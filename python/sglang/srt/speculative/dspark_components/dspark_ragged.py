from __future__ import annotations

import bisect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class RaggedVerifyMode(str):
    STATIC = "static"
    CAP_ACCEPT = "cap-accept"
    COMPACT = "compact"


def read_ragged_verify_mode() -> str:
    mode = str(envs.SGLANG_RAGGED_VERIFY_MODE.get())
    if mode not in (RaggedVerifyMode.STATIC, RaggedVerifyMode.CAP_ACCEPT, RaggedVerifyMode.COMPACT):
        raise ValueError(
            f"invalid SGLANG_RAGGED_VERIFY_MODE={mode!r}; expected "
            "'static', 'cap-accept', or 'compact'"
        )
    return mode


@dataclass(frozen=True)
class SpsTable:
    batch_tokens: list[int]
    steps_per_sec: list[float]

    def lookup(self, tokens: int) -> float:
        idx = max(0, bisect.bisect_right(self.batch_tokens, int(tokens)) - 1)
        return float(self.steps_per_sec[min(idx, len(self.steps_per_sec) - 1)])


def load_sps_table(path: Optional[str], max_batch_tokens: int) -> SpsTable:
    if not path:
        return SpsTable([1, max(1, int(max_batch_tokens))], [1.0, 1.0])
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "sample_batch_tokens" in data:
        xs, ys = data["sample_batch_tokens"], data["sample_steps_per_sec"]
    elif "bs_probes" in data:
        xs = data["bs_probes"]
        ys = [
            1.0 / max(
                1e-9,
                float(data.get("bias_seconds", 0.0))
                + float(alpha)
                + float(data["theta_seconds"][min(i, len(data["theta_seconds"]) - 1)]),
            )
            for i, alpha in enumerate(data["alpha_seconds"])
        ]
    else:
        raise ValueError(f"Unsupported DSpark SPS table format in {path}")
    if len(xs) != len(ys) or not xs:
        raise ValueError(f"Invalid DSpark SPS table in {path}")
    return SpsTable([int(x) for x in xs], [float(y) for y in ys])


def load_sts_temperatures(path: Optional[str], gamma: int) -> Optional[torch.Tensor]:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    temperatures = data.get("temperatures")
    if not isinstance(temperatures, list) or len(temperatures) != int(gamma):
        raise ValueError(
            f"DSpark STS calibration must contain gamma={gamma} temperatures"
        )
    if any(float(x) <= 0 for x in temperatures):
        raise ValueError("DSpark STS temperatures must be positive")
    return torch.tensor(temperatures, dtype=torch.float32)


def schedule_verify_lens(
    confidence: torch.Tensor, budget: int, gamma: int, max_verify_len: int
) -> torch.Tensor:
    survival = torch.cumprod(confidence.float(), dim=1)[:, :gamma]
    if int(budget) <= 0:
        return torch.ones((survival.shape[0],), dtype=torch.int32, device=confidence.device)
    flat = survival.reshape(-1)
    valid = flat >= 1e-6
    values = torch.where(valid, flat, torch.full_like(flat, -torch.inf))
    take = min(int(budget), int(values.numel()))
    selected = torch.topk(values, k=take, sorted=False).indices
    request_ids = selected // int(survival.shape[1])
    counts = torch.bincount(request_ids, minlength=survival.shape[0])
    return (counts + 1).clamp(min=1, max=int(max_verify_len)).to(torch.int32)


class DSparkRaggedPlanner:
    def __init__(self, *, worker, gamma: int, server_args) -> None:
        self.worker = worker
        self.gamma = int(gamma)
        self.mode = read_ragged_verify_mode()
        self.enabled = self.mode != RaggedVerifyMode.STATIC
        graph_runner = getattr(worker.model_runner, "graph_runner", None)
        backend = getattr(worker.model_runner, "attn_backend", None)
        self.graph_enabled = bool(
            self.mode == RaggedVerifyMode.COMPACT
            and getattr(backend, "supports_dspark_ragged_graph", False)
            and graph_runner is not None
            and not getattr(server_args, "disable_cuda_graph", False)
        )
        self.graph_slots = int(
            getattr(
                graph_runner,
                "max_bs",
                server_args.max_running_requests or 1,
            )
        )
        capture_bs = getattr(graph_runner, "capture_bs", [])
        width = self.gamma + 1
        self.graph_token_buckets = sorted(
            {
                int(bs) * width
                for bs in capture_bs
                if int(bs) * width >= self.graph_slots
            }
        )
        self.has_profile = server_args.speculative_dspark_sps_table_path is not None
        self.sps = load_sps_table(
            server_args.speculative_dspark_sps_table_path,
            max_batch_tokens=max(1, int(server_args.max_running_requests or 1) * self.gamma),
        )
        self.sts_temperatures = load_sts_temperatures(
            server_args.speculative_dspark_confidence_sts_path, self.gamma
        )
        if self.enabled and getattr(worker.draft_model, "confidence_head", None) is None:
            raise ValueError(
                f"DSpark {self.mode} mode requires a trained confidence_head; "
                "use SGLANG_RAGGED_VERIFY_MODE=static for a head-less checkpoint."
            )
        if self.enabled and server_args.speculative_dspark_sps_table_path is None:
            logger.warning(
                "DSpark %s mode has no SPS table; using verify-all budget. "
                "Profile --speculative-dspark-sps-table-path for scheduling gain.",
                self.mode,
            )
        if self.enabled and self.mode == RaggedVerifyMode.COMPACT and not self.graph_enabled:
            logger.info(
                "DSpark compact graph tiers are unavailable for attention backend %s; "
                "using eager ragged verify.",
                type(backend).__name__ if backend is not None else "unknown",
            )

    def prepare_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        confidence = torch.sigmoid(confidence.float())
        if self.sts_temperatures is not None:
            temperature = self.sts_temperatures.to(confidence.device)
            confidence = torch.sigmoid(
                torch.logit(confidence.clamp(1e-6, 1 - 1e-6)) / temperature
            )
        return confidence.clamp(0.0, 1.0)

    def plan(self, confidence: Optional[torch.Tensor], bs: int) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        if confidence is None:
            raise RuntimeError("DSpark ragged mode did not receive confidence output")
        confidence = self.prepare_confidence(confidence)
        survival = torch.cumprod(confidence, dim=1)
        full = int(bs) * self.gamma
        if not self.has_profile:
            budget = full
        else:
            sorted_survival = torch.sort(survival.reshape(-1), descending=True).values
            scores = []
            for extra in range(full + 1):
                tokens = int(bs) + extra
                expected = int(bs) + float(sorted_survival[:extra].sum().item())
                scores.append(expected * self.sps.lookup(tokens))
            budget = int(max(range(len(scores)), key=scores.__getitem__))
        return schedule_verify_lens(confidence, budget, self.gamma, self.gamma + 1)

    def graph_num_tokens(self, verify_lens: torch.Tensor, bs: int) -> Optional[int]:
        if not self.graph_enabled or not self.graph_token_buckets:
            return None
        total = int(verify_lens.to(torch.int64).sum().item())
        required = total + max(0, self.graph_slots - int(bs))
        for bucket in self.graph_token_buckets:
            padded_actual = bucket - max(0, self.graph_slots - int(bs))
            if padded_actual >= total and padded_actual <= int(bs) * (self.gamma + 1):
                return bucket
        return None
