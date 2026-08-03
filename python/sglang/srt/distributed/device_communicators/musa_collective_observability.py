# SPDX-License-Identifier: Apache-2.0

import contextvars
import json
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_phase: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sglang_musa_collective_phase", default="unscoped"
)
_capture: contextvars.ContextVar[Optional["CaptureRecord"]] = contextvars.ContextVar(
    "sglang_musa_collective_capture", default=None
)
_event_counts: Counter[tuple[str, ...]] = Counter()
_event_count = 0


def enabled() -> bool:
    return envs.SGLANG_MUSA_COLLECTIVE_OBSERVABILITY.get()


def current_phase() -> str:
    return _phase.get()


def _max_events() -> int:
    return envs.SGLANG_MUSA_COLLECTIVE_OBSERVABILITY_MAX_EVENTS.get()


def _json_log(kind: str, payload: dict[str, Any]) -> None:
    logger.info(
        "[MUSA_COLLECTIVE] %s",
        json.dumps({"kind": kind, **payload}, sort_keys=True),
    )


def _group_fields(group: Any) -> dict[str, Any]:
    ranks = list(getattr(group, "ranks", ()))
    return {
        "group": getattr(group, "group_name", "unknown"),
        "group_id": getattr(group, "unique_name", None),
        "ranks": ranks,
        "world_size": getattr(group, "world_size", None),
    }


def _tensor_fields(output: Any, input: Any) -> dict[str, Any]:
    return {
        "input_shape": list(input.shape),
        "output_shape": list(output.shape),
        "dtype": str(input.dtype),
    }


def startup_event(event: str, group: Any, **fields: Any) -> None:
    if not enabled():
        return
    _json_log(event, {**_group_fields(group), **fields})


@contextmanager
def phase_scope(name: str) -> Iterator[None]:
    if not enabled():
        yield
        return

    token = _phase.set(name)
    before = _event_count
    _json_log("phase_begin", {"phase": name})
    try:
        yield
    finally:
        _json_log(
            "phase_end",
            {"phase": name, "route_events": _event_count - before},
        )
        _phase.reset(token)


@dataclass
class CaptureRecord:
    key: dict[str, Any]
    routes: Counter[tuple[str, ...]] = field(default_factory=Counter)
    manifest: Optional[dict[str, Any]] = None


@contextmanager
def graph_capture_scope(**key: Any) -> Iterator[CaptureRecord]:
    if not enabled():
        record = CaptureRecord(key={})
        yield record
        return

    record = CaptureRecord(key=key)
    capture_token = _capture.set(record)
    phase_token = None
    if "phase" in key:
        phase_token = _phase.set(str(key["phase"]))
    try:
        yield record
    finally:
        record.manifest = {
            "key": key,
            "routes": [
                {"count": count, "route": list(route)}
                for route, count in sorted(record.routes.items())
            ],
        }
        _json_log("graph_capture_manifest", record.manifest)
        if phase_token is not None:
            _phase.reset(phase_token)
        _capture.reset(capture_token)


def record_graph_replay(manifest: Optional[dict[str, Any]], **fields: Any) -> None:
    if not enabled() or manifest is None:
        return
    _json_log("graph_replay", {"manifest": manifest, **fields})


def record_route(
    *,
    op: str,
    route: str,
    reason: str,
    group: Any,
    output: Any,
    input: Any,
) -> None:
    global _event_count
    if not enabled():
        return

    phase = current_phase()
    group_fields = _group_fields(group)
    event_key = (
        phase,
        op,
        route,
        reason,
        group_fields["group"],
        ",".join(str(rank) for rank in group_fields["ranks"]),
        str(group_fields["world_size"]),
    )
    _event_counts[event_key] += 1
    _event_count += 1

    capture = _capture.get()
    if capture is not None:
        capture.routes[event_key] += 1

    if _event_count <= _max_events():
        _json_log(
            "route",
            {
                "phase": phase,
                "op": op,
                "route": route,
                "reason": reason,
                "graph_state": "capture" if capture is not None else "eager",
                **group_fields,
                **_tensor_fields(output, input),
            },
        )
    elif _event_count == _max_events() + 1:
        _json_log("route_limit_reached", {"max_events": _max_events()})


def route_summary() -> dict[str, int]:
    return {"|".join(key): count for key, count in sorted(_event_counts.items())}
