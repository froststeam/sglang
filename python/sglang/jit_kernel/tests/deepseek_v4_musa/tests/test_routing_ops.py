from __future__ import annotations

import os

import pytest
import torch

from sglang.test.ci.ci_register import register_musa_ci
from ..utils import MUSA_OPS, REPO_ROOT, get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)

hash_topk_musa = MUSA_OPS.hash_topk_musa
mask_topk_ids_musa = MUSA_OPS.mask_topk_ids_musa
topk_ids_logical_to_physical_static_musa = MUSA_OPS.topk_ids_logical_to_physical_static_musa


def _require_real_tilelang_musa_test() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip(
            "set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 "
            "to run real TileLang MUSA kernel validation"
        )
    pytest.importorskip("tilelang")


def _assert_repeat_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if torch.equal(actual, expected):
        return
    mismatch_count = int((actual != expected).sum().item())
    if actual.dtype.is_floating_point:
        max_abs_diff = float((actual.float() - expected.float()).abs().max().item())
        detail = f", max_abs_diff={max_abs_diff:.6g}"
    else:
        detail = ""
    pytest.fail(
        f"{name} is not repeat-deterministic: "
        f"mismatch_count={mismatch_count}/{actual.numel()}{detail}"
    )


def tilelang_hash_topk_unavailable(*args, **kwargs) -> None:
    del args, kwargs
    return None


def load_moe_align_block_size_for_test():
    import importlib.util
    import sys
    import types

    saved_modules = {name: sys.modules.get(name) for name in ("sglang", "sglang.srt", "sglang.srt.utils")}
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.__dict__["is_cuda"] = lambda: False
    utils_module.__dict__["is_hip"] = lambda: False
    utils_module.__dict__["is_musa"] = lambda: True
    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = srt_module
    sys.modules["sglang.srt.utils"] = utils_module
    sglang_module.__dict__["srt"] = srt_module
    srt_module.__dict__["utils"] = utils_module
    try:
        align_path = REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "moe" / "fused_moe_triton" / "moe_align_block_size.py"
        if not align_path.exists():
            pytest.skip("legacy fused_moe_triton/moe_align_block_size.py is not present in this SGLang tree")
        spec = importlib.util.spec_from_file_location("test_moe_align_block_size", align_path)
        assert spec is not None and spec.loader is not None
        align_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(align_module)
        except ImportError as exc:
            pytest.skip(f"sgl_kernel.moe_align_block_size is required: {exc}")
        return align_module.moe_align_block_size
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_moe_align_block_size_missing_sgl_kernel_fails_before_dispatch(monkeypatch) -> None:
    import builtins
    import importlib.util
    import sys
    import types

    saved_modules = {name: sys.modules.get(name) for name in ("sglang", "sglang.srt", "sglang.srt.utils")}
    sglang_module = types.ModuleType("sglang")
    srt_module = types.ModuleType("sglang.srt")
    utils_module = types.ModuleType("sglang.srt.utils")
    utils_module.__dict__["is_cuda"] = lambda: False
    utils_module.__dict__["is_hip"] = lambda: False
    utils_module.__dict__["is_musa"] = lambda: True
    sys.modules["sglang"] = sglang_module
    sys.modules["sglang.srt"] = srt_module
    sys.modules["sglang.srt.utils"] = utils_module
    sglang_module.__dict__["srt"] = srt_module
    srt_module.__dict__["utils"] = utils_module

    real_import = builtins.__import__

    def import_without_sgl_kernel(name, *args, **kwargs):
        if name == "sgl_kernel":
            raise ImportError("sgl_kernel unavailable for MP31 moe_align_block_size")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sgl_kernel)
    try:
        align_path = REPO_ROOT / "python" / "sglang" / "srt" / "layers" / "moe" / "fused_moe_triton" / "moe_align_block_size.py"
        if not align_path.exists():
            pytest.skip("legacy fused_moe_triton/moe_align_block_size.py is not present in this SGLang tree")
        spec = importlib.util.spec_from_file_location("test_moe_align_block_size_missing_sgl_kernel", align_path)
        assert spec is not None and spec.loader is not None
        align_module = importlib.util.module_from_spec(spec)
        with pytest.raises(ImportError, match="sgl_kernel unavailable for MP31 moe_align_block_size"):
            spec.loader.exec_module(align_module)
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_fused_moe_kernels_for_debug_test():
    import importlib.util
    import sys
    import types

    module_names = [
        "sglang",
        "sglang.srt",
        "sglang.srt.debug_utils",
        "sglang.srt.debug_utils.deepseek_v4_debug_utils",
        "sglang.srt.layers",
        "sglang.srt.layers.quantization",
        "sglang.srt.layers.quantization.fp8_kernel",
        "sglang.srt.layers.quantization.int8_kernel",
        "sglang.srt.utils",
    ]
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules[name] = types.ModuleType(name)

    sys.modules["sglang"].__dict__["srt"] = sys.modules["sglang.srt"]
    sys.modules["sglang.srt"].__dict__["debug_utils"] = sys.modules["sglang.srt.debug_utils"]
    sys.modules["sglang.srt"].__dict__["layers"] = sys.modules["sglang.srt.layers"]
    sys.modules["sglang.srt"].__dict__["utils"] = sys.modules["sglang.srt.utils"]
    sys.modules["sglang.srt.debug_utils"].__dict__["deepseek_v4_debug_utils"] = sys.modules[
        "sglang.srt.debug_utils.deepseek_v4_debug_utils"
    ]
    sys.modules["sglang.srt.layers"].__dict__["quantization"] = sys.modules["sglang.srt.layers.quantization"]
    sys.modules["sglang.srt.layers.quantization"].__dict__["fp8_kernel"] = sys.modules[
        "sglang.srt.layers.quantization.fp8_kernel"
    ]
    sys.modules["sglang.srt.layers.quantization"].__dict__["int8_kernel"] = sys.modules[
        "sglang.srt.layers.quantization.int8_kernel"
    ]

    def deepseek_v4_moe_code_path_checker_stub(*args, **kwargs):
        del args, kwargs

    sys.modules[
        "sglang.srt.debug_utils.deepseek_v4_debug_utils"
    ].__dict__["deepseek_v4_moe_code_path_checker"] = deepseek_v4_moe_code_path_checker_stub

    fp8 = sys.modules["sglang.srt.layers.quantization.fp8_kernel"]
    fp8.__dict__["per_token_group_quant_fp8"] = lambda A, block_k: (
        A.to(torch.float8_e4m3fn),
        torch.ones((A.shape[0], (A.shape[1] + block_k - 1) // block_k), device=A.device, dtype=torch.float32),
    )

    def scaled_fp8_quant_stub(A, A_scale, use_per_token_if_dynamic=False):
        del A_scale, use_per_token_if_dynamic
        return A, torch.ones((A.shape[0], 1), device=A.device, dtype=torch.float32)

    fp8.__dict__["scaled_fp8_quant"] = scaled_fp8_quant_stub
    fp8.__dict__["sglang_per_token_group_quant_fp8"] = fp8.__dict__["per_token_group_quant_fp8"]

    int8 = sys.modules["sglang.srt.layers.quantization.int8_kernel"]
    int8.__dict__["per_token_group_quant_int8"] = lambda A, block_k: (
        A.to(torch.int8),
        torch.ones((A.shape[0], (A.shape[1] + block_k - 1) // block_k), device=A.device, dtype=torch.float32),
    )
    int8.__dict__["per_token_quant_int8"] = lambda A: (
        A.to(torch.int8),
        torch.ones((A.shape[0], 1), device=A.device, dtype=torch.float32),
    )
    int8.__dict__["sglang_per_token_group_quant_int8"] = int8.__dict__["per_token_group_quant_int8"]

    utils = sys.modules["sglang.srt.utils"]
    utils.__dict__["cpu_has_amx_support"] = lambda: False
    utils.__dict__["get_bool_env_var"] = lambda name, default=False: str(os.getenv(name, default)).strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "t",
        "on",
    )
    utils.__dict__["get_device_name"] = lambda: "MUSA"
    utils.__dict__["is_cpu"] = lambda: False
    utils.__dict__["is_cuda"] = lambda: False
    utils.__dict__["is_hip"] = lambda: False
    utils.__dict__["is_sm90_supported"] = lambda: False

    try:
        kernels_path = (
            REPO_ROOT
            / "python"
            / "sglang"
            / "srt"
            / "layers"
            / "moe"
            / "fused_moe_triton"
            / "fused_moe_triton_kernels.py"
        )
        if not kernels_path.exists():
            pytest.skip("legacy fused_moe_triton/fused_moe_triton_kernels.py is not present in this SGLang tree")
        spec = importlib.util.spec_from_file_location("test_fused_moe_triton_kernels_debug", kernels_path)
        assert spec is not None and spec.loader is not None
        kernels = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernels)
        return kernels
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _rand_fp8(shape: tuple[int, ...], scale: float = 0.01) -> torch.Tensor:
    return (torch.randn(shape, device=get_musa_device(), dtype=torch.bfloat16) * scale).to(torch.float8_e4m3fn)


def _invoke_stub_fp8_fused_moe(
    kernels,
    A: torch.Tensor,
    B: torch.Tensor,
    top_k: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    debug_moe_stage: str,
) -> torch.Tensor:
    import triton.language as tl

    block_k = 128
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
    }
    C = torch.empty((A.shape[0], top_k, B.shape[1]), device=A.device, dtype=torch.bfloat16)
    B_scale = torch.ones(
        (B.shape[0], (B.shape[1] + 127) // 128, (B.shape[2] + block_k - 1) // block_k),
        device=A.device,
        dtype=torch.float32,
    )
    topk_weights = torch.ones((A.shape[0], top_k), device=A.device, dtype=torch.bfloat16)
    topk_ids = torch.zeros((A.shape[0], top_k), device=A.device, dtype=torch.int32)
    num_tokens_post_padded = torch.tensor([64], device=A.device, dtype=torch.int32)

    kernels.invoke_fused_moe_kernel(
        A,
        B,
        None,
        C,
        None,
        B_scale,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        top_k,
        config,
        tl.bfloat16,
        True,
        False,
        False,
        False,
        False,
        block_shape=[128, block_k],
        debug_moe_stage=debug_moe_stage,
    )
    getattr(torch, "musa").synchronize()
    assert torch.isfinite(C.float()).all()
    return C


def test_hash_topk_musa_invokes_tilelang_fixed_expert_path(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_hash_topk_musa(
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tid2eid: torch.Tensor,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor))
        return (
            torch.full((router_logits.shape[0], tid2eid.shape[1]), 0.5, dtype=torch.float32),
            torch.full((router_logits.shape[0], tid2eid.shape[1]), 7, dtype=torch.int64),
        )

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", fake_try_tilelang_hash_topk_musa)

    weights, ids = hash_topk_musa(
        torch.zeros((2, 8), dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
    )

    assert len(calls) == 1
    assert calls[0][3] == 0
    assert calls[0][4] == 1.0
    torch.testing.assert_close(weights, torch.full((2, 2), 0.5, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(ids, torch.full((2, 2), 7, dtype=torch.int64), rtol=0, atol=0)


def test_hash_topk_musa_keeps_int32_input_ids_for_tilelang(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_hash_topk_musa(
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tid2eid: torch.Tensor,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(input_ids)
        return (
            torch.full((router_logits.shape[0], tid2eid.shape[1]), 0.5, dtype=torch.float32),
            torch.full((router_logits.shape[0], tid2eid.shape[1]), 7, dtype=torch.int64),
        )

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", fake_try_tilelang_hash_topk_musa)

    weights, ids = hash_topk_musa(
        torch.zeros((2, 8), dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
    )

    assert len(calls) == 1
    assert calls[0].dtype == torch.int32
    torch.testing.assert_close(weights, torch.full((2, 2), 0.5, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(ids, torch.full((2, 2), 7, dtype=torch.int64), rtol=0, atol=0)


def test_hash_topk_musa_invokes_tilelang_fixed_expert_path_with_shared(monkeypatch) -> None:
    calls = []

    def fake_try_tilelang_hash_topk_musa(
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        tid2eid: torch.Tensor,
        num_fused_shared_experts: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append((router_logits, input_ids, tid2eid, num_fused_shared_experts, routed_scaling_factor))
        output_topk = tid2eid.shape[1] + num_fused_shared_experts
        return (
            torch.full((router_logits.shape[0], output_topk), 0.25, dtype=torch.float32),
            torch.full((router_logits.shape[0], output_topk), 9, dtype=torch.int64),
        )

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", fake_try_tilelang_hash_topk_musa)

    weights, ids = hash_topk_musa(
        torch.zeros((2, 8), dtype=torch.float32),
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        num_fused_shared_experts=1,
        routed_scaling_factor=2.0,
    )

    assert len(calls) == 1
    assert calls[0][3] == 1
    assert calls[0][4] == 2.0
    torch.testing.assert_close(weights, torch.full((2, 3), 0.25, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(ids, torch.full((2, 3), 9, dtype=torch.int64), rtol=0, atol=0)


def test_hash_topk_musa_with_shared_experts_uses_reference_semantics(monkeypatch) -> None:
    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", tilelang_hash_topk_unavailable)

    router_logits = torch.tensor([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]], dtype=torch.float32)
    input_ids = torch.tensor([1, 0], dtype=torch.int64)
    tid2eid = torch.tensor([[2, 1], [0, 2]], dtype=torch.int64)

    weights, ids = hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.0,
    )

    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    routed_scores = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    expected_ids = torch.cat([routed_ids, torch.tensor([[3], [3]], dtype=torch.int64)], dim=1)
    expected_weights = torch.cat(
        [routed_scores.to(torch.float32), torch.full((2, 1), 0.5, dtype=torch.float32)], dim=1
    )

    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids)


def test_hash_topk_musa_applies_routed_scaling_factor_to_shared_experts(monkeypatch) -> None:
    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", tilelang_hash_topk_unavailable)

    router_logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    input_ids = torch.tensor([0], dtype=torch.int64)
    tid2eid = torch.tensor([[3, 1]], dtype=torch.int64)

    weights, ids = hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=2,
        routed_scaling_factor=4.0,
    )

    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, tid2eid.long())
    expected_routed = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    expected_weights = torch.cat([expected_routed, torch.full((1, 2), 0.25)], dim=1)
    expected_ids = torch.tensor([[3, 1, 4, 5]], dtype=torch.int64)
    torch.testing.assert_close(weights, expected_weights.to(torch.float32))
    torch.testing.assert_close(ids, expected_ids)


def test_hash_topk_musa_real_tilelang_matches_reference() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    router_logits = torch.tensor(
        [[0.0, 1.0, -1.0, 0.5], [2.0, -2.0, 0.5, 1.5]],
        device=device,
        dtype=torch.float32,
    )
    input_ids = torch.tensor([1, 0], device=device, dtype=torch.int64)
    tid2eid = torch.tensor([[2, 1], [0, 3]], device=device, dtype=torch.int64)

    weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.0,
    )

    assert weights is not None
    assert ids is not None
    assert ids.dtype == torch.int64
    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    routed_scores = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    expected_ids = torch.cat([routed_ids, torch.full((2, 1), 4, device=device, dtype=torch.int64)], dim=1)
    expected_weights = torch.cat(
        [routed_scores.to(torch.float32), torch.full((2, 1), 0.5, device=device, dtype=torch.float32)], dim=1
    )

    torch.testing.assert_close(weights.cpu(), expected_weights.cpu())
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu())


def test_hash_topk_musa_real_tilelang_matches_hash_topk_log_shapes() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_experts = 256
    topk = 6
    num_tid = 32000
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 37) % num_experts

    for num_tokens in (128, 1):
        router_logits = torch.linspace(
            -4.0,
            4.0,
            steps=num_tokens * num_experts,
            device=device,
            dtype=torch.float32,
        ).reshape(num_tokens, num_experts)
        input_ids = torch.arange(num_tokens, device=device, dtype=torch.int64) % num_tid

        weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
        )

        assert weights is not None
        assert ids is not None
        routed_ids = tid2eid[input_ids].to(torch.int64)
        routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
        expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
        assert tuple(weights.shape) == (num_tokens, topk)
        assert tuple(ids.shape) == (num_tokens, topk)
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int64
        assert int(ids.cpu().min()) >= 0
        assert int(ids.cpu().max()) < num_experts
        torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(ids.cpu(), routed_ids.cpu())


def test_hash_topk_musa_real_tilelang_accepts_int32_tid2eid_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_experts = 256
    topk = 6
    num_tid = 32000
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int32).reshape(num_tid, topk) * 37) % num_experts

    for num_tokens in (128, 1):
        router_logits = torch.linspace(
            -4.0,
            4.0,
            steps=num_tokens * num_experts,
            device=device,
            dtype=torch.float32,
        ).reshape(num_tokens, num_experts)
        input_ids = torch.arange(num_tokens, device=device, dtype=torch.int64) % num_tid

        weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
        )

        assert weights is not None
        assert ids is not None
        routed_ids = tid2eid[input_ids].to(torch.int64)
        routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
        expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
        assert tuple(weights.shape) == (num_tokens, topk)
        assert tuple(ids.shape) == (num_tokens, topk)
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int64
        torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(ids.cpu(), routed_ids.cpu())


def test_hash_topk_musa_real_tilelang_bfloat16_accepts_int32_tid2eid_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 4
    num_experts = 256
    topk = 6
    num_tid = 128
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int32).reshape(num_tid, topk) * 19) % num_experts
    input_ids = torch.tensor([0, 11, 63, 127], device=device, dtype=torch.int64)
    router_logits = (
        (torch.arange(num_tokens * num_experts, device=device, dtype=torch.float32).reshape(num_tokens, num_experts) / 257.0)
        - 2.0
    ).to(torch.bfloat16)

    result = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert result is not None
    weights, ids = result

    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits.float()).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    torch.testing.assert_close(ids.cpu(), routed_ids.cpu())
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-3, atol=2e-3)


def test_hash_topk_musa_real_tilelang_matches_strided_reference() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 17
    num_experts = 32
    topk = 4
    num_tid = 64

    router_storage = torch.linspace(
        -3.0,
        3.0,
        steps=num_tokens * num_experts * 2 + 11,
        device=device,
        dtype=torch.float32,
    )
    router_logits = torch.as_strided(router_storage, (num_tokens, num_experts), (num_experts * 2, 2), storage_offset=5)
    input_storage = torch.arange(num_tokens * 3 + 5, device=device, dtype=torch.int64) % num_tid
    input_ids = torch.as_strided(input_storage, (num_tokens,), (3,), storage_offset=2)
    tid_storage = (torch.arange(num_tid * topk * 2 + 13, device=device, dtype=torch.int64) * 11) % num_experts
    tid2eid = torch.as_strided(tid_storage, (num_tid, topk), (topk * 2, 2), storage_offset=7)

    weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=2,
        routed_scaling_factor=4.0,
    )

    assert weights is not None
    assert ids is not None
    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    routed_scores = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    expected_ids = torch.cat(
        [routed_ids, torch.arange(num_experts, num_experts + 2, device=device, dtype=torch.int64).expand(num_tokens, 2)],
        dim=1,
    )
    expected_weights = torch.cat(
        [routed_scores.to(torch.float32), torch.full((num_tokens, 2), 0.25, device=device, dtype=torch.float32)],
        dim=1,
    )

    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu())


def test_hash_topk_to_moe_align_valid_prefix_matches_dispatch_log_shapes() -> None:
    import triton

    device = get_musa_device()
    moe_align_block_size = load_moe_align_block_size_for_test()

    num_experts = 256
    topk = 6
    num_tid = 32000
    block_size = 64
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 37) % num_experts

    for num_tokens in (128, 1):
        router_logits = torch.linspace(
            -4.0,
            4.0,
            steps=num_tokens * num_experts,
            device=device,
            dtype=torch.float32,
        ).reshape(num_tokens, num_experts)
        input_ids = torch.arange(num_tokens, device=device, dtype=torch.int64) % num_tid

        _, topk_ids = hash_topk_musa(router_logits, input_ids, tid2eid)
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size, num_experts)

        num_tokens_post_padded_value = int(num_tokens_post_padded.item())
        valid_blocks = triton.cdiv(num_tokens_post_padded_value, block_size)
        active_expert_ids = expert_ids[:valid_blocks]
        active_sorted_token_ids = sorted_token_ids[:num_tokens_post_padded_value]

        assert tuple(topk_ids.shape) == (num_tokens, topk)
        assert topk_ids.dtype == torch.int64
        assert num_tokens_post_padded_value % block_size == 0
        assert valid_blocks > 0
        assert valid_blocks <= expert_ids.numel()
        assert int(active_expert_ids.cpu().min()) >= 0
        assert int(active_expert_ids.cpu().max()) < num_experts
        assert int(active_sorted_token_ids.cpu().min()) >= 0
        assert int(active_sorted_token_ids.cpu().max()) <= topk_ids.numel()


def test_decode_dispatch_log_operator_chain_shapes(monkeypatch) -> None:
    import triton

    from ..utils import assert_sm90_aligned_scale_contract, reference_grouped_fp8_quant

    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", tilelang_hash_topk_unavailable)
    device = get_musa_device()
    moe_align_block_size = load_moe_align_block_size_for_test()
    num_experts = 256
    topk = 6
    num_tid = 32000
    block_size = 64
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 37) % num_experts
    router_logits = torch.linspace(-4.0, 4.0, steps=num_experts, device=device, dtype=torch.float32).reshape(1, num_experts)
    input_ids = torch.tensor([31999], device=device, dtype=torch.int64)

    weights, topk_ids = hash_topk_musa(router_logits, input_ids, tid2eid)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size, num_experts)
    num_tokens_post_padded_value = int(num_tokens_post_padded.item())
    valid_blocks = triton.cdiv(num_tokens_post_padded_value, block_size)

    assert tuple(weights.shape) == (1, topk)
    assert tuple(topk_ids.shape) == (1, topk)
    assert topk_ids.dtype == torch.int64
    assert int(topk_ids.cpu().min()) >= 0
    assert int(topk_ids.cpu().max()) < num_experts
    assert num_tokens_post_padded_value % block_size == 0
    assert int(expert_ids[:valid_blocks].cpu().min()) >= 0
    assert int(expert_ids[:valid_blocks].cpu().max()) < num_experts
    assert int(sorted_token_ids[:num_tokens_post_padded_value].cpu().max()) <= topk_ids.numel()

    for rows, cols, expected_groups in ((1, 4096, 32), (128, 4096, 32), (768, 2048, 16)):
        value = torch.linspace(-2.0, 2.0, steps=rows * cols, device=device, dtype=torch.float32).reshape(rows, cols)
        output = torch.empty_like(value, dtype=torch.float8_e4m3fn)
        output_scale = torch.empty((rows, expected_groups), device=device, dtype=torch.float32)
        MUSA_OPS._quantize_fp8_grouped(
            value,
            output,
            output_scale,
            quant_group_size=128,
            scale_ue8m0=False,
            transposed=False,
        )
        ref_quantized, ref_scale = reference_grouped_fp8_quant(value, 128)
        assert output.dtype == torch.float8_e4m3fn
        assert_sm90_aligned_scale_contract(output_scale, (rows, expected_groups))
        torch.testing.assert_close(output_scale.cpu(), ref_scale.cpu(), rtol=1e-3, atol=1e-6)
        torch.testing.assert_close(output.float().cpu(), ref_quantized.float().cpu(), rtol=0, atol=0)


def test_musa_moe_decode_stub_fp8_operator_chain_boundary() -> None:
    torch.manual_seed(6)
    kernels = load_fused_moe_kernels_for_debug_test()
    hidden = torch.randn((1, 4096), device=get_musa_device(), dtype=torch.bfloat16)
    sorted_w1 = torch.arange(6, device=hidden.device, dtype=torch.int32)
    sorted_w1 = torch.nn.functional.pad(sorted_w1, (0, 58), value=6).to(torch.int32)
    expert_ids = torch.tensor([0], device=hidden.device, dtype=torch.int32)

    w1_out = _invoke_stub_fp8_fused_moe(
        kernels,
        hidden,
        _rand_fp8((256, 256, 4096)),
        6,
        sorted_w1,
        expert_ids,
        "decode_chain_w1_curr_hidden_states",
    )
    assert tuple(w1_out.shape) == (1, 6, 256)

    gate, up = w1_out.view(-1, 256).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    getattr(torch, "musa").synchronize()
    assert tuple(activated.shape) == (6, 128)
    assert activated.dtype == torch.bfloat16
    assert torch.isfinite(activated.float()).all()

    sorted_w2 = torch.arange(6, device=hidden.device, dtype=torch.int32)
    sorted_w2 = torch.nn.functional.pad(sorted_w2, (0, 58), value=6).to(torch.int32)
    w2_out = _invoke_stub_fp8_fused_moe(
        kernels,
        activated.contiguous(),
        _rand_fp8((256, 512, 128)),
        1,
        sorted_w2,
        expert_ids,
        "decode_chain_w2_intermediate_cache2",
    )
    assert tuple(w2_out.shape) == (6, 1, 512)

    reduce_input = w2_out.view(1, 6, 512).contiguous()
    output = torch.empty((1, 512), device=hidden.device, dtype=torch.bfloat16)
    kernels.moe_sum_reduce_triton(reduce_input, output, 1.0)
    getattr(torch, "musa").synchronize()
    assert tuple(output.shape) == (1, 512)
    assert torch.isfinite(output.float()).all()


def test_hash_topk_musa_fallback_matches_dispatch_log_shapes(monkeypatch) -> None:
    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", tilelang_hash_topk_unavailable)

    num_experts = 256
    topk = 6
    num_tid = 32000
    tid2eid = (torch.arange(num_tid * topk, dtype=torch.int64).reshape(num_tid, topk) * 37) % num_experts

    for num_tokens in (128, 1):
        router_logits = torch.linspace(-4.0, 4.0, steps=num_tokens * num_experts, dtype=torch.float32).reshape(
            num_tokens, num_experts
        )
        input_ids = torch.arange(num_tokens, dtype=torch.int64) % num_tid

        weights, ids = hash_topk_musa(router_logits, input_ids, tid2eid)

        routed_ids = tid2eid[input_ids].to(torch.int64)
        routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
        expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
        assert tuple(weights.shape) == (num_tokens, topk)
        assert tuple(ids.shape) == (num_tokens, topk)
        assert weights.dtype == torch.float32
        assert ids.dtype == torch.int64
        assert int(ids.min()) >= 0
        assert int(ids.max()) < num_experts
        torch.testing.assert_close(weights, expected_weights.to(torch.float32))
        torch.testing.assert_close(ids, routed_ids)



def test_hash_topk_musa_fallback_keeps_large_int64_routed_ids(monkeypatch) -> None:
    monkeypatch.setattr(MUSA_OPS, "_try_tilelang_hash_topk_musa", tilelang_hash_topk_unavailable)

    router_logits = torch.linspace(-2.0, 2.0, steps=3 * 8, dtype=torch.float32).reshape(3, 8)
    input_ids = torch.tensor([0, 31999, 65535], dtype=torch.int64)
    tid2eid = torch.empty((65536, 3), dtype=torch.int64)
    tid2eid[:, 0] = torch.arange(65536, dtype=torch.int64) % 8
    tid2eid[:, 1] = (torch.arange(65536, dtype=torch.int64) * 3 + 1) % 8
    tid2eid[:, 2] = (torch.arange(65536, dtype=torch.int64) * 5 + 2) % 8

    weights, ids = hash_topk_musa(router_logits, input_ids, tid2eid)

    routed_ids = tid2eid[input_ids]
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    assert ids.dtype == torch.int64
    torch.testing.assert_close(ids, routed_ids)
    torch.testing.assert_close(weights, expected_weights.to(torch.float32))


def test_hash_topk_musa_real_tilelang_handles_large_input_ids_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_experts = 256
    topk = 6
    num_tid = 65536
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 37) % num_experts
    input_ids = torch.tensor([0, 31999, 65535], device=device, dtype=torch.int64)
    router_logits = torch.linspace(-4.0, 4.0, steps=input_ids.numel() * num_experts, device=device, dtype=torch.float32).reshape(input_ids.numel(), num_experts)

    weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )

    assert weights is not None
    assert ids is not None
    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True)
    assert ids.dtype == torch.int64
    torch.testing.assert_close(ids.cpu(), routed_ids.cpu())
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)


def test_hash_topk_musa_real_tilelang_topk16_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 5
    num_experts = 64
    num_tid = 32
    topk = 16
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 17) % num_experts
    input_ids = torch.tensor([0, 7, 13, 19, 31], device=device, dtype=torch.int64)
    router_logits = (
        torch.arange(num_tokens * num_experts, device=device, dtype=torch.float32).reshape(num_tokens, num_experts) / 127.0
    ) - 1.0

    weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )

    assert weights is not None
    assert ids is not None
    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    torch.testing.assert_close(ids.cpu(), routed_ids.cpu())
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)


def test_hash_topk_musa_real_tilelang_bfloat16_matches_reference_on_musa() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 4
    num_experts = 256
    topk = 6
    num_tid = 128
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 19) % num_experts
    input_ids = torch.tensor([0, 11, 63, 127], device=device, dtype=torch.int64)
    router_logits = (
        (torch.arange(num_tokens * num_experts, device=device, dtype=torch.float32).reshape(num_tokens, num_experts) / 257.0)
        - 2.0
    ).to(torch.bfloat16)

    result = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert result is not None
    weights, ids = result

    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits.float()).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    torch.testing.assert_close(ids.cpu(), routed_ids.cpu())
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=2e-3, atol=2e-3)


def test_hash_topk_musa_real_tilelang_is_repeat_deterministic_on_musa() -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20261625)
    num_tokens = 32
    num_experts = 256
    topk = 6
    num_tid = 256
    tid2eid = (
        torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(
            num_tid, topk
        )
        * 19
    ) % num_experts
    input_ids = (torch.arange(num_tokens, device=device, dtype=torch.int64) * 7) % num_tid
    router_logits = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )

    expected_weights, expected_ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert expected_weights is not None
    assert expected_ids is not None
    torch.musa.synchronize()
    for repeat_idx in range(20):
        weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
            router_logits,
            input_ids,
            tid2eid,
            num_fused_shared_experts=0,
            routed_scaling_factor=1.0,
        )
        assert weights is not None
        assert ids is not None
        torch.musa.synchronize()
        prefix = f"repeat={repeat_idx}"
        _assert_repeat_exact(f"hash_topk weights {prefix}", weights, expected_weights)
        _assert_repeat_exact(f"hash_topk ids {prefix}", ids, expected_ids)


@pytest.mark.parametrize("num_tokens", [1, 32, 65, 129])
def test_hash_topk_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20261725 + num_tokens)
    num_experts = 256
    topk = 6
    num_tid = 256
    tid2eid = (
        torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(
            num_tid, topk
        )
        * 17
    ) % num_experts
    probe_logits = torch.randn((num_experts,), device=device, dtype=torch.float32)
    probe_input_id = torch.tensor(37, device=device, dtype=torch.int64)

    baseline_weights, baseline_ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        probe_logits.view(1, num_experts),
        probe_input_id.view(1),
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert baseline_weights is not None
    assert baseline_ids is not None
    torch.musa.synchronize()
    expected_weights = baseline_weights[0].detach().clone()
    expected_ids = baseline_ids[0].detach().clone()

    router_logits = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    input_ids = (torch.arange(num_tokens, device=device, dtype=torch.int64) * 11) % num_tid
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        router_logits[pos].copy_(probe_logits)
        input_ids[pos].copy_(probe_input_id)

    weights, ids = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert weights is not None
    assert ids is not None
    torch.musa.synchronize()
    for pos in probe_positions:
        prefix = f"num_tokens={num_tokens}, pos={pos}"
        torch.testing.assert_close(weights[pos].cpu(), expected_weights.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ids[pos].cpu(), expected_ids.cpu(), rtol=0, atol=0)


def _ordered_sqrtsoftplus_topk_ref(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.nn.functional.softplus(gating_output).sqrt()
    choice = scores + correction_bias.unsqueeze(0)
    work = choice.clone()
    ids = []
    weights = []
    for _ in range(topk):
        max_values = work.max(dim=-1, keepdim=True).values
        # Match the fused kernels: repeated max selection, lower expert id wins ties.
        selected = torch.argmax((work == max_values).to(torch.int32), dim=-1).to(torch.int32)
        ids.append(selected)
        weights.append(scores.gather(1, selected.to(torch.int64).view(-1, 1)).squeeze(1))
        work.scatter_(1, selected.to(torch.int64).view(-1, 1), float("-inf"))
    topk_ids = torch.stack(ids, dim=1)
    topk_scores = torch.stack(weights, dim=1).to(torch.float32)
    topk_weights = topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    return topk_weights, topk_ids


def test_moe_fused_gate_musa_real_tilelang_matches_biased_topk_reference() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 17
    num_experts = 256
    topk = 8
    gating_output = torch.linspace(
        -4.0,
        4.0,
        steps=num_tokens * num_experts,
        device=device,
        dtype=torch.float32,
    ).reshape(num_tokens, num_experts)
    correction_bias = torch.linspace(-0.2, 0.2, steps=num_experts, device=device, dtype=torch.float32)

    result = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=None,
        apply_routed_scaling_factor_on_output=False,
    )
    assert result is not None
    weights, ids = result

    scores = torch.nn.functional.softplus(gating_output).sqrt()
    _, expected_ids = torch.topk(scores + correction_bias.unsqueeze(0), k=topk, dim=-1, sorted=False)
    expected_scores = scores.gather(1, expected_ids)
    expected_weights = expected_scores / expected_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    torch.testing.assert_close(ids.cpu().sort(dim=-1).values, expected_ids.to(torch.int32).cpu().sort(dim=-1).values)
    actual_by_id = torch.gather(weights, 1, ids.argsort(dim=-1))
    ref_by_id = torch.gather(expected_weights, 1, expected_ids.argsort(dim=-1))
    torch.testing.assert_close(actual_by_id.cpu(), ref_by_id.cpu(), rtol=1e-4, atol=1e-6)


def test_moe_fused_gate_musa_real_tilelang_matches_order_and_tie_break() -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    topk = 6
    num_experts = 256
    base = torch.linspace(-1.0, 1.0, steps=4 * num_experts, device=device, dtype=torch.float32).reshape(4, num_experts)
    # Rows 0/1 contain deliberate score ties after bias to guard output order.
    base[0, 3] = base[0, 7]
    base[1, 11] = base[1, 13]
    correction_bias = torch.zeros((num_experts,), device=device, dtype=torch.float32)

    result = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
        base,
        correction_bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=None,
        apply_routed_scaling_factor_on_output=False,
    )
    assert result is not None
    weights, ids = result
    expected_weights, expected_ids = _ordered_sqrtsoftplus_topk_ref(base, correction_bias, topk)
    torch.testing.assert_close(ids.cpu(), expected_ids.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)


def test_moe_fused_gate_musa_real_tilelang_is_repeat_deterministic_on_musa() -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20261825)
    num_tokens = 32
    num_experts = 256
    topk = 8
    gating_output = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    correction_bias = torch.linspace(
        -0.2, 0.2, steps=num_experts, device=device, dtype=torch.float32
    )

    expected = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=None,
        apply_routed_scaling_factor_on_output=False,
    )
    assert expected is not None
    expected_weights, expected_ids = expected
    torch.musa.synchronize()
    for repeat_idx in range(20):
        result = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
            gating_output,
            correction_bias,
            topk=topk,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=0,
            renormalize=True,
            routed_scaling_factor=None,
            apply_routed_scaling_factor_on_output=False,
        )
        assert result is not None
        weights, ids = result
        torch.musa.synchronize()
        prefix = f"repeat={repeat_idx}"
        _assert_repeat_exact(f"moe_fused_gate weights {prefix}", weights, expected_weights)
        _assert_repeat_exact(f"moe_fused_gate ids {prefix}", ids, expected_ids)


@pytest.mark.parametrize("num_tokens", [1, 32, 65, 129])
def test_moe_fused_gate_musa_real_tilelang_probe_row_is_batch_shape_invariant_on_musa(
    num_tokens: int,
) -> None:
    _require_real_tilelang_musa_test()
    device = get_musa_device()
    torch.manual_seed(20261925 + num_tokens)
    num_experts = 256
    topk = 8
    correction_bias = torch.linspace(
        -0.2, 0.2, steps=num_experts, device=device, dtype=torch.float32
    )
    probe_logits = torch.randn((num_experts,), device=device, dtype=torch.float32)

    baseline = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
        probe_logits.view(1, num_experts),
        correction_bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=None,
        apply_routed_scaling_factor_on_output=False,
    )
    assert baseline is not None
    baseline_weights, baseline_ids = baseline
    torch.musa.synchronize()
    expected_weights = baseline_weights[0].detach().clone()
    expected_ids = baseline_ids[0].detach().clone()

    gating_output = torch.randn(
        (num_tokens, num_experts), device=device, dtype=torch.float32
    )
    probe_positions = sorted({0, min(17, num_tokens - 1), num_tokens - 1})
    for pos in probe_positions:
        gating_output[pos].copy_(probe_logits)

    result = MUSA_OPS._try_tilelang_moe_fused_gate_musa(
        gating_output,
        correction_bias,
        topk=topk,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        renormalize=True,
        routed_scaling_factor=None,
        apply_routed_scaling_factor_on_output=False,
    )
    assert result is not None
    weights, ids = result
    torch.musa.synchronize()
    for pos in probe_positions:
        torch.testing.assert_close(weights[pos].cpu(), expected_weights.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(ids[pos].cpu(), expected_ids.cpu(), rtol=0, atol=0)


def test_biased_topk_tilelang_musa_miss_fails_closed_in_graph(monkeypatch) -> None:
    device = get_musa_device()
    from sglang.srt.hardware_backend.layers.deepseek_v4_musa import ops as dsv4_musa_ops
    from sglang.srt.layers.moe import topk as topk_module

    monkeypatch.setattr(dsv4_musa_ops, "tilelang_moe_fused_gate_musa", lambda *args, **kwargs: None)
    monkeypatch.setattr(dsv4_musa_ops, "_musa_graph_capture_enabled", lambda: True)

    hidden_states = torch.empty((2, 4), device=device, dtype=torch.float32)
    gating_output = torch.empty((2, 8), device=device, dtype=torch.float32)
    correction_bias = torch.zeros((8,), device=device, dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="during graph capture"):
        topk_module.biased_topk_tilelang_musa_impl(
            hidden_states=hidden_states,
            gating_output=gating_output,
            correction_bias=correction_bias,
            topk=2,
            renormalize=True,
            scoring_func="sqrtsoftplus",
            num_fused_shared_experts=0,
            routed_scaling_factor=None,
            num_token_non_padded=None,
            expert_location_dispatch_info=None,
            apply_routed_scaling_factor_on_output=False,
        )


@pytest.mark.parametrize("topk", [32, 64])
def test_hash_topk_musa_real_tilelang_large_topk_matches_reference_on_musa(topk: int) -> None:
    if os.environ.get("SGLANG_RUN_REAL_TILELANG_MUSA_TEST") != "1":
        pytest.skip("set SGLANG_RUN_REAL_TILELANG_MUSA_TEST=1 to run real TileLang MUSA kernel validation")
    pytest.importorskip("tilelang")
    device = get_musa_device()
    num_tokens = 4
    num_experts = 256
    num_tid = 128
    tid2eid = (torch.arange(num_tid * topk, device=device, dtype=torch.int64).reshape(num_tid, topk) * 19) % num_experts
    input_ids = torch.tensor([0, 11, 63, 127], device=device, dtype=torch.int64)
    router_logits = (
        torch.arange(num_tokens * num_experts, device=device, dtype=torch.float32).reshape(num_tokens, num_experts) / 257.0
    ) - 2.0

    result = MUSA_OPS._try_tilelang_hash_topk_musa(
        router_logits,
        input_ids,
        tid2eid,
        num_fused_shared_experts=0,
        routed_scaling_factor=1.0,
    )
    assert result is not None
    weights, ids = result

    routed_ids = tid2eid[input_ids].to(torch.int64)
    routed_scores = torch.nn.functional.softplus(router_logits).sqrt().gather(1, routed_ids.long())
    expected_weights = routed_scores / routed_scores.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    torch.testing.assert_close(ids.cpu(), routed_ids.cpu())
    torch.testing.assert_close(weights.cpu(), expected_weights.cpu(), rtol=1e-4, atol=1e-6)


def test_moe_align_block_size_handles_nopad_and_ep_shape_edges() -> None:
    import triton

    device = get_musa_device()
    moe_align_block_size = load_moe_align_block_size_for_test()
    num_experts = 256
    block_size = 64

    for topk_ids in (
        torch.empty((0, 6), device=device, dtype=torch.int64),
        torch.tensor([[1, 7, 13, 19, 23, 29]], device=device, dtype=torch.int64),
        (torch.arange(128 * 6, device=device, dtype=torch.int64).reshape(128, 6) * 17) % num_experts,
    ):
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, block_size, num_experts)
        num_tokens_post_padded_value = int(num_tokens_post_padded.item())
        assert num_tokens_post_padded_value % block_size == 0
        assert sorted_token_ids.dtype in (torch.int32, torch.int64)
        assert expert_ids.dtype in (torch.int32, torch.int64)
        if topk_ids.numel() == 0:
            assert num_tokens_post_padded_value == 0
            continue
        valid_blocks = triton.cdiv(num_tokens_post_padded_value, block_size)
        assert valid_blocks > 0
        assert int(expert_ids[:valid_blocks].cpu().min()) >= 0
        assert int(expert_ids[:valid_blocks].cpu().max()) < num_experts
        valid_sorted = sorted_token_ids[:num_tokens_post_padded_value]
        assert int(valid_sorted.cpu().min()) >= 0
        assert int(valid_sorted.cpu().max()) <= topk_ids.numel()

def test_mask_topk_ids_musa_masks_padded_rows() -> None:
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    got = mask_topk_ids_musa(topk_ids, torch.tensor(2, dtype=torch.int32))
    expected = torch.tensor([[1, 2], [3, 4], [-1, -1]], dtype=torch.int32)
    torch.testing.assert_close(got, expected)


def test_mask_topk_ids_musa_masks_all_rows_when_no_tokens() -> None:
    topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    got = mask_topk_ids_musa(topk_ids, torch.tensor(0, dtype=torch.int32))
    torch.testing.assert_close(got, torch.full_like(topk_ids, -1))


def test_mask_topk_ids_musa_keeps_all_rows_when_unpadded() -> None:
    topk_ids = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    got = mask_topk_ids_musa(topk_ids, torch.tensor(3, dtype=torch.int32))
    torch.testing.assert_close(got, torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32))


def test_topk_ids_logical_to_physical_static_musa_maps_int32_ids() -> None:
    device = get_musa_device()
    topk_ids = torch.tensor([[0, 2, -1], [3, 1, 0]], device=device, dtype=torch.int32)
    mapping = torch.tensor([10, 11, 12, 13], device=device, dtype=torch.int32)

    got = topk_ids_logical_to_physical_static_musa(topk_ids, mapping)

    expected = torch.tensor([[10, 12, -1], [13, 11, 10]], device=device, dtype=torch.int32)
    torch.testing.assert_close(got, expected)


def test_topk_ids_logical_to_physical_static_musa_maps_int64_ids() -> None:
    device = get_musa_device()
    topk_ids = torch.tensor([[0, 2, -1], [3, 1, 0]], device=device, dtype=torch.int64)
    mapping = torch.tensor(
        [2**33, 2**33 + 1, 2**33 + 2, 2**33 + 3],
        device=device,
        dtype=torch.int64,
    )

    got = topk_ids_logical_to_physical_static_musa(topk_ids, mapping)

    expected = torch.tensor(
        [[2**33, 2**33 + 2, -1], [2**33 + 3, 2**33 + 1, 2**33]],
        device=device,
        dtype=torch.int64,
    )
    torch.testing.assert_close(got, expected)
