from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_musa_ci
from sglang.srt.hardware_backend.layers.deepseek_v4_musa.ops import cache_ops as CACHE_OPS

from ..utils import MUSA_OPS, get_musa_device

register_musa_ci(
    est_time=1,
    suite="stage-a-test-1-gpu-musa-smoke",
    disabled="DeepSeek V4 MUSA operator test is opt-in outside smoke CI",
)


fused_store_cache_musa = MUSA_OPS.fused_store_cache_musa


def _indexer_store_ref(
    input: torch.Tensor,
    indices: torch.Tensor,
    num_pages: int,
    page_size: int,
) -> torch.Tensor:
    head_dim = 128
    row_bytes = head_dim + 4
    cache = torch.full(
        (num_pages, page_size * row_bytes),
        0xCD,
        dtype=torch.uint8,
        device=input.device,
    )
    values = input.float()
    scale = values.abs().amax(dim=-1).clamp(min=1.0e-4) / 448.0
    quantized = torch.clamp(values / scale.unsqueeze(-1), -448.0, 448.0).to(torch.float8_e4m3fn)
    packed = torch.cat(
        [
            quantized.view(torch.uint8).reshape(input.shape[0], head_dim),
            scale.contiguous().view(torch.uint8).reshape(input.shape[0], 4),
        ],
        dim=-1,
    )
    for row, loc in enumerate(indices.tolist()):
        page = loc // page_size
        offset = loc % page_size
        start = offset * row_bytes
        cache[page, start : start + row_bytes] = packed[row]
    return cache


def _cache_row(cache: torch.Tensor, loc: int, page_size: int) -> torch.Tensor:
    row_bytes = 128 + 4
    page = loc // page_size
    offset = loc % page_size
    start = offset * row_bytes
    return cache[page, start : start + row_bytes]


def _flashmla_page_bytes(page_size: int) -> int:
    bytes_per_token = 448 + 64 * 2 + 8
    return ((page_size * bytes_per_token + 575) // 576) * 576


def _flashmla_input_storage(input: torch.Tensor) -> torch.Tensor:
    return input.as_strided(
        (input.untyped_storage().nbytes() // input.element_size(),),
        (1,),
        storage_offset=0,
    )


def _flashmla_direct_tile_parallel_store(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> None:
    input_dtype = "float32" if input.dtype == torch.float32 else "bfloat16"
    kernel = CACHE_OPS._tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel(
        input_dtype,
        cache.shape[1],
        page_size,
        tokens_per_cta=CACHE_OPS._FLASHMLA_PREFILL_TILE_PARALLEL_TOKENS_PER_CTA,
        compile_profile="dsa_full",
        full_tiles=(input.shape[0] % CACHE_OPS._FLASHMLA_PREFILL_TILE_PARALLEL_TOKENS_PER_CTA == 0),
    )
    kernel(
        _flashmla_input_storage(input),
        cache,
        cache.view(torch.uint32),
        indices,
        int(input.storage_offset()),
        int(input.stride(0)),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_fused_store_cache_prefill_ignores_invalid_indices(dtype: torch.dtype) -> None:
    device = get_musa_device()
    torch.manual_seed(20260603)
    num_tokens = 129
    page_size = 64
    num_pages = 2
    input = (torch.randn((num_tokens, 128), device=device, dtype=torch.float32) * 3.0).to(dtype)
    capacity = num_pages * page_size
    indices = torch.empty((num_tokens,), device=device, dtype=torch.int32)
    indices[0::3] = -1
    indices[1::3] = capacity
    indices[2::3] = capacity + 17
    cache = torch.full(
        (num_pages, page_size * (128 + 4)),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )

    fused_store_cache_musa(input, cache, indices, page_size=page_size, type="indexer")
    torch.musa.synchronize()

    assert torch.equal(cache, torch.full_like(cache, 0xCD))


@pytest.mark.parametrize("page_size", [2, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flashmla_fused_store_cache_prefill_ignores_invalid_indices(
    page_size: int,
    dtype: torch.dtype,
) -> None:
    device = get_musa_device()
    torch.manual_seed(20260604 + page_size)
    num_tokens = 129
    num_pages = 4
    input = (torch.randn((num_tokens, 512), device=device, dtype=torch.float32) * 3.0).to(dtype)
    capacity = num_pages * page_size
    indices = torch.empty((num_tokens,), device=device, dtype=torch.int32)
    indices[0::3] = -1
    indices[1::3] = capacity
    indices[2::3] = capacity + 17
    page_bytes = page_size * (448 + 64 * 2) + page_size * 8
    cache = torch.full(
        (num_pages, page_bytes),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )

    fused_store_cache_musa(input, cache, indices, page_size=page_size, type="flashmla")
    torch.musa.synchronize()

    assert torch.equal(cache, torch.full_like(cache, 0xCD))


@pytest.mark.parametrize("page_size", [2, 64, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flashmla_fused_store_cache_prefill_dispatches_tile_parallel_x8(
    page_size: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = get_musa_device()
    torch.manual_seed(20260605 + page_size)
    num_tokens = 256
    num_pages = (num_tokens + page_size - 1) // page_size + 16
    input = (torch.randn((num_tokens, 512), device=device, dtype=torch.float32) * 0.2).to(dtype)
    indices = (torch.arange(num_tokens - 1, -1, -1, device=device, dtype=torch.int32) + 3).contiguous()
    cache = torch.empty(
        (num_pages, _flashmla_page_bytes(page_size)),
        dtype=torch.uint8,
        device=device,
    )
    calls: list[dict[str, object]] = []
    original_factory = CACHE_OPS._tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel

    def recording_factory(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(
        CACHE_OPS,
        "_tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel",
        recording_factory,
    )

    fused_store_cache_musa(input, cache, indices, page_size=page_size, type="flashmla")
    torch.musa.synchronize()

    assert len(calls) == 1
    call = calls[0]
    input_dtype = "float32" if dtype == torch.float32 else "bfloat16"
    assert call["args"][:3] == (input_dtype, cache.shape[1], page_size)
    assert call["kwargs"]["tokens_per_cta"] == 8
    assert call["kwargs"]["compile_profile"] == "dsa_full"
    assert call["kwargs"]["full_tiles"] is True


@pytest.mark.parametrize("page_size", [2, 64, 256])
@pytest.mark.parametrize("num_tokens", [256, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flashmla_fused_store_cache_prefill_tile_parallel_x8_matches_direct_kernel(
    page_size: int,
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    device = get_musa_device()
    torch.manual_seed(20260606 + page_size + num_tokens)
    num_pages = (num_tokens + page_size - 1) // page_size + 16
    input = (torch.randn((num_tokens, 512), device=device, dtype=torch.float32) * 0.2).to(dtype)
    indices = (torch.arange(num_tokens - 1, -1, -1, device=device, dtype=torch.int32) + 3).contiguous()
    page_bytes = _flashmla_page_bytes(page_size)
    actual = torch.full((num_pages, page_bytes), 0xCD, dtype=torch.uint8, device=device)
    expected = torch.full_like(actual, 0xCD)

    fused_store_cache_musa(input, actual, indices, page_size=page_size, type="flashmla")
    _flashmla_direct_tile_parallel_store(input, expected, indices, page_size)
    torch.musa.synchronize()

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_tokens", [*range(1, 17), 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_fused_store_cache_decode_m1_m16(num_tokens: int, dtype: torch.dtype) -> None:
    device = get_musa_device()
    torch.manual_seed(20260521 + num_tokens)
    page_size = 64
    num_pages = 2
    input = (torch.randn((num_tokens, 128), device=device, dtype=torch.float32) * 3.0).to(dtype)
    # Use distinct non-contiguous locations to cover page/offset calculation.
    indices = ((torch.arange(num_tokens, device=device, dtype=torch.int32) * 7 + 3) % (num_pages * page_size)).contiguous()
    cache = torch.full(
        (num_pages, page_size * (128 + 4)),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )
    expected = _indexer_store_ref(input, indices, num_pages, page_size)

    fused_store_cache_musa(input, cache, indices, page_size=page_size, type="indexer")

    torch.testing.assert_close(cache.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 16, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_fused_store_cache_is_repeat_deterministic(
    num_tokens: int,
    dtype: torch.dtype,
) -> None:
    device = get_musa_device()
    torch.manual_seed(20261325 + num_tokens)
    page_size = 64
    num_pages = 4
    input = (torch.randn((num_tokens, 128), device=device, dtype=torch.float32) * 3.0).to(dtype)
    indices = (
        (torch.arange(num_tokens, device=device, dtype=torch.int32) * 11 + 5)
        % (num_pages * page_size)
    ).contiguous()

    def run_once() -> torch.Tensor:
        cache = torch.full(
            (num_pages, page_size * (128 + 4)),
            0xCD,
            dtype=torch.uint8,
            device=device,
        )
        fused_store_cache_musa(input, cache, indices, page_size=page_size, type="indexer")
        torch.musa.synchronize()
        return cache

    expected = run_once()
    for repeat_idx in range(20):
        actual = run_once()
        assert torch.equal(actual, expected), (
            f"fused_store_cache_musa is not repeat-deterministic at "
            f"repeat={repeat_idx}, num_tokens={num_tokens}, dtype={dtype}"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_indexer_fused_store_cache_probe_row_is_batch_shape_invariant(
    dtype: torch.dtype,
) -> None:
    device = get_musa_device()
    torch.manual_seed(20261425)
    page_size = 64
    num_pages = 4
    num_tokens = 128
    probe = (torch.randn((128,), device=device, dtype=torch.float32) * 3.0).to(dtype)

    baseline_input = probe.view(1, 128).contiguous()
    baseline_index = torch.tensor([5], device=device, dtype=torch.int32)
    baseline_cache = torch.full(
        (num_pages, page_size * (128 + 4)),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )
    fused_store_cache_musa(
        baseline_input,
        baseline_cache,
        baseline_index,
        page_size=page_size,
        type="indexer",
    )
    torch.musa.synchronize()
    expected_row = _cache_row(baseline_cache, int(baseline_index.item()), page_size).clone()

    input = (torch.randn((num_tokens, 128), device=device, dtype=torch.float32) * 3.0).to(dtype)
    indices = (torch.arange(num_tokens, device=device, dtype=torch.int32) + 128).contiguous()
    probe_positions = [0, 17, num_tokens - 1]
    probe_locs = [5, 77, 96]
    for pos, loc in zip(probe_positions, probe_locs, strict=True):
        input[pos].copy_(probe)
        indices[pos] = loc

    cache = torch.full(
        (num_pages, page_size * (128 + 4)),
        0xCD,
        dtype=torch.uint8,
        device=device,
    )
    fused_store_cache_musa(input, cache, indices, page_size=page_size, type="indexer")
    torch.musa.synchronize()
    for pos, loc in zip(probe_positions, probe_locs, strict=True):
        assert torch.equal(_cache_row(cache, loc, page_size), expected_row), (
            f"cache row differs for embedded probe pos={pos}, loc={loc}, dtype={dtype}"
        )
