import os
from typing import Literal

import torch

from ..kernels.cache_kernels import (
    _tilelang_pack_store_flashmla_cache_kernel,
    _tilelang_pack_store_flashmla_cache_decode_x4_fp32_kernel,
    _tilelang_pack_store_flashmla_cache_decode_x4_kernel,
    _tilelang_pack_store_indexer_cache_decode_x4_kernel,
    _tilelang_pack_store_indexer_cache_prefill_x8_kernel,
    _tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel,
    _tilelang_pack_store_flashmla_cache_prefill_subwarp16_kernel,
    _tilelang_store_flashmla_cache_kernel,
)
from .ops_common import _debug_musa_allow_torch_fallback, _debug_musa_torch_fallback

_FLASHMLA_PACK_IMPL_ENV = "SGLANG_DEEPSEEK_V4_MUSA_FLASHMLA_PACK_IMPL"
_FLASHMLA_DEFAULT_DECODE_IMPL = "decode_x4"
_FLASHMLA_DECODE_IMPLS = {
    "decode_x4",
    "decode_x4_fp32",
}
_FLASHMLA_PACK_IMPLS = {"auto"} | _FLASHMLA_DECODE_IMPLS
_FLASHMLA_PREFILL_SUBWARP16_TOKENS_PER_CTA = 8
_FLASHMLA_PREFILL_TILE_PARALLEL_TOKENS_PER_CTA = 8

def _flashmla_auto_decode_max_tokens() -> int:
    return 128

def _flashmla_auto_decode_impl() -> str:
    # Keep the auto decode path on the validated block-per-token x4 kernel.
    return _FLASHMLA_DEFAULT_DECODE_IMPL


def _flashmla_auto_decode_fp32_impl(page_size: int) -> str:
    return "decode_x4_fp32"


def _flashmla_trace_dispatch(message: str) -> None:
    return


def _flashmla_debug_sync(label: str) -> None:
    if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_FLASHMLA_SYNC_AFTER_LAUNCH") != "1":
        return
    try:
        torch.musa.synchronize()
    except Exception as exc:
        if "captur" in str(exc):
            _flashmla_trace_dispatch(
                f"sync_after_launch_skip_capture label={label} exc={type(exc).__name__}: {exc}"
            )
            return
        _flashmla_trace_dispatch(
            f"sync_after_launch_fail label={label} exc={type(exc).__name__}: {exc}"
        )
        raise
    _flashmla_trace_dispatch(f"sync_after_launch_ok label={label}")


def _flashmla_dispatch_context(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
    input_dtype: str,
    pack_impl: str,
) -> str:
    return (
        f"pack_impl={pack_impl} dtype={input_dtype} tokens={input.shape[0]} "
        f"input_shape={tuple(input.shape)} input_stride={tuple(input.stride())} "
        f"input_offset={int(input.storage_offset())} input_contig={input.is_contiguous()} "
        f"cache_shape={tuple(cache.shape)} page_size={page_size} "
        f"indices_shape={tuple(indices.shape)} indices_stride={tuple(indices.stride())} "
        f"indices_contig={indices.is_contiguous()}"
    )

def _flashmla_pack_impl() -> str:
    impl = os.environ.get(_FLASHMLA_PACK_IMPL_ENV, "auto").strip().lower()
    if impl in {"", "default"}:
        return "auto"
    if impl not in _FLASHMLA_PACK_IMPLS:
        raise ValueError(
            f"Unsupported {_FLASHMLA_PACK_IMPL_ENV}={impl!r}; "
            f"expected one of {sorted(_FLASHMLA_PACK_IMPLS)}"
        )
    return impl

def _compute_grouped_fp8_scale(value: torch.Tensor, quant_group_size: int) -> torch.Tensor:
    return value.float().reshape(value.shape[0], -1, quant_group_size).abs().amax(dim=-1).clamp(min=1e-4) / 448.0

def _pack_uint8_rows(values: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(v.shape[0], -1).to(torch.uint8) for v in values], dim=1)

def _flashmla_scale_pow2(value: torch.Tensor) -> torch.Tensor:
    """
    Compute 2^ceil(log2(clamp_min(value, 1e-4))) entirely on GPU.
    Uses bit manipulation for CUDA Graph compatibility.

    For 2^n in IEEE 754 float32:
    - exponent field (bits 23-30) = n + 127 (bias)
    - mantissa field (bits 0-22) = 0
    - sign bit (bit 31) = 0

    So the binary representation is (n + 127) << 23.
    """
    # Clamp, log2, ceil all on GPU
    clamped = torch.clamp_min(value, 1e-4)
    log2_val = torch.log2(clamped)
    ceil_val = torch.ceil(log2_val)

    # Convert to int32 for bit manipulation (ceil_val is always integer)
    n_int = ceil_val.to(torch.int32)

    # Construct IEEE 754 float32 representation of 2^n
    # exponent = n + 127 (float32 bias), shifted to bits 23-30
    exponent_bits = (n_int + 127) << 23

    # View as float32 to get the actual 2^n value
    scale = exponent_bits.view(torch.float32)

    return scale

def _pack_pow2_scales_to_ue8m0(scale_pow2_fp32: torch.Tensor) -> torch.Tensor:
    return (scale_pow2_fp32.contiguous().view(torch.int32) >> 23).to(torch.uint8)

def _pack_flashmla_cache_rows(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_nope, k_rope = input.split([448, 64], dim=-1)
    nope_tiles = k_nope.contiguous().view(-1, 7, 64)
    scale_pow2_fp32 = _flashmla_scale_pow2(nope_tiles.abs().amax(dim=-1).float() / 448.0)
    k_nope_fp8 = (nope_tiles.float() / scale_pow2_fp32.unsqueeze(-1)).to(torch.float8_e4m3fn).view(-1, 448)
    scale_k_nope_ue8m0 = _pack_pow2_scales_to_ue8m0(scale_pow2_fp32)
    return k_nope_fp8, k_rope.to(torch.bfloat16).contiguous(), scale_k_nope_ue8m0

def _try_tilelang_pack_store_indexer_cache_musa(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> bool:
    if input.device.type != "musa" or cache.device.type != "musa" or indices.device.type != "musa":
        return False
    if input.device != cache.device or indices.device != cache.device:
        raise NotImplementedError("DeepSeekV4 MUSA indexer fused_store_cache expects input/cache/indices on the same device")
    if input.dtype == torch.float32:
        input_dtype = "float32"
    elif input.dtype == torch.bfloat16:
        input_dtype = "bfloat16"
    else:
        return False
    if input.dim() != 2 or input.shape[-1] != 128 or input.stride(-1) != 1:
        return False
    if cache.dtype != torch.uint8 or cache.dim() != 2 or not cache.is_contiguous():
        return False
    if indices.dtype != torch.int32 or indices.dim() != 1 or indices.shape[0] != input.shape[0]:
        return False
    if not indices.is_contiguous():
        return False
    input_storage = input.as_strided((input.untyped_storage().nbytes() // input.element_size(),), (1,), storage_offset=0)
    # The x4/x8 paths avoid the shared-memory AllReduce used by the legacy
    # prefill kernel. Keep decode on x4; use half-warp/token x8 only for
    # larger prefill rows to avoid decode TPOT regressions.
    use_vectorized_pack = page_size > 0 and cache.shape[1] % 4 == 0
    use_prefill_x8 = use_vectorized_pack and input.shape[0] >= 128
    if use_prefill_x8:
        kernel = _tilelang_pack_store_indexer_cache_prefill_x8_kernel(
            input_dtype,
            cache.shape[1],
            page_size,
        )
        kernel_args = (
            input_storage,
            cache,
            cache.view(torch.uint32),
            indices,
            int(input.storage_offset()),
            int(input.stride(0)),
        )
    elif use_vectorized_pack:
        kernel = _tilelang_pack_store_indexer_cache_decode_x4_kernel(
            input_dtype,
            cache.shape[1],
            page_size,
        )
        kernel_args = (
            input_storage,
            cache,
            cache.view(torch.uint32),
            indices,
            int(input.storage_offset()),
            int(input.stride(0)),
        )
    else:
        _flashmla_trace_dispatch(
            "indexer_tilelang_miss reason=unsupported_cache_layout "
            f"input_dtype={input.dtype} input_shape={tuple(input.shape)} "
            f"cache_shape={tuple(cache.shape)} indices_shape={tuple(indices.shape)} "
            f"page_size={page_size}"
        )
        return False
    try:
        kernel(*kernel_args)
    except Exception as exc:
        _flashmla_trace_dispatch(
            "indexer_tilelang_miss "
            f"exc={type(exc).__name__}: {exc}; "
            f"input_dtype={input.dtype} input_shape={tuple(input.shape)} "
            f"cache_shape={tuple(cache.shape)} indices_shape={tuple(indices.shape)} "
            f"page_size={page_size}"
        )
        return False
    _flashmla_debug_sync("indexer_tilelang")
    return True

def _try_jit_pack_store_indexer_cache_musa(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> bool:
    if input.device.type != "musa" or cache.device.type != "musa" or indices.device.type != "musa":
        return False
    if input.device != cache.device or indices.device != cache.device:
        return False
    if input.dtype not in (torch.float32, torch.bfloat16):
        return False
    if input.dim() != 2 or input.shape[-1] != 128 or not input.is_contiguous():
        return False
    if cache.dtype != torch.uint8 or cache.dim() != 2 or not cache.is_contiguous():
        return False
    if indices.dtype != torch.int32 or indices.dim() != 1 or not indices.is_contiguous():
        return False
    if indices.shape[0] != input.shape[0]:
        return False
    if page_size <= 0 or (page_size & (page_size - 1)) != 0:
        return False
    if cache.shape[1] != page_size * (128 + 4):
        return False
    try:
        from sglang.jit_kernel.deepseek_v4 import _jit_fused_store_module

        module = _jit_fused_store_module(
            "indexer",
            input.dtype,
            indices.dtype,
            page_size,
        )
        module.run(input, cache, indices)
        _flashmla_debug_sync("indexer_jit")
    except Exception as exc:
        _flashmla_trace_dispatch(
            "indexer_jit_fallback_miss "
            f"exc={type(exc).__name__}: {exc}; "
            f"input_dtype={input.dtype} input_shape={tuple(input.shape)} "
            f"cache_shape={tuple(cache.shape)} indices_shape={tuple(indices.shape)} "
            f"page_size={page_size}"
        )
        return False
    return True

def _try_jit_pack_store_flashmla_cache_musa(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> bool:
    if input.device.type != "musa" or cache.device.type != "musa" or indices.device.type != "musa":
        return False
    if input.device != cache.device or indices.device != cache.device:
        return False
    if input.dtype not in (torch.float32, torch.bfloat16):
        return False
    if input.dim() != 2 or input.shape[-1] != 512 or not input.is_contiguous():
        return False
    if cache.dtype != torch.uint8 or cache.dim() != 2 or not cache.is_contiguous():
        return False
    if indices.dtype != torch.int32 or indices.dim() != 1 or not indices.is_contiguous():
        return False
    if indices.shape[0] != input.shape[0]:
        return False
    if page_size <= 0 or (page_size & (page_size - 1)) != 0:
        return False
    page_bytes = ((584 * int(page_size) + 575) // 576) * 576
    if cache.shape[1] != page_bytes:
        return False
    try:
        from sglang.jit_kernel.deepseek_v4 import _jit_fused_store_module

        module = _jit_fused_store_module(
            "flashmla",
            input.dtype,
            indices.dtype,
            page_size,
        )
        module.run(input, cache, indices)
        _flashmla_debug_sync("flashmla_jit")
    except Exception as exc:
        _flashmla_trace_dispatch(
            "flashmla_jit_fallback_miss "
            f"exc={type(exc).__name__}: {exc}; "
            f"input_dtype={input.dtype} input_shape={tuple(input.shape)} "
            f"cache_shape={tuple(cache.shape)} indices_shape={tuple(indices.shape)} "
            f"page_size={page_size}"
        )
        return False
    return True

def _tilelang_pack_store_flashmla_cache_musa(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> bool:
    # The bf16 warp-column path uses warp-level max reduction for each 64-wide
    # NoPE tile; keep the row-wise path as a compatibility fallback.
    if input.device.type != "musa" or cache.device.type != "musa" or indices.device.type != "musa":
        return False
    if input.device != cache.device or indices.device != cache.device:
        raise NotImplementedError("DeepSeekV4 MUSA flashmla fused_store_cache expects input/cache/indices on the same device")
    if input.dtype == torch.float32:
        input_dtype = "float32"
    elif input.dtype == torch.bfloat16:
        input_dtype = "bfloat16"
    else:
        return False
    if input.dim() != 2 or input.shape[-1] != 512:
        return False
    if cache.dtype != torch.uint8 or cache.dim() != 2 or not cache.is_contiguous():
        return False
    if indices.dtype != torch.int32 or indices.dim() != 1 or indices.shape[0] != input.shape[0]:
        return False
    if not indices.is_contiguous():
        return False
    if input.stride(-1) != 1:
        return False
    input_storage = input.as_strided((input.untyped_storage().nbytes() // input.element_size(),), (1,), storage_offset=0)
    # bf16 uses one row per lane. A 128-thread launch with blk_m=64 lowers to
    # threadIdx.x & 63 plus a half-block guard, wasting half the block on MUSA.
    threads = 64 if input_dtype == "bfloat16" else (256 if input.shape[0] >= 8192 else 128)
    blk_m = 64
    input_base_offset = int(input.storage_offset())
    input_row_stride = int(input.stride(0))
    rope_128_env = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_ENABLE_BF16_ROPE_128")
    rope_store_128_aligned = (
        input_dtype == "bfloat16"
        and (input_base_offset + 448) % 8 == 0
        and input_row_stride % 8 == 0
    )
    # Hardware supports 128-bit bf16 rope stores. On the current TileLang MUSA
    # image this helps large FlashMLA pack/store shapes, but it regresses small
    # 8K/32K-token shapes and older TileLang builds may fail to lower ldg128/stg128.
    rope_store_128 = rope_store_128_aligned and (
        rope_128_env == "1" or (rope_128_env != "0" and input.shape[0] >= 65536)
    )
    pack_impl = _flashmla_pack_impl()
    if input_dtype != "bfloat16" and pack_impl not in {"auto", "decode_x4_fp32"}:
        raise NotImplementedError(
            f"{_FLASHMLA_PACK_IMPL_ENV}={pack_impl!r} is only supported for bf16 FlashMLA input "
            "or the fp32 auto/decode_x4_fp32 paths"
        )
    def run_decode_impl(decode_impl: str) -> bool:
        if decode_impl == "decode_x4_fp32":
            if input_dtype != "float32":
                raise NotImplementedError(
                    f"FlashMLA decode impl {decode_impl!r} is only supported for fp32 input"
                )
            if not input.is_contiguous():
                raise NotImplementedError(
                    f"FlashMLA decode impl {decode_impl!r} requires contiguous input"
                )
            if page_size not in {2, 64}:
                raise NotImplementedError(
                    f"FlashMLA decode impl {decode_impl!r} is only enabled for page_size=2/64"
                )
            decode_x4_fp32_kernel = _tilelang_pack_store_flashmla_cache_decode_x4_fp32_kernel(
                cache.shape[1],
                page_size,
                input_base_offset,
                input_row_stride,
            )
            decode_x4_fp32_kernel(
                input_storage,
                cache,
                cache.view(torch.uint32),
                indices,
            )
            _flashmla_debug_sync("flashmla_decode_x4_fp32")
            return True
        if input_dtype != "bfloat16":
            raise NotImplementedError(
                f"FlashMLA decode impl {decode_impl!r} is only supported for bf16 input"
            )
        if not input.is_contiguous():
            raise NotImplementedError(
                f"FlashMLA decode impl {decode_impl!r} requires contiguous input"
            )
        use_i32_addresses = False
        dsa_compile_flags = "0"
        if decode_impl == "decode_x4":
            decode_x4_kernel = _tilelang_pack_store_flashmla_cache_decode_x4_kernel(
                cache.shape[1],
                page_size,
                input_base_offset,
                input_row_stride,
                use_i32_addresses=use_i32_addresses,
                dsa_compile_flags=dsa_compile_flags,
            )
            decode_x4_kernel(
                input_storage,
                cache,
                cache.view(torch.uint32),
                indices,
            )
            _flashmla_debug_sync(f"flashmla_{decode_impl}")
            return True
        raise ValueError(
            "Unsupported FlashMLA decode impl="
            f"{decode_impl!r}; expected one of {sorted(_FLASHMLA_DECODE_IMPLS)}"
        )

    if pack_impl in _FLASHMLA_DECODE_IMPLS:
        run_decode_impl(pack_impl)
        return True

    fp32_decode_supported = input_dtype == "float32" and page_size in {2, 64}
    dtype_decode_supported = input_dtype == "bfloat16" or fp32_decode_supported
    decode_reasons = []
    if pack_impl != "auto":
        decode_reasons.append(f"pack_impl={pack_impl}")
    if input_dtype == "float32" and page_size not in {2, 64}:
        decode_reasons.append(f"fp32_page_size={page_size}")
    elif not dtype_decode_supported:
        decode_reasons.append(f"dtype={input_dtype}")
    if not input.is_contiguous():
        decode_reasons.append(f"input_not_contiguous_stride={tuple(input.stride())}")
    decode_max_tokens = _flashmla_auto_decode_max_tokens()
    if input.shape[0] > decode_max_tokens:
        decode_reasons.append(f"tokens>{decode_max_tokens}")
    auto_decode = (
        pack_impl == "auto"
        and dtype_decode_supported
        and input.is_contiguous()
        and input.shape[0] <= decode_max_tokens
    )
    if auto_decode:
        decode_impl = (
            _flashmla_auto_decode_fp32_impl(page_size)
            if input_dtype == "float32"
            else _flashmla_auto_decode_impl()
        )
        run_decode_impl(decode_impl)
        return True
    prefill_tokens_per_cta = _FLASHMLA_PREFILL_SUBWARP16_TOKENS_PER_CTA
    use_prefill_subwarp16 = (
        pack_impl == "auto"
        and input.shape[0] >= 128
        and page_size > 0
        and cache.shape[1] % 4 == 0
        and input_dtype in {"bfloat16", "float32"}
    )
    if use_prefill_subwarp16:
        use_swa_subwarp16_ls = input_dtype == "bfloat16" and page_size == 256
        use_tile_parallel = not use_swa_subwarp16_ls
        prefill_compile_profile = "ls" if use_swa_subwarp16_ls else "dsa_full"
        if use_tile_parallel:
            # The 128-bit RoPE store candidate is kept for direct experiments,
            # but do not auto-dispatch it until it beats the scalar RoPE path
            # on large prefill shapes.
            rope_store_128 = False
            prefill_kernel = _tilelang_pack_store_flashmla_cache_prefill_tile_parallel_kernel(
                input_dtype,
                cache.shape[1],
                page_size,
                tokens_per_cta=_FLASHMLA_PREFILL_TILE_PARALLEL_TOKENS_PER_CTA,
                compile_profile=prefill_compile_profile,
                full_tiles=(input.shape[0] % _FLASHMLA_PREFILL_TILE_PARALLEL_TOKENS_PER_CTA == 0),
                rope_store_128=rope_store_128,
            )
        else:
            prefill_kernel = _tilelang_pack_store_flashmla_cache_prefill_subwarp16_kernel(
                input_dtype,
                cache.shape[1],
                page_size,
                tokens_per_cta=prefill_tokens_per_cta,
                compile_profile=prefill_compile_profile,
                full_tiles=(input.shape[0] % prefill_tokens_per_cta == 0),
            )
        try:
            prefill_kernel(
                input_storage,
                cache,
                cache.view(torch.uint32),
                indices,
                int(input_base_offset),
                int(input_row_stride),
            )
            _flashmla_debug_sync("flashmla_prefill_tile_parallel" if use_tile_parallel else "flashmla_prefill_subwarp16")
            return True
        except Exception:
            pass
    _flashmla_trace_dispatch(
        "flashmla_pack_auto_no_supported_tilelang_path "
        f"{_flashmla_dispatch_context(input, cache, indices, page_size, input_dtype, pack_impl)}"
    )
    return False

def _try_tilelang_store_flashmla_cache_musa(
    cache: torch.Tensor,
    indices: torch.Tensor,
    k_nope_fp8: torch.Tensor,
    k_rope_bf16: torch.Tensor,
    scale_k_nope_ue8m0: torch.Tensor,
    page_size: int,
) -> bool:
    if cache.device.type != "musa" or cache.dtype != torch.uint8 or not cache.is_contiguous():
        return False
    if indices.device != cache.device or indices.dtype != torch.int32 or indices.dim() != 1:
        return False
    if k_nope_fp8.device != cache.device or k_nope_fp8.dtype != torch.float8_e4m3fn:
        return False
    if k_rope_bf16.device != cache.device or k_rope_bf16.dtype != torch.bfloat16:
        return False
    if scale_k_nope_ue8m0.device != cache.device or scale_k_nope_ue8m0.dtype != torch.uint8:
        return False
    if k_nope_fp8.shape != (indices.shape[0], 448):
        return False
    if k_rope_bf16.shape != (indices.shape[0], 64):
        return False
    if scale_k_nope_ue8m0.shape != (indices.shape[0], 7):
        return False
    try:
        kernel = _tilelang_store_flashmla_cache_kernel(
            cache.shape[1],
            page_size,
            full_tiles=(indices.shape[0] % 64 == 0),
        )
        kernel(
            cache,
            cache.view(torch.bfloat16),
            indices.contiguous(),
            k_nope_fp8.contiguous().view(torch.uint8),
            k_rope_bf16.contiguous(),
            scale_k_nope_ue8m0.contiguous(),
        )
    except Exception:
        return False
    return True

def _store_flashmla_cache_rows(
    cache: torch.Tensor,
    indices: torch.Tensor,
    k_nope_fp8: torch.Tensor,
    k_rope_bf16: torch.Tensor,
    scale_k_nope_ue8m0: torch.Tensor,
    page_size: int,
) -> None:
    nope_dim = k_nope_fp8.shape[1]
    rope_dim = k_rope_bf16.shape[1]
    scale_dim = scale_k_nope_ue8m0.shape[1]
    buf_numel_per_page = cache.shape[1]

    buf_fp8_u8 = cache.view(torch.uint8).flatten()
    buf_bf16 = cache.view(torch.bfloat16).flatten()
    buf_scale = cache.view(torch.uint8).flatten()

    loc = indices.to(torch.int64).contiguous()
    loc_page_index = loc // page_size
    loc_token_offset_in_page = loc % page_size
    s_offset_nbytes_in_page = page_size * (nope_dim + rope_dim * 2)

    nope_offset = loc_page_index * buf_numel_per_page + loc_token_offset_in_page * (nope_dim + rope_dim * 2)
    rope_offset = loc_page_index * buf_numel_per_page // 2 + (
        loc_token_offset_in_page * (nope_dim + rope_dim * 2) + nope_dim
    ) // 2
    s_offset = loc_page_index * buf_numel_per_page + s_offset_nbytes_in_page + loc_token_offset_in_page * (scale_dim + 1)

    nope_cols = torch.arange(nope_dim, device=cache.device, dtype=torch.int64)
    rope_cols = torch.arange(rope_dim, device=cache.device, dtype=torch.int64)
    scale_cols = torch.arange(scale_dim, device=cache.device, dtype=torch.int64)

    buf_fp8_u8[(nope_offset[:, None] + nope_cols).reshape(-1)] = k_nope_fp8.contiguous().view(torch.uint8).reshape(-1)
    buf_bf16[(rope_offset[:, None] + rope_cols).reshape(-1)] = k_rope_bf16.contiguous().reshape(-1)
    buf_scale[(s_offset[:, None] + scale_cols).reshape(-1)] = scale_k_nope_ue8m0.contiguous().reshape(-1)

def _flashmla_no_copy_fallback_message(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    page_size: int,
) -> str:
    return (
        "DeepSeekV4 MUSA flashmla fused_store_cache TileLang fused pack/store path unsupported or failed; "
        "no copy fallback on MUSA. "
        f"input=device:{input.device},dtype:{input.dtype},shape:{tuple(input.shape)},stride:{tuple(input.stride())},"
        f"contiguous:{input.is_contiguous()},storage_offset:{input.storage_offset()}; "
        f"cache=device:{cache.device},dtype:{cache.dtype},shape:{tuple(cache.shape)},stride:{tuple(cache.stride())},"
        f"contiguous:{cache.is_contiguous()},storage_offset:{cache.storage_offset()}; "
        f"indices=device:{indices.device},dtype:{indices.dtype},shape:{tuple(indices.shape)},stride:{tuple(indices.stride())},"
        f"contiguous:{indices.is_contiguous()},storage_offset:{indices.storage_offset()}; "
        f"page_size={page_size}"
    )

def fused_store_cache_musa(
    input: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
    type: Literal["flashmla", "indexer"],
) -> None:
    if type == "indexer":
        if input.shape[-1] != 128:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA indexer fused_store_cache expects head_dim=128, got {input.shape[-1]}"
            )
        # The TileLang x4 indexer path can raise illegal-address on short
        # prefill C4 shapes before Python can fall back. Route small rows to
        # the existing graph-safe MUSA JIT store kernel first.
        if input.shape[0] < 128 and _try_jit_pack_store_indexer_cache_musa(input, cache, indices, page_size):
            return
        if _try_tilelang_pack_store_indexer_cache_musa(input, cache, indices, page_size):
            return
        if _try_jit_pack_store_indexer_cache_musa(input, cache, indices, page_size):
            return
        if input.device.type == "musa" or cache.device.type == "musa" or indices.device.type == "musa":
            message = (
                "DeepSeekV4 MUSA indexer fused_store_cache has no torch fallback by default; "
                "set SGLANG_MUSA_ALLOW_TORCH_FALLBACK=1 to enable the torch fallback. "
                f"input=device:{input.device},dtype:{input.dtype},shape:{tuple(input.shape)},"
                f"stride:{input.stride()},contiguous:{input.is_contiguous()}; "
                f"cache=device:{cache.device},dtype:{cache.dtype},shape:{tuple(cache.shape)},"
                f"stride:{cache.stride()},contiguous:{cache.is_contiguous()}; "
                f"indices=device:{indices.device},dtype:{indices.dtype},shape:{tuple(indices.shape)},"
                f"stride:{indices.stride()},contiguous:{indices.is_contiguous()}; page_size={page_size}"
            )
            if not _debug_musa_allow_torch_fallback():
                raise NotImplementedError(message)
            _debug_musa_torch_fallback(message)
        values = input.float()
        scale = _compute_grouped_fp8_scale(values, quant_group_size=128)
        quantized = torch.clamp(
            values.reshape(values.shape[0], -1, 128) / scale.unsqueeze(-1), -448.0, 448.0
        ).to(torch.float8_e4m3fn)
        packed = _pack_uint8_rows(
            [quantized.reshape(values.shape[0], 128).view(torch.uint8), scale.to(torch.float32).view(torch.uint8)]
        )
    elif type == "flashmla":
        if input.dtype not in (torch.bfloat16, torch.float32):
            raise NotImplementedError(
                f"DeepSeekV4 MUSA flashmla fused_store_cache expects bf16/fp32 input, got {input.dtype}"
            )
        if input.dim() != 2 or input.shape[-1] != 512:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA flashmla fused_store_cache expects [N,512] input, got {tuple(input.shape)}"
            )
        if cache.dtype != torch.uint8:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA flashmla fused_store_cache expects uint8 cache, got {cache.dtype}"
            )
        if cache.dim() != 2:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA flashmla fused_store_cache expects [num_pages,page_bytes] cache, got {tuple(cache.shape)}"
            )
        if not cache.is_contiguous():
            raise NotImplementedError(
                "DeepSeekV4 MUSA flashmla fused_store_cache expects contiguous cache layout"
            )
        if input.shape[0] < 128 and _try_jit_pack_store_flashmla_cache_musa(input, cache, indices, page_size):
            return
        if _tilelang_pack_store_flashmla_cache_musa(input, cache, indices, page_size):
            return
        if _try_jit_pack_store_flashmla_cache_musa(input, cache, indices, page_size):
            return
        if input.device.type == "musa" or cache.device.type == "musa" or indices.device.type == "musa":
            message = _flashmla_no_copy_fallback_message(input, cache, indices, page_size)
            if not _debug_musa_allow_torch_fallback():
                raise NotImplementedError(message)
            _debug_musa_torch_fallback(message)
        k_nope_fp8, k_rope_bf16, scale_k_nope_ue8m0 = _pack_flashmla_cache_rows(input.contiguous())
        if _try_tilelang_store_flashmla_cache_musa(
            cache,
            indices,
            k_nope_fp8,
            k_rope_bf16,
            scale_k_nope_ue8m0,
            page_size,
        ):
            return
        _store_flashmla_cache_rows(
            cache,
            indices.contiguous(),
            k_nope_fp8,
            k_rope_bf16,
            scale_k_nope_ue8m0,
            page_size,
        )
        return
    else:
        raise ValueError(f"Unsupported DeepSeekV4 MUSA fused_store_cache type={type!r}")

    page_idx = torch.div(indices.to(torch.int64), page_size, rounding_mode="floor")
    offset = torch.remainder(indices.to(torch.int64), page_size)
    row_bytes = packed.shape[1]
    columns = offset.view(-1, 1) * row_bytes + torch.arange(row_bytes, device=packed.device, dtype=torch.int64).view(1, -1)
    cache[page_idx.view(-1, 1), columns] = packed

__all__ = [
    '_compute_grouped_fp8_scale',
    '_pack_uint8_rows',
    '_flashmla_scale_pow2',
    '_pack_pow2_scales_to_ue8m0',
    '_pack_flashmla_cache_rows',
    '_try_tilelang_pack_store_indexer_cache_musa',
    '_try_jit_pack_store_indexer_cache_musa',
    '_try_jit_pack_store_flashmla_cache_musa',
    '_tilelang_pack_store_flashmla_cache_musa',
    '_try_tilelang_store_flashmla_cache_musa',
    '_store_flashmla_cache_rows',
    'fused_store_cache_musa',
]
