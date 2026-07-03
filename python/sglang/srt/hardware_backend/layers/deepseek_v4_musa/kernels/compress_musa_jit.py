from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch


@lru_cache(maxsize=1)
def _load_c4_reduce_module():
    from torch_musa.utils.musa_extension import load

    source = Path(__file__).resolve().parents[1] / "csrc" / "c4_reduce_musa_jit.mu"
    return load(
        name="dsv4_c4_reduce_musa_jit_v3",
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_musa_cflags=["-O3"],
        verbose=False,
    )


def try_c4_page_reduce_musa_jit(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    extra_data: torch.Tensor,
    compress_rows: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
) -> tuple[bool, Optional[str]]:
    if head_dim not in {128, 512}:
        return False, "head_dim"
    tensors = (kv_score_buffer, kv_score_input, ape, extra_data, compress_rows, out)
    if any(t.device.type != "musa" for t in tensors):
        return False, "device"
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32:
        return False, "kv dtype"
    if ape.dtype != torch.float32 or out.dtype != torch.float32:
        return False, "ape/out dtype"
    if extra_data.dtype != torch.int32 or compress_rows.dtype != torch.int32:
        return False, "metadata dtype"
    width = head_dim * 4
    if kv_score_buffer.dim() != 3 or kv_score_buffer.shape[1:] != (4, width):
        return False, "kv_score_buffer shape"
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != width:
        return False, "kv_score_input shape"
    if ape.shape != (8, head_dim) or out.dim() != 2 or out.shape[1] != head_dim:
        return False, "ape/out shape"
    if extra_data.dim() != 2 or extra_data.shape[1] < 2:
        return False, "extra_data shape"
    if compress_rows.dim() != 2 or compress_rows.shape[1] != 4:
        return False, "compress_rows shape"

    # The kernel vector-loads four contiguous floats across the last dimension.
    if kv_score_buffer.stride(1) != width or kv_score_buffer.stride(2) != 1:
        return False, "kv_score_buffer stride"
    if kv_score_input.stride(1) != 1 or ape.stride(1) != 1 or out.stride(1) != 1:
        return False, "last dim stride"
    if extra_data.stride(1) != 1 or compress_rows.stride(1) != 1:
        return False, "metadata stride"

    try:
        _load_c4_reduce_module().c4_page_reduce_float(
            kv_score_buffer,
            kv_score_input,
            ape,
            extra_data,
            compress_rows,
            out,
        )
    except Exception as exc:
        return False, f"musa jit launch: {exc}"
    return True, None


def _check_c4_page_common(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    extra_data: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
) -> tuple[bool, Optional[str]]:
    if head_dim not in {128, 512}:
        return False, "head_dim"
    if any(t.device.type != "musa" for t in (kv_score_buffer, kv_score_input, ape, extra_data, out)):
        return False, "device"
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32:
        return False, "kv dtype"
    if ape.dtype != torch.float32 or out.dtype != torch.float32:
        return False, "ape/out dtype"
    if extra_data.dtype != torch.int32:
        return False, "extra_data dtype"
    width = head_dim * 4
    if kv_score_buffer.dim() != 3 or kv_score_buffer.shape[1:] != (4, width):
        return False, "kv_score_buffer shape"
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != width:
        return False, "kv_score_input shape"
    if ape.shape != (8, head_dim) or out.dim() != 2 or out.shape[1] != head_dim:
        return False, "ape/out shape"
    if extra_data.dim() != 2 or extra_data.shape[1] < 4:
        return False, "extra_data shape"
    if kv_score_buffer.stride(1) != width or kv_score_buffer.stride(2) != 1:
        return False, "kv_score_buffer stride"
    if kv_score_input.stride(1) != 1 or ape.stride(1) != 1 or out.stride(1) != 1:
        return False, "last dim stride"
    if extra_data.stride(1) != 1:
        return False, "extra_data stride"
    return True, None


def try_c4_page_prefill_musa_jit(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    extra_data: torch.Tensor,
    compress_rows: torch.Tensor,
    write_rows: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
) -> tuple[bool, Optional[str]]:
    ok, reason = _check_c4_page_common(kv_score_buffer, kv_score_input, ape, extra_data, out, head_dim)
    if not ok:
        return False, reason
    if indices.device.type != "musa" or compress_rows.device.type != "musa" or write_rows.device.type != "musa":
        return False, "metadata device"
    if indices.dtype != torch.int32 or compress_rows.dtype != torch.int32 or write_rows.dtype != torch.int32:
        return False, "metadata dtype"
    if indices.dim() != 1:
        return False, "indices shape"
    if compress_rows.dim() != 2 or compress_rows.shape[1] != 4:
        return False, "compress_rows shape"
    if write_rows.dim() != 2 or write_rows.shape[1] != 4:
        return False, "write_rows shape"
    if compress_rows.stride(1) != 1 or write_rows.stride(1) != 1:
        return False, "rows stride"

    try:
        _load_c4_reduce_module().c4_page_prefill_float(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            extra_data,
            compress_rows,
            write_rows,
            out,
        )
    except Exception as exc:
        return False, f"musa jit launch: {exc}"
    return True, None


__all__ = ["try_c4_page_prefill_musa_jit", "try_c4_page_reduce_musa_jit"]
