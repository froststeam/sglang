import importlib
import logging
import os
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F

from ..kernels.compress_kernels import _tilelang_compress_forward_ratio128_decode_flat_kernel, _tilelang_compress_forward_ratio128_decode_flat_parallel_kernel, _tilelang_compress_forward_ratio128_decode_kernel, _tilelang_compress_forward_ratio128_decode_parallel_kernel, _tilelang_compress_forward_ratio4_decode_flat_kernel, _tilelang_compress_forward_ratio4_decode_kernel, _tilelang_compress_forward_ratio4_decode_page_kernel, _tilelang_compress_prefill_zero_kernel, _tilelang_compress_ratio128_prefill_reduce_kernel, _tilelang_compress_ratio128_prefill_reduce_parallel_kernel, _tilelang_compress_ratio128_prefill_write_kernel, _tilelang_compress_ratio128_prefill_write_vec4_kernel, _tilelang_compress_ratio4_prefill_flat_reduce_kernel, _tilelang_compress_ratio4_prefill_flat_write_kernel, _tilelang_compress_ratio4_prefill_flat_write_vec4_kernel, _tilelang_compress_ratio4_prefill_page_reduce_kernel, _tilelang_compress_ratio4_prefill_page_reduce_cached_kernel, _tilelang_compress_ratio4_prefill_page_write_kernel, _tilelang_compress_ratio4_prefill_page_write_vec4_kernel, _tilelang_compress_ratio4_prefill_reduce_kernel, _tilelang_compress_ratio4_prefill_write_kernel, _tilelang_compress_ratio4_prefill_write_vec4_kernel
from ..kernels.compress_musa_jit import try_c4_page_prefill_musa_jit
from .ops_common import _debug_musa_allow_torch_fallback, _debug_musa_torch_fallback, _has_musa_tensor, _is_musa_tensor, _musa_graph_capture_enabled

_MetadataPlan = Union[torch.Tensor, tuple[int, torch.Tensor], object]

def _is_last_dim_contiguous_2d(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride(-1) == 1

def _is_strided_1d(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 1 and tensor.stride(0) > 0

def _is_strided_2d(tensor: torch.Tensor) -> bool:
    return tensor.dim() == 2 and tensor.stride(0) > 0 and tensor.stride(1) > 0

def _compress_threads(kind: str, default: int = 128) -> int:
    names = (
        f"SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_{kind}_THREADS",
        "SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_THREADS",
    )
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            threads = int(value)
        except ValueError:
            return default
        if threads in (64, 128, 256, 512):
            return threads
    return default

def _compress_vector_write_enabled() -> bool:
    return os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_VECTOR_WRITE", "1") != "0"

def _compress_vector_write_min_rows() -> int:
    value = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_VECTOR_WRITE_MIN_ROWS")
    if value is None:
        return 16
    try:
        return max(0, int(value))
    except ValueError:
        return 16

def _compress_c4_prefill_musa_jit_enabled() -> bool:
    return os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C4_PREFILL_MUSA_JIT_REDUCE", "1") != "0"

def _compress_raise_prefill_miss() -> bool:
    return os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_RAISE_PREFILL_MISS") == "1"

def _compress_c128_parallel_reduce_enabled() -> bool:
    return os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_REDUCE") == "1"

def _compress_c128_decode_parallel_reduce_enabled() -> bool:
    value = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_DECODE_PARALLEL_REDUCE")
    if value is None:
        value = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_REDUCE", "1")
    return value not in {"0", "false", "False"}

def _compress_c128_parallel_final_merge(kind: Literal["decode", "prefill"]) -> str:
    mode = os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_C128_PARALLEL_FINAL", "shared").lower()
    if mode == "auto":
        # Shared merge is currently both faster on serving shapes and cleaner at
        # process teardown; keep warp merge as an explicit tuning/debug opt-in.
        return "shared"
    if mode in ("shared", "warp"):
        return mode
    return "shared"

def _compress_tilelang_zero_enabled() -> bool:
    return os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_TILELANG_ZERO") == "1"

def _try_tilelang_compress_forward_ratio4_decode_musa(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> tuple[bool, str | None]:
    def tensor_summary(name: str, tensor: Optional[torch.Tensor]) -> str:
        if tensor is None:
            return f"{name}=None"
        return (
            f"{name}=device:{tensor.device},dtype:{tensor.dtype},shape:{tuple(tensor.shape)},"
            f"stride:{tensor.stride()},contiguous:{tensor.is_contiguous()}"
        )

    def fail(reason: str) -> tuple[bool, str]:
        return False, (
            f"{reason}; "
            f"{tensor_summary('kv_score_buffer', kv_score_buffer)}; "
            f"{tensor_summary('kv_score_input', kv_score_input)}; "
            f"{tensor_summary('ape', ape)}; "
            f"{tensor_summary('indices', indices)}; "
            f"{tensor_summary('seq_lens', seq_lens)}; "
            f"{tensor_summary('extra_data', extra_data)}; "
            f"{tensor_summary('out', out)}; "
            f"head_dim={head_dim}; compress_ratio=4"
        )

    if kv_score_buffer.device.type != "musa" or kv_score_input.device.type != "musa":
        return fail("kv_score_buffer/kv_score_input device")
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32 or out.dtype != torch.float32:
        return fail("kv_score_buffer/kv_score_input/out dtype")
    if ape.device != kv_score_input.device or ape.dtype != torch.float32:
        return fail("ape device/dtype")
    if indices.device != kv_score_input.device or seq_lens.device != kv_score_input.device:
        return fail("indices/seq_lens device")
    if indices.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return fail("indices/seq_lens dtype")
    if extra_data is not None and (extra_data.device != kv_score_input.device or extra_data.dtype != torch.int32):
        return fail("extra_data device/dtype")
    if not kv_score_buffer.is_contiguous():
        return fail("kv_score_buffer contiguity")
    if not _is_last_dim_contiguous_2d(kv_score_input) or not _is_last_dim_contiguous_2d(out):
        return fail("kv_score_input/out layout")
    if not _is_last_dim_contiguous_2d(ape) or not _is_strided_1d(indices) or not _is_strided_1d(seq_lens):
        return fail("ape/indices/seq_lens layout")
    if extra_data is not None and not _is_strided_2d(extra_data):
        return fail("extra_data layout")
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 4:
        return fail("kv_score_input shape")
    if ape.shape != (8, head_dim) or out.shape != (kv_score_input.shape[0], head_dim):
        return fail("ape/out shape")
    if indices.shape != (kv_score_input.shape[0],) or seq_lens.shape != (kv_score_input.shape[0],):
        return fail("indices/seq_lens shape")
    if extra_data is not None and (
        extra_data.dim() != 2
        or extra_data.shape[0] < kv_score_input.shape[0]
        or extra_data.shape[1] not in (1, 4)
    ):
        return fail("extra_data shape")

    try:
        if extra_data is not None:
            if kv_score_buffer.dim() != 3 or kv_score_buffer.shape[1:] != (4, head_dim * 4):
                return fail("kv_score_buffer page4 shape")
            kernel = _tilelang_compress_forward_ratio4_decode_page_kernel(
                head_dim, extra_data.shape[1], _compress_threads("C4_DECODE")
            )
            kernel(
                kv_score_buffer,
                kv_score_input,
                ape,
                indices,
                seq_lens,
                extra_data,
                out,
            )
            return True, None
        if kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (8, head_dim * 4):
            kernel = _tilelang_compress_forward_ratio4_decode_kernel(head_dim, _compress_threads("C4_DECODE"))
        elif kv_score_buffer.dim() == 2 and kv_score_buffer.shape[1] == head_dim * 4:
            kernel = _tilelang_compress_forward_ratio4_decode_flat_kernel(head_dim, _compress_threads("C4_DECODE"))
        else:
            return fail("kv_score_buffer shape")
        kernel(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            out,
        )
    except Exception as exc:
        return fail(f"kernel exception {type(exc).__name__}: {exc}")
    return True, None

def _try_tilelang_compress_forward_ratio128_decode_musa(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
) -> tuple[bool, str | None]:
    def tensor_summary(name: str, tensor: torch.Tensor) -> str:
        return (
            f"{name}=device:{tensor.device},dtype:{tensor.dtype},shape:{tuple(tensor.shape)},"
            f"stride:{tensor.stride()},contiguous:{tensor.is_contiguous()}"
        )

    def fail(reason: str) -> tuple[bool, str]:
        return False, (
            f"guard_reason={reason}; "
            f"function_path=sglang.srt.hardware_backend.layers.deepseek_v4_musa_ops.compress_forward_musa.decode.ratio128; "
            f"plan_type=CompressorDecodePlan; "
            f"{tensor_summary('kv_score_buffer', kv_score_buffer)}; "
            f"{tensor_summary('kv_score_input', kv_score_input)}; "
            f"{tensor_summary('ape', ape)}; "
            f"{tensor_summary('indices', indices)}; "
            f"{tensor_summary('seq_lens', seq_lens)}; "
            f"{tensor_summary('out', out)}; "
            f"head_dim={head_dim}; compress_ratio=128"
        )

    if kv_score_buffer.device.type != "musa" or kv_score_input.device.type != "musa":
        return fail("kv_score_buffer/kv_score_input device")
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32 or out.dtype != torch.float32:
        return fail("kv_score_buffer/kv_score_input/out dtype")
    if ape.device != kv_score_input.device or ape.dtype != torch.float32:
        return fail("ape device/dtype")
    if indices.device != kv_score_input.device or seq_lens.device != kv_score_input.device:
        return fail("indices/seq_lens device")
    if indices.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        return fail("indices/seq_lens dtype")
    if not kv_score_buffer.is_contiguous():
        return fail("kv_score_buffer contiguity")
    if not _is_last_dim_contiguous_2d(kv_score_input) or not _is_last_dim_contiguous_2d(out):
        return fail("kv_score_input/out layout")
    if not _is_last_dim_contiguous_2d(ape) or not _is_strided_1d(indices) or not _is_strided_1d(seq_lens):
        return fail("ape/indices/seq_lens layout")
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 2:
        return fail("kv_score_input shape")
    if ape.shape != (128, head_dim) or out.shape != (kv_score_input.shape[0], head_dim):
        return fail("ape/out shape")
    if indices.shape != (kv_score_input.shape[0],) or seq_lens.shape != (kv_score_input.shape[0],):
        return fail("indices/seq_lens shape")
    try:
        if kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (128, head_dim * 2):
            if _compress_c128_decode_parallel_reduce_enabled() and head_dim % 64 == 0:
                kernel = _tilelang_compress_forward_ratio128_decode_parallel_kernel(
                    head_dim,
                    _compress_c128_parallel_final_merge("decode"),
                )
            else:
                kernel = _tilelang_compress_forward_ratio128_decode_kernel(head_dim, _compress_threads("C128_DECODE"))
        elif kv_score_buffer.dim() == 2 and kv_score_buffer.shape[1] == head_dim * 2:
            if _compress_c128_decode_parallel_reduce_enabled() and head_dim % 64 == 0:
                kernel = _tilelang_compress_forward_ratio128_decode_flat_parallel_kernel(
                    head_dim,
                    _compress_c128_parallel_final_merge("decode"),
                )
            else:
                kernel = _tilelang_compress_forward_ratio128_decode_flat_kernel(head_dim, _compress_threads("C128_DECODE"))
        else:
            return fail("kv_score_buffer shape")
        kernel(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            out,
        )
    except Exception as exc:
        return fail(f"kernel exception {type(exc).__name__}: {exc}")
    return True, None

def _compress_softmax_reduce(
    kv: torch.Tensor,
    score: torch.Tensor,
    score_bias: torch.Tensor,
    debug_info: Optional[dict] = None,
) -> torch.Tensor:
    """Compute softmax-weighted reduction of kv using scores and bias.

    Args:
        kv: [window_size, head_dim] tensor
        score: [window_size, head_dim] tensor
        score_bias: [window_size, head_dim] tensor (ape)
        debug_info: Optional dict with seq_len, compress_ratio, row for debugging

    Returns:
        [head_dim] tensor
    """
    logits = score.float() + score_bias.float()

    logits = torch.clamp(logits, min=-1e9)
    weights = torch.softmax(logits, dim=0)
    weights = torch.where(torch.isnan(weights), torch.ones_like(weights) / weights.shape[0], weights)
    result = torch.sum(kv.float() * weights, dim=0)
    return torch.where(torch.isnan(result) | torch.isinf(result), torch.zeros_like(result), result)

def _compress_forward_ratio4_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if _musa_graph_capture_enabled() and _has_musa_tensor(
        kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out
    ):
        raise NotImplementedError("DeepSeekV4 MUSA ratio4 decode has no torch fallback during graph capture")

    # Handle 2D buffer by reshaping to 3D view for RingBuffer mode
    # For paged (extra_data provided): buffer should be [N, 4, head_dim*4]
    # For non-paged (extra_data None): buffer should be viewed as [N, 8, head_dim*4]
    #    where each index in the original 2D [N, head_dim*4] corresponds to 8 slots
    paged = extra_data is not None
    element_size = head_dim * 4  # 512 for head_dim=128

    if kv_score_buffer.dim() == 2 and kv_score_buffer.shape[-1] == element_size:
        # 2D buffer: reshape to treat consecutive rows as slots for RingBuffer
        # For RingBuffer mode, each index needs 8 slots of element_size
        num_rows = kv_score_buffer.shape[0]
        if paged:
            # For paged mode with 2D, reshape to [num_rows/4, 4, element_size]
            if num_rows % 4 != 0:
                raise ValueError(
                    f"DeepSeekV4 MUSA c4 paged decode expects kv_score_buffer rows divisible by 4, got {num_rows}"
                )
            kv_score_buffer = kv_score_buffer.view(num_rows // 4, 4, element_size)
        else:
            # For non-paged RingBuffer mode, treat each row as 8 virtual slots
            # Actually the kernel treats 2D buffer with stride index * 8 * element_size
            # So we need to interpret the flat buffer differently
            # For simplicity, reshape to [num_rows, 1, element_size] and adjust access logic
            # Or: treat buffer as [num_rows, 8, element_size/8] which doesn't match
            # Best: keep as 2D and use flat indexing: kv_score_buffer[index * 8 + slot]
            pass  # Will handle 2D case with flat indexing below
    elif kv_score_buffer.dim() != 3 or kv_score_buffer.shape[1] not in (4, 8):
        raise ValueError(
            f"DeepSeekV4 MUSA c4 decode expects kv_score_buffer shape [N,4|8,{element_size}] or [N,{element_size}], got {tuple(kv_score_buffer.shape)}"
        )

    if kv_score_input.shape[-1] != element_size:
        raise ValueError(
            f"DeepSeekV4 MUSA c4 decode expects kv_score_input trailing dim {element_size}, got {kv_score_input.shape[-1]}"
        )
    if ape.shape != (8, head_dim):
        raise ValueError(f"DeepSeekV4 MUSA c4 decode expects ape shape {(8, head_dim)}, got {tuple(ape.shape)}")

    out.zero_()

    # Determine buffer access mode and validate 3D layout matches paged flag
    use_flat_indexing = kv_score_buffer.dim() == 2
    if kv_score_buffer.dim() == 3:
        expected_slots = 4 if paged else 8
        if kv_score_buffer.shape[1] != expected_slots:
            raise ValueError(
                f"DeepSeekV4 MUSA c4 decode expects {'paged' if paged else 'non-paged'} "
                f"kv_score_buffer with {expected_slots} slots, got {kv_score_buffer.shape[1]}"
            )

    for row in range(kv_score_input.shape[0]):
        seq_len = int(seq_lens[row].item())
        index = int(indices[row].item())
        current = kv_score_input[row].reshape(4, head_dim)

        write_pos = (seq_len + (3 if paged else 7)) % (4 if paged else 8)

        if use_flat_indexing:
            # RingBuffer mode with 2D buffer: each index maps to 8 rows
            # kv_score_buffer[index * 8 + write_pos] = current
            flat_index = index * 8 + write_pos
            if flat_index < kv_score_buffer.shape[0]:
                kv_score_buffer[flat_index].copy_(current.reshape(-1))
        else:
            kv_score_buffer[index, write_pos].copy_(current.reshape(-1))

        if seq_len % 4 != 0:
            continue

        kv_window = []
        score_window = []
        for i in range(8):
            is_overlap = i < 4
            if paged:
                if is_overlap:
                    if extra_data is None:
                        raise ValueError("DeepSeekV4 MUSA paged c4 decode expects overlap extra_data")
                    overlap_index = int(extra_data[row, 0].item())
                    src = kv_score_buffer[overlap_index, i]
                else:
                    src = kv_score_buffer[index, i % 4]
            elif use_flat_indexing:
                # RingBuffer mode with 2D buffer
                slot_pos = (seq_len + i) % 8
                flat_src_index = index * 8 + slot_pos
                if flat_src_index < kv_score_buffer.shape[0]:
                    src = kv_score_buffer[flat_src_index]
                else:
                    src = torch.zeros(element_size, dtype=kv_score_buffer.dtype, device=kv_score_buffer.device)
            else:
                src = kv_score_buffer[index, (seq_len + i) % 8]

            src = src.reshape(4, head_dim)
            kv_window.append(src[0] if is_overlap else src[1])
            score_window.append(src[2] if is_overlap else src[3])

        if seq_len == 4:
            for i in range(4):
                kv_window[i] = torch.zeros_like(kv_window[i])
                score_window[i] = torch.full_like(score_window[i], -1e9)

        kv_tensor = torch.stack(kv_window, dim=0)
        score_tensor = torch.stack(score_window, dim=0)
        debug_info = {"seq_len": seq_len, "ratio": 4, "row": row}
        out[row].copy_(_compress_softmax_reduce(kv_tensor, score_tensor, ape, debug_info).to(out.dtype))
    return out

def _compress_forward_ratio128_decode(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if _musa_graph_capture_enabled() and _has_musa_tensor(kv_score_buffer, kv_score_input, ape, indices, seq_lens, out):
        raise NotImplementedError("DeepSeekV4 MUSA ratio128 decode has no torch fallback during graph capture")

    # Handle 2D buffer for RingBuffer mode (similar to c4)
    # For non-paged: buffer is 2D [N, head_dim*2] where each index maps to 128 slots
    element_size = head_dim * 2
    ring_size = 128

    # Determine buffer layout
    if kv_score_buffer.dim() == 2 and kv_score_buffer.shape[-1] == element_size:
        use_flat_indexing = True
    elif kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1] == ring_size and kv_score_buffer.shape[2] == element_size:
        use_flat_indexing = False
    else:
        raise ValueError(
            f"DeepSeekV4 MUSA c128 decode expects kv_score_buffer shape [N,{ring_size},{element_size}] or [N,{element_size}], got {tuple(kv_score_buffer.shape)}"
        )

    if kv_score_input.shape[-1] != element_size:
        raise ValueError(
            f"DeepSeekV4 MUSA c128 decode expects kv_score_input trailing dim {element_size}, got {kv_score_input.shape[-1]}"
        )
    if ape.shape != (128, head_dim):
        raise ValueError(f"DeepSeekV4 MUSA c128 decode expects ape shape {(128, head_dim)}, got {tuple(ape.shape)}")

    out.zero_()
    for row in range(kv_score_input.shape[0]):
        seq_len = int(seq_lens[row].item())
        index = int(indices[row].item())
        current = kv_score_input[row].reshape(2, head_dim)

        write_pos = (seq_len + 127) % 128

        if use_flat_indexing:
            # RingBuffer mode with 2D buffer: each index maps to 128 rows
            flat_index = index * ring_size + write_pos
            if flat_index < kv_score_buffer.shape[0]:
                kv_score_buffer[flat_index].copy_(current.reshape(-1))
        else:
            kv_score_buffer[index, write_pos].copy_(current.reshape(-1))

        if seq_len % 128 != 0:
            continue

        # Gather 128 slots for compression
        kv_window = []
        score_window = []
        for i in range(128):
            if use_flat_indexing:
                slot_pos = (seq_len + i) % 128
                flat_src_index = index * ring_size + slot_pos
                if flat_src_index < kv_score_buffer.shape[0]:
                    src = kv_score_buffer[flat_src_index]
                else:
                    src = torch.zeros(element_size, dtype=kv_score_buffer.dtype, device=kv_score_buffer.device)
            else:
                src = kv_score_buffer[index, (seq_len + i) % 128]

            src = src.reshape(2, head_dim)
            kv_window.append(src[0])
            score_window.append(src[1])

        kv_tensor = torch.stack(kv_window, dim=0)
        score_tensor = torch.stack(score_window, dim=0)
        debug_info = {"seq_len": seq_len, "ratio": 128, "row": row}
        out[row].copy_(_compress_softmax_reduce(kv_tensor, score_tensor, ape, debug_info).to(out.dtype))
    return out

def _validate_prefill_plan_rows(plan: torch.Tensor) -> None:
    if plan.dim() != 2 or plan.shape[1] != 16 or plan.dtype != torch.uint8:
        raise ValueError(f"DeepSeekV4 MUSA prefill plan expects uint8 shape [N,16], got {tuple(plan.shape)} {plan.dtype}")

def _is_prefill_plan_int32_view_compatible(plan: torch.Tensor) -> bool:
    if plan.dim() != 2 or plan.shape[1] != 16 or plan.dtype != torch.uint8:
        return False
    if plan.stride(1) != 1 or plan.stride(0) <= 0:
        return False
    if plan.storage_offset() % 4 != 0 or plan.stride(0) % 4 != 0:
        return False
    return True

def _prefill_plan_rows(plan: torch.Tensor) -> torch.Tensor:
    _validate_prefill_plan_rows(plan)
    if _is_prefill_plan_int32_view_compatible(plan):
        return plan.view(torch.int32).reshape(plan.shape[0], 4)
    return plan.contiguous().view(torch.int32).reshape(-1, 4)

def _prefill_plan_summary(name: str, plan: torch.Tensor, compatible: bool) -> str:
    return (
        f"{name}:dtype={plan.dtype},shape={tuple(plan.shape)},stride={tuple(plan.stride())},"
        f"offset={int(plan.storage_offset())},contig={plan.is_contiguous()},i32_compatible={compatible}"
    )

def _try_tilelang_compress_forward_ratio4_prefill_musa(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> bool:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (kv_score_buffer, kv_score_input, ape, indices, compress_plan, write_plan, out)):
        return False
    paged = extra_data is not None
    if paged and not isinstance(extra_data, torch.Tensor):
        return False
    if kv_score_buffer.device.type != "musa" or kv_score_input.device.type != "musa":
        return False
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32 or out.dtype != torch.float32:
        return False
    if ape.device != kv_score_input.device or ape.dtype != torch.float32:
        return False
    if indices.device != kv_score_input.device or indices.dtype != torch.int32:
        return False
    if paged and (extra_data.device != kv_score_input.device or extra_data.dtype != torch.int32):
        return False
    if compress_plan.device != kv_score_input.device or write_plan.device != kv_score_input.device:
        return False
    if compress_plan.dtype != torch.uint8 or write_plan.dtype != torch.uint8:
        return False
    if not kv_score_buffer.is_contiguous() or not kv_score_input.is_contiguous() or not out.is_contiguous():
        return False
    if not ape.is_contiguous() or not indices.is_contiguous():
        return False
    if paged and not extra_data.is_contiguous():
        return False
    compress_plan_i32_compatible = _is_prefill_plan_int32_view_compatible(compress_plan)
    write_plan_i32_compatible = _is_prefill_plan_int32_view_compatible(write_plan)
    if not compress_plan_i32_compatible or not write_plan_i32_compatible:
        if paged and _compress_c4_prefill_musa_jit_enabled() and _compress_raise_prefill_miss():
            raise RuntimeError(
                "C4 prefill MUSA JIT not entered: unsupported prefill plan layout; "
                f"{_prefill_plan_summary('compress_plan', compress_plan, compress_plan_i32_compatible)}; "
                f"{_prefill_plan_summary('write_plan', write_plan, write_plan_i32_compatible)}"
            )
        return False
    if paged:
        if kv_score_buffer.dim() != 3 or kv_score_buffer.shape[1:] != (4, head_dim * 4):
            return False
        kv_score_buffer_view = kv_score_buffer
        use_flat_kernel = False
    elif kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (8, head_dim * 4):
        kv_score_buffer_view = kv_score_buffer
        use_flat_kernel = False
    elif kv_score_buffer.dim() == 2 and kv_score_buffer.shape[1] == head_dim * 4:
        kv_score_buffer_view = kv_score_buffer
        use_flat_kernel = True
    else:
        return False
    if paged and (extra_data.dim() != 2 or extra_data.shape[1] != 4):
        return False
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 4:
        return False
    if ape.shape != (8, head_dim) or out.shape != (kv_score_input.shape[0], head_dim):
        return False
    if indices.dim() != 1 or compress_plan.dim() != 2 or write_plan.dim() != 2:
        return False
    if compress_plan.shape[1] != 16 or write_plan.shape[1] != 16:
        return False

    compress_rows = _prefill_plan_rows(compress_plan)
    write_rows = _prefill_plan_rows(write_plan)
    use_vector_write = (
        _compress_vector_write_enabled()
        and (head_dim * 4) % 4 == 0
        and write_rows.shape[0] >= _compress_vector_write_min_rows()
    )
    try:
        if _compress_tilelang_zero_enabled():
            _tilelang_compress_prefill_zero_kernel(head_dim, _compress_threads("PREFILL_ZERO"))(out)
        else:
            out.zero_()
        if paged:
            if _compress_c4_prefill_musa_jit_enabled():
                ok, reason = try_c4_page_prefill_musa_jit(
                    kv_score_buffer_view,
                    kv_score_input,
                    ape,
                    indices,
                    extra_data,
                    compress_rows,
                    write_rows,
                    out,
                    head_dim,
                )
                if ok:
                    return True
                if _compress_raise_prefill_miss():
                    raise RuntimeError(f"C4 prefill MUSA JIT miss: {reason}")
            if compress_rows.shape[0] != 0:
                reduce_kernel_factory = (
                    _tilelang_compress_ratio4_prefill_page_reduce_cached_kernel
                    if compress_rows.shape[0] >= 1024
                    else _tilelang_compress_ratio4_prefill_page_reduce_kernel
                )
                reduce_kernel_factory(head_dim, _compress_threads("C4_PREFILL_REDUCE"))(
                    kv_score_buffer_view,
                    kv_score_input,
                    ape,
                    indices,
                    extra_data,
                    compress_rows,
                    out,
                )
            if write_rows.shape[0] != 0:
                if use_vector_write:
                    try:
                        _tilelang_compress_ratio4_prefill_page_write_vec4_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            extra_data,
                            write_rows,
                        )
                    except Exception:
                        _tilelang_compress_ratio4_prefill_page_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            extra_data,
                            write_rows,
                        )
                else:
                    _tilelang_compress_ratio4_prefill_page_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                        kv_score_buffer_view,
                        kv_score_input,
                        indices,
                        extra_data,
                        write_rows,
                    )
        elif use_flat_kernel:
            if write_rows.shape[0] != 0:
                if use_vector_write:
                    try:
                        _tilelang_compress_ratio4_prefill_flat_write_vec4_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            write_rows,
                        )
                    except Exception:
                        _tilelang_compress_ratio4_prefill_flat_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            write_rows,
                        )
                else:
                    _tilelang_compress_ratio4_prefill_flat_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                        kv_score_buffer_view,
                        kv_score_input,
                        indices,
                        write_rows,
                    )
            if compress_rows.shape[0] != 0:
                _tilelang_compress_ratio4_prefill_flat_reduce_kernel(head_dim, _compress_threads("C4_PREFILL_REDUCE"))(
                    kv_score_buffer_view,
                    kv_score_input,
                    ape,
                    indices,
                    compress_rows,
                    out,
                )
        else:
            if write_rows.shape[0] != 0:
                if use_vector_write:
                    try:
                        _tilelang_compress_ratio4_prefill_write_vec4_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            write_rows,
                        )
                    except Exception:
                        _tilelang_compress_ratio4_prefill_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                            kv_score_buffer_view,
                            kv_score_input,
                            indices,
                            write_rows,
                        )
                else:
                    _tilelang_compress_ratio4_prefill_write_kernel(head_dim, _compress_threads("C4_PREFILL_WRITE"))(
                        kv_score_buffer_view,
                        kv_score_input,
                        indices,
                        write_rows,
                    )
            if compress_rows.shape[0] != 0:
                _tilelang_compress_ratio4_prefill_reduce_kernel(head_dim, _compress_threads("C4_PREFILL_REDUCE"))(
                    kv_score_buffer_view,
                    kv_score_input,
                    ape,
                    indices,
                    compress_rows,
                    out,
                )
    except Exception:
        if os.environ.get("SGLANG_DEEPSEEK_V4_MUSA_COMPRESS_RAISE_PREFILL_MISS") == "1":
            raise
        return False
    return True

def _try_tilelang_compress_forward_ratio128_prefill_musa(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> bool:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (kv_score_buffer, kv_score_input, ape, indices, compress_plan, write_plan, out)):
        return False
    if extra_data is not None and not isinstance(extra_data, torch.Tensor):
        return False
    if kv_score_buffer.device.type != "musa" or kv_score_input.device.type != "musa":
        return False
    if kv_score_buffer.dtype != torch.float32 or kv_score_input.dtype != torch.float32 or out.dtype != torch.float32:
        return False
    if ape.device != kv_score_input.device or ape.dtype != torch.float32:
        return False
    if indices.device != kv_score_input.device or indices.dtype != torch.int32:
        return False
    if extra_data is not None and (extra_data.device != kv_score_input.device or extra_data.dtype != torch.int32):
        return False
    if compress_plan.device != kv_score_input.device or write_plan.device != kv_score_input.device:
        return False
    if compress_plan.dtype != torch.uint8 or write_plan.dtype != torch.uint8:
        return False
    if not kv_score_buffer.is_contiguous() or not kv_score_input.is_contiguous() or not out.is_contiguous():
        return False
    if not ape.is_contiguous() or not indices.is_contiguous():
        return False
    if extra_data is not None and not extra_data.is_contiguous():
        return False
    if not _is_prefill_plan_int32_view_compatible(compress_plan) or not _is_prefill_plan_int32_view_compatible(write_plan):
        return False
    if kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (128, head_dim * 2):
        kv_score_buffer_view = kv_score_buffer
    elif kv_score_buffer.dim() == 2 and kv_score_buffer.shape[0] % 128 == 0 and kv_score_buffer.shape[1] == head_dim * 2:
        kv_score_buffer_view = kv_score_buffer.view(-1, 128, head_dim * 2)
    else:
        return False
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 2:
        return False
    if ape.shape != (128, head_dim) or out.shape != (kv_score_input.shape[0], head_dim):
        return False
    if indices.dim() != 1 or compress_plan.dim() != 2 or write_plan.dim() != 2:
        return False
    if extra_data is not None:
        if extra_data.dim() == 2 and extra_data.shape[1] == 1:
            load_indices = extra_data.reshape(-1)
        elif extra_data.dim() == 1:
            load_indices = extra_data
        else:
            return False
        if load_indices.shape != indices.shape:
            return False
    else:
        load_indices = indices
    if compress_plan.shape[1] != 16 or write_plan.shape[1] != 16:
        return False

    compress_rows = _prefill_plan_rows(compress_plan)
    write_rows = _prefill_plan_rows(write_plan)
    use_vector_write = (
        _compress_vector_write_enabled()
        and (head_dim * 2) % 4 == 0
        and write_rows.shape[0] >= _compress_vector_write_min_rows()
    )
    try:
        if _compress_tilelang_zero_enabled():
            _tilelang_compress_prefill_zero_kernel(head_dim, _compress_threads("PREFILL_ZERO"))(out)
        else:
            out.zero_()
        if write_rows.shape[0] != 0:
            if use_vector_write:
                try:
                    _tilelang_compress_ratio128_prefill_write_vec4_kernel(head_dim, _compress_threads("C128_PREFILL_WRITE"))(
                        kv_score_buffer_view,
                        kv_score_input,
                        indices,
                        write_rows,
                    )
                except Exception:
                    _tilelang_compress_ratio128_prefill_write_kernel(head_dim, _compress_threads("C128_PREFILL_WRITE"))(
                        kv_score_buffer_view,
                        kv_score_input,
                        indices,
                        write_rows,
                    )
            else:
                _tilelang_compress_ratio128_prefill_write_kernel(head_dim, _compress_threads("C128_PREFILL_WRITE"))(
                    kv_score_buffer_view,
                    kv_score_input,
                    indices,
                    write_rows,
                )
        if compress_rows.shape[0] != 0:
            reduce_kernel = (
                _tilelang_compress_ratio128_prefill_reduce_parallel_kernel(
                    head_dim,
                    _compress_c128_parallel_final_merge("prefill"),
                )
                if _compress_c128_parallel_reduce_enabled() and head_dim % 64 == 0
                else _tilelang_compress_ratio128_prefill_reduce_kernel(head_dim, _compress_threads("C128_PREFILL_REDUCE"))
            )
            reduce_kernel(
                kv_score_buffer_view,
                kv_score_input,
                ape,
                indices,
                load_indices,
                compress_rows,
                out,
            )
    except Exception:
        return False
    return True

def _compress_page4_buffer_layout(
    kv_score_buffer: torch.Tensor,
    element_size: int,
    name: str,
) -> None:
    if kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (4, element_size):
        return
    raise ValueError(f"{name} expects kv_score_buffer shape [N,4,{element_size}], got {tuple(kv_score_buffer.shape)}")

def _compress_ring_buffer_layout(
    kv_score_buffer: torch.Tensor,
    ring_size: int,
    element_size: int,
    name: str,
) -> Literal["logical", "flat"]:
    if kv_score_buffer.dim() == 3 and kv_score_buffer.shape[1:] == (ring_size, element_size):
        return "logical"
    if kv_score_buffer.dim() == 2 and kv_score_buffer.shape[1] == element_size:
        return "flat"
    raise ValueError(
        f"{name} expects kv_score_buffer shape [N,{ring_size},{element_size}] or [N*{ring_size},{element_size}], got {tuple(kv_score_buffer.shape)}"
    )

def _compress_ring_buffer_slot(
    kv_score_buffer: torch.Tensor,
    layout: Literal["logical", "flat"],
    index: int,
    slot: int,
    ring_size: int,
) -> torch.Tensor:
    if layout == "logical":
        return kv_score_buffer[index, slot]
    return kv_score_buffer[index * ring_size + slot]

def _compress_forward_ratio4_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if _musa_graph_capture_enabled() and _has_musa_tensor(
        kv_score_buffer, kv_score_input, ape, indices, compress_plan, write_plan, extra_data, out
    ):
        raise NotImplementedError("DeepSeekV4 MUSA ratio4 prefill has no torch fallback during graph capture")

    paged = extra_data is not None
    if paged:
        _compress_page4_buffer_layout(
            kv_score_buffer,
            element_size=head_dim * 4,
            name="DeepSeekV4 MUSA c4 prefill page4",
        )
        if extra_data.dim() != 2 or extra_data.shape[1] != 4:
            raise ValueError(f"DeepSeekV4 MUSA c4 prefill page4 expects extra_data shape [N,4], got {tuple(extra_data.shape)}")
        layout = None
    else:
        layout = _compress_ring_buffer_layout(
            kv_score_buffer,
            ring_size=8,
            element_size=head_dim * 4,
            name="DeepSeekV4 MUSA c4 prefill",
        )
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 4:
        raise ValueError(
            f"DeepSeekV4 MUSA c4 prefill expects kv_score_input shape [N,{head_dim * 4}], got {tuple(kv_score_input.shape)}"
        )
    if ape.shape != (8, head_dim):
        raise ValueError(f"DeepSeekV4 MUSA c4 prefill expects ape shape {(8, head_dim)}, got {tuple(ape.shape)}")

    def run_write_rows() -> None:
        for ragged_id, batch_id, position, _window_len in _prefill_plan_rows(write_plan).cpu().tolist():
            if paged:
                block_id = int(indices[batch_id].item())
                if position < int(extra_data[batch_id, 3].item()):
                    block_id = int(extra_data[batch_id, 2].item())
                kv_score_buffer[block_id, position % 4].copy_(kv_score_input[ragged_id])
            else:
                index = int(indices[batch_id].item())
                _compress_ring_buffer_slot(kv_score_buffer, layout, index, position % 8, 8).copy_(kv_score_input[ragged_id])

    def run_compress_rows() -> None:
        for ragged_id, batch_id, position, window_len in _prefill_plan_rows(compress_plan).cpu().tolist():
            index = int(indices[batch_id].item())
            seq_len = position + 1
            kv_window = []
            score_window = []
            for i in range(8):
                is_overlap = i < 4
                if paged and i < window_len:
                    source_block = int(extra_data[batch_id, 1].item())
                    if window_len > 4 and i < 4:
                        source_block = int(extra_data[batch_id, 0].item())
                    src = kv_score_buffer[source_block, i % 4]
                elif i < window_len:
                    src = _compress_ring_buffer_slot(kv_score_buffer, layout, index, (seq_len + i) % 8, 8)
                else:
                    src = kv_score_input[ragged_id + i - 7]
                src = src.reshape(4, head_dim)
                kv_window.append(src[0] if is_overlap else src[1])
                score_window.append(src[2] if is_overlap else src[3])

            if seq_len == 4:
                for i in range(4):
                    kv_window[i] = torch.zeros_like(kv_window[i])
                    score_window[i] = torch.full_like(score_window[i], -1e9)

            kv_tensor = torch.stack(kv_window, dim=0)
            score_tensor = torch.stack(score_window, dim=0)
            debug_info = {"seq_len": seq_len, "ratio": 4, "row": ragged_id, "mode": "prefill"}
            out[ragged_id].copy_(_compress_softmax_reduce(kv_tensor, score_tensor, ape, debug_info).to(out.dtype))

    out.zero_()
    if paged:
        run_compress_rows()
        run_write_rows()
    else:
        run_write_rows()
        run_compress_rows()
    return out

def _compress_forward_ratio128_prefill(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    compress_plan: torch.Tensor,
    write_plan: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if _musa_graph_capture_enabled() and _has_musa_tensor(
        kv_score_buffer, kv_score_input, ape, indices, compress_plan, write_plan, extra_data, out
    ):
        raise NotImplementedError("DeepSeekV4 MUSA ratio128 prefill has no torch fallback during graph capture")

    layout = _compress_ring_buffer_layout(
        kv_score_buffer,
        ring_size=128,
        element_size=head_dim * 2,
        name="DeepSeekV4 MUSA c128 prefill",
    )
    if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 2:
        raise ValueError(
            f"DeepSeekV4 MUSA c128 prefill expects kv_score_input shape [N,{head_dim * 2}], got {tuple(kv_score_input.shape)}"
        )
    if ape.shape != (128, head_dim):
        raise ValueError(f"DeepSeekV4 MUSA c128 prefill expects ape shape {(128, head_dim)}, got {tuple(ape.shape)}")
    if extra_data is None:
        load_indices = indices
    else:
        if extra_data.dtype != torch.int32:
            raise ValueError(f"DeepSeekV4 MUSA c128 prefill load_indices expects int32, got {extra_data.dtype}")
        if extra_data.dim() == 2 and extra_data.shape[1] == 1:
            load_indices = extra_data.reshape(-1)
        elif extra_data.dim() == 1:
            load_indices = extra_data
        else:
            raise ValueError(f"DeepSeekV4 MUSA c128 prefill load_indices expects shape [N] or [N,1], got {tuple(extra_data.shape)}")
        if load_indices.shape != indices.shape:
            raise ValueError(
                f"DeepSeekV4 MUSA c128 prefill load_indices shape {tuple(load_indices.shape)} must match indices {tuple(indices.shape)}"
            )

    out.zero_()
    for ragged_id, batch_id, position, _window_len in _prefill_plan_rows(write_plan).cpu().tolist():
        index = int(indices[batch_id].item())
        _compress_ring_buffer_slot(kv_score_buffer, layout, index, position % 128, 128).copy_(kv_score_input[ragged_id])

    for ragged_id, batch_id, _position, window_len in _prefill_plan_rows(compress_plan).cpu().tolist():
        index = int(load_indices[batch_id].item())
        kv_window = []
        score_window = []
        for i in range(128):
            if i < window_len:
                src = _compress_ring_buffer_slot(kv_score_buffer, layout, index, i, 128)
            else:
                src = kv_score_input[ragged_id + i - 127]
            src = src.reshape(2, head_dim)
            kv_window.append(src[0])
            score_window.append(src[1])
        kv_tensor = torch.stack(kv_window, dim=0)
        score_tensor = torch.stack(score_window, dim=0)
        debug_info = {"seq_len": window_len, "ratio": 128, "row": ragged_id, "mode": "prefill"}
        out[ragged_id].copy_(_compress_softmax_reduce(kv_tensor, score_tensor, ape, debug_info).to(out.dtype))
    return out

def _flatten_decode_seq_lens(seq_lens: torch.Tensor) -> torch.Tensor:
    if _is_musa_tensor(seq_lens):
        if seq_lens.dtype != torch.int32:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA compress_forward decode expects int32 seq_lens on MUSA, got {seq_lens.dtype}"
            )
        if seq_lens.dim() == 1:
            return seq_lens
        if seq_lens.dim() > 1 and seq_lens.numel() != seq_lens.shape[0]:
            raise NotImplementedError(
                f"DeepSeekV4 MUSA compress_forward decode expects flat seq_lens, got {tuple(seq_lens.shape)}"
            )
        return seq_lens.as_strided((seq_lens.shape[0],), (seq_lens.stride(0),), storage_offset=seq_lens.storage_offset())
    return seq_lens.to(torch.int32).view(-1)

def _has_musa_compress_decode_input(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extra_data: Optional[torch.Tensor],
    out: torch.Tensor,
) -> bool:
    return (
        _is_musa_tensor(kv_score_buffer)
        or _is_musa_tensor(kv_score_input)
        or _is_musa_tensor(ape)
        or _is_musa_tensor(indices)
        or _is_musa_tensor(seq_lens)
        or _is_musa_tensor(extra_data)
        or _is_musa_tensor(out)
    )

def _try_tensor_paged_mqa_logits_metadata_musa(seq_lens: torch.Tensor, page_size: int, num_sm: int) -> Optional[torch.Tensor]:
    if page_size != 64 or num_sm <= 0:
        return None
    if seq_lens.dtype not in (torch.int32, torch.int64):
        return None
    seq_lens = seq_lens.view(-1).to(torch.int32)
    metadata = torch.empty((num_sm + 1, 2), dtype=torch.int32, device=seq_lens.device)
    if seq_lens.numel() == 0:
        metadata.zero_()
        return metadata

    work = torch.div(seq_lens + 255, 256, rounding_mode="floor")
    prefix_end = torch.cumsum(work, dim=0)
    total_work = prefix_end[-1]
    boundaries = torch.arange(num_sm + 1, dtype=torch.int32, device=seq_lens.device)
    avg = torch.div(total_work, num_sm, rounding_mode="floor")
    ret = total_work.remainder(num_sm)
    targets = boundaries * avg + torch.minimum(boundaries, ret)

    q = torch.searchsorted(prefix_end, targets, right=True).to(torch.int32)
    valid = q < seq_lens.numel()
    safe_q = q.clamp(max=seq_lens.numel() - 1).long()
    prefix_start = prefix_end[safe_q] - work[safe_q]

    metadata[:, 0] = torch.where(valid, q, torch.full_like(q, seq_lens.numel()))
    metadata[:, 1] = torch.where(valid, targets - prefix_start, torch.zeros_like(q))
    return metadata

def get_paged_mqa_logits_metadata_musa(
    seq_lens: torch.Tensor, page_size: int, num_sm: int
) -> torch.Tensor:
    tensor_result = _try_tensor_paged_mqa_logits_metadata_musa(seq_lens, page_size, num_sm)
    if tensor_result is not None:
        return tensor_result
    if page_size != 64:
        raise NotImplementedError(
            f"DeepSeekV4 MUSA paged MQA metadata expects page_size=64, got {page_size}"
        )
    raise NotImplementedError(
        "DeepSeekV4 MUSA paged MQA metadata requires int32/int64 tensor seq_lens and a positive num_sm"
    )

def compress_forward_musa(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    indices: torch.Tensor,
    plan: _MetadataPlan = None,
    extra_data: Optional[torch.Tensor] = None,
    *,
    head_dim: int,
    compress_ratio: Literal[4, 128],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = kv_score_input.new_empty((kv_score_input.shape[0], head_dim))
    if compress_ratio not in (4, 128):
        raise ValueError(f"Unsupported DeepSeekV4 MUSA compress_ratio={compress_ratio}")
    if plan is not None and hasattr(plan, "write_plan"):
        compress_plan = plan.compress_plan
        write_plan = plan.write_plan
        _validate_prefill_plan_rows(compress_plan)
        _validate_prefill_plan_rows(write_plan)
        if compress_ratio == 4:
            if extra_data is not None:
                _compress_page4_buffer_layout(
                    kv_score_buffer,
                    element_size=head_dim * 4,
                    name="DeepSeekV4 MUSA c4 prefill page4",
                )
                if extra_data.dim() != 2 or extra_data.shape[1] != 4:
                    raise ValueError(f"DeepSeekV4 MUSA c4 prefill page4 expects extra_data shape [N,4], got {tuple(extra_data.shape)}")
            else:
                _compress_ring_buffer_layout(
                    kv_score_buffer,
                    ring_size=8,
                    element_size=head_dim * 4,
                    name="DeepSeekV4 MUSA c4 prefill",
                )
            if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 4:
                raise ValueError(
                    f"DeepSeekV4 MUSA c4 prefill expects kv_score_input shape [N,{head_dim * 4}], got {tuple(kv_score_input.shape)}"
                )
            if ape.shape != (8, head_dim):
                raise ValueError(f"DeepSeekV4 MUSA c4 prefill expects ape shape {(8, head_dim)}, got {tuple(ape.shape)}")
        else:
            _compress_ring_buffer_layout(
                kv_score_buffer,
                ring_size=128,
                element_size=head_dim * 2,
                name="DeepSeekV4 MUSA c128 prefill",
            )
            if kv_score_input.dim() != 2 or kv_score_input.shape[1] != head_dim * 2:
                raise ValueError(
                    f"DeepSeekV4 MUSA c128 prefill expects kv_score_input shape [N,{head_dim * 2}], got {tuple(kv_score_input.shape)}"
                )
            if ape.shape != (128, head_dim):
                raise ValueError(f"DeepSeekV4 MUSA c128 prefill expects ape shape {(128, head_dim)}, got {tuple(ape.shape)}")
        if compress_ratio == 4:
            if _try_tilelang_compress_forward_ratio4_prefill_musa(
                kv_score_buffer,
                kv_score_input,
                ape,
                indices,
                compress_plan,
                write_plan,
                extra_data,
                out,
                head_dim,
            ):
                return out
        elif _try_tilelang_compress_forward_ratio128_prefill_musa(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            compress_plan,
            write_plan,
            extra_data,
            out,
            head_dim,
        ):
            return out
        if (
            _is_musa_tensor(kv_score_buffer)
            or _is_musa_tensor(kv_score_input)
            or _is_musa_tensor(ape)
            or _is_musa_tensor(indices)
            or _is_musa_tensor(compress_plan)
            or _is_musa_tensor(write_plan)
            or _is_musa_tensor(extra_data)
            or _is_musa_tensor(out)
        ):
            if _musa_graph_capture_enabled():
                raise NotImplementedError(
                    "DeepSeekV4 MUSA compress_forward prefill has no torch fallback during graph capture; "
                    "Python fallback is disabled for graph capture and TileLang prefill path is required"
                )
            if not _debug_musa_allow_torch_fallback():
                raise NotImplementedError(
                    "DeepSeekV4 MUSA compress_forward prefill has no torch fallback by default; "
                    "Python fallback is disabled on MUSA"
                )
            _debug_musa_torch_fallback("DeepSeekV4 MUSA compress_forward prefill using Python fallback outside graph capture")
        if compress_ratio == 4:
            result = _compress_forward_ratio4_prefill(
                kv_score_buffer,
                kv_score_input,
                ape,
                indices,
                compress_plan,
                write_plan,
                extra_data,
                out,
                head_dim,
            )
            return result
        result = _compress_forward_ratio128_prefill(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            compress_plan,
            write_plan,
            extra_data,
            out,
            head_dim,
        )
        return result
    if plan is None or not hasattr(plan, "seq_lens"):
        raise ValueError("DeepSeekV4 MUSA compress_forward decode expects a plan with seq_lens")

    seq_lens = _flatten_decode_seq_lens(plan.seq_lens)
    if kv_score_input.shape[0] != seq_lens.numel() or kv_score_input.shape[0] != indices.numel():
        raise ValueError(
            "DeepSeekV4 MUSA compress_forward expects kv_score_input, seq_lens, and indices to have matching batch size"
        )

    if compress_ratio == 4:
        tilelang_ok, tilelang_failure = _try_tilelang_compress_forward_ratio4_decode_musa(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            extra_data,
            out,
            head_dim,
        )
        if tilelang_ok:
            return out
        if _has_musa_compress_decode_input(kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out):
            if _musa_graph_capture_enabled():
                raise NotImplementedError(
                    "DeepSeekV4 MUSA compress_forward ratio4 decode has no torch fallback during graph capture; "
                    "Python fallback is disabled for graph capture and TileLang decode path is required: "
                    f"{tilelang_failure}"
                )
            if not _debug_musa_allow_torch_fallback():
                raise NotImplementedError(
                    "DeepSeekV4 MUSA compress_forward ratio4 decode has no torch fallback by default; "
                    "Python fallback is disabled on MUSA: "
                    f"{tilelang_failure}"
                )
            _debug_musa_torch_fallback(
                "DeepSeekV4 MUSA compress_forward ratio4 decode using Python fallback outside graph capture after TileLang miss: "
                f"{tilelang_failure}"
            )
        return _compress_forward_ratio4_decode(
            kv_score_buffer,
            kv_score_input,
            ape,
            indices,
            seq_lens,
            extra_data,
            out,
            head_dim,
        )
    tilelang_ok, tilelang_failure = _try_tilelang_compress_forward_ratio128_decode_musa(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        seq_lens,
        out,
        head_dim,
    )
    if tilelang_ok:
        return out
    if _has_musa_compress_decode_input(kv_score_buffer, kv_score_input, ape, indices, seq_lens, extra_data, out):
        if _musa_graph_capture_enabled():
            # TODO(dsv4-musa): capture production metadata for ratio128 decode and replace this with a real path.
            raise NotImplementedError(
                "DeepSeekV4 MUSA compress_forward ratio128 decode has no torch fallback during graph capture: "
                f"{tilelang_failure}"
            )
        if not _debug_musa_allow_torch_fallback():
            raise NotImplementedError(
                "DeepSeekV4 MUSA compress_forward ratio128 decode has no torch fallback by default; "
                "Python fallback is disabled on MUSA: "
                f"{tilelang_failure}"
            )
        _debug_musa_torch_fallback(
            "DeepSeekV4 MUSA compress_forward ratio128 decode using Python fallback outside graph capture after TileLang miss: "
            f"{tilelang_failure}"
        )
    return _compress_forward_ratio128_decode(
        kv_score_buffer,
        kv_score_input,
        ape,
        indices,
        seq_lens,
        out,
        head_dim,
    )

__all__ = [
    '_try_tilelang_compress_forward_ratio4_decode_musa',
    '_try_tilelang_compress_forward_ratio128_decode_musa',
    '_compress_softmax_reduce',
    '_compress_forward_ratio4_decode',
    '_compress_forward_ratio128_decode',
    '_validate_prefill_plan_rows',
    '_prefill_plan_rows',
    '_try_tilelang_compress_forward_ratio4_prefill_musa',
    '_try_tilelang_compress_forward_ratio128_prefill_musa',
    '_compress_page4_buffer_layout',
    '_compress_ring_buffer_layout',
    '_compress_ring_buffer_slot',
    '_compress_forward_ratio4_prefill',
    '_compress_forward_ratio128_prefill',
    '_flatten_decode_seq_lens',
    '_has_musa_compress_decode_input',
    '_try_tensor_paged_mqa_logits_metadata_musa',
    'get_paged_mqa_logits_metadata_musa',
    'compress_forward_musa',
]
