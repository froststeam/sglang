import importlib
import logging
import os
from functools import lru_cache
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F


@lru_cache(maxsize=None)
def _tilelang_silu_and_mul_kernel(hidden2: int, input_dtype: str, threads: int = 256):
    import tilelang
    import tilelang.language as T

    from sglang.srt.hardware_backend.layers.deepseek_v4_musa.kernels.kernel_common import (
        _tilelang_jit,
        _tilelang_musa_aggressive_pass_configs,
    )

    if hidden2 % 2 != 0:
        raise ValueError(f"SwiGLU input hidden dimension must be even, got {hidden2}")
    hidden = hidden2 // 2
    num_rows = T.dynamic("num_rows")

    @_tilelang_jit(
        tilelang,
        f"dsv4_moe_silu_and_mul_h{hidden}_t{threads}_{input_dtype}",
        pass_configs=_tilelang_musa_aggressive_pass_configs(
            tilelang, disable_index_promotion=True
        ),
    )
    def silu_and_mul_kernel(
        x: T.Tensor[(num_rows, hidden2), input_dtype],
        out: T.Tensor[(num_rows, hidden), input_dtype],
        swiglu_limit: T.float32,
    ):
        with T.Kernel(T.ceildiv(num_rows * hidden, threads), threads=threads) as (bid,):
            tx = T.get_thread_binding()
            linear = bid * threads + tx
            row = linear // T.int32(hidden)
            col = linear - row * T.int32(hidden)
            gate = T.alloc_local((1,), "float32")
            up = T.alloc_local((1,), "float32")

            if linear < num_rows * hidden:
                gate[0] = T.cast(x[row, col], "float32")
                up[0] = T.cast(x[row, col + hidden], "float32")
                if swiglu_limit >= 0.0:
                    gate[0] = T.min(gate[0], swiglu_limit)
                    up[0] = T.min(T.max(up[0], -swiglu_limit), swiglu_limit)
                out[row, col] = T.cast(
                    gate[0] / (1.0 + T.exp(-gate[0])) * up[0],
                    input_dtype,
                )

    return silu_and_mul_kernel


def silu_and_mul_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: Optional[float] = None,
) -> None:
    if input.dim() != 2 or output.dim() != 2:
        raise ValueError(
            f"DeepSeekV4 MUSA TileLang SwiGLU expects 2D tensors, got input={tuple(input.shape)} output={tuple(output.shape)}"
        )
    if input.shape[1] % 2 != 0 or output.shape != (input.shape[0], input.shape[1] // 2):
        raise ValueError(
            f"DeepSeekV4 MUSA TileLang SwiGLU shape mismatch: input={tuple(input.shape)} output={tuple(output.shape)}"
        )
    if not input.is_contiguous() or not output.is_contiguous():
        raise ValueError("DeepSeekV4 MUSA TileLang SwiGLU requires contiguous input/output")
    if input.dtype != output.dtype or input.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"DeepSeekV4 MUSA TileLang SwiGLU requires matching bf16/fp16 dtype, got input={input.dtype} output={output.dtype}"
        )

    dtype = "bfloat16" if input.dtype == torch.bfloat16 else "float16"
    kernel = _tilelang_silu_and_mul_kernel(input.shape[1], dtype)
    kernel(input, output, -1.0 if swiglu_limit is None else float(swiglu_limit))


def _try_tile_swiglu_per_token_cast_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    swiglu_limit: Optional[float],
) -> bool:
    if input.dim() != 2 or output.dim() != 2 or output_scale.dim() != 2:
        return False
    if output.shape != (input.shape[0], input.shape[1] // 2):
        return False
    if output_scale.numel() != input.shape[0] * (input.shape[1] // (2 * quant_group_size)):
        return False

    try:
        tile_quant = importlib.import_module("tile_kernels.quant")
        quantized, scale = tile_quant.swiglu_forward_and_per_token_cast(
            x=input.contiguous(),
            fmt="e4m3",
            num_per_channels=quant_group_size,
            pos_to_expert=None,
            pos_to_token_topk=None,
            topk_weights=None,
            swiglu_clamp_value=swiglu_limit,
            use_tma_aligned_col_major_sf=False,
            round_sf=False,
            use_packed_ue8m0=False,
            clamped_count=None,
        )
    except Exception:
        return False

    if quantized.shape != output.shape or scale.numel() != output_scale.numel():
        return False

    output.copy_(quantized.reshape_as(output))
    output_scale.copy_(scale.to(output_scale.dtype).reshape_as(output_scale))
    return True

def _try_tile_swiglu_per_token_cast_prealloc_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    swiglu_limit: Optional[float],
) -> bool:
    if input.dim() != 2 or output.dim() != 2 or output_scale.dim() != 2:
        return False
    if input.device.type != "musa":
        return False
    if not input.is_contiguous() or not output.is_contiguous() or not output_scale.is_contiguous():
        return False
    hidden = input.shape[1] // 2
    if input.shape[1] != hidden * 2 or output.shape != (input.shape[0], hidden):
        return False
    if output_scale.shape != (input.shape[0], hidden // quant_group_size):
        return False
    if output.dtype != torch.float8_e4m3fn or output_scale.dtype != torch.float32:
        return False
    if quant_group_size != 128 and quant_group_size != hidden:
        return False

    try:
        import tilelang.language as T
        quant_common = importlib.import_module("tile_kernels.quant.common")
        quant_kernel = importlib.import_module(
            "tile_kernels.quant.swiglu_forward_and_per_token_cast_kernel"
        )

        out_config = quant_common.get_cast_output_config(
            "e4m3",
            (1, quant_group_size),
            False,
            False,
            False,
        )
        kernel = quant_kernel.get_swiglu_forward_and_per_token_cast_kernel(
            hidden,
            False,
            False,
            swiglu_limit is not None,
            False,
            in_dtype=T.dtype(input.dtype),
            out_config=out_config,
            num_sms=None,
        )
        kernel(
            input,
            output,
            output_scale,
            None,
            None,
            None,
            None,
            0.0 if swiglu_limit is None else float(swiglu_limit),
        )
    except Exception:
        return False
    return True

def _try_tile_swiglu_expert_post_quant_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    swiglu_limit: Optional[float],
) -> bool:
    # The packed TileKernels route below builds a compact expert-token list via
    # bool-mask indexing. On MUSA this lowers to aten::nonzero/AddBaseSumFusedNonzero
    # plus host-visible count handling, which is too expensive for EP decode.
    # Keep this helper CPU-test-only until a native masked kernel consumes
    # masked_m directly without materializing valid rows.
    if input.device.type == "musa":
        return False

    if input.dim() != 3 or output.dim() != 3 or output_scale.dim() != 3:
        return False
    if masked_m.dim() != 1 or masked_m.shape[0] != input.shape[0]:
        return False
    if output.shape != (input.shape[0], input.shape[1], input.shape[2] // 2):
        return False
    if output_scale.shape != (input.shape[0], input.shape[1], input.shape[2] // (2 * quant_group_size)):
        return False

    valid_counts = masked_m.to(torch.int64)
    if input.device.type != "musa" and (torch.any(valid_counts < 0) or torch.any(valid_counts > input.shape[1])):
        return False

    expert_ids = torch.arange(input.shape[0], device=input.device, dtype=torch.int32).view(-1, 1)
    token_ids = torch.arange(input.shape[1], device=input.device, dtype=torch.int64).view(1, -1)
    valid_mask = token_ids < valid_counts.view(-1, 1)
    if input.device.type != "musa" and not torch.any(valid_mask):
        return True

    packed_input = input[valid_mask].contiguous()
    packed_pos_to_expert = expert_ids.expand(-1, input.shape[1])[valid_mask]
    total_valid = packed_input.shape[0]

    try:
        tile_quant = importlib.import_module("tile_kernels.quant")
        quantized, scale = tile_quant.swiglu_forward_and_per_token_cast(
            x=packed_input,
            fmt="e4m3",
            num_per_channels=quant_group_size,
            pos_to_expert=packed_pos_to_expert,
            pos_to_token_topk=None,
            topk_weights=None,
            swiglu_clamp_value=swiglu_limit,
            use_tma_aligned_col_major_sf=False,
            round_sf=False,
            use_packed_ue8m0=False,
            clamped_count=None,
        )
    except Exception:
        return False

    if quantized.shape != (total_valid, output.shape[-1]) or scale.numel() != total_valid * output_scale.shape[-1]:
        return False

    output.view(torch.uint8)[valid_mask] = quantized.reshape(total_valid, output.shape[-1]).view(torch.uint8)
    output_scale[valid_mask] = scale.to(output_scale.dtype).reshape(total_valid, output_scale.shape[-1])
    return True

def _tile_swiglu_forward(
    input: torch.Tensor,
    swiglu_limit: Optional[float],
) -> torch.Tensor:
    try:
        tile_swiglu = importlib.import_module("tile_kernels.torch.swiglu")
        return tile_swiglu.swiglu_forward(
            input.contiguous(),
            swiglu_clamp_value=swiglu_limit,
        )
    except Exception:
        gate, up = input.chunk(2, dim=-1)
        gate = gate.float()
        up = up.float()
        if swiglu_limit is not None:
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        return F.silu(gate) * up

def silu_and_mul_clamp_musa(input: torch.Tensor, output: torch.Tensor, swiglu_limit: float) -> None:
    silu_and_mul_musa(input, output, swiglu_limit)

def _quantize_fp8_grouped(
    value: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool,
    transposed: bool,
) -> None:
    # TODO(dsv4-musa): defer UE8M0/transposed scale layouts until a SM100-style
    # MUSA target is required; current MUSA alignment target is SM90.
    if scale_ue8m0:
        raise NotImplementedError("DeepSeekV4 MUSA UE8M0 scale quantization is not implemented yet")
    if transposed:
        raise NotImplementedError("DeepSeekV4 MUSA transposed FP8 scale layout is not implemented yet")

    used_tile_cast = False
    if value.dim() == 2 and output.dim() == 2 and output_scale.dim() == 2:
        try:
            tile_cast = importlib.import_module("tile_kernels.torch.cast")
            quantized, scale = tile_cast.cast(
                value.contiguous(),
                "e4m3",
                block_size=(1, quant_group_size),
                round_sf=False,
                use_tma_aligned_col_major_sf=False,
                use_packed_ue8m0=False,
            )
            output.copy_(quantized.reshape_as(output))
            output_scale.copy_(scale.to(output_scale.dtype).reshape_as(output_scale))
            used_tile_cast = True
        except Exception:
            used_tile_cast = False

    if used_tile_cast:
        return

    reshaped = value.float().reshape(value.shape[0], -1, quant_group_size)
    scale = reshaped.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
    quantized = torch.clamp(reshaped / scale.unsqueeze(-1), -448.0, 448.0).reshape_as(output)
    output.copy_(quantized.to(output.dtype))
    output_scale.copy_(scale.to(output_scale.dtype).reshape_as(output_scale))

def silu_and_mul_contig_post_quant_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    scale_ue8m0: bool = False,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    # TODO(dsv4-musa): replace tile_kernels dependency with a native MUSA swiglu quant path.
    if swizzle:
        raise NotImplementedError("DeepSeekV4 MUSA swizzled contiguous Silu+quant is not implemented yet")
    if scale_ue8m0:
        raise NotImplementedError("DeepSeekV4 MUSA UE8M0 scale quantization is not implemented yet")
    if transposed:
        raise NotImplementedError("DeepSeekV4 MUSA transposed FP8 scale layout is not implemented yet")
    if _try_tile_swiglu_per_token_cast_prealloc_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        swiglu_limit,
    ):
        return
    if _try_tile_swiglu_per_token_cast_musa(
        input,
        output,
        output_scale,
        quant_group_size,
        swiglu_limit,
    ):
        return
    if input.device.type == "musa":
        raise NotImplementedError(
            "DeepSeekV4 MUSA contiguous Silu+quant requires tile_kernels.quant.swiglu_forward_and_per_token_cast"
        )
    value = _tile_swiglu_forward(input, swiglu_limit)
    _quantize_fp8_grouped(value, output, output_scale, quant_group_size, scale_ue8m0, transposed)

def silu_and_mul_masked_post_quant_musa(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: Optional[float] = None,
    swizzle: bool = False,
) -> None:
    _ = topk
    # TODO(dsv4-musa): add native masked swizzle support. Keep UE8M0/transposed
    # scale layouts fail-closed while current MUSA target aligns to SM90.
    if swizzle:
        raise NotImplementedError("DeepSeekV4 MUSA swizzled masked Silu+quant is not implemented yet")
    if scale_ue8m0:
        raise NotImplementedError("DeepSeekV4 MUSA UE8M0 scale quantization is not implemented yet")
    if transposed:
        raise NotImplementedError("DeepSeekV4 MUSA transposed FP8 scale layout is not implemented yet")
    if masked_m.numel() == 1:
        if input.dim() == 2 and output.dim() == 2 and output_scale.dim() == 2:
            if _try_tile_swiglu_expert_post_quant_musa(
                input.unsqueeze(0),
                output.unsqueeze(0),
                output_scale.unsqueeze(0),
                quant_group_size,
                masked_m.reshape(1),
                swiglu_limit,
            ):
                return
            if input.device.type == "musa":
                raise NotImplementedError(
                    "DeepSeekV4 MUSA scalar masked Silu+quant requires tile_kernels.quant.swiglu_forward_and_per_token_cast"
                )
        valid_rows = int(masked_m.item())
        if valid_rows <= 0:
            return
        silu_and_mul_contig_post_quant_musa(
            input[:valid_rows],
            output[:valid_rows],
            output_scale[:valid_rows],
            quant_group_size,
            scale_ue8m0,
            transposed,
            swiglu_limit,
            swizzle,
        )
        return

    if input.dim() != 3 or output.dim() != 3 or output_scale.dim() != 3:
        raise NotImplementedError(
            "DeepSeekV4 MUSA masked Silu+quant expects 3D expert-major tensors when masked_m is per-expert"
        )
    if masked_m.shape[0] != input.shape[0]:
        raise ValueError(
            f"DeepSeekV4 MUSA masked Silu+quant expected masked_m shape ({input.shape[0]},), got {tuple(masked_m.shape)}"
        )

    if not scale_ue8m0 and not transposed:
        if _try_tile_swiglu_expert_post_quant_musa(
            input,
            output,
            output_scale,
            quant_group_size,
            masked_m,
            swiglu_limit,
        ):
            return
        if input.device.type == "musa":
            raise NotImplementedError(
                "DeepSeekV4 MUSA masked Silu+quant requires tile_kernels.quant.swiglu_forward_and_per_token_cast"
            )

    row_ids = torch.arange(input.shape[1], device=input.device, dtype=masked_m.dtype).view(1, -1)
    valid_mask = row_ids < masked_m.to(device=input.device, dtype=row_ids.dtype).view(-1, 1)
    packed_input = input[valid_mask]
    if packed_input.numel() == 0:
        return

    value = _tile_swiglu_forward(packed_input, swiglu_limit)
    reshaped = value.float().reshape(value.shape[0], -1, quant_group_size)
    scale = reshaped.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
    quantized = torch.clamp(reshaped / scale.unsqueeze(-1), -448.0, 448.0).reshape(packed_input.shape[0], output.shape[-1])
    output.view(torch.uint8).reshape(*output.shape)[valid_mask] = quantized.to(output.dtype).view(torch.uint8).reshape(packed_input.shape[0], output.shape[-1])
    output_scale[valid_mask] = scale.to(output_scale.dtype).reshape(packed_input.shape[0], output_scale.shape[-1])

__all__ = [
    '_try_tile_swiglu_per_token_cast_musa',
    '_try_tile_swiglu_per_token_cast_prealloc_musa',
    '_try_tile_swiglu_expert_post_quant_musa',
    '_tile_swiglu_forward',
    'silu_and_mul_musa',
    'silu_and_mul_clamp_musa',
    '_quantize_fp8_grouped',
    'silu_and_mul_contig_post_quant_musa',
    'silu_and_mul_masked_post_quant_musa',
]
