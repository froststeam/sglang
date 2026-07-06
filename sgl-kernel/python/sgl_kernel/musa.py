from typing import Optional, Union

import torch
from sgl_kernel.utils import _to_tensor_scalar_tuple


def musa_batched_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
) -> None:
    return torch.ops.sgl_kernel.musa_batched_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        rot_dim,
        cos_sin_cache_offsets,
    )


def musa_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return torch.ops.sgl_kernel.musa_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )


def musa_fused_mul_add(
    self: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: Optional[float],
    accurate: bool = True,
):
    # if accurate == False, then we call inplace op: bias += (self * scale)
    if not accurate:
        bias.add_(self, alpha=scale)
        return bias

    # otherwise, we call custom outplace op, act: output = self * scale + bias
    output = torch.empty_like(self)
    torch.ops.sgl_kernel.musa_fused_mul_add(output, self, bias, scale)

    return output


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_k_renorm_probs.default(
        probs, renorm_probs, maybe_top_k_arr, top_k_val
    )
    return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


def _top_p_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    renorm_probs = torch.empty_like(probs)
    torch.ops.sgl_kernel.top_p_renorm_probs.default(
        probs, renorm_probs, maybe_top_p_arr, top_p_val
    )
    return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.top_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
        generator,
    )
    return samples


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(top_p), deterministic, generator
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
    maybe_top_p_arr = maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.musa_top_k_top_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
        generator,
    )
    return samples


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs,
            top_p,
            indices,
            deterministic,
            generator=generator,
            check_nan=check_nan,
        )
    if filter_apply_order == "joint":
        if check_nan and torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
        return _top_k_top_p_sampling_from_probs_internal(
            probs,
            indices,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
            generator,
        )
    raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def _min_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor],
    maybe_min_p_arr: Optional[torch.Tensor],
    min_p_val: float,
    deterministic: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    device = probs.device
    probs = probs.float()
    maybe_min_p_arr = maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
    samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.min_p_sampling_from_probs.default(
        probs,
        samples,
        indices,
        maybe_min_p_arr,
        min_p_val,
        deterministic,
        generator,
    )
    return samples


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
) -> torch.Tensor:
    if check_nan and torch.any(torch.isnan(probs)):
        raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, indices, *_to_tensor_scalar_tuple(min_p), deterministic, generator
    )
