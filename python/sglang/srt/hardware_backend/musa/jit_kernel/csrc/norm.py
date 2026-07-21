from __future__ import annotations

from pathlib import Path

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.memory import (
    store_cache as memory_store_cache,
)
from sglang.srt.utils.custom_op import register_custom_op


@cache_once
def _tilelang_musa_cflags() -> tuple[str, ...]:
    import tilelang

    tilelang_dir = Path(tilelang.__file__).resolve().parent
    return (
        f"-I{(tilelang_dir / 'src').resolve()}",
        f"-I{(tilelang_dir / '3rdparty' / 'mutlass' / 'include').resolve()}",
        "-Wno-error=address-of-temporary",
        "-fmusa-flush-denormals-to-zero",
        "-fno-signed-zeros",
        "-mllvm",
        "-mtgpu-opt-level=1",
        "-mllvm",
        "-mtgpu-load-store-opt=1",
        "-mllvm",
        "-mtgpu-fold-global-ldst=1",
        "-mllvm",
        "-mtgpu-load-cluster-mutation=1",
        "-mllvm",
        "-mtgpu-store-cluster-mutation=1",
        "-mllvm",
        "-mtgpu-memory-sched-mutation=1",
        "-mllvm",
        "-mtgpu-alloc-shared-memory-from-zero=1",
    )


@cache_once
def _rmsnorm_module():
    return load_musa_jit(
        "sglang_musa_norm_rmsnorm",
        ("norm/rmsnorm.mu",),
        extra_musa_cflags=_tilelang_musa_cflags(),
    )


@cache_once
def _qk_mrope_module():
    return load_musa_jit(
        "sglang_musa_norm_qk_mrope",
        ("norm/qk_mrope.mu",),
        extra_musa_cflags=_tilelang_musa_cflags(),
    )


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_custom(input, weight, out, float(eps), False)
    return out


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
    enable_pdl: bool | None = None,
) -> torch.Tensor:
    _ = enable_pdl
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm_custom(input, weight, out, float(eps), True)
    return out


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    _ = enable_pdl
    _fused_add_rmsnorm_custom(input, residual, weight, float(eps), False)


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool | None = None,
) -> None:
    _ = enable_pdl
    _fused_add_rmsnorm_custom(input, residual, weight, float(eps), True)


def fused_qk_rmsnorm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _fused_qk_rmsnorm_mrope_custom(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )
    return q_out, k_out


def fused_qk_rmsnorm_mrope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
    gemma: bool = False,
) -> torch.Tensor:
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _fused_qk_rmsnorm_mrope_cache_custom(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        True,
        float(eps),
        bool(gemma),
    )
    return q_out


def fused_qk_mrope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.shape[0] == 0:
        return torch.empty_like(q), torch.empty_like(k)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    # The no-norm specialization ignores these two arguments. Reusing an
    # existing contiguous head avoids allocating dummy tensors during graph
    # capture while preserving the shared JIT entrypoint's input contract.
    q_weight_placeholder = q[0, 0]
    k_weight_placeholder = k[0, 0]
    _fused_qk_rmsnorm_mrope_cache_out_custom(
        q,
        k,
        v,
        q_weight_placeholder,
        k_weight_placeholder,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        False,
        0.0,
        False,
    )
    return q_out, k_out


def fused_qk_rmsnorm_mrope_cache_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float = 1e-6,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_out = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _fused_qk_rmsnorm_mrope_cache_out_custom(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        True,
        float(eps),
        bool(gemma),
    )
    return q_out, k_out


def _try_fused_qk_rmsnorm_mrope_no_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    mrope_section: list[int] | tuple[int, int, int] | None,
    is_interleaved: bool,
    mrope_interleaved_glm: bool = False,
    eps: float = 1e-6,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    rot_dim = int(cos_sin_cache.size(1)) if cos_sin_cache.dim() == 2 else 0
    if not (
        num_heads > 0
        and num_kv_heads > 0
        and head_dim > 0
        and positions.dim() == 2
        and positions.size(0) == 3
        and mrope_section is not None
        and len(mrope_section) == 3
        and not mrope_interleaved_glm
        and q.dim() == 2
        and k.dim() == 2
        and q.shape[0] == k.shape[0]
        and positions.size(1) == q.shape[0]
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and cos_sin_cache.dtype == q.dtype
        and q.size(-1) == num_heads * head_dim
        and k.size(-1) == num_kv_heads * head_dim
        and q_weight.numel() == head_dim
        and k_weight.numel() == head_dim
        and rot_dim > 0
        and rot_dim <= head_dim
        and rot_dim % 2 == 0
        and sum(mrope_section) == rot_dim // 2
    ):
        return None

    num_tokens = q.shape[0]
    fused_q, fused_k = fused_qk_rmsnorm_mrope(
        q=q.reshape(num_tokens, num_heads, head_dim),
        k=k.reshape(num_tokens, num_kv_heads, head_dim),
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox,
        mrope_section_t=mrope_section[0],
        mrope_section_h=mrope_section[1],
        mrope_section_w=mrope_section[2],
        is_interleaved=is_interleaved,
        eps=eps,
        gemma=gemma,
    )
    return fused_q.reshape(q.shape), fused_k.reshape(k.shape)


def _try_fused_qk_rmsnorm_mrope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    mrope_section: list[int] | tuple[int, int, int] | None,
    is_interleaved: bool,
    mrope_interleaved_glm: bool = False,
    eps: float = 1e-6,
    gemma: bool = False,
) -> torch.Tensor | None:
    row_dim = int(num_kv_heads) * int(head_dim)
    rot_dim = int(cos_sin_cache.size(1)) if cos_sin_cache.dim() == 2 else 0
    if not (
        indices is not None
        and row_dim > 0
        and num_heads > 0
        and positions.dim() == 2
        and positions.size(0) == 3
        and mrope_section is not None
        and len(mrope_section) == 3
        and not mrope_interleaved_glm
        and q.dim() == 2
        and k.dim() == 2
        and v.dim() == 2
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and k_cache.dtype == q.dtype
        and v_cache.dtype == q.dtype
        and q.size(-1) == num_heads * head_dim
        and k.size(-1) == num_kv_heads * head_dim
        and v.size(-1) == num_kv_heads * head_dim
        and q_weight.numel() == head_dim
        and k_weight.numel() == head_dim
        and indices.dim() == 1
        and indices.numel() >= q.shape[0]
        and k_cache.numel() % row_dim == 0
        and v_cache.numel() % row_dim == 0
        and rot_dim > 0
        and rot_dim <= head_dim
        and rot_dim % 2 == 0
        and sum(mrope_section) == rot_dim // 2
    ):
        return None

    num_tokens = q.shape[0]
    return fused_qk_rmsnorm_mrope_cache(
        q=q.reshape(num_tokens, num_heads, head_dim),
        k=k.reshape(num_tokens, num_kv_heads, head_dim),
        v=v.reshape(num_tokens, num_kv_heads, head_dim),
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache.view(-1, row_dim),
        v_cache=v_cache.view(-1, row_dim),
        indices=indices,
        is_neox=is_neox,
        mrope_section_t=mrope_section[0],
        mrope_section_h=mrope_section[1],
        mrope_section_w=mrope_section[2],
        is_interleaved=is_interleaved,
        eps=eps,
        gemma=gemma,
    ).reshape(q.shape)


def _try_fused_qk_rmsnorm_mrope_cache_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    mrope_section: list[int] | tuple[int, int, int] | None,
    is_interleaved: bool,
    mrope_interleaved_glm: bool = False,
    eps: float = 1e-6,
    gemma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    row_dim = int(num_kv_heads) * int(head_dim)
    rot_dim = int(cos_sin_cache.size(1)) if cos_sin_cache.dim() == 2 else 0
    if not (
        indices is not None
        and row_dim > 0
        and num_heads > 0
        and positions.dim() == 2
        and positions.size(0) == 3
        and mrope_section is not None
        and len(mrope_section) == 3
        and not mrope_interleaved_glm
        and q.dim() == 2
        and k.dim() == 2
        and v.dim() == 2
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.dtype == torch.bfloat16
        and k.dtype == q.dtype
        and v.dtype == q.dtype
        and q_weight.dtype == q.dtype
        and k_weight.dtype == q.dtype
        and k_cache.dtype == q.dtype
        and v_cache.dtype == q.dtype
        and q.size(-1) == num_heads * head_dim
        and k.size(-1) == num_kv_heads * head_dim
        and v.size(-1) == num_kv_heads * head_dim
        and q_weight.numel() == head_dim
        and k_weight.numel() == head_dim
        and indices.dim() == 1
        and indices.numel() >= q.shape[0]
        and k_cache.numel() % row_dim == 0
        and v_cache.numel() % row_dim == 0
        and rot_dim > 0
        and rot_dim <= head_dim
        and rot_dim % 2 == 0
        and sum(mrope_section) == rot_dim // 2
    ):
        return None

    num_tokens = q.shape[0]
    fused_q, fused_k = fused_qk_rmsnorm_mrope_cache_out(
        q=q.reshape(num_tokens, num_heads, head_dim),
        k=k.reshape(num_tokens, num_kv_heads, head_dim),
        v=v.reshape(num_tokens, num_kv_heads, head_dim),
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        k_cache=k_cache.view(-1, row_dim),
        v_cache=v_cache.view(-1, row_dim),
        indices=indices,
        is_neox=is_neox,
        mrope_section_t=mrope_section[0],
        mrope_section_h=mrope_section[1],
        mrope_section_w=mrope_section[2],
        is_interleaved=is_interleaved,
        eps=eps,
        gemma=gemma,
    )
    return fused_q.reshape(q.shape), fused_k.reshape(k.shape)


def try_fused_qk_rmsnorm_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    mrope_section: list[int] | tuple[int, int, int] | None,
    is_interleaved: bool,
    mrope_interleaved_glm: bool = False,
    eps: float = 1e-6,
    gemma: bool = False,
    v: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    return_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    has_cache_args = (
        v is not None
        and k_cache is not None
        and v_cache is not None
        and indices is not None
    )
    if has_cache_args:
        cache_kwargs = dict(
            q=q,
            k=k,
            v=v,
            q_weight=q_weight,
            k_weight=k_weight,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            k_cache=k_cache,
            v_cache=v_cache,
            indices=indices,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_neox=is_neox,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
            mrope_interleaved_glm=mrope_interleaved_glm,
            eps=eps,
            gemma=gemma,
        )
        if return_k:
            return _try_fused_qk_rmsnorm_mrope_cache_out(**cache_kwargs)
        fused_q = _try_fused_qk_rmsnorm_mrope_cache(**cache_kwargs)
        if fused_q is None:
            return None
        return fused_q, None

    if any(x is not None for x in (v, k_cache, v_cache, indices)):
        return None

    return _try_fused_qk_rmsnorm_mrope_no_cache(
        q=q,
        k=k,
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_neox=is_neox,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        mrope_interleaved_glm=mrope_interleaved_glm,
        eps=eps,
        gemma=gemma,
    )


def try_fused_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_neox: bool,
    eps: float = 1e-6,
    gemma: bool = False,
    v: torch.Tensor | None = None,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    return_k: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None] | None:
    if not (
        positions.dim() == 1
        and cos_sin_cache.dim() == 2
        and is_neox
        and cos_sin_cache.size(1) % 2 == 0
    ):
        return None

    positions_3d = positions.unsqueeze(0).expand(3, -1)
    rot_half = int(cos_sin_cache.size(1)) // 2
    return try_fused_qk_rmsnorm_mrope(
        q=q,
        k=k,
        q_weight=q_weight,
        k_weight=k_weight,
        positions=positions_3d,
        cos_sin_cache=cos_sin_cache,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        is_neox=is_neox,
        mrope_section=(rot_half, 0, 0),
        is_interleaved=False,
        eps=eps,
        gemma=gemma,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        indices=indices,
        return_k=return_k,
    )


def store_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    memory_store_cache(k, v, k_cache, v_cache, indices)


@register_custom_op(
    op_name="musa_csrc_rmsnorm",
    mutates_args=["out"],
)
def _rmsnorm_custom(
    input: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _rmsnorm_module().sgl_musa_rmsnorm(input, weight, out, float(eps), bool(gemma))


@register_custom_op(
    op_name="musa_csrc_fused_add_rmsnorm",
    mutates_args=["input", "residual"],
)
def _fused_add_rmsnorm_custom(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gemma: bool,
) -> None:
    _rmsnorm_module().sgl_musa_fused_add_rmsnorm(
        input, residual, weight, float(eps), bool(gemma)
    )


@register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope",
    mutates_args=["q_out", "k_out"],
)
def _fused_qk_rmsnorm_mrope_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    eps: float,
    gemma: bool,
) -> None:
    _qk_mrope_module().sgl_musa_fused_qk_rmsnorm_mrope(
        q,
        k,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        float(eps),
        bool(gemma),
    )


@register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope_cache",
    mutates_args=["q_out", "k_cache", "v_cache"],
)
def _fused_qk_rmsnorm_mrope_cache_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    apply_rmsnorm: bool,
    eps: float,
    gemma: bool,
) -> None:
    _qk_mrope_module().sgl_musa_fused_qk_rmsnorm_mrope_cache(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        bool(apply_rmsnorm),
        float(eps),
        bool(gemma),
    )


@register_custom_op(
    op_name="musa_csrc_fused_qk_rmsnorm_mrope_cache_out",
    mutates_args=["q_out", "k_out", "k_cache", "v_cache"],
)
def _fused_qk_rmsnorm_mrope_cache_out_custom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    is_neox: bool,
    mrope_section_t: int,
    mrope_section_h: int,
    mrope_section_w: int,
    is_interleaved: bool,
    apply_rmsnorm: bool,
    eps: float,
    gemma: bool,
) -> None:
    _qk_mrope_module().sgl_musa_fused_qk_rmsnorm_mrope_cache_out(
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        k_cache,
        v_cache,
        indices,
        bool(is_neox),
        int(mrope_section_t),
        int(mrope_section_h),
        int(mrope_section_w),
        bool(is_interleaved),
        bool(apply_rmsnorm),
        float(eps),
        bool(gemma),
    )
