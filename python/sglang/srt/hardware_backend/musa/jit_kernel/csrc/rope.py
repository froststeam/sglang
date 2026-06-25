from __future__ import annotations

import os

import torch

from sglang.jit_kernel.utils import cache_once
from sglang.srt.hardware_backend.musa.jit_kernel.csrc.jit import load_musa_jit
from sglang.srt.utils.custom_op import register_custom_op


def _layout_strides(
    tensor: torch.Tensor,
    positions_ndim: int,
    head_size: int,
) -> tuple[int, int, int, int]:
    if tensor.dim() not in (positions_ndim + 1, positions_ndim + 2):
        raise ValueError(
            "tensor must have shape [..., hidden_size] "
            "or [..., num_heads, head_size]"
        )

    batch_stride = tensor.stride(0) if positions_ndim == 2 else 0
    token_stride = tensor.stride(positions_ndim - 1)
    if tensor.dim() == positions_ndim + 2:
        head_stride = tensor.stride(-2)
        dim_stride = tensor.stride(-1)
    else:
        dim_stride = tensor.stride(-1)
        head_stride = head_size * dim_stride
    return batch_stride, token_stride, head_stride, dim_stride


def _store_hint(num_tokens: int, rot_dim: int) -> str:
    hint = os.getenv("SGLANG_MUSA_ROPE_STORE_HINT", "auto").lower()
    if hint in {"default", "stwb", "stcg", "stcs"}:
        return hint
    if hint != "auto":
        return "stwb"
    return "stwb"


def _musa_arch_tag() -> str:
    try:
        major, minor = torch.musa.get_device_capability()
        return f"mp{int(major)}{int(minor)}"
    except Exception:
        return "mp31"


@cache_once
def _rope_module(arch_tag: str, store_hint: str):
    hint_value = {"default": 0, "stwb": 1, "stcg": 2, "stcs": 3}[store_hint]
    arch_mp31 = 1 if arch_tag == "mp31" else 0
    return load_musa_jit(
        f"sglang_musa_rope_{arch_tag}_{store_hint}",
        ("rope/rotary_embedding.mu",),
        extra_musa_cflags=(
            f"-DSGLANG_MUSA_ROPE_STORE_HINT={hint_value}",
            f"-DSGLANG_MUSA_ROPE_ARCH_MP31={arch_mp31}",
        ),
    )


def _rotary_embedding_kernel(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_tokens: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    seq_len: int,
    query_batch_stride: int,
    key_batch_stride: int,
    query_stride: int,
    key_stride: int,
    query_head_stride: int,
    key_head_stride: int,
    query_dim_stride: int,
    key_dim_stride: int,
    is_neox: bool,
    has_key: bool,
) -> None:
    _rope_module(
        _musa_arch_tag(), _store_hint(int(num_tokens), int(rot_dim))
    ).sgl_rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(seq_len),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
        bool(has_key),
    )


def _rotary_embedding_cache_kernel(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    query_stride: int,
    key_stride: int,
    query_head_stride: int,
    key_head_stride: int,
    query_dim_stride: int,
    key_dim_stride: int,
    is_neox: bool,
) -> None:
    _rope_module(
        _musa_arch_tag(), _store_hint(int(num_tokens), int(rot_dim))
    ).sgl_rotary_embedding_cache(
        positions,
        query,
        key,
        value,
        cos_sin_cache,
        k_cache,
        v_cache,
        indices,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
    )


def _rotary_embedding_cache_qout_kernel(
    positions: torch.Tensor,
    query: torch.Tensor,
    query_out: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    num_tokens: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    rot_dim: int,
    query_stride: int,
    key_stride: int,
    query_head_stride: int,
    key_head_stride: int,
    query_dim_stride: int,
    key_dim_stride: int,
    is_neox: bool,
) -> None:
    _rope_module(
        _musa_arch_tag(), _store_hint(int(num_tokens), int(rot_dim))
    ).sgl_rotary_embedding_cache_qout(
        positions,
        query,
        query_out,
        key,
        value,
        cos_sin_cache,
        k_cache,
        v_cache,
        indices,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
    )


def _rotary_pos_emb_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    num_tokens: int,
    seq_len: int,
    head_size: int,
    num_heads: int,
    num_kv_heads: int,
    query_batch_stride: int,
    key_batch_stride: int,
    query_out_batch_stride: int,
    key_out_batch_stride: int,
    query_stride: int,
    key_stride: int,
    query_out_stride: int,
    key_out_stride: int,
    query_head_stride: int,
    key_head_stride: int,
    query_out_head_stride: int,
    key_out_head_stride: int,
    query_dim_stride: int,
    key_dim_stride: int,
    query_out_dim_stride: int,
    key_out_dim_stride: int,
    cos_batch_stride: int,
    sin_batch_stride: int,
    cos_stride: int,
    sin_stride: int,
    cos_dim_stride: int,
    sin_dim_stride: int,
) -> None:
    _rope_module(
        _musa_arch_tag(), _store_hint(int(num_tokens), int(head_size))
    ).sgl_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
        query_out,
        key_out,
        int(num_tokens),
        int(seq_len),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_out_batch_stride),
        int(key_out_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_out_stride),
        int(key_out_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_out_head_stride),
        int(key_out_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        int(query_out_dim_stride),
        int(key_out_dim_stride),
        int(cos_batch_stride),
        int(sin_batch_stride),
        int(cos_stride),
        int(sin_stride),
        int(cos_dim_stride),
        int(sin_dim_stride),
    )


def _rotary_embedding_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    num_tokens = positions.numel()
    positions_ndim = positions.dim()
    if positions_ndim not in (1, 2):
        raise ValueError(
            "positions must have shape [num_tokens] or [batch_size, seq_len]"
        )
    if positions_ndim == 1:
        if query.size(0) != positions.size(0) or (
            key is not None and key.size(0) != positions.size(0)
        ):
            raise ValueError(
                "query, key and positions must have the same number of tokens"
            )
    else:
        if (
            query.size(0) != positions.size(0)
            or query.size(1) != positions.size(1)
            or (key is not None and key.size(0) != positions.size(0))
            or (key is not None and key.size(1) != positions.size(1))
        ):
            raise ValueError(
                "query, key and positions must have the same batch_size and seq_len"
            )

    query_hidden_size = query.numel() // num_tokens
    key_hidden_size = key.numel() // num_tokens if key is not None else 0
    if query_hidden_size % head_size != 0:
        raise ValueError("query hidden size must be divisible by head_size")
    if key is not None and key_hidden_size % head_size != 0:
        raise ValueError("key hidden size must be divisible by head_size")

    num_heads = query_hidden_size // head_size
    num_kv_heads = key_hidden_size // head_size if key is not None else num_heads
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    rot_dim = cos_sin_cache.size(1)
    if rot_dim % 2 != 0 or rot_dim > head_size:
        raise ValueError("rot_dim must be even and no larger than head_size")
    if positions.dtype != torch.long:
        raise TypeError("positions must be torch.int64")
    if query.dtype != cos_sin_cache.dtype or (
        key is not None and key.dtype != query.dtype
    ):
        raise TypeError("query, key and cos_sin_cache must have the same dtype")
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()

    (
        query_batch_stride,
        query_stride,
        query_head_stride,
        query_dim_stride,
    ) = _layout_strides(query, positions_ndim, head_size)
    if key is not None:
        (
            key_batch_stride,
            key_stride,
            key_head_stride,
            key_dim_stride,
        ) = _layout_strides(key, positions_ndim, head_size)
    else:
        key_batch_stride = query_batch_stride
        key_stride = query_stride
        key_head_stride = query_head_stride
        key_dim_stride = query_dim_stride

    seq_len = positions.size(1) if positions_ndim == 2 else 0
    _rotary_embedding_kernel(
        positions.reshape(-1),
        query,
        key if key is not None else query,
        cos_sin_cache,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(seq_len),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
        key is not None,
    )


def _rotary_pos_emb_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    unsqueeze_dim: int,
) -> None:
    if query.dim() == 3 and key.dim() == 3:
        batch_size = 1
        seq_len = query.size(0)
        num_heads = query.size(1)
        num_kv_heads = key.size(1)
        head_size = query.size(2)
        query_batch_stride = key_batch_stride = 0
        query_out_batch_stride = key_out_batch_stride = 0
        query_stride, query_head_stride, query_dim_stride = query.stride()
        key_stride, key_head_stride, key_dim_stride = key.stride()
        query_out_stride, query_out_head_stride, query_out_dim_stride = (
            query_out.stride()
        )
        key_out_stride, key_out_head_stride, key_out_dim_stride = key_out.stride()
    elif query.dim() == 4 and key.dim() == 4 and unsqueeze_dim == 2:
        batch_size, seq_len, num_heads, head_size = query.shape
        num_kv_heads = key.size(2)
        if key.shape[:2] != (batch_size, seq_len) or key.size(3) != head_size:
            raise ValueError("query and key shapes are incompatible")
        (
            query_batch_stride,
            query_stride,
            query_head_stride,
            query_dim_stride,
        ) = query.stride()
        key_batch_stride, key_stride, key_head_stride, key_dim_stride = key.stride()
        (
            query_out_batch_stride,
            query_out_stride,
            query_out_head_stride,
            query_out_dim_stride,
        ) = query_out.stride()
        (
            key_out_batch_stride,
            key_out_stride,
            key_out_head_stride,
            key_out_dim_stride,
        ) = key_out.stride()
    elif query.dim() == 4 and key.dim() == 4 and unsqueeze_dim == 1:
        batch_size, num_heads, seq_len, head_size = query.shape
        num_kv_heads = key.size(1)
        if (
            key.size(0) != batch_size
            or key.size(2) != seq_len
            or key.size(3) != head_size
        ):
            raise ValueError("query and key shapes are incompatible")
        (
            query_batch_stride,
            query_head_stride,
            query_stride,
            query_dim_stride,
        ) = query.stride()
        key_batch_stride, key_head_stride, key_stride, key_dim_stride = key.stride()
        (
            query_out_batch_stride,
            query_out_head_stride,
            query_out_stride,
            query_out_dim_stride,
        ) = query_out.stride()
        (
            key_out_batch_stride,
            key_out_head_stride,
            key_out_stride,
            key_out_dim_stride,
        ) = key_out.stride()
    else:
        raise ValueError("query and key must have 3D or supported 4D HF layout")

    num_tokens = batch_size * seq_len
    if cos.dim() == 2:
        if cos.shape != (seq_len, head_size) or sin.shape != cos.shape:
            raise ValueError("cos and sin shapes are incompatible")
        cos_batch_stride = sin_batch_stride = 0
        cos_stride, cos_dim_stride = cos.stride()
        sin_stride, sin_dim_stride = sin.stride()
    elif cos.dim() == 3:
        if cos.shape != (batch_size, seq_len, head_size) or sin.shape != cos.shape:
            raise ValueError("cos and sin shapes are incompatible")
        cos_batch_stride, cos_stride, cos_dim_stride = cos.stride()
        sin_batch_stride, sin_stride, sin_dim_stride = sin.stride()
    else:
        raise ValueError(
            "cos and sin must have shape [seq_len, head_size] or [batch, seq_len, head_size]"
        )

    if head_size % 2 != 0:
        raise ValueError("head_size must be even")
    if not (
        query.dtype
        == key.dtype
        == cos.dtype
        == sin.dtype
        == query_out.dtype
        == key_out.dtype
    ):
        raise TypeError("query, key, cos, sin and outputs dtype mismatch")
    if not cos.is_contiguous():
        cos = cos.contiguous()
    if not sin.is_contiguous():
        sin = sin.contiguous()

    _rotary_pos_emb_kernel(
        query,
        key,
        cos,
        sin,
        query_out,
        key_out,
        int(num_tokens),
        int(seq_len),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(query_batch_stride),
        int(key_batch_stride),
        int(query_out_batch_stride),
        int(key_out_batch_stride),
        int(query_stride),
        int(key_stride),
        int(query_out_stride),
        int(key_out_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_out_head_stride),
        int(key_out_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        int(query_out_dim_stride),
        int(key_out_dim_stride),
        int(cos_batch_stride),
        int(sin_batch_stride),
        int(cos_stride),
        int(sin_stride),
        int(cos_dim_stride),
        int(sin_dim_stride),
    )


def _rotary_embedding_cache_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    if positions.dim() != 1:
        raise ValueError("positions must have shape [num_tokens]")
    num_tokens = positions.numel()
    if query.size(0) != num_tokens or key.size(0) != num_tokens:
        raise ValueError("query, key and positions must have the same tokens")

    query_hidden_size = query.numel() // num_tokens
    key_hidden_size = key.numel() // num_tokens
    if query_hidden_size % head_size != 0:
        raise ValueError("query hidden size must be divisible by head_size")
    if key_hidden_size % head_size != 0:
        raise ValueError("key hidden size must be divisible by head_size")

    num_heads = query_hidden_size // head_size
    num_kv_heads = key_hidden_size // head_size
    row_dim = num_kv_heads * head_size
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if value.numel() != num_tokens * row_dim:
        raise ValueError("value shape must match key hidden size")

    rot_dim = cos_sin_cache.size(1)
    if rot_dim % 2 != 0 or rot_dim > head_size:
        raise ValueError("rot_dim must be even and no larger than head_size")
    if positions.dtype != torch.long:
        raise TypeError("positions must be torch.int64")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be torch.int32 or torch.int64")
    if not (
        query.dtype
        == key.dtype
        == value.dtype
        == cos_sin_cache.dtype
        == k_cache.dtype
        == v_cache.dtype
    ):
        raise TypeError("query, key, value, cache and cos_sin_cache dtype mismatch")
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    (
        _query_batch_stride,
        query_stride,
        query_head_stride,
        query_dim_stride,
    ) = _layout_strides(query, 1, head_size)
    (
        _key_batch_stride,
        key_stride,
        key_head_stride,
        key_dim_stride,
    ) = _layout_strides(key, 1, head_size)

    _rotary_embedding_cache_kernel(
        positions,
        query,
        key,
        value.reshape(num_tokens, row_dim),
        cos_sin_cache,
        k_cache.view(-1, row_dim),
        v_cache.view(-1, row_dim),
        indices,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
    )


def _rotary_embedding_cache_qout_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    query_out: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    if positions.dim() != 1:
        raise ValueError("positions must have shape [num_tokens]")
    num_tokens = positions.numel()
    if query.size(0) != num_tokens or key.size(0) != num_tokens:
        raise ValueError("query, key and positions must have the same tokens")

    query_hidden_size = query.numel() // num_tokens
    key_hidden_size = key.numel() // num_tokens
    if query_hidden_size % head_size != 0:
        raise ValueError("query hidden size must be divisible by head_size")
    if key_hidden_size % head_size != 0:
        raise ValueError("key hidden size must be divisible by head_size")

    num_heads = query_hidden_size // head_size
    num_kv_heads = key_hidden_size // head_size
    row_dim = num_kv_heads * head_size
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if value.numel() != num_tokens * row_dim:
        raise ValueError("value shape must match key hidden size")
    if query_out.shape != (num_tokens, query_hidden_size):
        raise ValueError("query_out must have shape [num_tokens, query hidden size]")
    if not query_out.is_contiguous():
        raise ValueError("query_out must be contiguous")

    rot_dim = cos_sin_cache.size(1)
    if rot_dim != head_size or rot_dim not in (64, 128):
        raise ValueError("qout cache path requires rot_dim == head_size in (64, 128)")
    if not is_neox:
        raise ValueError("qout cache path requires neox rotary style")
    if query.dtype != torch.bfloat16:
        raise TypeError("qout cache path requires bfloat16")
    if positions.dtype != torch.long:
        raise TypeError("positions must be torch.int64")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be torch.int32 or torch.int64")
    if not (
        query.dtype
        == query_out.dtype
        == key.dtype
        == value.dtype
        == cos_sin_cache.dtype
        == k_cache.dtype
        == v_cache.dtype
    ):
        raise TypeError(
            "query, output, key, value, cache and cos_sin_cache dtype mismatch"
        )
    if not positions.is_contiguous():
        positions = positions.contiguous()
    if not cos_sin_cache.is_contiguous():
        cos_sin_cache = cos_sin_cache.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    (
        _query_batch_stride,
        query_stride,
        query_head_stride,
        query_dim_stride,
    ) = _layout_strides(query, 1, head_size)
    (
        _key_batch_stride,
        key_stride,
        key_head_stride,
        key_dim_stride,
    ) = _layout_strides(key, 1, head_size)
    if query_dim_stride != 1 or key_dim_stride != 1:
        raise ValueError("qout cache path requires head dim stride 1")

    _rotary_embedding_cache_qout_kernel(
        positions,
        query,
        query_out,
        key,
        value.reshape(num_tokens, row_dim),
        cos_sin_cache,
        k_cache.view(-1, row_dim),
        v_cache.view(-1, row_dim),
        indices,
        int(num_tokens),
        int(head_size),
        int(num_heads),
        int(num_kv_heads),
        int(rot_dim),
        int(query_stride),
        int(key_stride),
        int(query_head_stride),
        int(key_head_stride),
        int(query_dim_stride),
        int(key_dim_stride),
        bool(is_neox),
    )


@register_custom_op(
    op_name="musa_rotary_pos_emb",
    mutates_args=["query_out", "key_out"],
)
def _rotary_pos_emb_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    query_out: torch.Tensor,
    key_out: torch.Tensor,
    unsqueeze_dim: int,
) -> None:
    _rotary_pos_emb_impl(query, key, cos, sin, query_out, key_out, unsqueeze_dim)


@register_custom_op(
    op_name="musa_rotary_embedding",
    mutates_args=["query", "key"],
)
def _rotary_embedding_custom(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    _rotary_embedding_impl(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )


@register_custom_op(
    op_name="musa_rotary_embedding_cache",
    mutates_args=["query", "key", "k_cache", "v_cache"],
)
def _rotary_embedding_cache_custom(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _rotary_embedding_cache_impl(
        positions,
        query,
        key,
        value,
        head_size,
        cos_sin_cache,
        is_neox,
        k_cache,
        v_cache,
        indices,
    )


@register_custom_op(
    op_name="musa_rotary_embedding_cache_qout",
    mutates_args=["query_out", "key", "k_cache", "v_cache"],
)
def _rotary_embedding_cache_qout_custom(
    positions: torch.Tensor,
    query: torch.Tensor,
    query_out: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _rotary_embedding_cache_qout_impl(
        positions,
        query,
        query_out,
        key,
        value,
        head_size,
        cos_sin_cache,
        is_neox,
        k_cache,
        v_cache,
        indices,
    )


def rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)
    _rotary_pos_emb_custom(query, key, cos, sin, query_out, key_out, unsqueeze_dim)
    return query_out, key_out


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    _rotary_embedding_custom(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )


def rotary_embedding_cache(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _rotary_embedding_cache_custom(
        positions,
        query,
        key,
        value,
        head_size,
        cos_sin_cache,
        is_neox,
        k_cache,
        v_cache,
        indices,
    )


def rotary_embedding_cache_qout(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if positions.dim() != 1:
        raise ValueError("positions must have shape [num_tokens]")
    num_tokens = positions.numel()
    query_hidden_size = query.numel() // num_tokens
    query_out = torch.empty(
        (num_tokens, query_hidden_size), device=query.device, dtype=query.dtype
    )
    _rotary_embedding_cache_qout_custom(
        positions,
        query,
        query_out,
        key,
        value,
        head_size,
        cos_sin_cache,
        is_neox,
        k_cache,
        v_cache,
        indices,
    )
    return query_out
