from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from mate import flash_attn_with_kvcache as _mate_flash_attn_with_kvcache
from mate.mha_interface import get_scheduler_metadata

from sglang.srt.distributed import get_pp_group, get_pp_indices
from sglang.srt.environ import envs

MATE_MLA_WORKSPACE_BUFFER: torch.tensor | None = None
MATE_NO_MLA_SCHEDULER_MATEDATA_DICT = dict()


# MATE's FA3-compatible wrapper; generates scheduler metadata for optimal performance.
def mate_flash_attn_with_kvcache_wrapper(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse: bool = False,
    **kwargs,
):
    layer = kwargs["layer"]
    device = kwargs["device"]
    prefix = kwargs["prefix"]
    use_mla = kwargs["use_mla"]
    can_run_tbo = kwargs["can_run_tbo"]
    max_seqlen_k = kwargs["max_seqlen_k"]
    num_hidden_layers = kwargs["num_hidden_layers"]
    first_k_dense_replace = kwargs["first_k_dense_replace"]
    full_attention_interval = kwargs["full_attention_interval"]

    current_layer_id = layer.layer_id
    batch_size = cu_seqlens_q.shape[-1] - 1
    page_size = k_cache.shape[1] if k_cache is not None else 1

    # Note: Scheduler metadata must be updated on the first call to `_mate_flash_attn_with_kvcache`
    # - Each pipeline rank updates metadata once during pipeline parallelism.
    # - Front dense layers in two-batch overlap skip this update (they don't call this method).
    # - Skip metadata update for linear attention layers in Qwen3-Next-like models based on full attention interval.
    should_update = True
    pp_group = get_pp_group()
    pp_rank = pp_group.rank_in_group
    start_layer_id, _ = get_pp_indices(
        num_hidden_layers, pp_group.rank_in_group, pp_group.world_size
    )
    if can_run_tbo and pp_rank == 0:
        start_layer_id += (
            first_k_dense_replace if first_k_dense_replace is not None else 0
        )

    if full_attention_interval is not None:
        start_layer_id += full_attention_interval - 1

    if current_layer_id > start_layer_id:
        should_update = False

    if envs.SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA.get():
        should_update = True

    if use_mla:
        global MATE_MLA_WORKSPACE_BUFFER
        if MATE_MLA_WORKSPACE_BUFFER == None:
            MATE_MLA_WORKSPACE_BUFFER = torch.empty(
                128 * 1024 * 1024, device=device, dtype=torch.uint8
            )
        scheduler_metadata = (MATE_MLA_WORKSPACE_BUFFER, should_update)
    else:
        global MATE_NO_MLA_SCHEDULER_MATEDATA_DICT
        if should_update or prefix not in MATE_NO_MLA_SCHEDULER_MATEDATA_DICT:
            MATE_NO_MLA_SCHEDULER_MATEDATA_DICT[prefix] = get_scheduler_metadata(
                batch_size=batch_size,
                num_heads_q=layer.tp_q_head_num,
                num_heads_kv=layer.tp_k_head_num,
                headdim=layer.qk_head_dim,
                headdim_v=layer.v_head_dim,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                page_size=page_size,
                causal=causal,
                window_size=window_size,
                num_splits=num_splits,
            )
        scheduler_metadata = MATE_NO_MLA_SCHEDULER_MATEDATA_DICT[prefix]

    return _mate_flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        qv=qv,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
    )


flash_attn_with_kvcache = mate_flash_attn_with_kvcache_wrapper
