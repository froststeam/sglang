from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from flash_attn_interface import flash_attn_varlen_func
from flash_attn_interface import flash_attn_with_kvcache as mate_flash_attn_with_kvcache
from flash_attn_interface import get_scheduler_metadata

from sglang.srt.distributed import get_pp_group, get_pp_indices
from sglang.srt.environ import envs
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMultiStepBackend,
    merge_state_v2_wrapper,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import (
    cp_allgather_and_save_kv_cache,
    cp_attn_forward_extend,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

MATE_MLA_WORKSPACE_SIZE_BYTES = 128 * 1024 * 1024


def flash_attn_with_kvcache(
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
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[Union[torch.Tensor, Tuple[torch.Tensor, bool]]] = None,
    num_splits: int = 0,
    pack_gqa=None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
    ver=3,
):
    """MUSA flash_attn_with_kvcache wrapper.

    MUSA FA3 callers pass scheduler metadata explicitly at each call site.
    """
    if ver != 3:
        raise ValueError("Only ver=3 is supported for MUSA FA3.")
    if score_mod is not None or aux_tensors is not None:
        raise NotImplementedError(
            "score_mod and aux_tensors are not supported by the MUSA FA3 backend."
        )

    # MATE FA3 requires scheduler metadata for paged KV-cache attention.
    # Letting this fall back to None can select an unsafe scheduling path and
    # produce numerical mismatches, so fail at the wrapper boundary.
    assert (
        scheduler_metadata is not None
    ), "MUSA MATE FA3 flash_attn_with_kvcache requires scheduler_metadata."

    return mate_flash_attn_with_kvcache(
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
        sinks=sinks,
    )


class MusaFlashAttentionBackend(FlashAttentionBackend):
    def __init__(self, model_runner: ModelRunner, **kwargs):
        fa_impl_ver = kwargs.get("fa_impl_ver", 3)
        if fa_impl_ver != 3:
            raise ValueError("MUSA flash attention backend only supports FA3.")

        super().__init__(model_runner, **kwargs)

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self.num_hidden_layers = model_runner.model_config.num_hidden_layers
        self.first_k_dense_replace = model_runner.model_config.first_k_dense_replace
        self.full_attention_interval = model_runner.model_config.full_attention_interval
        self.head_dim = model_runner.model_config.head_dim
        self.v_head_dim = model_runner.model_config.v_head_dim
        self.num_attention_heads = model_runner.model_config.get_num_attention_heads(
            model_runner.tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            model_runner.tp_size
        )
        _softcapping = getattr(
            model_runner.model_config.hf_text_config, "attn_logit_softcapping", None
        )
        self.has_softcap = _softcapping is not None and _softcapping > 0.0
        self._mate_mla_workspace_buffer = (
            torch.empty(
                MATE_MLA_WORKSPACE_SIZE_BYTES,
                device=self.device,
                dtype=torch.uint8,
            )
            if self.use_mla
            else None
        )
        # Disable default scheduler metadata for fa3
        self._get_scheduler_metadata = None

    @staticmethod
    def _has_extend_prefix(forward_batch: ForwardBatch) -> bool:
        extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        return extend_prefix_lens_cpu is not None and any(extend_prefix_lens_cpu)

    def _needs_no_mla_kvcache_scheduler_metadata(
        self, forward_batch: ForwardBatch
    ) -> bool:
        forward_mode = forward_batch.forward_mode
        return (
            forward_mode.is_decode_or_idle()
            or forward_mode.is_target_verify()
            or forward_mode.is_draft_extend()
            or self._has_extend_prefix(forward_batch)
        )

    # Scheduler metadata decision table.
    #
    # MATE FA3 kvcache attention requires scheduler_metadata. Varlen attention
    # does not use it. For no-MLA, precompute metadata only for paths that can
    # later call flash_attn_with_kvcache, while keeping separate metadata
    # objects for calls whose shape/page-table parameters differ. For MLA, do
    # not precompute at init time because the metadata tuple's skip_update flag
    # depends on the current layer.
    #
    # no-MLA paths:
    # | Forward path | Runtime call | Metadata |
    # |---|---|---|
    # | decode / idle | kvcache | `metadata.scheduler_metadata` |
    # | target verify | kvcache + expand | base + expand metadata |
    # | draft extend | kvcache | `metadata.scheduler_metadata` |
    # | extend with prefix | kvcache | `metadata.scheduler_metadata` |
    # | plain extend without prefix | varlen | clear base metadata |
    # | local attention on kvcache path | kvcache | local metadata |
    # | cross attention on kvcache path | kvcache | encoder metadata |
    # | context-parallel extend | CP kvcache | CP prev + CP next metadata |
    # | context-parallel local/SWA | CP kvcache | recompute at call site |
    #
    # Notes:
    # - base metadata: `metadata.scheduler_metadata`
    # - expand metadata: `forward_metadata_spec_decode_expand.scheduler_metadata`
    # - local metadata: `local_attn_metadata.scheduler_metadata`
    # - encoder metadata: `metadata.encoder_scheduler_metadata`
    # - CP prev/next: `metadata.cp_scheduler_metadata_prev/next`
    # - CP local/SWA recomputes at the call site because cu_seqlens, window, or
    #   page parameters differ from the precomputed base/CP metadata.
    #
    # MLA paths:
    # | Forward path | Runtime call | Metadata |
    # |---|---|---|
    # | decode / idle | MLA kvcache | call-site workspace tuple |
    # | target verify | MLA kvcache + expand | call-site workspace tuple |
    # | draft extend / draft extend v2 | MLA kvcache | call-site workspace tuple |
    # | normal MLA extend | MLA kvcache | call-site workspace tuple |
    # | prefix-cache varlen path | varlen | clear stale base metadata |
    #
    # MLA call-site workspace tuple means:
    # `(self._mate_mla_workspace_buffer, skip_update)`.
    #
    # Only the first effective MLA layer should update the MATE scheduler
    # metadata. Later layers pass skip_update=True, avoiding expensive repeated
    # metadata updates in both eager execution and cuda graph replay. The first
    # effective layer accounts for PP rank, TBO, first_k_dense_replace, and
    # full_attention_interval; SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA overrides
    # the skip decision for debugging.
    #
    # _init_mla_scheduler_metadata only clears stale metadata fields. Actual
    # MLA scheduler metadata is built at the kvcache call site, because
    # skip_update depends on layer_id.
    def _compute_no_mla_scheduler_metadata(
        self,
        *,
        max_seqlen_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k_new: Optional[torch.Tensor],
        max_seqlen_q: int,
        causal: bool,
        window_size: Tuple[int, int],
        page_size: int,
    ) -> Optional[torch.Tensor]:
        assert not self.use_mla
        if cache_seqlens is None or cu_seqlens_q is None:
            return None
        return get_scheduler_metadata(
            batch_size=cu_seqlens_q.shape[-1] - 1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max(max_seqlen_k, 1),
            num_heads_q=self.num_attention_heads,
            num_heads_kv=self.num_kv_heads,
            headdim=self.head_dim,
            headdim_v=self.v_head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k_new,
            page_size=page_size,
            causal=causal,
            window_size=window_size,
            has_softcap=self.has_softcap,
            num_splits=self.num_splits,
        )

    def _should_skip_mla_scheduler_metadata_update(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        assert self.use_mla

        should_update = True
        pp_group = get_pp_group()
        pp_rank = pp_group.rank_in_group
        start_layer_id, _ = get_pp_indices(
            self.num_hidden_layers, pp_rank, pp_group.world_size
        )
        if getattr(forward_batch, "can_run_tbo", False) and pp_rank == 0:
            start_layer_id += (
                self.first_k_dense_replace
                if self.first_k_dense_replace is not None
                else 0
            )

        if self.full_attention_interval is not None:
            start_layer_id += self.full_attention_interval - 1

        if layer.layer_id > start_layer_id:
            should_update = False

        if envs.SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA.get():
            should_update = True

        return not should_update

    def _mla_scheduler_metadata_for_kvcache(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> Tuple[torch.Tensor, bool]:
        assert self._mate_mla_workspace_buffer is not None
        return (
            self._mate_mla_workspace_buffer,
            self._should_skip_mla_scheduler_metadata_update(layer, forward_batch),
        )

    def _init_mla_scheduler_metadata(self):
        if self.forward_metadata is None:
            return

        # MLA scheduler metadata is layer-dependent. Clear stale metadata here;
        # kvcache call sites build the actual `(workspace, skip_update)` tuple.
        setattr(self.forward_metadata, "scheduler_metadata", None)
        metadata_expand = self.forward_metadata_spec_decode_expand
        if metadata_expand is not None:
            setattr(metadata_expand, "scheduler_metadata", None)

    @staticmethod
    def _set_scheduler_metadata(owner, attr: str, new_metadata):
        old_metadata = getattr(owner, attr, None)
        if isinstance(old_metadata, torch.Tensor) and isinstance(
            new_metadata, torch.Tensor
        ):
            old_metadata.copy_(new_metadata)
        else:
            setattr(owner, attr, new_metadata)

    @staticmethod
    def _max_seqlen_from_metadata(
        max_seqlen_k: int, cache_seqlens: torch.Tensor
    ) -> int:
        if max_seqlen_k and max_seqlen_k > 0:
            return max_seqlen_k
        return int(cache_seqlens.max().item())

    def _init_no_mla_scheduler_metadata(self, forward_batch: ForwardBatch):
        metadata = self.forward_metadata
        if metadata is None:
            return

        is_decode = forward_batch.forward_mode.is_decode_or_idle()
        use_cascade = self.topk > 1 and (
            (is_decode and forward_batch.spec_info is not None)
            or forward_batch.forward_mode.is_target_verify()
        )
        needs_kvcache_metadata = self._needs_no_mla_kvcache_scheduler_metadata(
            forward_batch
        )
        if needs_kvcache_metadata:
            cu_seqlens_k_new = None if is_decode else metadata.cu_seqlens_k
            self._set_scheduler_metadata(
                metadata,
                "scheduler_metadata",
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=metadata.max_seq_len_k,
                    cache_seqlens=metadata.cache_seqlens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k_new,
                    max_seqlen_q=metadata.max_seq_len_q,
                    causal=not use_cascade,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                ),
            )
        else:
            setattr(metadata, "scheduler_metadata", None)

        local_metadata = getattr(metadata, "local_attn_metadata", None)
        if needs_kvcache_metadata and local_metadata is not None:
            self._set_scheduler_metadata(
                local_metadata,
                "scheduler_metadata",
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=local_metadata.local_max_seq_len,
                    cache_seqlens=local_metadata.local_seqused_k,
                    cu_seqlens_q=local_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_metadata.local_max_query_len,
                    causal=True,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                ),
            )
        elif local_metadata is not None:
            setattr(local_metadata, "scheduler_metadata", None)

        if (
            forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
            and self.attn_cp_size > 1
        ):
            cp_meta = forward_batch.attn_cp_metadata
            device = metadata.cache_seqlens_int32.device
            cu_seqlens_q_prev = torch.tensor(
                [0, cp_meta.actual_seq_q_prev], device=device, dtype=torch.int32
            )
            cu_seqlens_q_next = torch.tensor(
                [0, cp_meta.actual_seq_q_next], device=device, dtype=torch.int32
            )

            def _cp_scheduler_metadata(
                max_seqlen_k, cache_seqlens, cu_seqlens_q, max_seqlen_q, window_size
            ):
                return self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=max_seqlen_k,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=metadata.cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    causal=True,
                    window_size=window_size,
                    page_size=self.page_size,
                )

            self._set_scheduler_metadata(
                metadata,
                "cp_scheduler_metadata_prev",
                _cp_scheduler_metadata(
                    int(cp_meta.kv_len_prev),
                    cp_meta.kv_len_prev_tensor,
                    cu_seqlens_q_prev,
                    cp_meta.actual_seq_q_prev,
                    (-1, -1),
                ),
            )
            self._set_scheduler_metadata(
                metadata,
                "cp_scheduler_metadata_next",
                _cp_scheduler_metadata(
                    int(cp_meta.kv_len_next),
                    cp_meta.kv_len_next_tensor,
                    cu_seqlens_q_next,
                    cp_meta.actual_seq_q_next,
                    (-1, -1),
                ),
            )
        else:
            setattr(metadata, "cp_scheduler_metadata_prev", None)
            setattr(metadata, "cp_scheduler_metadata_next", None)

        if needs_kvcache_metadata and metadata.encoder_lens_int32 is not None:
            encoder_max_seqlen_q = 1 if is_decode else metadata.max_seq_len_q
            self._set_scheduler_metadata(
                metadata,
                "encoder_scheduler_metadata",
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=metadata.encoder_max_seq_len_k,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=encoder_max_seqlen_q,
                    causal=False,
                    window_size=(-1, -1),
                    page_size=self.page_size,
                ),
            )
        else:
            setattr(metadata, "encoder_scheduler_metadata", None)

        metadata_expand = (
            self.forward_metadata_spec_decode_expand if use_cascade else None
        )
        if metadata_expand is not None:
            expand_page_size = (
                1 if forward_batch.forward_mode.is_target_verify() else self.page_size
            )
            self._set_scheduler_metadata(
                metadata_expand,
                "scheduler_metadata",
                self._compute_no_mla_scheduler_metadata(
                    max_seqlen_k=self._max_seqlen_from_metadata(
                        metadata_expand.max_seq_len_k,
                        metadata_expand.cache_seqlens_int32,
                    ),
                    cache_seqlens=metadata_expand.cache_seqlens_int32,
                    cu_seqlens_q=metadata_expand.cu_seqlens_q,
                    cu_seqlens_k_new=metadata_expand.cu_seqlens_k,
                    max_seqlen_q=metadata_expand.max_seq_len_q,
                    causal=False,
                    window_size=(-1, -1),
                    page_size=expand_page_size,
                ),
            )
        elif self.forward_metadata_spec_decode_expand is not None:
            setattr(
                self.forward_metadata_spec_decode_expand, "scheduler_metadata", None
            )

    def _init_scheduler_metadata(self, forward_batch: ForwardBatch):
        if self.use_mla:
            self._init_mla_scheduler_metadata()
        else:
            self._init_no_mla_scheduler_metadata(forward_batch)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._init_scheduler_metadata(forward_batch)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
    ):
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self._init_scheduler_metadata(
            SimpleNamespace(
                forward_mode=forward_mode,
                spec_info=spec_info,
                attn_cp_metadata=None,
            )
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
            out_cache_loc,
        )
        self._init_scheduler_metadata(
            SimpleNamespace(
                forward_mode=forward_mode,
                spec_info=spec_info,
                attn_cp_metadata=None,
            )
        )

    def _scheduler_metadata_for_kvcache(
        self,
        metadata,
        *,
        is_swa_layer: bool = False,
        is_cross_attention: bool = False,
        local_metadata=None,
        max_seqlen_k: Optional[int] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k_new: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        causal: bool = True,
        window_size: Tuple[int, int] = (-1, -1),
        page_size: Optional[int] = None,
    ):
        if local_metadata is not None:
            return getattr(local_metadata, "scheduler_metadata", None)
        if is_swa_layer:
            assert max_seqlen_q is not None
            assert cache_seqlens is not None
            assert cu_seqlens_q is not None
            if max_seqlen_k is None:
                max_seqlen_k = self._max_seqlen_from_metadata(0, cache_seqlens)
            return self._compute_no_mla_scheduler_metadata(
                max_seqlen_k=max_seqlen_k,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q,
                causal=causal,
                window_size=window_size,
                page_size=page_size if page_size is not None else self.page_size,
            )
        if is_cross_attention:
            return getattr(metadata, "encoder_scheduler_metadata", None)
        return getattr(metadata, "scheduler_metadata", None)

    def _cp_scheduler_metadata_for_kvcache(
        self,
        metadata,
        forward_batch: ForwardBatch,
        cache_seqlens_cp,
        is_swa_layer: bool,
        cu_seqlens_q_cp: Optional[torch.Tensor] = None,
        max_seqlen_q_cp: Optional[int] = None,
        cu_seqlens_k_new: Optional[torch.Tensor] = None,
        window_size: Tuple[int, int] = (-1, -1),
    ):
        assert not self.use_mla
        if is_swa_layer or cu_seqlens_k_new is None:
            assert cu_seqlens_q_cp is not None
            assert max_seqlen_q_cp is not None
            return self._compute_no_mla_scheduler_metadata(
                max_seqlen_k=self._max_seqlen_from_metadata(0, cache_seqlens_cp),
                cache_seqlens=cache_seqlens_cp,
                cu_seqlens_q=cu_seqlens_q_cp,
                cu_seqlens_k_new=cu_seqlens_k_new,
                max_seqlen_q=max_seqlen_q_cp,
                causal=True,
                window_size=window_size,
                page_size=self.page_size,
            )
        is_cp_prev = (
            cache_seqlens_cp is forward_batch.attn_cp_metadata.kv_len_prev_tensor
        )
        return getattr(
            metadata,
            (
                "cp_scheduler_metadata_prev"
                if is_cp_prev
                else "cp_scheduler_metadata_next"
            ),
            None,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None

            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            )

            if save_kv_cache and not is_cp_mode:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
            if is_cp_mode:
                cp_allgather_and_save_kv_cache(
                    forward_batch, layer, k, v, self.attn_cp_size
                )

        metadata = self.forward_metadata

        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_swa_layer
        )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_swa_layer and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            if is_swa_layer and self.use_sliding_window_kv_pool:
                if metadata.swa_page_table is not None:
                    page_table = metadata.swa_page_table
                else:
                    page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                        metadata.page_table
                    )
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        if not self.use_mla:
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            if (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
            ):

                def _fa_cp_attn(
                    q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
                ):
                    scheduler_metadata = self._cp_scheduler_metadata_for_kvcache(
                        metadata,
                        forward_batch,
                        cache_seqlens_cp,
                        is_swa_layer,
                        cu_seqlens_q_cp=cu_seqlens_q_cp,
                        max_seqlen_q_cp=max_seqlen_q_cp,
                        cu_seqlens_k_new=(cu_seqlens_k if not use_local_attn else None),
                        window_size=window_size,
                    )
                    return flash_attn_with_kvcache(
                        q=q_chunk,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens_cp,
                        cu_seqlens_q=cu_seqlens_q_cp,
                        cu_seqlens_k_new=(cu_seqlens_k if not use_local_attn else None),
                        max_seqlen_q=max_seqlen_q_cp,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        scheduler_metadata=scheduler_metadata,
                        num_splits=self.num_splits,
                        **kwargs,
                    )

                result = cp_attn_forward_extend(
                    forward_batch,
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.device,
                    _fa_cp_attn,
                )
            elif (
                (
                    forward_batch.extend_prefix_lens_cpu is not None
                    and any(forward_batch.extend_prefix_lens_cpu)
                )
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend()
            ):
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        is_swa_layer=is_swa_layer,
                        is_cross_attention=layer.is_cross_attention,
                        local_metadata=local_metadata if use_local_attn else None,
                        max_seqlen_k=(
                            metadata.encoder_max_seq_len_k
                            if layer.is_cross_attention
                            else self._max_seqlen_from_metadata(
                                metadata.max_seq_len_k, cache_seqlens
                            )
                        ),
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
            else:
                output = flash_attn_varlen_func(
                    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k=metadata.cu_seqlens_q,
                    max_seqlen_q=metadata.max_seq_len_q,
                    max_seqlen_k=metadata.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=True,
                    return_softmax_lse=forward_batch.mha_return_lse,
                    **kwargs,
                )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return (
                        output.view(-1, layer.tp_q_head_num * layer.v_head_dim),
                        lse,
                    )
                return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                    v_cache=value_cache.view(
                        -1, 1, layer.tp_v_head_num, layer.head_dim
                    ),
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        self.forward_metadata_spec_decode_expand,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        **kwargs,
                    )
                else:
                    cu_seqlens_k = (
                        metadata.cu_seqlens_q
                        if not forward_batch.mha_one_shot
                        else metadata.cu_seqlens_k
                    )
                    max_seqlen_k = (
                        metadata.max_seq_len_q
                        if not forward_batch.mha_one_shot
                        else metadata.max_seq_len_k
                    )
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                        **kwargs,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                        layer,
                        forward_batch,
                    ),
                    num_splits=self.num_splits,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                                layer,
                                forward_batch,
                            ),
                            num_splits=self.num_splits,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        is_swa_layer = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None

        if not self.use_mla:
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if layer.is_cross_attention:
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        is_cross_attention=True,
                        max_seqlen_k=metadata.encoder_max_seq_len_k,
                        cache_seqlens=metadata.encoder_lens_int32,
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                        max_seqlen_q=1,
                        causal=False,
                        window_size=(-1, -1),
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
            elif use_local_attn:
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        local_metadata=local_attn_metadata,
                        max_seqlen_k=local_attn_metadata.local_max_seq_len,
                        cache_seqlens=local_attn_metadata.local_seqused_k,
                        cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                        cu_seqlens_k_new=None,
                        max_seqlen_q=local_attn_metadata.local_max_query_len,
                        causal=True,
                        window_size=(-1, -1),
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is not None:
                        page_table = metadata.swa_page_table
                    else:
                        page_table = (
                            self.token_to_kv_pool.translate_loc_from_full_to_swa(
                                metadata.page_table
                            )
                        )
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                result = flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    scheduler_metadata=self._scheduler_metadata_for_kvcache(
                        metadata,
                        is_swa_layer=is_swa_layer,
                        max_seqlen_k=self._max_seqlen_from_metadata(
                            metadata.max_seq_len_k, cache_seqlens
                        ),
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k_new=None,
                        max_seqlen_q=max_seqlen_q,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                    ),
                    num_splits=self.num_splits,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            scheduler_metadata=self._scheduler_metadata_for_kvcache(
                                self.forward_metadata_spec_decode_expand,
                            ),
                            num_splits=self.num_splits,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,
                scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                    layer,
                    forward_batch,
                ),
                num_splits=self.num_splits,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    scheduler_metadata=self._mla_scheduler_metadata_for_kvcache(
                        layer,
                        forward_batch,
                    ),
                    num_splits=self.num_splits,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class MusaFlashAttentionMultiStepBackend(FlashAttentionMultiStepBackend):

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        if fa_impl_ver != 3:
            raise ValueError("MUSA flash attention backend only supports FA3.")

        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                MusaFlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )
