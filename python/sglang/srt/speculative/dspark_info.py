from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.speculative.dflash_utils import top_k_renorm_prob, top_p_renorm_prob
from sglang.srt.speculative.triton_ops.dspark_reject_sampling import (
    chain_speculative_sampling_triton,
)


def _compute_paged_keep_slots(
    *,
    prefix_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    draft_token_num: int | torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    if page_size <= 1:
        raise ValueError(f"Expected page_size > 1, got {page_size}.")

    seq_dtype = prefix_lens.dtype
    extended_lens = prefix_lens + (
        int(draft_token_num)
        if not isinstance(draft_token_num, torch.Tensor)
        else draft_token_num.to(prefix_lens.device, dtype=prefix_lens.dtype)
    )
    new_lens = prefix_lens + commit_lens.to(seq_dtype)
    aligned_new_lens = ((new_lens + page_size - 1) // page_size) * page_size
    keep_lens = torch.minimum(aligned_new_lens, extended_lens)
    keep_slots = (keep_lens - prefix_lens).to(torch.int64)
    if not isinstance(draft_token_num, torch.Tensor):
        keep_slots.clamp_(min=0, max=int(draft_token_num))
    else:
        keep_slots.clamp_(min=0)
        keep_slots = torch.minimum(
            keep_slots,
            draft_token_num.to(keep_slots.device, dtype=keep_slots.dtype),
        )
    return keep_slots


def _compute_dspark_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, verify_token_num = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if verify_token_num <= 1:
        raise ValueError(
            f"DSpark verify_token_num must be greater than 1, got {verify_token_num}."
        )

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len, bonus.to(torch.int64)


@dataclass
class DSparkDraftInput(SpecInput):
    bonus_tokens: torch.Tensor
    target_hidden: torch.Tensor
    ctx_lens: torch.Tensor
    draft_seq_lens: torch.Tensor

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DSPARK_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return (1, 1)

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        old_ctx_lens = self.ctx_lens
        old_target_hidden = self.target_hidden

        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.ctx_lens = old_ctx_lens[new_indices]
        self.draft_seq_lens = self.draft_seq_lens[new_indices]

        if old_target_hidden is None or old_target_hidden.numel() == 0:
            self.target_hidden = old_target_hidden
            return

        old_bs = int(old_ctx_lens.shape[0])
        offsets = torch.zeros(
            (old_bs + 1,), dtype=torch.int64, device=old_ctx_lens.device
        )
        offsets[1:].copy_(old_ctx_lens.to(torch.int64).cumsum(0))

        seg_start = offsets[:-1][new_indices]
        seg_lens = old_ctx_lens[new_indices].to(torch.int64)
        max_len = int(seg_lens.max().item()) if seg_lens.numel() > 0 else 0
        if max_len <= 0:
            self.target_hidden = old_target_hidden[:0]
            return

        r = torch.arange(max_len, device=old_ctx_lens.device, dtype=torch.int64)[
            None, :
        ]
        pos2d = seg_start[:, None] + r
        mask = r < seg_lens[:, None]
        flat_pos = pos2d[mask]
        self.target_hidden = (
            old_target_hidden.index_select(0, flat_pos)
            if flat_pos.numel() > 0
            else old_target_hidden[:0]
        )

    def merge_batch(self, spec_info: "DSparkDraftInput"):
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.ctx_lens = torch.cat([self.ctx_lens, spec_info.ctx_lens], dim=0)
        self.draft_seq_lens = torch.cat(
            [self.draft_seq_lens, spec_info.draft_seq_lens], dim=0
        )
        if self.target_hidden is None or self.target_hidden.numel() == 0:
            self.target_hidden = spec_info.target_hidden
        elif (
            spec_info.target_hidden is not None and spec_info.target_hidden.numel() > 0
        ):
            self.target_hidden = torch.cat(
                [self.target_hidden, spec_info.target_hidden], dim=0
            )


@dataclass
class DSparkVerifyInput(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    draft_probs: torch.Tensor | None = None
    confidence_threshold: float = 0.0
    topk: int = 1
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    num_tokens_per_batch: int = -1
    verify_lens: Optional[torch.Tensor] = None
    accept_lens: Optional[torch.Tensor] = None
    graph_num_tokens: Optional[int] = None
    graph_slots: Optional[int] = None
    graph_verify_lens: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DSPARK_VERIFY)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def _prepare_full_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool,
    ) -> None:
        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 1:
            raise ValueError(
                f"DSPARK draft_token_num must be greater than 1, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(
            q_len, device=batch.device, dtype=torch.int32
        ).unsqueeze(1)
        for prefix_len in batch.seq_lens_cpu.tolist():
            prefix_len_i = int(prefix_len)
            kv_len = prefix_len_i + q_len
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            allow = k_idx <= (prefix_len_i + q_idx)
            mask_chunks.append(allow.flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        page_size: int,
        *,
        build_custom_mask: bool = True,
    ):
        if batch.forward_mode.is_idle():
            return

        if self.verify_lens is None:
            self._prepare_full_verify(
                batch, page_size, build_custom_mask=build_custom_mask
            )
            return

        q_lens = self.verify_lens
        assert q_lens is not None
        q_lens = q_lens.to(device=batch.device, dtype=torch.int32)
        if self.graph_num_tokens is not None:
            q_lens_list = q_lens.to(device="cpu", dtype=torch.int32).tolist()
            dummy_rows = max(
                0, int(self.graph_slots or batch.batch_size()) - batch.batch_size()
            )
            target_tokens = int(self.graph_num_tokens) - dummy_rows
            extra = target_tokens - sum(q_lens_list)
            for row in range(len(q_lens_list)):
                if extra <= 0:
                    break
                room = self.draft_token_num - int(q_lens_list[row])
                add = min(room, extra)
                q_lens_list[row] += add
                extra -= add
            if extra != 0:
                raise ValueError(
                    f"DSPARK graph tier {self.graph_num_tokens} cannot fit "
                    f"verify lengths {q_lens_list}"
                )
            q_lens = torch.tensor(q_lens_list, dtype=torch.int32, device=batch.device)
            self.graph_verify_lens = torch.cat(
                [
                    q_lens,
                    torch.ones(dummy_rows, dtype=torch.int32, device=batch.device),
                ]
            )
        self.verify_lens = q_lens
        if (
            int(q_lens.min().item()) < 1
            or int(q_lens.max().item()) > self.draft_token_num
        ):
            raise ValueError(
                f"DSPARK verify_lens must be in [1, {self.draft_token_num}], "
                f"got {q_lens.tolist()}"
            )
        candidates = self.draft_token.view(batch.batch_size(), self.draft_token_num)
        packed_input_ids = torch.cat(
            [candidates[i, : int(q_lens[i].item())] for i in range(batch.batch_size())],
            dim=0,
        )
        packed_positions = torch.cat(
            [
                self.positions.view(batch.batch_size(), self.draft_token_num)[
                    i, : int(q_lens[i].item())
                ]
                for i in range(batch.batch_size())
            ],
            dim=0,
        )
        batch.input_ids = packed_input_ids
        self.positions = packed_positions

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache, len(batch.input_ids)
            )
            end_offset = batch.seq_lens + q_lens
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + q_lens
            end_offset_cpu = prefix_lens_cpu + q_lens.to(prefix_lens_cpu.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )
        if self.graph_num_tokens is not None:
            dummy_tokens = int(self.graph_num_tokens) - len(batch.input_ids)
            if dummy_tokens < 0:
                raise ValueError(
                    f"DSPARK graph input has {len(batch.input_ids)} tokens, "
                    f"exceeding tier {self.graph_num_tokens}"
                )
            if dummy_tokens:
                batch.input_ids = torch.cat(
                    [
                        batch.input_ids,
                        torch.zeros(
                            dummy_tokens, dtype=batch.input_ids.dtype, device=batch.device
                        ),
                    ]
                )
                self.positions = torch.cat(
                    [
                        self.positions,
                        torch.zeros(
                            dummy_tokens, dtype=self.positions.dtype, device=batch.device
                        ),
                    ]
                )
                batch.out_cache_loc = torch.cat(
                    [
                        batch.out_cache_loc,
                        torch.zeros(
                            dummy_tokens,
                            dtype=batch.out_cache_loc.dtype,
                            device=batch.device,
                        ),
                    ]
                )

        if not build_custom_mask:
            self.custom_mask = None
            return

        if self.draft_token_num <= 1:
            raise ValueError(
                f"DSPARK draft_token_num must be greater than 1, got {self.draft_token_num}."
            )
        mask_chunks: List[torch.Tensor] = []
        q_lens_cpu = q_lens.to(device="cpu", dtype=torch.int32).tolist()
        q_len = int(self.draft_token_num)
        q_idx = torch.arange(q_len, device=batch.device, dtype=torch.int32).unsqueeze(1)
        for row, prefix_len in enumerate(batch.seq_lens_cpu.tolist()):
            prefix_len_i = int(prefix_len)
            row_q_len = int(q_lens_cpu[row])
            kv_len = prefix_len_i + row_q_len
            k_idx = torch.arange(
                kv_len, device=batch.device, dtype=torch.int32
            ).unsqueeze(0)
            allow = k_idx[:, :].expand(row_q_len, -1) <= (
                prefix_len_i + torch.arange(
                    row_q_len, device=batch.device, dtype=torch.int32
                ).unsqueeze(1)
            )
            mask_chunks.append(allow.flatten())
        self.custom_mask = (
            torch.cat(mask_chunks, dim=0)
            if mask_chunks
            else torch.empty((0,), dtype=torch.bool, device=batch.device)
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        if self.verify_lens is None and self.graph_verify_lens is None:
            qo_indptr = torch.arange(
                0,
                (bs + 1) * self.draft_token_num,
                step=self.draft_token_num,
                dtype=torch.int32,
                device=device,
            )
            cum_kv_seq_len = torch.zeros(
                (bs + 1,), dtype=torch.int32, device=device
            )
            paged_kernel_lens = paged_kernel_lens + self.draft_token_num
            cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indices = torch.empty(
                paged_kernel_lens_sum + self.draft_token_num * bs,
                dtype=torch.int32,
                device=device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                cum_kv_seq_len,
                None,
                kv_indices,
                req_to_token.size(1),
            )
            mask = self.custom_mask
            if mask is not None:
                mask_numel = (
                    paged_kernel_lens_sum * self.draft_token_num
                    + (self.draft_token_num**2) * bs
                )
                if mask.numel() < mask_numel:
                    mask = torch.cat(
                        [
                            mask,
                            torch.full(
                                (mask_numel - mask.numel(),),
                                True,
                                dtype=torch.bool,
                                device=device,
                            ),
                        ],
                        dim=0,
                    )
                    self.custom_mask = mask
            return kv_indices, cum_kv_seq_len, qo_indptr, mask

        q_lens = (
            self.graph_verify_lens
            if self.graph_verify_lens is not None
            else self.verify_lens
        )
        if q_lens is None:
            q_lens = torch.full(
                (bs,), self.draft_token_num, dtype=torch.int32, device=device
            )
        q_lens = q_lens.to(device=device, dtype=torch.int32)
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        qo_indptr[1:] = torch.cumsum(q_lens, dim=0)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + q_lens
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            int(paged_kernel_lens.sum().item()),
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = int(
                (paged_kernel_lens * q_lens).sum().item()
            )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask

    def _verify_sampling(
        self,
        *,
        candidates: torch.Tensor,
        target_logits: torch.Tensor,
        draft_probs: torch.Tensor,
        sampling_info,
        verify_lens_cpu: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lossless rejection sampling for packed, per-request verify rows."""
        bs = candidates.shape[0]
        device = target_logits.device
        vocab_size = int(target_logits.shape[-1])
        correct_len = torch.zeros((bs,), dtype=torch.int32, device=device)
        bonus = torch.empty((bs,), dtype=torch.int64, device=device)
        starts = [0]
        for length in verify_lens_cpu:
            starts.append(starts[-1] + int(length))

        for q_len in sorted(set(int(x) for x in verify_lens_cpu)):
            rows = [i for i, x in enumerate(verify_lens_cpu) if int(x) == q_len]
            row_ids = torch.tensor(rows, dtype=torch.long, device=device)
            logits = torch.stack(
                [target_logits[starts[row] : starts[row] + q_len] for row in rows],
                dim=0,
            )
            temps = sampling_info.temperatures.index_select(0, row_ids).float()
            target_probs = F.softmax(
                logits / temps[:, None, None].clamp_min(1e-5), dim=-1
            )
            if sampling_info.need_top_k_sampling:
                target_probs = top_k_renorm_prob(
                    target_probs.flatten(0, 1),
                    sampling_info.top_ks.index_select(0, row_ids).repeat_interleave(q_len),
                ).view(len(rows), q_len, vocab_size)
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs.flatten(0, 1),
                    sampling_info.top_ps.index_select(0, row_ids).repeat_interleave(q_len),
                ).view(len(rows), q_len, vocab_size)
            if q_len == 1:
                group_bonus = torch.multinomial(
                    target_probs[:, 0, :], num_samples=1
                ).squeeze(-1)
                bonus.index_copy_(0, row_ids, group_bonus.to(torch.int64))
                continue

            group_bs = len(rows)
            group_candidates = candidates.index_select(0, row_ids)[:, :q_len]
            group_draft_probs = draft_probs.index_select(0, row_ids)[:, : q_len - 1]
            retrieve_index = torch.arange(
                group_bs * q_len, dtype=torch.int32, device=device
            ).view(group_bs, q_len)
            retrieve_next_token = torch.full_like(retrieve_index, -1)
            retrieve_next_token[:, :-1] = torch.arange(
                1, q_len, dtype=torch.int32, device=device
            )
            retrieve_next_sibling = torch.full_like(retrieve_index, -1)
            predicts = torch.empty(
                (group_bs * q_len,), dtype=torch.int64, device=device
            )
            accept_index = torch.empty_like(retrieve_index)
            group_correct_len = torch.empty(
                (group_bs,), dtype=torch.int32, device=device
            )
            chain_speculative_sampling_triton(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=group_correct_len,
                candidates=group_candidates.to(torch.int64),
                retrive_index=retrieve_index,
                retrive_next_token=retrieve_next_token,
                retrive_next_sibling=retrieve_next_sibling,
                uniform_samples=torch.rand(
                    (group_bs, q_len - 1), dtype=torch.float32, device=device
                ),
                uniform_samples_for_final_sampling=torch.rand(
                    (group_bs,), dtype=torch.float32, device=device
                ),
                target_probs=target_probs,
                draft_probs=group_draft_probs,
                threshold_single=1.0,
                threshold_acc=1.0,
                deterministic=True,
            )
            group_rows = torch.arange(group_bs, dtype=torch.long, device=device)
            accept_pos = accept_index[
                group_rows, group_correct_len.to(torch.long)
            ].to(torch.long)
            correct_len.index_copy_(0, row_ids, group_correct_len)
            bonus.index_copy_(0, row_ids, predicts[accept_pos].to(torch.int64))
        return correct_len, bonus

    def verify(
        self,
        *,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        if batch.forward_mode.is_idle():
            empty = torch.empty((0,), dtype=torch.int64, device=batch.device)
            return empty, empty.to(torch.int32), empty, []

        bs = batch.batch_size()
        device = logits_output.next_token_logits.device
        candidates = self.draft_token.view(bs, self.draft_token_num)
        if self.verify_lens is None:
            model_lens_cpu = [self.draft_token_num] * bs
        else:
            model_lens_cpu = self.verify_lens.to(
                device="cpu", dtype=torch.int32
            ).tolist()
        if self.accept_lens is None:
            verify_lens_cpu = model_lens_cpu
        else:
            verify_lens_cpu = self.accept_lens.to(
                device="cpu", dtype=torch.int32
            ).tolist()
        target_logits = logits_output.next_token_logits
        if len(set(model_lens_cpu)) == 1 and model_lens_cpu[0] == self.draft_token_num:
            target_predict = torch.argmax(target_logits, dim=-1).view(bs, self.draft_token_num)
        else:
            target_predict = torch.full(
                (bs, self.draft_token_num),
                -1,
                dtype=torch.long,
                device=device,
            )
            cursor = 0
            for row, row_len in enumerate(model_lens_cpu):
                target_predict[row, :row_len] = torch.argmax(
                    target_logits[cursor : cursor + row_len], dim=-1
                )
                cursor += row_len
        greedy_correct_len, greedy_bonus = _compute_dspark_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        sampling_info = batch.sampling_info
        ragged_sampling_result = None
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and self.verify_lens is not None
            and (
                len(set(verify_lens_cpu)) > 1
                or verify_lens_cpu[0] != self.draft_token_num
            )
            and self.draft_probs is not None
        ):
            ragged_sampling_result = self._verify_sampling(
                candidates=candidates,
                target_logits=target_logits,
                draft_probs=self.draft_probs,
                sampling_info=sampling_info,
                verify_lens_cpu=verify_lens_cpu,
            )
            ragged_sampling_result = (
                torch.where(
                    (sampling_info.top_ks <= 1).view(-1),
                    greedy_correct_len,
                    ragged_sampling_result[0],
                ),
                torch.where(
                    (sampling_info.top_ks <= 1).view(-1),
                    greedy_bonus,
                    ragged_sampling_result[1],
                ),
            )
            packed_target_logits = target_logits
            target_logits = target_logits.new_zeros(
                (bs * self.draft_token_num, target_logits.shape[-1])
            )
            cursor = 0
            for row, row_len in enumerate(verify_lens_cpu):
                target_logits[
                    row * self.draft_token_num : row * self.draft_token_num + row_len
                ].copy_(packed_target_logits[cursor : cursor + row_len])
                cursor += row_len
        if sampling_info is None or sampling_info.is_all_greedy:
            correct_len, bonus = greedy_correct_len, greedy_bonus
        else:
            if self.draft_probs is None:
                raise RuntimeError(
                    "DSPARK sampling verify requires proposal draft probabilities."
                )
            vocab_size = int(target_logits.shape[-1])
            sampling_target_logits = target_logits
            if not (
                len(set(verify_lens_cpu)) == 1
                and verify_lens_cpu[0] == self.draft_token_num
            ):
                sampling_target_logits = target_logits.new_zeros(
                    (bs * self.draft_token_num, vocab_size)
                )
                cursor = 0
                for row, row_len in enumerate(verify_lens_cpu):
                    sampling_target_logits[
                        row * self.draft_token_num : row * self.draft_token_num + row_len
                    ].copy_(target_logits[cursor : cursor + row_len])
                    cursor += row_len
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures.to(torch.float32).clamp_min(1e-5),
                self.draft_token_num,
                dim=0,
            )
            target_probs = F.softmax(
                sampling_target_logits.float() / expanded_temperature, dim=-1
            )
            if sampling_info.need_top_k_sampling:
                target_probs = top_k_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ks, self.draft_token_num, dim=0
                    ),
                )
            if sampling_info.need_top_p_sampling:
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.view(
                bs, self.draft_token_num, vocab_size
            ).contiguous()
            draft_probs = self.draft_probs
            if draft_probs.shape != (
                bs,
                self.draft_token_num - 1,
                vocab_size,
            ):
                raise RuntimeError(
                    "DSPARK draft/target probability shape mismatch: "
                    f"draft={tuple(draft_probs.shape)}, "
                    f"expected={(bs, self.draft_token_num - 1, vocab_size)}."
                )

            retrieve_index = torch.arange(
                bs * self.draft_token_num, dtype=torch.int32, device=device
            ).view(bs, self.draft_token_num)
            retrieve_next_token = torch.full_like(retrieve_index, -1)
            retrieve_next_token[:, :-1] = torch.arange(
                1, self.draft_token_num, dtype=torch.int32, device=device
            )
            retrieve_next_sibling = torch.full_like(retrieve_index, -1)
            predicts = torch.empty(
                (bs * self.draft_token_num,), dtype=torch.int64, device=device
            )
            accept_index = torch.empty_like(retrieve_index)
            sampling_correct_len = torch.empty(
                (bs,), dtype=torch.int32, device=device
            )
            chain_speculative_sampling_triton(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=sampling_correct_len,
                candidates=candidates.to(torch.int64),
                retrive_index=retrieve_index,
                retrive_next_token=retrieve_next_token,
                retrive_next_sibling=retrieve_next_sibling,
                uniform_samples=torch.rand(
                    (bs, self.draft_token_num - 1),
                    dtype=torch.float32,
                    device=device,
                ),
                uniform_samples_for_final_sampling=torch.rand(
                    (bs,), dtype=torch.float32, device=device
                ),
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=1.0,
                threshold_acc=1.0,
                deterministic=True,
            )
            row_ids = torch.arange(bs, dtype=torch.long, device=device)
            accept_pos = accept_index[
                row_ids, sampling_correct_len.to(torch.long)
            ].to(torch.long)
            sampling_bonus = predicts[accept_pos].to(torch.int64)

            greedy_mask = (sampling_info.top_ks <= 1).view(-1)
            correct_len = torch.where(
                greedy_mask, greedy_correct_len, sampling_correct_len
            )
            bonus = torch.where(greedy_mask, greedy_bonus, sampling_bonus)
        if ragged_sampling_result is not None:
            correct_len, bonus = ragged_sampling_result
        cutoff_lens = (
            self.accept_lens
            if self.accept_lens is not None
            else self.verify_lens
        )
        if cutoff_lens is not None:
            cap = cutoff_lens.to(device=device, dtype=torch.int32) - 1
            original_correct = correct_len
            capped_correct = torch.minimum(correct_len, cap)
            row_ids = torch.arange(bs, device=device, dtype=torch.long)
            if sampling_info is not None and not sampling_info.is_all_greedy:
                sampled_bonus = torch.multinomial(
                    target_probs.view(bs, self.draft_token_num, -1)[
                        row_ids, capped_correct.to(torch.long)
                    ],
                    num_samples=1,
                ).squeeze(-1)
                greedy_mask = (sampling_info.top_ks <= 1).view(-1)
                sampled_bonus = torch.where(
                    greedy_mask,
                    target_predict[row_ids, capped_correct.to(torch.long)],
                    sampled_bonus,
                )
                bonus = torch.where(
                    original_correct > cap, sampled_bonus, bonus
                )
            else:
                bonus = target_predict[row_ids, capped_correct.to(torch.long)]
            correct_len = capped_correct
        max_correct = self.draft_token_num - 1
        packed = torch.cat(
            [candidates[:, 1:], correct_len.unsqueeze(1), bonus.unsqueeze(1)], dim=1
        ).cpu()

        num_correct_drafts_per_req_cpu: List[int] = []
        commit_lens_cpu: List[int] = []
        new_bonus_tokens_list: List[int] = []

        for i, req in enumerate(batch.reqs):
            num_correct_drafts = min(
                int(packed[i, max_correct].item()), max(0, verify_lens_cpu[i] - 1)
            )
            accept_tokens = packed[i, :num_correct_drafts].tolist() + [
                int(packed[i, max_correct + 1].item())
            ]

            appended = 0
            for token in accept_tokens:
                token = int(token)
                req.output_ids.append(token)
                appended += 1
                req.check_finished()
                if req.finished():
                    break
                if req.grammar is not None:
                    req.grammar.accept_token(token)

            if req.output_ids:
                new_bonus_token = int(req.output_ids[-1])
            elif req.origin_input_ids:
                new_bonus_token = int(req.origin_input_ids[-1])
            else:
                raise RuntimeError(
                    "DSPARK verify cannot determine current token: both output_ids and origin_input_ids are empty."
                )

            commit_lens_cpu.append(appended)
            new_bonus_tokens_list.append(new_bonus_token)
            num_correct_drafts_per_req_cpu.append(max(0, appended - 1))
            req.spec_verify_ct += 1
            req.spec_num_correct_drafts += num_correct_drafts_per_req_cpu[-1]

        commit_lens = torch.tensor(commit_lens_cpu, dtype=torch.int32, device=device)
        new_bonus_tokens = torch.tensor(
            new_bonus_tokens_list, dtype=torch.int64, device=device
        )

        if self.verify_lens is None:
            out_cache_loc = batch.out_cache_loc.view(bs, self.draft_token_num)
            row_offsets = torch.arange(self.draft_token_num, device=device)[None, :]
            if page_size == 1:
                keep_mask = row_offsets < commit_lens[:, None]
                batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
                batch.out_cache_loc = out_cache_loc[keep_mask]
            else:
                keep_slots = _compute_paged_keep_slots(
                    prefix_lens=batch.seq_lens,
                    commit_lens=commit_lens,
                    draft_token_num=self.draft_token_num,
                    page_size=page_size,
                )
                free_mask = row_offsets >= keep_slots[:, None]
                batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])
                keep_mask = row_offsets < commit_lens[:, None]
                batch.out_cache_loc = out_cache_loc[keep_mask]
        else:
            q_lens_device = torch.tensor(
                model_lens_cpu, dtype=torch.int32, device=device
            )
            row_ids = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=torch.int64), q_lens_device
            )
            row_offsets = torch.cat(
                [
                    torch.arange(int(row_len), device=device, dtype=torch.int64)
                    for row_len in model_lens_cpu
                ],
                dim=0,
            )
            out_cache_loc = batch.out_cache_loc[: sum(model_lens_cpu)]
            if page_size == 1:
                keep_mask = row_offsets < commit_lens[row_ids]
                batch.token_to_kv_pool_allocator.free(out_cache_loc[~keep_mask])
                batch.out_cache_loc = out_cache_loc[keep_mask]
            else:
                keep_slots = _compute_paged_keep_slots(
                    prefix_lens=batch.seq_lens,
                    commit_lens=commit_lens,
                    draft_token_num=q_lens_device,
                    page_size=page_size,
                )
                free_mask = row_offsets >= keep_slots[row_ids]
                batch.token_to_kv_pool_allocator.free(out_cache_loc[free_mask])
                keep_mask = row_offsets < commit_lens[row_ids]
                batch.out_cache_loc = out_cache_loc[keep_mask]

        for req, commit_len in zip(batch.reqs, commit_lens_cpu, strict=True):
            req.kv_committed_len += commit_len
            req.kv_allocated_len = req.kv_committed_len

        end_offset = batch.seq_lens + commit_lens.to(batch.seq_lens.dtype)
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

        batch.seq_lens.add_(commit_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(
            torch.tensor(commit_lens_cpu, dtype=batch.seq_lens_cpu.dtype)
        )
        batch.seq_lens_sum += sum(commit_lens_cpu)

        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSPARK verify requires target hidden states, but got None.")
        if len(set(model_lens_cpu)) == 1 and model_lens_cpu[0] == self.draft_token_num:
            hidden = hidden.view(bs, self.draft_token_num, -1)
        else:
            hidden_rows = []
            cursor = 0
            for row_len in model_lens_cpu:
                row_hidden = hidden[cursor : cursor + row_len]
                if row_len < self.draft_token_num:
                    row_hidden = torch.cat(
                        [row_hidden, hidden.new_zeros((self.draft_token_num - row_len, hidden.shape[-1]))],
                        dim=0,
                    )
                hidden_rows.append(row_hidden)
                cursor += row_len
            hidden = torch.stack(hidden_rows, dim=0)
        segments: List[torch.Tensor] = []
        for i, ln in enumerate(commit_lens_cpu):
            if ln > 0:
                segments.append(hidden[i, :ln, :])
        next_target_hidden = torch.cat(segments, dim=0) if segments else hidden[:0]
        logits_output.hidden_states = None

        return (
            new_bonus_tokens,
            commit_lens,
            next_target_hidden,
            num_correct_drafts_per_req_cpu,
        )
