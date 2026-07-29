"""MiniMax-M3 Sparse Attention (MSA) — Triton kernels (in-sglang, mate-independent).

Drop-in replacement for the mate (tilelang) MSA path: same public signatures as
`mate.msa_indexer.msa_block_topk_indices`, `mate.msa_attention.msa_block_sparse_attention`,
and `mate.msa_attention.msa_decode_attention`. Selected via the `triton` backend
(SGLANG_MUSA_M3_MSA_{INDEXER,ATTN}_BACKEND=triton) in models/minimax_m3.py.

Algorithm (oracle: operators/msa_attention{,_decode}/*_reference.py):
  index head h == GQA group g (Hi == Hk); idx_k is ONE head shared across index heads.
    score[h,t,j]       = (idx_q[t,h] . idx_k[j]) * idx_scale            (causal j <= t)
    block_score[h,t,b] = max_{j in block b, j<=t} score                 (score_type="max")
    selected(t)        = {0..init-1} U {cur-local+1..cur} U top-`topk`(block_score)  (union)
  The `group` q-heads of group g attend (full causal softmax, scale) ONLY to keys in
  selected(t) blocks of group g. Bit-equal to full causal attention when topk+forced >=
  num_blocks (seq <= ~2048) — the short-prompt correctness gate is unaffected.

Kernels: range-first q-tile flash (avoids the tilelang fixed-first-slice hang); fp32
accumulate; masked-row guards so unselected blocks contribute exactly 0.

Naming: dims are parameterized throughout so every kernel runs at BOTH TP-sharded per-rank
serving shapes (Hq=8,Hk=1,group=8,D=128) and single-GPU global shapes (64/4/16). The paged
decode kernels (msa_decode_attention_paged + its _msa_paged_* helpers) are tensorcore
`tl.dot` GQA-batched (the index-heads / q-group fill the MMA's 16-row minimum) and
cuda-graph-capturable (all-device-tensor inputs, static grids, in-kernel iterative argmax —
no host sync, no torch.topk/sort).
"""
from __future__ import annotations

import logging
import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# -inf is inlined as -float("inf") (Triton kernels can't read module globals)

# ----------------------------------------------------------------------------------------
# Indexer — block scores (Triton) + selection (torch, exact, matches the oracle)
# ----------------------------------------------------------------------------------------


@triton.jit
def _block_score_kernel(
    idx_q_ptr, idx_k_ptr, bs_ptr,
    T, Hk, num_blocks, idx_scale,
    DI: tl.constexpr, BQ: tl.constexpr, BLK: tl.constexpr,
):
    """block_scores[h, t, b] = max_{j in block b, j<=t} (idx_q[t,h]·idx_k[j])*idx_scale.

    idx_q: [T, Hk, DI] contiguous; idx_k: [T, DI] contiguous; bs_ptr: [Hk, T, num_blocks] f32.
    grid = (ceil(T/BQ), Hk).
    """
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, DI)
    q = tl.load(
        idx_q_ptr + offs_q[:, None] * (Hk * DI) + h * DI + offs_d[None, :],
        mask=offs_q[:, None] < T, other=0.0,
    )  # [BQ, DI] bf16 (native TensorCore dot; fp32-input dot is flaky on MUSA here)
    for b in range(0, num_blocks):
        offs_k = b * BLK + tl.arange(0, BLK)
        kk = tl.load(
            idx_k_ptr + offs_k[:, None] * DI + offs_d[None, :],
            mask=offs_k[:, None] < T, other=0.0,
        )  # [BLK, DI] bf16
        s = tl.dot(q, tl.trans(kk)).to(tl.float32) * idx_scale  # [BQ, BLK] fp32 accum
        ok = (offs_k[None, :] <= offs_q[:, None]) & (offs_k[None, :] < T)
        # finite -1e38 sentinel (NOT -inf): tl.max over an all--inf row returns garbage
        # (+inf) on the MUSA Triton backend; -1e38 keeps the reduce finite. The selector
        # treats <= -1e30 as a future/invalid block (>-1e30 check).
        s = tl.where(ok, s, -1e38)
        bscore = tl.max(s, axis=1)  # [BQ]
        # Guard against sporadic MUSA tl.dot garbage (+inf / huge / nan): map any
        # out-of-range score to the -1e38 sentinel so it reads as an invalid block
        # (never selected). Real scores are O(1) << 1e30; future blocks are -1e38.
        bscore = tl.where((bscore > -1e30) & (bscore < 1e30), bscore, -1e38)
        tl.store(
            bs_ptr + h * (T * num_blocks) + offs_q * num_blocks + b,
            bscore, mask=offs_q < T,
        )


def _block_scores_triton(idx_q: torch.Tensor, idx_k: torch.Tensor, block_size: int, idx_scale: float) -> torch.Tensor:
    T, Hk, DI = idx_q.shape
    num_blocks = (T + block_size - 1) // block_size
    bs = torch.full((Hk, T, num_blocks), -float("inf"), dtype=torch.float32, device=idx_q.device)
    BQ = 128
    grid = (triton.cdiv(T, BQ), Hk)
    _block_score_kernel[grid](
        idx_q.contiguous(), idx_k.contiguous(), bs,
        T, Hk, num_blocks, float(idx_scale),
        DI=DI, BQ=BQ, BLK=block_size,
    )
    return bs


def _block_scores_torch(idx_q: torch.Tensor, idx_k: torch.Tensor, block_size: int, idx_scale: float) -> torch.Tensor:
    T, Hk, DI = idx_q.shape
    nb = (T + block_size - 1) // block_size
    qpos = torch.arange(T, device=idx_q.device)
    causal = qpos.view(1, -1) <= qpos.view(-1, 1)  # [T,T] key<=query
    out = torch.full((Hk, T, nb), -float("inf"), dtype=torch.float32, device=idx_q.device)
    ik = idx_k.float()  # [T, DI]
    for h in range(Hk):
        s = (idx_q[:, h, :].float() @ ik.T) * idx_scale  # [T,T]
        s = s.masked_fill(~causal, -float("inf"))
        for b in range(nb):
            j0, j1 = b * block_size, min((b + 1) * block_size, T)
            out[h, :, b] = s[:, j0:j1].max(dim=-1).values
    return out


def _select_from_scores(block_scores: torch.Tensor, block_size: int, topk: int, init_blocks: int, local_blocks: int):
    """[Hk,T,nb] f32 (-inf future) -> indices [T,Hk,W] int32 (ascending, -1 pad), counts [T,Hk]."""
    Hk, T, nb = block_scores.shape
    dev = block_scores.device
    bidx = torch.arange(nb, device=dev)
    cur = torch.arange(T, device=dev) // block_size  # [T]
    # "valid" = causally-real block. Use > -1e30 (not ~isinf): on MUSA torch_musa,
    # `max` over an all--inf row returns -FLT_MAX (-3.4e38), not -inf — so isinf misses
    # the future/masked blocks. The Triton kernel emits true -inf; both are caught here.
    valid = block_scores > -1e30  # [Hk,T,nb]
    sink = (bidx[None, :] < init_blocks) & (bidx[None, :] <= cur[:, None])  # [T,nb]
    lo = (cur - (local_blocks - 1)).clamp(min=0)
    local = (bidx[None, :] >= lo[:, None]) & (bidx[None, :] <= cur[:, None])
    forced = (sink | local)[None].expand(Hk, T, nb) & valid
    cand = valid & ~forced
    sc = torch.where(cand, block_scores, torch.full_like(block_scores, -float("inf")))
    k = min(topk, nb)
    top = sc.topk(k, dim=-1).indices  # [Hk,T,k]
    topmask = torch.zeros_like(valid)
    topmask.scatter_(-1, top, torch.ones_like(top, dtype=torch.bool))
    topmask = topmask & cand
    sel = (forced | topmask) & valid  # [Hk,T,nb]
    counts = sel.sum(-1).transpose(0, 1).to(torch.int32).contiguous()  # [T,Hk]
    W = topk + init_blocks + local_blocks
    big = nb + 1
    rank = torch.where(sel, bidx[None, None, :].expand(Hk, T, nb), torch.full((Hk, T, nb), big, device=dev, dtype=bidx.dtype))
    sorted_ids, _ = rank.sort(dim=-1)
    sorted_ids = sorted_ids[..., :W]
    indices = torch.where(sorted_ids < nb, sorted_ids, torch.full_like(sorted_ids, -1)).to(torch.int32)
    return indices.transpose(0, 1).contiguous(), counts  # [T,Hk,W], [T,Hk]


def msa_block_topk_indices(
    idx_q: torch.Tensor,            # [T, Hk, D]
    idx_k: torch.Tensor,            # [T, D] or [T, 1, D]
    *,
    block_size: int = 128,
    topk: int = 16,
    init_blocks: int = 1,
    local_blocks: int = 1,
    idx_scale: Optional[float] = None,
    backend: str = "triton",
    return_block_scores: bool = False,
):
    if idx_k.dim() == 3:
        idx_k = idx_k[:, 0, :]
    D = idx_q.shape[-1]
    if idx_scale is None:
        idx_scale = 1.0 / math.sqrt(D)
    if backend == "torch" or not idx_q.is_cuda:
        bs = _block_scores_torch(idx_q, idx_k, block_size, idx_scale)
    else:
        bs = _block_scores_triton(idx_q, idx_k, block_size, idx_scale)
    indices, counts = _select_from_scores(bs, block_size, topk, init_blocks, local_blocks)
    if return_block_scores:
        return indices, counts, bs
    return indices, counts


# ----------------------------------------------------------------------------------------
# Prefill — block-sparse flash attention (Triton)
# ----------------------------------------------------------------------------------------


def _keep_from_indices(indices: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """indices [T,Hk,W] (ascending, -1 pad) -> keep [T,Hk,num_blocks] int32 {0,1}."""
    T, Hk, W = indices.shape
    dev = indices.device
    valid = indices >= 0
    tgt = torch.where(valid, indices.long(), torch.full_like(indices, num_blocks, dtype=torch.long))
    keep_ext = torch.zeros((T, Hk, num_blocks + 1), dtype=torch.int32, device=dev)
    keep_ext.scatter_(2, tgt, torch.ones_like(tgt, dtype=torch.int32))
    return keep_ext[..., :num_blocks].contiguous()



def _block_sparse_attn_torch(q, k, v, indices, block_size, scale):
    """Oracle-style dense reference (fallback / cross-check)."""
    T, Hq, D = q.shape
    Hk = k.shape[1]
    group = Hq // Hk
    nb = (T + block_size - 1) // block_size
    keep = _keep_from_indices(indices, nb).bool()  # [T,Hk,nb]
    out = torch.empty_like(q)
    kpos = torch.arange(T, device=q.device)
    for g in range(Hk):
        keymask = torch.zeros((T, T), dtype=torch.bool, device=q.device)
        for b in range(nb):
            j0, j1 = b * block_size, min((b + 1) * block_size, T)
            keymask[keep[:, g, b], j0:j1] = True
        keymask = keymask & (kpos.view(1, -1) <= kpos.view(-1, 1))
        addmask = torch.where(keymask, 0.0, -float("inf")).float()
        for hh in range(group):
            h = g * group + hh
            s = (q[:, h, :].float() @ k[:, g, :].float().T) * scale + addmask
            p = torch.softmax(s, dim=-1)
            out[:, h, :] = (p @ v[:, g, :].float()).to(q.dtype)
    return out


# ----------------------------------------------------------------------------------------
# Per-token block-sparse prefill flash. Per (query token, kv-head) it batches the `group`
# q-heads of that kv-head into a 16-row MMA (HT=16, valid rows < group) and loops ONLY the
# W selected blocks in `indices` (exact sparsity, tensorcore dot), at ~O(T*W) instead of
# O(T*num_blocks). PARAMETERIZED dims so it runs at BOTH TP-sharded per-rank (Hq=8,Hk=1,
# group=8) and global (64/4/16). grid = (T, Hk). Self-attention, causal.


@triton.jit
def _msa_prefill_attn_kernel(
    q_ptr, k_ptr, v_ptr, indices_ptr, out_ptr,
    T, Hq, Hk, group, scale,
    D: tl.constexpr, BLK: tl.constexpr, W: tl.constexpr, HT: tl.constexpr,
):
    t = tl.program_id(0)
    g = tl.program_id(1)
    hrow = tl.arange(0, HT)
    hmask = hrow < group                     # valid q-head rows (group<=HT=16; rest zero-padded)
    q_head = g * group + hrow
    dim = tl.arange(0, D)
    lane = tl.arange(0, BLK)
    q_tile = tl.load(
        q_ptr + t * (Hq * D) + q_head[:, None] * D + dim[None, :],
        mask=hmask[:, None], other=0.0,
    )  # [HT, D] bf16 (padded rows masked -> not read, discarded on store)
    m_i = tl.full([HT], -float("inf"), tl.float32)
    l_i = tl.zeros([HT], tl.float32)
    acc = tl.zeros([HT, D], tl.float32)
    for w in range(W):
        blk = tl.load(indices_ptr + t * (Hk * W) + g * W + w)   # selected block id or -1 pad
        token = blk * BLK + lane
        valid = (blk >= 0) & (token < T) & (token <= t)         # causal + selected
        kk = tl.load(
            k_ptr + token[None, :] * (Hk * D) + g * D + dim[:, None],
            mask=valid[None, :], other=0.0,
        )  # [D, BLK] bf16 (transposed)
        logits = tl.dot(q_tile, kk).to(tl.float32) * scale       # [HT, BLK] fp32 accum
        logits = tl.where(valid[None, :], logits, -float("inf"))
        new_max = tl.maximum(m_i, tl.max(logits, axis=1))
        m_safe = tl.where(new_max < -1e30, 0.0, new_max)         # finite sentinel (MUSA -FLT_MAX)
        p = tl.where(valid[None, :], tl.exp(logits - m_safe[:, None]), 0.0)   # [HT, BLK]
        alpha = tl.where(m_i < -1e30, 0.0, tl.exp(m_i - m_safe))              # [HT]
        vv = tl.load(
            v_ptr + token[:, None] * (Hk * D) + g * D + dim[None, :],
            mask=valid[:, None], other=0.0,
        )  # [BLK, D] bf16
        acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = new_max
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(
        out_ptr + t * (Hq * D) + q_head[:, None] * D + dim[None, :],
        out, mask=hmask[:, None],
    )


def msa_block_sparse_attention(
    q: torch.Tensor,        # [T, Hq, D]
    k: torch.Tensor,        # [T, Hk, D]
    v: torch.Tensor,        # [T, Hk, D]
    indices: torch.Tensor,  # [T, Hk, W] int32
    counts: Optional[torch.Tensor] = None,
    *,
    block_size: int = 128,
    scale: Optional[float] = None,
    backend: str = "triton",
) -> torch.Tensor:
    T, Hq, D = q.shape
    Hk = k.shape[1]
    group = Hq // Hk
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if backend == "torch" or q.device.type not in ("cuda", "musa"):
        # NOTE: use device.type (not q.is_cuda) — torch_musa tensors report is_cuda=False, so the
        # old `not q.is_cuda` gate sent every standalone MUSA call to the slow torch path.
        return _block_sparse_attn_torch(q, k, v, indices, block_size, scale)
    # Per-token sparse-flash (uses `indices` directly, per-rank capable).
    W = indices.shape[-1]
    out = torch.empty((T, Hq, D), dtype=torch.float32, device=q.device)
    _msa_prefill_attn_kernel[(T, Hk)](
        q.contiguous(), k.contiguous(), v.contiguous(),
        indices.contiguous().to(torch.int32), out,
        T, Hq, Hk, group, float(scale),
        D=D, BLK=block_size, W=W, HT=16,
    )
    return out.to(q.dtype)


# ----------------------------------------------------------------------------------------
# Decode — q=1 block-sparse flash attention (Triton)
# ----------------------------------------------------------------------------------------


@triton.jit
def _decode_block_score_kernel(
    idxq_ptr, idxk_ptr, bs_ptr,
    S, Hk, num_blocks, idx_scale,
    D: tl.constexpr, BLK: tl.constexpr,
):
    """q=1: bscore[h,b] = max_{j in block b} (idx_q[h]·idx_k[j])*idx_scale. grid=(num_blocks,Hk)."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_d = tl.arange(0, D)
    q = tl.load(idxq_ptr + h * D + offs_d).to(tl.float32)  # [D]
    offs_k = b * BLK + tl.arange(0, BLK)
    kk = tl.load(idxk_ptr + offs_k[:, None] * D + offs_d[None, :], mask=offs_k[:, None] < S, other=0.0).to(tl.float32)
    s = tl.sum(q[None, :] * kk, axis=1) * idx_scale  # [BLK]
    s = tl.where(offs_k < S, s, -1e38)  # finite sentinel (MUSA tl.max-over--inf is garbage)
    bsv = tl.max(s, axis=0)
    bsv = tl.where((bsv > -1e30) & (bsv < 1e30), bsv, -1e38)  # neutralize tl.dot garbage
    tl.store(bs_ptr + h * num_blocks + b, bsv)


@triton.jit
def _decode_attn_kernel(
    q_ptr, k_ptr, v_ptr, keep_ptr, out_ptr,
    S, Hq, Hk, num_blocks, group, scale,
    D: tl.constexpr, BLK: tl.constexpr,
):
    """q=1 flash over selected blocks. grid=(Hq,). q:[Hq,D] k/v:[S,Hk,D] keep:[Hk,num_blocks] i32 -> out:[Hq,D]."""
    hq = tl.program_id(0)
    g = hq // group
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + hq * D + offs_d).to(tl.float32)  # [D]
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)
    for b in range(0, num_blocks):
        keepv = tl.load(keep_ptr + g * num_blocks + b)
        if keepv > 0:
            offs_k = b * BLK + tl.arange(0, BLK)
            kk = tl.load(k_ptr + offs_k[:, None] * (Hk * D) + g * D + offs_d[None, :], mask=offs_k[:, None] < S, other=0.0).to(tl.float32)  # [BLK,D]
            vv = tl.load(v_ptr + offs_k[:, None] * (Hk * D) + g * D + offs_d[None, :], mask=offs_k[:, None] < S, other=0.0).to(tl.float32)
            s = tl.sum(q[None, :] * kk, axis=1) * scale  # [BLK]
            s = tl.where(offs_k < S, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.where(offs_k < S, tl.exp(s - m_safe), 0.0)  # [BLK]
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m_i = m_new
    out = acc / (l_i if l_i > 0.0 else 1.0)
    tl.store(out_ptr + hq * D + offs_d, out)


def msa_decode_attention(
    q: torch.Tensor,        # [Hq, D]
    idx_q: torch.Tensor,    # [Hk, D]
    K: torch.Tensor,        # [S, Hk, D]
    V: torch.Tensor,        # [S, Hk, D]
    idx_k: torch.Tensor,    # [S, D] or [S, 1, D]
    *,
    block_size: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 1,
    scale: Optional[float] = None,
    idx_scale: Optional[float] = None,
) -> torch.Tensor:
    if idx_k.dim() == 3:
        idx_k = idx_k[:, 0, :]
    S, Hk, D = K.shape
    Hq = q.shape[0]
    group = Hq // Hk
    nb = (S + block_size - 1) // block_size
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if idx_scale is None:
        idx_scale = 1.0 / math.sqrt(D)

    # indexer (q=1) -> block scores [Hk, nb] -> select -> keep [Hk, nb]
    bs = torch.full((Hk, nb), -float("inf"), dtype=torch.float32, device=q.device)
    _decode_block_score_kernel[(nb, Hk)](
        idx_q.contiguous(), idx_k.contiguous(), bs, S, Hk, nb, float(idx_scale), D=D, BLK=block_size,
    )
    cur = (S - 1) // block_size
    blk = torch.arange(nb, device=q.device)
    forced = (blk < init_blocks) | ((blk >= max(cur - (local_blocks - 1), 0)) & (blk <= cur))  # [nb]
    keep = torch.zeros((Hk, nb), dtype=torch.int32, device=q.device)
    k = min(topk, nb)
    for h in range(Hk):
        cand = ~forced
        sc = torch.where(cand, bs[h], torch.full_like(bs[h], -float("inf")))
        topi = sc.topk(k).indices
        tm = torch.zeros(nb, dtype=torch.bool, device=q.device)
        tm[topi] = True
        tm = tm & cand
        keep[h] = (forced | tm).to(torch.int32)

    out = torch.empty((Hq, q.shape[-1]), dtype=torch.float32, device=q.device)
    _decode_attn_kernel[(Hq,)](
        q.contiguous(), K.contiguous(), V.contiguous(), keep.contiguous(), out,
        S, Hq, Hk, nb, group, float(scale), D=D, BLK=block_size,
    )
    return out.to(q.dtype)



# ----------------------------------------------------------------------------------------
# Paged q=1 block-sparse decode (cuda-graph-capturable). A tensorcore `tl.dot` implementation
# of the paged decode: the indexer block-max AND the attention both use `tl.dot`, batching the
# index-heads / GQA q-group into the MMA's 16-row minimum (grid (NB,bs)/(Hk,bs), Hk index-heads
# + `group` q-heads padded to 16 rows) — vs the per-request eager decode's elementwise `tl.sum`.
# PARAMETERIZED dims (Hq,Hk,group,D) so it runs at BOTH TP-sharded per-rank (Hq=8,Hk=1,group=8)
# and global (64/4/16). Reads seq_lens / req_pool_indices / req_to_token / KV-pool as DEVICE
# tensors and masks in-kernel: no host syncs, no dynamic shapes, fixed grid -> the whole decode
# forward is cuda-graph-capturable. Selection is capture-safe (iterative argmax + cumsum
# compaction; NO torch.topk/sort, which can host-sync under graph capture on MUSA).
# HARDENED: the select kernel's candidate/exhaustion logic uses a finite -1e38 sentinel + `> -1e30`
# guard (NOT `> -inf`) because MUSA `tl.max` over an all--inf row returns -FLT_MAX.


@triton.jit
def _msa_paged_index_kernel(idx_q, idxk_pool, req_to_token,
                                req_pool_indices, seq_lens, scores, Hk,
                                MAX_CTX: tl.constexpr, MAX_NB: tl.constexpr,
                                DI: tl.constexpr, BLK: tl.constexpr, HTI: tl.constexpr):
    block = tl.program_id(0)
    batch = tl.program_id(1)
    length = tl.load(seq_lens + batch)
    # O(seq_len)-not-O(max_ctx): the grid is a fixed NB=cdiv(max_ctx,BLK) (capture-safe), but at
    # runtime skip blocks past this request's seq_len. Correctness: the select kernel's `candidate`
    # mask only reads scores for blocks < cdiv(length,BLK)-1 (all in-range, all computed here), and
    # loads `other=-1e38` for the rest -> out-of-range scores are never read, so no sentinel store
    # is needed. This is THE lever that lets sparse decode beat dense (dense is also O(seq_len)).
    if block * BLK >= length:
        return
    dim = tl.arange(0, DI)
    lane = tl.arange(0, BLK)
    token = block * BLK + lane
    head = tl.arange(0, HTI)                                  # index-head MMA rows (Hk<=HTI=16)
    request = tl.load(req_pool_indices + batch)
    slot = tl.load(req_to_token + request * MAX_CTX + token,
                   mask=token < length, other=0)
    q = tl.load(idx_q + batch * (Hk * DI) + head[:, None] * DI + dim[None, :],
                mask=head[:, None] < Hk, other=0.0)           # [HTI, DI] (Hk valid rows)
    kt = tl.load(idxk_pool + slot[None, :] * DI + dim[:, None],
                 mask=token[None, :] < length, other=0.0)     # [DI, BLK]
    dots = tl.dot(q, kt, out_dtype=tl.float32)                # bf16 inputs -> fp32 accum (MUSA rule)
    dots = tl.where(token[None, :] < length, dots, -float("inf"))
    maximum = tl.max(dots, axis=1)
    tl.store(scores + (batch * Hk + head) * MAX_NB + block, maximum, mask=head < Hk)


@triton.jit
def _msa_paged_select_kernel(scores, seq_lens, selected, Hk,
                                 MAX_NB: tl.constexpr, BLOCK_NB: tl.constexpr,
                                 BLK: tl.constexpr, TOPK: tl.constexpr, W: tl.constexpr):
    head = tl.program_id(0)
    batch = tl.program_id(1)
    block = tl.arange(0, BLOCK_NB)
    length = tl.load(seq_lens + batch)
    num_blocks = (length + BLK - 1) // BLK
    candidate = (block > 0) & (block < num_blocks - 1)
    values = tl.load(scores + (batch * Hk + head) * MAX_NB + block,
                     mask=candidate, other=-1e38)          # finite sentinel (MUSA)
    base = (batch * Hk + head) * W
    tl.store(selected + base, 0)                            # sink block 0 (init_blocks=1)
    for rank in range(TOPK):
        # ONE fused reduction for both value and index (was two: tl.max + tl.argmax = 32
        # reductions/call over BLOCK_NB=1024; now 16). The TOPK iterations stay serial (each masks
        # the previous pick), but each is half the work.
        best, index = tl.max(values, axis=0, return_indices=True)
        # MUSA tl.max over all--inf returns -FLT_MAX, so guard on the finite sentinel:
        # when the candidate pool is exhausted emit -1 so the attn kernel skips the rank.
        chosen = tl.where(best > -1e30, index, -1)
        tl.store(selected + base + rank + 1, chosen)
        values = tl.where(block == index, -1e38, values)
    tl.store(selected + base + W - 1, num_blocks - 1)       # local block (local_blocks=1)


@triton.jit
def _msa_paged_attn_kernel(q, k_pool, v_pool, req_to_token,
                               req_pool_indices, seq_lens, selected, out, Hq, Hk, group,
                               MAX_CTX: tl.constexpr, SCALE: tl.constexpr,
                               D: tl.constexpr, BLK: tl.constexpr, W: tl.constexpr, HT: tl.constexpr):
    kv_head = tl.program_id(0)
    batch = tl.program_id(1)
    hrow = tl.arange(0, HT)
    hmask = hrow < group                                     # valid q-head rows (group<=HT=16)
    q_head = kv_head * group + hrow
    dim = tl.arange(0, D)
    lane = tl.arange(0, BLK)
    length = tl.load(seq_lens + batch)
    request = tl.load(req_pool_indices + batch)
    q_tile = tl.load(q + batch * (Hq * D) + q_head[:, None] * D + dim[None, :],
                     mask=hmask[:, None], other=0.0)         # [HT, D]
    running_max = tl.full((HT,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HT,), tl.float32)
    accumulator = tl.zeros((HT, D), tl.float32)
    selection_base = (batch * Hk + kv_head) * W
    for sb in range(W):
        block = tl.load(selected + selection_base + sb)
        token = block * BLK + lane
        valid = (block >= 0) & (token < length)            # block=-1 rank -> contributes 0
        slot = tl.load(req_to_token + request * MAX_CTX + token, mask=valid, other=0)
        kt = tl.load(k_pool + slot[None, :] * (Hk * D) + kv_head * D + dim[:, None],
                     mask=valid[None, :], other=0.0)
        logits = tl.dot(q_tile, kt, out_dtype=tl.float32) * SCALE
        logits = tl.where(valid[None, :], logits, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(logits, axis=1))
        m_safe = tl.where(new_max < -1e30, 0.0, new_max)    # finite sentinel (MUSA -FLT_MAX)
        probabilities = tl.where(valid[None, :], tl.exp(logits - m_safe[:, None]), 0.0)
        alpha = tl.where(running_max < -1e30, 0.0, tl.exp(running_max - m_safe))
        values = tl.load(v_pool + slot[:, None] * (Hk * D) + kv_head * D + dim[None, :],
                         mask=valid[:, None], other=0.0)
        accumulator = accumulator * alpha[:, None] + tl.dot(
            probabilities.to(tl.bfloat16), values, out_dtype=tl.float32)
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max
    result = accumulator / tl.where(running_sum > 0.0, running_sum, 1.0)[:, None]
    tl.store(out + batch * (Hq * D) + q_head[:, None] * D + dim[None, :], result, mask=hmask[:, None])


@triton.jit
def _msa_paged_attn_partial_kernel(q, k_pool, v_pool, req_to_token,
                                       req_pool_indices, seq_lens, selected,
                                       partial_o, partial_m, partial_l, Hq, Hk, group,
                                       MAX_CTX: tl.constexpr, SCALE: tl.constexpr,
                                       D: tl.constexpr, BLK: tl.constexpr, W: tl.constexpr,
                                       HT: tl.constexpr, NSPLIT: tl.constexpr, BPS: tl.constexpr):
    # Flash-decode KV-split: each (kv_head, batch, split) CTA attends only its BPS-block slice of
    # the W selected blocks and writes an UNNORMALIZED partial (acc, running_max, running_sum); the
    # combine kernel merges the NSPLIT partials. This lifts the bs1 sparse-attn off a SINGLE CTA
    # (209us/call, 30% of the decode step: an 18-deep serial online-softmax chain + 1.5MB loaded
    # through one SM) onto NSPLIT parallel CTAs (short serial chain + memory-level parallelism).
    kv_head = tl.program_id(0)
    batch = tl.program_id(1)
    split = tl.program_id(2)
    hrow = tl.arange(0, HT)
    hmask = hrow < group                                     # valid q-head rows (group<=HT=16)
    q_head = kv_head * group + hrow
    dim = tl.arange(0, D)
    lane = tl.arange(0, BLK)
    length = tl.load(seq_lens + batch)
    request = tl.load(req_pool_indices + batch)
    q_tile = tl.load(q + batch * (Hq * D) + q_head[:, None] * D + dim[None, :],
                     mask=hmask[:, None], other=0.0)         # [HT, D]
    running_max = tl.full((HT,), -float("inf"), tl.float32)
    running_sum = tl.zeros((HT,), tl.float32)
    accumulator = tl.zeros((HT, D), tl.float32)
    selection_base = (batch * Hk + kv_head) * W
    for j in range(BPS):
        sb = split * BPS + j
        if sb < W:                                           # this split owns blocks [split*BPS, +BPS)
            block = tl.load(selected + selection_base + sb)
            token = block * BLK + lane
            valid = (block >= 0) & (token < length)          # block=-1 rank -> contributes 0
            slot = tl.load(req_to_token + request * MAX_CTX + token, mask=valid, other=0)
            kt = tl.load(k_pool + slot[None, :] * (Hk * D) + kv_head * D + dim[:, None],
                         mask=valid[None, :], other=0.0)
            logits = tl.dot(q_tile, kt, out_dtype=tl.float32) * SCALE
            logits = tl.where(valid[None, :], logits, -float("inf"))
            new_max = tl.maximum(running_max, tl.max(logits, axis=1))
            m_safe = tl.where(new_max < -1e30, 0.0, new_max)
            probabilities = tl.where(valid[None, :], tl.exp(logits - m_safe[:, None]), 0.0)
            alpha = tl.where(running_max < -1e30, 0.0, tl.exp(running_max - m_safe))
            values = tl.load(v_pool + slot[:, None] * (Hk * D) + kv_head * D + dim[None, :],
                             mask=valid[:, None], other=0.0)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(tl.bfloat16), values, out_dtype=tl.float32)
            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            running_max = new_max
    prow = ((batch * Hk + kv_head) * NSPLIT + split) * HT + hrow
    tl.store(partial_o + prow[:, None] * D + dim[None, :], accumulator)
    tl.store(partial_m + prow, running_max)
    tl.store(partial_l + prow, running_sum)


@triton.jit
def _msa_paged_attn_combine_kernel(partial_o, partial_m, partial_l, out, Hq, Hk, group,
                                       D: tl.constexpr, HT: tl.constexpr, NSPLIT: tl.constexpr,
                                       DTILE: tl.constexpr):
    # Merge the NSPLIT flash partials for one (kv_head, batch) into the final normalized output.
    # PARALLELIZED over D-tiles (grid dim 2): at TP8 bs1 the base grid is (Hk=1,bs=1) = ONE CTA,
    # which serialized the whole merge on a single SM (~27us/call, ~as costly as the 18-CTA partial
    # in the in-graph profile). The per-row max/sum (m,l) are D-independent, so each D-tile CTA
    # recomputes the cheap [HT] m/l and owns a D/DTILE slice of the [HT,D] accumulator -> D//DTILE
    # concurrent CTAs, no cross-tile dependency.
    kv_head = tl.program_id(0)
    batch = tl.program_id(1)
    dtile = tl.program_id(2)
    hrow = tl.arange(0, HT)
    hmask = hrow < group
    q_head = kv_head * group + hrow
    dim = dtile * DTILE + tl.arange(0, DTILE)                # this CTA's D-slice
    m = tl.full((HT,), -float("inf"), tl.float32)
    l = tl.zeros((HT,), tl.float32)
    acc = tl.zeros((HT, DTILE), tl.float32)
    base = (batch * Hk + kv_head) * NSPLIT
    for s in range(NSPLIT):
        prow = (base + s) * HT + hrow
        pm = tl.load(partial_m + prow)                       # [HT] this split's max
        pl = tl.load(partial_l + prow)                       # [HT] this split's sum
        po = tl.load(partial_o + prow[:, None] * D + dim[None, :])   # [HT, DTILE] unnormalized
        new_m = tl.maximum(m, pm)
        m_safe = tl.where(new_m < -1e30, 0.0, new_m)
        alpha = tl.where(m < -1e30, 0.0, tl.exp(m - m_safe))         # rescale running
        beta = tl.where(pm < -1e30, 0.0, tl.exp(pm - m_safe))       # rescale this split
        acc = acc * alpha[:, None] + po * beta[:, None]
        l = l * alpha + pl * beta
        m = new_m
    result = acc / tl.where(l > 0.0, l, 1.0)[:, None]
    tl.store(out + batch * (Hq * D) + q_head[:, None] * D + dim[None, :], result, mask=hmask[:, None])


def _msa_paged_decode(q, idx_q, k_pool, v_pool, idxk_pool, req_to_token,
                          req_pool_indices, seq_lens, scale, topk, NB):
    """Tensorcore GQA-batched paged q=1 decode (parameterized dims). Capture-safe: static grids
    (NB,bs)/(Hk,bs), in-kernel iterative argmax (no topk/sort)."""
    bs, Hq, D = q.shape
    Hk = k_pool.shape[1]
    DI = idx_q.shape[-1]
    group = Hq // Hk
    max_ctx = req_to_token.shape[1]
    W = topk + 2                                            # init_blocks=1 + local_blocks=1
    scores = torch.empty((bs, Hk, NB), dtype=torch.float32, device=q.device)
    selected = torch.empty((bs, Hk, W), dtype=torch.int32, device=q.device)
    out = torch.empty((bs, Hq, D), dtype=q.dtype, device=q.device)
    _msa_paged_index_kernel[(NB, bs)](
        idx_q.contiguous(), idxk_pool, req_to_token, req_pool_indices, seq_lens, scores, Hk,
        MAX_CTX=max_ctx, MAX_NB=NB, DI=DI, BLK=128, HTI=16, num_warps=8,
    )
    # Single-CTA iterative-argmax select. A two-level (chunked local-topk + merge) parallelization
    # was tried and REVERTED: the cost is the TOPK serial argmax passes (inherent to top-k), NOT the
    # per-pass reduction width, so chunking BLOCK_NB doesn't cut the serial chain and just adds a
    # second kernel + scratch -> net slower (32.7 -> 31.0 tok/s bs1). tl.max over 1024 w/ 8 warps is
    # already near the single-CTA floor.
    _msa_paged_select_kernel[(Hk, bs)](
        scores, seq_lens, selected, Hk, MAX_NB=NB, BLOCK_NB=triton.next_power_of_2(NB),
        BLK=128, TOPK=topk, W=W, num_warps=8,
    )
    # Sparse-attn: default to the flash KV-SPLIT path (parallelize the W blocks across CTAs) —
    # the single-CTA kernel is 209us/call (30% of the bs1 decode step). Env SGLANG_MUSA_MSA_ATTN_SPLIT:
    #   1 = original single-CTA; <=0 (default) = W splits (BPS=1, max parallelism); n = min(n,W) splits.
    _raw = int(os.environ.get("SGLANG_MUSA_MSA_ATTN_SPLIT", "0"))
    if _raw == 1:
        _msa_paged_attn_kernel[(Hk, bs)](
            q.contiguous(), k_pool, v_pool, req_to_token, req_pool_indices, seq_lens, selected, out,
            Hq, Hk, group, MAX_CTX=max_ctx, SCALE=float(scale), D=D, BLK=128, W=W, HT=16, num_warps=8,
        )
        return out
    nsplit_req = W if _raw <= 0 else min(_raw, W)
    BPS = (W + nsplit_req - 1) // nsplit_req                 # blocks per split
    NSPLIT = (W + BPS - 1) // BPS                            # tighten -> no empty trailing splits
    HT = 16
    qc = q.contiguous()
    partial_o = torch.empty((bs, Hk, NSPLIT, HT, D), dtype=torch.float32, device=q.device)
    partial_m = torch.empty((bs, Hk, NSPLIT, HT), dtype=torch.float32, device=q.device)
    partial_l = torch.empty((bs, Hk, NSPLIT, HT), dtype=torch.float32, device=q.device)
    _msa_paged_attn_partial_kernel[(Hk, bs, NSPLIT)](
        qc, k_pool, v_pool, req_to_token, req_pool_indices, seq_lens, selected,
        partial_o, partial_m, partial_l, Hq, Hk, group,
        MAX_CTX=max_ctx, SCALE=float(scale), D=D, BLK=128, W=W, HT=HT,
        NSPLIT=NSPLIT, BPS=BPS, num_warps=4,
    )
    # combine parallelized over D-tiles so it isn't a single CTA at bs1 (see kernel comment).
    DTILE = 32 if D % 32 == 0 else D
    _msa_paged_attn_combine_kernel[(Hk, bs, D // DTILE)](
        partial_o, partial_m, partial_l, out, Hq, Hk, group,
        D=D, HT=HT, NSPLIT=NSPLIT, DTILE=DTILE, num_warps=4,
    )
    return out


def msa_decode_attention_paged(
    q: torch.Tensor,                  # [bs, Hq, D] (or [bs, Hq*D])
    idx_q: torch.Tensor,              # [bs, Hk, DI]
    k_pool: torch.Tensor,             # [cap, Hk, D]
    v_pool: torch.Tensor,             # [cap, Hk, D]
    idxk_pool: torch.Tensor,          # [cap, DI]
    req_to_token: torch.Tensor,       # [num_reqs, max_ctx] int32
    req_pool_indices: torch.Tensor,   # [bs] int64
    seq_lens: torch.Tensor,           # [bs] int (DEVICE)
    *,
    block_size: int,
    topk: int,
    init_blocks: int = 1,
    local_blocks: int = 1,
    scale: Optional[float] = None,
    idx_scale: Optional[float] = None,
    num_blocks: Optional[int] = None,
) -> torch.Tensor:
    """Batched, paged, cuda-graph-capturable q=1 block-sparse decode. All inputs are
    device tensors; the only host-known sizes are static (bs, Hq, Hk, D, DI, W, NB).
    `num_blocks` (NB) MUST be a fixed model constant (default cdiv(max_ctx, block_size))
    so the captured grid is valid for every replay seq_len up to the pool's max context."""
    cap, Hk, D = k_pool.shape
    bs = q.shape[0]
    q = q.view(bs, -1, D)                      # [bs, Hq, D]
    max_ctx = req_to_token.shape[1]
    NB = num_blocks if num_blocks is not None else (max_ctx + block_size - 1) // block_size
    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Tensorcore GQA-batched paged decode (parameterized dims). Validated in-server
    # (4/4 gate, >2048 needle, inf_burst, AIME25) on the M3-FP8 stack. See _msa_paged_decode.
    return _msa_paged_decode(
        q, idx_q, k_pool, v_pool, idxk_pool,
        req_to_token, req_pool_indices, seq_lens, scale, topk, NB,
    )
