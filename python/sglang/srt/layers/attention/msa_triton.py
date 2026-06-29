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
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

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


@triton.jit
def _block_sparse_attn_kernel(
    q_ptr, k_ptr, v_ptr, keep_ptr, out_ptr,
    T, Hq, Hk, num_blocks, group, scale,
    D: tl.constexpr, BQ: tl.constexpr, BLK: tl.constexpr,
):
    """Range-first q-tile flash. grid = (ceil(T/BQ), Hq). out: [T,Hq,D] f32-accum -> store q.dtype.
    q:[T,Hq,D] k/v:[T,Hk,D] keep:[T,Hk,num_blocks] i32."""
    pid_t = tl.program_id(0)
    hq = tl.program_id(1)
    g = hq // group
    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + offs_q[:, None] * (Hq * D) + hq * D + offs_d[None, :],
        mask=offs_q[:, None] < T, other=0.0,
    )  # [BQ,D] bf16 (native TensorCore dot; fp32-input tl.dot is flaky on MUSA)
    m_i = tl.full([BQ], -float("inf"), tl.float32)
    l_i = tl.zeros([BQ], tl.float32)
    acc = tl.zeros([BQ, D], tl.float32)
    for b in range(0, num_blocks):
        keepv = tl.load(
            keep_ptr + offs_q * (Hk * num_blocks) + g * num_blocks + b,
            mask=offs_q < T, other=0,
        )  # [BQ] {0,1}
        if tl.max(keepv) > 0:
            offs_k = b * BLK + tl.arange(0, BLK)
            kk = tl.load(
                k_ptr + offs_k[:, None] * (Hk * D) + g * D + offs_d[None, :],
                mask=offs_k[:, None] < T, other=0.0,
            )  # [BLK,D] bf16
            vv = tl.load(
                v_ptr + offs_k[:, None] * (Hk * D) + g * D + offs_d[None, :],
                mask=offs_k[:, None] < T, other=0.0,
            )  # [BLK,D] bf16
            s = tl.dot(q, tl.trans(kk)).to(tl.float32) * scale  # [BQ,BLK] fp32 accum
            mask = (offs_k[None, :] <= offs_q[:, None]) & (offs_k[None, :] < T) & (keepv[:, None] > 0)
            s = tl.where(mask, s, -float("inf"))
            m_blk = tl.max(s, axis=1)  # [BQ]
            m_new = tl.maximum(m_i, m_blk)
            # guard masked rows (m_new == -inf -> no contribution)
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.where(mask, tl.exp(s - m_safe[:, None]), 0.0)  # [BQ,BLK]
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))  # [BQ]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(vv.dtype), vv)
            m_i = m_new
    out = acc / tl.where(l_i > 0.0, l_i, 1.0)[:, None]
    tl.store(
        out_ptr + offs_q[:, None] * (Hq * D) + hq * D + offs_d[None, :],
        out, mask=offs_q[:, None] < T,
    )


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
    if backend == "torch" or not q.is_cuda:
        return _block_sparse_attn_torch(q, k, v, indices, block_size, scale)
    nb = (T + block_size - 1) // block_size
    keep = _keep_from_indices(indices, nb)  # [T,Hk,nb] i32
    out = torch.empty((T, Hq, D), dtype=torch.float32, device=q.device)
    BQ = 64
    grid = (triton.cdiv(T, BQ), Hq)
    _block_sparse_attn_kernel[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), keep, out,
        T, Hq, Hk, nb, group, float(scale),
        D=D, BQ=BQ, BLK=block_size,
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
# Decode — BATCHED + PAGED (cuda-graph-capturable). Reads seq_lens / req_pool_indices /
# req_to_token / KV-pool as DEVICE tensors and masks in-kernel: no host syncs, no
# dynamic shapes, fixed grid -> the whole decode forward can be cuda-graph captured.
# Selection is capture-safe (iterative argmax + cumsum compaction; NO torch.topk/sort,
# which can host-sync under graph capture on MUSA). Used by models/minimax_m3._msa_decode.
# ----------------------------------------------------------------------------------------


@triton.jit
def _decode_block_score_paged_kernel(
    idxq_ptr,          # [bs, Hk, DI] bf16   (this forward's decode tokens, one per req)
    idxk_pool_ptr,     # [cap, DI]    bf16   (model-owned idx_k aux cache)
    rt_ptr,            # [num_reqs, max_ctx] int32  (req_to_token)
    seqlens_ptr,       # [bs] int32/64
    reqidx_ptr,        # [bs] int64        (req_pool_indices)
    bs_ptr,            # [bs, Hk, NB] f32 out
    cap, max_ctx, Hk, NB, idx_scale,
    DI: tl.constexpr, BLK: tl.constexpr,
):
    """bscore[i,h,b] = max_{j in block b, j<seq_len[i]} (idx_q[i,h]·idx_k_pool[slot(i,j)])*idx_scale.
    grid = (bs, Hk, NB). Blocks past the request's length write the -1e38 invalid sentinel."""
    i = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    out_off = i * (Hk * NB) + h * NB + b
    slen = tl.load(seqlens_ptr + i).to(tl.int32)
    nb_i = (slen + BLK - 1) // BLK
    if b >= nb_i:
        tl.store(bs_ptr + out_off, -1e38)  # future/invalid block -> never selected
        return
    req = tl.load(reqidx_ptr + i).to(tl.int64)
    offs_d = tl.arange(0, DI)
    q = tl.load(idxq_ptr + i * (Hk * DI) + h * DI + offs_d).to(tl.float32)  # [DI]
    offs_j = b * BLK + tl.arange(0, BLK)                                    # token positions
    jmask = offs_j < slen
    slots = tl.load(rt_ptr + req * max_ctx + offs_j, mask=jmask, other=0).to(tl.int64)
    slots = tl.minimum(tl.maximum(slots, 0), cap - 1)                       # clamp (capture-safe)
    kk = tl.load(
        idxk_pool_ptr + slots[:, None] * DI + offs_d[None, :],
        mask=jmask[:, None], other=0.0,
    ).to(tl.float32)                                                        # [BLK, DI]
    s = tl.sum(q[None, :] * kk, axis=1) * idx_scale                         # [BLK] (q=1: no tl.dot)
    s = tl.where(jmask, s, -1e38)
    bsv = tl.max(s, axis=0)
    bsv = tl.where((bsv > -1e30) & (bsv < 1e30), bsv, -1e38)                # neutralize garbage
    tl.store(bs_ptr + out_off, bsv)


def _select_paged(block_scores, seq_lens, block_size, topk, init_blocks, local_blocks):
    """Capture-safe selection: [bs,Hk,NB] f32 (-1e38 invalid) + seq_lens [bs] ->
    indices [bs,Hk,W] int32 (ascending block ids, -1 pad), counts [bs,Hk] int32.
    Uses iterative argmax (k passes) + cumsum compaction — NO topk/sort (host-sync free)."""
    bs, Hk, NB = block_scores.shape
    dev = block_scores.device
    bidx = torch.arange(NB, device=dev)
    cur = ((seq_lens.to(torch.int64) - 1) // block_size).clamp(min=0)          # [bs]
    valid = block_scores > -1e30                                              # [bs,Hk,NB]
    sink = bidx[None, :] < init_blocks                                        # [1,NB]
    lo = (cur - (local_blocks - 1)).clamp(min=0)                              # [bs]
    local = (bidx[None, :] >= lo[:, None]) & (bidx[None, :] <= cur[:, None])  # [bs,NB]
    forced = (sink | local)[:, None, :] & valid                              # [bs,Hk,NB]
    cand = valid & ~forced
    sc = torch.where(cand, block_scores, torch.full_like(block_scores, -float("inf")))
    chosen = torch.zeros_like(valid)
    k = min(topk, NB)
    neg = torch.full((bs, Hk, 1), -float("inf"), device=dev, dtype=block_scores.dtype)
    for _ in range(k):
        am = sc.argmax(dim=-1, keepdim=True)               # [bs,Hk,1]
        take = sc.gather(-1, am) > -1e30                    # real candidate?
        chosen = chosen | torch.zeros_like(chosen).scatter(-1, am, take)
        sc.scatter_(-1, am, neg)                            # remove from future argmaxes
    sel = forced | chosen                                  # [bs,Hk,NB] (chosen ⊆ cand, disjoint)
    W = topk + init_blocks + local_blocks
    sel_i = sel.to(torch.int32)
    counts = sel_i.sum(-1).to(torch.int32)                 # [bs,Hk]
    rank = sel_i.cumsum(-1) - 1                             # [bs,Hk,NB] ascending rank of selected
    tgt = torch.where(sel & (rank < W), rank, torch.full_like(rank, W))  # non-selected -> overflow
    ind_ext = torch.full((bs, Hk, W + 1), -1, dtype=torch.int32, device=dev)
    src = bidx[None, None, :].expand(bs, Hk, NB).to(torch.int32)
    ind_ext.scatter_(-1, tgt, src)
    return ind_ext[..., :W].contiguous(), counts


@triton.jit
def _decode_attn_paged_kernel(
    q_ptr,             # [bs, Hq, D] (this forward's decode q, one token per req)
    k_pool_ptr,        # [cap, Hk, D]
    v_pool_ptr,        # [cap, Hk, D]
    rt_ptr,            # [num_reqs, max_ctx] int32
    seqlens_ptr,       # [bs]
    reqidx_ptr,        # [bs] int64
    indices_ptr,       # [bs, Hk, W] int32 (block ids, -1 pad)
    out_ptr,           # [bs, Hq, D] out
    cap, max_ctx, Hq, Hk, group, scale,
    D: tl.constexpr, BLK: tl.constexpr, W: tl.constexpr,
):
    """q=1 flash-decode over the W selected blocks (paged gather). grid = (bs, Hq)."""
    i = tl.program_id(0)
    hq = tl.program_id(1)
    g = hq // group
    slen = tl.load(seqlens_ptr + i).to(tl.int32)
    req = tl.load(reqidx_ptr + i).to(tl.int64)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + i * (Hq * D) + hq * D + offs_d).to(tl.float32)  # [D]
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], tl.float32)
    for w in range(0, W):
        blk = tl.load(indices_ptr + i * (Hk * W) + g * W + w)          # block id or -1
        if blk >= 0:
            offs_j = blk * BLK + tl.arange(0, BLK)
            jmask = offs_j < slen
            slots = tl.load(rt_ptr + req * max_ctx + offs_j, mask=jmask, other=0).to(tl.int64)
            slots = tl.minimum(tl.maximum(slots, 0), cap - 1)
            kk = tl.load(
                k_pool_ptr + slots[:, None] * (Hk * D) + g * D + offs_d[None, :],
                mask=jmask[:, None], other=0.0,
            ).to(tl.float32)                                            # [BLK, D]
            vv = tl.load(
                v_pool_ptr + slots[:, None] * (Hk * D) + g * D + offs_d[None, :],
                mask=jmask[:, None], other=0.0,
            ).to(tl.float32)
            s = tl.sum(q[None, :] * kk, axis=1) * scale                # [BLK]
            s = tl.where(jmask, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=0))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.where(jmask, tl.exp(s - m_safe), 0.0)               # [BLK]
            alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * vv, axis=0)
            m_i = m_new
    out = acc / (l_i if l_i > 0.0 else 1.0)
    tl.store(out_ptr + i * (Hq * D) + hq * D + offs_d, out)


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
    Hq = q.shape[1]
    DI = idx_q.shape[-1]
    group = Hq // Hk
    max_ctx = req_to_token.shape[1]
    NB = num_blocks if num_blocks is not None else (max_ctx + block_size - 1) // block_size
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if idx_scale is None:
        idx_scale = 1.0 / math.sqrt(DI)
    dev = q.device

    bscore = torch.empty((bs, Hk, NB), dtype=torch.float32, device=dev)
    _decode_block_score_paged_kernel[(bs, Hk, NB)](
        idx_q.contiguous(), idxk_pool, req_to_token, seq_lens, req_pool_indices, bscore,
        cap, max_ctx, Hk, NB, float(idx_scale),
        DI=DI, BLK=block_size,
    )
    indices, _counts = _select_paged(
        bscore, seq_lens, block_size, topk, init_blocks, local_blocks
    )
    W = indices.shape[-1]
    out = torch.empty((bs, Hq, D), dtype=q.dtype, device=dev)
    _decode_attn_paged_kernel[(bs, Hq)](
        q.contiguous(), k_pool, v_pool, req_to_token, seq_lens, req_pool_indices,
        indices, out,
        cap, max_ctx, Hq, Hk, group, float(scale),
        D=D, BLK=block_size, W=W,
    )
    return out
