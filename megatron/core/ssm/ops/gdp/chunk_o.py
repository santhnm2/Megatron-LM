# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_product/chunk_deltaproduct_o.py` in
# flash-linear-attention v0.5.1 (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file at the repository root.

"""Output pass for the Gated Delta Product prefill.

Forked from `chunk_gated_delta_product_fwd_o`. Changes: forward-only,
varlen-only, no autotuning, caller-supplied `chunk_indices`, and the output is
zero-initialized instead of upstream's `-inf` fill. The zero fill is the
padding contract: token slots that belong to no sequence (the rounded-up tail of
a CUDA-graph batch) are never written by any chunk program, so they must read
back as zeros rather than `-inf`.
"""

import torch

from .common import HAVE_TRITON, exp2, tl, triton


@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    """Inter-chunk contribution `q @ h` plus the intra-chunk attention term."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    i_tg = i_t
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * num_householder * H + i_h) * K
    v += (bos * num_householder * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
        b_m = tl.where(m_A, exp2(b_g[:, None] - b_g[None, :]), 0)
        b_o = b_o * exp2(b_g)[:, None]
    else:
        b_m = ((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)).to(tl.float32)

    for i_dp in range(num_householder):
        b_A = tl.zeros([BT, BT], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_k = tl.make_block_ptr(
                k + i_dp * H * K,
                (K, T),
                (1, num_householder * H * K),
                (i_k * BK, i_t * BT),
                (BK, BT),
                (0, 1),
            )
            # [BT, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            # [BK, BT]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BT, BK] @ [BK, BT] -> [BT, BT]
            b_A += tl.dot(b_q, b_k)
        b_A = b_A * b_m
        p_v = tl.make_block_ptr(
            v + i_dp * H * V,
            (T, V),
            (H * V * num_householder, 1),
            (i_t * BT, i_v * BV),
            (BT, BV),
            (1, 0),
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_o += tl.dot(b_A.to(b_v.dtype), b_v)
    b_o = b_o * scale
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    num_householder: int,
    chunk_size: int,
) -> torch.Tensor:
    """Compute the Gated Delta Product outputs.

    Args:
        q: Queries `[B, T, H, K]` on the unexpanded token stream.
        k: Keys `[B, T*M, H, K]` on the Householder-expanded stream.
        v: Corrected values `[B, T*M, H, V]` from `chunk_h`.
        h: State at each chunk boundary, from `chunk_h`.
        g: Within-chunk cumulative log2 decays `[B, T, H]`, or `None`.
        scale: Score scale.
        cu_seqlens: Sequence boundaries `[N+1]` on the unexpanded stream.
        chunk_indices: Precomputed chunk descriptors for the unexpanded stream.
        num_householder: Number of Householder copies `M`.
        chunk_size: Chunk length.

    Returns the outputs `[B, T, H, V]`, zero at padding token positions.
    """
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert (
        q.shape[1] * num_householder == k.shape[1]
    ), "q.shape[1] * num_householder must be equal to k.shape[1]"

    # Zeros, not empty: padded token positions belong to no chunk program.
    o = v.new_zeros(B, T, H, V)
    BK, BV = 64, 64
    grid = (triton.cdiv(V, BV), chunk_indices.shape[0], B * H)
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BK=BK,
        BV=BV,
        USE_G=g is not None,
        num_warps=4,
        num_stages=2,
    )
    return o
