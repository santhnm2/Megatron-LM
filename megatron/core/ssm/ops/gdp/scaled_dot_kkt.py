# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/common/chunk_scaled_dot_kkt.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in this directory.

"""Strictly lower-triangular `beta * K K^T` per chunk (the WY `A` matrix).

Forked from `chunk_scaled_dot_kkt_fwd`. Changes: forward-only, varlen-only,
no autotuning, and caller-supplied `chunk_indices`. See `cumsum.py` for why.
"""

import torch

from .common import HAVE_TRITON, exp2, tl, triton


@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
):
    """Compute one chunk's `beta * K K^T`, masked to below the diagonal."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_h = i_bh % HV
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A *= exp2(b_g_diff)
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * HV + i_h) * BT, (T, BT), (BT * HV, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute `beta * K K^T` per chunk, decayed by `g` and masked below the diagonal.

    Args:
        k: Keys `[B, T, H, K]`, where `H` is the number of query/key heads.
        g: Within-chunk cumulative log2 decays `[B, T, HV]`, or `None`.
        beta: Betas `[B, T, HV]`, where `HV` is the number of value/output heads.
            For GVA, `H < HV` and `HV % H == 0`; otherwise `H == HV`.
        cu_seqlens: Sequence boundaries `[N+1]`.
        chunk_indices: Precomputed chunk descriptors.
        chunk_size: Chunk length.
        output_dtype: Result dtype.

    Returns `[B, T, HV, chunk_size]`, the per-chunk lower-triangular block.
    """
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H, K, HV = *k.shape, beta.shape[2]
    A = torch.empty(B, T, HV, chunk_size, device=k.device, dtype=output_dtype)
    grid = (chunk_indices.shape[0], B * HV)
    chunk_scaled_dot_kkt_fwd_kernel[grid](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=chunk_size,
        BK=64,
        USE_G=g is not None,
        num_warps=4,
        num_stages=3,
    )
    return A
