# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from
# https://github.com/fla-org/flash-linear-attention/ (v0.4.2,
# `fla/ops/common/chunk_scaled_dot_kkt.py`).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of that source tree.

"""Strictly lower-triangular `beta * K K^T` per chunk (the WY `A` matrix).

Forked from `chunk_scaled_dot_kkt_fwd`. Changes: forward-only, varlen-only,
no autotuning, and caller-supplied `chunk_indices`. See `cumsum.py` for why.
"""

import torch

from .common import HAVE_TRITON, exp, tl, triton


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
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    USE_G: tl.constexpr,
):
    """Compute the strictly lower-triangular block `A = tril(beta * K K^T, -1)`."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_b = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_A += tl.dot(b_k, tl.trans(b_k))

    if USE_G:
        p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_A *= exp(b_g[:, None] - b_g[None, :])
    b_A *= b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)
    p_A = tl.make_block_ptr(
        A + (bos * H + i_h) * BT, (T, BT), (BT * H, 1), (i_t * BT, 0), (BT, BT), (1, 0)
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
    """Returns `beta * K K^T` of shape `[B, T, H, chunk_size]`."""
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H, K = k.shape
    A = torch.empty(B, T, H, chunk_size, device=k.device, dtype=output_dtype)
    grid = (chunk_indices.shape[0], B * H)
    chunk_scaled_dot_kkt_fwd_kernel[grid](
        k=k,
        g=g,
        beta=beta,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=chunk_size,
        BK=64,
        USE_G=g is not None,
        num_warps=4,
        num_stages=3,
    )
    return A
