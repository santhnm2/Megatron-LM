# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_rule/wy_fast.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in this directory.

"""Recompute the WY representation (`w`, `u`) from the inverted `A`.

Forked from `recompute_w_u_fwd`. Changes: forward-only, varlen-only, no
autotuning, and caller-supplied `chunk_indices`. See `cumsum.py` for why.
"""

import torch

from .common import HAVE_TRITON, exp2, tl, triton


@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
):
    """Apply the inverted transition block to the betas-scaled keys and values."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_h = i_bh % HV
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    p_b = tl.make_block_ptr(beta + bos * HV + i_h, (T,), (HV,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(
        A + (bos * HV + i_h) * BT, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(
            v + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + (bos * HV + i_h) * V, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    if USE_G:
        p_g = tl.make_block_ptr(g + (bos * HV + i_h), (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_g = exp2(tl.load(p_g, boundary_check=(0,)))

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h // (HV // H)) * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        p_w = tl.make_block_ptr(
            w + (bos * HV + i_h) * K, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0)
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]
        if USE_G:
            b_kb *= b_g[:, None]
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the WY factors `w` and `u` for every chunk.

    Args:
        k: Keys `[B, T, H, K]`, where `H` is the number of query/key heads.
        v: Values `[B, T, HV, V]`, where `HV` is the number of value/output heads.
            For GVA, `H < HV` and `HV % H == 0`; otherwise `H == HV`.
        beta: Betas `[B, T, HV]`.
        A: Inverted transition blocks `[B, T, HV, BT]` from `solve_tril`.
        g: Within-chunk cumulative log2 decays `[B, T, HV]`, or `None`.
        cu_seqlens: Sequence boundaries `[N+1]`.
        chunk_indices: Precomputed chunk descriptors.

    Returns `(w, u)`, shaped `[B, T, HV, K]` and like `v` respectively.
    """
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = A.shape[-1]

    w = k.new_empty(B, T, HV, K)
    u = torch.empty_like(v)
    grid = (chunk_indices.shape[0], B * HV)
    recompute_w_u_fwd_kernel[grid](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BK=64,
        BV=64,
        USE_G=g is not None,
        num_warps=4,
        num_stages=3,
    )
    return w, u
