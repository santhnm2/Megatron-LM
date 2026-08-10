# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from
# https://github.com/fla-org/flash-linear-attention/ (v0.4.2,
# `fla/ops/gated_delta_product/chunk_deltaproduct_h.py`).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of that source tree.

"""Chunk-level recurrent state pass for the Gated Delta Product prefill.

Forked from `chunk_gated_delta_product_fwd_h`. Changes:

* Forward-only, varlen-only, no autotuning, caller-supplied `chunk_offsets`.
* The final state is written directly into the slot-indexed per-request cache
  via `state_indices`, in place, instead of being returned as a dense tensor
  for the caller to scatter. `-1` marks a padding request: nothing is written
  for it, which is what lets a graph captured at a padded prefill count replay
  correctly for fewer real requests.
"""

import torch

from .common import HAVE_TRITON, exp, tl, triton


@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_product_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    ht_slot_stride,
    ht_head_stride,
    state_indices,
    cu_seqlens,
    chunk_offsets,
    T,
    num_householder: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
):
    """One program per (value block, sequence, head): scan the sequence's chunks,
    emitting the per-chunk state `h` and the corrected values `v_new`."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos
    NT = tl.cdiv(T, BT)
    boh = tl.load(chunk_offsets + i_n).to(tl.int32)

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h) * K * V
    v += (bos * H + i_h) * V
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    v_new += (bos * H + i_h) * V
    stride_v = H * V
    stride_h = H * K * V
    stride_k = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        if i_t % num_householder == 0:
            i_t_true = i_t // num_householder
            p_h1 = tl.make_block_ptr(
                h + i_t_true * stride_h, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_h2 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_h3 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_h4 = tl.make_block_ptr(
                    h + i_t_true * stride_h, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v_new = tl.zeros([BT, BV], dtype=tl.float32)
        p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v_new += tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (stride_k, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_new += tl.dot(b_w, b_h4.to(b_w.dtype))
        b_v_new = -b_v_new + tl.load(p_v, boundary_check=(0, 1))

        p_v_new = tl.make_block_ptr(
            v_new, (T, V), (stride_v, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            last_idx = min((i_t + 1) * BT, T) - 1
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            b_v_new = b_v_new * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 = b_h1 * b_g_last
            if K > 64:
                b_h2 = b_h2 * b_g_last
            if K > 128:
                b_h3 = b_h3 * b_g_last
            if K > 192:
                b_h4 = b_h4 * b_g_last
        b_v_new = b_v_new.to(k.dtype.element_ty)
        p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v_new)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v_new)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v_new)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v_new)

    # epilogue: write the final state into this request's cache slot
    if STORE_FINAL_STATE:
        if HAS_STATE_INDICES:
            i_s = tl.load(state_indices + i_n).to(tl.int64)
        else:
            i_s = i_n
        # A padding request (-1) owns no slot; leave the cache untouched.
        if i_s >= 0:
            ht = ht + i_s * ht_slot_stride + i_h * ht_head_stride
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 64:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 128:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
            if K > 192:
                p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_product_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    num_chunks: int,
    num_householder: int,
    chunk_size: int,
    state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunk-level state pass.

    Args:
        k, w, u: `[B, T*M, H, ...]` Householder-expanded tensors.
        g: Chunk-local cumulative log decays over the expanded stream, or None.
        cu_seqlens: Sequence boundaries of the *expanded* stream, `[N+1]`.
        chunk_offsets: Cumulative unexpanded chunk counts per sequence, `[N+1]`.
        num_chunks: Total number of unexpanded chunks (the fixed, padded count).
        num_householder: Number of Householder copies `M`.
        chunk_size: Chunk length (64).
        state: `[S, H, K, V]` per-request state cache, written in place at
            `state_indices`. `None` skips the final-state write.
        state_indices: `[N]` slot per sequence; `-1` marks padding.
        initial_state: Optional dense `[N, H, K, V]` starting state.

    Returns:
        `(h, v_new)`: the per-chunk states and the corrected values.
    """
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H, K = k.shape
    V = u.shape[-1]
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    N = cu_seqlens.shape[0] - 1

    h = k.new_empty(B, num_chunks, H, K, V)
    v_new = torch.empty_like(u)

    if state is not None:
        assert state.shape[1:] == (H, K, V), (
            f"state is expected to have shape [num_slots, {H}, {K}, {V}], "
            f"got {tuple(state.shape)}"
        )
        assert (
            state.stride(3) == 1 and state.stride(2) == V
        ), "the last two dimensions of the state cache must be contiguous"

    BV = 64
    grid = (triton.cdiv(V, BV), N * H)
    chunk_gated_delta_product_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=state,
        ht_slot_stride=state.stride(0) if state is not None else 0,
        ht_head_stride=state.stride(1) if state is not None else 0,
        state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        num_householder=num_householder,
        H=H,
        K=K,
        V=V,
        BT=chunk_size,
        BV=BV,
        USE_G=g is not None,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=state is not None,
        HAS_STATE_INDICES=state_indices is not None,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new
