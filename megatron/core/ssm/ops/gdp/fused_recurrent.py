# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_rule/fused_recurrent.py` in flash-linear-attention
# v0.5.1 (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in this directory.

"""Fused recurrent Gated Delta Rule step, used by the decode path.

Gated Delta Product decode reaches this kernel by folding the `M` Householder
copies into the sequence dimension, so a single decode token becomes an
`M`-length sequence with the query placed on the last copy and the decay on the
first; the caller slices the answer back out.

The fork exists for the same reasons as the Mamba2 kernel fork in the sibling
`mamba2` package:

1. **Slot-indexed recurrent state.** The recurrent state lives in a persistent
   per-request cache owned by `DynamicInferenceContext`; this kernel reads and
   writes it *in place* at the rows named by `state_indices`, so no gather of
   the initial state and no scatter of the final state is needed.
2. **CUDA-graph friendliness.** `state_indices` may contain `-1` for padding
   requests (batch shapes are rounded up to a captured graph size). Padded rows
   write zeros to the output and touch no state, so a replayed graph with a
   partially filled batch produces the same result as an eager run with only the
   real requests.

Deliberately dropped relative to upstream (not needed for decode): `gk`/`gv`
decays, varlen `cu_seqlens`, the transposed state layout, and the autograd
wrapper (the backward pass is unimplemented upstream anyway).
"""

import torch

from .common import HAVE_TRITON, tl, triton


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "HAS_STATE_INDICES": lambda args: args["state_indices"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_update_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h,
    h_slot_stride,
    h_head_stride,
    h_k_stride,
    h_v_stride,
    state_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
):
    """Recurrent gated delta rule over `T` tokens per request, in place on a
    slot-indexed state cache."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_o = o + (i_n * T * HV + i_hv) * V + o_v

    # State slot mapping (handles dynamic batching slot allocation).
    if HAS_STATE_INDICES:
        i_s = tl.load(state_indices + i_n).to(tl.int64)
    else:
        i_s = i_n

    # Skip padding requests: zero their outputs and leave the state cache alone.
    if i_s < 0:
        for _ in tl.range(0, T):
            tl.store(p_o, 0.0, mask=mask_v)
            p_o += HV * V
        return

    bos = i_n * T
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_h = (
        h
        + i_s * h_slot_stride
        + i_hv * h_head_stride
        + o_k[:, None] * h_k_stride
        + o_v[None, :] * h_v_stride
    )
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)

        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> torch.Tensor:
    """Decode step of the gated delta rule, updating a slot-indexed state cache.

    Unlike `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule`, the
    recurrent state is neither gathered on input nor returned on output: `state`
    is the persistent cache and is updated in place at the rows given by
    `state_indices`.

    Args:
        q: Queries of shape `[B, T, H, K]`.
        k: Keys of shape `[B, T, H, K]`.
        v: Values of shape `[B, T, HV, V]`. GVA is applied when `HV > H`.
        state: Recurrent state cache of shape `[S, HV, K, V]`, read and written
            in place. `S` is the number of cache slots, which may exceed `B`.
        g: Decays of shape `[B, T, HV]`.
        beta: Betas of shape `[B, T, HV]` (headwise) or `[B, T, HV, V]`.
            Defaults to ones.
        scale: Score scale; defaults to `K ** -0.5`.
        state_indices: `[B]` map from batch position to cache slot. `-1` marks
            a padding request: its output rows are zeroed and no state slot is
            read or written. `None` means batch position `i` uses slot `i`.
        use_qk_l2norm_in_kernel: Whether to L2-normalize `q`/`k` in kernel.

    Returns:
        Outputs of shape `[B, T, HV, V]`.
    """
    assert HAVE_TRITON, "Triton is required for fused_recurrent_gated_delta_rule_update."
    # The kernel indexes with raw pointer arithmetic and would read garbage from
    # a strided input, so force every tensor argument contiguous.
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()
    if g is not None:
        g = g.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    assert state.shape[1:] == (
        HV,
        K,
        V,
    ), f"state is expected to have shape [num_slots, {HV}, {K}, {V}], got {tuple(state.shape)}"
    if state_indices is not None:
        assert state_indices.shape == (
            B,
        ), f"state_indices is expected to have shape [{B}], got {tuple(state_indices.shape)}"
    if scale is None:
        scale = K**-0.5
    if beta is None:
        beta = torch.ones_like(v[..., 0])
    # The kernel indexes q/k/v/g/beta/o with packed strides, so require contiguity
    # here rather than silently reading the wrong elements. (Checking layout is
    # host-side metadata only: no synchronization, so this stays graph-safe.)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert tensor is None or tensor.is_contiguous(), f"{name} must be contiguous"

    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    grid = (NV, B * HV)
    fused_recurrent_gated_delta_rule_update_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h=state,
        h_slot_stride=state.stride(0),
        h_head_stride=state.stride(1),
        h_k_stride=state.stride(2),
        h_v_stride=state.stride(3),
        state_indices=state_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        num_warps=1,
        num_stages=3,
    )
    return o
