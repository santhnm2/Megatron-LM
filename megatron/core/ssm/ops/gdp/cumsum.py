# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/utils/cumsum.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file at the repository root.

"""Within-chunk cumulative sum of the scalar (per-head) log decays.

Only the scalar, non-reversed, variable-length path is provided -- the one the
Gated Delta Product prefill calls. Two further changes make it CUDA-graph
capturable: there is no autotuning (a fixed launch config, because autotuning
benchmarks on first call and must not happen inside a capture), and
`chunk_indices` is required from the caller rather than derived from
`cu_seqlens` -- deriving it costs a device-to-host sync and yields a
data-dependent size, both of which are fatal to graph capture.
"""

import torch

from .common import HAVE_TRITON, tl, triton


@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s, o, scale, cu_seqlens, chunk_indices, T, H: tl.constexpr, BT: tl.constexpr
):
    """Cumulative sum of `s` within each chunk, scaled by `scale`, written to `o`."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_h = i_bh % H
    i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
    i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
    bos = tl.load(cu_seqlens + i_n).to(tl.int32)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    p_s = tl.make_block_ptr(s + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_o = tl.make_block_ptr(o + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
    scale: float = 1.0,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Chunk-local cumulative sum of `g` of shape `[B, T, H]`, scaled by `scale`."""
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H = g.shape
    out = torch.empty_like(g, dtype=output_dtype)
    grid = (chunk_indices.shape[0], B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g,
        o=out,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=chunk_size,
        num_warps=2,
    )
    return out
