# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from
# https://github.com/fla-org/flash-linear-attention/ (v0.4.2,
# `fla/ops/utils/cumsum.py`).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of that source tree.

"""Within-chunk cumulative sum of the scalar (per-head) log decays.

Forked from `chunk_local_cumsum_scalar`. Changes: forward-only, varlen-only,
no autotuning (a fixed launch config, because autotuning benchmarks on first
call and must not happen inside a CUDA-graph capture), and `chunk_indices` is
required from the caller rather than derived from `cu_seqlens` -- deriving it
costs a device-to-host sync and yields a data-dependent size, both of which are
fatal to graph capture.
"""

import torch

from .common import HAVE_TRITON, tl, triton


@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s, o, cu_seqlens, chunk_indices, T, H: tl.constexpr, BT: tl.constexpr
):
    """Cumulative sum of `s` within each chunk, written to `o`."""
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
    b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Chunk-local cumulative sum of `g` of shape `[B, T, H]`."""
    assert HAVE_TRITON, "Triton is required for the forked GDP prefill kernels."
    B, T, H = g.shape
    out = torch.empty_like(g, dtype=output_dtype)
    grid = (chunk_indices.shape[0], B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g,
        o=out,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=chunk_size,
        num_warps=2,
    )
    return out
