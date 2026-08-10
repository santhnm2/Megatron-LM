# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from
# https://github.com/fla-org/flash-linear-attention/ (v0.4.2,
# `fla/utils.py`, `fla/ops/utils/op.py`, `fla/ops/utils/index.py` and
# `fla/modules/l2norm.py`).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of that source tree.

"""Helpers the forked Gated Delta Product kernels pull out of `fla.utils`.

The kernel modules in this package are near-verbatim copies of their upstream
counterparts, which means they reference a handful of names that live outside
the kernel files themselves: the hardware-capability probes that select
autotune configurations, the chunk-descriptor builders, `exp`, and the L2
normalization applied to the queries and keys. Vendoring them here keeps the
package self-contained without importing `fla` at run time.

The probes are reproduced rather than simplified because they choose which
autotune configurations exist, so changing them would change which kernel
variants get benchmarked and picked.
"""

import functools
import os

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


# Upstream FLA fixes the Gated Delta Product chunk length at 64: `solve_tril`
# merges 16x16 blocks up to 64x64, the WY representation is built on 64-wide
# blocks, and the h kernel stores one state block per `num_householder`
# expanded chunks.
CHUNK_SIZE = 64


# ----------------------------------------------------------------------------
# Hardware capability probes (`fla.utils`).
# ----------------------------------------------------------------------------
def _is_nvidia() -> bool:
    return torch.cuda.is_available() and torch.version.hip is None


IS_NVIDIA = _is_nvidia()
IS_NVIDIA_HOPPER = IS_NVIDIA and (
    'NVIDIA H' in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9
)
IS_NVIDIA_BLACKWELL = IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10
IS_TMA_SUPPORTED = (
    IS_NVIDIA
    and torch.cuda.get_device_capability(0)[0] >= 9
    and os.environ.get('FLA_USE_TMA', '0') == '1'
    and (
        hasattr(tl, '_experimental_make_tensor_descriptor') or hasattr(tl, 'make_tensor_descriptor')
    )
)

if hasattr(tl, '_experimental_make_tensor_descriptor'):
    # Triton 3.3.x
    make_tensor_descriptor = tl._experimental_make_tensor_descriptor
elif hasattr(tl, 'make_tensor_descriptor'):
    # Triton 3.4.x and later
    make_tensor_descriptor = tl.make_tensor_descriptor
else:

    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        """Stub for Triton builds without TMA; only keeps the compiler happy."""
        return None


# Shared-memory thresholds per architecture, from `fla.utils.Backend`.
_SHARED_MEM_BY_ARCH = {
    'ada': 101376,  # RTX 4090
    'ampere': 166912,  # A100
    'hopper': 232448,  # H100
    'none': 102400,  # default
}


@functools.cache
def check_shared_mem(arch: str = "none") -> bool:
    """Whether the current device has at least `arch`'s shared memory budget."""
    try:
        return torch.cuda.get_device_properties(
            0
        ).shared_memory_per_block_optin >= _SHARED_MEM_BY_ARCH.get(
            arch.lower(), _SHARED_MEM_BY_ARCH['none']
        )
    except Exception:
        return False


# ----------------------------------------------------------------------------
# `fla.ops.utils.op`
# ----------------------------------------------------------------------------
@triton.jit
def exp(x):
    """Exponentiate in fp32 regardless of the input dtype."""
    return tl.exp(x.to(tl.float32))


# ----------------------------------------------------------------------------
# `fla.ops.utils.index`
# ----------------------------------------------------------------------------
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Flattened `(sequence, chunk-within-sequence)` pairs, one per chunk.

    Note that this synchronizes on the device: the per-sequence chunk counts
    come back to the host via `.tolist()` so the result's length can depend on
    them. That is upstream FLA's behavior, preserved here unchanged.
    """
    lens = torch.diff(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(lens, chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Per-sequence prefix sum of chunk counts, with a leading zero."""
    lens = torch.diff(cu_seqlens)
    return torch.nn.functional.pad(triton.cdiv(lens, chunk_size), (1, 0), value=0).cumsum(-1)


# ----------------------------------------------------------------------------
# `fla.modules.l2norm`
# ----------------------------------------------------------------------------
_BT_LIST = [8, 16, 32, 64, 128]


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(x, y, rstd, eps, D, BD: tl.constexpr):
    """Row-per-program L2 normalization, used when `D > 512`."""
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in _BT_LIST
    ],
    key=["D", "NB"],
)
@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel(
    x, y, rstd, eps, T, D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr
):
    """Block-of-rows L2 normalization, used when `D <= 512`."""
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise L2 normalization over the last dimension, computed in fp32.

    Upstream also returns `rstd`, which only the backward pass consumes. This
    fork is forward-only, so the buffer is still written by the kernel (to keep
    the kernel itself unmodified) but is not returned.
    """
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue the fused kernel.
    max_fused_size = 65536 // x.element_size()
    BD = min(max_fused_size, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        # Tolerate a wide range of T before recompiling, to limit autotuning.
        NB = triton.cdiv(T, 2048 * 32)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](x=x, y=y, rstd=rstd, eps=eps, T=T, D=D, BD=BD, NB=NB)
    else:
        l2norm_fwd_kernel1[(T,)](x=x, y=y, rstd=rstd, eps=eps, D=D, BD=BD)
    return y.view(x_shape_og)
