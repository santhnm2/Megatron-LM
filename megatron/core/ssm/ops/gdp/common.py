# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/utils/op.py` and `fla/modules/l2norm.py` in
# flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in this directory.

"""Shared helpers for the forked, inference-only Gated Delta Product kernels.

The kernel modules in this package reference a handful of names that do not
belong to any one of them: the chunk length, the base-2 exponential they
exponentiate with, and the L2 normalization applied to the queries and keys.
Keeping them here makes the package self-contained, with no `fla` import at run
time.

Upstream's hardware-capability probes and chunk-descriptor builders are absent
because nothing here needs them: the kernels have fixed launch configs rather
than autotune configurations to select between, and the descriptors are built
outside the graph by `metadata`.
"""

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


# The Gated Delta Product chunk length is fixed at 64: `solve_tril`
# merges 16x16 blocks up to 64x64, the WY representation is built on 64-wide
# blocks, and the h kernel stores one state block per `num_householder`
# expanded chunks. Upstream FLA makes the same choice for this operator.
CHUNK_SIZE = 64

# 1/ln(2), used to convert natural-log decays into the base-2 space the chunked
# kernels exponentiate in. Best fp32 approximation (hex 0x3FB8AA3B).
RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    """Base-2 exponentiate in fp32 regardless of the input dtype."""
    return tl.math.exp2(x.to(tl.float32))


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise L2 normalization over the last dimension, computed in fp32.

    Replaces `fla.modules.l2norm.l2norm_fwd`. The forward pass is a single
    normalization, so plain PyTorch ops are used instead of a forked Triton
    kernel: they are CUDA-graph safe, allocate no `rstd` buffer (only the
    backward pass needs it, and this path is inference-only), and match the
    upstream kernel's math -- fp32 accumulation, `rstd = 1/sqrt(sum(x^2)+eps)`
    -- and the in-kernel normalization the decode path applies.
    """
    x_fp32 = x.float()
    rstd = torch.rsqrt((x_fp32 * x_fp32).sum(-1, keepdim=True) + eps)
    return (x_fp32 * rstd).to(x.dtype)
