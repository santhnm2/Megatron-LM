# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Some of this code was adopted from
# https://github.com/fla-org/flash-linear-attention/ (v0.4.2).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of that source tree.

"""Shared helpers for the forked, inference-only Gated Delta Product kernels."""

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


# The forked chunk kernels are wired to a chunk length of 64: `solve_tril`
# only has a 64x64 merge path here, the WY representation is built on 64-wide
# blocks, and the h kernel stores one state block per `num_householder`
# expanded chunks. Upstream FLA makes the same choice for this operator.
CHUNK_SIZE = 64


@triton.jit
def exp(x):
    """`fla.ops.utils.op.exp`: exponentiate in fp32 regardless of input dtype."""
    return tl.exp(x.to(tl.float32))


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
