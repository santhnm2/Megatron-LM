# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Inference-only fork of the Gated Delta Product kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.4.2), mirroring the Mamba2 kernel fork in the sibling `mamba2` package.

Training and the static-batching inference path keep calling the pip
`flash-linear-attention` kernels, which own the backward pass. Only the
dynamic-batching decode and prefill steps route here:

* `fused_recurrent_gated_delta_rule_update` -- decode.
* `chunk_gated_delta_product_varlen` -- prefill.

The fork is forward-only: the backward kernels, the `torch.autograd.Function`
wrappers and the paths this caller never reaches (fixed-length batching, the
non-gated delta rule, chunk lengths other than 64, the `gk` / `gv` gates) are
dropped. What remains is upstream's code, unmodified -- same math, same
autotune configurations, same host-synchronizing chunk-descriptor derivation.
Making these kernels CUDA-graph capturable is a separate change on top.
"""

from .chunk import chunk_gated_delta_product_varlen
from .fused_recurrent import fused_recurrent_gated_delta_rule_update

__all__ = ["chunk_gated_delta_product_varlen", "fused_recurrent_gated_delta_rule_update"]
