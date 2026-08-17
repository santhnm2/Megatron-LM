# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The modules in this package are forked from the Gated Delta Product kernels
# in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention), licensed under the MIT
# license. See the LICENSE file at the repository root.

"""Inference-only fork of the Gated Delta Product kernels from
`flash-linear-attention <https://github.com/fla-org/flash-linear-attention>`_
(v0.5.1).

Training and the static-batching inference path call the pip
`flash-linear-attention` kernels, which own the backward pass. Only the
dynamic-batching decode and prefill steps route here, because those are the ones
that must be CUDA-graph capturable and padding-aware:

* `fused_recurrent` -- decode. Reads and writes the recurrent state in place at
  slots named by `state_indices`.
* `chunk` -- prefill. Takes precomputed chunk descriptors (see `metadata`)
  instead of deriving them with a device-to-host sync, and writes the final
  state in place.

Both entry points are forward-only, and both treat `-1` in the slot indices as a
padding request: zero output, no state access. That is what lets a graph
captured at a rounded-up batch shape replay correctly for a step with fewer real
requests.
"""

from .chunk import chunk_gated_delta_product_varlen
from .fused_recurrent import fused_recurrent_gated_delta_rule_update
from .metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts

__all__ = [
    "chunk_gated_delta_product_varlen",
    "fused_recurrent_gated_delta_rule_update",
    "build_gdp_chunk_descriptors",
    "max_gdp_chunk_counts",
]
