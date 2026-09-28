# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from vLLM Mixer2RMSNormGated (Apache-2.0).

"""Grouped gated RMSNorm with the reference vLLM operation/rounding order."""

import torch
from torch.nn import functional as F


def grouped_gated_rmsnorm(module, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Match vLLM's local-group path, including compiler specialization.

    Pass the module rather than standalone tensor/scalar arguments so that
    torch.compile treats its parameter shape and group size as constants,
    as in the reference. Only the activation batch dimension is dynamic.
    """
    input_dtype = x.dtype
    x = x * F.silu(gate.to(torch.float32))
    *prefix_dims, hidden_dim = x.shape
    group_count = hidden_dim // module.group_size
    x_grouped = x.view(*prefix_dims, group_count, module.group_size)
    variance = x_grouped.pow(2).mean(-1, keepdim=True)
    x_grouped = x_grouped * torch.rsqrt(variance + module.eps)
    x = x_grouped.view(*prefix_dims, hidden_dim)
    return module.weight * x.to(input_dtype)


_compiled_grouped_gated_rmsnorm = torch.compile(
    grouped_gated_rmsnorm,
    fullgraph=True,
    dynamic=False,
    # Match the reference despite nemoRL's policy-worker global override.
    options={"autotune_local_cache": True},
)


def compiled_grouped_gated_rmsnorm(module, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Specialize group width and epsilon as in the reference compilation."""
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.mark_dynamic(gate, 0)
    return _compiled_grouped_gated_rmsnorm(module, x, gate)
