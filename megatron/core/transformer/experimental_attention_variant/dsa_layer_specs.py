# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hybrid-stack specs for CuTe DSA over grouped-query attention.

Each spec is its hybrid counterpart with the ordinary attention layer's
self-attention swapped for ``DSGroupedSelfAttention``. Wide-residual specs reuse the
attention-swapped stack for their MTP submodules, so MTP attention is swapped too.
"""

import copy

from megatron.core.models.hybrid.hybrid_layer_specs import (
    _get_wide_residual_hybrid_stack_spec,
    gated_delta_product_inference_stack_spec,
    gated_delta_product_stack_spec,
    hybrid_inference_stack_spec,
    hybrid_stack_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import DSGroupedSelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec


def _with_dsa_gqa_attention(stack_spec: ModuleSpec) -> ModuleSpec:
    """Replace only a hybrid stack's ordinary attention implementation."""
    spec = copy.deepcopy(stack_spec)
    spec.submodules.attention_layer.submodules.self_attention.module = DSGroupedSelfAttention
    return spec


dsa_stack_spec = _with_dsa_gqa_attention(hybrid_stack_spec)
gdp_dsa_stack_spec = _with_dsa_gqa_attention(gated_delta_product_stack_spec)
dsa_inference_stack_spec = _with_dsa_gqa_attention(hybrid_inference_stack_spec)
gdp_dsa_inference_stack_spec = _with_dsa_gqa_attention(gated_delta_product_inference_stack_spec)

wide_residual_dsa_stack_spec = _get_wide_residual_hybrid_stack_spec(dsa_stack_spec)
wide_residual_gdp_dsa_stack_spec = _get_wide_residual_hybrid_stack_spec(gdp_dsa_stack_spec)
wide_residual_dsa_inference_stack_spec = _get_wide_residual_hybrid_stack_spec(
    dsa_inference_stack_spec
)
wide_residual_gdp_dsa_inference_stack_spec = _get_wide_residual_hybrid_stack_spec(
    gdp_dsa_inference_stack_spec
)

__all__ = [
    "dsa_stack_spec",
    "gdp_dsa_stack_spec",
    "dsa_inference_stack_spec",
    "gdp_dsa_inference_stack_spec",
    "wide_residual_dsa_stack_spec",
    "wide_residual_gdp_dsa_stack_spec",
    "wide_residual_dsa_inference_stack_spec",
    "wide_residual_gdp_dsa_inference_stack_spec",
]
