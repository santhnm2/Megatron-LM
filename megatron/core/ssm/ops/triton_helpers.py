# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from vLLM (Apache-2.0).

"""Numerical expressions shared with the reference SSD forward kernels."""

import triton
import triton.language as tl


@triton.jit
def fast_exp(x):
    """Use the reference's explicit exp2 expression, including signed zeros."""
    log2e = tl.constexpr(1.4426950408889634)
    return tl.math.exp2(log2e * x)
