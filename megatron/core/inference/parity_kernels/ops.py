# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Build the small local CUDA extension before inference or graph capture."""

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_ops():
    """Load local kernels; compilation requires the CUDA toolkit and PyTorch 2.11+."""
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent / 'csrc'
    load(
        name='megatron_parity_cuda',
        sources=[
            str(root / name)
            for name in (
                'bindings.cpp',
                'libtorch_stable/layernorm_kernels.cu',
                'libtorch_stable/moe/moe_align_sum_kernels.cu',
                'libtorch_stable/moe/grouped_topk_kernels.cu',
                'libtorch_stable/custom_all_reduce.cu',
            )
        ],
        extra_include_paths=[str(root), str(root / 'libtorch_stable')],
        extra_cflags=[
            '-O3',
            '-std=c++20',
            '-DUSE_CUDA',
            '-fvisibility=hidden',
            '-DTORCH_TARGET_VERSION=0x020B000000000000ULL',
        ],
        extra_cuda_cflags=[
            '-O3',
            '-std=c++20',
            '-DUSE_CUDA',
            '-DENABLE_FP8',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
            '-U__CUDA_NO_HALF2_OPERATORS__',
            '-Xcompiler=-fvisibility=hidden',
            '-DTORCH_TARGET_VERSION=0x020B000000000000ULL',
        ],
        is_python_module=False,
        extra_ldflags=['-lcuda'],
    )
    return torch.ops.mcore_parity
