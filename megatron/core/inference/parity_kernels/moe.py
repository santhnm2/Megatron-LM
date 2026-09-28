# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Megatron: BF16, single expert group, ReLU-squared only.
"""Reference routing, GEMM policy and rounding without the vLLM runtime."""

import functools
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .moe_gemm import fused_moe_kernel
from .ops import load_ops


@functools.lru_cache
def get_configs(experts, width):
    """Use the GB300 fallback or an explicitly supplied reference table directory."""
    device = torch.cuda.get_device_name().replace(' ', '_')
    if 'H200' in device.split('_'):
        device = 'NVIDIA_H200'
    filename = f'E={experts},N={width},device_name={device}.json'
    config_dir = os.environ.get('MEGATRON_PARITY_MOE_CONFIG_DIR')
    if config_dir:
        path = Path(config_dir) / filename
        if path.is_file():
            values = json.loads(path.read_text())
            values.pop('triton_version', None)
            return {int(k): v for k, v in values.items()}
    elif device != 'NVIDIA_GB300':
        raise ValueError(
            'The bundled MoE policy covers GB300, which has no reference tuning table. '
            'For another device, set MEGATRON_PARITY_MOE_CONFIG_DIR to the pinned '
            "reference's complete BF16 configuration directory."
        )
    return None


def get_config(tokens, experts, width):
    """Match vLLM 0.25.1's BF16 fallback, including stage/warp selection."""
    configs = get_configs(experts, width)
    if configs:
        return configs[min(configs, key=lambda k: abs(k - tokens))]
    return dict(
        BLOCK_SIZE_M=16 if tokens <= 32 else 32 if tokens <= 96 else 64 if tokens <= 512 else 128,
        BLOCK_SIZE_N=64 if tokens <= 64 else 128,
        BLOCK_SIZE_K=128 if tokens <= 64 else 64,
        GROUP_SIZE_M=16 if tokens // max(experts, 1) > 128 else 1,
        SPLIT_K=1,
        num_warps=4 if tokens <= 128 else 8,
        num_stages=4 if tokens <= 32 else 3,
    )


def grouped_topk(logits, bias, topk):
    """Select with biased sigmoid scores; normalize the unbiased weights."""
    return load_ops().grouped_topk(logits, 1, 1, topk, True, 1.0, bias, 1)


def assignments(ids, block, experts):
    """Keep the reference's small-batch direct assignment and aligned path."""
    if ids.numel() * 4 <= experts:
        return (
            None,
            ids.view(-1),
            torch.full((1,), ids.numel() * block, dtype=torch.int32, device=ids.device),
        )
    capacity = ids.numel() + experts * (block - 1)
    if ids.numel() < experts:
        capacity = min(ids.numel() * block, capacity)
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=ids.device)
    expert_ids = torch.empty(triton.cdiv(capacity, block), dtype=torch.int32, device=ids.device)
    padded = torch.empty(1, dtype=torch.int32, device=ids.device)
    load_ops().moe_align_block_size(ids, experts, block, sorted_ids, expert_ids, padded, None)
    return sorted_ids, expert_ids, padded


def matmul(a, b, c, weights, sorted_ids, expert_ids, padded, weighted, topk, config):
    """Launch the original Triton kernel with the BF16 specialization flags."""
    valid = a.shape[0] * topk
    em = valid * config['BLOCK_SIZE_M'] if sorted_ids is None else sorted_ids.numel()
    if sorted_ids is not None and a.shape[0] < config['BLOCK_SIZE_M']:
        em = min(em, valid * config['BLOCK_SIZE_M'])
    grid = (
        triton.cdiv(em, config['BLOCK_SIZE_M']) * triton.cdiv(b.shape[1], config['BLOCK_SIZE_N']),
    )
    fused_moe_kernel[grid](
        a,
        b,
        c,
        None,
        None,
        None,
        weights,
        sorted_ids,
        expert_ids,
        padded,
        b.shape[1],
        b.shape[2],
        em,
        valid,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        naive_block_assignment=sorted_ids is None,
        MUL_ROUTED_WEIGHT=weighted,
        top_k=topk,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        HAS_BIAS=False,
        SWAP_AB=False,
        **config,
    )


def fused_experts(x, w1, w2, weights, ids):
    """Preserve FC1 BF16 rounding, BF16 ReLU/square, and ordered expert sum."""
    assert x.dtype == w1.dtype == w2.dtype == torch.bfloat16
    assert x.is_contiguous() and w1.stride(-1) == w2.stride(-1) == 1
    tokens, topk = ids.shape
    experts, up, hidden = w1.shape
    assert x.shape == (tokens, hidden) and w2.shape == (experts, hidden, up)
    config = get_config(tokens, experts, up)
    storage = torch.empty(tokens * topk * max(up, hidden), dtype=x.dtype, device=x.device)
    first = storage[: tokens * topk * up].view(tokens, topk, up)
    second = storage[: tokens * topk * hidden].view(tokens, topk, hidden)
    activated = torch.empty((tokens * topk, up), dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    sorted_ids, expert_ids, padded = assignments(ids, config['BLOCK_SIZE_M'], experts)
    matmul(x, w1, first, weights, sorted_ids, expert_ids, padded, False, topk, config)
    F.relu(first.view(-1, up), inplace=True)
    torch.square(first.view(-1, up), out=activated)
    matmul(activated, w2, second, weights, sorted_ids, expert_ids, padded, True, 1, config)
    load_ops().moe_sum(second, output)
    return output
