# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exact eager/graph replay with changing inputs and no vLLM imports.

Run on the audited four-GPU NVLink/CUDA 13 profile with the inference-parity
dependencies installed. This file also supports direct torchrun execution,
which preserves the production NCCL settings instead of unit-suite defaults::

    torchrun --standalone --nproc-per-node=4 \
        tests/unit_tests/determinism/kernels/test_local_parity_kernels.py

These replay tests detect stale graph inputs and arithmetic changes between
eager and graph execution. Reference-engine equivalence is a separate audit.
"""

import importlib.abc
import os
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
import torch.distributed as dist


def has_parity_runtime():
    """The optional profile is separate from the default development extra."""
    try:
        return (
            torch.cuda.is_available()
            and torch.__version__.split('+')[0] == '2.11.0'
            and torch.cuda.get_device_capability()[0] == 10
            and version('triton') == '3.6.0'
            and version('nvidia-cutlass-dsl') == '4.5.2'
            and version('nvidia-cutlass-dsl-libs-cu13') == '4.5.2'
            and version('quack-kernels') == '0.4.1'
        )
    except PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not has_parity_runtime(), reason='requires the audited Blackwell inference-parity runtime'
)


@contextmanager
def without_vllm():
    """Fail even if a dependency attempts a lazy import during graph replay."""

    class BlockVllm(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == 'vllm' or fullname.startswith('vllm.'):
                raise AssertionError(f'Unexpected vLLM runtime dependency: {fullname}')

    assert not any(n == 'vllm' or n.startswith('vllm.') for n in sys.modules)
    blocker = BlockVllm()
    sys.meta_path.insert(0, blocker)
    try:
        yield
        assert not any(n == 'vllm' or n.startswith('vllm.') for n in sys.modules)
    finally:
        sys.meta_path.remove(blocker)


def exact(actual, expected):
    """Compare every byte, including signs of zero; reject nonfinite outputs."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def capture(function):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    return graph, output


def test_local_norm_graph_replay():
    with without_vllm():
        from megatron.core.inference.parity_kernels.ops import load_ops

        ops = load_ops()
        torch.manual_seed(41)
        for rows in (1, 127, 128, 129, 5808):
            x = torch.randn(rows, 2688, dtype=torch.bfloat16, device='cuda')
            residual = torch.randn_like(x)
            weight = torch.randn(2688, dtype=x.dtype, device=x.device)

            def run():
                plain = torch.empty_like(x)
                ops.rms_norm(plain, x, weight, 1e-5)
                fused, summed = x.clone(), residual.clone()
                ops.fused_add_rms_norm(fused, summed, weight, 1e-5)
                return plain, fused, summed

            graph, outputs = capture(run)
            for _ in range(4):
                x.normal_()
                residual.normal_()
                expected = run()
                graph.replay()
                for actual, reference in zip(outputs, expected):
                    exact(actual, reference)


def test_compiled_parity_helpers_graph_replay():
    """Compiler-specialized norms and MoE combine retain live graph inputs."""
    from types import SimpleNamespace

    from megatron.core.inference.vllm_parity import (
        compiled_combine_shared_routed,
        compiled_residual_rmsnorm,
    )
    from megatron.core.ssm.ops.vllm_grouped_rmsnorm import compiled_grouped_gated_rmsnorm

    with without_vllm(), torch.no_grad():
        torch.manual_seed(45)
        weight = torch.randn(1536, dtype=torch.bfloat16, device='cuda')
        norm = SimpleNamespace(weight=weight, group_size=768, eps=1e-5)
        layer = SimpleNamespace(config=SimpleNamespace(moe_router_topk_scaling_factor=2.5))
        # Compile with the audited prefill width and projection-view gate stride.
        for rows in (8480, 4, 128):
            x = torch.randn(rows, 1536, dtype=weight.dtype, device=weight.device)
            gate = torch.randn(rows, 2576, dtype=weight.dtype, device=weight.device)[:, :1536]
            residual = torch.randn_like(x)

            def run():
                gated = compiled_grouped_gated_rmsnorm(norm, x, gate)
                normalized, carried = compiled_residual_rmsnorm(
                    x, residual, weight, 1e-5, keep_residual_fp32=True
                )
                combined = compiled_combine_shared_routed(layer, x.clone(), residual)
                return gated, normalized, *carried, combined

            graph, outputs = capture(run)
            for _ in range(3):
                x.normal_()
                gate.normal_()
                residual.normal_()
                expected = run()
                graph.replay()
                for actual, reference in zip(outputs, expected):
                    exact(actual, reference)


def test_local_moe_graph_replay():
    with without_vllm():
        from megatron.core.inference.parity_kernels.moe import fused_experts, grouped_topk

        torch.manual_seed(42)
        experts, hidden, intermediate, topk = 128, 2688, 464, 6
        w1 = torch.randn(experts, intermediate, hidden, dtype=torch.bfloat16, device='cuda') * 0.02
        w2 = torch.randn(experts, hidden, intermediate, dtype=w1.dtype, device=w1.device) * 0.02
        bias = torch.randn(experts, device='cuda') * 0.1
        for rows in (1, 32, 128, 129, 8480):
            x = torch.randn(rows, hidden, dtype=w1.dtype, device=w1.device)
            logits = torch.randn(rows, experts, device='cuda')

            def run():
                weights, ids = grouped_topk(logits, bias, topk)
                return fused_experts(x, w1, w2, weights, ids)

            graph, output = capture(run)
            previous = output.clone()
            for _ in range(4):
                x.normal_()
                logits.normal_()
                expected = run()
                graph.replay()
                exact(output, expected)
                assert not torch.equal(output, previous)
                previous.copy_(output)


def test_local_attention_graph_replay():
    with without_vllm():
        from megatron.core.inference.parity_kernels.cute.interface import _flash_attn_fwd

        torch.manual_seed(43)
        q = torch.randn(1, 8, 128, dtype=torch.bfloat16, device='cuda')
        k = torch.randn(768, 256, 1, 128, dtype=q.dtype, device=q.device)
        v = torch.randn_like(k)
        lengths = torch.tensor([196480], dtype=torch.int32, device='cuda')
        metadata = dict(
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device='cuda'),
            seqused_k=lengths,
            max_seqlen_q=1,
            max_seqlen_k=196608,
            page_table=torch.arange(768, dtype=torch.int32, device='cuda').view(1, -1),
            causal=True,
            softmax_scale=128**-0.5,
            num_splits=0,
            return_lse=False,
        )

        def run():
            return _flash_attn_fwd(q, k, v, **metadata)[0]

        graph, output = capture(run)
        for length in (1, 255, 256, 257, 196480):
            lengths.fill_(length)
            q.normal_()
            expected = run()
            graph.replay()
            exact(output, expected)


def test_local_collective_graph_replay():
    if not dist.is_initialized() or dist.get_world_size() != 4:
        pytest.skip('requires four ranks on one NVLink node')
    with without_vllm():
        from megatron.core.inference.parity_kernels.collectives import CudaCommunicator

        group = dist.new_group(backend='gloo')
        comm = CudaCommunicator(group, torch.device('cuda', torch.cuda.current_device()))
        torch.manual_seed(44 + dist.get_rank())
        # Custom, symmetric-memory and NCCL dispatch on the audited GB300 profile.
        for rows in (4, 16, 128, 608, 8480):
            x = torch.randn(rows, 2688, dtype=torch.bfloat16, device='cuda')
            comm.all_reduce(x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with comm.ca_comm.capture():
                with torch.cuda.graph(graph):
                    output = comm.all_reduce(x)
            for _ in range(4):
                x.normal_()
                expected = comm.all_reduce(x)
                graph.replay()
                exact(output, expected)
            del graph, output
        torch.cuda.synchronize()
        dist.barrier(group)
        del comm
        dist.destroy_process_group(group)


if __name__ == '__main__':
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    assert has_parity_runtime(), 'Install the audited inference-parity runtime before running'
    dist.init_process_group('gloo')
    for test in (
        test_local_norm_graph_replay,
        test_compiled_parity_helpers_graph_replay,
        test_local_moe_graph_replay,
        test_local_attention_graph_replay,
        test_local_collective_graph_replay,
    ):
        test()
        print(f'PASS rank={dist.get_rank()} {test.__name__}', flush=True)
    dist.barrier()
    dist.destroy_process_group()
