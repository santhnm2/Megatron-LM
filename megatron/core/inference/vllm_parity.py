# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental Nemotron-H parity with the pinned vLLM numerical profile.

This adapter keeps Megatron parameters and state management. It deliberately
uses locally maintained reference primitives and their normal dispatch/config
selection. No Triton tuning winner is pinned here.
"""

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn import functional as F

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols


def configure_ssd_autotune_cache():
    """Match the reference disk-cache default despite earlier Megatron imports.

    vLLM sets this environment default before importing its SSD autotuners.
    Megatron imports its kernels before the parity adapter, so their existing
    objects need the same setting. Explicit environment overrides still apply.
    Candidate lists, benchmark selection and existing winners are preserved.
    """
    from megatron.core.ssm.ops.mamba2 import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    enabled = os.environ.get('TRITON_CACHE_AUTOTUNING', '1') == '1'
    for kernel in (
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_chunk_scan._chunk_scan_fwd_kernel,
        ssd_state_passing._state_passing_fwd_kernel,
        ssd_bmm._bmm_chunk_fwd_kernel,
    ):
        kernel.cache_results = enabled


def combine_shared_routed(layer, shared, routed):
    """Keep the reference compiler's scale/add boundary before TP reduction."""
    shared.copy_(shared + routed * layer.config.moe_router_topk_scaling_factor)
    return shared


# nemoRL disables this cache globally in policy workers. The reference
# generation worker keeps it enabled; scope the override to parity helpers.
_compiled_combine_shared_routed = torch.compile(
    combine_shared_routed, fullgraph=True, dynamic=False, options={"autotune_local_cache": True}
)


def compiled_combine_shared_routed(layer, shared, routed):
    """Specialize width and scaling factor, with a dynamic token dimension."""
    torch._dynamo.mark_dynamic(shared, 0)
    torch._dynamo.mark_dynamic(routed, 0)
    return _compiled_combine_shared_routed(layer, shared, routed)


def residual_rmsnorm(x, residual, weight, epsilon, keep_residual_fp32=False):
    """Mirror vLLM IR normalization and its compiled residual boundaries.

    Adapted from vLLM ir/ops/layernorm.py (Apache-2.0). Across an opaque MoE,
    the reference retains the two BF16 components and recomputes their FP32
    sum in the next norm. Materializing an extra FP32 output here changes
    Inductor's arithmetic fusion. Mamba/attention boundaries store BF16.
    """
    dtype = x.dtype
    if residual is None:
        residual = x
        x = x.float()
    else:
        previous = (
            residual[0].float() + residual[1].float()
            if isinstance(residual, tuple)
            else residual.float()
        )
        summed = x.float() + previous
        # Preserve the unrounded sum without adding a store to the norm-only
        # compiler partition. The next norm reconstructs it from these inputs.
        residual = (x, residual) if keep_residual_fp32 else summed.to(dtype)
        x = summed
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return (x.to(weight.dtype) * weight).to(dtype), residual


def residual_first_rmsnorm(x, residual, weight, epsilon, keep_residual_fp32=False, final=False):
    """Expose the residual first, matching the reference partition outputs.

    Output order affects Inductor's placement of the residual store relative
    to the variance reduction, which can change its floating-point fusion.
    """
    normalized, residual = residual_rmsnorm(x, residual, weight, epsilon, keep_residual_fp32)
    if final:
        # The final reference partition reuses the hidden buffer and has no
        # residual output. An extra output changes the native tuning workload.
        x.copy_(normalized)
        return None, x
    return residual, normalized


_compiled_residual_first_rmsnorm = torch.compile(
    residual_first_rmsnorm, fullgraph=True, dynamic=False, options={"autotune_local_cache": True}
)


def compiled_residual_rmsnorm(x, residual, weight, epsilon, keep_residual_fp32=False, final=False):
    """Keep batch size dynamic and the model's width/epsilon specialized."""

    def mark_batch(value):
        if isinstance(value, tuple):
            for component in value:
                mark_batch(component)
        elif value is not None:
            torch._dynamo.mark_dynamic(value, 0)

    mark_batch(x)
    mark_batch(residual)
    residual, normalized = _compiled_residual_first_rmsnorm(
        x, residual, weight, epsilon, keep_residual_fp32, final
    )
    return normalized, residual


class VllmHybridParity:
    """Execute the supported hybrid blocks with vLLM rounding boundaries."""

    def __init__(self, stack):
        config = stack.config
        if (
            config.pipeline_model_parallel_size != 1
            or config.context_parallel_size != 1
            or config.expert_model_parallel_size != 1
            or config.expert_tensor_parallel_size != config.tensor_model_parallel_size
            or config.params_dtype != torch.bfloat16
            or config.fp8
            or config.fp4
            or config.add_bias_linear
            or config.gated_linear_unit
            or not stack.post_process
            or not stack.pre_process
        ):
            raise ValueError('vLLM parity requires BF16 Nemotron-H, PP=CP=EP=1 and ETP=TP')
        if set(stack.layer_type_list) - {Symbols.MAMBA, Symbols.MOE, Symbols.ATTENTION}:
            raise ValueError('vLLM parity supports Mamba, ReLU-squared MoE, and attention only')

        from megatron.core.inference.parity_kernels.collectives import CudaCommunicator
        from megatron.core.inference.parity_kernels.ops import load_ops

        configure_ssd_autotune_cache()
        self.ops = load_ops()
        self.compile_norm = config.inference_vllm_compile_norm
        self.tp_group = stack.pg_collection.tp
        self.tp_size = dist.get_world_size(self.tp_group)
        self.tp_rank = dist.get_rank(self.tp_group)
        ranks = dist.get_process_group_ranks(self.tp_group)
        self.cpu_group = dist.new_group(ranks=ranks, backend='gloo', use_local_synchronization=True)
        self.communicator = CudaCommunicator(
            self.cpu_group, torch.device('cuda', torch.cuda.current_device())
        )
        self.observer = None
        self._compiled_norms_warmed = False
        self._row_projections = []
        for symbol, layer in zip(stack.layer_type_list, stack.layers):
            if symbol == Symbols.MAMBA:
                layer.mixer.in_proj._vllm_normalized_replicated_input = True
                layer.mixer.out_proj._vllm_communicator = self.communicator
                self._row_projections.append(layer.mixer.out_proj)
                layer.mixer.norm.inference_vllm_compile_norm = config.inference_vllm_compile_norm
            elif symbol == Symbols.ATTENTION:
                layer.self_attention.linear_proj._vllm_communicator = self.communicator
                self._row_projections.append(layer.self_attention.linear_proj)

    def observe(self, name: str, **tensors) -> None:
        """Optional diagnostic observer; normally absent from the execution path."""
        if self.observer is not None:
            self.observer(name, tensors)

    def cuda_graph_capture_context(self):
        """Register reference collective IPC addresses after CUDA capture ends."""
        communicator = self.communicator.ca_comm
        return communicator.capture() if communicator is not None else nullcontext()

    @torch.no_grad()
    def warmup_compiled_norms(self, stack, max_tokens):
        """Trace norms at the reference's prefill compilation shape.

        vLLM compiles its model using the scheduler's maximum token budget.
        Inductor uses that first shape to select reduction implementations,
        even for subsequent dynamic batches. Match the compilation input
        before Megatron warms up smaller CUDA graph shapes. The two engines
        must be configured with the same maximum token budget.
        """
        if self._compiled_norms_warmed or not self.compile_norm:
            return
        from megatron.core.ssm.ops.vllm_grouped_rmsnorm import compiled_grouped_gated_rmsnorm

        for symbol, layer in zip(stack.layer_type_list, stack.layers):
            if symbol != Symbols.MAMBA:
                continue
            mixer = layer.mixer
            weight = mixer.norm.weight
            width = weight.numel()
            projected_width = mixer.in_proj.weight.shape[0]
            x = torch.zeros((max_tokens, width), dtype=weight.dtype, device=weight.device)
            gate = torch.empty_strided(
                (max_tokens, width), (projected_width, 1), dtype=weight.dtype, device=weight.device
            )
            gate.zero_()
            compiled_grouped_gated_rmsnorm(mixer.norm, x, gate)
        residual = None
        for symbol, layer in zip(stack.layer_type_list, stack.layers):
            if symbol == Symbols.MAMBA:
                weight = layer.mixer.in_proj.layer_norm_weight
            elif symbol == Symbols.ATTENTION:
                weight = layer.self_attention.linear_qkv.layer_norm_weight
            else:
                weight = layer.pre_mlp_layernorm.weight
            # Each block's hidden output has distinct storage from the carry.
            # Preserve that alias relationship in the compilation inputs.
            x = torch.zeros((max_tokens, weight.numel()), dtype=weight.dtype, device=weight.device)
            _, residual = compiled_residual_rmsnorm(
                x,
                residual,
                weight,
                stack.config.layernorm_epsilon,
                keep_residual_fp32=symbol == Symbols.MOE,
            )
        x = torch.zeros_like(x)
        compiled_residual_rmsnorm(
            x, residual, stack.final_norm.weight, stack.config.layernorm_epsilon, final=True
        )
        if stack.config.inference_vllm_compile_moe:
            for symbol, layer in zip(stack.layer_type_list, stack.layers):
                if symbol == Symbols.MOE:
                    # Match the reference's initial scheduler-budget shape and
                    # reuse of the shared-expert output before TP reduction.
                    shared = torch.zeros_like(x)
                    routed = torch.zeros_like(x)
                    compiled_combine_shared_routed(layer, shared, routed)
                    break
        self._compiled_norms_warmed = True

    def norm(self, x, residual, weight, eps, keep_residual_fp32=False, final=False):
        """Retain vLLM's fused residual-add/RMSNorm boundary."""
        x = x.reshape(-1, x.shape[-1])
        if self.compile_norm:
            return compiled_residual_rmsnorm(x, residual, weight, eps, keep_residual_fp32, final)
        if residual is None:
            residual = x
            normalized = torch.empty_like(x)
            self.ops.rms_norm(normalized, x, weight, eps)
        else:
            normalized = x.clone()
            # The reference mutates both arguments and returns both tensors.
            self.ops.fused_add_rms_norm(normalized, residual, weight, eps)
        return normalized, residual

    def moe(self, layer, x):
        """TP-sharded experts, with reference routing/activation/weighting order."""
        from megatron.core.inference.parity_kernels.moe import fused_experts, grouped_topk

        router = layer.router
        logits = torch.mm(x, router.weight.T, out_dtype=torch.float32)
        weights, ids = grouped_topk(logits, router.expert_bias, layer.config.moe_router_topk)
        experts = layer.experts
        if not experts._concatenated_weights_built:
            experts._build_concatenated_weights()
            experts._concatenated_weights_built = True
        routed = fused_experts(x, experts._fc1_weight, experts._fc2_weight, weights, ids)
        shared = layer.shared_experts
        shared_up = F.linear(x, shared.linear_fc1.weight)
        shared_out = F.linear(F.relu(shared_up).square(), shared.linear_fc2.weight)
        self.observe(
            'moe.local', logits=logits, weights=weights, ids=ids, routed=routed, shared=shared_out
        )
        combine = (
            compiled_combine_shared_routed
            if layer.config.inference_vllm_compile_moe
            else combine_shared_routed
        )
        return self.communicator.all_reduce(combine(layer, shared_out, routed))

    def forward(self, stack, hidden_states, context):
        """Carry the residual separately through the hybrid stack."""
        if context is None or not context.is_dynamic_batching():
            raise ValueError('vLLM parity currently requires the dynamic inference context')
        if stack.training:
            raise RuntimeError('The parity path is inference only')
        self.warmup_compiled_norms(stack, context.max_tokens)
        # Ungraphed reference prefill primitives use actual token counts.
        # Padding changes router GEMM selection and NCCL's reduction order.
        # Decode retains the physical graph shape used during capture.
        real_tokens = None if context.is_decode_only() else context.active_token_count
        for projection in self._row_projections:
            projection._vllm_token_count = real_tokens
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        if stack.config.sequence_parallel and self.tp_size > 1:
            gathered = torch.empty(
                (x.shape[0] * self.tp_size, x.shape[1]), dtype=x.dtype, device=x.device
            )
            dist.all_gather_into_tensor(gathered, x.contiguous(), group=self.tp_group)
            x = gathered
        residual = None
        for index, (symbol, layer) in enumerate(zip(stack.layer_type_list, stack.layers)):
            if symbol == Symbols.MAMBA:
                weight = layer.mixer.in_proj.layer_norm_weight
            elif symbol == Symbols.ATTENTION:
                weight = layer.self_attention.linear_qkv.layer_norm_weight
            else:
                weight = layer.pre_mlp_layernorm.weight
            self.observe(f'layer.{index}.input', hidden=x, residual=residual)
            x, residual = self.norm(
                x,
                residual,
                weight,
                stack.config.layernorm_epsilon,
                keep_residual_fp32=symbol == Symbols.MOE,
            )
            self.observe(f'layer.{index}.normalized', hidden=x, residual=residual)
            if symbol == Symbols.MAMBA:
                x, bias = layer.mixer(x.unsqueeze(1), inference_context=context)
            elif symbol == Symbols.ATTENTION:
                x, bias = layer.self_attention(
                    x.unsqueeze(1), attention_mask=None, inference_context=context
                )
            else:
                physical_tokens = x.shape[0]
                x, bias = self.moe(layer.mlp, x[:real_tokens]), None
                if x.shape[0] < physical_tokens:
                    x = F.pad(x, (0, 0, 0, physical_tokens - x.shape[0]))
            if bias is not None:
                raise RuntimeError('The parity profile does not support projection bias')
            x = x.reshape(-1, shape[-1])
            self.observe(f'layer.{index}.output', hidden=x, residual=residual)
        x, residual = self.norm(
            x, residual, stack.final_norm.weight, stack.config.layernorm_epsilon, final=True
        )
        self.observe('final_norm', hidden=x)
        if stack.config.sequence_parallel and self.tp_size > 1:
            x = x.chunk(self.tp_size, dim=0)[self.tp_rank].contiguous()
        return x.view(shape)


def attention_qkv(attention, hidden_states):
    """Map checkpoint QKV shards to vLLM's replicated-KV projection layout.

    Rebuild from live parameters on every call so weight refits are visible,
    including when the call is captured by a CUDA graph.
    """
    config = attention.config
    group = attention.pg_collection.tp
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    weight = attention.linear_qkv.weight
    full = torch.empty(
        (weight.shape[0] * size, weight.shape[1]), dtype=weight.dtype, device=weight.device
    )
    dist.all_gather_into_tensor(full, weight.contiguous(), group=group)
    head_dim = config.kv_channels
    query_heads_per_group = config.num_attention_heads // config.num_query_groups
    grouped = full.view(config.num_query_groups, (query_heads_per_group + 2) * head_dim, -1)
    queries = grouped[:, : query_heads_per_group * head_dim].reshape(-1, weight.shape[1])
    keys = grouped[
        :, query_heads_per_group * head_dim : (query_heads_per_group + 1) * head_dim
    ].reshape(-1, weight.shape[1])
    values = grouped[:, (query_heads_per_group + 1) * head_dim :].reshape(-1, weight.shape[1])
    q_width = config.num_attention_heads // size * head_dim
    if config.num_query_groups >= size:
        kv_width = config.num_query_groups // size * head_dim
        kv_start = rank * kv_width
    else:
        kv_width = head_dim
        kv_start = rank // (size // config.num_query_groups) * kv_width
    local_weight = torch.cat(
        (
            queries[rank * q_width : (rank + 1) * q_width],
            keys[kv_start : kv_start + kv_width],
            values[kv_start : kv_start + kv_width],
        )
    )
    projected = F.linear(hidden_states, local_weight)
    q, k, v = projected.split((q_width, kv_width, kv_width), dim=-1)
    prefix = hidden_states.shape[:-1]
    return tuple(t.reshape(*prefix, -1, head_dim) for t in (q, k, v))
