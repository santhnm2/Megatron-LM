# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""End-to-end test for Mamba prefix caching with a real hybrid model.

This test exercises the 4 key indices within a Mamba prefill:

  1. num_mamba_matched — how many blocks have cached Mamba state.
     Determines how many tokens the prefill can skip.

  2. num_kv_matched — how many KV blocks are shared with prior
     requests. Can exceed num_mamba_matched, since KV blocks are
     always registered for every completed block, while Mamba
     state is only cached at divergence and last-aligned blocks.

  3. last_aligned_block — the last full-block boundary in the
     prompt: floor(prompt_len / block_size) * block_size. Mamba
     state is always cached here (if it falls within the
     effective prefill). This is the "end of the known prefix"
     state that future requests can restore from.

  4. end_of_sequence — the actual prompt length. When prompt_len
     is block-aligned (prompt_len == last_aligned), the final
     Mamba state is cached via the EOS path (copy from live
     buffer). When not aligned, there's a gap between
     last_aligned and end_of_sequence that doesn't get cached.

5 requests with overlapping prefixes are processed in a specific
order so that each request sees a different combination of these
indices. The test verifies both internal state (mamba cache
registration, skip counts) and output correctness (generated
tokens match between pc=off and pc=on).
"""

import os
import random
import types

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    PrefixCachingEvictionPolicy,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.mamba_mixer import _check_mamba_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import CudaGraphManager, _CudagraphGlobalRecord
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils

BLOCK_SIZE = 256
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 2048
NUM_TOKENS_TO_GENERATE = 16
# multi-group uses 4x the requests (20 vs 5), creating larger batch
# composition differences between pc=off and pc=on. reduce decode steps
# to stay within the safe bf16 rounding margin.
MULTI_GROUP_TOKENS_TO_GENERATE = 8
NUM_GROUPS = 4
GROUP_TOKEN_STRIDE = 2000


def skip_if_mamba_sequence_packing_not_available():
    sequence_packing_available, reason = _check_mamba_sequence_packing_support()
    if not sequence_packing_available:
        pytest.skip(reason)


def set_rounder(value):
    DynamicInferenceContext.ROUNDER = value
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class _HybridPCHelpers:
    """Model, engine, and request fixtures shared by the hybrid prefix-caching tests.

    Not collected by pytest (no ``Test`` prefix); mixed into the test classes
    below so they share one hybrid model definition and engine builder.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        random.seed(123)
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(
            seed=123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def _create_model(self, num_cuda_graphs=None):
        transformer_config = TransformerConfig(
            params_dtype=torch.bfloat16,
            num_layers=3,
            hidden_size=256,
            mamba_num_heads=16,
            num_attention_heads=16,
            use_cpu_initialization=True,
            cuda_graph_impl="local" if num_cuda_graphs else "none",
            inference_rng_tracker=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            add_bias_linear=True,
            is_hybrid_model=True,
        )
        model = HybridModel(
            config=transformer_config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB_SIZE,
            max_sequence_length=MAX_SEQ_LEN,
            parallel_output=True,
            hybrid_layer_pattern="M*-",
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(transformer_config.params_dtype)
        model.eval()
        return model

    def _create_prompts(self, offset=0):
        """Build 5 prompts with carefully designed prefix sharing.

        Each prompt uses disjoint token ID ranges for unique segments
        so that parent-chained block hashes differ where content differs.

        The prompts are designed so that each request hits a different
        combination of the 4 key indices:

        req 0 (300 tokens): seed request, no matches. last_aligned=256
        req 1 (800 tokens): 1 KV match, 1 Mamba match (block 0 from req 0)
        req 2 (800 tokens): 2 KV matches, but only 1 Mamba match
        req 3 (800 tokens): 2 KV matches, 2 Mamba matches
        req 4 (1100 tokens): 3 KV matches, 3 Mamba matches
        """
        device = torch.cuda.current_device()
        base = torch.arange(offset, offset + 256, dtype=torch.int64, device=device)
        seg_B = torch.arange(offset + 256, offset + 512, dtype=torch.int64, device=device)
        seg_1rest = torch.arange(offset + 512, offset + 800, dtype=torch.int64, device=device)
        seg_2rest = torch.arange(offset + 800, offset + 1088, dtype=torch.int64, device=device)
        seg_3rest = torch.arange(offset + 1088, offset + 1376, dtype=torch.int64, device=device)
        seg_4ext = torch.arange(offset + 1376, offset + 1708, dtype=torch.int64, device=device)
        extra_0 = torch.arange(offset + 1708, offset + 1752, dtype=torch.int64, device=device)

        prompts = [
            torch.cat([base, extra_0]),  # 300
            torch.cat([base, seg_B, seg_1rest]),  # 800
            torch.cat([base, seg_B, seg_2rest]),  # 800
            torch.cat([base, seg_B, seg_3rest]),  # 800
            torch.cat([base, seg_B, seg_1rest[:256], seg_4ext]),  # 1100
        ]
        assert [len(p) for p in prompts] == [300, 800, 800, 800, 1100]
        return prompts

    def _build_engine(
        self,
        model,
        mamba_config,
        enable_prefix_caching,
        buffer_size_gb=0.5,
        # max_requests is not capped, so it auto-derives from buffer_size_gb. The
        # Mamba cache budget must cover the per-step extraction scratch (which scales
        # with max_requests) on top of the durable cache, so it needs enough headroom.
        prefix_caching_mamba_gb=2.0,
        request_rounder=4,
        num_cuda_graphs=None,
        enable_chunked_prefill=False,
        max_tokens=None,
        max_requests=None,
    ):
        set_rounder(request_rounder)
        inference_config_kwargs = dict(
            max_sequence_length=MAX_SEQ_LEN,
            buffer_size_gb=buffer_size_gb,
            block_size_tokens=BLOCK_SIZE,
            mamba_inference_state_config=mamba_config,
            materialize_only_last_token_logits=False,
            enable_prefix_caching=enable_prefix_caching,
            unified_memory_level=0,
            num_cuda_graphs=num_cuda_graphs,
            sampling_backend='torch',
            enable_chunked_prefill=enable_chunked_prefill,
        )
        if max_tokens is not None:
            inference_config_kwargs['max_tokens'] = max_tokens
        if max_requests is not None:
            inference_config_kwargs['max_requests'] = max_requests
        if enable_prefix_caching:
            inference_config_kwargs.update(
                prefix_caching_eviction_policy=PrefixCachingEvictionPolicy.LRU,
                prefix_caching_mamba_gb=prefix_caching_mamba_gb,
            )
        context = DynamicInferenceContext(
            model_config=model.config, inference_config=InferenceConfig(**inference_config_kwargs)
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        _CudagraphGlobalRecord.cudagraph_created = False
        _CudagraphGlobalRecord.cudagraph_record = []
        _CudagraphGlobalRecord.cudagraph_inference_record = []
        CudaGraphManager.global_mempool = None
        for module in model.modules():
            if isinstance(module, CudaGraphManager):
                module.cudagraph_runners.clear()
                module.custom_cudagraphs_lookup_table.clear()
        return DynamicInferenceEngine(controller, context)

    def _make_request(self, req_id, prompt, enable_pc, num_tokens=NUM_TOKENS_TO_GENERATE):
        return DynamicInferenceRequest(
            request_id=req_id,
            prompt_tokens=prompt,
            sampling_params=SamplingParams(
                num_tokens_to_generate=num_tokens, termination_id=-1, top_k=1
            ),
            block_size_tokens=BLOCK_SIZE if enable_pc else None,
            enable_prefix_caching=enable_pc,
        )

    def _run_simple(
        self,
        model,
        mamba_config,
        prompts,
        enable_pc,
        base_req_id=0,
        num_tokens=NUM_TOKENS_TO_GENERATE,
        **engine_kwargs,
    ):
        """Run all prompts with given pc setting, return (finished_dict, lifetime_prefill)."""
        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=enable_pc, **engine_kwargs
        )
        for i, prompt in enumerate(prompts):
            engine._add_request(self._make_request(base_req_id + i, prompt, enable_pc, num_tokens))
        finished = {}
        while engine.has_unfinished_requests():
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)
        return finished, engine.context.lifetime_prefill_token_count


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestMambaPrefixCachingE2E(_HybridPCHelpers):
    """End-to-end test for Mamba prefix caching with a real hybrid model."""

    def _get_ref_count(self, alloc, block_hash):
        bid = alloc.kv_hash_to_block_id.get(block_hash)
        return 0 if bid is None else alloc.block_ref_counts[bid].item()

    def _assert_step(self, step, reqs_by_group, alloc, step_prefill, num_groups, ctx=None):
        """Shared per-step verification for single-group and multi-group runs."""
        G = num_groups
        if step == 1:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[0]._mamba_num_matched_blocks == 0, f"step 1 group {g}"
                assert r[0].precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G
            assert step_prefill == G * 300, f"step 1: expected {G * 300}, got {step_prefill}"
            if G == 1:
                assert (
                    self._get_ref_count(alloc, reqs_by_group[0][0].precomputed_block_hashes[0]) == 1
                )

        elif step == 2:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[1]._mamba_num_matched_blocks == 1, f"step 2 group {g}"
                assert r[1].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 2
            assert step_prefill == G * 544, f"step 2: expected {G * 544}, got {step_prefill}"
            if G == 1:
                assert (
                    self._get_ref_count(alloc, reqs_by_group[0][0].precomputed_block_hashes[0]) == 2
                )

        elif step == 3:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[2]._mamba_num_matched_blocks == 1, f"step 3 group {g} req 2"
                assert r[4]._mamba_num_matched_blocks == 3, f"step 3 group {g} req 4"
                assert r[2].precomputed_block_hashes[1] in ctx.mamba_slot_allocator.hash_to_block_id
                assert r[2].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
                assert r[4].precomputed_block_hashes[3] in ctx.mamba_slot_allocator.hash_to_block_id
                h0 = r[0].precomputed_block_hashes[0]
                h1 = r[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h0) == 4, f"step 3 group {g}"
                assert self._get_ref_count(alloc, h1) == 3, f"step 3 group {g}"
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 5
            assert step_prefill == G * (
                544 + 332
            ), f"step 3: expected {G * 876}, got {step_prefill}"

        elif step == 4:
            for g in range(G):
                r = reqs_by_group[g]
                assert r[3]._mamba_num_matched_blocks == 2, f"step 4 group {g}"
                assert r[3].precomputed_block_hashes[2] in ctx.mamba_slot_allocator.hash_to_block_id
                h0 = r[0].precomputed_block_hashes[0]
                h1 = r[1].precomputed_block_hashes[1]
                assert self._get_ref_count(alloc, h0) == 5, f"step 4 group {g}"
                assert self._get_ref_count(alloc, h1) == 4, f"step 4 group {g}"
            assert len(ctx.mamba_slot_allocator.hash_to_block_id) == G * 6
            assert step_prefill == G * 288, f"step 4: expected {G * 288}, got {step_prefill}"

    def _run_pc_on(self, model, mamba_config, prompts):
        """Run requests with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = {i: self._make_request(i, p, True) for i, p in enumerate(prompts)}
        for i in [0, 1, 2, 4]:
            engine._add_request(reqs[i])

        step = 0
        req3_added = False
        finished = {}
        prev_prefill = 0
        reqs_by_group = [{k: reqs[k] for k in reqs}]

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step <= 2 or (step == 3 and not req3_added) or (step == 4 and req3_added):
                self._assert_step(step, reqs_by_group, alloc, step_prefill, 1, ctx)
            if step == 3 and not req3_added:
                engine._add_request(reqs[3])
                req3_added = True

        return finished, ctx.lifetime_prefill_token_count

    def _run_multi_pc_on(self, model, mamba_config, all_prompts, num_cuda_graphs=None):
        """Run 4 groups with prefix caching enabled, verifying per-step state."""
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=2.0,
            # Large buffer auto-derives many max_requests, so the extraction scratch
            # is large; give the Mamba cache enough budget to cover it plus durable.
            prefix_caching_mamba_gb=4.0,
            num_cuda_graphs=num_cuda_graphs,
        )
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = []
        for g, prompts in enumerate(all_prompts):
            group_reqs = {}
            for lid, prompt in enumerate(prompts):
                rid = g * 5 + lid
                group_reqs[lid] = self._make_request(
                    rid, prompt, True, MULTI_GROUP_TOKENS_TO_GENERATE
                )
            reqs.append(group_reqs)

        for g in range(NUM_GROUPS):
            for lid in [0, 1, 2, 4]:
                engine._add_request(reqs[g][lid])

        step = 0
        req3_added = False
        finished = {}
        prev_prefill = 0

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step <= 2 or (step == 3 and not req3_added) or (step == 4 and req3_added):
                self._assert_step(step, reqs, alloc, step_prefill, NUM_GROUPS, ctx)
            if step == 3 and not req3_added:
                for g in range(NUM_GROUPS):
                    engine._add_request(reqs[g][3])
                req3_added = True

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_prefix_caching_e2e(self):
        """Verify output tokens match between pc=off and pc=on."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_prompts()

        off_outputs, off_prefill = self._run_simple(model, mamba_config, prompts, False)
        on_outputs, on_prefill = self._run_pc_on(model, mamba_config, prompts)

        for req_id in range(5):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"
        assert off_prefill == 3800 and on_prefill == 2008 and on_prefill < off_prefill

    @pytest.mark.parametrize("num_cuda_graphs", [None, 2])
    @torch.inference_mode()
    def test_mamba_prefix_caching_multi_group_e2e(self, num_cuda_graphs):
        """Verify multi-group prefix caching with 4 independent groups."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model(num_cuda_graphs=num_cuda_graphs)
        mamba_config = MambaInferenceStateConfig.from_model(model)
        all_prompts = [self._create_prompts(g * GROUP_TOKEN_STRIDE) for g in range(NUM_GROUPS)]

        _, off_prefill = self._run_simple(
            model,
            mamba_config,
            [p for group in all_prompts for p in group],
            False,
            num_tokens=MULTI_GROUP_TOKENS_TO_GENERATE,
            num_cuda_graphs=num_cuda_graphs,
            buffer_size_gb=2.0,
            prefix_caching_mamba_gb=0.2,
        )
        on_outputs, on_prefill = self._run_multi_pc_on(
            model, mamba_config, all_prompts, num_cuda_graphs=num_cuda_graphs
        )

        # verify per-group outputs match independent runs
        for g in range(NUM_GROUPS):
            ref_outputs, _ = self._run_simple(
                model,
                mamba_config,
                all_prompts[g],
                True,
                base_req_id=g * 5,
                num_tokens=MULTI_GROUP_TOKENS_TO_GENERATE,
                num_cuda_graphs=num_cuda_graphs,
            )
            for lid in range(5):
                rid = g * 5 + lid
                assert (
                    on_outputs[rid] == ref_outputs[rid]
                ), f"group {g} req {lid}: multi {on_outputs[rid]} != per-group {ref_outputs[rid]}"

        assert off_prefill == NUM_GROUPS * 3800
        assert on_prefill == NUM_GROUPS * 2008 and on_prefill < off_prefill

    def _create_block_aligned_prompts(self):
        """Build 4 prompts with block-aligned lengths for EOS path testing."""
        device = torch.cuda.current_device()
        seg_0 = torch.arange(8000, 8256, dtype=torch.int64, device=device)
        seg_1 = torch.arange(8256, 8512, dtype=torch.int64, device=device)
        prompts = [
            seg_0.clone(),
            seg_0.clone(),
            torch.cat([seg_0, seg_1]),
            torch.cat([seg_0, seg_1]),
        ]
        assert [len(p) for p in prompts] == [256, 256, 512, 512]
        return prompts

    def _run_eos_pc_on(self, model, mamba_config, prompts):
        """Run block-aligned prompts with pc=on, per-step assertions.

        Scheduling with pending_block_hashes coordination:
          - step 1: A scheduled (B, C, D deferred: h0 pending)
          - step 2: B + C co-scheduled (D deferred: h1 pending from C)
          - step 3: D scheduled
        """
        engine = self._build_engine(model, mamba_config, enable_prefix_caching=True)
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        reqs = {i: self._make_request(i, p, True) for i, p in enumerate(prompts)}
        for i in range(4):
            engine._add_request(reqs[i])

        step = 0
        finished = {}
        prev_prefill = 0

        while engine.has_unfinished_requests():
            result = engine.step_modern()
            step += 1
            step_prefill = ctx.lifetime_prefill_token_count - prev_prefill
            prev_prefill = ctx.lifetime_prefill_token_count
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

            if step == 1:
                assert reqs[0]._mamba_num_matched_blocks == 0, f"step 1"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1
                assert (
                    reqs[0].precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
                )
                assert step_prefill == 256
            elif step == 2:
                # B: 1 mamba match but raw_skip >= chunk_length, back off to 0 blocks, full recompute (256)
                # C: 1 mamba match, skip 256, effective 256
                assert reqs[1]._mamba_num_matched_blocks == 1, f"step 2 B"
                assert reqs[2]._mamba_num_matched_blocks == 1, f"step 2 C"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 2
                assert (
                    reqs[2].precomputed_block_hashes[1] in ctx.mamba_slot_allocator.hash_to_block_id
                )
                assert step_prefill == 512  # B=256 (back-off recompute) + C=256
            elif step == 3:
                # D: 2 mamba matches, raw_skip >= chunk_length, back off to block 0, skip 256, effective 256
                assert reqs[3]._mamba_num_matched_blocks == 2, f"step 3 D"
                assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 2
                assert step_prefill == 256

        return finished, ctx.lifetime_prefill_token_count

    @torch.inference_mode()
    def test_mamba_block_aligned_eos_e2e(self):
        """Verify block-aligned EOS caching and recompute-based back-off."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_block_aligned_prompts()

        off_outputs, off_prefill = self._run_simple(model, mamba_config, prompts, False)
        on_outputs, on_prefill = self._run_eos_pc_on(model, mamba_config, prompts)

        for req_id in range(4):
            assert (
                off_outputs[req_id] == on_outputs[req_id]
            ), f"req {req_id}: pc=off {off_outputs[req_id]} != pc=on {on_outputs[req_id]}"
        assert off_prefill == 1536 and on_prefill == 1024 and on_prefill < off_prefill

    def _create_eviction_prompts(self):
        device = torch.cuda.current_device()
        return [
            torch.arange(8000, 8300, dtype=torch.int64, device=device),
            torch.arange(8300, 8600, dtype=torch.int64, device=device),
            torch.arange(8000, 8300, dtype=torch.int64, device=device),  # identical to E
        ]

    @torch.inference_mode()
    def test_mamba_lru_eviction_e2e(self):
        """Verify KV eviction invalidates mamba state via invalidate_mamba_state_for_block."""
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        prompts = self._create_eviction_prompts()

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=0.002,
            prefix_caching_mamba_gb=0.05,
            request_rounder=1,
        )
        alloc = engine.context.kv_block_allocator
        ctx = engine.context

        assert alloc.total_count == 3, f"expected 3 total blocks, got {alloc.total_count}"
        assert ctx.max_requests >= 1

        finished = {}

        def _run_one(req_id, prompt):
            # Use num_tokens_to_generate=2 so the request survives the prefill
            # step (commit_mamba_intermediate_states runs after update_requests)
            req = self._make_request(req_id, prompt, True, num_tokens=2)
            engine._add_request(req)
            while engine.has_unfinished_requests():
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    finished[merged.request_id] = list(merged.generated_tokens)
            return req

        # E: seed request
        req_E = _run_one(0, prompts[0])
        h_E0 = req_E.precomputed_block_hashes[0]
        assert (
            h_E0 in ctx.mamba_slot_allocator.hash_to_block_id and h_E0 in alloc.kv_hash_to_block_id
        )
        # One block holds E's cached prefix and one is unused. total_avail counts
        # the cached block too, since LRU can reclaim it.
        assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1
        assert alloc.free_count == 1 and alloc.total_avail == 2

        # F: disjoint prefix, forces eviction of E's cached block
        req_F = _run_one(1, prompts[1])
        assert req_F.precomputed_block_hashes[0] in ctx.mamba_slot_allocator.hash_to_block_id
        assert (
            h_E0 not in alloc.kv_hash_to_block_id
            and h_E0 not in ctx.mamba_slot_allocator.hash_to_block_id
        )
        assert len(ctx.mamba_slot_allocator.hash_to_block_id) == 1

        # G: identical to E, but E's state was evicted
        req_G = _run_one(2, prompts[2])
        assert req_G._mamba_num_matched_blocks == 0
        assert h_E0 in ctx.mamba_slot_allocator.hash_to_block_id
        assert finished[0] == finished[2]

    @torch.inference_mode()
    def test_mamba_chunked_prefill_unaligned_boundary_snapshot(self):
        """Chunked prefill snapshots Mamba state at the last block boundary.

        ``compute_and_store_offsets`` records a Mamba state snapshot at a KV-block
        boundary only when that boundary is a whole multiple of the SSM chunk size
        measured from the start of the current prefill chunk. Because the chunk
        start equals ``finished_chunk_token_count`` on continuation chunks, this
        holds exactly when every chunk boundary is block-aligned.

        Here ``max_tokens`` (300) is intentionally not a multiple of the block size
        (256), so the request spans several chunks and its last full-block boundary
        (token 768) lands in a continuation chunk. The scheduler keeps each chunk
        boundary block-aligned, so the final chunk begins at token 512 and the
        token-768 snapshot is extracted and committed. A second request sharing the
        768-token prefix then restores that state and skips those blocks.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        device = torch.cuda.current_device()
        # 800-token prompt -> 3 full blocks (256/512/768) + a 32-token tail.
        # The last full-block boundary (768) falls in the final continuation chunk.
        prompt = torch.arange(9000, 9800, dtype=torch.int64, device=device)
        assert len(prompt) == 800

        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_tokens=300,  # not a multiple of BLOCK_SIZE (256) -> forces unaligned cuts
            max_requests=4,
            request_rounder=4,
        )
        ctx = engine.context
        # Sanity: the prompt genuinely spans multiple prefill chunks.
        assert ctx.max_tokens < len(prompt)

        # --- Seed request: fills the cache, no prior matches. ---
        seed = self._make_request(0, prompt, enable_pc=True, num_tokens=4)
        engine._add_request(seed)
        while engine.has_unfinished_requests():
            engine.step_modern()
        # Seed has no prior cache, so no Mamba blocks are matched during its prefill.
        assert seed._mamba_num_matched_blocks == 0

        # block index 2 == the boundary at token 768 (768 // 256 - 1). The final
        # chunk begins block-aligned at token 512, so (768 - 512) % 128 == 0 and
        # the state at this boundary is extracted and committed.
        assert len(seed.precomputed_block_hashes) == 3
        last_block_hash = seed.precomputed_block_hashes[2]
        assert (
            last_block_hash in ctx.mamba_slot_allocator.hash_to_block_id
        ), "Mamba snapshot at the last block boundary (token 768) was not recorded."

        # --- Reuse request: shares the full 768-token prefix, should restore the
        # cached Mamba state and skip those blocks entirely. ---
        reuse_prompt = torch.cat(
            [prompt[:768], torch.arange(9800, 9900, dtype=torch.int64, device=device)]
        )
        reuse = self._make_request(1, reuse_prompt, enable_pc=True, num_tokens=4)
        engine._add_request(reuse)
        while engine.has_unfinished_requests():
            engine.step_modern()

        assert reuse._mamba_num_matched_blocks == 3, (
            "Reuse request should restore Mamba state from the token-768 snapshot "
            f"(3 matched blocks), got {reuse._mamba_num_matched_blocks}."
        )


EVICTION_TOKENS_TO_GENERATE = 8
# Segment indices carve the vocabulary into disjoint block-sized token ranges:
# 0-2 are the shared prefix blocks, and the rest are per-prompt tails.
SEG_SEED_TAIL = 3  # 3 seed prompts -> indices 3, 4, 5
SEG_PROBE_TAIL = 6  # probe prompts -> indices 6, 7
SEG_STRESS_POOL = 31  # stress prompts draw whole blocks from indices 0..30
SEG_STRESS_TAIL = 31  # partial tail shared by every stress prompt


@pytest.mark.internal
@pytest.mark.skipif(not is_fa_min_version("2.7.3"), reason="need flash attn")
class TestHybridPrefixCachingEvictionEquivalence(_HybridPCHelpers):
    """Generated tokens must not depend on the state of the prefix cache.

    Every case runs one prompt sequence twice against the same hybrid model --
    once with prefix caching disabled, once enabled -- and requires identical
    output tokens. Requests are issued one at a time and drained to completion,
    so batch composition is identical between the two runs and any divergence is
    attributable to the cache path alone.

    Between requests in the caching run, the cache is driven into a specific
    state through the LRU eviction entry points, so each case covers a different
    combination of surviving KV blocks and surviving Mamba snapshots. Each case
    runs both as a single prefill chunk per request and under chunked prefill,
    where a prompt's blocks are matched across several chunks.
    """

    # ------------------------------------------------------------------ prompts

    _buffer_fit: dict = {}

    def _calibrate_buffer_gb(
        self, model, mamba_config, num_blocks, max_requests=None, request_rounder=4
    ):
        """Return the buffer size that yields ``num_blocks`` KV blocks.

        On a hybrid model the buffer covers the live Mamba state for
        ``max_requests`` concurrent requests before anything is left for KV
        blocks, so block count is affine in buffer size, not proportional. Bytes
        per block and that fixed reservation both depend on the model shape, so
        fit the line from two probe engines and solve it rather than assuming
        either the slope or the offset. A pool size written as a GB figure is a
        guess that drifts silently when the model or the request count changes.
        """
        cls = type(self)
        key = (max_requests, request_rounder)
        if key not in cls._buffer_fit:
            samples = []
            for probe_gb in (0.02, 0.05):
                probe = self._build_engine(
                    model,
                    mamba_config,
                    enable_prefix_caching=True,
                    buffer_size_gb=probe_gb,
                    max_requests=max_requests,
                    request_rounder=request_rounder,
                )
                samples.append((probe_gb, probe.context.kv_block_allocator.total_count))
            (gb_lo, blocks_lo), (gb_hi, blocks_hi) = samples
            assert blocks_hi > blocks_lo, "probe engines did not scale with buffer size"
            gb_per_block = (gb_hi - gb_lo) / (blocks_hi - blocks_lo)
            reserved_gb = gb_lo - blocks_lo * gb_per_block
            cls._buffer_fit[key] = (gb_per_block, reserved_gb)
        gb_per_block, reserved_gb = cls._buffer_fit[key]
        return reserved_gb + num_blocks * gb_per_block

    def _pause_prompts(self, chunked, count):
        """Block-aligned prompts sized so the chunked variant genuinely chunks.

        A prompt shorter than the per-step token budget is never split, and a
        one-block prompt sits under the 300-token budget the chunked runs use, so
        it would take the same single-chunk path while merely lowering how many
        requests are in flight. Two blocks against that budget cut into two
        block-aligned chunks instead.
        """
        blocks = 2 if chunked else 1
        return [
            torch.cat([self._seg(i * blocks + b) for b in range(blocks)]) for i in range(count)
        ]

    @staticmethod
    def _seg(index, length=BLOCK_SIZE):
        """A block-sized (or shorter) run of token ids unique to ``index``."""
        start = index * BLOCK_SIZE
        assert start + length <= VOCAB_SIZE, (
            f"segment {index} (+{length} tokens) runs past the vocabulary; "
            f"prompt tokens must be valid embedding indices"
        )
        return torch.arange(
            start, start + length, dtype=torch.int64, device=torch.cuda.current_device()
        )

    def _seed_prompts(self):
        """Three prompts that populate Mamba snapshots at block boundaries 0, 1 and 2.

        A prompt caches Mamba state at its last full-block boundary, so prompt
        lengths are chosen to place that boundary one block further along each
        time. They share a common prefix, so each also matches the blocks the
        previous ones registered.
        """
        seg0, seg1, seg2 = self._seg(0), self._seg(1), self._seg(2)
        return [
            torch.cat([seg0, self._seg(SEG_SEED_TAIL, 44)]),  # 300 tokens -> boundary at block 0
            torch.cat([seg0, seg1, self._seg(SEG_SEED_TAIL + 1, 44)]),  # 556 tokens -> boundary at block 1
            torch.cat(
                [seg0, seg1, seg2, self._seg(SEG_SEED_TAIL + 2, 44)]
            ),  # 812 tokens -> boundary at block 2
        ]

    def _probe(self, num_shared_blocks, tail_index, tail_len=44):
        """A prompt sharing ``num_shared_blocks`` leading blocks with the seeds."""
        shared = [self._seg(i) for i in range(num_shared_blocks)]
        return torch.cat(shared + [self._seg(tail_index, tail_len)])

    # ----------------------------------------------------------------- eviction

    @staticmethod
    def _kv_map(engine):
        return engine.context.kv_block_allocator.kv_hash_to_block_id

    @staticmethod
    def _mamba_map(engine):
        return engine.context.mamba_slot_allocator.hash_to_block_id

    @classmethod
    def _evict_kv(cls, count, log=None, optional=False):
        """Evict the ``count`` least-recently-used cached KV blocks.

        Verifies the blocks really left the cache, and that any Mamba snapshot
        that went with them belonged to one of the evicted blocks. Hashes of the
        evicted blocks are appended to ``log``. With ``optional``, tolerate there
        being too few evictable blocks.
        """

        def action(engine, requests):
            alloc = engine.context.kv_block_allocator
            msa = engine.context.mamba_slot_allocator
            before_kv = set(alloc.kv_hash_to_block_id)
            before_mamba = set(msa.hash_to_block_id)

            evicted = alloc.evict_lru_blocks(count)

            removed_kv = before_kv - set(alloc.kv_hash_to_block_id)
            removed_mamba = before_mamba - set(msa.hash_to_block_id)
            if not optional:
                assert evicted, f"expected at least {count} evictable KV blocks"
                assert len(removed_kv) == count, (
                    f"expected {count} blocks to leave the KV cache, {len(removed_kv)} did"
                )
            # a Mamba snapshot may only disappear alongside its own KV block
            assert removed_mamba <= removed_kv, (
                "Mamba snapshots were dropped for blocks that are still cached"
            )
            if log is not None:
                log.extend(removed_kv)

        return action

    @classmethod
    def _evict_mamba(cls, count, log=None, optional=False):
        """Evict the ``count`` least-recently-used Mamba snapshots, keeping their KV blocks.

        Verifies the snapshots really left the Mamba cache and that their KV
        blocks stayed. Hashes of the evicted snapshots are appended to ``log``.
        With ``optional``, evict whatever is cached (possibly nothing).
        """

        def action(engine, requests):
            msa = engine.context.mamba_slot_allocator
            kv = engine.context.kv_block_allocator.kv_hash_to_block_id
            available = len(msa.hash_to_block_id)
            if optional:
                count_now = min(count, available)
                if count_now == 0:
                    return
            else:
                assert (
                    available >= count
                ), f"expected at least {count} cached Mamba snapshots, have {available}"
                count_now = count

            before = set(msa.hash_to_block_id)
            slots = msa._evict_lru_slots_batch(count_now)
            # _evict_lru_slots_batch hands ownership of the freed slots to its
            # caller; return them to the free pool the way allocate_slots_batch
            # would after taking them.
            for slot in slots:
                msa.free_slots[msa.free_count] = slot
                msa.free_count += 1

            removed = before - set(msa.hash_to_block_id)
            assert len(removed) == count_now, (
                f"expected {count_now} snapshots to leave the Mamba cache, {len(removed)} did"
            )
            for block_hash in removed:
                assert block_hash in kv, "a Mamba-only eviction must leave the KV block cached"
            if log is not None:
                log.extend(removed)

        return action

    @staticmethod
    def _evict_mamba_for_hash(get_hash):
        """Drop the Mamba snapshot for one specific block, keeping its KV block cached.

        Mirrors the bookkeeping of the LRU slot eviction path, but targets a
        chosen block so a case can place a hole at a known depth of the prefix
        chain. ``get_hash`` is called with the finished requests so far and
        returns the block hash to drop.
        """

        def action(engine, requests):
            msa = engine.context.mamba_slot_allocator
            block_hash = get_hash(requests)
            assert block_hash in msa.hash_to_block_id, "target block has no cached Mamba snapshot"
            block_id = msa.hash_to_block_id.pop(block_hash)
            slot = msa.block_to_slot[block_id].item()
            assert slot >= 0
            msa.block_to_slot[block_id] = -1
            msa.slot_to_block[slot] = -1
            msa.free_slots[msa.free_count] = slot
            msa.free_count += 1
            # the KV block itself stays cached and matchable
            assert block_hash in engine.context.kv_block_allocator.kv_hash_to_block_id

        return action

    @staticmethod
    def _chain(*actions):
        def action(engine, requests):
            for a in actions:
                a(engine, requests)

        return action

    # -------------------------------------------------------------------- runner

    def _engine_kwargs(self, chunked):
        if not chunked:
            return {}
        # max_tokens below the prompt lengths forces each request across several
        # prefill chunks. 300 is not a multiple of BLOCK_SIZE, so the scheduler
        # has to choose the cuts rather than inheriting them from the prompt.
        return dict(
            enable_chunked_prefill=True, max_tokens=300, max_requests=4, request_rounder=4
        )

    def _run(
        self,
        model,
        mamba_config,
        prompts,
        enable_pc,
        actions=None,
        on_engine=None,
        generate=EVICTION_TOKENS_TO_GENERATE,
        admit=1,
        stats=None,
        **engine_kwargs,
    ):
        """Issue the prompts in groups of ``admit``, draining each group before the next.

        With the default ``admit=1`` every request runs alone, so batch
        composition is identical between runs. A larger group admits requests
        together, which is what lets them compete for blocks and be paused.

        ``actions`` maps a prompt index to a callable invoked with the engine
        once that prompt's group has finished. Actions run only when prefix
        caching is on, since they manipulate cache state that does not otherwise
        exist. ``stats`` collects observations about what the engine had to do.
        """
        engine = self._build_engine(
            model, mamba_config, enable_prefix_caching=enable_pc, **engine_kwargs
        )
        if on_engine is not None:
            on_engine(engine)
        if stats is not None:
            self._instrument_resume(engine, stats)
        outputs, requests = {}, []
        for start in range(0, len(prompts), admit):
            group = range(start, min(start + admit, len(prompts)))
            for i in group:
                req = self._make_request(i, prompts[i], enable_pc, num_tokens=generate)
                engine._add_request(req)
                requests.append(req)
            steps = 0
            while engine.has_unfinished_requests():
                steps += 1
                assert steps < 500, (
                    f"engine stopped making progress with {len(outputs)} of "
                    f"{len(prompts)} requests finished"
                )
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    outputs[merged.request_id] = list(merged.generated_tokens)
            if enable_pc and actions:
                for i in group:
                    if i in actions:
                        actions[i](engine, requests)
        return outputs, requests, engine

    def _assert_equivalent(
        self,
        prompts,
        actions=None,
        chunked=False,
        on_engine=None,
        generate=EVICTION_TOKENS_TO_GENERATE,
        admit=1,
        reference_admit=1,
        stats=None,
        buffer_blocks=None,
        **engine_overrides,
    ):
        """Require the caching run to reproduce an uncached single-pass reference.

        The reference deliberately runs without chunked prefill and without
        memory pressure: one forward over each whole prompt. Chunked prefill cuts
        a prompt at different points than a single pass, and the SSM scan is not
        bitwise invariant to where it is cut, so an uncached *chunked* run is not
        a stable baseline -- it diverges from the single-pass answer on its own,
        with caching out of the picture entirely. Comparing against the
        single-pass answer is both stabler and a stronger claim: whatever the
        cache skipped, restored or evicted, and however the prompt was chunked,
        the tokens must match a plain uncached forward.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        if buffer_blocks is not None:
            engine_overrides["buffer_size_gb"] = self._calibrate_buffer_gb(
                model,
                mamba_config,
                buffer_blocks,
                max_requests=engine_overrides.get("max_requests"),
                request_rounder=engine_overrides.get("request_rounder", 4),
            )

        reference, _, _ = self._run(
            model, mamba_config, prompts, False, generate=generate, admit=reference_admit
        )
        cached, requests, engine = self._run(
            model,
            mamba_config,
            prompts,
            True,
            actions=actions,
            on_engine=on_engine,
            generate=generate,
            admit=admit,
            stats=stats,
            **{**self._engine_kwargs(chunked), **engine_overrides},
        )

        expected_ids = list(range(len(prompts)))
        assert sorted(reference) == expected_ids, (
            f"the reference run finished {sorted(reference)}, expected {expected_ids}"
        )
        assert sorted(cached) == expected_ids, (
            f"the caching run finished {sorted(cached)}, expected {expected_ids}. "
            f"Requests that never fit are dropped by the scheduler rather than "
            f"raising, so an empty list means the pool is too small to run them."
        )
        for req_id in reference:
            assert reference[req_id] == cached[req_id], (
                f"request {req_id}: uncached reference produced {reference[req_id]}, "
                f"caching enabled produced {cached[req_id]}"
            )
        return requests, engine

    # ---------------------------------------------------------------- the cases

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_single_block_match(self, chunked):
        """A request sharing one leading block with a cached prompt, nothing evicted."""
        prompts = self._seed_prompts() + [self._probe(1, tail_index=SEG_PROBE_TAIL)]
        requests, engine = self._assert_equivalent(prompts, chunked=chunked)

        # nothing was evicted: every block the seeds registered is still cached,
        # in both maps
        for block_hash in requests[2].precomputed_block_hashes:
            assert block_hash in self._kv_map(engine)
            assert block_hash in self._mamba_map(engine)
        if not chunked:
            assert requests[-1]._mamba_num_matched_blocks == 1

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_kv_block_evicted(self, chunked):
        """A cached KV block is evicted before a request that would have matched it.

        Evicting a KV block also drops the Mamba snapshot anchored to it, while
        snapshots on the blocks that survive are left in place.
        """
        prompts = self._seed_prompts() + [self._probe(3, tail_index=SEG_PROBE_TAIL)]
        evicted = []

        def check_cascade(engine, requests):
            seed_hashes = requests[2].precomputed_block_hashes
            # blocks 0-2 form a chain, so the only leaf is the deepest one
            assert evicted == [seed_hashes[2]], f"expected block 2 to be evicted, got {evicted}"
            assert seed_hashes[2] not in self._kv_map(engine)
            assert seed_hashes[2] not in self._mamba_map(engine), "snapshot outlived its KV block"
            # its ancestors are untouched in both caches
            for block_hash in seed_hashes[:2]:
                assert block_hash in self._kv_map(engine)
                assert block_hash in self._mamba_map(engine)

        actions = {2: self._chain(self._evict_kv(1, log=evicted), check_cascade)}
        requests, engine = self._assert_equivalent(prompts, actions=actions, chunked=chunked)

        if not chunked:
            # the probe matched the two surviving blocks rather than all three
            assert requests[-1]._mamba_num_matched_blocks == 2

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_mamba_snapshot_evicted(self, chunked):
        """A Mamba snapshot is evicted while its KV block stays cached.

        The request that follows can still match the KV block, but has to resume
        the SSM from a shallower boundary -- or recompute from the start if no
        snapshot survives below its match.
        """
        prompts = self._seed_prompts() + [self._probe(3, tail_index=SEG_PROBE_TAIL)]
        evicted = []

        def check_kv_survives(engine, requests):
            assert len(evicted) == 1, f"expected one snapshot evicted, got {evicted}"
            assert evicted[0] not in self._mamba_map(engine)
            assert evicted[0] in self._kv_map(engine), "the KV block should have stayed cached"
            # every block the seeds registered is still matchable
            for block_hash in requests[2].precomputed_block_hashes:
                assert block_hash in self._kv_map(engine)

        actions = {2: self._chain(self._evict_mamba(1, log=evicted), check_kv_survives)}
        self._assert_equivalent(prompts, actions=actions, chunked=chunked)

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_kv_block_and_mamba_snapshot_evicted(self, chunked):
        """Both a KV block and an unrelated Mamba snapshot are evicted."""
        prompts = self._seed_prompts() + [self._probe(3, tail_index=SEG_PROBE_TAIL)]
        kv_evicted = []

        def check_both(engine, requests):
            seed_hashes = requests[2].precomputed_block_hashes
            # block 2 is the chain's only leaf, so the block eviction took it,
            # and its snapshot went with it
            assert kv_evicted == [seed_hashes[2]], f"expected block 2 evicted, got {kv_evicted}"
            assert seed_hashes[2] not in self._kv_map(engine)
            assert seed_hashes[2] not in self._mamba_map(engine)
            # block 1 lost only its snapshot; the block itself is still cached
            assert seed_hashes[1] in self._kv_map(engine)
            assert seed_hashes[1] not in self._mamba_map(engine)
            # block 0 is untouched in both caches
            assert seed_hashes[0] in self._kv_map(engine)
            assert seed_hashes[0] in self._mamba_map(engine)

        actions = {
            2: self._chain(
                # Target block 1 rather than taking the LRU snapshot: the block
                # eviction below removes the leaf (block 2), and evicting that
                # same block's snapshot first would collapse the two cases into
                # one instead of leaving a block that is cached without state.
                self._evict_mamba_for_hash(
                    lambda requests: requests[2].precomputed_block_hashes[1]
                ),
                self._evict_kv(1, log=kv_evicted),
                check_both,
            )
        }
        requests, engine = self._assert_equivalent(prompts, actions=actions, chunked=chunked)

        if not chunked:
            # only blocks 0-1 survive in KV, and only block 0 has state, so the
            # probe resumes from the first boundary
            assert requests[-1]._mamba_num_matched_blocks == 1

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_kv_block_evicted_then_recomputed(self, chunked):
        """An evicted KV block is recomputed by one request and matched by the next."""
        prompts = self._seed_prompts() + [
            self._probe(3, tail_index=SEG_PROBE_TAIL),  # recomputes what was evicted
            self._probe(3, tail_index=SEG_PROBE_TAIL + 1),  # matches the recomputed blocks
        ]
        evicted = []

        def check_gone(engine, requests):
            assert len(evicted) == 1, f"expected one block evicted, got {evicted}"
            assert evicted[0] not in self._kv_map(engine)

        actions = {2: self._chain(self._evict_kv(1, log=evicted), check_gone)}
        requests, engine = self._assert_equivalent(prompts, actions=actions, chunked=chunked)

        # the evicted block was recomputed and is cached again, along with the
        # rest of the prefix, by the time the last request runs
        assert evicted[0] in self._kv_map(engine), "the evicted block was never recomputed"
        for block_hash in requests[-1].precomputed_block_hashes:
            assert block_hash in self._kv_map(engine)

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_mamba_snapshot_evicted_then_recomputed(self, chunked):
        """An evicted Mamba snapshot is re-extracted by one request and used by the next."""
        prompts = self._seed_prompts() + [
            self._probe(3, tail_index=SEG_PROBE_TAIL),  # re-extracts a snapshot on its divergence boundary
            self._probe(3, tail_index=SEG_PROBE_TAIL + 1),  # resumes from the re-extracted snapshot
        ]
        evicted = []

        def check_gone(engine, requests):
            assert len(evicted) == 1, f"expected one snapshot evicted, got {evicted}"
            assert evicted[0] not in self._mamba_map(engine)
            assert evicted[0] in self._kv_map(engine)

        actions = {2: self._chain(self._evict_mamba(1, log=evicted), check_gone)}
        requests, engine = self._assert_equivalent(prompts, actions=actions, chunked=chunked)

        # the snapshot was re-extracted and is available again
        assert evicted[0] in self._mamba_map(engine), "the snapshot was never re-extracted"
        if not chunked:
            assert requests[-1]._mamba_num_matched_blocks > 0

    @torch.inference_mode()
    def test_snapshot_hole_under_tight_prefill_chunk(self):
        """A prompt one token past its last shared block, with a snapshot hole beneath it.

        The deepest cached snapshot covers every token but one of this prompt, so
        the skip has to give back a block to leave something to compute. The
        boundary it gives back to has had its snapshot evicted, so the resume
        point is a block shallower still.
        """
        seeds = self._seed_prompts()
        # 3 shared blocks + 1 token: skipping to the block-2 boundary would leave
        # a single token for this chunk, so the skip has to give that block back.
        probe = self._probe(3, tail_index=SEG_PROBE_TAIL, tail_len=1)
        assert len(probe) == 3 * BLOCK_SIZE + 1

        # Drop the snapshot at block index 1 -- the boundary the skip falls back
        # to -- so the usable resume point is block 0, two boundaries below the
        # deepest snapshot the prefix matches.
        actions = {
            2: self._evict_mamba_for_hash(lambda requests: requests[2].precomputed_block_hashes[1])
        }
        requests, engine = self._assert_equivalent(seeds + [probe], actions=actions)

        # the prefix still matches all three blocks; only the resume point moved
        assert requests[-1]._mamba_num_matched_blocks == 3
        assert requests[-1].precomputed_block_hashes[1] not in self._mamba_map(engine)

    @torch.inference_mode()
    def test_eviction_stress(self):
        """Many overlapping prompts through a cache far too small to hold them.

        Blocks are evicted and recomputed continuously while requests are also
        losing Mamba snapshots, so the run covers a long sequence of cache states
        rather than one constructed one.

        Each request generates a single token, which is the one the prefill state
        produces directly and so is the token that a mishandled skip or restore
        corrupts. Later tokens are not compared here: with chunked prefill on, a
        prompt that gets no cache hit is still split into chunks the uncached
        single-pass reference does not use, and the SSM scan is not bitwise
        invariant to where it is split. That difference is present with caching
        disabled entirely, so comparing decode steps across the two would be
        measuring chunking, not the cache. The structured cases above keep the
        full-sequence comparison, where caching makes the two runs do the same
        work and the tokens match exactly.
        """
        # Each prompt is 1-4 whole blocks drawn from a pool of segments, then a
        # partial tail every prompt shares. Only complete blocks are registered,
        # so the whole blocks are what fill the cache; the tail contributes
        # nothing to evict. Most prompts start from an earlier prompt's prefix so
        # matches, evictions and recomputes all land at varying depths. Because
        # block hashes are parent-chained, the same segment after a different
        # prefix is a different cache entry, so a modest pool of segments still
        # produces far more distinct blocks than the pool of KV blocks holds.
        rng = random.Random(20260728)
        num_prompts = 64
        segment_pool = SEG_STRESS_POOL
        block_lists, prompts = [], []
        for i in range(num_prompts):
            depth = 1 + (i % 4)
            if block_lists and rng.random() < 0.6:
                base = block_lists[rng.randrange(len(block_lists))]
                blocks = base[: rng.randint(1, len(base))]
            else:
                blocks = []
            while len(blocks) < depth:
                blocks.append(rng.randrange(segment_pool))
            blocks = blocks[:depth]
            block_lists.append(blocks)
            prompts.append(
                torch.cat(
                    [self._seg(b) for b in blocks] + [self._seg(SEG_STRESS_TAIL, 44)]
                )
            )

        evicted_blocks, mamba_cascades = [], []

        def instrument(engine):
            alloc = engine.context.kv_block_allocator
            msa = engine.context.mamba_slot_allocator
            deregister = alloc._deregister_blocks
            cascade = alloc.on_blocks_deregistered

            def counting_deregister(block_ids):
                evicted_blocks.append(int(block_ids.numel()))
                return deregister(block_ids)

            def counting_cascade(block_ids_list, hashes_to_delete):
                # count the evictions that actually took Mamba state with them
                if hashes_to_delete & msa.hash_to_block_id.keys():
                    mamba_cascades.append(len(hashes_to_delete))
                return cascade(block_ids_list, hashes_to_delete)

            alloc._deregister_blocks = counting_deregister
            alloc.on_blocks_deregistered = counting_cascade

        # Drop a Mamba snapshot every third request and force an extra block
        # eviction every fifth, on top of the churn the small pool causes on its
        # own, so slot eviction and block eviction interleave throughout.
        mamba_evicted, kv_evicted = [], []
        actions = {}
        for i in range(num_prompts):
            if i % 3 == 2:
                actions[i] = self._evict_mamba(1, log=mamba_evicted, optional=True)
            elif i % 5 == 4:
                actions[i] = self._evict_kv(1, log=kv_evicted, optional=True)

        _, engine = self._assert_equivalent(
            prompts,
            actions=actions,
            chunked=True,
            on_engine=instrument,
            generate=1,
            # roughly 15 blocks, far fewer than these prompts register, so the
            # pool turns over many times across the run
            buffer_size_gb=0.01,
        )

        # KV blocks were evicted continuously, not just once or twice
        assert sum(evicted_blocks) >= num_prompts, (
            f"expected sustained KV eviction across {num_prompts} requests, saw "
            f"{sum(evicted_blocks)} blocks evicted. Only complete blocks are "
            f"registered, so raise the prompt count or the blocks per prompt "
            f"rather than the tail length."
        )
        # Mamba state left the cache both ways: dropped on its own with the KV
        # block still resident, and carried out by a KV block eviction.
        assert len(mamba_evicted) >= num_prompts // 8, (
            f"expected repeated Mamba-only eviction, saw {len(mamba_evicted)}"
        )
        assert len(mamba_cascades) >= num_prompts // 8, (
            f"expected KV eviction to carry Mamba state with it repeatedly, "
            f"saw {len(mamba_cascades)}"
        )
        # and the forced block evictions found something to take each time
        assert kv_evicted, "the periodic KV eviction never had a block to evict"

    @staticmethod
    def _instrument_resume(engine, stats):
        """Record what the resume path saw each time it ran.

        A request that fills its last block is pause-marked unconditionally and,
        when capacity allows, resumed inside the same ``update_requests`` call.
        A paused count sampled after a step therefore usually reads zero even
        though requests were paused and resumed during it. Wrapping the resume
        entry point observes that transient state instead.
        """
        ctx = engine.context
        alloc = ctx.kv_block_allocator
        original = ctx.resume_paused_requests

        def wrapped(active_request_count, newly_paused_request_ids):
            paused = ctx.paused_request_count
            # update_requests pause-marks boundary-crossing requests before
            # calling this, so the rise over what was left paused last time is
            # how many requests just crossed a block boundary.
            stats["pause_marks"] = stats.get("pause_marks", 0) + max(
                0, paused - stats.get("_left_after_resume", 0)
            )
            if paused > 0:
                stats["resume_with_paused"] = stats.get("resume_with_paused", 0) + 1
                stats["max_paused"] = max(stats.get("max_paused", 0), paused)
                # The free pool is empty, so any block a request resumes into has
                # to come from evicting a cached one.
                if alloc.free_count == 0 and int(alloc.get_evictable_block_count()) > 0:
                    stats["resume_needing_eviction"] = stats.get("resume_needing_eviction", 0) + 1
            result = original(active_request_count, newly_paused_request_ids)
            stats["_left_after_resume"] = ctx.paused_request_count
            if ctx.paused_request_count > 0:
                # resumption could not take every paused request this step: one
                # of them wanted a block no amount of reclaiming could supply
                stats["resume_left_paused"] = stats.get("resume_left_paused", 0) + 1
                stats["max_left_paused"] = max(
                    stats.get("max_left_paused", 0), ctx.paused_request_count
                )
            return result

        ctx.resume_paused_requests = wrapped

    def _assert_paused(self, stats):
        """Fail unless requests were actually paused and resumed."""
        assert stats.get("resume_with_paused", 0) > 0, (
            "the resume path never ran with a paused request. Prompts must fill "
            "their last block, and requests must outlive the step that fills it: "
            "finished requests are released before pause-marking runs, so a "
            "request generating a single token is gone before it can be paused."
        )

    def _assert_paused_under_scarcity(self, stats):
        """Fail unless resumption ran out of blocks and had to leave requests waiting.

        Distinct from :meth:`_assert_paused_under_eviction`: that one wants
        capacity to exist but only behind an eviction, which requires registered
        cached blocks to reclaim. A workload whose prompts never fill a block
        registers nothing, so its contention shows up as requests that simply
        cannot be resumed yet.
        """
        self._assert_paused(stats)
        assert stats.get("resume_left_paused", 0) > 0, (
            "every paused request was resumed immediately, so the pool never "
            "actually ran out of blocks. Lower the pool size or raise the "
            "request count."
        )

    def _assert_paused_under_eviction(self, stats):
        """Fail unless resumption had to reclaim a cached block to proceed."""
        self._assert_paused(stats)
        assert stats.get("resume_needing_eviction", 0) > 0, (
            "requests resumed, but the free pool always had a spare block, so "
            "resumption never depended on reclaiming a cached one. Lower "
            "buffer_size_gb or raise the request count."
        )

    @torch.inference_mode()
    def test_decode_crossing_many_block_boundaries(self):
        """Generating past a block's worth pauses a request repeatedly, without changing its output.

        Every other pause case here crosses exactly one block boundary, the one
        immediately after prefill, because the prompts are block-aligned. That is
        a single special case. Here the prompts are deliberately *not*
        block-aligned and each request generates well over a block's worth, so
        the crossings land in the middle of decode and recur, and each request is
        paused and resumed several times while others draw on the same pool.

        Correctness is checked by running the same batch twice against pools of
        different sizes. A roomy pool resumes every request immediately; a tight
        one makes them wait on blocks that only eviction can release. Prompts,
        batching and generation length are identical, so pausing under contention
        is the only difference between the runs and any divergence in the output
        is attributable to it. That is a tighter control than the uncached
        single-pass reference used elsewhere, which would also differ in how the
        prefill itself was computed.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        num_requests, prompt_len, generate = 6, 200, 400
        assert prompt_len % BLOCK_SIZE != 0, "prompts must not end on a block boundary"
        prompts = [self._seg(i, prompt_len) for i in range(num_requests)]

        # Boundaries strictly inside (prompt_len, prompt_len + generate) are the
        # ones a request crosses while decoding.
        crossings = [
            b
            for b in range(BLOCK_SIZE, prompt_len + generate + 1, BLOCK_SIZE)
            if prompt_len < b <= prompt_len + generate
        ]
        assert len(crossings) >= 2, f"expected repeated mid-decode crossings, got {crossings}"
        total_blocks = -(-(prompt_len + generate) // BLOCK_SIZE)

        def run(pool_blocks, stats):
            engine = self._build_engine(
                model,
                mamba_config,
                enable_prefix_caching=True,
                buffer_size_gb=self._calibrate_buffer_gb(
                    model, mamba_config, pool_blocks, max_requests=num_requests, request_rounder=4
                ),
                max_requests=num_requests,
                request_rounder=4,
            )
            self._instrument_resume(engine, stats)
            for i, prompt in enumerate(prompts):
                engine._add_request(
                    self._make_request(i, prompt, enable_pc=True, num_tokens=generate)
                )
            outputs, steps = {}, 0
            while engine.has_unfinished_requests():
                steps += 1
                assert steps < 20000, (
                    f"engine stopped making progress after {len(outputs)} of "
                    f"{num_requests} requests finished"
                )
                result = engine.step_modern()
                for record in result["finished_request_records"]:
                    merged = record.merge()
                    outputs[merged.request_id] = list(merged.generated_tokens)
            assert sorted(outputs) == list(range(num_requests))
            for req_id, tokens in outputs.items():
                assert (
                    len(tokens) == generate
                ), f"request {req_id} produced {len(tokens)} tokens, expected {generate}"
            return outputs

        # Roomy: every request fits at once, so nothing ever waits on a block.
        roomy_stats = {}
        roomy_out = run(num_requests * total_blocks + 4, roomy_stats)
        # Tight: room for about two requests, so the rest contend.
        tight_stats = {}
        tight_out = run(2 * total_blocks + 1, tight_stats)

        # The two runs differ in the intended way and only in that way. These
        # prompts are shorter than a block, so no complete block is ever
        # registered and there is nothing cached to reclaim; contention here is
        # requests waiting on a pool with no free block at all, which is what
        # resume_left_paused records.
        assert roomy_stats.get("resume_left_paused", 0) == 0, (
            "the roomy pool was not roomy: a request had to wait for a block, so "
            "it is not a contention-free control"
        )
        self._assert_paused_under_scarcity(tight_stats)

        # Each request crosses every boundary in its span, and a crossing is what
        # pause-marks it, so neither run can have marked fewer than that. Counting
        # marks rather than the ids returned by a step matters: resumption removes
        # the requests it takes, so a step only reports the ones still waiting,
        # which is zero whenever the pool has room.
        expected = num_requests * len(crossings)
        for label, run_stats in (("roomy", roomy_stats), ("tight", tight_stats)):
            marks = run_stats.get("pause_marks", 0)
            assert marks >= expected, (
                f"{label} run: expected at least {expected} pause marks "
                f"({num_requests} requests x {len(crossings)} crossings), saw {marks}"
            )

        for req_id in roomy_out:
            assert roomy_out[req_id] == tight_out[req_id], (
                f"request {req_id} generated different tokens when it had to wait "
                f"for blocks: uncontended {roomy_out[req_id][:8]}... vs "
                f"contended {tight_out[req_id][:8]}..."
            )

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_paused_requests(self, chunked):
        """Requests paused mid-decode and resumed still produce the reference tokens.

        Every prompt is exactly one block, so each request needs a second block
        the moment it generates, and they are admitted together against a pool
        that cannot serve them all at once. Requests are therefore paused and
        resumed while the cache holds evictable blocks.

        One token per request is compared: admitting as a batch means the batch
        composition necessarily differs from the sequential reference, and that
        alone perturbs decode-step numerics. The first token is the one the
        prefill state and a correct resume determine, which is the property
        pausing puts at risk.
        """
        num_requests = 16
        prompts = self._pause_prompts(chunked, num_requests)
        assert all(len(p) % BLOCK_SIZE == 0 for p in prompts), "prompts must be block-aligned"

        # Each request holds its prompt blocks plus one more for the tokens it
        # generates, and the pool is sized so the rest have to wait on blocks
        # that only eviction can release. Chunked prefill admits at most one
        # chunked request at a time, so fewer requests are ever in flight and a
        # pool sized for two of them still leaves spares; size it for one
        # resident request plus a margin instead.
        blocks_per_request = len(prompts[0]) // BLOCK_SIZE + 1
        pool_blocks = blocks_per_request + 2 if chunked else 2 * blocks_per_request + 1

        stats = {}
        _, engine = self._assert_equivalent(
            prompts,
            chunked=chunked,
            # Three tokens: the first is produced by prefill, the request is
            # pause-marked at the block boundary during the update that follows,
            # and the rest are produced after it resumes. Comparing only the
            # first token would say nothing about the resume, since pausing
            # happens after it is sampled.
            generate=3,
            admit=num_requests,
            reference_admit=num_requests,
            stats=stats,
            buffer_blocks=pool_blocks,
            max_requests=num_requests,
            request_rounder=4,
        )
        self._assert_paused_under_eviction(stats)
        assert engine.context.kv_block_allocator.total_count <= pool_blocks + 1

    @pytest.mark.parametrize("chunked", [False, True], ids=["single_chunk", "chunked_prefill"])
    @torch.inference_mode()
    def test_paused_requests_sharing_cached_blocks(self, chunked):
        """Pausing while the blocks under contention are shared and cache-pinned.

        Prompts come in identical pairs, so the second of each pair matches the
        first's block and raises its reference count instead of allocating. A
        pinned block cannot be evicted to relieve the pressure, which changes
        which requests can resume and when, so the resume accounting is exercised
        against a pool that is part shared, part reclaimable.
        """
        pairs = self._pause_prompts(chunked, 8)
        prompts = [pairs[i // 2] for i in range(16)]
        assert all(len(p) % BLOCK_SIZE == 0 for p in prompts)
        blocks_per_request = len(prompts[0]) // BLOCK_SIZE + 1
        pool_blocks = blocks_per_request + 2 if chunked else 2 * blocks_per_request + 1

        stats = {}
        _, engine = self._assert_equivalent(
            prompts,
            chunked=chunked,
            generate=3,
            admit=len(prompts),
            reference_admit=len(prompts),
            stats=stats,
            buffer_blocks=pool_blocks,
            max_requests=len(prompts),
            request_rounder=4,
        )
        self._assert_paused_under_eviction(stats)
        # the duplicated prompts really did share a block rather than each
        # allocating their own
        assert len(self._kv_map(engine)) <= len(pairs)

    @torch.inference_mode()
    def test_concurrent_block_aligned_requests_pause_and_resume(self):
        """Requests that exhaust the pool mid-decode are paused and later resume.

        Every prompt is exactly one block long, so each request needs a *second*
        block the moment it generates its first token. Several run concurrently
        against a pool that cannot hand out that second block to all of them at
        once, so requests get paused and resumed as capacity frees up.

        This is the state prefill-time eviction never reaches. Under LRU, a
        finished request's complete blocks stay registered rather than returning
        to the free pool, so at the moment a paused request wants to resume the
        free pool can be empty while the capacity it needs sits in
        reclaimable-but-still-cached blocks. Resumption has to count that
        capacity, not just the free pool.

        The assertion is progress and accounting rather than token equality:
        requests are admitted as a batch here, so batch composition differs from
        an uncached run by construction.
        """
        skip_if_mamba_sequence_packing_not_available()
        model = self._create_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)

        num_requests, generate = 16, 4
        prompts = [self._seg(i) for i in range(num_requests)]
        assert all(len(p) == BLOCK_SIZE for p in prompts), "prompts must be exactly one block"

        # Two blocks per request (prompt plus one for what it generates); a pool
        # of roughly two requests' worth forces the rest to resume into blocks
        # that only eviction can free.
        pool_blocks = 5
        engine = self._build_engine(
            model,
            mamba_config,
            enable_prefix_caching=True,
            buffer_size_gb=self._calibrate_buffer_gb(
                model, mamba_config, pool_blocks, max_requests=num_requests, request_rounder=4
            ),
            max_requests=num_requests,
            request_rounder=4,
        )
        stats = {}
        self._instrument_resume(engine, stats)

        for i, prompt in enumerate(prompts):
            engine._add_request(self._make_request(i, prompt, enable_pc=True, num_tokens=generate))

        finished, steps = {}, 0
        while engine.has_unfinished_requests():
            steps += 1
            assert steps < 500, (
                f"engine stopped making progress after {len(finished)} of "
                f"{num_requests} requests finished"
            )
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = list(merged.generated_tokens)

        assert sorted(finished) == list(
            range(num_requests)
        ), f"only {sorted(finished)} of {num_requests} requests finished"
        for req_id, tokens in finished.items():
            assert (
                len(tokens) == generate
            ), f"request {req_id} produced {len(tokens)} tokens, expected {generate}"
        self._assert_paused_under_eviction(stats)
