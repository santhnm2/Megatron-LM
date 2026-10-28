# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer."""

import numpy as np
import pytest
import torch

from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import (
    gated_delta_product_inference_stack_spec,
    hybrid_inference_stack_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import (
    dsa_gqa_flashinfer,
    dsa_layer_specs,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import DSGroupedSelfAttention
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import _hybrid_inference_stack_spec
from tests.unit_tests.test_utilities import Utils

HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 256
INDEX_DIM = 128
TOPK = 512
DSA_SPECS = "megatron.core.transformer.experimental_attention_variant.dsa_layer_specs"


def _config(**overrides) -> TransformerConfig:
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=1,
        kv_channels=HEAD_DIM,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        experimental_attention_variant="dsa",
        dsa_indexer_mode="simplified",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=INDEX_DIM,
        dsa_indexer_topk=TOPK,
        dsa_kernel_backend="reference",
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _flashinfer_supported(topk=TOPK):
    device = torch.device("cuda", torch.cuda.current_device())
    return (
        dsa_gqa_flashinfer.HAVE_FLASHINFER_DSA
        and dsa_gqa_flashinfer.HAVE_DEEP_SELECT
        and dsa_gqa_flashinfer.flashinfer_dsa_supported(device, topk, INDEX_DIM, 256, NUM_HEADS)
    )


requires_flashinfer_dsa = pytest.mark.skipif(
    not torch.cuda.is_available() or not _flashinfer_supported(),
    reason="FlashInfer's DSA kernels are not installed or not supported on this device",
)


@pytest.fixture(scope="module", autouse=True)
def initialize_distributed():
    Utils.initialize_distributed()


# Layer specs.


@pytest.mark.parametrize(
    "derived, upstream",
    [
        (dsa_layer_specs.dsa_inference_stack_spec, hybrid_inference_stack_spec),
        (dsa_layer_specs.gdp_dsa_inference_stack_spec, gated_delta_product_inference_stack_spec),
    ],
)
def test_inference_stack_specs_swap_only_the_attention(derived, upstream):
    attention = derived.submodules.attention_layer.submodules.self_attention
    upstream_attention = upstream.submodules.attention_layer.submodules.self_attention
    assert attention.module is DSGroupedSelfAttention
    assert upstream_attention.module is not DSGroupedSelfAttention
    assert attention.submodules.linear_qkv is upstream_attention.submodules.linear_qkv
    assert derived.submodules.mamba_layer == upstream.submodules.mamba_layer


@pytest.mark.parametrize("name", ["dsa_stack_spec", "gdp_dsa_stack_spec"])
@pytest.mark.parametrize("wide_residual", [False, True])
def test_inference_optimized_resolves_dsa_training_specs(name, wide_residual):
    """--transformer-impl inference_optimized must keep DSA rather than fall back to dense."""
    resolved = _hybrid_inference_stack_spec([DSA_SPECS, name], wide_residual=wide_residual)
    expected = name.removesuffix("_stack_spec") + "_inference_stack_spec"
    if wide_residual:
        expected = "wide_residual_" + expected
    assert resolved is getattr(dsa_layer_specs, expected)
    attention = resolved.submodules.attention_layer.submodules.self_attention
    assert attention.module is DSGroupedSelfAttention


# Step plan and cache kernels.


def test_flashinfer_step_plan():
    """Leading single-token requests take the decode path; the rest fill 128-row tiles whose
    8-row groups are consecutive, gap-free union-core batches; sizes follow the padded batch."""
    # Two decode requests, a 300-token chunk, a single-token request after it (prefill path),
    # padded to 400 rows and 8 requests.
    plan = dsa_gqa_flashinfer.build_step_plan(
        [1, 1, 300, 1],
        [50, 9000, 1300, 77],
        padded_rows=400,
        padded_requests=8,
        decode_only=False,
        heads=NUM_HEADS,
    )
    sizes = dsa_gqa_flashinfer.plan_sizes(400, 8, False)
    assert sizes == {"decode_rows": 8, "tiles": 4 + 8, "prefill_requests": 8}
    assert plan["decode_lengths"].tolist() == [50, 9000] + [0] * 6
    assert plan["decode_requests"][:2].tolist() == [0, 1]
    assert len(plan["query_rows"]) == 12 * 128 and len(plan["batch_starts"]) == 12 * 16 + 1
    rows = plan["query_rows"][plan["query_rows"] >= 0]
    assert rows.tolist() == list(range(2, 303))
    # Request 2's rows end at keys 1001 .. 1300; request 3's single row at key 77.
    ends = plan["row_ends"][plan["query_rows"] >= 0]
    assert ends.tolist() == list(range(1001, 1301)) + [77]
    assert plan["tile_requests"][:4].tolist() == [2, 2, 2, 3]
    starts = plan["batch_starts"]
    assert (np.diff(starts) >= 0).all() and starts[0] == 2 and starts[-1] == 303
    assert plan["sample_ends"][:2].tolist() == [1300 // 128 * 3, 77 // 128 * 3]
    decode_plan = dsa_gqa_flashinfer.build_step_plan(
        [1] * 3, [10, 20, 30], padded_rows=4, padded_requests=4, decode_only=True, heads=NUM_HEADS
    )
    assert decode_plan["decode_lengths"].tolist() == [10, 20, 30, 0]
    assert decode_plan["query_rows"].size == 0


@pytest.mark.skipif(not dsa_gqa_flashinfer.HAVE_TRITON, reason="Triton is required")
def test_write_paged_rows_and_page_table():
    """Token rows land at their (block, position) slots, padding tokens write nothing, and the
    page table splits each block into 64-key pages."""
    device = torch.device("cuda", torch.cuda.current_device())
    cache = torch.zeros(6, 256, INDEX_DIM, device=device, dtype=torch.bfloat16)
    rows = torch.randn(5, INDEX_DIM, device=device, dtype=torch.bfloat16)
    blocks = torch.tensor([3, 3, 0, -1, 5], device=device, dtype=torch.int64)
    positions = torch.tensor([0, 255, 7, 9, 128], device=device, dtype=torch.int32)
    dsa_gqa_flashinfer.write_paged_rows(rows, cache, blocks, positions)
    expected = torch.zeros_like(cache)
    for t in (0, 1, 2, 4):
        expected[blocks[t], positions[t]] = rows[t]
    assert torch.equal(cache, expected)
    table = torch.tensor([[2, 0], [5, 1]], device=device, dtype=torch.int32)
    pages = torch.full((3, 10), -7, device=device, dtype=torch.int32)
    dsa_gqa_flashinfer.fill_page_table(table, 256, pages)
    assert pages[:2, :8].tolist() == [[8, 9, 10, 11, 0, 1, 2, 3], [20, 21, 22, 23, 4, 5, 6, 7]]
    assert (pages[:, 8:] == -7).all() and (pages[2] == -7).all()


def _rows_of(query_lengths, kv_lengths):
    """Each step row's request and causal key count."""
    requests, lengths = [], []
    for request, (q_len, kv_len) in enumerate(zip(query_lengths, kv_lengths)):
        requests += [request] * q_len
        lengths += list(range(kv_len - q_len + 1, kv_len + 1))
    return requests, lengths


def _reference_rows(query, index_query, keys, values, index_keys, table, requests, lengths):
    """Exact float64 top-k selection of each row's causal keys, then attention over them."""
    page_size = keys.size(1)
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row, (request, length) in enumerate(zip(requests, lengths)):
        positions = torch.arange(length, device=query.device)
        pages = table[request, positions // page_size].long()
        slots = positions % page_size
        scores = index_keys[pages, slots].double() @ index_query[row].double()
        selected = scores.topk(min(TOPK, length)).indices
        k = keys[pages[selected], slots[selected], 0].float()
        v = values[pages[selected], slots[selected], 0].float()
        probs = (query[row].float() @ k.T * HEAD_DIM**-0.5).softmax(-1)
        out[row] = probs @ v
    return out


@requires_flashinfer_dsa
@pytest.mark.parametrize("decode_only", [True, False])
def test_flashinfer_replays_in_cuda_graph(decode_only):
    """A captured layer, replayed for new batches of one padded size (new keys, queries, tables
    and lengths, plan buffers refilled in place), matches an exact PyTorch reference, and
    padding rows stay zero."""
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(71)
    page_size, bound = 256, 9216
    if decode_only:
        padded_rows = padded_requests = 4
        batches = [([1, 1, 1], [700, 9000, 2600]), ([1, 1, 1, 1], [9000, 513, 4000, 1])]
    else:
        padded_rows, padded_requests = 640, 4
        batches = [([1, 1, 300, 37], [700, 9000, 3000, 37]), ([1, 520], [8000, 9000])]
    max_pages = bound // page_size
    num_pages = padded_requests * max_pages
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.zeros(padded_requests, max_pages, device=device, dtype=torch.int32)
    query = torch.zeros(padded_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(padded_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    sizes = dsa_gqa_flashinfer.plan_sizes(padded_rows, padded_requests, decode_only)
    empty = dsa_gqa_flashinfer.build_step_plan(
        [1],
        [1],
        padded_rows=padded_rows,
        padded_requests=padded_requests,
        decode_only=decode_only,
        heads=NUM_HEADS,
    )
    buffers = {
        name: torch.zeros(len(v), device=device, dtype=torch.int32) for name, v in empty.items()
    }
    page_table = torch.zeros(
        padded_requests,
        dsa_gqa_flashinfer.page_table_columns(max_pages, page_size, bound, TOPK),
        device=device,
        dtype=torch.int32,
    )
    plan = dsa_gqa_flashinfer.StepPlan(
        decode_rows=sizes["decode_rows"],
        num_tiles=sizes["tiles"],
        max_context=bound,
        page_table=page_table,
        **buffers,
    )

    def fill(query_lengths, kv_lengths):
        arrays = dsa_gqa_flashinfer.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_rows,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=NUM_HEADS,
        )
        for name, values_np in arrays.items():
            buffers[name].copy_(torch.from_numpy(values_np))
        real = sum(query_lengths)
        query.zero_()
        index_query.zero_()
        query[:real] = torch.randn(real, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        index_query[:real] = torch.randn(real, INDEX_DIM, device=device, dtype=torch.bfloat16)
        keys.copy_(torch.randn_like(keys))
        index_keys.copy_(torch.randn_like(index_keys))
        table.copy_(torch.randperm(num_pages, device=device).view(padded_requests, max_pages))
        dsa_gqa_flashinfer.fill_page_table(table, page_size, page_table)
        return real

    def attend():
        return dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            plan=plan,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    fill(*batches[0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attend()  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = attend()

    for query_lengths, kv_lengths in batches:
        real = fill(query_lengths, kv_lengths)
        graph.replay()
        requests, lengths = _rows_of(query_lengths, kv_lengths)
        expected = _reference_rows(
            query[:real], index_query[:real], keys, values, index_keys, table, requests, lengths
        )
        torch.testing.assert_close(graph_output[:real].float(), expected, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[real:]) == 0


# Dynamic inference context and attention layer.


def _inference_config(**overrides) -> InferenceConfig:
    kwargs = dict(
        max_sequence_length=2048,
        max_requests=4,
        max_tokens=4096,
        block_size_tokens=64,
        buffer_size_gb=1,
        num_cuda_graphs=None,
        unified_memory_level=0,
    )
    kwargs.update(overrides)
    return InferenceConfig(**kwargs)


class TestDynamicSparseAttention:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def test_speculative_decoding_is_rejected(self):
        with pytest.raises(NotImplementedError, match="speculative decoding"):
            DynamicInferenceContext(
                model_config=_config(), inference_config=_inference_config(num_speculative_tokens=1)
            )

    def _attention(self, config: TransformerConfig) -> DSGroupedSelfAttention:
        stack = (
            dsa_layer_specs.dsa_inference_stack_spec
            if config.transformer_impl == "inference_optimized"
            else dsa_layer_specs.dsa_stack_spec
        )
        spec = stack.submodules.attention_layer.submodules.self_attention
        return build_module(spec, config=config, layer_number=1).cuda().eval()

    @requires_flashinfer_dsa
    def test_context_pages_indexer_keys_with_the_kv_cache(self):
        config = _config()
        plain = DynamicInferenceContext(
            model_config=_config(experimental_attention_variant=None, dsa_indexer_mode="standard"),
            inference_config=_inference_config(),
        )
        context = DynamicInferenceContext(model_config=config, inference_config=_inference_config())
        assert context.cache_dsa_indexer_keys and not plain.cache_dsa_indexer_keys
        assert context.dsa_memory_buffer.shape == (
            1,
            context.kv_block_allocator.pool_size,
            context.block_size_tokens,
            INDEX_DIM,
        )
        # Each block also holds one indexer key per token, so a block costs more bytes.
        assert context.block_size_bytes == plain.block_size_bytes + (
            config.params_dtype.itemsize * context.block_size_tokens * INDEX_DIM
        )

    @requires_flashinfer_dsa
    @pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
    def test_mixed_prefill_and_decode_match_uncached_forward(self, transformer_impl):
        """Cached prefill, decode, and mixed batches match uncached sparse attention.

        Prompts longer than the routing top-k make sparse selection matter.
        """
        config = _config(transformer_impl=transformer_impl)
        context = DynamicInferenceContext(model_config=config, inference_config=_inference_config())
        attention = self._attention(config)

        def add_request(request_id, prompt_length):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(prompt_length, dtype=torch.long),
                    sampling_params=SamplingParams(num_tokens_to_generate=8),
                )
            )

        def step():
            hidden = torch.randn(
                context.padded_active_token_count,
                1,
                HIDDEN_SIZE,
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.inference_mode(), InferenceMode.active():
                output, _ = attention(
                    hidden_states=hidden, attention_mask=None, inference_context=context
                )
            return output, hidden

        def uncached(history):
            with torch.inference_mode():
                output, _ = attention(hidden_states=history, attention_mask=None)
            return output

        histories = {}
        prompt_lengths = {0: 700, 1: 1100}
        for request_id, length in prompt_lengths.items():
            add_request(request_id, length)
        context.initialize_attention_state()
        output, hidden = step()
        start = 0
        for request_id, length in prompt_lengths.items():
            histories[request_id] = hidden[start : start + length]
            torch.testing.assert_close(
                output[start : start + length],
                uncached(histories[request_id]),
                atol=2e-2,
                rtol=2e-2,
            )
            start += length

        for decode_step in range(3):
            active = sorted(histories)
            context.update_requests(
                active_requests_mask=torch.ones(len(active), dtype=torch.int32),
                new_tokens=torch.full((len(active),), 7, dtype=torch.long),
            )
            if decode_step == 1:
                # Mix a fresh prefill into a decode step.
                add_request(2, 37)
            context.initialize_attention_state()
            output, hidden = step()
            for row, request_id in enumerate(active):
                histories[request_id] = torch.cat((histories[request_id], hidden[row : row + 1]))
                torch.testing.assert_close(
                    output[row], uncached(histories[request_id])[-1], atol=2e-2, rtol=2e-2
                )
            if decode_step == 1:
                histories[2] = hidden[len(active) : len(active) + 37]
                torch.testing.assert_close(
                    output[len(active) : len(active) + 37],
                    uncached(histories[2]),
                    atol=2e-2,
                    rtol=2e-2,
                )
