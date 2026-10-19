# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.tensor_parallel.inference_layers import (
    InferenceLayerNormColumnParallelLinear,
    InferenceRowParallelLinear,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    dynamic_sparse_attention,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_decode_kernel import (
    HAVE_CUTEDSL,
    dsa_gqa_decode_attention,
    dsa_gqa_decode_indexer_score,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HAVE_SSA = importlib.util.find_spec("simplified_sparse_attention") is not None

pytestmark = pytest.mark.skipif(not HAVE_CUTEDSL, reason="CuTeDSL is required")
requires_ssa = pytest.mark.skipif(
    not HAVE_SSA, reason="The simplified-sparse-attention wheel is required"
)

HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 256
INDEX_DIM = 128
TOPK = 512


def _config_kwargs(**overrides):
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
        dsa_simplified_use_learned_k=True,
        dsa_min_memory_backend="cute",
        dsa_indexer_n_heads=1,
        dsa_indexer_head_dim=INDEX_DIM,
        dsa_indexer_topk=TOPK,
        dsa_indexer_rotate_activation=False,
        dsa_indexer_scoring_relu=False,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module", autouse=True)
def initialize_decode_test_distributed():
    Utils.initialize_distributed()


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(dsa_indexer_topk=256), "top-K"),
        (dict(num_query_groups=2), "MQA1"),
        (dict(dsa_indexer_scoring_relu=True), "unweighted"),
        (dict(dsa_simplified_use_learned_k=False), "learned-Q/K"),
        (dict(experimental_attention_variant=None), "experimental_attention_variant='dsa'"),
        (dict(dsa_min_memory_backend="reference"), "only implemented by"),
    ],
)
def test_config_rejects_configs_outside_the_cute_contract(overrides, message):
    with pytest.raises(ValueError, match=message):
        TransformerConfig(**_config_kwargs(**overrides))


def test_missing_sparse_attention_package_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "simplified_sparse_attention", None)
    with pytest.raises(ImportError, match="simplified_sparse_attention"):
        dsa_cute_kernels.run_sparse_attention_inference(
            query=None,
            key=None,
            value=None,
            q_indexer=None,
            k_indexer=None,
            key_offset=0,
            indexer_topk=TOPK,
        )


def test_speculative_decode_steps_are_rejected():
    context = SimpleNamespace(num_speculative_tokens=2)
    with pytest.raises(NotImplementedError, match="MTP speculative decoding"):
        dynamic_sparse_attention(
            query=None,
            key_cache=None,
            value_cache=None,
            index_query=None,
            index_key_cache=None,
            block_table=None,
            inference_context=context,
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )


@pytest.mark.parametrize("head_dim", [128, 192, 256])
def test_paged_decode_kernels_match_reference(head_dim):
    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, pages_per_request, page_size, heads = 4, 3, 64, 4
    lengths = torch.tensor([3, 63, 65, 137], dtype=torch.int32, device=device)
    block_table = (
        torch.randperm(batch * pages_per_request, device=device)
        .reshape(batch, pages_per_request)
        .to(torch.int32)
    )
    keys = torch.randn(
        batch * pages_per_request, page_size, 1, head_dim, device=device, dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    index_q = torch.randn(batch, head_dim, device=device, dtype=torch.bfloat16)
    scores = torch.empty(batch, 151, device=device, dtype=torch.float32)
    dsa_gqa_decode_indexer_score(index_q, keys[:, :, 0], block_table, lengths, out=scores)
    expected_scores = torch.full_like(scores, -torch.inf)
    for request, length in enumerate(lengths.tolist()):
        positions = torch.arange(length, device=device)
        k = keys[block_table[request, positions // page_size].long(), positions % page_size, 0]
        expected_scores[request, :length] = k.float() @ index_q[request].float()
    torch.testing.assert_close(scores, expected_scores, atol=2e-4, rtol=2e-4)

    indices = scores.topk(16, dim=-1).indices.to(torch.int32)
    # Attention must accept a query view with the pitched token stride produced by QKV.
    q = torch.randn(batch, heads + 2, head_dim, device=device, dtype=torch.bfloat16)[:, :heads]
    output = torch.empty(batch, heads, head_dim, device=device, dtype=torch.bfloat16)
    scale = head_dim**-0.5
    dsa_gqa_decode_attention(q, keys, values, block_table, indices, lengths, scale, out=output)
    expected = torch.empty_like(output)
    for request, length in enumerate(lengths.tolist()):
        selected = indices[request, : min(length, indices.size(1))].long()
        pages = block_table[request, selected // page_size].long()
        k = keys[pages, selected % page_size, 0]
        v = values[pages, selected % page_size, 0]
        probs = (q[request].float() @ k.float().T * scale).softmax(-1).to(torch.bfloat16)
        expected[request] = (probs.float() @ v.float()).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)


def test_decode_kernels_compile_once_across_batch_sizes(monkeypatch):
    """A changing decode batch, score width, and query pitch must reuse one compilation."""
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_gqa_decode_kernel as kernels,
    )

    for name in ("_COMPILED", "_OPERATORS", "_ATTENTION_COMPILED", "_ATTENTION_OPERATORS"):
        monkeypatch.setattr(kernels, name, {})
    compile_calls = []
    original_compile = kernels.cute.compile

    def count_compile(*args, **kwargs):
        compile_calls.append(1)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(kernels.cute, "compile", count_compile)
    torch.manual_seed(23)
    device = torch.device("cuda", torch.cuda.current_device())
    num_pages, page_size, max_pages, heads, topk = 32, 64, 4, NUM_HEADS, 16
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5
    for batch in (1, 2, 5, 3, 8, 1):
        table = (
            torch.randperm(num_pages, device=device)[: batch * max_pages]
            .view(batch, max_pages)
            .to(torch.int32)
        )
        lengths = torch.randint(1, max_pages * page_size + 1, (batch,), device=device)
        lengths = lengths.to(torch.int32)
        width = int(lengths.max().item())
        index_q = torch.randn(batch, INDEX_DIM, device=device, dtype=torch.bfloat16)
        scores = torch.empty(batch, width, device=device, dtype=torch.float32)
        dsa_gqa_decode_indexer_score(index_q, index_keys, table, lengths, out=scores)
        positions = torch.arange(width, device=device)
        logical_keys = index_keys[table[:, positions // page_size].long(), positions % page_size]
        expected_scores = torch.einsum("bd,bkd->bk", index_q.float(), logical_keys.float())
        expected_scores.masked_fill_(positions[None, :] >= lengths[:, None], -torch.inf)
        torch.testing.assert_close(scores, expected_scores, atol=2e-4, rtol=2e-4)

        indices = scores.topk(min(topk, width), dim=-1).indices
        indices = torch.nn.functional.pad(indices, (0, topk - indices.size(1)), value=-1)
        indices = indices.to(torch.int32).contiguous()
        # Pitched query rows, as produced by slicing a fused QKV projection.
        query = torch.randn(batch, heads + 3, HEAD_DIM, device=device, dtype=torch.bfloat16)
        query = query[:, :heads]
        output = torch.empty(batch, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
        dsa_gqa_decode_attention(query, keys, values, table, indices, lengths, scale, out=output)
        for request in range(batch):
            selected = indices[request]
            selected = selected[(selected >= 0) & (selected < lengths[request])].long()
            pages = table[request, selected // page_size].long()
            k = keys[pages, selected % page_size, 0].float()
            v = values[pages, selected % page_size, 0].float()
            probs = (query[request].float() @ k.T * scale).softmax(-1).to(torch.bfloat16)
            torch.testing.assert_close(
                output[request], (probs.float() @ v).to(torch.bfloat16), atol=2e-2, rtol=2e-2
            )
    assert len(compile_calls) == 2
    assert len(kernels._COMPILED) == len(kernels._ATTENTION_COMPILED) == 1


def test_sparse_decode_replays_in_cuda_graph():
    """Graph replays with new lengths and padded rows must match eager decode."""
    from megatron.core.transformer.experimental_attention_variant.dsa_gqa import _sparse_decode

    torch.manual_seed(31)
    device = torch.device("cuda", torch.cuda.current_device())
    num_pages, page_size, max_pages, batch, topk = 32, 64, 4, 4, 64
    graph_width = max_pages * page_size
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.zeros(batch, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(batch, INDEX_DIM, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(batch, max_pages, device=device, dtype=torch.int32)
    kv_lengths = torch.ones(batch, device=device, dtype=torch.int32)

    def decode(width):
        return _sparse_decode(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=block_table,
            kv_lengths=kv_lengths,
            max_context_length=width,
            index_topk=topk,
            softmax_scale=HEAD_DIM**-0.5,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode(graph_width)  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = decode(graph_width)

    for lengths in ([1, 256, 0, 0], [200, 17, 64, 0], [65, 3, 256, 129]):
        query.copy_(torch.randn_like(query))
        index_query.copy_(torch.randn_like(index_query))
        block_table.copy_(
            torch.randperm(num_pages, device=device)[: batch * max_pages].view(batch, max_pages)
        )
        kv_lengths.copy_(torch.tensor(lengths, device=device, dtype=torch.int32))
        graph.replay()
        eager = decode(max(lengths))
        live = kv_lengths > 0
        torch.testing.assert_close(graph_output[live], eager[live], atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[~live]) == 0


@requires_ssa
@pytest.mark.parametrize("key_offset", [0, 1, 300, 777])
def test_rectangular_prefill_matches_full_sequence_rows(key_offset):
    """A chunk over its cached prefix must equal the matching rows of a full prefill."""
    torch.manual_seed(5)
    device = torch.device("cuda", torch.cuda.current_device())
    total, chunk = key_offset + 400, 400

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)

    query = randn(total, NUM_HEADS, HEAD_DIM)
    key, value = randn(total, 1, HEAD_DIM), randn(total, 1, HEAD_DIM)
    q_idx, k_idx = randn(total, INDEX_DIM), randn(total, INDEX_DIM)
    full = dsa_cute_kernels.run_sparse_attention_inference(
        query=query,
        key=key,
        value=value,
        q_indexer=q_idx,
        k_indexer=k_idx,
        key_offset=0,
        indexer_topk=TOPK,
    )
    tail = dsa_cute_kernels.run_sparse_attention_inference(
        query=query[key_offset:],
        key=key,
        value=value,
        q_indexer=q_idx[key_offset:],
        k_indexer=k_idx,
        key_offset=key_offset,
        indexer_topk=TOPK,
    )
    assert tail.shape == (chunk, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(tail, full[key_offset:], atol=2e-2, rtol=2e-2)


@requires_ssa
class TestDynamicSparseAttention:
    @pytest.fixture(scope="function", autouse=True)
    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    def _config(self, transformer_impl: str) -> TransformerConfig:
        return TransformerConfig(**_config_kwargs(transformer_impl=transformer_impl))

    def _attention(self, config: TransformerConfig) -> DSGroupedSelfAttention:
        inference_layers = config.transformer_impl == "inference_optimized"
        return DSGroupedSelfAttention(
            config=config,
            submodules=SelfAttentionSubmodules(
                linear_qkv=(
                    InferenceLayerNormColumnParallelLinear
                    if inference_layers
                    else TELayerNormColumnParallelLinear
                ),
                core_attention=TEDotProductAttention,
                linear_proj=InferenceRowParallelLinear if inference_layers else TERowParallelLinear,
            ),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        ).eval()

    @pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
    def test_indexer_reads_the_fused_qkv_norm_output(self, transformer_impl):
        """The QKV projection returns its RMSNorm output, which feeds the indexer."""
        attention = self._attention(self._config(transformer_impl))
        assert set(attention.state_dict()) >= {
            "core_attention.indexer.linear_q.weight",
            "core_attention.indexer.linear_k.weight",
        }
        assert not any(
            key.startswith("core_attention.") and ".indexer." not in key
            for key in attention.state_dict()
        )
        hidden = torch.randn(32, 1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            output, bias, normalized = attention.linear_qkv(hidden)
        assert bias is None
        weight = attention.linear_qkv.layer_norm_weight.float()
        expected = hidden.float() * torch.rsqrt(
            hidden.float().square().mean(-1, keepdim=True) + attention.config.layernorm_epsilon
        )
        torch.testing.assert_close(
            normalized.float(), expected * weight, atol=2e-2, rtol=2e-2
        )
        assert output.shape[-1] == (NUM_HEADS + 2) * HEAD_DIM

    def test_static_batching_context_is_rejected(self):
        attention = self._attention(self._config("transformer_engine"))
        context = SimpleNamespace(is_dynamic_batching=lambda: False)
        with pytest.raises(NotImplementedError, match="dynamic batching only"):
            attention(hidden_states=None, attention_mask=None, inference_context=context)

    @pytest.mark.parametrize("transformer_impl", ["transformer_engine", "inference_optimized"])
    def test_mixed_prefill_and_decode_match_training_forward(self, transformer_impl):
        """Cached decode and mixed batches must match uncached attention.

        Prompts longer than the routing top-k make sparse selection matter.
        """
        config = self._config(transformer_impl)
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=2048,
                max_requests=4,
                max_tokens=4096,
                block_size_tokens=64,
                buffer_size_gb=1,
                num_cuda_graphs=None,
                unified_memory_level=0,
            ),
        )
        assert context.cache_dsa_indexer_keys
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

        histories = {}
        prompt_lengths = {0: 700, 1: 1100}
        for request_id, length in prompt_lengths.items():
            add_request(request_id, length)
        context.initialize_attention_state()
        output, hidden = step()
        start = 0
        for request_id, length in prompt_lengths.items():
            histories[request_id] = hidden[start : start + length]
            with torch.inference_mode():
                expected, _ = attention(hidden_states=histories[request_id], attention_mask=None)
            torch.testing.assert_close(
                output[start : start + length], expected, atol=2e-2, rtol=2e-2
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
                with torch.inference_mode():
                    expected, _ = attention(
                        hidden_states=histories[request_id], attention_mask=None
                    )
                torch.testing.assert_close(output[row], expected[-1], atol=2e-2, rtol=2e-2)
            if decode_step == 1:
                histories[2] = hidden[len(active) : len(active) + 37]
                with torch.inference_mode():
                    expected, _ = attention(hidden_states=histories[2], attention_mask=None)
                torch.testing.assert_close(
                    output[len(active) : len(active) + 37], expected, atol=2e-2, rtol=2e-2
                )
