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
from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels, dsa_gqa
from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    DSGroupedSelfAttention,
    _deep_select_decode_supported,
    _sparse_attention_rows,
    dynamic_sparse_attention,
)
from megatron.core.transformer.experimental_attention_variant.dsa_gqa_inference_kernels import (
    HAVE_TRITON,
    SCORE_ROW_BLOCK,
    build_row_metadata,
    dsa_gqa_indexer_scores,
    dsa_gqa_sparse_attention,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HAVE_SSA = importlib.util.find_spec("simplified_sparse_attention") is not None

pytestmark = pytest.mark.skipif(not HAVE_TRITON, reason="Triton is required")
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


def _rows_with_padding(query_lengths, kv_lengths, padding_rows, device):
    """Row metadata for requests plus empty padding rows and blocks, on ``device``."""
    row_requests, row_lengths, block_starts, block_rows, block_requests = build_row_metadata(
        torch.tensor(query_lengths), torch.tensor(kv_lengths)
    )
    pad = torch.zeros(padding_rows, dtype=torch.int32)
    row_requests = torch.cat((row_requests, pad - 1))
    row_lengths = torch.cat((row_lengths, pad))
    block_pad = torch.zeros(2, dtype=torch.int32)
    blocks = [
        torch.cat((tensor, block_pad)) for tensor in (block_starts, block_rows, block_requests)
    ]
    return [tensor.to(device) for tensor in (row_requests, row_lengths, *blocks)]


def _reference_scores(index_q, index_keys, table, row_requests, row_lengths, page_size):
    """Float64 scores of each row over its request's keys [0, length)."""
    scores = []
    for row in range(index_q.size(0)):
        length = int(row_lengths[row])
        positions = torch.arange(length, device=index_q.device)
        pages = table[max(int(row_requests[row]), 0), positions // page_size].long()
        keys = index_keys[pages, positions % page_size].double()
        scores.append(keys @ index_q[row].double())
    return scores


def _reference_attention(query, keys, values, table, row_requests, row_lengths, indices, scale):
    """Per-row attention over its valid selected keys; probabilities rounded to BF16 like PV."""
    page_size, groups = keys.size(1), keys.size(2)
    heads_per_group = query.size(1) // groups
    out = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    for row in range(query.size(0)):
        selected = indices[row]
        selected = selected[(selected >= 0) & (selected < row_lengths[row])].long()
        if selected.numel() == 0:
            continue
        pages = table[int(row_requests[row]), selected // page_size].long()
        for head in range(query.size(1)):
            group = head // heads_per_group
            k = keys[pages, selected % page_size, group].float()
            v = values[pages, selected % page_size, group].float()
            probs = (query[row, head].float() @ k.T * scale).softmax(-1)
            out[row, head] = probs @ v
    return out


def test_build_row_metadata():
    rows = build_row_metadata(torch.tensor([1, 70, 1]), torch.tensor([9, 135, 4]), row_block=32)
    row_requests, row_lengths, block_starts, block_rows, block_requests = rows
    assert row_requests.tolist() == [0] + [1] * 70 + [2]
    # Request 1 appends 70 tokens after 65 cached ones: lengths 66 .. 135.
    assert row_lengths.tolist() == [9] + list(range(66, 136)) + [4]
    assert block_starts.tolist() == [0, 1, 33, 65, 71]
    assert block_rows.tolist() == [1, 32, 32, 6, 1]
    assert block_requests.tolist() == [0, 1, 1, 1, 2]


@pytest.mark.parametrize("head_dim", [128, 192, 256])
def test_inference_kernels_match_reference_on_mixed_rows(head_dim):
    """Decode rows, a chunked-prefill block with a cached prefix, and padding rows."""
    torch.manual_seed(17)
    device = torch.device("cuda", torch.cuda.current_device())
    page_size, max_pages, heads, topk = 64, 4, 4, 16
    query_lengths, kv_lengths = [1, 70, 1], [3, 135, 137]
    row_requests, row_lengths, block_starts, block_rows, block_requests = _rows_with_padding(
        query_lengths, kv_lengths, padding_rows=3, device=device
    )
    num_rows = row_requests.numel()
    num_pages = len(query_lengths) * max_pages
    table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
    index_keys = torch.randn(num_pages, page_size, head_dim, device=device, dtype=torch.bfloat16)
    index_q = torch.randn(num_rows, head_dim, device=device, dtype=torch.bfloat16)

    scores = torch.empty(num_rows, 151, device=device, dtype=torch.float32)
    dsa_gqa_indexer_scores(
        index_q,
        index_keys,
        table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out=scores,
        fill_invalid=True,
    )
    expected = _reference_scores(index_q, index_keys, table, row_requests, row_lengths, page_size)
    # Padding rows belong to no score block, so neither mode writes them.
    real_rows = int((row_requests >= 0).sum())
    for row in range(real_rows):
        length = int(row_lengths[row])
        torch.testing.assert_close(
            scores[row, :length].double(), expected[row], atol=2e-4, rtol=2e-4
        )
        assert bool(torch.isneginf(scores[row, length:]).all())

    # Without filling, positions past each length keep whatever the buffer held.
    unfilled = torch.full_like(scores, 7.0)
    dsa_gqa_indexer_scores(
        index_q,
        index_keys,
        table,
        row_lengths,
        block_starts,
        block_rows,
        block_requests,
        out=unfilled,
        fill_invalid=False,
    )
    tail = torch.arange(scores.size(1), device=device)[None, :] >= row_lengths[:, None]
    assert bool((unfilled[tail] == 7.0).all())
    torch.testing.assert_close(unfilled[~tail], scores[~tail])

    indices = scores.topk(topk, dim=-1).indices.to(torch.int32)
    keys = torch.randn(num_pages, page_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    # Attention must accept a query view with the pitched token stride produced by QKV.
    query = torch.randn(num_rows, heads + 2, head_dim, device=device, dtype=torch.bfloat16)
    query = query[:, :heads]
    output = torch.full((num_rows, heads, head_dim), 9.0, device=device, dtype=torch.bfloat16)
    scale = head_dim**-0.5
    dsa_gqa_sparse_attention(
        query, keys, values, table, row_requests, row_lengths, indices, scale, out=output
    )
    expected = _reference_attention(
        query, keys, values, table, row_requests, row_lengths, indices, scale
    )
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(output[row_lengths == 0]) == 0


def test_inference_kernels_match_reference_across_batches():
    """Changing batches, prefill chunk sizes, score widths, and query pitches."""
    torch.manual_seed(23)
    device = torch.device("cuda", torch.cuda.current_device())
    page_size, max_pages, topk = 64, 8, 64
    scale = HEAD_DIM**-0.5
    for query_lengths in ([1], [1, 1, 1, 1, 1], [90], [1, 1, 33], [1, 200, 1, 7], [1] * 8):
        num_requests = len(query_lengths)
        kv_lengths = [
            int(torch.randint(q, max_pages * page_size + 1, ()).item()) for q in query_lengths
        ]
        row_requests, row_lengths, block_starts, block_rows, block_requests = _rows_with_padding(
            query_lengths, kv_lengths, padding_rows=1, device=device
        )
        num_rows = row_requests.numel()
        num_pages = num_requests * max_pages
        table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
        index_keys = torch.randn(
            num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16
        )
        keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        index_q = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
        width = max(kv_lengths)
        scores = torch.empty(num_rows, width, device=device, dtype=torch.float32)
        dsa_gqa_indexer_scores(
            index_q,
            index_keys,
            table,
            row_lengths,
            block_starts,
            block_rows,
            block_requests,
            out=scores,
            fill_invalid=True,
        )
        expected = _reference_scores(
            index_q, index_keys, table, row_requests, row_lengths, page_size
        )
        for row in range(num_rows):
            length = int(row_lengths[row])
            torch.testing.assert_close(
                scores[row, :length].double(), expected[row], atol=2e-4, rtol=2e-4
            )
        indices = scores.topk(min(topk, width), dim=-1).indices
        indices = torch.nn.functional.pad(indices, (0, topk - indices.size(1)), value=-1)
        indices = indices.to(torch.int32).contiguous()
        query = torch.randn(num_rows, NUM_HEADS + 3, HEAD_DIM, device=device, dtype=torch.bfloat16)
        query = query[:, :NUM_HEADS]
        output = torch.empty(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        dsa_gqa_sparse_attention(
            query, keys, values, table, row_requests, row_lengths, indices, scale, out=output
        )
        expected = _reference_attention(
            query, keys, values, table, row_requests, row_lengths, indices, scale
        )
        torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("deep_select_backend", [False, True])
def test_sparse_attention_rows_replay_in_cuda_graph(deep_select_backend, monkeypatch):
    """Graph replays with new mixed batches and padding must match eager runs."""
    device = torch.device("cuda", torch.cuda.current_device())
    if deep_select_backend and not _deep_select_decode_supported(device, 64):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    if not deep_select_backend:
        monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.manual_seed(31)
    num_pages, page_size, max_pages, topk = 64, 64, 4, 64
    max_requests, max_rows = 4, 48
    num_blocks = max_requests + -(-max_rows // SCORE_ROW_BLOCK)
    graph_width = max_pages * page_size
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.zeros(max_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.zeros(max_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    block_table = torch.zeros(max_requests, max_pages, device=device, dtype=torch.int32)
    row_requests = torch.full((max_rows,), -1, device=device, dtype=torch.int32)
    row_lengths = torch.zeros(max_rows, device=device, dtype=torch.int32)
    blocks = [torch.zeros(num_blocks, device=device, dtype=torch.int32) for _ in range(3)]

    def attend(width):
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=block_table,
            row_metadata=(row_requests, row_lengths, *blocks, num_blocks),
            max_context_length=width,
            index_topk=topk,
            softmax_scale=HEAD_DIM**-0.5,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        attend(graph_width)  # Compile outside capture, as engine warmup does.
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = attend(graph_width)

    # (query lengths, kv lengths): decode-only, mixed with a cached-prefix chunk, prefill-only.
    for query_lengths, kv_lengths in (
        ([1, 1, 1], [1, 256, 17]),
        ([1, 40, 1], [200, 140, 64]),
        ([45], [45]),
    ):
        metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
        real_rows, real_blocks = metadata[0].numel(), metadata[2].numel()
        row_requests.fill_(-1)
        row_lengths.zero_()
        for buffer in blocks:
            buffer.zero_()
        row_requests[:real_rows] = metadata[0].to(device)
        row_lengths[:real_rows] = metadata[1].to(device)
        for buffer, values_cpu in zip(blocks, metadata[2:]):
            buffer[:real_blocks] = values_cpu.to(device)
        query.copy_(torch.randn_like(query))
        index_query.copy_(torch.randn_like(index_query))
        block_table.copy_(
            torch.randperm(num_pages, device=device)[: max_requests * max_pages].view(
                max_requests, max_pages
            )
        )
        graph.replay()
        eager = attend(max(kv_lengths))
        live = row_lengths > 0
        torch.testing.assert_close(graph_output[live], eager[live], atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(graph_output[~live]) == 0


@pytest.mark.parametrize("deep_select_backend", [False, True])
def test_score_windows_match_one_window(deep_select_backend, monkeypatch):
    """Scoring rows in small windows (not aligned to score blocks) must not change the output."""
    device = torch.device("cuda", torch.cuda.current_device())
    if deep_select_backend and not _deep_select_decode_supported(device, 64):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    if not deep_select_backend:
        monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.manual_seed(43)
    page_size, max_pages, topk = 64, 8, 64
    query_lengths, kv_lengths = [1, 70, 1, 45], [300, 400, 9, 120]
    metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
    row_metadata = (*[tensor.to(device) for tensor in metadata], metadata[2].numel())
    num_rows, num_pages = metadata[0].numel(), len(query_lengths) * max_pages
    table = torch.randperm(num_pages, device=device).view(-1, max_pages).to(torch.int32)
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)

    def attend():
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            row_metadata=row_metadata,
            max_context_length=max(kv_lengths),
            index_topk=topk,
            softmax_scale=HEAD_DIM**-0.5,
        )

    one_window = attend()
    # A 28 KiB buffer holds 14-17 score rows at this width, so windows straddle the 32-row
    # score blocks and the request boundaries.
    monkeypatch.setattr(dsa_gqa, "SCORE_BUFFER_BYTES", 7 * 1024 * 4)
    torch.testing.assert_close(attend(), one_window, atol=2e-2, rtol=2e-2)


def test_decode_attention_splits_groups_and_skips_invalid_slots():
    """Multi-slice, multi-group rows must match a reference over the valid selected keys.

    Selections hold padding (-1), positions past the length, and an all-padding row.
    """
    torch.manual_seed(37)
    device = torch.device("cuda", torch.cuda.current_device())
    batch, groups, heads, page_size, max_pages, topk = 5, 2, 8, 64, 8, 512
    lengths = torch.tensor([1, 70, 300, 512, 0], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 1, 2, 3, -1], dtype=torch.int32, device=device)
    num_pages = batch * max_pages
    keys = torch.randn(num_pages, page_size, groups, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    table = torch.randperm(num_pages, device=device).view(batch, max_pages).to(torch.int32)
    indices = torch.full((batch, topk), -1, dtype=torch.int32, device=device)
    for row, length in enumerate(lengths.tolist()):
        chosen = torch.randperm(max(length, 1), device=device)[: min(length, topk - 8)]
        indices[row, : chosen.numel()] = chosen.to(torch.int32)
        # Positions at or past the length must be ignored.
        indices[row, -4:] = length + torch.arange(4, device=device, dtype=torch.int32)
    query = torch.randn(batch, heads, HEAD_DIM, device=device, dtype=torch.bfloat16)
    output = torch.full((batch, heads, HEAD_DIM), 9.0, device=device, dtype=torch.bfloat16)
    scale = HEAD_DIM**-0.5
    dsa_gqa_sparse_attention(
        query, keys, values, table, requests, lengths, indices, scale, out=output
    )
    expected = _reference_attention(
        query, keys, values, table, requests.clamp(min=0), lengths, indices, scale
    )
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.count_nonzero(output[-1]) == 0


def test_deep_select_rows_match_torch_topk(monkeypatch):
    """DeepSelect over unfilled score tails must select what torch.topk selects."""
    device = torch.device("cuda", torch.cuda.current_device())
    if not _deep_select_decode_supported(device, TOPK):
        pytest.skip("DeepSelect is not installed or does not support this GPU")
    torch.manual_seed(41)
    num_pages, page_size, max_pages = 96, 64, 32
    query_lengths, kv_lengths = [1, 1, 1, 1, 1, 100, 1], [1, 100, 511, 512, 513, 700, 2048]
    metadata = build_row_metadata(torch.tensor(query_lengths), torch.tensor(kv_lengths))
    num_rows, num_blocks = metadata[0].numel(), metadata[2].numel()
    row_metadata = (*[tensor.to(device) for tensor in metadata], num_blocks)
    keys = torch.randn(num_pages, page_size, 1, HEAD_DIM, device=device, dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    index_keys = torch.randn(num_pages, page_size, INDEX_DIM, device=device, dtype=torch.bfloat16)
    query = torch.randn(num_rows, NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    index_query = torch.randn(num_rows, INDEX_DIM, device=device, dtype=torch.bfloat16)
    table = torch.randint(0, num_pages, (len(query_lengths), max_pages), device=device)
    table = table.to(torch.int32)

    def attend():
        return _sparse_attention_rows(
            query=query,
            key_cache=keys,
            value_cache=values,
            index_query=index_query,
            index_key_cache=index_keys,
            block_table=table,
            row_metadata=row_metadata,
            max_context_length=max(kv_lengths),
            index_topk=TOPK,
            softmax_scale=HEAD_DIM**-0.5,
        )

    with_deep_select = attend()
    monkeypatch.setattr(dsa_gqa, "HAVE_DEEP_SELECT", False)
    torch.testing.assert_close(with_deep_select, attend(), atol=2e-2, rtol=2e-2)


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
        torch.testing.assert_close(normalized.float(), expected * weight, atol=2e-2, rtol=2e-2)
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
