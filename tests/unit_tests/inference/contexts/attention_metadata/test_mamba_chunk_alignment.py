# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.contexts.attention_context.mamba_metadata import MambaMetadata


@pytest.mark.parametrize(
    'lengths, offsets, expected, last',
    [
        ([8480], [8480], [0, 96, *range(224, 8480, 128), 8480], [66]),
        ([2004], [16960], [0, 64, *range(192, 2004, 128), 2004], [16]),
        ([130, 130], [127, 127], [0, 1, 129, 130, 131, 259, 260], [2, 5]),
        ([4, 7], [126, 127], [0, 2, 4, 5, 11], [1, 3]),
        ([130, 130], None, [0, 128, 130, 258, 260], [1, 3]),
    ],
)
def test_cpu_ssd_chunks_preserve_prompt_alignment(lengths, offsets, expected, last):
    """Cover traced chunked prefills, mixed offsets, padding, and legacy metadata."""
    count, tokens = len(lengths), sum(lengths)
    padded_tokens = (tokens + 63) // 64 * 64
    metadata = MambaMetadata(
        max_requests=4,
        max_tokens=padded_tokens,
        max_intermediate_count=1,
        align_chunk_boundaries=offsets is not None,
    )
    fields = (
        'batch_indices_decode',
        'batch_indices_prefill',
        'seq_idx',
        'cu_seqlens',
        'cu_chunk_seqlens',
        'last_chunk_indices',
        'seq_idx_for_varlen',
        'conv_seq_idx',
        'conv_seq_start',
    )
    buffers = {name: getattr(metadata, '_' + name + '_buffer').cpu() for name in fields}
    metadata.bind_cpu_buffers(buffers)
    cu_query = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    transfer = metadata.compute_cpu_metadata(
        active_mamba_indices=torch.arange(count, dtype=torch.int32),
        token_to_request_idx=torch.arange(count).repeat_interleave(torch.tensor(lengths)),
        cpu_cu_query=cu_query,
        batch_dimensions=InferenceBatchDimensions(token_count=tokens, prefill_req_count=count),
        padded_batch_dimensions=InferenceBatchDimensions(
            token_count=padded_tokens, prefill_req_count=4
        ),
        enable_chunked_prefill=True,
        prefill_context_lengths=None if offsets is None else torch.tensor(offsets),
    )
    assert buffers['cu_chunk_seqlens'][: len(expected)].tolist() == expected
    assert buffers['last_chunk_indices'][:count].tolist() == last
    assert buffers['last_chunk_indices'][count:4].tolist() == list(
        range(last[-1] + 1, last[-1] + 1 + 4 - count)
    )
    assert (
        buffers['cu_chunk_seqlens'][len(expected) : transfer['padded_max_chunks'] + 1]
        .eq(tokens)
        .all()
    )
    chunk_counts = torch.tensor([last[0] + 1, *[b - a for a, b in zip(last, last[1:])]])
    assert buffers['seq_idx_for_varlen'][: last[-1] + 1].tolist() == (
        torch.arange(count).repeat_interleave(chunk_counts).tolist()
    )
