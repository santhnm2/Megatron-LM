# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dynamic-batching inference for DSA over GQA with the simplified indexer.

Each request's attention K/V and its indexer keys live in paged caches that share one block
table (``DynamicInferenceContext.dsa_memory_buffer``). ``dynamic_sparse_attention`` routes every
query token of a decode, prefill, or mixed step to the top-k keys of its causal prefix and attends
to them, through the kernels selected by ``InferenceConfig.dsa_backend``:

- ``"flashinfer"``: FlashInfer's DSA kernels (``dsa_gqa_flashinfer``), from a per-step plan that
  the dynamic context builds on the host (``DSAMetadata``), so every step type can run as a CUDA
  graph.
"""

import torch

from megatron.core.inference.contexts import BaseInferenceContext


@torch.no_grad()
def dynamic_sparse_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    inference_context: BaseInferenceContext,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """DSA attention over paged caches for a decode, prefill, or mixed dynamic batch.

    The caches already hold this step's keys, values, and indexer keys.

    Args:
        query: Queries ``[T,1,Hq,D]`` for the step's padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,1,D]``.
        value_cache: Paged values ``[blocks,block_size,1,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``.
        block_table: Per-request physical block ids ``[requests,max_blocks]``.
        inference_context: The step's dynamic inference context.
        index_topk: Number of keys each query attends to.
        softmax_scale: Attention softmax scale.

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows of padding tokens are zero.
    """
    num_tokens, _, num_query_heads, _ = query.shape
    if key_cache.size(2) != 1:
        raise NotImplementedError(
            f"DSA-over-GQA inference needs one local KV group, got {key_cache.size(2)}."
        )
    backend = inference_context.dsa_backend
    if backend == "flashinfer":
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer

        del block_table  # The step plan carries FlashInfer's page table, derived from it.
        if not dsa_gqa_flashinfer.flashinfer_dsa_supported(
            query.device, index_topk, index_query.size(-1), key_cache.size(1), num_query_heads
        ):
            raise NotImplementedError(
                "FlashInfer's DSA kernels need SM100/SM103, top-k 512/1024/2048, indexer head "
                "dim 128, at most 128 query heads per rank, and a KV block size that is a "
                "multiple of 64."
            )
        output = dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query[:, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query.reshape(num_tokens, -1),
            index_key_cache=index_key_cache,
            plan=inference_context.dsa_metadata.plan(),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
            deterministic=inference_context.batch_invariant_mode,
        )
    else:
        raise ValueError(f"Unknown DSA inference backend {backend!r}.")
    return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))
