# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference adapter for the CuTe sparse-attention kernels behind DSA over GQA.

Prefill and uncached forwards run the training kernels from the optional
``simplified_sparse_attention`` package (SM103), so inference applies exactly the
row layout and causal top-k routing the model was trained with. The package is
imported on first use; decode uses the paged kernels in ``dsa_gqa_decode_triton``.
"""

from __future__ import annotations

import torch

SUPPORTED_QUERY_HEADS = (16, 32, 96)
KV_HEADS = 1
HEAD_DIM = 256
INDEX_DIM = 128
SUPPORTED_TOPKS = (512, 1024, 2048)


def _import_sparse_attention():
    """Import the optional kernel package, failing with a clear message when absent."""
    try:
        from simplified_sparse_attention import RowMetadata, sparse_attention
    except ImportError as error:
        raise ImportError(
            "DSA over GQA prefill requires the 'simplified_sparse_attention' package "
            "(CuTe sparse-attention kernels for SM103). Add it to PYTHONPATH."
        ) from error
    return RowMetadata, sparse_attention


def _validate_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    indexer_topk: int,
) -> None:
    if query.ndim != 3 or query.shape[1] not in SUPPORTED_QUERY_HEADS or query.shape[2] != HEAD_DIM:
        raise NotImplementedError(
            f"CuTe DSA requires query [Tq,Hq,{HEAD_DIM}] with Hq in {SUPPORTED_QUERY_HEADS}, "
            f"got {tuple(query.shape)}."
        )
    for name, tensor in (("key", key), ("value", value)):
        if tensor.ndim != 3 or tensor.shape[1:] != (KV_HEADS, HEAD_DIM):
            raise NotImplementedError(
                f"CuTe DSA requires {name} [Tk,{KV_HEADS},{HEAD_DIM}], got {tuple(tensor.shape)}."
            )
    if q_indexer.shape != (query.size(0), INDEX_DIM):
        raise NotImplementedError(
            f"CuTe DSA requires indexer query [Tq,{INDEX_DIM}], got {tuple(q_indexer.shape)}."
        )
    if k_indexer.shape != (key.size(0), INDEX_DIM):
        raise NotImplementedError(
            f"CuTe DSA requires indexer key [Tk,{INDEX_DIM}], got {tuple(k_indexer.shape)}."
        )
    if indexer_topk not in SUPPORTED_TOPKS:
        raise NotImplementedError(
            f"CuTe DSA requires top-K in {SUPPORTED_TOPKS}, got {indexer_topk}."
        )


@torch.no_grad()
def run_sparse_attention_inference(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    key_offset: int,
    indexer_topk: int,
) -> torch.Tensor:
    """Run CuTe sparse attention for one request's new queries over its cached prefix.

    Query row ``i`` sits at absolute position ``key_offset + i`` and attends causally to
    keys ``[0, key_offset + i]`` (itself included), routed to the indexer's top-k keys, so
    fresh prompts, chunked prefill, and prefix-cache hits share the training kernels'
    row-bounded layout. The softmax scale is fixed by the kernels at ``HEAD_DIM**-0.5``.

    Args:
        query: Query ``[Tq,Hq,256]``.
        key: Keys ``[Tk,1,256]`` with ``Tk == key_offset + Tq``.
        value: Values ``[Tk,1,256]``.
        q_indexer: Indexer queries ``[Tq,128]``.
        k_indexer: Indexer keys ``[Tk,128]``.
        key_offset: Number of cached tokens that precede the first query.
        indexer_topk: Routing top-k.

    Returns:
        Attention output ``[Tq,Hq,256]``.
    """
    RowMetadata, sparse_attention = _import_sparse_attention()
    _validate_shapes(query, key, value, q_indexer, k_indexer, indexer_topk)
    num_queries, num_keys = query.size(0), key.size(0)
    if num_keys != key_offset + num_queries:
        raise ValueError(
            f"CuTe DSA inference expects {key_offset} cached keys before {num_queries} "
            f"queries, got {num_keys} keys."
        )
    device = query.device
    layout = RowMetadata(
        torch.zeros(num_queries, dtype=torch.int64, device=device),
        torch.arange(key_offset + 1, num_keys + 1, dtype=torch.int64, device=device),
        torch.ones(num_queries, dtype=torch.bool, device=device),
        torch.ones(1, dtype=torch.float32, device=device),
        triangular_scores=key_offset == 0,
        mask_invalid_rows=False,
    )
    output, _ = sparse_attention(
        query,
        key,
        value,
        q_indexer.contiguous(),
        k_indexer.contiguous(),
        layout,
        indexer_topk,
        loss_coeff=0.0,
    )
    return output


__all__ = ["run_sparse_attention_inference"]
