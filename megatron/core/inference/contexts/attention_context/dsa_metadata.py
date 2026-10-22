# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import torch

from .metadata_base import MetadataBase


@dataclass(frozen=True)
class DSAPrefillSpan:
    """One prefill request of a step: its query rows and its key count after the step."""

    request: int  # active-request index (row of the block table)
    query_start: int
    query_end: int
    kv_length: int


@dataclass(frozen=True)
class DSAStepInfo:
    """Host-side facts about a step that DSA layers need, so they read no lengths back."""

    num_decode_requests: int
    max_decode_query_length: int  # 0 without decode requests
    max_decode_kv_length: int  # 0 without decode requests
    max_kv_length: int  # over the step's real requests; 0 for an empty step
    prefill_spans: Tuple[DSAPrefillSpan, ...]


class DSAMetadata(MetadataBase):
    """Per-step metadata of DSA-over-GQA (simplified indexer) dynamic-batching inference.

    ``update`` runs on the CPU once per step from the step's request lengths and records a
    ``DSAStepInfo`` (decode-request count, longest contexts, each prefill request's query rows and
    key count), which every backend reads instead of syncing lengths from the GPU.

    With ``backend="flashinfer"`` it also holds the step plan of ``dsa_gqa_flashinfer``
    (decode-path rows, 128-row prefill tiles, their union-core groups, threshold-sample counts) and
    FlashInfer's 64-key page table, in static buffers at their largest sizes: ``update`` builds the
    plan on the CPU; ``transfer`` copies it to the GPU and derives the page table from the step's
    block table (once per step, outside CUDA graphs); ``plan`` returns the views a DSA layer reads.
    A step's view sizes depend only on its padded batch, so captured graphs replay on any batch
    that fits them.
    """

    def __init__(
        self,
        *,
        backend: Literal["flashinfer", "cute"],
        max_requests: int,
        max_tokens: int,
        max_kv_blocks: int,
        block_size_tokens: int,
        max_sequence_length: int,
        query_heads: int,
        index_topk: int,
    ):
        super().__init__()
        if backend not in ("flashinfer", "cute"):
            raise ValueError(f"Unknown DSA backend {backend!r}; expected 'flashinfer' or 'cute'.")
        self.backend = backend
        self.block_size_tokens = block_size_tokens
        self.max_sequence_length = max_sequence_length
        self.query_heads = query_heads
        self._fi = None
        if backend == "flashinfer":
            self._init_flashinfer_plan(max_requests, max_tokens, max_kv_blocks, index_topk)
        self.reset()

    def _init_flashinfer_plan(
        self, max_requests: int, max_tokens: int, max_kv_blocks: int, index_topk: int
    ) -> None:
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer

        dsa_gqa_flashinfer.require_flashinfer_dsa()
        self._fi = dsa_gqa_flashinfer
        device = torch.cuda.current_device()
        tiles = dsa_gqa_flashinfer.plan_sizes(max_tokens, max_requests, False)["tiles"]
        tiled = tiles * dsa_gqa_flashinfer.TILE_ROWS
        self._capacity = {
            "decode_requests": max_requests,
            "decode_lengths": max_requests,
            "tile_requests": tiles,
            "tile_ordinals": tiles,
            "row_ends": tiled,
            "query_rows": tiled,
            "request_rows": max_requests,
            "sample_ends": max_requests,
            "batch_starts": tiled // dsa_gqa_flashinfer.union_group(self.query_heads) + 1,
        }
        total = sum(self._capacity.values())
        self._cpu_buf = torch.zeros(total, dtype=torch.int32, pin_memory=True)
        self._gpu_buf = torch.zeros(total, dtype=torch.int32, device=device)
        self._cpu_views, self._gpu_views, offset = {}, {}, 0
        for name, size in self._capacity.items():
            self._cpu_views[name] = self._cpu_buf[offset : offset + size]
            self._gpu_views[name] = self._gpu_buf[offset : offset + size]
            offset += size
        self.page_table = torch.zeros(
            max_requests,
            dsa_gqa_flashinfer.page_table_columns(
                max_kv_blocks, self.block_size_tokens, self.max_sequence_length, index_topk
            ),
            dtype=torch.int32,
            device=device,
        )

    def update(
        self,
        query_lengths: Sequence[int],
        kv_lengths: Sequence[int],
        *,
        num_decode_requests: int,
        padded_tokens: int,
        padded_requests: int,
        decode_only: bool,
    ) -> None:
        """Record the step for requests that add ``query_lengths[i]`` tokens, the last at key
        ``kv_lengths[i] - 1``; decode requests come first. With FlashInfer, also build the plan."""
        if len(query_lengths) != len(kv_lengths):
            raise ValueError("query_lengths and kv_lengths must have one entry per request.")
        spans = []
        query_start = 0
        for request, (query_length, kv_length) in enumerate(zip(query_lengths, kv_lengths)):
            if request >= num_decode_requests:
                spans.append(
                    DSAPrefillSpan(request, query_start, query_start + query_length, kv_length)
                )
            query_start += query_length
        self._step = DSAStepInfo(
            num_decode_requests=num_decode_requests,
            max_decode_query_length=max(query_lengths[:num_decode_requests], default=0),
            max_decode_kv_length=max(kv_lengths[:num_decode_requests], default=0),
            max_kv_length=max(kv_lengths, default=0),
            prefill_spans=tuple(spans),
        )
        if self._fi is None:
            return
        arrays = self._fi.build_step_plan(
            query_lengths,
            kv_lengths,
            padded_rows=padded_tokens,
            padded_requests=padded_requests,
            decode_only=decode_only,
            heads=self.query_heads,
        )
        for name, values in arrays.items():
            assert values.size <= self._capacity[name], name
            self._cpu_views[name][: values.size] = torch.from_numpy(values)
            self._lengths[name] = values.size
        self._sizes = self._fi.plan_sizes(padded_tokens, padded_requests, decode_only)
        self._page_rows = padded_requests

    def transfer(self, block_table: torch.Tensor, non_blocking: bool) -> None:
        """Copy the FlashInfer plan to the GPU and fill its page table from the step's GPU block
        table. The CuTe backend keeps everything on the host."""
        if self._fi is None:
            return
        self._gpu_buf.copy_(self._cpu_buf, non_blocking=non_blocking)
        self._fi.fill_page_table(block_table, self.block_size_tokens, self.page_table)

    def step(self) -> DSAStepInfo:
        """The step's host-side facts, recorded by ``update``."""
        return self._step

    def plan(self):
        """The step's ``dsa_gqa_flashinfer.StepPlan``; kernel shapes use the configured bound."""
        if self._fi is None:
            raise RuntimeError("The DSA step plan exists only for the FlashInfer backend.")
        return self._fi.StepPlan(
            decode_rows=self._sizes["decode_rows"],
            num_tiles=self._sizes["tiles"],
            max_context=self.max_sequence_length,
            page_table=self.page_table[: self._page_rows],
            **{name: view[: self._lengths[name]] for name, view in self._gpu_views.items()},
        )

    def reset(self) -> None:
        """Empty step: no requests, rows, tiles."""
        self._step = DSAStepInfo(
            num_decode_requests=0,
            max_decode_query_length=0,
            max_decode_kv_length=0,
            max_kv_length=0,
            prefill_spans=(),
        )
        if self._fi is None:
            return
        self._lengths = {name: 0 for name in self._capacity}
        self._sizes = {"decode_rows": 0, "tiles": 0}
        self._page_rows = 0
