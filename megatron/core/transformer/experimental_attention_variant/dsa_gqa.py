# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA over grouped-query attention with the simplified learned-Q/K indexer.

Each attention layer keeps its ordinary GQA projections. A one-head indexer projects
the fused-norm output of ``linear_qkv`` to a learned query and key; every query attends
to the top-k keys, by unscaled indexer dot product, among the causal keys up to and
including itself (all of them when there are at most k).

Dynamic-batching inference keeps the indexer keys in a paged cache that shares the
attention KV block table. Every query token of a step, decode or prefill, is a row that scores
its causal prefix's indexer keys, selects their top-k (DeepSelect when installed, else
``torch.topk``), and attends to them, all through the Triton kernels in
``dsa_gqa_inference_kernels``; so decode, prefill, and mixed steps share one path and can all
run as CUDA graphs. The uncached forward runs the training kernels through
``dsa_cute_kernels``.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from functools import cache, partial
from typing import Optional, Union

import torch
import torch.nn.functional as F

try:
    import deep_select

    HAVE_DEEP_SELECT = True
except ImportError:
    deep_select = None
    HAVE_DEEP_SELECT = False

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.fp8_utils import get_fp8_disabled_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedDSGQAIndexerSubmodules:
    """Projection submodules for the simplified indexer."""

    linear_q: Union[ModuleSpec, type]
    linear_k: Union[ModuleSpec, type]


class SimplifiedDSGQAIndexer(MegatronModule):
    """One-head learned Q/K indexer for the NoPE CuTe contract."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SimplifiedDSGQAIndexerSubmodules,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        self.hidden_size = config.hidden_size
        self.index_head_dim = config.dsa_indexer_head_dim
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection

        # These small BF16 projections are replicated across TP ranks and never quantized.
        indexer_config = copy.copy(config)
        indexer_config.fp8 = indexer_config.fp4 = None
        indexer_config.fp8_param = indexer_config.fp4_param = False
        indexer_config.quant_recipe = None
        indexer_config.params_dtype = torch.bfloat16
        with get_fp8_disabled_context(config, is_init=True):
            self.linear_q = build_module(
                submodules.linear_q,
                self.hidden_size,
                self.index_head_dim,
                config=indexer_config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )
            self.linear_k = build_module(
                submodules.linear_k,
                self.hidden_size,
                self.index_head_dim,
                config=indexer_config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                skip_weight_param_allocation=False,
                parallel_mode="duplicated",
            )

    def forward_qk(
        self, hidden_states: torch.Tensor, use_rope: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project normalized attention input ``[T,B,H]`` to indexer Q and K ``[T,B,1,Di]``."""
        if use_rope:
            raise NotImplementedError(
                "The simplified CuTe DSA indexer supports NoPE only; "
                "use position_embedding_type='none'."
            )
        if self.pg_collection.tp.size() != 1:
            raise NotImplementedError("The simplified CuTe DSA indexer requires TP1.")
        with get_fp8_disabled_context(self.config):
            q, _ = self.linear_q(hidden_states)
            k, _ = self.linear_k(hidden_states)
        return (
            q.reshape(q.size(0), q.size(1), 1, self.index_head_dim),
            k.reshape(k.size(0), k.size(1), 1, self.index_head_dim),
        )


@dataclass
class SimplifiedDSACoreAttentionSubmodules:
    """Submodules for the simplified DSA core attention."""

    indexer: Union[ModuleSpec, type]


class SimplifiedDSACoreAttention(MegatronModule):
    """Sparse core attention routed by the simplified indexer.

    Owns the indexer, so checkpoint keys are ``core_attention.indexer.linear_{q,k}.weight``.
    ``forward`` covers uncached causal forwards; dynamic-batching inference runs through
    ``DSGroupedSelfAttention._dynamic_core_attention_forward`` against the paged caches.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SimplifiedDSACoreAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: Optional[str] = None,
        softmax_scale: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        super().__init__(config=config)
        from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels

        del cp_comm_type
        if attention_type != "self":
            raise NotImplementedError("Simplified DSA supports self-attention only.")
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.index_topk = config.dsa_indexer_topk
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(config.kv_channels)
        expected_softmax_scale = dsa_cute_kernels.HEAD_DIM**-0.5
        if not math.isclose(softmax_scale, expected_softmax_scale, rel_tol=0.0, abs_tol=1.0e-12):
            raise NotImplementedError(
                "The CuTe DSA kernels hard-code attention softmax scale "
                f"{expected_softmax_scale}, got {softmax_scale}."
            )
        self.softmax_scale = softmax_scale
        self.indexer = build_module(submodules.indexer, config=config, pg_collection=pg_collection)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        x: torch.Tensor,
        qr: torch.Tensor,
        use_indexer_rope: bool,
    ) -> torch.Tensor:
        """Uncached causal sparse attention over one sequence ``[T,1,H,D] -> [T,1,H*D]``."""
        from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels

        del attention_mask, qr
        if torch.is_grad_enabled():
            raise NotImplementedError("Simplified DSA core attention is inference-only here.")
        if packed_seq_params is not None:
            raise NotImplementedError("Simplified DSA uncached forward does not take packed input.")
        if attention_bias is not None:
            raise NotImplementedError("Simplified DSA does not support attention bias.")
        mask_type = self.attn_mask_type if attn_mask_type is None else attn_mask_type
        if mask_type != AttnMaskType.causal:
            raise NotImplementedError("Simplified DSA supports causal attention only.")
        if query.size(1) != 1:
            raise NotImplementedError("Simplified DSA uncached forward requires batch size 1.")
        index_query, index_key = self.indexer.forward_qk(x, use_rope=use_indexer_rope)
        output = dsa_cute_kernels.run_sparse_attention_inference(
            query=query[:, 0],
            key=key[:, 0],
            value=value[:, 0],
            q_indexer=index_query[:, 0, 0],
            k_indexer=index_key[:, 0, 0],
            key_offset=0,
            indexer_topk=self.index_topk,
        )
        return output.reshape(output.size(0), 1, -1)


class DSGroupedSelfAttention(SelfAttention):
    """GQA self-attention whose core attention is routed by the simplified DSA indexer."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: Optional[int] = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        if config.experimental_attention_variant != "dsa" or config.dsa_min_memory_backend != (
            "cute"
        ):
            raise ValueError(
                "DSGroupedSelfAttention requires experimental_attention_variant='dsa' and "
                "dsa_min_memory_backend='cute'."
            )
        submodules = copy.copy(submodules)
        submodules.linear_qkv = partial(submodules.linear_qkv, return_layernorm_output=True)
        submodules.core_attention = partial(
            SimplifiedDSACoreAttention,
            submodules=SimplifiedDSACoreAttentionSubmodules(
                indexer=ModuleSpec(
                    module=SimplifiedDSGQAIndexer,
                    submodules=SimplifiedDSGQAIndexerSubmodules(
                        linear_q=TELinear, linear_k=TELinear
                    ),
                )
            ),
        )
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )

    def _needs_qkv_layernorm_output(self) -> bool:
        return True

    def forward_pre_attn_and_core_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        rotary_pos_cos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        packed_sequence_cp_metadata=None,
    ) -> torch.Tensor:
        """Reject static-batching caches; dynamic batching uses the paged sparse path."""
        context = inference_context if inference_context is not None else inference_params
        if context is not None and not context.is_dynamic_batching():
            raise NotImplementedError(
                "Simplified CuTe DSA supports decode caching with dynamic batching only."
            )
        return super().forward_pre_attn_and_core_attn(
            hidden_states,
            attention_mask,
            key_value_states=key_value_states,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            inference_params=inference_params,
            packed_sequence_cp_metadata=packed_sequence_cp_metadata,
        )

    def _get_core_attention_extra_kwargs(
        self,
        *,
        hidden_states: torch.Tensor,
        qkv_layernorm_output: Optional[torch.Tensor],
        rotary_pos_emb,
        rotary_pos_cos: Optional[torch.Tensor],
        rotary_pos_sin: Optional[torch.Tensor],
        rotary_pos_cos_sin: Optional[torch.Tensor],
    ) -> dict[str, object]:
        del hidden_states
        if qkv_layernorm_output is None:
            raise RuntimeError(
                "CuTe DSA requested the fused QKV norm output, but none was returned."
            )
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        use_indexer_rope = not no_rope and any(
            embedding is not None
            for embedding in (rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, rotary_pos_cos_sin)
        )
        return {
            "x": qkv_layernorm_output,
            "qr": qkv_layernorm_output,
            "use_indexer_rope": use_indexer_rope,
        }

    def _dynamic_core_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        inference_context: BaseInferenceContext,
        block_table: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
        *,
        x: torch.Tensor,
        qr: torch.Tensor,
        use_indexer_rope: bool,
    ) -> torch.Tensor:
        """Run sparse attention for a dynamic batch against the paged KV and indexer caches.

        Every token, decode or prefill, routes to the top-k of its causal prefix through the
        Triton inference kernels, which covers fresh prompts, chunked prefill, and mixed
        batches, eager or CUDA-graphed.
        """
        del attention_mask, attn_mask_type, packed_seq_params, qr
        if attention_bias is not None:
            raise NotImplementedError("Simplified DSA does not support attention bias.")
        layer_number = self.layer_number - self._get_pp_layer_offset_for_inference()
        with torch.no_grad():
            index_query, index_key = self.core_attention.indexer.forward_qk(
                x, use_rope=use_indexer_rope
            )
        inference_context.append_dsa_key_cache(layer_number, index_key)
        index_key_cache, _ = inference_context.dsa_key_cache(layer_number)
        return dynamic_sparse_attention(
            query=query,
            key_cache=key,
            value_cache=value,
            index_query=index_query,
            index_key_cache=index_key_cache,
            block_table=block_table,
            inference_context=inference_context,
            index_topk=self.core_attention.index_topk,
            softmax_scale=self.core_attention.softmax_scale,
            backend=self.config.dsa_inference_backend,
        )


# Bound on the indexer score buffer of one step: rows are scored and selected in windows that
# fit it (2048 prefill rows at 1M context would otherwise need 8 GiB).
SCORE_BUFFER_BYTES = 1 << 30


@cache
def _log_decode_topk_backend(use_deep_select: bool) -> None:
    """Say once per backend which top-k decode uses, since the fallback is much slower."""
    if use_deep_select:
        logger.info("DSA decode selects its top-k with DeepSelect.")
    else:
        logger.warning(
            "DSA decode selects its top-k with torch.topk over the whole score buffer; "
            "install DeepSelect (SM100/SM103, top-k <= 4096) for decode cost that follows "
            "the real context lengths."
        )


@cache
def _log_fused_topk() -> None:
    logger.info("DSA selects its top-k with the fused selection (no score buffer).")


def _deep_select_decode_supported(device: torch.device, index_topk: int) -> bool:
    """DeepSelect's CUDA build has SM100a/SM103a kernels for top-k up to 4096."""
    return (
        HAVE_DEEP_SELECT
        and 0 < index_topk <= 4096
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    )


def _select_topk(
    scores: torch.Tensor, row_lengths: torch.Tensor, index_topk: int, use_deep_select: bool
) -> torch.Tensor:
    """Each row's top-k positions among its first ``row_lengths[t]`` scores, -1 padded."""
    if use_deep_select:
        _, topk_indices = deep_select.topk(
            scores,
            index_topk,
            end=row_lengths,
            indices_type=torch.int32,
            return_value=False,
            idx_oob_fill_value=-1,
        )
        return topk_indices
    width = scores.size(1)
    topk_indices = scores.topk(min(index_topk, width), dim=-1).indices.to(torch.int32)
    return F.pad(topk_indices, (0, index_topk - topk_indices.size(1)), value=-1)


@torch.no_grad()
def _sparse_attention_rows(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_metadata: tuple,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
    filter_blocks: Optional[tuple] = None,
    decode_only: bool = False,
) -> torch.Tensor:
    """Route every row to its top-k cached keys and attend to them.

    ``row_metadata`` is ``(row_requests, row_lengths, block_starts, block_rows,
    block_requests, num_blocks)`` (``DynamicInferenceContext.dsa_row_metadata``), and
    ``max_context_length`` bounds every row length. DeepSelect reads each score row only up
    to its length, so the indexer skips the rest and the cost follows the real lengths;
    ``torch.topk`` reads whole rows, so the indexer fills positions past each length with
    negative infinity.

    Given ``filter_blocks`` (``DynamicInferenceContext.dsa_filter_blocks``), contexts bounded
    by at least ``FUSED_MIN_CONTEXT`` keys take the fused selection of
    ``dsa_gqa_fused_select`` where the device supports it, which never materializes the
    scores; ``decode_only`` says every row is its request's only query token this step.

    Returns:
        Attention output ``[rows,Hq,Dv]``.
    """
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_gqa_fused_select,
        dsa_gqa_inference_kernels,
    )

    row_requests, row_lengths, block_starts, block_rows, block_requests, _ = row_metadata
    num_rows, num_query_heads, _ = query.shape
    block_table = block_table.to(torch.int32).contiguous()
    index_query = index_query.contiguous()
    use_fused = (
        filter_blocks is not None
        and max_context_length >= dsa_gqa_fused_select.FUSED_MIN_CONTEXT
        and dsa_gqa_fused_select.fused_select_supported(
            query.device, index_topk, index_query.size(-1), index_key_cache.size(1)
        )
    )
    if use_fused:
        _log_fused_topk()
        topk_indices = dsa_gqa_fused_select.fused_select_topk(
            index_query,
            index_key_cache,
            block_table,
            row_requests,
            row_lengths,
            (block_starts, block_rows, block_requests),
            filter_blocks,
            width=max_context_length,
            index_topk=index_topk,
            decode_only=decode_only,
        )
    else:
        topk_indices = _materialized_topk(
            index_query=index_query,
            index_key_cache=index_key_cache,
            block_table=block_table,
            row_metadata=row_metadata,
            max_context_length=max_context_length,
            index_topk=index_topk,
        )
    output = query.new_empty((num_rows, num_query_heads, value_cache.size(-1)))
    dsa_gqa_inference_kernels.dsa_gqa_sparse_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        row_requests,
        row_lengths,
        topk_indices,
        softmax_scale,
        out=output,
    )
    return output


def _materialized_topk(
    *,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    row_metadata: tuple,
    max_context_length: int,
    index_topk: int,
) -> torch.Tensor:
    """Each row's top-k by scoring its keys into a buffer and selecting from it."""
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_inference_kernels

    row_requests, row_lengths, block_starts, block_rows, block_requests, _ = row_metadata
    num_rows = row_requests.numel()
    use_deep_select = _deep_select_decode_supported(index_query.device, index_topk)
    _log_decode_topk_backend(use_deep_select)
    width = max_context_length
    if use_deep_select:
        # DeepSelect needs row strides aligned to its requirement and at least top-k
        # columns; the indexer never writes past a row's length.
        alignment = deep_select.get_stride_requirement()[0] // torch.float32.itemsize
        width = -(-max(width, index_topk) // alignment) * alignment
    # Score and select in windows of rows so the score buffer stays bounded at long context;
    # the window count depends only on the padded rows and the width (CUDA-graph safe).
    window = max(1, min(num_rows, SCORE_BUFFER_BYTES // (width * torch.float32.itemsize)))
    scores = torch.empty((window, width), dtype=torch.float32, device=index_query.device)
    topk_indices = torch.empty((num_rows, index_topk), dtype=torch.int32, device=index_query.device)
    for start in range(0, num_rows, window):
        stop = min(start + window, num_rows)
        window_scores = scores[: stop - start]
        dsa_gqa_inference_kernels.dsa_gqa_indexer_scores(
            index_query,
            index_key_cache,
            block_table,
            row_lengths,
            block_starts,
            block_rows,
            block_requests,
            out=window_scores,
            fill_invalid=not use_deep_select,
            row_offset=start,
        )
        topk_indices[start:stop] = _select_topk(
            window_scores, row_lengths[start:stop], index_topk, use_deep_select
        )
    return topk_indices


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
    backend: str = "triton",
) -> torch.Tensor:
    """Simplified-DSA attention over paged caches for a decode, prefill, or mixed batch.

    Every token is a row that routes to the top-k of its causal prefix
    (``DynamicInferenceContext.dsa_row_metadata``), so all step types share one path with
    launch sizes fixed by the padded batch and can run as CUDA graphs. The caches already
    hold this step's keys, values, and indexer keys.

    Args:
        query: Query ``[T,1,Hq,D]`` or ``[T,Hq,D]`` for the padded active tokens.
        key_cache: Paged keys ``[blocks,block_size,G,D]``.
        value_cache: Paged values ``[blocks,block_size,G,Dv]``.
        index_query: Indexer queries ``[T,1,1,Di]``.
        index_key_cache: Paged indexer keys ``[blocks,block_size,Di]``.
        block_table: Per-request physical block ids ``[requests,max_blocks]``.
        inference_context: Dynamic inference context for this step.
        index_topk: Number of keys each query attends to.
        softmax_scale: Attention softmax scale.
        backend: ``triton`` (Megatron's kernels) or ``flashinfer`` (``dsa_gqa_flashinfer``).

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows for padding tokens are zero.
    """
    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "DSA inference decodes one token per request; MTP speculative decoding is not "
            "supported."
        )
    if query.dim() == 3:
        query = query.unsqueeze(1)
    num_tokens, _, num_query_heads, _ = query.shape
    row_metadata = inference_context.dsa_row_metadata()
    if row_metadata[0].numel() != num_tokens:
        raise RuntimeError(
            f"DSA row metadata covers {row_metadata[0].numel()} tokens, but the step has "
            f"{num_tokens}."
        )
    # Graphed steps use the graph's static bound; eager steps the batch's longest context.
    max_context_length = inference_context.active_attn_metadata["mha_metadata"].state_data[
        "max_seqlen_k"
    ]
    if backend == "flashinfer":
        from megatron.core.transformer.experimental_attention_variant import dsa_gqa_flashinfer

        if not dsa_gqa_flashinfer.flashinfer_dsa_supported(
            query.device, index_topk, index_query.size(-1), key_cache.size(1), num_query_heads
        ):
            raise RuntimeError(
                "dsa_inference_backend='flashinfer' needs FlashInfer with DSA kernels, CuTe DSL "
                "4.7+, FA4, DeepSelect and Triton on SM100/SM103, top-k 512/1024/2048, indexer "
                "dim 128, and a KV block size that is a multiple of 64."
            )
        output = dsa_gqa_flashinfer.flashinfer_sparse_attention(
            query=query[:, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query.reshape(num_tokens, -1),
            index_key_cache=index_key_cache,
            block_table=block_table.to(torch.int32),
            # The configured bound, eager or graphed: FlashInfer compiles per launch shape.
            plan=inference_context.dsa_flashinfer_plan(inference_context.max_sequence_length),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        )
        return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))
    output = _sparse_attention_rows(
        query=query[:, 0],
        key_cache=key_cache,
        value_cache=value_cache,
        index_query=index_query.reshape(num_tokens, -1),
        index_key_cache=index_key_cache,
        block_table=block_table,
        row_metadata=row_metadata,
        max_context_length=max_context_length,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        filter_blocks=inference_context.dsa_filter_blocks(),
        decode_only=inference_context.is_decode_only(),
    )
    return output.view(num_tokens, 1, num_query_heads * value_cache.size(-1))


__all__ = [
    "DSGroupedSelfAttention",
    "SimplifiedDSACoreAttention",
    "SimplifiedDSGQAIndexer",
    "dynamic_sparse_attention",
]
