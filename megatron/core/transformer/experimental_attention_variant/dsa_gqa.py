# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA over grouped-query attention with the simplified learned-Q/K indexer.

Each attention layer keeps its ordinary GQA projections. A one-head indexer projects
the fused-norm output of ``linear_qkv`` to a learned query and key; every query attends
to the top-k keys, by unscaled indexer dot product, among the causal keys up to and
including itself (all of them when there are at most k).

Dynamic-batching inference keeps the indexer keys in a paged cache that shares the
attention KV block table. Decode runs the paged CuTe kernels in
``dsa_gqa_decode_kernel``; prefill runs the training kernels through
``dsa_cute_kernels``.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Optional, Union

import torch
import torch.nn.functional as F

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
        self.indexer = build_module(
            submodules.indexer, config=config, pg_collection=pg_collection
        )

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
            raise RuntimeError("CuTe DSA requested the fused QKV norm output, but none was returned.")
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

        Decode requests use the paged CuTe decode kernels. Each prefill request runs the
        training CuTe sparse kernel over its cached prefix, which covers fresh prompts and
        chunked prefill.
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
        )


def _paged_cache_rows(
    cache: torch.Tensor, block_table_row: torch.Tensor, num_tokens: int, block_size_tokens: int
) -> torch.Tensor:
    """Gather logical positions ``[0, num_tokens)`` from one request's paged cache."""
    positions = torch.arange(num_tokens, device=cache.device)
    physical_blocks = block_table_row[positions // block_size_tokens].long()
    return cache[physical_blocks, positions % block_size_tokens]


@torch.no_grad()
def _sparse_decode(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    index_query: torch.Tensor,
    index_key_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lengths: torch.Tensor,
    max_context_length: int,
    index_topk: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Route each single-token decode query to its top-k cached keys and attend to them.

    ``max_context_length`` must bound every entry of ``kv_lengths``; scores past a
    request's length are negative infinity and their indices are masked by the kernel.

    Returns:
        Attention output ``[requests,Hq,Dv]``.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_gqa_decode_kernel

    num_requests, num_query_heads, _ = query.shape
    kv_lengths = kv_lengths.to(torch.int32).contiguous()
    block_table = block_table.to(torch.int32).contiguous()
    scores = torch.empty(
        (num_requests, max_context_length), dtype=torch.float32, device=query.device
    )
    dsa_gqa_decode_kernel.dsa_gqa_decode_indexer_score(
        index_query.contiguous(), index_key_cache, block_table, kv_lengths, out=scores
    )
    topk_indices = scores.topk(min(index_topk, max_context_length), dim=-1).indices
    # A fixed top-k width keeps the attention kernel's shared-memory layout constant.
    topk_indices = F.pad(topk_indices, (0, index_topk - topk_indices.size(1)), value=-1)
    output = query.new_empty((num_requests, num_query_heads, value_cache.size(-1)))
    dsa_gqa_decode_kernel.dsa_gqa_decode_attention(
        query.contiguous(),
        key_cache,
        value_cache,
        block_table,
        topk_indices.to(torch.int32).contiguous(),
        kv_lengths,
        softmax_scale,
        out=output,
    )
    return output


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
    """Simplified-DSA attention over paged caches for a mixed decode/prefill batch.

    The dynamic context orders decode requests before prefill requests, and its
    caches already hold this step's keys, values, and indexer keys.

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

    Returns:
        Attention output ``[T,1,Hq*Dv]``; rows for padding tokens are zero.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa_cute_kernels

    if inference_context.num_speculative_tokens:
        raise NotImplementedError(
            "Simplified CuTe DSA decodes one token per request; "
            "MTP speculative decoding is not supported."
        )
    if query.dim() == 3:
        query = query.unsqueeze(1)
    num_tokens, _, num_query_heads, _ = query.shape
    value_head_dim = value_cache.size(-1)
    output = query.new_zeros((num_tokens, 1, num_query_heads * value_head_dim))
    state_data = inference_context.active_attn_metadata["mha_metadata"].state_data
    index_query = index_query.reshape(num_tokens, -1)

    if inference_context.using_cuda_graph_this_step():
        # A captured graph replays on later steps, so use only sizes fixed by the graph's
        # padded batch and never read per-step metadata on the host. Padded requests have
        # zero KV length and produce zero output.
        if not inference_context.is_decode_only():
            raise NotImplementedError(
                "Simplified CuTe DSA supports decode-only CUDA graphs; "
                "use --decode-only-cuda-graphs."
            )
        num_decode = inference_context.padded_batch_dimensions.decode_req_count
        output[:num_decode, 0] = _sparse_decode(
            query=query[:num_decode, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query[:num_decode],
            index_key_cache=index_key_cache,
            block_table=block_table[:num_decode],
            kv_lengths=state_data["kv_seq_lengths"][:num_decode],
            max_context_length=state_data["max_seqlen_k"],
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        ).flatten(1)
        return output

    request_count = inference_context.get_active_request_count()
    if request_count == 0:
        return output
    query_lengths = state_data["query_lengths"][:request_count].tolist()
    kv_lengths = state_data["kv_seq_lengths"][:request_count]

    num_decode = inference_context.num_decode_requests
    if num_decode:
        if any(length != 1 for length in query_lengths[:num_decode]):
            raise NotImplementedError("CuTe DSA decode supports one query token per request.")
        output[:num_decode, 0] = _sparse_decode(
            query=query[:num_decode, 0],
            key_cache=key_cache,
            value_cache=value_cache,
            index_query=index_query[:num_decode],
            index_key_cache=index_key_cache,
            block_table=block_table[:num_decode],
            kv_lengths=kv_lengths[:num_decode],
            max_context_length=int(kv_lengths[:num_decode].max().item()),
            index_topk=index_topk,
            softmax_scale=softmax_scale,
        ).flatten(1)

    query_start = num_decode
    block_size_tokens = inference_context.block_size_tokens
    for request_idx in range(num_decode, request_count):
        query_length = query_lengths[request_idx]
        kv_length = int(kv_lengths[request_idx].item())
        query_end = query_start + query_length
        request_blocks = block_table[request_idx]
        request_output = dsa_cute_kernels.run_sparse_attention_inference(
            query=query[query_start:query_end, 0],
            key=_paged_cache_rows(key_cache, request_blocks, kv_length, block_size_tokens),
            value=_paged_cache_rows(value_cache, request_blocks, kv_length, block_size_tokens),
            q_indexer=index_query[query_start:query_end],
            k_indexer=_paged_cache_rows(
                index_key_cache, request_blocks, kv_length, block_size_tokens
            ),
            key_offset=kv_length - query_length,
            indexer_topk=index_topk,
        )
        output[query_start:query_end, 0] = request_output.flatten(1)
        query_start = query_end
    return output


__all__ = [
    "DSGroupedSelfAttention",
    "SimplifiedDSACoreAttention",
    "SimplifiedDSGQAIndexer",
    "dynamic_sparse_attention",
]
