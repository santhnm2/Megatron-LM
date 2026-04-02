# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import math
import warnings
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_attr_wrapped_model, get_model_config

if TYPE_CHECKING:
    from megatron.core.transformer import TransformerConfig


def compute_kv_block_size_bytes(
    *,
    kv_dtype_size: int,
    num_attention_layers: int,
    block_size_tokens: int,
    cache_mla_latent: bool,
    kv_reduced_dim: int = 0,
    heads_per_partition: int = 0,
    head_size: int = 0,
) -> int:
    """Compute the byte size of a single KV cache block.

    This is the single source of truth used by both ``InferenceConfig``
    and ``DynamicInferenceContext``.
    """
    if cache_mla_latent:
        return kv_dtype_size * num_attention_layers * block_size_tokens * kv_reduced_dim
    return (
        kv_dtype_size
        * 2  # key + value
        * num_attention_layers
        * block_size_tokens
        * heads_per_partition
        * head_size
    )


def compute_mamba_memory_per_request(
    mamba_config: "MambaInferenceStateConfig",
    num_mamba_layers: int,
    num_speculative_tokens: int = 0,
) -> int:
    """Compute the per-request Mamba state memory in bytes.

    This is the single source of truth used by both ``InferenceConfig``
    and ``DynamicInferenceContext``.
    """
    if num_mamba_layers == 0:
        return 0

    conv_bytes = math.prod(mamba_config.conv_states_shape) * mamba_config.conv_states_dtype.itemsize
    ssm_bytes = math.prod(mamba_config.ssm_states_shape) * mamba_config.ssm_states_dtype.itemsize
    per_layer = conv_bytes + ssm_bytes

    total = per_layer * num_mamba_layers
    if num_speculative_tokens > 0:
        total += per_layer * num_mamba_layers * (num_speculative_tokens + 1)
    return total


def measure_peak_activation_memory(model: MegatronModule, max_tokens: int, tp_size: int = 1) -> int:
    """Measure peak transient activation memory by running a dummy forward pass.

    Runs the model with ``max_tokens`` tokens (no inference context, no KV cache)
    and returns the peak GPU memory increase observed during the forward pass.

    Args:
        model: The model, already on GPU and in eval mode.
        max_tokens: Number of tokens in the dummy forward pass.
        tp_size: Tensor-parallel size. The token count is rounded up to a
            multiple of ``tp_size`` for sequence-parallel compatibility.

    Returns:
        Peak activation memory in bytes.
    """
    num_tokens = max(tp_size, max_tokens)
    num_tokens = ((num_tokens + tp_size - 1) // tp_size) * tp_size

    device = torch.cuda.current_device()
    tokens = torch.zeros((1, num_tokens), dtype=torch.long, device=device)
    position_ids = torch.zeros((1, num_tokens), dtype=torch.long, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    with torch.inference_mode():
        model(tokens, position_ids, attention_mask=None)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    del tokens, position_ids
    torch.cuda.empty_cache()

    return max(0, peak - baseline)


@dataclass
class MambaInferenceStateConfig:
    """
    Config for initializing Mamba model inference state tensors.

    Note that we maintain separate metadata for decode, regular prefill, and
    chunked prefill requests because the Mamba kernels do not yet support mixing
    these. Once the kernels have been updated we can simplify this code.
    """

    layer_type_list: List[str]
    """
    A list of strings that indicates the layer type (Mamba / Attention / MLP) for each layer.
    See `megatron/core/ssm/mamba_hybrid_layer_allocation.py` for the list of symbols.
    """

    conv_states_shape: Tuple[int]
    """Mamba conv states shape per request."""

    ssm_states_shape: Tuple[int]
    """Mamba SSM states shape per request."""

    conv_states_dtype: torch.dtype
    """The dtype to use for the Mamba conv state tensor. Defaults to the model dtype."""

    ssm_states_dtype: torch.dtype
    """The dtype to use for the Mamba SSM state tensor. Defaults to the model dtype."""

    mamba_chunk_size: int = 128
    """The chunk size used by the Mamba SSM Triton kernels."""

    @classmethod
    def from_model(
        cls,
        model: MegatronModule,
        conv_states_dtype: Optional[torch.dtype] = None,
        ssm_states_dtype: Optional[torch.dtype] = None,
    ) -> Optional["MambaInferenceStateConfig"]:
        """Returns Mamba inference state config from the model if it is a hybrid model."""
        from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols

        decoder = get_attr_wrapped_model(model, "decoder")
        layer_type_list = getattr(decoder, "layer_type_list", None)
        if layer_type_list is not None and Symbols.MAMBA in layer_type_list:
            (mamba_conv_states_shape, mamba_ssm_states_shape) = (
                decoder.mamba_state_shapes_per_request()
            )
            if conv_states_dtype is None:
                conv_states_dtype = model.config.params_dtype
            if ssm_states_dtype is None:
                ssm_states_dtype = model.config.params_dtype
            mamba_chunk_size = 128
            for layer_type, layer in zip(decoder.layer_type_list, decoder.layers):
                if layer_type == Symbols.MAMBA and hasattr(layer, 'mixer'):
                    mamba_chunk_size = layer.mixer.chunk_size
                    break
            return cls(
                layer_type_list=layer_type_list,
                conv_states_shape=mamba_conv_states_shape,
                ssm_states_shape=mamba_ssm_states_shape,
                conv_states_dtype=conv_states_dtype,
                ssm_states_dtype=ssm_states_dtype,
                mamba_chunk_size=mamba_chunk_size,
            )
        return None


class PrefixCachingEvictionPolicy(str, Enum):
    """Eviction policy for prefix caching blocks.

    Only applies when enable_prefix_caching is True.
    """

    REF_ZERO = "ref_zero"
    """Deregister blocks immediately when ref_count hits 0. No caching after release."""

    LRU = "lru"
    """Keep released blocks in hash table. Evict oldest ref=0 blocks when space is needed."""


class PrefixCachingCoordinatorPolicy(str, Enum):
    """Routing policy for the DP inference coordinator with prefix caching."""

    LONGEST_PREFIX = "longest_prefix"
    """Route to the rank with the longest consecutive prefix match."""

    FIRST_PREFIX_BLOCK = "first_prefix_block"
    """Route to the rank that has the first block hash cached. O(ranks) check."""

    ROUND_ROBIN = "round_robin"
    """Route requests to ranks in round-robin order, ignoring prefix affinity."""


class KVCacheManagementMode(str, Enum):
    """Mode for handling large tensors (KV cache, Mamba states) during suspend/resume."""

    PERSIST = "persist"
    """Do not deallocate and reallocate large tensors; keep them on GPU."""

    OFFLOAD = "offload"
    """Offload large tensors to CPU during deallocation; onload during allocation."""

    RECOMPUTE = "recompute"
    """Deallocate large tensors and recompute them from scratch during allocation."""


@dataclass
class InferenceConfig:
    """
    Config for inference.

    When ``model_config`` is provided, fields left as ``None`` are auto-computed
    from the model architecture and available GPU memory.  Fields set explicitly
    are used as-is (an exception is raised if the constraints are impossible to
    satisfy).

    NOTE: Must remain mutually exclusive with the `TransformerConfig`.
    """

    # =================================
    # KV cache and Mamba states config
    # =================================
    block_size_tokens: int = 256
    """Size of KV cache block size."""

    buffer_size_gb: Optional[float] = None
    """
    Buffer size reserved on the GPU for the KV cache.
    Auto-computed from GPU memory when ``model_config`` is provided and this is
    ``None``.  Must be set explicitly when ``model_config`` is not provided.
    If `unified_memory_level` >= 1, then CPU memory is additionally utilized, resulting in a total
    buffer size of `buffer_size_gb + paused_buffer_size_gb`.
    """

    paused_buffer_size_gb: Optional[int] = None
    """
    Portion of buffer reserved for paused requests. Active requests are paused when there are not
    enough active blocks available to continue generating a request. The total buffer size
    (active + paused) depends on `unified_memory_level` (uvm):
        - uvm 0: buffer_size_gb (paused buffer is inclusive)
        - uvm 1: buffer_size_gb + paused_buffer_size_gb
    """

    mamba_inference_state_config: Optional[MambaInferenceStateConfig] = None
    """The Mamba inference state config if the model is a hybrid model."""

    mamba_memory_ratio: Optional[float] = None
    """
    Percentage of memory buffer to allocate for Mamba states.  Auto-computed as
    the natural proportion of Mamba-to-total per-request memory when
    ``model_config`` is provided and this is ``None``.  Only used for hybrid
    models.
    """

    max_requests: Optional[int] = None
    """
    Max number of active requests to use for decode-only forward passes.
    Auto-computed to maximise throughput within the memory budget when
    ``model_config`` is provided and this is ``None``.
    """

    max_tokens: int = 16384
    """
    Max number of tokens to use for forward passes. This is primarily limited by prefill activation
    memory usage.
    """

    unified_memory_level: int = 0
    """
    Sets unified memory usage within the dynamic inference context.
    The levels are:
        0) no unified memory (default)
        1) allocate `memory_buffer` in unified memory.
    Eventually, additional levels will be included to control other tensors within the context.
    """

    kv_cache_management_mode: KVCacheManagementMode = KVCacheManagementMode.PERSIST
    """
    Mode used to determine how large tensors are handled by the allocate and deallocate methods.
    See `KVCacheManagementMode` for options.
    """

    # =================================
    # CUDA graph config
    # =================================
    num_cuda_graphs: Optional[int] = None
    """
    Maximum number of cuda graphs to capture, where the cuda graph batch sizes range from 1 to
    `max_requests`. Due to rounding, the actual number of cuda graphs may not equal this argument.
    """

    cuda_graph_mixed_prefill_count: Optional[int] = 16
    """
    The number of mixed prefill graphs to capture if mixed prefill/decode graphs are enabled.
    """

    use_cuda_graphs_for_non_decode_steps: bool = True
    """
    Whether to use CUDA graphs for non-decode steps.
    """

    static_kv_memory_pointers: bool = False
    """
    Whether the KV cache (and Mamba states) will reside at the same memory addresses
    after suspend/resume as before. When True, CUDA graphs that reference these buffers
    remain valid across suspend/resume cycles and do not need to be recaptured.
    Requires either UVM or `torch_memory_saver` when `kv_cache_management_mode` is not PERSIST.
    """

    # =================================
    # Model config
    # =================================
    max_sequence_length: int = 2560
    """Max possible sequence length (prompt + output) that will occur."""

    pg_collection: Optional[ProcessGroupCollection] = None
    """A `ProcessGroupCollection` for distributed execution."""

    use_flashinfer_fused_rope: Optional[bool] = False
    """
    If True, use flashinfer's fused rope implementation.
    If None, defaults to using flash-infer if available.
    """

    materialize_only_last_token_logits: bool = True
    """
    Whether to only materialize logits for the last token. This should be set to False
    if returning log probs.
    """

    # =================================
    # Engine config
    # =================================
    enable_chunked_prefill: bool = False
    """Whether to enable chunked prefill."""

    num_speculative_tokens: int = 0
    """The number of speculative tokens to generate for decode steps."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for KV cache block sharing."""

    prefix_caching_eviction_policy: PrefixCachingEvictionPolicy = (
        PrefixCachingEvictionPolicy.REF_ZERO
    )
    """Eviction policy for prefix caching blocks. See `PrefixCachingEvictionPolicy` for options.

    Only applies when enable_prefix_caching is True.
    """

    prefix_caching_coordinator_policy: PrefixCachingCoordinatorPolicy = (
        PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    )
    """Routing policy for the DP inference coordinator. See
    `PrefixCachingCoordinatorPolicy` for options.

    Only applies when enable_prefix_caching is True and using a coordinator.
    """

    prefix_caching_routing_alpha: float = 0.5
    """Weight for prefix-aware scoring: score = alpha * match + (1 - alpha) * normalized_load.
    Higher alpha favors prefix cache hits; lower alpha favors load balance.
    Must be in [0, 1]. Only applies when enable_prefix_caching is True and using a coordinator.
    """

    prefix_caching_mamba_gb: Optional[float] = None
    """GPU memory budget (in GB) for the Mamba state cache used by prefix caching
    on hybrid models. Each cache slot stores SSM and conv states for all Mamba layers
    at a single block boundary. When set, Mamba states at KV divergence and last-aligned
    block boundaries are cached and reused across requests with matching prefixes."""

    # =================================
    # Logging config
    # =================================
    track_paused_request_events: bool = False
    """
    Whether to track paused request events. If True, `add_event_pause()` is called on
    requests when they are paused during bookkeeping.
    """

    track_generated_token_events: bool = False
    """
    Whether to track per-token events with timestamps for each generated token.
    When enabled, each generated token creates a GENERATED_TOKEN event with a
    timestamp, useful for per-token latency analysis.
    """

    metrics_writer: Optional["WandbModule"] = None
    """Wandb module for writing metrics."""

    logging_step_interval: int = 0
    """
    The step interval at which to log inference metrics to wandb.
    Defaults to 0, which means no logging.
    """

    request_metadata_types: Optional[List[Tuple[str, torch.dtype, bool]]] = None
    """
    A list of the per-request metadata types to track. Each entry is a tuple
    consisting of the string label, the target dtype, and whether to store the data on GPU.
    """

    use_synchronous_zmq_collectives: bool = False
    """Whether to use synchronous ZMQ collectives for inference. If True, the
    all_reduce_max operation will be performed synchronously, which can help reduce
    performance variability for MoEs.
    """

    # =================================
    # Auto-configuration inputs (not stored on the instance)
    # =================================
    model_config: InitVar[Optional["TransformerConfig"]] = None
    """When provided, ``None``-valued fields (``buffer_size_gb``,
    ``mamba_memory_ratio``, ``max_requests``) are auto-computed from
    the model architecture and available GPU memory."""

    model: InitVar[Optional[MegatronModule]] = None
    """Optional model instance (on GPU, eval mode).  When provided alongside
    ``model_config``, a dummy forward pass measures peak activation memory so
    the KV-cache buffer fills remaining GPU memory exactly."""

    gpu_memory_fraction: InitVar[float] = 0.90
    """Fraction of free GPU memory to use for the KV-cache buffer.  Only used
    when ``model`` is *not* provided."""

    gpu_memory_budget_gb: InitVar[Optional[float]] = None
    """Explicit GPU memory budget in GB.  Overrides both GPU probing and
    activation measurement.  Useful for offline planning."""

    def __post_init__(self, model, gpu_memory_fraction, gpu_memory_budget_gb):
        self.model_config = get_model_config(model)

        # --- static validation ---
        if not (0.0 <= self.prefix_caching_routing_alpha <= 1.0):
            raise ValueError(
                f"prefix_caching_routing_alpha must be in [0, 1], "
                f"got {self.prefix_caching_routing_alpha}"
            )

        if model_config is None:
            # Legacy path: no auto-computation, require buffer_size_gb.
            if self.buffer_size_gb is None:
                raise ValueError(
                    "buffer_size_gb must be set explicitly when model_config " "is not provided."
                )
            self._log_config()
            return

        # --- auto-configuration from model_config ---
        from megatron.core.transformer import MLATransformerConfig

        is_mla = isinstance(model_config, MLATransformerConfig) and model_config.cache_mla_latents
        kv_dtype_size = model_config.params_dtype.itemsize
        num_kv_heads = model_config.num_query_groups or model_config.num_attention_heads
        tp_size = model_config.tensor_model_parallel_size
        pp_size = model_config.pipeline_model_parallel_size

        # Layer counts.
        if self.mamba_inference_state_config is not None:
            from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols

            ltl = self.mamba_inference_state_config.layer_type_list
            num_attn_layers = max(1, sum(1 for lt in ltl if lt == Symbols.ATTENTION) // pp_size)
            num_mamba_layers = max(0, sum(1 for lt in ltl if lt == Symbols.MAMBA) // pp_size)
        else:
            num_attn_layers = model_config.num_layers // pp_size
            num_mamba_layers = 0

        # Per-block KV cache bytes.
        head_size = model_config.kv_channels or (
            model_config.hidden_size // model_config.num_attention_heads
        )
        heads_per_partition = num_kv_heads // tp_size if num_kv_heads >= tp_size else 1
        block_size_bytes = compute_kv_block_size_bytes(
            kv_dtype_size=kv_dtype_size,
            num_attention_layers=num_attn_layers,
            block_size_tokens=self.block_size_tokens,
            cache_mla_latent=is_mla,
            kv_reduced_dim=(
                model_config.kv_lora_rank + model_config.qk_pos_emb_head_dim if is_mla else 0
            ),
            heads_per_partition=heads_per_partition,
            head_size=head_size,
        )

        # Per-request Mamba state bytes.
        mamba_bytes_per_request = 0
        if self.mamba_inference_state_config is not None:
            mamba_bytes_per_request = compute_mamba_memory_per_request(
                self.mamba_inference_state_config, num_mamba_layers, self.num_speculative_tokens
            )

        # --- mamba_memory_ratio ---
        if self.mamba_memory_ratio is None and mamba_bytes_per_request > 0:
            blocks_per_request = math.ceil(self.max_sequence_length / self.block_size_tokens)
            kv_per_request = blocks_per_request * block_size_bytes
            self.mamba_memory_ratio = mamba_bytes_per_request / (
                kv_per_request + mamba_bytes_per_request
            )

        # --- buffer_size_gb ---
        activation_bytes = None
        print(
            f"self.buffer_size_gb={self.buffer_size_gb}, gpu_memory_budget_gb={gpu_memory_budget_gb}"
        )
        if self.buffer_size_gb is None:
            if gpu_memory_budget_gb is not None:
                budget_bytes = int(gpu_memory_budget_gb * 1024**3)
            elif model is not None:
                activation_bytes = measure_peak_activation_memory(
                    model, self.max_tokens, tp_size=tp_size
                )
                print(
                    f"Measured activation_bytes={activation_bytes} bytes with max_tokens={self.max_tokens}"
                )
                free_mem, _ = torch.cuda.mem_get_info()
                print(f"Free mem={free_mem} bytes")
                budget_bytes = max(0, free_mem - activation_bytes)
                print(f"Budget bytes={budget_bytes} bytes")
            else:
                try:
                    free_mem, _ = torch.cuda.mem_get_info()
                    budget_bytes = int(free_mem * gpu_memory_fraction)
                except (RuntimeError, AssertionError):
                    raise RuntimeError(
                        "Cannot determine GPU memory.  Pass buffer_size_gb or "
                        "gpu_memory_budget_gb explicitly."
                    )

            # Reserve Mamba prefix-caching memory.
            if self.prefix_caching_mamba_gb is not None and self.prefix_caching_mamba_gb > 0:
                budget_bytes = max(0, budget_bytes - int(self.prefix_caching_mamba_gb * 1024**3))

            self.buffer_size_gb = budget_bytes / (1024**3)

        # --- max_requests ---
        if self.max_requests is None:
            budget_bytes = int(self.buffer_size_gb * 1024**3)
            REQUEST_ROUNDER = 4

            if self.mamba_memory_ratio is not None:
                kv_budget = int(budget_bytes * (1.0 - self.mamba_memory_ratio))
                mamba_budget = int(budget_bytes * self.mamba_memory_ratio)
                block_count = max(2, kv_budget // block_size_bytes)
                mamba_max = int(mamba_budget // mamba_bytes_per_request)
                computed = min(block_count - 1, mamba_max)
            else:
                effective = block_size_bytes + mamba_bytes_per_request
                block_count = max(2, budget_bytes // effective)
                computed = block_count - 1

            computed = (computed // tp_size) * tp_size
            computed = (computed // REQUEST_ROUNDER) * REQUEST_ROUNDER
            self.max_requests = max(REQUEST_ROUNDER, computed)

        # --- validate explicit max_requests fits the budget ---
        else:
            blocks_per_request = math.ceil(self.max_sequence_length / self.block_size_tokens)
            blocks_needed = self.max_requests * blocks_per_request + 1
            if self.mamba_memory_ratio is not None:
                kv_needed = blocks_needed * block_size_bytes
                needed_bytes = int(kv_needed / (1.0 - self.mamba_memory_ratio))
            else:
                needed_bytes = blocks_needed * (block_size_bytes + mamba_bytes_per_request)
            budget_bytes = int(self.buffer_size_gb * 1024**3)
            if needed_bytes > budget_bytes:
                raise ValueError(
                    f"max_requests={self.max_requests} requires "
                    f"{needed_bytes / 1024**3:.1f} GB but buffer_size_gb is "
                    f"{self.buffer_size_gb:.1f} GB."
                )

        self._log_config(
            activation_bytes=activation_bytes,
            block_size_bytes=block_size_bytes,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            mamba_bytes_per_request=mamba_bytes_per_request,
        )

    def _log_config(
        self,
        *,
        activation_bytes=None,
        block_size_bytes=None,
        num_attn_layers=None,
        num_mamba_layers=None,
        mamba_bytes_per_request=None,
    ):
        """Log the resolved inference configuration on rank 0."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return

        lines = [
            "InferenceConfig:",
            f"  buffer_size_gb        = {self.buffer_size_gb}",
            f"  max_requests          = {self.max_requests}",
            f"  max_tokens            = {self.max_tokens}",
            f"  max_sequence_length   = {self.max_sequence_length}",
            f"  block_size_tokens     = {self.block_size_tokens}",
        ]
        if block_size_bytes is not None:
            lines.append(f"  kv_block_size_bytes   = {block_size_bytes}")
            if self.buffer_size_gb is not None and self.mamba_memory_ratio is not None:
                kv_budget = int(self.buffer_size_gb * 1024**3 * (1.0 - self.mamba_memory_ratio))
            elif self.buffer_size_gb is not None:
                kv_budget = int(self.buffer_size_gb * 1024**3)
            else:
                kv_budget = 0
            if kv_budget > 0:
                num_kv_blocks = kv_budget // block_size_bytes
                lines.append(f"  num_kv_blocks         = {num_kv_blocks}")
        if num_attn_layers is not None:
            lines.append(f"  num_attention_layers  = {num_attn_layers}")
        if activation_bytes is not None:
            lines.append(f"  activation_memory     = {activation_bytes / 1024**3:.2f} GB")
        if self.mamba_memory_ratio is not None:
            lines.append(f"  mamba_memory_ratio    = {self.mamba_memory_ratio:.4f}")
        if num_mamba_layers is not None and num_mamba_layers > 0:
            mc = self.mamba_inference_state_config
            lines.append(f"  num_mamba_layers      = {num_mamba_layers}")
            lines.append(f"  mamba_bytes_per_req   = {mamba_bytes_per_request}")
            lines.append(f"  mamba_conv_shape      = {mc.conv_states_shape}")
            lines.append(f"  mamba_ssm_shape       = {mc.ssm_states_shape}")
            lines.append(f"  mamba_conv_dtype      = {mc.conv_states_dtype}")
            lines.append(f"  mamba_ssm_dtype       = {mc.ssm_states_dtype}")
            lines.append(f"  mamba_chunk_size      = {mc.mamba_chunk_size}")
        if self.num_speculative_tokens > 0:
            lines.append(f"  num_speculative_tokens = {self.num_speculative_tokens}")
        if self.prefix_caching_mamba_gb is not None:
            lines.append(f"  prefix_caching_mamba_gb = {self.prefix_caching_mamba_gb}")
        if self.num_cuda_graphs is not None:
            lines.append(f"  num_cuda_graphs       = {self.num_cuda_graphs}")
        logging.info("\n".join(lines))
