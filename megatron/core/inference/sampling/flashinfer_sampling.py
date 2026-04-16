# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""FlashInfer-based sampling backend with CUDA graph support."""

from typing import Optional

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

from megatron.core.inference.sampling.base import Sampling
from megatron.core.transformer.cuda_graphs import CudaGraphManager


class FlashInferSampling(Sampling):
    """Fused FlashInfer sampling with optional CUDA graph capture/replay.

    Unlike :class:`TorchSampling`, FlashInfer kernels accept per-row parameter
    tensors (temperature, top_k, top_p) directly and handle filtering internally,
    so no bucketing is required.

    We always use ``top_k_top_p_sampling_from_probs`` with sentinel values
    (``top_k=vocab_size``, ``top_p=1.0``) to disable filtering when a request
    doesn't need it.  This avoids a D2H sync to branch on filtered vs. unfiltered
    kernels and halves the number of CUDA graph variants.

    CUDA graph capture/replay is managed internally — the controller just passes
    ``use_graph=True`` and this class handles the cache.
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, enable_cuda_graph: bool = False
    ) -> None:
        self._vocab_size = vocab_size
        self._rng = rng
        self._enable_cuda_graph = enable_cuda_graph
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def pre_forward_bookkeeping(self, context) -> None:
        pass

    def _sample_impl(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> None:
        """Raw FlashInfer sampling kernel — no graph wrapping."""
        md = context.active_request_metadata
        if token_to_request_index is None:
            temperature = md["temperature"][:n]
            top_k = md["top_k"][:n]
            top_p = md["top_p"][:n]
        else:
            temperature = md["temperature"][token_to_request_index]
            top_k = md["top_k"][token_to_request_index]
            top_p = md["top_p"][token_to_request_index]

        if gather_indices is None:
            scaled = logits[:n].to(torch.float32, copy=True)
        else:
            scaled = logits[gather_indices[:n], :].to(torch.float32, copy=True)

        scaled.div_(temperature.unsqueeze(1))
        probs = torch.softmax(scaled, dim=-1)

        top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
        top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
        output[:n].copy_(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k_safe, top_p_safe, generator=self._rng
            )
        )

    def sample(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
        use_graph: bool = False,
    ) -> None:
        use_graph = use_graph and self._enable_cuda_graph
        if not use_graph:
            self._sample_impl(
                logits, n, output, context,
                gather_indices=gather_indices,
                token_to_request_index=token_to_request_index,
            )
            return

        padded_n = context.padded_active_request_count

        def kernel():
            self._sample_impl(
                logits, padded_n, output, context,
                gather_indices=gather_indices,
                token_to_request_index=token_to_request_index,
            )

        self._run_in_graph(("sample", padded_n), kernel)

    # ------------------------------------------------------------------
    # Speculative decoding
    # ------------------------------------------------------------------

    def sample_and_verify(
        self,
        logits: Tensor,
        input_ids: Tensor,
        num_speculative_tokens: int,
        context,
        *,
        sampled_tokens: Tensor,
        last_accepted_indices: Tensor,
        accepted_tokens: Tensor,
        accepted_counts: Tensor,
        use_graph: bool = False,
    ) -> None:
        use_graph = use_graph and self._enable_cuda_graph
        if use_graph:
            num_decode = context.padded_batch_dimensions.decode_req_count
            num_prefill = context.padded_batch_dimensions.prefill_req_count
        else:
            num_decode = context.num_decode_requests
            num_prefill = context.num_prefill_requests

        def kernel():
            self._sample_and_verify_core(
                self._sample_impl,
                logits,
                input_ids,
                num_speculative_tokens,
                num_decode,
                num_prefill,
                context,
                sampled_tokens=sampled_tokens,
                last_accepted_indices=last_accepted_indices,
                accepted_tokens=accepted_tokens,
                accepted_counts=accepted_counts,
            )

        if use_graph:
            self._run_in_graph(("verify", num_decode, num_prefill), kernel)
        else:
            kernel()

    # ------------------------------------------------------------------
    # CUDA graph cache
    # ------------------------------------------------------------------

    def _run_in_graph(self, key: tuple, fn) -> None:
        if key in self._graphs:
            self._graphs[key].replay()
            return

        # Warmup run.
        fn()

        # Capture run.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=CudaGraphManager.global_mempool):
            fn()
        self._graphs[key] = g
