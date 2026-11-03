# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

# Fused temperature softmax (`softmax(logits, temperature=...)`).
_HAVE_FI_SOFTMAX = flashinfer is not None and hasattr(
    getattr(flashinfer, "sampling", None), "softmax"
)

from megatron.core.inference.sampling.base import Sampling
from megatron.core.inference.sampling.sampling_params_triton import (
    HAVE_TRITON,
    gather_sampling_params,
)
from megatron.core.inference.sampling_params import (
    MIN_SAMPLING_TEMPERATURE,
    is_no_op_top_k,
    is_no_op_top_p,
)


class FlashInferSampling(Sampling):
    """FlashInfer sampling with per-step unfiltered / top-p-only / top-k-only / joint dispatch.

    Each step selects a kernel from the batch's active filters: the logits kernel
    when nothing filters, the dedicated exact top-p or top-k kernel when only one filter is in use,
    and the joint kernel only for genuinely mixed batches.
    The dispatch flags are read from the pinned CPU sampling metadata.

    The sampler runs eagerly. Its kernel choice is data-dependent (it varies with
    which filters the batch uses), so it cannot be captured in a CUDA graph; running
    eagerly also lets the controller's seeded RNG generator advance its philox offset
    normally between steps -- fresh randomness per step, reproducible from the seed.
    (FlashInfer bakes the philox state into a graph as a by-value constant at capture,
    so a captured sampler replays identical random numbers; see
    https://www.linkedin.com/pulse/pinned-rng-drifting-crash-from-cuda-graph-chenyang-zhao-csuac/)
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, config=None, enable_cuda_graph: bool = False
    ) -> None:
        # `config` / `enable_cuda_graph` are accepted for factory API symmetry but
        # intentionally unused: the sampler is never graphed (see class docstring).
        del config, enable_cuda_graph
        self._vocab_size = vocab_size
        self._rng = rng
        self._param_bufs = {}

    # ------------------------------------------------------------------
    # Fused sampling prelude.
    # ------------------------------------------------------------------
    def _param_buffers(self, slot: str, num_rows: int, device) -> Tuple[Tensor, Tensor, Tensor]:
        """Persistent per-row (temperature, top_k, top_p) buffers, grown on demand.

        One buffer set per `slot`, so parameters prepared for the draft depths survive the
        verify-path sample of the same step.
        """
        bufs = self._param_bufs.get(slot)
        if bufs is None or bufs[0].numel() < num_rows or bufs[0].device != device:
            capacity = max(num_rows, 64)
            bufs = (
                torch.empty(capacity, dtype=torch.float32, device=device),
                torch.empty(capacity, dtype=torch.int32, device=device),
                torch.empty(capacity, dtype=torch.float32, device=device),
            )
            self._param_bufs[slot] = bufs
        return tuple(b[:num_rows] for b in bufs)

    def _row_params(
        self,
        context,
        num_rows: int,
        device,
        *,
        slot: str,
        token_to_request_index: Optional[Tensor] = None,
        num_decode_requests: Optional[int] = None,
        rows_per_decode_request: int = 1,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Per-row sampler parameters, built in one launch (see `gather_sampling_params`).

        Returns the clamped temperature and the top-k / top-p with their no-op sentinels
        replaced, exactly as the unfused gather / clamp / masked_fill ops produced them.
        """
        temperature, top_k, top_p = self._param_buffers(slot, num_rows, device)
        gv = context.gpu_view
        if HAVE_TRITON and device.type == "cuda":
            gather_sampling_params(
                gv.temperature,
                gv.top_k,
                gv.top_p,
                temperature,
                top_k,
                top_p,
                num_rows,
                vocab_size=self._vocab_size,
                min_temperature=MIN_SAMPLING_TEMPERATURE,
                num_decode_requests=num_decode_requests,
                rows_per_decode_request=rows_per_decode_request,
                row_to_request=token_to_request_index,
            )
            return temperature, top_k, top_p
        # Reference path (no Triton).
        if token_to_request_index is None:
            if num_decode_requests is None:
                num_decode_requests = -(-num_rows // rows_per_decode_request)
            num_decode_rows = min(num_decode_requests * rows_per_decode_request, num_rows)
            token_to_request_index = torch.cat(
                [
                    torch.arange(num_decode_requests, device=device).repeat_interleave(
                        rows_per_decode_request
                    )[:num_decode_rows],
                    torch.arange(
                        num_decode_requests,
                        num_decode_requests + num_rows - num_decode_rows,
                        device=device,
                    ),
                ]
            )
        t = gv.temperature[token_to_request_index].clamp(min=MIN_SAMPLING_TEMPERATURE)
        k = gv.top_k[token_to_request_index]
        p = gv.top_p[token_to_request_index]
        temperature.copy_(t)
        top_k.copy_(k.masked_fill(is_no_op_top_k(k), self._vocab_size))
        top_p.copy_(p.masked_fill(is_no_op_top_p(p), 1.0))
        return temperature, top_k, top_p

    @staticmethod
    def _temperature_softmax(rows: Tensor, temperature: Tensor) -> Tensor:
        """fp32 `softmax(rows / temperature)` in one FlashInfer launch.

        Deterministic and RNG-free. FlashInfer upcasts non-fp32 logits before its kernel; with
        fp32 logits this is a single fused launch.
        """
        if _HAVE_FI_SOFTMAX:
            return flashinfer.sampling.softmax(rows.contiguous(), temperature=temperature)
        return torch.softmax(rows / temperature.unsqueeze(1), dim=-1)

    def _sample_rows(
        self,
        logits: Tensor,
        n: int,
        params: Tuple[Tensor, Tensor, Tensor],
        *,
        no_top_k: bool,
        no_top_p: bool,
        gather_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample `n` rows with prepared per-row parameters. Returns FlashInfer's int32 ids.

        The FlashInfer sampling call itself, its arguments, and the generator it advances are
        unchanged from the unfused path; it always runs eagerly.
        """
        temperature, top_k, top_p = params
        rows = logits[:n] if gather_indices is None else logits[gather_indices[:n], :]

        # `no_top_k` / `no_top_p` are the caller-supplied batch-level dispatch flags:
        # a filter is absent only when NO active request uses it. Per-row sentinels
        # disable a filter for a row (top_k=vocab keeps all tokens, top_p=1.0 keeps
        # the full mass). Every kernel gets `self._rng` so sampling is seeded and its
        # philox offset advances per launch.
        if no_top_k and no_top_p:
            # No filtering: sample the temperature-scaled dist with the Gumbel-race logits
            # kernel. `bf16 logits / fp32 temperature` promotes to fp32 -- the sampling math
            # must run in fp32; the assert pins that guarantee.
            scaled = rows / temperature.unsqueeze(1)
            assert scaled.dtype == torch.float32, f"sampling math must be fp32, got {scaled.dtype}"
            return flashinfer.sampling.sampling_from_logits(
                scaled, deterministic=True, generator=self._rng
            )
        if no_top_k:
            # Top-p only -> dedicated exact nucleus kernel on fp32 probs.
            probs = self._temperature_softmax(rows, temperature)
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p, deterministic=True, generator=self._rng
            )
        if no_top_p:
            # Top-k only -> dedicated exact top-k kernel on fp32 probs.
            probs = self._temperature_softmax(rows, temperature)
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k, deterministic=True, generator=self._rng
            )
        # Mixed batch (some top-k, some top-p, or requests using both) -> joint
        # kernel, fed the fp32 temperature-scaled logits.
        scaled = rows / temperature.unsqueeze(1)
        assert scaled.dtype == torch.float32, f"sampling math must be fp32, got {scaled.dtype}"
        return flashinfer.sampling.top_k_top_p_sampling_from_logits(
            scaled, top_k, top_p, deterministic=True, generator=self._rng
        )

    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        no_top_k: bool,
        no_top_p: bool,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
        output: Optional[Tensor] = None,
        sequence_lengths: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
        num_decode_requests: Optional[int] = None,
        rows_per_decode_request: int = 1,
        row_params: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        return_int64: bool = True,
    ) -> Tensor:
        """Sample tokens, dispatching top-p-only / top-k-only / joint by filter flags.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            no_top_k, no_top_p: Required batch-level dispatch flags (whether NO active
                request uses top-k / top-p). The caller computes them once from the
                pinned CPU sampling metadata (the context's
                `active_sampling_filter_flags`).
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, sampling parameters are gathered
                per-token rather than per-request (speculative decoding path).
            output: Optional caller-owned destination tensor of shape `[n]`.
            eager, cache_key: Accepted for API symmetry; ignored (no CUDA graph).
            num_decode_requests, rows_per_decode_request: Speculative row layout, as an
                alternative to `token_to_request_index`: `rows_per_decode_request` rows per
                decode request, then one row per remaining request. The map is computed
                in-kernel.
            row_params: Per-row (temperature, top_k, top_p) already prepared for these rows
                (see `prepare_draft_sampling_params`); skips the parameter launch.
            return_int64: When False and `output` is None, return FlashInfer's int32 ids
                without the int64 widening copy.

        Returns:
            Sampled token IDs in `output`, or a newly allocated tensor when it is not provided.
        """
        del eager, cache_key
        if row_params is None:
            slot = "rows" if num_decode_requests is None else "speculative"
            row_params = self._row_params(
                context,
                n,
                logits.device,
                slot=slot,
                token_to_request_index=token_to_request_index,
                num_decode_requests=num_decode_requests,
                rows_per_decode_request=rows_per_decode_request,
            )
        sampled_tokens = self._sample_rows(
            logits,
            n,
            row_params,
            no_top_k=no_top_k,
            no_top_p=no_top_p,
            gather_indices=gather_indices,
        )
        if output is None:
            return sampled_tokens.long() if return_int64 else sampled_tokens
        output.copy_(sampled_tokens)
        return output

    def sample_speculative(
        self,
        required_logits: Tensor,
        num_decode: int,
        num_prefill: int,
        num_speculative_tokens: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Speculative-verify sampling with the token -> request map computed in-kernel.

        Same rows, parameters, dispatch and sampler call as `Sampling.sample_speculative`, but
        the per-token parameter prelude is a single launch and the result is FlashInfer's int32
        token ids (no int64 widening copy; the verify kernel reads either width).
        """
        del eager, cache_key
        stride = 1 + num_speculative_tokens
        no_top_k, no_top_p = self._speculative_filter_flags(context)
        return self.sample_kernel(
            required_logits,
            num_decode * stride + num_prefill,
            context,
            no_top_k=no_top_k,
            no_top_p=no_top_p,
            gather_indices=gather_indices,
            eager=True,
            num_decode_requests=num_decode,
            rows_per_decode_request=stride,
            return_int64=False,
        )

    def prepare_draft_sampling_params(self, context, n: int) -> Optional[Any]:
        """Build the per-request parameters once for every MTP draft depth of this step."""
        if n <= 0:
            return None
        device = torch.device("cuda", torch.cuda.current_device())
        return self._row_params(context, n, device, slot="draft")

    def sample_draft(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        prepared: Optional[Any],
        no_top_k: bool,
        no_top_p: bool,
    ) -> Tensor:
        """Sample one draft depth with `prepare_draft_sampling_params`; returns int32 ids."""
        if prepared is not None and prepared[0].shape[0] != n:
            prepared = None
        return self.sample_kernel(
            logits,
            n,
            context,
            no_top_k=no_top_k,
            no_top_p=no_top_p,
            eager=True,
            row_params=prepared,
            return_int64=False,
        )

    def log_probs_kernel(
        self, logits: Tensor, context, *, token_to_request_index: Optional[Tensor] = None
    ) -> Tensor:
        """Per-row log-probs of the FlashInfer top-k / top-p sampling distribution.

        Args:
            logits (Tensor): Raw logits with shape `[num_rows, vocab_size]`.
            context: Active dynamic inference context providing GPU sampling metadata.
            token_to_request_index (Optional[Tensor]): Optional mapping from each
                logits row to its request index.

        Returns:
            Tensor: Per-row log probabilities for the processed distribution.
        """
        gpu_view = context.gpu_view
        if token_to_request_index is None:
            num_rows = logits.size(0)
            temperature = gpu_view.temperature[:num_rows]
            top_k = gpu_view.top_k[:num_rows]
            top_p = gpu_view.top_p[:num_rows]
        else:
            token_to_request_index = token_to_request_index.to(logits.device, non_blocking=True)
            temperature = gpu_view.temperature[token_to_request_index]
            top_k = gpu_view.top_k[token_to_request_index]
            top_p = gpu_view.top_p[token_to_request_index]

        temperature = temperature.clamp(min=MIN_SAMPLING_TEMPERATURE)
        scaled = logits / temperature.unsqueeze(1)

        # Batch-level no-op check.
        no_top_k_batch, no_top_p_batch = context.active_sampling_filter_flags()
        if no_top_k_batch and no_top_p_batch:
            return torch.log_softmax(scaled, dim=-1)

        # Sentinel / no-op values disable filtering:
        # top_k=vocab_size keeps all tokens, top_p=1.0 keeps the full probability mass.
        no_top_k = is_no_op_top_k(top_k) | (top_k >= self._vocab_size)
        no_top_p = is_no_op_top_p(top_p)
        top_k_safe = top_k.masked_fill(no_top_k, self._vocab_size)
        top_p_safe = top_p.masked_fill(no_top_p, 1.0)

        probs = torch.softmax(scaled, dim=-1)
        # Renormalize to the kept set (top-k first, then top-p) to match
        renormed = flashinfer.sampling.top_k_renorm_probs(probs, top_k_safe)
        renormed = flashinfer.sampling.top_p_renorm_probs(renormed, top_p_safe)
        # Unfiltered rows of a mixed batch bypass the renorm rounding entirely.
        return torch.where(
            (no_top_k & no_top_p).unsqueeze(1),
            torch.log_softmax(scaled, dim=-1),
            torch.log(renormed),
        )
