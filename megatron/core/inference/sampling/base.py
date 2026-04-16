# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base class for inference sampling backends."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Base class for inference sampling backends.

    Subclasses implement :meth:`sample` (the raw sampling kernel) and optionally
    override :meth:`sample_and_verify` for speculative decoding.  CUDA graph
    management, if any, is internal to the subclass.
    """

    @abstractmethod
    def pre_forward_bookkeeping(self, context) -> None:
        """Prepare sampling state before the forward pass."""
        ...

    @abstractmethod
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
        """Sample ``n`` tokens from ``logits`` into ``output[:n]``.

        Args:
            logits: Logits tensor of shape ``[>=n, vocab_size]``.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active :class:`DynamicInferenceContext`.
            gather_indices: If provided, gather ``logits[gather_indices[:n], :]``
                before sampling (used when ``materialize_only_last_token_logits``
                is ``False``).
            token_to_request_index: Per-token → per-request mapping used by the
                speculative path to expand per-request params to per-token.
            use_graph: Hint that this call may be captured in a CUDA graph.
                Backends that support graphs will use padded dimensions and
                manage capture/replay internally.  Ignored by backends that
                do not support graphs.
        """
        ...

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
        """Sample from speculative logits and verify against input tokens.

        Default implementation: eager, no CUDA graph capture.
        Subclasses may override to add graph support.
        """
        num_decode = context.num_decode_requests
        num_prefill = context.num_prefill_requests
        self._sample_and_verify_core(
            self.sample,
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

    def _sample_and_verify_core(
        self,
        sample_fn,
        logits: Tensor,
        input_ids: Tensor,
        num_speculative_tokens: int,
        num_decode: int,
        num_prefill: int,
        context,
        *,
        sampled_tokens: Tensor,
        last_accepted_indices: Tensor,
        accepted_tokens: Tensor,
        accepted_counts: Tensor,
    ) -> None:
        """Shared sample-then-verify logic.

        ``sample_fn`` must be callable with the same signature as
        :meth:`sample` (minus ``use_graph``).
        """
        active_request_count = num_decode + num_prefill
        device = logits.device

        # --- Compute required logit indices ---
        decode_token_count = num_decode * (num_speculative_tokens + 1)
        decode_indices = torch.arange(decode_token_count, device=device)
        query_lengths = context.active_request_query_lengths[:active_request_count]
        cumsum = torch.cumsum(query_lengths, dim=0)
        prefill_last_indices = cumsum[num_decode:] - 1
        required_logit_indices = torch.cat([decode_indices, prefill_last_indices])

        if context.config.materialize_only_last_token_logits:
            required_logits = logits.squeeze(0)
        else:
            required_logits = logits.squeeze(0)[required_logit_indices, :]

        # --- Sample ---
        num_tokens = decode_token_count + num_prefill
        token_to_request_index = torch.cat(
            [
                torch.arange(num_decode, device=device).repeat_interleave(
                    1 + num_speculative_tokens, output_size=decode_token_count
                ),
                torch.arange(num_decode, num_decode + num_prefill, device=device),
            ]
        )
        output_tokens = torch.empty(num_tokens, device=device, dtype=torch.int64)
        sample_fn(
            required_logits,
            num_tokens,
            output_tokens,
            context,
            token_to_request_index=token_to_request_index,
        )

        # --- Verify ---
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
        input_tokens_required = input_ids[required_logit_indices]

        last_one_indices = self._verify_tokens(
            output_tokens,
            input_tokens_required,
            num_decode,
            num_prefill,
            num_speculative_tokens,
            active_request_count,
            accepted_tokens,
            accepted_counts,
        )

        sampled_tokens[:active_request_count] = output_tokens[last_one_indices]
        last_accepted_indices[:active_request_count] = required_logit_indices[
            last_one_indices
        ]

    @staticmethod
    def _verify_tokens(
        output_tokens: Tensor,
        input_tokens: Tensor,
        num_decode: int,
        num_prefill: int,
        num_speculative_tokens: int,
        active_request_count: int,
        accepted_tokens: Tensor,
        accepted_counts: Tensor,
    ) -> Tensor:
        """Verify speculative tokens and return last-accepted indices."""
        device = output_tokens.device
        decode_len = num_decode * (num_speculative_tokens + 1)

        # Decode verification.
        decode_inputs = input_tokens[:decode_len].reshape(
            num_decode, num_speculative_tokens + 1
        )
        decode_outputs = output_tokens[:decode_len].reshape(
            num_decode, num_speculative_tokens + 1
        )
        decode_outputs_shifted = decode_outputs.roll(1, dims=1)
        decode_mask_2d = decode_inputs == decode_outputs_shifted
        decode_mask_2d[:, 0] = True
        decode_mask_2d = decode_mask_2d.cummin(dim=1).values

        last_one_indices = torch.zeros(
            active_request_count, device=device, dtype=torch.long
        )
        local_last_indices = decode_mask_2d.sum(dim=1) - 1
        row_offsets = torch.arange(num_decode, device=device) * (
            num_speculative_tokens + 1
        )
        last_one_indices[:num_decode] = row_offsets + local_last_indices

        last_one_indices[num_decode : num_decode + num_prefill] = (
            torch.arange(num_prefill, device=device) + decode_len
        )

        # Extract accepted tokens into output buffers.
        decode_accepted = decode_inputs.masked_fill(~decode_mask_2d, -1)
        accepted_tokens[:num_decode, :] = decode_accepted[:, 1:]
        accepted_counts.copy_((accepted_tokens != -1).sum(dim=1))

        return last_one_indices
