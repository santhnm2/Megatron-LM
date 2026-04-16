# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Torch-based sampling backend."""

from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.sampling.base import Sampling


class TorchSampling(Sampling):
    """Per-bucket ``torch.multinomial`` sampling.

    Requests are grouped ("bucketed") by identical ``(temperature, top_k, top_p)``
    tuples so that each group can be sampled in a single kernel launch.
    """

    def __init__(self, rng: torch.Generator, vocab_size: int) -> None:
        self._rng = rng
        self._vocab_size = vocab_size
        self._buckets: List[Tuple] = []

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def pre_forward_bookkeeping(self, context) -> None:
        active_request_count = context.total_request_count - context.paused_request_count
        bucket_map = defaultdict(list)

        temp = context.active_request_metadata["temperature"][:active_request_count].tolist()
        top_k = context.active_request_metadata["top_k"][:active_request_count].tolist()
        top_p = context.active_request_metadata["top_p"][:active_request_count].tolist()

        for request_index, (t, k, p) in enumerate(zip(temp, top_k, top_p)):
            bucket_map[(t, k, p)].append(request_index)

        self._buckets = [
            (indices, *sampling_params) for sampling_params, indices in bucket_map.items()
        ]

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
        if gather_indices is not None:
            logits = logits[gather_indices[:n], :]

        token_list = []
        indices_list = []
        for request_indices, temp, top_k, top_p in self._buckets:
            indices_tensor = torch.tensor(
                request_indices, device=logits.device, dtype=torch.long
            )
            if token_to_request_index is not None:
                indices_tensor = torch.where(
                    torch.isin(token_to_request_index, indices_tensor)
                )[0]
            token_list.append(
                self._sampling_func(logits[indices_tensor, :], temp, top_k, top_p)
            )
            indices_list.append(indices_tensor)

        sampled_tokens = torch.cat(token_list, dim=0)
        sampled_indices = torch.cat(indices_list, dim=0)
        output[sampled_indices] = sampled_tokens

    # ------------------------------------------------------------------
    # Core sampling kernel
    # ------------------------------------------------------------------

    def _sampling_func(
        self,
        last_token_logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tensor:
        """Sample from logits with temperature, top-k, and top-p filtering.

        Args:
            last_token_logits: Logits of shape ``[batch_size, vocab_size]``.
            temperature: Temperature for sampling.
            top_k: Top-k value (0 disables).
            top_p: Top-p value (0.0 disables).

        Returns:
            Sampled token ids of shape ``[batch_size]``.
        """
        assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"

        if top_k == 1:
            return torch.argmax(last_token_logits, dim=-1)

        last_token_logits = last_token_logits.clone()
        if temperature != 1.0:
            last_token_logits.div_(temperature)

        if top_k > 1:
            filter_ = last_token_logits < torch.topk(last_token_logits, top_k)[0][..., -1, None]
            last_token_logits.masked_fill_(filter_, float("-Inf"))
        elif top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            filter_ = cumulative_probs > top_p
            filter_[:, 1:] = filter_[:, :-1].clone()
            filter_[..., 0] = 0
            filter_ = filter_.scatter(1, sorted_indices, filter_)
            last_token_logits.masked_fill_(filter_, float("-Inf"))

        probabilities = last_token_logits.softmax(dim=-1)
        sampled = torch.multinomial(
            probabilities, num_samples=1, generator=self._rng
        ).view(-1)

        return torch.clamp(sampled, min=0, max=(self._vocab_size - 1))
