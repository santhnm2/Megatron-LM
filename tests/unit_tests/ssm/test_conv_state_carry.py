# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the carrying conv-state update used by SSM chunked prefill.

Under chunked prefill a request's conv state has to survive being handed a
slice at a time. Deriving the new state from the slice alone is fine while
every slice is at least `d_conv` tokens long, and wrong the moment one is not
-- which chunked prefill makes reachable, since a prompt's final chunk can be
as short as two tokens.
"""

import pytest
import torch

from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_carry_states


def _reference(x, cu_seqlens, previous_states):
    """Right-align `[previous_tokens..., slice_tokens...]` into a d_conv window."""
    num_requests, conv_dim, d_conv = previous_states.shape
    out = torch.empty_like(previous_states)
    for i in range(num_requests):
        start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        # The previous state already holds the d_conv tokens before the slice.
        history = torch.cat([previous_states[i], x[start:end].transpose(0, 1)], dim=1)
        out[i] = history[:, -d_conv:]
    return out


def _run(lengths, d_conv=4, conv_dim=3, seed=0):
    torch.manual_seed(seed)
    cu_seqlens = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32)
    x = torch.randn(int(cu_seqlens[-1]), conv_dim)
    previous_states = torch.randn(len(lengths), conv_dim, d_conv)
    got = causal_conv1d_varlen_carry_states(x, cu_seqlens, previous_states)
    return got, _reference(x, cu_seqlens, previous_states)


@pytest.mark.internal
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 9])
def test_matches_reference_for_every_slice_length(length):
    """Slices shorter, equal to, and longer than d_conv all round-trip."""
    got, expected = _run([length])
    torch.testing.assert_close(got, expected)


@pytest.mark.internal
def test_long_slice_ignores_previous_state():
    """A slice of at least d_conv tokens fully determines the new state.

    This is the case the non-carrying `causal_conv1d_varlen_states` also gets
    right, so it pins the equivalence: nothing of the old state leaks through.
    """
    torch.manual_seed(1)
    cu_seqlens = torch.tensor([0, 7], dtype=torch.int32)
    x = torch.randn(7, 3)
    first = causal_conv1d_varlen_carry_states(x, cu_seqlens, torch.randn(1, 3, 4))
    second = causal_conv1d_varlen_carry_states(x, cu_seqlens, torch.randn(1, 3, 4))
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[0], x[-4:].transpose(0, 1))


@pytest.mark.internal
def test_short_slice_carries_history():
    """A 2-token slice keeps the two taps that predate it.

    Deriving the state from the slice alone would zero-fill those two columns,
    which is exactly what corrupts the first decode step after a short final
    prefill chunk.
    """
    torch.manual_seed(2)
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)
    x = torch.randn(2, 3)
    previous = torch.randn(1, 3, 4)
    got = causal_conv1d_varlen_carry_states(x, cu_seqlens, previous)

    torch.testing.assert_close(got[0, :, :2], previous[0, :, 2:])
    torch.testing.assert_close(got[0, :, 2:], x.transpose(0, 1))
    assert not torch.allclose(got[0, :, :2], torch.zeros_like(got[0, :, :2]))


@pytest.mark.internal
def test_mixed_batch_including_padding_requests():
    """Zero-length padding requests keep their state; real ones update."""
    lengths = [0, 2, 5, 0, 130]
    got, expected = _run(lengths)
    torch.testing.assert_close(got, expected)


@pytest.mark.internal
def test_zero_length_request_is_a_no_op():
    torch.manual_seed(3)
    cu_seqlens = torch.tensor([0, 0], dtype=torch.int32)
    previous = torch.randn(1, 3, 4)
    got = causal_conv1d_varlen_carry_states(torch.randn(0, 3), cu_seqlens, previous)
    torch.testing.assert_close(got, previous)
