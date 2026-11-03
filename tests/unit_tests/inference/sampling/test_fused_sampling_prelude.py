# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The fused sampling prelude must reproduce the unfused FlashInfer path.

`FlashInferSampling` builds per-row sampling parameters with one Triton launch
(`gather_sampling_params`) and computes the top-p / top-k probabilities with FlashInfer's fused
temperature softmax. These tests pin that against a verbatim copy of the previous eager path:

* the per-row parameters are bit-identical (any layout: plain rows, speculative decode + prefill
  rows for D = 1..4, an explicit map, paused requests in the context);
* the pre-sampling probabilities match within fp32 rounding;
* greedy tokens are bit-identical, and sampled tokens are identical under the same seed;
* the sampler consumes the generator identically (same philox offset afterwards).
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.sampling.sampling_params_triton import gather_sampling_params
from megatron.core.inference.sampling_params import (
    MIN_SAMPLING_TEMPERATURE,
    is_no_op_top_k,
    is_no_op_top_p,
)

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

try:
    import flashinfer

    HAVE_FLASHINFER = True
except ImportError:
    flashinfer = None
    HAVE_FLASHINFER = False

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
)
needs_flashinfer = pytest.mark.skipif(not HAVE_FLASHINFER, reason="needs flashinfer")

DEVICE = "cuda"
VOCAB = 4096


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _request_params(num_requests, kind, seed=0):
    """Per-request (temperature, top_k, top_p) on the GPU, covering every sentinel."""
    g = torch.Generator().manual_seed(seed)
    if kind == "greedy":
        temperature = torch.zeros(num_requests)
        top_k = torch.ones(num_requests, dtype=torch.int32)
        top_p = torch.zeros(num_requests)
    elif kind == "top_p":
        temperature = torch.ones(num_requests)
        top_k = torch.zeros(num_requests, dtype=torch.int32)
        top_p = torch.full((num_requests,), 0.95)
        top_p[::3] = 0.0  # sentinel: disabled for some rows
    elif kind == "top_k":
        temperature = torch.rand(num_requests, generator=g) + 0.25
        top_k = torch.randint(1, 50, (num_requests,), generator=g, dtype=torch.int32)
        top_k[::4] = 0  # sentinel
        top_p = torch.zeros(num_requests)
    elif kind == "none":
        temperature = torch.rand(num_requests, generator=g) + 0.5
        top_k = torch.zeros(num_requests, dtype=torch.int32)
        top_p = torch.zeros(num_requests)
    elif kind == "mixed":
        # Greedy, top-p, top-k, joint and unfiltered rows side by side.
        temperature = torch.tensor([0.0, 1.0, 0.7, 1.3, 2.0, 1e-9, 0.5, 1.0])[
            torch.arange(num_requests) % 8
        ]
        top_k = torch.tensor([1, 0, 20, 5, 0, 0, -1, 7], dtype=torch.int32)[
            torch.arange(num_requests) % 8
        ]
        top_p = torch.tensor([0.0, 0.95, 0.0, 0.9, 1.0, 0.5, 1.5, 0.0])[
            torch.arange(num_requests) % 8
        ]
    else:
        raise ValueError(kind)
    return temperature.to(DEVICE), top_k.to(DEVICE), top_p.to(DEVICE)


def _context(temperature, top_k, top_p, paused=0):
    """Stand-in for the fields the FlashInfer sampler reads from DynamicInferenceContext.

    `gpu_view` and `active_request_metadata` are indexed by ACTIVE request; paused requests only
    shift `total_request_count`.
    """
    n = temperature.numel()
    return SimpleNamespace(
        gpu_view=SimpleNamespace(temperature=temperature, top_k=top_k, top_p=top_p),
        active_request_metadata={
            "temperature": temperature.cpu(),
            "top_k": top_k.cpu(),
            "top_p": top_p.cpu(),
        },
        total_request_count=n + paused,
        paused_request_count=paused,
    )


def _spec_map(num_decode, num_prefill, num_spec):
    """The token -> request map `Sampling.sample_speculative` used to build eagerly."""
    stride = 1 + num_spec
    return torch.cat(
        [
            torch.arange(num_decode, device=DEVICE).repeat_interleave(
                stride, output_size=num_decode * stride
            ),
            torch.arange(num_decode, num_decode + num_prefill, device=DEVICE),
        ]
    )


def _reference_params(temperature, top_k, top_p, idx):
    """The unfused per-row parameter ops."""
    t = temperature[idx].clamp(min=MIN_SAMPLING_TEMPERATURE)
    k = top_k[idx]
    p = top_p[idx]
    return (t, k.masked_fill(is_no_op_top_k(k), VOCAB), p.masked_fill(is_no_op_top_p(p), 1.0))


def _reference_sample(logits, n, context, idx, no_top_k, no_top_p, generator):
    """Verbatim copy of the previous `FlashInferSampling.sample_kernel` body.

    Returns (int64 tokens, the pre-sampling tensor: probs or scaled logits).
    """
    gv = context.gpu_view
    temperature = gv.temperature[idx]
    top_k = gv.top_k[idx]
    top_p = gv.top_p[idx]
    temperature = temperature.clamp(min=MIN_SAMPLING_TEMPERATURE)
    scaled = logits[:n] / temperature.unsqueeze(1)
    if no_top_k and no_top_p:
        out = flashinfer.sampling.sampling_from_logits(
            scaled, deterministic=True, generator=generator
        )
        return out.long(), scaled
    if no_top_k:
        probs = torch.softmax(scaled, dim=-1)
        top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
        out = flashinfer.sampling.top_p_sampling_from_probs(
            probs, top_p_safe, deterministic=True, generator=generator
        )
        return out.long(), probs
    if no_top_p:
        probs = torch.softmax(scaled, dim=-1)
        top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), VOCAB)
        out = flashinfer.sampling.top_k_sampling_from_probs(
            probs, top_k_safe, deterministic=True, generator=generator
        )
        return out.long(), probs
    top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), VOCAB)
    top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
    out = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        scaled, top_k_safe, top_p_safe, deterministic=True, generator=generator
    )
    return out.long(), scaled


def _logits(n, dtype, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # Peaky rows, like real LM logits, with a few exact ties.
    logits = torch.randn(n, VOCAB, generator=g, device=DEVICE) * 4.0
    logits[:, 7] = logits[:, 11]
    return logits.to(dtype)


def _generator(seed):
    return torch.Generator(device=DEVICE).manual_seed(seed)


# ---------------------------------------------------------------------------
# gather_sampling_params: bit-exact against the torch ops it replaces
# ---------------------------------------------------------------------------
class TestGatherSamplingParams:
    @pytest.mark.parametrize("kind", ["greedy", "top_p", "top_k", "none", "mixed"])
    @pytest.mark.parametrize(
        "num_decode,num_prefill,num_spec",
        [(5, 0, 0), (5, 0, 1), (3, 2, 2), (7, 3, 3), (4, 4, 4), (0, 3, 3), (1, 0, 4)],
    )
    def test_speculative_layout(self, kind, num_decode, num_prefill, num_spec):
        params = _request_params(num_decode + num_prefill, kind)
        n = num_decode * (1 + num_spec) + num_prefill
        out = (
            torch.full((n + 3,), -7.0, device=DEVICE),
            torch.full((n + 3,), -7, dtype=torch.int32, device=DEVICE),
            torch.full((n + 3,), -7.0, device=DEVICE),
        )
        gather_sampling_params(
            *params,
            *out,
            n,
            vocab_size=VOCAB,
            min_temperature=MIN_SAMPLING_TEMPERATURE,
            num_decode_requests=num_decode,
            rows_per_decode_request=1 + num_spec,
        )
        ref = _reference_params(*params, _spec_map(num_decode, num_prefill, num_spec))
        for got, want in zip(out, ref):
            assert torch.equal(got[:n], want)
            # Rows past n are untouched.
            assert (got[n:] == -7).all()

    @pytest.mark.parametrize("n", [1, 9, 1500])
    def test_plain_rows_and_explicit_map(self, n):
        params = _request_params(n, "mixed", seed=n)
        out = tuple(torch.empty_like(p) for p in params)
        gather_sampling_params(
            *params, *out, n, vocab_size=VOCAB, min_temperature=MIN_SAMPLING_TEMPERATURE
        )
        ref = _reference_params(*params, torch.arange(n, device=DEVICE))
        for got, want in zip(out, ref):
            assert torch.equal(got, want)

        perm = torch.randperm(n, device=DEVICE)
        gather_sampling_params(
            *params,
            *out,
            n,
            vocab_size=VOCAB,
            min_temperature=MIN_SAMPLING_TEMPERATURE,
            row_to_request=perm,
        )
        ref = _reference_params(*params, perm)
        for got, want in zip(out, ref):
            assert torch.equal(got, want)

    def test_replays_bit_exact(self):
        """Determinism: repeated launches write identical bits."""
        params = _request_params(300, "mixed", seed=3)
        outs = []
        for _ in range(3):
            out = tuple(torch.empty(1200, dtype=p.dtype, device=DEVICE) for p in params)
            gather_sampling_params(
                *params,
                *out,
                1200,
                vocab_size=VOCAB,
                min_temperature=MIN_SAMPLING_TEMPERATURE,
                num_decode_requests=300,
                rows_per_decode_request=4,
            )
            outs.append(out)
        for out in outs[1:]:
            for a, b in zip(outs[0], out):
                assert torch.equal(a.view(torch.int32), b.view(torch.int32))


# ---------------------------------------------------------------------------
# FlashInferSampling: new path vs the previous eager path
# ---------------------------------------------------------------------------
@needs_flashinfer
class TestFlashInferFusedPrelude:
    def _sampling(self, seed):
        from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling

        return FlashInferSampling(VOCAB, _generator(seed))

    @staticmethod
    def _flags(kind, context):
        from megatron.core.inference.sampling.base import Sampling

        return Sampling._speculative_filter_flags(context)

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("kind", ["greedy", "top_p", "top_k", "none", "mixed"])
    @pytest.mark.parametrize(
        "num_decode,num_prefill,num_spec", [(6, 0, 1), (5, 2, 2), (8, 0, 3), (3, 3, 4)]
    )
    @pytest.mark.parametrize("paused", [0, 2])
    def test_sample_speculative_matches_unfused(
        self, dtype, kind, num_decode, num_prefill, num_spec, paused
    ):
        num_requests = num_decode + num_prefill
        params = _request_params(num_requests, kind)
        context = _context(*params, paused=paused)
        n = num_decode * (1 + num_spec) + num_prefill
        logits = _logits(n + 5, dtype, seed=n)  # trailing padded rows are never sampled
        no_top_k, no_top_p = self._flags(kind, context)

        seed = 1234
        ref_gen = _generator(seed)
        ref_tokens, ref_pre = _reference_sample(
            logits,
            n,
            context,
            _spec_map(num_decode, num_prefill, num_spec),
            no_top_k,
            no_top_p,
            ref_gen,
        )

        sampling = self._sampling(seed)
        tokens = sampling.sample_speculative(logits, num_decode, num_prefill, num_spec, context)

        assert tokens.shape == (n,)
        assert torch.equal(tokens.long(), ref_tokens), (tokens.long() != ref_tokens).nonzero()
        # Same RNG consumption: the next draw starts from the same philox state.
        assert torch.equal(sampling._rng.get_state(), ref_gen.get_state())

        # Pre-sampling tensors.
        temperature, top_k, top_p = sampling._param_bufs["speculative"]
        if no_top_k != no_top_p:
            probs = sampling._temperature_softmax(logits[:n], temperature[:n])
            assert probs.dtype == torch.float32
            torch.testing.assert_close(probs, ref_pre, rtol=2e-5, atol=1e-7)
        else:
            scaled = logits[:n] / temperature[:n].unsqueeze(1)
            assert torch.equal(scaled, ref_pre)

    @pytest.mark.parametrize("kind", ["greedy", "top_p", "mixed"])
    def test_greedy_rows_bit_identical_across_seeds(self, kind):
        """top_k=1 rows pick the argmax regardless of seed, in both paths."""
        num_decode, num_spec = 16, 3
        params = _request_params(num_decode, kind)
        context = _context(*params)
        n = num_decode * (1 + num_spec)
        logits = _logits(n, torch.bfloat16, seed=5)
        no_top_k, no_top_p = self._flags(kind, context)
        greedy_rows = (params[1] == 1).repeat_interleave(1 + num_spec)
        for seed in (0, 1, 2):
            ref, _ = _reference_sample(
                logits,
                n,
                context,
                _spec_map(num_decode, 0, num_spec),
                no_top_k,
                no_top_p,
                _generator(seed),
            )
            got = self._sampling(seed).sample_speculative(logits, num_decode, 0, num_spec, context)
            assert torch.equal(got.long()[greedy_rows], ref[greedy_rows])
            assert torch.equal(got.long(), ref)

    @pytest.mark.parametrize("kind", ["greedy", "top_p", "top_k", "none", "mixed"])
    @pytest.mark.parametrize("num_requests", [1, 7, 64])
    def test_draft_depths_match_unfused(self, kind, num_requests):
        """Draft sampling with params prepared once per step == per-depth unfused sampling."""
        params = _request_params(num_requests, kind)
        context = _context(*params, paused=1)
        from megatron.core.inference.sampling_params import is_no_op_top_k as nk
        from megatron.core.inference.sampling_params import is_no_op_top_p as np_

        # The draft path's flags come from `active_sampling_filter_flags`.
        no_top_k = bool(nk(context.active_request_metadata["top_k"]).all())
        no_top_p = bool(np_(context.active_request_metadata["top_p"]).all())

        seed = 77
        ref_gen = _generator(seed)
        sampling = self._sampling(seed)
        prepared = sampling.prepare_draft_sampling_params(context, num_requests)
        idx = torch.arange(num_requests, device=DEVICE)
        for depth in range(4):
            logits = _logits(num_requests, torch.bfloat16, seed=100 + depth)
            ref, _ = _reference_sample(
                logits, num_requests, context, idx, no_top_k, no_top_p, ref_gen
            )
            got = sampling.sample_draft(
                logits,
                num_requests,
                context,
                prepared=prepared,
                no_top_k=no_top_k,
                no_top_p=no_top_p,
            )
            assert torch.equal(got.long(), ref), f"depth {depth}"
        assert torch.equal(sampling._rng.get_state(), ref_gen.get_state())

    @pytest.mark.parametrize("kind", ["top_p", "mixed"])
    def test_sample_kernel_output_and_dtype_unchanged(self, kind):
        """`sample_kernel` keeps its contract: int64 ids, or a copy into `output`."""
        n = 12
        params = _request_params(n, kind)
        context = _context(*params)
        logits = _logits(n, torch.bfloat16, seed=9)
        from megatron.core.inference.sampling_params import is_no_op_top_k as nk
        from megatron.core.inference.sampling_params import is_no_op_top_p as np_

        no_top_k = bool(nk(context.active_request_metadata["top_k"]).all())
        no_top_p = bool(np_(context.active_request_metadata["top_p"]).all())
        ref, _ = _reference_sample(
            logits, n, context, torch.arange(n, device=DEVICE), no_top_k, no_top_p, _generator(3)
        )
        got = self._sampling(3).sample_kernel(
            logits, n, context, no_top_k=no_top_k, no_top_p=no_top_p
        )
        assert got.dtype == torch.int64 and torch.equal(got, ref)
        out = torch.full((n,), -1, dtype=torch.int64, device=DEVICE)
        self._sampling(3).sample_kernel(
            logits, n, context, no_top_k=no_top_k, no_top_p=no_top_p, output=out
        )
        assert torch.equal(out, ref)

        # Gathered rows (materialize_only_last_token_logits=False layout).
        gather = torch.randperm(n, device=DEVICE).to(torch.int32)
        ref_g, _ = _reference_sample(
            logits[gather.long()],
            n,
            context,
            torch.arange(n, device=DEVICE),
            no_top_k,
            no_top_p,
            _generator(4),
        )
        got_g = self._sampling(4).sample_kernel(
            logits, n, context, no_top_k=no_top_k, no_top_p=no_top_p, gather_indices=gather
        )
        assert torch.equal(got_g, ref_g)
