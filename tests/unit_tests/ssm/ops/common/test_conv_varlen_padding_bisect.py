# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stage 3 diagnostic: is the varlen conv padding-invariant on the GDP path?

`causal_conv1d_varlen_fn` is the one stage between the GDP layer input and the
chunk kernels that the prefill stage bisect does not cover, and it is the only
place where GDP exercises code Mamba never reaches:

* `bias=None` -> `HAS_BIAS=False`. `MambaMixer` defaults `conv_bias=True` and
  asserts it (`mamba_mixer.py:276`), so the bias-free branch added in
  `3a2b0a05b8` has exactly one caller in the tree: `GatedDeltaProductMixer`,
  which defaults `conv_bias=False` (`gated_delta_product.py:172`). The branch
  aliases `bias = x` as a dummy pointer, so any read of `bias_ptr` that is not
  gated by `HAS_BIAS` silently reads activations.
* `initial_states=None` -> `HAS_INITIAL_STATES=False`. GDP prefill has no conv
  history (no prefix caching yet); Mamba always passes a real tensor.

The geometry is the failing e2e step: prompts 70 and 129, 199 real tokens, and
`padded_token_count` 200 (what the eager arm rounds to) versus 1720 (the CUDA
graph bucket), so request 1 is the last real sequence and sits directly against
the padded tail.

Not a real test -- delete once the question is settled.
"""

import pytest
import torch

from megatron.core.ssm.ops.common.causal_conv1d_varlen import (
    _causal_conv1d_varlen_simple,
    causal_conv1d_varlen_fn,
)
from tests.unit_tests.test_utilities import Utils

# GDP mixer geometry: VKQ carries d_inner*M + (M+1)*ngroups*d_state channels.
# With mamba_num_heads=8, mamba_head_dim=32, mamba_num_groups=8,
# mamba_state_dim=64, M=2: 256*2 + 3*8*64 = 2048.
CONV_DIM = 2048
D_CONV = 4
SEQ_LENS = [70, 129]
REAL_TOKENS = sum(SEQ_LENS)


def _metadata(seq_lens, padded_tokens, padded_reqs, device):
    """`cu_seqlens` plus the per-token conv metadata, exactly as MambaMetadata builds it.

    Mirrors `mamba_metadata.py:377-397`: request ids and request start offsets are
    prefill-relative, and the padded tail is filled with the `0` sentinel.
    """
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    real_tokens = cu[-1]
    cu += [cu[-1]] * (padded_reqs - len(seq_lens))

    seq_idx = torch.zeros(padded_tokens, dtype=torch.int32, device=device)
    seq_start = torch.zeros(padded_tokens, dtype=torch.int32, device=device)
    for i, n in enumerate(seq_lens):
        seq_idx[cu[i] : cu[i + 1]] = i
        seq_start[cu[i] : cu[i + 1]] = cu[i]
    # Padded tail keeps the 0/0 sentinel the metadata writes.
    assert real_tokens <= padded_tokens
    return (
        torch.tensor(cu, dtype=torch.int32, device=device),
        seq_idx,
        seq_start,
        real_tokens,
    )


def _inputs(padded_tokens, padded_reqs, pad_fill, use_bias, use_initial_states, seed=0):
    device = torch.cuda.current_device()
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.bfloat16)

    x = rand(padded_tokens, CONV_DIM)
    if padded_tokens > REAL_TOKENS:
        x[REAL_TOKENS:] = pad_fill
    weight = rand(CONV_DIM, D_CONV)
    bias = rand(CONV_DIM) if use_bias else None
    initial_states = rand(padded_reqs, CONV_DIM, D_CONV - 1) if use_initial_states else None
    return x, weight, bias, initial_states


def _report(label, ref, cmp_, real_tokens):
    a = ref[:real_tokens].float()
    b = cmp_[:real_tokens].float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    d = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    worst = float(d.max())
    nonfinite = int((torch.isfinite(a) != torch.isfinite(b)).sum())
    # Per-sequence, so "which request broke" is immediate.
    per_seq = []
    start = 0
    for i, n in enumerate(SEQ_LENS):
        seg = d[start : start + n]
        per_seq.append(f"seq{i}(len {n})={float(seg.max()):.3e}")
        start += n
    print(
        f"  {label:<44} max|diff|={worst:.3e}  nonfinite_mismatch={nonfinite}\n"
        f"      {'  '.join(per_seq)}"
    )
    return worst == 0 and nonfinite == 0


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestConvVarlenPaddingBisect:
    """Does token/request padding move the real region of the varlen conv?"""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("padded_tokens", [200, 1720], ids=["eager200", "graph1720"])
    @pytest.mark.parametrize("padded_reqs", [2, 4], ids=["reqs2", "reqs4"])
    @pytest.mark.parametrize("use_bias", [False, True], ids=["nobias", "bias"])
    @pytest.mark.parametrize("use_initial_states", [False, True], ids=["noinit", "init"])
    @pytest.mark.parametrize("pad_fill", [0.0, 7.0, float("nan")], ids=["zero", "big", "nan"])
    @torch.inference_mode()
    def test_conv_padding_invariance(
        self, padded_tokens, padded_reqs, use_bias, use_initial_states, pad_fill
    ):
        """The conv output at real token positions must not depend on the padding."""
        device = torch.cuda.current_device()
        x, weight, bias, initial_states = _inputs(
            max(padded_tokens, REAL_TOKENS), padded_reqs, pad_fill, use_bias, use_initial_states
        )
        cu_pad, idx_pad, start_pad, real_tokens = _metadata(
            SEQ_LENS, padded_tokens, padded_reqs, device
        )
        out_pad = causal_conv1d_varlen_fn(
            x=x.contiguous(),
            weight=weight,
            bias=bias,
            cu_seqlens=cu_pad,
            initial_states=initial_states,
            activation="silu",
            precomputed_seq_idx=idx_pad,
            precomputed_seq_start=start_pad,
        )

        # Tight reference: only the real requests, no token or request padding.
        cu_tight, idx_tight, start_tight, _ = _metadata(
            SEQ_LENS, REAL_TOKENS, len(SEQ_LENS), device
        )
        out_tight = causal_conv1d_varlen_fn(
            x=x[:REAL_TOKENS].contiguous(),
            weight=weight,
            bias=bias,
            cu_seqlens=cu_tight,
            initial_states=(
                initial_states[: len(SEQ_LENS)].contiguous()
                if initial_states is not None
                else None
            ),
            activation="silu",
            precomputed_seq_idx=idx_tight,
            precomputed_seq_start=start_tight,
        )

        print(
            f"\n=== conv padding: tokens {REAL_TOKENS}->{padded_tokens}, "
            f"reqs {len(SEQ_LENS)}->{padded_reqs}, bias={use_bias}, "
            f"init={use_initial_states}, pad_fill={pad_fill} ==="
        )
        ok = _report("padded vs tight", out_tight, out_pad, REAL_TOKENS)

        # Also check against the PyTorch reference, so a shared kernel bug in
        # both arms cannot hide behind their agreement.
        ref = torch.empty_like(x[:REAL_TOKENS])
        _causal_conv1d_varlen_simple(
            x[:REAL_TOKENS].contiguous(),
            weight,
            bias,
            cu_tight,
            initial_states[: len(SEQ_LENS)].contiguous() if initial_states is not None else None,
            ref,
        )
        a = ref.float()
        b = out_tight[:REAL_TOKENS].float()
        print(f"  {'tight vs pytorch reference':<44} max|diff|={float((a - b).abs().max()):.3e}")

        assert ok, "padding changed the real region of the conv output"

    @pytest.mark.parametrize("warm_up_first", [True, False], ids=["warm", "cold"])
    @torch.inference_mode()
    def test_conv_graph_replay(self, warm_up_first):
        """Capture and replay the conv; `cold` skips the pre-capture warmup.

        The kernel autotunes on `key=["conv_dim"]` only, and its grid is derived
        from the autotuned `BLOCK_T`/`BLOCK_C`. If a `conv_dim` is first seen
        during capture, `do_bench` runs inside the capture. In the e2e test the
        eager arm always warms the cache first, so `cold` is the case that is
        never exercised there.
        """
        device = torch.cuda.current_device()
        padded_tokens, padded_reqs = 1720, 2
        x, weight, bias, initial_states = _inputs(
            padded_tokens, padded_reqs, 0.0, use_bias=False, use_initial_states=False
        )
        cu, idx, start, _ = _metadata(SEQ_LENS, padded_tokens, padded_reqs, device)

        def run():
            return causal_conv1d_varlen_fn(
                x=x,
                weight=weight,
                bias=bias,
                cu_seqlens=cu,
                initial_states=initial_states,
                activation="silu",
                precomputed_seq_idx=idx,
                precomputed_seq_start=start,
            )

        eager = run().clone()

        if warm_up_first:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = run()
        graph.replay()
        torch.cuda.synchronize()

        print(f"\n=== conv graph replay ({'warm' if warm_up_first else 'cold'}) ===")
        ok = _report("replay vs eager", eager, static_out, REAL_TOKENS)
        assert ok, "graph replay changed the real region of the conv output"
