# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stage 2 diagnostic: which stage of the GDP prefill chain breaks?

`chunk_gated_delta_product_varlen` is a chain of seven kernel launches. This
file runs that chain twice, stage by stage, and diffs every intermediate on the
region that belongs to real tokens:

* `test_stage_padding_invariance` -- tight batch vs. a batch padded out to a
  CUDA-graph bucket (extra padded tokens, extra zero-length requests with `-1`
  slots, longer chunk descriptors). No CUDA graph involved: this isolates the
  padding contract from graph capture.
* `test_stage_graph_replay` -- the same chain captured in a CUDA graph and
  replayed, diffed against the eager run of identical inputs. Stage tensors are
  returned from the capture, so their storage lives in the graph pool and shows
  the replayed values.

The geometry matches `test_gdp_cuda_graph_e2e`: sequence lengths 70/129/33/200,
none a multiple of the 64-token chunk, and `M = 2` Householder copies.

Not a real test -- delete once the question is settled.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.chunk_h import chunk_gated_delta_product_fwd_h
from megatron.core.ssm.ops.gdp.chunk_o import chunk_gated_delta_product_fwd_o
from megatron.core.ssm.ops.gdp.common import CHUNK_SIZE, RCP_LN2, l2norm_fwd
from megatron.core.ssm.ops.gdp.cumsum import chunk_local_cumsum
from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors
from megatron.core.ssm.ops.gdp.scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from megatron.core.ssm.ops.gdp.solve_tril import solve_tril
from megatron.core.ssm.ops.gdp.wy_fast import recompute_w_u_fwd
from tests.unit_tests.test_utilities import Utils

# The e2e model: mamba_num_heads=8, mamba_state_dim=64, mamba_head_dim=32, M=2.
H, K, V, M = 8, 64, 32, 2
SEQ_LENS = [70, 129, 33, 200]
REAL_TOKENS = sum(SEQ_LENS)
NUM_SLOTS = 16
REAL_SLOTS = [5, 1, 11, 2]


def _descriptors(seq_lens, padded_tokens, padded_reqs, device):
    """Device-side chunk descriptors for `seq_lens` padded to a graph bucket."""
    cu_list = [0]
    for n in seq_lens:
        cu_list.append(cu_list[-1] + n)
    cu_list += [cu_list[-1]] * (padded_reqs - len(seq_lens))
    indices, indices_dp, offsets, n_c, n_c_dp = build_gdp_chunk_descriptors(
        cu_list, padded_reqs, M, padded_tokens
    )

    def dev(values, shape, dtype=torch.int32):
        return torch.tensor(values, device=device, dtype=dtype).view(*shape)

    return cu_list, dict(
        cu_seqlens=dev(cu_list, (len(cu_list),)),
        chunk_indices=dev(indices, (n_c, 2)),
        chunk_indices_dp=dev(indices_dp, (n_c_dp, 2)),
        chunk_offsets=dev(offsets, (len(offsets),)),
    )


def _inputs(padded_tokens, pad_fill, seed=0):
    """Packed prefill activations; the padded tail is filled with `pad_fill`."""
    device = torch.cuda.current_device()
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    q = rand(1, padded_tokens, H, K)
    k = rand(1, padded_tokens * M, H, K)
    v = rand(1, padded_tokens * M, H, V)
    g = -rand(1, padded_tokens, H, dtype=torch.float32).abs()
    beta = rand(1, padded_tokens * M, H).sigmoid()
    if padded_tokens > REAL_TOKENS:
        # What a real step's padded tail holds: whatever the buffer had before.
        for t, stride in ((q, 1), (g, 1), (k, M), (v, M), (beta, M)):
            t[:, REAL_TOKENS * stride :] = pad_fill
    return q, k, v, g, beta


def _chain(q, k, v, g, beta, state, state_indices, desc):
    """`chunk_gated_delta_product_varlen`, unrolled so every stage is visible."""
    B, T, _, _ = q.shape
    cu_seqlens = desc["cu_seqlens"]
    chunk_indices = desc["chunk_indices"]
    chunk_indices_dp = desc["chunk_indices_dp"]
    cu_seqlens_dp = cu_seqlens * M

    qn = l2norm_fwd(q)
    kn = l2norm_fwd(k)

    g_interleaved = g.new_zeros(B, T, M, H, dtype=torch.float32)
    g_interleaved[:, :, 0] = g
    g_interleaved = g_interleaved.view(B, T * M, H).contiguous()
    g_cs = chunk_local_cumsum(
        g,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices,
    )
    g_dp_cs = chunk_local_cumsum(
        g_interleaved,
        chunk_size=CHUNK_SIZE,
        scale=RCP_LN2,
        cu_seqlens=cu_seqlens_dp,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )

    A = chunk_scaled_dot_kkt_fwd(
        k=kn,
        g=g_dp_cs,
        beta=beta,
        cu_seqlens=cu_seqlens_dp,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )
    Ai = solve_tril(
        A=A, cu_seqlens=cu_seqlens_dp, chunk_indices=chunk_indices_dp, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=kn,
        v=v,
        beta=beta,
        A=Ai,
        g=g_dp_cs,
        cu_seqlens=cu_seqlens_dp,
        chunk_indices=chunk_indices_dp,
    )
    h, v_new, _ = chunk_gated_delta_product_fwd_h(
        k=kn,
        w=w,
        u=u,
        g=g_dp_cs,
        initial_state=None,
        cu_seqlens=cu_seqlens_dp,
        num_householder=M,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
        chunk_offsets=desc["chunk_offsets"],
        state=state,
        state_indices=state_indices,
    )
    o = chunk_gated_delta_product_fwd_o(
        q=qn,
        k=kn,
        v=v_new,
        h=h,
        g=g_cs,
        scale=K**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
        num_householder=M,
        chunk_indices=chunk_indices,
    )
    # (name, tensor, stream): "t" indexes the unexpanded stream, "dp" the
    # Householder-expanded one, "chunk" the per-chunk state buffer, "" the whole
    # tensor.
    return [
        ("q_l2norm", qn, "t"),
        ("k_l2norm", kn, "dp"),
        ("g_cumsum", g_cs, "t"),
        ("g_dp_cumsum", g_dp_cs, "dp"),
        ("A", A, "dp"),
        ("Ai", Ai, "dp"),
        ("w", w, "dp"),
        ("u", u, "dp"),
        ("h", h, "chunk"),
        ("v_new", v_new, "dp"),
        ("o", o, "t"),
        ("state", state, ""),
    ]


def _real_view(tensor, stream, num_real_chunks):
    if stream == "t":
        return tensor[:, :REAL_TOKENS]
    if stream == "dp":
        return tensor[:, : REAL_TOKENS * M]
    if stream == "chunk":
        return tensor[:, :num_real_chunks]
    return tensor


def _report(label, ref_stages, cmp_stages, num_real_chunks):
    """Print the first stage whose real region differs, and everything after it."""
    print(f"\n=== {label} ===")
    first_bad = None
    for (name, ref, stream), (_, cmp_, _) in zip(ref_stages, cmp_stages):
        a = _real_view(ref, stream, num_real_chunks).float()
        b = _real_view(cmp_, stream, num_real_chunks).float()
        finite = torch.isfinite(a) & torch.isfinite(b)
        d = (a - b).abs()
        d = torch.where(finite, d, torch.zeros_like(d))
        worst = float(d.max()) if d.numel() else 0.0
        nan_mismatch = int((torch.isfinite(a) != torch.isfinite(b)).sum())
        scale = float(a[finite].abs().max()) if finite.any() else 0.0
        print(
            f"  {name:<12} max|diff|={worst:.3e}  scale={scale:.3e}  "
            f"nonfinite_mismatch={nan_mismatch}"
        )
        if (worst > 0 or nan_mismatch > 0) and first_bad is None:
            first_bad = name
            print(f"      ^^^ first divergent stage: {name}")
    print(f"  first divergent stage: {first_bad}")
    return first_bad


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestGDPPrefillStageBisect:
    """Localize the divergence to a single stage of the prefill chain."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _state_and_slots(padded_reqs, device, seed=3):
        gen = torch.Generator(device=device).manual_seed(seed)
        state = torch.randn(
            NUM_SLOTS, H, K, V, generator=gen, device=device, dtype=torch.bfloat16
        )
        slots = torch.full((padded_reqs,), -1, device=device, dtype=torch.int32)
        slots[: len(REAL_SLOTS)] = torch.tensor(REAL_SLOTS, device=device, dtype=torch.int32)
        return state, slots

    @pytest.mark.parametrize("padded_tokens", [432, 512, 1024], ids=["tight", "512", "1024"])
    @pytest.mark.parametrize("padded_reqs", [4, 8], ids=["reqs4", "reqs8"])
    @pytest.mark.parametrize("pad_fill", [0.0, 7.0, float("nan")], ids=["zero", "big", "nan"])
    @torch.inference_mode()
    def test_stage_padding_invariance(self, padded_tokens, padded_reqs, pad_fill):
        """Padding the batch must not move a single bit of the real region."""
        device = torch.cuda.current_device()
        num_real_chunks = sum((n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in SEQ_LENS)

        # Reference: exactly the real requests, no token or request padding.
        q, k, v, g, beta = _inputs(max(padded_tokens, REAL_TOKENS), pad_fill)
        _, ref_desc = _descriptors(SEQ_LENS, REAL_TOKENS, len(SEQ_LENS), device)
        ref_state, ref_slots = self._state_and_slots(len(SEQ_LENS), device)
        ref_stages = _chain(
            q[:, :REAL_TOKENS].contiguous(),
            k[:, : REAL_TOKENS * M].contiguous(),
            v[:, : REAL_TOKENS * M].contiguous(),
            g[:, :REAL_TOKENS].contiguous(),
            beta[:, : REAL_TOKENS * M].contiguous(),
            ref_state,
            ref_slots,
            ref_desc,
        )

        # Padded: the same real data plus a garbage tail and -1 request slots.
        _, pad_desc = _descriptors(SEQ_LENS, padded_tokens, padded_reqs, device)
        pad_state, pad_slots = self._state_and_slots(padded_reqs, device)
        pad_stages = _chain(q, k, v, g, beta, pad_state, pad_slots, pad_desc)

        first_bad = _report(
            f"padding invariance: tokens {REAL_TOKENS}->{padded_tokens}, "
            f"reqs {len(SEQ_LENS)}->{padded_reqs}, pad_fill={pad_fill}",
            ref_stages,
            pad_stages,
            num_real_chunks,
        )
        assert first_bad is None, f"padding changed the real region, starting at {first_bad}"

    @torch.inference_mode()
    def test_stage_graph_replay(self):
        """A graph captured at the padded shape must replay bit-identically."""
        device = torch.cuda.current_device()
        padded_tokens, padded_reqs = 512, 8
        num_real_chunks = sum((n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in SEQ_LENS)

        q, k, v, g, beta = _inputs(padded_tokens, 0.0)
        _, desc = _descriptors(SEQ_LENS, padded_tokens, padded_reqs, device)
        state, slots = self._state_and_slots(padded_reqs, device)
        state_init = state.clone()

        eager_state = state_init.clone()
        eager_stages = [
            (name, t.clone(), stream)
            for name, t, stream in _chain(q, k, v, g, beta, eager_state, slots, desc)
        ]

        # Warm up on a side stream so every Triton autotune + JIT happens before
        # capture, then restore the cache the warmup mutated.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _chain(q, k, v, g, beta, state, slots, desc)
        torch.cuda.current_stream().wait_stream(stream)
        state.copy_(state_init)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_stages = _chain(q, k, v, g, beta, state, slots, desc)
        state.copy_(state_init)
        graph.replay()
        torch.cuda.synchronize()

        first_bad = _report(
            "graph replay vs eager (identical inputs)",
            eager_stages,
            graph_stages,
            num_real_chunks,
        )
        assert first_bad is None, f"graph replay changed the result, starting at {first_bad}"
