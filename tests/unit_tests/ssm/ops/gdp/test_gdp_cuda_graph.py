# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA-graph correctness tests for the Gated Delta Product inference path.

Covers both halves of the forked kernel set, mirroring what exists for the
Mamba2 inference path (`test_causal_conv1d_triton.py` for a forked kernel,
plus the padding semantics the dynamic-batching path relies on):

1. `ops/gdp/fused_recurrent.py` -- the decode recurrence, against a naive
   PyTorch reference, including slot-indexed state and `-1` padding slots.
2. `ops/gdp/metadata.py` -- the chunk descriptors that replace FLA's
   host-synchronizing `prepare_chunk_indices`.
3. `ops/gdp/chunk.py` -- the chunked prefill, against the pip FLA kernels the
   training path still uses.
4. `GatedDeltaProductMixer.ssm_decode` replayed from a CUDA graph captured at
   a padded batch size, against the same call run eagerly.

The padding contract is what makes CUDA graphs work: a graph is captured for a
rounded-up batch shape, and steps with fewer real requests mark the leftover
rows with `-1` in `batch_indices` (and, for prefill, as zero-length
sequences). Those rows must produce zero output and must not read or write any
state slot.

Run with::

    torchrun --nproc_per_node=1 -m pytest \\
        tests/unit_tests/ssm/test_gdp_cuda_graph.py -m internal -v
"""

from __future__ import annotations

import pytest
import torch

from megatron.core.ssm.ops.gdp.chunk import chunk_gated_delta_product_varlen
from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update
from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

try:
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_product import (
        GatedDeltaProductMixer,
        GatedDeltaProductMixerSubmodules,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    HAVE_MIXER_DEPS = True
except ImportError:
    HAVE_MIXER_DEPS = False

try:
    import einops  # noqa: F401
    import mamba_ssm  # noqa: F401

    HAVE_MAMBA_DEPS = True
except ImportError:
    HAVE_MAMBA_DEPS = False

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


pytestmark = [pytest.mark.internal]

# The chunk-descriptor builder is pure Python and runs anywhere; everything that
# launches a kernel needs a GPU.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module", autouse=True)
def distributed_environment():
    """Bind every rank to its own GPU before anything allocates or captures.

    `torch.cuda.set_device()` lives inside `Utils.initialize_distributed()`,
    so without this each rank of a multi-rank run would allocate on GPU 0 and
    four processes would capture CUDA graphs on the same device. Module-scoped
    (not per-method) so the current device never changes mid-process, which
    would strand earlier allocations and the graph memory pool on another GPU.
    """
    if not torch.cuda.is_available():
        # The chunk-descriptor tests are pure Python and need no GPU or ranks.
        yield
        return
    Utils.initialize_model_parallel()
    clear_nvte_env_vars()
    yield
    Utils.destroy_model_parallel()


# --------------------------------------------------------------------------- #
# Reference implementation
# --------------------------------------------------------------------------- #


def _gated_delta_rule_ref(q, k, v, g, beta, state, state_indices, use_qk_l2norm, scale=None):
    """Naive per-request gated delta rule; updates `state` in place.

    Deliberately written as an explicit Python loop over requests, heads and
    time steps so it is obviously independent of the Triton kernel's blocking.
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if scale is None:
        scale = K**-0.5
    o = torch.zeros_like(v)

    for b in range(B):
        slot = b if state_indices is None else int(state_indices[b])
        if slot < 0:
            # Padding request: zero output, no state access.
            continue
        for i_hv in range(HV):
            i_h = i_hv // (HV // H)
            h = state[slot, i_hv].float()
            for t in range(T):
                q_t = q[b, t, i_h].float()
                k_t = k[b, t, i_h].float()
                v_t = v[b, t, i_hv].float()
                if use_qk_l2norm:
                    q_t = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                    k_t = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                q_t = q_t * scale
                h = h * torch.exp(g[b, t, i_hv].float())
                v_new = beta[b, t, i_hv].float() * (v_t - (h * k_t[:, None]).sum(0))
                h = h + k_t[:, None] * v_new
                o[b, t, i_hv] = (h * q_t[:, None]).sum(0).to(o.dtype)
            state[slot, i_hv] = h.to(state.dtype)
    return o


def _capture(fn, restore):
    """Warm up, capture `fn` into a CUDA graph, and undo warmup side effects.

    `restore` maps a mutated tensor to the contents it should be reset to; the
    warmup iterations run the real kernels (and so mutate the state caches),
    while graph capture itself only records.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    for tensor, contents in restore.items():
        tensor.copy_(contents)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = fn()
    for tensor, contents in restore.items():
        tensor.copy_(contents)
    return graph, static_out


def _to_dev(values, shape):
    """Materialize a flat descriptor list as an int32 CUDA tensor."""
    return torch.tensor(values, device="cuda", dtype=torch.int32).view(*shape)


def _random_inputs(B, T, H, HV, K, V, num_slots, device="cuda", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    # Decays are log-space and non-positive, as produced by the mixer.
    g = -torch.rand(B, T, HV, device=device, dtype=torch.float32)
    beta = torch.rand(B, T, HV, device=device, dtype=dtype).sigmoid()
    state = torch.randn(num_slots, HV, K, V, device=device, dtype=dtype)
    return q, k, v, g, beta, state


# --------------------------------------------------------------------------- #
# Kernel-level tests
# --------------------------------------------------------------------------- #


@requires_cuda
class TestFusedRecurrentGatedDeltaRuleUpdate:
    """Forked, slot-indexed recurrent kernel used by the GDP decode step."""

    SHAPE = dict(B=6, T=3, H=2, HV=4, K=16, V=32, num_slots=11)

    @pytest.mark.parametrize("use_qk_l2norm", [False, True])
    def test_matches_reference_with_slot_indices(self, use_qk_l2norm):
        """Permuted (non-identity) slots read and write the right cache rows."""
        q, k, v, g, beta, state = _random_inputs(**self.SHAPE)
        # Slots deliberately out of order and not covering the whole cache.
        indices = torch.tensor([7, 0, 3, 10, 1, 5], device="cuda", dtype=torch.int32)
        state_ref = state.clone()

        out = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            state=state,
            g=g,
            beta=beta,
            state_indices=indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        out_ref = _gated_delta_rule_ref(q, k, v, g, beta, state_ref, indices, use_qk_l2norm)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-4)

    def test_no_state_indices_uses_identity_slots(self):
        q, k, v, g, beta, state = _random_inputs(**{**self.SHAPE, "num_slots": self.SHAPE["B"]})
        state_ref = state.clone()

        out = fused_recurrent_gated_delta_rule_update(
            q, k, v, state=state, g=g, beta=beta, state_indices=None
        )
        out_ref = _gated_delta_rule_ref(q, k, v, g, beta, state_ref, None, False)

        torch.testing.assert_close(out, out_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-4)

    def test_padding_slots_zero_output_and_preserve_state(self):
        """`-1` rows: zero output, untouched cache, real rows unaffected."""
        q, k, v, g, beta, state = _random_inputs(**self.SHAPE)
        # Rows 1 and 4 are padding; the rest map to real slots.
        indices = torch.tensor([7, -1, 3, 10, -1, 5], device="cuda", dtype=torch.int32)
        state_before = state.clone()
        state_ref = state.clone()

        out = fused_recurrent_gated_delta_rule_update(
            q, k, v, state=state, g=g, beta=beta, state_indices=indices
        )
        out_ref = _gated_delta_rule_ref(q, k, v, g, beta, state_ref, indices, False)

        pad_rows = [1, 4]
        real_rows = [0, 2, 3, 5]
        assert torch.count_nonzero(out[pad_rows]) == 0, "padding rows must produce zero output"
        torch.testing.assert_close(out[real_rows], out_ref[real_rows], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state, state_ref, atol=1e-4, rtol=1e-4)

        # Slots not named by any real row are bit-identical to before the call.
        untouched = [s for s in range(self.SHAPE["num_slots"]) if s not in {7, 3, 10, 5}]
        torch.testing.assert_close(
            state[untouched], state_before[untouched], atol=0, rtol=0, equal_nan=True
        )

    def test_padding_does_not_change_real_rows(self):
        """A padded batch gives the real rows exactly what an unpadded one does."""
        shape = dict(self.SHAPE)
        q, k, v, g, beta, state = _random_inputs(**shape)
        real = 4
        padded_indices = torch.tensor([2, 6, 0, 9, -1, -1], device="cuda", dtype=torch.int32)

        state_padded = state.clone()
        out_padded = fused_recurrent_gated_delta_rule_update(
            q, k, v, state=state_padded, g=g, beta=beta, state_indices=padded_indices
        )

        state_real = state.clone()
        out_real = fused_recurrent_gated_delta_rule_update(
            q[:real],
            k[:real],
            v[:real],
            state=state_real,
            g=g[:real],
            beta=beta[:real],
            state_indices=padded_indices[:real],
        )

        torch.testing.assert_close(out_padded[:real], out_real, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(state_padded, state_real, atol=0, rtol=0, equal_nan=True)

    @pytest.mark.skipif(not HAVE_FLA, reason="parity check requires flash-linear-attention")
    def test_matches_upstream_fla(self):
        """Fork parity with the pip FLA kernel the training path still uses."""
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule

        q, k, v, g, beta, state = _random_inputs(**self.SHAPE)
        indices = torch.tensor([7, 0, 3, 10, 1, 5], device="cuda", dtype=torch.int32)

        state_fork = state.clone()
        out_fork = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            state=state_fork,
            g=g,
            beta=beta,
            state_indices=indices,
            use_qk_l2norm_in_kernel=True,
        )

        # Upstream needs the initial states gathered in, and returns the final
        # states to be scattered back out -- exactly what the fork removes.
        out_fla, final_state = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=state[indices.long()],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        state_fla = state.clone()
        state_fla[indices.long()] = final_state.to(state.dtype)

        torch.testing.assert_close(out_fork, out_fla, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_fork, state_fla, atol=1e-4, rtol=1e-4)

    def test_cuda_graph_capture_and_replay(self):
        """Capture once at the padded size, replay with new data and new slots."""
        q, k, v, g, beta, state = _random_inputs(**self.SHAPE)
        indices = torch.tensor([7, 0, 3, 10, -1, -1], device="cuda", dtype=torch.int32)
        state_init = state.clone()

        def run():
            return fused_recurrent_gated_delta_rule_update(
                q,
                k,
                v,
                state=state,
                g=g,
                beta=beta,
                state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )

        graph, static_out = _capture(run, restore={state: state_init})

        # Replay 1: same inputs as the eager reference.
        state.copy_(state_init)
        graph.replay()
        torch.cuda.synchronize()
        out_replay, state_replay = static_out.clone(), state.clone()

        state_eager = state_init.clone()
        out_eager = _gated_delta_rule_ref(q, k, v, g, beta, state_eager, indices, True)
        torch.testing.assert_close(out_replay, out_eager, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state_replay, state_eager, atol=1e-4, rtol=1e-4)

        # Replay 2: new token data and a different number of padded rows, written
        # into the same static buffers. This is what a decode step does.
        q2, k2, v2, g2, beta2, _ = _random_inputs(**self.SHAPE, seed=7)
        indices2 = torch.tensor([1, 4, -1, -1, -1, -1], device="cuda", dtype=torch.int32)
        for dst, src in ((q, q2), (k, k2), (v, v2), (g, g2), (beta, beta2)):
            dst.copy_(src)
        indices.copy_(indices2)
        state.copy_(state_init)
        graph.replay()
        torch.cuda.synchronize()

        state_eager2 = state_init.clone()
        out_eager2 = _gated_delta_rule_ref(q, k, v, g, beta, state_eager2, indices, True)
        torch.testing.assert_close(static_out, out_eager2, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(state, state_eager2, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- #
# Mixer-level tests
# --------------------------------------------------------------------------- #


@requires_cuda
@pytest.mark.skipif(not HAVE_MIXER_DEPS, reason="GDP mixer requires transformer_engine")
@pytest.mark.skipif(not HAVE_MAMBA_DEPS, reason="GDP mixer requires mamba_ssm + einops")
@pytest.mark.skipif(not HAVE_FLA, reason="GDP mixer requires fla")
class TestGDPCudaGraphDecode:
    """`GatedDeltaProductMixer.ssm_decode` under CUDA-graph capture."""

    NUM_SLOTS = 12
    PADDED_REQUESTS = 6
    REAL_REQUESTS = 4

    def setup_method(self, method):
        model_parallel_cuda_manual_seed(123)

    def _build_mixer(self):
        # Seed here, not just in setup_method: model_parallel_cuda_manual_seed
        # derives per-rank seeds, so without this every rank builds a different
        # mixer and the tests compare against different numerics per rank.
        torch.manual_seed(1234)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_query_groups=4,
            ffn_hidden_size=128,
            normalization="RMSNorm",
            bf16=True,
            params_dtype=torch.bfloat16,
            mamba_num_heads=4,
            mamba_head_dim=16,
            mamba_num_groups=4,
            mamba_state_dim=16,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = GatedDeltaProductMixer(
            config=config,
            submodules=GatedDeltaProductMixerSubmodules(
                in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
            ),
            d_model=config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
            name="decoder.layers.0.mixer",
        )
        return mixer.cuda().bfloat16(), config

    def _decode_inputs(self, mixer, num_requests, seed=0):
        """Random decode-step activations plus fresh conv/SSM caches."""
        torch.manual_seed(seed)
        proj_dim = mixer.in_proj.weight.shape[0]
        # `ssm_decode` takes the mixin's batch-first layout: [n, seq_len, proj_dim].
        zVKQba = torch.randn(num_requests, 1, proj_dim, device="cuda", dtype=torch.bfloat16)
        conv_shape, ssm_shape = mixer.mamba_state_shapes_per_request()
        conv_state = torch.randn(self.NUM_SLOTS, *conv_shape, device="cuda", dtype=torch.bfloat16)
        ssm_state = torch.randn(self.NUM_SLOTS, *ssm_shape, device="cuda", dtype=torch.bfloat16)
        return zVKQba, conv_state, ssm_state

    def test_padded_decode_matches_unpadded(self):
        """Padding rows are inert: real rows and caches match a tight batch."""
        mixer, _ = self._build_mixer()
        padded, real = self.PADDED_REQUESTS, self.REAL_REQUESTS
        zVKQba, conv_state, ssm_state = self._decode_inputs(mixer, padded)
        indices = torch.full((padded,), -1, device="cuda", dtype=torch.int32)
        indices[:real] = torch.tensor([5, 0, 9, 2], device="cuda", dtype=torch.int32)

        conv_padded, ssm_padded = conv_state.clone(), ssm_state.clone()
        conv_real, ssm_real = conv_state.clone(), ssm_state.clone()

        with torch.no_grad():
            y_padded = mixer.ssm_decode(zVKQba, conv_padded, ssm_padded, indices)
            y_real = mixer.ssm_decode(zVKQba[:real], conv_real, ssm_real, indices[:real])

        assert y_padded.shape == (padded, 1, y_real.shape[-1])
        assert torch.count_nonzero(y_padded[real:]) == 0, "padded rows must be zeroed"
        torch.testing.assert_close(y_padded[:real], y_real, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(conv_padded, conv_real, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(ssm_padded, ssm_real, atol=0, rtol=0, equal_nan=True)

    def test_cuda_graph_replay_matches_eager(self):
        """A graph captured at the padded size replays bit-identically."""
        mixer, _ = self._build_mixer()
        padded, real = self.PADDED_REQUESTS, self.REAL_REQUESTS
        zVKQba, conv_state, ssm_state = self._decode_inputs(mixer, padded)
        indices = torch.full((padded,), -1, device="cuda", dtype=torch.int32)
        indices[:real] = torch.tensor([5, 0, 9, 2], device="cuda", dtype=torch.int32)

        conv_init, ssm_init = conv_state.clone(), ssm_state.clone()

        # Eager reference on copies of the caches.
        conv_eager, ssm_eager = conv_init.clone(), ssm_init.clone()
        with torch.no_grad():
            y_eager = mixer.ssm_decode(zVKQba, conv_eager, ssm_eager, indices).clone()

        with torch.no_grad():
            graph, y_static = _capture(
                lambda: mixer.ssm_decode(zVKQba, conv_state, ssm_state, indices),
                restore={conv_state: conv_init, ssm_state: ssm_init},
            )
            graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(y_static, y_eager, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(conv_state, conv_eager, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(ssm_state, ssm_eager, atol=0, rtol=0, equal_nan=True)

    def test_cuda_graph_replay_with_new_batch_composition(self):
        """One captured graph serves decode steps with different real counts."""
        mixer, _ = self._build_mixer()
        padded = self.PADDED_REQUESTS
        zVKQba, conv_state, ssm_state = self._decode_inputs(mixer, padded)
        indices = torch.full((padded,), -1, device="cuda", dtype=torch.int32)
        indices[:4] = torch.tensor([5, 0, 9, 2], device="cuda", dtype=torch.int32)

        conv_init, ssm_init = conv_state.clone(), ssm_state.clone()
        with torch.no_grad():
            graph, y_static = _capture(
                lambda: mixer.ssm_decode(zVKQba, conv_state, ssm_state, indices),
                restore={conv_state: conv_init, ssm_state: ssm_init},
            )

        # A later step: different activations, different slots, fewer requests.
        zVKQba_next, _, _ = self._decode_inputs(mixer, padded, seed=13)
        indices_next = torch.full((padded,), -1, device="cuda", dtype=torch.int32)
        indices_next[:2] = torch.tensor([3, 11], device="cuda", dtype=torch.int32)

        conv_eager, ssm_eager = conv_init.clone(), ssm_init.clone()
        with torch.no_grad():
            y_eager = mixer.ssm_decode(zVKQba_next, conv_eager, ssm_eager, indices_next).clone()

        zVKQba.copy_(zVKQba_next)
        indices.copy_(indices_next)
        conv_state.copy_(conv_init)
        ssm_state.copy_(ssm_init)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(y_static, y_eager, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(conv_state, conv_eager, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(ssm_state, ssm_eager, atol=0, rtol=0, equal_nan=True)
        assert torch.count_nonzero(y_static[:, 2:]) == 0, "padded rows must be zeroed"


# --------------------------------------------------------------------------- #
# Prefill: chunk descriptors
# --------------------------------------------------------------------------- #


class TestGDPChunkDescriptors:
    """Pure-Python builder for the descriptors the forked chunk kernels read.

    No CUDA needed: this is the piece that replaces FLA's `prepare_chunk_indices`
    (which synchronizes on the device and returns a data-dependent shape).
    """

    def test_descriptors_match_per_sequence_chunk_counts(self):
        seq_lens = [130, 64, 7, 0]
        cu_seqlens = [0]
        for n in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + n)
        num_householder, padded_tokens = 3, 256

        indices, indices_dp, offsets, num_chunks, num_chunks_dp = build_gdp_chunk_descriptors(
            cu_seqlens, len(seq_lens), num_householder, padded_tokens
        )
        pairs = [tuple(indices[i : i + 2]) for i in range(0, len(indices), 2)]
        pairs_dp = [tuple(indices_dp[i : i + 2]) for i in range(0, len(indices_dp), 2)]

        # Unexpanded: ceil(L/64) chunks per sequence, in sequence order.
        expected = []
        for i, n in enumerate(seq_lens):
            expected.extend((i, j) for j in range((n + 63) // 64))
        assert pairs[: len(expected)] == expected
        assert offsets == [0, 3, 4, 5, 5]
        assert offsets[-1] == len(expected)

        # Expanded: ceil(L*M/64), which is *not* M * ceil(L/64) in general.
        expected_dp = []
        for i, n in enumerate(seq_lens):
            expected_dp.extend((i, j) for j in range((n * num_householder + 63) // 64))
        assert pairs_dp[: len(expected_dp)] == expected_dp
        assert (7 * 3 + 63) // 64 != 3 * ((7 + 63) // 64), "the two chunkings must differ here"

    def test_padding_rows_are_addressed_out_of_range(self):
        """Padding descriptor rows must land past the end of sequence 0 so their
        programs load zeros and store nothing."""
        seq_lens = [96, 32]
        cu_seqlens = [0, 96, 128]
        num_householder, padded_tokens = 2, 192

        indices, indices_dp, _, num_chunks, num_chunks_dp = build_gdp_chunk_descriptors(
            cu_seqlens, len(seq_lens), num_householder, padded_tokens
        )
        assert len(indices) // 2 == num_chunks
        assert len(indices_dp) // 2 == num_chunks_dp

        real = sum((n + 63) // 64 for n in seq_lens)
        for i in range(real, num_chunks):
            seq, chunk = indices[2 * i], indices[2 * i + 1]
            assert chunk * 64 >= seq_lens[seq], "padding chunk overlaps a real chunk"

        real_dp = sum((n * num_householder + 63) // 64 for n in seq_lens)
        for i in range(real_dp, num_chunks_dp):
            seq, chunk = indices_dp[2 * i], indices_dp[2 * i + 1]
            assert chunk * 64 >= seq_lens[seq] * num_householder, "padding overlaps a real chunk"

    def test_buffer_sizing_covers_worst_case(self):
        max_tokens, max_requests, num_householder = 512, 8, 4
        max_chunks, max_chunks_dp = max_gdp_chunk_counts(max_tokens, max_requests, num_householder)
        # Worst case for chunk count: as many short sequences as possible.
        seq_lens = [1] * (max_requests - 1)
        seq_lens.append(max_tokens - sum(seq_lens))
        assert sum((n + 63) // 64 for n in seq_lens) <= max_chunks
        assert sum((n * num_householder + 63) // 64 for n in seq_lens) <= max_chunks_dp


# --------------------------------------------------------------------------- #
# Prefill: forked chunk kernels
# --------------------------------------------------------------------------- #


def _prefill_inputs(seq_lens, H, K, V, num_householder, num_slots, padded_tokens=None, seed=0):
    """Random packed prefill activations plus the matching chunk descriptors."""
    torch.manual_seed(seed)
    total = sum(seq_lens)
    padded_tokens = total if padded_tokens is None else padded_tokens
    M = num_householder

    q = torch.randn(1, padded_tokens, H, K, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, padded_tokens * M, H, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, padded_tokens * M, H, V, device="cuda", dtype=torch.bfloat16)
    g = -torch.rand(1, padded_tokens, H, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, padded_tokens * M, H, device="cuda", dtype=torch.bfloat16).sigmoid()

    cu_list = [0]
    for n in seq_lens:
        cu_list.append(cu_list[-1] + n)
    cu_seqlens = torch.tensor(cu_list, device="cuda", dtype=torch.int32)

    indices, indices_dp, offsets, num_chunks, num_chunks_dp = build_gdp_chunk_descriptors(
        cu_list, len(seq_lens), M, padded_tokens
    )
    descriptors = dict(
        cu_seqlens=cu_seqlens,
        chunk_indices=_to_dev(indices, (num_chunks, 2)),
        chunk_indices_dp=_to_dev(indices_dp, (num_chunks_dp, 2)),
        chunk_offsets=_to_dev(offsets, (len(offsets),)),
    )
    state = torch.randn(num_slots, H, K, V, device="cuda", dtype=torch.bfloat16)
    return q, k, v, g, beta, state, descriptors


@requires_cuda
class TestChunkGatedDeltaProductVarlen:
    """Forked chunked prefill: parity, padding semantics, graph capture."""

    SHAPE = dict(H=4, K=64, V=64, num_householder=2, num_slots=8)

    @pytest.mark.skipif(not HAVE_FLA, reason="parity check requires flash-linear-attention")
    def test_matches_upstream_fla(self):
        """The fork must agree with the pip kernels the training path still uses."""
        from fla.ops.gated_delta_product import chunk_gated_delta_product

        seq_lens = [128, 65, 32]
        q, k, v, g, beta, state, desc = _prefill_inputs(seq_lens, **self.SHAPE)
        M = self.SHAPE["num_householder"]
        slots = torch.tensor([3, 0, 6], device="cuda", dtype=torch.int32)

        state_fork = state.clone()
        o_fork = chunk_gated_delta_product_varlen(
            q,
            k,
            v,
            g=g,
            beta=beta,
            num_householder=M,
            state=state_fork,
            state_indices=slots,
            use_qk_l2norm_in_kernel=True,
            **desc,
        )

        o_fla, final_state = chunk_gated_delta_product(
            q,
            k,
            v,
            g=g,
            beta=beta,
            num_householder=M,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=desc["cu_seqlens"].long(),
        )
        state_fla = state.clone()
        state_fla[slots.long()] = final_state.to(state.dtype)

        torch.testing.assert_close(o_fork, o_fla, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state_fork, state_fla, atol=2e-2, rtol=2e-2)

    def test_padding_requests_are_inert(self):
        """Zero-length padded requests with -1 slots change nothing."""
        real_lens = [96, 33]
        padded_lens = real_lens + [0, 0]
        padded_tokens = 192

        q, k, v, g, beta, state, desc = _prefill_inputs(
            padded_lens, padded_tokens=padded_tokens, **self.SHAPE
        )
        M = self.SHAPE["num_householder"]
        slots_padded = torch.tensor([5, 1, -1, -1], device="cuda", dtype=torch.int32)

        state_padded = state.clone()
        o_padded = chunk_gated_delta_product_varlen(
            q,
            k,
            v,
            g=g,
            beta=beta,
            num_householder=M,
            state=state_padded,
            state_indices=slots_padded,
            use_qk_l2norm_in_kernel=True,
            **desc,
        )

        # Same activations, but only the real requests and no token padding.
        real_tokens = sum(real_lens)
        cu_list = [0]
        for n in real_lens:
            cu_list.append(cu_list[-1] + n)
        indices, indices_dp, offsets, n_c, n_c_dp = build_gdp_chunk_descriptors(
            cu_list, len(real_lens), M, real_tokens
        )
        state_real = state.clone()
        o_real = chunk_gated_delta_product_varlen(
            q[:, :real_tokens].contiguous(),
            k[:, : real_tokens * M].contiguous(),
            v[:, : real_tokens * M].contiguous(),
            g=g[:, :real_tokens].contiguous(),
            beta=beta[:, : real_tokens * M].contiguous(),
            num_householder=M,
            cu_seqlens=torch.tensor(cu_list, device="cuda", dtype=torch.int32),
            chunk_indices=_to_dev(indices, (n_c, 2)),
            chunk_indices_dp=_to_dev(indices_dp, (n_c_dp, 2)),
            chunk_offsets=_to_dev(offsets, (len(offsets),)),
            state=state_real,
            state_indices=slots_padded[: len(real_lens)],
            use_qk_l2norm_in_kernel=True,
        )

        torch.testing.assert_close(
            o_padded[:, :real_tokens], o_real, atol=0, rtol=0, equal_nan=True
        )
        assert torch.count_nonzero(o_padded[:, real_tokens:]) == 0, "padded tokens must be zero"
        torch.testing.assert_close(state_padded, state_real, atol=0, rtol=0, equal_nan=True)

        # Slots owned by no real request are byte-identical to before the call.
        untouched = [s for s in range(self.SHAPE["num_slots"]) if s not in {5, 1}]
        torch.testing.assert_close(
            state_padded[untouched], state[untouched], atol=0, rtol=0, equal_nan=True
        )

    def test_cuda_graph_replay_matches_eager(self):
        """Capture at the padded shape, then replay with a different real batch."""
        padded_lens = [96, 33, 0, 0]
        padded_tokens = 192
        q, k, v, g, beta, state, desc = _prefill_inputs(
            padded_lens, padded_tokens=padded_tokens, **self.SHAPE
        )
        M = self.SHAPE["num_householder"]
        slots = torch.tensor([5, 1, -1, -1], device="cuda", dtype=torch.int32)
        state_init = state.clone()

        def run():
            return chunk_gated_delta_product_varlen(
                q,
                k,
                v,
                g=g,
                beta=beta,
                num_householder=M,
                state=state,
                state_indices=slots,
                use_qk_l2norm_in_kernel=True,
                **desc,
            )

        state_eager = state_init.clone()
        with torch.no_grad():
            o_eager = chunk_gated_delta_product_varlen(
                q,
                k,
                v,
                g=g,
                beta=beta,
                num_householder=M,
                state=state_eager,
                state_indices=slots,
                use_qk_l2norm_in_kernel=True,
                **desc,
            ).clone()

            graph, o_static = _capture(run, restore={state: state_init})
            state.copy_(state_init)
            graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(o_static, o_eager, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(state, state_eager, atol=0, rtol=0, equal_nan=True)
