# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests confirming CUDA-graph-unsafe patterns in the NVLS token dispatcher.

Three bugs where Python-side values are frozen at CUDA-graph capture time:

Bug 1 — Graph misalignment across EP ranks:
    When EP ranks independently select different captured CUDA graphs
    (because DP replicas have different batch sizes), the multicast
    operations inside those graphs are incompatible, causing a hang.
    Fix: EP-wide reduce-max of token counts before graph selection.

Bug 2 — Frozen local_tokens in _fused_metadata_kernel:
    local_tokens is passed as a Python int to the Triton kernel. Under
    CUDA graph replay the kernel perpetually writes the warmup value,
    so step_metadata (valid_tokens, rank_token_offset, ep_max_tokens)
    is wrong for every post-warmup step.
    Fix: read local_tokens from a pre-allocated GPU buffer.

Bug 3 — Frozen hidden_shape / _local_tokens on dispatcher:
    dispatch_preprocess stores hidden_states.shape (Python tuple) and
    hidden_states.shape[0] (Python int). token_combine uses the int
    for the RSV output allocation; combine_postprocess uses the tuple
    for the final view. Both are baked into the graph.
    Fix: use GPU-side tensors for these values.

Run with:
    uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -xvs \
        tests/unit_tests/inference/test_nvls_cuda_graph_freezing.py
"""

import gc

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

_ENGINE_MAX_TOKENS = 512
_HIDDEN_SIZE = 128
_TOPK = 6
_NUM_EXPERTS = 8

_NANOV3_BASE = dict(
    num_layers=4,
    hidden_size=_HIDDEN_SIZE,
    ffn_hidden_size=_HIDDEN_SIZE,
    num_attention_heads=4,
    num_query_groups=2,
    num_moe_experts=_NUM_EXPERTS,
    moe_ffn_hidden_size=_HIDDEN_SIZE,
    moe_router_topk=_TOPK,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_topk_scaling_factor=2.5,
    moe_shared_expert_intermediate_size=256,
    moe_router_dtype='fp32',
    moe_shared_expert_overlap=False,
    moe_grouped_gemm=True,
    moe_token_dispatcher_type="alltoall",
    moe_aux_loss_coeff=0.01,
    normalization="RMSNorm",
    add_bias_linear=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    transformer_impl="inference_optimized",
    expert_tensor_parallel_size=1,
    use_cpu_initialization=True,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration_inference",
    moe_pad_experts_for_cuda_graph_inference=False,
    mamba_state_dim=128,
    mamba_head_dim=64,
    mamba_num_groups=8,
    mamba_num_heads=64,
)


def _make_config(**overrides):
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    params = {**_NANOV3_BASE, "attention_backend": AttnBackend.local, **overrides}
    return TransformerConfig(**params)


@pytest.mark.internal
class TestNVLSCudaGraphFreezing:
    """Reproduce CUDA-graph-unsafe patterns in the NVLS token dispatcher.

    Each test captures a CUDA graph containing the NVLS metadata + AGV + RSV
    path, then verifies that frozen Python-side values cause incorrect
    behaviour on replay with different token counts.
    """

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel(1, 1, expert_model_parallel_size=Utils.world_size)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    @classmethod
    def teardown_class(cls):
        NVLSAllGatherVDispatcher._delete_buffers()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        gc.collect()
        torch.cuda.empty_cache()

    # ── helpers ───────────────────────────────────────────────────────────

    def _make_dispatcher(self):
        from megatron.core.parallel_state import get_expert_model_parallel_group
        from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

        config = _make_config(expert_model_parallel_size=Utils.world_size)
        num_local_experts = config.num_moe_experts // Utils.world_size
        ep_rank = dist.get_rank() if Utils.world_size > 1 else 0
        local_expert_indices = [ep_rank * num_local_experts + i for i in range(num_local_experts)]
        ep_group = get_expert_model_parallel_group()

        NVLSAllGatherVDispatcher.allocate_buffers(
            per_rank_worst_case_token_count=_ENGINE_MAX_TOKENS,
            topk=_TOPK,
            hidden_size=_HIDDEN_SIZE,
            ep_group=ep_group,
        )

        return NVLSAllGatherVDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            pg_collection=get_default_pg_collection(),
            runs_metadata_sync=True,
        )

    def _warmup_and_capture(self, dispatcher, local_tokens):
        """Warmup 3× then capture a CUDA graph at the given token count.

        Returns (graph, combined_output_tensor, dispatched_hidden_tensor).
        The returned tensors are graph-owned; their contents are updated
        on each graph.replay().
        """
        hidden = torch.randn(local_tokens, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        probs = torch.randn(local_tokens, _TOPK, device="cuda", dtype=torch.float32)
        routing_map = torch.randint(0, _NUM_EXPERTS, (local_tokens, _TOPK), device="cuda")

        with torch.no_grad():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    dispatcher.routing_map = routing_map
                    dispatcher._local_tokens = local_tokens
                    d_h, _ = dispatcher.token_dispatch(hidden, probs)
                    dispatcher.token_combine(d_h.clone())
            torch.cuda.current_stream().wait_stream(s)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dispatcher.routing_map = routing_map
            dispatcher._local_tokens = local_tokens
            d_h, _ = dispatcher.token_dispatch(hidden, probs)
            combined = dispatcher.token_combine(d_h.clone())

        return graph, combined, d_h

    # ── Bug 1: graph misalignment across EP ranks ────────────────────────

    @pytest.mark.timeout(30)
    def test_graph_misalignment_hang(self):
        """Different EP ranks replaying different captured CUDA graphs.

        Captures graph_a (16 tokens) and graph_b (8 tokens) on ALL ranks,
        then even ranks replay graph_a while odd ranks replay graph_b.

        The multicast barrier inside multimem_all_gatherv_3tensor requires
        all EP ranks to execute the same graph.  When different ranks run
        graphs captured at different token counts, the multicast store
        patterns into the shared symmetric buffer are incompatible.

        Expected: hangs (test times out at 30 s) when the bug is present.
        When the EP token-count sync fix is in place, both ranks should
        always select the same graph and this test should pass.
        """
        if Utils.world_size < 2:
            pytest.skip("Requires EP >= 2 to test graph misalignment across ranks")

        dispatcher = self._make_dispatcher()

        graph_a, _, _ = self._warmup_and_capture(dispatcher, local_tokens=16)
        graph_b, _, _ = self._warmup_and_capture(dispatcher, local_tokens=8)

        rank = dist.get_rank()
        if rank % 2 == 0:
            graph_a.replay()
        else:
            graph_b.replay()

        torch.cuda.synchronize()

    # ── Bug 2: frozen local_tokens in metadata kernel ────────────────────

    def test_metadata_frozen_overwrites_runtime_update(self):
        """_fused_metadata_kernel's local_tokens arg is frozen at capture time.

        Procedure:
        1. Capture graph with local_tokens = 16.
        2. Before replay, manually write 7 into the symmetric metadata
           buffer at this rank's slot — simulating a runtime where the
           actual token count has changed.
        3. Replay the graph.
        4. Verify _step_metadata[2] (ep_max_tokens) == 16, proving the
           frozen metadata kernel overwrote the 7 with its baked-in 16.

        When the fix is in place (metadata kernel reads from a GPU buffer
        instead of a frozen int), step 4 would read 7 instead of 16.
        """
        dispatcher = self._make_dispatcher()
        ep_size = dispatcher.ep_size

        capture_tokens = 16
        graph, _, _ = self._warmup_and_capture(dispatcher, local_tokens=capture_tokens)

        rank = dist.get_rank() if ep_size > 1 else 0
        NVLSAllGatherVDispatcher._symm_metadata["tensor"][rank] = 7
        torch.cuda.synchronize()
        if ep_size > 1:
            dist.barrier()

        graph.replay()
        torch.cuda.synchronize()

        metadata = NVLSAllGatherVDispatcher._step_metadata.cpu()
        ep_max_tokens = metadata[2].item()
        valid_tokens = metadata[0].item()
        rank_offset = metadata[1].item()

        assert ep_max_tokens == capture_tokens, (
            f"Expected frozen ep_max_tokens={capture_tokens}, got {ep_max_tokens}. "
            f"If ep_max_tokens==7, the metadata kernel now reads from a GPU buffer "
            f"and this bug may be fixed."
        )
        assert valid_tokens == capture_tokens * ep_size, (
            f"Expected frozen valid_tokens={capture_tokens * ep_size}, "
            f"got {valid_tokens}."
        )
        assert rank_offset == rank * capture_tokens, (
            f"Expected frozen rank_offset={rank * capture_tokens}, "
            f"got {rank_offset}."
        )

    def test_metadata_frozen_across_graph_switches(self):
        """Each captured graph freezes its own local_tokens value.

        Captures graph_16 (16 tokens) and graph_8 (8 tokens) on all ranks.
        Replaying them alternately shows that ep_max_tokens always matches
        the graph's capture-time value, never the "current" token count.

        This means the inference engine cannot dynamically adjust the
        metadata by replaying a different graph — each graph is stuck at
        its own frozen token count.  When DP replicas pick different graphs
        (bug 1), they write different frozen values to the shared symmetric
        buffer, corrupting the offsets.
        """
        dispatcher = self._make_dispatcher()

        graph_16, _, _ = self._warmup_and_capture(dispatcher, local_tokens=16)
        graph_8, _, _ = self._warmup_and_capture(dispatcher, local_tokens=8)

        graph_16.replay()
        torch.cuda.synchronize()
        assert NVLSAllGatherVDispatcher._step_metadata[2].item() == 16

        graph_8.replay()
        torch.cuda.synchronize()
        assert NVLSAllGatherVDispatcher._step_metadata[2].item() == 8

        graph_16.replay()
        torch.cuda.synchronize()
        assert NVLSAllGatherVDispatcher._step_metadata[2].item() == 16, (
            "ep_max_tokens should revert to 16 when graph_16 is replayed, "
            "because local_tokens=16 is frozen in that graph's metadata kernel."
        )

    # ── Bug 3: frozen _local_tokens / hidden_shape ───────────────────────

    def test_combine_output_size_frozen(self):
        """token_combine output allocation uses frozen self._local_tokens.

        The RSV output is always [capture_tokens, hidden_size], regardless
        of the actual batch size at replay time, because torch.empty() in
        token_combine was captured with a frozen Python int.
        """
        dispatcher = self._make_dispatcher()

        capture_tokens = 16
        graph, combined, _ = self._warmup_and_capture(dispatcher, local_tokens=capture_tokens)

        graph.replay()
        torch.cuda.synchronize()

        assert combined.shape == (capture_tokens, _HIDDEN_SIZE), (
            f"Expected frozen output shape ({capture_tokens}, {_HIDDEN_SIZE}), "
            f"got {combined.shape}."
        )

        different_tokens = 7
        graph_7, combined_7, _ = self._warmup_and_capture(dispatcher, local_tokens=different_tokens)
        graph_7.replay()
        torch.cuda.synchronize()

        assert combined_7.shape == (different_tokens, _HIDDEN_SIZE), (
            f"Graph captured at {different_tokens} tokens should produce "
            f"output of that size, got {combined_7.shape}."
        )

        graph.replay()
        torch.cuda.synchronize()
        assert combined.shape[0] == capture_tokens, (
            "After replaying graph_16 again, the output should still be "
            f"[{capture_tokens}, {_HIDDEN_SIZE}], not [{different_tokens}, ...]."
        )

    @pytest.mark.timeout(30)
    def test_mismatched_token_counts_across_ranks(self):
        """Simulates the full failure: ranks with different actual token counts.

        Captures one shared graph at 8 tokens (warmup).  Then ranks
        would ideally process different actual counts (7, 7, 6, 6 for a
        4-GPU EP group), but the graph replays with local_tokens=8 on
        every rank.

        After replay, verifies that _step_metadata is frozen at 8 (not
        the intended per-rank counts), proving that the metadata kernel
        cannot reflect runtime token-count changes.

        On an EP group where the frozen metadata causes the AGV to write
        at wrong offsets while different ranks expect different layouts,
        this can escalate to a hang in multimem_all_gatherv_3tensor.
        """
        dispatcher = self._make_dispatcher()
        ep_size = dispatcher.ep_size
        rank = dist.get_rank() if ep_size > 1 else 0

        warmup_tokens = 8
        graph, combined, _ = self._warmup_and_capture(dispatcher, local_tokens=warmup_tokens)

        intended_per_rank = [7, 7, 6, 6, 5, 5, 4, 4][:ep_size]
        intended_local = intended_per_rank[rank]

        graph.replay()
        torch.cuda.synchronize()

        metadata = NVLSAllGatherVDispatcher._step_metadata.cpu()
        ep_max_tokens = metadata[2].item()

        assert ep_max_tokens == warmup_tokens, (
            f"Rank {rank}: intended local_tokens={intended_local} but "
            f"ep_max_tokens is frozen at {warmup_tokens} (got {ep_max_tokens}). "
            f"The metadata kernel's Python int arg is baked into the graph."
        )

        assert combined.shape[0] == warmup_tokens, (
            f"Rank {rank}: intended {intended_local} output tokens but "
            f"token_combine produced {combined.shape[0]} (frozen at warmup)."
        )

        assert metadata[0].item() == warmup_tokens * ep_size, (
            f"valid_tokens frozen at {warmup_tokens * ep_size}, "
            f"but should be {sum(intended_per_rank)} if metadata were dynamic."
        )
