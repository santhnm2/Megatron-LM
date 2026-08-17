# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP (Gated Delta Product) inference equivalence tests.

`GatedDeltaProductMixer` supports two inference paths: static batching
(`StaticInferenceContext`) and dynamic batching (`DynamicInferenceContext`).
These tests assert that both are numerically equivalent to a plain
full-sequence forward, and therefore to each other.

The reference is a single full-sequence `model.forward` (the
`chunk_gated_delta_product` path). Dynamic prefill runs the same chunk kernel
over a packed var-len layout, and static-batching prefill runs the same chunk
kernel with the recurrent cache seeded; both must reproduce the reference's
last-token logits.

The single-forward equivalence tests run at TP=1 (they compare raw logits that
are sequence-sharded under sequence-parallel, and the static path does not
support SP). The end-to-end engine tests sweep TP (`_TP_SIZES`) with SP enabled
at TP>1, covering dynamic inference under tensor + sequence parallelism; TP>1
variants skip when the world has too few GPUs.

The second half of the file drops below the model and covers the in-tree kernels
the dynamic path runs on, and their CUDA-graph behaviour:

1. `ops/gdp/fused_recurrent.py` -- the decode recurrence, against a naive
   PyTorch reference, including slot-indexed state and `-1` padding slots.
2. `ops/gdp/metadata.py` -- the chunk descriptors that stand in for FLA's
   host-synchronizing `prepare_chunk_indices`.
3. `ops/gdp/chunk.py` -- the chunked prefill, against the pip FLA kernels the
   training path runs on.
4. `GatedDeltaProductMixer.ssm_decode` replayed from a CUDA graph captured at a
   padded batch size, against the same call run eagerly.

The padding contract is what makes CUDA graphs work: a graph is captured for a
rounded-up batch shape, and steps with fewer real requests mark the leftover
rows with `-1` in `batch_indices` (and, for prefill, as zero-length sequences).
Those rows must produce zero output and must not read or write any state slot.

Whole-model graph capture is covered separately, in
`tests/unit_tests/inference/engines/test_gdp_cuda_graph_e2e.py`.
"""

from __future__ import annotations

import random
import types
from typing import Dict, List, Optional, Sequence, Tuple

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest, Status
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import gated_delta_product_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.ops.gdp.chunk import chunk_gated_delta_product_varlen
from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update
from megatron.core.ssm.ops.gdp.metadata import build_gdp_chunk_descriptors, max_gdp_chunk_counts
from megatron.core.ssm.packed_seq_helpers import check_fla_sequence_packing_support
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.utils import is_fa_min_version
from tests.unit_tests.test_utilities import Utils, clear_nvte_env_vars

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

try:
    import einops  # noqa: F401
    import mamba_ssm  # noqa: F401

    HAVE_MAMBA_DEPS = True
except ImportError:
    HAVE_MAMBA_DEPS = False

# The model-level tests need the whole stack; the kernel-level ones below need
# only the pieces they actually touch.
HAVE_GDP_DEPS = HAVE_FLA and HAVE_MAMBA_DEPS

try:
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
        TERowParallelLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixerSubmodules

    HAVE_MIXER_DEPS = True
except ImportError:
    HAVE_MIXER_DEPS = False

# GDP dynamic inference relies on the same packed-sequence conv1d kernel as the
# training/prefill path (`causal_conv1d_fn(seq_idx=...)`, added in 1.4.0).
_PACKING_OK, _PACKING_REASON = check_fla_sequence_packing_support()

# Only `internal` is module-wide. The model-level requirements (the full GDP
# dependency set, packed-sequence support) sit on the classes that need them:
# the kernel tests below reach past the model and have their own, lighter guards
# -- the chunk-descriptor builder is pure Python and needs neither a GPU nor fla.
pytestmark = [pytest.mark.internal]

requires_gdp_model = [
    pytest.mark.skipif(not HAVE_GDP_DEPS, reason="GDP requires fla, mamba_ssm, and einops"),
    pytest.mark.skipif(not _PACKING_OK, reason=_PACKING_REASON or "packed-seq support missing"),
]

# The chunk-descriptor builder is pure Python and runs anywhere; everything that
# launches a kernel needs a GPU.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def gdp_kernel_distributed_environment():
    """Bind every rank to its own GPU before anything allocates or captures.

    `torch.cuda.set_device()` lives inside `Utils.initialize_distributed()`, so
    without this each rank of a multi-rank run would allocate on GPU 0 and four
    processes would capture CUDA graphs on the same device. Module-scoped (not
    per-method) so the current device never changes mid-process, which would
    strand earlier allocations and the graph memory pool on another GPU.

    Requested explicitly by the kernel-level classes rather than `autouse`,
    because the model-level classes above manage `Utils` per method and would
    tear this down underneath it. Those classes all sort before the kernel ones
    in file order, so this fixture is only created once they are done.
    """
    if not torch.cuda.is_available():
        # The chunk-descriptor tests are pure Python and need no GPU or ranks.
        yield
        return
    Utils.initialize_model_parallel()
    clear_nvte_env_vars()
    yield
    Utils.destroy_model_parallel()


# A short single-chunk prompt is enough to exercise the packed-varlen dynamic
# path; the sizes are kept small to keep the test fast.
_VOCAB_SIZE = 128
_MAX_SEQ_LEN = 512
_PROMPT_LEN = 64

# bf16 chunk-vs-chunk tolerance. Full-forward and prefill run the same
# `chunk_gated_delta_product` kernel, so they differ only by the packed var-len
# layout and floating-point accumulation order.
_ATOL = 5e-2
_RTOL = 5e-2

# Looser tolerance for the decode-step check: it compares the recurrent decode
# kernel against a full-sequence chunk-kernel recompute (different kernels), so
# it drifts more than the chunk-vs-chunk prefill comparison. Still far tighter
# than the O(1)+ deviations a genuinely broken decode/state-handoff would show.
_DECODE_ATOL = 1e-1
_DECODE_RTOL = 1e-1

# `DynamicInferenceContext` requires at least one attention layer, so the model
# pattern is GDP mixer + attention + MLP. Dynamic batching needs a recent
# flash-attention.
_LAYER_PATTERN = "M*-"
_NUM_LAYERS = len(_LAYER_PATTERN)
requires_dynamic_batching = pytest.mark.skipif(
    not is_fa_min_version("2.7.3"), reason="need flash-attn >= 2.7.3 for dynamic batching"
)

# Tensor-parallel sizes swept by the end-to-end engine tests. TP>1 requires
# sequence-parallel (the inference-optimized linears assert it), which GDP's
# dynamic path supports; static inference does not, so the single-forward
# equivalence tests above stay TP=1.
_TP_SIZES = [1, 2]


def _make_config(tp: int = 1) -> TransformerConfig:
    """A small but shape-valid GDP config, sharded across `tp` tensor-parallel ranks.

    The in_proj output width (`zVKQba`) is column-parallel, so each rank sees
    `proj_dim / tp` channels. The packed-prefill conv slices a channels-last view
    out of that and `causal_conv1d_fn` requires its stride (the per-rank width) to
    be a multiple of 8. With mamba_num_heads=16 the full width is
    `(1+M)*d_inner + (M+1)*ngroups*d_state + (M+1)*nheads = 3*256 + 3*64 + 3*16
    = 1008`, so per-rank widths are 1008 (tp=1) and 504 (tp=2), both aligned.
    Production configs satisfy this by having much larger, aligned dimensions.
    """
    return TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        mamba_num_heads=16,
        mamba_head_dim=16,
        mamba_num_groups=4,
        mamba_state_dim=16,
        gdp_num_householder=2,
        is_hybrid_model=True,  # needed for correct out_proj init
        tensor_model_parallel_size=tp,
        sequence_parallel=False,
        context_parallel_size=1,
    )


def _build_model(tp: int = 1) -> HybridModel:
    """Build a small GDP hybrid model (mixer + attention + MLP), eval on CUDA."""
    model_parallel_cuda_manual_seed(123)
    model = HybridModel(
        config=_make_config(tp),
        hybrid_stack_spec=gated_delta_product_inference_stack_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_MAX_SEQ_LEN,
        hybrid_layer_pattern=_LAYER_PATTERN,
    )
    return model.cuda().eval()


@requires_dynamic_batching
@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="GDP requires fla, mamba_ssm, and einops")
@pytest.mark.skipif(not _PACKING_OK, reason=_PACKING_REASON or "packed-seq support missing")
class TestGDPDynamicInference:
    """Static/dynamic GDP inference equivalence against a full-sequence forward.

    These compare raw `model.forward` logits, which are sequence-sharded under
    sequence-parallel; combined with the static path not supporting SP, they run
    at TP=1 only. TP>1 dynamic inference is covered end-to-end by the engine
    tests below, which handle SP through the inference wrapper.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        self.model = _build_model(tp=1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _input_ids(self) -> torch.Tensor:
        """A deterministic single-request prompt: shape [1, _PROMPT_LEN]."""
        return torch.arange(_PROMPT_LEN, device="cuda", dtype=torch.long).unsqueeze(0)

    @torch.inference_mode()
    def _full_forward_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Reference: plain full-sequence forward -> last-token logits [1, V]."""
        # No inference_context: GDP runs the training / chunk_gated_delta_product
        # path. This is the ground truth both inference modes must reproduce.
        InferenceMode.unset_active()
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            runtime_gather_output=True,
        )
        return logits[:, -1, :].float()

    def _build_dynamic_context(self) -> DynamicInferenceContext:
        mamba_config = MambaInferenceStateConfig.from_model(self.model)
        assert mamba_config is not None, "GDP hybrid model should expose Mamba inference state"
        return DynamicInferenceContext(
            model_config=self.model.config,
            inference_config=InferenceConfig(
                max_sequence_length=_MAX_SEQ_LEN,
                buffer_size_gb=1.0,
                block_size_tokens=256,
                # Materialize all tokens so we can read the prompt's last-token
                # logits directly (static batching always uses last-token only).
                materialize_only_last_token_logits=False,
                mamba_inference_state_config=mamba_config,
                num_cuda_graphs=0,
                use_cuda_graphs_for_non_decode_steps=False,
                max_requests=4,
                max_tokens=128,
            ),
        )

    @torch.inference_mode()
    def _dynamic_prefill_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Dynamic-batching prefill -> last-token logits [1, V]."""
        ctx = self._build_dynamic_context()
        request = DynamicInferenceRequest(
            request_id=0,
            prompt_tokens=input_ids.cpu().squeeze(0),
            sampling_params=SamplingParams(num_tokens_to_generate=1, termination_id=-1),
        )
        ctx.add_request(request)
        ctx.initialize_attention_state()
        with InferenceMode.active():
            logits = self.model(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=None,
                inference_context=ctx,
                runtime_gather_output=True,
            )
        # materialize_only_last_token_logits=False -> [1, prompt_len, V].
        return logits[:, -1, :].float()

    @torch.inference_mode()
    def _static_prefill_last_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Static-batching prefill -> last-token logits [1, V]."""
        ctx = StaticInferenceContext(max_batch_size=1, max_sequence_length=_MAX_SEQ_LEN)
        ctx.sequence_len_offset = 0
        position_ids = torch.arange(input_ids.shape[1], device="cuda").unsqueeze(0)
        with InferenceMode.active():
            logits = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                inference_context=ctx,
                runtime_gather_output=True,
            )
        # StaticInferenceContext forces materialize_only_last_token_logits=True,
        # so the sequence dimension is already collapsed to the last token.
        assert logits.shape[1] == 1
        return logits[:, 0, :].float()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_constructor(self):
        """The GDP stack spec wires a GatedDeltaProductMixer into the mamba layer."""
        assert isinstance(self.model, HybridModel)
        mixers = [
            layer.mixer
            for layer in self.model.decoder.layers
            if hasattr(layer, "mixer") and layer.mixer is not None
        ]
        assert len(mixers) == 1, f"pattern {_LAYER_PATTERN!r} should yield exactly one mixer layer"
        assert isinstance(mixers[0], GatedDeltaProductMixer)

    def test_full_forward_shape(self):
        """Sanity check: plain forward returns [batch, seq, vocab]."""
        input_ids = self._input_ids()
        InferenceMode.unset_active()
        position_ids = torch.arange(_PROMPT_LEN, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                runtime_gather_output=True,
            )
        assert logits.shape == (1, _PROMPT_LEN, _VOCAB_SIZE)

    def test_dynamic_prefill_matches_full_forward(self):
        """Dynamic-batching prefill reproduces the full-sequence forward."""
        input_ids = self._input_ids()
        reference = self._full_forward_last_logits(input_ids)
        dynamic = self._dynamic_prefill_last_logits(input_ids)
        torch.testing.assert_close(dynamic, reference, atol=_ATOL, rtol=_RTOL)

    def test_static_prefill_matches_full_forward(self):
        """Static-batching prefill reproduces the full-sequence forward."""
        input_ids = self._input_ids()
        reference = self._full_forward_last_logits(input_ids)
        static = self._static_prefill_last_logits(input_ids)
        torch.testing.assert_close(static, reference, atol=_ATOL, rtol=_RTOL)

    def test_static_and_dynamic_prefill_agree(self):
        """Static and dynamic inference produce equivalent logits.

        This is the central invariant: the two batching strategies must agree.
        Anchoring each to the full-sequence forward (above) guarantees this
        transitively, but assert it directly as well so a regression in either
        path that happens to drift in the same direction is still caught.
        """
        input_ids = self._input_ids()
        static = self._static_prefill_last_logits(input_ids)
        dynamic = self._dynamic_prefill_last_logits(input_ids)
        torch.testing.assert_close(static, dynamic, atol=_ATOL, rtol=_RTOL)

    @torch.inference_mode()
    def test_decode_step_matches_recompute(self):
        """One decode step matches a full-sequence recompute (decode-path check).

        This validates the recurrent decode kernel and the prefill->decode
        conv/SSM state handoff independently of any golden snapshot: after
        prefilling the prompt, decoding one more token must produce the same
        next-token logits as a plain forward over prompt+token. Compared at the
        logit level with tolerance, so it is robust to the bf16 numerics that
        make exact greedy token equality across the recurrent/chunk kernels
        fragile. Uses `StaticInferenceContext`, whose decode calls the same
        `ssm_decode` recurrent kernel as the dynamic engine.
        """
        prompt = self._input_ids()  # [1, P]

        # Ground truth: the next token from the prompt, and the full-recompute
        # distribution for the token after it (chunk kernel over prompt+token).
        next_token = int(self._full_forward_last_logits(prompt).argmax(dim=-1).item())
        extended = torch.cat(
            [prompt, torch.tensor([[next_token]], dtype=torch.int64, device="cuda")], dim=1
        )
        recompute = self._full_forward_last_logits(extended)  # [1, V]

        # Incremental path: prefill the prompt, then a single decode step.
        ctx = StaticInferenceContext(max_batch_size=1, max_sequence_length=_MAX_SEQ_LEN)
        prompt_length = prompt.shape[1]
        with InferenceMode.active():
            ctx.sequence_len_offset = 0
            self.model(
                input_ids=prompt,
                position_ids=torch.arange(prompt_length, device="cuda").unsqueeze(0),
                attention_mask=None,
                inference_context=ctx,
                runtime_gather_output=True,
            )
            ctx.sequence_len_offset = prompt_length
            decode_logits = self.model(
                input_ids=torch.tensor([[next_token]], dtype=torch.int64, device="cuda"),
                position_ids=torch.tensor([[prompt_length]], dtype=torch.int64, device="cuda"),
                attention_mask=None,
                inference_context=ctx,
                runtime_gather_output=True,
            )
        decode_last = decode_logits[:, -1, :].float()

        torch.testing.assert_close(decode_last, recompute, atol=_DECODE_ATOL, rtol=_DECODE_RTOL)


# ======================================================================
# End-to-end engine tests.
#
# The tests above exercise a single forward pass. These drive the full
# `DynamicInferenceEngine` (add requests -> schedule -> prefill -> decode ->
# finish) so that GDP is validated through the same runtime path production
# inference uses: the text-generation controller, the inference-wrapped model,
# the KV/Mamba-state cache, and the request scheduler.
#
# Decoding is greedy (`top_k=1`) so outputs are deterministic. Decode
# correctness (the recurrent kernel `fused_recurrent_gated_delta_rule` plus the
# slot-indexed conv/SSM cache, distinct from the prefill chunk kernel) is
# validated by a byte-for-byte match against committed golden token ids,
# mirroring the Mamba2 `test_dynamic_engine.py::test_simple` style. The golden
# constant is captured from a reference GPU run (see `_GOLDEN_*` below); until it
# is populated the test self-captures and skips with the observed ids.
# ======================================================================

# Fixed prompts for the golden-token test. Deterministic (not random) so the
# committed golden ids below are reproducible across machines. Varying lengths
# exercise the scheduler's mixed-length prefill batching.
_GOLDEN_PROMPTS: List[List[int]] = [
    [3, 14, 15, 92, 65, 35, 89, 79],
    [2, 71, 82, 81, 8],
    [11, 22, 33, 44, 55, 66],
    [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7],
]
_GOLDEN_NUM_TOKENS_TO_GENERATE = 12

# Golden generated-token ids per TP size, one list per prompt in `_GOLDEN_PROMPTS`,
# captured from a reference GPU run. TP shards the weights differently, so each TP
# size has its own goldens. Environment-sensitive (FLA / causal_conv1d kernel
# build, GPU arch); re-capture if the kernels or config change. While an entry is
# None, the test self-captures: it prints the observed ids and skips instead of
# failing. Paste them in (see the skip message) to turn it into a hard assertion.
_GOLDEN_GENERATED_TOKENS: Dict[int, Optional[List[List[int]]]] = {
    1: [
        [32, 126, 35, 125, 52, 116, 55, 38, 39, 4, 53, 100],
        [88, 10, 105, 95, 105, 44, 2, 100, 127, 59, 23, 18],
        [29, 9, 61, 2, 100, 2, 69, 75, 36, 80, 103, 26],
        [2, 55, 4, 108, 116, 120, 24, 113, 100, 48, 111, 22],
    ],
    2: [
        [12, 51, 63, 22, 2, 40, 45, 30, 55, 10, 31, 29],
        [38, 54, 29, 17, 33, 13, 10, 45, 1, 22, 21, 37],
        [24, 55, 38, 55, 35, 61, 26, 25, 31, 20, 62, 56],
        [53, 26, 31, 61, 16, 19, 42, 41, 49, 18, 53, 26],
    ],
}


def _make_engine_config(tp: int = 1) -> TransformerConfig:
    """GDP config for the engine tests: same shape as `_make_config`, plus the
    deterministic inference sampling knobs greedy decoding needs. TP>1 turns on
    sequence-parallel, which the inference-optimized linears require."""
    config = _make_config(tp)
    config.sequence_parallel = tp > 1
    config.inference_rng_tracker = True
    config.inference_sampling_seed = 123
    return config


@pytest.mark.internal
@requires_dynamic_batching
@pytest.mark.skipif(not HAVE_GDP_DEPS, reason="GDP requires fla, mamba_ssm, and einops")
@pytest.mark.skipif(not _PACKING_OK, reason=_PACKING_REASON or "packed-seq support missing")
class TestGDPDynamicInferenceEngine:
    """End-to-end GDP decoding through `DynamicInferenceEngine`."""

    SEED = 123
    VOCAB_SIZE = _VOCAB_SIZE

    def teardown_method(self, method):
        delete_cuda_graphs()
        Utils.destroy_model_parallel()

    # ------------------------------------------------------------------
    # Harness
    # ------------------------------------------------------------------

    def _build_engine(
        self,
        *,
        num_tokens_to_generate: int,
        tp: int = 1,
        num_requests: Optional[int] = None,
        prompt_length: Optional[int] = None,
        prompts: Optional[Sequence[Sequence[int]]] = None,
    ) -> Tuple[DynamicInferenceEngine, List[DynamicInferenceRequest]]:
        """Build a greedy GDP engine plus its requests at TP=`tp`.

        Either pass explicit `prompts` (deterministic token lists) or
        `num_requests` + `prompt_length` (random prompts of a fixed length).
        Skips if the world is too small for the requested TP size.
        """
        if Utils.world_size < tp:
            pytest.skip(f"TP={tp} requires at least {tp} GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=1
        )
        clear_nvte_env_vars()
        random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        model_parallel_cuda_manual_seed(
            seed=self.SEED, inference_rng_tracker=True, force_reset_rng=True
        )

        if prompts is not None:
            prompt_tensors = [torch.tensor(p, dtype=torch.int64, device="cuda") for p in prompts]
        else:
            assert num_requests is not None and prompt_length is not None
            prompt_tensors = [
                torch.randint(
                    0, self.VOCAB_SIZE - 1, (prompt_length,), dtype=torch.int64, device="cuda"
                )
                for _ in range(num_requests)
            ]

        max_prompt_length = max(int(p.numel()) for p in prompt_tensors)
        max_sequence_length = max_prompt_length + num_tokens_to_generate
        config = _make_engine_config(tp)
        model = HybridModel(
            config=config,
            hybrid_stack_spec=gated_delta_product_inference_stack_spec,
            vocab_size=self.VOCAB_SIZE,
            max_sequence_length=max_sequence_length,
            parallel_output=True,
            hybrid_layer_pattern=_LAYER_PATTERN,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
        ).cuda()
        for param in model.parameters():
            param.data = param.data.to(config.params_dtype)
        model.eval()

        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=max_sequence_length,
                buffer_size_gb=0.1,
                block_size_tokens=256,
                materialize_only_last_token_logits=True,
                mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
                num_cuda_graphs=None,
                use_cuda_graphs_for_non_decode_steps=False,
                max_requests=32,
                max_tokens=1024,
            ),
        )

        wrapped_model = GPTInferenceWrapper(model, context)
        wrapped_model.model_is_pipeline_parallel = not (
            parallel_state.is_pipeline_first_stage() and parallel_state.is_pipeline_last_stage()
        )
        controller = TextGenerationController(
            inference_wrapped_model=wrapped_model,
            tokenizer=types.SimpleNamespace(
                vocab_size=self.VOCAB_SIZE, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        delete_cuda_graphs()
        engine = DynamicInferenceEngine(controller, context)

        requests = [
            DynamicInferenceRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                sampling_params=SamplingParams(
                    num_tokens_to_generate=num_tokens_to_generate,
                    termination_id=-1,  # never terminate early -> fixed output length
                    top_k=1,  # greedy -> deterministic
                ),
            )
            for request_id, prompt_tokens in enumerate(prompt_tensors)
        ]
        return engine, requests

    @staticmethod
    @torch.inference_mode()
    def _run_to_completion(
        engine: DynamicInferenceEngine, requests: List[DynamicInferenceRequest]
    ) -> Dict[int, DynamicInferenceRequest]:
        """Add every request, step until the engine drains, return finished requests by id."""
        for request in requests:
            engine._add_request(request)

        finished: Dict[int, DynamicInferenceRequest] = {}
        # Bound the loop so a scheduling regression fails loudly instead of hanging.
        for _ in range(1000):
            result = engine.step_modern()
            for record in result["finished_request_records"]:
                merged = record.merge()
                finished[merged.request_id] = merged
            if not engine.has_unfinished_requests():
                break
        assert not engine.has_unfinished_requests(), "engine did not drain within step budget"
        return finished

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("tp", _TP_SIZES)
    def test_engine_runs_to_completion(self, tp):
        """Every request completes and yields exactly the requested token count."""
        num_tokens_to_generate = 8
        engine, requests = self._build_engine(
            tp=tp, num_requests=4, prompt_length=8, num_tokens_to_generate=num_tokens_to_generate
        )
        finished = self._run_to_completion(engine, requests)

        assert len(finished) == len(requests)
        for request in requests:
            merged = finished[request.request_id]
            assert merged.status == Status.COMPLETED
            # termination_id=-1 disables early stop, so the length is exact.
            assert len(merged.generated_tokens) == num_tokens_to_generate

    @pytest.mark.parametrize("tp", _TP_SIZES)
    def test_engine_greedy_matches_golden(self, tp):
        """Greedy decode reproduces committed golden token ids (Mamba2-style).

        Deterministic fixed prompts + greedy sampling make the output a stable
        fingerprint of the GDP prefill+decode path. Until the TP entry in
        `_GOLDEN_GENERATED_TOKENS` is captured from a reference GPU run, the test
        prints the observed ids and skips instead of failing.
        """
        engine, requests = self._build_engine(
            tp=tp, prompts=_GOLDEN_PROMPTS, num_tokens_to_generate=_GOLDEN_NUM_TOKENS_TO_GENERATE
        )
        finished = self._run_to_completion(engine, requests)
        observed = [finished[r.request_id].generated_tokens for r in requests]

        golden = _GOLDEN_GENERATED_TOKENS.get(tp)
        if golden is None:
            pytest.skip(
                f"golden tokens for TP={tp} not captured yet; paste the following into "
                f"_GOLDEN_GENERATED_TOKENS[{tp}]:\n{observed!r}"
            )

        assert observed == golden, (
            f"generated tokens != golden (TP={tp}):\n  golden   = {golden}\n"
            f"  observed = {observed}"
        )

    @pytest.mark.parametrize("tp", _TP_SIZES)
    def test_generate_over_multiple_prompts(self, tp):
        """`engine.generate` drives several prompts through to completion at once."""
        engine, requests = self._build_engine(
            tp=tp, num_requests=4, prompt_length=8, num_tokens_to_generate=4
        )

        prompts = [f"prompt{i}" for i in range(len(requests))]

        def mock_tokenize_prompt(tokenizer, prompt, add_BOS=False):
            prompt_num = int(prompt[-1])
            return [10 + i for i in range(prompt_num + 2)]

        engine.controller.tokenize_prompt = mock_tokenize_prompt

        finished_records = engine.generate(prompts, requests[0].sampling_params)
        finished = [record.merge() for record in finished_records]

        assert len(finished) == len(prompts)
        # generate() returns finished requests in request-id order.
        assert [r.request_id for r in finished] == sorted(r.request_id for r in finished)
        for request in finished:
            assert request.status == Status.COMPLETED
            assert len(request.generated_tokens) > 0


# --------------------------------------------------------------------------- #
# Reference implementation
# --------------------------------------------------------------------------- #


def _gated_delta_rule_ref(q, k, v, g, beta, state, state_indices, use_qk_l2norm, scale=None):
    """Naive per-request gated delta rule; updates `state` in place.

    Deliberately written as an explicit Python loop over requests, heads and
    time steps so it is obviously independent of the Triton kernel's blocking.
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
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


@pytest.mark.usefixtures("gdp_kernel_distributed_environment")
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

        out, _ = fused_recurrent_gated_delta_rule_update(
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

        out, _ = fused_recurrent_gated_delta_rule_update(
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

        out, _ = fused_recurrent_gated_delta_rule_update(
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
        out_padded, _ = fused_recurrent_gated_delta_rule_update(
            q, k, v, state=state_padded, g=g, beta=beta, state_indices=padded_indices
        )

        state_real = state.clone()
        out_real, _ = fused_recurrent_gated_delta_rule_update(
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
        out_fork, _ = fused_recurrent_gated_delta_rule_update(
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
            out, _ = fused_recurrent_gated_delta_rule_update(
                q,
                k,
                v,
                state=state,
                g=g,
                beta=beta,
                state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
            return out

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


@pytest.mark.usefixtures("gdp_kernel_distributed_environment")
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


@pytest.mark.usefixtures("gdp_kernel_distributed_environment")
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
        o_fork, _ = chunk_gated_delta_product_varlen(
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
        o_padded, _ = chunk_gated_delta_product_varlen(
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
        o_real, _ = chunk_gated_delta_product_varlen(
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
            out, _ = chunk_gated_delta_product_varlen(
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
            return out

        state_eager = state_init.clone()
        with torch.no_grad():
            o_eager, _ = chunk_gated_delta_product_varlen(
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
