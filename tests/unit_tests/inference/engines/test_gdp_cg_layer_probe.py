# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stage 4 diagnostic: which *layer* diverges, and is it a near-tie argmax flip?

Every GDP kernel stage is now cleared: the prefill chain and the varlen conv are
both bit-identical under token/request padding and under graph replay. So the
divergence either enters the GDP layer from outside, or it comes from a layer
that is not GDP at all.

This probe runs one prefill step in each arm with `register_forward_pre_hook` /
`register_forward_hook` on all three decoder layers -- hooks fire inside
`nn.Module._call_impl`, before `CudaGraphManager` takes over, so they observe
both arms -- and reports, per layer, the max absolute difference of its input
and output over the real token positions.

It then compares the last-token logits per request and asks the question the
token assertion cannot: is the logit difference merely larger than the gap
between the top two candidates? If so, the tokens diverge because greedy
decoding amplifies a small numeric difference, and the e2e test's premise of
bit-exact tokens across a 199 -> 1720 shape change is not achievable.

Not a real test -- delete once the question is settled.
"""

import pytest
import torch

from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (  # noqa: F401
    NUM_CUDA_GRAPHS,
    TestGDPCudaGraphE2E,
)

# The failing e2e step: prompts 0 and 1 prefill together, 199 real tokens.
PROMPT_LENS = [70, 129]
REAL_TOKENS = sum(PROMPT_LENS)


def _first(x):
    """Layer inputs/outputs are tensors or tuples; take the hidden states."""
    while isinstance(x, (tuple, list)):
        if not x:
            return None
        x = x[0]
    return x if isinstance(x, torch.Tensor) else None


def _align(a, b, n):
    """Trim both tensors to the first `n` real tokens, whichever axis that is.

    The two arms pad to different token counts (200 vs 1720), so the token axis
    is the one whose size differs -- dim 0 for `[S, B, H]` hidden states, dim 1
    for the embedding's `[B, S]` input ids. Returns `(None, None, None)` when the
    shapes differ in more than one axis, which is not something to compare.
    """
    if a.shape == b.shape:
        # Same shape in both arms: the token axis is not identifiable from the
        # difference, so fall back to the first axis long enough to hold them.
        axis = next((i for i, s in enumerate(a.shape) if s >= n), None)
        if axis is None:
            return None, None, None
        return a.narrow(axis, 0, n), b.narrow(axis, 0, n), axis
    if a.ndim != b.ndim:
        return None, None, None
    diff = [i for i, (x, y) in enumerate(zip(a.shape, b.shape)) if x != y]
    if len(diff) != 1:
        return None, None, None
    axis = diff[0]
    if a.shape[axis] < n or b.shape[axis] < n:
        return None, None, None
    return a.narrow(axis, 0, n), b.narrow(axis, 0, n), axis


def _attach(model, store):
    """Record input/output hidden states for every module outside the captured layers.

    Submodules *inside* a decoder layer are skipped on purpose: with
    `cuda_graph_impl="local"` a captured layer replays kernels without running
    Python, so their hooks never fire in the graph arm and the comparison would
    silently drop them. The layers themselves are hooked at their boundary
    (`nn.Module._call_impl` runs hooks before `CudaGraphManager` takes over), and
    everything after the last layer -- the final norm and the output layer --
    runs eagerly in both arms.
    """
    handles = []

    def add(name, module):
        def pre_hook(module, args, name=name):
            t = _first(args)
            if t is not None:
                store[f"{name}.in"] = t.detach().float().clone()

        def post_hook(module, args, output, name=name):
            t = _first(output)
            if t is not None:
                store[f"{name}.out"] = t.detach().float().clone()

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    for i, layer in enumerate(model.decoder.layers):
        add(f"2_layer{i}", layer)

    for name, module in model.named_modules():
        if not name or ".layers." in name or name.endswith(".layers"):
            continue
        # Sort key prefix so the report reads in rough execution order.
        prefix = "1_" if "embedding" in name else ("3_" if "decoder" != name else "2z_")
        add(f"{prefix}{name}", module)

    return handles


class TestGDPLayerProbe(TestGDPCudaGraphE2E):
    """Localize the divergence to a layer, then judge whether it is a tie-break."""

    @pytest.mark.skip(reason="inherited from the e2e class; not part of this probe")
    def test_cuda_graphs_do_not_change_generated_tokens(self, *args, **kwargs):
        """Neutralised so the probe runs on its own."""

    @pytest.mark.parametrize("batch_invariant", [False, True], ids=["cublas", "batchinv"])
    @torch.inference_mode()
    def test_locate_divergent_layer(self, batch_invariant):
        if batch_invariant:
            # Global GEMM mode only. `config.batch_invariant_mode` is a different,
            # stronger switch that routes Mamba decode through
            # MambaBatchInvariantDecode, which GDP does not implement.
            # backend="triton" avoids requiring DeepGEMM bf16 bindings.
            from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
                disable_batch_invariant_mode,
                enable_batch_invariant_mode,
            )

            try:
                enable_batch_invariant_mode(backend="triton")
            except (RuntimeError, ValueError, ImportError) as exc:
                pytest.skip(f"batch-invariant mode unavailable: {exc}")
        print(
            f"\n########## GEMM mode: "
            f"{'BATCH-INVARIANT (triton)' if batch_invariant else 'default (cuBLAS/TE)'} "
            f"##########"
        )
        try:
            self._compare_arms()
        finally:
            if batch_invariant:
                disable_batch_invariant_mode()

    @torch.inference_mode()
    def test_eager_shape_sensitivity(self):
        """Two EAGER engines at different padded token counts. No graphs anywhere.

        This is the control the whole investigation needs. Every hook fires in
        both arms (nothing is captured, so nothing skips Python), which finally
        makes per-layer localization possible, and it answers the prior question
        directly: if the layers already disagree between a 200-token and a
        1720-token padded batch with graphs switched off entirely, then the
        divergence is shape sensitivity in the layer GEMMs and CUDA graphs are
        merely what changes the shape.
        """
        from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

        model = self._create_model()
        prompts = self._create_prompts()[:2]

        acts, logits = {}, {}
        original_token_rounder = DynamicInferenceContext.TOKEN_ROUNDER
        try:
            for label, token_rounder in (("pad200", 4), ("pad1720", 1720)):
                # Only the token axis moves; REQUEST_ROUNDER stays at 4 so the
                # padded prefill request count is identical in both arms.
                DynamicInferenceContext.TOKEN_ROUNDER = token_rounder
                engine = self._build_engine(model, num_cuda_graphs=None)
                for i, prompt in enumerate(prompts):
                    engine._add_request(self._make_request(i, prompt))

                store = {}
                handles = _attach(model, store)
                try:
                    engine.step_modern()
                finally:
                    for h in handles:
                        h.remove()

                ctx = engine.context
                print(
                    f"\n[{label}] dims={ctx.batch_dimensions} "
                    f"padded={ctx.padded_batch_dimensions} "
                    f"graphed={ctx.using_cuda_graph_this_step()}"
                )
                assert not ctx.using_cuda_graph_this_step(), "this control must stay eager"
                acts[label] = store
                logits[label] = engine.controller._all_logits_cuda[0, :REAL_TOKENS].float().clone()
        finally:
            DynamicInferenceContext.TOKEN_ROUNDER = original_token_rounder

        self._report(acts, logits, "pad200", "pad1720")

    def _compare_arms(self):
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            is_batch_invariant_mode_enabled,
        )

        model = self._create_model()
        prompts = self._create_prompts()[:2]
        assert [len(p) for p in prompts] == PROMPT_LENS
        print(
            f"  batch_invariant_enabled={is_batch_invariant_mode_enabled()}  "
            f"output_layer={type(model.output_layer).__name__}  "
            f"logit_scale={getattr(model, 'logit_scale', None)}"
        )

        acts, logits = {}, {}
        for label, ngraphs in (("eager", None), ("graph", NUM_CUDA_GRAPHS)):
            engine = self._build_engine(model, num_cuda_graphs=ngraphs)
            for i, prompt in enumerate(prompts):
                engine._add_request(self._make_request(i, prompt))

            store = {}
            handles = _attach(model, store)
            try:
                engine.step_modern()
            finally:
                for h in handles:
                    h.remove()

            ctx = engine.context
            print(
                f"\n[{label}] dims={ctx.batch_dimensions} padded={ctx.padded_batch_dimensions} "
                f"graphed={ctx.using_cuda_graph_this_step()}"
            )
            acts[label] = store
            all_logits = engine.controller._all_logits_cuda
            logits[label] = all_logits[0, :REAL_TOKENS].float().clone()

        self._report(acts, logits, "eager", "graph")

    @staticmethod
    def _report(acts, logits, a_label, b_label):
        """Print per-module divergence between two arms, then the logit tie analysis."""
        print(f"\n=== per-module divergence, {a_label} vs {b_label}, 199 real tokens ===")
        print("  (2_layer0 = GDP/mamba, 2_layer1 = attention, 2_layer2 = MLP)")
        only_a = sorted(set(acts[a_label]) - set(acts[b_label]))
        only_b = sorted(set(acts[b_label]) - set(acts[a_label]))
        if only_a or only_b:
            print(f"  NOT COMPARED (one arm only): {a_label}={only_a} {b_label}={only_b}")
        first_bad = None
        for name in sorted(set(acts[a_label]) & set(acts[b_label])):
            ta, tb = acts[a_label][name], acts[b_label][name]
            a, b, axis = _align(ta, tb, REAL_TOKENS)
            if a is None:
                print(
                    f"  {name:<32} SKIPPED: shapes {tuple(ta.shape)} vs {tuple(tb.shape)} "
                    "differ on more than the token axis"
                )
                continue
            d = (a - b).abs()
            worst = float(d.max())
            scale = float(a.abs().max())
            # Per sequence, so we can see whether only request 1 moves.
            seq0 = float(d.narrow(axis, 0, PROMPT_LENS[0]).max())
            seq1 = float(d.narrow(axis, PROMPT_LENS[0], PROMPT_LENS[1]).max())
            print(
                f"  {name:<32} dim{axis} max|diff|={worst:.3e} "
                f"(rel {worst / max(scale, 1e-30):.3e})  seq0={seq0:.3e}  seq1={seq1:.3e}"
            )
            if worst > 0 and first_bad is None:
                first_bad = name
        print(f"  first divergent tensor: {first_bad}")

        print("\n=== last-token logits: is the token flip a near-tie? ===")
        for rid, pos in enumerate([PROMPT_LENS[0] - 1, REAL_TOKENS - 1]):
            la, lb = logits[a_label][pos], logits[b_label][pos]
            delta = float((la - lb).abs().max())
            top2_a = torch.topk(la, 2).values
            top2_b = torch.topk(lb, 2).values
            gap_a = float(top2_a[0] - top2_a[1])
            arg_a, arg_b = int(la.argmax()), int(lb.argmax())
            print(
                f"  req {rid} (pos {pos}): max|dlogit|={delta:.3e}  "
                f"{a_label} top-2 gap={gap_a:.3e}  "
                f"{b_label} top-2 gap={float(top2_b[0] - top2_b[1]):.3e}\n"
                f"      argmax {a_label}={arg_a} {b_label}={arg_b} "
                f"{'FLIPPED' if arg_a != arg_b else 'same'}  "
                f"=> {'tie-break (|dlogit| > gap)' if delta > gap_a else 'gap survives |dlogit|'}"
            )
