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


def _attach(model, store):
    """Record each decoder layer's input and output hidden states."""
    handles = []
    for i, layer in enumerate(model.decoder.layers):

        def pre_hook(module, args, i=i):
            t = _first(args)
            if t is not None:
                store[f"layer{i}.in"] = t.detach().float().clone()

        def post_hook(module, args, output, i=i):
            t = _first(output)
            if t is not None:
                store[f"layer{i}.out"] = t.detach().float().clone()

        handles.append(layer.register_forward_pre_hook(pre_hook))
        handles.append(layer.register_forward_hook(post_hook))
    return handles


class TestGDPLayerProbe(TestGDPCudaGraphE2E):
    """Localize the divergence to a layer, then judge whether it is a tie-break."""

    @pytest.mark.skip(reason="inherited from the e2e class; not part of this probe")
    def test_cuda_graphs_do_not_change_generated_tokens(self, *args, **kwargs):
        """Neutralised so the probe runs on its own."""

    @torch.inference_mode()
    def test_locate_divergent_layer(self):
        model = self._create_model()
        prompts = self._create_prompts()[:2]
        assert [len(p) for p in prompts] == PROMPT_LENS

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

        print("\n=== per-layer divergence over the 199 real token positions ===")
        print("  (layer 0 = GDP/mamba, layer 1 = attention, layer 2 = MLP)")
        first_bad = None
        for name in sorted(set(acts["eager"]) & set(acts["graph"])):
            a = acts["eager"][name][:REAL_TOKENS]
            b = acts["graph"][name][:REAL_TOKENS]
            d = (a - b).abs()
            worst = float(d.max())
            scale = float(a.abs().max())
            # Per sequence, so we can see whether only request 1 moves.
            seq0 = float(d[: PROMPT_LENS[0]].max())
            seq1 = float(d[PROMPT_LENS[0] : REAL_TOKENS].max())
            print(
                f"  {name:<12} max|diff|={worst:.3e} (rel {worst / max(scale, 1e-30):.3e})  "
                f"seq0={seq0:.3e}  seq1={seq1:.3e}"
            )
            if worst > 0 and first_bad is None:
                first_bad = name
        print(f"  first divergent tensor: {first_bad}")

        print("\n=== last-token logits: is the token flip a near-tie? ===")
        ends = [PROMPT_LENS[0] - 1, REAL_TOKENS - 1]
        for rid, pos in enumerate(ends):
            le, lg = logits["eager"][pos], logits["graph"][pos]
            delta = float((le - lg).abs().max())
            top2_e = torch.topk(le, 2).values
            top2_g = torch.topk(lg, 2).values
            gap_e = float(top2_e[0] - top2_e[1])
            arg_e, arg_g = int(le.argmax()), int(lg.argmax())
            print(
                f"  req {rid} (pos {pos}): max|dlogit|={delta:.3e}  "
                f"eager top-2 gap={gap_e:.3e}  graph top-2 gap={float(top2_g[0] - top2_g[1]):.3e}\n"
                f"      argmax eager={arg_e} graph={arg_g} "
                f"{'FLIPPED' if arg_e != arg_g else 'same'}  "
                f"=> {'tie-break (|dlogit| > gap)' if delta > gap_e else 'gap survives |dlogit|'}"
            )
