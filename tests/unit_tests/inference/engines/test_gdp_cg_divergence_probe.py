# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stage 1 diagnostic: is the graph-vs-eager divergence in the output or the state?

Runs the two engines from `test_gdp_cuda_graph_e2e` in lockstep and, after every
step, reports for each live request:

* whether the generated token prefix still agrees,
* the max absolute difference in that request's conv and SSM cache rows.

The point is to separate three stories the e2e assertion cannot tell apart:

1. The prefill step already produces different logits (tokens diverge on the
   very first step, states may or may not differ). -> chunked prefill chain.
2. Prefill tokens agree but the prefill-written state differs. -> the state
   write in `chunk_h`, or the conv state write.
3. Both agree after prefill and drift starts on a later decode step.
   -> `fused_recurrent`/`causal_conv1d_update`.

Not a real test -- delete once the question is settled.
"""

import pytest
import torch

from tests.unit_tests.inference.engines.test_gdp_cuda_graph_e2e import (  # noqa: F401
    NUM_CUDA_GRAPHS,
    TestGDPCudaGraphE2E,
)


def _request_to_slot(engine):
    """`{request_id: mamba cache slot}` for the requests live right now."""
    ctx = engine.context
    lo, hi = ctx.paused_request_count, ctx.total_request_count
    if hi <= lo:
        return {}
    ids = ctx.request_ids[lo:hi].tolist()
    slots = ctx.mamba_metadata.request_to_mamba_state_idx[lo:hi].tolist()
    return {rid: slot for rid, slot in zip(ids, slots) if rid >= 0 and slot >= 0}


def _tokens(engine, result):
    """`{request_id: generated tokens}` for everything touched by this step."""
    out = {}
    for rid in result["active_request_ids"]:
        out[int(rid)] = list(engine.get_request(int(rid)).generated_tokens)
    for record in result["finished_request_records"]:
        merged = record.merge()
        out[int(merged.request_id)] = list(merged.generated_tokens)
    return out


class TestGDPDivergenceProbe(TestGDPCudaGraphE2E):
    """Report where the divergence first appears, rather than asserting on tokens."""

    @pytest.mark.skip(reason="inherited from the e2e class; not part of this probe")
    def test_cuda_graphs_do_not_change_generated_tokens(self, *args, **kwargs):
        """Neutralised so the probe runs on its own."""

    @torch.inference_mode()
    def test_locate_first_divergence(self):
        model = self._create_model()
        prompts = self._create_prompts()

        arms = {}
        for label, ngraphs in (("eager", None), ("graph", NUM_CUDA_GRAPHS)):
            engine = self._build_engine(model, num_cuda_graphs=ngraphs)
            for i, prompt in enumerate(prompts):
                engine._add_request(self._make_request(i, prompt))
            arms[label] = engine

        tokens = {"eager": {}, "graph": {}}
        first_bad_token_step = None
        first_bad_state_step = None

        step = 0
        while any(e.has_unfinished_requests() for e in arms.values()):
            step += 1
            pre_slots, post = {}, {}
            for label, engine in arms.items():
                pre_slots[label] = _request_to_slot(engine)
                result = engine.step_modern()
                tokens[label].update(_tokens(engine, result))
                ctx = engine.context
                post[label] = dict(
                    dims=ctx.batch_dimensions,
                    padded=ctx.padded_batch_dimensions,
                    graphed=ctx.using_cuda_graph_this_step(),
                    conv=ctx.mamba_conv_states,
                    ssm=ctx.mamba_ssm_states,
                )

            print(
                f"\n=== step {step} ===\n"
                f"  eager: dims={post['eager']['dims']} padded={post['eager']['padded']} "
                f"graphed={post['eager']['graphed']}\n"
                f"  graph: dims={post['graph']['dims']} padded={post['graph']['padded']} "
                f"graphed={post['graph']['graphed']}"
            )

            shared = sorted(set(tokens["eager"]) & set(tokens["graph"]))
            for rid in shared:
                te, tg = tokens["eager"][rid], tokens["graph"][rid]
                n = min(len(te), len(tg))
                token_ok = te[:n] == tg[:n]
                if not token_ok and first_bad_token_step is None:
                    first_bad_token_step = (step, rid)

                se = pre_slots["eager"].get(rid)
                sg = pre_slots["graph"].get(rid)
                if se is None or sg is None:
                    print(f"  req {rid}: tokens_match={token_ok} (slot retired, no state diff)")
                    continue

                conv_d = (
                    (post["eager"]["conv"][:, se].float() - post["graph"]["conv"][:, sg].float())
                    .abs()
                    .max()
                )
                ssm_d = (
                    (post["eager"]["ssm"][:, se].float() - post["graph"]["ssm"][:, sg].float())
                    .abs()
                    .max()
                )
                ssm_scale = post["eager"]["ssm"][:, se].float().abs().max().clamp(min=1e-30)
                if (float(conv_d) > 0 or float(ssm_d) > 0) and first_bad_state_step is None:
                    first_bad_state_step = (step, rid)
                print(
                    f"  req {rid}: tokens_match={token_ok} n={n} "
                    f"slots=({se},{sg}) "
                    f"max|dconv|={float(conv_d):.3e} max|dssm|={float(ssm_d):.3e} "
                    f"(rel {float(ssm_d / ssm_scale):.3e})"
                )
                if not token_ok:
                    at = next(i for i in range(n) if te[i] != tg[i])
                    print(f"      first token diff at index {at}: {te[at]} vs {tg[at]}")

        print(
            f"\n=== verdict ===\n"
            f"  first step with a token divergence: {first_bad_token_step}\n"
            f"  first step with a state divergence: {first_bad_state_step}"
        )
