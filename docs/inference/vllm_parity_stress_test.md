# Recorded-history stress test of self-contained parity inference

## Result — September 26, 2026

The self-contained parity implementation completed **1,636 / 1,636 requests**
through nemoRL's live Megatron inference server on four GB300 GPUs. There were
**zero request errors and no CUDA faults found in the engine or collected Ray
logs**, including no illegal memory accesses, device assertions or misaligned
accesses. The supervisor completed and shut down the server cleanly.

The primary Megatron worker's preflight reported **`vllm_available: false`**.
There was no reference-worker Python path mount or diagnostic capture overlay.
The numerical code corresponds to self-contained commit `fca03b4910`; the
launched snapshot and per-file hashes are retained in the audit workspace.
Final differences from that snapshot are documentation, packaging, tests,
whitespace, equivalent merged imports, and a warning message; executable
Python AST / C++ token comparisons preserve the tested computations.

| Measurement | Self-contained run 4022383 |
|---|---:|
| Recorded conversations / SWE tasks | 64 / 32 |
| Peak concurrent client requests | 64 |
| Prompt length range | 5,776–196,480 tokens |
| Total prompt tokens | 130,753,329 |
| Fresh generated tokens | 370,177 |
| Stop / length finishes | 1,320 / 316 |
| Replay duration, excluding startup | 1,557.8 seconds |
| Maximum sampled GPU memory per device | 99,177 MiB |

Decode graph buckets of 4, 8, 16, 24 and 32 rows appear in the engine logs.
The largest **sampled actual engine batch** was 31 requests, including 30
decodes. Client concurrency is not the same as engine batch size. Status lines
were sampled every 100 steps and are not a complete graph-replay count.

The baseline using imported vLLM primitives (job 4021949, commit `1541eba45b`)
also completed all 1,636 requests without request errors or logged CUDA faults.
It generated 380,823 tokens in 1,676.7 seconds. These independent stochastic
runs are not a controlled speed comparison or a token-equality test.

## Replay protocol

The inputs are pre-recorded, multi-turn SWE trajectories from rkirby's two
historical step-8 arms, with 64 conversations across 32 tasks. Exact token tapes
contain the original assistant responses, tool outputs and chat delimiters.
Each request ends at a recorded assistant generation boundary.

Fresh server outputs are saved, then the next turn advances using the original
recorded response and tool result. **Fresh outputs never enter subsequent
history, and generated commands are never executed.** All 1,636 request IDs,
prefix hashes, prompt lengths, generation limits and recorded turn boundaries
match the baseline replay. The client also validates each server-echoed prompt.

Seven phases use client concurrency 1, 2, 4, 8, 16, 32 and 64, with respectively
4, 4, 6, 8, 8, 12 and 16 consecutive turns per conversation. Early, middle and
late windows exercise different context lengths. Output limits rotate through
128, 512, 1,024 and 2,048 tokens, clipped to the remaining context capacity.
Length finishes and partially emitted tool calls at those limits are expected.

The server uses frozen BF16 Nemotron-H step18 weights, TP4/EP1/ETP4, PP=CP=1,
an 8,480-token scheduler budget, native autotuning, compiled primitives and
automatic decode graphs. Prefix caching and full prefill graphs are disabled.
Sampling is temperature 1, top-p 1, top-k disabled, with no request seed.
Requests use `/v1/completions` token IDs through the actual nemoRL server.

## Manual generation review

Reviewed **41 fresh outputs covering all 32 tasks**: the first and longest
output in every load phase, the three longest contexts, and the first completed
response for tasks not already represented. This is a qualitative sample;
all 1,636 outputs are saved for further inspection.

Many responses are relevant debugging continuations with coherent language and
plausible tool calls. At 195,982 and 196,480 tokens, the xarray responses correctly
interpret recorded passing tests and propose relevant DataArray/Dataset checks.

The sample also contains real quality problems:

- Long Julia and path-validation responses repeat contradictory hypotheses.
- One proposed Rust macro uses a repeated metavariable outside repetition.
- One response overlooks a recorded shell history-expansion error.
- A purported deep-equality test actually uses JavaScript object identity.
- Another initial search selects the wrong source language.

There was no obvious garbled text or catastrophic numerical corruption in the
reviewed sample. **This is not a blanket generation-quality pass.** Generated
patches were not executed, and neither software correctness nor SWE success
rate was evaluated. The automated compression screen flagged no responses,
including the repetitive reasoning above, so it cannot replace manual review.
These quality issues have not been attributed to an inference mismatch.

## Separate parity and packaging checks

- Primitive comparison job 4022228: 19 MoE shapes, 7 attention shapes, 12 norm
  checks, 12 MoE graph replays, and 26 collective checks on each of four ranks
  match the pinned vLLM reference.
- Integrated job 4022390 versus reference 4020425: all seven scenarios pass
  byte-for-byte, including 92 logits, 2,116 Mamba states, 552 KV states, returned
  tokens/logprobs and 184 observer controls. Equality is conditioned on matched
  autotuning choices; the production selection policies remain equivalent.
- Regression job 4022522: changing-input norm, MoE, attention and collective
  eager/graph replay passes on all four ranks with vLLM imports blocked.
- Packaging job 4022546: a Python 3.13 wheel contains all 217 local package
  files, including CUDA sources, tuning tables and licenses, matching its frozen
  build input. Subsequent whitespace cleanup preserves AST/C++ token identity.

See [runtime setup and parity scope](vllm_numerical_parity.md) for dependencies
and configuration. Existing parity flags still apply; no new model config is
required.

## Reproduction and retained evidence

The shared audit workspace contains the server launch, client, source snapshot
and prepared private inputs under `swe_inference_audit/self_contained_stress_v1/`.
From that workspace's repository root:

```bash
sbatch swe_inference_audit/self_contained_stress_v1/run.sbatch
python3 swe_inference_audit/self_contained_stress_v1/analyze.py \
  swe_inference_audit/self_contained_stress_v1/results/<job>
```

The allocation is bounded at two hours and replay at 95 minutes. The client
does not retry failed requests. The launch requires the recorded internal
container/checkpoint paths and prepared token tapes; these private inputs are
not distributed with the code branch.

Run 4022383 retains `manifest.json`, `worker_environment.log`, `responses.jsonl`,
`events.jsonl`, `engine.log`, `ray_logs/`, `gpu_metrics.csv`, `summary.json`,
`run_status.json`, `analysis.json`, `history_replay_validation.json`,
`REVIEW_CANDIDATES.md`, and the individual judgments in `manual_review.json`.
The final analyzer records `stability_pass: true`.

This finite run exercises scheduling, prefill, decode, cache reuse and graph
replay. It does not establish the absence of every memory-access corner case,
nor cover chat parsing, live tools, optimizer updates, repeated weight refits,
or the cause of the original RL loop-rate discrepancy.
