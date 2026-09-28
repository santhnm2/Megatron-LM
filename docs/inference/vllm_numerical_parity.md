# Experimental vLLM numerical parity branch

The reference is **vLLM**, including its finite-precision rounding. Agreement
with the training forward pass, or greater arithmetic accuracy, is not an
acceptance criterion for this branch. **The seven audited scenarios now pass
byte-for-byte under matched autotuning choices, with equivalent native selection
policies.** This result covers the TP4/EP1/ETP4 profile described below.

## Port to current main (2026-09-28)

The main-based branch starts at `c035a426e7447e5c22742b2af62d2180537dfe53`.
It carries only the parity changes after the original audit base, preserving
main's GDP/MTP/GTP support and sampler. SSM imports follow main's `ops/common`
and `ops/mamba2` layout. All 52 vendored runtime files are unchanged.

- Primitive/reference and blocked-vLLM graph checks pass (job 4073252).
- Sampling, padding, chunk alignment, SSD replay, QKV layout, convolution,
  determinism registry, dependency export and wheel checks pass (4073252/4073680).
- The seven-case full-model rerun against reference 4020425 passes exactly
  (job 4074593): 92 logits, 2,116 Mamba-state pairs, 552 KV pairs, 184 observer
  controls, and all returned tokens/logprobs.
- Native run 4074731 passes all seven observer controls, 44 compiled-policy
  comparisons and 20 SSD-policy comparisons. All 32 parity helper ASTs (plus
  eight other cached helpers) match controlled run 4074593 after source
  filename normalization. Production tuning winners remain native.
- The recorded-history stress rerun is still pending.

The frozen runtime matches all 651 MCore files in both validation wheels.
Test-only launch compatibility adapts the older pinned NeMo-RL/Bridge container
to main's moved APIs. Dataset helpers are built from main in a writable build
mount; checkpoint writes are disabled. These adapters do not replace model
forwards or checkpoint loading and are not part of the package.

## Self-contained runtime (2026-09-26)

Parity mode no longer imports the vLLM package or its compiled extensions.
The original adapter reused reference primitives to establish the numerical
execution order. Those primitives now live in
`megatron/core/inference/parity_kernels`: BF16 MoE routing/GEMMs, RMSNorm,
custom/symmetric-memory/NCCL collectives, and the pinned FlashAttention 4
implementation. Licenses, source revisions and original hashes are included.
The existing `inference_vllm_*` option names still select the same numerical
profile. No new Megatron model config is required.

The CUDA extension compiles locally before graph capture. It needs a CUDA
toolkit with nvcc, a C++20 compiler, Ninja, and PyTorch 2.11+. The validated
runtime remains PyTorch 2.11.0+cu130, Triton 3.6.0, CUDA 13, and GB300.
The bundled attention implementation contains only BF16 Blackwell SM10x
forward/combine kernels and their dependencies, for head dimensions up to 128.
Backward, benchmark utilities and unused architecture/MLA kernels are excluded.
The bundled MoE policy uses the reference's GB300 fallback; no GB300 tuning
table exists in the pinned reference. Other GPU profiles must explicitly provide
its complete BF16 table directory through `MEGATRON_PARITY_MOE_CONFIG_DIR`.
No source downloads occur at inference runtime. Provision the independent
dependencies inside the policy-worker container before starting Ray:

```bash
MEGATRON_WORKER_PY=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
MEGATRON_SOURCE=/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
uv pip install --python "$MEGATRON_WORKER_PY" \
  -r "$MEGATRON_SOURCE/requirements/inference-parity.txt"
"$MEGATRON_WORKER_PY" -c \
  'from megatron.core.inference.parity_kernels.ops import load_ops; load_ops()'
```

Keep the container's matching CUDA-enabled PyTorch/NCCL build. The dependency
file supplies the CUDA 13 CUTLASS DSL libraries explicitly. `TORCH_EXTENSIONS_DIR`
can point at a writable build cache, and `MAX_JOBS` bounds compiler concurrency.
Every node must have the same dependencies and source. Building an image with
these installed avoids changing ephemeral worker environments at each launch.
The old reference-worker `site.addsitedir` / `.pth` mount is unnecessary.

For a package installation from this branch checkout, select the equivalent
`.[inference-parity]` extra. With the repository uv project, select
`--extra inference-parity`; its torch/triton overrides intentionally retain
the container-provided packages. Current main's `dev` and `ssm` extras require
TVM FFI 0.1.11, while this audited worker uses 0.1.9. Each conflicts with the
`inference-parity` extra; select these environments separately. Main's FFI and
TileLang pins are preserved. The lockfile advances the three CUTLASS DSL packages
from 4.5.0 to the audited 4.5.2, within the development dependencies' allowed
range. Provision the parity profile in the validated worker image, retaining
its existing Mamba package.

Validation of this port is separate from the original audit:

- Job 4022228: 19 MoE shapes, 7 attention shapes, 12 norm checks, 12 MoE graph
  replays, and 26 collective comparisons on each of four ranks match vLLM.
  Local execution ran before exposing the reference package.
- Job 4022390 versus reference 4020425: all seven scenarios again pass
  byte-for-byte, including 92 logit events, 2,116 Mamba states, 552 KV states,
  returned tokens/logprobs and 184 observer controls. Autotuning controls and
  the scope of the original native-policy validation remain explicit.
- Portable regression checks are in
  `tests/unit_tests/determinism/kernels/test_local_parity_kernels.py` and block
  vLLM imports throughout changing-input eager/graph replay. Install pytest in
  the inference worker environment, then run on four GPUs:

  ```bash
  torchrun --standalone --nproc-per-node=4 \
    tests/unit_tests/determinism/kernels/test_local_parity_kernels.py
  ```

- Native live stress job 4022383 completed 1,636 requests across 64 recorded
  conversations with no request errors or logged CUDA faults, without an
  importable vLLM package. It generated 370,177 tokens with contexts up to
  196,480 tokens. Manual review covered 41 outputs from all 32 tasks and found
  both coherent continuations and some incorrect/repetitive debugging. See
  [the full stress report](vllm_parity_stress_test.md) for evidence and limits.

The reference profile is native autotuning, BF16, full attention, ReLU-squared
experts, and one NVLink node with EP1/ETP=TP. Alternative reference environment
profiles (including batch invariance and forced all-reduce algorithms) are
outside the audited default. vLLM is needed only for optional reference-engine
comparison scripts, never for serving this parity mode.

## Completed audit (2026-09-26)

Source-v21, candidate job 4020742 versus reference 4020425, passes prompts of
127, 128, 129, 5,808 and 18,964 tokens with one generated token, plus eight-token
trajectories from 128- and 5,808-token prompts. Every full-vocabulary logit event,
returned token/logprob, 2,116 Mamba cache comparisons and 552 attention cache
comparisons matches exactly. All 184 observer controls pass.

Native worker 4020595 and policy inspection 4020619 validate 44 compiled-policy
comparisons and all five SSD policies on four ranks. All 32 generated helper
modules in the final run match that native worker's ASTs after source filenames
are normalized. The compiler may represent mutated buffers with different
pointer arguments than vLLM; native benchmark timings and independent winners
need not match. Production selection remains native.

The actual nemoRL SWE sampler separately passes 28 rank/batch fixtures for
tokens, RNG advancement and selected raw/processed logprobs on common logits.
Its production files are unchanged since that validation. The runtime requires
the matching PTX assembler environment described below. Original EP4 layout
equivalence and the cause of the RL loop-rate gap are outside this result.

The remaining sections record the investigation chronologically. Earlier
pending or failing results describe those intermediate snapshots.

## Base and reference

- Original audit base: Robert Kirby's `880de0fce84321c04a81a052722f1b909014bac6`.
- The separately committed prefill generated-logprob guard preserves the
  working-tree change present in the audited runtime.
- Reference installation: vLLM `0.25.1`, source revision `752a3a504`, PyTorch
  `2.11.0+cu130`, Triton `3.6.0`, NVIDIA GB300, BF16 unquantized Nemotron-H.
- Initial model profile: hidden size 2,688; TP=4; 16 local Mamba heads;
  two local Mamba groups with 128 state values per group; 128-token scan chunks;
  128 routed experts, top-6, ReLU squared, scale 2.5.
- Primary target: production compilation and graph execution. Eager execution
  is a separate diagnostic target. Existing prefill captures are eager; passing
  their replay does not validate a compiled whole-model forward.
- Existing engine layouts differ: MINF uses EP=4 and sequence parallelism;
  vLLM uses EP=1 with TP-sharded experts. Resolving that difference is part of
  the work, not an assumption of equivalence.

Record source hashes, weight mapping, dtype/stride, kernel launch configuration,
padding, batch composition, cache state, and chunk boundaries with each test.
Pin the reference installation and execution profile. **Match vLLM's actual
Triton autotuning policy**, including candidate configurations and order, keys,
launch defaults, and cache behavior. Do not pin a production kernel to a measured
winner unless vLLM itself does so. Hyperparameter pinning is allowed only in
separately labeled diagnostic controls. Equality may be conditioned on fixed autotuning behavior when the native
autotuning policies are equivalent (user clarification, 2026-09-26). Independent
native winners may differ; preserve that evidence separately from fixed-choice
byte comparisons.

## Acceptance criteria

1. Compare matching logical tokens at each operation boundary, first with
   identical inputs and weights, then in the actual complete forward pass.
2. Require **byte equality**, with dtype and shape equality, for the tested
   activations, recurrent/KV state, logits, and returned logprobs. Report numeric
   error for diagnosis; do not turn a small tolerance into an exactness claim.
3. Validate prefill, partial-chunk continuation, cached decode, and production
   graph/compilation paths. Include short and long histories, mixed lengths,
   and batching controls. Name the profile covered by each result.
4. Sampling scope follows the nemoRL SWE flow: temperature 1, top_p 1,
   top_k disabled, no request seed, and no speculation. Match its actual
   backend dispatch, probabilities, random variates, RNG consumption, and
   selected logprobs on common logits. General filter combinations and
   per-request seed features are outside this audit's scope.
5. Keep raw and sampling-distribution logprobs distinct in the harness, and
   verify their equivalence for this unfiltered temperature-1 profile. Equal
   tokens alone do not establish parity; independent production RNG streams
   need not produce identical trajectories.

These are forward/sampling tests. They do not establish the cause of an RL
training or evaluation discrepancy.

## Initial implementation

The initial inference prefill changes implement the verified product-rounding
mechanism and align native launch/autotuning policies with the reference:

- `causal_conv1d_varlen.py` rounds each convolution product to the input dtype
  before FP32 accumulation, matching the pinned vLLM kernel. The previous
  implementation retained FP32 products.
- The convolution uses the reference's fixed 8-token by 256-channel tile,
  two pipeline stages, and default four warps. vLLM does not autotune this
  prefill convolution.
- The five corresponding forward-scan autotuners use vLLM's complete candidate
  lists in the same order, with the same keys and default options. The lists
  have 6 cumsum, 14 chunk-state, 23 chunk-scan, 6 state-passing, and 9 BMM
  configurations. Megatron-specific deterministic candidate filtering is
  removed from these five decorators. There is **no fixed cumsum tile**.
  The auxiliary `_chunk_state_varlen_kernel`, which has no corresponding
  reference decorator and is outside this tested path, is unchanged.

These changes affect this branch's inference defaults. The training kernel is
not changed. Cached single-token convolution and state update need separate
validation; a successful prefill test must not be extrapolated to decode.

`examples/inference/parity/replay_mamba_prefill.py` verifies the installed vLLM
entrypoint source hashes against the supplied capture metadata and reproduces
historical vLLM calls using their recorded configurations. That is a diagnostic
check of capture fidelity. The harness restores the native configuration lists
and caches before comparing engines with identical convolution/scan inputs.
It asserts runtime equality of the five candidate lists and reports the native
winning configurations alongside source hashes and byte comparisons.

`--diagnostic-pin-cumsum` optionally adds a control that temporarily gives MINF
the configuration selected by native vLLM. The harness restores MINF's native
configuration and cache afterward. This control is reported separately and
**cannot make a native-parity failure pass**. The command fails if historical
replay or any native comparison differs.

Run in a CUDA environment exposing both the branch and the pinned vLLM package:

```bash
uv run python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/inference/parity/replay_mamba_prefill.py \
  --minf /path/to/minf-capture-run \
  --vllm /path/to/vllm-capture-run \
  --out-dir /path/to/results
```

The four workers replay independent TP shards; this command does not validate
distributed model collectives. Capture data and private model weights are not
part of the branch.

## Initial validation (2026-09-25)

GB300 job 4013022 completed on four independent shards. Each comparison uses
identical inputs, weights, and incoming state. Five related histories cover
127/128/129-token prefixes, a 5,808-token control, and an 18,964-token history,
including incoming states at their actual engine prefill chunk boundaries.

| Check | Result |
|---|---|
| Historical vLLM call replay, recorded configurations | 56 / 56 calls exact |
| Convolution, reference's native fixed launch | 52 / 52 outputs byte-identical |
| Scan, native autotuning in both engines | 52 / 52 outputs and boundary states byte-identical |
| Separately labeled cumsum-pinning diagnostic | 52 / 52 outputs and states byte-identical |

Both native autotuners selected cumsum `BLOCK_SIZE_H=2` in this run. That choice
was not pinned for the native comparisons. The complete five candidate lists
are checked at runtime against vLLM; their source decorators also match by AST.
This is one hardware/runtime/tuning run, not a guarantee of equal winners in
future processes. The earlier fixed-tile job 4012983 was cancelled and provides
no validation result.

The check stops before gate/normalization and uses common intermediate inputs.
It does not establish complete Mamba-layer, full-model, decode, graph, or
sampling parity. All saved results use zero differing bytes as the criterion;
no numeric tolerance or token-only shortcut was used.

## Grouped gated normalization (2026-09-25)

Inference with gate-before-norm now follows the reference's Torch operation
order and compiles the gate, group reduction, cast, and weight multiplication
together. The weight and group size remain module attributes, so their compiler
specialization matches vLLM's. Passing them as independent dynamic arguments
produced a few differing bytes in the first diagnostic and was rejected.
Training and gate-after-norm retain their existing path.

GB300 job 4013222 passed 96 short-input/shape-control cases in eager and compiled
mode. Job 4013575 then tested the actual `ExtendedRMSNorm.forward` integration
on all 52 captured inputs, including full long-context chunks and original
strides: every compiled output was byte-identical to the independently compiled
installed vLLM method. Eager reference comparisons also passed. The singleton
shape controls are not actual cached-decode validation.

`examples/inference/parity/replay_mamba_norm.py` accepts `--minf`, `--vllm`, and
`--out-dir` for full-tensor capture runs. It preserves input strides, records
reference source hashes and runtime versions, and fails on any differing byte.
Whole-model compilation can change fusion boundaries; its validation remains
separate and pending.

## Resumed native-tuning audit (2026-09-25)

The initial native pass above does not generalize to independent tuning runs.
The later integrated eager run still differs from vLLM in its first Mamba
block. A replay prepared during that investigation incorrectly reshaped the
256 B/C channels as one group of 256 states. The actual model has two groups
of 128 states. Correcting the diagnostic removed its large replay error;
small differences against the recorded live outputs remained.

GB300 diagnostic job 4014418 tested all 23 scan configurations on four TP
shards. Both implementations were byte-identical in all 92 comparisons when
given the same inputs and settings. Job 4014640 then varied only the cumsum
configuration. On ranks 2 and 3, `BLOCK_SIZE_H=2` reproduces the recorded vLLM
scan output, while `BLOCK_SIZE_H=4` reproduces the recorded Megatron output;
each differs from the other by one BF16 value. Both implementations agree
under each shared configuration. These are explicitly pinned diagnostic
controls and do not count as native parity passes.

Fresh native runs also select different cumsum winners, including H=2, H=4,
and H=8. Matching the candidate lists is insufficient to ensure equal native
winners. The runtime policy has additional gaps: the installed vLLM warms a
128-token profile before cache allocation, with and without initial states,
and passes BF16 dt bias. Megatron's current prefill call passes FP32 dt bias,
which changes Triton's implicit dtype cache key. These differences still
need resolution without pinning a production winner.

Actual compiled vLLM job 4013854 completed seven cases. Its 23 saved rank-0
logit tensors are byte-identical between capture-off and capture-on controls.
Its final-prefill logits differ from eager vLLM on all five tested histories,
so the eager reference cannot substitute for the production target.

The old hook also admitted auxiliary Megatron decode forwards whose reused
input buffers still contained prompt tokens. Such captures can have different
padded shapes across ranks and must not be interpreted as matching distributed
forward calls. Prefill capture now requires the actual inference context to
be in prefill mode. Guarded Megatron job 4014731 and vLLM job 4014596 validate
that guard on all four ranks: one 128-token prefill, with embeddings, input
norm, projection, convolution, and scan input values agreeing. Megatron
natively selected cumsum H=4 on every rank; vLLM selected H=2. Scan outputs
differ by 27/2/1/1 bytes, and the first block output differs by 496 bytes on
each rank. Both endpoint capture controls pass, but cross-engine raw logprobs
differ despite choosing the same token. Full-model, production Megatron,
cached-decode, and complete sampler/RNG parity remain unestablished.

## Remaining work

| Stage | Required proof |
|---|---|
| Embedding, input normalization, projections | Full-token, identical-input equality with matching GEMM shapes and padding |
| Mamba gate/RMSNorm | Actual compiled reference, gate/rounding placement, weights, group reduction |
| Mamba output projection and residual | Partial GEMM, TP reduction, residual-add rounding separately |
| Mamba decode and cache | Same history/state, chunk position, cache precision, graph execution |
| MoE routing | Weight mapping, FP32 logits, bias, top-k IDs/probabilities and tie handling |
| Routed/shared experts | FC1 storage, ReLU squared, FC2 weighting, routed scale, summation, TP/EP layout |
| Attention | Weight/head mapping, Q/K/V, attention backend and cache, projection/reduction |
| Final norm and LM head | Full logits, vocabulary slicing, output dtype and raw logprobs |
| Sampling | Actual backend, processed distribution, shared draws, then RNG/scheduling parity |
| Whole model | Fixed histories and forced tokens first, then greedy and stochastic generation |

Do not claim a row is complete solely because sampled tokens match or an
upstream numerical difference becomes smaller. Preserve intermediate results
so the first remaining divergence can be identified after each change.


## Continued production audit

The current experimental adapter runs TP4/EP1/ETP4 and uses the reference TP
collective implementation. CUDA graph capture now registers its custom
all-reduce buffers before replay. Production jobs have completed all graph
profiles and the seven prefill/decode cases, but full byte parity is pending.

Actual exported vLLM compiler kernels identified additional rounding boundaries:

- MoE shared/routed scaling and addition must be compiled together.
- Residual RMSNorm follows the vLLM IR formula. The compiler keeps the residual
  in FP32 across an intervening MoE and materializes BF16 at a Mamba or attention
  partition boundary.
- Grouped gated RMSNorm depends on the compiler's initial shape and cache.
  Warm the helper at the reference scheduler token budget before smaller graph
  batches. The tested reference budget is 8,480 tokens. Match the scheduler
  budget too, so long prompts have the same chunk boundaries.

The isolated reference-method compilation used in the earlier gate tests did
not cover the full-model compiler decisions. Exported-kernel controls now cover
that distinction. Native autotuners remain unpinned; independent reference
processes can select cumsum tiles that change bytes.

For chunked prefill, sampler observers also record intermediate calls whose
sampled tokens the scheduler discards. Compare the suffix returned to the
client, verify its token IDs, and compare decode logits only while preceding
histories match. Same-run observer controls pass full logits in the completed
production runs. Sampling/RNG lifecycle and cached recurrent/KV state remain
separate unfinished acceptance gates.

## nemoRL scope and latest controls (2026-09-26)

Sampling coverage is limited to the SWE flow under investigation: temperature
1, top-p 1, disabled top-k, no request seed, and no speculative decoding.
Job 4016659 compared the actual Megatron batch sampler with vLLM's Sampler
using common captured logits and aligned RNG states. All 28 rank/batch cases
agreed on tokens, RNG advancement, and raw/processed selected logprobs.
Request scheduling and model outputs remain separate checks; broader sampling
features are outside this investigation.

The grouped gate must retain its projection stride (2,576), including through
the prefill reshapes. Copying it to a compact tensor changes the compiler
specialization. Actual captured gate output now matches the exported
whole-model reference kernel on all four ranks.

The 8,480-token scheduler budget requires storage for 8,512 padded tokens.
The context now separates logical scheduling capacity from padded buffer
capacity. Distributed regression tests pass for MHA and hybrid contexts at
both 65 and 8,480 logical tokens. The synchronized 18,964-token prompt and
seven-case production run complete without the previous KV append fault.
Full logits still differ. A scan-configuration diagnostic control now matches
the entire first Mamba block, including its collective; the first remaining
difference is in the following MoE block.

## Runtime normalization and full-forward controls

The audit now observes the existing compiled GEMM calls and actual Inductor
normalization launches. Output bytes map those calls to all 52 model layers.
Both observers pass capture-off/on full-logit equality on all four ranks.
This replaces ambiguous lookup by exported kernel name: the first MoE norm
uses the `_2` occurrence, while the previously selected `_0` occurrence
belongs to normalization after attention.

The remaining layer39 normalization difference came from compiled output
order: exposing the residual first changes Inductor's residual store placement
and floating-point fusion. Source-v9 preserves the public return order outside
the compiled helper. All53 common-input normalization controls now pass on all
four ranks (job4017484); all23 grouped gated norms also pass (92 comparisons).

With reference cumsum and per-layer normalization launch choices controlled
only in the diagnostic worker, job4017485 matches every normalized input,
reduced block output, and final hidden state across all52 layers and four
ranks. This is a diagnostic pass. Native production job4017486 completes all
seven cases with CUDA graphs, but full logits still differ; both eight-step
decode histories remain aligned. Native tuning and cached-state parity are
still pending.

A reference repeatability check also matters: two independent native vLLM
runs produce identical full logits, while a third differs145934 bytes for
the same checkpoint and prefix. Capture-off/on controls pass for all three.
Native tuning variability must be reported alongside any controlled parity
result; a controlled pass cannot replace the production acceptance checks.


## Decode convolution and cached-state controls

The decode convolution previously multiplied and accumulated in FP32. vLLM
rounds each product to input dtype before sequential FP32 accumulation from
the bias. Source-v10 enables that rounding and the reference SiLU expression
only in the parity profile. On23actual layers ×8constructed updates ×4ranks,
all736 convolution outputs and logical histories now match; isolated SSU
outputs and FP32 states also match. The27convolution unit cases pass perrank,
including indexed batches1/4/64, inactive slots, and CUDAgraph replay.

Live production graph captures map physical cache slots to the same logical
request. Megatron's width4 convolution cache is compared through the final3
history entries read by the next update. All128 observer-off/on full-logit
controls pass. After the fix, first-layer SSM states remain exact through all
8steps wherever the prefill state matched (3ranks at128tokens;2at5808tokens).
Other ranks already differ at prefill, and later-layer cached states still
differ. This verifies the decode fix in the actual flow without claiming
full native or cached-state parity.


### Graph sizing scope correction

The original SWE MINF recipe uses `num_cuda_graphs=-1`. The audit's bounded
four-graph override created a152-row minimum decode graph. Automatic graph
sizing includes4-row graphs atTP4. The MoE replay verifies that1versus4rows
is byte-exact at all6stages for128common input rows on everyrank. Padding to
152changes router logits/weights and some routed/combined outputs. This
specific152-row decode mismatch belongs to the probe configuration and is
not evidence about the original RL divergence. Subsequent full-flow checks
restore automatic graph sizing; their matched8480-token budget andTP4/EP1
adapter remain explicit audit conditions.


### Prefill graph policy

The experimental parity profile now disables full prefill graphs while retaining
compiled primitives and decode graphs. The installed vLLM profile captures full
graphs only for decode. Padding a 128-token prefill to 152 rows changes the FP32
router GEMM and propagates through routing weights and expert outputs; the
common-input replay verifies this on every rank. The context applies this policy
when `inference_vllm_parity` is enabled, even if non-decode graphs were requested.

Jobs 4018277 and 4018332 established the unpadded behavior using a launch override
before the context change. They agree at prefill and move the first observed
Mamba state difference from layer 2 to layer 21 for the 128-token prefix. Full
logits still differ. Automatic graph sizing remains necessary: the old explicit
four-graph override is not representative of rkirby's recipe. The EP1 adapter
results do not establish equivalence with the original EP4 rollout flow.


### Residual components across MoE

The residual remains mathematically FP32 across MoE, but its physical storage
also matters. The reference retains two BF16 components across the opaque MoE
and recomputes their sum in the following norm. Storing an extra FP32 carry
from the preceding norm changes Inductor fusion. On fresh exact inputs, the
old helper differs by one byte at layer 36 on ranks 0 and 1 (R0_BLOCK=4096).
The same layer matches on ranks 2 and 3 (R0_BLOCK=2048); the MoE collective
then propagates the two ranks' difference.

The updated helper retains BF16 components and exposes only the normalized
computation at the MoE input boundary. Replay 4019600 matches all 53 actual
reference calls on all four ranks with recorded launch choices, including
residual outputs where present. Native replay still differs where tuning
selects different arithmetic. These controlled results do not establish full
native parity. Full prefill and cached-generation validation follows.


### Residual compilation policy and final norm

The norm helpers now specialize width and epsilon while keeping batch size
dynamic. They warm at the logical scheduler budget before decode graph capture;
the audited budget is 8,480 tokens. This matches the reference's size hint of
16,384, replacing the candidate's previous 608-row initial call (hint 1,024)
and runtime epsilon. Native candidate lists and benchmark selection are intact.
Warmup preserves distinct hidden/carry storage to avoid compiling an alias-only
specialization.

The final norm reuses its hidden buffer and has no residual output, matching
the reference's actual final compiler partition. Replay 4019715 passes all
212 matched-launch norm checks; native final-norm results match on all four
ranks. Full native fixture counts are 43/40/40/53 of 53 by rank. Independent
tuning still selects different tiles in some earlier norms. Source-v13 full
model checks are required before claiming these changes reach native parity.

### Attention request capacity (source-v14)

The reference FA4 heuristic selects splits using the graph's request count.
TP4 aligned model buffers previously forced a minimum four-request attention
shape, while vLLM captures one and two. On common inputs, short-context output
matches but long-context output changes; page sizes 256/1056 are equivalent in
these fixtures. Diagnostic 4019867 validates this on all 24 rank/query/case
fixtures and verifies eager/graph agreement.

The parity context now captures additional one/two-request attention variants
while retaining four-row model/Mamba/SP buffers. The graph key includes attention
capacity and selection rejects variants too small for the real batch. Attention
slices queries to that capacity and zero-pads its output back to model storage.
FA4 still uses its unmodified native split heuristic; no split count is pinned.
Prefill and other profiles retain their existing graph dimensions. Full native
and controlled integration validation is pending; do not infer full parity from
this structural fix or the isolated diagnostic.

### Native cache policy (source-v15)

The actual nemoRL Megatron worker calls `configure_dynamo_cache()`, which sets
Inductor `autotune_local_cache=False`. The installed vLLM generation worker uses
`True`. Source-v15 sets the compile option to `True` on the three parity helper
compilations (residual norm, grouped gate and MoE combine), matching reference
cache policy without changing the worker's global training settings or fixing
any tuning winner. Source-v14's successful fixed-choice full-flow comparison is
preserved; policy validation and final seven-case checks remain pending.

### Additional native policies (source-v16/v17, pending verification)

The parity adapter applies vLLM's default SSD disk autotuning cache policy to
Megatron's five already-imported forward autotuners, honoring explicit
TRITON_CACHE_AUTOTUNING overrides. Their candidates and winners remain native.

Grouped gate compilation specializes group width and epsilon with dynamic token
count. MoE combine similarly specializes the scaling factor and hidden width,
reuses the shared-output buffer, and first compiles at the maximum scheduler
budget. These match the reference's workload and alias behavior for tuning.
The seven-case conditional gate still fails prefixes 127/129/18964 at source-v14;
first twelve-layer captures are running to localize those independent gaps.

### Real prefill rows and SSD signed zeros (source-v18)

Prefill MoE primitives and row-parallel projections/collectives now use the
actual token count, then restore padded model storage. On the recorded
8480-row first chunk, local projection results are identical at 8480/8512 rows,
but native NCCL reduction changes 6,346,419 bytes when padded. Replaying the
8480-row collective reproduces the reference exactly on all four ranks.
At 127/129 rows, padding changes router GEMM and routed-expert rounding.
Decode retains its captured physical shape.

The SSD forward kernels use the reference's explicit exp2(log2(e)*x) expression.
A common-input second-chunk scan differed in one signed-zero byte with tl.exp;
all earlier intermediates matched. The explicit expression restores the byte.
Full-model validation follows with all five SSD choices controlled per prefill
chunk, accounting for native choices that vary with the initial-state dtype key.

### Chunk alignment and worker initialization (source-v19/v20)

SSD cache policy is configured before `MambaInferenceStateConfig.from_model`
warms the kernels. Initializing it in the lazily constructed adapter was too
late. A native worker verifies all five policies before its first warmup on
all four ranks.

Source-v18 passes six of the seven complete scenarios, including all logits
and caches through both eight-token trajectories. The long prompt's first
8,480-token prefill also matches throughout. Its later prefills revealed that
vLLM aligns SSD chunks to the original sequence: after 8,480 tokens, the next
chunk starts with 96 tokens; after 16,960, it starts with 64. Megatron previously
restarted at 128 on every prefill. Source-v20 supplies prior context lengths to
CPU metadata construction and reserves room for an additional partial chunk
per request. The complete source-v20 validation is pending.

The nemoRL worker environments must also use the same compiler toolchain.
The audited container provides PTX assembler 13.2, while the Megatron worker
otherwise falls back to Triton's bundled Blackwell assembler 12.9. Propagate
the reference compiler into the worker environment before initialization:

```bash
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas
```

This is a runtime prerequisite for this pinned container, not a kernel winner
override. Actual generated helper policies now match all recorded selection
fields, including compiler fingerprint (native worker 4020595 and policy
inspection 4020619). The audited
profile uses TP4/EP1/ETP4, prefix caching disabled, and a logical token budget
of 8,480. It does not establish equivalence with the original EP4 layout.

### Prefill attention request count (source-v21)

Source-v20 matches all logits and caches through the first two long-prefill
chunks (16,960 tokens). The final 2,004-token chunk first differs after attention
5. Prefill metadata still described four padded requests, while vLLM described
one. With 18,964 cached tokens, FA4 selects two splits for one request and no
SplitKV for four. Common-query replay on all four GPUs verifies that matching
the actual request count restores exact output with native selection; page
sizes 256/1056 and extra query rows alone do not change the result.

Source-v21 uses actual request counts for ungraphed parity attention and slices
prefill queries to the real token count. It restores model-buffer padding after
the call. Decode graph buckets retain their separate attention capacities.
Full-model validation is pending. The native source-v20 worker's 44 compiled
policy comparisons and all five SSD policy comparisons per rank now pass,
including compiler identity. Source-v21 does not change these helpers or their
warmup/selection policies.
