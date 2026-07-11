# SPEED-Bench MTP Acceptance-Rate Benchmark (Megatron)

Measure MTP speculative-decoding **acceptance rate / acceptance length** for
Megatron dynamic inference on the
[NVIDIA SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)
qualitative split (880 rows × 11 categories). It mirrors the reference vLLM
`specdec_bench` setup but drives a Megatron server via `--engine MEGATRON`.

---

## What you need

| Thing | What / where | Notes |
|---|---|---|
| **Megatron-LM** | this repo, on the branch with the SPEED-bench server changes | Provides the server + these scripts. The server runs from your checkout (mounted into the container), so **your working tree is what executes** — make sure it has the changes below. |
| **specdec_bench** | a clone on the `megatron_inference` branch (has the `MegatronModel` client) | Path passed via `SPECDEC_BENCH`. |
| **mcore inference container** | an enroot `.sqsh` with torch/TE/Megatron deps | Path via `CONTAINER_IMAGE`. |
| **MTP checkpoint** | a Megatron dist-checkpoint trained with MTP heads | Path via `CKPT`. The example is Nemotron-3-Super-120B. |
| **HF tokenizer** | must match the checkpoint | `TOKENIZER`; cached under `HF_HOME`. |
| **Resolved SPEED-Bench data** | a local parquet (step 1 stages it) | `SPEED_BENCH_DATA`. |
| **SLURM** | a GPU partition + an account/QOS you can submit to | Edit the `#SBATCH` lines or override on the CLI. |

### Required Megatron-LM changes (server side)
The server returns per-request acceptance data that the client consumes. These
must be present in your checkout (they are on the SPEED-bench branch):
- `inference_request.py` — `acceptance_step_lengths` field (+ carried through `merge()`).
- `engines/dynamic_engine.py` — records per-step emitted-token counts.
- `.../dynamic_text_gen_server/endpoints/completions.py` — returns
  `acceptance_step_lengths`, `ttft`, `tpot` on each choice.
- The **MTP + tensor-parallel acceptance fix** (SP/MTP all-gathers) — without it,
  acceptance is badly degraded at `TP > 1`.

---

## Step 1 — Stage the data (once, no internet needed)

```bash
REF_SPEED_DATA=<dir-with-resolved-qualitative-parquet> \
  bash scripts/speed_bench/prepare_speed_data.sh
# -> copies the resolved qualitative parquet to
#    $HOME/speed-bench-data/speed/qualitative/  (override with OUTPUT_DIR)
```

Why copy instead of download: re-resolving SPEED-Bench pulls from ~14 source
datasets, several **gated on the Hub** (e.g. `cais/hle`) that fail without
per-dataset access. The pre-resolved parquet sidesteps that. To force a full
re-download+resolve (needs internet + gated access): `RESOLVE=1 bash ...`
(it bootstraps a Python ≥3.10 venv, since specdec_bench needs 3.10+).

---

## Step 2 — Run the benchmark (server + client, one allocation)

```bash
cd <your Megatron-LM checkout>

SPEC_TOKENS=2 \
CKPT=<path-to-your-mtp-checkpoint> \
CONTAINER_IMAGE=<path-to-mcore-container.sqsh> \
SPECDEC_BENCH=<path-to-your-specdec_bench-checkout> \
SPEED_BENCH_DATA=$HOME/speed-bench-data/speed/qualitative \
EXP_NAME=super_mtp_ar \
sbatch -A <your_account> --gres=gpu:8 scripts/speed_bench/run_speed_bench.slurm
```

The job: starts the 8-GPU MTP server → waits for `http://0.0.0.0:5000` → runs the
specdec_bench acceptance client against localhost → tears the server down.

**Smoke test first:** add `NUM_REQUESTS=8` for a ~2-min end-to-end check (first 8
rows) before the full 880-row sweep.

**Match the vLLM reference draft length:** the reference used `draft_length=11`.
The example checkpoint has one trained MTP head + `--mtp-use-repeated-layer`, so
you can set `SPEC_TOKENS=11` to repeat it to 11 draft positions (expect a large
memory increase — see OOM tuning).

### Knobs (all env vars; override on the `sbatch` line)

| Var | Default | Meaning |
|---|---|---|
| `SPEC_TOKENS` | `2` | MTP draft tokens. Repeated-layer allows > trained heads. |
| `TOKENIZER` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` | HF tokenizer; **must match `CKPT`** (client sends token ids). |
| `TP` / `EP` | `2` / `8` | Tensor / expert parallel. Product-of-parallelism must fit `--gres`. |
| `NUM_GPUS_PER_NODE` | `8` | Must equal `--gres=gpu:N`. |
| `BUFFER_GB` | `12` | Dynamic-batching buffer (GB). **Lower if OOM.** |
| `MAX_REQUESTS` | `32` | Max concurrent server requests. **Lower if OOM** (esp. high `SPEC_TOKENS`). |
| `CONCURRENCY` | `8` | Client in-flight requests. Doesn't affect AR (greedy) — only speed. Effective server capacity ≈ `DP × MAX_REQUESTS`. |
| `OSL` | `512` | Max output tokens per request. |
| `NUM_REQUESTS` | (all) | Cap request count (smoke test). |
| `SPEED_BENCH_CATEGORY` | (all) | Single-category filter. |
| `SPECDEC_BENCH` | **(required)** | Client checkout (must be on `megatron_inference`). |
| `SPEED_BENCH_DATA` | **(required)** | Staged parquet dir (from step 1). |
| `CKPT` | **(required)** | Megatron MTP checkpoint dir. |
| `CONTAINER_IMAGE` | shared mcore image | Override with your container. |
| `EXTRA_MOUNTS` | `/home:/home` | Extra `src:dst` mounts (the example ckpt is under `/home`). |
| `SAVE_DIR` | `<EXP_DIR>/ar_results` | Where results/plots land. |

---

## Step 3 — Results & interpretation

Under `SAVE_DIR` (default `<EXP_DIR>/ar_results/`):
- `specbench_results.json` — per-request + per-category AR, overall average,
  acceptance-length histogram.
- `acceptance_rate_analysis.png` — AR by category / length / distribution.
- `specbench_responses.jsonl` — generated responses.
- `<EXP_DIR>/ar_benchmark.log` — client stdout incl. the per-category table.
- `<EXP_DIR>/server_rank0.log` — server log, incl. cumulative
  `spec (cumul): accept X% [t1=.. t2=..]` per-position lines.

**AR = acceptance *length*** (mean tokens emitted per step, in `[1, SPEC_TOKENS+1]`),
matching vLLM specdec_bench. The server log's `accept X%` is an acceptance
**rate** (`accepted/proposed`) — different metric, related by
`length ≈ 1 + (rate/100) × SPEC_TOKENS`.

Watch the per-position rates (`t1, t2, …`): a healthy MTP model **decays
monotonically** (e.g. `t1=73% → t2=48% → …`). A collapse — especially only at
`TP>1` — indicates the MTP+TP acceptance bug (needs the SP/MTP all-gather fix).

---

## Porting to another cluster / model

- **Cluster:** override `-A/-p/-q`, `--gres`, `CONTAINER_IMAGE`, and the
  `HF_HOME`/path defaults. The CPU request auto-derives from `SLURM_CPUS_ON_NODE`.
  If your checkpoint isn't under `/lustre`, add it to `EXTRA_MOUNTS`.
- **Model:** the `MODEL_ARGS` block in `speed_bench_worker.sh` is specific to
  Nemotron-3-Super-120B. For a different MTP checkpoint, replace `MODEL_ARGS`
  with that model's arch flags (mirror one of `scripts/run_eval_server_*.sh`) and
  set `CKPT`/`TOKENIZER`/`TP`/`EP`.

---

## Troubleshooting (things that bit us)

| Symptom | Cause / fix |
|---|---|
| `sbatch: Invalid qos specification` | Your account doesn't grant that QOS. Default here is `-q normal`; check `sacctmgr -n -p show assoc user=$USER format=QOS`. |
| `srun: More processors requested than permitted` | The node has fewer CPUs than requested. The script auto-derives `cpus-per-task` from `SLURM_CPUS_ON_NODE`; if it still fails, the allocation is short on CPUs — check your `--gres`/partition. |
| `>> '--exit-on-missing-checkpoint' set ... exiting.` | The container can't see `CKPT`. If it's outside `/lustre` (e.g. `/home`), add it to `EXTRA_MOUNTS`. |
| `CUDA out of memory` during engine init | Lower `BUFFER_GB` (e.g. 8) and/or `MAX_REQUESTS` (e.g. 16). High `SPEC_TOKENS` multiplies per-request SSM speculative-state memory. |
| `DatasetNotFoundError: cais/hle ... gated` (data prep) | You ran `RESOLVE=1` without gated access. Use the default copy path instead. |
| `TypeError: unsupported operand type(s) for |` (data prep) | Login-node Python < 3.10. The prep script bootstraps a 3.10+ venv automatically; or set `PYTHON=python3.12`. |
| Client `--model_dir` required error | Handled — the worker passes `--model_dir` (informational for the MEGATRON engine). |
| Acceptance much lower at `TP>1` than `TP=1` | The MTP+TP (SP/MTP all-gather) fix isn't in your checkout. |
