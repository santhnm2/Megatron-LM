#!/bin/bash
# Per-rank worker for the SPEED-Bench MTP acceptance-rate benchmark.
#
# Launched once per GPU by scripts/speed_bench/run_speed_bench.slurm (via srun,
# inside the mcore container). Every rank starts a Megatron dynamic inference
# server process (they form one distributed engine); rank 0 additionally waits
# for the server to be ready, runs the specdec_bench acceptance-rate client
# against it, then tears the server down.
#
# All configuration comes from environment variables (exported by the sbatch
# script). See scripts/speed_bench/README.md for the full list.
set -uo pipefail

# --- Distributed / per-rank setup -------------------------------------------
export RANK=${RANK:-${SLURM_PROCID:-0}}
LOCAL_RANK=${LOCAL_RANK:-${SLURM_LOCALID:-0}}
export WORLD_SIZE=${WORLD_SIZE:-${SLURM_NTASKS:-1}}
# Rendezvous: default to the first node in the allocation (works for 1 node and
# multi-node). Only set if the launcher didn't already provide them.
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" 2>/dev/null | head -n1)}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-6000}
# Per-rank caches so ranks don't collide writing JIT artifacts.
export TRITON_CACHE_DIR=/tmp/triton_cache_${RANK}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache_${RANK}
export TORCH_HOME=/tmp/torch_home_${RANK}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# --- Model / server configuration (env-overridable) -------------------------
CKPT=${CKPT:?set CKPT to your Megatron MTP checkpoint directory}
TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16}
TP=${TP:-2}
EP=${EP:-8}
SPEC_TOKENS=${SPEC_TOKENS:-2}          # MTP draft tokens (repeated-layer allows > trained heads)
BUFFER_GB=${BUFFER_GB:-12}             # dynamic-batching buffer; lower if OOM
MAX_REQUESTS=${MAX_REQUESTS:-32}       # max concurrent server requests; lower if OOM
OSL=${OSL:-512}
PORT=${PORT:-5000}
PARSERS=${PARSERS:-deepseek-r1-reasoning qwen3-coder-tool}

# Model architecture args. This block is specific to Nemotron-3-Super-120B (the
# worked example). To benchmark a DIFFERENT MTP checkpoint, replace MODEL_ARGS
# with that model's arch flags (mirror one of megatron-lm/scripts/run_eval_server_*.sh)
# and set CKPT / TOKENIZER / TP / EP accordingly.
MODEL_ARGS="\
    --hidden-size 4096 --ffn-hidden-size 2688 --seq-length 1048576 \
    --num-attention-heads 32 --num-query-groups 2 --group-query-attention --kv-channels 128 \
    --max-position-embeddings 1048576 --position-embedding-type none \
    --rotary-base 10000 --rotary-percent 1.0 --disable-bias-linear --squared-relu \
    --untie-embeddings-and-output-weights --normalization RMSNorm \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --mtp-hybrid-override-pattern none --mtp-use-repeated-layer \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --num-experts 512 --moe-layer-freq 1 --moe-ffn-hidden-size 2688 --moe-router-topk 22 \
    --moe-grouped-gemm --moe-shared-expert-intermediate-size 5376 \
    --moe-router-score-function sigmoid --moe-router-enable-expert-bias --moe-router-topk-scaling-factor 5.0 \
    --mamba-state-dim 128 --mamba-head-dim 64 --mamba-num-groups 8 --mamba-num-heads 128 \
    --hybrid-layer-pattern 'MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME/*E/*E' \
    --moe-latent-size 1024 --padded-vocab-size 131072 --model-provider hybrid \
    --inference-max-seq-length 131072"

# Parallelism, checkpoint, tokenizer, inference/perf flags.
SERVER_ARGS="\
    --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \
    --tensor-model-parallel-size ${TP} --expert-model-parallel-size ${EP} --expert-tensor-parallel-size 1 \
    --pipeline-model-parallel-size 1 --sequence-parallel \
    --load ${CKPT} --use-checkpoint-args --dist-ckpt-strictness log_unexpected \
    --tokenizer-type HuggingFaceTokenizer --tokenizer-model ${TOKENIZER} --no-use-tokenizer-model-from-checkpoint-args \
    --moe-router-dtype fp32 --moe-token-dispatcher-type alltoall --moe-permute-fusion \
    --attention-backend flash --transformer-impl inference_optimized \
    --inference-grouped-gemm-backend vllm --inference-use-synchronous-zmq-collectives --moe-shared-expert-overlap \
    --inference-dynamic-batching --inference-dynamic-batching-unified-memory-level 0 \
    --inference-dynamic-batching-max-tokens 2048 --inference-dynamic-batching-mamba-memory-ratio 0.21 \
    --enable-chunked-prefill --use-flashinfer-fused-rope \
    --inference-dynamic-batching-buffer-size-gb ${BUFFER_GB} --inference-dynamic-batching-max-requests ${MAX_REQUESTS} \
    --inference-dynamic-batching-num-cuda-graphs -1 --cuda-graph-impl local --inference-cuda-graph-scope block \
    --inference-logging-step-interval 1000 \
    --parsers ${PARSERS} \
    --host 0.0.0.0 --port ${PORT}"

# MTP speculative decoding. mtp-num-layers is the (possibly repeated) draft depth.
SPEC_ARGS=""
if [ "${SPEC_TOKENS}" -gt 0 ]; then
    SPEC_ARGS="--mtp-num-layers ${SPEC_TOKENS} --num-speculative-tokens ${SPEC_TOKENS}"
fi

SERVER_CMD="python -m tools.run_dynamic_text_generation_server ${MODEL_ARGS} ${SERVER_ARGS} ${SPEC_ARGS}"

# --- Launch the server (every rank) -----------------------------------------
if [ "${RANK}" == "0" ]; then
    { echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"; echo "WORLD_SIZE=${WORLD_SIZE}"; \
      echo "CKPT=${CKPT}"; echo "TOKENIZER=${TOKENIZER}"; echo "TP=${TP} EP=${EP}"; \
      echo "SPEC_TOKENS=${SPEC_TOKENS} BUFFER_GB=${BUFFER_GB} MAX_REQUESTS=${MAX_REQUESTS}"; \
      echo "CMD=${SERVER_CMD}"; } > "${EXP_DIR}/config.env"
fi

echo "[rank ${RANK}] ${SERVER_CMD}"
eval "${SERVER_CMD}" > "${EXP_DIR}/server_rank${RANK}.log" 2>&1 &
SERVER_PID=$!

# --- Rank 0: wait for readiness, run the client, tear down ------------------
if [ "${RANK}" == "0" ]; then
    echo "Waiting for server to be ready (this includes checkpoint load; can take ~10 min)..."
    until grep -q "Running on http://0.0.0.0:${PORT}" "${EXP_DIR}/server_rank0.log" 2>/dev/null; do
        # Bail out early if the server process died during init (e.g. OOM).
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "ERROR: server process exited before becoming ready. See ${EXP_DIR}/server_rank0.log" >&2
            echo "failed" > "${EXP_DIR}/done_status"
            exit 1
        fi
        sleep 5
    done
    echo "Server ready. Running specdec_bench acceptance-rate client..."

    SPECDEC_BENCH=${SPECDEC_BENCH:?set SPECDEC_BENCH to your specdec_bench checkout}
    # SAVE_DIR must be absolute (client runs from a `cd` subshell).
    SAVE_DIR=${SAVE_DIR:-$(pwd)/${EXP_DIR}/ar_results}
    mkdir -p "${SAVE_DIR}"
    # Client deps (the server ranks already have torch/TE): specdec_bench's
    # pinned requirements + aiohttp for the async HTTP client.
    pip install -r "${SPECDEC_BENCH}/requirements.txt" aiohttp

    EXTRA_ARGS=""
    [ -n "${NUM_REQUESTS:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --num_requests ${NUM_REQUESTS}"
    [ -n "${SPEED_BENCH_CATEGORY:-}" ] && EXTRA_ARGS="${EXTRA_ARGS} --category ${SPEED_BENCH_CATEGORY}"

    ( cd "${SPECDEC_BENCH}" && python3 -u run.py \
        --engine MEGATRON \
        --base_url "http://localhost:${PORT}" \
        --model_dir "${TOKENIZER}" \
        --dataset speed \
        --dataset_path "${SPEED_BENCH_DATA}" \
        --tokenizer "${TOKENIZER}" \
        --speculative_algorithm MTP \
        --draft_length "${SPEC_TOKENS}" \
        --output_length "${OSL}" \
        --concurrency "${CONCURRENCY:-8}" \
        --show_progress \
        --save_dir "${SAVE_DIR}" \
        ${EXTRA_ARGS} ) 2>&1 | tee "${EXP_DIR}/ar_benchmark.log"

    echo "done" > "${EXP_DIR}/done_status"
    echo "Results in: ${SAVE_DIR}"
else
    # Non-zero ranks: keep the server alive until rank 0 signals completion.
    until [ -f "${EXP_DIR}/done_status" ]; do sleep 30; done
fi

# Gracefully stop the server.
kill "${SERVER_PID}" 2>/dev/null
wait "${SERVER_PID}" 2>/dev/null
