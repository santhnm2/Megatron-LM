#!/bin/bash
# Stage the SPEED-Bench qualitative split for the acceptance-rate benchmark.
#
# By default this COPIES the already-resolved parquet from a reference location
# into your own space -- reproducible, offline, and it avoids re-resolving the
# external source datasets (several are GATED on the Hub, e.g. cais/hle, and
# fail without per-dataset access). Run this once before the benchmark.
#
# Output: ${OUTPUT_DIR}/speed/${CONFIG}/*.parquet
# Point the benchmark at it:  SPEED_BENCH_DATA=${OUTPUT_DIR}/speed/${CONFIG}
#
# Usage:
#   bash prepare_speed_data.sh                          # copy qualitative (default)
#   REF_SPEED_DATA=/path/to/qualitative bash ...        # copy from a different source
#   RESOLVE=1 bash prepare_speed_data.sh                # re-download+resolve instead
#                                                       # (needs internet + gated HF access)
set -euo pipefail

CONFIG="${CONFIG:-qualitative}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/speed-bench-data}"
# Directory holding an already-resolved SPEED-Bench '${CONFIG}' parquet to copy
# from (fully resolved: 880 rows, no placeholders). Required unless RESOLVE=1.
REF_SPEED_DATA="${REF_SPEED_DATA:-}"
DEST="${OUTPUT_DIR}/speed/${CONFIG}"

if [ "${RESOLVE:-0}" != "1" ]; then
    if [ -z "${REF_SPEED_DATA}" ] || ! ls "${REF_SPEED_DATA}"/*.parquet >/dev/null 2>&1; then
        echo "ERROR: no resolved parquet found (REF_SPEED_DATA='${REF_SPEED_DATA}')."
        echo "Set REF_SPEED_DATA to a dir with the resolved SPEED-Bench '${CONFIG}' parquet,"
        echo "or run with RESOLVE=1 to re-download+resolve (needs internet + gated HF access)."
        exit 1
    fi
    mkdir -p "${DEST}"
    cp -v "${REF_SPEED_DATA}"/*.parquet "${DEST}/"
    echo "Done. Set SPEED_BENCH_DATA=${DEST}"
    exit 0
fi

# --- Fallback: re-download + resolve via specdec_bench ----------------------
# specdec_bench uses `str | Path` unions -> needs Python >= 3.10. Login nodes
# often default to 3.9, so bootstrap a venv with a >=3.10 interpreter.
SPECDEC_BENCH="${SPECDEC_BENCH:?set SPECDEC_BENCH to your specdec_bench checkout}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
VENV_DIR="${VENV_DIR:-${SPECDEC_BENCH}/.venv-prep}"
pick_python() {
    for p in "${PYTHON:-}" python3.12 python3.11 python3.10; do
        [ -z "$p" ] && continue
        if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
            echo "$p"; return 0
        fi
    done
    return 1
}
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    PY="$(pick_python)" || { echo "ERROR: need Python >= 3.10 (set PYTHON=...)"; exit 1; }
    echo "Bootstrapping venv at ${VENV_DIR} with ${PY} ($(${PY} --version 2>&1))"
    "${PY}" -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install -r "${SPECDEC_BENCH}/requirements.txt"
fi
echo "Resolving SPEED-Bench config '${CONFIG}' -> ${DEST}"
cd "${SPECDEC_BENCH}"
"${VENV_DIR}/bin/python" prepare_data.py --dataset speed --config "${CONFIG}" --output_dir "${OUTPUT_DIR}"
echo "Done. Set SPEED_BENCH_DATA=${DEST}"
