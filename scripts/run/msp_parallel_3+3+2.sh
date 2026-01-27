#!/usr/bin/env bash
set -e

PYTHON=python
SCRIPT=benchmark_orion_msp_dynamic.py

DATA_ROOT=limix
OUT_ROOT=results/msp_parallel_3+3+2
mkdir -p "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# （可选）把 HF 缓存放到指定目录，避免写到 ~/.cache
# export HF_HOME=/path/to/hf_home
# export HUGGINGFACE_HUB_CACHE=/path/to/hf_home/hub

CKPT_VERSION=Orion-MSP-v1.0.ckpt
LOCAL_CKPT_PATH="./checkpoints/${CKPT_VERSION}"
mkdir -p "$(dirname "${LOCAL_CKPT_PATH}")"

# --- Warmup: 先预下载一次，避免并行 workers 同时抢下载锁看起来“0%卡住” ---
${PYTHON} - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(repo_id="Lexsi/Orion-MSP", filename="Orion-MSP-v1.0.ckpt")
print("Downloaded/cached at:", p)
PY

COMMON_ARGS="
  --device cuda:0
  --batch-size 4
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --ckpt ${LOCAL_CKPT_PATH}
  --verbose
"

echo "🚀 Launching TALENT(0,1,2) + TabZilla(3,4,5) + OpenML-CC18(6,7) in parallel..."

# TALENT
(
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/talent_csv" \
    --out-dir "${OUT_ROOT}/talent" \
    --all-out "${OUT_ROOT}/msp_talent.ALL.csv" \
    --summary-txt "${OUT_ROOT}/msp_talent.summary.txt" \
    --workers 3 \
    --gpus 0,3,2 \
    ${COMMON_ARGS}
) &

# TabZilla
(
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/tabzilla_csv" \
    --out-dir "${OUT_ROOT}/tabzilla" \
    --all-out "${OUT_ROOT}/msp_tabzilla.ALL.csv" \
    --summary-txt "${OUT_ROOT}/msp_tabzilla.summary.txt" \
    --workers 3 \
    --gpus 1,4,7 \
    ${COMMON_ARGS}
) &

# OpenML-CC18
(
  ${PYTHON} ${SCRIPT} \
    --root "${DATA_ROOT}/openml_cc18_csv" \
    --out-dir "${OUT_ROOT}/openml_cc18" \
    --all-out "${OUT_ROOT}/msp_openml_cc18.ALL.csv" \
    --summary-txt "${OUT_ROOT}/msp_openml_cc18.summary.txt" \
    --workers 2 \
    --gpus 6,5 \
    ${COMMON_ARGS}
) &

wait
echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
