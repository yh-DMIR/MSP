#!/usr/bin/env bash
set -e

# =========================
# 基本配置
# =========================
PYTHON=python
SCRIPT=benchmark_orion_msp_dynamic.py

DATA_ROOT=limix
OUT_ROOT=results/msp_dynamic
mkdir -p "${OUT_ROOT}"

# =========================
# 多进程时建议限制 CPU 线程争用
# =========================
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =========================
# Orion-MSP 本地 ckpt 路径（离线机器必填）
# =========================
LOCAL_CKPT_PATH="./checkpoints/Orion-MSP-v1.0.ckpt"

# =========================
# GPU/worker 设置（8 卡 AMD）
# 如果你想指定具体用哪些卡，比如 0-7：
#   GPUS="0,1,2,3,4,5,6,7"
# =========================
WORKERS=8
GPUS="0,1,2,3,4,5,6,7"

# =========================
# 通用运行参数
# device 建议用 cuda:0（在每个 worker 里，它指的是该 worker 可见的那一张卡）
# =========================
COMMON_ARGS="
  --workers ${WORKERS}
  --gpus ${GPUS}
  --device cuda:0
  --batch-size 4
  --n-estimators 32
  --norm-methods none,power
  --feat-shuffle latin
  --softmax-temp 0.9
  --ckpt ${LOCAL_CKPT_PATH}
  --verbose
"

# =========================
# 1️⃣ TALENT
# =========================
echo "===== Running Orion-MSP: TALENT (dynamic ${WORKERS} workers) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/talent_csv" \
  --out-dir "${OUT_ROOT}/talent" \
  --all-out "${OUT_ROOT}/msp_talent.ALL.csv" \
  --summary-txt "${OUT_ROOT}/msp_talent.summary.txt" \
  ${COMMON_ARGS}

# =========================
# 2️⃣ OpenML-CC18
# =========================
echo "===== Running Orion-MSP: OpenML-CC18 (dynamic ${WORKERS} workers) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/openml_cc18_csv" \
  --out-dir "${OUT_ROOT}/openml_cc18" \
  --all-out "${OUT_ROOT}/msp_openml_cc18.ALL.csv" \
  --summary-txt "${OUT_ROOT}/msp_openml_cc18.summary.txt" \
  ${COMMON_ARGS}

# =========================
# 3️⃣ TabZilla
# =========================
echo "===== Running Orion-MSP: TabZilla (dynamic ${WORKERS} workers) ====="
${PYTHON} ${SCRIPT} \
  --root "${DATA_ROOT}/tabzilla_csv" \
  --out-dir "${OUT_ROOT}/tabzilla" \
  --all-out "${OUT_ROOT}/msp_tabzilla.ALL.csv" \
  --summary-txt "${OUT_ROOT}/msp_tabzilla.summary.txt" \
  ${COMMON_ARGS}


echo "✅ All datasets finished."
echo "Results saved in: ${OUT_ROOT}"
