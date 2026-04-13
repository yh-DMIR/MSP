#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_EXTERNAL_DATASET_ROOT="/vast/users/guangyi.chen/causal_group/zijian.li/dmir_crl/lyh/TabDPT/dataset"

if [[ -z "${DATASET_ROOT:-}" ]]; then
  if [[ -d "${DEFAULT_EXTERNAL_DATASET_ROOT}" ]]; then
    DATASET_ROOT="${DEFAULT_EXTERNAL_DATASET_ROOT}"
  else
    DATASET_ROOT="${REPO_ROOT}/dataset"
  fi
fi

mkdir -p /tmp/$USER/comgr
export TMPDIR=/tmp/$USER
export TEMP=/tmp/$USER
export TMP=/tmp/$USER
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_ROOT}/src"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-${REPO_ROOT}/benchmark_orion_msp_classification_failed_amd.py}
ROOT=${ROOT:-${REPO_ROOT}}
BENCHMARKS=${BENCHMARKS:-openml_cc18_csv=${DATASET_ROOT}/openml_cc18_72,tabzilla_csv=${DATASET_ROOT}/tabzilla35}
DATASET_NAMES=${DATASET_NAMES:-OpenML-ID-1468.csv,OpenML-ID-1485.csv,OpenML-ID-41147.csv}
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/ckpt/Orion-MSP-v1.0.ckpt}
OUT_DIR=${OUT_DIR:-${REPO_ROOT}/results/OrionMSP_official_classification_failed_rerun}
WORKERS=${WORKERS:-4}
GPUS=${GPUS:-0,1,2,3}

echo "Re-running datasets: ${DATASET_NAMES}"
echo "Benchmarks: ${BENCHMARKS}"
echo "Results will be written to: ${OUT_DIR}"

${PYTHON} ${SCRIPT} \
  --root "${ROOT}" \
  --benchmarks "${BENCHMARKS}" \
  --dataset-names "${DATASET_NAMES}" \
  --model-path "${MODEL_PATH}" \
  --checkpoint-version Orion-MSP-v1.0.ckpt \
  --out-dir "${OUT_DIR}" \
  --workers "${WORKERS}" \
  --gpus "${GPUS}" \
  --device cuda:0 \
  --batch-size 4 \
  --n-estimators 32 \
  --norm-methods none,power \
  --feat-shuffle latin \
  --softmax-temp 0.9 \
  --test-size 0.2 \
  --verbose
