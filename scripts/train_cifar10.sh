#!/bin/bash
# Launch the CIFAR-10 training sweeps used in the paper.
# Override defaults by exporting HOURS/GPUS/DATASET_DIR/RESULTS_DIR/OUTPUT_DIR before calling this script.

set -euo pipefail

HOURS=${HOURS:-10}
GPUS=${GPUS:-1}
CONFIG=${CONFIG:-configs/cifar10.yml}
DATASET_DIR=${DATASET_DIR:-datasets}
RESULTS_DIR=${RESULTS_DIR:-results}
OUTPUT_DIR=${OUTPUT_DIR:-jobs_outputs}
ID_NAME=${ID_NAME:-cifar10}

python scripts/generate_submit_slurm_jobs.py \
  --hours "${HOURS}" \
  --n-gpus "${GPUS}" \
  --config "${CONFIG}" \
  --id-name "${ID_NAME}" \
  --dataset-dir "${DATASET_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}"

python scripts/generate_submit_slurm_jobs_wd.py \
  --hours "${HOURS}" \
  --n-gpus "${GPUS}" \
  --config "${CONFIG}" \
  --id-name "${ID_NAME}" \
  --dataset-dir "${DATASET_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}"
