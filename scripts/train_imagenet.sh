#!/bin/bash
# Launch the ImageNet training sweeps used in the LPC paper.
# Override defaults by exporting HOURS/GPUS/DATASET_DIR/RESULTS_DIR/OUTPUT_DIR before invoking.

set -euo pipefail

HOURS=${HOURS:-20}
GPUS=${GPUS:-2}
CONFIG=${CONFIG:-configs/imagenet.yml}
DATASET_DIR=${DATASET_DIR:-/path/to/imagenet}
RESULTS_DIR=${RESULTS_DIR:-results}
OUTPUT_DIR=${OUTPUT_DIR:-jobs_outputs}
ID_NAME=${ID_NAME:-imagenet}

python scripts/imagenet_generate_submit_slurm_jobs.py \
  --hours "${HOURS}" \
  --n-gpus "${GPUS}" \
  --config "${CONFIG}" \
  --id-name "${ID_NAME}" \
  --dataset-dir "${DATASET_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --skip-directory-prompt
