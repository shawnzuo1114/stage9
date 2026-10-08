#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VMAMBA_PRETRAINED_PATH="${VMAMBA_PRETRAINED_PATH:-../vm.pth}"
LOCAL_MAMBA_ROOT="${LOCAL_MAMBA_ROOT:-../mamba-1p1p1}"
LOCAL_CAUSAL_CONV1D_ROOT="${LOCAL_CAUSAL_CONV1D_ROOT:-../causal-conv1d}"
SAVE_ROOT="${SAVE_ROOT:-save_model/stage9_htfr}"
RUN_NAME="${RUN_NAME:-stage9_htfr}"

python train.py \
  --stage stage9_htfr \
  --enable-stage9-htfr true \
  --htfr-mode full \
  --arch stage9_htfr \
  --dataset VCM \
  --img-h 288 \
  --img-w 144 \
  --seq-len 12 \
  --num-segments 4 \
  --segment-len 3 \
  --batch-size 8 \
  --num-pos 2 \
  --vmamba-variant vmamba_tiny \
  --vmamba-pretrained "${VMAMBA_PRETRAINED_PATH}" \
  --local-mamba-root "${LOCAL_MAMBA_ROOT}" \
  --local-causal-conv1d-root "${LOCAL_CAUSAL_CONV1D_ROOT}" \
  --disable-sie false \
  --sth-num-hubs 4 \
  --sth-start-epoch 5 \
  --sth-warmup-epoch 10 \
  --irm-enable-ir-srm true \
  --irm-srm-alpha-ir 0.20 \
  --irm-srm-alpha-rgb 0.08 \
  --irm-srm-start-epoch 5 \
  --irm-srm-warmup-epoch 10 \
  --irm-lambda-struct 0.005 \
  --htfr-gate-max 0.20 \
  --htfr-temporal-temperature 1.0 \
  --htfr-freq-id-lambda 0.05 \
  --htfr-freq-id-start-epoch 5 \
  --htfr-freq-id-warmup-epoch 5 \
  --htfr-freq-xm-lambda 0.05 \
  --htfr-freq-xm-start-epoch 10 \
  --htfr-freq-xm-warmup-epoch 10 \
  --htfr-fusion-start-epoch 15 \
  --htfr-fusion-warmup-epoch 10 \
  --enable-batch-xm-retrieval true \
  --enable-xm-proto true \
  --lambda-xm-proto 0.10 \
  --rerank none \
  --max-epoch 120 \
  --lr 0.005 \
  --workers 4 \
  --use-tqdm true \
  --amp true \
  --grad-checkpoint true \
  --seed 3407 \
  --run-name "${RUN_NAME}" \
  --save-root "${SAVE_ROOT}" \
  "$@"
