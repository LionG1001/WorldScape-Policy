#!/usr/bin/env bash
set -euo pipefail

# MUSA / torchada, interactive text, native RoboCOIN simulated EEF6D data.
# Runs the full schedule by default; set RUN_STEPS for a short smoke test.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/recipes/common/musa_runtime.sh"
: "${DATA_ROOT:?Set DATA_ROOT to the RoboCOIN LeRobot dataset root}"
: "${PRETRAINED_MODEL_PATH:?Set PRETRAINED_MODEL_PATH to the full native WSP checkpoint}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to the local umt5-xxl tokenizer}"
export DATA_ROOT PRETRAINED_MODEL_PATH TOKENIZER_DIR
export WSP_USE_TORCHADA=1 TORCHADA_FORCE_CUDA_AVAILABLE=1 DS_ACCELERATOR=musa
export WSP_PYTHON="${WSP_PYTHON:-$(command -v python)}"
export WSP_AUTO_DOWNLOAD=false WSP_MODE=interactive VISUAL_PROMPT=none
export NUM_GPUS="${NUM_GPUS:-8}"
export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}" MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export DEEPSPEED_MODE="${DEEPSPEED_MODE:-zero2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
export MAX_STEPS="${MAX_STEPS:-50000}" RUN_STEPS="${RUN_STEPS:-null}"
export LEARNING_RATE="${LEARNING_RATE:-6e-5}" LR_SCHEDULE="${LR_SCHEDULE:-linear}"
export ADAM_BETA1="${ADAM_BETA1:-0.9}" ADAM_BETA2="${ADAM_BETA2:-0.999}"
export SEED="${SEED:-62}" SAVE_STEPS="${SAVE_STEPS:-5000}" SAVE_AT_END="${SAVE_AT_END:-true}"
export RUN_NAME="${RUN_NAME:-musa-train}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/checkpoints/agilex-fold-shirt-text/$RUN_NAME}"
export WANDB_ENABLED="${WANDB_ENABLED:-false}" WANDB_MODE="${WANDB_MODE:-offline}"
export VLM_ATTN_IMPLEMENTATION=sdpa
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

[[ -f "$DATA_ROOT/meta/episodes.jsonl" ]] || { echo "Missing LeRobot episode metadata" >&2; exit 1; }
[[ -d "$PRETRAINED_MODEL_PATH" && -d "$TOKENIZER_DIR" ]] || { echo "Missing local model/tokenizer" >&2; exit 1; }
"$WSP_PYTHON" -c 'import torchada, torch; assert torch.cuda.is_available() and torch.musa.device_count() > 0, "MUSA unavailable"'
source "$ROOT/recipes/common/launcher_common.sh"
wsp_distributed_train_launch "$ROOT" "$ROOT/configs/posttrain/agilex_robocoin_musa.yaml" "$@"
