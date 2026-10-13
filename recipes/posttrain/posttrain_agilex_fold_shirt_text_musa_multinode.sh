#!/usr/bin/env bash
set -euo pipefail
# Run once per node. Communication defaults follow the referenced starVLA
# gl_musa-dev launcher; HCA names must be supplied for each physical host.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MASTER_ADDR:?Set MASTER_ADDR to node 0's reachable address}"
: "${NODE_RANK:?Set NODE_RANK to the zero-based node index}"
: "${MCCL_IB_HCA:?Set MCCL_IB_HCA to this node's training RoCE adapters}"
export NNODES="${NNODES:-2}" NUM_GPUS="${NUM_GPUS:-8}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if ! [[ "$NNODES" =~ ^[0-9]+$ && "$NODE_RANK" =~ ^[0-9]+$ ]] ||
   (( NNODES < 2 || NODE_RANK >= NNODES )); then
    echo "Invalid NNODES/NODE_RANK" >&2; exit 2
fi
export MLP_WORKER_NUM="$NNODES" MLP_ROLE_INDEX="$NODE_RANK"
export MLP_WORKER_0_HOST="$MASTER_ADDR" MLP_WORKER_0_PORT="${MASTER_PORT:-29620}"
export RUN_STEPS="${RUN_STEPS:-50}"
# Tune multi-node channels separately; do not cap the single-node recipe.
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
printf 'MUSA nodes=%s node_rank=%s local_gpus=%s master=%s:%s HCA=%s\n' \
    "$NNODES" "$NODE_RANK" "$NUM_GPUS" "$MASTER_ADDR" "$MLP_WORKER_0_PORT" "$MCCL_IB_HCA"
exec bash "$ROOT/recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh" "$@"
