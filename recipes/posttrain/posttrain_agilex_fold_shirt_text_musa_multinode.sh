#!/usr/bin/env bash
set -euo pipefail
# Run once on the controller with HOSTFILE. --node is the internal node mode.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" != "--node" ]]; then
    exec python3 "$ROOT/recipes/common/musa_multinode_launch.py" "$@"
fi
shift
: "${MASTER_ADDR:?Set MASTER_ADDR to a reachable node 0 address}"
: "${NODE_RANK:?Set NODE_RANK to the zero-based node index}"
export NNODES="${NNODES:-2}" NUM_GPUS="${NUM_GPUS:-8}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
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
    "$NNODES" "$NODE_RANK" "$NUM_GPUS" "$MASTER_ADDR" "$MLP_WORKER_0_PORT" "${MCCL_IB_HCA:-auto}"
exec bash "$ROOT/recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh" "$@"
