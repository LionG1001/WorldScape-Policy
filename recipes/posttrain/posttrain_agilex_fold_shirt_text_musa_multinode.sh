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
export MCCL_IB_DISABLE="${MCCL_IB_DISABLE:-0}"
export MCCL_IB_GID_INDEX="${MCCL_IB_GID_INDEX:-3}"
export MCCL_IB_ROCE_VERSION_NUM="${MCCL_IB_ROCE_VERSION_NUM:-2}"
export MCCL_SOCKET_IFNAME="${MCCL_SOCKET_IFNAME:-bond0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$MCCL_SOCKET_IFNAME}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}" MCCL_ALGOS="${MCCL_ALGOS:-1}"
export MCCL_BUFFSIZE="${MCCL_BUFFSIZE:-20971520}"
# Additional communication defaults from ms-swift v4.0.2-musa's node launcher.
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-14}"
export MCCL_CHECK_POINTERS="${MCCL_CHECK_POINTERS:-0}"
export MCCL_NET_SHARED_BUFFERS="${MCCL_NET_SHARED_BUFFERS:-0}"
export MUSA_BLOCK_SCHEDULE_MODE="${MUSA_BLOCK_SCHEDULE_MODE:-1}"
export MUSA_LAUNCH_BLOCKING="${MUSA_LAUNCH_BLOCKING:-0}"
export MUSA_DEVICE_MAX_CONNECTIONS="${MUSA_DEVICE_MAX_CONNECTIONS:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export MUSA_KERNEL_TIMEOUT="${MUSA_KERNEL_TIMEOUT:-3200000}"
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
export MCCL_DEBUG="${MCCL_DEBUG:-INFO}"
printf 'MUSA nodes=%s node_rank=%s local_gpus=%s master=%s:%s HCA=%s\n' \
    "$NNODES" "$NODE_RANK" "$NUM_GPUS" "$MASTER_ADDR" "$MLP_WORKER_0_PORT" "$MCCL_IB_HCA"
exec bash "$ROOT/recipes/posttrain/posttrain_agilex_fold_shirt_text_musa.sh" "$@"
