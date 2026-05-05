#!/usr/bin/env bash
set -euo pipefail

# Multi-node example (two nodes, 8 GPUs each): on each node, set NODE_RANK and run:
#   NNODES=2 NODE_RANK=0 NPROC_PER_NODE=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 ./scripts/run_pretrain.sh
#   NNODES=2 NODE_RANK=1 NPROC_PER_NODE=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 ./scripts/run_pretrain.sh
#
# MPICH: use ./scripts/run_pretrain_mpich.sh (sets XFORMER_USE_MPI_LAUNCHER=1 per rank).

if [[ "${XFORMER_USE_MPI_LAUNCHER:-0}" == 1 ]]; then
  exec python -m xformer_pretrain.train "$@"
fi

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  -m xformer_pretrain.train "$@"
