#!/usr/bin/env bash
# One MPI rank → one training process. Maps MPICH PMI env to PyTorch distributed vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${PMI_RANK:?MPICH did not set PMI_RANK (use mpirun/mpiexec from MPICH)}"
: "${PMI_SIZE:?MPICH did not set PMI_SIZE}"

PPN="${XFORMER_PPN:-8}"
export RANK="${PMI_RANK}"
export WORLD_SIZE="${PMI_SIZE}"

if [[ -n "${PMI_LOCAL_RANK:-}" ]]; then
  export LOCAL_RANK="${PMI_LOCAL_RANK}"
elif [[ -n "${HYDRA_LOCAL_RANK:-}" ]]; then
  export LOCAL_RANK="${HYDRA_LOCAL_RANK}"
else
  export LOCAL_RANK=$((PMI_RANK % PPN))
fi

export LOCAL_WORLD_SIZE="${PPN}"
export XFORMER_USE_MPI_LAUNCHER=1

exec "${SCRIPT_DIR}/run_pretrain.sh" "$@"
