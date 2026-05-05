#!/usr/bin/env bash
# Launch training under MPICH mpirun from an interactive Slurm allocation.
# Expects:
#   SLURM_NTASKS  — total MPI ranks (should match nodes × PPN)
#   hosts_8ppn   — comma-separated hostnames, or a file with one host per line
# Optional:
#   MASTER_ADDR  — rendezvous host (default: first host in hosts_8ppn)
#   MASTER_PORT  — default 29500
#   XFORMER_PPN  — processes (GPUs) per node; default 8
#   MPIRUN       — default mpirun
#
# Example:
#   export hosts_8ppn=node1,node2,node3
#   export SLURM_NTASKS=24
#   ./scripts/run_pretrain_mpich.sh --backend torch --steps 100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${SLURM_NTASKS:?export SLURM_NTASKS (total ranks)}"
: "${hosts_8ppn:?export hosts_8ppn (comma-separated hosts or path to a hostfile)}"

PPN="${XFORMER_PPN:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MPIRUN="${MPIRUN:-mpirun}"

if [[ -f "${hosts_8ppn}" ]]; then
  HOSTS_CSV="$(grep -v '^[[:space:]]*$' "${hosts_8ppn}" | tr '\n' ',' | sed 's/,$//')"
  FIRST_HOST="$(grep -v '^[[:space:]]*$' "${hosts_8ppn}" | head -1 | tr -d '[:space:]')"
else
  # Comma- or space-separated
  HOSTS_CSV="$(echo "${hosts_8ppn}" | tr ' ' ',' | tr -s ',' | sed 's/^,//;s/,$//')"
  FIRST_HOST="$(echo "${HOSTS_CSV}" | cut -d, -f1 | tr -d '[:space:]')"
fi

: "${FIRST_HOST:?could not determine first host from hosts_8ppn}"
MASTER_ADDR="${MASTER_ADDR:-${FIRST_HOST}}"

export MASTER_ADDR MASTER_PORT XFORMER_PPN="${PPN}"

cd "${REPO_ROOT}"

exec "${MPIRUN}" \
  -np "${SLURM_NTASKS}" \
  -hosts "${HOSTS_CSV}" \
  -ppn "${PPN}" \
  "${SCRIPT_DIR}/run_pretrain_mpi_worker.sh" \
  "$@"
