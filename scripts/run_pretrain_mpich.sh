#!/usr/bin/env bash
# Launch training under MPICH mpirun from an interactive Slurm allocation.
#
# Host list (first match wins):
#   1) hosts_8ppn  — comma-separated hosts or path to a hostfile (must be exported)
#   2) HOSTS_8PPN  — same (some sites export uppercase)
#   3) SLURM_NODELIST — expanded with `scontrol show hostnames` when scontrol exists
#
# Total ranks:
#   SLURM_NTASKS — if unset, defaults to (#unique hosts × XFORMER_PPN)
#
# Optional: MASTER_ADDR, MASTER_PORT, XFORMER_PPN (default 8), MPIRUN
#
# If you see "Missing host list", the variable is unset in the script environment.
# Use: export hosts_8ppn=node1,node2,node3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PPN="${XFORMER_PPN:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
MPIRUN="${MPIRUN:-mpirun}"

_raw_hosts=""
if [[ -n "${hosts_8ppn:-}" ]]; then
  _raw_hosts="${hosts_8ppn}"
elif [[ -n "${HOSTS_8PPN:-}" ]]; then
  _raw_hosts="${HOSTS_8PPN}"
elif [[ -n "${SLURM_NODELIST:-}" ]] && command -v scontrol >/dev/null 2>&1; then
  _raw_hosts="$(scontrol show hostnames "${SLURM_NODELIST}" | tr '\n' ',' | sed 's/,$//')"
else
  echo "Missing host list for mpirun -hosts." >&2
  echo "  Export one of: hosts_8ppn, HOSTS_8PPN" >&2
  echo "  Or run inside Slurm with SLURM_NODELIST and scontrol on PATH." >&2
  echo "  Note: bash only passes exported variables to child processes; use 'export var=...'." >&2
  exit 1
fi

if [[ -f "${_raw_hosts}" ]]; then
  HOSTS_CSV="$(grep -v '^[[:space:]]*$' "${_raw_hosts}" | tr '\n' ',' | sed 's/,$//')"
  FIRST_HOST="$(grep -v '^[[:space:]]*$' "${_raw_hosts}" | head -1 | tr -d '[:space:]')"
else
  HOSTS_CSV="$(echo "${_raw_hosts}" | tr ' ' ',' | tr -s ',' | sed 's/^,//;s/,$//')"
  FIRST_HOST="$(echo "${HOSTS_CSV}" | cut -d, -f1 | tr -d '[:space:]')"
fi

: "${FIRST_HOST:?could not determine first host from host list}"
MASTER_ADDR="${MASTER_ADDR:-${FIRST_HOST}}"

if [[ -z "${SLURM_NTASKS:-}" ]]; then
  _nhosts="$(echo "${HOSTS_CSV}" | tr ',' '\n' | sed '/^$/d' | wc -l | tr -d '[:space:]')"
  SLURM_NTASKS=$((_nhosts * PPN))
  echo "SLURM_NTASKS unset; using ${_nhosts} hosts × PPN=${PPN} => SLURM_NTASKS=${SLURM_NTASKS}" >&2
fi

export MASTER_ADDR MASTER_PORT XFORMER_PPN="${PPN}"

cd "${REPO_ROOT}"

exec "${MPIRUN}" \
  -np "${SLURM_NTASKS}" \
  -hosts "${HOSTS_CSV}" \
  -ppn "${PPN}" \
  "${SCRIPT_DIR}/run_pretrain_mpi_worker.sh" \
  "$@"
