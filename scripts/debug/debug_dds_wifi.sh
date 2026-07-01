#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate tv
fi

unset PYTHONPATH || true

NETWORK_INTERFACE="${NETWORK_INTERFACE:-wlo1}"

cd "${REPO_DIR}"

exec python tests/diagnostics/debug_dds_topics.py \
  --network-interface "${NETWORK_INTERFACE}" \
  "$@"
