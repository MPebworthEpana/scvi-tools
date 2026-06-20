#!/usr/bin/env bash
# Run MultiVI idle-gap benchmark in Ubuntu WSL with the RAPIDS conda env.
#
# Usage (from Windows PowerShell or WSL):
#   wsl bash scripts/run_benchmark_wsl.sh
#   wsl bash scripts/run_benchmark_wsl.sh --limit-train-batches 200
#
# Environment variables:
#   CONDA_ENV   conda env name (default: scatlas_rapids2410)
#   CONDA_ROOT  miniforge/miniconda root (default: ~/miniforge3)

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-scatlas_rapids2410}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "conda.sh not found at ${CONDA_ROOT}; set CONDA_ROOT to your miniforge/miniconda install." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "Conda env: ${CONDA_ENV} ($(which python))"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import scvi; print('scvi', scvi.__version__)" 2>/dev/null || true
echo "Repo: ${REPO_ROOT}"
echo

cd "${REPO_ROOT}"
exec python scripts/benchmark_multivi_idle_gap.py \
  --output-json scripts/benchmark_multivi_idle_gap_wsl_results.json \
  "$@"
