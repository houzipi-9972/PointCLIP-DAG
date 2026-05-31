#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/zhangshuai/anaconda3}"
ENV_NAME="${ENV_NAME:-PointCLIP}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
set +u
conda activate "${ENV_NAME}"
set -u

# Keep CUDA libraries from the active PointCLIP environment ahead of stale
# libraries from UniDSeg or system CUDA. This fixes cublas/cublasLt symbol
# mismatches during `import torch`.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export POINTCLIP_SHOW_XFORMERS_WARNING="${POINTCLIP_SHOW_XFORMERS_WARNING:-0}"

cd "${PROJECT_ROOT}"
echo "PointCLIP environment is active."
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
