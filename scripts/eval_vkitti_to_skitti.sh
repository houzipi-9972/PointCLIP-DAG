#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_path> [extra eval.py args...]" >&2
  exit 2
fi

CKPT="$1"
shift

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/activate_pointclip.sh"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/eval.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --ckpt "${CKPT}" \
  "$@"
