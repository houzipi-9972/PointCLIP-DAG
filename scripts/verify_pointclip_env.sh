#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_ROOT}/scripts/activate_pointclip.sh" >/dev/null

python - <<'PY'
import importlib
import os

print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"))
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))

mods = ["torch", "torchvision", "spconv", "clip", "timm", "yaml", "PIL", "tqdm"]
for name in mods:
    mod = importlib.import_module(name)
    print(f"{name}: {getattr(mod, '__version__', 'available')}")

try:
    import xformers  # noqa: F401
    print("xformers: available")
except Exception:
    print("xformers: not installed; DepthAnything/DINOv2 will use standard attention fallback")

try:
    import nuscenes  # noqa: F401
    print("nuscenes-devkit: available")
except Exception as exc:
    print(f"nuscenes-devkit: missing ({type(exc).__name__}); required only for nuScenes experiments")

try:
    import mmcv  # noqa: F401
    print("mmcv:", getattr(mmcv, "__version__", "available"))
except Exception as exc:
    print(f"mmcv: missing ({type(exc).__name__}); not required for current raw vKITTI->SemanticKITTI adapter")

try:
    import mmseg  # noqa: F401
    print("mmseg:", getattr(mmseg, "__version__", "available"))
except Exception as exc:
    print(f"mmseg: missing ({type(exc).__name__}); not required for current raw vKITTI->SemanticKITTI adapter")

import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
PY

python pointclip_dag/scripts/check_train_ready.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --check-model
