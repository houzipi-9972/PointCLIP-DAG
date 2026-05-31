from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def run():
    print(f"python: {sys.version.split()[0]}")
    torch = _import_optional("torch")
    if torch is None:
        print("torch: not available")
        print("cuda available: false")
    else:
        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda device: {torch.cuda.get_device_name(0)}")
    for name in [
        "yaml",
        "numpy",
        "torchvision",
        "timm",
        "open_clip",
        "clip",
        "nuscenes",
        "mmcv",
        "mmseg",
        "spconv",
        "MinkowskiEngine",
    ]:
        print(f"{name}: {_version(name)}")


def _version(module_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "available")
    except Exception as exc:
        return f"not available ({exc.__class__.__name__})"


def _import_optional(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


if __name__ == "__main__":
    run()
