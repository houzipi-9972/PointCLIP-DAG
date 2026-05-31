#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-PointCLIP}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
CUDA_TOOLKIT="${CUDA_TOOLKIT:-11.1}"
TORCH_VERSION="${TORCH_VERSION:-1.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.10.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-0.9.1}"

echo "[1/7] Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y

echo "[2/7] Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[3/7] Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_TOOLKIT}"
conda install -y -c pytorch -c conda-forge \
  "pytorch=${TORCH_VERSION}" \
  "torchvision=${TORCHVISION_VERSION}" \
  "torchaudio=${TORCHAUDIO_VERSION}" \
  "cudatoolkit=${CUDA_TOOLKIT}"

echo "[4/7] Installing base Python packages"
python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  numpy==1.24.3 \
  pyyaml \
  pillow \
  scipy \
  matplotlib \
  opencv-python-headless \
  tqdm \
  ftfy \
  regex \
  prettytable \
  tensorboard \
  yacs \
  timm==0.4.12 \
  nuscenes-devkit \
  yapf==0.40.1

echo "[5/7] Installing OpenMMLab dependencies"
python -m pip install -U openmim
mim install "mmengine"
mim install "mmcv==2.1.0"
python -m pip install "mmsegmentation==1.2.2"

echo "[6/7] Installing sparse convolution and CLIP packages"
python -m pip install "spconv-cu111==2.1.25"
python -m pip install "git+https://github.com/openai/CLIP.git"

echo "[7/7] Verifying key imports"
python - <<'PY'
import importlib

mods = [
    "torch",
    "torchvision",
    "spconv",
    "mmcv",
    "mmseg",
    "clip",
    "nuscenes",
    "yaml",
    "timm",
]

for name in mods:
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'available')}")

import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device:", torch.cuda.get_device_name(0))
PY

echo
echo "Environment ${ENV_NAME} is ready."
echo "Installing activation hooks to avoid CUDA library path conflicts."
"$(dirname "$0")/install_pointclip_conda_hooks.sh"
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Then prepare local project weights:"
echo "  cd /home/zhangshuai/Sysu_4T/houych/PointCLIP-DAG"
echo "  python pointclip_dag/scripts/prepare_weights.py --clip-model ViT-L/14 --download-root weights/clip"
