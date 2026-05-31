from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def freeze_module(module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def trainable_parameters(modules: Iterable[torch.nn.Module]):
    for module in modules:
        yield from (p for p in module.parameters() if p.requires_grad)
