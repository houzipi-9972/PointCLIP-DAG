from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dict config with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    cfg_raw = load_yaml(config_path)
    default_name = cfg_raw.pop("_base_", "../default.yaml")
    default_path = (config_path.parent / default_name).resolve()
    if default_path.exists():
        cfg_raw = _merge(load_yaml(default_path), cfg_raw)
    cfg_raw["config_path"] = str(config_path)
    cfg_raw["project_root"] = str(_find_project_root(config_path))
    _resolve_project_paths(cfg_raw, Path(cfg_raw["project_root"]))
    return _to_config(cfg_raw)


def setup_external_paths(cfg: Config) -> None:
    """Register external source roots in one place."""
    for path in cfg.get("external", {}).get("python_paths", []):
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def save_config(cfg: Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(_plain_dict(cfg), handle, sort_keys=False)


def _plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_dict(v) for v in value]
    return value


def _find_project_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pointclip_dag").is_dir() and (parent / "configs").is_dir():
            return parent
    return config_path.parents[2] if len(config_path.parents) > 2 else config_path.parent


def _resolve_project_paths(cfg: dict, project_root: Path) -> None:
    path_keys = {
        "download_root",
        "image_encoder_weight_path",
        "dino_weight_path",
        "depth_weight_path",
        "repo_path",
        "dino_repo_path",
        "depth_repo_path",
    }

    def visit(value):
        if isinstance(value, dict):
            for key, item in list(value.items()):
                if key in path_keys and isinstance(item, str) and item:
                    path = Path(item).expanduser()
                    if not path.is_absolute():
                        value[key] = str(project_root / path)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(cfg)
