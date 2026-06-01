from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def norm_name(text: str) -> str:
    return str(text).lower().replace("_", " ").replace("-", " ").strip()


@dataclass(frozen=True)
class DatasetLabelSpace:
    key: str
    aliases: tuple[str, ...]
    raw_to_canonical: dict[int, str]
    ignore_names: tuple[str, ...]

    def matches(self, dataset_name: str) -> bool:
        name = norm_name(dataset_name)
        return name == norm_name(self.key) or name in {norm_name(alias) for alias in self.aliases}


@dataclass(frozen=True)
class VocabMap:
    name: str
    canonical_to_vocab: dict[str, str]

    def map_name(self, canonical_name: str) -> str | None:
        key = norm_name(canonical_name)
        return self.canonical_to_vocab.get(key, canonical_name)


class TaskLabelMapper:
    """Explicit raw-label-to-open-vocabulary mapper.

    This class owns only label translation. It never defines model output
    dimensionality; the active vocabulary still decides the final column ids.
    """

    def __init__(self, task_mapping_path: str | Path, project_root: str | Path | None = None):
        self.task_mapping_path = _resolve(project_root, task_mapping_path)
        with open(self.task_mapping_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        self.task_name = raw.get("task", self.task_mapping_path.stem)
        self.dataset_spaces = {
            key: _load_dataset_space(_resolve(project_root, path))
            for key, path in raw.get("datasets", {}).items()
        }
        self.vocab_maps = {
            key: _load_vocab_map(_resolve(project_root, path))
            for key, path in raw.get("vocab_maps", {}).items()
        }
        self.dataset_vocab_map = raw.get("dataset_vocab_map", {})
        self.source_dataset = raw.get("source_dataset", "")
        self.target_dataset = raw.get("target_dataset", "")
        self._cache: dict[tuple[str, tuple[str, ...], int], torch.Tensor] = {}
        if not self.dataset_spaces:
            raise ValueError(f"No datasets defined in task mapping: {self.task_mapping_path}")

    def map_raw_to_vocab(
        self,
        raw_labels: torch.Tensor,
        dataset_name: str,
        vocabulary,
        ignore_index: int = -100,
        vocab_role: str | None = None,
    ) -> torch.Tensor:
        if norm_name(dataset_name) == "syntheticdataset" or norm_name(dataset_name) == "synthetic":
            labels = raw_labels.long()
            mapped = torch.full_like(labels, fill_value=ignore_index)
            valid = (labels >= 0) & (labels < vocabulary.num_classes)
            mapped[valid] = labels[valid]
            return mapped
        dataset_key = self.resolve_dataset_key(dataset_name)
        mapping = self.raw_to_vocab_ids(dataset_key, vocabulary, ignore_index=ignore_index, vocab_role=vocab_role)
        mapping = mapping.to(device=raw_labels.device)
        labels = raw_labels.long()
        mapped = torch.full_like(labels, fill_value=ignore_index)
        valid = (labels >= 0) & (labels < mapping.numel())
        if bool(valid.any()):
            mapped[valid] = mapping[labels[valid]]
        return mapped

    def raw_to_vocab_ids(
        self,
        dataset_key: str,
        vocabulary,
        ignore_index: int = -100,
        vocab_role: str | None = None,
    ) -> torch.Tensor:
        dataset_key = self.resolve_dataset_key(dataset_key)
        cache_key = (dataset_key, tuple(vocabulary.names), int(ignore_index))
        if cache_key in self._cache:
            return self._cache[cache_key]
        space = self.dataset_spaces[dataset_key]
        vocab_lookup = _vocab_lookup(vocabulary)
        vocab_map = self._select_vocab_map(dataset_key, vocabulary, vocab_role=vocab_role)
        max_raw_id = max(space.raw_to_canonical.keys()) if space.raw_to_canonical else 0
        table = torch.full((max_raw_id + 1,), fill_value=int(ignore_index), dtype=torch.long)
        ignore_names = {norm_name(name) for name in space.ignore_names}
        for raw_id, canonical in space.raw_to_canonical.items():
            if norm_name(canonical) in ignore_names:
                continue
            vocab_name = vocab_map.map_name(canonical) if vocab_map is not None else canonical
            vocab_idx = vocab_lookup.get(norm_name(vocab_name))
            if vocab_idx is None:
                vocab_idx = vocab_lookup.get(norm_name(canonical))
            if vocab_idx is not None:
                table[int(raw_id)] = int(vocab_idx)
        self._cache[cache_key] = table
        return table

    def resolve_dataset_key(self, dataset_name: str) -> str:
        for key, space in self.dataset_spaces.items():
            if space.matches(dataset_name) or norm_name(key) == norm_name(dataset_name):
                return key
        available = ", ".join(sorted(self.dataset_spaces))
        raise KeyError(f"Unknown dataset '{dataset_name}' for task mapping {self.task_name}; available: {available}")

    def coverage_rows(self, dataset_name: str, vocabulary, vocab_role: str | None = None) -> list[dict[str, Any]]:
        dataset_key = self.resolve_dataset_key(dataset_name)
        space = self.dataset_spaces[dataset_key]
        vocab_lookup = _vocab_lookup(vocabulary)
        vocab_map = self._select_vocab_map(dataset_key, vocabulary, vocab_role=vocab_role)
        rows = []
        ignore_names = {norm_name(name) for name in space.ignore_names}
        for raw_id in sorted(space.raw_to_canonical):
            canonical = space.raw_to_canonical[raw_id]
            vocab_name = vocab_map.map_name(canonical) if vocab_map is not None else canonical
            vocab_id = vocab_lookup.get(norm_name(vocab_name), None)
            ignored = norm_name(canonical) in ignore_names or vocab_id is None
            rows.append(
                {
                    "raw_id": int(raw_id),
                    "canonical": canonical,
                    "vocab_name": vocab_name,
                    "vocab_id": "" if vocab_id is None else int(vocab_id),
                    "ignored": bool(ignored),
                }
            )
        return rows

    def unmapped_vocab_names(self, dataset_name: str, vocabulary, vocab_role: str | None = None) -> list[str]:
        rows = self.coverage_rows(dataset_name, vocabulary, vocab_role=vocab_role)
        mapped = {norm_name(row["vocab_name"]) for row in rows if not row["ignored"]}
        return [name for name in vocabulary.names if norm_name(name) not in mapped]

    def _select_vocab_map(self, dataset_key: str, vocabulary, vocab_role: str | None = None) -> VocabMap | None:
        role = vocab_role or _infer_vocab_role(vocabulary)
        role_map = self.dataset_vocab_map.get(dataset_key, {})
        map_key = role_map.get(role) or role_map.get("eval") or role_map.get("train")
        if map_key is None and role in self.vocab_maps:
            map_key = role
        return self.vocab_maps.get(map_key)


def build_label_mapper(cfg, project_root: str | Path | None = None) -> TaskLabelMapper | None:
    mapping_cfg = cfg.get("mapping", {})
    path = mapping_cfg.get("task_mapping_path", "")
    if not path:
        return None
    root = project_root or cfg.get("project_root", None)
    return TaskLabelMapper(path, project_root=root)


def _load_dataset_space(path: Path) -> DatasetLabelSpace:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    raw_labels = {int(key): str(value) for key, value in raw.get("raw_labels", {}).items()}
    return DatasetLabelSpace(
        key=str(raw["dataset"]),
        aliases=tuple(raw.get("aliases", [])),
        raw_to_canonical=raw_labels,
        ignore_names=tuple(raw.get("ignore_names", [])),
    )


def _load_vocab_map(path: Path) -> VocabMap:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    mapping = {norm_name(key): str(value) for key, value in raw.get("canonical_to_vocab", {}).items()}
    return VocabMap(name=str(raw.get("name", path.stem)), canonical_to_vocab=mapping)


def _vocab_lookup(vocabulary) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, item in enumerate(vocabulary.classes):
        for text in [item.name, *item.aliases]:
            lookup.setdefault(norm_name(text), idx)
    return lookup


def _infer_vocab_role(vocabulary) -> str:
    names = {norm_name(name) for name in vocabulary.names}
    if "semantic probe" in names:
        return "semantic_probe"
    if {"lane marking", "curb", "sky", "bridge", "tunnel"} & names:
        return "semantic_probe"
    if len(names) > 16:
        return "eval"
    return "train"


def _resolve(project_root: str | Path | None, path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if project_root is None:
        return path
    return Path(project_root).expanduser() / path
