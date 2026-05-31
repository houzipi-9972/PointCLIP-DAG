from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from pointclip_dag.data.raw_label_spaces import build_raw_to_name

@dataclass(frozen=True)
class VocabClass:
    name: str
    train_id: Optional[int]
    aliases: tuple[str, ...]
    seen: bool


class Vocabulary:
    def __init__(self, classes: list[VocabClass], prompt_templates: list[str] | None = None):
        self.classes = list(classes)
        self.prompt_templates = prompt_templates or ["a photo of a {}."]
        self.names = [item.name for item in self.classes]
        self.train_ids = [item.train_id for item in self.classes]
        self.seen_mask = torch.tensor([item.seen for item in self.classes], dtype=torch.bool)
        train_ids_with_labels = [train_id for train_id in self.train_ids if train_id is not None]
        if len(set(train_ids_with_labels)) != len(train_ids_with_labels):
            raise ValueError("Vocabulary train_id values must be unique.")
        self.label_to_index = {
            int(train_id): idx for idx, train_id in enumerate(self.train_ids) if train_id is not None
        }

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_text_groups(self) -> list[list[str]]:
        groups = []
        for item in self.classes:
            names = [item.name, *item.aliases]
            groups.append([template.format(name) for name in names for template in self.prompt_templates])
        return groups

    def to_label_mapping(self) -> dict[int, int]:
        return dict(self.label_to_index)

    def map_labels(self, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        """Legacy direct train_id mapping. Prefer map_labels_to_vocab with dataset_name."""
        mapped = torch.full_like(labels, fill_value=ignore_index)
        for raw_id, vocab_idx in self.label_to_index.items():
            mapped[labels == raw_id] = vocab_idx
        return mapped


def build_vocabulary(path: str | Path) -> Vocabulary:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    classes = [
        VocabClass(
            name=item["name"],
            train_id=None if item.get("train_id", None) is None else int(item["train_id"]),
            aliases=tuple(item.get("aliases", [])),
            seen=bool(item.get("seen", True)),
        )
        for item in raw["classes"]
    ]
    return Vocabulary(classes, prompt_templates=raw.get("prompt_templates"))


def build_vocabulary_from_names(names: list[str], prompt_templates: list[str] | None = None) -> Vocabulary:
    classes = [VocabClass(name=name.strip(), train_id=None, aliases=tuple(), seen=False) for name in names if name.strip()]
    return Vocabulary(classes, prompt_templates=prompt_templates)


def load_vocab(yaml_path: str | Path) -> Vocabulary:
    return build_vocabulary(yaml_path)


def build_text_prompts(vocab: Vocabulary, templates: list[str] | None = None) -> list[list[str]]:
    if templates is None:
        return vocab.class_text_groups()
    groups = []
    for item in vocab.classes:
        names = [item.name, *item.aliases]
        groups.append([template.format(name) for name in names for template in templates])
    return groups


def build_raw_to_vocab_mapping(dataset_name: str, vocab: Vocabulary, ignore_index: int = -100) -> dict[int, int]:
    """Map raw dataset label IDs to current vocabulary column IDs.

    Matching uses explicit `train_id` first, then normalized class names and aliases.
    This keeps dataset label space separate from open vocabulary column space.
    """
    raw_to_name = build_raw_to_name(dataset_name)
    text_to_vocab = {}
    for vocab_idx, item in enumerate(vocab.classes):
        for text in [item.name, *item.aliases]:
            text_to_vocab[_norm(text)] = vocab_idx

    mapping = {}
    for raw_id, raw_name in raw_to_name.items():
        raw_key = _norm(raw_name)
        mapping[raw_id] = text_to_vocab.get(raw_key, ignore_index)

    for vocab_idx, item in enumerate(vocab.classes):
        if item.train_id is not None:
            mapping[int(item.train_id)] = vocab_idx
    return mapping


def map_labels_to_vocab(raw_labels: torch.Tensor, dataset_name: str, vocab: Vocabulary, ignore_index: int = -100) -> torch.Tensor:
    mapping = build_raw_to_vocab_mapping(dataset_name, vocab, ignore_index=ignore_index)
    mapped = torch.full_like(raw_labels, fill_value=ignore_index)
    for raw_id, vocab_idx in mapping.items():
        if vocab_idx != ignore_index:
            mapped[raw_labels == int(raw_id)] = int(vocab_idx)
    return mapped


def get_seen_unseen_masks(vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    seen = vocab.seen_mask.clone()
    return seen, ~seen


def _norm(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ").strip()
