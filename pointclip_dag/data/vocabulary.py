from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

try:
    import torch
except ModuleNotFoundError:  # Keep vocab inspection tools usable without the training stack.
    torch = None

@dataclass(frozen=True)
class VocabClass:
    name: str
    raw_labels: tuple[int, ...]
    mapping_label: str
    aliases: tuple[str, ...]
    seen: bool


class Vocabulary:
    def __init__(self, classes: list[VocabClass], prompt_templates: list[str] | None = None):
        self.classes = list(classes)
        self.prompt_templates = prompt_templates or ["a photo of a {}."]
        self.names = [item.name for item in self.classes]
        self.seen_mask = (
            torch.tensor([item.seen for item in self.classes], dtype=torch.bool)
            if torch is not None
            else [item.seen for item in self.classes]
        )
        self.raw_label_to_index = {}
        for idx, item in enumerate(self.classes):
            for raw_label in item.raw_labels:
                raw_label = int(raw_label)
                if raw_label in self.raw_label_to_index:
                    other = self.classes[self.raw_label_to_index[raw_label]].name
                    raise ValueError(f"raw_label {raw_label} is mapped to both {other} and {item.name}.")
                self.raw_label_to_index[raw_label] = idx
        self.label_to_index = dict(self.raw_label_to_index)

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
        return dict(self.raw_label_to_index)

    def map_labels(self, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        if torch is None:
            raise ImportError("torch is required to map tensor labels with Vocabulary.map_labels().")
        mapped = torch.full_like(labels, fill_value=ignore_index)
        for raw_id, vocab_idx in self.raw_label_to_index.items():
            mapped[labels == raw_id] = vocab_idx
        return mapped


def build_vocabulary(path: str | Path) -> Vocabulary:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    classes = [
        VocabClass(
            name=item["name"],
            raw_labels=_parse_raw_labels(item),
            mapping_label=str(item.get("mapping_label", item["name"])),
            aliases=tuple(item.get("aliases", [])),
            seen=bool(item.get("seen", True)),
        )
        for item in raw["classes"]
    ]
    return Vocabulary(classes, prompt_templates=raw.get("prompt_templates"))


def build_vocabulary_from_names(names: list[str], prompt_templates: list[str] | None = None) -> Vocabulary:
    classes = [
        VocabClass(
            name=name.strip(),
            raw_labels=tuple(),
            mapping_label=name.strip(),
            aliases=tuple(),
            seen=False,
        )
        for name in names
        if name.strip()
    ]
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
    return vocab.to_label_mapping()


def map_labels_to_vocab(raw_labels: torch.Tensor, dataset_name: str, vocab: Vocabulary, ignore_index: int = -100) -> torch.Tensor:
    return vocab.map_labels(raw_labels, ignore_index=ignore_index)


def get_seen_unseen_masks(vocab: Vocabulary) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ImportError("torch is required to build seen/unseen tensor masks.")
    seen = vocab.seen_mask.clone()
    return seen, ~seen


def _parse_raw_labels(item: dict) -> tuple[int, ...]:
    raw = item.get("raw_labels", item.get("raw_label", None))
    if raw is None:
        return tuple()
    if isinstance(raw, (list, tuple)):
        return tuple(int(value) for value in raw if value is not None)
    return (int(raw),)
