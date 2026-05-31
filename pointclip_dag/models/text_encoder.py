from __future__ import annotations

import hashlib
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """CLIP text encoder with a deterministic fallback for environment checks."""

    def __init__(self, cfg):
        super().__init__()
        self.backend = cfg.get("backend", "open_clip")
        self.model_name = cfg.get("model_name", "ViT-B-32")
        self.pretrained = cfg.get("pretrained", "openai")
        self.download_root = cfg.get("download_root", None)
        self.embed_dim = int(cfg.get("embed_dim", 512))
        self.cache_enabled = bool(cfg.get("cache", True))
        self.cache = {}
        self.model = None
        self.tokenizer = None
        self._init_backend()
        if self.backend != "none" and self.model is None and not cfg.get("allow_hash_fallback", False):
            raise RuntimeError(
                f"Failed to initialize text encoder backend={self.backend}, model_name={self.model_name}. "
                "Install/load CLIP correctly or set backend: none only for smoke tests."
            )
        if cfg.get("freeze", True):
            self.freeze()

    def _init_backend(self) -> None:
        if self.backend == "none":
            return
        if self.backend == "open_clip":
            try:
                import open_clip

                self.model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
                self.tokenizer = open_clip.get_tokenizer(self.model_name)
                self.embed_dim = int(self.model.text_projection.shape[1]) if hasattr(self.model, "text_projection") else self.embed_dim
                return
            except Exception:
                self.model = None
        if self.backend in {"clip", "openai_clip", "open_clip"}:
            try:
                import clip

                self.model, _ = clip.load(
                    _openai_clip_name(self.model_name),
                    device="cpu",
                    jit=False,
                    download_root=self.download_root,
                )
                self.tokenizer = clip.tokenize
                self.embed_dim = int(self.model.text_projection.shape[1])
            except Exception:
                self.model = None

    def freeze(self) -> None:
        if self.model is not None:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def encode_groups(self, text_groups: Sequence[Sequence[str]], device=None) -> torch.Tensor:
        key = tuple(tuple(group) for group in text_groups)
        device = device or next(self.parameters(), torch.empty(0)).device
        if self.cache_enabled and key in self.cache:
            return self.cache[key].to(device)
        embeddings = []
        for texts in text_groups:
            text_features = self._encode_texts(list(texts), device)
            text_features = F.normalize(text_features, dim=-1)
            embeddings.append(F.normalize(text_features.mean(dim=0), dim=-1))
        out = torch.stack(embeddings, dim=0)
        if self.cache_enabled:
            self.cache[key] = out.detach().cpu()
        return out

    def _encode_texts(self, texts: list[str], device) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            return self._hash_embeddings(texts, device)
        self.model.to(device)
        tokens = self.tokenizer(texts).to(device)
        features = self.model.encode_text(tokens)
        return features.float()

    def _hash_embeddings(self, texts: list[str], device) -> torch.Tensor:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = torch.tensor(list(digest), dtype=torch.float32, device=device)
            repeat = (self.embed_dim + values.numel() - 1) // values.numel()
            vector = values.repeat(repeat)[: self.embed_dim]
            vector = (vector - vector.mean()) / (vector.std() + 1e-6)
            vectors.append(vector)
        return torch.stack(vectors, dim=0)


def _openai_clip_name(model_name: str) -> str:
    aliases = {
        "ViT-B-32": "ViT-B/32",
        "ViT-B-16": "ViT-B/16",
        "ViT-L-14": "ViT-L/14",
        "ViT-L-14-336px": "ViT-L/14@336px",
    }
    return aliases.get(model_name, model_name)
