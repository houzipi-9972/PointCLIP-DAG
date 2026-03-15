from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_clip_ckpt(backbone_2d: str) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    if backbone_2d == "ViT-B-16":
        return str(repo_root / "pretrained" / "ViT-B-16.pt")
    # Use ViT-L/14 text encoder for both ViT-L-14 and SAM_ViT-L visual backbones.
    return str(repo_root / "pretrained" / "ViT-L-14.pt")


class OpenVocabClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        class_names: Sequence[str],
        backbone_2d: str = "ViT-L-14",
        prompt_templates: Sequence[str] = ("a photo of a {}",),
        logit_scale: float = 30.0,
        clip_ckpt_path: str = "",
    ):
        super().__init__()
        if not class_names:
            raise ValueError("`class_names` must not be empty in open-vocabulary mode.")
        if not prompt_templates:
            raise ValueError("`prompt_templates` must not be empty in open-vocabulary mode.")

        self.class_names = tuple(class_names)
        self.prompt_templates = tuple(prompt_templates)
        self.logit_scale = float(logit_scale)

        text_features = self._build_text_features(
            class_names=self.class_names,
            prompt_templates=self.prompt_templates,
            clip_ckpt_path=clip_ckpt_path or _default_clip_ckpt(backbone_2d),
        )
        text_dim = text_features.shape[1]
        self.proj = nn.Linear(in_channels, text_dim)
        self.register_buffer("text_features", text_features, persistent=True)

    @staticmethod
    def _build_text_features(
        class_names: Sequence[str],
        prompt_templates: Sequence[str],
        clip_ckpt_path: str,
    ) -> torch.Tensor:
        try:
            import clip
        except Exception as exc:
            raise ImportError(
                "Open-vocabulary mode requires the `clip` package. "
                "Install OpenAI CLIP and retry."
            ) from exc

        device = "cpu"
        text_model, _ = clip.load(clip_ckpt_path, device=device, jit=False)
        text_model.eval()

        with torch.no_grad():
            template_features = []
            for template in prompt_templates:
                texts = [template.format(name) for name in class_names]
                tokens = clip.tokenize(texts, truncate=True).to(device)
                feats = text_model.encode_text(tokens).float()
                feats = F.normalize(feats, dim=1)
                template_features.append(feats)
            text_features = torch.stack(template_features, dim=0).mean(dim=0)
            text_features = F.normalize(text_features, dim=1)
        return text_features

    def encode_features(self, feats: torch.Tensor) -> torch.Tensor:
        proj_feats = self.proj(feats)
        return F.normalize(proj_feats, dim=1)

    def compute_logits_from_encoded(self, encoded_feats: torch.Tensor) -> torch.Tensor:
        return self.logit_scale * encoded_feats @ self.text_features.t()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        encoded_feats = self.encode_features(feats)
        return self.compute_logits_from_encoded(encoded_feats)
