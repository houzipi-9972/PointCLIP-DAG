from __future__ import annotations

import sys
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Vireo2DOVBranch(nn.Module):
    """Vireo-style 2D open-vocabulary branch for projected point supervision.

    The implementation keeps the Vireo mechanisms that are useful for
    PointCLIP-DAG without importing the MMSeg Mask2Former stack:
    frozen VFM image features, DepthAnything/sparse-depth geometry, text-aware
    token refinement, coarse mask prior, and final text-space pixel embeddings.
    """

    def __init__(self, cfg, text_dim: int):
        super().__init__()
        width = int(cfg.get("width", 96))
        self.embed_dim = text_dim
        self.width = width
        self.enable_depth_vfm = bool(cfg.get("enable_depth", cfg.get("enable_depth_vfm", False)))
        self.depth_mode = "sparse_depth_adapter"
        self.allow_depth_fallback = bool(cfg.get("allow_depth_fallback", True))
        self.enable_text_feature_refinement = bool(cfg.get("enable_text_feature_refinement", True))
        self.enable_geotext_query = bool(cfg.get("enable_geotext_query", True))
        self.enable_cmpe = bool(cfg.get("enable_cmpe", True))
        self.enable_dov_head = bool(cfg.get("enable_dov_head", True))
        self.image_encoder_type = cfg.get("image_encoder", "none")
        self.image_encoder = None
        self.depth_vfm = None

        if self.image_encoder_type.startswith("dinov2_"):
            image_cfg = {
                "image_encoder": self.image_encoder_type,
                "pretrained": cfg.get("pretrained", True),
                "weight_path": cfg.get("image_encoder_weight_path", cfg.get("dino_weight_path", "")),
                "repo_path": cfg.get("dino_repo_path", cfg.get("depth_vfm", {}).get("repo_path", "")),
                "freeze": cfg.get("freeze_image_encoder", True),
                "input_size": cfg.get("image_encoder_input_size", 518),
            }
            self.image_encoder = DINOv2ImageEncoderAdapter(image_cfg)
            image_dim = self.image_encoder.feature_dim
        elif self.image_encoder_type.startswith("clip_"):
            image_cfg = {
                "image_encoder": self.image_encoder_type,
                "pretrained": cfg.get("pretrained", True),
                "download_root": cfg.get("clip_download_root", ""),
                "freeze": cfg.get("freeze_image_encoder", True),
                "input_size": cfg.get("image_encoder_input_size", 224),
            }
            self.image_encoder = CLIPImageEncoderAdapter(image_cfg)
            image_dim = self.image_encoder.feature_dim
        elif self.image_encoder_type in {"none", "rgb_stem"}:
            image_dim = width
            if bool(cfg.get("pretrained", False)):
                raise ValueError("branch2d.pretrained=true requires a real image_encoder, not rgb_stem/none.")
        else:
            raise ValueError(f"Unsupported branch2d.image_encoder: {self.image_encoder_type}")

        depth_vfm_cfg = dict(cfg.get("depth_vfm", {}))
        depth_backend = cfg.get("depth_mode", depth_vfm_cfg.get("backend", "none"))
        if cfg.get("depth_weight_path", ""):
            depth_vfm_cfg["depth_weight_path"] = cfg.get("depth_weight_path")
        if cfg.get("depth_repo_path", ""):
            depth_vfm_cfg["repo_path"] = cfg.get("depth_repo_path")
        if cfg.get("depth_encoder", ""):
            depth_vfm_cfg["encoder"] = cfg.get("depth_encoder")
        if self.enable_depth_vfm and depth_backend == "depth_anything_v2":
            try:
                self.depth_vfm = DepthAnythingV2Adapter(depth_vfm_cfg)
                self.depth_mode = f"depth_anything_v2/{self.depth_vfm.encoder}"
            except Exception:
                if not self.allow_depth_fallback:
                    raise
                print(
                    "[model:2d] WARNING: DepthAnythingV2 failed to initialize; "
                    "falling back to sparse_depth_adapter because allow_depth_fallback=true.",
                    flush=True,
                )
        else:
            if self.enable_depth_vfm:
                print(
                    "[model:2d] enable_depth_vfm=true, but no external DepthAnything/VFM checkpoint is configured; "
                    "using sparse_depth_adapter fallback.",
                    flush=True,
                )

        self.rgb_stem = nn.Sequential(
            ConvBlock(3, width // 2),
            ConvBlock(width // 2, width, stride=2),
            ConvBlock(width, width, stride=2),
        )
        self.vfm_proj = nn.Sequential(
            nn.Conv2d(image_dim, width, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.depth_adapter = nn.Sequential(
            ConvBlock(1, width // 2),
            ConvBlock(width // 2, width, stride=2),
            ConvBlock(width, width, stride=2),
        )
        self.geotext = GeoTextTokenRefiner(
            width=width,
            text_dim=text_dim,
            num_tokens=int(cfg.get("num_geotext_tokens", 64)),
        )
        self.cmpe = CoarseMaskPriorEmbedding(width=width, text_dim=text_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(width * 3 + text_dim, width * 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        self.text_film = nn.Linear(text_dim, width * 2)
        self.dov_head = TextConditionedEmbeddingHead(width=width, text_dim=text_dim)
        self.out_proj = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, text_dim, 1),
        )
        self.structure_name = (
            "frozen RGB image VFM encoder + RGB local stem + dense/sparse depth adapter "
            "+ GeoTextTokenRefiner + CoarseMaskPriorEmbedding + text-conditioned embedding head"
        )

    def forward(self, image: torch.Tensor, sparse_depth: torch.Tensor, text_embeddings: torch.Tensor):
        input_size = image.shape[-2:]
        rgb_local = self.rgb_stem(image)

        dense_depth = None
        if self.image_encoder is not None:
            image_feat = self.vfm_proj(self.image_encoder(image))
        else:
            image_feat = self.vfm_proj(rgb_local)

        if self.depth_vfm is not None:
            dense_depth = self.depth_vfm(image)

        if image_feat.shape[-2:] != rgb_local.shape[-2:]:
            image_feat = F.interpolate(image_feat, size=rgb_local.shape[-2:], mode="bilinear", align_corners=False)

        depth_input = sparse_depth
        if dense_depth is not None:
            sparse = _normalize_depth(sparse_depth)
            sparse_mask = (sparse_depth > 0).float()
            if self.depth_vfm.blend_sparse_depth:
                depth_input = dense_depth * (1.0 - sparse_mask) + sparse * sparse_mask
            else:
                depth_input = dense_depth

        depth_feat = self.depth_adapter(depth_input)
        if depth_feat.shape[-2:] != image_feat.shape[-2:]:
            depth_feat = F.interpolate(depth_feat, size=image_feat.shape[-2:], mode="bilinear", align_corners=False)

        image_feat = image_feat + rgb_local
        refined = self.geotext(image_feat, depth_feat, text_embeddings) if self.enable_geotext_query else image_feat
        if self.enable_cmpe:
            coarse_logits, text_prior_map, query_context_map = self.cmpe(refined, text_embeddings)
        else:
            coarse_logits = torch.einsum(
                "bdhw,kd->bkhw",
                F.normalize(self.out_proj(refined), dim=1),
                F.normalize(text_embeddings, dim=-1),
            )
            text_prior_map = refined.new_zeros(refined.shape[0], self.embed_dim, refined.shape[2], refined.shape[3])
            query_context_map = refined.new_zeros(refined.shape)

        fused = torch.cat([refined, depth_feat, query_context_map, text_prior_map], dim=1)
        fused = self.fusion(fused)
        if self.enable_text_feature_refinement:
            text_global = F.normalize(text_embeddings, dim=-1).mean(dim=0)
            gamma, beta = self.text_film(text_global).chunk(2, dim=-1)
            fused = fused * (1.0 + torch.tanh(gamma).view(1, -1, 1, 1)) + beta.view(1, -1, 1, 1)
        if self.enable_dov_head:
            z2d, dov_attention = self.dov_head(fused, text_embeddings)
        else:
            z2d = self.out_proj(fused)
            dov_attention = None
        z2d = F.interpolate(z2d, size=input_size, mode="bilinear", align_corners=False)
        coarse_logits = F.interpolate(coarse_logits, size=input_size, mode="bilinear", align_corners=False)
        out = {
            "z2d_map": F.normalize(z2d, dim=1),
            "coarse_logits2d_map": coarse_logits,
            "depth_mode": self.depth_mode,
        }
        if dov_attention is not None:
            out["dov_attention_maps"] = F.interpolate(
                dov_attention, size=input_size, mode="bilinear", align_corners=False
            )
        return out


class GeoTextTokenRefiner(nn.Module):
    """Lightweight version of Vireo's token-based image/depth/text refinement."""

    def __init__(self, width: int, text_dim: int, num_tokens: int = 64):
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(num_tokens, width))
        self.text_proj = nn.Linear(text_dim, width)
        self.token_to_feat = nn.Linear(width, width)
        self.delta = nn.Linear(width, width)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Parameter(torch.tensor(0.001))
        self.norm = nn.LayerNorm(width)
        nn.init.xavier_uniform_(self.tokens)

    def forward(self, image_feat, depth_feat, text_embeddings):
        bsz, channels, height, width = image_feat.shape
        img = image_feat.flatten(2).transpose(1, 2)
        dep = depth_feat.flatten(2).transpose(1, 2)
        text = F.normalize(self.text_proj(text_embeddings), dim=-1)
        tokens = self.tokens
        token_text_attn = torch.softmax(tokens @ text.t(), dim=-1)
        text_context = token_text_attn @ text
        tokens = F.normalize(tokens + text_context, dim=-1)

        alpha = torch.sigmoid(self.fusion_weight)
        attn = (img @ tokens.t()) * alpha + (dep @ tokens.t()) * (1.0 - alpha)
        attn = torch.softmax(attn * (channels ** -0.5), dim=-1)
        delta = attn @ self.token_to_feat(tokens)
        delta = self.delta(self.norm(delta + img))
        out = img + self.scale * delta
        return out.transpose(1, 2).reshape(bsz, channels, height, width)


class CoarseMaskPriorEmbedding(nn.Module):
    """CMPE-style dynamic text prior without a fixed class-count layer."""

    def __init__(self, width: int, text_dim: int):
        super().__init__()
        self.feat_to_text = nn.Conv2d(width, text_dim, 1, bias=False)
        self.query_proj = nn.Linear(width, width)
        self.query_to_map = nn.Conv2d(width, width, 1, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, text_embeddings):
        text = F.normalize(text_embeddings, dim=-1)
        feat_text = F.normalize(self.feat_to_text(feat), dim=1)
        coarse = torch.einsum("bdhw,kd->bkhw", feat_text, text) * self.temperature.exp().clamp(max=100.0)

        class_spatial = torch.softmax(coarse.flatten(2), dim=-1)
        feat_flat = feat.flatten(2)
        class_feats = torch.einsum("bkn,bcn->bkc", class_spatial, feat_flat)
        class_feats = self.query_proj(class_feats)
        class_weights = torch.softmax(coarse, dim=1)
        query_context = torch.einsum("bkhw,bkc->bchw", class_weights, class_feats)
        query_context = self.query_to_map(query_context)

        text_prior_map = torch.einsum("bkhw,kd->bdhw", class_weights, text)
        return coarse, text_prior_map, query_context


class TextConditionedEmbeddingHead(nn.Module):
    """DOV-style dynamic text-conditioned pixel embedding head."""

    def __init__(self, width: int, text_dim: int):
        super().__init__()
        self.visual_proj = nn.Conv2d(width, text_dim, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(text_dim * 2, text_dim, 1, bias=False),
            nn.BatchNorm2d(text_dim),
            nn.GELU(),
            nn.Conv2d(text_dim, text_dim, 1),
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, text_embeddings):
        text = F.normalize(text_embeddings, dim=-1)
        visual = self.visual_proj(feat)
        visual_norm = F.normalize(visual, dim=1)
        attention = torch.einsum("bdhw,kd->bkhw", visual_norm, text)
        attention = torch.softmax(attention * self.temperature.exp().clamp(max=100.0), dim=1)
        semantic = torch.einsum("bkhw,kd->bdhw", attention, text)
        return self.fuse(torch.cat([visual, semantic], dim=1)), attention


class DINOv2ImageEncoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_name = cfg.get("image_encoder", "dinov2_vitl14")
        self.encoder = _parse_dinov2_encoder(self.encoder_name)
        self.input_size = int(cfg.get("input_size", 518))
        self.freeze = bool(cfg.get("freeze", True))
        self.pretrained = bool(cfg.get("pretrained", True))
        self.weight_path = str(Path(cfg.get("weight_path", "")).expanduser())
        self.pretrained_loaded = False
        self.missing_keys = []
        self.unexpected_keys = []

        repo_path = Path(cfg.get("repo_path", "")).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"DINOv2 repo_path does not exist: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        _configure_xformers_logging()
        try:
            from depth_anything_v2.dinov2 import DINOv2
        except Exception as exc:
            raise ImportError(f"Could not import DINOv2 from {repo_path}.") from exc

        self.feature_dim = _dinov2_feature_dim(self.encoder)
        self.model = DINOv2(self.encoder)
        if self.pretrained:
            if not self.weight_path:
                raise ValueError(
                    "branch2d.pretrained=true requires branch2d.image_encoder_weight_path "
                    "or branch2d.dino_weight_path."
                )
            path = Path(self.weight_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"DINOv2 image encoder weight is missing: {path}. "
                    "Depth Anything checkpoint is not a replacement for this file."
                )
            state = _clean_state_dict(torch.load(str(path), map_location="cpu"))
            info = self.model.load_state_dict(state, strict=False)
            self.missing_keys = list(info.missing_keys)
            self.unexpected_keys = list(info.unexpected_keys)
            self.pretrained_loaded = True
        if self.freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        print(
            f"[model:2d] image_encoder={self.encoder_name} pretrained_loaded={self.pretrained_loaded} "
            f"weight_path={self.weight_path}",
            flush=True,
        )
        print(
            f"[model:2d] image_encoder missing_keys={len(self.missing_keys)} "
            f"unexpected_keys={len(self.unexpected_keys)}",
            flush=True,
        )

    def forward(self, image):
        size = _multiple_of_14(self.input_size)
        x = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        x = _imagenet_normalize(x.clamp(0.0, 1.0))
        context = torch.no_grad() if self.freeze else _null_context()
        with context:
            features = self.model.get_intermediate_layers(
                x,
                n=1,
                reshape=True,
                return_class_token=False,
                norm=True,
            )[0]
        return features


class CLIPImageEncoderAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder_name = cfg.get("image_encoder", "clip_vitl14")
        self.model_name = _parse_clip_image_encoder(self.encoder_name)
        self.input_size = int(cfg.get("input_size", 224))
        self.freeze = bool(cfg.get("freeze", True))
        self.pretrained = bool(cfg.get("pretrained", True))
        self.download_root = str(Path(cfg.get("download_root", "")).expanduser()) if cfg.get("download_root", "") else None
        self.weight_path = _clip_cache_path(self.model_name, self.download_root)
        self.pretrained_loaded = False
        self.missing_keys = []
        self.unexpected_keys = []
        if not self.pretrained:
            raise ValueError("CLIP image encoder does not support random initialization in PointCLIP-DAG.")
        try:
            import clip
        except Exception as exc:
            raise ImportError("OpenAI CLIP is required for branch2d.image_encoder=clip_*") from exc
        model, _ = clip.load(self.model_name, device="cpu", jit=False, download_root=self.download_root)
        self.visual = model.visual.float()
        self.feature_dim = int(model.visual.proj.shape[1]) if getattr(model.visual, "proj", None) is not None else int(model.visual.ln_post.normalized_shape[0])
        self.pretrained_loaded = True
        if self.freeze:
            self.visual.eval()
            for param in self.visual.parameters():
                param.requires_grad = False
        print(
            f"[model:2d] image_encoder={self.encoder_name} pretrained_loaded=true weight_path={self.weight_path}",
            flush=True,
        )

    def forward(self, image):
        x = F.interpolate(image, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = _imagenet_normalize(x.clamp(0.0, 1.0))
        context = torch.no_grad() if self.freeze else _null_context()
        with context:
            visual = self.visual
            x = visual.conv1(x.type(visual.conv1.weight.dtype))
            bsz, channels, grid_h, grid_w = x.shape
            x = x.reshape(bsz, channels, -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype) + torch.zeros(
                bsz, 1, channels, dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls, x], dim=1)
            pos = visual.positional_embedding.to(x.dtype)
            if pos.shape[0] != x.shape[1]:
                pos = _resize_clip_pos_embed(pos, grid_h, grid_w)
            x = x + pos
            x = visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = visual.transformer(x)
            x = x.permute(1, 0, 2)
            x = visual.ln_post(x[:, 1:, :])
            if getattr(visual, "proj", None) is not None:
                x = x @ visual.proj
            x = x.permute(0, 2, 1).reshape(bsz, self.feature_dim, grid_h, grid_w)
        return x.float()


class DepthAnythingV2Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = cfg.get("encoder", "vits")
        self.input_size = int(cfg.get("input_size", 518))
        self.freeze = bool(cfg.get("freeze", True))
        self.blend_sparse_depth = bool(cfg.get("blend_sparse_depth", True))
        self.weight_path = str(Path(cfg.get("depth_weight_path", cfg.get("checkpoint_path", ""))).expanduser())
        repo_path = Path(cfg.get("repo_path", "")).expanduser()
        checkpoint_path = Path(self.weight_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"DepthAnythingV2 repo_path does not exist: {repo_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"DepthAnythingV2 checkpoint_path does not exist: {checkpoint_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        _configure_xformers_logging()
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except Exception as exc:
            raise ImportError(
                "Could not import depth_anything_v2.dpt.DepthAnythingV2. "
                f"Clone the official repo into {repo_path}."
            ) from exc
        model_cfg = _depth_anything_v2_model_cfg(self.encoder)
        self.feature_dim = _depth_anything_v2_feature_dim(self.encoder)
        self.model = DepthAnythingV2(**model_cfg)
        state = torch.load(str(checkpoint_path), map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        print(
            f"[model:2d] loaded DepthAnythingV2 encoder={self.encoder} checkpoint={checkpoint_path}",
            flush=True,
        )

    def forward(self, image):
        h, w = image.shape[-2:]
        size = _multiple_of_14(self.input_size)
        x = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
        x = _imagenet_normalize(x.clamp(0.0, 1.0))
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        context = torch.no_grad() if self.freeze else _null_context()
        with context:
            layers = self.model.pretrained.get_intermediate_layers(
                x, self.model.intermediate_layer_idx[self.encoder], return_class_token=True
            )
            depth = F.relu(self.model.depth_head(layers, patch_h, patch_w))
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        return _normalize_depth(depth)


def _depth_anything_v2_model_cfg(encoder):
    configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    if encoder not in configs:
        raise ValueError(f"Unsupported DepthAnythingV2 encoder: {encoder}")
    return configs[encoder]


def _depth_anything_v2_feature_dim(encoder):
    return {"vits": 384, "vitb": 768, "vitl": 1024}[encoder]


def _parse_dinov2_encoder(name):
    mapping = {
        "dinov2_vits14": "vits",
        "dinov2_vitb14": "vitb",
        "dinov2_vitl14": "vitl",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported DINOv2 image encoder: {name}")
    return mapping[name]


def _parse_clip_image_encoder(name):
    mapping = {
        "clip_vitl14": "ViT-L/14",
        "clip_vitb16": "ViT-B/16",
        "clip_vitb32": "ViT-B/32",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported CLIP image encoder: {name}")
    return mapping[name]


def _clip_cache_path(model_name, download_root):
    root = Path(download_root or "~/.cache/clip").expanduser()
    return str(root / (model_name.replace("/", "-").replace("@", "-") + ".pt"))


def _resize_clip_pos_embed(pos_embed, grid_h, grid_w):
    cls_pos = pos_embed[:1]
    patch_pos = pos_embed[1:]
    old_size = int(patch_pos.shape[0] ** 0.5)
    patch_pos = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
    return torch.cat([cls_pos, patch_pos], dim=0)


def _dinov2_feature_dim(encoder):
    return {"vits": 384, "vitb": 768, "vitl": 1024}[encoder]


def _clean_state_dict(state):
    if isinstance(state, dict):
        for key in ["state_dict", "model", "teacher", "student"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    cleaned = {}
    for key, value in state.items():
        clean_key = key
        for prefix in ["module.", "backbone.", "encoder.", "teacher.", "student."]:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        cleaned[clean_key] = value
    return cleaned


def _imagenet_normalize(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (image - mean) / std


def _normalize_depth(depth):
    flat = depth.flatten(2)
    min_value = flat.min(dim=-1).values.view(depth.shape[0], 1, 1, 1)
    max_value = flat.max(dim=-1).values.view(depth.shape[0], 1, 1, 1)
    return (depth - min_value) / (max_value - min_value + 1e-6)


def _multiple_of_14(value):
    value = int(value)
    return max(14, int(round(value / 14.0)) * 14)


def _configure_xformers_logging():
    if os.environ.get("POINTCLIP_SHOW_XFORMERS_WARNING", "0") != "1":
        logging.getLogger("dinov2").setLevel(logging.ERROR)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


VireoStyle2DBranch = Vireo2DOVBranch
