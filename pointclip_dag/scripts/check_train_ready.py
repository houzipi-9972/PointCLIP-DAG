from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pointclip_dag.config import load_config
from pointclip_dag.data import build_dataloader, build_vocabulary
from pointclip_dag.models.text_encoder import _openai_clip_name


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiments" / "vkitti_to_skitti.yaml"))
    parser.add_argument("--check-batch", action="store_true")
    parser.add_argument("--check-model", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"config: {args.config}")
    _check_imports()
    _check_paths(cfg)
    _check_clip_cache(cfg)
    _check_branch2d_weights(cfg)

    vocab = build_vocabulary(_resolve(PROJECT_ROOT, _vocab_path(cfg, "train")))
    eval_vocab = build_vocabulary(_resolve(PROJECT_ROOT, _vocab_path(cfg, "eval")))
    print(f"train vocabulary: {vocab.num_classes} classes -> {vocab.names}")
    print(f"eval vocabulary: {eval_vocab.num_classes} classes -> {eval_vocab.names}")

    if args.check_batch:
        loader = build_dataloader(cfg, "source", "train", vocabulary=vocab)
        batch = next(iter(loader))
        print("source batch:")
        print(f"  points: {tuple(batch['points'].shape)}")
        print(f"  features_3d: {tuple(batch['features_3d'].shape)}")
        print(f"  labels_3d: {tuple(batch['labels_3d'].shape)}")
        print(f"  image: {tuple(batch['image'].shape)}")
        print(f"  sparse_depth: {tuple(batch['sparse_depth'].shape)}")
        valid_points = sum(int(mask.sum().item()) for mask in batch["valid_mask"])
        total_points = sum(int(mask.numel()) for mask in batch["valid_mask"])
        print(f"  valid projected points: {valid_points}/{total_points} ({valid_points / max(total_points, 1):.4f})")

    if args.check_model:
        import torch
        from pointclip_dag.models import build_model

        model = build_model(cfg, vocabulary=vocab)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model built: {model.__class__.__name__}, trainable params={num_trainable:,}")
        print(f"text_encoder trainable params = {_count_trainable(model.text_encoder):,}")
        print(f"projection_head trainable params = {_count_trainable(model.head3d):,}")
        print(f"branch3d trainable params = {_count_trainable(model.branch3d):,}")
        print(f"branch2d trainable params = {_count_trainable(model.branch2d):,}")
        print(f"Actual 3D backend: {model.branch3d.backend}")
        print(f"MLP fallback used: {str(getattr(model.branch3d, 'fallback_used', False)).lower()}")
        print(f"2D branch class: {model.branch2d.__class__.__name__}")
        print(f"2D branch structure: {getattr(model.branch2d, 'structure_name', 'unknown')}")
        image_encoder = getattr(model.branch2d, "image_encoder", None)
        print(f"2D image encoder type: {getattr(model.branch2d, 'image_encoder_type', 'unknown')}")
        print(f"2D image encoder weight path: {getattr(image_encoder, 'weight_path', '')}")
        print(f"2D image encoder pretrained weights loaded: {str(getattr(image_encoder, 'pretrained_loaded', False)).lower()}")
        print(f"2D image encoder missing keys: {_format_keys(getattr(image_encoder, 'missing_keys', []))}")
        print(f"2D image encoder unexpected keys: {_format_keys(getattr(image_encoder, 'unexpected_keys', []))}")
        print(f"2D image encoder trainable params = {_count_trainable(image_encoder):,}")
        print(f"2D adapter/head trainable params = {_count_branch2d_adapter_trainable(model.branch2d):,}")
        print(f"2D depth mode: {getattr(model.branch2d, 'depth_mode', 'unknown')}")
        print(f"2D depth weight path: {getattr(getattr(model.branch2d, 'depth_vfm', None), 'weight_path', '')}")
        print(f"2D uses DepthAnything depth branch: {str(getattr(model.branch2d, 'depth_vfm', None) is not None).lower()}")
        print(f"2D has GeoTextTokenRefiner: {str(hasattr(model.branch2d, 'geotext')).lower()}")
        print(f"2D has CoarseMaskPriorEmbedding: {str(hasattr(model.branch2d, 'cmpe')).lower()}")
        print(f"2D DepthAnything trainable params = {_count_trainable(getattr(model.branch2d, 'depth_vfm', None)):,}")
        print(f"cuda available: {torch.cuda.is_available()}")

    print("ready check finished")


def _check_imports():
    for name in ["torch", "torchvision", "clip", "spconv", "mmcv", "mmseg", "nuscenes"]:
        try:
            module = importlib.import_module(name)
            print(f"{name}: {getattr(module, '__version__', 'available')}")
        except Exception as exc:
            status = "required for nuScenes" if name == "nuscenes" else "missing"
            print(f"{name}: {status} ({exc.__class__.__name__})")


def _check_paths(cfg):
    for domain in ["source", "target"]:
        data_cfg = cfg.data[domain]
        kwargs = data_cfg.get("kwargs", {})
        for key in ["preprocess_dir", "virtual_kitti_dir", "semantic_kitti_dir", "nuscenes_dir"]:
            if key in kwargs:
                path = Path(kwargs[key])
                print(f"{domain}.{key}: {path} {'OK' if path.exists() else 'MISSING'}")
        for split_key in ["train", "val", "test"]:
            if split_key in data_cfg and "preprocess_dir" in kwargs:
                for split in data_cfg[split_key]:
                    pkl = Path(kwargs["preprocess_dir"]) / f"{split}.pkl"
                    print(f"{domain}.{split}.pkl: {pkl} {'OK' if pkl.exists() else 'MISSING'}")


def _check_clip_cache(cfg):
    text_cfg = cfg.model.text_encoder
    if text_cfg.backend not in {"clip", "openai_clip", "open_clip"}:
        print(f"clip cache: skipped for backend={text_cfg.backend}")
        return
    root = Path(text_cfg.get("download_root", "~/.cache/clip")).expanduser()
    model_name = _openai_clip_name(text_cfg.model_name)
    cache_name = model_name.replace("/", "-").replace("@", "-")
    candidates = sorted(root.glob(f"{cache_name}*.pt"))
    if candidates:
        print(f"clip cache: {candidates[0]} OK")
    else:
        print(f"clip cache: missing for {model_name} under {root}; run pointclip_dag/scripts/prepare_weights.py")


def _check_branch2d_weights(cfg):
    branch_cfg = cfg.model.get("branch2d", {})
    image_encoder = branch_cfg.get("image_encoder", "none")
    print(f"branch2d.image_encoder: {image_encoder}")
    if branch_cfg.get("pretrained", False) and image_encoder.startswith("dinov2_"):
        dino_path = Path(branch_cfg.get("image_encoder_weight_path", branch_cfg.get("dino_weight_path", ""))).expanduser()
        print(f"branch2d.image_encoder_weight_path: {dino_path} {'OK' if dino_path.exists() else 'MISSING'}")
    if branch_cfg.get("pretrained", False) and image_encoder.startswith("clip_"):
        root = Path(branch_cfg.get("clip_download_root", "~/.cache/clip")).expanduser()
        print(f"branch2d.clip_download_root: {root} {'OK' if root.exists() else 'MISSING'}")
    depth_cfg = branch_cfg.get("depth_vfm", {})
    depth_path = Path(
        branch_cfg.get("depth_weight_path", depth_cfg.get("depth_weight_path", depth_cfg.get("checkpoint_path", "")))
    ).expanduser()
    if branch_cfg.get("enable_depth", branch_cfg.get("enable_depth_vfm", False)):
        print(f"branch2d.depth_weight_path: {depth_path} {'OK' if depth_path.exists() else 'MISSING'}")


def _resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def _vocab_path(cfg, split):
    vocab_cfg = cfg.get("vocabulary", {})
    if split == "train":
        return vocab_cfg.get("train_vocab_path", cfg.vocab.train)
    return vocab_cfg.get("eval_vocab_path", cfg.vocab.eval)


def _count_trainable(module):
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _count_branch2d_adapter_trainable(branch2d):
    total = 0
    for name, param in branch2d.named_parameters():
        if name.startswith("image_encoder.") or name.startswith("depth_vfm."):
            continue
        if param.requires_grad:
            total += param.numel()
    return total


def _format_keys(keys, max_items=20):
    keys = list(keys or [])
    if len(keys) <= max_items:
        return keys
    return keys[:max_items] + [f"... ({len(keys) - max_items} more)"]


if __name__ == "__main__":
    run()
