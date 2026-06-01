from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pointclip_dag.config import load_config, save_config
from pointclip_dag.data import build_dataloader, build_vocabulary
from pointclip_dag.engine import Trainer
from pointclip_dag.losses import build_loss
from pointclip_dag.models import build_model
from pointclip_dag.utils.checkpoint import load_checkpoint
from pointclip_dag.utils.logger import build_logger
from pointclip_dag.utils.seed import set_random_seed


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_random_seed(int(cfg.seed), deterministic=bool(cfg.get("deterministic", False)))
    run_dir = _build_run_dir(cfg, args)
    logger = build_logger("pointclip_dag", str(run_dir / "logs" / "train.log"))
    save_config(cfg, run_dir / "config.yaml")
    logger.info("project_root=%s", cfg.project_root)
    logger.info("run_dir=%s", run_dir)

    train_vocab = build_vocabulary(_resolve(PROJECT_ROOT, _vocab_path(cfg, "train")))
    eval_vocab = build_vocabulary(_resolve(PROJECT_ROOT, _vocab_path(cfg, "eval")))
    logger.info("train_vocab=%s classes=%d names=%s", _vocab_path(cfg, "train"), train_vocab.num_classes, train_vocab.names)
    logger.info("eval_vocab=%s classes=%d names=%s", _vocab_path(cfg, "eval"), eval_vocab.num_classes, eval_vocab.names)
    logger.info("train_raw_label_mapping=%s", train_vocab.to_label_mapping())
    logger.info("eval_raw_label_mapping=%s", eval_vocab.to_label_mapping())
    probe_path = cfg.get("vocabulary", {}).get("semantic_probe_vocab_path", "")
    if probe_path:
        probe_vocab = build_vocabulary(_resolve(PROJECT_ROOT, probe_path))
        logger.info("semantic_probe_vocab=%s classes=%d names=%s", probe_path, probe_vocab.num_classes, probe_vocab.names)

    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = build_model(cfg, vocabulary=train_vocab).to(device)
    _log_model_sanity(logger, model)
    loss_fn = build_loss(cfg)
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)

    train_loader = build_dataloader(cfg, "source", "train", vocabulary=train_vocab)
    val_loader = None
    if cfg.eval.get("enabled", True):
        val_loader = build_dataloader(cfg, "target", cfg.eval.get("split", "val"), vocabulary=eval_vocab)

    start_step = 0
    best_miou = -1.0
    if args.resume:
        payload = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_step = int(payload.get("global_step", 0))
        best_miou = float(payload.get("best_miou", -1.0))
        logger.info("resumed checkpoint %s at step %s", args.resume, payload.get("global_step", "unknown"))

    trainer = Trainer(
        cfg,
        model,
        loss_fn,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_vocab,
        eval_vocab,
        device,
        run_dir,
        logger,
        start_step=start_step,
        best_miou=best_miou,
    )
    trainer.train()


def _build_optimizer(cfg, model):
    params = [param for param in model.parameters() if param.requires_grad]
    opt_cfg = cfg.optimizer
    if opt_cfg.type.lower() == "adamw":
        return torch.optim.AdamW(params, lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))
    if opt_cfg.type.lower() == "sgd":
        return torch.optim.SGD(params, lr=float(opt_cfg.lr), momentum=0.9, weight_decay=float(opt_cfg.weight_decay))
    return torch.optim.Adam(params, lr=float(opt_cfg.lr), weight_decay=float(opt_cfg.weight_decay))


def _build_scheduler(cfg, optimizer):
    sched_cfg = cfg.get("scheduler", {})
    if not sched_cfg or sched_cfg.get("type", "none") == "none":
        return None
    if sched_cfg.type == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sched_cfg.get("milestones", [])),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
    if sched_cfg.type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(sched_cfg.t_max))
    raise ValueError(f"Unsupported scheduler: {sched_cfg.type}")


def _log_model_sanity(logger, model):
    groups = {
        "text_encoder": model.text_encoder,
        "projection_head": model.head3d,
        "branch3d": model.branch3d,
        "branch2d": model.branch2d,
    }
    for name, module in groups.items():
        logger.info("%s trainable params = %d", name, _count_trainable(module))
    logger.info("Actual 3D backend: %s", model.branch3d.backend)
    logger.info("MLP fallback used: %s", str(getattr(model.branch3d, "fallback_used", False)).lower())
    logger.info("2D branch class: %s", model.branch2d.__class__.__name__)
    logger.info("2D branch structure: %s", getattr(model.branch2d, "structure_name", "unknown"))
    image_encoder = getattr(model.branch2d, "image_encoder", None)
    logger.info("2D image encoder type: %s", getattr(model.branch2d, "image_encoder_type", "unknown"))
    logger.info("2D image encoder weight path: %s", getattr(image_encoder, "weight_path", ""))
    logger.info("2D image encoder pretrained weights loaded: %s", str(getattr(image_encoder, "pretrained_loaded", False)).lower())
    logger.info("2D image encoder missing keys: %s", _format_keys(getattr(image_encoder, "missing_keys", [])))
    logger.info("2D image encoder unexpected keys: %s", _format_keys(getattr(image_encoder, "unexpected_keys", [])))
    logger.info("2D image encoder trainable params = %d", _count_trainable(image_encoder))
    logger.info("2D adapter/head trainable params = %d", _count_branch2d_adapter_trainable(model.branch2d))
    logger.info("2D depth mode: %s", getattr(model.branch2d, "depth_mode", "unknown"))
    logger.info("2D depth weight path: %s", getattr(getattr(model.branch2d, "depth_vfm", None), "weight_path", ""))
    logger.info("2D uses DepthAnything depth branch: %s", str(getattr(model.branch2d, "depth_vfm", None) is not None).lower())
    logger.info("2D has GeoTextTokenRefiner: %s", str(hasattr(model.branch2d, "geotext")).lower())
    logger.info("2D has CoarseMaskPriorEmbedding: %s", str(hasattr(model.branch2d, "cmpe")).lower())
    logger.info("2D DepthAnything trainable params = %d", _count_trainable(getattr(model.branch2d, "depth_vfm", None)))
    if getattr(model.branch3d, "fallback_used", False):
        raise RuntimeError("Formal training aborted because branch3d used MLP fallback.")
    if _count_trainable(model.text_encoder) != 0:
        raise RuntimeError("Formal training aborted because text_encoder has trainable parameters.")


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


def _resolve(root: Path, path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _vocab_path(cfg, split: str) -> str:
    vocab_cfg = cfg.get("vocabulary", {})
    if split == "train":
        return vocab_cfg.get("train_vocab_path", cfg.vocab.train)
    return vocab_cfg.get("eval_vocab_path", cfg.vocab.eval)


def _build_run_dir(cfg, args) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    if args.resume:
        ckpt_path = Path(args.resume).resolve()
        if ckpt_path.parent.name == "checkpoints":
            return ckpt_path.parent.parent
    root = Path(cfg.output.run_root) / cfg.experiment_name
    use_timestamp = bool(cfg.output.get("timestamp_subdir", True))
    if not use_timestamp:
        return root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / timestamp


if __name__ == "__main__":
    run()
