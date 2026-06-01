from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pointclip_dag.config import load_config
from pointclip_dag.data import build_dataloader, build_label_mapper, build_vocabulary, build_vocabulary_from_names
from pointclip_dag.engine import Evaluator
from pointclip_dag.models import build_model
from pointclip_dag.utils.checkpoint import load_checkpoint
from pointclip_dag.utils.logger import build_logger


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval-vocab", default="", help="Override cfg.vocab.eval with another vocabulary yaml.")
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated arbitrary text classes. These have no GT train_id unless written in a yaml vocab.",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.classes:
        eval_vocab = build_vocabulary_from_names(
            args.classes.split(","),
            prompt_templates=cfg.get("vocabulary", {}).get(
                "prompt_templates",
                ["a photo of a {}.", "a point cloud of a {}.", "a driving scene with {}."],
            ),
        )
    else:
        vocab_path = args.eval_vocab or cfg.get("vocabulary", {}).get("eval_vocab_path", cfg.vocab.eval)
        eval_vocab = build_vocabulary(_resolve(PROJECT_ROOT, vocab_path))
    train_vocab = build_vocabulary(_resolve(PROJECT_ROOT, _train_vocab_path(cfg)))
    probe_vocab = _build_probe_vocab(cfg)
    label_mapper = build_label_mapper(cfg, PROJECT_ROOT)
    loader = build_dataloader(cfg, "target", cfg.eval.get("test_split", "test"), vocabulary=eval_vocab)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = build_model(cfg, vocabulary=eval_vocab, label_mapper=label_mapper).to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    run_dir = _eval_out_dir(cfg, args)
    logger = build_logger("pointclip_dag_eval", str(run_dir / "eval.log"))
    evaluator = Evaluator(
        cfg,
        model,
        loader,
        eval_vocab,
        device,
        logger=logger,
        out_dir=run_dir,
        semantic_probe_vocabulary=probe_vocab,
        train_vocabulary=train_vocab,
    )
    evaluator.evaluate(save_predictions=args.save_predictions)


def _resolve(root: Path, path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def _eval_out_dir(cfg, args) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    ckpt_path = Path(args.ckpt).resolve()
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent / "eval"
    return Path(cfg.output.run_root) / cfg.experiment_name / "eval"


def _train_vocab_path(cfg) -> str:
    return cfg.get("vocabulary", {}).get("train_vocab_path", cfg.vocab.train)


def _build_probe_vocab(cfg):
    path = cfg.get("vocabulary", {}).get("semantic_probe_vocab_path", "")
    if not path:
        return None
    return build_vocabulary(_resolve(PROJECT_ROOT, path))


if __name__ == "__main__":
    run()
