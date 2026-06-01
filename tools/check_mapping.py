from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointclip_dag.config import load_config
from pointclip_dag.data import build_label_mapper, build_vocabulary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a vocab class has no raw-label coverage.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mapper = build_label_mapper(cfg, PROJECT_ROOT)
    if mapper is None:
        raise SystemExit("No mapping.task_mapping_path configured.")

    train_vocab = build_vocabulary(_resolve(PROJECT_ROOT, cfg.vocabulary.train_vocab_path))
    eval_vocab = build_vocabulary(_resolve(PROJECT_ROOT, cfg.vocabulary.eval_vocab_path))
    probe_vocab = None
    if cfg.get("vocabulary", {}).get("semantic_probe_vocab_path", ""):
        probe_vocab = build_vocabulary(_resolve(PROJECT_ROOT, cfg.vocabulary.semantic_probe_vocab_path))

    checks = [
        ("train", cfg.data.source.type, train_vocab),
        ("eval", cfg.data.target.type, eval_vocab),
    ]
    if probe_vocab is not None:
        checks.append(("semantic_probe", cfg.data.target.type, probe_vocab))

    failed = False
    print(f"task_mapping: {mapper.task_mapping_path}")
    print(f"task: {mapper.task_name}")
    for role, dataset_name, vocab in checks:
        rows = mapper.coverage_rows(dataset_name, vocab, vocab_role=role)
        mapped = [row for row in rows if not row["ignored"]]
        unmapped_vocab = mapper.unmapped_vocab_names(dataset_name, vocab, vocab_role=role)
        print("")
        print(f"[{role}] dataset={dataset_name} vocab_classes={vocab.num_classes}")
        print(f"mapped raw labels: {len(mapped)}/{len(rows)}")
        print(f"unmapped vocab classes: {unmapped_vocab}")
        if role in {"train", "eval"} and unmapped_vocab and args.strict:
            failed = True
    if failed:
        raise SystemExit(2)


def _resolve(root: Path, path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    run()
