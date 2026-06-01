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
    args = parser.parse_args()

    cfg = load_config(args.config)
    mapper = build_label_mapper(cfg, PROJECT_ROOT)
    if mapper is None:
        raise SystemExit("No mapping.task_mapping_path configured.")
    train_vocab = build_vocabulary(_resolve(PROJECT_ROOT, cfg.vocabulary.train_vocab_path))
    eval_vocab = build_vocabulary(_resolve(PROJECT_ROOT, cfg.vocabulary.eval_vocab_path))
    _print_table("Train Mapping", mapper.coverage_rows(cfg.data.source.type, train_vocab, vocab_role="train"))
    _print_table("Eval Mapping", mapper.coverage_rows(cfg.data.target.type, eval_vocab, vocab_role="eval"))


def _print_table(title, rows):
    print("")
    print(title)
    print("=" * len(title))
    headers = ["raw_id", "canonical", "observed_count", "vocab_name", "vocab_id", "ignored"]
    table = [[row.get(key, "") for key in headers] for row in rows]
    print("\n".join(_format_table(headers, table)))


def _format_table(headers, rows):
    rows = [tuple(str(cell) for cell in row) for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep, "| " + " | ".join(str(header).ljust(width) for header, width in zip(headers, widths)) + " |", sep]
    for row in rows:
        out.append("| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |")
    out.append(sep)
    return out


def _resolve(root: Path, path: str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


if __name__ == "__main__":
    run()
