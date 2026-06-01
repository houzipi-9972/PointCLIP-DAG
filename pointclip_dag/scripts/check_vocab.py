from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointclip_dag.data import build_vocabulary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab", help="Path to a vocabulary yaml.")
    args = parser.parse_args()

    vocab_path = Path(args.vocab)
    if not vocab_path.is_absolute():
        vocab_path = vocab_path if vocab_path.exists() else PROJECT_ROOT / vocab_path
    vocab = build_vocabulary(vocab_path)

    labeled = [item for item in vocab.classes if item.raw_labels]
    text_only = [item for item in vocab.classes if not item.raw_labels]
    seen = [item for item in vocab.classes if item.seen]
    unseen = [item for item in vocab.classes if not item.seen]

    print(f"vocab: {vocab_path}")
    print(f"classes: {vocab.num_classes}")
    print(f"with raw_label: {len(labeled)}")
    print(f"text-only queries: {len(text_only)}")
    print(f"seen: {len(seen)}")
    print(f"unseen: {len(unseen)}")
    print()
    for idx, item in enumerate(vocab.classes):
        raw_label = "none" if not item.raw_labels else list(item.raw_labels)
        split = "seen" if item.seen else "unseen"
        alias_count = len(item.aliases)
        print(
            f"{idx:03d} raw_label={raw_label} mapping_label={item.mapping_label} "
            f"{split:6s} aliases={alias_count:02d} name={item.name}"
        )


if __name__ == "__main__":
    run()
