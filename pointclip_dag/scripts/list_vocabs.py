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
    parser.add_argument("--task", default="", help="Optional task folder, e.g. vkitti_skitti or nuscenes.")
    args = parser.parse_args()

    root = PROJECT_ROOT / "configs" / "vocab"
    search_root = root / args.task if args.task else root
    for path in sorted(search_root.rglob("*.yaml")):
        vocab = build_vocabulary(path)
        rel = path.relative_to(root)
        labeled = sum(item.train_id is not None for item in vocab.classes)
        text_only = vocab.num_classes - labeled
        seen = sum(item.seen for item in vocab.classes)
        unseen = vocab.num_classes - seen
        print(f"{rel}: classes={vocab.num_classes}, labeled={labeled}, text_only={text_only}, seen={seen}, unseen={unseen}")


if __name__ == "__main__":
    run()
