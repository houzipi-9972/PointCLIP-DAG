from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if not args.eval_only:
        subprocess.check_call([sys.executable, str(PROJECT_ROOT / "tools" / "train.py"), "--config", args.config])
    if args.ckpt:
        subprocess.check_call(
            [sys.executable, str(PROJECT_ROOT / "tools" / "eval.py"), "--config", args.config, "--ckpt", args.ckpt]
        )


if __name__ == "__main__":
    run()
