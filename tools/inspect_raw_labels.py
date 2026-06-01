from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path

import numpy as np


def run():
    parser = argparse.ArgumentParser(description="Inspect raw label ids from local dataset files.")
    parser.add_argument("--dataset", required=True, choices=["semantic_kitti", "virtual_kitti", "nuscenes"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional file/sample limit for quick checks.")
    args = parser.parse_args()

    root = Path(args.root)
    if args.dataset == "semantic_kitti":
        counts, units = _scan_semantic_kitti(root, args.limit)
    elif args.dataset == "virtual_kitti":
        counts, units = _scan_virtual_kitti(root, args.limit)
    else:
        counts, units = _scan_nuscenes(root, args.limit)
    print(f"dataset={args.dataset} scanned={units}")
    for label_id, count in sorted(counts.items()):
        print(f"{label_id}: {count}")


def _scan_semantic_kitti(root: Path, limit: int):
    files = list(root.glob("dataset/sequences/*/labels/*.label"))
    if not files:
        files = list(root.glob("**/*.label"))
    if limit > 0:
        files = files[:limit]
    counts = collections.Counter()
    for path in files:
        labels = np.fromfile(path, dtype=np.uint32) & 0xFFFF
        _update_counts(counts, labels)
    return counts, len(files)


def _scan_virtual_kitti(root: Path, limit: int):
    pkl_path = root / "preprocess" / "train.pkl"
    if root.suffix == ".pkl":
        pkl_path = root
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)
    if limit > 0:
        data = data[:limit]
    counts = collections.Counter()
    for item in data:
        _update_counts(counts, np.asarray(item["seg_labels"], dtype=np.int64))
    return counts, len(data)


def _scan_nuscenes(root: Path, limit: int):
    files = list(root.glob("nuScenes-lidarseg-all-v1.0/lidarseg/**/*_lidarseg.bin"))
    if not files:
        files = list(root.glob("**/*_lidarseg.bin"))
    if limit > 0:
        files = files[:limit]
    counts = collections.Counter()
    for path in files:
        _update_counts(counts, np.fromfile(path, dtype=np.uint8))
    return counts, len(files)


def _update_counts(counts, labels):
    unique, unique_counts = np.unique(labels, return_counts=True)
    counts.update(dict(zip(map(int, unique), map(int, unique_counts))))


if __name__ == "__main__":
    run()
