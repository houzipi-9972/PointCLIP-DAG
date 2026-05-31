from __future__ import annotations

import time

from torch.utils.data import DataLoader

from pointclip_dag.config import setup_external_paths
from pointclip_dag.data.collate import collate_pointclip
from pointclip_dag.data.unidseg_adapter import SemanticKITTIRawSCN, SyntheticDataset, UniDSegDatasetAdapter, VirtualKITTIRawSCN


def build_dataset(cfg, domain: str, mode: str, vocabulary=None):
    start = time.time()
    setup_external_paths(cfg)
    data_cfg = cfg.data[domain]
    dataset_type = data_cfg.type
    print(f"[data] building {domain}/{mode}: type={dataset_type}", flush=True)
    if dataset_type == "SyntheticDataset":
        kwargs = dict(data_cfg.get("kwargs", {}))
        if vocabulary is not None:
            kwargs.setdefault("num_classes", vocabulary.num_classes)
        dataset = SyntheticDataset(**kwargs)
        print(f"[data] built {domain}/{mode}: samples={len(dataset)} time={time.time() - start:.1f}s", flush=True)
        return dataset
    split = data_cfg.get(mode, data_cfg.get("split"))
    kwargs = dict(data_cfg.get("kwargs", {}))
    if dataset_type == "SemanticKITTIRawSCN":
        dataset = SemanticKITTIRawSCN(split=tuple(split), **kwargs)
    elif dataset_type == "VirtualKITTIRawSCN":
        dataset = VirtualKITTIRawSCN(split=tuple(split), **kwargs)
    else:
        dataset = UniDSegDatasetAdapter(dataset_type, split=split, dataset_kwargs=kwargs, mode=mode)
    print(f"[data] built {domain}/{mode}: samples={len(dataset)} time={time.time() - start:.1f}s", flush=True)
    return dataset


def build_dataloader(cfg, domain: str, mode: str, vocabulary=None) -> DataLoader:
    dataset = build_dataset(cfg, domain, mode, vocabulary=vocabulary)
    is_train = mode == "train"
    loader_cfg = cfg.dataloader
    batch_size = cfg.train.batch_size if is_train else cfg.eval.batch_size
    print(
        f"[data] dataloader {domain}/{mode}: batch_size={batch_size} "
        f"num_workers={loader_cfg.num_workers} drop_last={is_train and loader_cfg.get('drop_last', True)}",
        flush=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.get("pin_memory", True),
        drop_last=is_train and loader_cfg.get("drop_last", True),
        collate_fn=collate_pointclip,
    )
