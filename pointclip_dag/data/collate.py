from __future__ import annotations

import torch


def collate_pointclip(batch: list[dict]) -> dict:
    locs = []
    feats = []
    labels = []
    images = []
    depths = []
    point2img = []
    valid_masks = []
    scene_ids = []
    frame_ids = []
    dataset_names = []

    for batch_idx, item in enumerate(batch):
        coords = _tensor(item["coords"], torch.long)
        batch_col = torch.full((coords.shape[0], 1), batch_idx, dtype=torch.long)
        locs.append(torch.cat([coords, batch_col], dim=1))
        feats.append(_tensor(item["features_3d"], torch.float32))
        labels.append(_tensor(item["labels_3d"], torch.long))
        images.append(_tensor(item["image"], torch.float32))
        depths.append(_tensor(item["sparse_depth"], torch.float32))
        xy = _tensor(item["point2img"], torch.long)
        point2img.append(xy)
        valid_masks.append(_tensor(item.get("valid_mask", torch.ones(xy.shape[0], dtype=torch.bool)), torch.bool))
        scene_ids.append(item.get("scene_id", ""))
        frame_ids.append(item.get("frame_id", ""))
        dataset_names.append(item.get("dataset_name", ""))

    return {
        "x": [torch.cat(locs, dim=0), torch.cat(feats, dim=0)],
        "points": torch.cat([x[:, :3] for x in locs], dim=0),
        "features_3d": torch.cat(feats, dim=0),
        "labels_3d": torch.cat(labels, dim=0),
        "image": torch.stack(images, dim=0),
        "sparse_depth": torch.stack(depths, dim=0),
        "point2img": point2img,
        "valid_mask": valid_masks,
        "scene_id": scene_ids,
        "frame_id": frame_ids,
        "dataset_name": dataset_names,
    }


def _tensor(value, dtype):
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)
