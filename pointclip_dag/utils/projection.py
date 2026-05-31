from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_image_features(feature_map: torch.Tensor, point_xy, valid_mask=None):
    """Sample [B, C, H, W] features at row/col point coordinates.

    UniDSeg dataloaders provide img_indices as [row, col]. The adapter keeps the
    same convention and passes a list with one tensor per batch item.
    """
    samples = []
    masks = []
    coords = []
    batch_size, _, height, width = feature_map.shape
    for batch_idx in range(batch_size):
        xy = point_xy[batch_idx]
        if not torch.is_tensor(xy):
            xy = torch.as_tensor(xy, device=feature_map.device)
        xy = xy.to(feature_map.device).long()
        if xy.numel() == 0:
            masks.append(torch.zeros(0, dtype=torch.bool, device=feature_map.device))
            continue
        mask = (xy[:, 0] >= 0) & (xy[:, 0] < height) & (xy[:, 1] >= 0) & (xy[:, 1] < width)
        if valid_mask is not None:
            vm = valid_mask[batch_idx].to(feature_map.device).bool()
            mask &= vm
        rows = xy[mask, 0].float()
        cols = xy[mask, 1].float()
        grid_x = cols / max(width - 1, 1) * 2.0 - 1.0
        grid_y = rows / max(height - 1, 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            feature_map[batch_idx : batch_idx + 1],
            grid,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).squeeze(-1).transpose(0, 1)
        samples.append(sampled)
        masks.append(mask)
        coords.append(xy[mask])
    if samples:
        return torch.cat(samples, dim=0), masks, torch.cat(coords, dim=0) if coords else None
    return feature_map.new_zeros((0, feature_map.shape[1])), masks, None
