from __future__ import annotations

import torch
import torch.nn as nn

from pointclip_dag.data.vocabulary import map_labels_to_vocab
from pointclip_dag.models.point3d import Point3DEncoder
from pointclip_dag.models.projection_head import ProjectionHead
from pointclip_dag.models.text_encoder import TextEncoder
from pointclip_dag.models.vireo2d import Vireo2DOVBranch
from pointclip_dag.utils.projection import sample_image_features


class PointCLIPDAG(nn.Module):
    def __init__(self, cfg, vocabulary=None):
        super().__init__()
        self.cfg = cfg
        self.vocabulary = vocabulary
        self.text_encoder = TextEncoder(cfg.model.text_encoder)
        text_dim = self.text_encoder.embed_dim
        self.branch2d = Vireo2DOVBranch(cfg.model.branch2d, text_dim=text_dim)
        self.branch3d = Point3DEncoder(cfg.model.branch3d)
        self.head3d = ProjectionHead(
            self.branch3d.out_dim,
            text_dim,
            hidden_dim=cfg.model.projection_head.get("hidden_dim", None),
            dropout=float(cfg.model.projection_head.get("dropout", 0.0)),
        )
        self.logit_scale = float(cfg.loss.get("logit_scale", 20.0))

    def set_vocabulary(self, vocabulary) -> None:
        self.vocabulary = vocabulary

    def encode_text(self, device):
        if self.vocabulary is None:
            raise ValueError("PointCLIPDAG needs a vocabulary before forward.")
        return self.text_encoder.encode_groups(self.vocabulary.class_text_groups(), device=device)

    def forward(self, batch):
        device = batch["image"].device
        text_embeddings = torch.nn.functional.normalize(self.encode_text(device), dim=-1)
        feat3d = self.branch3d(batch)
        z3d = self.head3d(feat3d)
        logits3d = self.logit_scale * z3d @ text_embeddings.t()

        branch2d_outputs = self.branch2d(batch["image"], batch["sparse_depth"], text_embeddings)
        z2d_map = branch2d_outputs["z2d_map"]
        z2d_points, masks, point_xy = sample_image_features(z2d_map, batch["point2img"], batch.get("valid_mask"))
        valid_point_mask = torch.cat(masks, dim=0) if masks else torch.zeros(0, dtype=torch.bool, device=device)
        logits2d_points = self.logit_scale * z2d_points @ text_embeddings.t()
        logits2d_map = self.logit_scale * torch.einsum("bdhw,kd->bkhw", z2d_map, text_embeddings)

        dataset_name = _get_dataset_name(batch)
        point_labels = map_labels_to_vocab(
            batch["labels_3d"],
            dataset_name,
            self.vocabulary,
            ignore_index=int(self.cfg.loss.ignore_index),
        )

        outputs = {
            "text_embeddings": text_embeddings,
            "z3d": z3d,
            "logits3d": logits3d,
            "z2d_points": z2d_points,
            "logits2d_points": logits2d_points,
            "valid_point_mask": valid_point_mask,
            "point_labels": point_labels,
            "labels_vocab": point_labels,
            "raw_point_labels": batch["labels_3d"],
            "point_to_image_xy": point_xy,
            "z2d_map": z2d_map,
            "logits2d_map": logits2d_map,
            "coarse_logits2d_map": branch2d_outputs.get("coarse_logits2d_map", None),
            "dov_attention_maps": branch2d_outputs.get("dov_attention_maps", None),
            "branch2d_depth_mode": branch2d_outputs.get("depth_mode", ""),
        }
        if valid_point_mask.numel() == z3d.shape[0]:
            outputs["z3d_valid"] = z3d[valid_point_mask]
            outputs["logits3d_valid"] = logits3d[valid_point_mask]
            outputs["labels_valid"] = point_labels[valid_point_mask]
        else:
            outputs["z3d_valid"] = z3d.new_zeros((0, z3d.shape[1]))
            outputs["logits3d_valid"] = logits3d.new_zeros((0, logits3d.shape[1]))
            outputs["labels_valid"] = batch["labels_3d"].new_zeros((0,))
        return outputs


def _get_dataset_name(batch) -> str:
    dataset_name = batch.get("dataset_name", "")
    if isinstance(dataset_name, (list, tuple)):
        return dataset_name[0] if dataset_name else ""
    return dataset_name
