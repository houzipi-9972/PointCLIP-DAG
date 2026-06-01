from __future__ import annotations

from pointclip_dag.config import setup_external_paths
from pointclip_dag.models.pointclip_dag import PointCLIPDAG
from pointclip_dag.utils.misc import freeze_module


def build_model(cfg, vocabulary=None):
    setup_external_paths(cfg)
    model = PointCLIPDAG(cfg, vocabulary=vocabulary)
    freeze = cfg.model.get("freeze", {})
    if freeze.get("text_encoder", True):
        freeze_module(model.text_encoder)
    if freeze.get("branch2d", False):
        freeze_module(model.branch2d)
    if freeze.get("branch3d", False):
        freeze_module(model.branch3d)
    return model
