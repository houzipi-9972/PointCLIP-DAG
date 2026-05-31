from __future__ import annotations

import torch.nn as nn


class Point3DEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backend = cfg.get("backend", "unidseg_spconv")
        self.fallback_used = False
        self.out_dim = int(cfg.get("out_dim", 64))
        self.encoder = None
        self.import_error = None
        if self.backend == "unidseg_spconv":
            self._init_unidseg(cfg)
        if self.encoder is None:
            if not cfg.get("allow_mlp_fallback", True):
                raise ImportError(
                    "UniDSeg SpUNetBase could not be imported and MLP fallback is disabled. "
                    "Formal training requires the UniDSeg 3D stack and spconv in the active "
                    "conda environment. For the PointCLIP environment, install the CUDA-matched "
                    "package, for example: python -m pip install spconv-cu111==2.1.25. "
                    f"Original import error: {self.import_error.__class__.__name__}: {self.import_error}"
                ) from self.import_error
            self.backend = "mlp"
            self.fallback_used = True
            in_channels = int(cfg.get("in_channels", 1))
            hidden = int(cfg.get("mlp_hidden", 128))
            self.out_dim = int(cfg.get("out_dim", hidden))
            self.encoder = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.out_dim),
                nn.GELU(),
            )

    def _init_unidseg(self, cfg) -> None:
        try:
            from xmuda.models.spconv_unet_v1m1_base import SpUNetBase

            channels = tuple(cfg.get("channels", [32, 64, 128, 256, 256, 128, 96, 64]))
            self.encoder = SpUNetBase(
                in_channels=int(cfg.get("in_channels", 1)),
                num_classes=0,
                base_channels=int(cfg.get("base_channels", 32)),
                channels=channels,
                layers=tuple(cfg.get("layers", [2, 3, 4, 6, 2, 2, 2, 2])),
            )
            self.out_dim = int(channels[-1])
        except Exception as exc:
            self.import_error = exc
            self.encoder = None

    def forward(self, batch):
        if self.backend == "unidseg_spconv":
            skips, _ = self.encoder.encoder_forward(batch["x"])
            feats, _, _ = self.encoder.decoder_forward(skips)
            return feats
        return self.encoder(batch["features_3d"])
