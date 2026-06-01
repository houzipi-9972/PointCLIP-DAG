from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pointclip_dag.config import load_config
from pointclip_dag.data import build_dataloader
from pointclip_dag.data.vocabulary import Vocabulary, VocabClass
from pointclip_dag.losses import build_loss
from pointclip_dag.models import build_model
from pointclip_dag.utils.misc import move_to_device


def run():
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    cfg.model.text_encoder.backend = "none"
    cfg.model.branch3d.backend = "mlp"
    cfg.model.branch3d.allow_mlp_fallback = True
    cfg.train.batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for num_classes in [12, 19, 30]:
        cfg.data.source.type = "SyntheticDataset"
        cfg.data.source.kwargs.num_classes = num_classes
        cfg.data.target.type = "SyntheticDataset"
        cfg.data.target.kwargs.num_classes = num_classes
        vocab = _synthetic_vocab(num_classes)
        loader = build_dataloader(cfg, "source", "train", vocabulary=vocab)
        model = build_model(cfg, vocabulary=vocab).to(device)
        loss_fn = build_loss(cfg)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        batch = move_to_device(next(iter(loader)), device)
        outputs = model(batch)
        losses = loss_fn(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optimizer.step()
        values = {key: float(value.detach().cpu()) for key, value in losses.items()}
        print(f"classes={num_classes}", values)


def _synthetic_vocab(num_classes: int) -> Vocabulary:
    classes = [
        VocabClass(
            name=f"synthetic class {idx}",
            raw_labels=(idx,),
            mapping_label=f"synthetic class {idx}",
            aliases=tuple(),
            seen=idx % 2 == 0,
        )
        for idx in range(num_classes)
    ]
    return Vocabulary(classes, prompt_templates=["a point cloud of a {}."])


if __name__ == "__main__":
    run()
