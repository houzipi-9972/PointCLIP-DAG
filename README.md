# PointCLIP-DAG

Domain-Generalized Open-Vocabulary 3D Semantic Segmentation.

This first version implements text-aligned 3D point embeddings, a lightweight
Vireo-style 2D open-vocabulary teacher, and 2D-to-3D distillation. It does not
use a fixed Linear classifier, coarse/fine hierarchy, parent-child label tree,
or closed-set inference head.

## What Is Reused

- UniDSeg data preprocessing/dataloaders are wrapped by
  `pointclip_dag/data/unidseg_adapter.py`.
- UniDSeg `SpUNetBase` is used by `pointclip_dag/models/point3d.py` when
  `spconv` and the UniDSeg python path are available.
- Vireo is not imported as an mmseg model in this version because its full stack
  is tied to mmseg/mmdet Mask2Former, Hungarian matching, and dense 2D mask
  labels. `pointclip_dag/models/vireo2d.py` implements `Vireo2DOVBranch`, a
  lightweight wrapper that keeps the transferable Vireo mechanisms: frozen
  DepthAnything/DINO features, depth-aware GeoText token refinement, coarse mask
  prior embedding, and dynamic text-dot-product output.
- CLIP/open_clip official pretrained text encoders are supported, but Vireo
  Cityscapes/ACDC/DELIVER task checkpoints are intentionally not loaded.

## Project Layout

```text
configs/
  default.yaml
  experiments/
    vkitti_to_skitti.yaml
    nuscenes_day_to_night.yaml
    nuscenes_usa_to_singapore.yaml
  vocab/
    vkitti_skitti/
      vkitti_raw_train.yaml
      semantic_kitti_gt_aligned_19.yaml
      semantic_probe_extended.yaml
    nuscenes/
      raw_lidarseg_16.yaml
      driving_queries.yaml
pointclip_dag/
  data/ models/ losses/ engine/ utils/ scripts/
tools/
  train.py eval.py run_experiment.py
```

## Environment Check

```bash
cd /home/zhangshuai/Sysu_4T/houych/PointCLIP-DAG
/home/zhangshuai/anaconda3/envs/UniDSeg/bin/python pointclip_dag/scripts/check_env.py
```

For real training, use an environment compatible with UniDSeg first: PyTorch,
CUDA, spconv, mmcv/mmseg if your UniDSeg install needs them, plus
OpenAI CLIP. nuScenes experiments require `nuscenes-devkit`; the adapter checks
for it before loading nuScenes.

```bash
pip install -r requirements.txt
python -m pip install timm==0.4.12
python -m pip install spconv-cu111==2.1.25
```

`spconv` is required for formal training because
`configs/experiments/vkitti_to_skitti.yaml` uses `branch3d.backend:
unidseg_spconv` and `allow_mlp_fallback: false`. If `python -c "import
spconv.pytorch"` fails, training will stop before model construction instead of
silently falling back to an MLP.

The Aliyun PyPI mirror currently exposes `spconv-cu111` up to `2.1.25`, so this
project pins `2.1.25` for the `PointCLIP` environment. A newer `2.3.x` build is
also acceptable if it is already installed in another UniDSeg-compatible
environment.

The current `PointCLIP` conda environment on this machine already has:

- `torch 1.9.1+cu111`
- `torchvision 0.10.1+cu111`
- `spconv 2.1.25+`
- OpenAI `clip`
- `nuscenes-devkit`

`mmcv/mmseg` are not required by the current raw vKITTI -> SemanticKITTI
adapter. Do not install or upgrade them unless you explicitly switch back to
MMSeg-based Vireo modules.

`open_clip` is optional for this first vKITTI -> SemanticKITTI run because the
default config uses OpenAI CLIP.

To avoid CUDA library conflicts, activate through the project script or install
the conda hooks:

```bash
source scripts/activate_pointclip.sh
# or once:
./scripts/install_pointclip_conda_hooks.sh
```

The active `LD_LIBRARY_PATH` should be exactly:

```text
/home/zhangshuai/anaconda3/envs/PointCLIP/lib
```

`xFormers` is intentionally optional for this PyTorch 1.9/CUDA 11.1 stack.
DepthAnything/DINOv2 uses the standard attention fallback; the project suppresses
the noisy `xFormers not available` warning by default.

## Required Weights

For the vKITTI -> SemanticKITTI training run, the 2D branch uses separate
pretrained weights for text, RGB image VFM, and optional depth VFM.

- CLIP `ViT-L/14`: `weights/clip/ViT-L-14.pt`
- DINOv2 RGB image encoder `dinov2_vitl14`:
  `weights/dinov2/dinov2_vitl14_pretrain_torch19.pth`
- Depth Anything V2 Small depth branch:
  `weights/depth_anything_v2/depth_anything_v2_vits.pth`

CLIP text weights can be verified or downloaded through OpenAI CLIP:

```bash
python \
  pointclip_dag/scripts/prepare_weights.py \
  --clip-model ViT-L/14 \
  --download-root weights/clip
```

The RGB image encoder weight is independent from Depth Anything. Depth Anything
weights must not be used as `image_encoder_weight_path` or `dino_weight_path`.

```bash
cd /home/zhangshuai/Sysu_4T/houych/PointCLIP-DAG
python pointclip_dag/scripts/prepare_weights.py \
  --skip-clip \
  --dinov2 \
  --dinov2-encoder vitl14 \
  --dinov2-dir weights/dinov2
```

If the server cannot access the network, manually place the DINOv2 checkpoint at
the configured `model.branch2d.image_encoder_weight_path`. With `pretrained:
true`, the trainer raises an error if that file is missing instead of silently
using random weights.

The active `PointCLIP` environment uses torch 1.9.1. If the official DINOv2
checkpoint cannot be read by torch 1.9, convert it once with a newer local torch
environment:

```bash
/home/zhangshuai/anaconda3/envs/PointNext/bin/python -c \
  "import torch; inp='weights/dinov2/dinov2_vitl14_pretrain.pth'; out='weights/dinov2/dinov2_vitl14_pretrain_torch19.pth'; state=torch.load(inp,map_location='cpu'); torch.save(state,out,_use_new_zipfile_serialization=False)"
```

If DINOv2 weights are not available yet, you can explicitly switch the RGB
image encoder to OpenAI CLIP, which still loads pretrained visual weights:

```yaml
model:
  branch2d:
    image_encoder: clip_vitl14
    pretrained: true
    clip_download_root: weights/clip
    freeze_image_encoder: true
```

This is a fallback for getting a pretrained RGB image VFM running. The default
formal config uses `dinov2_vitl14`.

Optional weights not used by the current config:

- SAM `sam_vit_l_0b3195.pth`: only needed if a later config enables SAM.
- Depth Anything can be disabled by setting `enable_depth_vfm: false`, in which
  case the sparse LiDAR depth adapter is used and no Depth Anything checkpoint
  is needed.
- Vireo task checkpoints are intentionally not needed and should not be loaded
  for this project.

The current vKITTI -> SemanticKITTI config enables Depth Anything V2 Small as
the 2D dense-depth VFM:

```bash
cd /home/zhangshuai/Sysu_4T/houych/PointCLIP-DAG
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
python pointclip_dag/scripts/prepare_weights.py \
  --skip-clip \
  --depth-anything-v2 \
  --depth-anything-encoder vits \
  --depth-anything-dir weights/depth_anything_v2
```

The config expects:

```text
third_party/Depth-Anything-V2/
weights/clip/ViT-L-14.pt
weights/dinov2/dinov2_vitl14_pretrain_torch19.pth
weights/depth_anything_v2/depth_anything_v2_vits.pth
```

Depth Anything V2 Small is about 99 MB and uses the Apache-2.0 license. Base
and Large are also supported by the code path, but the official Large checkpoint
is about 1.34 GB and uses CC-BY-NC-4.0.

## Data Paths

The experiment yaml files currently point to:

- `/home/zhangshuai/Sysu_4T/houych/VirtualKITTI`
- `/home/zhangshuai/Sysu_4T/houych/SemanticKITTI`
- `/home/zhangshuai/Sysu_4T/houych/nuscenes`

UniDSeg loaders expect preprocessed pickle files. If your pkl files are stored
elsewhere, edit only `data.*.kwargs.preprocess_dir` in the yaml files.

On this machine the vKITTI -> SemanticKITTI required files are present:

- `/home/zhangshuai/Sysu_4T/houych/VirtualKITTI/preprocess/train.pkl`
- `/home/zhangshuai/Sysu_4T/houych/SemanticKITTI/preprocess/val.pkl`
- `/home/zhangshuai/Sysu_4T/houych/SemanticKITTI/preprocess/test.pkl`

Note: `VirtualKITTI/preprocess/train.pkl` is about 19 GB, and UniDSeg loads the
pickle into memory. Run training on a node with enough free RAM.

## Train-Ready Check

Fast check without loading the 19 GB training pickle:

```bash
/home/zhangshuai/anaconda3/envs/UniDSeg/bin/python \
  pointclip_dag/scripts/check_train_ready.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --check-model
```

Optional first-batch check, which loads the large vKITTI pickle and can take a
long time:

```bash
/home/zhangshuai/anaconda3/envs/UniDSeg/bin/python \
  pointclip_dag/scripts/check_train_ready.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --check-batch
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhangshuai/anaconda3/envs/UniDSeg/bin/python \
  tools/train.py --config configs/experiments/vkitti_to_skitti.yaml
```

The vKITTI -> SemanticKITTI config is epoch-driven for a formal run:
`epochs: 200` and `max_iters: 0`, so training runs full epochs instead of being
cut off by an iteration cap. The terminal shows a tqdm progress bar with total
loss, 3D CE, 2D CE, feature distillation, KL, and learning rate.

Later nuScenes DG experiments:

```bash
CUDA_VISIBLE_DEVICES=0 /home/zhangshuai/anaconda3/envs/UniDSeg/bin/python \
  tools/train.py --config configs/experiments/nuscenes_day_to_night.yaml

CUDA_VISIBLE_DEVICES=0 /home/zhangshuai/anaconda3/envs/UniDSeg/bin/python \
  tools/train.py --config configs/experiments/nuscenes_usa_to_singapore.yaml
```

Outputs are written to:

```text
runs/<experiment_name>/<YYYYmmdd_HHMMSS>/
  config.yaml
  logs/train.log
  logs/loss_history.csv
  logs/loss_curves.png
  logs/loss_curves/loss.png
  logs/loss_curves/loss_3d_ce.png
  logs/loss_curves/loss_2d_ce.png
  logs/loss_curves/loss_feat.png
  logs/loss_curves/loss_kl.png
  checkpoints/
  eval/
```

Timestamped run directories are enabled by default through
`output.timestamp_subdir: true`, so a new training command will not overwrite
previous checkpoints or metrics. To resume a checkpoint, pass `--resume`; the
trainer writes back into that checkpoint's run directory. To force a specific
output directory, pass `--run-dir`.

The loss CSV also records 2D projected diagnostics:

- `metric_2d_projected_acc`
- `metric_2d_projected_miou`
- `metric_valid_projected_ratio`
- `metric_ignored_projected_label_ratio`

## Evaluation

```bash
python tools/eval.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --ckpt runs/vkitti_to_skitti_ov_dg/<YYYYmmdd_HHMMSS>/checkpoints/best.pth
```

By default, evaluation writes metrics next to the checkpoint:
`runs/<experiment_name>/<YYYYmmdd_HHMMSS>/eval/`. Use `--out-dir` only when you
want a separate evaluation directory.

Evaluation rebuilds text embeddings from `vocabulary.eval_vocab_path`, so the
evaluation vocabulary can differ from the training vocabulary as long as dataset
labels are mapped to the desired train IDs. A third
`semantic_probe_vocab_path` can hold richer text probes; it is not used for
standard GT mIoU.

Metrics saved in `eval/metrics.json`:

- all mIoU
- seen mIoU
- unseen mIoU
- per-class IoU
- 3D branch mIoU
- 2D projected-point mIoU
- 2D/3D ensemble mIoU
- 2D projected prediction / GT histograms
- valid projected point ratio
- ignored projected label ratio
- present-class mIoU and seen/unseen present mIoU
- semantic similarity score, near-miss semantic accuracy, text-alignment margin
- 2D/3D agreement and KL diagnostics
- prompt consistency diagnostics
- ensemble alpha sweep over configured 2D/3D weights
- confusion matrix

Evaluation also writes `eval/metrics_summary.txt`. This keeps the full JSON
untouched and adds readable tables for:

- training vocabulary labels
- evaluation vocabulary labels
- per-label `iou3d`, `iou2d`, best ensemble IoU, best ensemble alpha, GT count,
  and seen/unseen flag
- global ensemble alpha sweep

## Vocabulary

Training and evaluation classes are controlled by yaml files. To customize the
classes for an experiment, create or edit a vocab yaml and point the experiment
config to it:

```yaml
vocabulary:
  train_vocab_path: configs/vocab/vkitti_skitti/vkitti_raw_train.yaml
  eval_vocab_path: configs/vocab/vkitti_skitti/semantic_kitti_gt_aligned_19.yaml
  semantic_probe_vocab_path: configs/vocab/vkitti_skitti/semantic_probe_extended.yaml
```

Each vocabulary yaml defines class text, aliases, optional dataset label IDs,
and seen/unseen membership:

```yaml
prompt_templates:
  - "a photo of a {}."
  - "a point cloud of a {}."
classes:
  - name: car
    train_id: 11
    aliases: ["vehicle car", "automobile"]
    seen: true
  - name: traffic light
    train_id: null
    aliases: ["signal light"]
    seen: false
```

Multiple aliases and prompt templates are encoded and averaged into one
normalized class text embedding.

`train_id` maps a text class to the raw dataset GT label for CE/mIoU. It is not
the logits column index. PointCLIP-DAG maps raw dataset labels to active
vocabulary columns at runtime using `dataset_name + vocab`.

- In a training vocab, classes with `train_id` are supervised by CE.
- Labels not present in the active training vocab are ignored for CE.
- In an eval vocab, classes with `train_id` can contribute to mIoU.
- Classes with `train_id: null` are valid open-vocabulary text queries, but do
  not contribute to mIoU because there is no GT label to compare against.

Validate a custom yaml before running:

```bash
python pointclip_dag/scripts/check_vocab.py configs/vocab/vkitti_skitti/semantic_kitti_gt_aligned_19.yaml
```

List all available vocab banks:

```bash
python pointclip_dag/scripts/list_vocabs.py
python pointclip_dag/scripts/list_vocabs.py --task vkitti_skitti
```

UniDSeg six-class merged labels are not the PointCLIP-DAG category limit. They
are only a legacy closed-set baseline. The default vKITTI -> SemanticKITTI
experiment uses raw label spaces:

- Train vocabulary:
  `configs/vocab/vkitti_skitti/vkitti_raw_train.yaml`
  contains 12 VirtualKITTI raw supervised classes.
- Eval vocabulary:
  `configs/vocab/vkitti_skitti/semantic_kitti_gt_aligned_19.yaml`
  contains the GT-aligned SemanticKITTI eval classes. `bus` is not evaluated as
  an independent class; it is an alias of `other vehicle` because the current
  SemanticKITTI raw mapping used here does not provide bus GT as a separate
  present class.
- Semantic probe vocabulary:
  `configs/vocab/vkitti_skitti/semantic_probe_extended.yaml`
  contains extra driving-scene text queries for semantic diagnostics, not
  standard mIoU.

Changing either yaml changes the dynamic output dimension `K`; no checkpoint
classification head depends on those class counts.

For quick ad-hoc inference, evaluation can still bypass yaml with `--classes`:

```bash
python tools/eval.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --ckpt runs/vkitti_to_skitti_ov_dg/checkpoints/best.pth \
  --classes "car,road,pole,traffic light,sidewalk,tree" \
  --save-predictions
```

Classes passed through `--classes` are pure text queries. They do not need to be
known during training and do not need a fixed classifier head. Because they have
no dataset `train_id`, this mode is prediction-oriented; mIoU is meaningful only
for classes that have a label id in a yaml vocabulary.

For SemanticKITTI raw-class evaluation, use the same experiment config and
override the eval vocabulary if needed:

```bash
python tools/eval.py \
  --config configs/experiments/vkitti_to_skitti.yaml \
  --ckpt runs/vkitti_to_skitti_ov_dg/checkpoints/best.pth \
  --eval-vocab configs/vocab/vkitti_skitti/semantic_kitti_gt_aligned_19.yaml
```

This uses `SemanticKITTIRawSCN`, keeps raw SemanticKITTI label ids such as
`car=10`, `road=40`, `pole=80`, and evaluates a GT-aligned text vocabulary instead
of the UniDSeg 6-class merged target labels.

Semantic consistency metrics are auxiliary diagnostics. Standard GT-based mIoU
remains the primary segmentation metric; semantic similarity and near-miss
scores are intended to show whether open-vocabulary errors are semantically
nearby or arbitrary.

## Smoke Test

This runs one synthetic batch through data, model, loss, and backward. It uses
hash text embeddings and an MLP 3D fallback, so it does not need real datasets
or spconv, but it still requires PyTorch.

```bash
python pointclip_dag/scripts/smoke_test.py
```

## Version Control

The repository is initialized on the `main` branch. Large local artifacts are
ignored by `.gitignore`, including `weights/`, `runs/`, `third_party/`, and
dataset files.

To create a GitHub repository from this local version, install/login GitHub CLI
once and run:

```bash
gh auth login
VISIBILITY=private ./scripts/create_github_repo.sh PointCLIP-DAG
```

Use `VISIBILITY=public` only when you are ready to publish the code. The script
does not upload local pretrained weights or training outputs.

## Core Open-Vocabulary Contract

The model returns:

```python
{
    "text_embeddings": ...,   # [K, D]
    "z3d": ...,               # [N, D]
    "logits3d": ...,          # [N, K]
    "z2d_points": ...,        # [N_valid, D]
    "logits2d_points": ...,   # [N_valid, K]
    "valid_point_mask": ...,  # [N]
    "point_labels": ...,      # [N]
    "point_to_image_xy": ..., # [N_valid, 2]
}
```

All embeddings are L2-normalized. Classification is always:

```python
logits = logit_scale * embedding @ text_embeddings.T
```

No fixed class Linear head is used.
