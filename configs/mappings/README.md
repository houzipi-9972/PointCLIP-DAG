PointCLIP-DAG label mapping files keep dataset labels separate from open vocabulary
columns.

Mapping chain:

1. dataset raw label id -> canonical class name
2. canonical class name -> vocab class name
3. vocab class name -> dynamic vocabulary column id

The final vocabulary column id is always determined by the active vocab yaml.
These mapping files never define model output classes and never create a fixed
classifier head.

For a new experiment, create one self-contained task file under
`configs/mappings/`. It should contain the source/target raw labels and their
maps to `train_vocab`, `eval_vocab`, and optional `semantic_probe_vocab` in the
same yaml. Point the experiment yaml to it with:

```yaml
mapping:
  task_mapping_path: configs/mappings/vkitti_to_skitti.yaml
```

Before editing a mapping, inspect the local raw labels instead of guessing:

```bash
python tools/inspect_raw_labels.py --dataset semantic_kitti --root /path/to/SemanticKITTI --limit 200
python tools/inspect_raw_labels.py --dataset virtual_kitti --root /path/to/VirtualKITTI
python tools/inspect_raw_labels.py --dataset nuscenes --root /path/to/nuscenes --limit 200
```

Use `tools/check_mapping.py` before training to verify train/eval/probe coverage.
