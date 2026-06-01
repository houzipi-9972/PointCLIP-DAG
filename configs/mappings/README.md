PointCLIP-DAG label mapping files keep dataset labels separate from open vocabulary
columns.

Mapping chain:

1. dataset raw label id -> canonical class name
2. canonical class name -> vocab class name
3. vocab class name -> dynamic vocabulary column id

The final vocabulary column id is always determined by the active vocab yaml.
These mapping files never define model output classes and never create a fixed
classifier head.

For a new experiment, create one task file under `configs/mappings/tasks/` and
point the experiment yaml to it with:

```yaml
mapping:
  task_mapping_path: configs/mappings/tasks/vkitti_to_skitti.yaml
```

Use `tools/check_mapping.py` before training to verify train/eval/probe coverage.
