# Reproduction workflow scripts

| Directory | Purpose |
|---|---|
| `serving/` | LingBot model serving and inference checks |
| `eval/` | RoboTwin evaluation, paired fixed-seed evaluation, four-GPU parallel evaluation, optional video, and idle-GPU discovery |
| `checkpoints/` | LoRA and full-parameter SFT checkpoint conversion |
| `training/` | Full-parameter SFT and single-task LoRA launchers |
| `data/` | LeRobot data conversion and host-side preparation of the external-data mount |

Training YAML files live under `configs/lingbot_vla_v2_6b_robotwin/`. The Dockerfile copies runtime helpers into `/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/` in one `COPY` step. The root `.dockerignore` excludes the two `data/` scripts that run only on the host. The LoRA and SFT notebooks call the container's helpers and coordinate training, conversion, and evaluation in their own cells.

Prepare the default image's mounted data on the host:

```bash
bash scripts/data/prepare_external_data.sh --dry-run
bash scripts/data/prepare_external_data.sh
```

The script prepares `/models/robotwin-persistent` directly on the host: it downloads simulation assets, 50 clean demonstration tasks, and base models; converts missing tasks to LeRobot v3; validates the result; and writes the training manifest. When Python dependencies are missing, it creates isolated environments under `.preparation_runtime/` in the target directory and keeps caches and conversion logs there. It does not require a running container or GPU evaluation.
