# LingBot-VLA-v2-6B × RoboTwin ROCm Reproduction Guide

[Chinese (Simplified)](./Reproduce_Guide.zh-CN.md)

## Pinned versions

| Component | Version or commit |
|---|---|
| Base image | `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| Default image | `robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| FlashAttention2 | Dao-AILab `9c6d9297e5d43653c2ea5837ed9e0d10f76ed231` (PR #2900) |
| RoboTwin | `e154702de28cac4b9bfe0299044303f507236bb3` |
| RoboTwin submodule XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| SAPIEN | `3.0.0b1` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` |
| Official base-model revision | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` |
| Qwen configuration revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| MoGe-2 ViT-B normal revision | `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368` |
| RoboTwin 2.0 data revision | `a967b852afa21a9cbf19a198f7e653109042e87c` |
| PyTorch / ROCm | 2.9.1 / 7.2.1 |
| LeRobot | 0.6.0 |

RoboTwin `e154702...` fixes valid-episode counting when seeds are unstable. The image uses this commit and its pinned XPolicyLab submodule. A compatibility patch connects RoboTwin's evaluation script to the LingBot model service.

On AMD, disable CuRobo and use MPLib for end-effector planning. Label closed-loop results with `ROCm + MPLib + expert_check=true`; do not compare them directly with CUDA/CuRobo results.

Use `HIP_VISIBLE_DEVICES` to select a ROCm compute device and `DRI_PRIME` to select a SAPIEN render device. The evaluation scripts set `DRI_PRIME` automatically. Do not also set `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, or `GPU_DEVICE_ORDINAL`.

Single-task and multi-task evaluation both use RoboTwin's `scripts/eval_policy_xpolicylab.py`. The multi-task `run_robotwin_benchmark.py` starts model services and assigns GPUs. FP32 inference and expert-seed filtering are enabled by default; task configuration chooses the instruction category. Formal evaluation runs 100 valid rollouts per task. Use `--episodes 10` for a short check. For example, inside the container:

```bash
python /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --model-path /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b \
  --task-config demo_clean --episodes 100 --gpu-count 4
```

Record RoboTwin `e154702...`, SAPIEN `3.0.0b1`, ROCm/MPLib, and the evaluation configuration with results. Check those conditions before comparing against results from older LingBot clients.

# Part 1: Using the built Docker image

The default image pins RoboTwin, XPolicyLab, and LingBot source revisions. Simulation assets, models, and training data stay on the host under `/models/robotwin-persistent`; mount `/models` when starting the container.

The running path in the image is:

```text
/RoboTwin                   source tree and default working directory
/RoboTwin/data/demo_clean   extracted demonstrations for 50 tasks
/RoboTwin/data/lerobot      converted LeRobot v3 datasets
/workspace/runtime          logs, evaluation results, and training checkpoints
/opt/robotwin-env           Python environment for inference, training, evaluation, LeRobot v3, and Jupyter
```

The `/workspace` persistent volume on the referenced Radeon Cloud Global setup is 100 GB. Before training or evaluation, check free space on both the root disk and persistent workspace, along with the size of runtime outputs:

```bash
df -h / /workspace
du -d1 -h /workspace/runtime/outputs
```

Put large temporary compilation files, caches, or intermediate results on the root disk to reduce pressure on `/workspace`. The root disk is generally lost when the instance is destroyed. Copy final checkpoints, logs, evaluation results, and models you need to keep to persistent `/workspace`, and verify the copy before destroying the instance.

## 1. Build the image

The image does not include large assets, datasets, or models. Prepare them on the host under:

```text
/models/robotwin-persistent/
├── assets/       # RoboTwin simulation assets
├── data/         # demo_clean, lerobot, and the training manifest
└── models/       # LingBot-VLA models and Qwen tokenizer/configuration
```

Build command:

```bash
chmod +x docker/external-data/build.sh
./docker/external-data/build.sh
```

Prepare mounted simulation assets, demonstrations for 50 tasks, and base models on the host. Data preparation does not require the image and can run before or after the image build:

```bash
bash scripts/data/prepare_external_data.sh
```

The script downloads pinned revisions, installs Python dependencies as needed, and converts demonstrations to LeRobot v3. It writes results to `/models/robotwin-persistent/` and keeps its source, isolated environments, caches, and conversion logs under `.preparation_runtime/`. Run it with `--dry-run` to inspect the steps without downloading anything.

The default image tag is `robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`.

In the container, `assets`, `data`, and `models` are symbolic links to the mounted tree, so subsequent commands use the standard RoboTwin paths. The entrypoint checks the assets, the 50-task training manifest, and the model index, and reports any missing path.

To override the image name or pip source:

```bash
IMAGE_NAME=my-registry/robotwin-lingbot:v1 \
PIP_INDEX_URL=https://pypi.org/simple \
./docker/external-data/build.sh
```

## 2. Start the container

### 2.1 Local Docker

Set default image and data mount:

```bash
IMAGE=robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

Start the container:

```bash
mkdir -p /workspace/robotwin-runtime

docker run --name robotwin-lingbot-vla-v2 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=32g \
  --security-opt seccomp=unconfined \
  --cap-add=SYS_PTRACE \
  --network=host \
  -v /models:/models \
  -v /workspace/robotwin-runtime:/workspace/runtime \
  -it "$IMAGE" \
  bash
```

`/workspace/runtime` is a normal directory. The bind mount preserves evaluation results, training checkpoints, logs, and the Hugging Face cache after the container is removed. Omit that `-v` option if persistence is unnecessary. Manage permissions on the host directory yourself.

Re-enter an existing container:

```bash
docker start robotwin-lingbot-vla-v2
docker exec -it robotwin-lingbot-vla-v2 bash
```

### 2.2 Radeon Cloud

Validate the image on [Radeon Cloud Global](https://radeon-global.anruicloud.com/). For platform operations, see the [Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09/blob/main/Radeon-Cloud-User-Guide/README.md). Select the **default image** built and uploaded from this project:

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

1. Sign in to Radeon Cloud. If it opens the **Classic** interface, click **Switch to the new design** at the lower right to use the **New** interface described here. If you do not have an SSH key, generate one with `ssh-keygen -t ed25519`, open **Settings → New SSH Key**, and paste `~/.ssh/id_ed25519.pub`. Upload only the `.pub` public key, never the private key.
2. On the instance page, click **Customize** and select **4 GPUs** or **8 GPUs**. Choose **robotwin** under **Image**, the competition's **Dev** resource pool, **Persistent /workspace** under **Workspace Storage**, and **Devzone** under **Mount a model**. Without Devzone, the instance will lack the image's required data.
3. **Persistent /workspace** preserves checkpoints, logs, and evaluation results. **Devzone** mounts prepared content at `/models` automatically; its required tree is `/models/robotwin-persistent`:

Check free `/workspace` space before and after each run. LoRA and full-SFT checkpoints can take tens of gigabytes. Remove obsolete intermediates when appropriate, and copy any results kept temporarily on the root disk to persistent `/workspace` before destroying the instance.

   ```text
   /models/robotwin-persistent/
   ├── assets/
   ├── data/
   └── models/
   ```

4. Start the configured instance and wait for **Your workspace is ready**. Click **Open Notebook** to enter JupyterLab. Once the instance shows **Instance running**, you can also copy the SSH host, port, and user from its SSH panel and connect from a local terminal:

   ```bash
   ssh <user>@<host> -p <port>
   ```

Radeon Cloud's JupyterLab file browser opens `/workspace` by default. Open a terminal in JupyterLab and run:

   ```bash
   ln -sfn /RoboTwin /workspace/RoboTwin
   ```

Refresh the file browser and open the `RoboTwin` link. It points to the image's `/RoboTwin` source tree without copying it. The image also contains an interactive entry point:

   ```text
   /RoboTwin/notebooks/RoboTwin_ROCm_Reproduction.ipynb
   ```

Open this notebook in JupyterLab for mount and GPU checks; model serving on port 13400; a 10-episode `adjust_bottle` closed-loop evaluation; and optional clean/randomized evaluation of 100 tasks × 10 episodes on four or eight GPUs. Training cells cover four- or eight-GPU LoRA fine-tuning and merging, full-parameter SFT and DCP conversion, and restarting the merged model on port 13400 for closed-loop evaluation. The closed-loop benchmark and LoRA cells default to eight GPUs; adjust their GPU settings for four-GPU instances. Long GPU jobs do not start automatically: review their settings and run the relevant cells deliberately.
5. After entering the instance, first check the current directory and background mount:

   ```bash
   pwd
   findmnt -T /models/robotwin-persistent
   ls -l /RoboTwin/assets \
     /RoboTwin/data \
     /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models
   ```

`/RoboTwin` is already in the image; do not clone or create it again. Its asset, data, and model paths are symbolic links to the mounted tree.
6. Do not run a second Docker daemon inside the cloud instance. Along with the checks in Section 3, confirm that AMD GPUs are exposed:

   ```bash
   ls -l /dev/kfd /dev/dri
   rocminfo | head
   ```

## 3. Environment and image content inspection

The Dockerfile sets runtime variables globally with `ENV`. The container entrypoint also loads `/usr/local/bin/robotwin-env`, and login shells can load the same settings through `/etc/profile.d/robotwin-env.sh`:

- `AITER_TRITON_ONLY=1`: Let AITER use the Triton operator path supported by the current environment.
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`: Selects the AMD Triton FlashAttention2 implementation.
- `PYTHONPATH=/opt/aiter`: Let Python find the AITER source code.
- `TMPDIR=/workspace/runtime/tmp`: Directs runtime temporary files to this directory.
- `HF_HOME=/workspace/runtime/.cache/huggingface`: Sets the Hugging Face cache directory.

Check the current values in the container:

```bash
env | grep -E '^(AITER_TRITON_ONLY|FLASH_ATTENTION_TRITON_AMD_ENABLE|PYTHONPATH|TMPDIR|HF_HOME)='
```

```bash
pwd
# /RoboTwin

git rev-parse HEAD
git -C XPolicyLab rev-parse HEAD
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD

/opt/robotwin-env/bin/python - <<'PY'
import aiter, flash_attn, torch, triton
import open3d, sapien, mplib, lerobot
print(torch.__version__, torch.version.hip)
print("Triton:", triton.__version__, "FlashAttention2:", flash_attn.__version__)
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
assert flash_attn.__version__ == "2.8.4"
x = torch.randn(1024, 1024, device="cuda")
print((x @ x).shape)
PY
```

Check that all three Git commits match the pinned revisions at the start of this guide.

## 4. Check the mounted training data

The host preparation script downloads the pinned `demo_clean.zip` archives for 50 tasks and converts them into 50 independent `<task>_joint_v30` LeRobot datasets. After mounting the data and starting the container, check:

```bash
test "$(find /RoboTwin/data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50
test "$(find /RoboTwin/data/lerobot -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50
test -f /RoboTwin/data/robotwin_demo_clean_joint_v30.txt
test "$(wc -l </RoboTwin/data/robotwin_demo_clean_joint_v30.txt)" -eq 50
test -z "$(find /RoboTwin/data -type f -name '*.zip' -print -quit)"
du -sh /RoboTwin/data/demo_clean
du -sch /RoboTwin/data/lerobot/* | tail -1
```

`robotwin_demo_clean_joint_v30.txt` is a manifest listing paths to 50 LeRobot datasets, not a dataset directory. Individual task directories use the `<task>_joint_v30` form, such as `adjust_bottle_joint_v30`.

The RoboTwin evaluator writes per-episode results to `/RoboTwin/eval_result`, which the image links to `/workspace/runtime/eval_result`. Training checkpoints and logs also go under `/workspace/runtime`. For local Docker, mount a host directory there to preserve outputs; otherwise they remain in the container's writable layer and disappear when it is removed. Inputs come from the `/models` mount, so retain the host's `/models/robotwin-persistent` tree.

## 5. Official model inference

The default model is the official `robbyant/lingbot-vla-v2-6b` base checkpoint. Serving, closed-loop evaluation, and training initialization use it unless a different checkpoint is passed explicitly. The RoboTwin-trained checkpoint is used only in the post-training validation commands that select it.

Terminal 1:

```bash
cd /RoboTwin
mkdir -p /workspace/runtime/outputs/{logs,benchmarks}

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/official_server.log False
```

Terminal 2:

```bash
cd /RoboTwin
until curl -fsS http://127.0.0.1:13400/healthz; do sleep 2; done

source /opt/robotwin-env/bin/activate
export PYTHONPATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2:/RoboTwin

python experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/benchmark_official_inference.py \
  --port 13400 \
  --repeats 3 \
  --batch-sizes 1 2 4 \
  --output /workspace/runtime/outputs/benchmarks/official_inference.json
```

The model service's `USE_LENGTH` controls action sequence length. With the current default of 50, a single response is usually shaped `[50, 14]`; check `action_shape` in the result for the actual shape. The first request includes model warm-up, so use later requests for performance comparisons.

## 6. RoboTwin closed-loop evaluation

The official `robbyant/lingbot-vla-v2-6b` base checkpoint is useful for checking serving, rendering, planning, and evaluation. Its RoboTwin success rate may be low; one failure does not establish a deployment problem. For task performance, you can also evaluate LingBot's official RoboTwin-trained checkpoint:

- [Hugging Face: robbyant/lingbot-vla-v2-6b-robotwin](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)
- [Official LingBot-VLA-v2 repository](https://github.com/robbyant/lingbot-vla-v2)

The base-checkpoint evaluation comes first below, followed by the corresponding RoboTwin-checkpoint command. Reproduction or competition training must start from the base checkpoint, not the already trained RoboTwin checkpoint.

The commands below disable video by default to reduce frame handling, FFmpeg encoding, and disk writes.

**Single-task evaluation (`scripts/eval_policy_xpolicylab.py`)** runs multiple episodes for one task. The examples use `--additional_info eval_video_log=false`; change it to `--additional_info eval_video_log=true` to save video.

**Multi-task benchmark (`experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py`)** uses `--no-video` in the examples; replace it with `--video` to save video. Removing `--no-video` alone leaves the script's no-video default. The LoRA and full-SFT evaluations in Sections 7 and 8 follow the same rules.

Keep the model server running and evaluate 10 `adjust_bottle` episodes in another terminal:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

python scripts/eval_policy_xpolicylab.py \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

To evaluate the official RoboTwin-trained checkpoint, stop the base-model server first, then start the RoboTwin-checkpoint server:

```bash
cd /RoboTwin

ROBOTWIN_CHECKPOINT_ROOT=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b-robotwin
ROBOTWIN_CHECKPOINT="$ROBOTWIN_CHECKPOINT_ROOT/checkpoints/global_step_50000/hf_ckpt"
export LINGBOTVLA_TRAINING_CONFIG="$ROBOTWIN_CHECKPOINT_ROOT/lingbotvla_cli.yaml"

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/robotwin_checkpoint_server.log False \
  "$ROBOTWIN_CHECKPOINT"
```

`ROBOTWIN_CHECKPOINT_ROOT` contains the checkpoint's training YAML; weights are under `checkpoints/global_step_50000/hf_ckpt/`. The image's `models/` path is a symbolic link to `/models/robotwin-persistent/models/`. The launcher honors an exported `LINGBOTVLA_TRAINING_CONFIG`. Without it, the launcher uses this repository's `/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml` instead of the checkpoint's YAML; the model and data settings differ.

Then evaluate the checkpoint with the same single-task closed-loop parameters used for the base model:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

python scripts/eval_policy_xpolicylab.py \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

You can run the same parameters through RoboTwin's wrapper; it still calls `scripts/eval_policy_xpolicylab.py`:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

bash scripts/eval_policy.sh \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

Evaluation results go to `/workspace/runtime/eval_result`. For an initial environment check, temporarily use `--test_num 1`; switch to `--test_num 10` after confirming rendering, MPLib planning, and WebSocket communication. Mount `/workspace/runtime` to preserve results on the host; otherwise they remain only in the container.

For the 50 tasks in RoboTwin's `/RoboTwin/env_cfg/eval/all_tasks.yml`, the script runs both `demo_clean` and `demo_randomized`. It starts four or eight model services with the same number of evaluation workers. Each GPU hosts one model service and one simulator. A dynamic queue assigns the next task to whichever GPU finishes first. Check that ports `13400`–`13403` are free for four GPUs, or `13400`–`13407` for eight. By default, the script reads each HIP GPU's PCI address and sets the matching `DRI_PRIME`. To override detection, set `ROBOTWIN_DRI_PRIME_MAP` in HIP GPU order or pass `--dri-prime-map`.

Four-GPU run:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 4 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --run-name both100x10_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

The script:

- Starts model services on GPUs 0–3 using ports `13400`–`13403`;
- Runs 50 clean and 50 randomized tasks, 10 episodes each, for 1,000 episodes total;
- Writes per-task logs, timing, failures, and completion markers under `/workspace/runtime/outputs/both100x10_4gpu`;
- Skips completed tasks using `done/<task_config>__<task>.done`, so `--resume` can continue an interrupted run;
- Summarizes success rates after completion and stops the model services it started.

For eight GPUs, change the GPU count and run name:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 8 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --run-name both100x10_8gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

For the same four- or eight-GPU benchmark with the official RoboTwin-trained checkpoint, point `LINGBOTVLA_TRAINING_CONFIG` to its YAML under the checkpoint root and pass `checkpoints/global_step_50000/hf_ckpt` with `--model-path`. If that variable is omitted, the launcher falls back to this repository's training YAML. Ensure the selected ports are free before starting.

Four-GPU RoboTwin-checkpoint benchmark:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

ROBOTWIN_CHECKPOINT_ROOT=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b-robotwin
ROBOTWIN_CHECKPOINT="$ROBOTWIN_CHECKPOINT_ROOT/checkpoints/global_step_50000/hf_ckpt"
export LINGBOTVLA_TRAINING_CONFIG="$ROBOTWIN_CHECKPOINT_ROOT/lingbotvla_cli.yaml"

python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 4 \
  --episodes 10 \
  --model-path "$ROBOTWIN_CHECKPOINT" \
  --expert-check \
  --no-video \
  --run-name robotwin_checkpoint_both100x10_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

For eight GPUs, use:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 8 \
  --episodes 10 \
  --model-path "$ROBOTWIN_CHECKPOINT" \
  --expert-check \
  --no-video \
  --run-name robotwin_checkpoint_both100x10_8gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

Measured wall times for clean, randomized, and combined runs:

| Machine | Configuration | Completed | Wall time |
|---|---|---:|---:|
| 4× W7900 | clean | 50 tasks / 500 episodes | 08:26:41 |
| 4× W7900 | randomized | 50 tasks / 500 episodes | 10:29:07 |
| 4× W7900 | total | 100 tasks / 1,000 episodes | 18:55:48 |
| 8× W7900 | clean | 50 tasks / 500 episodes | 06:34:23 |
| 8× W7900 | randomized | 50 tasks / 500 episodes | 08:37:58 |
| 8× W7900 | total | 100 tasks / 1,000 episodes | 15:12:21 |

## 7. LoRA training, checkpoint merging, and evaluation

### 7.1 LoRA training

Four-GPU training:

```bash
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
source /opt/robotwin-env/bin/activate
mkdir -p /workspace/runtime/outputs/logs

export HIP_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 4 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 4 \
  --train.max_steps 100 \
  --train.save_steps 100 \
  --train.output_dir /workspace/runtime/outputs/lora_100steps_4gpu \
  2>&1 | tee /workspace/runtime/outputs/logs/lora_100steps_4gpu.log
```

Eight-GPU training:

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=8 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 8 \
  --train.gradient_accumulation_steps 1 \
  --train.global_batch_size 8 \
  --train.max_steps 100 \
  --train.save_steps 100 \
  --train.output_dir /workspace/runtime/outputs/lora_100steps_8gpu \
  2>&1 | tee /workspace/runtime/outputs/logs/lora_100steps_8gpu.log
```

Run distributed training through the active Python environment (`python -m torch.distributed.run`), since a standalone `torchrun` on `PATH` may use a different interpreter. LoRA training uses PyTorch Distributed Data Parallel. The number of GPUs in `HIP_VISIBLE_DEVICES` must match `--nproc-per-node`. The four-GPU and eight-GPU examples have effective global batch sizes of 4 and 8, respectively; each satisfies `global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps`. Checkpoints are written to:

```text
/workspace/runtime/outputs/lora_100steps_4gpu
/workspace/runtime/outputs/lora_100steps_8gpu
```

There are three attention configurations with similar names but different purposes:

- `model.attn_implementation: flash_attention_2`: attention in the Hugging Face/Qwen model.
- `model.vit_attn_implementation: flash_attention_2`: attention in the vision encoder.
- `train.attention_implementation: flex_cached`: joint attention between the VLM and action expert. This path uses a custom two-dimensional block mask; setting it to `flash_attention_2` raises `Invalid attention implementation`.

The YAML sets `train.max_steps: 100` to check data loading, forward and backward passes, checkpoint saving, and merging. This short run is unlikely to give representative fine-tuning results. In the notebook, change:

```python
TRAIN_STEPS = 100   # workflow validation
# TRAIN_STEPS = 1000  # initial experiments
# TRAIN_STEPS = 5000  # longer training; choose final steps using closed-loop validation
```

The notebook uses `TRAIN_STEPS` to override `--train.max_steps`, `--train.save_steps`, and `--train.output_dir`, then selects the matching `global_step_${TRAIN_STEPS}` for merging. For command-line training, add equivalent overrides to the command:

```bash
--train.max_steps 1000 \
--train.save_steps 1000 \
--train.output_dir /workspace/runtime/outputs/lora_1000steps_8gpu
```

### 7.2 Merge the LoRA checkpoint and evaluate it

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

MERGED=/workspace/runtime/outputs/lora_100steps_8gpu/merged_checkpoint/global_step_100/hf_ckpt

python experiments/lingbot_vla_v2_6b_robotwin/scripts/checkpoints/merge_lora_dcp.py \
  --checkpoint /workspace/runtime/outputs/lora_100steps_8gpu/checkpoints/global_step_100 \
  --training-output /workspace/runtime/outputs/lora_100steps_8gpu \
  --base-model /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b \
  --output "$MERGED" \
  --rank 8 \
  --alpha 16

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/merged_server.log False "$MERGED"
```

Stop the model server currently using port 13400 before starting the merged checkpoint server. In another terminal, wait for the health check:

```bash
until curl -fsS http://127.0.0.1:13400/healthz; do sleep 2; done
```

Then evaluate the merged model with the RoboTwin closed-loop entry point:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

bash scripts/eval_policy.sh \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

After the `adjust_bottle` check, you can evaluate the merged LoRA model on 100 tasks with 10 episodes each. Stop the single model server on port 13400 first: the benchmark launcher starts its own model servers. The notebook defaults to eight GPUs; command-line examples for both GPU counts follow. Four GPUs:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 4 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --model-path "$MERGED" \
  --run-name lora_100steps_both100x10_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

Eight GPUs:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 8 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --model-path "$MERGED" \
  --run-name lora_100steps_both100x10_8gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

`--resume` reads `done/<task>.done` in the run directory and skips completed tasks. To start an independent evaluation, use a new `--run-name` so old results are not mixed with new ones.

## 8. Full SFT, checkpoint conversion, and evaluation

This full-parameter SFT workflow is separate from the LoRA workflow in Section 7. The default enables depth and video teachers. SFT without teachers lacks that supervision and is not recommended for formal reproduction. The training checkpoint includes alignment heads that inference does not use; the deployment patch excludes those heads while strictly checking the remaining weights. Stop any model server occupying the training GPUs before starting.

### 8.1 Default: training with depth and video teachers

The official base checkpoint contains LingBot-Depth and DINO-VIDEO. The persistent model directory must also contain the separately downloaded `moge-2-vitb-normal/model.pt`. The commands below enable current and future depth plus the DINO-video teacher. They use AdamW and a micro batch of 16, as in the capacity measurements:

|Number of GPUs| `data_parallel_shard_size` | `micro_batch_size` | `gradient_accumulation_steps` | `global_batch_size` |
|---:|---:|---:|---:|---:|
|8 (default)| 8 | 16 | 2 | 256 |
| 4 | 4 | 16 | 4 | 256 |

Four GPUs:

```bash
cd /RoboTwin

TEACHER_MODE=full \
GPU_COUNT=4 \
OPTIMIZER=adamw \
MICRO_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
TIMEOUT_SECONDS=0 \
MAX_STEPS=10 SAVE_STEPS=10 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_full_4gpu_10steps \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

Eight GPUs:

```bash
cd /RoboTwin

TEACHER_MODE=full \
GPU_COUNT=8 \
OPTIMIZER=adamw \
MICRO_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
TIMEOUT_SECONDS=0 \
MAX_STEPS=10 SAVE_STEPS=10 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_full_8gpu_10steps \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

`TIMEOUT_SECONDS` limits the run time of a smoke test; `0` disables that limit. Set `MAX_STEPS`, `SAVE_STEPS`, and `OUTPUT_DIR` for the intended run. Setting `SAVE_STEPS` below `MAX_STEPS` saves intermediate checkpoints.

The depth/video teacher was tested up to a micro batch of 16 on both GPU counts. The notebook defaults to eight GPUs (`16 × 8 × 2 = 256`); the four-GPU setting uses `16 × 4 × 4 = 256`. The script derives the shard size and gradient accumulation from `GPU_COUNT`:

```text
use_lora=false
data_parallel_mode=fsdp2
data_parallel_replicate_size=1
data_parallel_shard_size=GPU_COUNT
micro_batch_size=16              # GPU_COUNT=4 or 8
gradient_accumulation_steps=256/(GPU_COUNT*micro_batch_size)
global_batch_size=256
enable_gradient_checkpointing=true
enable_full_shard=true
```

Capacity measurements for four or eight W7900 GPUs (48 GiB each) follow. Gradient checkpointing and FSDP2 full sharding were enabled. Step time is the mean of optimizer steps 2 and 3. Peak GPU memory was sampled once per second across model loading, forward pass, backward pass, and optimizer update. These measurements establish the capacity limit; the default remains micro batch 16 and global batch 256.

| Platform | Micro batch per GPU | Gradient accumulation | Steady-state step time | Peak GPU memory per GPU | Result |
|---|---:|---:|---:|---:|---|
| 4× W7900 48 GiB | 1 | 64 |245.580 seconds| 40.329–40.375 GiB | Pass |
| 4× W7900 48 GiB | 2 | 32 |136.748 seconds| 41.337–41.349 GiB | Pass |
| 4× W7900 48 GiB | 4 | 16 |76.010 seconds| 43.509–43.521 GiB | Pass |
| 4× W7900 48 GiB | 8 | 8 |54.031 seconds| 47.729–47.761 GiB | Pass, little headroom |
| 4× W7900 48 GiB | 16 | 4 |41.392 seconds| 47.890–47.975 GiB | Pass, close to memory limit |
| 4× W7900 48 GiB | 32 | 2 | — |47.648–47.956 GiB, leaving only 62–286 MiB| HIP OOM |
| 8× W7900 48 GiB | 1 | 32 |143.665 seconds| 27.520–27.558 GiB | Completed |
| 8× W7900 48 GiB | 2 | 16 |80.319 seconds| 27.818–27.832 GiB | Pass |
| 8× W7900 48 GiB | 4 | 8 |47.055 seconds| 29.498–29.523 GiB | Pass |
| 8× W7900 48 GiB | 8 | 4 |27.108 seconds| 33.004–33.386 GiB | Pass |
| 8× W7900 48 GiB | 16 | 2 |24.682 seconds| 41.344–41.356 GiB | Pass |
| 8× W7900 48 GiB | 32 | 1 | — |47.878–47.947 GiB, leaving only about 0.12–0.14 GiB| HIP OOM |

Memory ranges show the lowest and highest per-GPU peaks within each run. The step times average optimizer steps 2 and 3. The measurements do not change the default four- or eight-GPU training configurations.

A full AdamW DCP checkpoint takes about 70 GB across all ranks, not per GPU. It stores model weights, optimizer moments, the learning-rate scheduler, RNG state, and dataloader state. Changing the GPU count changes the number and size of shards, but does not proportionally reduce the total size. Check free space under `/workspace/runtime` before training. Once the converted Hugging Face checkpoint loads successfully, an unneeded DCP checkpoint can be removed if training will not resume from it.

### 8.2 Optional: full SFT without teachers

The same launcher supports four or eight GPUs without teachers. This omits the depth/video supervision and is not recommended for formal reproduction. Use the commands from Section 8.1 with `TEACHER_MODE=none`.

Four GPUs:

```bash
cd /RoboTwin
TEACHER_MODE=none \
GPU_COUNT=4 \
OPTIMIZER=adamw \
MICRO_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
TIMEOUT_SECONDS=0 \
MAX_STEPS=10 SAVE_STEPS=10 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_none_4gpu_10steps \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

Eight GPUs:

```bash
cd /RoboTwin
TEACHER_MODE=none \
GPU_COUNT=8 \
OPTIMIZER=adamw \
MICRO_BATCH_SIZE=16 \
GLOBAL_BATCH_SIZE=256 \
TIMEOUT_SECONDS=0 \
MAX_STEPS=10 SAVE_STEPS=10 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_none_8gpu_10steps \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

### 8.3 Selection of `enable_full_shard`

This parameter is passed to FSDP2 as `reshard_after_forward`; it does not turn FSDP on or off:

- `true`: reshard each FSDP module after the forward pass. This lowers peak memory but requires another all-gather before the backward pass. This is the script default.
- `false`: keep the module parameters until backward. This can save an all-gather and may run faster, but generally uses more peak GPU memory. FSDP2 sharding remains enabled.

The following three configurations can be used, and the first one is selected by default for official reproduction:

| Configuration | Key parameters | Purpose |
|---|---|---|
| Base full SFT | `data_parallel_shard_size=GPU_COUNT`, global batch 256, full sharding | Default configuration in this section |
| No reshard | `data_parallel_shard_size=GPU_COUNT`, global batch 256, `enable_full_shard=false` | Compare speed and memory usage with fewer all-gathers |
| FP32/future-image variant | Shard size `GPU_COUNT`, full sharding, `enable_fp32=true`, `use_future_image=true` | Check additional training switches; `align_params={}` does not enable full depth/video alignment |

All three configurations set `use_lora=false` and enable gradient checkpointing; they train all parameters. They are alternative configurations, not consecutive phases.

To compare the no-reshard setting, override one environment variable:

Four GPUs:

```bash
GPU_COUNT=4 \
ENABLE_FULL_SHARD=false \
TEACHER_MODE=full \
MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_4gpu_no_reshard_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

Eight GPUs:

```bash
GPU_COUNT=8 \
TEACHER_MODE=full \
ENABLE_FULL_SHARD=false \
MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_8gpu_no_reshard_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

### 8.4 Convert the full DCP checkpoint and evaluate it

Full SFT has no LoRA adapter. Convert the distributed DCP checkpoint to Hugging Face format:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

TRAIN_OUTPUT=/workspace/runtime/outputs/full_sft_full_8gpu_10steps
CHECKPOINT="$TRAIN_OUTPUT/checkpoints/global_step_10"
FULL_SFT_MODEL="$TRAIN_OUTPUT/merged_checkpoint/global_step_10/hf_ckpt"

python experiments/lingbot_vla_v2_6b_robotwin/scripts/checkpoints/convert_full_sft_dcp.py \
  --checkpoint "$CHECKPOINT" \
  --training-output "$TRAIN_OUTPUT" \
  --output "$FULL_SFT_MODEL"
```

Start the converted full-SFT model server on port 13400:

```bash
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/full_sft_server.log False "$FULL_SFT_MODEL"
```

In another terminal, wait for the server and run a closed-loop evaluation:

```bash
until curl -fsS http://127.0.0.1:13400/healthz; do sleep 2; done

cd /RoboTwin
source /opt/robotwin-env/bin/activate

bash scripts/eval_policy.sh \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

After the single-task check, you can evaluate the converted model on 100 tasks with 10 episodes each. Stop the single model server on port 13400 before the benchmark launcher starts its own servers. Four GPUs:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 4 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --model-path "$FULL_SFT_MODEL" \
  --run-name full_sft_100steps_both100x10_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

Eight GPUs:

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --gpu-count 8 \
  --episodes 10 \
  --expert-check \
  --no-video \
  --model-path "$FULL_SFT_MODEL" \
  --run-name full_sft_100steps_both100x10_8gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

A full evaluation takes time. With `--resume`, the launcher runs only tasks without completion markers under the same `--run-name`. Use a new run name for an independent evaluation of the same checkpoint.

# Part 2: Install from a clean base image or on a host

Use this section when installing without the project image. If you use the image from Part 1, skip these installation steps.

## 9. Basic environment

Docker:

```bash
docker run --name robotwin-lingbot-vla-v2 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --ipc=host \
  --shm-size=32g \
  --network=host \
  -v /workspace/robotwin-clean:/RoboTwin \
  -it rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  bash
```

A host installation requires working ROCm 7.2.1, PyTorch 2.9.1, and Vulkan/Mesa drivers. Run the following commands in the container or host shell.

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs curl wget ffmpeg \
  libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
  build-essential ninja-build cmake pkg-config
git lfs install
```

Check GPU access:

```bash
ls -l /dev/kfd /dev/dri
rocminfo | head
python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.device_count())
assert torch.cuda.is_available() and torch.version.hip is not None
PY
```

## 10. Check out pinned sources and apply patches

```bash
git clone --no-checkout https://github.com/RoboTwin-Platform/RoboTwin.git /RoboTwin
cd /RoboTwin
git checkout e154702de28cac4b9bfe0299044303f507236bb3
git submodule update --init --recursive
test "$(git -C XPolicyLab rev-parse HEAD)" = c37109c500be67d0dea6b36bf7337bbd26e763cd

mkdir -p experiments/lingbot_vla_v2_6b_robotwin/source
git clone https://github.com/robbyant/lingbot-vla-v2.git \
  experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
  checkout 951475ae1b1d87553e7dc47c97b53a3d695c0d13
```

Apply the three compatibility patches from this repository to RoboTwin, XPolicyLab, and LingBot-VLA-v2:

```bash
git apply /path/to/RoboTwin-radeon-cloud/patches/robotwin-rocm-reproduction.patch
git -C XPolicyLab apply \
  /path/to/RoboTwin-radeon-cloud/patches/xpolicylab-lerobot-v30.patch
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 apply \
  /path/to/RoboTwin-radeon-cloud/patches/lingbot-vla-v2-rocm.patch

REPRO_ROOT=/path/to/RoboTwin-radeon-cloud
EXP_ROOT=experiments/lingbot_vla_v2_6b_robotwin
mkdir -p "$EXP_ROOT/scripts" "$EXP_ROOT/training"
cp -a "$REPRO_ROOT/scripts/." "$EXP_ROOT/scripts/"
cp "$REPRO_ROOT/configs/lingbot_vla_v2_6b_robotwin/lingbotvla_cli.yaml" "$EXP_ROOT/training/"
```

See [patches/README.md](./patches/README.md) for the files changed and applicable revisions. In brief:

- `robotwin-rocm-reproduction.patch`: ROCm compatibility, the CuRobo fallback, MPLib planning, and the LingBot websocket evaluation protocol. It retains the pinned upstream revision's valid-episode counting and unstable-seed filtering order.
- `xpolicylab-lerobot-v30.patch`: LeRobot 0.6 data conversion and optional end-effector data format.
- `lingbot-vla-v2-rocm.patch`: ROCm/LeRobot training, inference, and action-expert LoRA compatibility.

## 11. Create the model environment

The base ROCm image provides PyTorch under `/opt/venv`. Create `/opt/robotwin-env` to match the project image. The user running these commands needs write access to `/opt`; create the directory as root or with `sudo` if necessary.

Install the pinned dependencies from the same list used by the Dockerfile:

```bash
REPRO_ROOT=/path/to/RoboTwin-radeon-cloud

/opt/venv/bin/python -m venv --system-site-packages /opt/robotwin-env
source /opt/robotwin-env/bin/activate

MODEL_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' \
  /opt/venv/lib/python3.12/site-packages \
  /opt/venv/local/lib/python3.12/dist-packages \
  /opt/venv/lib/python3/dist-packages \
  /opt/venv/lib/python3.12/dist-packages \
  > "$MODEL_SITE/rocm_image_venv.pth"

awk '/^(pip|setuptools|wheel)==/{print}' \
  "$REPRO_ROOT/docker/requirements.txt" > /tmp/robotwin-build-tools.txt
python -m pip install --no-deps -r /tmp/robotwin-build-tools.txt
python -m pip install --no-deps --no-build-isolation --prefer-binary \
  -r "$REPRO_ROOT/docker/requirements.txt"

python -m pip install --no-deps --no-build-isolation \
  -e /RoboTwin/XPolicyLab \
  -e /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
```

Do not install upstream requirements files that could replace ROCm PyTorch, Triton, or FlashAttention with CUDA/PyPI builds. The pinned packages are installed with `--no-deps` into one environment for model work and Jupyter. XPolicyLab and LingBot are editable installs. LeRobot 0.6 declares NumPy ≥2 and huggingface-hub ≥1, which conflict with this LingBot/Transformers combination. The pinned NumPy 1.26 and hub 0.34 have been used to read this project's LeRobot v3 data, but they do not satisfy LeRobot's package metadata, so `pip check` is not a valid success criterion for this environment.

### 11.1 Install ROCm FlashAttention2

Install the pinned AITER revision using the Triton included with ROCm PyTorch:

```bash
git clone https://github.com/ROCm/aiter.git /opt/aiter
git -C /opt/aiter checkout 9bab8388c35936814a659b4ebd245c491e1b940a
test "$(git -C /opt/aiter rev-parse HEAD)" = \
  9bab8388c35936814a659b4ebd245c491e1b940a

cd /opt/aiter
AITER_USE_SYSTEM_TRITON=1 \
  /opt/robotwin-env/bin/python setup.py develop
```

Install the pinned Dao-AILab FlashAttention2 commit that includes PR #2900. During installation, `AITER_TRITON_ONLY=1` uses AITER's Triton path, avoiding AOT/C++ operator loading and GPU-driver queries in a build environment without a GPU. `--no-deps` and `FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE` prevent pip from installing another AITER, Triton, or PyTorch:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git \
  /opt/flash-attention-source
git -C /opt/flash-attention-source checkout \
  9c6d9297e5d43653c2ea5837ed9e0d10f76ed231
test "$(git -C /opt/flash-attention-source rev-parse HEAD)" = \
  9c6d9297e5d43653c2ea5837ed9e0d10f76ed231

cd /opt/flash-attention-source
PYTHONPATH=/opt/aiter \
AITER_TRITON_ONLY=1 \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE \
  /opt/robotwin-env/bin/python -m pip install \
  --no-build-isolation --no-deps .
```

`AITER_USE_SYSTEM_TRITON=1` and `FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE` are installation settings. At runtime, `AITER_TRITON_ONLY=1` prevents the pinned Triton 3.5.1 setup from importing AITER Gluon modules that need Triton 3.6 or newer. Set both `AITER_TRITON_ONLY=1` and `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for inference and training. On a host installation, also add the editable AITER source at `/opt/aiter` to `PYTHONPATH`, then check imports:

```bash
export AITER_TRITON_ONLY=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONPATH=/opt/aiter${PYTHONPATH:+:${PYTHONPATH}}

/opt/robotwin-env/bin/python - <<'PY'
import aiter
import flash_attn
import torch
import triton

print("torch:", torch.__version__, "HIP:", torch.version.hip)
print("triton:", triton.__version__)
print("flash_attn:", flash_attn.__version__)
assert torch.version.hip is not None
assert flash_attn.__version__ == "2.8.4"
PY
```

Keep the Triton version paired with the base PyTorch/ROCm image; do not change it solely to satisfy an AITER version prompt. AITER may prefer Triton 3.6, or FlashAttention may report a missing `flash_attn_2_cuda` before using its AMD Triton fallback. These messages do not necessarily indicate installation failure. The model patch preserves the upstream attention default, and the image training configuration uses `flash_attention_2`. For diagnosis of a specific incompatible operator, set the corresponding experiment configuration to `eager`.

Download RoboTwin simulation assets after preparing the model environment:

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate
bash scripts/_download_assets.sh
test -f assets/objects/objaverse/list.json
```

## 12. Create a data conversion environment

Create `/opt/lerobot-env` only when converting data manually from a clean base image or host. The project image uses data converted on the host and does not contain this environment:

```bash
/opt/venv/bin/python -m venv --system-site-packages /opt/lerobot-env
source /opt/lerobot-env/bin/activate

DATA_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' \
  /opt/venv/lib/python3.12/site-packages \
  /opt/venv/local/lib/python3.12/dist-packages \
  /opt/venv/lib/python3/dist-packages \
  /opt/venv/lib/python3.12/dist-packages \
  > "$DATA_SITE/rocm_image_venv.pth"

python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip install 'lerobot[dataset]==0.6.0' 'h5py==3.14.0'
```

Check the converter imports:

```bash
cd /RoboTwin
python - <<'PY'
import h5py, cv2, numpy, pandas, pyarrow, lerobot
from XPolicyLab.utils.data_loader import load
print("data conversion imports OK")
PY
```

## 13. Download the model

```bash
source /opt/robotwin-env/bin/activate
cd /RoboTwin
MODEL_ROOT=$PWD/experiments/lingbot_vla_v2_6b_robotwin/models
mkdir -p "$MODEL_ROOT"

huggingface-cli download robbyant/lingbot-vla-v2-6b \
  --revision 11c703bf6a5c1f45b3b69168482da11fdbba53d7 \
  --local-dir "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b"

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
  --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct-config-tokenizer"

huggingface-cli download Ruicheng/moge-2-vitb-normal \
  --revision ca5f0e07ff01d3e5a364c1d954ed12ee1814b368 \
  --local-dir "$MODEL_ROOT/moge-2-vitb-normal"
```

Check the weight shards:

```bash
find "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b" -maxdepth 1 \
  -name 'model-*.safetensors' | wc -l
# expected: 6

test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/depth/model.pt"
test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/dino_video/config.yaml"
test -f "$MODEL_ROOT/moge-2-vitb-normal/model.pt"
```

The base checkpoint includes LingBot-Depth and DINO-VIDEO checkpoints/configurations but not the MoGe-2 weights. Only the Qwen tokenizer and configuration are needed; the 4B weights need not be downloaded again. Standard inference and LoRA/SFT without alignment loss do not load the teachers. Full training with depth/video teachers uses additional GPU memory and compute.

## 14. Data, inference, and training

Set the paths:

```bash
export ROBOTWIN_ROOT=/RoboTwin
export LINGBOT_VLA_SOURCE=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
export QWEN3VL_PATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer
export LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python
export HF_LEROBOT_HOME=/RoboTwin/data/lerobot
export ROBOTWIN_DISABLE_CUROBO=1
export ROBOTWIN_EE_PLANNER=mplib
export PYOPENGL_PLATFORM=egl
source /usr/local/bin/robotwin-env
```

With a manual installation, source the environment in every new shell used for inference or training. You can also copy and source `docker/robotwin-env.sh` on the host. `AITER_USE_SYSTEM_TRITON` and `FLASH_ATTENTION_USE_SYSTEM_AITER` are needed only during installation.

Download and extract the demonstration archives, then remove the ZIP files:

```bash
cd /RoboTwin
ROBOTWIN_DATA_ROOT=/RoboTwin/data \
HF_ARCHIVE_CACHE=/workspace/download-cache \
HF_REVISION=a967b852afa21a9cbf19a198f7e653109042e87c \
HF_KEEP_ARCHIVES=0 \
bash scripts/download_xpolicylab_data.sh

test "$(find data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50
test -z "$(find data -type f -name '*.zip' -print -quit)"
```

Convert all tasks in parallel using the data environment. For the image workflow, use RoboTwin's batch conversion script:

```bash
source /usr/local/bin/robotwin-env
LEROBOT_PYTHON=/opt/lerobot-env/bin/python \
CONVERSION_JOBS=8 \
bash /path/to/RoboTwin-radeon-cloud/scripts/data/convert_all_data.sh
```

The conversion script puts temporary files, logs, and caches under `/workspace/runtime` by default, avoiding a potentially small `/tmp`. Override these paths with `ROBOTWIN_RUNTIME_ROOT`, `CONVERSION_CACHE_ROOT`, or `CONVERSION_LOG_ROOT`. For example, to place the cache on another disk:

```bash
CONVERSION_CACHE_ROOT=/root/conv-cache \
CONVERSION_LOG_ROOT=/workspace/runtime/outputs/logs/robotwin-lerobot-conversion \
  bash /path/to/RoboTwin-radeon-cloud/scripts/data/convert_all_data.sh
```

During image builds, the Dockerfile points `CONVERSION_CACHE_ROOT` at a BuildKit cache mount to keep conversion cache files out of the final image layer.

Create the runtime directories, then follow Sections 5–8 of Part 1. The environment and output paths match those in the image:

```bash
mkdir -p /workspace/runtime/eval_result \
  /workspace/runtime/outputs \
  /workspace/runtime/.cache/huggingface
```

## 15. FAQ

### `ModuleNotFoundError: h5py`

This error means the manually created data conversion environment lacks a dependency. Install it with:

```bash
source /opt/lerobot-env/bin/activate
python -m pip install h5py==3.14.0
```

The project image already includes `h5py` in `/opt/robotwin-env`. This command applies only to a manually created `/opt/lerobot-env`.

### `assets/norm_stats/robotwin.json` not found

In older images, `configs/robot_configs/robotwin.yaml` contains a path relative to the LingBot-VLA repository. Starting training from `/RoboTwin` can resolve it against the wrong directory. Change to the LingBot source directory before launching training:

```bash
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
```

The current patch resolves the path relative to the LingBot-VLA repository, regardless of the working directory.

### PyTorch can't see AMD GPU

```bash
ls -l /dev/kfd /dev/dri
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

Check that the container has access to `/dev/kfd` and `/dev/dri`, and that a CUDA/PyPI PyTorch package has not replaced the ROCm build.

### SAPIEN and svulkan2 logs

Legacy code may see this when launching the SAPIEN renderer in a ROCm/AMD environment:

```text
[svulkan2] [error] CUDA Error: cudaErrorInsufficientDriver
[svulkan2] [error] Failed to initialize denoiser
```

These messages come from svulkan2 attempting to initialize its optional NVIDIA CUDA denoiser; they do not show that ROCm evaluation failed. The RoboTwin patch disables the denoiser before SAPIEN creates an engine, scene, or renderer, and sets separate core and renderer log levels:

```python
import sapien.core as sapien

sapien.set_log_level("warning")
sapien.render.set_ray_tracing_denoiser("none")
sapien.render.set_log_level("critical")
```

`sapien.set_log_level()` controls SAPIEN core and physics logs, while `sapien.render.set_log_level()` controls the `[svulkan2]` messages. Set these options before creating `sapien.Engine()`, `sapien.Scene()`, `sapien.render.RenderSystem(...)`, or `sapien.SapienRenderer()`. Do not re-enable `set_ray_tracing_denoiser("oidn")` later in `setup_scene()`.

In one `adjust_bottle` closed-loop test, the patched task succeeded (1/1), with no `cudaErrorInsufficientDriver` or `Failed to initialize denoiser` log entries. These settings disable only optional ray-tracing denoising and change log levels; Vulkan rendering and model inference remain enabled.

If the renderer still fails to initialize or the camera produces no image, run:

```bash
export PYOPENGL_PLATFORM=egl
vulkaninfo --summary
```

Check that `/dev/dri` is available inside the container and install `libvulkan1` and `mesa-vulkan-drivers`.

### Training process uses wrong Python

After activating the model environment, invoke the distributed launcher through Python:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 ...
```

### Checkpoint is not persisted

To keep checkpoints after deleting the container, mount persistent host storage at `/workspace/runtime` and set the training `output_dir` under `/workspace/runtime/outputs`. The image does not link `/workspace/runtime` to the external data directory.
