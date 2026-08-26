# LingBot-VLA-v2-6B × RoboTwin ROCm 复现指南

固定版本：

| 组件 | 版本或提交 |
|---|---|
| 基础镜像 | `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| 完整镜像 | `robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| RoboTwin | `266f3aadf505a4f7fe9af0faa41a20f5f47cd123` |
| XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` |
| 官方模型 revision | `0451855729ec904f970600e0aec8b84661423afe` |
| Qwen 配置 revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| RoboTwin2.0 数据 revision | `a967b852afa21a9cbf19a198f7e653109042e87c` |
| PyTorch / ROCm | 2.9.1 / 7.2.1 |
| LeRobot | 0.6.0 |

AMD 环境关闭 CuRobo，使用 MPLib 做末端位姿规划。闭环结果应注明
`ROCm + MPLib + expert_check=false`，不要直接与 CUDA/CuRobo 结果比较。

# 第一部分：使用构建好的 Docker 镜像

full 和 external-data 镜像包含相同的源码、兼容补丁及 Python 环境，后续推理、
评测和训练命令完全相同。两者只在大型数据的保存位置上有区别：

| 镜像 | 仿真资产、模型和训练数据 | 启动时的数据挂载 |
|---|---|---|
| full | 全部内置在镜像中 | 不需要 |
| external-data | 位于外置数据目录 | `/models` 挂载后在 `/models/robotwin-persistent` 可用 |

两个镜像中的公共运行路径为：

```text
/RoboTwin                   源码和默认工作目录
/RoboTwin/data/demo_clean   50 个任务的原始解压数据
/RoboTwin/data/lerobot      转换完成的 LeRobot v3 数据
/workspace/runtime          日志、评测结果和训练 checkpoint
/opt/robotwin-env           推理、训练和评测环境
/opt/lerobot-env            LeRobot v3 数据转换环境
```

## 1. 构建镜像

### 1.1 full 镜像

full 镜像包含仿真资产、官方模型、Qwen tokenizer/config、全部 50 个任务的解压
数据和转换完成的 LeRobot v3 数据，启动后不需要下载或转换。在仓库根目录执行：

```bash
chmod +x docker/full/build.sh
./docker/full/build.sh
```

默认生成：

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

构建使用持久 BuildKit cache 保存 apt 包、pip wheels、仿真资产压缩包、模型权重、
50 个原始数据 ZIP，以及按任务划分的 LeRobot 转换结果。这些缓存不会进入最终
镜像；不要在频繁重建期间执行 `docker builder prune`。

### 1.2 external-data 镜像

external-data 镜像不包含大型资产、数据和模型，它们必须位于宿主机：

```text
/models/robotwin-persistent/
├── assets/       # RoboTwin 仿真资产
├── data/         # demo_clean、lerobot 和训练清单
└── models/       # LingBot-VLA 模型与 Qwen tokenizer/config
```

构建命令：

```bash
chmod +x docker/external-data/build.sh
./docker/external-data/build.sh
```

默认生成：

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1-external-data
```

容器内的 `assets`、`data` 和 `models` 使用软链接指向上述挂载目录，因此后续命令
无需修改路径。entrypoint 会检查资产、50 任务训练清单和模型索引，未挂载或内容
不完整时会直接报出缺失路径。

如需覆盖镜像名或 pip 源，两个构建脚本均支持：

```bash
IMAGE_NAME=my-registry/robotwin-lingbot:v1 \
PIP_INDEX_URL=https://pypi.org/simple \
./docker/full/build.sh
```

## 2. 启动容器

### 2.1 本地 Docker

先根据镜像类型设置变量。

使用 full 镜像：

```bash
IMAGE=robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
DATA_MOUNTS=()
```

使用 external-data 镜像：

```bash
IMAGE=robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1-external-data
DATA_MOUNTS=(-v /models:/models)
```

然后使用同一条命令启动，两种镜像的容器名都固定为
`robotwin-lingbot-vla-v2`：

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
  "${DATA_MOUNTS[@]}" \
  -v /workspace/robotwin-runtime:/workspace/runtime \
  -it "$IMAGE" \
  bash
```

`/workspace/runtime` 对两种镜像都是普通目录。上述挂载用于持久化评测结果、
训练 checkpoint、日志和 Hugging Face 缓存；如果不需要在删除容器后保留输出，
可以删除该 `-v` 参数。宿主机挂载目录的权限由宿主机负责。

重新进入已有容器：

```bash
docker start robotwin-lingbot-vla-v2
docker exec -it robotwin-lingbot-vla-v2 bash
```

### 2.2 Radeon Cloud

云端入口参考 [Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/tree/main/Radeon-Cloud-User%20Guide)。
Radeon Cloud 使用本项目构建并上传到镜像仓库的 **external-data 镜像**，不是 full
镜像，也不是仅启动原始 ROCm PyTorch 基础镜像：

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1-external-data
```

1. 登录 Radeon Cloud，进入 **Profile → My Templates → Add Template**。
2. **Container Image** 填写上述 external-data 镜像在云端镜像仓库中的完整地址。
3. 平台后台已经固定挂载整个 `/models`，用户不需要手动挂载。external-data
   所需内容位于其中的 `/models/robotwin-persistent`，目录结构为：

   ```text
   /models/robotwin-persistent/
   ├── assets/
   ├── data/
   └── models/
   ```

4. Launch 后可以使用 JupyterLab Terminal；如果模板提前启用了
   **SSH Access (advanced)**，并在 Profile 中保存了公钥，也可以使用页面提供的
   host、port 和 user 通过 SSH 登录。Radeon Cloud 的 JupyterLab 文件浏览器默认
   打开 `/workspace`，因此可以先打开 Terminal（命令行终端）并执行：

   ```bash
   ln -sfn /RoboTwin /workspace/RoboTwin
   ```

   然后刷新左侧文件列表，进入 `RoboTwin` 文件夹操作。该目录是指向镜像内
   `/RoboTwin` 的软链接，不会复制源码。external-data 镜像还内置了交互式入口：

   ```text
   /RoboTwin/RoboTwin_ROCm_Reproduction.ipynb
   ```

   可在 JupyterLab 中打开该文件，依次完成挂载/GPU 检查、启动 13400 模型服务、
   `adjust_bottle` 的 10-episode 闭环评测、单卡/双卡/四卡 LoRA 微调、checkpoint
   合并，以及在同一 13400 端口重新启动合并模型并再次闭环评测。Notebook 中的
   长时间 GPU 单元不会自动执行，需要用户手动运行。
5. 进入实例后先检查当前目录及后台挂载：

   ```bash
   pwd
   findmnt -T /models/robotwin-persistent
   ls -l /RoboTwin/assets \
     /RoboTwin/data \
     /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models
   ```

   `/RoboTwin` 已经内置于 external-data 镜像，不需要重新 clone，也不
   需要手工建立该目录；资产、数据和模型由镜像中的软链接映射到后台挂载目录。
6. 云端实例中不要再启动第二层 Docker。执行以下检查以及第 3 节自检，确认实例
   已暴露 AMD GPU：

   ```bash
   ls -l /dev/kfd /dev/dri
   rocminfo | head
   ```

模型 server 与评测 client 位于同一云端实例时继续使用 `127.0.0.1:13400`。需要
从外部访问 WebSocket 时，可以按照云端指南安装 `rc-tunnel` 并暴露 server 端口；
公开地址没有自动鉴权，不应暴露无认证的管理服务。

## 3. 环境和镜像内容检查

```bash
pwd
# /RoboTwin

git rev-parse HEAD
git -C XPolicyLab rev-parse HEAD
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 rev-parse HEAD

/opt/robotwin-env/bin/python - <<'PY'
import torch, open3d, sapien, mplib, lerobot
print(torch.__version__, torch.version.hip)
print("GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device="cuda")
print((x @ x).shape)
PY
```

三个 Git commit 应依次与本文开头的固定版本一致。

## 4. 检查内置训练数据

镜像构建时下载固定 revision 的全部 50 个 `demo_clean.zip`，解压后立即删除 ZIP，
然后并行转换为 50 个独立的 `<task>_joint_v30` LeRobot 数据集。训练清单包含
全部数据集。容器启动后只需检查：

```bash
test "$(find /RoboTwin/data/demo_clean -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50
test "$(find /RoboTwin/data/lerobot -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 50
test -f /RoboTwin/data/robotwin_demo_clean_joint_v30.txt
test "$(wc -l </RoboTwin/data/robotwin_demo_clean_joint_v30.txt)" -eq 50
test -z "$(find /RoboTwin/data -type f -name '*.zip' -print -quit)"
du -sh /RoboTwin/data/demo_clean
du -sch /RoboTwin/data/lerobot/* | tail -1
```

`robotwin_demo_clean_joint_v30.txt` 是包含 50 个独立 LeRobot 数据集路径的训练
清单，不是同名数据集目录；具体任务目录命名为 `<task>_joint_v30`，例如
`adjust_bottle_joint_v30`。

`/RoboTwin/eval_result` 链接到 `/workspace/runtime/eval_result`；训练
checkpoint 和日志也写入 `/workspace/runtime`。本地启动容器时，只有把宿主目录
挂载到 `/workspace/runtime`，这些输出才会持久化；该目录未挂载时，输出保存在
容器可写层，删除容器后会丢失。镜像内置输入数据是只随镜像分发的公共数据，
不需要持久卷保存。

## 5. 官方模型推理

终端 1：

```bash
cd /RoboTwin
mkdir -p /workspace/runtime/outputs/{logs,benchmarks}

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/official_server.log False
```

终端 2：

```bash
cd /RoboTwin
until curl -fsS http://127.0.0.1:13400/healthz; do sleep 2; done

source /opt/robotwin-env/bin/activate
export PYTHONPATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2:/RoboTwin

python experiments/lingbot_vla_v2_6b_robotwin/scripts/benchmark_official_inference.py \
  --port 13400 \
  --repeats 3 \
  --batch-sizes 1 2 4 \
  --output /workspace/runtime/outputs/benchmarks/official_inference.json
```

正常响应的 action 形状为 `[25, 14]`。第一次请求包含模型 warm-up，性能比较应使用后续请求。

## 6. RoboTwin 闭环评测

保持模型 server 运行，在模型环境中执行：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate
export ROBOTWIN_DISABLE_CUROBO=1
export ROBOTWIN_EE_PLANNER=mplib
export PYOPENGL_PLATFORM=egl
```

模型 server 保持运行，在另一个终端执行 `adjust_bottle` 的 10 个 episode：

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
  --device_id 0 \
  --seed 0 \
  --test_num 10 \
  --expert_check false \
  --eval_batch false
```

也可以通过统一入口运行同一组参数：

```bash
bash scripts/eval_policy.sh \
  --task_name adjust_bottle \
  --task_config demo_clean \
  --policy_name LingBot-VLA-v2 \
  --protocol lingbot_vla_v2 \
  --host 127.0.0.1 \
  --port 13400 \
  --device_id 0 \
  --seed 0 \
  --test_num 10 \
  --expert_check false \
  --eval_batch false
```

评测结果写入 `/workspace/runtime/eval_result`。首次检查环境时可以临时改成
`--test_num 1`，确认渲染、MPLib 规划器和 WebSocket 通信正常后再使用默认的
`--test_num 10`。如果启动时挂载了
`/workspace/runtime`，结果会持久化到对应的宿主目录；未挂载时仅保存在
容器内。

测试 `env_cfg/eval/all_tasks.yml` 中列出的全部 50 个任务时，可以复用同一个
13400 模型服务并顺序执行，避免多个仿真进程同时争用 GPU 显存：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate
mkdir -p /workspace/runtime/outputs/logs
set -o pipefail

mapfile -t TASKS < <(python - <<'PY'
import yaml

with open("env_cfg/eval/all_tasks.yml", encoding="utf-8") as file:
    print(*yaml.safe_load(file)["tasks"], sep="\n")
PY
)

for TASK in "${TASKS[@]}"; do
  echo "===== evaluating ${TASK} ====="
  bash scripts/eval_policy.sh \
    --task_name "${TASK}" \
    --task_config demo_clean \
    --policy_name LingBot-VLA-v2 \
    --protocol lingbot_vla_v2 \
    --host 127.0.0.1 \
    --port 13400 \
    --device_id 0 \
    --seed 0 \
    --test_num 10 \
    --expert_check false \
    --eval_batch false \
    2>&1 | tee "/workspace/runtime/outputs/logs/eval_${TASK}.log"
done
```

这会执行 50 × 10 个 episode。需要中断后继续时，可以直接把 `TASKS` 替换为尚未
完成的任务名列表。

## 7. LoRA 训练

默认使用单卡训练：

```bash
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
source /opt/robotwin-env/bin/activate
mkdir -p /workspace/runtime/outputs/logs

export HIP_VISIBLE_DEVICES=0
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=1 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/reproduction_100steps/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 1 \
  --train.gradient_accumulation_steps 4 \
  2>&1 | tee /workspace/runtime/outputs/logs/lora_100steps_1gpu.log
```

双卡训练只需把可见 GPU 和进程数改为 2：

```bash
export HIP_VISIBLE_DEVICES=0,1
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=2 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/reproduction_100steps/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 2 \
  --train.gradient_accumulation_steps 2 \
  2>&1 | tee /workspace/runtime/outputs/logs/lora_100steps_2gpu.log
```

四卡训练：

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  -m tasks.vla.train_lingbotvla \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/reproduction_100steps/lingbotvla_cli.yaml \
  --train.data_parallel_shard_size 4 \
  --train.gradient_accumulation_steps 1 \
  2>&1 | tee /workspace/runtime/outputs/logs/lora_100steps_4gpu.log
```

不要直接调用 PATH 中的 `torchrun`，它可能绑定错误的 Python。多卡训练使用
PyTorch Distributed Data Parallel；`HIP_VISIBLE_DEVICES` 中的 GPU 数量必须与
`--nproc-per-node` 一致。这里保持有效全局 batch size 为 4，因此还必须满足
`global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps`：
单卡、双卡、四卡的梯度累积步数分别是 4、2、1。训练输出位于：

```text
/workspace/runtime/outputs/reproduction_100steps
```

YAML 中的 `train.max_steps: 100` 只用于验证数据读取、前后向、保存和合并链路，
通常不足以获得有代表性的微调效果。Notebook 可以直接修改：

```python
TRAIN_STEPS = 100   # 流程验证
# TRAIN_STEPS = 1000  # 初步实验
# TRAIN_STEPS = 5000  # 更长训练，最终步数应根据闭环验证集选择
```

Notebook 会同步覆盖 `--train.max_steps`、`--train.save_steps` 和
`--train.output_dir`，合并步骤也会自动选择对应的
`global_step_${TRAIN_STEPS}`。命令行训练时可在原命令末尾加入同样的覆盖项，例如：

```bash
--train.max_steps 1000 \
--train.save_steps 1000 \
--train.output_dir /workspace/runtime/outputs/reproduction_1000steps
```

## 8. 合并 checkpoint 并重新推理

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

MERGED=/workspace/runtime/outputs/reproduction_100steps/merged_checkpoint/global_step_100/hf_ckpt

python experiments/lingbot_vla_v2_6b_robotwin/scripts/merge_lora_dcp.py \
  --checkpoint /workspace/runtime/outputs/reproduction_100steps/checkpoints/global_step_100 \
  --training-output /workspace/runtime/outputs/reproduction_100steps \
  --base-model /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt \
  --output "$MERGED" \
  --rank 8 \
  --alpha 16

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/merged_server.log False "$MERGED"
```

启动合并模型前，先停止原来占用 13400 端口的官方模型 server。合并模型 server
启动后，在另一个终端等待健康检查通过：

```bash
until curl -fsS http://127.0.0.1:13400/healthz; do sleep 2; done
```

然后按 RoboTwin 闭环入口评测合并后的模型：

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
  --device_id 0 \
  --seed 0 \
  --test_num 10 \
  --expert_check false \
  --eval_batch false
```

确认 `adjust_bottle` 正常后，可使用第 6 节的全部任务循环，在同一个 13400 服务上
评测合并后的 checkpoint。建议将旧模型和合并模型的 `/workspace/runtime/eval_result`
结果分别备份，避免同任务、同 seed 的结果混淆。

# 第二部分：从干净基础镜像或本机安装

本部分用于不使用完整镜像的环境。若已经使用第一部分的镜像，不要重复执行。

## 9. 基础环境

Docker 方式：

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

本机方式需要预先安装可工作的 ROCm 7.2.1、PyTorch 2.9.1 和 Vulkan/Mesa 驱动。
以下命令在容器或本机 shell 中执行。

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git git-lfs curl wget ffmpeg \
  libgl1 libglib2.0-0 libvulkan1 vulkan-tools mesa-vulkan-drivers \
  build-essential ninja-build cmake pkg-config
git lfs install
```

确认 GPU：

```bash
ls -l /dev/kfd /dev/dri
rocminfo | head
python - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.device_count())
assert torch.cuda.is_available() and torch.version.hip is not None
PY
```

## 10. 下载固定源码并应用补丁

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git /RoboTwin
cd /RoboTwin
git checkout 266f3aadf505a4f7fe9af0faa41a20f5f47cd123
git submodule update --init --recursive
test "$(git -C XPolicyLab rev-parse HEAD)" = c37109c500be67d0dea6b36bf7337bbd26e763cd

mkdir -p experiments/lingbot_vla_v2_6b_robotwin/source
git clone https://github.com/robbyant/lingbot-vla-v2.git \
  experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
  checkout 951475ae1b1d87553e7dc47c97b53a3d695c0d13
```

从本复现仓库应用三个兼容补丁：

```bash
git apply /path/to/RoboTwin-radeon-cloud/docker/patches/robotwin-rocm.patch
git -C XPolicyLab apply \
  /path/to/RoboTwin-radeon-cloud/docker/patches/xpolicylab-lerobot-v30.patch
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 apply \
  /path/to/RoboTwin-radeon-cloud/docker/patches/lingbot-vla-v2-rocm.patch

cp -a /path/to/RoboTwin-radeon-cloud/docker/assets/experiments/lingbot_vla_v2_6b_robotwin/. \
  experiments/lingbot_vla_v2_6b_robotwin/
```

这些补丁完成：

- 禁用 CUDA FlashAttention2，使用 eager attention；
- 使用 MPLib 替代 CuRobo；
- 兼容 LeRobot 0.6 数据读取接口；
- 注册 action-expert LoRA 参数；
- 修正 AMD 环境的评测和训练入口。

## 11. 创建模型环境

基础 ROCm 镜像的 torch 位于 `/opt/venv`。为了让后续命令与构建好的镜像完全
一致，模型环境固定创建在 `/opt/robotwin-env`。在本机执行时，当前用户必须拥有
`/opt` 的写权限，否则需要用 root 或 sudo 预先创建目录。

复用本复现仓库中与 Dockerfile 相同的固定依赖清单：

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

python -m pip install --upgrade 'pip<26' 'setuptools<81' wheel
python -m pip install --prefer-binary \
  -r "$REPRO_ROOT/docker/requirements.txt"
python -m pip install --no-deps \
  -r "$REPRO_ROOT/docker/requirements-no-deps.txt"
python -m pip install --prefer-binary open3d==0.19.0

python -m pip install -e /RoboTwin/XPolicyLab
python -m pip install --no-deps -e \
  /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
```

不要安装上游完整 requirements 文件，其中固定的 CUDA/PyPI torch、triton 或
flash-attn 会覆盖 ROCm torch。`requirements-no-deps.txt` 必须保留 `--no-deps`，
否则 LeRobot 的依赖解析会替换主清单中已经验证的版本。

模型环境准备完成后下载 RoboTwin 仿真资产：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate
bash scripts/_download_assets.sh
test -f assets/objects/objaverse/list.json
```

## 12. 创建数据转换环境

数据转换环境同样使用与镜像一致的 `/opt/lerobot-env`：

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

验证完整导入链：

```bash
cd /RoboTwin
python - <<'PY'
import h5py, cv2, numpy, pandas, pyarrow, lerobot
from XPolicyLab.utils.data_loader import load
print("data conversion imports OK")
PY
```

## 13. 下载模型

```bash
source /opt/robotwin-env/bin/activate
cd /RoboTwin
MODEL_ROOT=$PWD/experiments/lingbot_vla_v2_6b_robotwin/models
mkdir -p "$MODEL_ROOT"

huggingface-cli download robbyant/lingbot-vla-v2-6b-robotwin \
  --revision 0451855729ec904f970600e0aec8b84661423afe \
  --local-dir "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b-robotwin"

huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --include '*.json' '*.txt' '*.jinja' merges.txt vocab.json \
  --local-dir "$MODEL_ROOT/Qwen3-VL-4B-Instruct-config-tokenizer"
```

检查权重分片：

```bash
find "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt" \
  -name 'model-*.safetensors' | wc -l
# 应为 6
```

## 14. 数据、推理和训练

设置路径：

```bash
export ROBOTWIN_ROOT=/RoboTwin
export LINGBOT_VLA_SOURCE=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
export QWEN3VL_PATH=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/Qwen3-VL-4B-Instruct-config-tokenizer
export LINGBOT_VLA_PYTHON=/opt/robotwin-env/bin/python
export HF_LEROBOT_HOME=/RoboTwin/data/lerobot
export ROBOTWIN_DISABLE_CUROBO=1
export ROBOTWIN_EE_PLANNER=mplib
export PYOPENGL_PLATFORM=egl
```

下载、解压并删除全部原始 ZIP：

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

使用数据环境并行转换全部任务。构建镜像时直接调用仓库提供的批处理脚本：

```bash
LEROBOT_PYTHON=/opt/lerobot-env/bin/python \
CONVERSION_JOBS=8 \
bash /path/to/RoboTwin-radeon-cloud/docker/full/convert_all_data.sh
```

创建运行目录后，直接按第一部分第 5～8 节执行；虚拟环境和运行输出路径与镜像
保持一致，不再需要替换命令中的路径：

```bash
mkdir -p /workspace/runtime/eval_result \
  /workspace/runtime/outputs \
  /workspace/runtime/.cache/huggingface
```

## 15. 常见问题

### `ModuleNotFoundError: h5py`

说明正在使用缺少数据转换依赖的旧环境。执行：

```bash
source /opt/lerobot-env/bin/activate
python -m pip install h5py==3.14.0
```

重新构建的完整镜像已经包含该依赖。

### 找不到 `assets/norm_stats/robotwin.json`

旧镜像中的 `configs/robot_configs/robotwin.yaml` 使用相对于 LingBot-VLA 仓库根
目录的路径；如果从 `/RoboTwin` 启动训练，上游代码会错误地相对于
当前目录解析。先进入源码目录再启动：

```bash
cd /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
```

新版补丁会始终相对于 LingBot-VLA 仓库根目录解析该路径，不再依赖启动目录。

### PyTorch 看不到 AMD GPU

```bash
ls -l /dev/kfd /dev/dri
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())'
```

确认容器启动时传入 `/dev/kfd`、`/dev/dri`，并且没有安装 CUDA/PyPI torch 覆盖
ROCm 版本。

### Vulkan 或无头渲染失败

在 ROCm/AMD 环境启动 SAPIEN 渲染器时，可能看到：

```text
[svulkan2] [error] CUDA Error: cudaErrorInsufficientDriver
[svulkan2] [error] Failed to initialize denoiser
```

这是正常现象。svulkan2 会尝试初始化 NVIDIA CUDA denoiser，但当前环境使用 AMD
ROCm，没有 NVIDIA CUDA driver，因此该可选降噪器初始化失败。只要程序随后继续
运行、Vulkan 场景能够创建并且评测正常产生图像，就可以忽略这两行，不影响
RoboTwin 闭环评测。

只有在这两行之后程序退出、无法创建 renderer 或没有输出相机图像时，才属于真正
的渲染故障，此时执行：

```bash
export PYOPENGL_PLATFORM=egl
vulkaninfo --summary
```

确认 `/dev/dri` 已传入容器，并安装 `libvulkan1`、`mesa-vulkan-drivers`。

### 训练进程使用错误的 Python

不要直接调用 `torchrun`。激活模型环境后使用：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 ...
```

### checkpoint 没有持久化

需要在删除容器后保留 checkpoint 时，完整镜像和 external-data 镜像都应把宿主
目录挂到 `/workspace/runtime`，训练配置的 `output_dir` 必须位于
`/workspace/runtime/outputs`。external-data 镜像不会再把 `/workspace/runtime`
链接到外置数据目录。
