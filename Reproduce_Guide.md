# LingBot-VLA-v2-6B × RoboTwin ROCm 复现指南

固定版本：

| 组件 | 版本或提交 |
|---|---|
| 基础镜像 | `rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| 默认镜像 | `robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1` |
| FlashAttention2 | Dao-AILab `9c6d9297e5d43653c2ea5837ed9e0d10f76ed231`（PR #2900） |
| RoboTwin | `e154702de28cac4b9bfe0299044303f507236bb3` |
| RoboTwin 子模块 XPolicyLab | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| SAPIEN | `3.0.0b1` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` |
| 官方基础模型 revision | `11c703bf6a5c1f45b3b69168482da11fdbba53d7` |
| Qwen 配置 revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| MoGe-2 ViT-B normal revision | `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368` |
| RoboTwin2.0 数据 revision | `a967b852afa21a9cbf19a198f7e653109042e87c` |
| PyTorch / ROCm | 2.9.1 / 7.2.1 |
| LeRobot | 0.6.0 |

RoboTwin `e154702...` 修复了不稳定 seed 的有效 episode 计数。镜像使用该提交
及其固定的 XPolicyLab 子模块版本；兼容补丁让 RoboTwin 的评测脚本连接 LingBot
模型服务。

AMD 环境关闭 CuRobo，使用 MPLib 做末端位姿规划。闭环结果应注明
`ROCm + MPLib + expert_check=true`，不要直接与 CUDA/CuRobo 结果比较。

AMD 计算设备使用 `HIP_VISIBLE_DEVICES`，SAPIEN 渲染设备使用 `DRI_PRIME`。
评测脚本会自动设置后者；不要同时设置 `ROCR_VISIBLE_DEVICES`、
`CUDA_VISIBLE_DEVICES` 或 `GPU_DEVICE_ORDINAL`。

单任务和多任务评测都调用 RoboTwin 的 `scripts/eval_policy_xpolicylab.py`；
多任务脚本 `run_robotwin_benchmark.py` 负责启动模型服务和分配 GPU。它默认使用
FP32 推理、expert seed 筛选和任务配置中的指令类别。正式评测每任务运行
100 次有效 rollout；`--episodes 10` 可用于短程检查。例如在容器内：

```bash
python /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py \
  --model-path /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b \
  --task-config demo_clean --episodes 100 --gpu-count 4
```

记录结果时请注明 RoboTwin `e154702...`、SAPIEN `3.0.0b1`、ROCm/MPLib
及评测配置；与旧版 LingBot 客户端的结果比较时，应核对这些条件。

# 第一部分：使用构建好的 Docker 镜像

默认镜像使用固定版本的 RoboTwin、XPolicyLab 与 LingBot 源码。仿真资产、模型和
训练数据位于宿主机 `/models/robotwin-persistent`，启动容器时挂载 `/models`。

镜像中的运行路径为：

```text
/RoboTwin                   源码和默认工作目录
/RoboTwin/data/demo_clean   50 个任务的原始解压数据
/RoboTwin/data/lerobot      转换完成的 LeRobot v3 数据
/workspace/runtime          日志、评测结果和训练 checkpoint
/opt/robotwin-env           推理、训练、评测、LeRobot v3 读取和 Jupyter 环境
```

Radeon Cloud Global 当前的持久化 `/workspace` 空间只有 100 GB。启动训练或评测前，
建议先检查根盘和持久化 workspace 的剩余空间，以及 runtime 输出目录的占用：

```bash
df -h / /workspace
du -d1 -h /workspace/runtime/outputs
```

较大的中间编译文件、缓存或临时结果可以先放在 `/` 下的临时目录，以减轻
`/workspace` 的压力；但根盘内容通常不会随实例销毁而持久保存。最终 checkpoint、日志、
评测结果以及需要保留的模型必须复制或移动到 `/workspace` 持久化空间，并在销毁实例前确认
文件已经写入该目录。

## 1. 构建镜像

镜像不包含大型资产、数据和模型，它们必须位于宿主机：

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

在宿主机准备镜像需要挂载的资产、50 个任务的数据和基础模型。
此步骤不依赖上述镜像，镜像构建可在数据准备前后进行：

```bash
bash scripts/data/prepare_external_data.sh
```

脚本直接在宿主机下载固定 revision，按需安装 Python 依赖并转换 LeRobot v3 数据，
结果固定写入 `/models/robotwin-persistent/`；源码、隔离环境、缓存和转换日志存放在
`.preparation_runtime/`。可先加 `--dry-run` 查看步骤，不开始下载。

默认生成 `robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1`。

容器内的 `assets`、`data` 和 `models` 使用软链接指向上述挂载目录，因此后续命令
无需修改路径。entrypoint 会检查资产、50 任务训练清单和模型索引，未挂载或内容
不完整时会直接报出缺失路径。

如需覆盖镜像名或 pip 源：

```bash
IMAGE_NAME=my-registry/robotwin-lingbot:v1 \
PIP_INDEX_URL=https://pypi.org/simple \
./docker/external-data/build.sh
```

## 2. 启动容器

### 2.1 本地 Docker

设置默认镜像和数据挂载：

```bash
IMAGE=robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

启动容器：

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

`/workspace/runtime` 是普通目录。上述挂载用于持久化评测结果、
训练 checkpoint、日志和 Hugging Face 缓存；如果不需要在删除容器后保留输出，
可以删除该 `-v` 参数。宿主机挂载目录的权限由宿主机负责。

重新进入已有容器：

```bash
docker start robotwin-lingbot-vla-v2
docker exec -it robotwin-lingbot-vla-v2 bash
```

### 2.2 Radeon Cloud

当前需要在 [Radeon Cloud Global](https://radeon-global.anruicloud.com/) 上验证。云端
操作参考 [Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Embodied-AI-Challenge-AMD-Platform-2026-09/blob/main/Radeon-Cloud-User-Guide/README.md)。
Radeon Cloud 使用本项目构建并上传到镜像仓库的**默认镜像**：

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

1. 登录 Radeon Cloud。登录后默认进入 **Classic** 风格界面；点击页面右下角的
   **Switch to the new design** 切换到 **New** 风格界面，后续实例配置步骤以新版界面为准。
   如果本地还没有 SSH 密钥，先执行
   `ssh-keygen -t ed25519` 生成密钥，然后在平台中点击 **Settings → New SSH Key**，
   粘贴 `~/.ssh/id_ed25519.pub` 的内容并保存。只能上传 `.pub` 公钥，不要上传私钥。
2. 在实例配置页面点击 **Customize**，根据需要选择 **4 GPUs** 或 **8 GPUs**；在
   **Image** 中选择 **robotwin**，在 **Resource Pool** 中选择本次比赛对应的资源池 **Dev**，
   在 **Workspace Storage** 中选择 **Persistent /workspace**，最后在 **Mount a model**
   中选择 **Devzone**。如果没有选择 Devzone，实例内不会挂载镜像所需的数据。
3. 选择 **Persistent /workspace** 后，训练 checkpoint、日志和评测结果会保存在持久化
   workspace 中；选择 **Devzone** 后，平台后台会把相应内容挂载到 `/models`，用户不需要手动
   挂载。镜像所需内容位于其中的 `/models/robotwin-persistent`，目录结构为：

   每次运行前后都应检查 `/workspace` 的剩余空间。LoRA 和 Full-SFT checkpoint 可能占用
   数十 GB；不再需要的旧 LoRA 结果和中间输出应及时清理。若临时将中间结果放在根盘，
   必须在实例销毁前将最终结果保存到持久化 `/workspace`。

   ```text
   /models/robotwin-persistent/
   ├── assets/
   ├── data/
   └── models/
   ```

4. 配置完成并启动实例，等待页面显示 **Your workspace is ready**。点击
   **Open Notebook** 可以进入 JupyterLab；第一步已经添加公钥后，实例显示
   **Instance running** 时也可以直接复制页面 SSH 窗口中的 host、port 和 user，
   在本地终端登录：

   ```bash
   ssh <user>@<host> -p <port>
   ```

   Radeon Cloud 的 JupyterLab 文件浏览器默认打开 `/workspace`，因此可以先打开
   Terminal（命令行终端）并执行：

   ```bash
   ln -sfn /RoboTwin /workspace/RoboTwin
   ```

   然后刷新左侧文件列表，进入 `RoboTwin` 文件夹操作。该目录是指向镜像内
   `/RoboTwin` 的软链接，不会复制源码。镜像还内置了交互式入口：

   ```text
   /RoboTwin/notebooks/RoboTwin_ROCm_Reproduction.ipynb
   ```

   可在 JupyterLab 中打开该文件，依次完成挂载/GPU 检查、启动端口号为 13400 的模型服务、
   `adjust_bottle` 的 10-episode 闭环评测，以及可选的四卡/八卡 clean + randomized
   100-task × 10-episode 全量评测。训练流程包括四卡/八卡 LoRA 微调、LoRA checkpoint 合并、
   四卡/八卡全参数 SFT、全参数 DCP checkpoint 合并，以及分别在同一 13400 端口
   重新启动合并模型并再次进行闭环评测。Notebook 的闭环 benchmark、LoRA 训练和 LoRA
   benchmark 默认使用 8 卡；四卡分支仍然保留，只需修改对应 GPU 配置。Notebook 中的
   长时间 GPU 单元不会自动执行，需要用户根据实例 GPU 数量确认配置后手动运行。
5. 进入实例后先检查当前目录及后台挂载：

   ```bash
   pwd
   findmnt -T /models/robotwin-persistent
   ls -l /RoboTwin/assets \
     /RoboTwin/data \
     /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models
   ```

   `/RoboTwin` 已经内置于镜像，不需要重新 clone，也不
   需要手工建立该目录；资产、数据和模型由镜像中的软链接映射到后台挂载目录。
6. 云端实例中不要再启动第二层 Docker。执行以下检查以及第 3 节自检，确认实例
   已暴露 AMD GPU：

   ```bash
   ls -l /dev/kfd /dev/dri
   rocminfo | head
   ```

## 3. 环境和镜像内容检查

默认镜像通过 Dockerfile 的 `ENV` 全局设置运行变量；容器入口
也会加载 `/usr/local/bin/robotwin-env`，登录 shell 可通过 `/etc/profile.d/robotwin-env.sh`
加载同一配置：

- `AITER_TRITON_ONLY=1`：让 AITER 使用当前环境支持的 Triton 算子路径。
- `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`：启用 FlashAttention2 的 AMD Triton 实现。
- `PYTHONPATH=/opt/aiter`：让 Python 找到 AITER 源码。
- `TMPDIR=/workspace/runtime/tmp`：存放运行时临时文件。
- `HF_HOME=/workspace/runtime/.cache/huggingface`：存放 Hugging Face 缓存。

在容器内查看当前值：

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

三个 Git commit 应依次与本文开头的固定版本一致。

## 4. 检查挂载的训练数据

宿主机准备脚本下载固定 revision 的 50 个 `demo_clean.zip`，并将数据转换为
50 个独立的 `<task>_joint_v30` LeRobot 数据集。训练清单包含全部数据集。
挂载数据并启动容器后检查：

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

RoboTwin 评测器把逐 episode 结果写入 `/RoboTwin/eval_result`。镜像将该目录
链接到 `/workspace/runtime/eval_result`；训练 checkpoint 和日志也写入
`/workspace/runtime`。本地启动容器时，只有把宿主目录
挂载到 `/workspace/runtime`，这些输出才会持久化；该目录未挂载时，输出保存在
容器可写层，删除容器后会丢失。输入数据通过 `/models` 挂载，需保留宿主机的
`/models/robotwin-persistent`。

## 5. 官方模型推理

默认模型是官方 `robbyant/lingbot-vla-v2-6b` 基础 checkpoint。默认 server、默认
闭环评测和训练初始化都使用该基础 checkpoint；RoboTwin 后训练 checkpoint 不作为默认
模型，只有在对应的训练后验证命令中显式传入时才会使用。

终端 1：

```bash
cd /RoboTwin
mkdir -p /workspace/runtime/outputs/{logs,benchmarks}

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/official_server.log False
```

终端 2：

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

action 长度由模型服务的 `USE_LENGTH` 决定；当前默认值为 50，单样本响应通常为
`[50, 14]`，实际形状以结果中的 `action_shape` 为准。第一次请求包含模型 warm-up，
性能比较应使用后续请求。

## 6. RoboTwin 闭环评测

官方 `robbyant/lingbot-vla-v2-6b` baseline checkpoint 主要用于验证模型服务、渲染、规划器和评测链路；它在 RoboTwin 上的实际成功率可能不高，因此单次结果不应直接当作部署失败。需要更好的 RoboTwin 任务效果时，可以评测 LingBot 官方已经针对 RoboTwin 训练过的 checkpoint：

- [Hugging Face：robbyant/lingbot-vla-v2-6b-robotwin](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)
- [LingBot-VLA-v2 官方仓库](https://github.com/robbyant/lingbot-vla-v2)

下面先保留 baseline 评测命令，再给出 RoboTwin checkpoint 的对应命令。比赛或复现训练必须从 baseline checkpoint 开始，不能从已经训练好的 RoboTwin checkpoint 开始。

以下命令默认不录视频，以减少视频帧处理、ffmpeg 编码和磁盘写入开销。

**单任务评测（`scripts/eval_policy_xpolicylab.py`）**：一个任务可以运行多个 episode。
示例中的 `--additional_info eval_video_log=false` 关闭视频；需要保存视频时改为
`--additional_info eval_video_log=true`。

**多任务 benchmark（`experiments/lingbot_vla_v2_6b_robotwin/scripts/eval/run_robotwin_benchmark.py`）**：
示例中的 `--no-video` 关闭视频；需要保存视频时改为 `--video`。仅删除 `--no-video`
仍会使用脚本默认的不录视频设置。第 7 节 LoRA 和第 8 节 Full-SFT 的评测也遵循这些规则。

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
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

如果需要评测官方已经针对 RoboTwin 训练过的 checkpoint，先停止正在运行的 baseline server，再启动 RoboTwin checkpoint server：

```bash
cd /RoboTwin

ROBOTWIN_CHECKPOINT_ROOT=/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/models/robbyant_lingbot-vla-v2-6b-robotwin
ROBOTWIN_CHECKPOINT="$ROBOTWIN_CHECKPOINT_ROOT/checkpoints/global_step_50000/hf_ckpt"
export LINGBOTVLA_TRAINING_CONFIG="$ROBOTWIN_CHECKPOINT_ROOT/lingbotvla_cli.yaml"

bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/robotwin_checkpoint_server.log False \
  "$ROBOTWIN_CHECKPOINT"
```

`ROBOTWIN_CHECKPOINT_ROOT` 保存训练配置，实际模型权重在其
`checkpoints/global_step_50000/hf_ckpt/` 下。镜像中的 `models/`
是指向 `/models/robotwin-persistent/models/` 的软链接。启动脚本会优先读取导出的
`LINGBOTVLA_TRAINING_CONFIG`。此处若省略，启动脚本会改用本复现仓库的
`/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/training/lingbotvla_cli.yaml`，
而不是 RoboTwin checkpoint 自带的 YAML；两份配置的模型和数据设置不同。

然后使用与 baseline 相同的单任务闭环评测参数：

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

也可以通过 RoboTwin 的统一入口运行同一组参数；此处最终仍调用
`scripts/eval_policy_xpolicylab.py`：

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

评测结果写入 `/workspace/runtime/eval_result`。首次检查环境时可以临时改成
`--test_num 1`，确认渲染、MPLib 规划器和 WebSocket 通信正常后再使用默认的
`--test_num 10`。如果启动时挂载了
`/workspace/runtime`，结果会持久化到对应的宿主目录；未挂载时仅保存在
容器内。

测试 RoboTwin 仓库 `/RoboTwin/env_cfg/eval/all_tasks.yml` 列出的 50 个任务时，
脚本分别使用 `demo_clean` 和 `demo_randomized` 两种任务配置。统一脚本会启动
四个或八个模型服务和对应数量的评测 worker。每张 GPU 同时运行一个模型服务和一个仿真进程；
任务由动态队列分配，先完成的 GPU 会继续领取下一个任务，减少长短任务不均造成的
尾部等待。四卡运行前确认 `13400`～`13403` 没有被其他模型服务占用，八卡运行前确认
`13400`～`13407` 没有被其他模型服务占用。脚本默认通过 HIP 运行时自动取得
每张卡的 PCI 地址，并据此设置 `DRI_PRIME`；如需手工覆盖，可按 HIP GPU 顺序
设置 `ROBOTWIN_DRI_PRIME_MAP` 或使用 `--dri-prime-map`。

四卡运行：

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

脚本会自动完成以下操作：

- 在 GPU 0～3 上分别启动模型服务，端口为 `13400`～`13403`；
- 执行 50 个 `demo_clean` 和 50 个 `demo_randomized` 任务，每个任务 10 episodes，共 1000 episodes；
- 将每个任务的日志、耗时、失败记录和完成标记写入
  `/workspace/runtime/outputs/both100x10_4gpu`；
- 通过 `done/<task_config>__<task>.done` 跳过已完成任务，因此同一命令中断后可用 `--resume`
  继续；
- 全部完成后检查每个日志的 `Final success rate`，汇总总体成功率，并自动停止脚本
  启动的模型服务。

八卡运行时只需修改 GPU 数量和运行目录名：

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

如果要使用官方 RoboTwin checkpoint 运行同样的四卡/八卡长 benchmark，保持
`LINGBOTVLA_TRAINING_CONFIG` 指向 checkpoint 仓库根目录的配置，并通过
`--model-path` 传入实际权重目录 `checkpoints/global_step_50000/hf_ckpt`。启动脚本会读取
该环境变量；如果省略，会回退到前述本复现仓库的训练配置。先确认没有其他 server 占用对应端口。

四卡 RoboTwin checkpoint benchmark：

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

八卡时改为：

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

实测 clean、randomized 以及四卡/八卡合计墙钟时间如下：

| 机器 | 配置 | 完成度 | 墙钟时间 |
|---|---|---:|---:|
| 4× W7900 | clean | 50 tasks / 500 episodes | 8小时26分41秒 |
| 4× W7900 | randomized | 50 tasks / 500 episodes | 10小时29分07秒 |
| 4× W7900 | 合计 | 100 tasks / 1000 episodes | 18小时55分48秒 |
| 8× W7900 | clean | 50 tasks / 500 episodes | 6小时34分23秒 |
| 8× W7900 | randomized | 50 tasks / 500 episodes | 8小时37分58秒 |
| 8× W7900 | 合计 | 100 tasks / 1000 episodes | 15小时12分21秒 |

## 7. LoRA 训练、合并 checkpoint 并重新推理

### 7.1 LoRA 训练

四卡训练：

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

八卡训练：

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

不要直接调用 PATH 中的 `torchrun`，它可能绑定错误的 Python。多卡训练使用
PyTorch Distributed Data Parallel；`HIP_VISIBLE_DEVICES` 中的 GPU 数量必须与
`--nproc-per-node` 一致。这里四卡保持有效全局 batch size 为 4，八卡保持有效全局
batch size 为 8，因此还必须满足
`global_batch_size = micro_batch_size × data_parallel_size × gradient_accumulation_steps`：
四卡、八卡的梯度累积步数按对应的 global batch 和数据并行规模设置。训练输出位于：

```text
/workspace/runtime/outputs/lora_100steps_4gpu
/workspace/runtime/outputs/lora_100steps_8gpu
```

这里有三项名称相近但用途不同的 attention 配置：

- `model.attn_implementation: flash_attention_2`：用于 Hugging Face/Qwen 模型。
- `model.vit_attn_implementation: flash_attention_2`：用于视觉编码器。
- `train.attention_implementation: flex_cached`：用于 VLM 与 action expert 的联合注意力；
  该路径使用自定义二维 block mask，不能设置为 `flash_attention_2`，否则会报
  `Invalid attention implementation`。

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
--train.output_dir /workspace/runtime/outputs/lora_1000steps_8gpu
```

### 7.2 合并 LoRA checkpoint 并重新推理

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
  --seed 0 \
  --test_num 10 \
  --expert_check true \
  --eval_batch false \
  --additional_info eval_video_log=false
```

确认 `adjust_bottle` 正常后，可以对合并后的 LoRA 模型运行完整 100-task ×
10-episode 评测；开始前先停止上面占用 13400 端口的合并模型服务，因为脚本会自行启动
一组 4/8 卡模型服务。Notebook 默认使用八卡；命令行仍分别给出四卡和八卡示例。四卡运行：

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

八卡运行：

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

`--resume` 会读取同一运行目录下的 `done/<task>.done`，跳过已经完成的任务，适合长时间
评测中断后继续。若要从头独立复测，不要复用已有运行目录，应修改 `--run-name`。

## 8. 全量 SFT、转换 checkpoint 并重新推理

这是与第 7 节 LoRA 相互独立的训练流程。默认训练启用完整 depth/video teacher；不带
teacher 的普通 SFT 可以运行，但会缺少对应的深度/视频监督，效果较差，不推荐用于正式复现。
完整 teacher 会在训练 checkpoint 中保存仅用于 alignment loss 的额外 head；转换后的推理
模型不使用这些 head，部署 patch 会在保持其余权重严格校验的前提下过滤这些训练专用参数。
先停止占用 GPU 的模型 server，然后运行统一训练入口：

### 8.1 默认：完整 depth/video teacher 训练

官方基础 checkpoint 已包含 LingBot-Depth 和 DINO-VIDEO，持久模型目录还必须有单独下载的
`moge-2-vitb-normal/model.pt`。以下命令启用 current/future depth 与 DINO-video teacher，
完成若干真实 forward/backward/optimizer steps 后由 timeout 停止。默认使用 AdamW，micro
batch 16 与容量表一致：

| GPU 数 | `data_parallel_shard_size` | `micro_batch_size` | `gradient_accumulation_steps` | `global_batch_size` |
|---:|---:|---:|---:|---:|
| 8（默认） | 8 | 16 | 2 | 256 |
| 4 | 4 | 16 | 4 | 256 |

四卡运行：

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

八卡运行：

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

`TIMEOUT_SECONDS` 只控制 smoke test 的最长运行时间；正式训练时将其设为 `0`，并根据需要
调整训练步数、保存步数和输出目录。用户可以根据训练需求自行调整 `MAX_STEPS` 和
`SAVE_STEPS`；通常将两者设为相同值，需要更频繁保存 checkpoint 时可以单独减小
`SAVE_STEPS`。

四卡和八卡的完整 teacher 容量测试均验证到 micro batch 16，故 notebook 默认使用八卡
`16 × 8 × 2 = 256`；四卡配置为 `16 × 4 × 4 = 256`。脚本使用以下关键参数，其中 shard size
和梯度累积由 `GPU_COUNT` 自动计算：

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

完整 depth/video teacher 在四张或八张 W7900（每张 48 GiB）上的容量测试如下。测试启用
gradient checkpointing 和 FSDP2 full-shard；稳态 step time 取第 2、3 个 optimizer step
的平均值，峰值显存按每秒采样覆盖模型加载、forward、backward 和 optimizer update，取各卡
中的最高值。这里的测试 sweep 用于确认容量边界；正式默认配置仍为 micro batch 16、
global batch 256。

| 平台 | 每卡 micro batch | 梯度累积 | 稳态 step time | 峰值显存/卡 | 结果 |
|---|---:|---:|---:|---:|---|
| 4× W7900 48 GiB | 1 | 64 | 245.580 秒 | 40.329–40.375 GiB | 通过 |
| 4× W7900 48 GiB | 2 | 32 | 136.748 秒 | 41.337–41.349 GiB | 通过 |
| 4× W7900 48 GiB | 4 | 16 | 76.010 秒 | 43.509–43.521 GiB | 通过 |
| 4× W7900 48 GiB | 8 | 8 | 54.031 秒 | 47.729–47.761 GiB | 通过，余量有限 |
| 4× W7900 48 GiB | 16 | 4 | 41.392 秒 | 47.890–47.975 GiB | 通过，接近显存上限 |
| 4× W7900 48 GiB | 32 | 2 | — | 47.648–47.956 GiB，仅余 62–286 MiB | HIP OOM |
| 8× W7900 48 GiB | 1 | 32 | 143.665 秒 | 27.520–27.558 GiB | 完成计算 |
| 8× W7900 48 GiB | 2 | 16 | 80.319 秒 | 27.818–27.832 GiB | 通过 |
| 8× W7900 48 GiB | 4 | 8 | 47.055 秒 | 29.498–29.523 GiB | 通过 |
| 8× W7900 48 GiB | 8 | 4 | 27.108 秒 | 33.004–33.386 GiB | 通过 |
| 8× W7900 48 GiB | 16 | 2 | 24.682 秒 | 41.344–41.356 GiB | 通过 |
| 8× W7900 48 GiB | 32 | 1 | — | 47.878–47.947 GiB，仅余约 0.12–0.14 GiB | HIP OOM |

显存区间表示对应 GPU 数量的峰值显存最小值到最大值。稳态 step time 取 sweep 中第 2、3
个 optimizer step 的平均值。
上述数据是容量测试记录，不改变四卡和八卡默认训练配置。

默认四卡/八卡 AdamW 的全量 DCP checkpoint 目录约为 70 GB 级别。这是所有 rank 写出的
分片文件合计，不是每张卡各写对应大小。checkpoint 同时保存训练所需的完整模型参数和
optimizer 状态；AdamW 通常为每个参数保存一阶、二阶矩，因此 optimizer 部分往往比模型
权重本身更大。此外还有学习率调度器、随机数和 dataloader 状态。改变 GPU 数主要改变分片
文件的数量和单片大小，不会按比例降低整个 checkpoint 的总容量。应提前检查
`/workspace/runtime` 的可用空间；如果只需要部署，可以在转换出 Hugging Face checkpoint
并确认可加载后删除不再需要的 DCP。

### 8.2 可选：不带 teacher 的全量 SFT 训练

普通 SFT 入口支持 4 或 8 卡，但不启用 teacher，效果不如默认的完整 teacher 训练，不推荐作为
正式复现配置。命令格式与 8.1 保持一致，只将 `TEACHER_MODE` 改为 `none`。

四卡运行：

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

八卡运行：

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

### 8.3 `enable_full_shard` 的选择

该参数在 FSDP2 中传给 `reshard_after_forward`，不是“是否启用 FSDP”的开关：

- `true`：每个 FSDP 模块完成 forward 后重新分片；峰值显存更低，但 backward 前需要再次
  all-gather。复现脚本默认使用此设置。
- `false`：forward 后保留当前模块的完整参数直到 backward；可以减少一次 all-gather，
  可能更快，但通常占用更多峰值显存。模型整体仍然使用 FSDP2 分片。

可以使用以下三种配置，正式复现默认选择第一种：

| 配置 | 关键参数 | 用途 |
|---|---|---|
| 基础全量 SFT | `data_parallel_shard_size=GPU_COUNT`、global batch 256、full shard | 本节默认配置 |
| no-reshard | `data_parallel_shard_size=GPU_COUNT`、global batch 256、`enable_full_shard=false` | 比较少一次参数 all-gather 的速度和显存代价 |
| FP32/future-image 变体 | shard `GPU_COUNT`、full shard、`enable_fp32=true`、`use_future_image=true` | 检查额外训练开关；若 `align_params={}`，它不代表完整 depth/video alignment 训练 |

三种配置都设置 `use_lora=false`、gradient checkpointing，并执行全参数
forward、backward 和 optimizer step。它们是可选配置，不是三个连续训练阶段，也不需要
全部运行。

需要比较 no-reshard 时，只覆盖一个环境变量：

四卡运行：

```bash
GPU_COUNT=4 \
ENABLE_FULL_SHARD=false \
TEACHER_MODE=full \
MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_4gpu_no_reshard_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

八卡运行：

```bash
GPU_COUNT=8 \
TEACHER_MODE=full \
ENABLE_FULL_SHARD=false \
MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_8gpu_no_reshard_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/training/train_full_sft.sh
```

### 8.4 将全量 DCP 转换为推理 checkpoint 并重新推理

全量 SFT 没有 LoRA adapter，因此这里是把分布式 DCP 聚合并保存为 Hugging Face 格式，
不是把 adapter 合并回基础模型：

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

启动全量 SFT 模型，继续复用 13400 接口：

```bash
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/serving/launch_official_server.sh \
  0 13400 /workspace/runtime/outputs/logs/full_sft_server.log False "$FULL_SFT_MODEL"
```

在另一个终端等待服务就绪并运行闭环评测：

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

确认单任务正常后，可以对转换后的全量 SFT 模型运行完整 100-task × 10-episode
评测；开始前先停止上面占用 13400 端口的全量 SFT 服务。四卡运行：

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

八卡运行：

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

完整评测耗时很长。`--resume` 只补跑同一 `--run-name` 下尚未生成完成标记的任务；需要
对相同 checkpoint 重新进行一轮独立评测时，应使用新的 `--run-name`，避免与旧结果混合。

# 第二部分：从干净基础镜像或本机安装

本部分用于不使用本项目镜像的环境。若已经使用第一部分的镜像，不要重复执行。

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

从本复现仓库应用三个兼容补丁：RoboTwin ROCm 与 LingBot 评测补丁、
XPolicyLab 数据补丁，以及 LingBot-VLA-v2 补丁：

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

各补丁的修改文件、作用及适用版本见 [patches/README.md](./patches/README.md)。主要用途：

- `robotwin-rocm-reproduction.patch`：RoboTwin 的 ROCm 环境兼容、CuRobo fallback、
  MPLib 规划及 LingBot websocket 评测协议；基于上游已修复 seed 计数的 `e154702...`，
  保留其不稳定 seed 的筛选顺序；
- `xpolicylab-lerobot-v30.patch`：兼容 LeRobot 0.6 的数据转换和 EE 数据格式；
- `lingbot-vla-v2-rocm.patch`：兼容 ROCm/LeRobot 训练、推理和 action-expert LoRA。

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

awk '/^(pip|setuptools|wheel)==/{print}' \
  "$REPRO_ROOT/docker/requirements.txt" > /tmp/robotwin-build-tools.txt
python -m pip install --no-deps -r /tmp/robotwin-build-tools.txt
python -m pip install --no-deps --no-build-isolation --prefer-binary \
  -r "$REPRO_ROOT/docker/requirements.txt"

python -m pip install --no-deps --no-build-isolation \
  -e /RoboTwin/XPolicyLab \
  -e /RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2
```

不要安装上游完整 requirements 文件，其中固定的 CUDA/PyPI torch、triton 或
flash-attn 会覆盖 ROCm torch。统一清单中的普通包按名称排序，并以 `--no-deps`
装入同一个模型和 Jupyter 环境；XPolicyLab、LingBot 从固定源码直接 editable 安装。
LeRobot 0.6
声明要求 NumPy ≥2 和 huggingface-hub ≥1，与当前 LingBot/Transformers 组合冲突。
固定的 NumPy 1.26 与 hub 0.34 已实际读取本项目的 LeRobot v3 数据；包元数据
仍不满足 LeRobot 的声明约束，不能把 `pip check` 当作此组合的通过条件。

### 11.1 安装 ROCm FlashAttention2

先安装固定版本的 AITER，并强制复用 ROCm PyTorch 环境自带的 Triton：

```bash
git clone https://github.com/ROCm/aiter.git /opt/aiter
git -C /opt/aiter checkout 9bab8388c35936814a659b4ebd245c491e1b940a
test "$(git -C /opt/aiter rev-parse HEAD)" = \
  9bab8388c35936814a659b4ebd245c491e1b940a

cd /opt/aiter
AITER_USE_SYSTEM_TRITON=1 \
  /opt/robotwin-env/bin/python setup.py develop
```

然后安装已合入 PR #2900 的 Dao-AILab FlashAttention2 主仓库固定提交。
`AITER_TRITON_ONLY=1` 让安装期只使用
AITER 的 Triton 实现，避免在无 GPU 的镜像构建环境中加载 AITER AOT/C++ 算子并查询
GPU driver；因此不需要再修改 FlashAttention 的 `setup.py`。`--no-deps` 和
`FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE` 用于防止 pip 安装第二份 AITER、Triton 或
PyTorch：

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

`AITER_USE_SYSTEM_TRITON=1` 和 `FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE` 仅用于安装。
`AITER_TRITON_ONLY=1` 还会被 AITER 的 `__init__.py` 在导入时读取：对于本文固定的
Triton 3.5.1，它能阻止 AITER 继续加载要求 Triton 3.6 以上的 Gluon 模块。因此运行
LingBot-VLA-v2 推理或训练时，必须同时设置 `AITER_TRITON_ONLY=1` 和
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`。裸机安装还需让 Python 找到 `/opt/aiter` 中的
editable AITER 源码，然后执行导入检查：

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

当前基础镜像自带的 Triton 与 PyTorch/ROCm 配套，不能为了满足 AITER 的版本提示而单独
升级或降级 Triton。该组合可能提示 AITER 更偏好 Triton 3.6，也可能提示找不到
`flash_attn_2_cuda` 而使用 Triton AMD fallback；这不代表安装失败。模型补丁不再覆盖上游
attention 默认值，镜像内置训练配置也使用 `flash_attention_2`。若某个独立算子不兼容，
可在对应实验配置中显式改回 `eager` 进行诊断。

模型环境准备完成后下载 RoboTwin 仿真资产：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate
bash scripts/_download_assets.sh
test -f assets/objects/objaverse/list.json
```

## 12. 创建数据转换环境

仅在从干净基础镜像或裸机手工转换数据时，才需要创建独立的 `/opt/lerobot-env`。
本项目默认镜像使用宿主机预先转换的数据，不包含这个环境：

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

检查权重分片：

```bash
find "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b" -maxdepth 1 \
  -name 'model-*.safetensors' | wc -l
# 应为 6

test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/depth/model.pt"
test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth"
test -f "$MODEL_ROOT/robbyant_lingbot-vla-v2-6b/dino_video/config.yaml"
test -f "$MODEL_ROOT/moge-2-vitb-normal/model.pt"
```

基础 checkpoint 自带 LingBot-Depth 和 DINO-VIDEO checkpoint/config，但不自带
MoGe-2 权重；Qwen 下载仅保留 tokenizer/config，不需要重复下载一份 4B 权重。普通推理和
不启用 alignment loss 的 LoRA/SFT 不加载这些 teacher，只有完整 depth/video teacher
训练才会增加相应显存和计算开销。

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
source /usr/local/bin/robotwin-env
```

从干净基础镜像或本机安装时，每个用于推理或训练的新 shell 都要执行上面的 `source`；
也可以把 `docker/robotwin-env.sh` 复制到本机后 source。不要在运行期继续设置仅用于安装的
`AITER_USE_SYSTEM_TRITON` 和 `FLASH_ATTENTION_USE_SYSTEM_AITER`。

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
source /usr/local/bin/robotwin-env
LEROBOT_PYTHON=/opt/lerobot-env/bin/python \
CONVERSION_JOBS=8 \
bash /path/to/RoboTwin-radeon-cloud/scripts/data/convert_all_data.sh
```

转换脚本默认将临时目录、日志和缓存放在 `/workspace/runtime`，不再使用通常只有几 GB
的 `/tmp`。可以通过
`ROBOTWIN_RUNTIME_ROOT`、`CONVERSION_CACHE_ROOT` 和 `CONVERSION_LOG_ROOT` 覆盖这些路径；
例如需要把缓存放到单独持久盘时：

```bash
CONVERSION_CACHE_ROOT=/root/conv-cache \
CONVERSION_LOG_ROOT=/workspace/runtime/outputs/logs/robotwin-lerobot-conversion \
  bash /path/to/RoboTwin-radeon-cloud/scripts/data/convert_all_data.sh
```

Dockerfile 构建阶段仍会显式把 `CONVERSION_CACHE_ROOT` 指向 BuildKit cache mount，避免把
构建缓存写入最终镜像层。

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

默认镜像的 `/opt/robotwin-env` 已包含 `h5py`；这里的安装命令仅适用于手工创建的
`/opt/lerobot-env`。

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

### SAPIEN 与 svulkan2 日志

旧版代码在 ROCm/AMD 环境启动 SAPIEN 渲染器时可能看到：

```text
[svulkan2] [error] CUDA Error: cudaErrorInsufficientDriver
[svulkan2] [error] Failed to initialize denoiser
```

这是 svulkan2 尝试初始化可选 NVIDIA CUDA denoiser 时产生的日志，不表示 ROCm
评测失败。当前 Docker 镜像的 RoboTwin 补丁已经在创建任何 SAPIEN engine、scene 或
renderer 之前关闭该 denoiser，并分别设置主体日志和渲染日志：

```python
import sapien.core as sapien

sapien.set_log_level("warning")
sapien.render.set_ray_tracing_denoiser("none")
sapien.render.set_log_level("critical")
```

`sapien.set_log_level()` 控制 SAPIEN 主体及物理相关日志；真正控制上述
`[svulkan2]` 消息的是 `sapien.render.set_log_level()`。代码必须位于
`sapien.Engine()`、`sapien.Scene()`、`sapien.render.RenderSystem(...)` 或
`sapien.SapienRenderer()` 之前。同时不得在后面的 `setup_scene()` 中再次调用
`set_ray_tracing_denoiser("oidn")`，否则会重新启用 CUDA denoiser。

在真实 `adjust_bottle` 单 episode 闭环测试中，修改后任务成功率为 1/1，日志中的
`cudaErrorInsufficientDriver` 和 `Failed to initialize denoiser` 均为 0 条。这个设置只
关闭可选光线追踪去噪器并调整日志级别，不关闭 Vulkan renderer，也不修改模型推理。

如果仍然无法创建 renderer 或没有输出相机图像，执行：

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

需要在删除容器后保留 checkpoint 时，应把宿主
目录挂到 `/workspace/runtime`，训练配置的 `output_dir` 必须位于
`/workspace/runtime/outputs`。默认镜像不会把 `/workspace/runtime`
链接到外置数据目录。
