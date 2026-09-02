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

当前需要在 [Radeon Cloud Global](https://radeon-global.anruicloud.com/) 上验证。云端
操作参考 [Radeon Cloud User Guide](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/tree/main/Radeon-Cloud-User%20Guide)。
Radeon Cloud 使用本项目构建并上传到镜像仓库的 **external-data 镜像**，不是 full
镜像，也不是仅启动原始 ROCm PyTorch 基础镜像：

```text
robotwin-lingbot-vla-v2:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1-external-data
```

1. 登录 Radeon Cloud，进入 **Profile → My Templates → Add Template**。
2. **Container Image** 填写上述 external-data 镜像在云端镜像仓库中的完整地址。
   Add Template 最后的 **Model Directory** 选项必须选择并拉取 **Devzone**；如果
   没有选择 Devzone，实例内不会挂载 external-data 镜像所需的数据。
3. 选择 **Devzone** 后，平台后台会把相应内容挂载到 `/models`，用户不需要手动
   挂载。external-data
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
   `adjust_bottle` 的 10-episode 闭环评测、可选的单卡/双卡/四卡 50-task ×
   50-episode 全量评测、单卡/双卡/四卡 LoRA 微调、checkpoint 合并，以及在同一
   13400 端口重新启动合并模型并再次闭环评测。Notebook 中的长时间 GPU 单元不会
   自动执行，需要用户手动运行。
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

full 和 external-data 镜像都已经通过 Dockerfile 的 `ENV` 持久设置
`AITER_TRITON_ONLY=1` 和 `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`，因此进入新 shell 后
不需要手工重复设置。前者避免 AITER 顶层导入当前任务不需要且要求更高 Triton 版本的
Gluon/AOT 算子，后者使 FlashAttention2 使用 AMD Triton kernel。

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

测试 `env_cfg/eval/all_tasks.yml` 中列出的全部 50 个任务时，使用统一脚本启动
四个模型服务和四个评测 worker。每张 GPU 同时运行一个模型服务和一个仿真进程；
任务由动态队列分配，先完成的 GPU 会继续领取下一个任务，减少长短任务不均造成的
尾部等待。运行前确认 `13400`～`13403` 没有被其他模型服务占用：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 4 \
  --episodes 50 \
  --run-name clean50x50_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

脚本会自动完成以下操作：

- 在 GPU 0～3 上分别启动模型服务，端口为 `13400`～`13403`；
- 执行全部 50 个 `demo_clean` 任务，每个任务 50 episodes，共 2500 episodes；
- 将每个任务的日志、耗时、失败记录和完成标记写入
  `/workspace/runtime/outputs/clean50x50_4gpu`；
- 通过 `done/<task>.done` 跳过已完成任务，因此同一命令中断后可用 `--resume`
  继续；
- 全部完成后检查每个日志的 `Final success rate`，汇总总体成功率，并自动停止脚本
  启动的模型服务。

同一个脚本也支持单卡和双卡，只需同时修改卡数和运行目录名：

```bash
# 单卡
python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 1 --episodes 50 --run-name clean50x50_1gpu \
  --runtime-dir /workspace/runtime --resume

# 双卡
python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 2 --episodes 50 --run-name clean50x50_2gpu \
  --runtime-dir /workspace/runtime --resume
```

根据已经完成的 50 任务 × 50 episodes 运行，时间参考如下：

| GPU 数量 | 时间 |
|---:|---:|
| 1 | 约 25 小时 27 分钟 |
| 2 | 约 13 小时 40 分钟 |
| 4 | 约 6 小时 50 分钟 |

## 7. LoRA 训练、合并 checkpoint 并重新推理

### 7.1 LoRA 训练

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

这里有三项名称相近但用途不同的 attention 配置：

- `model.attn_implementation: flash_attention_2`：用于 Hugging Face/Qwen 模型。
- `model.vit_attn_implementation: flash_attention_2`：用于视觉编码器。
- `train.attention_implementation: flex_cached`：用于 VLM 与 action expert 的联合注意力；
  该路径使用自定义二维 block mask，不能设置为 `flash_attention_2`，否则会报
  `Invalid attention implementation`。

旧版数据清单中如果仍含 `/workspace/RoboTwin/data/...`，镜像内的数据加载兼容补丁会在
对应 `/RoboTwin/data/...` 目录确实存在时自动映射，无需修改持久化数据文件。

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

### 7.2 合并 LoRA checkpoint 并重新推理

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

确认 `adjust_bottle` 正常后，可以对合并后的 LoRA 模型运行完整 50-task ×
50-episode 评测。该过程四卡约需 6 小时 50 分钟，单卡或双卡会更久；开始前先停止
上面占用 13400 端口的单卡合并模型服务，因为脚本会自行启动一组模型服务：

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 4 \
  --episodes 50 \
  --model-path "$MERGED" \
  --run-name lora_100steps_clean50x50_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

`--resume` 会读取同一运行目录下的 `done/<task>.done`，跳过已经完成的任务，适合长时间
评测中断后继续。若要从头独立复测，不要复用已有运行目录，应修改 `--run-name`。

## 8. 全量 SFT、转换 checkpoint 并重新推理

这是与第 7 节 LoRA 相互独立的训练流程。全量 SFT 会更新模型的全部可训练参数，不能使用
LoRA 的 `merge_lora_dcp.py`。先停止占用 GPU 的模型 server，然后运行 1 step smoke test：

### 8.1 全量 SFT 训练

脚本支持 1、2、4 卡，默认使用 4 卡。与 LoRA 训练相同，有效 global batch 始终保持为 4：

| GPU 数 | `data_parallel_shard_size` | `gradient_accumulation_steps` | `global_batch_size` |
|---:|---:|---:|---:|
| 1 | 1 | 4 | 4 |
| 2 | 2 | 2 | 4 |
| 4（默认） | 4 | 1 | 4 |

单卡 1-step smoke test：

```bash
cd /RoboTwin
GPU_COUNT=1 MAX_STEPS=1 SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_1gpu_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

双卡 1-step smoke test：

```bash
cd /RoboTwin
GPU_COUNT=2 MAX_STEPS=1 SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_2gpu_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

双卡 48 GB 可以优先按这条命令评估，建议保持 `enable_full_shard=true` 和 gradient
checkpointing，并在另一个终端持续观察两张卡的显存。卡数减少后，每张卡承担的参数、梯度
和 optimizer 分片都会增大，因此不能从四卡结果直接保证双卡一定不 OOM；只有 1 step 完整
通过 forward、backward 和 optimizer step，才能继续增加训练步数。

默认四卡 1-step smoke test：

```bash
cd /RoboTwin

MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_4gpu_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

脚本使用以下关键参数，其中 shard size 和梯度累积由 `GPU_COUNT` 自动计算：

```text
use_lora=false
data_parallel_mode=fsdp2
data_parallel_replicate_size=1
data_parallel_shard_size=GPU_COUNT
micro_batch_size=1
gradient_accumulation_steps=4/GPU_COUNT
global_batch_size=4
enable_gradient_checkpointing=true
enable_full_shard=true
```

确认四个 rank 都完成 forward、backward、optimizer step 和 DCP 保存后，再运行所需步数：

```bash
GPU_COUNT=4 \
MAX_STEPS=100 \
SAVE_STEPS=100 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_4gpu_100steps \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

`MAX_STEPS=100` 仍主要用于验证完整流程。正式微调时根据验证集结果增加训练步数，并用较小的
`SAVE_STEPS` 定期保存。

每个全量 DCP checkpoint 目录约为 70 GB，这是所有 rank 写出的分片文件合计，不是每张卡
各写 70 GB。checkpoint 同时保存训练所需的完整模型参数和 AdamW optimizer 状态；AdamW
通常为每个参数保存一阶、二阶矩，因此 optimizer 部分往往比模型权重本身更大。此外还有
学习率调度器、随机数和 dataloader 状态。改变 GPU 数主要改变分片文件的数量和单片大小，
不会按比例降低整个 checkpoint 的总容量。应提前检查 `/workspace/runtime` 的可用空间；如果
只需要部署，可以在转换出 Hugging Face checkpoint 并确认可加载后删除不再需要的 DCP。

### 8.2 `enable_full_shard` 的选择

该参数在 FSDP2 中传给 `reshard_after_forward`，不是“是否启用 FSDP”的开关：

- `true`：每个 FSDP 模块完成 forward 后重新分片；峰值显存更低，但 backward 前需要再次
  all-gather。复现脚本默认使用此设置。
- `false`：forward 后保留当前模块的完整参数直到 backward；可以减少一次 all-gather，
  可能更快，但通常占用更多峰值显存。模型整体仍然使用 FSDP2 分片。

四卡可以使用以下三种配置，正式复现默认选择第一种：

| 配置 | 关键参数 | 用途 |
|---|---|---|
| 四卡基础全量 SFT | shard 4、global batch 4、full shard | 本节默认的四卡复现配置 |
| 四卡 no-reshard | shard 4、global batch 4、`enable_full_shard=false` | 比较少一次参数 all-gather 的速度和显存代价 |
| 四卡 FP32/future-image 变体 | shard 4、full shard、`enable_fp32=true`、`use_future_image=true` | 检查额外训练开关；若 `align_params={}`，它不代表完整 depth/video alignment 训练 |

三种配置都设置 `use_lora=false`、micro batch 1、gradient checkpointing，并执行全参数
forward、backward 和 optimizer step。它们是可选配置，不是三个连续训练阶段，也不需要
全部运行。

需要比较 no-reshard 时，只覆盖一个环境变量：

```bash
ENABLE_FULL_SHARD=false \
MAX_STEPS=1 \
SAVE_STEPS=1 \
OUTPUT_DIR=/workspace/runtime/outputs/full_sft_4gpu_no_reshard_1step \
bash experiments/lingbot_vla_v2_6b_robotwin/training/train_full_sft.sh
```

### 8.3 将全量 DCP 转换为推理 checkpoint 并重新推理

全量 SFT 没有 LoRA adapter，因此这里是把分布式 DCP 聚合并保存为 Hugging Face 格式，
不是把 adapter 合并回基础模型：

```bash
cd /RoboTwin
source /opt/robotwin-env/bin/activate

TRAIN_OUTPUT=/workspace/runtime/outputs/full_sft_4gpu_100steps
CHECKPOINT="$TRAIN_OUTPUT/checkpoints/global_step_100"
FULL_SFT_MODEL="$TRAIN_OUTPUT/merged_checkpoint/global_step_100/hf_ckpt"

python experiments/lingbot_vla_v2_6b_robotwin/scripts/convert_full_sft_dcp.py \
  --checkpoint "$CHECKPOINT" \
  --training-output "$TRAIN_OUTPUT" \
  --output "$FULL_SFT_MODEL"
```

启动全量 SFT 模型，继续复用 13400 接口：

```bash
bash experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh \
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
  --device_id 0 \
  --seed 0 \
  --test_num 10 \
  --expert_check false \
  --eval_batch false
```

确认单任务正常后，可以对转换后的全量 SFT 模型运行完整 50-task × 50-episode
评测。该过程四卡约需 6 小时 50 分钟，单卡或双卡会更久；开始前先停止上面占用
13400 端口的单卡全量 SFT 服务：

```bash
python experiments/lingbot_vla_v2_6b_robotwin/scripts/run_clean_benchmark.py \
  --gpu-count 4 \
  --episodes 50 \
  --model-path "$FULL_SFT_MODEL" \
  --run-name full_sft_100steps_clean50x50_4gpu \
  --runtime-dir /workspace/runtime \
  --resume
```

完整评测耗时很长。`--resume` 只补跑同一 `--run-name` 下尚未生成完成标记的任务；需要
对相同 checkpoint 重新进行一轮独立评测时，应使用新的 `--run-name`，避免与旧结果混合。

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

然后安装固定的 AMD FlashAttention2 fork。`AITER_TRITON_ONLY=1` 让安装期只使用
AITER 的 Triton 实现，避免在无 GPU 的镜像构建环境中加载 AITER AOT/C++ 算子并查询
GPU driver；因此不需要再修改 FlashAttention 的 `setup.py`。`--no-deps` 和
`FLASH_ATTENTION_USE_SYSTEM_AITER=TRUE` 用于防止 pip 安装第二份 AITER、Triton 或
PyTorch：

```bash
git clone https://github.com/ZiguanWang/flash-attention.git \
  /opt/flash-attention-source
git -C /opt/flash-attention-source checkout \
  bc76302fbb24c0158207978930db030ca1eca5ca
test "$(git -C /opt/flash-attention-source rev-parse HEAD)" = \
  bc76302fbb24c0158207978930db030ca1eca5ca

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
export AITER_TRITON_ONLY=1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export PYTHONPATH=/opt/aiter${PYTHONPATH:+:${PYTHONPATH}}
```

从干净基础镜像或本机安装时，每个用于推理或训练的新 shell 都要设置上面最后三个变量；
也可以将它们写入所使用 shell 的启动文件。不要在运行期继续设置仅用于安装的
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
