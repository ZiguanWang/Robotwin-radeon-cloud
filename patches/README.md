# 源码补丁

Docker 构建和手动复现都以以下固定源码版本为基准。补丁在源码检出后、editable 安装前应用；不要直接应用到其他提交。

| 仓库 | 固定提交 | 补丁 |
|---|---|---|
| RoboTwin | `e154702de28cac4b9bfe0299044303f507236bb3` | `robotwin-rocm-reproduction.patch` |
| XPolicyLab（RoboTwin 子模块） | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | `xpolicylab-lerobot-v30.patch` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` | `lingbot-vla-v2-rocm.patch` |

## RoboTwin：`robotwin-rocm-reproduction.patch`

| 修改文件 | 作用 |
|---|---|
| `envs/_base_task.py` | 降低 SAPIEN 日志量，禁用当前 RADV 路径不支持的 OIDN 光追降噪。 |
| `envs/robot/planner.py` | CuRobo 不可用时允许 policy rollout 初始化，并为 MPLib 实现按候选目标逐个规划的 `plan_batch` 兼容接口。CuRobo 专属操作仍会显式报错。 |
| `envs/robot/robot.py` | 根据 `ROBOTWIN_EE_PLANNER` 选用 MPLib，并让左右臂的单目标、批量末端位姿规划进入对应 MPLib planner。 |
| `scripts/eval_policy_xpolicylab.py` | 接入 LingBot 的 `WebsocketClientPolicy` 与观测／动作协议；支持显式 seed 起点、指令及视频覆盖，记录推理耗时，并校验 LingBot 协议只能使用非 batch 评测。保留固定上游提交已有的不稳定 seed 计数逻辑。 |

闭环评测、ROCm 渲染与 MPLib 路径仍使用此补丁，不能整份删除。日志级别和推理计时是辅助改动；它们并不构成删除其他兼容修改的理由。

## XPolicyLab：`xpolicylab-lerobot-v30.patch`

只修改 `scripts/transform_lerobot_v30_format.py`：

- 对接 LeRobot 0.6 的 `RGBEncoderConfig`、dataset writer 和 `finalize()` API。
- 用 PyAV 读取视频、H.264 CPU 编码和明确的 CRF 配置替代原来的 NVIDIA NVENC 默认值；关闭多相机 streaming 编码，避免已观察到的丢帧。
- 保留默认的 joint state/action 转换，并增加可选的 `--control_mode ee`，将双臂末端位姿与夹爪整理为 `[pose7, gripper] × 2`。

宿主机 `scripts/data/prepare_external_data.sh` 使用其 joint 模式转换 50 个任务，因此**整个补丁仍需要**。EE 分支在当前默认流程中不执行，但删除它会移除手工转换 EE 数据的能力；若决定只维护 joint 格式，可单独裁剪该分支并重新验证转换器。

## LingBot-VLA-v2：`lingbot-vla-v2-rocm.patch`

| 修改文件 | 作用 |
|---|---|
| `deploy/lingbot_vla_v2_policy.py` | 忽略推理模型不使用的 Full-SFT alignment 权重，同时拒绝其他未知权重；按显式路径或实验配置查找训练 YAML；关闭单形状 ViT 网格缓存；接收 episode reset seed。workshop 的 MoE 回退只在指定环境变量下启用。 |
| `lingbotvla/data/vla_data/base_dataset.py` | 适配 LeRobot 0.6 的 transform 导入、DatasetReader 与 PyAV 视频读取。 |
| `lingbotvla/data/vla_data/multi_vla_dataset.py` | 兼容旧清单中的 `/workspace/RoboTwin` 绝对路径，将其解析到当前 `/RoboTwin`。 |
| `lingbotvla/data/vla_data/utils.py` | 将相对的机器人归一化统计文件路径解析到 LingBot 仓库，避免从 `/RoboTwin` 启动时找错目录。 |
| `lingbotvla/utils/arguments.py` | 防止 YAML 中的 `null` 或空列表覆盖训练参数的 dataclass 默认值。 |
| `lingbotvla/utils/lora_utils.py` | 支持按正则限定 LoRA 目标模块，并返回注入 adapter 后的模型。 |
| `tasks/vla/train_lingbotvla.py` | 加入 action-expert 或所有 attention 层的 LoRA 训练选项，冻结基础权重并报告可训练参数。 |
| `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py` | 定义上游异常回退及 sliding-window 提示已引用却未初始化的 `logger`，避免这些分支报 `NameError`。 |

模型服务、LoRA 训练及 LeRobot v3 数据读取都依赖这些修改，不能整份删除。旧路径兼容是可单独审视的历史支持；移除前须确认所有运行清单均已迁移。

## 手动应用与检查

在各仓库检出上表提交后执行：

```bash
REPRO_ROOT=/path/to/RoboTwin-radeon-cloud
cd /RoboTwin

git apply --check "$REPRO_ROOT/patches/robotwin-rocm-reproduction.patch"
git apply "$REPRO_ROOT/patches/robotwin-rocm-reproduction.patch"
git -C XPolicyLab apply --check "$REPRO_ROOT/patches/xpolicylab-lerobot-v30.patch"
git -C XPolicyLab apply "$REPRO_ROOT/patches/xpolicylab-lerobot-v30.patch"
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
  apply --check "$REPRO_ROOT/patches/lingbot-vla-v2-rocm.patch"
git -C experiments/lingbot_vla_v2_6b_robotwin/source/lingbot-vla-v2 \
  apply "$REPRO_ROOT/patches/lingbot-vla-v2-rocm.patch"
```

Dockerfile 执行同一组补丁；宿主机数据准备只需应用 XPolicyLab 的转换补丁。
