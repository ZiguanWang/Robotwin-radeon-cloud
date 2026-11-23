# 可复现流程脚本

| 目录 | 用途 |
|---|---|
| `serving/` | LingBot 模型服务与推理测试 |
| `eval/` | RoboTwin 评测、固定 seed 配对评测、四卡并行评测、可选视频及空闲 GPU 自动发现 |
| `checkpoints/` | LoRA 和全参数 SFT checkpoint 转换 |
| `training/` | 全参数 SFT 与单任务 LoRA 启动 |
| `data/` | LeRobot 数据转换及本机 external-data 挂载目录准备 |

训练 YAML 位于 `configs/lingbot_vla_v2_6b_robotwin/`。Dockerfile 用一次 `COPY`
将运行脚本放到 `/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin/scripts/`；
根目录 `.dockerignore` 排除了仅在宿主机使用的两个 `data/` 脚本。
LoRA/SFT Notebook 在各自的代码单元中调用容器内脚本并编排训练、转换与评测。

本机准备默认镜像的挂载数据：

```bash
bash scripts/data/prepare_external_data.sh --dry-run
bash scripts/data/prepare_external_data.sh
```

脚本直接在宿主机准备 `/models/robotwin-persistent`：下载仿真资产、50 个 clean 示范
任务及基础模型，转换缺失的 LeRobot v3 任务，验证后生成训练清单。缺少 Python 依赖时，
会在目标目录的 `.preparation_runtime/` 中创建隔离环境并安装；缓存和转换日志也存放在此。
无需运行容器或 GPU 评测。
