# RoboTwin ROCm 复现资料

本仓库用于在 AMD ROCm 环境下复现 LingBot-VLA-v2 与 RoboTwin 的推理、闭环评测、LoRA 训练和全参数 SFT 流程。默认面向 Radeon Cloud 上的 4× 或 8× W7900 实例。

## 文档和 Notebook

### [Reproduce_Guide.md](./Reproduce_Guide.md)

复现指南是完整的命令行参考，包含：

- Radeon Cloud、Docker 和环境检查；
- 官方 LingBot-VLA-v2 基础 checkpoint 的启动与推理；
- MPLib、`expert_check=true` 以及 clean/randomized 全量评测；
- 四卡/八卡 LoRA 训练、checkpoint 合并和评测；
- 四卡/八卡完整 depth/video teacher 全参数 SFT；
- DCP checkpoint 转换、模型重新启动和评测；
- Docker patch 的手动应用方式及各 patch 的作用。

需要复制命令到终端执行，或了解完整环境、参数和排错流程时，以这份文档为准。

### [RoboTwin_ROCm_Reproduction.ipynb](./RoboTwin_ROCm_Reproduction.ipynb)

Notebook 是交互式复现入口，与指南中的配置保持同步，按顺序覆盖：

1. 环境和 GPU 检查；
2. 官方基础模型服务与 `adjust_bottle` 10-episode 评测；
3. clean + randomized 的 4/8 卡、100 tasks × 10 episodes benchmark；
4. 4/8 卡 LoRA 训练、合并和评测；
5. 4/8 卡完整 teacher 全参数 SFT、DCP 转换和评测。

Notebook 中的长时间训练和全量 benchmark 默认不会自动执行，需要在对应代码单元中手动打开开关，并确认当前实例 GPU 数量、模型数据和持久化目录。Notebook 适合交互式运行、观察日志和复用中间变量；需要一次性、可审计的命令行流程时，请使用复现指南。

## 推荐使用顺序

1. 先阅读 [Reproduce_Guide.md](./Reproduce_Guide.md) 的环境、镜像和模型说明。
2. 按指南完成 Docker 或 Radeon Cloud 实例准备。
3. 在 JupyterLab 中打开 [RoboTwin_ROCm_Reproduction.ipynb](./RoboTwin_ROCm_Reproduction.ipynb)，按顺序执行检查、smoke test 和所需评测单元。
4. 需要正式训练或长时间 benchmark 时，使用 Notebook 中确认过的参数，切换到指南中的命令行入口并将输出写入 `/workspace/runtime`。

默认模型是官方 `robbyant/lingbot-vla-v2-6b` 基础 checkpoint；RoboTwin 后训练 checkpoint 只在显式指定训练后模型进行验证时使用。默认评测使用 MPLib、`expert_check=true`，并在 `play_once` 失败时保留可用的渲染信息继续评测。
