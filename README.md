# RoboTwin ROCm Reproduction Resources

[Chinese (Simplified)](./README.zh-CN.md)

This repository reproduces LingBot-VLA-v2 inference, closed-loop RoboTwin evaluation, LoRA training, and full-parameter SFT on AMD ROCm. The default target is a Radeon Cloud instance with four or eight W7900 GPUs.

The top-level `scripts/` directory groups serving, evaluation, training, checkpoint conversion, and data preparation helpers. Training YAML files live in `configs/`. Run `bash scripts/data/prepare_external_data.sh` on the host to prepare the assets, data, and base models mounted by the default image.

## Documentation and notebooks

### [Reproduce_Guide.md](./Reproduce_Guide.md)

The reproduction guide is the complete command-line reference. It covers:

- Radeon Cloud, Docker, and environment checks;
- serving and testing the official LingBot-VLA-v2 base checkpoint;
- MPLib, `expert_check=true`, and full clean/randomized evaluation;
- four- and eight-GPU LoRA training, checkpoint merging, and evaluation;
- four- and eight-GPU full-parameter SFT with depth/video teachers;
- DCP checkpoint conversion, model restart, and evaluation;
- the purpose and manual application of [source patches](./patches/README.md).

Use the guide when you need copyable terminal commands or details about the environment, parameters, and troubleshooting. A [Chinese version](./Reproduce_Guide.zh-CN.md) is also available.

### [RoboTwin_ROCm_Reproduction.ipynb](./notebooks/RoboTwin_ROCm_Reproduction.ipynb)

The notebook is the interactive reproduction entry point. Its settings match the guide, and its cells cover:

1. Environment and GPU checks;
2. The official base-model server and a 10-episode `adjust_bottle` evaluation;
3. A clean and randomized benchmark across 100 tasks × 10 episodes on four or eight GPUs;
4. LoRA training, merging, and evaluation on four or eight GPUs;
5. Full-parameter SFT with complete teachers, DCP conversion, and evaluation on four or eight GPUs.

Long training runs and full benchmarks do not execute automatically. Enable their cells deliberately after checking the GPU count, model data, and persistent output path. Use the notebook for interactive runs and inspecting logs; use the guide for a command-line workflow that is easier to audit.

### Single-task workshops

- [LoRA workshop](./notebooks/lora_workshop.ipynb): automatically selects one idle AMD ROCm GPU (such as an R9700 or W7900), trains for 25 steps on 50 clean `click_alarmclock` demonstrations, and compares results before and after training on 10 fixed seeds in each of the clean and randomized scenes.
- [Action-expert SFT workshop](./notebooks/sft_workshop.ipynb): trains the `turn_switch` action expert on four GPUs, compares the base model, trained model, and official RoboTwin checkpoint on the same fixed seeds, then records a successful video with the official checkpoint.

## Recommended order

1. Read the environment, image, and model sections of [Reproduce_Guide.md](./Reproduce_Guide.md).
2. Prepare Docker or a Radeon Cloud instance as described in the guide.
3. Open [RoboTwin_ROCm_Reproduction.ipynb](./notebooks/RoboTwin_ROCm_Reproduction.ipynb) in JupyterLab and run the checks, smoke test, and evaluation cells you need.
4. For formal training or long benchmarks, carry the parameters checked in the notebook into the guide's command-line entry points and write outputs under `/workspace/runtime`.

The default model is the official `robbyant/lingbot-vla-v2-6b` base checkpoint. Use the RoboTwin-trained checkpoint only when you select it explicitly. Default evaluation uses MPLib and `expert_check=true` and skips seeds that fail the expert check or have unstable environment initialization.
