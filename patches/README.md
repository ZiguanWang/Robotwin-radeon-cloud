# Source patches

[Chinese (Simplified)](./README.zh-CN.md)

Docker builds and manual reproduction use the pinned source revisions below. Apply each patch after checking out its source and before the editable installation. Do not apply these patches to different revisions without reviewing them first.

| Repository | Pinned commit | Patch |
|---|---|---|
| RoboTwin | `e154702de28cac4b9bfe0299044303f507236bb3` | `robotwin-rocm-reproduction.patch` |
| XPolicyLab (RoboTwin submodule) | `c37109c500be67d0dea6b36bf7337bbd26e763cd` | `xpolicylab-lerobot-v30.patch` |
| LingBot-VLA-v2 | `951475ae1b1d87553e7dc47c97b53a3d695c0d13` | `lingbot-vla-v2-rocm.patch` |

## RoboTwin: `robotwin-rocm-reproduction.patch`

| Modified file | Purpose |
|---|---|
| `envs/_base_task.py` | Reduce SAPIEN log volume and disable OIDN ray-tracing denoising unsupported by the current RADV path. |
| `envs/robot/planner.py` | Allow policy rollout initialization without CuRobo, and provide an MPLib-compatible `plan_batch` that plans candidate targets one by one. CuRobo-only operations still fail explicitly. |
| `envs/robot/robot.py` | Select MPLib with `ROBOTWIN_EE_PLANNER` and route single-target and batched end-effector planning for both arms through the corresponding MPLib planner. |
| `scripts/eval_policy_xpolicylab.py` | Integrate LingBot's `WebsocketClientPolicy` and observation/action protocol; support explicit seed start, instruction and video overrides; record inference timing; and require non-batch evaluation for the LingBot protocol. Preserve the pinned upstream revision's unstable-seed counting behavior. |

Closed-loop evaluation, ROCm rendering, and the MPLib path still depend on this patch; it cannot be removed as a whole. Log-level and inference-timing changes are auxiliary and do not make the compatibility changes unnecessary.

## XPolicyLab: `xpolicylab-lerobot-v30.patch`

This patch changes only `scripts/transform_lerobot_v30_format.py`:

- Adapt the LeRobot 0.6 `RGBEncoderConfig`, dataset writer, and `finalize()` APIs.
- Use PyAV for reading, CPU H.264 encoding, and an explicit CRF setting in place of the NVIDIA NVENC default; disable multicamera streaming encoding to avoid observed dropped frames.
- Keep joint state/action conversion as the default, and add optional `--control_mode ee` conversion that arranges each arm's end-effector pose and gripper as `[pose7, gripper] × 2`.

The host-side `scripts/data/prepare_external_data.sh` uses the joint conversion mode for 50 tasks, so the patch is still needed in full. The EE branch is not used by the default workflow. Removing it would remove manual EE conversion; if only joint format will be maintained, trim that branch separately and revalidate the converter.

## LingBot-VLA-v2: `lingbot-vla-v2-rocm.patch`

| Modified file | Purpose |
|---|---|
| `deploy/lingbot_vla_v2_policy.py` | Ignore full-SFT alignment weights unused by the inference model while rejecting other unknown weights; find the training YAML from an explicit path or experiment configuration; disable the single-shape ViT grid cache; accept episode reset seeds. The workshop MoE fallback activates only with its specified environment variable. |
| `lingbotvla/data/vla_data/base_dataset.py` | Adapt LeRobot 0.6 transform imports, `DatasetReader`, and PyAV video reading. |
| `lingbotvla/data/vla_data/multi_vla_dataset.py` | Resolve legacy `/workspace/RoboTwin` absolute paths in old manifests to the current `/RoboTwin`. |
| `lingbotvla/data/vla_data/utils.py` | Resolve relative robot normalization-statistics paths against the LingBot repository instead of the caller's current directory. |
| `lingbotvla/utils/arguments.py` | Prevent `null` and empty YAML lists from overriding dataclass defaults for training arguments. |
| `lingbotvla/utils/lora_utils.py` | Restrict LoRA target modules with a regular expression and return the model after adapter injection. |
| `tasks/vla/train_lingbotvla.py` | Add LoRA choices for the action expert or all attention layers, freeze base weights, and report trainable-parameter counts. |
| `lingbotvla/models/vla/lingbot_vla/qwen2_action_expert.py` | Define the logger already referenced by upstream exception and sliding-window branches to avoid `NameError`. |

Model serving, LoRA training, and LeRobot v3 reading depend on these changes, so the patch cannot be removed as a whole. Legacy-path compatibility can be reviewed separately; confirm all runtime manifests have migrated before removing it.

## Manual application and checks

After checking out the pinned revisions above:

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

The Dockerfile applies the same three patches. Host-side data preparation needs only the XPolicyLab conversion patch.
