#!/usr/bin/env python3
"""Convert a full-parameter LingBot-VLA DCP checkpoint to deployable HF weights."""

import argparse
import shutil
from pathlib import Path

from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)
    model_assets = args.training_output / "model_assets"
    if not model_assets.is_dir():
        raise FileNotFoundError(model_assets)

    state = ckpt_to_state_dict(args.checkpoint, args.training_output, "dcp", False)
    lora_keys = [key for key in state if ".lora_" in key or ".base_layer." in key]
    if lora_keys:
        raise ValueError(
            "This is a LoRA checkpoint; use merge_lora_dcp.py instead: "
            f"{lora_keys[:5]}"
        )

    args.output.mkdir(parents=True)
    for asset in model_assets.iterdir():
        if asset.is_file() and not asset.name.endswith((".safetensors", ".bin", ".index.json")):
            shutil.copy2(asset, args.output / asset.name)
    save_model_weights(args.output, state)
    print(f"Converted full-SFT checkpoint to {args.output}")


if __name__ == "__main__":
    main()
