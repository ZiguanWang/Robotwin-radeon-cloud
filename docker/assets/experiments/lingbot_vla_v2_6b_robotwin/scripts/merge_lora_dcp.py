#!/usr/bin/env python3
"""Convert a LingBot-VLA LoRA DCP checkpoint into deployable merged HF weights."""

import argparse
import shutil
from pathlib import Path

import torch

from lingbotvla.checkpoint import ckpt_to_state_dict
from lingbotvla.models import save_model_weights


def merge_lora_state_dict(state_dict, alpha: float, rank: int):
    suffix = ".lora_A.default.weight"
    adapter_prefixes = [key[: -len(suffix)] for key in state_dict if key.endswith(suffix)]
    if not adapter_prefixes:
        raise ValueError("No LoRA adapter tensors found in checkpoint")
    merged = dict(state_dict)
    for prefix in adapter_prefixes:
        base_key = prefix + ".base_layer.weight"
        a_key = prefix + suffix
        b_key = prefix + ".lora_B.default.weight"
        if base_key not in merged or b_key not in merged:
            raise KeyError(f"Incomplete LoRA tensors for {prefix}")
        base = merged.pop(base_key)
        a = merged.pop(a_key).to(torch.float32)
        b = merged.pop(b_key).to(torch.float32)
        merged[prefix + ".weight"] = (base.to(torch.float32) + (alpha / rank) * (b @ a)).to(base.dtype)
        base_bias_key = prefix + ".base_layer.bias"
        if base_bias_key in merged:
            merged[prefix + ".bias"] = merged.pop(base_bias_key)
    leftovers = [key for key in merged if ".lora_" in key or ".base_layer." in key]
    if leftovers:
        raise ValueError(f"Unmerged PEFT keys remain: {leftovers[:5]}")
    return merged, len(adapter_prefixes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()

    state = ckpt_to_state_dict(args.checkpoint, args.training_output, "dcp", False)
    state, count = merge_lora_state_dict(state, args.alpha, args.rank)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for asset in args.base_model.iterdir():
        if asset.is_file() and not asset.name.endswith((".safetensors", ".bin", ".index.json")):
            shutil.copy2(asset, args.output / asset.name)
    save_model_weights(args.output, state)
    print(f"Merged {count} LoRA layers into {args.output}")


if __name__ == "__main__":
    main()

