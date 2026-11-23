"""Merge a small LoRA adapter file into sharded base weights with bounded RAM.

This is for a constrained single-W7900 container. It matches the LoRA update
used by merge_lora_dcp.py while reading only one base safetensors shard at a time.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TRAINING_ONLY_PREFIXES = (
    "model.current_video_align_",
    "model.depth_align_",
    "model.future_depth_align_",
    "model.future_video_align_",
    "model.current_shared_task_proj",
    "model.future_shared_task_proj",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with safe_open(args.adapters, framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        rank = int(meta["rank"])
        alpha = float(meta["alpha"])
        adapters = {key: handle.get_tensor(key) for key in handle.keys()}
    a_suffix = ".lora_A.default.weight"
    prefixes = {key[: -len(a_suffix)] for key in adapters if key.endswith(a_suffix)}
    if len(prefixes) * 2 != len(adapters):
        raise RuntimeError("Incomplete or unexpected LoRA adapter keys")
    index = json.loads((args.base_model / "model.safetensors.index.json").read_text())
    output_map: dict[str, str] = {}
    used: set[str] = set()
    total_bytes = 0
    args.output.mkdir(parents=True)
    for asset in args.base_model.iterdir():
        if asset.is_file() and not asset.name.endswith((".safetensors", ".bin", ".index.json")):
            shutil.copy2(asset, args.output / asset.name)
    for shard in sorted(set(index["weight_map"].values())):
        merged: dict[str, torch.Tensor] = {}
        with safe_open(args.base_model / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(TRAINING_ONLY_PREFIXES):
                    continue
                base = handle.get_tensor(key)
                if base.dtype != torch.float32:
                    raise TypeError(f"Expected FP32 base weight: {key} {base.dtype}")
                prefix = key[:-7] if key.endswith(".weight") else ""
                if prefix in prefixes:
                    a = adapters[prefix + a_suffix]
                    b = adapters[prefix + ".lora_B.default.weight"]
                    base = base + (alpha / rank) * (b @ a)
                    used.add(prefix)
                result = base.to(torch.bfloat16).contiguous()
                merged[key] = result
                output_map[key] = shard
                total_bytes += result.numel() * result.element_size()
        save_file(merged, args.output / shard)
        print(f"Wrote {shard}: {len(merged)} tensors", flush=True)
        del merged
    if used != prefixes or len(output_map) != 1624:
        raise RuntimeError(
            f"Merge coverage mismatch: used {len(used)}/{len(prefixes)}, model keys {len(output_map)}"
        )
    output_index = {"metadata": {"total_size": total_bytes}, "weight_map": output_map}
    (args.output / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2) + "\n"
    )
    print(f"Merged {len(used)} LoRA layers; {len(output_map)} tensors; {total_bytes} bytes")


if __name__ == "__main__":
    main()
