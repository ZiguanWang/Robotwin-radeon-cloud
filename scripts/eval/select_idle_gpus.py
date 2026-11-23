"""Select idle ROCm GPUs in HIP order and report their model and architecture."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


VISIBILITY_VARS = (
    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
)


def rocm_json(*options: str) -> dict:
    return json.loads(subprocess.check_output(
        ["rocm-smi", *options, "--json"], text=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--max-used-gib", type=float, default=1.0)
    args = parser.parse_args()
    if args.count < 1 or args.max_used_gib <= 0:
        raise ValueError("Select a positive GPU count and memory threshold")

    for name in VISIBILITY_VARS:
        os.environ.pop(name, None)
    resolver = Path(__file__).with_name("resolve_dri_prime.py")
    selectors = subprocess.check_output(
        [sys.executable, str(resolver), "--all"], text=True).strip().split(",")
    buses = rocm_json("--showbus")
    memory = rocm_json("--showmeminfo", "vram")
    products = rocm_json("--showproductname")

    try:
        import torch
    except ImportError:
        torch = None
    cards = {}
    idle = []
    for gpu, selector in enumerate(selectors):
        parts = selector.removeprefix("pci-").removesuffix("!").split("_")
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected HIP render selector: {selector}")
        bus = f"{parts[0]}:{parts[1]}:{parts[2]}.{parts[3]}"
        card = next((name for name, value in buses.items()
                     if value["PCI Bus"].lower() == bus.lower()), None)
        if card is None:
            raise RuntimeError(f"No rocm-smi card for HIP GPU {gpu} / {bus}")
        used = int(memory[card]["VRAM Total Used Memory (B)"]) / 2**30
        product = products.get(card, {})
        properties = (torch.cuda.get_device_properties(gpu)
                      if torch is not None and torch.cuda.is_available() else None)
        model = (product.get("Card Series") or
                 (getattr(properties, "name", None) if properties is not None else None) or
                 product.get("Card Model") or "unknown")
        arch = (product.get("GFX Version") or
                (getattr(properties, "gcnArchName", None) if properties is not None else None) or
                "unknown")
        cards[str(gpu)] = {"card": card, "pci_bus": bus, "dri_prime": selector,
                           "model": model, "arch": arch,
                           "vram_used_gib": round(used, 3)}
        if used < args.max_used_gib:
            idle.append(gpu)
    if len(idle) < args.count:
        raise RuntimeError(f"Need {args.count} idle ROCm GPUs; found {len(idle)} of {len(selectors)}")
    selected = idle[:args.count]
    print(json.dumps({"gpu_ids": selected,
                      "gpu_cards": {str(gpu): cards[str(gpu)] for gpu in selected}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
