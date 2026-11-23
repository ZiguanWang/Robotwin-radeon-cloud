"""Load only LoRA tensors from a large DCP model checkpoint.

The full checkpoint retains frozen base-model weights, so this avoids loading
those weights into a small W7900 container's RAM.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader, load_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reader = FileSystemReader(str(args.model_dir))
    metadata = reader.read_metadata().state_dict_metadata
    prefix = "state.model."
    selected = {
        fqn[len(prefix):]: torch.empty(item.size, dtype=item.properties.dtype)
        for fqn, item in metadata.items()
        if fqn.startswith(prefix) and (".lora_A." in fqn or ".lora_B." in fqn)
    }
    if not selected or len(selected) % 2:
        raise RuntimeError(f"Unexpected adapter count: {len(selected)}")
    state = {"state": {"model": selected}}
    load_state_dict(
        state,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
        no_dist=True,
    )
    tensors = {key: value.contiguous() for key, value in selected.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        args.output,
        metadata={"rank": str(args.rank), "alpha": str(args.alpha)},
    )
    print(f"Extracted {len(tensors)} LoRA tensors to {args.output}")


if __name__ == "__main__":
    main()
