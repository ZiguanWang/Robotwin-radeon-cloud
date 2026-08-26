#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.websocket_client_policy import WebsocketClientPolicy


def observation(seed: int):
    rng = np.random.default_rng(seed)
    return {
        "observation.images.cam_high": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.cam_left_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.images.cam_right_wrist": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.state": np.zeros(14, dtype=np.float32),
        "task": "adjust the bottle upright",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=13330)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = WebsocketClientPolicy("127.0.0.1", args.port)
    records = []
    for batch_size in args.batch_sizes:
        for repeat in range(args.repeats):
            client.infer({"reset": True, "robo_name": "robotwin"})
            items = [observation(1000 * batch_size + 10 * repeat + i) for i in range(batch_size)]
            request = items[0] if batch_size == 1 else {"batch": items}
            started = time.perf_counter()
            response = client.infer(request)
            wall_ms = (time.perf_counter() - started) * 1000
            action = np.asarray(response["action"])
            record = {
                "batch_size": batch_size,
                "repeat": repeat,
                "wall_ms": wall_ms,
                "server_infer_ms": response["server_timing"]["infer_ms"],
                "action_shape": list(action.shape),
                "env_steps_per_s": batch_size * 25 / (wall_ms / 1000),
                "action_chunks_per_s": batch_size / (wall_ms / 1000),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

