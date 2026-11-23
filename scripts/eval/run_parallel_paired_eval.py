"""Evaluate paired clean/randomized seeds on four ROCm GPUs concurrently."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


WORKER = Path(__file__).with_name("run_paired_eval.py")


def rows(path: Path, expected_seeds: list[int]) -> list[dict]:
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if (len(values) != len(expected_seeds) or
            [int(value["seed"]) for value in values] != expected_seeds or
            any(value.get("error") is not None for value in values)):
        raise RuntimeError(f"Invalid paired episode results: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-tag", default="no_video")
    parser.add_argument("--gpu-ids", required=True, help="Four comma-separated physical HIP IDs")
    parser.add_argument("--port-base", type=int, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--method", choices=("baseline", "sft"), required=True)
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    if len(gpu_ids) != 4 or len(set(gpu_ids)) != 4:
        raise ValueError("Parallel evaluation needs four distinct GPU IDs")
    if args.count < 2 or args.count % 2:
        raise ValueError("Use an even number of paired seeds, normally 10")
    run_dir = args.run_dir.resolve()
    all_seeds = [int(value) for value in json.loads(
        (run_dir / "paired_eval_seeds.json").read_text())]
    if args.count > len(all_seeds) or len(set(all_seeds)) != len(all_seeds):
        raise ValueError("Invalid paired seed list or count")
    seeds = all_seeds[:args.count]
    results = run_dir / "evaluations" / args.result_tag
    results.mkdir(parents=True, exist_ok=True)
    (results / "paired_eval_seeds.json").write_text(json.dumps(seeds, indent=2) + "\n")
    labels = (f"{args.method}_clean", f"{args.method}_random")
    if all((results / f"{label}_episodes.jsonl").is_file() for label in labels):
        for label in labels:
            rows(results / f"{label}_episodes.jsonl", seeds)
        print(f"Verified existing four-GPU {args.method} results: {results}", flush=True)
        return

    started = time.monotonic()
    assignments = (
        ("clean", gpu_ids[0], seeds[::2], args.port_base),
        ("clean", gpu_ids[1], seeds[1::2], args.port_base + 1),
        ("random", gpu_ids[2], seeds[::2], args.port_base + 2),
        ("random", gpu_ids[3], seeds[1::2], args.port_base + 3),
    )
    workers = []
    try:
        for scene, gpu, shard_seeds, port in assignments:
            name = f"{args.method}_{scene}_gpu{gpu}"
            shard = results / "shards" / name
            shard.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(WORKER),
                       "--run-dir", str(run_dir),
                       "--result-tag", f"{args.result_tag}/shards/{name}",
                       "--hip-gpu", str(gpu), "--port", str(port),
                       "--count", str(len(shard_seeds)), "--steps", str(args.steps),
                       "--task-name", args.task_name, "--method", args.method,
                       "--scene", scene,
                       "--seeds", ",".join(map(str, shard_seeds))]
            log = (shard / "worker_console.log").open("w")
            try:
                process = subprocess.Popen(command, cwd=WORKER.parent,
                                           env=os.environ.copy(), stdout=log,
                                           stderr=subprocess.STDOUT)
            except Exception:
                log.close()
                raise
            workers.append((scene, gpu, shard_seeds, shard, process, log))
        failures = []
        for scene, gpu, _, shard, process, _ in workers:
            code = process.wait()
            if code:
                failures.append(f"{scene} GPU {gpu}: rc={code}; see {shard / 'worker_console.log'}")
        if failures:
            raise RuntimeError("Parallel evaluation failed: " + "; ".join(failures))
    finally:
        for _, _, _, _, _, log in workers:
            log.close()

    combined = {}
    groups = {}
    for scene in ("clean", "random"):
        label = f"{args.method}_{scene}"
        by_seed = {}
        shard_records = []
        for assigned_scene, gpu, shard_seeds, shard, _, _ in workers:
            if assigned_scene != scene:
                continue
            shard_rows = rows(shard / f"{label}_episodes.jsonl", shard_seeds)
            for row in shard_rows:
                seed = int(row["seed"])
                if seed in by_seed:
                    raise RuntimeError(f"Duplicate {scene} seed {seed}")
                by_seed[seed] = row
            timing = json.loads((shard / "timing.json").read_text())
            shard_records.append({"gpu": gpu, "seeds": shard_seeds,
                                  "timing": timing["groups"][0]})
        if set(by_seed) != set(seeds):
            raise RuntimeError(f"Missing {scene} seeds after four-GPU evaluation")
        ordered = [by_seed[seed] for seed in seeds]
        (results / f"{label}_episodes.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))
        combined[label] = sum(bool(row["success"]) for row in ordered)
        groups[label] = {"label": label, "success": combined[label],
                         "episodes": len(ordered), "shards": shard_records}

    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update(combined)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    timing_path = results / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.is_file() else {"groups": []}
    previous = {item["label"]: item for item in timing["groups"]}
    previous.update(groups)
    timing_path.write_text(json.dumps({
        "last_invocation_s": round(time.monotonic() - started, 1),
        "groups": [previous[label] for label in
                   ("baseline_clean", "baseline_random", "sft_clean", "sft_random")
                   if label in previous]}, indent=2) + "\n")
    (results / f"{args.method}_parallel_manifest.json").write_text(json.dumps({
        "gpu_ids": gpu_ids, "seeds": seeds, "assignments": [
            {"scene": scene, "gpu": gpu, "seeds": shard_seeds, "port": port}
            for scene, gpu, shard_seeds, port in assignments],
        "success": combined}, indent=2) + "\n")
    print(f"Four-GPU {args.method}: {combined}; "
          f"wall {time.monotonic() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
