"""Run paired single-task rollouts inside the prepared RoboTwin image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
from video_files import find_video, name_videos

EXP = Path("/RoboTwin/experiments/lingbot_vla_v2_6b_robotwin")
BASE = EXP / "models/robbyant_lingbot-vla-v2-6b"
ROBOTWIN_CONFIG = EXP / "models/robbyant_lingbot-vla-v2-6b-robotwin/lingbotvla_cli.yaml"
ROBOTWIN_MODEL = EXP / "models/robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
PYTHON = Path(sys.executable)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def port_listening(port: int) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.5)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def resolve_dri(gpu: int) -> str:
    env = os.environ.copy()
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(key, None)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    script = Path(__file__).with_name("resolve_dri_prime.py")
    result = subprocess.run(
        [str(PYTHON), str(script), "--visible-one"],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not (result.startswith("pci-") and result.endswith("!")):
        raise RuntimeError(f"Cannot resolve ROCm GPU {gpu} render device: {result!r}")
    return result


def gpu_environment(gpu: int, cache: Path, dri: str | None = None,
                    training_config: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    short_tmp = Path("/tmp") / ("rtw_ev_" + hashlib.sha256(
        str(cache).encode()).hexdigest()[:10])
    short_tmp.mkdir(parents=True, exist_ok=True)
    for key in ("ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        env.pop(key, None)
    env.update({
        "HIP_VISIBLE_DEVICES": str(gpu),
        "AITER_USE_SYSTEM_TRITON": "1",
        "AITER_TRITON_ONLY": "1",
        "FLASH_ATTENTION_TRITON_AMD_ENABLE": "TRUE",
        "SETUPTOOLS_SCM_PRETEND_VERSION": "0.0.0",
        "QWEN3VL_PATH": str(EXP / "models/Qwen3-VL-4B-Instruct-config-tokenizer"),
        "LINGBOTVLA_TRAINING_CONFIG": str(
            training_config or EXP / "training/lingbotvla_cli.yaml"),
        "TMPDIR": str(short_tmp),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
    })
    if dri is not None:
        env.update({"DRI_PRIME": dri, "MESA_VK_DEVICE_SELECT": "", "ROBOTWIN_DISABLE_CUROBO": "1"})
    return env


def wait_for_server(process: subprocess.Popen, port: int) -> float:
    started = time.monotonic()
    while time.monotonic() - started < 900:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited before opening port {port}: rc={process.returncode}")
        if port_listening(port):
            return time.monotonic() - started
        time.sleep(2)
    raise TimeoutError(f"Policy server did not open port {port}")


def validate(path: Path, seeds: list[int]) -> tuple[int, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(seeds) or [int(row["seed"]) for row in rows] != seeds:
        raise RuntimeError(f"Seed/count mismatch in {path}")
    errors = [row for row in rows if row.get("error") is not None]
    if errors:
        raise RuntimeError(f"{len(errors)} episode errors in {path}")
    return sum(bool(row["success"]) for row in rows), rows


def run_one(
    label: str, condition: str, model: Path, *, task_name: str, output: Path, results: Path,
    source: Path, evaluator: Path, gpu: int, dri: str, port: int,
    seeds: list[int], video: bool, training_config: Path | None = None,
) -> dict:
    episodes = results / f"{label}_episodes.jsonl"
    rollouts = results / "rollouts" / label
    if episodes.exists():
        try:
            success, rows = validate(episodes, seeds)
        except (RuntimeError, json.JSONDecodeError, KeyError) as error:
            archived = episodes.with_name(f"{episodes.stem}.incomplete_{int(time.time())}.jsonl")
            episodes.rename(archived)
            print(f"{label}: archived incomplete result ({error})", flush=True)
        else:
            videos = sorted(p for p in rollouts.rglob("*.mp4") if p.stat().st_size > 0)
            if video and len(videos) != len(seeds):
                raise RuntimeError(f"{label}: expected {len(seeds)} videos, found {len(videos)}")
            if video:
                for row in rows:
                    find_video(rollouts, row)
            print(f"{label}: existing {success}/{len(seeds)} verified", flush=True)
            return {"label": label, "success": success, "episodes": len(seeds),
                    "video_files": [str(p) for p in videos], "resumed": True}

    cache = output / "cache" / label
    for subdir in ("tmp", "triton", "torch_extensions"):
        (cache / subdir).mkdir(parents=True, exist_ok=True)
    rollouts.mkdir(parents=True, exist_ok=True)
    server_env = gpu_environment(gpu, cache, training_config=training_config)
    server_env["ROBOTWIN_WORKSHOP_DISABLE_FUSED_MOE"] = "1"
    eval_env = gpu_environment(gpu, cache, dri, training_config=training_config)
    eval_env["PYTHONPATH"] = "/RoboTwin" + (
        ":" + eval_env["PYTHONPATH"] if eval_env.get("PYTHONPATH") else "")
    eval_env["ROBOTWIN_ROOT"] = "/RoboTwin"
    eval_env["ROBOTWIN_EVAL_OUTPUT_ROOT"] = str(rollouts)
    server_command = [str(PYTHON), "-u", "-m", "deploy.lingbot_vla_v2_policy",
        "--model_path", str(model), "--use_length", "50", "--use_bf16", "True",
        "--use_fp32", "False", "--use_compile", "False", "--port", str(port)]
    server_log = results / f"{label}_server.log"
    with server_log.open("w") as server_output:
        server = subprocess.Popen(
            server_command, cwd=source, env=server_env,
            stdout=server_output, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            startup_s = wait_for_server(server, port)
            eval_command = [str(PYTHON), "-u", str(evaluator),
                "--task_name", task_name, "--task_config", condition,
                "--instruction_type", "seen" if condition == "demo_clean" else "unseen",
                "--policy_name", "LingBot-VLA-v2",
                "--protocol", "lingbot_vla_v2", "--host", "127.0.0.1",
                "--port", str(port), "--seed", "0",
                "--seed_list", str(results / "paired_eval_seeds.json"),
                "--test_num", str(len(seeds)), "--expert_check", "false",
                "--eval_batch", "false",
                "--additional_info", f"eval_video_log={str(video).lower()}",
                "--timing_output", str(episodes)]
            started = time.monotonic()
            eval_log = results / f"{label}_eval.log"
            with eval_log.open("w") as eval_output:
                completed = subprocess.run(eval_command, cwd="/RoboTwin", env=eval_env,
                    stdout=eval_output, stderr=subprocess.STDOUT)
            eval_s = time.monotonic() - started
            if completed.returncode:
                raise RuntimeError(f"{label}: evaluator rc={completed.returncode}; see {eval_log}")
            success, rows = validate(episodes, seeds)
            videos = (name_videos(rollouts, rows) if video else
                      sorted(p for p in rollouts.rglob("*.mp4") if p.stat().st_size > 0))
            if video and len(videos) != len(seeds):
                raise RuntimeError(f"{label}: expected {len(seeds)} MP4s, found {len(videos)}; see {eval_log}")
            record = {"label": label, "condition": condition, "model": str(model),
                "success": success, "episodes": len(seeds), "server_startup_s": round(startup_s, 1),
                "eval_s": round(eval_s, 1), "gpu": gpu, "dri_prime": dri,
                "video_files": [str(p) for p in videos]}
            print(f"{label}: {success}/{len(seeds)}, video files {len(videos)}, "
                  f"startup {startup_s:.0f}s, rollout {eval_s:.0f}s", flush=True)
            return record
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--result-tag", required=True)
    parser.add_argument("--hip-gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=13690)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--method", choices=("baseline", "lora", "sft", "robotwin"), required=True)
    parser.add_argument("--scene", choices=("both", "clean", "random"), default="both")
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, help="Evaluate one seed from the paired list with --count 1")
    seed_group.add_argument("--seeds", help="Comma-separated subset from the paired seed list")
    parser.add_argument("--video", action="store_true", help="Export MP4s for the selected scene(s)")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    results = run_dir / "evaluations" / args.result_tag
    results.mkdir(parents=True, exist_ok=True)
    source = EXP / "source/lingbot-vla-v2"
    evaluator = REPO / "eval/paired_seed_evaluator.py"
    lora_model = run_dir / f"merged_checkpoint/global_step_{args.steps}/hf_ckpt_streamed"
    sft_model = run_dir / f"merged_checkpoint/global_step_{args.steps}/hf_ckpt"
    required = [BASE / "model.safetensors.index.json",
                source / "deploy/lingbot_vla_v2_policy.py", evaluator]
    if args.method == "lora":
        required.append(lora_model / "model.safetensors.index.json")
    elif args.method == "sft":
        required.append(sft_model / "model.safetensors.index.json")
    elif args.method == "robotwin":
        required.extend((ROBOTWIN_MODEL / "model.safetensors.index.json", ROBOTWIN_CONFIG))
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    seed_file = run_dir / "paired_eval_seeds.json"
    if not seed_file.is_file():
        seed_file = run_dir / "paired_20_valid_seeds.json"
    all_seeds = [int(value) for value in json.loads(seed_file.read_text())]
    if len(set(all_seeds)) != len(all_seeds) or not 1 <= args.count <= len(all_seeds):
        raise ValueError("Invalid unique seed count")
    if args.seeds is not None:
        try:
            seeds = [int(value) for value in args.seeds.split(",")]
        except ValueError as error:
            raise ValueError("--seeds must contain comma-separated integers") from error
        if (len(seeds) != args.count or len(set(seeds)) != len(seeds) or
                any(seed not in all_seeds for seed in seeds)):
            raise ValueError("--seeds must select the requested number of unique paired seeds")
    elif args.seed is None:
        seeds = all_seeds[:args.count]
    else:
        if args.count != 1 or args.seed not in all_seeds:
            raise ValueError("--seed requires --count 1 and a seed from the paired list")
        seeds = [args.seed]
    (results / "paired_eval_seeds.json").write_text(json.dumps(seeds, indent=2) + "\n")
    if port_listening(args.port):
        raise RuntimeError(f"Port {args.port} is already in use")
    dri = resolve_dri(args.hip_gpu)
    manifest = {"run_dir": str(run_dir), "result_tag": args.result_tag,
        "task_name": args.task_name,
        "gpu": args.hip_gpu, "dri_prime": dri, "seeds": seeds,
        "scene": args.scene, "selected_seed": args.seed,
        "base_index_sha256": sha256(BASE / "model.safetensors.index.json"),
        "policy_sha256": sha256(source / "deploy/lingbot_vla_v2_policy.py"),
        "evaluator_sha256": sha256(evaluator), "video": args.video}
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    model = {"baseline": BASE, "lora": lora_model, "sft": sft_model,
             "robotwin": ROBOTWIN_MODEL}[args.method]
    training_config = ROBOTWIN_CONFIG if args.method == "robotwin" else None
    records = []
    conditions = (("demo_clean", "clean"), ("demo_randomized", "random"))
    for condition, suffix in conditions:
        if args.scene != "both" and suffix != args.scene:
            continue
        label = f"{args.method}_{suffix}"
        records.append(run_one(label, condition, model, task_name=args.task_name,
            output=results, results=results,
            source=source, evaluator=evaluator, gpu=args.hip_gpu, dri=dri,
            port=args.port, seeds=seeds, video=args.video,
            training_config=training_config))
    summary_path = results / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update({record["label"]: record["success"] for record in records})
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    timing_path = results / "timing.json"
    previous = json.loads(timing_path.read_text()) if timing_path.is_file() else {"groups": []}
    groups = {item["label"]: item for item in previous["groups"]}
    for record in records:
        if not (record.get("resumed") and record["label"] in groups):
            groups[record["label"]] = record
    ordered = [groups[name] for name in ("baseline_clean", "baseline_random",
                                        "lora_clean", "lora_random",
                                        "sft_clean", "sft_random",
                                        "robotwin_clean", "robotwin_random") if name in groups]
    timing_path.write_text(json.dumps({
        "total_s": round(sum(item.get("server_startup_s", 0) + item.get("eval_s", 0)
                             for item in ordered), 1),
        "last_invocation_s": round(time.monotonic() - started, 1),
        "groups": ordered}, indent=2) + "\n")
    print("summary", summary, flush=True)


if __name__ == "__main__":
    main()
