#!/usr/bin/env python3
"""Run every RoboTwin clean task with one model server per GPU."""

from __future__ import annotations

import argparse
import os
import queue
import re
import signal
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all RoboTwin clean tasks with a dynamic 1/2/4-GPU queue."
    )
    parser.add_argument("--gpu-count", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runtime-dir", type=Path, default=Path("/workspace/runtime"))
    parser.add_argument("--robotwin-root", type=Path, default=Path("/RoboTwin"))
    parser.add_argument("--python", default="/opt/robotwin-env/bin/python")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Hugging Face checkpoint directory; defaults to the official LingBot-VLA-v2 model.",
    )
    parser.add_argument("--base-port", type=int, default=13400)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_line(path: Path, value: str, lock: threading.Lock) -> None:
    with lock, path.open("a", encoding="utf-8") as file:
        file.write(value + "\n")


def stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)


def wait_for_server(process: subprocess.Popen[bytes], port: int, log_path: Path) -> None:
    for _ in range(600):
        if process.poll() is not None:
            raise RuntimeError(
                f"Model server on port {port} exited with code {process.returncode}; "
                f"inspect {log_path}"
            )
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2).read()
            return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Model server on port {port} did not become ready; inspect {log_path}")


def read_result(log_path: Path) -> tuple[int, int]:
    pattern = re.compile(r"Final success rate:\s*(\d+)/(\d+)\s*=\s*([0-9.]+)%")
    text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Missing final success rate in {log_path}")
    task_successes, task_episodes, _ = matches[-1]
    return int(task_successes), int(task_episodes)


def collect_results(tasks: list[str], logs_dir: Path) -> tuple[int, int]:
    successes = 0
    episodes = 0
    for task in tasks:
        log_path = logs_dir / f"eval_{task}.log"
        task_successes, task_episodes = read_result(log_path)
        successes += task_successes
        episodes += task_episodes
    return successes, episodes


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    root = args.robotwin_root.resolve()
    run_name = args.run_name or f"clean50x50_{args.gpu_count}gpu"
    run_dir = args.runtime_dir.resolve() / "outputs" / run_name
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"{run_dir} is not empty; choose a new --run-name or pass --resume")
    logs_dir = run_dir / "logs"
    done_dir = run_dir / "done"
    logs_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    with (root / "env_cfg/eval/all_tasks.yml").open(encoding="utf-8") as file:
        tasks = list(yaml.safe_load(file)["tasks"])
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise RuntimeError(f"Expected 50 unique tasks, got {len(tasks)}")

    gpu_count = int(
        subprocess.check_output(
            [args.python, "-c", "import torch; print(torch.cuda.device_count())"], text=True
        ).strip()
    )
    if gpu_count < args.gpu_count:
        raise RuntimeError(f"Requested {args.gpu_count} GPUs, found {gpu_count}")

    completed = {path.stem for path in done_dir.glob("*.done")}
    unknown_markers = completed - set(tasks)
    if unknown_markers:
        raise RuntimeError(f"Unknown completion markers: {sorted(unknown_markers)}")
    pending = [task for task in tasks if task not in completed]
    print(f"run={run_name} completed={len(completed)} pending={len(pending)}")

    if not pending:
        successes, episode_count = collect_results(tasks, logs_dir)
        print(f"overall success: {successes}/{episode_count} = {100 * successes / episode_count:.2f}%")
        print(f"results: {run_dir}")
        return

    timing_file = run_dir / "task_times.tsv"
    events_file = run_dir / "events.log"
    failures_file = run_dir / "failures.tsv"
    if not (run_dir / "run_start_iso").exists():
        (run_dir / "run_start_iso").write_text(utc_now().isoformat() + "\n", encoding="utf-8")

    server_script = root / "experiments/lingbot_vla_v2_6b_robotwin/scripts/launch_official_server.sh"
    model_path = (
        args.model_path.expanduser().resolve()
        if args.model_path is not None
        else root
        / "experiments/lingbot_vla_v2_6b_robotwin/models/"
        "robbyant_lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt"
    )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    server_processes: list[subprocess.Popen[bytes]] = []
    server_handles = []
    eval_processes: set[subprocess.Popen[bytes]] = set()
    process_lock = threading.Lock()
    write_lock = threading.Lock()
    stop_event = threading.Event()

    def cleanup() -> None:
        stop_event.set()
        with process_lock:
            running_evals = list(eval_processes)
        for process in running_evals:
            stop_process_group(process)
        for process in server_processes:
            stop_process_group(process)
        for handle in server_handles:
            handle.close()

    def evaluate(gpu_id: int, task: str) -> None:
        port = args.base_port + gpu_id
        started = utc_now()
        append_line(events_file, f"{started.isoformat()} gpu={gpu_id} task={task} start", write_lock)
        log_path = logs_dir / f"eval_{task}.log"
        command = [
            args.python,
            str(root / "scripts/eval_policy_xpolicylab.py"),
            "--task_name",
            task,
            "--task_config",
            "demo_clean",
            "--policy_name",
            "LingBot-VLA-v2",
            "--protocol",
            "lingbot_vla_v2",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device_id",
            "0",
            "--seed",
            "0",
            "--test_num",
            str(args.episodes),
            "--expert_check",
            "false",
            "--eval_batch",
            "false",
        ]
        env = os.environ.copy()
        env.update(
            HIP_VISIBLE_DEVICES=str(gpu_id),
            ROBOTWIN_DISABLE_CUROBO="1",
            ROBOTWIN_EE_PLANNER="mplib",
            PYOPENGL_PLATFORM="egl",
        )
        env.pop("ROCR_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        with log_path.open("wb") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            with process_lock:
                eval_processes.add(process)
            status = process.wait()
            with process_lock:
                eval_processes.discard(process)

        ended = utc_now()
        elapsed = int((ended - started).total_seconds())
        append_line(
            events_file,
            f"{ended.isoformat()} gpu={gpu_id} task={task} elapsed={elapsed}s status={status}",
            write_lock,
        )
        if status != 0:
            append_line(failures_file, f"{task}\t{gpu_id}\t{elapsed}\t{status}", write_lock)
            raise RuntimeError(f"{task} failed on GPU {gpu_id}; inspect {log_path}")
        _, completed_episodes = read_result(log_path)
        if completed_episodes != args.episodes:
            raise RuntimeError(
                f"{task} reported {completed_episodes} episodes, expected {args.episodes}"
            )
        append_line(
            timing_file,
            f"{task}\t{gpu_id}\t{started.isoformat()}\t{ended.isoformat()}\t{elapsed}\t0",
            write_lock,
        )
        (done_dir / f"{task}.done").touch()
        print(f"[{len(list(done_dir.glob('*.done')))}/50] {task}: {elapsed}s on GPU {gpu_id}")

    task_queue: queue.Queue[str] = queue.Queue()
    for task in pending:
        task_queue.put(task)

    def worker(gpu_id: int) -> None:
        while not stop_event.is_set():
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                evaluate(gpu_id, task)
            except Exception:
                stop_event.set()
                raise
            finally:
                task_queue.task_done()

    started_perf = time.perf_counter()
    try:
        for gpu_id in range(args.gpu_count):
            port = args.base_port + gpu_id
            log_path = logs_dir / f"server_gpu{gpu_id}.log"
            handle = log_path.open("ab")
            server_handles.append(handle)
            command = [
                "bash",
                str(server_script),
                str(gpu_id),
                str(port),
                str(log_path),
                str(args.use_compile),
                str(model_path),
            ]
            server_processes.append(
                subprocess.Popen(
                    command,
                    cwd=root,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        for gpu_id, process in enumerate(server_processes):
            port = args.base_port + gpu_id
            wait_for_server(process, port, logs_dir / f"server_gpu{gpu_id}.log")
            print(f"GPU {gpu_id} server ready on port {port}")

        with ThreadPoolExecutor(max_workers=args.gpu_count) as executor:
            futures = [executor.submit(worker, gpu_id) for gpu_id in range(args.gpu_count)]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                stop_event.set()
                with process_lock:
                    running_evals = list(eval_processes)
                for process in running_evals:
                    stop_process_group(process)
                for future in futures:
                    future.cancel()
                raise
    except KeyboardInterrupt:
        print("Interrupted; completed tasks can be resumed with --resume")
        raise
    finally:
        cleanup()

    markers = {path.stem for path in done_dir.glob("*.done")}
    missing = sorted(set(tasks) - markers)
    if missing:
        raise RuntimeError(f"Benchmark incomplete; missing tasks: {missing}")
    successes, episode_count = collect_results(tasks, logs_dir)
    expected_episodes = len(tasks) * args.episodes
    if episode_count != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} episodes, found {episode_count}")
    elapsed = time.perf_counter() - started_perf
    print(f"benchmark invocation: {elapsed / 3600:.2f} h")
    print(f"overall success: {successes}/{episode_count} = {100 * successes / episode_count:.2f}%")
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
