#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from benchmark_orion_msp_classification_amd import (
    DEFAULT_BENCHMARKS,
    OFFICIAL_CKPT_NAME,
    ResultRow,
    build_tasks,
    collect_worker_outputs,
    parse_benchmark_specs,
    run_worker,
    sanitize_dataset_id,
    write_summary,
)


DEFAULT_FAILED_DATASETS = [
    "OpenML-ID-1468.csv",
    "OpenML-ID-1485.csv",
    "OpenML-ID-41147.csv",
]


def filter_tasks_by_dataset_names(tasks, dataset_names: List[str]):
    normalized_names = set()
    for name in dataset_names:
        cleaned = name.strip()
        if not cleaned:
            continue
        normalized_names.add(cleaned)
        if not cleaned.endswith(".csv"):
            normalized_names.add(f"{cleaned}.csv")

    filtered = [
        task
        for task in tasks
        if Path(task.csv_path).name in normalized_names or sanitize_dataset_id(Path(task.csv_path)) in normalized_names
    ]

    discovered: Dict[str, int] = {}
    for task in filtered:
        discovered[task.benchmark] = discovered.get(task.benchmark, 0) + 1
    return filtered, discovered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--dataset-names", default=",".join(DEFAULT_FAILED_DATASETS))
    parser.add_argument("--out-dir", default="results/OrionMSP_official_classification_failed_rerun")
    parser.add_argument("--model-path", default=f"ckpt/{OFFICIAL_CKPT_NAME}")
    parser.add_argument("--checkpoint-version", default=OFFICIAL_CKPT_NAME)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=32)
    parser.add_argument("--norm-methods", default="none,power")
    parser.add_argument("--feat-shuffle", default="latin")
    parser.add_argument("--softmax-temp", type=float, default=0.9)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-shift", action="store_true")
    parser.add_argument("--no-average-logits", action="store_true")
    parser.add_argument("--no-hierarchical", action="store_true")
    parser.add_argument("--no-auto-download", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    try:
        root = root.resolve()
    except Exception:
        pass
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    model_path = Path(args.model_path).expanduser()
    try:
        model_path = model_path.resolve()
    except Exception:
        pass

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    benchmark_names = [name for name, _ in parse_benchmark_specs(root, benchmark_specs)]
    tasks, _ = build_tasks(root, benchmark_specs)
    if not tasks:
        raise FileNotFoundError("No single-file classification CSVs found in the configured benchmark directories.")

    dataset_names = [x.strip() for x in args.dataset_names.split(",") if x.strip()]
    tasks, discovered = filter_tasks_by_dataset_names(tasks, dataset_names)
    discovered = {name: discovered.get(name, 0) for name in benchmark_names}
    if not tasks:
        raise FileNotFoundError("No datasets matched --dataset-names: " + ", ".join(dataset_names))

    if args.verbose:
        print(
            "task_scheduling: dynamic_queue priority=(rows desc, cols desc, benchmark asc, path asc) "
            f"task_count={len(tasks)}"
        )

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    if len(gpu_ids) != args.workers:
        raise ValueError(f"--gpus must contain exactly {args.workers} ids")

    norm_methods = [x.strip() for x in args.norm_methods.split(",") if x.strip()]
    model_kwargs: Dict = {
        "model_path": str(model_path),
        "allow_auto_download": not args.no_auto_download,
        "checkpoint_version": args.checkpoint_version,
        "batch_size": args.batch_size,
        "n_estimators": args.n_estimators,
        "norm_methods": norm_methods,
        "feat_shuffle_method": args.feat_shuffle,
        "class_shift": not args.no_class_shift,
        "softmax_temperature": args.softmax_temp,
        "average_logits": not args.no_average_logits,
        "use_hierarchical": not args.no_hierarchical,
        "use_amp": not args.no_amp,
        "device": args.device,
        "verbose": False,
        "random_state": args.random_state,
    }

    if not model_path.exists() and args.no_auto_download:
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    start_time = time.time()
    task_queue: mp.Queue = mp.Queue()
    ready_queue: mp.Queue = mp.Queue()
    start_event = mp.Event()
    processes: List[mp.Process] = []

    for task in tasks:
        task_queue.put((task.benchmark, task.csv_path, task.n_rows, task.n_cols))
    for _ in range(args.workers):
        task_queue.put(None)

    for worker_id in range(args.workers):
        proc = mp.Process(
            target=run_worker,
            args=(
                worker_id,
                gpu_ids[worker_id],
                task_queue,
                ready_queue,
                start_event,
                str(out_dir / f"worker_{worker_id}.csv"),
                dict(model_kwargs),
                args.test_size,
                args.random_state,
                args.verbose,
            ),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    ready_workers: set[int] = set()
    while len(ready_workers) < args.workers:
        try:
            message = ready_queue.get(timeout=10)
        except Exception:
            dead_workers = [
                str(idx)
                for idx, proc in enumerate(processes)
                if not proc.is_alive() and idx not in ready_workers
            ]
            if dead_workers:
                raise RuntimeError(
                    "Some workers exited before initialization completed: " + ", ".join(dead_workers)
                )
            continue

        if message.get("status") == "ready":
            ready_workers.add(int(message["worker_id"]))
            if args.verbose:
                print(
                    f"[worker {message['worker_id']} | gpu {message['gpu_id']}] "
                    f"ready scheduling={message.get('schedule_mode', 'dynamic_queue')}"
                )
            continue

        if message.get("status") == "crash":
            raise RuntimeError(
                f"Worker {message['worker_id']} on gpu {message['gpu_id']} crashed "
                f"during initialization:\n{message.get('error', '(no traceback)')}"
            )

    start_event.set()

    for proc in processes:
        proc.join()

    dfs = collect_worker_outputs(out_dir, args.workers)
    columns = list(ResultRow.__annotations__.keys())
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)
    all_csv = out_dir / "all_classification_results.csv"
    all_df.to_csv(all_csv, index=False)

    wall_seconds = time.time() - start_time
    write_summary(out_dir / "summary.txt", all_df, sum(discovered.values()), wall_seconds)

    for benchmark in benchmark_names:
        benchmark_dir = out_dir / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        benchmark_df = (
            all_df[all_df["benchmark"] == benchmark].copy()
            if len(all_df)
            else pd.DataFrame(columns=columns)
        )
        benchmark_df.to_csv(benchmark_dir / "all_classification_results.csv", index=False)
        write_summary(benchmark_dir / "summary.txt", benchmark_df, discovered.get(benchmark, 0), wall_seconds)

    print(f"saved_all_csv: {all_csv}")
    print(f"saved_summary: {out_dir / 'summary.txt'}")
    for benchmark in benchmark_names:
        print(f"{benchmark}: {out_dir / benchmark / 'summary.txt'}")


if __name__ == "__main__":
    main()
