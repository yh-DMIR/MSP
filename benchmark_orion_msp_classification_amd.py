#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_BENCHMARKS = [
    "openml_cc18_csv=dataset/openml_cc18_72",
    "tabarena_cls=dataset/tabarena/cls",
    "tabzilla_csv=dataset/tabzilla35",
    "talent_cls=dataset/talent_cls",
]
TARGET_CANDIDATES = ["target", "label", "class", "y", "TARGET", "Label", "Class", "Y"]
OFFICIAL_CKPT_NAME = "Orion-MSP-v1.0.ckpt"


@dataclass
class ResultRow:
    benchmark: str
    dataset_id: str
    dataset_dir: str
    dataset_name: str
    n_train: int
    n_test: int
    n_features: int
    n_classes: Optional[int]
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    logloss: Optional[float]
    fit_seconds: float
    predict_seconds: float
    status: str
    error: Optional[str]


@dataclass(frozen=True)
class DatasetTask:
    benchmark: str
    csv_path: str
    n_rows: int
    n_cols: int


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def sanitize_dataset_id(csv_path: Path) -> str:
    match = re.search(r"(OpenML-ID-\d+)", str(csv_path))
    return match.group(1) if match else csv_path.stem


def infer_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return df.columns[-1]


def parse_benchmark_specs(root: Path, specs: List[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            name, rel_path = spec.split("=", 1)
            benchmark_name = name.strip()
            benchmark_path = Path(rel_path.strip())
        else:
            benchmark_path = Path(spec.strip())
            benchmark_name = benchmark_path.name

        if not benchmark_path.is_absolute():
            benchmark_path = (root / benchmark_path).resolve()
        else:
            benchmark_path = benchmark_path.resolve()
        parsed.append((benchmark_name, benchmark_path))
    return parsed


def discover_csv_files(benchmark_dir: Path) -> List[Path]:
    csv_files: List[Path] = []
    for csv_path in benchmark_dir.rglob("*.csv"):
        name = csv_path.name
        if name.endswith("_train.csv") or name.endswith("_test.csv"):
            continue
        csv_files.append(csv_path)
    return sorted(csv_files, key=lambda p: p.name)


def probe_csv_shape(csv_path: Path) -> Tuple[int, int]:
    try:
        n_cols = int(pd.read_csv(csv_path, nrows=0).shape[1])
    except Exception:
        n_cols = -1

    try:
        newline_count = 0
        saw_any_bytes = False
        last_byte = b""
        with csv_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                saw_any_bytes = True
                last_byte = chunk[-1:]
                newline_count += chunk.count(b"\n")
        physical_lines = newline_count
        if saw_any_bytes and last_byte not in (b"\n", b"\r"):
            physical_lines += 1
        n_rows = max(0, physical_lines - 1)
    except Exception:
        n_rows = -1

    return n_rows, n_cols


def build_tasks(root: Path, benchmark_specs: List[str]) -> Tuple[List[DatasetTask], Dict[str, int]]:
    tasks: List[DatasetTask] = []
    discovered: Dict[str, int] = {}
    for benchmark_name, benchmark_dir in parse_benchmark_specs(root, benchmark_specs):
        csv_files = discover_csv_files(benchmark_dir) if benchmark_dir.exists() else []
        discovered[benchmark_name] = len(csv_files)
        for csv_path in csv_files:
            n_rows, n_cols = probe_csv_shape(csv_path)
            tasks.append(
                DatasetTask(
                    benchmark=benchmark_name,
                    csv_path=str(csv_path),
                    n_rows=n_rows,
                    n_cols=n_cols,
                )
            )
    tasks.sort(key=lambda task: (-task.n_rows, -task.n_cols, task.benchmark, task.csv_path))
    return tasks, discovered


def evaluate_one_dataset(
    clf,
    benchmark: str,
    csv_path: Path,
    test_size: float,
    random_state: int,
) -> ResultRow:
    dataset_id = sanitize_dataset_id(csv_path)
    try:
        df = pd.read_csv(csv_path)
        target_col = infer_target_column(df)
        df = df.dropna(subset=[target_col])

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if len(X) < 2:
            raise ValueError("Not enough valid rows after dropping missing target.")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )

        t0 = time.time()
        clf.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        t1 = time.time()
        try:
            proba = clf.predict_proba(X_test)
            y_pred = clf.classes_[np.argmax(proba, axis=1)]
            ll = log_loss(y_test, proba, labels=clf.classes_)
        except Exception:
            y_pred = clf.predict(X_test)
            ll = None
        predict_seconds = time.time() - t1

        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=int(len(X_train)),
            n_test=int(len(X_test)),
            n_features=int(X_train.shape[1]),
            n_classes=int(y_train.nunique()),
            accuracy=float(accuracy_score(y_test, y_pred)),
            f1_weighted=float(f1_score(y_test, y_pred, average="weighted")),
            logloss=float(ll) if ll is not None else None,
            fit_seconds=float(fit_seconds),
            predict_seconds=float(predict_seconds),
            status="ok",
            error=None,
        )
    except Exception as exc:
        return ResultRow(
            benchmark=benchmark,
            dataset_id=dataset_id,
            dataset_dir=csv_path.parent.as_posix(),
            dataset_name=csv_path.name,
            n_train=0,
            n_test=0,
            n_features=0,
            n_classes=None,
            accuracy=None,
            f1_weighted=None,
            logloss=None,
            fit_seconds=0.0,
            predict_seconds=0.0,
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def run_worker(
    worker_id: int,
    gpu_id: int,
    task_queue,
    ready_queue,
    start_event,
    worker_out_csv: str,
    model_kwargs: Dict,
    test_size: float,
    random_state: int,
    verbose: bool,
) -> None:
    try:
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("GPU_DEVICE_ORDINAL", None)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        import torch
        from orion_msp.sklearn.classifier import OrionMSPClassifier

        worker_kwargs = dict(model_kwargs)
        if worker_kwargs.get("device", "cuda:0").startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("GPU backend is not available in this worker.")

        clf = OrionMSPClassifier(**worker_kwargs)

        ready_queue.put(
            {
                "worker_id": worker_id,
                "gpu_id": gpu_id,
                "status": "ready",
                "schedule_mode": "dynamic_queue",
            }
        )
        start_event.wait()

        rows: List[ResultRow] = []
        while True:
            item = task_queue.get()
            if item is None:
                break

            benchmark, csv_path_str, queued_rows, queued_cols = item
            row = evaluate_one_dataset(
                clf,
                benchmark=benchmark,
                csv_path=Path(csv_path_str),
                test_size=test_size,
                random_state=random_state,
            )
            rows.append(row)

            if verbose:
                if row.status == "ok":
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[ok] {benchmark}/{row.dataset_name} "
                        f"rows={queued_rows} cols={queued_cols} acc={row.accuracy:.6f}"
                    )
                else:
                    print(
                        f"[worker {worker_id} | gpu {gpu_id}] "
                        f"[fail] {benchmark}/{row.dataset_name} "
                        f"rows={queued_rows} cols={queued_cols} error={row.error}"
                    )
            clear_torch_cache()

        columns = list(ResultRow.__annotations__.keys())
        worker_df = pd.DataFrame([asdict(row) for row in rows]) if rows else pd.DataFrame(columns=columns)
        Path(worker_out_csv).parent.mkdir(parents=True, exist_ok=True)
        worker_df.to_csv(worker_out_csv, index=False)
    except Exception:
        error_text = traceback.format_exc()
        try:
            ready_queue.put(
                {
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "status": "crash",
                    "error": error_text,
                }
            )
        except Exception:
            pass
        pd.DataFrame(
            [
                {
                    "benchmark": "__worker__",
                    "dataset_id": f"__WORKER_CRASH__{worker_id}",
                    "dataset_dir": "__worker__",
                    "dataset_name": f"__WORKER_CRASH__{worker_id}",
                    "n_train": 0,
                    "n_test": 0,
                    "n_features": 0,
                    "n_classes": None,
                    "accuracy": None,
                    "f1_weighted": None,
                    "logloss": None,
                    "fit_seconds": 0.0,
                    "predict_seconds": 0.0,
                    "status": "fail",
                    "error": error_text,
                }
            ]
        ).to_csv(worker_out_csv, index=False)


def collect_worker_outputs(out_dir: Path, workers: int) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for worker_id in range(workers):
        worker_csv = out_dir / f"worker_{worker_id}.csv"
        if worker_csv.exists():
            try:
                dfs.append(pd.read_csv(worker_csv))
            except pd.errors.EmptyDataError:
                continue
    return dfs


def write_summary(summary_path: Path, result_df: pd.DataFrame, discovered_datasets: int, wall_seconds: float) -> None:
    ok_df = result_df[result_df["status"] == "ok"].copy() if len(result_df) else pd.DataFrame()
    failed_df = result_df[result_df["status"] == "fail"].copy() if len(result_df) else pd.DataFrame()

    lines = [
        f"discovered_datasets: {discovered_datasets}",
        f"processed_datasets: {len(result_df)}",
        f"ok_count: {len(ok_df)}",
        f"failed_count: {len(failed_df)}",
        (
            f"avg_accuracy_ok: {ok_df['accuracy'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["accuracy"].notna().any()
            else "avg_accuracy_ok: (none)"
        ),
        (
            f"avg_f1_weighted_ok: {ok_df['f1_weighted'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["f1_weighted"].notna().any()
            else "avg_f1_weighted_ok: (none)"
        ),
        (
            f"avg_logloss_ok: {ok_df['logloss'].dropna().mean():.6f}"
            if len(ok_df) and ok_df["logloss"].notna().any()
            else "avg_logloss_ok: (none)"
        ),
        f"wall_seconds: {wall_seconds:.3f}",
    ]

    if len(failed_df):
        failed_names = ", ".join(sorted(set(failed_df["dataset_name"].dropna().astype(str).tolist())))
        lines.append(f"failed_datasets: {failed_names}")
    else:
        lines.append("failed_datasets: (none)")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_BENCHMARKS))
    parser.add_argument("--out-dir", default="results/OrionMSP_official_classification")
    parser.add_argument("--model-path", default=f"ckpt/{OFFICIAL_CKPT_NAME}")
    parser.add_argument("--checkpoint-version", default=OFFICIAL_CKPT_NAME)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
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
    tasks, discovered = build_tasks(root, benchmark_specs)
    if not tasks:
        raise FileNotFoundError("No single-file classification CSVs found in the configured benchmark directories.")
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
