#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dynamic multi-GPU benchmark runner for Orion-MSP (AMD/ROCm friendly).

This script mirrors the output layout and multi-process scheduling strategy of
`benchmark_tabicl_dynamic.py`, but evaluates `OrionMSPClassifier`.

Key features:
- Dynamic scheduling (work stealing): N worker processes pull datasets from a shared queue.
- Each worker binds to one GPU via HIP_VISIBLE_DEVICES=<gpu_id> and uses device "cuda:0".
- Writes per-worker CSVs, merges into one ALL CSV, and writes ONE global summary TXT:
  - discovered_pairs: number of valid (train,test) pairs found
  - processed_pairs: number of attempted datasets (ok + fail rows written)
  - missing_test_datasets: train exists but test missing
  - failed_datasets: datasets with status=fail (including worker crash marker if any)
  - avg_accuracy_ok: mean accuracy over all OK results
  - avg_accuracy_ok_top_{27,63,154}: mean accuracy of top-N datasets by accuracy (descending),
    computed only if #OK-with-accuracy >= N.
  - wall_seconds: total elapsed wall time of the whole run (from start to end)
  - wall_time_hms: wall time formatted as H:MM:SS
  - started_at / finished_at: timestamps (local time)

Example:
  python benchmark_orion_msp_dynamic.py \
    --root limix/tabzilla_csv \
    --out-dir results/msp_dynamic/tabzilla \
    --workers 8 \
    --ckpt ./Orion-MSP-v1.0.ckpt \
    --no-auto-download \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------
# Helpers: dataset discovery
# -----------------------------

TARGET_CANDIDATES = [
    "target", "label", "class", "y",
    "TARGET", "Label", "Class", "Y",
]


def _fmt_hms(seconds: float) -> str:
    """Format seconds into H:MM:SS."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def sanitize_dataset_id(train_path: Path) -> str:
    m = re.search(r"(OpenML-ID-\d+)", str(train_path))
    if m:
        return m.group(1)
    return train_path.stem.replace("_train", "").replace("_TRAIN", "")


def discover_train_test_pairs(root: Path) -> Tuple[List[Tuple[Path, Path, str]], List[str]]:
    """
    Returns:
        pairs: list of (train_csv, test_csv, dataset_id)
        missing_test_ids: train exists but no matching test
    """
    root = root.expanduser().resolve()
    trains = sorted(root.rglob("*_train.csv")) + sorted(root.rglob("*_TRAIN.csv"))
    pairs: List[Tuple[Path, Path, str]] = []
    missing: List[str] = []

    for tr in trains:
        dsid = sanitize_dataset_id(tr)
        test1 = Path(str(tr).replace("_train.csv", "_test.csv"))
        test2 = Path(str(tr).replace("_TRAIN.csv", "_TEST.csv"))
        te = test1 if test1.exists() else (test2 if test2.exists() else None)
        if te is None:
            missing.append(dsid)
        else:
            pairs.append((tr, te, dsid))

    return pairs, missing


def find_target_column(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    for c in TARGET_CANDIDATES:
        if c in cols:
            return c
    # fallback: last column
    return cols[-1]


# -----------------------------
# MSP evaluation (single dataset)
# -----------------------------

def eval_one_dataset(
    train_csv: Path,
    test_csv: Path,
    dataset_id: str,
    clf_kwargs: Dict,
    verbose: bool = False,
) -> Dict:
    """
    Runs one dataset and returns dict of metrics.
    This function is called inside each worker process.
    """
    t0 = time.time()

    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        target_col = find_target_column(train_df)
        if target_col not in test_df.columns:
            target_col = find_target_column(test_df)

        y_train = train_df[target_col]
        X_train = train_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        X_test = test_df.drop(columns=[target_col])

        # Local import to avoid CUDA init in parent
        from orion_msp import OrionMSPClassifier  # type: ignore

        clf = OrionMSPClassifier(**clf_kwargs)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # logloss (if proba available)
        ll = None
        try:
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_test)
                ll = float(log_loss(y_test, y_proba))
        except Exception:
            ll = None

        used_sec = time.time() - t0
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(X_train.shape[1]),
            "target_col": str(target_col),
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "logloss": ll,
            "seconds": float(used_sec),
            "error": None,
            "traceback": None,
        }

    except Exception as e:
        used_sec = time.time() - t0
        tb = traceback.format_exc()
        if verbose:
            print(f"[FAIL] {dataset_id} -> {e}\n{tb}")
        return {
            "status": "fail",
            "dataset_id": dataset_id,
            "train_csv": str(train_csv),
            "test_csv": str(test_csv),
            "n_train": None,
            "n_test": None,
            "n_features": None,
            "target_col": None,
            "accuracy": None,
            "f1_macro": None,
            "logloss": None,
            "seconds": float(used_sec),
            "error": str(e),
            "traceback": tb,
        }


# -----------------------------
# Multiprocessing worker loop
# -----------------------------

@dataclass
class ResultRow:
    status: str
    dataset_id: str
    train_csv: str
    test_csv: str
    n_train: Optional[int]
    n_test: Optional[int]
    n_features: Optional[int]
    target_col: Optional[str]
    accuracy: Optional[float]
    f1_macro: Optional[float]
    logloss: Optional[float]
    seconds: float
    error: Optional[str]
    traceback: Optional[str]
    worker_id: int
    gpu_id: int


def worker_loop(
    worker_id: int,
    gpu_id: int,
    queue_path: Path,
    out_csv: Path,
    clf_kwargs: Dict,
    verbose: bool = False,
) -> None:
    """
    Each worker pops tasks from a shared JSONL queue file (simple file lock).
    Binds to one GPU by setting HIP_VISIBLE_DEVICES.
    """
    # bind GPU for ROCm
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    # IMPORTANT: inside worker, use cuda:0 after HIP_VISIBLE_DEVICES remap
    clf_kwargs = dict(clf_kwargs)
    clf_kwargs["device"] = "cuda:0"

    rows: List[Dict] = []

    # crude file lock via mkdir
    lock_dir = queue_path.parent / (queue_path.name + ".lock")

    while True:
        # acquire lock
        acquired = False
        for _ in range(2000):
            try:
                lock_dir.mkdir()
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.01)

        if not acquired:
            # mark worker crash
            rows.append(
                asdict(
                    ResultRow(
                        status="fail",
                        dataset_id=f"__worker_{worker_id}_lock_timeout__",
                        train_csv="",
                        test_csv="",
                        n_train=None,
                        n_test=None,
                        n_features=None,
                        target_col=None,
                        accuracy=None,
                        f1_macro=None,
                        logloss=None,
                        seconds=0.0,
                        error="lock_timeout",
                        traceback=None,
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                    )
                )
            )
            break

        try:
            if not queue_path.exists():
                break

            lines = queue_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                # done
                queue_path.unlink(missing_ok=True)
                break

            # pop one task
            first = lines[0]
            rest = lines[1:]
            queue_path.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")

        finally:
            # release lock
            try:
                lock_dir.rmdir()
            except Exception:
                pass

        task = json.loads(first)
        train_csv = Path(task["train_csv"])
        test_csv = Path(task["test_csv"])
        dataset_id = task["dataset_id"]

        result = eval_one_dataset(
            train_csv=train_csv,
            test_csv=test_csv,
            dataset_id=dataset_id,
            clf_kwargs=clf_kwargs,
            verbose=verbose,
        )

        rows.append(
            asdict(
                ResultRow(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    **result,
                )
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# -----------------------------
# Summary writer
# -----------------------------

def write_summary_txt(
    out_txt: Path,
    root: Path,
    discovered_pairs: int,
    processed_pairs: int,
    missing_test_ids: List[str],
    failed_ids: List[str],
    avg_acc: Optional[float],
    topn_avgs: Dict[int, float],
    wall_seconds: Optional[float] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
):
    lines: List[str] = []
    lines.append(f"root: {root}")
    lines.append(f"discovered_pairs: {discovered_pairs}")
    lines.append(f"processed_pairs: {processed_pairs}")

    # ---- timing ----
    if started_at is not None:
        lines.append(f"started_at: {started_at}")
    if finished_at is not None:
        lines.append(f"finished_at: {finished_at}")
    if wall_seconds is not None:
        lines.append(f"wall_seconds: {wall_seconds:.3f}")
        lines.append(f"wall_time_hms: {_fmt_hms(wall_seconds)}")

    lines.append(f"missing_test_count: {len(missing_test_ids)}")
    if missing_test_ids:
        lines.append("missing_test_datasets: " + ", ".join(missing_test_ids))
    else:
        lines.append("missing_test_datasets: (none)")

    lines.append(f"failed_count: {len(failed_ids)}")
    if failed_ids:
        lines.append("failed_datasets: " + ", ".join(failed_ids))
    else:
        lines.append("failed_datasets: (none)")

    if avg_acc is None:
        lines.append("avg_accuracy_ok: (none)")
    else:
        lines.append(f"avg_accuracy_ok: {avg_acc:.6f}")

    for n in (27, 63, 154):
        if n in topn_avgs:
            lines.append(f"avg_accuracy_ok_top_{n}: {topn_avgs[n]:.6f}")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True, help="Root folder containing *_train.csv and *_test.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-worker CSVs and merged results")
    ap.add_argument("--all-out", default=None, help="Path to merged ALL CSV (default: <out-dir>/msp_results.ALL.csv)")
    ap.add_argument("--summary-txt", default=None, help="Path to ONE global summary txt (default: <out-dir>/msp_results.summary.txt)")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker processes (usually #GPUs)")
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use (default: 0..workers-1)")

    ap.add_argument("--ckpt", required=True, help="Path to local Orion-MSP checkpoint (.ckpt)")
    ap.add_argument("--no-auto-download", action="store_true", help="Disable any auto-download behavior")

    ap.add_argument("--device", default="cuda:0", help='Device string in workers (recommend: "cuda:0")')
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-estimators", type=int, default=32)
    ap.add_argument("--norm-methods", default="none,power")
    ap.add_argument("--feat-shuffle", default="latin")
    ap.add_argument("--softmax-temp", type=float, default=0.9)

    ap.add_argument("--max-rows", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument("--only-datasets", type=str, default=None, help="Comma-separated dataset_ids to run")
    ap.add_argument("--skip-datasets", type=str, default=None, help="Comma-separated dataset_ids to skip")

    ap.add_argument("--no-logloss", action="store_true")
    ap.add_argument("--no-f1", action="store_true")
    ap.add_argument("--no-proba", action="store_true")
    ap.add_argument("--no-calibration", action="store_true")
    ap.add_argument("--no-postprocess", action="store_true")
    ap.add_argument("--no-feature-selection", action="store_true")
    ap.add_argument("--no-ensembling", action="store_true")
    ap.add_argument("--no-stacking", action="store_true")
    ap.add_argument("--no-iterative-imputation", action="store_true")
    ap.add_argument("--no-quick-imputation", action="store_true")
    ap.add_argument("--no-onehot", action="store_true")
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--no-hyperopt", action="store_true")
    ap.add_argument("--no-randsearch", action="store_true")
    ap.add_argument("--no-bagging", action="store_true")
    ap.add_argument("--no-boosting", action="store_true")
    ap.add_argument("--no-subspace", action="store_true")
    ap.add_argument("--no-earlystop", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # ---- run timing (aligned with benchmark_tabicl_dynamic.py) ----
    _t0 = time.time()
    _started_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    if args.all_out is None:
        all_out = out_dir / "msp_results.ALL.csv"
    else:
        all_out = Path(args.all_out)

    if args.summary_txt is None:
        summary_txt = out_dir / "msp_results.summary.txt"
    else:
        summary_txt = Path(args.summary_txt)

    pairs, missing_test_ids = discover_train_test_pairs(root)
    discovered_pairs = len(pairs)

    if args.only_datasets:
        only = {x.strip() for x in args.only_datasets.split(",") if x.strip()}
        pairs = [p for p in pairs if p[2] in only]

    if args.skip_datasets:
        skip = {x.strip() for x in args.skip_datasets.split(",") if x.strip()}
        pairs = [p for p in pairs if p[2] not in skip]

    if args.max_rows is not None:
        pairs = pairs[: args.max_rows]

    # build queue jsonl
    out_dir.mkdir(parents=True, exist_ok=True)
    queue_path = out_dir / "task_queue.jsonl"
    tasks = [{"train_csv": str(tr), "test_csv": str(te), "dataset_id": dsid} for tr, te, dsid in pairs]
    queue_path.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + ("\n" if tasks else ""), encoding="utf-8")

    # GPU ids
    if args.gpus is None:
        gpu_ids = list(range(args.workers))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) != args.workers:
            raise ValueError(f"--gpus length ({len(gpu_ids)}) must equal --workers ({args.workers})")

    # classifier kwargs
    clf_kwargs = dict(
        ckpt=args.ckpt,
        device=args.device,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        norm_methods=args.norm_methods,
        feat_shuffle=args.feat_shuffle,
        softmax_temp=args.softmax_temp,
        no_auto_download=bool(args.no_auto_download),
        random_state=args.random_state,
        no_logloss=bool(args.no_logloss),
        no_f1=bool(args.no_f1),
        no_proba=bool(args.no_proba),
        no_calibration=bool(args.no_calibration),
        no_postprocess=bool(args.no_postprocess),
        no_feature_selection=bool(args.no_feature_selection),
        no_ensembling=bool(args.no_ensembling),
        no_stacking=bool(args.no_stacking),
        no_iterative_imputation=bool(args.no_iterative_imputation),
        no_quick_imputation=bool(args.no_quick_imputation),
        no_onehot=bool(args.no_onehot),
        no_scaling=bool(args.no_scaling),
        no_hyperopt=bool(args.no_hyperopt),
        no_randsearch=bool(args.no_randsearch),
        no_bagging=bool(args.no_bagging),
        no_boosting=bool(args.no_boosting),
        no_subspace=bool(args.no_subspace),
        no_earlystop=bool(args.no_earlystop),
        no_amp=bool(args.no_amp),
    )

    # spawn workers
    import multiprocessing as mp

    worker_csv_paths: List[Path] = []
    procs: List[mp.Process] = []

    for wid, gid in enumerate(gpu_ids):
        w_csv = out_dir / f"worker_{wid}_gpu_{gid}.csv"
        worker_csv_paths.append(w_csv)
        p = mp.Process(
            target=worker_loop,
            kwargs=dict(
                worker_id=wid,
                gpu_id=gid,
                queue_path=queue_path,
                out_csv=w_csv,
                clf_kwargs=clf_kwargs,
                verbose=bool(args.verbose),
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    dfs: List[pd.DataFrame] = []
    for w_csv in worker_csv_paths:
        if w_csv.exists():
            try:
                dfs.append(pd.read_csv(w_csv))
            except Exception:
                continue

    if dfs:
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = pd.DataFrame(columns=[f.name for f in ResultRow.__dataclass_fields__.values()])

    all_out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(all_out, index=False)

    processed_pairs = int(len(all_df))

    if len(all_df):
        ok_df = all_df[(all_df["status"] == "ok") & all_df["accuracy"].notna()].copy()
    else:
        ok_df = pd.DataFrame(columns=["accuracy", "status", "dataset_id"])

    avg_acc = float(ok_df["accuracy"].mean()) if len(ok_df) > 0 else None

    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_sorted = ok_df.sort_values("accuracy", ascending=False, kind="mergesort")
        ok_count = len(ok_sorted)
        for n in (27, 63, 154):
            if ok_count >= n:
                topn_avgs[n] = float(ok_sorted.head(n)["accuracy"].mean())

    failed_ids: List[str] = []
    if len(all_df):
        failed_ids = (
            all_df.loc[all_df["status"] == "fail", "dataset_id"]
            .dropna()
            .astype(str)
            .tolist()
        )
        failed_ids = sorted(set(failed_ids))

    _finished_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    _wall_seconds = time.time() - _t0

    write_summary_txt(
        out_txt=summary_txt,
        root=root,
        discovered_pairs=discovered_pairs,
        processed_pairs=processed_pairs,
        missing_test_ids=missing_test_ids,
        failed_ids=failed_ids,
        avg_acc=avg_acc,
        topn_avgs=topn_avgs,
        wall_seconds=_wall_seconds,
        started_at=_started_at,
        finished_at=_finished_at,
    )

    # Print config for reproducibility
    print("\nSaved per-worker CSVs to:", str(out_dir))
    print("Saved merged ALL CSV to:", str(all_out))
    print("Saved summary TXT to:", str(summary_txt))
    print("\nOrion-MSP kwargs:")
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
