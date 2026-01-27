#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic multi-GPU benchmark runner for Orion-MSP (ROCm friendly).

- Dynamic scheduling via N worker processes pulling tasks from a shared JSONL queue.
- Each worker binds to one GPU via HIP_VISIBLE_DEVICES=<gpu_id> and uses device "cuda:0".
- Writes per-worker CSVs, merges into one ALL CSV, and writes ONE global summary TXT.
- Adds TabICL-like timing summary:
  started_at / finished_at / wall_seconds / wall_time_hms

AUTO-DOWNLOAD POLICY (requested):
- By default: ALLOW auto-download checkpoint.
- If --no-auto-download is set: DISALLOW auto-download.
- If --ckpt is provided but does not exist:
    - default: ignore the path and allow auto-download
    - with --no-auto-download: raise error
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

TARGET_CANDIDATES = [
    "target",
    "label",
    "y",
    "class",
    "Class",
    "TARGET",
    "Label",
    "Y",
]


def sanitize_dataset_id(p: Path) -> str:
    s = p.stem
    s = re.sub(r"(_train|_TRAIN)$", "", s)
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s


def discover_train_test_pairs(root: Path) -> Tuple[List[Tuple[Path, Path, str]], List[str]]:
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
    return cols[-1]


def eval_one_dataset(
    train_csv: Path,
    test_csv: Path,
    dataset_id: str,
    clf_kwargs: Dict,
    verbose: bool = False,
) -> Dict:
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

        # Local import to avoid GPU init in parent process
        from orion_msp import OrionMSPClassifier  # type: ignore

        # ---- filter/compat kwargs (keep CLI flags in clf_kwargs, but only pass accepted ones) ----
        import inspect
        sig = inspect.signature(OrionMSPClassifier.__init__)
        accepted = set(sig.parameters.keys()) - {"self"}

        _kw = dict(clf_kwargs)
        # backward/alias support
        if "feat_shuffle" in _kw and "feat_shuffle_method" not in _kw:
            _kw["feat_shuffle_method"] = _kw.pop("feat_shuffle")
        if "softmax_temp" in _kw and "softmax_temperature" not in _kw:
            _kw["softmax_temperature"] = _kw.pop("softmax_temp")
        if "ckpt" in _kw and "model_path" not in _kw:
            _kw["model_path"] = _kw.pop("ckpt")
        if "no_auto_download" in _kw and "allow_auto_download" not in _kw:
            _kw["allow_auto_download"] = (not bool(_kw.pop("no_auto_download")))

        model_kwargs = {k: v for k, v in _kw.items() if k in accepted}

        clf = OrionMSPClassifier(**model_kwargs)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        # optional metrics (controlled by CLI flags; defaults compute all)
        no_f1 = bool(clf_kwargs.get("no_f1", False))
        no_logloss = bool(clf_kwargs.get("no_logloss", False))
        no_proba = bool(clf_kwargs.get("no_proba", False))

        f1 = None
        if not no_f1:
            f1 = float(f1_score(y_test, y_pred, average="macro"))

        ll = None
        if (not no_logloss) and (not no_proba):
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
            "f1_macro": (float(f1) if f1 is not None else None),
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
    # Bind GPU for ROCm; inside worker always use cuda:0 after remap.
    os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_id)

    clf_kwargs = dict(clf_kwargs)
    clf_kwargs["device"] = "cuda:0"

    rows: List[Dict] = []

    lock_dir = queue_path.parent / (queue_path.name + ".lock")

    while True:
        acquired = False
        for _ in range(2000):
            try:
                lock_dir.mkdir()
                acquired = True
                break
            except FileExistsError:
                time.sleep(0.01)

        if not acquired:
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
                        error="lock timeout",
                        traceback=None,
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                    )
                )
            )
            break

        try:
            lines = queue_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                break
            task = json.loads(lines[0])
            remaining = lines[1:]
            queue_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        finally:
            try:
                lock_dir.rmdir()
            except Exception:
                pass

        train_csv = Path(task["train_csv"])
        test_csv = Path(task["test_csv"])
        dataset_id = task["dataset_id"]

        if verbose:
            print(f"[W{worker_id}][GPU{gpu_id}] start {dataset_id}")

        r = eval_one_dataset(train_csv, test_csv, dataset_id, clf_kwargs, verbose=verbose)
        r["worker_id"] = int(worker_id)
        r["gpu_id"] = int(gpu_id)

        rows.append(r)

        df = pd.DataFrame(rows)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def format_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root with *_train.csv and *_test.csv pairs")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--all-out", type=str, default="", help="Merged ALL CSV path (default: <out_dir>/msp_results.ALL.csv)")
    ap.add_argument("--summary-txt", type=str, default="", help="Summary TXT path (default: <out_dir>/msp_results.summary.txt)")

    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    ap.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU ids; length must equal --workers")

    ap.add_argument("--device", type=str, default="cuda:0", help='Device string (worker will force to "cuda:0")')
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-estimators", type=int, default=32)

    ap.add_argument("--norm-methods", type=str, default="none,power", help="Comma list, e.g. none,power")
    ap.add_argument("--feat-shuffle", type=str, default="latin", choices=["latin", "random", "none"])
    ap.add_argument("--softmax-temp", type=float, default=0.9)

    ap.add_argument("--ckpt", type=str, default="", help="Local ckpt path (optional). If missing, may auto-download.")
    ap.add_argument("--no-auto-download", action="store_true", help="Disallow HF auto-download if ckpt missing")

    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--only-datasets", type=str, default="", help="Comma list of dataset ids to run")
    ap.add_argument("--skip-datasets", type=str, default="", help="Comma list of dataset ids to skip")

    # metric toggles
    ap.add_argument("--no-logloss", action="store_true")
    ap.add_argument("--no-f1", action="store_true")
    ap.add_argument("--no-proba", action="store_true")

    # keep these flags for compatibility (they are not passed into OrionMSPClassifier unless supported)
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

    # ---- timing (TabICL-like) ----
    _t0 = time.time()
    _started_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    all_out = Path(args.all_out) if args.all_out else (out_dir / "msp_results.ALL.csv")
    summary_txt = Path(args.summary_txt) if args.summary_txt else (out_dir / "msp_results.summary.txt")

    # ---- checkpoint policy: prefer auto-download ----
    ckpt_to_use: Optional[str] = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser()
        if ckpt_path.exists():
            ckpt_to_use = str(ckpt_path)
        else:
            if args.no_auto_download:
                raise FileNotFoundError(f"--ckpt provided but not found: {ckpt_path}")
            # otherwise ignore local ckpt and allow auto-download
            print(f"[WARN] --ckpt not found: {ckpt_path} ; will fall back to auto-download.")
            ckpt_to_use = None

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
    queue_path.write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + ("\n" if tasks else ""),
        encoding="utf-8",
    )

    # GPU ids
    if args.gpus is None:
        gpu_ids = list(range(args.workers))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if len(gpu_ids) != args.workers:
            raise ValueError(f"--gpus length ({len(gpu_ids)}) must equal --workers ({args.workers})")

    # classifier kwargs
    clf_kwargs: Dict = dict(
        device=args.device,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        norm_methods=[x.strip() for x in args.norm_methods.split(",") if x.strip()] if isinstance(args.norm_methods, str) else args.norm_methods,
        feat_shuffle_method=args.feat_shuffle,
        softmax_temperature=args.softmax_temp,
        # allow auto-download by default:
        allow_auto_download=not bool(args.no_auto_download),
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
    # only set model_path if we have an existing local file
    if ckpt_to_use is not None:
        clf_kwargs["model_path"] = ckpt_to_use

    print("[INFO] discovered_pairs =", discovered_pairs)
    if missing_test_ids:
        print("[WARN] missing test for:", ", ".join(missing_test_ids[:20]), (" ..." if len(missing_test_ids) > 20 else ""))

    print("[INFO] tasks_to_run =", len(pairs))
    print("[INFO] clf_kwargs =")
    print(json.dumps(clf_kwargs, indent=2, ensure_ascii=False))

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

    ok_df = all_df[(all_df.get("status") == "ok") & all_df.get("accuracy").notna()].copy() if len(all_df) else pd.DataFrame()
    avg_acc = float(ok_df["accuracy"].mean()) if len(ok_df) > 0 else None

    topn_avgs: Dict[int, float] = {}
    if len(ok_df) > 0:
        ok_df_sorted = ok_df.sort_values("accuracy", ascending=False)
        for n in [1, 3, 5, 10]:
            sub = ok_df_sorted.head(n)
            if len(sub) > 0:
                topn_avgs[n] = float(sub["accuracy"].mean())

    wall_seconds = float(time.time() - _t0)
    finished_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    summary_lines = [
        f"started_at: {_started_at}",
        f"finished_at: {finished_at}",
        f"wall_seconds: {wall_seconds:.3f}",
        f"wall_time_hms: {format_hms(wall_seconds)}",
        "",
        f"discovered_pairs: {discovered_pairs}",
        f"tasks_to_run: {len(pairs)}",
        f"processed_pairs: {processed_pairs}",
        f"ok_pairs: {int(len(ok_df)) if ok_df is not None else 0}",
        "",
        f"avg_accuracy: {avg_acc if avg_acc is not None else 'NA'}",
    ]
    if topn_avgs:
        summary_lines.append("topN_avg_accuracy:")
        for n, v in topn_avgs.items():
            summary_lines.append(f"  top{n}: {v:.6f}")

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("[DONE] wrote:", str(all_out))
    print("[DONE] wrote:", str(summary_txt))


if __name__ == "__main__":
    main()
