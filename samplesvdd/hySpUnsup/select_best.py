#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Row:
    run_dir: str
    score: float
    auc: float
    inside: float
    n_active: int
    dom_train: float
    margin_mean: float
    n_spheres: int
    best_epoch: int


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f
    except Exception:
        return default


def _load_rows(results_glob: str, w_inside: float, w_dom: float) -> list[Row]:
    rows: list[Row] = []
    for p in glob.glob(results_glob, recursive=True):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        auc = _safe_float(d.get("macro_auc_test_pruned_active_spheres"))
        if math.isnan(auc):
            continue

        union = d.get("union_test_metrics_active_spheres", {})
        inside = _safe_float(union.get("union_frac_inside"))
        margin_mean = _safe_float(union.get("union_margin_mean"))

        prune = d.get("prune", {})
        active = prune.get("active_cluster_mask", [])
        n_active = int(sum(1 for x in active if bool(x)))
        counts = prune.get("train_argmin_counts", [])
        if counts and sum(counts) > 0:
            dom_train = float(max(counts) / float(sum(counts)))
        else:
            dom_train = float("nan")

        inside_pen = abs(inside - 0.5) if not math.isnan(inside) else 1.0
        dom_pen = dom_train if not math.isnan(dom_train) else 1.0
        score = auc - w_inside * inside_pen - w_dom * dom_pen

        rows.append(
            Row(
                run_dir=os.path.dirname(p),
                score=score,
                auc=auc,
                inside=inside,
                n_active=n_active,
                dom_train=dom_train,
                margin_mean=margin_mean,
                n_spheres=int(d.get("n_spheres", 0) or 0),
                best_epoch=int(d.get("best_epoch", -1) or -1),
            )
        )
    rows.sort(key=lambda r: r.score, reverse=True)
    return rows


def _print_top(rows: list[Row], top_k: int) -> None:
    print("rank\tscore\tauc\tinside\tn_active\tdom_train\tmargin_mean\tn_spheres\tbest_epoch\trun")
    for i, r in enumerate(rows[:top_k], start=1):
        print(
            f"{i}\t{r.score:.6f}\t{r.auc:.6f}\t{r.inside:.6f}\t{r.n_active}\t"
            f"{r.dom_train:.6f}\t{r.margin_mean:.6f}\t{r.n_spheres}\t{r.best_epoch}\t{r.run_dir}"
        )


def _write_csv(rows: list[Row], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "run_dir",
                "score",
                "auc",
                "inside",
                "n_active",
                "dom_train",
                "margin_mean",
                "n_spheres",
                "best_epoch",
            ]
        )
        for i, r in enumerate(rows, start=1):
            w.writerow(
                [
                    i,
                    r.run_dir,
                    f"{r.score:.8f}",
                    f"{r.auc:.8f}",
                    f"{r.inside:.8f}",
                    r.n_active,
                    f"{r.dom_train:.8f}",
                    f"{r.margin_mean:.8f}",
                    r.n_spheres,
                    r.best_epoch,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser("Rank sensitivity runs and select best hyperparams")
    p.add_argument(
        "--runs_root",
        type=str,
        default="samplesvdd/hySpUnsup/runs/sensitivity_v3",
        help="Root directory containing run subfolders with results.json",
    )
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument(
        "--w_inside",
        type=float,
        default=0.15,
        help="Penalty weight for inside-frac degeneracy via abs(inside-0.5)",
    )
    p.add_argument(
        "--w_dom",
        type=float,
        default=0.10,
        help="Penalty weight for dominant-cluster ratio on train assignments",
    )
    p.add_argument(
        "--csv_out",
        type=str,
        default="samplesvdd/hySpUnsup/runs/sensitivity_v3/selection_summary.csv",
        help="CSV output path (set empty string to disable)",
    )
    args = p.parse_args()

    results_glob = os.path.join(args.runs_root, "**", "results.json")
    rows = _load_rows(results_glob, w_inside=args.w_inside, w_dom=args.w_dom)
    if not rows:
        raise SystemExit(f"No valid results found under: {args.runs_root}")

    _print_top(rows, top_k=max(1, args.top_k))
    print("\nBEST_RUN:", rows[0].run_dir)
    if args.csv_out.strip():
        _write_csv(rows, args.csv_out)
        print("CSV:", args.csv_out)


if __name__ == "__main__":
    main()
