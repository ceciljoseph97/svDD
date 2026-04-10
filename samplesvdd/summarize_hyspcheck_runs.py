#!/usr/bin/env python3
"""
Aggregate MNIST (hySpCheck) and CIFAR-10 (hySpCheckcifar) run folders into CSV summaries.

Per run, **training** metrics always come **only** from the root file:
  <run>/results.json
    - best_epoch, best_macro_auc, per_digit_best_auc (mapped to train_per_class_*), lambdas, etc.
  Nothing under eval_hyperbolic/ is used for train_* or per_digit_best_auc.

**Evaluation** (optional unless --train-only):
  - <run>/eval_hyperbolic/results.json or <run>/eval_test/results.json  -> eval_h_*, macro_svdd_auc, per_digit_svdd_auc
  - <run>/eval_euclidean/results.json  -> eval_e_* (optional)
  - <run>/eval_*/global_top_anomalies.csv (optional)

Default --out_dir is OUTSIDE individual run folders (sibling summaries/).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_first_json(paths: List[Path]) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    for p in paths:
        j = _load_json(p)
        if j is not None:
            return j, p
    return None, None


def _csv_global_anomaly_stats(path: Path) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not path.is_file():
        return None, None, None
    vals: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                vals.append(float(row["global_anom_score"]))
            except (KeyError, ValueError, TypeError):
                continue
    if not vals:
        return None, None, None
    top1 = vals[0]
    top10 = statistics.mean(vals[:10]) if len(vals) >= 10 else statistics.mean(vals)
    top100 = statistics.mean(vals)
    return top1, top10, top100


def _summarize_runs(runs_root: Path, dataset_label: str, *, include_eval: bool = True) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not runs_root.is_dir():
        return rows
    for child in sorted(runs_root.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        train_path = child / "results.json"
        train = _load_json(train_path)
        if train is None:
            continue
        eh = None
        eh_src: Optional[Path] = None
        ee = None
        h1 = h10 = h100 = None
        e1 = e10 = e100 = None
        if include_eval:
            # Some older runs use eval_test/ instead of eval_hyperbolic/.
            eh, eh_src = _load_first_json(
                [
                    child / "eval_hyperbolic" / "results.json",
                    child / "eval_test" / "results.json",
                ]
            )
            ee = _load_json(child / "eval_euclidean" / "results.json")
            if eh_src is not None and eh_src.parent.name in ("eval_hyperbolic", "eval_test"):
                h_csv_path = eh_src.parent / "global_top_anomalies.csv"
            else:
                h_csv_path = child / "eval_hyperbolic" / "global_top_anomalies.csv"
            h1, h10, h100 = _csv_global_anomaly_stats(h_csv_path)
            e1, e10, e100 = _csv_global_anomaly_stats(child / "eval_euclidean" / "global_top_anomalies.csv")

        row: Dict[str, Any] = {
            "dataset": dataset_label,
            "run_name": child.name,
            "train_results_json": str(train_path.resolve()),
            "train_best_epoch": train.get("best_epoch"),
            "train_best_macro_auc": train.get("best_macro_auc"),
            "curvature": train.get("curvature"),
            "objective": train.get("objective"),
            "lambda_excl": train.get("lambda_excl"),
            "lambda_overlap": train.get("lambda_overlap"),
            "eval_hyperbolic_macro_auc": eh.get("macro_svdd_auc") if eh else None,
            "eval_euclidean_macro_auc": ee.get("macro_svdd_auc") if ee else None,
            "eval_h_csv_top1_global": h1,
            "eval_h_csv_top10_mean_global": h10,
            "eval_h_csv_top100_mean_global": h100,
            "eval_e_csv_top1_global": e1,
            "eval_e_csv_top10_mean_global": e10,
            "eval_e_csv_top100_mean_global": e100,
            "eval_h_source": str(eh_src.parent.name) if eh_src else None,
            "run_path": str(child.resolve()),
        }
        # Per-class: train ONLY from root results.json (per_digit_best_auc); eval from eval_* only.
        per_digit_train = train.get("per_digit_best_auc") or {}
        per_digit_eh = (eh or {}).get("per_digit_svdd_auc") or {}
        per_digit_ee = (ee or {}).get("per_digit_svdd_auc") or {}
        for d in range(10):
            k = str(d)
            row[f"train_per_class_{k}"] = per_digit_train.get(k)
            row[f"eval_h_per_class_{k}"] = per_digit_eh.get(k)
            row[f"eval_e_per_class_{k}"] = per_digit_ee.get(k)
        rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md(path: Path, title: str, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_No runs with `results.json` found._")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.extend([header, sep])
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.6f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _expand_per_class_rows(base_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert per-run wide columns into a long table:
      one row per (run, class_idx)
    """
    long_rows: List[Dict[str, Any]] = []
    for r in base_rows:
        dataset = r.get("dataset")
        run_name = r.get("run_name")
        run_path = r.get("run_path")
        for d in range(10):
            long_rows.append(
                {
                    "dataset": dataset,
                    "run_name": run_name,
                    "class_idx": d,
                    "train_best_auc": r.get(f"train_per_class_{d}"),
                    "eval_hyperbolic_best_auc": r.get(f"eval_h_per_class_{d}"),
                    "eval_euclidean_best_auc": r.get(f"eval_e_per_class_{d}"),
                    "run_path": run_path,
                }
            )
    return long_rows


def _write_grouped_per_class_md(path: Path, title: str, base_rows: List[Dict[str, Any]], *, include_eval: bool = True) -> None:
    """
    Clean per-class markdown grouped by:
      dataset -> run_name

    Each run gets a small 10-row table (class 0..9).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [f"# {title}", ""]
    if not base_rows:
        lines.append("_No runs with `results.json` found._")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    # deterministic grouping
    def _key(r: Dict[str, Any]) -> Tuple[str, str]:
        return (str(r.get("dataset") or ""), str(r.get("run_name") or ""))

    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in base_rows:
        grouped[_key(r)] = r

    datasets = sorted({k[0] for k in grouped.keys()}, key=lambda s: s.lower())
    for ds in datasets:
        lines.append(f"## {ds}")
        lines.append("")
        run_names = sorted([k[1] for k in grouped.keys() if k[0] == ds], key=lambda s: s.lower())
        for rn in run_names:
            r = grouped[(ds, rn)]
            lines.append(f"### {rn}")
            lines.append("")
            rp = r.get("run_path")
            if rp:
                lines.append(f"`{rp}`")
                lines.append("")
            if include_eval:
                lines.append("| class_idx | train (results.json per_digit_best_auc) | eval_hyperbolic |")
                lines.append("| --- | --- | --- |")
            else:
                lines.append("| class_idx | train (results.json per_digit_best_auc) |")
                lines.append("| --- | --- |")
            for d in range(10):
                def _fmt(v: Any) -> str:
                    if v is None:
                        return ""
                    if isinstance(v, float):
                        return f"{v:.6f}"
                    return str(v)

                cells = [str(d), _fmt(r.get(f"train_per_class_{d}"))]
                if include_eval:
                    cells.append(_fmt(r.get(f"eval_h_per_class_{d}")))
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _write_pivot_md(path: Path, title: str, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _class_first_ablation_grouped_table(rows: List[Dict[str, Any]], dataset_name: str) -> List[str]:
    """
    Build one table per dataset:
      rows: class_idx 0..9
      grouped columns: each ablation has [train, test_hyp]
    """
    runs = sorted([str(r["run_name"]) for r in rows], key=lambda s: s.lower())
    by_name = {str(r["run_name"]): r for r in rows}
    out: List[str] = [f"## {dataset_name}", ""]

    # 2-row markdown header to visually group columns per ablation.
    h1 = ["class_idx"]
    h2 = ["metric"]
    sep = ["---"]
    for rn in runs:
        h1.extend([rn, ""])
        h2.extend(["train", "test_hyp"])
        sep.extend(["---", "---"])

    out.append("| " + " | ".join(h1) + " |")
    out.append("| " + " | ".join(h2) + " |")
    out.append("| " + " | ".join(sep) + " |")

    for d in range(10):
        row_cells = [str(d)]
        for rn in runs:
            r = by_name.get(rn, {})
            row_cells.append(_fmt_cell(r.get(f"train_per_class_{d}")))
            row_cells.append(_fmt_cell(r.get(f"eval_h_per_class_{d}")))
        out.append("| " + " | ".join(row_cells) + " |")
    out.append("")
    return out


def _single_combined_table(
    mnist_rows: List[Dict[str, Any]], cifar_rows: List[Dict[str, Any]], *, include_eval: bool = True
) -> List[str]:
    """
    One single table:
      rows: MNIST_0..9 then CIFAR_0..9
      columns: train (from results.json per_digit_best_auc) and optionally test_hyp (eval hyperbolic per-class).
    """
    out: List[str] = ["## Combined single table", ""]

    def _norm(name: str) -> str:
        low = name.lower()
        if low.startswith("ablate_mnist_"):
            return low.replace("ablate_mnist_", "")
        if low.startswith("ablate_cifar_"):
            return low.replace("ablate_cifar_", "")
        if low.startswith("ae_pretrain_"):
            return "ae_pretrain"
        if low in ("hyp_multi_cuda", "hyp_cifar_cuda"):
            return "baseline"
        return low

    mn_by_norm = {_norm(str(r["run_name"])): r for r in mnist_rows}
    cf_by_norm = {_norm(str(r["run_name"])): r for r in cifar_rows}
    union = sorted(set(mn_by_norm.keys()) | set(cf_by_norm.keys()))
    preferred = ["full", "no_excl", "no_ov", "indep", "baseline", "ae_pretrain"]
    ordered = [k for k in preferred if k in union] + [k for k in union if k not in preferred]

    h1 = ["dataset_class"]
    h2 = ["metric"]
    sep = ["---"]
    h1.extend(["train"] * len(ordered))
    h2.extend(ordered)
    sep.extend(["---"] * len(ordered))
    if include_eval:
        h1.extend(["test_hyp"] * len(ordered))
        h2.extend(ordered)
        sep.extend(["---"] * len(ordered))

    out.append("| " + " | ".join(h1) + " |")
    out.append("| " + " | ".join(h2) + " |")
    out.append("| " + " | ".join(sep) + " |")

    for ds_name, by_norm in [("MNIST", mn_by_norm), ("CIFAR", cf_by_norm)]:
        for d in range(10):
            cells = [f"{ds_name}_{d}"]

            train_vals: List[Optional[float]] = []
            test_vals: List[Optional[float]] = []
            for ab in ordered:
                r = by_norm.get(ab, {})
                tv = r.get(f"train_per_class_{d}")
                hv = r.get(f"eval_h_per_class_{d}")
                train_vals.append(tv if isinstance(tv, (int, float)) else None)
                if include_eval:
                    test_vals.append(hv if isinstance(hv, (int, float)) else None)

            best_train = max((v for v in train_vals if v is not None), default=None)
            best_test = max((v for v in test_vals if v is not None), default=None) if include_eval else None

            for v in train_vals:
                s = _fmt_cell(v)
                if v is not None and best_train is not None and v == best_train:
                    s = f"**{s}**"
                cells.append(s)

            if include_eval:
                for v in test_vals:
                    s = _fmt_cell(v)
                    if v is not None and best_test is not None and v == best_test:
                        s = f"**{s}**"
                    cells.append(s)

            out.append("| " + " | ".join(cells) + " |")

    out.append("")
    return out

def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Summarize hySpCheck MNIST + hySpCheckcifar run metrics.")
    p.add_argument(
        "--mnist_runs_dir",
        type=str,
        default=str(here / "hySpCheck" / "runs"),
        help="Folder containing MNIST run subdirs (each with results.json).",
    )
    p.add_argument(
        "--cifar_runs_dir",
        type=str,
        default=str(here / "hySpCheckcifar" / "runs"),
        help="Folder containing CIFAR run subdirs.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(here / "ablation_summaries"),
        help="Output directory OUTSIDE per-run folders (default: samplesvdd/ablation_summaries).",
    )
    p.add_argument("--no_md", action="store_true", help="Skip writing Markdown tables.")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Only read <run>/results.json (per_digit_best_auc, best_macro_auc). Do not read eval_hyperbolic/eval_euclidean.",
    )
    args = p.parse_args()
    include_eval = not args.train_only

    mnist_root = Path(args.mnist_runs_dir)
    cifar_root = Path(args.cifar_runs_dir)
    out = Path(args.out_dir)

    mnist_rows = _summarize_runs(mnist_root, "MNIST", include_eval=include_eval)
    cifar_rows = _summarize_runs(cifar_root, "CIFAR-10", include_eval=include_eval)

    base_fieldnames = [
        "dataset",
        "run_name",
        "train_results_json",
        "train_best_epoch",
        "train_best_macro_auc",
        "curvature",
        "objective",
        "lambda_excl",
        "lambda_overlap",
        "eval_hyperbolic_macro_auc",
        "eval_euclidean_macro_auc",
        "eval_h_source",
        "eval_h_csv_top1_global",
        "eval_h_csv_top10_mean_global",
        "eval_h_csv_top100_mean_global",
        "eval_e_csv_top1_global",
        "eval_e_csv_top10_mean_global",
        "eval_e_csv_top100_mean_global",
        "run_path",
    ]
    per_class_fields = []
    for d in range(10):
        k = str(d)
        per_class_fields.append(f"train_per_class_{k}")
        per_class_fields.append(f"eval_h_per_class_{k}")
        per_class_fields.append(f"eval_e_per_class_{k}")
    fieldnames = base_fieldnames + per_class_fields

    _write_csv(out / "summary_mnist_runs.csv", fieldnames, mnist_rows)
    _write_csv(out / "summary_cifar_runs.csv", fieldnames, cifar_rows)
    _write_csv(out / "summary_all_runs.csv", fieldnames, mnist_rows + cifar_rows)

    md_cols = [
        "run_name",
        "train_best_macro_auc",
        "lambda_excl",
        "lambda_overlap",
    ]
    if include_eval:
        md_cols.extend(
            [
                "eval_hyperbolic_macro_auc",
                "eval_euclidean_macro_auc",
                "eval_h_csv_top1_global",
                "eval_h_csv_top10_mean_global",
            ]
        )
    if not args.no_md:
        _write_md(out / "summary_mnist_runs.md", "MNIST (hySpCheck) runs", mnist_rows, ["dataset"] + md_cols)
        _write_md(out / "summary_cifar_runs.md", "CIFAR-10 (hySpCheckcifar) runs", cifar_rows, ["dataset"] + md_cols)
        _write_md(out / "summary_all_runs.md", "All runs", mnist_rows + cifar_rows, ["dataset"] + md_cols)
        # Grouped per-class summaries (clean: dataset -> run/ablation).
        _write_grouped_per_class_md(
            out / "summary_mnist_per_class_runs.md", "MNIST per-class AUC per run", mnist_rows, include_eval=include_eval
        )
        _write_grouped_per_class_md(
            out / "summary_cifar_per_class_runs.md", "CIFAR-10 per-class AUC per run", cifar_rows, include_eval=include_eval
        )
        _write_grouped_per_class_md(
            out / "summary_all_per_class_runs.md", "All datasets per-class AUC per run", mnist_rows + cifar_rows, include_eval=include_eval
        )

        # Pivot-style per-class table (single combined table).
        pivot_lines: List[str] = []
        if include_eval:
            pivot_lines.append(
                "Single table: MNIST_0..9 then CIFAR_0..9; **train** = per_digit_best_auc from each run's results.json; "
                "**test_hyp** = eval hyperbolic per-class AUC."
            )
        else:
            pivot_lines.append(
                "Single table: MNIST_0..9 then CIFAR_0..9; **train only** = per_digit_best_auc from each run's results.json (no eval folders read)."
            )
        pivot_lines.append("")
        pivot_lines.extend(_single_combined_table(mnist_rows, cifar_rows, include_eval=include_eval))

        pivot_title = (
            "Per-class single table (train from results.json; test_hyp eval)"
            if include_eval
            else "Per-class single table (train from results.json only)"
        )
        _write_pivot_md(out / "summary_pivot_per_class.md", pivot_title, pivot_lines)

    meta = {
        "mnist_runs_dir": str(mnist_root.resolve()),
        "cifar_runs_dir": str(cifar_root.resolve()),
        "out_dir": str(out.resolve()),
        "train_only": bool(args.train_only),
        "mnist_run_count": len(mnist_rows),
        "cifar_run_count": len(cifar_rows),
        "written": [
            str(out / "summary_mnist_runs.csv"),
            str(out / "summary_cifar_runs.csv"),
            str(out / "summary_all_runs.csv"),
        ],
    }
    if not args.no_md:
        meta["written"].extend(
            [
                str(out / "summary_mnist_runs.md"),
                str(out / "summary_cifar_runs.md"),
                str(out / "summary_all_runs.md"),
                str(out / "summary_mnist_per_class_runs.md"),
                str(out / "summary_cifar_per_class_runs.md"),
                str(out / "summary_all_per_class_runs.md"),
                str(out / "summary_pivot_per_class.md"),
            ]
        )
    with (out / "summary_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
