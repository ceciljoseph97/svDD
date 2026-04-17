#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_HYSP_CHECK = _HERE.parent / "hySpCheck"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HYSP_CHECK))

from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD  # noqa: E402
from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE  # noqa: E402
import train_hyp_mnist_unsup as unsup  # noqa: E402


def _wandb_log_run(
    run_dir: str,
    args,
    ck_meta: dict,
    eval_meta: dict,
    image_paths: list[str],
) -> None:
    if not args.use_wandb:
        return
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] unavailable for {run_dir}: {e}")
        return

    run_name = f"{args.wandb_name_prefix}{os.path.basename(run_dir)}"
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] if args.wandb_tags else None
    group = args.wandb_group if args.wandb_group else os.path.basename(args.runs_root.rstrip("/"))
    config = {
        "run_dir": run_dir,
        "replay_split": args.split,
        "auc_mode": args.auc_mode,
        **ck_meta,
    }
    wb = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        group=group,
        name=run_name,
        tags=tags,
        mode=args.wandb_mode,
        config=config,
        reinit=True,
    )
    payload = {}
    for k in ("train_macro_auc", "test_macro_auc"):
        if k in eval_meta:
            payload[k] = eval_meta[k]
    if payload:
        wandb.log(payload)

    for p in image_paths:
        if os.path.isfile(p):
            key = os.path.relpath(p, run_dir).replace(os.sep, "/")
            key = key.replace(".", "_")
            wandb.log({key: wandb.Image(p)})

    wb.finish()


def _build_dataset(
    mnist_processed_dir: str,
    split: str,
    train_fraction: float,
    digits: list[int],
    max_samples_per_class: int | None,
):
    raw = MNISTDigitsProcessedRawDataset(
        root_dir=mnist_processed_dir,
        split=split,
        train_fraction=train_fraction,
        digits=digits,
        max_samples=None,
        max_samples_per_class=max_samples_per_class,
    )
    return raw, unsup.UnsupervisedScaledDataset(raw, return_raw=False)


def _run_one(
    run_dir: str,
    args,
) -> None:
    ckpt_path = os.path.join(run_dir, "checkpoint_v2.pth")
    if not os.path.isfile(ckpt_path):
        return

    ck = torch.load(ckpt_path, map_location="cpu")
    rep_dim = int(ck.get("rep_dim", 32))
    z_dim = int(ck.get("z_dim", 16))
    curvature = float(ck.get("curvature", 1.0))
    n_spheres = int(ck.get("n_spheres", 10))
    objective = str(ck.get("objective", "union-soft"))
    eval_objective = "soft-boundary" if objective == "union-soft" else "one-class"
    train_digits = list(ck.get("train_normal_digits", list(range(10))))
    test_digits = list(ck.get("test_digits", list(range(10))))

    device = torch.device(args.device)
    backbone = MNIST_LeNet_SVDDIAE(rep_dim=rep_dim)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=rep_dim, z_dim=z_dim, n_digits=n_spheres, c=curvature
    ).to(device)
    model.load_state_dict(ck["model_state"], strict=True)
    c_h = ck["c_h"].to(device)
    R = ck["R"].to(device)
    active_mask = np.asarray(ck.get("active_cluster_mask", [True] * n_spheres), dtype=bool)
    ck_meta = {
        "n_spheres": n_spheres,
        "curvature": curvature,
        "objective": objective,
        "best_epoch": int(ck.get("best_epoch", -1) or -1),
    }
    image_paths: list[str] = []

    tr_raw, tr = _build_dataset(
        mnist_processed_dir=args.mnist_processed_dir,
        split="train",
        train_fraction=args.train_fraction,
        digits=train_digits,
        max_samples_per_class=args.max_samples_per_class,
    )
    te_raw, te = _build_dataset(
        mnist_processed_dir=args.mnist_processed_dir,
        split="test",
        train_fraction=args.train_fraction,
        digits=test_digits,
        max_samples_per_class=args.max_samples_per_class,
    )
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Eval from checkpoint on train/test (if requested)
    out_eval: dict = {
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "n_spheres": n_spheres,
        "curvature": curvature,
        "objective": objective,
        "active_cluster_mask": active_mask.astype(bool).tolist(),
    }
    if args.split in ("train", "both"):
        m_train, p_train = unsup.evaluate(
            model,
            c_h,
            R,
            tr_loader,
            device,
            eval_objective,
            curvature,
            active_cluster_mask=active_mask,
            auc_mode=args.auc_mode,
        )
        out_eval["train_macro_auc"] = m_train
        out_eval["train_per_digit_auc"] = p_train
    if args.split in ("test", "both"):
        m_test, p_test = unsup.evaluate(
            model,
            c_h,
            R,
            te_loader,
            device,
            eval_objective,
            curvature,
            active_cluster_mask=active_mask,
            auc_mode=args.auc_mode,
        )
        out_eval["test_macro_auc"] = m_test
        out_eval["test_per_digit_auc"] = p_test

    eval_out = os.path.join(run_dir, "eval_from_checkpoint.json")
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(out_eval, f, indent=2)

    # t-SNE regeneration
    if args.split in ("train", "both"):
        tsne_out = os.path.join(run_dir, "tsne_unsup_v2_nearest_z_train.png")
        unsup.plot_tsne_unsupervised(
            model,
            c_h,
            R,
            tr,
            device,
            curvature,
            tsne_out,
            max_samples=args.tsne_max_samples,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            seed=args.tsne_seed,
            eval_objective=eval_objective,
            active_cluster_mask=active_mask,
        )
        image_paths.append(tsne_out)
        if args.export_cluster_samples > 0:
            ex_dir = os.path.join(run_dir, "cluster_exports_train")
            ds_ex = unsup.UnsupervisedScaledDataset(tr_raw, return_raw=True)
            unsup.export_cluster_sample_images(
                model,
                c_h,
                R,
                ds_ex,
                device,
                curvature,
                ex_dir,
                samples_per_cluster=args.export_cluster_samples,
                batch_size=args.batch_size,
                seed=args.cluster_export_seed,
                split_name="train",
                active_cluster_mask=active_mask,
            )
            image_paths.append(os.path.join(ex_dir, "cluster_samples_overview.png"))
        if args.export_hotspot_analysis:
            hs_dir = os.path.join(run_dir, "hotspot_analysis_train")
            unsup.export_hotspot_class_density(
                model,
                c_h,
                R,
                tr,
                device,
                curvature,
                hs_dir,
                batch_size=args.batch_size,
                split_name="train",
                active_cluster_mask=active_mask,
            )
            image_paths.extend(
                [
                    os.path.join(hs_dir, "hotspot_cluster_digit_bars.png"),
                    os.path.join(hs_dir, "hotspot_P_digit_given_cluster.png"),
                    os.path.join(hs_dir, "hotspot_P_cluster_given_digit.png"),
                ]
            )
        if args.export_cluster_neural_hotspots:
            nh_dir = os.path.join(run_dir, "cluster_neural_hotspots_train")
            unsup.export_cluster_neural_hotspots(
                model,
                c_h,
                R,
                tr,
                device,
                curvature,
                nh_dir,
                batch_size=args.batch_size,
                split_name="train",
                max_samples_per_cluster=args.neural_hotspot_max_samples,
                seed=args.cluster_export_seed,
                active_cluster_mask=active_mask,
            )
            image_paths.append(os.path.join(nh_dir, "cluster_neural_hotspots_overview.png"))
    if args.split in ("test", "both"):
        tsne_out = os.path.join(run_dir, "tsne_unsup_v2_nearest_z_test.png")
        unsup.plot_tsne_unsupervised(
            model,
            c_h,
            R,
            te,
            device,
            curvature,
            tsne_out,
            max_samples=args.tsne_max_samples,
            perplexity=args.tsne_perplexity,
            n_iter=args.tsne_iter,
            seed=args.tsne_seed,
            eval_objective=eval_objective,
            active_cluster_mask=active_mask,
        )
        image_paths.append(tsne_out)
        if args.export_cluster_samples > 0:
            ex_dir = os.path.join(run_dir, "cluster_exports_test")
            ds_ex = unsup.UnsupervisedScaledDataset(te_raw, return_raw=True)
            unsup.export_cluster_sample_images(
                model,
                c_h,
                R,
                ds_ex,
                device,
                curvature,
                ex_dir,
                samples_per_cluster=args.export_cluster_samples,
                batch_size=args.batch_size,
                seed=args.cluster_export_seed,
                split_name="test",
                active_cluster_mask=active_mask,
            )
            image_paths.append(os.path.join(ex_dir, "cluster_samples_overview.png"))
        if args.export_hotspot_analysis:
            hs_dir = os.path.join(run_dir, "hotspot_analysis_test")
            unsup.export_hotspot_class_density(
                model,
                c_h,
                R,
                te,
                device,
                curvature,
                hs_dir,
                batch_size=args.batch_size,
                split_name="test",
                active_cluster_mask=active_mask,
            )
            image_paths.extend(
                [
                    os.path.join(hs_dir, "hotspot_cluster_digit_bars.png"),
                    os.path.join(hs_dir, "hotspot_P_digit_given_cluster.png"),
                    os.path.join(hs_dir, "hotspot_P_cluster_given_digit.png"),
                ]
            )
        if args.export_cluster_neural_hotspots:
            nh_dir = os.path.join(run_dir, "cluster_neural_hotspots_test")
            unsup.export_cluster_neural_hotspots(
                model,
                c_h,
                R,
                te,
                device,
                curvature,
                nh_dir,
                batch_size=args.batch_size,
                split_name="test",
                max_samples_per_cluster=args.neural_hotspot_max_samples,
                seed=args.cluster_export_seed,
                active_cluster_mask=active_mask,
            )
            image_paths.append(os.path.join(nh_dir, "cluster_neural_hotspots_overview.png"))

    _wandb_log_run(run_dir, args, ck_meta, out_eval, image_paths)

    print(f"[done] {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser("Replay viz/eval from existing checkpoint_v2.pth (no training)")
    p.add_argument(
        "--runs_root",
        type=str,
        default="samplesvdd/hySpUnsup/runs/sensitivity_v3",
    )
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--auc_mode", type=str, default="union", choices=["union", "per_sphere"])
    p.add_argument("--tsne_max_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument("--export_cluster_samples", type=int, default=0)
    p.add_argument("--cluster_export_seed", type=int, default=42)
    p.add_argument("--export_hotspot_analysis", action="store_true")
    p.add_argument("--export_cluster_neural_hotspots", action="store_true")
    p.add_argument("--neural_hotspot_max_samples", type=int, default=50)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="hyspunsup-sensitivity")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_group", type=str, default="")
    p.add_argument("--wandb_tags", type=str, default="sensitivity,replay")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_name_prefix", type=str, default="replay-")
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.runs_root, "**", "checkpoint_v2.pth"), recursive=True))
    if not ckpts:
        raise SystemExit(f"No checkpoint_v2.pth found under {args.runs_root}")
    for ck in ckpts:
        run_dir = os.path.dirname(ck)
        _run_one(run_dir, args)


if __name__ == "__main__":
    main()
