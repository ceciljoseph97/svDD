"""
Version 2: same training as train_hyp_mnist_unsup.py, plus post-hoc sphere pruning and union metrics.

- Prune spheres with too few train assignments (argmin nearest-sphere) and/or low mean silhouette
  (Euclidean on nearest-head z in R^{z_dim}; hyperbolic geometry is not used for silhouette).
- Union evaluation: min_k (d_h^2 - R_k^2) over *active* spheres only; report frac inside union, mean margin.
- t-SNE / exports / hotspots: after pruning, **nearest-sphere id = argmin over active heads only** (pruned indices excluded), so samples are reassigned among survivors (Deep-MSVDD-style).
- t-SNE: passes active_cluster_mask to grey out pruned clusters / skip hulls; union in/out uses active set.

Run from samplesvdd: python hySpUnsup/train_hyp_mnist_unsup_v2.py --mnist_processed_dir ...

`--export_hotspot_analysis`: same as v1 — `<xp_path>/hotspot_analysis/` (split = `--cluster_export_split`).

`--export_cluster_neural_hotspots`: `<xp_path>/cluster_neural_hotspots/` — saliency + conv2 activation maps per cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import silhouette_samples
from torch.utils.data import DataLoader

_HYSP_UNSUP = Path(__file__).resolve().parent
_HYSP_CHECK = _HYSP_UNSUP.parent / "hySpCheck"
sys.path.insert(0, str(_HYSP_CHECK))
sys.path.insert(0, str(_HYSP_UNSUP))

import train_hyp_mnist_unsup as unsup  # noqa: E402

from hyperbolic_multi_sphere import (  # noqa: E402
    HyperbolicMultiSphereSVDD,
    dist_sq_to_all_centers,
    svdd_loss_one_class,
    svdd_loss_soft_boundary,
    update_radii_unsupervised,
)
from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE, recon_mse_loss  # noqa: E402

UnsupervisedScaledDataset = unsup.UnsupervisedScaledDataset
init_centers_h_unsupervised = unsup.init_centers_h_unsupervised
evaluate = unsup.evaluate
sphere_overlap_penalty = unsup.sphere_overlap_penalty
plot_tsne_unsupervised = unsup.plot_tsne_unsupervised
export_cluster_sample_images = unsup.export_cluster_sample_images
export_hotspot_class_density = unsup.export_hotspot_class_density
export_cluster_neural_hotspots = unsup.export_cluster_neural_hotspots


@torch.no_grad()
def prune_spheres_from_train(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    tr_loader: DataLoader,
    device: torch.device,
    curvature: float,
    min_cluster_members: int,
    silhouette_min: float | None,
    silhouette_max_samples: int,
    silhouette_seed: int,
) -> tuple[np.ndarray, dict]:
    """
    Returns active_mask length 10 (True = keep sphere in union scoring).
    """
    counts = np.zeros(10, dtype=np.int64)
    z_parts = []
    k_parts = []

    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        k_near = dist_sq.argmin(dim=1)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near].cpu().numpy()
        kn = k_near.cpu().numpy()
        for kk in range(10):
            counts[kk] += int(np.sum(kn == kk))
        z_parts.append(z_pick)
        k_parts.append(kn)

    Z = np.concatenate(z_parts, axis=0)
    K = np.concatenate(k_parts, axis=0)

    active = np.ones(10, dtype=bool)
    if min_cluster_members > 0:
        active &= counts >= min_cluster_members

    sil_detail: dict = {str(k): None for k in range(10)}
    if silhouette_min is not None and Z.shape[0] >= 10:
        rng = np.random.RandomState(silhouette_seed)
        if Z.shape[0] > silhouette_max_samples:
            sub = rng.choice(Z.shape[0], silhouette_max_samples, replace=False)
            Zs, Ks = Z[sub], K[sub]
        else:
            Zs, Ks = Z, K
        if len(np.unique(Ks)) >= 2:
            try:
                sil = silhouette_samples(Zs, Ks, metric="euclidean")
                for k in range(10):
                    m = Ks == k
                    if np.sum(m) >= 2:
                        sil_detail[str(k)] = float(np.mean(sil[m]))
                    elif np.sum(m) == 1:
                        sil_detail[str(k)] = float(sil[m][0])
            except Exception as e:
                sil_detail["_error"] = str(e)
        for k in range(10):
            if sil_detail[str(k)] is not None and sil_detail[str(k)] < silhouette_min:
                active[k] = False

    if not np.any(active):
        active[counts.argmax()] = True

    info = {
        "train_argmin_counts": counts.tolist(),
        "silhouette_mean_per_cluster_euclidean_z": sil_detail,
        "active_cluster_mask": active.tolist(),
    }
    return active, info


@torch.no_grad()
def union_metrics_test(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    te_loader: DataLoader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
) -> dict:
    """min over active spheres of (d^2 - R^2); frac with min <= 0."""
    model.eval()
    margins = []
    for x_scaled, _ in te_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        margin = dist_sq - (R.unsqueeze(0) ** 2)
        am = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
        margin = margin.masked_fill(~am.unsqueeze(0), float("inf"))
        u = margin.min(dim=1).values
        margins.append(u.cpu().numpy())
    u = np.concatenate(margins, axis=0)
    finite = u[np.isfinite(u)]
    return {
        "union_frac_inside": float(np.mean(finite <= 0)),
        "union_margin_mean": float(np.mean(finite)),
        "union_margin_std": float(np.std(finite)),
    }


def main():
    default_ae = _HYSP_CHECK / "runs" / "ae_pretrain_mnist" / "ae_stage1.pth"
    p = argparse.ArgumentParser("hySpUnsup v2: unsup multi-sphere + prune + union metrics")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpUnsup/runs/mnist_unsup_hyp_v2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--digits", type=str, default="all")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--ae_n_epochs", type=int, default=10)
    p.add_argument("--svdd_n_epochs", type=int, default=25)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--objective", type=str, default="union-soft", choices=["union-one", "union-soft"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=5e-5)
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--skip_ae_pretrain", action="store_true")
    p.add_argument("--ae_stage1_checkpoint_path", type=str, default=str(default_ae))
    p.add_argument("--save_ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--skip_tsne", action="store_true")
    p.add_argument("--tsne_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--tsne_max_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument("--tsne_out", type=str, default=None)
    # v2
    p.add_argument(
        "--min_cluster_members",
        type=int,
        default=0,
        help="Drop sphere k from union if train argmin count < this (0 = disable).",
    )
    p.add_argument(
        "--silhouette_min",
        type=float,
        default=None,
        help="If set, drop sphere k when mean Euclidean silhouette (z_pick) on train subsample is below this.",
    )
    p.add_argument("--silhouette_max_samples", type=int, default=800)
    p.add_argument("--silhouette_seed", type=int, default=42)
    p.add_argument(
        "--auc_mode",
        type=str,
        default="union",
        choices=["union", "per_sphere"],
        help="Same as v1: union score = min_k(d^2-R^2) over (active) spheres.",
    )
    p.add_argument("--export_cluster_samples", type=int, default=0, help="Save N raw thumbnails per nearest-sphere cluster (0=off).")
    p.add_argument("--cluster_export_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--cluster_export_seed", type=int, default=42)
    p.add_argument(
        "--export_hotspot_analysis",
        action="store_true",
        help="Save class×cluster density heatmaps and JSON (uses same split as cluster_export_split).",
    )
    p.add_argument(
        "--export_cluster_neural_hotspots",
        action="store_true",
        help="Per-cluster mean saliency |∂union_margin/∂x| + conv2 maps (union over active spheres).",
    )
    p.add_argument(
        "--neural_hotspot_max_samples",
        type=int,
        default=48,
        help="Max images per cluster to average for neural hotspot maps.",
    )

    args = p.parse_args()
    os.makedirs(args.xp_path, exist_ok=True)
    device = torch.device(args.device)
    digits = list(range(10)) if args.digits == "all" else [int(x) for x in args.digits.split(",") if x.strip()]

    tr_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="train",
        train_fraction=args.train_fraction,
        digits=digits,
        max_samples=args.max_train_samples,
        max_samples_per_class=args.max_samples_per_class,
    )
    te_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="test",
        train_fraction=args.train_fraction,
        digits=digits,
        max_samples=args.max_test_samples,
        max_samples_per_class=args.max_samples_per_class,
    )
    tr = UnsupervisedScaledDataset(tr_raw)
    te = UnsupervisedScaledDataset(te_raw)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.n_jobs_dataloader)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    backbone = MNIST_LeNet_SVDDIAE(rep_dim=args.rep_dim)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=10, c=args.curvature
    ).to(device)

    if args.skip_ae_pretrain:
        ckpt_path = args.ae_stage1_checkpoint_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise SystemExit(f"--skip_ae_pretrain requires a valid --ae_stage1_checkpoint_path (got {ckpt_path!r})")
        ae_ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        model.load_state_dict(model_state, strict=True)
        print(f"[AE] skipped pretrain; loaded: {ckpt_path}")
    else:
        opt_ae = optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for x_scaled, _digits_b in tr_loader:
                x_scaled = x_scaled.to(device)
                _rep, recon = model(x_scaled)
                loss = recon_mse_loss(recon, x_scaled)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            out_path = args.save_ae_stage1_checkpoint_path
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            torch.save({"model_state": model.state_dict()}, out_path)
            print(f"[AE] saved stage-1 checkpoint: {out_path}")

    c_h = init_centers_h_unsupervised(model, tr_loader, device=device)
    R = torch.zeros((10,), device=device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_objective = "soft-boundary" if args.objective == "union-soft" else "one-class"
    best_train_loss = float("inf")
    best_epoch = -1
    macro_auc_at_best = float("nan")
    best_per_digit = {}

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        rec_losses = []
        svdd_losses = []
        ov_losses = []
        dist_sq_chunks = [[] for _ in range(10)]

        for x_scaled, _digits_b in tr_loader:
            x_scaled = x_scaled.to(device)
            rep, recon = model(x_scaled)
            z_all_h = model.project_all_h(rep)
            dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)
            rec = recon_mse_loss(recon, x_scaled)

            min_sq, k_idx = dist_sq_all.min(dim=1)
            if args.objective == "union-one":
                sv = svdd_loss_one_class(min_sq)
            else:
                R_sel = R[k_idx]
                sv = svdd_loss_soft_boundary(min_sq, R_sel, nu=args.nu)

            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            loss = rec + args.lambda_svdd * sv + args.lambda_overlap * ov
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            rec_losses.append(float(rec.item()))
            svdd_losses.append(float(sv.item()))
            ov_losses.append(float(ov.item()))

            if args.objective == "union-soft" and ep > args.warm_up_n_epochs:
                with torch.no_grad():
                    for k in range(10):
                        m = k_idx == k
                        if torch.any(m):
                            dist_sq_chunks[k].append(dist_sq_all[m, k].detach().cpu())

        if args.objective == "union-soft" and ep > args.warm_up_n_epochs:
            R = update_radii_unsupervised(dist_sq_chunks, nu=args.nu, device=torch.device("cpu")).to(device)

        epoch_loss_mean = float(np.mean(losses))
        print(
            f"[SVDD-H-UNSUP-v2] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={epoch_loss_mean:.6f} rec={np.mean(rec_losses):.6f} svdd={np.mean(svdd_losses):.6f} "
            f"overlap={np.mean(ov_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )

        if epoch_loss_mean < best_train_loss:
            best_train_loss = epoch_loss_mean
            best_epoch = ep
            macro_b, per_b = evaluate(
                model, c_h, R, te_loader, device, eval_objective, args.curvature, auc_mode=args.auc_mode
            )
            macro_auc_at_best = macro_b
            best_per_digit = per_b
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "c_h": c_h.detach().cpu(),
                    "R": R.detach().cpu(),
                    "rep_dim": args.rep_dim,
                    "z_dim": args.z_dim,
                    "curvature": args.curvature,
                    "objective": args.objective,
                    "unsupervised": True,
                    "version": 2,
                    "best_training_loss": best_train_loss,
                    "best_epoch": best_epoch,
                },
                os.path.join(args.xp_path, "checkpoint_best.pth"),
            )
            print(
                f"[BEST] min-loss checkpoint epoch={ep:03d} train_loss={best_train_loss:.6f} macro_auc(test)={macro_b}"
            )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_digit = evaluate(
                model, c_h, R, te_loader, device, eval_objective, args.curvature, auc_mode=args.auc_mode
            )
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro} (auc_mode={args.auc_mode}; monitoring)")
        torch.save(
            {
                "model_state": model.state_dict(),
                "c_h": c_h.detach().cpu(),
                "R": R.detach().cpu(),
                "rep_dim": args.rep_dim,
                "z_dim": args.z_dim,
                "curvature": args.curvature,
                "objective": args.objective,
                "unsupervised": True,
                "version": 2,
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    best_ckpt = os.path.join(args.xp_path, "checkpoint_best.pth")
    if os.path.isfile(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"], strict=True)
        c_h = ck["c_h"].to(device)
        R = ck["R"].to(device)

    active_mask, prune_info = prune_spheres_from_train(
        model,
        c_h,
        R,
        tr_loader,
        device,
        args.curvature,
        min_cluster_members=args.min_cluster_members,
        silhouette_min=args.silhouette_min,
        silhouette_max_samples=args.silhouette_max_samples,
        silhouette_seed=args.silhouette_seed,
    )
    union_m = union_metrics_test(model, c_h, R, te_loader, device, args.curvature, active_mask)

    macro_pruned, per_pruned = evaluate(
        model,
        c_h,
        R,
        te_loader,
        device,
        eval_objective,
        args.curvature,
        active_cluster_mask=active_mask,
        auc_mode=args.auc_mode,
    )

    print(f"[v2 PRUNE] active spheres: {active_mask.astype(np.int32).tolist()}  ({int(active_mask.sum())}/10)")
    print(f"[v2 UNION test] {union_m}")
    print(f"[v2 EVAL test + active_mask] macro_auc={macro_pruned} (auc_mode={args.auc_mode})")

    out = {
        "version": 2,
        "checkpoint_selection": "min_training_loss",
        "best_epoch": int(best_epoch),
        "best_training_loss": None if best_epoch < 0 else float(best_train_loss),
        "macro_auc_test_at_best_epoch": None if np.isnan(macro_auc_at_best) else float(macro_auc_at_best),
        "per_digit_auc_at_best_epoch": best_per_digit,
        "macro_auc_test_pruned_active_spheres": None if np.isnan(macro_pruned) else float(macro_pruned),
        "per_digit_auc_test_pruned_active_spheres": per_pruned,
        "curvature": args.curvature,
        "objective": args.objective,
        "unsupervised": True,
        "ae_checkpoint": args.ae_stage1_checkpoint_path if args.skip_ae_pretrain else None,
        "prune": prune_info,
        "union_test_metrics_active_spheres": union_m,
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    torch.save(
        {
            "model_state": model.state_dict(),
            "c_h": c_h.detach().cpu(),
            "R": R.detach().cpu(),
            "rep_dim": args.rep_dim,
            "z_dim": args.z_dim,
            "curvature": args.curvature,
            "objective": args.objective,
            "unsupervised": True,
            "version": 2,
            "active_cluster_mask": active_mask.tolist(),
            "prune_info": prune_info,
        },
        os.path.join(args.xp_path, "checkpoint_v2.pth"),
    )

    if (
        (not args.skip_tsne)
        or (args.export_cluster_samples > 0)
        or args.export_hotspot_analysis
        or args.export_cluster_neural_hotspots
    ):
        if not args.skip_tsne:
            tsne_ds = te if args.tsne_split == "test" else tr
            out_png = args.tsne_out if args.tsne_out else os.path.join(args.xp_path, "tsne_unsup_v2_nearest_z.png")
            plot_tsne_unsupervised(
                model,
                c_h,
                R,
                tsne_ds,
                device,
                args.curvature,
                out_png,
                max_samples=args.tsne_max_samples,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_iter,
                seed=args.tsne_seed,
                eval_objective=eval_objective,
                active_cluster_mask=active_mask,
            )
        if args.export_cluster_samples > 0:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_ex = UnsupervisedScaledDataset(raw_base, return_raw=True)
            ex_dir = os.path.join(args.xp_path, "cluster_exports")
            export_cluster_sample_images(
                model,
                c_h,
                R,
                ds_ex,
                device,
                args.curvature,
                ex_dir,
                samples_per_cluster=args.export_cluster_samples,
                batch_size=args.batch_size,
                seed=args.cluster_export_seed,
                split_name=args.cluster_export_split,
                active_cluster_mask=active_mask,
            )
        if args.export_hotspot_analysis:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_hs = UnsupervisedScaledDataset(raw_base, return_raw=False)
            hp_dir = os.path.join(args.xp_path, "hotspot_analysis")
            export_hotspot_class_density(
                model,
                c_h,
                R,
                ds_hs,
                device,
                args.curvature,
                hp_dir,
                batch_size=args.batch_size,
                split_name=args.cluster_export_split,
                active_cluster_mask=active_mask,
            )
        if args.export_cluster_neural_hotspots:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_nh = UnsupervisedScaledDataset(raw_base, return_raw=False)
            nh_dir = os.path.join(args.xp_path, "cluster_neural_hotspots")
            export_cluster_neural_hotspots(
                model,
                c_h,
                R,
                ds_nh,
                device,
                args.curvature,
                nh_dir,
                batch_size=args.batch_size,
                split_name=args.cluster_export_split,
                max_samples_per_cluster=args.neural_hotspot_max_samples,
                seed=args.cluster_export_seed,
                active_cluster_mask=active_mask,
            )


if __name__ == "__main__":
    main()
