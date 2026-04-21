"""
Unsupervised AE-bottleneck + Hyperbolic multi-sphere SVDD (MNIST_processed).

New approach vs train_hyp_mnist_unsup_v2.py:
- Stage 1 trains only autoencoder reconstruction.
- Stage 2 optimizes hyperbolic SVDD loss on bottleneck embeddings (no reconstruction term).
- Optional encoder freeze in stage 2 to make SVDD run on fixed AE embeddings.

Run from samplesvdd:
  python hySpUnsup/train_hyp_mnist_unsup_ae_bottleneck.py --mnist_processed_dir ...
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
from torch.utils.data import DataLoader

_HYSP_UNSUP = Path(__file__).resolve().parent
_HYSP_CHECK = _HYSP_UNSUP.parent / "hySpCheck"
sys.path.insert(0, str(_HYSP_CHECK))
sys.path.insert(0, str(_HYSP_UNSUP))
sys.path.insert(0, str(_HYSP_UNSUP.parent))

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


def _parse_digit_list(s: str) -> list[int]:
    v = s.strip().lower()
    if v == "all":
        return list(range(10))
    out: list[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        d = int(tok)
        if d < 0 or d > 9:
            raise SystemExit(f"digit out of range [0..9]: {d}")
        out.append(d)
    if not out:
        raise SystemExit(f"No digits parsed from {s!r}")
    return sorted(set(out))


def _set_backbone_trainable(model: HyperbolicMultiSphereSVDD, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)


def main():
    default_ae = _HYSP_CHECK / "runs" / "ae_pretrain_mnist" / "ae_stage1.pth"
    p = argparse.ArgumentParser("Unsupervised AE-bottleneck + Hyperbolic multi-sphere SVDD")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpUnsup/runs/mnist_unsup_ae_bottleneck")
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
    p.add_argument("--n_spheres", type=int, default=10)
    p.add_argument("--curvature", type=float, default=1.0)

    p.add_argument("--ae_n_epochs", type=int, default=15)
    p.add_argument("--svdd_n_epochs", type=int, default=30)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)

    p.add_argument("--objective", type=str, default="union-soft", choices=["union-one", "union-soft"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=1.0, help="Weight for SVDD term in stage-2.")
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument(
        "--freeze_encoder_stage2",
        action="store_true",
        help="If set, stage-2 trains projection heads only on fixed AE bottleneck embeddings.",
    )

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
    p.add_argument("--auc_mode", type=str, default="union", choices=["union", "per_sphere"])
    p.add_argument("--export_cluster_samples", type=int, default=0)
    p.add_argument("--cluster_export_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--cluster_export_seed", type=int, default=42)
    p.add_argument("--export_hotspot_analysis", action="store_true")
    p.add_argument("--export_cluster_neural_hotspots", action="store_true")
    p.add_argument("--neural_hotspot_max_samples", type=int, default=48)

    args = p.parse_args()
    if args.n_spheres < 1:
        raise SystemExit("--n_spheres must be >= 1")

    os.makedirs(args.xp_path, exist_ok=True)
    device = torch.device(args.device)
    digits = _parse_digit_list(args.digits)

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
        backbone=backbone,
        rep_dim=args.rep_dim,
        z_dim=args.z_dim,
        n_digits=args.n_spheres,
        c=args.curvature,
    ).to(device)

    # Stage-1: AE pretrain only.
    if args.skip_ae_pretrain:
        ckpt_path = args.ae_stage1_checkpoint_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise SystemExit(f"--skip_ae_pretrain requires a valid --ae_stage1_checkpoint_path (got {ckpt_path!r})")
        ae_ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("proj_heads.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("proj_heads.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch outside projection heads.\n"
                f"missing(non-proj)={bad_missing}\n"
                f"unexpected(non-proj)={bad_unexpected}"
            )
        print(f"[AE] loaded checkpoint: {ckpt_path}")
    else:
        opt_ae = optim.Adam(model.backbone.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for x_scaled, _digits_b in tr_loader:
                x_scaled = x_scaled.to(device)
                rep = model.backbone.encode(x_scaled)
                recon = model.backbone.decode(rep)
                loss = recon_mse_loss(recon, x_scaled)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            print(f"[AE-BOT] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            out_path = args.save_ae_stage1_checkpoint_path
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            torch.save({"model_state": model.state_dict()}, out_path)
            print(f"[AE] saved stage-1 checkpoint: {out_path}")

    # Stage-2: hyperbolic SVDD on bottleneck embeddings (no reconstruction loss).
    if args.freeze_encoder_stage2:
        _set_backbone_trainable(model, trainable=False)
        print("[SVDD] encoder frozen: stage-2 trains projection heads only.")
    else:
        _set_backbone_trainable(model, trainable=True)
        print("[SVDD] encoder trainable: stage-2 fine-tunes AE bottleneck with SVDD objective.")

    c_h = init_centers_h_unsupervised(model, tr_loader, device=device)
    R = torch.zeros((args.n_spheres,), device=device)
    eval_objective = "soft-boundary" if args.objective == "union-soft" else "one-class"

    if args.freeze_encoder_stage2:
        svdd_params = list(model.proj_heads.parameters())
    else:
        svdd_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(svdd_params, lr=args.lr, weight_decay=args.weight_decay)

    best_train_loss = float("inf")
    best_epoch = -1
    macro_auc_at_best = float("nan")
    best_per_digit = {}
    full_active_mask = np.ones(args.n_spheres, dtype=bool)

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        svdd_losses = []
        ov_losses = []
        dist_sq_chunks = [[] for _ in range(args.n_spheres)]

        for x_scaled, _digits_b in tr_loader:
            x_scaled = x_scaled.to(device)
            rep = model.backbone.encode(x_scaled)
            if args.freeze_encoder_stage2:
                rep = rep.detach()
            z_all_h = model.project_all_h(rep)
            dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)

            min_sq, k_idx = dist_sq_all.min(dim=1)
            if args.objective == "union-one":
                sv = svdd_loss_one_class(min_sq)
            else:
                R_sel = R[k_idx]
                sv = svdd_loss_soft_boundary(min_sq, R_sel, nu=args.nu)

            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            loss = args.lambda_svdd * sv + args.lambda_overlap * ov
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            svdd_losses.append(float(sv.item()))
            ov_losses.append(float(ov.item()))

            if args.objective == "union-soft" and ep > args.warm_up_n_epochs:
                with torch.no_grad():
                    for k in range(args.n_spheres):
                        m = k_idx == k
                        if torch.any(m):
                            dist_sq_chunks[k].append(dist_sq_all[m, k].detach().cpu())

        if args.objective == "union-soft" and ep > args.warm_up_n_epochs:
            R = update_radii_unsupervised(dist_sq_chunks, nu=args.nu, device=torch.device("cpu")).to(device)

        epoch_loss_mean = float(np.mean(losses))
        print(
            f"[SVDD-AE-BOT] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={epoch_loss_mean:.6f} svdd={np.mean(svdd_losses):.6f} "
            f"overlap={np.mean(ov_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )

        if epoch_loss_mean < best_train_loss:
            best_train_loss = epoch_loss_mean
            best_epoch = ep
            macro_b, per_b = evaluate(
                model,
                c_h,
                R,
                te_loader,
                device,
                eval_objective,
                args.curvature,
                active_cluster_mask=full_active_mask,
                auc_mode=args.auc_mode,
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
                    "n_spheres": args.n_spheres,
                    "objective": args.objective,
                    "unsupervised": True,
                    "embedding_source": "ae_bottleneck",
                    "freeze_encoder_stage2": bool(args.freeze_encoder_stage2),
                    "best_training_loss": best_train_loss,
                    "best_epoch": best_epoch,
                },
                os.path.join(args.xp_path, "checkpoint_best.pth"),
            )
            print(f"[BEST] epoch={ep:03d} train_loss={best_train_loss:.6f} macro_auc(test)={macro_b}")

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, _per_digit = evaluate(
                model,
                c_h,
                R,
                te_loader,
                device,
                eval_objective,
                args.curvature,
                active_cluster_mask=full_active_mask,
                auc_mode=args.auc_mode,
            )
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro} (auc_mode={args.auc_mode})")

        torch.save(
            {
                "model_state": model.state_dict(),
                "c_h": c_h.detach().cpu(),
                "R": R.detach().cpu(),
                "rep_dim": args.rep_dim,
                "z_dim": args.z_dim,
                "curvature": args.curvature,
                "n_spheres": args.n_spheres,
                "objective": args.objective,
                "unsupervised": True,
                "embedding_source": "ae_bottleneck",
                "freeze_encoder_stage2": bool(args.freeze_encoder_stage2),
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    out = {
        "variant": "unsup_ae_bottleneck_hypersvdd",
        "checkpoint_selection": "min_training_loss",
        "best_epoch": int(best_epoch),
        "best_training_loss": None if best_epoch < 0 else float(best_train_loss),
        "macro_auc_test_at_best_epoch": None if np.isnan(macro_auc_at_best) else float(macro_auc_at_best),
        "per_digit_auc_at_best_epoch": best_per_digit,
        "curvature": args.curvature,
        "n_spheres": args.n_spheres,
        "digits": digits,
        "objective": args.objective,
        "unsupervised": True,
        "embedding_source": "ae_bottleneck",
        "freeze_encoder_stage2": bool(args.freeze_encoder_stage2),
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    best_ckpt = os.path.join(args.xp_path, "checkpoint_best.pth")
    need_viz = (
        (not args.skip_tsne)
        or (args.export_cluster_samples > 0)
        or args.export_hotspot_analysis
        or args.export_cluster_neural_hotspots
    )
    if need_viz:
        if os.path.isfile(best_ckpt):
            ck = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ck["model_state"], strict=True)
            c_h_viz = ck["c_h"].to(device)
            R_viz = ck["R"].to(device)
        else:
            c_h_viz, R_viz = c_h, R

        if not args.skip_tsne:
            tsne_ds = te if args.tsne_split == "test" else tr
            out_png = args.tsne_out if args.tsne_out else os.path.join(args.xp_path, "tsne_unsup_ae_bottleneck_nearest_z.png")
            plot_tsne_unsupervised(
                model,
                c_h_viz,
                R_viz,
                tsne_ds,
                device,
                args.curvature,
                out_png,
                max_samples=args.tsne_max_samples,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_iter,
                seed=args.tsne_seed,
                eval_objective=eval_objective,
                active_cluster_mask=full_active_mask,
            )

        if args.export_cluster_samples > 0:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_ex = UnsupervisedScaledDataset(raw_base, return_raw=True)
            ex_dir = os.path.join(args.xp_path, "cluster_exports")
            export_cluster_sample_images(
                model,
                c_h_viz,
                R_viz,
                ds_ex,
                device,
                args.curvature,
                ex_dir,
                samples_per_cluster=args.export_cluster_samples,
                batch_size=args.batch_size,
                seed=args.cluster_export_seed,
                split_name=args.cluster_export_split,
                active_cluster_mask=full_active_mask,
            )

        if args.export_hotspot_analysis:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_hs = UnsupervisedScaledDataset(raw_base, return_raw=False)
            hp_dir = os.path.join(args.xp_path, "hotspot_analysis")
            export_hotspot_class_density(
                model,
                c_h_viz,
                R_viz,
                ds_hs,
                device,
                args.curvature,
                hp_dir,
                batch_size=args.batch_size,
                split_name=args.cluster_export_split,
                active_cluster_mask=full_active_mask,
            )

        if args.export_cluster_neural_hotspots:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_nh = UnsupervisedScaledDataset(raw_base, return_raw=False)
            nh_dir = os.path.join(args.xp_path, "cluster_neural_hotspots")
            export_cluster_neural_hotspots(
                model,
                c_h_viz,
                R_viz,
                ds_nh,
                device,
                args.curvature,
                nh_dir,
                batch_size=args.batch_size,
                split_name=args.cluster_export_split,
                max_samples_per_cluster=args.neural_hotspot_max_samples,
                seed=args.cluster_export_seed,
                active_cluster_mask=full_active_mask,
            )


if __name__ == "__main__":
    main()
