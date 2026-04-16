"""
Euclidean Deep Multi-sphere SVDD — aligned with Deep-Multi-Sphere-SVDD-master (Ghafoori & Leckie),
with a single Euclidean embedding z = W·h and Euclidean distances to K centers.

Compared to train_hyp_mnist_unsup.py (HyperbolicMultiSphereSVDD):
  - Here: **one** shared z per sample + **K free centers** (Euclidean MSVDD: one rep + K centers).
  - There: **K projection heads**, each mapping the same rep to a different point (digit-style).

Pipeline (reference-aligned):
  - AE pretrain (optional skip + load checkpoint).
  - K-means on z (first n_batches of train) to init centers; per-cluster (1−ν) quantile radii.
  - Training: NPR argmin_k ||z-c_k||²; hinge sum(max(0, d²−R_k²)) / (B·ν).
  - Each epoch after warm-up: update R from full-train NPR assignments (same rule as update_radii_unsupervised).

Run from repo:
  cd samplesvdd/hySpUnsup && python train_hyp_mnist_dmsvdd_hyp.py --mnist_processed_dir ../../CVAEChecked/Data/MNIST_processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

_HYSP_UNSUP = Path(__file__).resolve().parent
_HYSP_CHECK = _HYSP_UNSUP.parent / "hySpCheck"
sys.path.insert(0, str(_HYSP_CHECK))
sys.path.insert(0, str(_HYSP_UNSUP))

from euclidean_dmsvdd import EuclideanDMSVDD, dist_sq_z_to_centers  # noqa: E402
from hyperbolic_multi_sphere import update_radii_unsupervised  # noqa: E402
from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE, recon_mse_loss  # noqa: E402
from train_hyp_mnist_unsup import (  # noqa: E402
    UnsupervisedScaledDataset,
    export_cluster_neural_hotspots,
    export_cluster_sample_images,
    export_hotspot_class_density,
    plot_tsne_unsupervised,
)


class _DMSVDDVisAdapter(torch.nn.Module):
    """
    Adapter so DMSVDD can reuse unsup v2 visualization utilities.
    Exposes:
      - forward(x) -> (rep, recon)
      - project_all_h(rep) -> (B, K, z_dim) where z is repeated for each cluster
    """

    def __init__(self, model: EuclideanDMSVDD, n_vis_clusters: int | None = None):
        super().__init__()
        self.model = model
        self.backbone = model.backbone
        self.curvature = model.curvature
        self.n_clusters = int(model.n_clusters if n_vis_clusters is None else n_vis_clusters)

    def forward(self, x: torch.Tensor):
        rep, recon = self.model.backbone(x)
        return rep, recon

    def project_all_h(self, rep: torch.Tensor) -> torch.Tensor:
        z = self.model._to_tangent(rep)
        return z.unsqueeze(1).expand(-1, self.n_clusters, -1)


@torch.no_grad()
def evaluate_dmsvdd(
    model: EuclideanDMSVDD,
    R: torch.Tensor,
    loader: DataLoader,
    device: torch.device,
    curvature: float,
    n_clusters: int,
    active_cluster_mask: np.ndarray | None = None,
    auc_mode: str = "union",
):
    model.eval()
    if active_cluster_mask is None:
        active_cluster_mask = np.ones(n_clusters, dtype=bool)

    all_d, all_margins, all_dist_cols = [], [], []
    for x_scaled, digits in loader:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        z, _rep, _ = model.embed_h(x_scaled)
        c_h = model.c_h
        dist_sq_all = dist_sq_z_to_centers(z, c_h, curvature)
        margin_all = dist_sq_all - (R.unsqueeze(0) ** 2)
        am = torch.as_tensor(active_cluster_mask, device=device, dtype=torch.bool)
        margin_all = margin_all.masked_fill(~am.unsqueeze(0), float("inf"))
        union_score = margin_all.min(dim=1).values
        all_d.append(digits.cpu())
        all_margins.append(union_score.cpu())
        all_dist_cols.append(margin_all.cpu())

    digits_np = torch.cat(all_d).numpy()
    union_np = torch.cat(all_margins).numpy()
    margin_np = torch.cat(all_dist_cols).numpy()

    per_digit = {}
    aucs = []
    for k in range(10):
        y = (digits_np != k).astype(np.int32)
        if auc_mode == "union":
            scores_k = union_np
        else:
            if k >= n_clusters:
                per_digit[str(k)] = None
                continue
            scores_k = margin_np[:, k]
        try:
            auc = roc_auc_score(y, scores_k)
        except ValueError:
            auc = np.nan
        per_digit[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")
    return macro, per_digit


@torch.no_grad()
def kmeans_init_centers_and_R(
    model: EuclideanDMSVDD,
    loader: DataLoader,
    device: torch.device,
    curvature: float,
    nu: float,
    kmeans_batches: int,
) -> torch.Tensor:
    """
    Like reference initialize_c_kmeans: cluster in encoder space, then place centers in z_dim.

    K-means runs on backbone representations h (same as Euclidean DMSVDD on feature_layer output).
    Cluster centers are padded/truncated to z_dim and scaled into the ball for _c_raw init.
    Radii: (1−nu) quantile of Euclidean squared distance to assigned center on the same init batch.
    """
    reps = []
    bi = 0
    for x_scaled, _ in loader:
        x_scaled = x_scaled.to(device)
        rep, _recon = model.backbone(x_scaled)
        reps.append(rep.cpu().numpy())
        bi += 1
        if bi >= kmeans_batches:
            break
    H = np.concatenate(reps, axis=0)
    K = model.n_clusters
    z_dim = model.z_dim
    rep_dim = H.shape[1]
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(H)
    cc = km.cluster_centers_
    if z_dim <= rep_dim:
        cc_z = cc[:, :z_dim].copy()
    else:
        cc_z = np.pad(cc, ((0, 0), (0, z_dim - rep_dim)), mode="constant")

    cc_z = cc_z.astype(np.float32)
    model._c_raw.copy_(torch.from_numpy(cc_z).to(device))

    Ht = torch.from_numpy(H).float().to(device)
    z_init = []
    for i in range(0, Ht.size(0), 512):
        rep_b = Ht[i : i + 512]
        z_init.append(model._to_tangent(rep_b))
    Zt = torch.cat(z_init, dim=0)
    c_h = model.c_h
    dist_sq = dist_sq_z_to_centers(Zt, c_h, curvature)
    k_idx = dist_sq.argmin(dim=1)
    chunks = [[] for _ in range(K)]
    for i in range(Zt.size(0)):
        ki = int(k_idx[i].item())
        chunks[ki].append(dist_sq[i : i + 1, ki].detach().cpu())
    R_cpu = update_radii_unsupervised(chunks, nu=nu, device=torch.device("cpu"))
    return R_cpu.to(device)


def main():
    default_ae = _HYSP_CHECK / "runs" / "ae_pretrain_mnist" / "ae_stage1.pth"
    default_data = _HYSP_UNSUP.parents[1] / "CVAEChecked" / "Data" / "MNIST_processed"
    default_vis_ckpt = _HYSP_UNSUP / "hySpUnsup" / "runs" / "mnist_dmsvdd_hyp" / "checkpoint_best.pth"
    p = argparse.ArgumentParser("Euclidean DMSVDD (paper-style: 1 embedding + K centers)")
    p.add_argument("--mnist_processed_dir", type=str, default=str(default_data))
    p.add_argument("--xp_path", type=str, default="hySpUnsup/runs/mnist_dmsvdd_hyp")
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
    p.add_argument("--n_cluster", type=int, default=10, help="K spheres (paper uses n_cluster).")
    p.add_argument("--curvature", type=float, default=1.0, help="Unused in Euclidean mode (kept for CLI compatibility).")
    p.add_argument("--ae_n_epochs", type=int, default=10)
    p.add_argument("--svdd_n_epochs", type=int, default=25)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=1.0, help="Scale on hinge / (B*nu).")
    p.add_argument("--warm_up_n_epochs", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--kmeans_init_batches", type=int, default=50, help="Batches of train used for K-means + R init (cf. c_mean_init_n_batches).")
    p.add_argument("--skip_ae_pretrain", action="store_true")
    p.add_argument("--ae_stage1_checkpoint_path", type=str, default=str(default_ae))
    p.add_argument("--save_ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--auc_mode", type=str, default="union", choices=["union", "per_sphere"])
    p.add_argument("--skip_tsne", action="store_true")
    p.add_argument("--tsne_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--tsne_max_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument("--tsne_out", type=str, default=None)
    p.add_argument("--export_cluster_samples", type=int, default=0)
    p.add_argument("--cluster_export_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--cluster_export_seed", type=int, default=42)
    p.add_argument("--export_hotspot_analysis", action="store_true")
    p.add_argument("--export_cluster_neural_hotspots", action="store_true")
    p.add_argument("--neural_hotspot_max_samples", type=int, default=48)
    p.add_argument(
        "--vis_only",
        action="store_true",
        help="Skip training, load --vis_checkpoint_path and run visualization exports.",
    )
    p.add_argument(
        "--vis_checkpoint_path",
        type=str,
        default=str(default_vis_ckpt),
        help="DMSVDD checkpoint used for --vis_only.",
    )
    args = p.parse_args()

    os.makedirs(args.xp_path, exist_ok=True)
    device = torch.device(args.device)

    def _parse_digits(spec: str | None, fallback_all: bool = False) -> list[int]:
        if spec is None:
            return list(range(10)) if fallback_all else []
        s = str(spec).strip().lower()
        if s == "all":
            return list(range(10))
        return [int(x) for x in str(spec).split(",") if x.strip()]

    digits = _parse_digits(args.digits, fallback_all=True)
    train_digits = digits
    test_digits = digits
    K = int(args.n_cluster)

    tr_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="train",
        train_fraction=args.train_fraction,
        digits=train_digits,
        max_samples=args.max_train_samples,
        max_samples_per_class=args.max_samples_per_class,
    )
    te_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="test",
        train_fraction=args.train_fraction,
        digits=test_digits,
        max_samples=args.max_test_samples,
        max_samples_per_class=args.max_samples_per_class,
    )
    tr = UnsupervisedScaledDataset(tr_raw)
    te = UnsupervisedScaledDataset(te_raw)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.n_jobs_dataloader)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    backbone = MNIST_LeNet_SVDDIAE(rep_dim=args.rep_dim)
    model = EuclideanDMSVDD(
        backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_clusters=K, c=args.curvature
    ).to(device)

    def _run_visualizations(R_vis: torch.Tensor):
        n_vis = 10
        vis_adapter = _DMSVDDVisAdapter(model, n_vis_clusters=n_vis).to(device)
        vis_adapter.eval()
        c_raw = model.c_h.detach()
        if K < n_vis:
            pad_n = n_vis - K
            c_pad = c_raw[:1].expand(pad_n, -1).clone()
            c_h_vis = torch.cat([c_raw, c_pad], dim=0)
            R_pad = R_vis.new_zeros((pad_n,))
            R_vis_eff = torch.cat([R_vis, R_pad], dim=0)
            active_mask = np.array(([True] * K) + ([False] * pad_n), dtype=bool)
        elif K > n_vis:
            c_h_vis = c_raw[:n_vis]
            R_vis_eff = R_vis[:n_vis]
            active_mask = np.ones(n_vis, dtype=bool)
        else:
            c_h_vis = c_raw
            R_vis_eff = R_vis
            active_mask = np.ones(n_vis, dtype=bool)
        if not args.skip_tsne:
            tsne_ds = te if args.tsne_split == "test" else tr
            out_png = args.tsne_out if args.tsne_out else os.path.join(args.xp_path, "tsne_dmsvdd_hyp_nearest_z.png")
            plot_tsne_unsupervised(
                vis_adapter,
                c_h_vis,
                R_vis_eff,
                tsne_ds,
                device,
                args.curvature,
                out_png,
                max_samples=args.tsne_max_samples,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_iter,
                seed=args.tsne_seed,
                eval_objective="soft-boundary",
                active_cluster_mask=active_mask,
            )
        if args.export_cluster_samples > 0:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_ex = UnsupervisedScaledDataset(raw_base, return_raw=True)
            ex_dir = os.path.join(args.xp_path, "cluster_exports")
            export_cluster_sample_images(
                vis_adapter,
                c_h_vis,
                R_vis_eff,
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
            hs_dir = os.path.join(args.xp_path, "hotspot_analysis")
            export_hotspot_class_density(
                vis_adapter,
                c_h_vis,
                R_vis_eff,
                ds_hs,
                device,
                args.curvature,
                hs_dir,
                batch_size=args.batch_size,
                split_name=args.cluster_export_split,
                active_cluster_mask=active_mask,
            )
        if args.export_cluster_neural_hotspots:
            raw_base = te_raw if args.cluster_export_split == "test" else tr_raw
            ds_nh = UnsupervisedScaledDataset(raw_base, return_raw=False)
            nh_dir = os.path.join(args.xp_path, "cluster_neural_hotspots")
            export_cluster_neural_hotspots(
                vis_adapter,
                c_h_vis,
                R_vis_eff,
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

    if args.vis_only:
        if not os.path.isfile(args.vis_checkpoint_path):
            raise SystemExit(f"--vis_only requires existing --vis_checkpoint_path, got: {args.vis_checkpoint_path}")
        ck = torch.load(args.vis_checkpoint_path, map_location="cpu")
        model.load_state_dict(ck["model_state"], strict=True)
        R_ck = ck.get("R", None)
        if R_ck is None:
            raise SystemExit(f"Checkpoint missing 'R': {args.vis_checkpoint_path}")
        R = R_ck.to(device)
        print(f"[VIS] loaded checkpoint: {args.vis_checkpoint_path}")
        _run_visualizations(R)
        return

    if args.skip_ae_pretrain:
        ckpt_path = args.ae_stage1_checkpoint_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise SystemExit(f"--skip_ae_pretrain requires valid --ae_stage1_checkpoint_path (got {ckpt_path!r})")
        ae_ckpt = torch.load(ckpt_path, map_location="cpu")
        ms = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        # Partial load: only backbone keys from a full multi-head checkpoint
        own = model.state_dict()
        loaded = {k: v for k, v in ms.items() if k in own and v.shape == own[k].shape}
        model.load_state_dict(loaded, strict=False)
        print(f"[AE] partial load from {ckpt_path} ({len(loaded)} tensors); training heads from scratch.")
    else:
        opt_ae = optim.Adam(model.backbone.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for x_scaled, _ in tr_loader:
                x_scaled = x_scaled.to(device)
                _rep, recon = model.backbone(x_scaled)
                loss = recon_mse_loss(recon, x_scaled)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_ae_stage1_checkpoint_path)), exist_ok=True)
            torch.save({"model_state": model.state_dict()}, args.save_ae_stage1_checkpoint_path)
            print(f"[AE] saved {args.save_ae_stage1_checkpoint_path}")

    print("[init] K-means on z + quantile radii...")
    R = kmeans_init_centers_and_R(
        model,
        tr_loader,
        device,
        args.curvature,
        args.nu,
        args.kmeans_init_batches,
    )

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_macro = -1.0
    best_ep = -1
    best_per = {}

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        hinge_losses = []
        dist_chunks = [[] for _ in range(K)]

        for x_scaled, _ in tr_loader:
            x_scaled = x_scaled.to(device)
            z, _rep, recon = model.embed_h(x_scaled)
            c_h = model.c_h
            dist_sq_all = dist_sq_z_to_centers(z, c_h, args.curvature)
            k_idx = dist_sq_all.argmin(dim=1)
            d_sel = dist_sq_all[torch.arange(x_scaled.size(0), device=device), k_idx]
            hinge = F.relu(d_sel - R[k_idx].detach() ** 2)
            loss_h = hinge.sum() / (float(x_scaled.size(0)) * args.nu)
            loss = args.lambda_svdd * loss_h

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            hinge_losses.append(float(loss_h.item()))
            if ep > args.warm_up_n_epochs:
                with torch.no_grad():
                    for kk in range(K):
                        m = k_idx == kk
                        if torch.any(m):
                            dist_chunks[kk].append(dist_sq_all[m, kk].detach().cpu())

        if ep > args.warm_up_n_epochs:
            R = update_radii_unsupervised(dist_chunks, nu=args.nu, device=torch.device("cpu")).to(device)

        print(
            f"[DMSVDD-H] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={np.mean(losses):.6f} hinge={np.mean(hinge_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_d = evaluate_dmsvdd(
                model, R, te_loader, device, args.curvature, K, auc_mode=args.auc_mode
            )
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro}")
            if macro > best_macro:
                best_macro = macro
                best_ep = ep
                best_per = per_d
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "R": R.detach().cpu(),
                        "rep_dim": args.rep_dim,
                        "z_dim": args.z_dim,
                        "n_clusters": K,
                        "curvature": args.curvature,
                        "nu": args.nu,
                        "dmsvdd_hyp": True,
                    },
                    os.path.join(args.xp_path, "checkpoint_best.pth"),
                )

        torch.save(
            {
                "model_state": model.state_dict(),
                "R": R.detach().cpu(),
                "rep_dim": args.rep_dim,
                "z_dim": args.z_dim,
                "n_clusters": K,
                "curvature": args.curvature,
                "dmsvdd_hyp": True,
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_ep),
        "best_macro_auc": float(best_macro) if best_macro > -0.5 else None,
        "per_digit_best": best_per,
        "n_cluster": K,
        "curvature": args.curvature,
        "nu": args.nu,
        "lambda_svdd": args.lambda_svdd,
        "dmsvdd_hyp": True,
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    if (
        (not args.skip_tsne)
        or (args.export_cluster_samples > 0)
        or args.export_hotspot_analysis
        or args.export_cluster_neural_hotspots
    ):
        best_ckpt = os.path.join(args.xp_path, "checkpoint_best.pth")
        if os.path.isfile(best_ckpt):
            ck = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ck["model_state"], strict=True)
            R = ck["R"].to(device)
        _run_visualizations(R)


if __name__ == "__main__":
    main()
