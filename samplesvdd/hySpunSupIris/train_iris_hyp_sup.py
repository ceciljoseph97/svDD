"""
Supervised hyperbolic multi-sphere SVDD on Iris — aligned with hySpCheck/train_hyp_mnist_multi.py.

- Stage 1: AE reconstruction (labels unused in loss).
- Stage 2: per-sample true-class projection project_self_h(rep, y), SVDD on true sphere,
  inter-class exclusion on all spheres, overlap penalty; radii from labeled assignments after warmup.

Preprocessing: StandardScaler fit on train only (tabular; no per-class scaling like MNIST min-max).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HYCHECK_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "hySpCheck"))
import sys

if HYCHECK_DIR not in sys.path:
    sys.path.insert(0, HYCHECK_DIR)

from hyperbolic_multi_sphere import (
    HyperbolicMultiSphereSVDD,
    dist_sq_to_all_centers,
    hyp_dist_sq_to_centers,
    init_centers_h,
    svdd_loss_one_class,
    svdd_loss_soft_boundary,
    update_radii,
)
from hyperbolic_ops import hyp_distance

from iris_backbone import IrisMLPSVDDIAE, recon_mse_loss
from iris_plots import collect_embeddings_nearest, plot_inline_outline_poincare, plot_poincare_single, plot_tsne


class IrisDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


@torch.no_grad()
def evaluate(model, c_h, R, loader, device, objective: str, curvature: float, n_classes: int):
    """Hyperbolic scores (same geometry as training), macro one-vs-rest AUC per class index."""
    model.eval()
    all_d, all_s = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        rep, _ = model(xb)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        if objective == "soft-boundary":
            scores_all = dist_sq_all - (R.unsqueeze(0) ** 2)
        else:
            scores_all = dist_sq_all
        all_d.append(yb.cpu())
        all_s.append(scores_all.cpu())

    y_np = torch.cat(all_d).numpy()
    scores_np = torch.cat(all_s).numpy()
    per_k = {}
    aucs = []
    for k in range(n_classes):
        t = (y_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(t, scores_np[:, k])
        except ValueError:
            auc = np.nan
        per_k[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")
    return macro, per_k


def inter_class_exclusion_loss(dist_sq_all, R, y, margin: float) -> torch.Tensor:
    B, K = dist_sq_all.shape
    target = (R.unsqueeze(0) + float(margin)) ** 2
    viol = torch.relu(target - dist_sq_all)
    true_mask = torch.nn.functional.one_hot(y, num_classes=K).bool()
    viol = viol.masked_fill(true_mask, 0.0)
    return torch.mean(viol)


def sphere_overlap_penalty(c_h, R, curvature: float, margin: float) -> torch.Tensor:
    K = c_h.size(0)
    if K <= 1:
        return torch.tensor(0.0, device=c_h.device)
    penalties = []
    for a in range(K):
        for b in range(a + 1, K):
            d_ab = hyp_distance(c_h[a : a + 1], c_h[b : b + 1], c=curvature).squeeze(0)
            penalties.append(torch.relu((R[a] + R[b] + float(margin)) - d_ab))
    return torch.mean(torch.stack(penalties)) if penalties else torch.tensor(0.0, device=c_h.device)


def main():
    p = argparse.ArgumentParser("Supervised hyperbolic multi-sphere SVDD on Iris (hySpCheck-style)")
    p.add_argument("--xp_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--stratify_split", action="store_true", help="Stratified train/test split.")
    p.add_argument("--n_classes", type=int, default=3, help="Number of classes / hyperspheres (Iris = 3).")
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=8)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ae_n_epochs", type=int, default=80)
    p.add_argument("--svdd_n_epochs", type=int, default=120)
    p.add_argument("--ae_lr", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--objective", type=str, default="soft-boundary", choices=["one-class", "soft-boundary"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=1e-3)
    p.add_argument("--lambda_excl", type=float, default=1e-2)
    p.add_argument("--margin_excl", type=float, default=0.1)
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--skip_ae_pretrain", action="store_true")
    p.add_argument("--ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--save_ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--plot_embeddings", action="store_true", help="Save Poincaré + inline outline PNGs (like unsup).")
    p.add_argument("--plot_tsne", action="store_true")
    p.add_argument("--plot_out_dir", type=str, default=None, help="Override output dir for figures (default: xp_path).")
    p.add_argument(
        "--plot_color_pred",
        action="store_true",
        help="Color points by nearest-sphere assignment (default: color by true Iris species).",
    )
    args = p.parse_args()

    xp = args.xp_path or os.path.join(THIS_DIR, "runs", "iris_sup_hyp")
    os.makedirs(xp, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    K = args.n_classes
    iris = load_iris()
    X, y = iris.data.astype(np.float64), iris.target.astype(np.int64)
    if args.stratify_split:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    tr_ds = IrisDataset(X_tr, y_tr)
    te_ds = IrisDataset(X_te, y_te)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    tr_eval = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    in_dim = X_tr.shape[1]
    backbone = IrisMLPSVDDIAE(in_dim=in_dim, rep_dim=args.rep_dim, hidden=args.hidden)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=K, c=args.curvature
    ).to(device)

    if args.skip_ae_pretrain:
        if not args.ae_stage1_checkpoint_path:
            raise SystemExit("--skip_ae_pretrain requires --ae_stage1_checkpoint_path")
        ck = torch.load(args.ae_stage1_checkpoint_path, map_location="cpu")
        st = ck["model_state"] if isinstance(ck, dict) and "model_state" in ck else ck
        model.load_state_dict(st, strict=True)
        print(f"[AE] skipped; loaded {args.ae_stage1_checkpoint_path}")
    else:
        opt_ae = optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for xb, _y in tr_loader:
                xb = xb.to(device)
                _, recon = model(xb)
                loss = recon_mse_loss(recon, xb)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            if ep % 20 == 0 or ep == 1:
                print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_ae_stage1_checkpoint_path)) or ".", exist_ok=True)
            torch.save({"model_state": model.state_dict()}, args.save_ae_stage1_checkpoint_path)
            print(f"[AE] saved {args.save_ae_stage1_checkpoint_path}")

    c_h = init_centers_h(model, tr_loader, device=device)
    R = torch.zeros((K,), device=device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_macro, best_ep, best_per = -1e9, -1, {}

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses, recs, svs, excls, ovs = [], [], [], [], []
        dist_sq_by_class = [[] for _ in range(K)]

        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            rep, recon = model(xb)
            z_h = model.project_self_h(rep, yb)
            c_b = c_h[yb]
            dist_sq = hyp_dist_sq_to_centers(z_h, c_b, c=args.curvature)
            z_all_h = model.project_all_h(rep)
            dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)
            rec = recon_mse_loss(recon, xb)
            if args.objective == "soft-boundary":
                R_b = R[yb]
                sv = svdd_loss_soft_boundary(dist_sq, R_b, nu=args.nu)
            else:
                sv = svdd_loss_one_class(dist_sq)
            excl = inter_class_exclusion_loss(dist_sq_all, R, yb, margin=args.margin_excl)
            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            loss = rec + args.lambda_svdd * sv + args.lambda_excl * excl + args.lambda_overlap * ov
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            recs.append(float(rec.item()))
            svs.append(float(sv.item()))
            excls.append(float(excl.item()))
            ovs.append(float(ov.item()))
            if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
                for k in range(K):
                    m = yb == k
                    if torch.any(m):
                        dist_sq_by_class[k].append(dist_sq[m].detach().cpu())

        if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
            R = update_radii(dist_sq_by_class, nu=args.nu, device=device)

        print(
            f"[SVDD-H] {ep:03d}/{args.svdd_n_epochs} loss={np.mean(losses):.6f} rec={np.mean(recs):.6f} "
            f"svdd={np.mean(svs):.6f} excl={np.mean(excls):.6f} overlap={np.mean(ovs):.6f} R_mean={float(R.mean()):.4f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_k = evaluate(model, c_h, R, te_loader, device, args.objective, args.curvature, K)
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro}")
            metric = macro if not np.isnan(macro) else -1e12
            if metric > best_macro:
                best_macro, best_ep, best_per = macro, ep, per_k
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "c_h": c_h.detach().cpu(),
                        "R": R.detach().cpu(),
                        "rep_dim": args.rep_dim,
                        "z_dim": args.z_dim,
                        "n_classes": K,
                        "curvature": args.curvature,
                        "objective": args.objective,
                        "lambda_excl": args.lambda_excl,
                        "margin_excl": args.margin_excl,
                        "lambda_overlap": args.lambda_overlap,
                        "margin_overlap": args.margin_overlap,
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                    },
                    os.path.join(xp, "checkpoint_best.pth"),
                )
        torch.save(
            {
                "model_state": model.state_dict(),
                "c_h": c_h.detach().cpu(),
                "R": R.detach().cpu(),
                "n_classes": K,
                "curvature": args.curvature,
                "objective": args.objective,
                "lambda_excl": args.lambda_excl,
                "margin_excl": args.margin_excl,
                "lambda_overlap": args.lambda_overlap,
                "margin_overlap": args.margin_overlap,
            },
            os.path.join(xp, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_ep),
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "per_class_best_auc": best_per,
        "curvature": args.curvature,
        "objective": args.objective,
        "lambda_excl": args.lambda_excl,
        "margin_excl": args.margin_excl,
        "lambda_overlap": args.lambda_overlap,
        "margin_overlap": args.margin_overlap,
        "supervised": True,
    }
    with open(os.path.join(xp, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    if args.plot_embeddings or args.plot_tsne:
        plot_dir = args.plot_out_dir or xp
        os.makedirs(plot_dir, exist_ok=True)
        best_p = os.path.join(xp, "checkpoint_best.pth")
        if os.path.isfile(best_p):
            ck = torch.load(best_p, map_location=device)
            model.load_state_dict(ck["model_state"])
            c_h = ck["c_h"].to(device)
            R = ck["R"].to(device)
        model.eval()
        z_emb, assign, y_all = collect_embeddings_nearest(model, tr_eval, c_h, device, args.curvature)
        color_by_true = not bool(args.plot_color_pred)
        if args.plot_embeddings:
            colors = y_all.numpy() if color_by_true else assign.numpy()
            plot_poincare_single(
                z_emb.numpy(),
                c_h.detach().cpu().numpy(),
                R.detach().cpu().numpy(),
                os.path.join(plot_dir, "poincare_clusters.png"),
                "Iris supervised — Poincaré (PCA); points colored by " + ("true class" if color_by_true else "pred sphere"),
                colors=colors,
            )
            plot_inline_outline_poincare(
                z_emb,
                c_h,
                R,
                assign,
                y_all,
                os.path.join(plot_dir, "per_class_inline_outline.png"),
                color_by_true=color_by_true,
            )
        if args.plot_tsne:
            plot_tsne(
                z_emb.numpy(),
                assign.numpy(),
                y_all.numpy(),
                os.path.join(plot_dir, "tsne_clusters.png"),
                color_by_true=color_by_true,
            )
        print(f"[PLOT] figures saved under {plot_dir}")


if __name__ == "__main__":
    main()
