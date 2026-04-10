"""
Purely unsupervised hyperbolic multi-sphere SVDD on Iris (tabular).

- Label-free preprocessing: StandardScaler fit on train only.
- Label-free center init: per-head mean of hyperbolic embeddings over train (then proj_ball).
- Same loss recipe as hySpCheckunSup/train_hyp_mnist_multi.py (soft SVDD, sep, overlap, entropy).
- Labels only for metrics / optional plot coloring.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HYCHECK_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "hySpCheck"))
if HYCHECK_DIR not in sys.path:
    sys.path.insert(0, HYCHECK_DIR)

from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers
from hyperbolic_ops import hyp_distance, proj_ball

from iris_backbone import IrisMLPSVDDIAE, recon_mse_loss

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


@torch.no_grad()
def init_centers_unsup_head_means(model: HyperbolicMultiSphereSVDD, loader: DataLoader, device: torch.device, eps: float = 1e-5):
    """
    Unsupervised init: for each head k, mean of z_k = project_k(rep) mapped to ball, averaged over ALL training points.
    Then project to ball (same device as model).
    """
    K = model.n_digits
    z_dim = model.z_dim
    c = torch.zeros((K, z_dim), device=device)
    n = 0
    for xb, _ in loader:
        xb = xb.to(device)
        rep, _ = model(xb)
        zh = model.project_all_h(rep)
        c += zh.sum(dim=0)
        n += zh.size(0)
    c = c / float(max(n, 1))
    c = proj_ball(c, c=model.curvature, eps=eps)
    return c


@torch.no_grad()
def sort_clusters_flagged(c_h, R, all_dist, all_z_sel, assign, args, curvature: float):
    K = c_h.size(0)
    if K <= 1 or all_dist.numel() == 0:
        return c_h, R, {"n": 0, "under": 0, "over": 0, "populated": K, "reseeded": 0, "merged": 0}

    n = int(all_dist.size(0))
    counts = torch.bincount(assign, minlength=K)
    min_count = max(1, int(args.sort_min_frac * n))
    max_count = max(min_count + 1, int(args.sort_max_frac * n))
    under = [int(k) for k in range(K) if int(counts[k].item()) < min_count]
    over = [int(k) for k in range(K) if int(counts[k].item()) > max_count]
    populated = [int(k) for k in range(K) if int(counts[k].item()) >= min_count]
    reseeded = merged = 0

    for i, k_tiny in enumerate(under):
        if i >= len(over):
            break
        k_big = over[i]
        m = assign == k_big
        if not torch.any(m):
            continue
        idx_local = torch.argmax(all_dist[m, k_big])
        global_idx = torch.where(m)[0][idx_local]
        c_h[k_tiny] = all_z_sel[global_idx].to(c_h.device)
        R[k_tiny] = torch.clamp(R[k_big] * 0.5, min=0.0)
        reseeded += 1

    for k_tiny in under[len(over) :]:
        if not populated:
            break
        best_k, best_d = None, None
        for k_pop in populated:
            d_val = hyp_distance(c_h[k_tiny : k_tiny + 1], c_h[k_pop : k_pop + 1], c=curvature).item()
            if best_d is None or d_val < best_d:
                best_d, best_k = d_val, k_pop
        c_h[k_tiny] = c_h[best_k].detach().clone()
        R[k_tiny] = torch.clamp(R[best_k], min=0.0)
        merged += 1

    return c_h, R, {"n": n, "under": len(under), "over": len(over), "populated": len(populated), "reseeded": reseeded, "merged": merged}


class IrisDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


@torch.no_grad()
def evaluate(model, c_h, R, loader, device, objective: str, curvature: float, n_clusters: int):
    model.eval()
    all_d, all_s, all_dist = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        rep, _ = model(xb)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        if objective == "soft-boundary":
            scores_all = dist_sq_all - (R.unsqueeze(0) ** 2)
        else:
            scores_all = dist_sq_all
        all_d.append(yb.cpu())
        all_s.append(scores_all.cpu())
        all_dist.append(dist_sq_all.cpu())

    y_np = torch.cat(all_d).numpy()
    scores_np = torch.cat(all_s).numpy()
    dist_np = torch.cat(all_dist).numpy()

    per_k = {}
    aucs = []
    for k in range(n_clusters):
        t = (y_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(t, scores_np[:, k])
        except ValueError:
            auc = np.nan
        per_k[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")

    pred = np.argmin(dist_np, axis=1).astype(np.int64)
    nmi = float(normalized_mutual_info_score(y_np, pred))
    ari = float(adjusted_rand_score(y_np, pred))

    hung = float("nan")
    macro_aligned = float("nan")
    per_lbl_aligned = {}
    if linear_sum_assignment is not None and y_np.size > 0:
        u = np.unique(y_np)
        conf = np.zeros((u.size, n_clusters), dtype=np.int64)
        for i, lbl in enumerate(u):
            m = y_np == lbl
            conf[i, :] = np.bincount(pred[m], minlength=n_clusters)
        cost = conf.max() - conf
        ri, ci = linear_sum_assignment(cost)
        hung = float(conf[ri, ci].sum() / max(1, y_np.shape[0]))

        # One-vs-rest AUC per *true* class using the sphere column matched by Hungarian (fixes index permutation).
        aucs_al = []
        for k in range(len(ri)):
            lbl = int(u[ri[k]])
            j = int(ci[k])
            t = (y_np != lbl).astype(np.int32)
            try:
                a = roc_auc_score(t, scores_np[:, j])
            except ValueError:
                a = np.nan
            per_lbl_aligned[str(lbl)] = None if np.isnan(a) else float(a)
            if not np.isnan(a):
                aucs_al.append(float(a))
        macro_aligned = float(np.mean(aucs_al)) if aucs_al else float("nan")

    unsup = {
        "hungarian_acc": None if np.isnan(hung) else hung,
        "nmi": nmi,
        "ari": ari,
        "macro_auc_hungarian_aligned": None if np.isnan(macro_aligned) else float(macro_aligned),
        "per_class_auc_hungarian_aligned": per_lbl_aligned,
    }
    return macro, per_k, unsup


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


@torch.no_grad()
def collect_embeddings(model, loader, c_h, device, curvature: float):
    model.eval()
    Z, A, Y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        rep, _ = model(xb)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        assign = torch.argmin(dist_sq, dim=1)
        z_sel = z_all[torch.arange(z_all.size(0), device=device), assign]
        Z.append(z_sel.cpu())
        A.append(assign.cpu())
        Y.append(yb.clone())
    return torch.cat(Z, 0), torch.cat(A, 0), torch.cat(Y, 0)


def to_2d(z_np: np.ndarray) -> np.ndarray:
    if z_np.shape[1] > 2:
        return PCA(n_components=2).fit_transform(z_np)
    return z_np


def plot_poincare_single(z_np, c_np, R_vec, out_path: str, title: str, colors=None):
    import matplotlib.pyplot as plt

    z2 = to_2d(z_np)
    c2 = to_2d(c_np)
    fig, ax = plt.subplots(figsize=(6, 6))
    circ = plt.Circle((0, 0), 1.0, fill=False, color="black")
    ax.add_artist(circ)
    if colors is None:
        ax.scatter(z2[:, 0], z2[:, 1], s=40, alpha=0.7)
    else:
        col = np.asarray(colors)
        ax.scatter(z2[:, 0], z2[:, 1], c=col, s=40, cmap="Set1", vmin=0, vmax=max(2, float(col.max())), alpha=0.8)
    for i in range(c2.shape[0]):
        ax.scatter(c2[i, 0], c2[i, 1], marker="x", s=120, c="black")
        r = float(R_vec[i])
        ax.add_artist(plt.Circle((c2[i, 0], c2[i, 1]), r, fill=False, linestyle="--", color="gray"))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_inline_outline_poincare(
    z_per_sample: torch.Tensor,
    c_h: torch.Tensor,
    R: torch.Tensor,
    assign: torch.Tensor,
    y_true: torch.Tensor,
    out_path: str,
    color_by_true: bool,
):
    """
    One row, K panels: each panel k shows Poincaré disk (PCA2D), sphere k outline, points whose nearest center is k.
    Point color: pred cluster (default) or true label (verification).
    """
    import matplotlib.pyplot as plt

    z_np = z_per_sample.numpy()
    c_np = c_h.detach().cpu().numpy()
    R_vec = R.detach().cpu().numpy()
    assign_np = assign.numpy()
    y_np = y_true.numpy()
    K = c_h.size(0)

    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), squeeze=False)
    for k in range(K):
        ax = axes[0, k]
        m = assign_np == k
        if not np.any(m):
            pts = np.zeros((0, z_np.shape[1]))
            col = np.array([])
        else:
            pts = z_np[m]
            col = (y_np[m] if color_by_true else assign_np[m]).astype(np.float64)

        pts2 = to_2d(pts) if pts.shape[0] else np.zeros((0, 2))
        c2 = to_2d(c_np)
        disk = plt.Circle((0, 0), 1.0, fill=False, color="black")
        ax.add_artist(disk)
        if pts2.shape[0]:
            ax.scatter(pts2[:, 0], pts2[:, 1], c=col, s=45, cmap="Set1", vmin=0, vmax=max(K - 1, 2), alpha=0.85)
        ax.scatter(c2[k, 0], c2[k, 1], marker="x", s=150, c="black", zorder=5)
        r = float(R_vec[k])
        ax.add_artist(plt.Circle((c2[k, 0], c2[k, 1]), r, fill=False, linestyle="--", color="red", linewidth=1.5))
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(f"Sphere {k} (n={int(m.sum())})")

    plt.suptitle("Inline outline: one panel per hypersphere (PCA view; radii approximate)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_tsne(z_np: np.ndarray, assign: np.ndarray, y_true: np.ndarray, out_path: str, color_by_true: bool):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        return
    import matplotlib.pyplot as plt

    z2 = TSNE(n_components=2, init="pca", random_state=42).fit_transform(z_np)
    c = y_true if color_by_true else assign
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=c, cmap="Set1", s=50, alpha=0.8)
    ax.set_title("t-SNE of selected hyperbolic embeddings")
    fig.colorbar(sc, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("Iris — unsupervised hyperbolic multi-sphere SVDD")
    p.add_argument("--xp_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--test_size", type=float, default=0.25)
    p.add_argument("--stratify_split", action="store_true", help="Use stratified train/test split (uses labels for split only).")
    p.add_argument("--n_clusters", type=int, default=3, help="K hyperspheres (Iris has 3 species).")
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=8)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ae_n_epochs", type=int, default=180)
    p.add_argument("--svdd_n_epochs", type=int, default=220)
    p.add_argument("--ae_lr", type=float, default=1e-2)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--objective", type=str, default="soft-boundary", choices=["one-class", "soft-boundary"])
    p.add_argument("--nu", type=float, default=0.15)
    p.add_argument("--lambda_svdd", type=float, default=1e-3)
    p.add_argument("--lambda_sep", type=float, default=0.05)
    p.add_argument("--sep_margin", type=float, default=0.05)
    p.add_argument("--lambda_entropy", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.25)
    p.add_argument("--lambda_overlap", type=float, default=0.01)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=15)
    p.add_argument("--radius_ema", type=float, default=0.85)
    p.add_argument("--radius_max_step", type=float, default=0.08)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--enable_cluster_sorting", action="store_true")
    p.add_argument("--sort_every", type=int, default=1)
    p.add_argument("--sort_min_frac", type=float, default=0.05)
    p.add_argument("--sort_max_frac", type=float, default=0.45)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.005)
    p.add_argument("--plot_embeddings", action="store_true")
    p.add_argument("--plot_color_true", action="store_true", help="Color plots by true species (eval/visualization only).")
    p.add_argument("--plot_tsne", action="store_true")
    args = p.parse_args()

    xp = args.xp_path or os.path.join(THIS_DIR, "runs", "iris_unsup_hyp")
    os.makedirs(xp, exist_ok=True)

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)

    iris = load_iris()
    X, y = iris.data.astype(np.float64), iris.target.astype(np.int64)
    if args.stratify_split:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    K = args.n_clusters
    tr_ds = IrisDataset(X_tr, y_tr)
    te_ds = IrisDataset(X_te, y_te)
    g = torch.Generator()
    g.manual_seed(args.seed)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=g)
    tr_eval = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=g)
    te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, generator=g)

    in_dim = X_tr.shape[1]
    backbone = IrisMLPSVDDIAE(in_dim=in_dim, rep_dim=args.rep_dim, hidden=args.hidden)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=K, c=args.curvature
    ).to(device)

    opt_ae = optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
    for ep in range(1, args.ae_n_epochs + 1):
        model.train()
        losses = []
        for xb, _ in tr_loader:
            xb = xb.to(device)
            _, recon = model(xb)
            loss = recon_mse_loss(recon, xb)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
            losses.append(float(loss.item()))
        if ep % 20 == 0 or ep == 1:
            print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")

    torch.save({"model_state": model.state_dict(), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}, os.path.join(xp, "ae_pretrained.pth"))

    c_h = init_centers_unsup_head_means(model, tr_eval, device)
    R = torch.zeros((K,), device=device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_sel, best_ep, best_macro, best_perk, best_unsup = -1e12, -1, float("nan"), {}, {}
    no_improve = 0

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        for xb, _ in tr_loader:
            xb = xb.to(device)
            rep, recon = model(xb)
            z_all = model.project_all_h(rep)
            dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=args.curvature)
            p = torch.softmax(-dist_sq_all / args.temperature, dim=1)
            rec = recon_mse_loss(recon, xb)
            if args.objective == "soft-boundary":
                Re = R.unsqueeze(0)
                svdd_all = Re ** 2 + (1.0 / args.nu) * torch.relu(dist_sq_all - Re ** 2)
            else:
                svdd_all = dist_sq_all
            svdd = torch.mean(torch.sum(p * svdd_all, dim=1))
            top2 = torch.topk(dist_sq_all, k=min(2, K), largest=False)
            if K >= 2:
                sep = torch.mean(torch.relu(top2.values[:, 0] - top2.values[:, 1] + args.sep_margin))
            else:
                sep = torch.tensor(0.0, device=device)
            ov = sphere_overlap_penalty(c_h, R, args.curvature, args.margin_overlap)
            eps = 1e-8
            H = -torch.sum(p * torch.log(p + eps), dim=1)
            entropy_loss = -torch.mean(H)
            loss = rec + args.lambda_svdd * svdd + args.lambda_sep * sep + args.lambda_overlap * ov + args.lambda_entropy * entropy_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
            with torch.no_grad():
                all_d, all_z = [], []
                for xb, _ in tr_eval:
                    xb = xb.to(device)
                    rep, _ = model(xb)
                    z_all = model.project_all_h(rep)
                    d2 = dist_sq_to_all_centers(z_all, c_h, curvature=args.curvature)
                    ab = torch.argmin(d2, dim=1)
                    z_sel = z_all[torch.arange(z_all.size(0), device=device), ab]
                    all_d.append(d2.cpu())
                    all_z.append(z_sel.cpu())
                all_d = torch.cat(all_d, dim=0)
                all_z = torch.cat(all_z, dim=0)
                assign = torch.argmin(all_d, dim=1)
                new_R = torch.zeros_like(R)
                for k in range(K):
                    dk = all_d[assign == k, k]
                    new_R[k] = torch.quantile(dk.to(device), 1.0 - args.nu) if dk.numel() > 0 else 0.0
                ema = args.radius_ema * R + (1.0 - args.radius_ema) * new_R
                R = torch.clamp(R + torch.clamp(ema - R, min=-args.radius_max_step, max=args.radius_max_step), min=0.0)
                if args.enable_cluster_sorting and ep % max(1, args.sort_every) == 0:
                    c_h, R, info = sort_clusters_flagged(c_h, R, all_d, all_z, assign, args, args.curvature)
                    print(f"[SORT] ep={ep:03d} {info}")

        print(f"[UNSUP-H] {ep:03d}/{args.svdd_n_epochs} loss={np.mean(losses):.6f}")

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, perk, unsup = evaluate(model, c_h, R, te_loader, device, args.objective, args.curvature, K)
            ma_h = unsup.get("macro_auc_hungarian_aligned")
            print(
                f"[EVAL] ep={ep:03d} macro_auc_fixed={macro} macro_auc_hung={ma_h} "
                f"h_acc={unsup['hungarian_acc']} nmi={unsup['nmi']} ari={unsup['ari']}"
            )
            m = float(unsup["hungarian_acc"]) if unsup["hungarian_acc"] is not None else (macro if not np.isnan(macro) else -1e12)
            if m > best_sel + args.early_stop_min_delta:
                best_sel, best_ep, best_macro, best_perk, best_unsup = m, ep, macro, perk, unsup
                no_improve = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "c_h": c_h.cpu(),
                        "R": R.cpu(),
                        "n_clusters": K,
                        "curvature": args.curvature,
                        "scaler_mean": scaler.mean_.tolist(),
                        "scaler_scale": scaler.scale_.tolist(),
                    },
                    os.path.join(xp, "checkpoint_best.pth"),
                )
            else:
                no_improve += 1
                if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                    print(f"[EARLY-STOP] ep={ep} best_ep={best_ep}")
                    break

        torch.save(
            {"model_state": model.state_dict(), "c_h": c_h.cpu(), "R": R.cpu(), "n_clusters": K},
            os.path.join(xp, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_ep) if best_ep >= 0 else None,
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "best_macro_auc_hungarian_aligned": best_unsup.get("macro_auc_hungarian_aligned"),
        "per_class_auc": best_perk,
        "per_class_auc_hungarian_aligned": best_unsup.get("per_class_auc_hungarian_aligned"),
        "best_hungarian_acc": best_unsup.get("hungarian_acc"),
        "best_nmi": best_unsup.get("nmi"),
        "best_ari": best_unsup.get("ari"),
        "n_clusters": K,
        "curvature": args.curvature,
        "purely_unsupervised_train": True,
        "preprocess": "StandardScaler_train_only",
        "center_init": "unsup_head_means_proj_ball",
    }
    with open(os.path.join(xp, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    if args.plot_embeddings:
        z_emb, assign, y_all = collect_embeddings(model, tr_eval, c_h, device, args.curvature)
        color = y_all.numpy() if args.plot_color_true else assign.numpy()
        plot_poincare_single(
            z_emb.numpy(),
            c_h.cpu().numpy(),
            R.cpu().numpy(),
            os.path.join(xp, "poincare_clusters.png"),
            "Iris — Poincaré (PCA); unsupervised training",
            colors=color,
        )
        plot_inline_outline_poincare(
            z_emb,
            c_h,
            R,
            assign,
            y_all,
            os.path.join(xp, "per_class_inline_outline.png"),
            color_by_true=args.plot_color_true,
        )
        if args.plot_tsne:
            plot_tsne(z_emb.numpy(), assign.numpy(), y_all.numpy(), os.path.join(xp, "tsne_clusters.png"), args.plot_color_true)
        print(f"[PLOT] saved under {xp}")


if __name__ == "__main__":
    main()
