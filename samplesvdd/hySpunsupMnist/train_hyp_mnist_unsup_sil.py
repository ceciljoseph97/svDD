"""
Unsupervised hyperbolic multi-sphere SVDD on MNIST with silhouette-driven training.

- Optional preprocessing: per-digit minmax (label leak for scaling) vs label-free GCN only.
- Optional center init: labeled (digit-conditioned), unsup head means, KMeans on rep, random ball.
- Main training signal: differentiable silhouette surrogate on encoder `rep` with soft hyperbolic
  assignments (intra scatter minus weighted inter-mean separation). True sklearn silhouette logged.
- Optional sphere overlap penalty, optional cluster pruning (same maintenance as hySpCheckunSup).
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score, silhouette_score
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
HYCHECK_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "hySpCheck"))
if HYCHECK_DIR not in sys.path:
    sys.path.insert(0, HYCHECK_DIR)

from mnist_local import (
    MNISTDigitsProcessedRawDataset,
    MNIST_LeNet_SVDDIAE,
    global_contrast_normalization,
    preprocess_batch_by_digit_minmax,
    recon_mse_loss,
)
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers, init_centers_h
from hyperbolic_ops import hyp_distance, proj_ball

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


def seed_everything(seed: int, deterministic: bool = True) -> None:
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ScaledDigitDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_raw, d, _ = self.base[idx]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = preprocess_batch_by_digit_minmax(x_raw.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
        return x_scaled, d_t


class GCNOnlyDigitDataset(Dataset):
    """Label-free scaling: global contrast normalization only (no per-digit minmax)."""

    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_raw, d, _ = self.base[idx]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = global_contrast_normalization(x_raw.clone(), scale="l1")
        return x_scaled, d_t


@torch.no_grad()
def init_centers_unsup_head_means(model: HyperbolicMultiSphereSVDD, loader: DataLoader, device: torch.device, eps: float = 1e-5):
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
def init_centers_random(K: int, z_dim: int, device: torch.device, curvature: float, eps: float, scale: float = 0.05):
    c = torch.randn(K, z_dim, device=device) * float(scale)
    return proj_ball(c, c=curvature, eps=eps)


@torch.no_grad()
def init_centers_kmeans_on_rep(model: HyperbolicMultiSphereSVDD, loader: DataLoader, device: torch.device, eps: float = 1e-5):
    """KMeans on encoder rep; each head k gets mean hyperbolic embed z_k over points in cluster k."""
    K = model.n_digits
    all_reps: list[torch.Tensor] = []
    all_z: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            rep, _ = model(xb)
            za = model.project_all_h(rep)
            all_reps.append(rep.cpu())
            all_z.append(za.cpu())
    rep_all = torch.cat(all_reps, dim=0)
    z_all = torch.cat(all_z, dim=0)
    n = rep_all.size(0)
    if n < K:
        return init_centers_random(K, model.z_dim, device, model.curvature, eps)

    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(rep_all.numpy())
    lab = km.labels_
    z_dim = z_all.size(-1)
    c = torch.zeros(K, z_dim)
    for k in range(K):
        m = lab == k
        if not np.any(m):
            c[k] = torch.randn(z_dim) * 0.02
        else:
            c[k] = z_all[torch.from_numpy(m), k, :].mean(0)
    c = proj_ball(c.to(device), c=model.curvature, eps=eps)
    return c


def silhouette_surrogate_loss(rep: torch.Tensor, p_assign: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Differentiable proxy aligned with silhouette: minimize soft intra-cluster variance in `rep`,
    maximize separation between soft cluster means. Returns (loss_scalar, intra_term, inter_term).
    """
    B, K = p_assign.shape
    d = rep.size(1)
    p_sum = p_assign.sum(0).clamp_min(eps)
    mu = (p_assign.T @ rep) / p_sum.unsqueeze(1)
    diff = rep.unsqueeze(1) - mu.unsqueeze(0)
    per_pair = (diff**2).sum(-1)
    intra = (p_assign * per_pair).sum(dim=1).mean()

    if K <= 1:
        inter = torch.tensor(0.0, device=rep.device, dtype=rep.dtype)
    else:
        acc = torch.tensor(0.0, device=rep.device, dtype=rep.dtype)
        cnt = 0
        for a in range(K):
            for b in range(a + 1, K):
                acc = acc + (mu[a] - mu[b]).pow(2).sum()
                cnt += 1
        inter = acc / max(cnt, 1)
    loss = intra - inter
    return loss, intra, inter


@torch.no_grad()
def sort_clusters_flagged(c_h, R, all_dist, all_z_sel, assign, args, curvature: float):
    K = c_h.size(0)
    if K <= 1 or all_dist.numel() == 0:
        return c_h, R, {"n": 0, "under": 0, "over": 0, "populated": K, "reseeded": 0, "merged": 0}

    n = int(all_dist.size(0))
    counts = torch.bincount(assign, minlength=K)
    min_count = max(1, int(args.prune_min_frac * n))
    max_count = max(min_count + 1, int(args.prune_max_frac * n))
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
        d_big = all_dist[m, k_big]
        idx_local = torch.argmax(d_big)
        global_idx = torch.where(m)[0][idx_local]
        c_h[k_tiny] = all_z_sel[global_idx].to(c_h.device)
        R[k_tiny] = torch.clamp(R[k_big] * 0.5, min=0.0)
        reseeded += 1

    for k_tiny in under[len(over) :]:
        if not populated:
            break
        nearest = None
        nearest_d = None
        for k_pop in populated:
            d_val = hyp_distance(c_h[k_tiny : k_tiny + 1], c_h[k_pop : k_pop + 1], c=curvature).squeeze(0)
            dd = float(d_val.item())
            if nearest_d is None or dd < nearest_d:
                nearest_d = dd
                nearest = k_pop
        if nearest is not None:
            c_h[k_tiny] = c_h[nearest].detach().clone()
            R[k_tiny] = torch.clamp(R[nearest], min=0.0)
            merged += 1

    return c_h, R, {
        "n": n,
        "under": len(under),
        "over": len(over),
        "populated": len(populated),
        "reseeded": reseeded,
        "merged": merged,
    }


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
def evaluate(model, c_h, R, loader, device, objective, curvature):
    model.eval()
    all_d = []
    all_s = []
    all_dist = []
    for x_scaled, digits in loader:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        if objective == "soft-boundary":
            scores_all = dist_sq_all - (R.unsqueeze(0) ** 2)
        else:
            scores_all = dist_sq_all
        all_d.append(digits.cpu())
        all_s.append(scores_all.cpu())
        all_dist.append(dist_sq_all.cpu())

    digits_np = torch.cat(all_d).numpy()
    scores_np = torch.cat(all_s).numpy()
    dist_np = torch.cat(all_dist).numpy()
    per_digit = {}
    aucs = []
    for k in range(scores_np.shape[1]):
        y = (digits_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(y, scores_np[:, k])
        except ValueError:
            auc = np.nan
        per_digit[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")

    pred_cluster = np.argmin(dist_np, axis=1).astype(np.int64)
    nmi = float(normalized_mutual_info_score(digits_np, pred_cluster))
    ari = float(adjusted_rand_score(digits_np, pred_cluster))

    hungarian_acc = float("nan")
    if linear_sum_assignment is not None and digits_np.size > 0:
        true_labels = np.unique(digits_np)
        n_true = true_labels.size
        n_pred = dist_np.shape[1]
        conf = np.zeros((n_true, n_pred), dtype=np.int64)
        for i, lbl in enumerate(true_labels):
            m = digits_np == lbl
            counts = np.bincount(pred_cluster[m], minlength=n_pred)
            conf[i, :] = counts
        if conf.size > 0:
            cost = conf.max() - conf
            row_ind, col_ind = linear_sum_assignment(cost)
            matched = conf[row_ind, col_ind].sum()
            hungarian_acc = float(matched / max(1, digits_np.shape[0]))

    unsup = {
        "hungarian_acc": None if np.isnan(hungarian_acc) else hungarian_acc,
        "nmi": None if np.isnan(nmi) else nmi,
        "ari": None if np.isnan(ari) else ari,
    }
    return macro, per_digit, unsup, dist_np, pred_cluster


@torch.no_grad()
def silhouette_sklearn_on_loader(model, c_h, loader, device, curvature, max_samples: int, feature: str) -> float | None:
    """Euclidean silhouette on `rep` or selected hyperbolic z with labels = argmin hyperbolic dist."""
    model.eval()
    reps = []
    zs = []
    dist_rows = []
    n = 0
    for x_scaled, _ in loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        if feature == "rep":
            reps.append(rep.cpu().numpy())
        else:
            assign = torch.argmin(dist_sq_all, dim=1)
            z_sel = z_all[torch.arange(z_all.size(0), device=device), assign]
            zs.append(z_sel.cpu().numpy())
        dist_rows.append(dist_sq_all.cpu().numpy())
        n += x_scaled.size(0)
        if max_samples is not None and n >= max_samples:
            break

    if feature == "rep":
        X = np.concatenate(reps, axis=0)
    else:
        X = np.concatenate(zs, axis=0)
    dist_np = np.concatenate(dist_rows, axis=0)
    if max_samples is not None and X.shape[0] > max_samples:
        X = X[:max_samples]
        dist_np = dist_np[:max_samples]
    labels = np.argmin(dist_np, axis=1).astype(np.int64)
    uniq = np.unique(labels)
    if uniq.size < 2 or X.shape[0] <= uniq.size:
        return None
    try:
        return float(silhouette_score(X, labels, metric="euclidean"))
    except ValueError:
        return None


@torch.no_grad()
def collect_embeddings(model, loader, c_h, device, curvature):
    model.eval()
    all_z = []
    all_assign = []
    for x_scaled, _ in loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        assign = torch.argmin(dist_sq_all, dim=1)
        z_sel = z_all[torch.arange(z_all.size(0), device=device), assign]
        all_z.append(z_sel.cpu())
        all_assign.append(assign.cpu())
    return torch.cat(all_z, dim=0), torch.cat(all_assign, dim=0)


def to_2d(z_np):
    if z_np.shape[1] > 2:
        return PCA(n_components=2).fit_transform(z_np)
    return z_np


def plot_poincare(z_tensor, c_h, R, out_path=None, assign=None):
    import matplotlib.pyplot as plt

    z_np = z_tensor.detach().cpu().numpy()
    z_2d = to_2d(z_np)
    c_np = c_h.detach().cpu().numpy()
    c_2d = to_2d(c_np)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    disk = plt.Circle((0.0, 0.0), 1.0, fill=False, color="black")
    ax.add_artist(disk)

    if assign is None:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], s=5)
    else:
        ax.scatter(z_2d[:, 0], z_2d[:, 1], c=assign.numpy(), s=5, cmap="tab10")

    for i in range(c_2d.shape[0]):
        ax.scatter(c_2d[i, 0], c_2d[i, 1], marker="x", s=100)
        r = float(R[i].item())
        circle = plt.Circle((c_2d[i, 0], c_2d[i, 1]), r, fill=False, linestyle="--")
        ax.add_artist(circle)

    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_aspect("equal")
    ax.set_title("Hyperbolic Clusters (Poincare Disk)")

    if out_path:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser("MNIST unsupervised hyperbolic SVDD — silhouette primary + optional overlap/pruning")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpunsupMnist/runs/mnist_unsup_sil")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--digits", type=str, default="all")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--ae_n_epochs", type=int, default=10)
    p.add_argument("--svdd_n_epochs", type=int, default=25)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument(
        "--preprocess",
        type=str,
        default="digit_minmax",
        choices=["digit_minmax", "gcn_only"],
        help="digit_minmax uses label-conditioned scaling; gcn_only is label-free.",
    )
    p.add_argument(
        "--center_init",
        type=str,
        default="unsup_head_means",
        choices=["labeled", "unsup_head_means", "kmeans_rep", "random"],
        help="Hyperbolic center initialization (labeled uses digit-conditioned z).",
    )
    p.add_argument("--objective", type=str, default="soft-boundary", choices=["one-class", "soft-boundary"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_rec", type=float, default=0.5, help="Reconstruction weight (silhouette surrogate is primary).")
    p.add_argument("--lambda_svdd", type=float, default=5e-5)
    p.add_argument("--lambda_sep", type=float, default=0.05)
    p.add_argument("--sep_margin", type=float, default=0.1)
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--lambda_entropy", type=float, default=0.01)
    p.add_argument(
        "--lambda_sil",
        type=float,
        default=1.0,
        help="Weight for differentiable silhouette surrogate (intra - inter on rep, soft assign).",
    )
    p.add_argument(
        "--lambda_sil_inter",
        type=float,
        default=1.0,
        help="Multiplier for inter-mean separation term inside surrogate (loss = intra - lambda_sil_inter * inter).",
    )
    p.add_argument("--warm_up_n_epochs", type=int, default=5)
    p.add_argument("--radius_ema", type=float, default=0.9)
    p.add_argument("--radius_max_step", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--sil_eval_max_samples", type=int, default=8000, help="Cap samples for sklearn silhouette (train loader).")
    p.add_argument(
        "--sil_eval_feature",
        type=str,
        default="rep",
        choices=["rep", "z_sel"],
        help="Feature space for logged sklearn silhouette.",
    )
    p.add_argument(
        "--select_metric",
        type=str,
        default="silhouette",
        choices=["silhouette", "hungarian", "macro_auc"],
        help="Checkpoint selection on validation.",
    )
    p.add_argument("--plot_embeddings", action="store_true")
    p.add_argument("--plot_out", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--enable_pruning", action="store_true", help="Cluster maintenance (tiny/oversized), same as hySpCheckunSup sort.")
    p.add_argument("--prune_every", type=int, default=1)
    p.add_argument("--prune_min_frac", type=float, default=0.02)
    p.add_argument("--prune_max_frac", type=float, default=0.25)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--skip_ae_pretrain", action="store_true")
    p.add_argument("--ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--save_ae_stage1_checkpoint_path", type=str, default=None)
    args = p.parse_args()

    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.makedirs(args.xp_path, exist_ok=True)
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    digits = list(range(10)) if args.digits == "all" else [int(x) for x in args.digits.split(",") if x.strip()]
    n_clusters = len(digits)

    tr_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="train",
        train_fraction=args.train_fraction,
        digits=digits,
        max_samples=args.max_train_samples,
    )
    te_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir,
        split="test",
        train_fraction=args.train_fraction,
        digits=digits,
        max_samples=args.max_test_samples,
    )
    DS = ScaledDigitDataset if args.preprocess == "digit_minmax" else GCNOnlyDigitDataset
    tr = DS(tr_raw)
    te = DS(te_raw)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    tr_loader = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_jobs_dataloader,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )
    tr_loader_eval = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_jobs_dataloader,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )
    te_loader = DataLoader(
        te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_jobs_dataloader,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )

    backbone = MNIST_LeNet_SVDDIAE(rep_dim=args.rep_dim)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=n_clusters, c=args.curvature
    ).to(device)

    if args.skip_ae_pretrain:
        if not args.ae_stage1_checkpoint_path:
            raise SystemExit("--skip_ae_pretrain requires --ae_stage1_checkpoint_path")
        ae_ckpt = torch.load(args.ae_stage1_checkpoint_path, map_location="cpu")
        model_state = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        model.load_state_dict(model_state, strict=True)
        print(f"[AE] skipped pretrain; loaded: {args.ae_stage1_checkpoint_path}")
    else:
        opt_ae = optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for x_scaled, _ in tr_loader:
                x_scaled = x_scaled.to(device)
                _, recon = model(x_scaled)
                loss = recon_mse_loss(recon, x_scaled)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            out_path = args.save_ae_stage1_checkpoint_path
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            torch.save({"model_state": model.state_dict()}, out_path)
            print(f"[AE] saved: {out_path}")

    if args.center_init == "labeled":
        c_h = init_centers_h(model, tr_loader_eval, device=device)[:n_clusters]
    elif args.center_init == "unsup_head_means":
        c_h = init_centers_unsup_head_means(model, tr_loader_eval, device=device)
    elif args.center_init == "kmeans_rep":
        c_h = init_centers_kmeans_on_rep(model, tr_loader_eval, device=device)
    else:
        c_h = init_centers_random(n_clusters, args.z_dim, device, args.curvature, eps=1e-5)

    R = torch.zeros((n_clusters,), device=device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro = -1e9
    best_epoch = -1
    best_per_digit = {}
    best_unsup = {}
    best_sil = None
    best_select_metric = -1e12
    no_improve_epochs = 0

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        rec_losses = []
        svdd_losses = []
        sep_losses = []
        ov_losses = []
        ent_losses = []
        sil_losses = []

        for x_scaled, _ in tr_loader:
            x_scaled = x_scaled.to(device)
            rep, recon = model(x_scaled)
            z_all_h = model.project_all_h(rep)
            dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)
            logits = -dist_sq_all / args.temperature
            p_assign = torch.softmax(logits, dim=1)

            rec = recon_mse_loss(recon, x_scaled)
            if args.objective == "soft-boundary":
                R_expand = R.unsqueeze(0)
                svdd_all = R_expand**2 + (1.0 / args.nu) * torch.relu(dist_sq_all - R_expand**2)
            else:
                svdd_all = dist_sq_all
            svdd = torch.mean(torch.sum(p_assign * svdd_all, dim=1))

            top2 = torch.topk(dist_sq_all, k=2, largest=False)
            d1 = top2.values[:, 0]
            d2 = top2.values[:, 1]
            sep = torch.mean(torch.relu(d1 - d2 + args.sep_margin))

            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            eps = 1e-8
            entropy = -torch.sum(p_assign * torch.log(p_assign + eps), dim=1)
            entropy_loss = -torch.mean(entropy)

            _, intra_s, inter_s = silhouette_surrogate_loss(rep, p_assign)
            sil_core = intra_s - args.lambda_sil_inter * inter_s

            loss = (
                args.lambda_rec * rec
                + args.lambda_svdd * svdd
                + args.lambda_sep * sep
                + args.lambda_overlap * ov
                + args.lambda_entropy * entropy_loss
                + args.lambda_sil * sil_core
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            rec_losses.append(float(rec.item()))
            svdd_losses.append(float(svdd.item()))
            sep_losses.append(float(sep.item()))
            ov_losses.append(float(ov.item()))
            ent_losses.append(float(entropy_loss.item()))
            sil_losses.append(float(sil_core.item()))

        if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
            with torch.no_grad():
                all_dist = []
                all_z_sel = []
                for x_scaled, _ in tr_loader_eval:
                    x_scaled = x_scaled.to(device)
                    rep, _ = model(x_scaled)
                    z_all_h = model.project_all_h(rep)
                    dist_sq = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)
                    assign_b = torch.argmin(dist_sq, dim=1)
                    z_sel_b = z_all_h[torch.arange(z_all_h.size(0), device=device), assign_b]
                    all_dist.append(dist_sq.cpu())
                    all_z_sel.append(z_sel_b.cpu())
                all_dist = torch.cat(all_dist, dim=0)
                all_z_sel = torch.cat(all_z_sel, dim=0)
                assign = torch.argmin(all_dist, dim=1)
                new_R = torch.zeros_like(R)
                for k in range(R.numel()):
                    d_k = all_dist[assign == k, k]
                    if d_k.numel() > 0:
                        new_R[k] = torch.quantile(d_k.to(device), 1.0 - args.nu)
                    else:
                        new_R[k] = 0.0
                ema_R = args.radius_ema * R + (1.0 - args.radius_ema) * new_R
                step = torch.clamp(ema_R - R, min=-args.radius_max_step, max=args.radius_max_step)
                R = torch.clamp(R + step, min=0.0)
                if args.enable_pruning and ep % max(1, args.prune_every) == 0:
                    c_h, R, pinfo = sort_clusters_flagged(c_h, R, all_dist, all_z_sel, assign, args, args.curvature)
                    print(
                        f"[PRUNE] epoch={ep:03d} n={pinfo['n']} under={pinfo['under']} over={pinfo['over']} "
                        f"populated={pinfo['populated']} reseeded={pinfo['reseeded']} merged={pinfo['merged']}"
                    )

        print(
            f"[UNSUP-SIL] {ep:03d}/{args.svdd_n_epochs} loss={np.mean(losses):.6f} rec={np.mean(rec_losses):.6f} "
            f"svdd={np.mean(svdd_losses):.6f} sep={np.mean(sep_losses):.6f} ov={np.mean(ov_losses):.6f} "
            f"ent={np.mean(ent_losses):.6f} sil_core={np.mean(sil_losses):.6f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_digit, unsup, _, _ = evaluate(model, c_h, R, te_loader, device, args.objective, args.curvature)
            sil_tr = silhouette_sklearn_on_loader(
                model, c_h, tr_loader_eval, device, args.curvature, args.sil_eval_max_samples, args.sil_eval_feature
            )
            sil_str = f"{sil_tr:.4f}" if sil_tr is not None else "nan"
            print(
                f"[EVAL] epoch={ep:03d} macro_auc={macro} sklearn_sil_train≈{sil_str} "
                f"h_acc={unsup['hungarian_acc']} nmi={unsup['nmi']} ari={unsup['ari']}"
            )

            if args.select_metric == "silhouette":
                metric = float(sil_tr) if sil_tr is not None else -1e12
            elif args.select_metric == "hungarian":
                metric = float(unsup["hungarian_acc"]) if unsup["hungarian_acc"] is not None else -1e12
            else:
                metric = float(macro) if not np.isnan(macro) else -1e12

            improved = metric > (best_select_metric + float(args.early_stop_min_delta))
            if improved:
                best_select_metric = metric
                best_macro = macro
                best_epoch = ep
                best_per_digit = per_digit
                best_unsup = unsup
                best_sil = sil_tr
                no_improve_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "c_h": c_h.detach().cpu(),
                        "R": R.detach().cpu(),
                        "rep_dim": args.rep_dim,
                        "z_dim": args.z_dim,
                        "curvature": args.curvature,
                        "center_init": args.center_init,
                        "preprocess": args.preprocess,
                        "lambda_sil": args.lambda_sil,
                        "lambda_sil_inter": args.lambda_sil_inter,
                        "lambda_overlap": args.lambda_overlap,
                        "eval_sklearn_silhouette_train": sil_tr,
                        "eval_hungarian_acc": unsup["hungarian_acc"],
                        "eval_nmi": unsup["nmi"],
                        "eval_ari": unsup["ari"],
                    },
                    os.path.join(args.xp_path, "checkpoint_best.pth"),
                )
            else:
                no_improve_epochs += 1
                if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                    print(f"[EARLY-STOP] epoch={ep:03d} best_epoch={best_epoch:03d} best_metric={best_select_metric}")
                    break

        torch.save(
            {
                "model_state": model.state_dict(),
                "c_h": c_h.detach().cpu(),
                "R": R.detach().cpu(),
                "rep_dim": args.rep_dim,
                "z_dim": args.z_dim,
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    if args.plot_embeddings:
        z_emb, assign = collect_embeddings(model, tr_loader_eval, c_h, device, args.curvature)
        plot_out = args.plot_out or os.path.join(args.xp_path, "poincare_clusters.png")
        plot_poincare(z_emb, c_h, R, out_path=plot_out, assign=assign)
        print(f"[PLOT] saved: {plot_out}")

    out = {
        "best_epoch": int(best_epoch),
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "per_digit_best_auc": best_per_digit,
        "best_sklearn_silhouette_train": best_sil,
        "best_hungarian_acc": best_unsup.get("hungarian_acc"),
        "best_nmi": best_unsup.get("nmi"),
        "best_ari": best_unsup.get("ari"),
        "select_metric": args.select_metric,
        "center_init": args.center_init,
        "preprocess": args.preprocess,
        "lambda_sil": args.lambda_sil,
        "lambda_sil_inter": args.lambda_sil_inter,
        "lambda_overlap": args.lambda_overlap,
        "margin_overlap": args.margin_overlap,
        "enable_pruning": args.enable_pruning,
        "seed": args.seed,
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
