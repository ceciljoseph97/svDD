"""Shared Poincaré / t-SNE plots for Iris hyperbolic SVDD trainers."""
from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.decomposition import PCA


def to_2d(z_np: np.ndarray) -> np.ndarray:
    if z_np.shape[1] > 2:
        return PCA(n_components=2).fit_transform(z_np)
    return z_np


@torch.no_grad()
def collect_embeddings_nearest(model, loader, c_h, device, curvature: float):
    from hyperbolic_multi_sphere import dist_sq_to_all_centers

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
        ax.scatter(
            z2[:, 0], z2[:, 1], c=col, s=40, cmap="Set1", vmin=0, vmax=max(2, float(col.max())), alpha=0.8
        )
    for i in range(c2.shape[0]):
        ax.scatter(c2[i, 0], c2[i, 1], marker="x", s=120, c="black")
        r = float(R_vec[i])
        ax.add_artist(plt.Circle((c2[i, 0], c2[i, 1]), r, fill=False, linestyle="--", color="gray"))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_inline_outline_poincare(z_per_sample, c_h, R, assign, y_true, out_path: str, color_by_true: bool):
    import matplotlib.pyplot as plt

    z_np = z_per_sample.numpy() if torch.is_tensor(z_per_sample) else z_per_sample
    c_np = c_h.detach().cpu().numpy()
    R_vec = R.detach().cpu().numpy()
    assign_np = assign.numpy() if torch.is_tensor(assign) else assign
    y_np = y_true.numpy() if torch.is_tensor(y_true) else y_true
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
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
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
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
