"""
Unsupervised hyperbolic multi-sphere training (MNIST_processed).

- AE + recon (DASVDD-style): reconstruction anchors the encoder (no collapse).
- No digit labels in the loss: inputs use label-free global scaling (GCN + global min-max).
- Multi-sphere: K projection heads; each point is mapped to K points in the Poincaré ball.
  Training minimizes distance to the *nearest* sphere center (union-of-balls objective),
  plus optional soft-boundary hinge on that nearest sphere (Deep multi-sphere SVDD flavor).
- Test AUC (default): one-vs-rest per digit k with **union** anomaly score S = min_j(d_h^2−R_j^2) (inside any sphere ⇒ S≤0). Legacy: `--auc_mode per_sphere` uses column k only.

Optional Stage-1 weights: hySpCheck/runs/ae_pretrain_mnist/ae_stage1.pth

After training, saves t-SNE of nearest-sphere hyperbolic embeddings to tsne_unsup_nearest_z.png (disable with --skip_tsne).

See train_hyp_mnist_unsup_v2.py for post-hoc sphere pruning + union test metrics + t-SNE with active mask.

`--export_cluster_samples N`: writes `<xp_path>/cluster_exports/` with `cluster_k_samples.png`, `cluster_samples_overview.png`, `cluster_export_meta.json` (raw MNIST, nearest-sphere assignment).

`--export_hotspot_analysis`: writes `<xp_path>/hotspot_analysis/` — P(d|k), P(k|d) heatmaps, bar charts, `hotspot_class_density.json`.

`--export_cluster_neural_hotspots`: writes `<xp_path>/cluster_neural_hotspots/` — per cluster: mean **input saliency** |∂(union margin)/∂x| with union margin = min_k(d_h²−R_k²), plus mean **encoder conv2** map, same split as `--cluster_export_split`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
import torch.optim as optim
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

# hySpCheck holds shared backbones and hyperbolic modules
_HYSP = Path(__file__).resolve().parent.parent / "hySpCheck"
sys.path.insert(0, str(_HYSP))

from hyperbolic_multi_sphere import (  # noqa: E402
    HyperbolicMultiSphereSVDD,
    dist_sq_to_all_centers,
    svdd_loss_one_class,
    svdd_loss_soft_boundary,
    update_radii_unsupervised,
)
from hyperbolic_ops import hyp_distance, proj_ball  # noqa: E402
from mnist_local import (  # noqa: E402
    MIN_MAX,
    MNISTDigitsProcessedRawDataset,
    MNIST_LeNet_SVDDIAE,
    global_contrast_normalization,
    recon_mse_loss,
)


def preprocess_batch_global_minmax(x_raw: torch.Tensor) -> torch.Tensor:
    """Label-free scaling: GCN then map to [0, 1] with global extrema over digit ranges."""
    x_gcn = global_contrast_normalization(x_raw.clone(), scale="l1")
    vmin = min(t[0] for t in MIN_MAX)
    vmax = max(t[1] for t in MIN_MAX)
    return (x_gcn - vmin) / (vmax - vmin + 1e-12)


class UnsupervisedScaledDataset(Dataset):
    def __init__(self, base_ds, return_raw: bool = False):
        self.base = base_ds
        self.return_raw = return_raw

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_raw, d, _ = self.base[idx]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = preprocess_batch_global_minmax(x_raw.unsqueeze(0)).squeeze(0)
        if self.return_raw:
            return x_scaled, d_t, x_raw.squeeze(0).clone()
        return x_scaled, d_t


@torch.no_grad()
def init_centers_h_unsupervised(model: HyperbolicMultiSphereSVDD, train_loader, device: torch.device, eps: float = 1e-5):
    """Fréchet-style init: per sphere k, mean of z_h^k over all training points (no labels)."""
    c = torch.zeros((model.n_digits, model.z_dim), device=device)
    n = 0
    for x_scaled, _ in train_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        c += z_all.sum(dim=0)
        n += z_all.size(0)
    c = c / float(max(n, 1))
    c = proj_ball(c, c=model.curvature, eps=eps)
    return c


@torch.no_grad()
def evaluate(
    model,
    c_h,
    R,
    loader,
    device,
    _objective,
    curvature: float,
    active_cluster_mask: np.ndarray | None = None,
    auc_mode: str = "union",
):
    """
    Test-split AUC (one-vs-rest per digit k).

    auc_mode:
      - union (default): anomaly score is the same for every k: S = min_j (d_h^2 - R_j^2) over *active* spheres
        (inactive j ignored). Inside any active sphere => S <= 0; outside all => S > 0. Higher S = more anomalous.
        y=1 iff digit != k. Aligns with "normal = inside any cluster".
      - per_sphere: legacy — score for task k is only distance to sphere k (column k).
    """
    model.eval()
    all_d = []
    all_margins = []
    all_dist_cols = []

    if active_cluster_mask is None:
        active_cluster_mask = np.ones(10, dtype=bool)

    for x_scaled, digits in loader:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq_all = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
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


def _cluster_class_summary(cluster_ids: np.ndarray, digits: np.ndarray, n_clusters: int = 10, n_classes: int = 10) -> dict:
    """Contingency + row-normalized class density per cluster (nearest-sphere assignment)."""
    counts = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for c in range(n_clusters):
        m = cluster_ids == c
        if not np.any(m):
            continue
        sub = digits[m]
        for d in range(n_classes):
            counts[c, d] = int(np.sum(sub == d))
    out: dict = {"n_samples": int(len(digits)), "per_cluster": {}}
    for c in range(n_clusters):
        n_c = int(counts[c].sum())
        dens = {str(d): (float(counts[c, d]) / n_c) if n_c > 0 else 0.0 for d in range(n_classes)}
        p = np.array([dens[str(d)] for d in range(n_classes)], dtype=np.float64)
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p + 1e-15))) if len(p) else 0.0
        out["per_cluster"][str(c)] = {
            "n": n_c,
            "class_counts": {str(d): int(counts[c, d]) for d in range(n_classes)},
            "class_density": dens,
            "entropy_nat": round(entropy, 4),
        }
    return out


def nearest_sphere_assignment(
    dist_sq: torch.Tensor,
    active_cluster_mask: np.ndarray | None,
) -> torch.Tensor:
    """
    Argmin over sphere index k of hyperbolic distance^2. If active_cluster_mask is set, only
    True columns compete (pruned / inactive masked +inf) — points are reassigned to the best active sphere.
    """
    if active_cluster_mask is None:
        return dist_sq.argmin(dim=1)
    am = np.asarray(active_cluster_mask).reshape(-1)
    if am.shape[0] != dist_sq.shape[1]:
        return dist_sq.argmin(dim=1)
    if bool(np.all(am)):
        return dist_sq.argmin(dim=1)
    t_am = torch.as_tensor(am, device=dist_sq.device, dtype=torch.bool)
    if not bool(torch.any(t_am)):
        return dist_sq.argmin(dim=1)
    masked = dist_sq.masked_fill(~t_am.unsqueeze(0), float("inf"))
    k = masked.argmin(dim=1)
    bad = torch.isinf(masked.min(dim=1).values)
    if bool(bad.any()):
        k_g = dist_sq.argmin(dim=1)
        k = torch.where(bad, k_g, k)
    return k


def _draw_cluster_hulls(ax, Z2: np.ndarray, cluster_ids: np.ndarray, base_colors: np.ndarray, alpha_fill: float = 0.18):
    """Shade approximate cluster regions in t-SNE space (convex hull per cluster)."""
    _draw_cluster_hulls_masked(ax, Z2, cluster_ids, base_colors, np.ones(10, dtype=bool), alpha_fill)


def _draw_cluster_hulls_masked(
    ax, Z2: np.ndarray, cluster_ids: np.ndarray, base_colors: np.ndarray, active: np.ndarray, alpha_fill: float = 0.18
):
    """Same as _draw_cluster_hulls but skip hulls for inactive (pruned) cluster indices."""
    for k in range(10):
        if not active[k]:
            continue
        m = cluster_ids == k
        pts = Z2[m]
        if pts.shape[0] < 3:
            continue
        try:
            hull = ConvexHull(pts)
            fc = tuple(float(x) for x in base_colors[k])
            poly = Polygon(
                pts[hull.vertices], closed=True, facecolor=fc, edgecolor=fc, alpha=alpha_fill, linewidth=0.8, zorder=1
            )
            ax.add_patch(poly)
        except Exception:
            continue


@torch.no_grad()
def plot_tsne_unsupervised(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    dataset: Dataset,
    device: torch.device,
    curvature: float,
    out_png: str,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
    eval_objective: str = "soft-boundary",
    active_cluster_mask: np.ndarray | None = None,
):
    """
    t-SNE on per-sample hyperbolic embedding from the *nearest* sphere (argmin_k d(z^k, c_k)).
    If active_cluster_mask has False entries, assignment is argmin over **active** k only (pruned heads ignored).
    Cluster id = k_near (unsupervised). Saves dual panel: (1) clusters + shaded hulls, (2) true digit + in/out union.
    If active_cluster_mask (length 10) is False for k, hull for k is skipped (v2 pruned clusters).
    """
    if active_cluster_mask is None:
        active_cluster_mask = np.ones(10, dtype=bool)
    model.eval()
    c_h = c_h.to(device)
    R = R.to(device)
    n = min(max_samples, len(dataset))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(dataset), size=n, replace=False).tolist()
    dl = DataLoader(Subset(dataset, idx), batch_size=256, shuffle=False, num_workers=0)

    zs = []
    digits_list = []
    inside_list = []
    cluster_list = []

    for x_scaled, digits in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        k_near = nearest_sphere_assignment(dist_sq, active_cluster_mask)
        b_idx = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b_idx, k_near]
        zs.append(z_pick.cpu())
        digits_list.append(digits.cpu())
        cluster_list.append(k_near.cpu())
        use_union_inside = (eval_objective == "soft-boundary") or (not np.all(active_cluster_mask))
        if use_union_inside:
            margin = dist_sq - (R.unsqueeze(0) ** 2)
            am = torch.as_tensor(active_cluster_mask, device=dist_sq.device, dtype=torch.bool)
            margin = margin.masked_fill(~am.unsqueeze(0), float("inf"))
            inside = margin.min(dim=1).values <= 0
        else:
            dist_near = dist_sq[b_idx, k_near]
            R_near = R[k_near]
            inside = dist_near <= (R_near**2)
        inside_list.append(inside.cpu())

    Z = torch.cat(zs, dim=0).numpy()
    D = torch.cat(digits_list, dim=0).numpy()
    INS = torch.cat(inside_list, dim=0).numpy()
    Kassign = torch.cat(cluster_list, dim=0).numpy()

    n_eff = Z.shape[0]
    perplexity = float(min(perplexity, max(5.0, 0.99 * max(n_eff - 1, 1))))
    # sklearn >= 1.5 requires max_iter >= 250
    n_iter_eff = max(250, int(n_iter))

    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, n_iter=n_iter_eff)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, max_iter=n_iter_eff)
    Z2 = tsne.fit_transform(Z)

    summary = _cluster_class_summary(Kassign, D)
    pruned_ids = [int(k) for k in range(10) if not active_cluster_mask[k]]
    summary["pruning"] = {
        "active_cluster_mask": [bool(active_cluster_mask[k]) for k in range(10)],
        "pruned_cluster_ids": pruned_ids,
        "n_active_spheres": int(np.sum(active_cluster_mask)),
        "nearest_sphere_assignment": "active_argmin"
        if (not bool(np.all(active_cluster_mask)))
        else "all_heads_argmin",
        "note": "per_cluster uses nearest active sphere (pruned heads excluded from argmin). Hull/legend grey-out uses active_cluster_mask.",
    }
    out_abs = os.path.abspath(out_png)
    _par = os.path.dirname(out_abs)
    if _par:
        os.makedirs(_par, exist_ok=True)
    summary_path = os.path.splitext(out_abs)[0] + "_cluster_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Compact table to stdout
    print("[t-SNE] cluster x class density (rows=nearest-sphere cluster, cols=digit)")
    hdr = "clu\\d " + " ".join(f"{d:>4}" for d in range(10))
    print(hdr)
    for c in range(10):
        row = summary["per_cluster"][str(c)]
        n_c = row["n"]
        if n_c == 0:
            print(f"{c:>4}   (empty)")
            continue
        parts = [f"{row['class_density'][str(d)] * 100.0:4.1f}" for d in range(10)]
        print(f"{c:>4}  " + " ".join(parts) + f"   n={n_c}  H={row['entropy_nat']}")

    try:
        cmap10 = plt.colormaps["tab10"]
    except AttributeError:
        cmap10 = plt.cm.get_cmap("tab10")
    cluster_colors = np.array([cmap10(i / 10.0)[:3] for i in range(10)])
    digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 7))

    _draw_cluster_hulls_masked(ax0, Z2, Kassign, cluster_colors, active_cluster_mask, alpha_fill=0.22)
    for k in range(10):
        m = Kassign == k
        if not np.any(m):
            continue
        col = tuple(cluster_colors[k]) if active_cluster_mask[k] else (0.55, 0.55, 0.55)
        ax0.scatter(
            Z2[m, 0],
            Z2[m, 1],
            s=14,
            color=col,
            marker="o",
            alpha=0.9 if active_cluster_mask[k] else 0.45,
            edgecolors="k",
            linewidths=0.2,
            zorder=3,
        )
    leg_c = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=cluster_colors[j],
            linestyle="None",
            markersize=7,
            label=f"cluster {j}" + ("" if active_cluster_mask[j] else " (pruned)"),
        )
        for j in range(10)
    ]
    ax0.legend(handles=leg_c, loc="best", fontsize=7, ncol=2, frameon=True)
    ax0.set_title("t-SNE: nearest-sphere cluster (shaded hulls)")
    ax0.set_xlabel("t-SNE-1")
    ax0.set_ylabel("t-SNE-2")

    for k in range(10):
        m = D == k
        if not np.any(m):
            continue
        m_in = m & INS
        m_out = m & (~INS)
        if np.any(m_in):
            ax1.scatter(Z2[m_in, 0], Z2[m_in, 1], s=12, marker="o", color=digit_colors[k], alpha=0.85)
        if np.any(m_out):
            ax1.scatter(Z2[m_out, 0], Z2[m_out, 1], s=12, marker="x", color=digit_colors[k], alpha=0.85)
    leg_digits = [Line2D([0], [0], marker="o", color=digit_colors[j], linestyle="None", markersize=6, label=f"{j}") for j in range(10)]
    leg_state = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="in union"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="outside"),
    ]
    leg1 = ax1.legend(handles=leg_state, loc="lower right", fontsize=9, frameon=True)
    leg2 = ax1.legend(handles=leg_digits, loc="upper left", fontsize=8, frameon=True, ncol=2)
    ax1.add_artist(leg1)
    ax1.set_title("Same embedding, true digit (eval) + union in/out")
    ax1.set_xlabel("t-SNE-1")
    ax1.set_ylabel("t-SNE-2")
    ax0.set_aspect("equal", adjustable="datalim")
    ax1.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[t-SNE] saved {out_png}")
    print(f"[t-SNE] cluster summary {summary_path}")


@torch.no_grad()
def export_cluster_sample_images(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    dataset: Dataset,
    device: torch.device,
    curvature: float,
    out_dir: str,
    samples_per_cluster: int,
    batch_size: int,
    seed: int,
    split_name: str,
    active_cluster_mask: np.ndarray | None = None,
) -> None:
    """Save raw MNIST thumbnails grouped by nearest-sphere cluster. If active_cluster_mask set, argmin over active spheres only."""
    from collections import defaultdict

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    c_h = c_h.to(device)
    R = R.to(device)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    buckets: dict[int, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for batch in dl:
        if len(batch) != 3:
            raise RuntimeError("export_cluster_sample_images requires UnsupervisedScaledDataset(return_raw=True)")
        x_scaled, digits, x_raw = batch
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        k_near = nearest_sphere_assignment(dist_sq, active_cluster_mask).cpu().numpy()
        digits_np = digits.numpy()
        x_raw_np = x_raw.cpu().numpy()
        for i in range(x_scaled.size(0)):
            buckets[int(k_near[i])].append((np.asarray(x_raw_np[i]), int(digits_np[i])))

    rng = np.random.RandomState(seed)
    picked_per_k: dict[int, list[tuple[np.ndarray, int]]] = {}
    for k in range(10):
        items = buckets[k]
        if len(items) > samples_per_cluster:
            idx = rng.choice(len(items), samples_per_cluster, replace=False)
            picked_per_k[k] = [items[i] for i in idx]
        else:
            picked_per_k[k] = list(items)

    meta: dict = {
        "split": split_name,
        "samples_per_cluster": samples_per_cluster,
        "nearest_sphere_assignment": "active_argmin"
        if (active_cluster_mask is not None and not bool(np.all(active_cluster_mask)))
        else "all_heads_argmin",
        "active_cluster_mask": None
        if active_cluster_mask is None
        else [bool(x) for x in np.asarray(active_cluster_mask).reshape(-1).tolist()],
        "clusters": {},
    }
    for k in range(10):
        picked = picked_per_k[k]
        dc = {str(d): 0 for d in range(10)}
        for _, d in picked:
            dc[str(d)] += 1
        meta["clusters"][str(k)] = {
            "digit_counts_exported": dc,
            "total_available_in_split": len(buckets[k]),
        }
    with open(os.path.join(out_dir, "cluster_export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for k in range(10):
        picked = picked_per_k[k]
        if not picked:
            continue
        n = len(picked)
        fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.6))
        if n == 1:
            axes = [axes]
        for j, (img, dg) in enumerate(picked):
            im = img.squeeze()
            axes[j].imshow(im, cmap="gray", vmin=0, vmax=1)
            axes[j].set_title(str(dg), fontsize=10)
            axes[j].axis("off")
        dc = meta["clusters"][str(k)]["digit_counts_exported"]
        fig.suptitle(f"cluster {k} (nearest sphere) | digit counts: {dc}", fontsize=10)
        plt.tight_layout()
        out_p = os.path.join(out_dir, f"cluster_{k}_samples.png")
        plt.savefig(out_p, dpi=140)
        plt.close()
        print(f"[export] {out_p}")

    fig, axes = plt.subplots(10, samples_per_cluster, figsize=(1.9 * samples_per_cluster, 2.05 * 10))
    for k in range(10):
        picked = picked_per_k[k]
        for j in range(samples_per_cluster):
            ax = axes[k, j]
            if j < len(picked):
                img, dg = picked[j]
                ax.imshow(np.asarray(img).squeeze(), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"{dg}", fontsize=7)
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
        axes[k, 0].set_ylabel(f"k={k}", fontsize=9)
    plt.suptitle(
        f"Nearest-sphere clusters ({split_name}) — digit labels for inspection only",
        fontsize=11,
        y=1.002,
    )
    plt.tight_layout()
    overview = os.path.join(out_dir, "cluster_samples_overview.png")
    plt.savefig(overview, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[export] {overview}")
    print(f"[export] meta {os.path.join(out_dir, 'cluster_export_meta.json')}")


@torch.no_grad()
def export_hotspot_class_density(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    dataset: Dataset,
    device: torch.device,
    curvature: float,
    out_dir: str,
    batch_size: int,
    split_name: str,
    active_cluster_mask: np.ndarray | None = None,
) -> None:
    """
    Full split: count(cluster=k, digit=d) for nearest-sphere assignment (active-only argmin if mask set).
    Saves empirical P(d|k), P(k|d), heatmaps, and JSON (entropy, argmax hotspots per row/col).
    Dataset: UnsupervisedScaledDataset (return_raw=False).
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    c_h = c_h.to(device)
    R = R.to(device)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    counts = np.zeros((10, 10), dtype=np.int64)

    for x_scaled, digits in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        k_near = nearest_sphere_assignment(dist_sq, active_cluster_mask).cpu().numpy()
        d_np = digits.cpu().numpy()
        for i in range(len(k_near)):
            counts[int(k_near[i]), int(d_np[i])] += 1

    row = counts.sum(axis=1, keepdims=True).clip(min=1)
    col = counts.sum(axis=0, keepdims=True).clip(min=1)
    p_d_given_k = counts / row
    p_k_given_d = counts / col

    def _entropy_row(p: np.ndarray) -> float:
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-15)))

    analysis: dict = {
        "split": split_name,
        "nearest_sphere_assignment": "active_argmin"
        if (active_cluster_mask is not None and not bool(np.all(active_cluster_mask)))
        else "all_heads_argmin",
        "active_cluster_mask": None
        if active_cluster_mask is None
        else [bool(x) for x in np.asarray(active_cluster_mask).reshape(-1).tolist()],
        "total_samples": int(counts.sum()),
        "counts_cluster_x_digit": counts.tolist(),
        "per_cluster": {},
        "per_digit": {},
    }
    for k in range(10):
        row_d = p_d_given_k[k]
        tot_k = int(counts[k].sum())
        dom = int(np.argmax(row_d))
        top3 = sorted([(d, float(row_d[d])) for d in range(10)], key=lambda x: -x[1])[:3]
        analysis["per_cluster"][str(k)] = {
            "n": tot_k,
            "entropy_digit_mix_nat": round(_entropy_row(row_d), 4),
            "hotspot_digit_argmax_P_d_given_k": dom,
            "top3_digits_by_density": {str(a[0]): round(a[1], 5) for a in top3},
        }
    for d in range(10):
        col_k = p_k_given_d[:, d]
        tot_d = int(counts[:, d].sum())
        dom = int(np.argmax(col_k))
        top3 = sorted([(k, float(col_k[k])) for k in range(10)], key=lambda x: -x[1])[:3]
        analysis["per_digit"][str(d)] = {
            "n": tot_d,
            "entropy_cluster_mix_nat": round(_entropy_row(col_k), 4),
            "hotspot_cluster_argmax_P_k_given_d": dom,
            "top3_clusters_by_density": {str(a[0]): round(a[1], 5) for a in top3},
        }

    json_path = os.path.join(out_dir, "hotspot_class_density.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    def _heatmap(data: np.ndarray, xlabel: str, ylabel: str, title: str, fname: str, fmt_cell: str = ".2f"):
        fig, ax = plt.subplots(figsize=(9, 7.5))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(float(data.max()), 0.05))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i in range(10):
            for j in range(10):
                v = data[i, j]
                if v >= 0.005:
                    ax.text(j, i, format(v, fmt_cell), ha="center", va="center", color="black", fontsize=7)
        plt.tight_layout()
        p = os.path.join(out_dir, fname)
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[hotspot] {p}")

    _heatmap(
        p_d_given_k,
        "true digit class",
        "nearest-sphere cluster k",
        f"P(digit | cluster) empirical density — {split_name}",
        "hotspot_P_digit_given_cluster.png",
    )
    _heatmap(
        p_k_given_d.T,
        "nearest-sphere cluster k",
        "true digit class",
        f"P(cluster | digit) — where each class lands — {split_name}",
        "hotspot_P_cluster_given_digit.png",
    )

    fig, axes = plt.subplots(2, 5, figsize=(16, 6))
    axes = axes.flatten()
    for k in range(10):
        ax = axes[k]
        ax.bar(range(10), p_d_given_k[k], color=plt.cm.tab10(np.linspace(0, 1, 10)))
        ax.set_xlabel("digit")
        ax.set_title(f"cluster k={k}  n={int(counts[k].sum())}")
        ax.set_ylim(0, min(1.05, float(p_d_given_k[k].max()) * 1.25 + 0.05))
    plt.suptitle(f"Class density inside each nearest-sphere cluster ({split_name})", fontsize=12)
    plt.tight_layout()
    barp = os.path.join(out_dir, "hotspot_cluster_digit_bars.png")
    plt.savefig(barp, dpi=140)
    plt.close()
    print(f"[hotspot] {barp}")
    print(f"[hotspot] {json_path}")


def _encoder_conv2_mean_activation_map(model: HyperbolicMultiSphereSVDD, x_scaled: torch.Tensor) -> torch.Tensor:
    """Mean |post-ReLU conv2| over channels, upsampled to input H×W. No grad."""
    bb = model.backbone
    x = bb.conv1(x_scaled)
    x = bb.pool(F.leaky_relu(bb.bn1(x)))
    x = bb.conv2(x)
    x = F.leaky_relu(bb.bn2(x))
    g = x.abs().mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(x_scaled.shape[-2], x_scaled.shape[-1]), mode="bilinear", align_corners=False)
    return g.squeeze(1)


def _saliency_union_margin(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    curvature: float,
    x_scaled_1b: torch.Tensor,
    active_cluster_mask: np.ndarray | None = None,
) -> torch.Tensor:
    """|∂(min over active k of (d_h²−R_k²))/∂x|; subgradient at argmin sphere (matches evaluate())."""
    x = x_scaled_1b.clone().detach().requires_grad_(True)
    rep, _ = model(x)
    z_all = model.project_all_h(rep)
    dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
    margins = dist_sq - (R**2).unsqueeze(0)
    if active_cluster_mask is not None:
        am = torch.as_tensor(active_cluster_mask, device=margins.device, dtype=torch.bool)
        margins = margins.masked_fill(~am.unsqueeze(0), float("inf"))
    union_m = margins.min(dim=1).values
    scalar = union_m.sum()
    grad_x = torch.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
    return grad_x.abs().mean(dim=1).squeeze(0).detach()


@torch.no_grad()
def _indices_by_nearest_cluster(
    model: HyperbolicMultiSphereSVDD,
    dataset: Dataset,
    device: torch.device,
    c_h: torch.Tensor,
    curvature: float,
    batch_size: int,
    active_cluster_mask: np.ndarray | None = None,
) -> list[list[int]]:
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    by_k: list[list[int]] = [[] for _ in range(10)]
    offset = 0
    c_h = c_h.to(device)
    model.eval()
    for x_scaled, _ in dl:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = dist_sq_to_all_centers(z_all, c_h, curvature=curvature)
        kn = nearest_sphere_assignment(dist_sq, active_cluster_mask).cpu().numpy()
        for i in range(x_scaled.size(0)):
            by_k[int(kn[i])].append(offset + i)
        offset += x_scaled.size(0)
    return by_k


def export_cluster_neural_hotspots(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    dataset: Dataset,
    device: torch.device,
    curvature: float,
    out_dir: str,
    batch_size: int,
    split_name: str,
    max_samples_per_cluster: int,
    seed: int,
    active_cluster_mask: np.ndarray | None = None,
) -> None:
    """
    Per nearest-sphere cluster k: mean saliency w.r.t. union margin min_j(d_h²−R_j²) (active spheres only if mask set), and mean encoder conv2 activation map.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    c_h = c_h.to(device)
    R = R.to(device)
    rng = np.random.RandomState(seed)

    by_k = _indices_by_nearest_cluster(
        model, dataset, device, c_h, curvature, batch_size, active_cluster_mask
    )
    picked: dict[int, list[int]] = {}
    for k in range(10):
        idxs = by_k[k]
        if len(idxs) > max_samples_per_cluster:
            sub = rng.choice(len(idxs), max_samples_per_cluster, replace=False)
            picked[k] = [idxs[i] for i in sub]
        else:
            picked[k] = list(idxs)

    meta: dict = {
        "split": split_name,
        "max_samples_per_cluster": max_samples_per_cluster,
        "saliency_target": "min_k (d_h^2(z_k,c_k) - R_k^2) over active k w.r.t. scaled input x (union SVDD margin; same as evaluate auc_mode=union)",
        "active_cluster_mask": None
        if active_cluster_mask is None
        else [bool(x) for x in np.asarray(active_cluster_mask).reshape(-1).tolist()],
        "activation": "mean |LeakyReLU(BN(conv2(x)))| over channels, bilinear upsample to input size",
        "nearest_sphere_assignment": "active_argmin"
        if (active_cluster_mask is not None and not bool(np.all(active_cluster_mask)))
        else "all_heads_argmin",
        "clusters": {},
    }

    def _norm01(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-12)

    cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}

    for k in range(10):
        sal_stack = []
        act_stack = []
        for idx in picked[k]:
            x_scaled, _ = dataset[idx]
            x1 = x_scaled.unsqueeze(0).to(device)
            with torch.no_grad():
                act_stack.append(_encoder_conv2_mean_activation_map(model, x1).squeeze(0).cpu().numpy())
            with torch.enable_grad():
                sal_stack.append(
                    _saliency_union_margin(model, c_h, R, curvature, x1, active_cluster_mask).cpu().numpy()
                )

        meta["clusters"][str(k)] = {
            "n_used": len(picked[k]),
            "n_available_nearest_sphere": len(by_k[k]),
        }

        if not sal_stack:
            continue

        sal_mean = np.mean(np.stack(sal_stack, axis=0), axis=0)
        act_mean = np.mean(np.stack(act_stack, axis=0), axis=0)
        sal_n = _norm01(sal_mean)
        act_n = _norm01(act_mean)
        cache[k] = (sal_n, act_n, len(sal_stack))

        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
        im0 = axes[0].imshow(sal_n, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title(f"mean saliency |∂union_margin/∂x| (n={len(sal_stack)})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        axes[0].axis("off")
        im1 = axes[1].imshow(act_n, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("mean encoder conv2 |act| (upsampled)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        axes[1].axis("off")
        fig.suptitle(f"cluster k={k} (nearest hyperbolic sphere) — {split_name}", fontsize=11)
        plt.tight_layout()
        outp = os.path.join(out_dir, f"cluster_{k}_neural_hotspot.png")
        plt.savefig(outp, dpi=150)
        plt.close()
        print(f"[neural_hotspot] {outp}")

    fig, axes = plt.subplots(10, 2, figsize=(5.2, 22))
    for k in range(10):
        if k not in cache:
            axes[k, 0].text(0.5, 0.5, "empty", ha="center", va="center", transform=axes[k, 0].transAxes)
            axes[k, 1].text(0.5, 0.5, "empty", ha="center", va="center", transform=axes[k, 1].transAxes)
            for ax in (axes[k, 0], axes[k, 1]):
                ax.axis("off")
            continue
        sal_n, act_n, _n = cache[k]
        axes[k, 0].imshow(sal_n, cmap="magma", vmin=0, vmax=1)
        axes[k, 0].set_ylabel(f"k={k}", fontsize=9)
        axes[k, 0].axis("off")
        axes[k, 1].imshow(act_n, cmap="viridis", vmin=0, vmax=1)
        axes[k, 1].axis("off")
    axes[0, 0].set_title("saliency", fontsize=9)
    axes[0, 1].set_title("conv2 |act|", fontsize=9)
    plt.suptitle(f"Neural hotspots by nearest-sphere cluster ({split_name})", fontsize=12, y=1.002)
    plt.tight_layout()
    ov = os.path.join(out_dir, "cluster_neural_hotspots_overview.png")
    plt.savefig(ov, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[neural_hotspot] {ov}")

    with open(os.path.join(out_dir, "neural_hotspot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[neural_hotspot] {os.path.join(out_dir, 'neural_hotspot_meta.json')}")


def main():
    default_ae = _HYSP / "runs" / "ae_pretrain_mnist" / "ae_stage1.pth"
    p = argparse.ArgumentParser("Unsupervised hyperbolic multi-sphere (union SVDD + AE), MNIST_processed")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpUnsup/runs/mnist_unsup_hyp")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--digits", type=str, default="all")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Balanced cap: keep at most this many images per digit (train and test).",
    )
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--ae_n_epochs", type=int, default=10)
    p.add_argument("--svdd_n_epochs", type=int, default=25)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument(
        "--objective",
        type=str,
        default="union-soft",
        choices=["union-one", "union-soft"],
        help="union-one: mean min_k d_sq (DASVDD + union). union-soft: soft-boundary on nearest sphere only.",
    )
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=5e-5)
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--skip_ae_pretrain", action="store_true", help="Load Stage-1 AE checkpoint instead of training AE.")
    p.add_argument(
        "--ae_stage1_checkpoint_path",
        type=str,
        default=str(default_ae),
        help="Checkpoint with model_state (full HyperbolicMultiSphereSVDD). Default: hySpCheck AE pretrain.",
    )
    p.add_argument("--save_ae_stage1_checkpoint_path", type=str, default=None)
    p.add_argument("--skip_tsne", action="store_true", help="Do not save t-SNE PNG after training.")
    p.add_argument("--tsne_split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--tsne_max_samples", type=int, default=2000)
    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--tsne_seed", type=int, default=42)
    p.add_argument(
        "--tsne_out",
        type=str,
        default=None,
        help="t-SNE output path (default: <xp_path>/tsne_unsup_nearest_z.png).",
    )
    p.add_argument(
        "--auc_mode",
        type=str,
        default="union",
        choices=["union", "per_sphere"],
        help="union: one score min_k(d^2-R^2) over spheres (normal=inside any). per_sphere: legacy column-k margin only.",
    )
    p.add_argument(
        "--export_cluster_samples",
        type=int,
        default=0,
        help="If >0, save raw MNIST thumbnails per nearest-sphere cluster (up to N per cluster).",
    )
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
        help="Per-cluster mean saliency |∂union_margin/∂x| (min_k d_h²−R_k²) + encoder conv2 maps.",
    )
    p.add_argument(
        "--neural_hotspot_max_samples",
        type=int,
        default=48,
        help="Max images per cluster to average for neural hotspot maps (subsample of nearest-sphere assignment).",
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
    best_macro = -1e9
    best_epoch = -1
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
            cpu_device = torch.device("cpu")
            R = update_radii_unsupervised(dist_sq_chunks, nu=args.nu, device=cpu_device).to(device)

        print(
            f"[SVDD-H-UNSUP] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={np.mean(losses):.6f} rec={np.mean(rec_losses):.6f} svdd={np.mean(svdd_losses):.6f} "
            f"overlap={np.mean(ov_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_digit = evaluate(
                model, c_h, R, te_loader, device, eval_objective, args.curvature, auc_mode=args.auc_mode
            )
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro}")
            metric = macro if not np.isnan(macro) else -1e12
            if metric > best_macro:
                best_macro = macro
                best_epoch = ep
                best_per_digit = per_digit
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
                    },
                    os.path.join(args.xp_path, "checkpoint_best.pth"),
                )
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
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_epoch),
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "per_digit_best_auc": best_per_digit,
        "curvature": args.curvature,
        "objective": args.objective,
        "unsupervised": True,
        "ae_checkpoint": args.ae_stage1_checkpoint_path if args.skip_ae_pretrain else None,
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
            out_png = args.tsne_out if args.tsne_out else os.path.join(args.xp_path, "tsne_unsup_nearest_z.png")
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
            )


if __name__ == "__main__":
    main()
