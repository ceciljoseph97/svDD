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
from sklearn.cluster import KMeans

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
from hyperbolic_ops import proj_ball  # noqa: E402
from euclidean_multi_sphere import EuclideanMultiSphereSVDD, dist_sq_to_all_centers_e, init_centers_e  # noqa: E402
from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE, recon_mse_loss  # noqa: E402

UnsupervisedScaledDataset = unsup.UnsupervisedScaledDataset
init_centers_h_unsupervised = unsup.init_centers_h_unsupervised
evaluate = unsup.evaluate
sphere_overlap_penalty = unsup.sphere_overlap_penalty
plot_tsne_unsupervised = unsup.plot_tsne_unsupervised
export_cluster_sample_images = unsup.export_cluster_sample_images
export_hotspot_class_density = unsup.export_hotspot_class_density
export_cluster_neural_hotspots = unsup.export_cluster_neural_hotspots


def _sphere_overlap_penalty_euclidean(c_h: torch.Tensor, R: torch.Tensor, margin: float) -> torch.Tensor:
    K = c_h.size(0)
    if K <= 1:
        return torch.tensor(0.0, device=c_h.device)
    penalties = []
    for a in range(K):
        for b in range(a + 1, K):
            d_ab = torch.norm(c_h[a] - c_h[b], p=2)
            penalties.append(torch.relu((R[a] + R[b] + float(margin)) - d_ab))
    return torch.mean(torch.stack(penalties)) if penalties else torch.tensor(0.0, device=c_h.device)


def _dist_sq_all_for_model(model, z_all, c_h, curvature: float):
    if isinstance(model, EuclideanMultiSphereSVDD):
        return dist_sq_to_all_centers_e(z_all, c_h)
    return dist_sq_to_all_centers(z_all, c_h, curvature=curvature)


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
    Returns active_mask length K (True = keep sphere in union scoring).
    """
    n_spheres = int(R.numel())
    counts = np.zeros(n_spheres, dtype=np.int64)
    z_parts = []
    k_parts = []

    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = _dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
        k_near = dist_sq.argmin(dim=1)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near].cpu().numpy()
        kn = k_near.cpu().numpy()
        for kk in range(n_spheres):
            counts[kk] += int(np.sum(kn == kk))
        z_parts.append(z_pick)
        k_parts.append(kn)

    Z = np.concatenate(z_parts, axis=0)
    K = np.concatenate(k_parts, axis=0)

    active = np.ones(n_spheres, dtype=bool)
    if min_cluster_members > 0:
        active &= counts >= min_cluster_members

    sil_detail: dict = {str(k): None for k in range(n_spheres)}
    if silhouette_min is not None and Z.shape[0] >= n_spheres:
        rng = np.random.RandomState(silhouette_seed)
        if Z.shape[0] > silhouette_max_samples:
            sub = rng.choice(Z.shape[0], silhouette_max_samples, replace=False)
            Zs, Ks = Z[sub], K[sub]
        else:
            Zs, Ks = Z, K
        if len(np.unique(Ks)) >= 2:
            try:
                sil = silhouette_samples(Zs, Ks, metric="euclidean")
                for k in range(n_spheres):
                    m = Ks == k
                    if np.sum(m) >= 2:
                        sil_detail[str(k)] = float(np.mean(sil[m]))
                    elif np.sum(m) == 1:
                        sil_detail[str(k)] = float(sil[m][0])
            except Exception as e:
                sil_detail["_error"] = str(e)
        for k in range(n_spheres):
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
        dist_sq = _dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
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


@torch.no_grad()
def split_overloaded_active_spheres(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    tr_loader: DataLoader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
    max_cluster_fraction: float,
    split_seed: int,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """
    Optional post-hoc split of overloaded active clusters into spare inactive spheres.
    Uses nearest-sphere assignments on train set and KMeans(2) iterative bisection.
    """
    info: dict = {"enabled": bool(max_cluster_fraction > 0), "max_cluster_fraction": float(max_cluster_fraction)}
    if max_cluster_fraction <= 0:
        info["applied"] = False
        return c_h, R, active_mask, info
    if max_cluster_fraction >= 1.0:
        info["applied"] = False
        info["note"] = "max_cluster_fraction >= 1.0 disables splitting in practice."
        return c_h, R, active_mask, info

    model.eval()
    z_parts = []
    k_parts = []
    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = _dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
        am = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
        dist_sq = dist_sq.masked_fill(~am.unsqueeze(0), float("inf"))
        k_near = dist_sq.argmin(dim=1)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near].cpu().numpy()
        z_parts.append(z_pick)
        k_parts.append(k_near.cpu().numpy())

    if not z_parts:
        info["applied"] = False
        info["note"] = "No train samples available."
        return c_h, R, active_mask, info

    Z = np.concatenate(z_parts, axis=0)
    K = np.concatenate(k_parts, axis=0)
    n_spheres = int(c_h.size(0))
    counts = np.bincount(K, minlength=n_spheres)
    n_train = int(Z.shape[0])
    cap = max(1, int(np.ceil(max_cluster_fraction * float(n_train))))

    overloaded = [k for k in range(n_spheres) if bool(active_mask[k]) and int(counts[k]) > cap]
    spare = [k for k in range(n_spheres) if not bool(active_mask[k])]
    info.update(
        {
            "applied": False,
            "n_train": n_train,
            "cap_count": cap,
            "counts_before": counts.tolist(),
            "overloaded_clusters": [int(k) for k in overloaded],
            "spare_inactive_clusters": [int(k) for k in spare],
            "splits": [],
        }
    )
    if not overloaded or not spare:
        return c_h, R, active_mask, info

    c_new = c_h.clone()
    R_new = R.clone()
    active_new = active_mask.copy()
    rng = np.random.RandomState(split_seed)

    for k in overloaded:
        idx_all = np.where(K == k)[0]
        if idx_all.size < 2:
            continue
        need_extra = int(np.ceil(float(idx_all.size) / float(cap))) - 1
        if need_extra <= 0:
            continue
        groups = [idx_all]
        for _ in range(need_extra):
            if not spare:
                break
            g_idx = int(np.argmax([len(g) for g in groups]))
            g = groups.pop(g_idx)
            if len(g) < 2:
                groups.append(g)
                break
            km = KMeans(n_clusters=2, random_state=int(rng.randint(1_000_000)), n_init=10)
            lab = km.fit_predict(Z[g])
            g0 = g[lab == 0]
            g1 = g[lab == 1]
            if len(g0) == 0 or len(g1) == 0:
                groups.append(g)
                break
            groups.append(g0)
            groups.append(g1)

        groups = sorted(groups, key=lambda x: len(x), reverse=True)
        base = groups[0]
        c_new[k] = torch.as_tensor(np.mean(Z[base], axis=0), dtype=c_new.dtype, device=c_new.device)
        R_new[k] = R_new[k] * 0.9
        used_spares = []
        for g in groups[1:]:
            if not spare:
                break
            s = int(spare.pop(0))
            c_new[s] = torch.as_tensor(np.mean(Z[g], axis=0), dtype=c_new.dtype, device=c_new.device)
            R_new[s] = R_new[k]
            active_new[s] = True
            used_spares.append(s)

        info["splits"].append(
            {
                "source_cluster": int(k),
                "source_count": int(idx_all.size),
                "groups_after_split": [int(len(g)) for g in groups],
                "new_clusters_activated": [int(s) for s in used_spares],
            }
        )

    if info["splits"]:
        info["applied"] = True
        info["active_cluster_mask_after_split"] = active_new.astype(bool).tolist()
    return c_new, R_new, active_new, info


@torch.no_grad()
def hard_cap_reassign_clusters(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    tr_loader: DataLoader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
    max_cluster_fraction: float,
    chaos_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """
    Hard-cap rebalance:
    - compute train assignments to active spheres
    - enforce per-cluster upper cap by re-centering spheres on capped subsets
    - overflow points are greedily reassigned to the nearest non-full active sphere
    chaos_factor in [0,1] adds a small bias against saturated spheres during overflow routing.
    """
    info: dict = {
        "enabled": bool(max_cluster_fraction > 0),
        "max_cluster_fraction": float(max_cluster_fraction),
        "chaos_factor": float(chaos_factor),
        "applied": False,
    }
    if max_cluster_fraction <= 0:
        return c_h, R, active_mask, info
    if max_cluster_fraction >= 1.0:
        info["note"] = "max_cluster_fraction >= 1.0 disables hard cap in practice."
        return c_h, R, active_mask, info

    model.eval()
    z_parts = []
    dist_parts = []
    assign_parts = []
    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = _dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
        am = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
        masked = dist_sq.masked_fill(~am.unsqueeze(0), float("inf"))
        k_near = masked.argmin(dim=1)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near].cpu().numpy()
        z_parts.append(z_pick)
        dist_parts.append(masked.cpu().numpy())
        assign_parts.append(k_near.cpu().numpy())

    if not z_parts:
        info["note"] = "No train samples available."
        return c_h, R, active_mask, info

    Z = np.concatenate(z_parts, axis=0)
    D = np.concatenate(dist_parts, axis=0)
    A = np.concatenate(assign_parts, axis=0)
    n_spheres = int(c_h.size(0))
    n_train = int(Z.shape[0])
    cap = max(1, int(np.ceil(max_cluster_fraction * float(n_train))))
    active_ids = [k for k in range(n_spheres) if bool(active_mask[k])]
    counts_before = np.bincount(A, minlength=n_spheres)
    if not active_ids:
        info["note"] = "No active spheres."
        return c_h, R, active_mask, info

    # Keep best-fit points in each sphere up to cap, overflow gets rerouted.
    keep_mask = np.zeros(n_train, dtype=bool)
    overflow_idx: list[int] = []
    for k in active_ids:
        idx = np.where(A == k)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(D[idx, k])]
        keep = order[:cap]
        spill = order[cap:]
        keep_mask[keep] = True
        overflow_idx.extend(spill.tolist())

    counts_after = np.zeros(n_spheres, dtype=np.int64)
    counts_after[A[keep_mask]] = np.bincount(A[keep_mask], minlength=n_spheres)[A[keep_mask]]
    counts_after = np.bincount(A[keep_mask], minlength=n_spheres)

    # Reassign overflow to nearest non-full active sphere with optional saturation penalty.
    A_new = A.copy()
    for idx in overflow_idx:
        candidates = []
        for k in active_ids:
            if counts_after[k] >= cap:
                continue
            sat = float(counts_after[k]) / float(cap)
            penalty = chaos_factor * sat * max(1e-6, float(np.nanmean(D[idx, active_ids][np.isfinite(D[idx, active_ids])])) )
            candidates.append((float(D[idx, k]) + penalty, k))
        if not candidates:
            # No room anywhere: keep original assignment.
            counts_after[A_new[idx]] += 1
            continue
        _, best_k = min(candidates, key=lambda x: x[0])
        A_new[idx] = int(best_k)
        counts_after[best_k] += 1

    c_new = c_h.clone()
    R_new = R.clone()
    for k in active_ids:
        idx = np.where(A_new == k)[0]
        if idx.size == 0:
            continue
        c_new[k] = torch.as_tensor(np.mean(Z[idx], axis=0), dtype=c_new.dtype, device=c_new.device)
        if counts_before[k] > cap:
            R_new[k] = R_new[k] * max(0.7, 1.0 - 0.3 * min(1.0, chaos_factor + 0.2))

    info.update(
        {
            "applied": bool(len(overflow_idx) > 0),
            "n_train": n_train,
            "cap_count": cap,
            "counts_before": counts_before.tolist(),
            "counts_after": np.bincount(A_new, minlength=n_spheres).tolist(),
            "overflow_reassigned": int(len(overflow_idx)),
        }
    )
    return c_new, R_new, active_mask.copy(), info


def hybrid_lb_ub_rebalance(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    tr_loader: DataLoader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
    lb_min_cluster_members: int,
    ub_max_cluster_fraction: float,
    split_seed: int,
    max_iters: int,
    hard_cap_enabled: bool,
    chaos_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """Iterative hybrid rebalance: LB pruning + UB subdivision."""
    info: dict = {
        "enabled": True,
        "lb_min_cluster_members": int(lb_min_cluster_members),
        "ub_max_cluster_fraction": float(ub_max_cluster_fraction),
        "max_iters": int(max_iters),
        "hard_cap_enabled": bool(hard_cap_enabled),
        "chaos_factor": float(chaos_factor),
        "iterations": [],
        "applied": False,
    }
    if max_iters < 1:
        info["enabled"] = False
        info["note"] = "max_iters < 1"
        return c_h, R, active_mask, info

    c_cur = c_h
    R_cur = R
    active_cur = active_mask.copy()
    any_change = False
    for it in range(1, max_iters + 1):
        iter_info: dict = {"iter": it}
        lb_info: dict = {"enabled": bool(lb_min_cluster_members > 0)}
        if lb_min_cluster_members > 0:
            active_next, lb_prune = prune_spheres_from_train(
                model,
                c_cur,
                R_cur,
                tr_loader,
                device,
                curvature,
                min_cluster_members=lb_min_cluster_members,
                silhouette_min=None,
                silhouette_max_samples=0,
                silhouette_seed=split_seed + it,
            )
            lb_info["counts"] = lb_prune.get("train_argmin_counts", [])
            lb_info["active_before"] = active_cur.astype(bool).tolist()
            lb_info["active_after"] = active_next.astype(bool).tolist()
            lb_info["changed"] = bool(not np.array_equal(active_next, active_cur))
            active_cur = active_next
        else:
            lb_info["changed"] = False
            lb_info["active_after"] = active_cur.astype(bool).tolist()

        c_new, R_new, active_new, ub_info = split_overloaded_active_spheres(
            model,
            c_cur,
            R_cur,
            tr_loader,
            device,
            curvature,
            active_mask=active_cur,
            max_cluster_fraction=ub_max_cluster_fraction,
            split_seed=split_seed + it,
        )
        ub_changed = bool(ub_info.get("applied", False))
        hc_info = {"enabled": False, "applied": False}
        hc_changed = False
        if hard_cap_enabled and ub_max_cluster_fraction > 0:
            c_new, R_new, active_new, hc_info = hard_cap_reassign_clusters(
                model,
                c_new,
                R_new,
                tr_loader,
                device,
                curvature,
                active_new,
                max_cluster_fraction=ub_max_cluster_fraction,
                chaos_factor=chaos_factor,
            )
            hc_changed = bool(hc_info.get("applied", False))
        iter_changed = bool(lb_info["changed"] or ub_changed or hc_changed)
        iter_info["lb"] = lb_info
        iter_info["ub"] = ub_info
        iter_info["hard_cap"] = hc_info
        iter_info["changed"] = iter_changed
        info["iterations"].append(iter_info)

        c_cur, R_cur, active_cur = c_new, R_new, active_new
        if iter_changed:
            any_change = True
        else:
            break

    info["applied"] = bool(any_change)
    info["active_cluster_mask_after_hybrid"] = active_cur.astype(bool).tolist()
    return c_cur, R_cur, active_cur, info


def _parse_digit_list(s: str, arg_name: str) -> list[int]:
    v = s.strip().lower()
    if v == "all":
        return list(range(10))
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            d = int(tok)
        except ValueError as e:
            raise SystemExit(f"{arg_name}: expected comma-separated ints in [0..9] or 'all' (got {s!r})") from e
        if d < 0 or d > 9:
            raise SystemExit(f"{arg_name}: digit out of range [0..9]: {d}")
        out.append(d)
    if not out:
        raise SystemExit(f"{arg_name}: no digits parsed from {s!r}")
    out = sorted(set(out))
    return out


def _write_ae_only_metadata(
    xp_path: str,
    args,
    train_digits: list[int],
    test_digits: list[int],
    checkpoint_path: str | None,
) -> None:
    out = {
        "mode": "ae_only",
        "version": 2,
        "ae_n_epochs": int(args.ae_n_epochs),
        "ae_lr": float(args.ae_lr),
        "weight_decay": float(args.weight_decay),
        "rep_dim": int(args.rep_dim),
        "z_dim": int(args.z_dim),
        "n_spheres": int(args.n_spheres),
        "curvature": float(args.curvature),
        "train_normal_digits": train_digits,
        "test_digits": test_digits,
        "used_checkpoint_input": args.ae_stage1_checkpoint_path if args.skip_ae_pretrain else None,
        "saved_checkpoint": checkpoint_path,
    }
    with open(os.path.join(xp_path, "results_ae_only.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _paper_radius_active_mask(assign_counts: np.ndarray, nu: float) -> np.ndarray:
    """
    Paper-style cluster survival rule:
      keep k iff n_k >= nu * max_j(n_j)
    """
    if assign_counts.size == 0:
        return np.array([], dtype=bool)
    max_count = int(np.max(assign_counts))
    if max_count <= 0:
        active = np.zeros_like(assign_counts, dtype=bool)
        active[0] = True
        return active
    threshold = float(max(0.0, nu)) * float(max_count)
    active = assign_counts.astype(np.float64) >= threshold
    if not np.any(active):
        active[int(np.argmax(assign_counts))] = True
    return active


@torch.no_grad()
def _recompute_centers_from_train(
    model,
    c_h: torch.Tensor,
    tr_loader: DataLoader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
) -> tuple[torch.Tensor, dict]:
    """
    Paper-style alternating step: with assignments fixed by nearest active sphere,
    update each active center to mean of its assigned head-specific embeddings.
    """
    model.eval()
    K = int(c_h.size(0))
    z_sums = torch.zeros_like(c_h)
    counts = np.zeros(K, dtype=np.int64)
    am_t = torch.as_tensor(active_mask, device=device, dtype=torch.bool)

    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = _dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
        dist_sq = dist_sq.masked_fill(~am_t.unsqueeze(0), float("inf"))
        k_near = dist_sq.argmin(dim=1)
        k_np = k_near.detach().cpu().numpy()
        counts += np.bincount(k_np, minlength=K)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near]
        for k in range(K):
            m = k_near == k
            if torch.any(m):
                z_sums[k] += z_pick[m].sum(dim=0)

    c_new = c_h.clone()
    for k in range(K):
        if bool(active_mask[k]) and counts[k] > 0:
            c_new[k] = z_sums[k] / float(counts[k])
            if not isinstance(model, EuclideanMultiSphereSVDD):
                c_new[k] = proj_ball(c_new[k : k + 1], c=curvature).squeeze(0)

    return c_new, {"assign_counts": counts.astype(int).tolist(), "active_cluster_mask": active_mask.astype(bool).tolist()}


@torch.no_grad()
def _init_centers_e_unsupervised(model: EuclideanMultiSphereSVDD, train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Unsupervised Euclidean init: for each head k, center is mean of z_k over all train samples.
    This avoids label-dependent collapse when training on a subset of digits (e.g., normal_digits=0).
    """
    c = torch.zeros((model.n_digits, model.z_dim), device=device)
    n = 0
    for x_scaled, _ in train_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all(rep)  # [B, K, Z]
        c += z_all.sum(dim=0)
        n += int(z_all.size(0))
    if n <= 0:
        return c
    return c / float(n)


def main():
    default_ae = _HYSP_CHECK / "runs" / "ae_pretrain_mnist" / "ae_stage1.pth"
    p = argparse.ArgumentParser("hySpUnsup v2: unsup multi-sphere + prune + union metrics")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpUnsup/runs/mnist_unsup_hyp_v2")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--digits", type=str, default="all")
    p.add_argument(
        "--normal_digits",
        type=str,
        default=None,
        help="Normal classes used for training only. Example: '0,1,2'. If omitted, uses --digits.",
    )
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--max_samples_per_class", type=int, default=None)
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--n_spheres", type=int, default=10)
    p.add_argument("--geometry", type=str, default="hyperbolic", choices=["hyperbolic", "euclidean"])
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
    p.add_argument("--ae_only", action="store_true", help="Run/load stage-1 AE only, then exit before any SVDD training.")
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
        "--max_cluster_fraction",
        type=float,
        default=0.0,
        help="If >0, post-prune split overloaded active clusters when train assignment count > fraction*N_train (uses spare inactive spheres).",
    )
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--hybrid_rebalance", action="store_true", help="Enable hybrid LB/UB rebalance (v3 style).")
    p.add_argument(
        "--lb_min_cluster_members",
        type=int,
        default=None,
        help="Lower-bound member cap for hybrid rebalance (defaults to --min_cluster_members).",
    )
    p.add_argument(
        "--ub_max_cluster_fraction",
        type=float,
        default=None,
        help="Upper-bound fraction cap for hybrid rebalance (defaults to --max_cluster_fraction).",
    )
    p.add_argument("--hybrid_max_iters", type=int, default=3)
    p.add_argument("--hard_cap_reassign", action="store_true", help="After UB split, hard-cap saturated clusters by rerouting overflow.")
    p.add_argument(
        "--inline_radius_prune",
        action="store_true",
        help="Paper-style inline cluster shutdown during training: set R_k=0 and deactivate sphere k when n_k < nu*max_j(n_j) each epoch.",
    )
    p.add_argument(
        "--inline_update_centers",
        action="store_true",
        help="Paper-style alternating update: recompute active centers from nearest-sphere assignments during training.",
    )
    p.add_argument(
        "--inline_update_centers_every",
        type=int,
        default=1,
        help="Apply inline center refresh every N epochs when --inline_update_centers is enabled.",
    )
    p.add_argument(
        "--chaos_factor",
        type=float,
        default=0.15,
        help="Small saturation penalty during hard-cap rerouting/subdivision. 0=deterministic nearest, ~0.1-0.3=gentle diversification.",
    )
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
    if args.n_spheres < 1:
        raise SystemExit("--n_spheres must be >= 1")
    os.makedirs(args.xp_path, exist_ok=True)
    device = torch.device(args.device)
    test_digits = _parse_digit_list(args.digits, "--digits")
    train_digits = _parse_digit_list(args.normal_digits, "--normal_digits") if args.normal_digits else list(test_digits)
    print(f"[DATA] train normal digits={train_digits} | test digits={test_digits}")

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
    if args.geometry == "hyperbolic":
        model = HyperbolicMultiSphereSVDD(
            backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=args.n_spheres, c=args.curvature
        ).to(device)
    else:
        model = EuclideanMultiSphereSVDD(
            backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=args.n_spheres
        ).to(device)
        # Compatibility shim for shared helper functions in train_hyp_mnist_unsup.py
        model.project_all_h = model.project_all  # type: ignore[attr-defined]
        unsup.dist_sq_to_all_centers = lambda z_all, c_h, curvature: dist_sq_to_all_centers_e(z_all, c_h)
    dist_fn = dist_sq_to_all_centers if args.geometry == "hyperbolic" else (lambda z_all, c_h, curvature: dist_sq_to_all_centers_e(z_all, c_h))
    overlap_fn = sphere_overlap_penalty if args.geometry == "hyperbolic" else (lambda c_h, R, curvature, margin: _sphere_overlap_penalty_euclidean(c_h, R, margin))

    if args.skip_ae_pretrain:
        ckpt_path = args.ae_stage1_checkpoint_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise SystemExit(f"--skip_ae_pretrain requires a valid --ae_stage1_checkpoint_path (got {ckpt_path!r})")
        ae_ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        missing = list(missing)
        unexpected = list(unexpected)
        bad_missing = [k for k in missing if not k.startswith("proj_heads.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("proj_heads.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Checkpoint/model mismatch outside projection heads.\n"
                f"missing(non-proj)={bad_missing}\n"
                f"unexpected(non-proj)={bad_unexpected}"
            )
        n_proj_missing = len([k for k in missing if k.startswith("proj_heads.")])
        n_proj_unexpected = len([k for k in unexpected if k.startswith("proj_heads.")])
        if n_proj_missing or n_proj_unexpected:
            print(
                "[AE] loaded with non-strict projection-head match "
                f"(missing proj params={n_proj_missing}, unexpected proj params={n_proj_unexpected})."
            )
            print(
                "[AE] This is expected when checkpoint head-count != --n_spheres; "
                "extra heads stay randomly initialized."
            )
        print(f"[AE] skipped pretrain; loaded: {ckpt_path}")
        if args.ae_only:
            _write_ae_only_metadata(args.xp_path, args, train_digits, test_digits, ckpt_path)
            print("[AE] ae_only=True; exiting after checkpoint load.")
            return
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
        else:
            out_path = None
        if args.ae_only:
            _write_ae_only_metadata(args.xp_path, args, train_digits, test_digits, out_path)
            print("[AE] ae_only=True; exiting before SVDD stage.")
            return

    if args.geometry == "hyperbolic":
        c_h = init_centers_h_unsupervised(model, tr_loader, device=device)
    else:
        c_h = _init_centers_e_unsupervised(model, tr_loader, device=device)
    R = torch.zeros((args.n_spheres,), device=device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_objective = "soft-boundary" if args.objective == "union-soft" else "one-class"
    best_train_loss = float("inf")
    best_epoch = -1
    macro_auc_at_best = float("nan")
    best_per_digit = {}
    training_active_mask = np.ones(args.n_spheres, dtype=bool)
    inline_history: list[dict] = []
    inline_center_history: list[dict] = []

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        rec_losses = []
        svdd_losses = []
        ov_losses = []
        dist_sq_chunks = [[] for _ in range(args.n_spheres)]
        assign_counts = np.zeros(args.n_spheres, dtype=np.int64)

        for x_scaled, _digits_b in tr_loader:
            x_scaled = x_scaled.to(device)
            rep, recon = model(x_scaled)
            z_all_h = model.project_all_h(rep)
            dist_sq_all = dist_fn(z_all_h, c_h, curvature=args.curvature)
            am_train = torch.as_tensor(training_active_mask, device=device, dtype=torch.bool)
            dist_sq_all = dist_sq_all.masked_fill(~am_train.unsqueeze(0), float("inf"))
            rec = recon_mse_loss(recon, x_scaled)

            min_sq, k_idx = dist_sq_all.min(dim=1)
            assign_counts += np.bincount(k_idx.detach().cpu().numpy(), minlength=args.n_spheres)
            if args.objective == "union-one":
                sv = svdd_loss_one_class(min_sq)
            else:
                R_sel = R[k_idx]
                sv = svdd_loss_soft_boundary(min_sq, R_sel, nu=args.nu)

            ov = overlap_fn(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
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
                    for k in range(args.n_spheres):
                        m = k_idx == k
                        if torch.any(m):
                            dist_sq_chunks[k].append(dist_sq_all[m, k].detach().cpu())

        if args.objective == "union-soft" and ep > args.warm_up_n_epochs:
            R = update_radii_unsupervised(dist_sq_chunks, nu=args.nu, device=torch.device("cpu")).to(device)
            if args.inline_radius_prune:
                training_active_mask = _paper_radius_active_mask(assign_counts, args.nu)
                R = R.masked_fill(~torch.as_tensor(training_active_mask, device=device, dtype=torch.bool), 0.0)
        elif args.inline_radius_prune:
            training_active_mask = _paper_radius_active_mask(assign_counts, args.nu)

        if args.inline_update_centers and ((ep % max(1, int(args.inline_update_centers_every))) == 0):
            c_h, center_info = _recompute_centers_from_train(
                model=model,
                c_h=c_h,
                tr_loader=tr_loader,
                device=device,
                curvature=args.curvature,
                active_mask=training_active_mask,
            )
            center_info["epoch"] = int(ep)
            inline_center_history.append(center_info)

        if args.inline_radius_prune:
            inline_history.append(
                {
                    "epoch": int(ep),
                    "assign_counts": assign_counts.astype(int).tolist(),
                    "active_cluster_mask": training_active_mask.astype(bool).tolist(),
                }
            )

        epoch_loss_mean = float(np.mean(losses))
        print(
            f"[SVDD-H-UNSUP-v2] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={epoch_loss_mean:.6f} rec={np.mean(rec_losses):.6f} svdd={np.mean(svdd_losses):.6f} "
            f"overlap={np.mean(ov_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )
        if args.inline_radius_prune:
            print(
                f"[INLINE-PRUNE] active spheres: {training_active_mask.astype(np.int32).tolist()} "
                f"({int(training_active_mask.sum())}/{args.n_spheres})"
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
                active_cluster_mask=training_active_mask,
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
                    "geometry": args.geometry,
                    "objective": args.objective,
                    "unsupervised": True,
                    "version": 2,
                    "best_training_loss": best_train_loss,
                    "best_epoch": best_epoch,
                    "active_cluster_mask": training_active_mask.tolist(),
                },
                os.path.join(args.xp_path, "checkpoint_best.pth"),
            )
            print(
                f"[BEST] min-loss checkpoint epoch={ep:03d} train_loss={best_train_loss:.6f} macro_auc(test)={macro_b}"
            )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_digit = evaluate(
                model,
                c_h,
                R,
                te_loader,
                device,
                eval_objective,
                args.curvature,
                active_cluster_mask=training_active_mask,
                auc_mode=args.auc_mode,
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
                "n_spheres": args.n_spheres,
                "geometry": args.geometry,
                "objective": args.objective,
                "unsupervised": True,
                "version": 2,
                "active_cluster_mask": training_active_mask.tolist(),
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    best_ckpt = os.path.join(args.xp_path, "checkpoint_best.pth")
    if os.path.isfile(best_ckpt):
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck["model_state"], strict=True)
        c_h = ck["c_h"].to(device)
        R = ck["R"].to(device)
        if "active_cluster_mask" in ck:
            training_active_mask = np.asarray(ck["active_cluster_mask"], dtype=bool)

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
    active_mask = np.logical_and(active_mask, training_active_mask)
    if not np.any(active_mask):
        active_mask[int(np.argmax(training_active_mask.astype(np.int32)))] = True
    prune_info["inline_active_cluster_mask"] = training_active_mask.astype(bool).tolist()
    prune_info["active_cluster_mask_after_inline_and_prune"] = active_mask.astype(bool).tolist()
    c_h, R, active_mask, split_info = split_overloaded_active_spheres(
        model,
        c_h,
        R,
        tr_loader,
        device,
        args.curvature,
        active_mask=active_mask,
        max_cluster_fraction=args.max_cluster_fraction,
        split_seed=args.split_seed,
    )
    hybrid_info = {"enabled": False, "applied": False}
    if args.hybrid_rebalance:
        lb_eff = args.lb_min_cluster_members if args.lb_min_cluster_members is not None else args.min_cluster_members
        ub_eff = args.ub_max_cluster_fraction if args.ub_max_cluster_fraction is not None else args.max_cluster_fraction
        c_h, R, active_mask, hybrid_info = hybrid_lb_ub_rebalance(
            model,
            c_h,
            R,
            tr_loader,
            device,
            args.curvature,
            active_mask=active_mask,
            lb_min_cluster_members=int(max(0, lb_eff)),
            ub_max_cluster_fraction=float(max(0.0, ub_eff)),
            split_seed=args.split_seed,
            max_iters=args.hybrid_max_iters,
            hard_cap_enabled=bool(args.hard_cap_reassign),
            chaos_factor=float(max(0.0, args.chaos_factor)),
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

    print(
        f"[v2 PRUNE] active spheres: {active_mask.astype(np.int32).tolist()}  ({int(active_mask.sum())}/{args.n_spheres})"
    )
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
        "n_spheres": args.n_spheres,
        "geometry": args.geometry,
        "train_normal_digits": train_digits,
        "test_digits": test_digits,
        "objective": args.objective,
        "unsupervised": True,
        "ae_checkpoint": args.ae_stage1_checkpoint_path if args.skip_ae_pretrain else None,
        "prune": prune_info,
        "split_overloaded": split_info,
        "hybrid_rebalance": hybrid_info,
        "union_test_metrics_active_spheres": union_m,
        "inline_radius_prune": {
            "enabled": bool(args.inline_radius_prune),
            "history": inline_history,
        },
        "inline_center_update": {
            "enabled": bool(args.inline_update_centers),
            "every_n_epochs": int(max(1, int(args.inline_update_centers_every))),
            "history": inline_center_history,
        },
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
            "n_spheres": args.n_spheres,
            "geometry": args.geometry,
            "train_normal_digits": train_digits,
            "test_digits": test_digits,
            "objective": args.objective,
            "unsupervised": True,
            "version": 2,
            "active_cluster_mask": active_mask.tolist(),
            "prune_info": prune_info,
            "split_info": split_info,
            "hybrid_info": hybrid_info,
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
