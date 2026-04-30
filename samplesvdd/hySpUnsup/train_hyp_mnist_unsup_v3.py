"""
v3 entrypoint: unsupervised adaptive multi-sphere clustering with:
- autoencoder-based representation learning (inherited from v2)
- K-means center initialization by default
- adaptive radii
- inline prune / split dynamics
- silhouette-guided cluster quality control

All v2 exports remain intact: checkpoints, t-SNE, hotspot analysis, cluster exports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import train_hyp_mnist_unsup_v2 as v2


@torch.no_grad()
def init_centers_h_unsupervised_agglomerative(
    model,
    train_loader,
    device: torch.device,
    eps: float = 1e-5,
    linkage: str = "ward",
):
    """
    Bottom-up init:
    1) cluster encoder reps with AgglomerativeClustering(K)
    2) for each cluster k, initialize center c_k using head-k embeddings of its members
    """
    from sklearn.cluster import AgglomerativeClustering
    from hyperbolic_ops import proj_ball

    reps_list: list[torch.Tensor] = []
    z_all_list: list[torch.Tensor] = []

    for x_scaled, _ in train_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        reps_list.append(rep.detach().cpu())
        z_all_list.append(z_all.detach().cpu())

    if not reps_list:
        c = torch.zeros((model.n_digits, model.z_dim), device=device)
        return proj_ball(c, c=model.curvature, eps=eps)

    reps = torch.cat(reps_list, dim=0).numpy()
    z_all_np = torch.cat(z_all_list, dim=0).numpy()  # [N, K, Z]

    n_samples = int(reps.shape[0])
    n_clusters = int(model.n_digits)
    if n_samples < n_clusters:
        # Safety fallback: not enough samples to run K-way agglomerative.
        c = torch.from_numpy(np.mean(z_all_np, axis=0)).to(device=device, dtype=torch.float32)
        return proj_ball(c, c=model.curvature, eps=eps)

    if linkage == "ward":
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        try:
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric="euclidean")
        except TypeError:
            # Backward compatibility with older sklearn API.
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="euclidean")
    labels = agg.fit_predict(reps)

    c_np = np.zeros((n_clusters, model.z_dim), dtype=np.float32)
    global_head_means = np.mean(z_all_np, axis=0).astype(np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            c_np[k] = np.mean(z_all_np[mask, k, :], axis=0)
        else:
            c_np[k] = global_head_means[k]

    c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
    c = proj_ball(c, c=model.curvature, eps=eps)
    return c


@torch.no_grad()
def init_centers_h_unsupervised_kmeans(
    model,
    train_loader,
    device: torch.device,
    eps: float = 1e-5,
    seed: int = 42,
):
    """
    K-means init:
    1) cluster encoder reps with KMeans(K)
    2) initialize center c_k from the corresponding head-k embeddings
    """
    from hyperbolic_ops import proj_ball

    reps_list: list[torch.Tensor] = []
    z_all_list: list[torch.Tensor] = []

    for x_scaled, _ in train_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        reps_list.append(rep.detach().cpu())
        z_all_list.append(z_all.detach().cpu())

    if not reps_list:
        c = torch.zeros((model.n_digits, model.z_dim), device=device)
        return proj_ball(c, c=model.curvature, eps=eps)

    reps = torch.cat(reps_list, dim=0).numpy()
    z_all_np = torch.cat(z_all_list, dim=0).numpy()

    n_samples = int(reps.shape[0])
    n_clusters = int(model.n_digits)
    if n_samples < n_clusters:
        c = torch.from_numpy(np.mean(z_all_np, axis=0)).to(device=device, dtype=torch.float32)
        return proj_ball(c, c=model.curvature, eps=eps)

    km = KMeans(n_clusters=n_clusters, random_state=int(seed), n_init=10)
    labels = km.fit_predict(reps)

    c_np = np.zeros((n_clusters, model.z_dim), dtype=np.float32)
    global_head_means = np.mean(z_all_np, axis=0).astype(np.float32)
    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            c_np[k] = np.mean(z_all_np[mask, k, :], axis=0)
        else:
            c_np[k] = global_head_means[k]

    c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
    c = proj_ball(c, c=model.curvature, eps=eps)
    return c


def _infer_n_spheres(argv: list[str], default: int = 10) -> int:
    if "--n_spheres" not in argv:
        return int(default)
    try:
        return int(argv[argv.index("--n_spheres") + 1])
    except Exception:
        return int(default)


def _infer_int_arg(argv: list[str], key: str, default: int) -> int:
    if key not in argv:
        return int(default)
    try:
        return int(argv[argv.index(key) + 1])
    except Exception:
        return int(default)


def _cluster_mean_intra_distance(z_k: np.ndarray, max_ref: int, rng: np.random.RandomState) -> np.ndarray:
    n_k = int(z_k.shape[0])
    if n_k <= 1:
        return np.zeros((n_k,), dtype=np.float32)
    if n_k <= max_ref:
        ref = z_k
    else:
        ref_idx = rng.choice(n_k, size=max_ref, replace=False)
        ref = z_k[ref_idx]
    diff = z_k[:, None, :] - ref[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    if ref.shape[0] == n_k:
        denom = max(1, n_k - 1)
        return (d.sum(axis=1) / float(denom)).astype(np.float32)
    return d.mean(axis=1).astype(np.float32)


@torch.no_grad()
def _collect_adaptive_cluster_snapshot(
    model,
    c_h: torch.Tensor,
    tr_loader,
    device: torch.device,
    curvature: float,
    active_mask: np.ndarray,
    nu: float,
    silhouette_ref_samples: int,
    silhouette_seed: int,
) -> dict:
    am_t = torch.as_tensor(active_mask, device=device, dtype=torch.bool)
    z_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    dist_parts: list[np.ndarray] = []
    all_dist_parts: list[np.ndarray] = []

    for x_scaled, _ in tr_loader:
        x_scaled = x_scaled.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)
        dist_sq = v2._dist_sq_all_for_model(model, z_all, c_h, curvature=curvature)
        masked = dist_sq.masked_fill(~am_t.unsqueeze(0), float("inf"))
        k_near = masked.argmin(dim=1)
        b = torch.arange(z_all.size(0), device=device)
        z_pick = z_all[b, k_near]
        d_pick = masked[b, k_near]
        z_parts.append(z_pick.detach().cpu().numpy())
        k_parts.append(k_near.detach().cpu().numpy())
        dist_parts.append(d_pick.detach().cpu().numpy())
        all_dist_parts.append(masked.detach().cpu().numpy())

    K_total = int(c_h.size(0))
    if not z_parts:
        zeros = np.zeros(K_total, dtype=np.int64)
        return {
            "Z": np.zeros((0, int(c_h.size(1))), dtype=np.float32),
            "A": np.zeros((0,), dtype=np.int64),
            "D_pick": np.zeros((0,), dtype=np.float32),
            "D_all": np.zeros((0, K_total), dtype=np.float32),
            "counts": zeros,
            "radius": np.zeros(K_total, dtype=np.float32),
            "sil_mean": np.zeros(K_total, dtype=np.float32),
            "neg_frac": np.zeros(K_total, dtype=np.float32),
        }

    Z = np.concatenate(z_parts, axis=0)
    A = np.concatenate(k_parts, axis=0).astype(np.int64)
    D_pick = np.concatenate(dist_parts, axis=0)
    D_all = np.concatenate(all_dist_parts, axis=0)
    counts = np.bincount(A, minlength=K_total)

    radius = np.zeros(K_total, dtype=np.float32)
    sil_mean = np.zeros(K_total, dtype=np.float32)
    neg_frac = np.zeros(K_total, dtype=np.float32)
    rng = np.random.RandomState(int(silhouette_seed))
    q = float(max(0.0, min(1.0, 1.0 - float(nu))))

    for k in range(K_total):
        mk = A == k
        n_k = int(np.sum(mk))
        if n_k <= 0:
            continue

        d_k = np.maximum(D_pick[mk], 0.0)
        radius[k] = float(np.sqrt(np.quantile(d_k, q=q)))

        z_k = Z[mk]
        a_i = _cluster_mean_intra_distance(z_k, max_ref=max(8, int(silhouette_ref_samples)), rng=rng)

        d_other = D_all[mk].copy()
        d_other[:, k] = np.inf
        b_sq = np.min(np.maximum(d_other, 0.0), axis=1)
        b_i = np.sqrt(b_sq)
        # When no alternative cluster is available (all inf), silhouette is undefined.
        # Use 0.0 to keep training/export metrics numerically stable.
        invalid = ~np.isfinite(b_i)
        denom = np.maximum(np.maximum(a_i, b_i), 1e-8)
        s_i = np.divide(
            (b_i - a_i),
            denom,
            out=np.zeros_like(a_i, dtype=np.float32),
            where=(~invalid) & np.isfinite(denom),
        )
        s_i = np.nan_to_num(s_i, nan=0.0, posinf=0.0, neginf=0.0)
        sil_mean[k] = float(np.mean(s_i))
        neg_frac[k] = float(np.mean(s_i < 0.0))

    return {
        "Z": Z,
        "A": A,
        "D_pick": D_pick,
        "D_all": D_all,
        "counts": counts.astype(np.int64),
        "radius": radius,
        "sil_mean": sil_mean,
        "neg_frac": neg_frac,
    }


def _project_center_if_needed(model, center: torch.Tensor, curvature: float) -> torch.Tensor:
    if isinstance(model, v2.EuclideanMultiSphereSVDD):
        return center
    return v2.proj_ball(center.unsqueeze(0), c=curvature).squeeze(0)


def _radius_from_points(model, z_points: torch.Tensor, center: torch.Tensor, curvature: float, nu: float) -> torch.Tensor:
    if z_points.size(0) == 0:
        return torch.tensor(0.0, device=center.device, dtype=center.dtype)
    if isinstance(model, v2.EuclideanMultiSphereSVDD):
        d_sq = ((z_points - center.unsqueeze(0)) ** 2).sum(dim=1)
    else:
        d_sq = v2.hyp_distance(z_points, center.unsqueeze(0).expand_as(z_points), c=curvature) ** 2
    q = float(max(0.0, min(1.0, 1.0 - float(nu))))
    return torch.sqrt(torch.quantile(d_sq, q=q).clamp_min(0.0))


def _make_silhouette_adaptive_splitter(config: argparse.Namespace):
    state = {"epoch_calls": 0, "prune_cooldown": None}

    @torch.no_grad()
    def _adaptive_split(
        model,
        c_h: torch.Tensor,
        R: torch.Tensor,
        tr_loader,
        device: torch.device,
        curvature: float,
        active_mask: np.ndarray,
        threshold: float,
        min_members: int,
        max_splits: int,
        seed: int,
        eps: float,
        nu: float,
    ):
        del threshold, eps  # v3 uses silhouette-guided control instead.
        state["epoch_calls"] += 1
        epoch_idx = int(state["epoch_calls"])

        info: dict = {
            "enabled": True,
            "applied": False,
            "epoch_call": epoch_idx,
            "silhouette_prune_threshold": float(config.silhouette_prune_threshold),
            "silhouette_split_threshold": float(config.silhouette_split_threshold),
            "silhouette_split_negative_fraction": float(config.silhouette_split_negative_fraction),
            "split_frequency": int(max(1, int(config.silhouette_split_frequency))),
            "k_max": int(config.k_max),
            "splits": [],
        }
        split_start = int(max(1, int(config.schedule_split_start_epoch)))
        split_every = int(max(1, int(config.schedule_split_every_epochs)))
        prune_start = int(max(int(config.prune_start_epoch), int(config.schedule_prune_start_epoch)))
        prune_every = int(max(1, int(config.schedule_prune_every_epochs)))
        wait_after_split = int(max(0, int(config.schedule_wait_after_split_epochs)))
        info["schedule"] = {
            "split_start_epoch": split_start,
            "split_every_epochs": split_every,
            "prune_start_epoch": prune_start,
            "prune_every_epochs": prune_every,
            "wait_after_split_epochs": wait_after_split,
            "svdd_total_epochs": int(config.svdd_total_epochs),
        }

        snapshot = _collect_adaptive_cluster_snapshot(
            model=model,
            c_h=c_h,
            tr_loader=tr_loader,
            device=device,
            curvature=curvature,
            active_mask=active_mask,
            nu=nu,
            silhouette_ref_samples=int(config.silhouette_ref_samples),
            silhouette_seed=int(seed),
        )
        counts = snapshot["counts"]
        radius = snapshot["radius"]
        sil_mean = snapshot["sil_mean"]
        neg_frac = snapshot["neg_frac"]
        if state["prune_cooldown"] is None or len(state["prune_cooldown"]) != int(c_h.size(0)):
            state["prune_cooldown"] = np.zeros(int(c_h.size(0)), dtype=np.int64)
        cooldown = state["prune_cooldown"]

        c_new = c_h.clone()
        R_new = R.clone()
        active_after_split = active_mask.copy()

        info["counts_before"] = counts.astype(int).tolist()
        info["radius_estimate_before"] = [float(x) for x in radius.tolist()]
        info["silhouette_mean_before"] = [float(x) for x in sil_mean.tolist()]
        info["negative_silhouette_fraction_before"] = [float(x) for x in neg_frac.tolist()]
        info["active_before"] = active_mask.astype(bool).tolist()
        info["prune_cooldown_before"] = cooldown.astype(int).tolist()

        should_split = (epoch_idx >= split_start) and (((epoch_idx - split_start) % split_every) == 0)
        info["split_step"] = bool(should_split)
        if should_split:
            current_active = int(np.sum(active_after_split))
            spare = [k for k in range(int(c_h.size(0))) if not bool(active_after_split[k])]
            split_budget = min(int(max_splits), max(0, int(config.k_max) - current_active), len(spare))
            if split_budget > 0:
                A = snapshot["A"]
                Z = snapshot["Z"]
                candidates = [
                    k
                    for k in range(int(c_h.size(0)))
                    if bool(active_after_split[k])
                    and int(counts[k]) >= max(2, int(min_members))
                    and (
                        float(sil_mean[k]) < float(config.silhouette_split_threshold)
                        or float(neg_frac[k]) > float(config.silhouette_split_negative_fraction)
                    )
                ]
                candidates = sorted(
                    candidates,
                    key=lambda k: (float(sil_mean[k]), -float(neg_frac[k]), -int(counts[k])),
                )
                info["split_budget"] = int(split_budget)
                info["split_candidates"] = [int(k) for k in candidates]
                rng = np.random.RandomState(int(seed))
                splits_done = 0
                protected_after_split: list[int] = []
                for src in candidates:
                    if splits_done >= split_budget or not spare:
                        break
                    idx = np.where(A == src)[0]
                    if idx.size < max(4, 2 * int(min_members)):
                        continue
                    Zk = Z[idx]
                    km = KMeans(n_clusters=2, random_state=int(rng.randint(0, 1_000_000)), n_init=10)
                    labels = km.fit_predict(Zk)
                    idx0 = idx[labels == 0]
                    idx1 = idx[labels == 1]
                    if idx0.size < max(2, int(min_members // 2)) or idx1.size < max(2, int(min_members // 2)):
                        continue

                    child = int(spare.pop(0))
                    z0 = torch.as_tensor(Z[idx0], dtype=c_h.dtype, device=device)
                    z1 = torch.as_tensor(Z[idx1], dtype=c_h.dtype, device=device)
                    c0 = _project_center_if_needed(model, z0.mean(dim=0), curvature)
                    c1 = _project_center_if_needed(model, z1.mean(dim=0), curvature)
                    r0 = _radius_from_points(model, z0, c0, curvature, nu)
                    r1 = _radius_from_points(model, z1, c1, curvature, nu)

                    c_new[src] = c0
                    c_new[child] = c1
                    R_new[src] = r0
                    R_new[child] = r1
                    active_after_split[child] = True
                    splits_done += 1
                    protected_after_split.extend([int(src), int(child)])
                    info["splits"].append(
                        {
                            "source_cluster": int(src),
                            "new_cluster": int(child),
                            "source_count": int(idx.size),
                            "split_counts": [int(idx0.size), int(idx1.size)],
                            "silhouette_mean_before": float(sil_mean[src]),
                            "negative_fraction_before": float(neg_frac[src]),
                        }
                    )
                info["applied"] = bool(splits_done > 0)
                info["n_splits"] = int(splits_done)
                if protected_after_split:
                    hold = int(max(0, int(config.post_split_stability_epochs))) + int(wait_after_split) + 1
                    for k in protected_after_split:
                        cooldown[k] = max(int(cooldown[k]), hold)
                    info["protected_after_split"] = sorted(set(int(x) for x in protected_after_split))
            else:
                info["split_budget"] = 0
                info["note"] = "No spare clusters available under k_max."

        # Recompute snapshot for pruning after any split update.
        prune_snapshot = _collect_adaptive_cluster_snapshot(
            model=model,
            c_h=c_new,
            tr_loader=tr_loader,
            device=device,
            curvature=curvature,
            active_mask=active_after_split,
            nu=nu,
            silhouette_ref_samples=int(config.silhouette_ref_samples),
            silhouette_seed=int(seed + 1337),
        )
        counts_p = prune_snapshot["counts"]
        radius_p = prune_snapshot["radius"]
        sil_mean_p = prune_snapshot["sil_mean"]
        neg_frac_p = prune_snapshot["neg_frac"]
        protected = cooldown > 0

        prune_step = (epoch_idx >= prune_start) and (((epoch_idx - prune_start) % prune_every) == 0)
        info["prune_step"] = bool(prune_step)
        if not prune_step:
            active_after_prune = active_after_split.copy()
            info["prune_skipped_until_epoch"] = int(prune_start)
        else:
            size_mask = v2._paper_radius_active_mask(counts_p, nu)
            sil_mask = sil_mean_p >= float(config.silhouette_prune_threshold)
            active_after_prune = active_after_split.copy() & size_mask & sil_mask
            active_after_prune |= (active_after_split.copy() & protected)

        min_active = max(1, int(config.min_active_clusters))
        n_active_now = int(np.sum(active_after_prune))
        if n_active_now < min_active:
            # Recover from collapse: allow re-activating high-support clusters,
            # not only those that were already active this epoch.
            ranked_all = sorted(range(len(counts_p)), key=lambda k: int(counts_p[k]), reverse=True)
            need = int(min_active - n_active_now)
            activated = []
            for k in ranked_all:
                if need <= 0:
                    break
                if bool(active_after_prune[k]):
                    continue
                # Only revive clusters that actually have assigned members.
                if int(counts_p[k]) <= 0:
                    continue
                active_after_prune[k] = True
                activated.append(int(k))
                need -= 1
            if activated:
                info["reactivated_for_min_active"] = activated

        if not np.any(active_after_prune) and counts_p.size > 0:
            active_after_prune[int(np.argmax(counts_p))] = True

        R_new = R_new.masked_fill(~torch.as_tensor(active_after_prune, device=device, dtype=torch.bool), 0.0)
        for k in range(len(radius_p)):
            if bool(active_after_prune[k]) and counts_p[k] > 0:
                R_new[k] = torch.as_tensor(radius_p[k], dtype=R_new.dtype, device=R_new.device)

        info["counts_for_prune"] = counts_p.astype(int).tolist()
        info["radius_estimate"] = [float(x) for x in radius_p.tolist()]
        info["silhouette_mean"] = [float(x) for x in sil_mean_p.tolist()]
        info["negative_silhouette_fraction"] = [float(x) for x in neg_frac_p.tolist()]
        info["max_count"] = int(np.max(counts_p)) if counts_p.size else 0
        info["active_after_split"] = active_after_split.astype(bool).tolist()
        info["active_after_prune"] = active_after_prune.astype(bool).tolist()
        info["prune_protected_mask"] = protected.astype(bool).tolist()

        cooldown = np.maximum(cooldown - 1, 0)
        state["prune_cooldown"] = cooldown
        info["prune_cooldown_after"] = cooldown.astype(int).tolist()
        return c_new, R_new, active_after_prune, info

    return _adaptive_split


def _parse_v3_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--init_method", type=str, default="kmeans", choices=["kmeans", "agglomerative", "mean"])
    p.add_argument("--init_linkage", type=str, default="ward", choices=["ward", "complete", "average", "single"])
    p.add_argument("--init_seed", type=int, default=42)
    p.add_argument(
        "--disable_agglomerative_init",
        action="store_true",
        help="Use v2 default mean init instead of agglomerative bottom-up init.",
    )
    p.add_argument("--silhouette_prune_threshold", type=float, default=0.0)
    p.add_argument("--silhouette_split_threshold", type=float, default=0.2)
    p.add_argument("--silhouette_split_negative_fraction", type=float, default=0.1)
    p.add_argument("--silhouette_split_frequency", type=int, default=5)
    p.add_argument("--silhouette_ref_samples", type=int, default=256)
    p.add_argument("--silhouette_split_max_per_iter", type=int, default=1)
    p.add_argument("--post_split_stability_epochs", type=int, default=3)
    p.add_argument("--prune_start_epoch", type=int, default=1)
    p.add_argument("--min_active_clusters", type=int, default=2)
    p.add_argument("--schedule_split_start_epoch", type=int, default=None)
    p.add_argument("--schedule_wait_after_split_epochs", type=int, default=None)
    p.add_argument("--schedule_split_every_epochs", type=int, default=None)
    p.add_argument("--schedule_prune_start_epoch", type=int, default=None)
    p.add_argument("--schedule_prune_every_epochs", type=int, default=1)
    p.add_argument("--k_max", type=int, default=None)
    return p.parse_known_args(argv)


def main() -> None:
    v3_args, passthrough = _parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]
    svdd_epochs = _infer_int_arg(passthrough, "--svdd_n_epochs", default=25)
    v3_args.svdd_total_epochs = int(max(1, svdd_epochs))
    auto_split_start = max(2, int(np.ceil(0.10 * v3_args.svdd_total_epochs)))
    auto_wait = max(1, int(np.ceil(0.05 * v3_args.svdd_total_epochs)))
    auto_split_every = max(2, int(np.ceil(0.08 * v3_args.svdd_total_epochs)))
    auto_prune_start = min(
        v3_args.svdd_total_epochs,
        max(1, auto_split_start + auto_wait),
    )
    if v3_args.schedule_split_start_epoch is None:
        v3_args.schedule_split_start_epoch = auto_split_start
    if v3_args.schedule_wait_after_split_epochs is None:
        v3_args.schedule_wait_after_split_epochs = auto_wait
    if v3_args.schedule_split_every_epochs is None:
        v3_args.schedule_split_every_epochs = auto_split_every
    if v3_args.schedule_prune_start_epoch is None:
        v3_args.schedule_prune_start_epoch = auto_prune_start
    n_spheres = _infer_n_spheres(passthrough, default=10)
    if v3_args.k_max is None:
        v3_args.k_max = 2 * int(n_spheres)
    v3_args.k_max = max(1, min(int(v3_args.k_max), int(n_spheres)))
    geometry = "hyperbolic"
    if "--geometry" in passthrough:
        try:
            geometry = passthrough[passthrough.index("--geometry") + 1]
        except Exception:
            geometry = "hyperbolic"

    # v3 defaults: turn on inline adaptive dynamics unless user explicitly sets them.
    if "--inline_radius_prune" not in sys.argv:
        sys.argv.append("--inline_radius_prune")
    if "--inline_update_centers" not in sys.argv:
        sys.argv.append("--inline_update_centers")
    # Do NOT force strict paper reporting here: it overrides `--digits` by design in v2.
    # Users can enable it explicitly if they want the normal-vs-rest evaluation mode.
    if "--inline_split_every" not in sys.argv:
        sys.argv.extend(["--inline_split_every", "1"])
    if "--inline_split_max_per_epoch" not in sys.argv:
        sys.argv.extend(["--inline_split_max_per_epoch", str(int(v3_args.silhouette_split_max_per_iter))])
    if "--inline_split_inverse_distance_threshold" not in sys.argv:
        sys.argv.extend(["--inline_split_inverse_distance_threshold", "1.0"])

    if geometry == "euclidean":
        print("[v3] geometry=euclidean; hyperbolic center init override is disabled")
    elif v3_args.disable_agglomerative_init or v3_args.init_method == "mean":
        print("[v3] using v2 mean init")
    elif v3_args.init_method == "agglomerative":
        def _wrapped_agglomerative_init(model, train_loader, device, eps=1e-5):
            return init_centers_h_unsupervised_agglomerative(
                model=model,
                train_loader=train_loader,
                device=device,
                eps=eps,
                linkage=v3_args.init_linkage,
            )

        v2.init_centers_h_unsupervised = _wrapped_agglomerative_init
        print(f"[v3] using agglomerative init (linkage={v3_args.init_linkage})")
    else:
        def _wrapped_kmeans_init(model, train_loader, device, eps=1e-5):
            return init_centers_h_unsupervised_kmeans(
                model=model,
                train_loader=train_loader,
                device=device,
                eps=eps,
                seed=int(v3_args.init_seed),
            )

        v2.init_centers_h_unsupervised = _wrapped_kmeans_init
        print(f"[v3] using k-means init (seed={v3_args.init_seed})")

    v2._inline_inverse_distance_split = _make_silhouette_adaptive_splitter(v3_args)
    print(
        "[v3] silhouette dynamics:"
        f" prune<{v3_args.silhouette_prune_threshold}"
        f" split<{v3_args.silhouette_split_threshold}"
        f" neg_frac>{v3_args.silhouette_split_negative_fraction}"
        f" split_schedule(start={v3_args.schedule_split_start_epoch},every={v3_args.schedule_split_every_epochs})"
        f" prune_schedule(start={v3_args.schedule_prune_start_epoch},every={v3_args.schedule_prune_every_epochs})"
        f" wait_after_split={v3_args.schedule_wait_after_split_epochs}"
        f" k_max={v3_args.k_max}"
    )

    v2.main()


if __name__ == "__main__":
    print("[DEPRECATED ENTRYPOINT] Use: hySpUnsup/train_hyp_mnist_unsup_unified.py")
    from train_hyp_mnist_unsup_unified import main as _main_unified

    _main_unified()
