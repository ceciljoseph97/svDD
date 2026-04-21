"""
v3 entrypoint: unsupervised hyperbolic multi-sphere with bottom-up agglomerative init.

Defaults over v2:
- enables `--hybrid_rebalance` and `--hard_cap_reassign`
- uses agglomerative (bottom-up) cluster-based center initialization by default
- keeps v2 behavior otherwise (same args, checkpoints, exports)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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


def _parse_v3_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--init_linkage", type=str, default="ward", choices=["ward", "complete", "average", "single"])
    p.add_argument(
        "--disable_agglomerative_init",
        action="store_true",
        help="Use v2 default mean init instead of agglomerative bottom-up init.",
    )
    return p.parse_known_args(argv)


def main() -> None:
    v3_args, passthrough = _parse_v3_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *passthrough]
    geometry = "hyperbolic"
    if "--geometry" in passthrough:
        try:
            geometry = passthrough[passthrough.index("--geometry") + 1]
        except Exception:
            geometry = "hyperbolic"

    # v3 defaults: turn on hybrid rebalance and hard-cap unless user explicitly sets them.
    if "--hybrid_rebalance" not in sys.argv:
        sys.argv.append("--hybrid_rebalance")
    if "--hard_cap_reassign" not in sys.argv:
        sys.argv.append("--hard_cap_reassign")

    if geometry == "euclidean":
        print("[v3] geometry=euclidean; agglomerative hyperbolic init is disabled")
    elif not v3_args.disable_agglomerative_init:
        def _wrapped_agglomerative_init(model, train_loader, device, eps=1e-5):
            return init_centers_h_unsupervised_agglomerative(
                model=model,
                train_loader=train_loader,
                device=device,
                eps=eps,
                linkage=v3_args.init_linkage,
            )

        v2.init_centers_h_unsupervised = _wrapped_agglomerative_init
        print(f"[v3] using agglomerative bottom-up init (linkage={v3_args.init_linkage})")
    else:
        print("[v3] agglomerative init disabled; using v2 mean init")

    v2.main()


if __name__ == "__main__":
    main()
