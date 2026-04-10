import argparse
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from cifar10_data import CIFAR10RawDataset
from cifar_backbone import CifarConvAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers


CIFAR10_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("t-SNE visualization for CIFAR10 hyperbolic multi-sphere")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_iter", type=int, default=1000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    backbone = CifarConvAE(rep_dim=int(ckpt["rep_dim"]))
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone,
        rep_dim=int(ckpt["rep_dim"]),
        z_dim=int(ckpt["z_dim"]),
        n_classes=10,
        c=float(ckpt["curvature"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    c_h = ckpt["c_h"].to(device)
    R = ckpt["R"].to(device)
    objective = ckpt.get("objective", "soft-boundary")

    ds = CIFAR10RawDataset(root=args.data_root, split="test", digits=list(range(10)), max_samples=args.max_samples, download=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    z_list, y_list, in_list = [], [], []
    for x, y, _idx in dl:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z_self = model.project_self_h(rep, y)
        z_all = model.project_all_h(rep)
        d2_all = dist_sq_to_all_centers(z_all, c_h, curvature=model.curvature)
        if objective == "soft-boundary":
            s_all = d2_all - (R.unsqueeze(0) ** 2)
            inside = s_all[torch.arange(s_all.size(0), device=s_all.device), y] <= 0
        else:
            inside = torch.ones((y.size(0),), dtype=torch.bool, device=y.device)
        z_list.append(z_self.cpu())
        y_list.append(y.cpu())
        in_list.append(inside.cpu())

    Z = torch.cat(z_list).numpy()
    Y = torch.cat(y_list).numpy()
    INS = torch.cat(in_list).numpy()

    try:
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=args.perplexity, n_iter=args.n_iter)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=args.seed, perplexity=args.perplexity, max_iter=args.n_iter)
    Z2 = tsne.fit_transform(Z)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    plt.figure(figsize=(10, 8))
    for k in range(10):
        m = Y == k
        if not np.any(m):
            continue
        m_in = m & INS
        m_out = m & (~INS)
        if np.any(m_in):
            plt.scatter(Z2[m_in, 0], Z2[m_in, 1], s=10, marker="o", color=colors[k], alpha=0.85)
        if np.any(m_out):
            plt.scatter(Z2[m_out, 0], Z2[m_out, 1], s=10, marker="x", color=colors[k], alpha=0.85)

    leg_state = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="inside own sphere"),
        Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="outside own sphere"),
    ]
    leg_names = [
        Line2D([0], [0], marker="o", color=colors[k], linestyle="None", markersize=6, label=CIFAR10_NAMES[k]) for k in range(10)
    ]
    l1 = plt.legend(handles=leg_state, loc="lower left", fontsize=9, frameon=True)
    ax = plt.gca()
    l2 = ax.legend(handles=leg_names, loc="upper left", fontsize=8, frameon=True, ncol=2)
    ax.add_artist(l1)
    plt.title("CIFAR10 hyperbolic z_self t-SNE")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    out = os.path.join(args.out_dir, "tsne_hyp_cifar_zself_inside_outside.png")
    plt.savefig(out, dpi=160)
    print("Saved:", out)


if __name__ == "__main__":
    main()

