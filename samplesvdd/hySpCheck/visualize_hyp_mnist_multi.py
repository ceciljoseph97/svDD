import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset

from mnist_local import MNISTDigitsProcessedRawDataset, preprocess_batch_by_digit_minmax, MNIST_LeNet_SVDDIAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD


class ScaledRawDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x_raw, d, idx = self.base[i]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = preprocess_batch_by_digit_minmax(x_raw.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
        return x_scaled, d_t, x_raw, idx


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("t-SNE visualization for hyperbolic multi-sphere")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--n_iter", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--embed_source",
        type=str,
        default="auto",
        choices=["auto", "hyperbolic_zself", "rep"],
        help="auto: use hyperbolic z_self if checkpoint has SVDD fields, else use encoder rep (AE-pretrain).",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

    has_svdd = all(k in ckpt for k in ("rep_dim", "z_dim", "curvature", "c_h", "R"))
    if args.embed_source == "auto":
        embed_mode = "hyperbolic_zself" if has_svdd else "rep"
    else:
        embed_mode = args.embed_source
    if embed_mode == "hyperbolic_zself" and not has_svdd:
        raise ValueError("Checkpoint has no SVDD fields (rep_dim/z_dim/c_h/R). Use --embed_source rep or auto.")

    if embed_mode == "hyperbolic_zself":
        backbone = MNIST_LeNet_SVDDIAE(rep_dim=int(ckpt["rep_dim"]))
        model = HyperbolicMultiSphereSVDD(
            backbone=backbone, rep_dim=int(ckpt["rep_dim"]), z_dim=int(ckpt["z_dim"]), n_digits=10, c=float(ckpt["curvature"])
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        c_h = ckpt["c_h"].to(device)
        R = ckpt["R"].to(device)
    else:
        # Stage-1 AE checkpoint compatibility: only encoder/decoder representation t-SNE.
        ms = ckpt["model_state"]
        if "rep_dim" in ckpt:
            rep_dim = int(ckpt["rep_dim"])
        elif "backbone.fc1.weight" in ms:
            rep_dim = int(ms["backbone.fc1.weight"].shape[0])
        elif "fc1.weight" in ms:
            rep_dim = int(ms["fc1.weight"].shape[0])
        else:
            rep_dim = 32
        model = MNIST_LeNet_SVDDIAE(rep_dim=rep_dim).to(device)
        if any(k.startswith("backbone.") for k in ms.keys()):
            stripped = OrderedDict()
            for k, v in ms.items():
                if k.startswith("backbone."):
                    stripped[k[len("backbone.") :]] = v
            model.load_state_dict(stripped, strict=False)
        else:
            model.load_state_dict(ms, strict=False)
        model.eval()

    base = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir, split=args.split, train_fraction=args.train_fraction, digits=list(range(10))
    )
    ds = ScaledRawDataset(base)
    n = min(args.max_samples, len(ds))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    z_all = []
    d_all = []
    in_all = []
    for x_scaled, digits, _x_raw, _idx in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        if embed_mode == "hyperbolic_zself":
            z_embed = model.project_self_h(rep, digits)
            c_b = c_h[digits]
            dist_sq = torch.sum((z_embed - c_b) ** 2, dim=1)
            score = dist_sq - (R[digits] ** 2)
            in_all.append((score <= 0).cpu())
        else:
            z_embed = rep
        z_all.append(z_embed.cpu())
        d_all.append(digits.cpu())
    Z = torch.cat(z_all).numpy()
    D = torch.cat(d_all).numpy()
    INS = torch.cat(in_all).numpy() if in_all else None

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
        m = D == k
        if not np.any(m):
            continue
        if INS is None:
            plt.scatter(Z2[m, 0], Z2[m, 1], s=8, marker="o", color=colors[k])
        else:
            m_in = m & INS
            m_out = m & (~INS)
            if np.any(m_in):
                plt.scatter(Z2[m_in, 0], Z2[m_in, 1], s=8, marker="o", color=colors[k])
            if np.any(m_out):
                plt.scatter(Z2[m_out, 0], Z2[m_out, 1], s=8, marker="x", color=colors[k])
    leg_digits = [Line2D([0], [0], marker="o", color=colors[k], linestyle="None", markersize=6, label=f"digit {k}") for k in range(10)]
    ax = plt.gca()
    if INS is None:
        ax.legend(handles=leg_digits, loc="upper left", fontsize=8, frameon=True, ncol=2)
        plt.title("MNIST t-SNE of encoder representation (AE-pretrain)")
    else:
        leg_state = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="inside"),
            Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="outside"),
        ]
        l1 = plt.legend(handles=leg_state, loc="best", fontsize=9, frameon=True)
        l2 = ax.legend(handles=leg_digits, loc="upper left", fontsize=8, frameon=True, ncol=2)
        ax.add_artist(l1)
        plt.title("Hyperbolic multi-sphere t-SNE of z_self")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    plt.tight_layout()
    out_name = "tsne_hyp_zself_inside_outside.png" if embed_mode == "hyperbolic_zself" else "tsne_rep_digits_only.png"
    out = os.path.join(args.out_dir, out_name)
    plt.savefig(out, dpi=150)
    print("Saved:", out)


if __name__ == "__main__":
    main()

