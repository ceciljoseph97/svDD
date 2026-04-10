import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from mnist_local import MNISTDigitsProcessedRawDataset, preprocess_batch_by_digit_minmax, MNIST_LeNet_SVDDIAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD
from hyperbolic_ops import hyp_distance, proj_ball


class ScaledRawDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x_raw, d, idx = self.base[i]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = preprocess_batch_by_digit_minmax(x_raw.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
        return x_scaled, d_t, idx


def geodesic_mds_3d(z_h_np, c=1.0, seed=42):
    from sklearn.manifold import MDS

    n = z_h_np.shape[0]
    z_t = torch.from_numpy(z_h_np).float()
    dmat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi = z_t[i : i + 1].repeat(n, 1)
        drow = hyp_distance(xi, z_t, c=c).cpu().numpy()
        dmat[i] = drow

    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=seed, normalized_stress="auto")
    y = mds.fit_transform(dmat)
    max_norm = np.max(np.linalg.norm(y, axis=1))
    if max_norm > 0:
        y = 0.95 * y / max_norm
    return y


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Interactive 3D Poincare-ball visualization (HTML)")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_mode", type=str, default="auto", choices=["auto", "direct3d", "geodesic_mds_3d"])
    p.add_argument("--use_r_vis", action="store_true", help="Use R_vis computed from sampled distances for inside/outside.")
    p.add_argument("--nu_vis", type=float, default=0.1, help="If --use_r_vis: quantile (1-nu_vis) defines R_vis per digit.")
    p.add_argument(
        "--inside_metric",
        type=str,
        default="auto",
        choices=["auto", "hyperbolic", "euclidean"],
        help="How to compute d^2 for inside/outside. auto=match MNIST training (hyperbolic).",
    )
    p.add_argument("--html_name", type=str, default="poincare_ball_3d_interactive.html")
    args = p.parse_args()

    import plotly.graph_objects as go
    import plotly.express as px

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt_path = args.checkpoint_path
    ckpt = torch.load(ckpt_path, map_location="cpu")

    rep_dim = int(ckpt["rep_dim"])
    z_dim = int(ckpt["z_dim"])
    curvature = float(ckpt["curvature"])
    r = 1.0 / np.sqrt(curvature)

    backbone = MNIST_LeNet_SVDDIAE(rep_dim=rep_dim)
    model = HyperbolicMultiSphereSVDD(backbone=backbone, rep_dim=rep_dim, z_dim=z_dim, n_digits=10, c=curvature).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    c_h = ckpt["c_h"].to(device)
    R = ckpt["R"].to(device)
    objective = ckpt.get("objective", "soft-boundary")

    # If best checkpoint has degenerate radii, try sibling latest checkpoint automatically.
    if torch.all(R <= 1e-12):
        latest_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "checkpoint_latest.pth")
        if os.path.isfile(latest_path):
            ckpt2 = torch.load(latest_path, map_location="cpu")
            if "model_state" in ckpt2 and "R" in ckpt2 and "c_h" in ckpt2:
                model.load_state_dict(ckpt2["model_state"], strict=True)
                c_h = ckpt2["c_h"].to(device)
                R = ckpt2["R"].to(device)
                objective = ckpt2.get("objective", objective)
                print(f"[info] checkpoint_best had degenerate R; using checkpoint_latest for visualization: {latest_path}")

    base = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir, split=args.split, train_fraction=args.train_fraction, digits=list(range(10))
    )
    n_take = min(args.max_samples, len(base))
    rng = np.random.RandomState(args.seed)
    idxs = rng.choice(len(base), size=n_take, replace=False).tolist()
    dl = DataLoader(Subset(ScaledRawDataset(base), idxs), batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    z_list, d_list, d2_list = [], [], []
    for x_scaled, digits, _idx in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_self_h = model.project_self_h(rep, digits)
        metric = args.inside_metric
        if metric == "auto":
            metric = "hyperbolic"
        if metric == "euclidean":
            d2 = torch.sum((z_self_h - c_h[digits]) ** 2, dim=1)
        else:
            d2 = hyp_distance(z_self_h, c_h[digits], c=curvature) ** 2
        z_list.append(z_self_h.cpu())
        d_list.append(digits.cpu())
        d2_list.append(d2.cpu())

    Z_h = torch.cat(z_list).numpy()
    D = torch.cat(d_list).numpy()
    d2_all = torch.cat(d2_list).numpy()

    # Inside/outside membership: CIFAR script behavior = checkpoint_R, unless user requests R_vis.
    if args.use_r_vis:
        R_vis = R.detach().cpu().numpy().astype(np.float64).copy()
        for k in range(10):
            mk = D == k
            if not np.any(mk):
                continue
            R_vis[k] = float(np.quantile(np.sqrt(d2_all[mk]), 1.0 - float(args.nu_vis)))
        INS = d2_all <= (R_vis[D] ** 2)
        inside_by = "R_vis"
    else:
        R_np = R.detach().cpu().numpy().astype(np.float64)
        INS = d2_all <= (R_np[D] ** 2)
        inside_by = "checkpoint_R"

    mode = args.embed_mode
    if mode == "auto":
        mode = "direct3d" if z_dim >= 3 else "geodesic_mds_3d"

    if mode == "direct3d":
        Y = Z_h[:, :3].copy()
        Y = proj_ball(torch.from_numpy(Y).float(), c=curvature, eps=1e-4).numpy()
    else:
        Y = geodesic_mds_3d(Z_h, c=curvature, seed=args.seed)

    fig = go.Figure()

    # sphere surface
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=zs,
            opacity=0.08,
            showscale=False,
            colorscale=[[0, "#666666"], [1, "#666666"]],
            name="Poincare ball boundary",
            hoverinfo="skip",
        )
    )

    palette = px.colors.qualitative.Plotly
    for k in range(10):
        m = D == k
        if not np.any(m):
            continue
        m_in = m & INS
        m_out = m & (~INS)
        color = palette[k % len(palette)]
        if np.any(m_in):
            fig.add_trace(
                go.Scatter3d(
                    x=Y[m_in, 0],
                    y=Y[m_in, 1],
                    z=Y[m_in, 2],
                    mode="markers",
                    marker=dict(size=3.5, color=color, symbol="circle"),
                    name=f"digit {k} inside",
                    legendgroup=f"digit{k}",
                )
            )
        if np.any(m_out):
            fig.add_trace(
                go.Scatter3d(
                    x=Y[m_out, 0],
                    y=Y[m_out, 1],
                    z=Y[m_out, 2],
                    mode="markers",
                    marker=dict(size=4.0, color=color, symbol="x"),
                    name=f"digit {k} outside",
                    legendgroup=f"digit{k}",
                )
            )

    fig.update_layout(
        title=f"Interactive 3D Poincare ball ({mode}), curvature={curvature}, inside_by={inside_by}, metric={metric}",
        scene=dict(
            xaxis=dict(range=[-1.05 * r, 1.05 * r], title="dim-1"),
            yaxis=dict(range=[-1.05 * r, 1.05 * r], title="dim-2"),
            zaxis=dict(range=[-1.05 * r, 1.05 * r], title="dim-3"),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"),
    )

    out_path = os.path.join(args.out_dir, args.html_name)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

