import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from cifar10_data import CIFAR10RawDataset
from cifar_backbone import CifarConvAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD
from hyperbolic_ops import hyp_distance, proj_ball

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


def geodesic_mds_3d(z_h_np, c=1.0, seed=42):
    from sklearn.manifold import MDS

    n = z_h_np.shape[0]
    z_t = torch.from_numpy(z_h_np).float()
    dmat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi = z_t[i : i + 1].repeat(n, 1)
        dmat[i] = hyp_distance(xi, z_t, c=c).cpu().numpy()
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=seed, normalized_stress="auto")
    y = mds.fit_transform(dmat)
    m = np.max(np.linalg.norm(y, axis=1))
    if m > 0:
        y = 0.95 * y / m
    return y


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Interactive 3D Poincare-ball for CIFAR10 hyperbolic model")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_mode", type=str, default="auto", choices=["auto", "direct3d", "geodesic_mds_3d"])
    p.add_argument("--html_name", type=str, default="poincare_ball_3d_interactive.html")
    args = p.parse_args()

    import plotly.graph_objects as go
    import plotly.express as px

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    curvature = float(ckpt["curvature"])
    r = 1.0 / np.sqrt(curvature)

    backbone = CifarConvAE(rep_dim=int(ckpt["rep_dim"]))
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone,
        rep_dim=int(ckpt["rep_dim"]),
        z_dim=int(ckpt["z_dim"]),
        n_classes=10,
        c=curvature,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    c_h = ckpt["c_h"].to(device)
    R = ckpt["R"].to(device)

    ds = CIFAR10RawDataset(root=args.data_root, split="test", digits=list(range(10)), max_samples=args.max_samples, download=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    z_all, y_all, in_all = [], [], []
    for x, y, _idx in dl:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z = model.project_self_h(rep, y)
        d2 = hyp_distance(z, c_h[y], c=curvature) ** 2
        inside = d2 <= (R[y] ** 2)
        z_all.append(z.cpu())
        y_all.append(y.cpu())
        in_all.append(inside.cpu())
    Z = torch.cat(z_all).numpy()
    Y = torch.cat(y_all).numpy()
    INS = torch.cat(in_all).numpy()

    mode = args.embed_mode
    if mode == "auto":
        mode = "direct3d" if int(ckpt["z_dim"]) >= 3 else "geodesic_mds_3d"
    if mode == "direct3d":
        P = proj_ball(torch.from_numpy(Z[:, :3]).float(), c=curvature, eps=1e-4).numpy()
    else:
        P = geodesic_mds_3d(Z, c=curvature, seed=args.seed)

    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = r * np.outer(np.cos(u), np.sin(v))
    ys = r * np.outer(np.sin(u), np.sin(v))
    zs = r * np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(x=xs, y=ys, z=zs, opacity=0.08, showscale=False, colorscale=[[0, "#666"], [1, "#666"]], hoverinfo="skip")
    )
    palette = px.colors.qualitative.Plotly
    for k in range(10):
        m = Y == k
        if not np.any(m):
            continue
        color = palette[k % len(palette)]
        mi = m & INS
        mo = m & (~INS)
        if np.any(mi):
            fig.add_trace(
                go.Scatter3d(
                    x=P[mi, 0],
                    y=P[mi, 1],
                    z=P[mi, 2],
                    mode="markers",
                    marker=dict(size=3.2, color=color),
                    name=f"{CIFAR10_NAMES[k]} in",
                )
            )
        if np.any(mo):
            fig.add_trace(
                go.Scatter3d(
                    x=P[mo, 0],
                    y=P[mo, 1],
                    z=P[mo, 2],
                    mode="markers",
                    marker=dict(size=4.0, color=color, symbol="x"),
                    name=f"{CIFAR10_NAMES[k]} out",
                )
            )

    fig.update_layout(
        title=f"CIFAR10 Hyperbolic 3D Poincare ({mode})",
        scene=dict(
            xaxis=dict(range=[-1.05 * r, 1.05 * r]),
            yaxis=dict(range=[-1.05 * r, 1.05 * r]),
            zaxis=dict(range=[-1.05 * r, 1.05 * r]),
            aspectmode="cube",
        ),
    )
    out = os.path.join(args.out_dir, args.html_name)
    fig.write_html(out, include_plotlyjs="cdn")
    print("Saved:", out)


if __name__ == "__main__":
    main()

