"""
One-shot MNIST hyperbolic multi-sphere visualization suite (hySpCheck).

Combines (for each checkpoint that supports it):
  - t-SNE (visualize_hyp_mnist_multi logic: z_self + in/out or AE rep)
  - Interactive 3D Poincaré ball HTML (visualize_hyp_poincare_3d_interactive)
  - In-class normal vs anomalous extremes (visualize_inclass_extremes_mnist)
  - Per-digit neural maps: mean |∂(s_k)/∂x| for true-class margin s_k = d_H^2 - R_k^2, plus mean |conv2 post-ReLU| (hySpUnsup-style)

AE-only checkpoints: t-SNE on rep + per-digit conv2 activation only (no SVDD saliency / poincare / inclass).

Run from hySpCheck:
  python visualize_hyp_mnist_suite.py --device cuda

Select checkpoints explicitly (overrides default ae/svdd pair):
  python visualize_hyp_mnist_suite.py --checkpoints svdd_best=runs/.../checkpoint_best.pth --skip_ae
  python visualize_hyp_mnist_suite.py --checkpoints run_a=path/to/a.pth run_b=path/to/b.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from collections import OrderedDict

from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, hyp_dist_sq_to_centers
from hyperbolic_ops import hyp_distance, proj_ball
from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE, preprocess_batch_by_digit_minmax

# Optional: in-class extremes
try:
    from visualize_hyp_poincare_3d_interactive import geodesic_mds_3d
except ImportError:
    geodesic_mds_3d = None

try:
    from visualize_inclass_extremes_mnist import (
        _run_all_classes,
        _save_combined_grid_mnist,
        _save_row_png,
    )
except ImportError:
    _run_all_classes = None
    _save_combined_grid_mnist = None
    _save_row_png = None


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


def _default_mnist_dir() -> Path | None:
    cand = HERE.parent.parent / "CVAEChecked" / "Data" / "MNIST_processed"
    return cand if cand.is_dir() else None


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Returns (model, c_h_or_none, R_or_none, has_svdd, rep_dim, z_dim, curvature, objective)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    has_svdd = all(k in ckpt for k in ("rep_dim", "z_dim", "curvature", "c_h", "R"))
    ms = ckpt["model_state"]
    if has_svdd:
        rep_dim = int(ckpt["rep_dim"])
        z_dim = int(ckpt["z_dim"])
        curv = float(ckpt["curvature"])
        backbone = MNIST_LeNet_SVDDIAE(rep_dim=rep_dim)
        model = HyperbolicMultiSphereSVDD(
            backbone=backbone, rep_dim=rep_dim, z_dim=z_dim, n_digits=10, c=curv
        ).to(device)
        model.load_state_dict(ms, strict=True)
        model.eval()
        c_h = ckpt["c_h"].to(device)
        R = ckpt["R"].to(device)
        objective = ckpt.get("objective", "soft-boundary")
        if torch.all(R <= 1e-12):
            latest = Path(checkpoint_path).resolve().parent / "checkpoint_latest.pth"
            if latest.is_file():
                ck2 = torch.load(latest, map_location="cpu")
                if "R" in ck2 and "c_h" in ck2:
                    model.load_state_dict(ck2["model_state"], strict=True)
                    c_h = ck2["c_h"].to(device)
                    R = ck2["R"].to(device)
                    print(f"[suite] {checkpoint_path}: degenerate R; using {latest.name} for viz.")
        return model, c_h, R, True, rep_dim, z_dim, curv, objective

    rep_dim = int(ckpt.get("rep_dim", 32))
    if "backbone.fc1.weight" in ms:
        rep_dim = int(ms["backbone.fc1.weight"].shape[0])
    elif "fc1.weight" in ms:
        rep_dim = int(ms["fc1.weight"].shape[0])
    model = MNIST_LeNet_SVDDIAE(rep_dim=rep_dim).to(device)
    if any(k.startswith("backbone.") for k in ms.keys()):
        stripped = OrderedDict((k[len("backbone.") :], v) for k, v in ms.items() if k.startswith("backbone."))
        model.load_state_dict(stripped, strict=False)
    else:
        model.load_state_dict(ms, strict=False)
    model.eval()
    return model, None, None, False, rep_dim, None, None, None


@torch.no_grad()
def run_tsne(
    model,
    c_h,
    R,
    has_svdd: bool,
    ds: Dataset,
    device: torch.device,
    out_path: str,
    max_samples: int,
    batch_size: int,
    n_workers: int,
    seed: int,
    perplexity: float,
    n_iter: int,
):
    model.eval()
    n = min(max_samples, len(ds))
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(ds), size=n, replace=False).tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=n_workers)

    z_all, d_all, in_all = [], [], []
    for x_scaled, digits, _xr, _i in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        if has_svdd:
            z_embed = model.project_self_h(rep, digits)
            c_b = c_h[digits]
            dist_sq = torch.sum((z_embed - c_b) ** 2, dim=1)
            score = dist_sq - (R[digits] ** 2)
            in_all.append((score <= 0).cpu())
        else:
            z_embed = rep
        z_all.append(z_embed.cpu())
        d_all.append(digits.cpu())

    Z = torch.cat(z_all).detach().numpy()
    D = torch.cat(d_all).detach().numpy()
    INS = torch.cat(in_all).detach().numpy() if in_all else None

    n_eff = Z.shape[0]
    perp = float(min(perplexity, max(5.0, 0.99 * max(n_eff - 1, 1))))
    n_iter_eff = max(250, int(n_iter))
    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp, n_iter=n_iter_eff)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp, max_iter=n_iter_eff)
    Z2 = tsne.fit_transform(Z)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    plt.figure(figsize=(10, 8))
    for k in range(10):
        m = D == k
        if not np.any(m):
            continue
        if INS is None:
            plt.scatter(Z2[m, 0], Z2[m, 1], s=8, marker="o", color=colors[k])
        else:
            mi = m & INS
            mo = m & (~INS)
            if np.any(mi):
                plt.scatter(Z2[mi, 0], Z2[mi, 1], s=8, marker="o", color=colors[k])
            if np.any(mo):
                plt.scatter(Z2[mo, 0], Z2[mo, 1], s=8, marker="x", color=colors[k])
    leg_digits = [Line2D([0], [0], marker="o", color=colors[k], linestyle="None", markersize=6, label=f"{k}") for k in range(10)]
    ax = plt.gca()
    if INS is None:
        ax.legend(handles=leg_digits, loc="upper left", fontsize=8, ncol=2)
        plt.title("t-SNE encoder rep (AE checkpoint)")
    else:
        leg_state = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="inside"),
            Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="outside"),
        ]
        l1 = plt.legend(handles=leg_state, loc="best", fontsize=9)
        l2 = ax.legend(handles=leg_digits, loc="upper left", fontsize=8, ncol=2)
        ax.add_artist(l1)
        plt.title("t-SNE z_self (hyperbolic multi-sphere)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[suite] t-SNE -> {out_path}")


@torch.no_grad()
def run_poincare_html(
    model: HyperbolicMultiSphereSVDD,
    c_h: torch.Tensor,
    R: torch.Tensor,
    curvature: float,
    z_dim: int,
    base_ds,
    device: torch.device,
    out_html: str,
    max_samples: int,
    batch_size: int,
    n_workers: int,
    seed: int,
):
    if geodesic_mds_3d is None:
        print("[suite] SKIP Poincare 3D: missing visualize_hyp_poincare_3d_interactive module.")
        return

    import plotly.express as px
    import plotly.graph_objects as go

    model.eval()
    r = 1.0 / np.sqrt(curvature)
    n_take = min(max_samples, len(base_ds))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(base_ds), size=n_take, replace=False).tolist()
    dl = DataLoader(Subset(ScaledRawDataset(base_ds), idxs), batch_size=batch_size, shuffle=False, num_workers=n_workers)

    z_list, d_list, d2_list = [], [], []
    for x_scaled, digits, *_ in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_self_h = model.project_self_h(rep, digits)
        d2 = hyp_distance(z_self_h, c_h[digits], c=curvature) ** 2
        z_list.append(z_self_h.cpu())
        d_list.append(digits.cpu())
        d2_list.append(d2.cpu())

    Z_h = torch.cat(z_list).numpy()
    D = torch.cat(d_list).numpy()
    d2_all = torch.cat(d2_list).numpy()
    R_np = R.detach().cpu().numpy().astype(np.float64)
    INS = d2_all <= (R_np[D] ** 2)

    mode = "direct3d" if z_dim >= 3 else "geodesic_mds_3d"
    if mode == "direct3d":
        Y = Z_h[:, :3].copy()
        Y = proj_ball(torch.from_numpy(Y).float(), c=curvature, eps=1e-4).numpy()
    else:
        Y = geodesic_mds_3d(Z_h, c=curvature, seed=seed)

    fig = go.Figure()
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
            name="ball",
            hoverinfo="skip",
        )
    )
    palette = px.colors.qualitative.Plotly
    for k in range(10):
        m = D == k
        if not np.any(m):
            continue
        mi, mo = m & INS, m & (~INS)
        color = palette[k % len(palette)]
        if np.any(mi):
            fig.add_trace(
                go.Scatter3d(
                    x=Y[mi, 0],
                    y=Y[mi, 1],
                    z=Y[mi, 2],
                    mode="markers",
                    marker=dict(size=3.5, color=color, symbol="circle"),
                    name=f"{k} in",
                    legendgroup=f"d{k}",
                )
            )
        if np.any(mo):
            fig.add_trace(
                go.Scatter3d(
                    x=Y[mo, 0],
                    y=Y[mo, 1],
                    z=Y[mo, 2],
                    mode="markers",
                    marker=dict(size=4.0, color=color, symbol="x"),
                    name=f"{k} out",
                    legendgroup=f"d{k}",
                )
            )
    fig.update_layout(
        title=f"Poincaré 3D ({mode}), c={curvature}",
        scene=dict(
            xaxis=dict(range=[-1.05 * r, 1.05 * r]),
            yaxis=dict(range=[-1.05 * r, 1.05 * r]),
            zaxis=dict(range=[-1.05 * r, 1.05 * r]),
            aspectmode="cube",
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_html)) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[suite] Poincaré HTML -> {out_html}")


def _encoder_conv2_mean_activation_map(model: HyperbolicMultiSphereSVDD | MNIST_LeNet_SVDDIAE, x_scaled: torch.Tensor) -> torch.Tensor:
    bb = model.backbone if hasattr(model, "backbone") else model
    x = bb.conv1(x_scaled)
    x = bb.pool(F.leaky_relu(bb.bn1(x)))
    x = bb.conv2(x)
    x = F.leaky_relu(bb.bn2(x))
    g = x.abs().mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(x_scaled.shape[-2], x_scaled.shape[-1]), mode="bilinear", align_corners=False)
    return g.squeeze(1)


def _saliency_true_class_margin(
    model: HyperbolicMultiSphereSVDD, c_h: torch.Tensor, R: torch.Tensor, curvature: float, digit_k: int, x1: torch.Tensor
) -> torch.Tensor:
    """|∂(d_h^2(z_k,c_k)-R_k^2)/∂x| mean over channels — true class k (supervised)."""
    x = x1.clone().detach().requires_grad_(True)
    rep, _ = model(x)
    z_all = model.project_all_h(rep)
    z_k = z_all[:, digit_k, :]
    ck = c_h[digit_k : digit_k + 1].expand_as(z_k)
    d2 = hyp_dist_sq_to_centers(z_k, ck, curvature)
    margin = (d2 - (R[digit_k] ** 2)).sum()
    gx = torch.autograd.grad(margin, x, retain_graph=False)[0]
    return gx.abs().mean(dim=1).squeeze(0).detach()


def _saliency_recon(model: MNIST_LeNet_SVDDIAE, x1: torch.Tensor) -> torch.Tensor:
    x = x1.clone().detach().requires_grad_(True)
    _rep, recon = model(x)
    loss = F.mse_loss(recon, x, reduction="sum")
    gx = torch.autograd.grad(loss, x)[0]
    return gx.abs().mean(dim=1).squeeze(0).detach()


def run_neural_activation_maps(
    model,
    c_h,
    R,
    has_svdd: bool,
    curvature: float,
    ds: Dataset,
    device: torch.device,
    out_dir: str,
    samples_per_digit: int,
    seed: int,
):
    """Mean saliency + conv2 maps per digit (test split, balanced subsample)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    by_d: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        _, d_t, *_ = ds[i]
        by_d[int(d_t.item())].append(i)
    picked: dict[int, list[int]] = {}
    for k in range(10):
        idxs = by_d[k]
        if len(idxs) > samples_per_digit:
            sub = rng.choice(len(idxs), samples_per_digit, replace=False)
            picked[k] = [idxs[i] for i in sub]
        else:
            picked[k] = list(idxs)

    meta = {"samples_per_digit": samples_per_digit, "has_svdd": has_svdd, "digits": {}}
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k in range(10):
        sal_stack, act_stack = [], []
        for idx in picked[k]:
            x_scaled, _d, *_ = ds[idx]
            x1 = x_scaled.unsqueeze(0).to(device)
            with torch.no_grad():
                act_stack.append(_encoder_conv2_mean_activation_map(model, x1).squeeze(0).cpu().numpy())
            with torch.enable_grad():
                if has_svdd:
                    sal_stack.append(_saliency_true_class_margin(model, c_h, R, curvature, k, x1).cpu().numpy())
                else:
                    sal_stack.append(_saliency_recon(model, x1).cpu().numpy())
        if not sal_stack:
            continue
        sm = np.mean(np.stack(sal_stack), axis=0)
        am = np.mean(np.stack(act_stack), axis=0)
        def _n01(a):
            a = np.asarray(a, dtype=np.float64)
            return (a - a.min()) / (a.max() - a.min() + 1e-12)

        cache[k] = (_n01(sm), _n01(am))
        meta["digits"][str(k)] = {"n": len(sal_stack)}

    for k, (sn, an) in cache.items():
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
        t0 = "|∂s_k/∂x| mean" if has_svdd else "|∂recon_mse/∂x| mean"
        axes[0].imshow(sn, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title(t0)
        axes[0].axis("off")
        axes[1].imshow(an, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("mean |conv2 act|")
        axes[1].axis("off")
        fig.suptitle(f"digit {k} (n={meta['digits'][str(k)]['n']})")
        plt.tight_layout()
        p = os.path.join(out_dir, f"digit_{k}_neural_activation.png")
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"[suite] {p}")

    fig, axes = plt.subplots(10, 2, figsize=(5.2, 22))
    for k in range(10):
        if k not in cache:
            for j in range(2):
                axes[k, j].axis("off")
            continue
        sn, an = cache[k]
        axes[k, 0].imshow(sn, cmap="magma", vmin=0, vmax=1)
        axes[k, 0].set_ylabel(str(k), fontsize=9)
        axes[k, 0].axis("off")
        axes[k, 1].imshow(an, cmap="viridis", vmin=0, vmax=1)
        axes[k, 1].axis("off")
    axes[0, 0].set_title("saliency", fontsize=9)
    axes[0, 1].set_title("conv2", fontsize=9)
    plt.suptitle("Neural activation overview (per digit)", fontsize=12)
    plt.tight_layout()
    ov = os.path.join(out_dir, "neural_activation_overview.png")
    plt.savefig(ov, dpi=130, bbox_inches="tight")
    plt.close()
    with open(os.path.join(out_dir, "neural_activation_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[suite] {ov}")


def run_one_checkpoint(
    tag: str,
    checkpoint_path: str,
    mnist_processed_dir: str,
    out_root: Path,
    device: torch.device,
    args: argparse.Namespace,
):
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        print(f"[suite] SKIP {tag}: missing file {checkpoint_path}")
        return
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model, c_h, R, has_svdd, rep_dim, z_dim, curvature, objective = load_model_from_checkpoint(str(ckpt_path), device)
    model.eval()

    base = MNISTDigitsProcessedRawDataset(
        root_dir=mnist_processed_dir,
        split=args.split,
        train_fraction=args.train_fraction,
        digits=list(range(10)),
    )
    ds = ScaledRawDataset(base)

    # t-SNE
    run_tsne(
        model,
        c_h,
        R,
        has_svdd,
        ds,
        device,
        str(out_dir / "tsne_suite.png"),
        args.max_samples_tsne,
        args.batch_size,
        args.n_jobs_dataloader,
        args.seed,
        args.perplexity,
        args.tsne_iter,
    )

    if has_svdd:
        run_poincare_html(
            model,
            c_h,
            R,
            curvature,
            z_dim,
            base,
            device,
            str(out_dir / "poincare_3d_interactive.html"),
            args.max_samples_poincare,
            args.batch_size,
            args.n_jobs_dataloader,
            args.seed,
        )
        if _run_all_classes is None or _save_combined_grid_mnist is None or _save_row_png is None:
            print("[suite] SKIP in-class extremes: missing visualize_inclass_extremes_mnist module.")
        else:
            best_img, worst_img, best_s, worst_s = _run_all_classes(
                mnist_processed_dir,
                str(ckpt_path),
                device,
                args.train_fraction,
                args.split,
                args.batch_size,
                args.n_jobs_dataloader,
            )
            inc_dir = out_dir / "inclass_extremes"
            inc_dir.mkdir(exist_ok=True)
            for k in range(10):
                op = inc_dir / f"inclass_extremes_mnist_k{k}.png"
                _save_row_png(best_img[k], worst_img[k], float(best_s[k]), float(worst_s[k]), k, str(op), args.dpi)
            _save_combined_grid_mnist(
                best_img,
                worst_img,
                best_s,
                worst_s,
                str(inc_dir / "inclass_extremes_mnist_combined.png"),
                args.dpi,
            )
            print(f"[suite] in-class extremes -> {inc_dir}")

    run_neural_activation_maps(
        model,
        c_h,
        R,
        has_svdd,
        float(curvature) if has_svdd else 1.0,
        ds,
        device,
        str(out_dir / "neural_activation"),
        args.samples_per_digit_activation,
        args.seed,
    )

    with open(out_dir / "suite_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tag": tag,
                "checkpoint": str(ckpt_path.resolve()),
                "has_svdd": has_svdd,
                "rep_dim": rep_dim,
                "z_dim": z_dim,
                "curvature": curvature,
                "objective": objective,
            },
            f,
            indent=2,
        )


def main():
    p = argparse.ArgumentParser("Combined MNIST hyperbolic multi-sphere visualization suite")
    md = _default_mnist_dir()
    p.add_argument("--mnist_processed_dir", type=str, default=str(md) if md else None)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(HERE / "runs" / "suite_mnist_multi_vis"),
        help="Subfolders per checkpoint tag are created here.",
    )
    p.add_argument(
        "--ae_checkpoint",
        type=str,
        default=str(HERE / "runs" / "ablate_mnist_full_t2_ae150" / "ae_stage1_ep150.pth"),
        help="AE / stage-1 checkpoint (optional).",
    )
    p.add_argument(
        "--svdd_checkpoint",
        type=str,
        default=str(HERE / "runs" / "ablate_mnist_full_t2_from_ae150" / "checkpoint_best.pth"),
        help="Full hyperbolic multi-sphere checkpoint.",
    )
    p.add_argument("--skip_ae", action="store_true", help="Do not process --ae_checkpoint.")
    p.add_argument("--skip_svdd", action="store_true", help="Do not process --svdd_checkpoint.")
    p.add_argument(
        "--checkpoints",
        nargs="*",
        metavar="TAG=PATH",
        default=None,
        help="Explicit runs: one or more TAG=PATH (split on first '='). Overrides --ae/--svdd when non-empty. "
        "Example: --checkpoints my_svdd=runs/foo/checkpoint_best.pth",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples_tsne", type=int, default=2000)
    p.add_argument("--max_samples_poincare", type=int, default=1200)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--samples_per_digit_activation", type=int, default=32)
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    if not args.mnist_processed_dir:
        p.error("Set --mnist_processed_dir (default path not found).")

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, str]] = []
    if args.checkpoints:
        for spec in args.checkpoints:
            if "=" not in spec:
                p.error(f"--checkpoints entry must be TAG=PATH, got: {spec!r}")
            tag, path = spec.split("=", 1)
            tag = tag.strip()
            path = path.strip().strip('"').strip("'")
            if not tag or not path:
                p.error(f"Invalid --checkpoints entry: {spec!r}")
            runs.append((tag, path))
    else:
        if not args.skip_ae:
            runs.append(("ae_stage1_ep150", args.ae_checkpoint))
        if not args.skip_svdd:
            runs.append(("svdd_checkpoint_best", args.svdd_checkpoint))

    if not runs:
        p.error("No checkpoints to run: use --checkpoints TAG=PATH ... or unset --skip_ae/--skip_svdd.")

    for tag, ck in runs:
        print(f"\n=== Suite: {tag} ===")
        run_one_checkpoint(tag, ck, args.mnist_processed_dir, out_root, device, args)

    print(f"\n[suite] Done. Outputs under {out_root.resolve()}")


if __name__ == "__main__":
    main()
