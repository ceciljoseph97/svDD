"""
One-shot CIFAR-10 hyperbolic multi-sphere visualization suite (hySpCheckcifar).

Combines (for each checkpoint that supports it):
  - t-SNE (same scoring as visualize_hyp_cifar_tsne: z_self + inside/out via soft-boundary margin)
  - Interactive 3D Poincaré ball HTML (visualize_hyp_cifar_poincare_3d_interactive logic)
  - In-class normal vs anomalous extremes (visualize_inclass_extremes_cifar)
  - Per-class neural maps: mean |∂(s_k)/∂x| for true-class margin s_k = d_H^2 - R_k^2, plus mean |encoder spatial activation|

AE-only checkpoints: t-SNE on rep + per-class encoder activation map only (no SVDD saliency / poincare / inclass).

Run from hySpCheckcifar:
  python visualize_hyp_cifar_suite.py --data_root data --device cuda

Select checkpoints explicitly:
  python visualize_hyp_cifar_suite.py --data_root data --checkpoints run_a=runs/foo/checkpoint_best.pth --skip_ae
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict
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

from cifar10_data import CIFAR10RawDataset
from cifar_backbone import CifarConvAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers
from hyperbolic_ops import hyp_distance, proj_ball
from visualize_hyp_cifar_poincare_3d_interactive import geodesic_mds_3d
from visualize_inclass_extremes_cifar import (
    CIFAR10_NAMES,
    _run_all_classes,
    _save_combined_grid_cifar,
    _save_row_png,
)

CIFAR_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def _default_cifar_data_root() -> Path | None:
    for p in (HERE / "data", HERE / "cifar_data", HERE.parent.parent / "CVAEChecked" / "Data" / "cifar_data"):
        p = p.resolve()
        if (p / "cifar-10-batches-py").is_dir():
            return p
    return None


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Returns (model, c_h_or_none, R_or_none, has_svdd, rep_dim, z_dim, curvature, objective)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    has_svdd = all(k in ckpt for k in ("rep_dim", "z_dim", "curvature", "c_h", "R"))
    ms = ckpt["model_state"]
    if has_svdd:
        rep_dim = int(ckpt["rep_dim"])
        z_dim = int(ckpt["z_dim"])
        curv = float(ckpt["curvature"])
        backbone = CifarConvAE(rep_dim=rep_dim)
        model = HyperbolicMultiSphereSVDD(
            backbone=backbone, rep_dim=rep_dim, z_dim=z_dim, n_classes=10, c=curv
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

    rep_dim = int(ckpt.get("rep_dim", 128))
    z_dim = None
    for k, v in ms.items():
        if k.endswith("proj_heads.0.weight"):
            z_dim = int(v.shape[0])
            rep_dim = int(v.shape[1])
            break
    if z_dim is None:
        if "backbone.fc_enc.weight" in ms:
            rep_dim = int(ms["backbone.fc_enc.weight"].shape[0])
        elif "fc_enc.weight" in ms:
            rep_dim = int(ms["fc_enc.weight"].shape[0])
        z_dim = int(ckpt.get("z_dim", 32))

    backbone = CifarConvAE(rep_dim=rep_dim)
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone, rep_dim=rep_dim, z_dim=z_dim, n_classes=10, c=float(ckpt.get("curvature", 1.0))
    ).to(device)
    model.load_state_dict(ms, strict=True)
    model.eval()
    return model, None, None, False, rep_dim, z_dim, None, None


@torch.no_grad()
def run_tsne(
    model,
    c_h,
    R,
    has_svdd: bool,
    objective: str | None,
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

    z_all, y_all, in_all = [], [], []
    for x, y, _i in dl:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        if has_svdd and c_h is not None and R is not None:
            z_embed = model.project_self_h(rep, y)
            z_all_h = model.project_all_h(rep)
            d2_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=model.curvature)
            if objective == "soft-boundary":
                s_all = d2_all - (R.unsqueeze(0) ** 2)
                inside = s_all[torch.arange(s_all.size(0), device=s_all.device), y] <= 0
            else:
                inside = torch.ones((y.size(0),), dtype=torch.bool, device=y.device)
            in_all.append(inside.cpu())
        else:
            z_embed = rep
        z_all.append(z_embed.cpu())
        y_all.append(y.cpu())

    Z = torch.cat(z_all).detach().numpy()
    Y = torch.cat(y_all).detach().numpy()
    INS = torch.cat(in_all).detach().numpy() if in_all else None

    n_eff = Z.shape[0]
    perp = float(min(perplexity, max(5.0, 0.99 * max(n_eff - 1, 1))))
    n_iter_eff = max(250, int(n_iter))
    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp, n_iter=n_iter_eff)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp, max_iter=n_iter_eff)
    Z2 = tsne.fit_transform(Z)

    plt.figure(figsize=(10, 8))
    for k in range(10):
        m = Y == k
        if not np.any(m):
            continue
        if INS is None:
            plt.scatter(Z2[m, 0], Z2[m, 1], s=8, marker="o", color=CIFAR_COLORS[k])
        else:
            mi = m & INS
            mo = m & (~INS)
            if np.any(mi):
                plt.scatter(Z2[mi, 0], Z2[mi, 1], s=8, marker="o", color=CIFAR_COLORS[k])
            if np.any(mo):
                plt.scatter(Z2[mo, 0], Z2[mo, 1], s=8, marker="x", color=CIFAR_COLORS[k])
    leg_names = [
        Line2D([0], [0], marker="o", color=CIFAR_COLORS[k], linestyle="None", markersize=6, label=CIFAR10_NAMES[k])
        for k in range(10)
    ]
    ax = plt.gca()
    if INS is None:
        ax.legend(handles=leg_names, loc="upper left", fontsize=8, ncol=2)
        plt.title("t-SNE encoder rep (AE checkpoint)")
    else:
        leg_state = [
            Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="inside"),
            Line2D([0], [0], marker="x", color="gray", linestyle="None", markersize=6, label="outside"),
        ]
        l1 = plt.legend(handles=leg_state, loc="lower left", fontsize=9)
        l2 = ax.legend(handles=leg_names, loc="upper left", fontsize=8, ncol=2)
        ax.add_artist(l1)
        plt.title("t-SNE z_self (CIFAR hyperbolic multi-sphere)")
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
    base_ds: CIFAR10RawDataset,
    device: torch.device,
    out_html: str,
    max_samples: int,
    batch_size: int,
    n_workers: int,
    seed: int,
):
    import plotly.express as px
    import plotly.graph_objects as go

    model.eval()
    r = 1.0 / np.sqrt(curvature)
    n_take = min(max_samples, len(base_ds))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(base_ds), size=n_take, replace=False).tolist()
    dl = DataLoader(Subset(base_ds, idxs), batch_size=batch_size, shuffle=False, num_workers=n_workers)

    z_list, y_list, d2_list = [], [], []
    for x, y, _ in dl:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z_self_h = model.project_self_h(rep, y)
        d2 = hyp_distance(z_self_h, c_h[y], c=curvature) ** 2
        z_list.append(z_self_h.cpu())
        y_list.append(y.cpu())
        d2_list.append(d2.cpu())

    Z_h = torch.cat(z_list).numpy()
    Y = torch.cat(y_list).numpy()
    d2_all = torch.cat(d2_list).numpy()
    R_np = R.detach().cpu().numpy().astype(np.float64)
    INS = d2_all <= (R_np[Y] ** 2)

    mode = "direct3d" if z_dim >= 3 else "geodesic_mds_3d"
    if mode == "direct3d":
        Yp = Z_h[:, :3].copy()
        Yp = proj_ball(torch.from_numpy(Yp).float(), c=curvature, eps=1e-4).numpy()
    else:
        Yp = geodesic_mds_3d(Z_h, c=curvature, seed=seed)

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
        m = Y == k
        if not np.any(m):
            continue
        mi, mo = m & INS, m & (~INS)
        color = palette[k % len(palette)]
        name = CIFAR10_NAMES[k]
        if np.any(mi):
            fig.add_trace(
                go.Scatter3d(
                    x=Yp[mi, 0],
                    y=Yp[mi, 1],
                    z=Yp[mi, 2],
                    mode="markers",
                    marker=dict(size=3.5, color=color, symbol="circle"),
                    name=f"{name} in",
                    legendgroup=f"c{k}",
                )
            )
        if np.any(mo):
            fig.add_trace(
                go.Scatter3d(
                    x=Yp[mo, 0],
                    y=Yp[mo, 1],
                    z=Yp[mo, 2],
                    mode="markers",
                    marker=dict(size=4.0, color=color, symbol="x"),
                    name=f"{name} out",
                    legendgroup=f"c{k}",
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


def _encoder_spatial_activation_map_cifar(model: HyperbolicMultiSphereSVDD | CifarConvAE, x: torch.Tensor) -> torch.Tensor:
    bb = model.backbone if hasattr(model, "backbone") else model
    h = bb.encoder(x)
    g = h.abs().mean(dim=1, keepdim=True)
    g = F.interpolate(g, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    return g.squeeze(1)


def _saliency_true_class_margin(
    model: HyperbolicMultiSphereSVDD, c_h: torch.Tensor, R: torch.Tensor, curvature: float, cls_k: int, x1: torch.Tensor
) -> torch.Tensor:
    x = x1.clone().detach().requires_grad_(True)
    rep, _ = model(x)
    z_all = model.project_all_h(rep)
    z_k = z_all[:, cls_k, :]
    ck = c_h[cls_k : cls_k + 1].expand_as(z_k)
    d2 = hyp_distance(z_k, ck, c=curvature) ** 2
    margin = (d2 - (R[cls_k] ** 2)).sum()
    gx = torch.autograd.grad(margin, x, retain_graph=False)[0]
    return gx.abs().mean(dim=1).squeeze(0).detach()


def _saliency_recon(model: CifarConvAE, x1: torch.Tensor) -> torch.Tensor:
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
    samples_per_class: int,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    by_c: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ds)):
        _x, d_t, _ = ds[i]
        by_c[int(d_t.item())].append(i)
    picked: dict[int, list[int]] = {}
    for k in range(10):
        idxs = by_c[k]
        if len(idxs) > samples_per_class:
            sub = rng.choice(len(idxs), samples_per_class, replace=False)
            picked[k] = [idxs[i] for i in sub]
        else:
            picked[k] = list(idxs)

    meta = {"samples_per_class": samples_per_class, "has_svdd": has_svdd, "classes": {}}
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for k in range(10):
        sal_stack, act_stack = [], []
        for idx in picked[k]:
            x, _d, _ = ds[idx]
            x1 = x.unsqueeze(0).to(device)
            with torch.no_grad():
                act_stack.append(_encoder_spatial_activation_map_cifar(model, x1).squeeze(0).cpu().numpy())
            with torch.enable_grad():
                if has_svdd:
                    sal_stack.append(_saliency_true_class_margin(model, c_h, R, curvature, k, x1).cpu().numpy())
                else:
                    sal_stack.append(_saliency_recon(model.backbone if hasattr(model, "backbone") else model, x1).cpu().numpy())
        if not sal_stack:
            continue

        def _n01(a):
            a = np.asarray(a, dtype=np.float64)
            return (a - a.min()) / (a.max() - a.min() + 1e-12)

        sm = np.mean(np.stack(sal_stack), axis=0)
        am = np.mean(np.stack(act_stack), axis=0)
        cache[k] = (_n01(sm), _n01(am))
        meta["classes"][str(k)] = {"n": len(sal_stack), "name": CIFAR10_NAMES[k]}

    for k, (sn, an) in cache.items():
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
        t0 = "|∂s_k/∂x| mean" if has_svdd else "|∂recon_mse/∂x| mean"
        axes[0].imshow(sn, cmap="magma", vmin=0, vmax=1)
        axes[0].set_title(t0)
        axes[0].axis("off")
        axes[1].imshow(an, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("mean |encoder act|")
        axes[1].axis("off")
        fig.suptitle(f"{CIFAR10_NAMES[k]} (class {k}, n={meta['classes'][str(k)]['n']})")
        plt.tight_layout()
        p = os.path.join(out_dir, f"class_{k}_neural_activation.png")
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
        axes[k, 0].set_ylabel(f"{k}", fontsize=9)
        axes[k, 0].axis("off")
        axes[k, 1].imshow(an, cmap="viridis", vmin=0, vmax=1)
        axes[k, 1].axis("off")
    axes[0, 0].set_title("saliency", fontsize=9)
    axes[0, 1].set_title("encoder", fontsize=9)
    plt.suptitle("Neural activation overview (per class)", fontsize=12)
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
    data_root: str,
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

    base = CIFAR10RawDataset(
        root=data_root,
        split=args.split,
        digits=list(range(10)),
        max_samples=None,
        download=not args.no_download,
    )

    run_tsne(
        model,
        c_h,
        R,
        has_svdd,
        objective,
        base,
        device,
        str(out_dir / "tsne_suite.png"),
        args.max_samples_tsne,
        args.batch_size,
        args.n_jobs_dataloader,
        args.seed,
        args.perplexity,
        args.tsne_iter,
    )

    if has_svdd and c_h is not None and R is not None and z_dim is not None and curvature is not None:
        run_poincare_html(
            model,
            c_h,
            R,
            float(curvature),
            int(z_dim),
            base,
            device,
            str(out_dir / "poincare_3d_interactive.html"),
            args.max_samples_poincare,
            args.batch_size,
            args.n_jobs_dataloader,
            args.seed,
        )
        best_img, worst_img, best_s, worst_s = _run_all_classes(
            data_root,
            str(ckpt_path),
            device,
            args.batch_size,
            args.n_jobs_dataloader,
            args.max_test_samples_inclass,
            not args.no_download,
        )
        inc_dir = out_dir / "inclass_extremes"
        inc_dir.mkdir(exist_ok=True)
        for k in range(10):
            op = inc_dir / f"inclass_extremes_cifar_k{k}.png"
            _save_row_png(best_img[k], worst_img[k], float(best_s[k]), float(worst_s[k]), k, str(op), args.dpi)
        _save_combined_grid_cifar(
            best_img,
            worst_img,
            best_s,
            worst_s,
            str(inc_dir / "inclass_extremes_cifar_combined.png"),
            args.dpi,
        )
        print(f"[suite] in-class extremes -> {inc_dir}")

    run_neural_activation_maps(
        model,
        c_h,
        R,
        has_svdd,
        float(curvature) if has_svdd and curvature is not None else 1.0,
        base,
        device,
        str(out_dir / "neural_activation"),
        args.samples_per_class_activation,
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
    dr = _default_cifar_data_root()
    p = argparse.ArgumentParser("Combined CIFAR-10 hyperbolic multi-sphere visualization suite")
    p.add_argument("--data_root", type=str, default=str(dr) if dr else None)
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(HERE / "runs" / "suite_cifar_multi_vis"),
        help="Subfolders per checkpoint tag are created here.",
    )
    p.add_argument(
        "--ae_checkpoint",
        type=str,
        default=str(HERE / "runs" / "hyp_cifar_cuda" / "ae_stage1.pth"),
        help="AE / stage-1 checkpoint (optional placeholder path).",
    )
    p.add_argument(
        "--svdd_checkpoint",
        type=str,
        default=str(HERE / "runs" / "hyp_cifar_cuda" / "checkpoint_best.pth"),
        help="Full hyperbolic multi-sphere checkpoint.",
    )
    p.add_argument("--skip_ae", action="store_true", help="Do not process --ae_checkpoint.")
    p.add_argument("--skip_svdd", action="store_true", help="Do not process --svdd_checkpoint.")
    p.add_argument(
        "--checkpoints",
        nargs="*",
        metavar="TAG=PATH",
        default=None,
        help="Explicit runs: TAG=PATH. Overrides default ae/svdd when non-empty.",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples_tsne", type=int, default=2000)
    p.add_argument("--max_samples_poincare", type=int, default=1200)
    p.add_argument("--max_test_samples_inclass", type=int, default=None)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--tsne_iter", type=int, default=1000)
    p.add_argument("--samples_per_class_activation", type=int, default=24)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--no_download", action="store_true", help="Do not download CIFAR if missing.")
    args = p.parse_args()

    if not args.data_root:
        p.error("Set --data_root (default path not found).")

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
            runs.append(("ae_stage1", args.ae_checkpoint))
        if not args.skip_svdd:
            runs.append(("svdd_checkpoint_best", args.svdd_checkpoint))

    if not runs:
        p.error("No checkpoints to run: use --checkpoints TAG=PATH ... or unset --skip_ae/--skip_svdd.")

    for tag, ck in runs:
        print(f"\n=== Suite: {tag} ===")
        run_one_checkpoint(tag, ck, args.data_root, out_root, device, args)

    print(f"\n[suite] Done. Outputs under {out_root.resolve()}")


if __name__ == "__main__":
    main()
