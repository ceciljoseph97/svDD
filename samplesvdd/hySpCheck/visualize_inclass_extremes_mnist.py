"""
MNIST: most normal (left) vs most anomalous (right) among in-class samples per digit k,
score s_k = d_H(z_k,c_k)^2 - R_k^2 (soft-boundary).

Use --all_classes to write one 1x2 PNG per digit (0..9) in one pass over the split.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "runs" / "hyp_multi_cuda" / "checkpoint_best.pth"
DEFAULT_OUT_DIR_ALL = HERE / "runs" / "hyp_multi_cuda" / "inclass_mnist_all"


def _default_mnist_processed_dir() -> Optional[str]:
    for rel in (
        HERE.parent.parent / "CVAEChecked" / "Data" / "MNIST_processed",
        HERE.parent / ".." / "CVAEChecked" / "Data" / "MNIST_processed",
    ):
        p = rel.resolve()
        if p.is_dir():
            return str(p)
    return None


class ScaledRawDataset(Dataset):
    def __init__(self, base):
        self.base = base
        from mnist_local import preprocess_batch_by_digit_minmax as _p

        self._preprocess = _p

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x_raw, d, idx = self.base[i]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = self._preprocess(x_raw.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
        return x_scaled, d_t, idx, x_raw


@torch.no_grad()
def _scores_matrix_all_k(model, c_h, R, x_scaled, objective: str, curvature: float) -> torch.Tensor:
    """Per-row, per-class soft scores s_k: (B, K)."""
    from hyperbolic_ops import hyp_distance

    rep, _ = model(x_scaled)
    z_all = model.project_all_h(rep)
    B, K, _ = z_all.shape
    out = torch.empty((B, K), device=x_scaled.device, dtype=x_scaled.dtype)
    for k in range(K):
        ck = c_h[k : k + 1].expand(B, -1)
        d2 = hyp_distance(z_all[:, k, :], ck, c=curvature) ** 2
        out[:, k] = d2 - (R[k] ** 2) if objective == "soft-boundary" else d2
    return out


@torch.no_grad()
def _load_model_and_loader(
    mnist_processed_dir: str,
    checkpoint_path: str,
    device: torch.device,
    train_fraction: float,
    split: str,
    batch_size: int,
    n_workers: int,
):
    sys.path.insert(0, str(HERE))
    from mnist_local import MNISTDigitsProcessedRawDataset, MNIST_LeNet_SVDDIAE
    from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    backbone = MNIST_LeNet_SVDDIAE(rep_dim=int(ckpt["rep_dim"]))
    model = HyperbolicMultiSphereSVDD(
        backbone=backbone,
        rep_dim=int(ckpt["rep_dim"]),
        z_dim=int(ckpt["z_dim"]),
        n_digits=10,
        c=float(ckpt["curvature"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    c_h = ckpt["c_h"].to(device)
    R = ckpt["R"].to(device)
    objective = ckpt.get("objective", "soft-boundary")
    curv = float(ckpt["curvature"])

    base = MNISTDigitsProcessedRawDataset(
        root_dir=mnist_processed_dir,
        split=split,
        train_fraction=train_fraction,
        digits=list(range(10)),
    )
    ds = ScaledRawDataset(base)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return model, c_h, R, objective, curv, dl


def _run_single_class(
    mnist_processed_dir: str,
    checkpoint_path: str,
    normal_class: int,
    device: torch.device,
    train_fraction: float,
    split: str,
    batch_size: int,
    n_workers: int,
):
    model, c_h, R, objective, curv, dl = _load_model_and_loader(
        mnist_processed_dir, checkpoint_path, device, train_fraction, split, batch_size, n_workers
    )
    k = int(normal_class)
    best_s, worst_s = float("inf"), float("-inf")
    best_img = worst_img = None
    for x_scaled, digits, _idx, x_raw in dl:
        x_b = x_scaled.to(device)
        S = _scores_matrix_all_k(model, c_h, R, x_b, objective, curv)
        for i in range(x_b.size(0)):
            if digits[i].item() != k:
                continue
            si = float(S[i, k].item())
            raw = x_raw[i].squeeze(0).numpy()
            if si < best_s:
                best_s, best_img = si, raw
            if si > worst_s:
                worst_s, worst_img = si, raw

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_img is None:
        raise RuntimeError(f"No in-class MNIST samples for digit {k} on split={split}.")
    return best_img, worst_img, best_s, worst_s, k


def _run_all_classes(
    mnist_processed_dir: str,
    checkpoint_path: str,
    device: torch.device,
    train_fraction: float,
    split: str,
    batch_size: int,
    n_workers: int,
):
    model, c_h, R, objective, curv, dl = _load_model_and_loader(
        mnist_processed_dir, checkpoint_path, device, train_fraction, split, batch_size, n_workers
    )
    K = 10
    best_s = np.full(K, np.inf, dtype=np.float64)
    worst_s = np.full(K, -np.inf, dtype=np.float64)
    best_img = [None] * K
    worst_img = [None] * K

    for x_scaled, digits, _idx, x_raw in dl:
        x_b = x_scaled.to(device)
        S = _scores_matrix_all_k(model, c_h, R, x_b, objective, curv).detach().cpu().numpy()
        for i in range(x_b.size(0)):
            y = int(digits[i].item())
            si = float(S[i, y])
            raw = x_raw[i].squeeze(0).numpy()
            if si < best_s[y]:
                best_s[y], best_img[y] = si, raw
            if si > worst_s[y]:
                worst_s[y], worst_img[y] = si, raw

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for k in range(K):
        if best_img[k] is None:
            raise RuntimeError(f"No in-class MNIST samples for digit {k} on split={split}.")
    return best_img, worst_img, best_s, worst_s


def _save_row_png(img_n, img_a, sn, sa, k, out_path: str, dpi: int):
    odir = os.path.dirname(os.path.abspath(out_path))
    if odir:
        os.makedirs(odir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.8), constrained_layout=True)
    fig.suptitle(f"MNIST digit {k}: in-class (soft-boundary $s_{{{k}}}$)", fontsize=11)
    axes[0].imshow(img_n, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"most normal\n$s_{{{k}}}={sn:.4f}$", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(img_a, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(f"most anomalous (in-class)\n$s_{{{k}}}={sa:.4f}$", fontsize=9)
    axes[1].axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_combined_grid_mnist(best_img, worst_img, best_s, worst_s, out_path: str, dpi: int):
    K = 10
    fig, axes = plt.subplots(K, 2, figsize=(4.8, 1.05 * K + 0.8), constrained_layout=True)
    fig.suptitle("MNIST — all classes: most normal (left) vs most anomalous in-class (right)", fontsize=11)
    for k in range(K):
        for j, (img, s) in enumerate(((best_img[k], best_s[k]), (worst_img[k], worst_s[k]))):
            axes[k, j].imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            axes[k, j].axis("off")
            if k == 0:
                axes[k, j].set_title(("Most normal", "Most anomalous (in-class)")[j], fontsize=9)
            axes[k, j].text(
                0.03,
                0.97,
                f"$s_{{{k}}}={float(s):.3f}$",
                transform=axes[k, j].transAxes,
                fontsize=7,
                va="top",
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.5, lw=0),
            )
        axes[k, 0].set_ylabel(str(k), fontsize=10, rotation=0, ha="right", va="center", labelpad=14)
    odir = os.path.dirname(os.path.abspath(out_path))
    if odir:
        os.makedirs(odir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("MNIST in-class most normal vs most anomalous (1x2 per class)")
    p.add_argument("--mnist_processed_dir", type=str, default=None, help="Default: ../CVAEChecked/Data/MNIST_processed if present.")
    p.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--all_classes", action="store_true", help="Write digits 0..9 to --out_dir (one PNG each).")
    p.add_argument("--normal_class", type=int, default=0, help="Digit 0..9 when not --all_classes.")
    p.add_argument("--out_path", type=str, default=str(HERE / "runs" / "inclass_extremes_mnist.png"))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR_ALL), help="Used with --all_classes.")
    p.add_argument(
        "--combined",
        action="store_true",
        help="With --all_classes: also write one 10×2 montage (all classes).",
    )
    p.add_argument(
        "--combined_path",
        type=str,
        default=None,
        help="Override combined PNG path (default: out_dir/inclass_extremes_mnist_combined.png).",
    )
    p.add_argument("--save_separate", action="store_true", help="With --all_classes: also write single-image PNGs per digit.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    mnist_dir = args.mnist_processed_dir or _default_mnist_processed_dir()
    if not mnist_dir:
        p.error("Pass --mnist_processed_dir or place MNIST_processed under CVAEChecked/Data/.")

    device = torch.device(args.device)

    if args.all_classes:
        os.makedirs(args.out_dir, exist_ok=True)
        best_img, worst_img, best_s, worst_s = _run_all_classes(
            mnist_dir,
            args.checkpoint_path,
            device,
            args.train_fraction,
            args.split,
            args.batch_size,
            args.n_jobs_dataloader,
        )
        for k in range(10):
            op = os.path.join(args.out_dir, f"inclass_extremes_mnist_k{k}.png")
            _save_row_png(best_img[k], worst_img[k], float(best_s[k]), float(worst_s[k]), k, op, args.dpi)
            print(f"Wrote {op}")
            if args.save_separate:
                base = Path(op).stem
                pdir = Path(op).parent
                plt.imsave(
                    pdir / f"{base}_most_normal_s{best_s[k]:.4f}.png",
                    best_img[k],
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
                plt.imsave(
                    pdir / f"{base}_most_anomalous_s{worst_s[k]:.4f}.png",
                    worst_img[k],
                    cmap="gray",
                    vmin=0.0,
                    vmax=1.0,
                )
        if args.combined:
            cpath = args.combined_path or os.path.join(args.out_dir, "inclass_extremes_mnist_combined.png")
            _save_combined_grid_mnist(best_img, worst_img, best_s, worst_s, cpath, args.dpi)
            print(f"Wrote {cpath}")
        return

    img_n, img_a, sn, sa, k = _run_single_class(
        mnist_dir,
        args.checkpoint_path,
        args.normal_class,
        device,
        args.train_fraction,
        args.split,
        args.batch_size,
        args.n_jobs_dataloader,
    )
    out_path = os.path.abspath(args.out_path)
    _save_row_png(img_n, img_a, sn, sa, k, out_path, args.dpi)
    if args.save_separate:
        stem = Path(out_path).stem
        parent = Path(out_path).parent
        plt.imsave(parent / f"{stem}_most_normal_k{k}_s{sn:.4f}.png", img_n, cmap="gray", vmin=0.0, vmax=1.0)
        plt.imsave(parent / f"{stem}_most_anomalous_k{k}_s{sa:.4f}.png", img_a, cmap="gray", vmin=0.0, vmax=1.0)
        print(f"Wrote singles under {parent}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
