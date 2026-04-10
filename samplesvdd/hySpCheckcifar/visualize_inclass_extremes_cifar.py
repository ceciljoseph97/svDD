"""
CIFAR-10: most normal (left) vs most anomalous (right) among in-class test samples per class k,
score s_k = d_H(z_k,c_k)^2 - R_k^2 (soft-boundary).

Use --all_classes to write one 1x2 PNG per class (0..9) in a single pass.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE / "runs" / "hyp_cifar_cuda" / "checkpoint_best.pth"
DEFAULT_OUT_DIR_ALL = HERE / "runs" / "hyp_cifar_cuda" / "inclass_cifar_all"

CIFAR10_NAMES = (
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
)


def _denorm_cifar_chw(t: np.ndarray) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
    x = t * std + mean
    x = np.clip(x, 0.0, 1.0)
    return np.transpose(x, (1, 2, 0))


def _default_data_root() -> Optional[str]:
    for p in (
        HERE / "data",
        HERE / "cifar_data",
        HERE.parent / "cifar_data",
        HERE.parent.parent / "cifar_data",
    ):
        p = p.resolve()
        if (p / "cifar-10-batches-py").is_dir():
            return str(p)
    return None


@torch.no_grad()
def _scores_matrix_all_k(model, c_h, R, x, objective: str) -> torch.Tensor:
    from hyperbolic_multi_sphere import dist_sq_to_all_centers

    rep, _ = model(x)
    z_all = model.project_all_h(rep)
    d2 = dist_sq_to_all_centers(z_all, c_h, curvature=model.curvature)
    if objective == "soft-boundary":
        return d2 - (R.unsqueeze(0) ** 2)
    return d2


@torch.no_grad()
def _load_model_and_loader(
    data_root: str,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    n_workers: int,
    max_test_samples: Optional[int],
    download: bool,
):
    sys.path.insert(0, str(HERE))
    from cifar10_data import CIFAR10RawDataset
    from cifar_backbone import CifarConvAE
    from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD

    ckpt = torch.load(checkpoint_path, map_location="cpu")
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

    te = CIFAR10RawDataset(
        root=data_root,
        split="test",
        digits=list(range(10)),
        max_samples=max_test_samples,
        download=download,
    )
    dl = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return model, c_h, R, objective, dl


def _tensor_to_show(x_chw: torch.Tensor) -> np.ndarray:
    return _denorm_cifar_chw(x_chw.detach().cpu().numpy())


def _run_single_class(
    data_root: str,
    checkpoint_path: str,
    normal_class: int,
    device: torch.device,
    batch_size: int,
    n_workers: int,
    max_test_samples: Optional[int],
    download: bool,
) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    model, c_h, R, objective, dl = _load_model_and_loader(
        data_root, checkpoint_path, device, batch_size, n_workers, max_test_samples, download
    )
    k = int(normal_class)
    best_s, worst_s = float("inf"), float("-inf")
    best_img = worst_img = None
    for x, y, _ in dl:
        x = x.to(device)
        S = _scores_matrix_all_k(model, c_h, R, x, objective).detach().cpu().numpy()
        for i in range(x.size(0)):
            if int(y[i].item()) != k:
                continue
            si = float(S[i, k])
            vis = _tensor_to_show(x[i])
            if si < best_s:
                best_s, best_img = si, vis
            if si > worst_s:
                worst_s, worst_img = si, vis

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_img is None:
        raise RuntimeError(f"No in-class CIFAR-10 samples for class {k}.")
    return best_img, worst_img, best_s, worst_s, k


def _run_all_classes(
    data_root: str,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    n_workers: int,
    max_test_samples: Optional[int],
    download: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    model, c_h, R, objective, dl = _load_model_and_loader(
        data_root, checkpoint_path, device, batch_size, n_workers, max_test_samples, download
    )
    K = 10
    best_s = np.full(K, np.inf, dtype=np.float64)
    worst_s = np.full(K, -np.inf, dtype=np.float64)
    best_img: List[Optional[np.ndarray]] = [None] * K
    worst_img: List[Optional[np.ndarray]] = [None] * K

    for x, y, _ in dl:
        x = x.to(device)
        S = _scores_matrix_all_k(model, c_h, R, x, objective).detach().cpu().numpy()
        for i in range(x.size(0)):
            c = int(y[i].item())
            si = float(S[i, c])
            vis = _tensor_to_show(x[i])
            if si < best_s[c]:
                best_s[c], best_img[c] = si, vis
            if si > worst_s[c]:
                worst_s[c], worst_img[c] = si, vis

    del model, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for k in range(K):
        if best_img[k] is None:
            raise RuntimeError(f"No in-class CIFAR-10 samples for class {k}.")
    return best_img, worst_img, best_s, worst_s  # type: ignore[return-value]


def _save_row_png(img_n, img_a, sn, sa, k, out_path: str, dpi: int):
    odir = os.path.dirname(os.path.abspath(out_path))
    if odir:
        os.makedirs(odir, exist_ok=True)
    name = CIFAR10_NAMES[k]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.9), constrained_layout=True)
    fig.suptitle(f"CIFAR-10 {name} (class {k}): in-class $s_{{{k}}}$", fontsize=11)
    axes[0].imshow(img_n)
    axes[0].set_title(f"most normal\n$s_{{{k}}}={sn:.4f}$", fontsize=9)
    axes[0].axis("off")
    axes[1].imshow(img_a)
    axes[1].set_title(f"most anomalous (in-class)\n$s_{{{k}}}={sa:.4f}$", fontsize=9)
    axes[1].axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_combined_grid_cifar(best_img, worst_img, best_s, worst_s, out_path: str, dpi: int):
    K = 10
    fig, axes = plt.subplots(K, 2, figsize=(5.2, 1.05 * K + 0.8), constrained_layout=True)
    fig.suptitle("CIFAR-10 — all classes: most normal (left) vs most anomalous in-class (right)", fontsize=11)
    for k in range(K):
        name = CIFAR10_NAMES[k]
        for j, (img, s) in enumerate(((best_img[k], best_s[k]), (worst_img[k], worst_s[k]))):
            axes[k, j].imshow(img, vmin=0.0, vmax=1.0)
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
        axes[k, 0].set_ylabel(f"{k}\n{name}", fontsize=8, rotation=0, ha="right", va="center", labelpad=18)
    odir = os.path.dirname(os.path.abspath(out_path))
    if odir:
        os.makedirs(odir, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser("CIFAR-10 in-class most normal vs most anomalous (1x2 per class)")
    p.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Folder containing cifar-10-batches-py (parent of that dir). Default tries hySpCheckcifar/data then cifar_data.",
    )
    p.add_argument("--checkpoint_path", type=str, default=str(DEFAULT_CKPT))
    p.add_argument("--all_classes", action="store_true")
    p.add_argument("--normal_class", type=int, default=0)
    p.add_argument("--out_path", type=str, default=str(HERE / "runs" / "inclass_extremes_cifar.png"))
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR_ALL))
    p.add_argument(
        "--combined",
        action="store_true",
        help="With --all_classes: also write one 10×2 montage (all classes).",
    )
    p.add_argument(
        "--combined_path",
        type=str,
        default=None,
        help="Override combined PNG path (default: out_dir/inclass_extremes_cifar_combined.png).",
    )
    p.add_argument("--save_separate", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--no_download", action="store_true", help="Fail if CIFAR is missing (no download).")
    p.add_argument("--dpi", type=int, default=220)
    args = p.parse_args()

    root = args.data_root or _default_data_root()
    if not root:
        p.error(
            "Pass --data_root (parent of cifar-10-batches-py), or place data under hySpCheckcifar/data or hySpCheckcifar/cifar_data."
        )

    device = torch.device(args.device)
    download = not args.no_download

    if args.all_classes:
        os.makedirs(args.out_dir, exist_ok=True)
        best_img, worst_img, best_s, worst_s = _run_all_classes(
            root,
            args.checkpoint_path,
            device,
            args.batch_size,
            args.n_jobs_dataloader,
            args.max_test_samples,
            download,
        )
        for k in range(10):
            op = os.path.join(args.out_dir, f"inclass_extremes_cifar_k{k}.png")
            _save_row_png(best_img[k], worst_img[k], float(best_s[k]), float(worst_s[k]), k, op, args.dpi)
            print(f"Wrote {op}")
            if args.save_separate:
                base = Path(op).stem
                pdir = Path(op).parent
                plt.imsave(pdir / f"{base}_most_normal_s{best_s[k]:.4f}.png", best_img[k], vmin=0.0, vmax=1.0)
                plt.imsave(pdir / f"{base}_most_anomalous_s{worst_s[k]:.4f}.png", worst_img[k], vmin=0.0, vmax=1.0)
        if args.combined:
            cpath = args.combined_path or os.path.join(args.out_dir, "inclass_extremes_cifar_combined.png")
            _save_combined_grid_cifar(best_img, worst_img, best_s, worst_s, cpath, args.dpi)
            print(f"Wrote {cpath}")
        return

    img_n, img_a, sn, sa, k = _run_single_class(
        root,
        args.checkpoint_path,
        args.normal_class,
        device,
        args.batch_size,
        args.n_jobs_dataloader,
        args.max_test_samples,
        download,
    )
    out_path = os.path.abspath(args.out_path)
    _save_row_png(img_n, img_a, sn, sa, k, out_path, args.dpi)
    if args.save_separate:
        stem = Path(out_path).stem
        parent = Path(out_path).parent
        plt.imsave(parent / f"{stem}_most_normal_k{k}_s{sn:.4f}.png", img_n, vmin=0.0, vmax=1.0)
        plt.imsave(parent / f"{stem}_most_anomalous_k{k}_s{sa:.4f}.png", img_a, vmin=0.0, vmax=1.0)
        print(f"Wrote singles under {parent}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
