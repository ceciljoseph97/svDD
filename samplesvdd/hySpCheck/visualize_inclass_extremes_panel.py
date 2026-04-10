"""
2x2 figure: [ MNIST most normal | MNIST most anomalous ]
            [ CIFAR most normal | CIFAR most anomalous ]
In-class = true label equals selected one-class id k. Scores use soft-boundary class-k SVDD:
  s_k(x) = d_H(z_k, c_k)^2 - R_k^2  (lower = more normal for that class).

Run from repo root or this folder; requires hySpCheckcifar as sibling of hySpCheck.
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

# --- paths ---
HERE = Path(__file__).resolve().parent
SAMPLESVDD = HERE.parent
CIFAR_DIR = SAMPLESVDD / "hySpCheckcifar"

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


def _denorm_cifar_chw(t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3, 1, 1)
    x = t.detach().cpu().numpy() * std + mean
    x = np.clip(x, 0.0, 1.0)
    return np.transpose(x, (1, 2, 0))


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
def _scores_class_k_mnist(model, c_h, R, x_scaled, k: int, objective: str, curvature: float):
    from hyperbolic_ops import hyp_distance

    device = x_scaled.device
    rep, _ = model(x_scaled)
    z_all = model.project_all_h(rep)
    B = z_all.size(0)
    ck = c_h[k : k + 1].expand(B, -1)
    d2 = hyp_distance(z_all[:, k, :], ck, c=curvature) ** 2
    if objective == "soft-boundary":
        return d2 - (R[k] ** 2)
    return d2


def _run_mnist_section(
    mnist_processed_dir: str,
    checkpoint_path: str,
    normal_class: int,
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

    k = int(normal_class)
    scores_ic = []
    raws_ic = []
    for x_scaled, digits, _idx, x_raw in dl:
        m = digits == k
        if not torch.any(m):
            continue
        x_b = x_scaled.to(device)
        s = _scores_class_k_mnist(model, c_h, R, x_b, k, objective, curv).detach().cpu()
        for i in range(x_b.size(0)):
            if digits[i].item() == k:
                scores_ic.append(float(s[i].item()))
                raws_ic.append(x_raw[i].squeeze(0).numpy())

    if len(scores_ic) == 0:
        raise RuntimeError(f"No in-class MNIST test samples for digit {k}.")

    scores_ic = np.array(scores_ic, dtype=np.float64)
    j_min = int(np.argmin(scores_ic))
    j_max = int(np.argmax(scores_ic))
    img_normal = raws_ic[j_min]
    img_anom = raws_ic[j_max]
    s_min = float(scores_ic[j_min])
    s_max = float(scores_ic[j_max])

    del model, backbone, ckpt, dl, ds
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return img_normal, img_anom, s_min, s_max


def _unload_cifar_modules():
    for name in (
        "hyperbolic_multi_sphere",
        "cifar_backbone",
        "cifar10_data",
    ):
        sys.modules.pop(name, None)


def _run_cifar_section(
    data_root: str,
    checkpoint_path: str,
    normal_class: int,
    device: torch.device,
    batch_size: int,
    n_workers: int,
    max_test_samples: Optional[int],
):
    _unload_cifar_modules()
    if str(CIFAR_DIR) not in sys.path:
        sys.path.insert(0, str(CIFAR_DIR))

    from cifar10_data import CIFAR10RawDataset
    from cifar_backbone import CifarConvAE
    from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers

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
        download=True,
    )
    dl = DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    k = int(normal_class)
    scores_ic = []
    tensors_ic = []
    for x, y, _idx in dl:
        m = y == k
        if not torch.any(m):
            continue
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z_all = model.project_all_h(rep)
        d2_all = dist_sq_to_all_centers(z_all, c_h, curvature=model.curvature)
        s_all = d2_all - (R.unsqueeze(0) ** 2) if objective == "soft-boundary" else d2_all
        s_col = s_all[:, k].detach().cpu()
        for i in range(x.size(0)):
            if y[i].item() == k:
                scores_ic.append(float(s_col[i].item()))
                tensors_ic.append(x[i].detach().cpu())

    if len(scores_ic) == 0:
        raise RuntimeError(f"No in-class CIFAR-10 test samples for class {k}.")

    scores_ic = np.array(scores_ic, dtype=np.float64)
    j_min = int(np.argmin(scores_ic))
    j_max = int(np.argmax(scores_ic))
    img_normal = _denorm_cifar_chw(tensors_ic[j_min])
    img_anom = _denorm_cifar_chw(tensors_ic[j_max])
    s_min = float(scores_ic[j_min])
    s_max = float(scores_ic[j_max])

    del model, backbone, ckpt, dl
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return img_normal, img_anom, s_min, s_max


def main():
    p = argparse.ArgumentParser("Panel: in-class most normal vs most anomalous (MNIST + CIFAR-10)")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--mnist_checkpoint", type=str, required=True)
    p.add_argument("--cifar_data_root", type=str, required=True)
    p.add_argument("--cifar_checkpoint", type=str, required=True)
    p.add_argument("--mnist_normal_class", type=int, default=0, help="Digit 0..9 for MNIST row.")
    p.add_argument("--cifar_normal_class", type=int, default=0, help="Class id 0..9 for CIFAR row.")
    p.add_argument("--out_path", type=str, default=str(HERE / "runs" / "inclass_extremes_panel.png"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_test_samples_cifar", type=int, default=None)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    if not CIFAR_DIR.is_dir():
        raise FileNotFoundError(f"Expected CIFAR code at {CIFAR_DIR}")

    device = torch.device(args.device)

    # MNIST first (imports hySpCheck modules only)
    mn_n, mn_a, mn_sn, mn_sa = _run_mnist_section(
        mnist_processed_dir=args.mnist_processed_dir,
        checkpoint_path=args.mnist_checkpoint,
        normal_class=args.mnist_normal_class,
        device=device,
        train_fraction=args.train_fraction,
        split=args.split,
        batch_size=args.batch_size,
        n_workers=args.n_jobs_dataloader,
    )

    # CIFAR (reload hyperbolic_multi_sphere from hySpCheckcifar)
    cf_n, cf_a, cf_sn, cf_sa = _run_cifar_section(
        data_root=args.cifar_data_root,
        checkpoint_path=args.cifar_checkpoint,
        normal_class=args.cifar_normal_class,
        device=device,
        batch_size=args.batch_size,
        n_workers=args.n_jobs_dataloader,
        max_test_samples=args.max_test_samples_cifar,
    )

    mk = int(args.mnist_normal_class)
    ck = int(args.cifar_normal_class)
    cname = CIFAR10_NAMES[ck]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    fig.suptitle(
        "Most normal (left) and most anomalous (right) in-class examples\n"
        f"(soft-boundary score $s_k$ for true class $k$; lower is more normal). "
        f"MNIST: digit {mk}; CIFAR-10: {cname} ({ck}).",
        fontsize=10,
    )

    axes[0, 0].imshow(mn_n, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title(f"MNIST — most normal\n$s_{{{mk}}}={mn_sn:.4f}$", fontsize=9)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(mn_a, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].set_title(f"MNIST — most anomalous (in-class)\n$s_{{{mk}}}={mn_sa:.4f}$", fontsize=9)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cf_n)
    axes[1, 0].set_title(f"CIFAR-10 — most normal\n$s_{{{ck}}}={cf_sn:.4f}$", fontsize=9)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cf_a)
    axes[1, 1].set_title(f"CIFAR-10 — most anomalous (in-class)\n$s_{{{ck}}}={cf_sa:.4f}$", fontsize=9)
    axes[1, 1].axis("off")

    _odir = os.path.dirname(os.path.abspath(args.out_path))
    if _odir:
        os.makedirs(_odir, exist_ok=True)
    fig.savefig(args.out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
