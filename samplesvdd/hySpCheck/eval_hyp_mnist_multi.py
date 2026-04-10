import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from mnist_local import MNISTDigitsProcessedRawDataset, preprocess_batch_by_digit_minmax, MNIST_LeNet_SVDDIAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD
from hyperbolic_ops import hyp_distance


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


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Eval Hyperbolic Multi-Sphere SVDD")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument(
        "--distance_metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "hyperbolic"],
        help="Distance used for SVDD scoring: Euclidean in mapped coords vs hyperbolic geodesic in Poincaré ball.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")

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

    base = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir, split=args.split, train_fraction=args.train_fraction, digits=list(range(10))
    )
    ds = ScaledRawDataset(base)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    all_digits = []
    all_scores = []
    all_idxs = []
    for x_scaled, digits, idx in dl:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)  # (B,K,z)
        if args.distance_metric == "euclidean":
            # Euclidean proxy in mapped coordinates
            dist_sq = torch.sum((z_all - c_h.unsqueeze(0)) ** 2, dim=2)
        else:
            c_all = c_h.unsqueeze(0).expand_as(z_all)  # (B,K,z)
            dist_sq = hyp_distance(z_all, c_all, c=model.curvature) ** 2
        if objective == "soft-boundary":
            scores = dist_sq - (R.unsqueeze(0) ** 2)
        else:
            scores = dist_sq
        all_digits.append(digits.cpu())
        all_scores.append(scores.cpu())
        all_idxs.append(idx)

    digits_np = torch.cat(all_digits).numpy()
    scores_np = torch.cat(all_scores).numpy()
    idxs_np = torch.cat(all_idxs).numpy()

    per_digit = {}
    aucs = []
    for k in range(10):
        y = (digits_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(y, scores_np[:, k])
        except ValueError:
            auc = np.nan
        per_digit[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")

    min_scores = np.min(scores_np, axis=1)
    global_anom = np.clip(min_scores, a_min=0.0, a_max=None)
    top_k = min(args.top_k, len(global_anom))
    top_idx = np.argsort(-global_anom)[:top_k]
    csv_path = os.path.join(args.out_dir, "global_top_anomalies.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sample_idx", "true_digit", "global_anom_score", "best_cluster"])
        for r, i in enumerate(top_idx, start=1):
            w.writerow([r, int(idxs_np[i]), int(digits_np[i]), float(global_anom[i]), int(np.argmin(scores_np[i]))])

    out = {"macro_svdd_auc": macro_auc, "per_digit_svdd_auc": per_digit, "csv": csv_path}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

