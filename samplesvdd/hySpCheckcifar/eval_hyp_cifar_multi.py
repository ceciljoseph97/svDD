import argparse
import csv
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cifar10_data import CIFAR10RawDataset
from cifar_backbone import CifarConvAE
from hyperbolic_multi_sphere import HyperbolicMultiSphereSVDD, dist_sq_to_all_centers


@torch.no_grad()
def main():
    p = argparse.ArgumentParser("Eval hyperbolic coupled multi-sphere on CIFAR10")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument(
        "--distance_metric",
        type=str,
        default="hyperbolic",
        choices=["euclidean", "hyperbolic"],
        help="Distance used for SVDD scoring: Euclidean in mapped coords vs hyperbolic geodesic in Poincaré ball.",
    )
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

    te = CIFAR10RawDataset(root=args.data_root, split="test", digits=list(range(10)), max_samples=args.max_test_samples, download=True)
    dl = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    ys, idxs, scores = [], [], []
    for x, y, idx in dl:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z_all = model.project_all_h(rep)
        if args.distance_metric == "euclidean":
            d2_all = torch.sum((z_all - c_h.unsqueeze(0)) ** 2, dim=2)
        else:
            d2_all = dist_sq_to_all_centers(z_all, c_h, curvature=model.curvature)
        s_all = d2_all - (R.unsqueeze(0) ** 2) if objective == "soft-boundary" else d2_all
        ys.append(y.cpu())
        idxs.append(idx)
        scores.append(s_all.cpu())

    y_np = torch.cat(ys).numpy()
    idx_np = torch.cat(idxs).numpy()
    s_np = torch.cat(scores).numpy()

    per, aucs = {}, []
    for k in range(10):
        yy = (y_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(yy, s_np[:, k])
        except ValueError:
            auc = np.nan
        per[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")

    min_scores = np.min(s_np, axis=1)
    global_anom = np.clip(min_scores, a_min=0.0, a_max=None)
    k = min(args.top_k, len(global_anom))
    top = np.argsort(-global_anom)[:k]
    csv_path = os.path.join(args.out_dir, "global_top_anomalies.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sample_idx", "true_class", "global_anom_score", "best_cluster"])
        for r, i in enumerate(top, start=1):
            w.writerow([r, int(idx_np[i]), int(y_np[i]), float(global_anom[i]), int(np.argmin(s_np[i]))])

    out = {"macro_svdd_auc": macro, "per_digit_svdd_auc": per, "csv": csv_path}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

