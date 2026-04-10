import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cifar10_data import CIFAR10RawDataset
from cifar_backbone import CifarConvAE, recon_mse_loss
from hyperbolic_multi_sphere import (
    HyperbolicMultiSphereSVDD,
    dist_sq_to_all_centers,
    init_centers_h,
    svdd_loss_one_class,
    svdd_loss_soft_boundary,
    update_radii,
)
from hyperbolic_ops import hyp_distance


def inter_class_exclusion_loss(dist_sq_all, R, y, margin):
    B, K = dist_sq_all.shape
    tgt = (R.unsqueeze(0) + float(margin)) ** 2
    viol = torch.relu(tgt - dist_sq_all)
    true_mask = F.one_hot(y, num_classes=K).bool()
    viol = viol.masked_fill(true_mask, 0.0)
    return torch.mean(viol)


def sphere_overlap_penalty(c_h, R, curvature, margin):
    K = c_h.size(0)
    vals = []
    for a in range(K):
        for b in range(a + 1, K):
            d = hyp_distance(c_h[a : a + 1], c_h[b : b + 1], c=curvature).squeeze(0)
            vals.append(torch.relu((R[a] + R[b] + float(margin)) - d))
    if not vals:
        return torch.tensor(0.0, device=c_h.device)
    return torch.mean(torch.stack(vals))


@torch.no_grad()
def evaluate(model, c_h, R, loader, device, objective):
    model.eval()
    ys, scores = [], []
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z_all = model.project_all_h(rep)
        d2_all = dist_sq_to_all_centers(z_all, c_h, curvature=model.curvature)
        s_all = d2_all - (R.unsqueeze(0) ** 2) if objective == "soft-boundary" else d2_all
        ys.append(y.cpu())
        scores.append(s_all.cpu())
    y_np = torch.cat(ys).numpy()
    s_np = torch.cat(scores).numpy()
    per, aucs = {}, []
    for k in range(s_np.shape[1]):
        yy = (y_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(yy, s_np[:, k])
        except ValueError:
            auc = np.nan
        per[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else float("nan"), per


def main():
    p = argparse.ArgumentParser("Hyperbolic coupled multi-sphere Deep SVDD for CIFAR10")
    p.add_argument("--data_root", type=str, required=True, help="Folder where cifar-10-python.tar.gz/batches live.")
    p.add_argument("--xp_path", type=str, default="hySpCheckcifar/runs/hyp_cifar_multi")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--classes", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--rep_dim", type=int, default=128)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--ae_n_epochs", type=int, default=20)
    p.add_argument("--svdd_n_epochs", type=int, default=40)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--objective", type=str, default="soft-boundary", choices=["one-class", "soft-boundary"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=5e-5)
    p.add_argument("--lambda_excl", type=float, default=1e-2)
    p.add_argument("--margin_excl", type=float, default=0.1)
    p.add_argument("--lambda_overlap", type=float, default=1e-2)
    p.add_argument("--margin_overlap", type=float, default=0.05)
    p.add_argument("--warm_up_n_epochs", type=int, default=8)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument(
        "--skip_ae_pretrain",
        action="store_true",
        help="Skip AE pretraining stage and load a saved Stage-1 checkpoint before SVDD training.",
    )
    p.add_argument(
        "--ae_stage1_checkpoint_path",
        type=str,
        default=None,
        help="Path to a checkpoint saved right after Stage-1 AE pretrain (contains model_state).",
    )
    p.add_argument(
        "--save_ae_stage1_checkpoint_path",
        type=str,
        default=None,
        help="If set, save a checkpoint right after Stage-1 AE pretrain to reuse for ablations.",
    )
    args = p.parse_args()

    os.makedirs(args.xp_path, exist_ok=True)
    device = torch.device(args.device)
    classes = list(range(10)) if args.classes == "all" else [int(x) for x in args.classes.split(",") if x.strip()]

    tr = CIFAR10RawDataset(root=args.data_root, split="train", digits=classes, max_samples=args.max_train_samples, download=True)
    te = CIFAR10RawDataset(root=args.data_root, split="test", digits=classes, max_samples=args.max_test_samples, download=True)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.n_jobs_dataloader)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=args.n_jobs_dataloader)

    backbone = CifarConvAE(rep_dim=args.rep_dim)
    model = HyperbolicMultiSphereSVDD(backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_classes=10, c=args.curvature).to(device)

    # stage1 AE
    if args.skip_ae_pretrain:
        if not args.ae_stage1_checkpoint_path:
            raise SystemExit("--skip_ae_pretrain requires --ae_stage1_checkpoint_path")
        ae_ckpt = torch.load(args.ae_stage1_checkpoint_path, map_location="cpu")
        model_state = ae_ckpt["model_state"] if isinstance(ae_ckpt, dict) and "model_state" in ae_ckpt else ae_ckpt
        model.load_state_dict(model_state, strict=True)
        print(f"[AE] skipped pretrain; loaded stage-1 checkpoint: {args.ae_stage1_checkpoint_path}")
    else:
        opt_ae = optim.Adam(model.parameters(), lr=args.ae_lr, weight_decay=args.weight_decay)
        for ep in range(1, args.ae_n_epochs + 1):
            model.train()
            losses = []
            for x, y, _ in tr_loader:
                x = x.to(device)
                _y = y.to(device)
                _, recon = model(x)
                loss = recon_mse_loss(recon, x)
                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                losses.append(float(loss.item()))
            print(f"[AE] {ep:03d}/{args.ae_n_epochs} recon={np.mean(losses):.6f}")
        if args.save_ae_stage1_checkpoint_path:
            out_path = args.save_ae_stage1_checkpoint_path
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            torch.save({"model_state": model.state_dict()}, out_path)
            print(f"[AE] saved stage-1 checkpoint: {out_path}")

    c_h = init_centers_h(model, tr_loader, device=device)
    R = torch.zeros((10,), device=device)

    # stage2 coupled
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_macro = -1e9
    best_epoch = -1
    best_per = {}
    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        logs = {"total": [], "rec": [], "svdd": [], "excl": [], "ov": []}
        dist_by_cls = [[] for _ in range(10)]
        for x, y, _ in tr_loader:
            x = x.to(device)
            y = y.to(device)
            rep, recon = model(x)
            z_self = model.project_self_h(rep, y)
            c_y = c_h[y]
            d2_self = hyp_distance(z_self, c_y, c=args.curvature) ** 2
            z_all = model.project_all_h(rep)
            d2_all = dist_sq_to_all_centers(z_all, c_h, curvature=args.curvature)

            rec = recon_mse_loss(recon, x)
            svdd = svdd_loss_soft_boundary(d2_self, R[y], nu=args.nu) if args.objective == "soft-boundary" else svdd_loss_one_class(d2_self)
            excl = inter_class_exclusion_loss(d2_all, R, y, margin=args.margin_excl)
            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            loss = rec + args.lambda_svdd * svdd + args.lambda_excl * excl + args.lambda_overlap * ov

            opt.zero_grad()
            loss.backward()
            opt.step()

            logs["total"].append(float(loss.item()))
            logs["rec"].append(float(rec.item()))
            logs["svdd"].append(float(svdd.item()))
            logs["excl"].append(float(excl.item()))
            logs["ov"].append(float(ov.item()))

            if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
                for k in range(10):
                    m = y == k
                    if torch.any(m):
                        dist_by_cls[k].append(d2_self[m].detach().cpu())

        if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
            R = update_radii(dist_by_cls, nu=args.nu, device=device)
        print(
            f"[SVDD-H] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={np.mean(logs['total']):.6f} rec={np.mean(logs['rec']):.6f} "
            f"svdd={np.mean(logs['svdd']):.6f} excl={np.mean(logs['excl']):.6f} ov={np.mean(logs['ov']):.6f} "
            f"R_mean={float(R.mean().item()):.4f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per = evaluate(model, c_h, R, te_loader, device, args.objective)
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro}")
            metric = macro if not np.isnan(macro) else -1e12
            if metric > best_macro:
                best_macro = macro
                best_epoch = ep
                best_per = per
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "c_h": c_h.detach().cpu(),
                        "R": R.detach().cpu(),
                        "rep_dim": args.rep_dim,
                        "z_dim": args.z_dim,
                        "curvature": args.curvature,
                        "objective": args.objective,
                        "lambda_excl": args.lambda_excl,
                        "margin_excl": args.margin_excl,
                        "lambda_overlap": args.lambda_overlap,
                        "margin_overlap": args.margin_overlap,
                    },
                    os.path.join(args.xp_path, "checkpoint_best.pth"),
                )

        torch.save(
            {
                "model_state": model.state_dict(),
                "c_h": c_h.detach().cpu(),
                "R": R.detach().cpu(),
                "rep_dim": args.rep_dim,
                "z_dim": args.z_dim,
                "curvature": args.curvature,
                "objective": args.objective,
                "lambda_excl": args.lambda_excl,
                "margin_excl": args.margin_excl,
                "lambda_overlap": args.lambda_overlap,
                "margin_overlap": args.margin_overlap,
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_epoch),
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "per_digit_best_auc": best_per,
        "curvature": args.curvature,
        "objective": args.objective,
        "lambda_excl": args.lambda_excl,
        "margin_excl": args.margin_excl,
        "lambda_overlap": args.lambda_overlap,
        "margin_overlap": args.margin_overlap,
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

