import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from mnist_local import MNISTDigitsProcessedRawDataset, preprocess_batch_by_digit_minmax, MNIST_LeNet_SVDDIAE, recon_mse_loss

from hyperbolic_multi_sphere import (
    HyperbolicMultiSphereSVDD,
    dist_sq_to_all_centers,
    hyp_dist_sq_to_centers,
    init_centers_h,
    svdd_loss_one_class,
    svdd_loss_soft_boundary,
    update_radii,
)
from hyperbolic_ops import hyp_distance


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ScaledDigitDataset(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_raw, d, _ = self.base[idx]
        d_t = torch.tensor(d, dtype=torch.long)
        x_scaled = preprocess_batch_by_digit_minmax(x_raw.unsqueeze(0), d_t.unsqueeze(0)).squeeze(0)
        return x_scaled, d_t


@torch.no_grad()
def evaluate(model, c_h, R, loader, device, objective):
    model.eval()
    all_d = []
    all_s = []
    for x_scaled, digits in loader:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        z_all = model.project_all_h(rep)  # (B,K,z)
        c_all = c_h.unsqueeze(0).expand(z_all.size(0), -1, -1)
        dist_sq_all = torch.sum((z_all - c_all) ** 2, dim=2)  # proxy for speed in eval
        if objective == "soft-boundary":
            scores_all = dist_sq_all - (R.unsqueeze(0) ** 2)
        else:
            scores_all = dist_sq_all
        all_d.append(digits.cpu())
        all_s.append(scores_all.cpu())

    digits_np = torch.cat(all_d).numpy()
    scores_np = torch.cat(all_s).numpy()
    per_digit = {}
    aucs = []
    for k in range(scores_np.shape[1]):
        y = (digits_np != k).astype(np.int32)
        try:
            auc = roc_auc_score(y, scores_np[:, k])
        except ValueError:
            auc = np.nan
        per_digit[str(k)] = None if np.isnan(auc) else float(auc)
        if not np.isnan(auc):
            aucs.append(float(auc))
    macro = float(np.mean(aucs)) if aucs else float("nan")
    return macro, per_digit


def inter_class_exclusion_loss(dist_sq_all, R, y, margin: float) -> torch.Tensor:
    """
    Encourage sample to lie outside all non-true class spheres (squared form):
      dist_sq(x, k) >= (R_k + margin)^2,  for k != y
    """
    B, K = dist_sq_all.shape
    target = (R.unsqueeze(0) + float(margin)) ** 2
    viol = torch.relu(target - dist_sq_all)  # (B,K)
    true_mask = torch.nn.functional.one_hot(y, num_classes=K).bool()
    viol = viol.masked_fill(true_mask, 0.0)
    return torch.mean(viol)


def sphere_overlap_penalty(c_h, R, curvature: float, margin: float) -> torch.Tensor:
    """
    Penalize overlap of class spheres in hyperbolic geometry:
      d(c_a, c_b) >= R_a + R_b + margin
    """
    K = c_h.size(0)
    if K <= 1:
        return torch.tensor(0.0, device=c_h.device)
    penalties = []
    for a in range(K):
        for b in range(a + 1, K):
            d_ab = hyp_distance(c_h[a : a + 1], c_h[b : b + 1], c=curvature).squeeze(0)
            penalties.append(torch.relu((R[a] + R[b] + float(margin)) - d_ab))
    return torch.mean(torch.stack(penalties)) if penalties else torch.tensor(0.0, device=c_h.device)


def main():
    p = argparse.ArgumentParser("Hyperbolic Multi-Sphere SVDD on MNIST_processed")
    p.add_argument("--mnist_processed_dir", type=str, required=True)
    p.add_argument("--xp_path", type=str, default="hySpCheck/runs/hyp_multi")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--digits", type=str, default="all")
    p.add_argument("--train_fraction", type=float, default=0.8)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_jobs_dataloader", type=int, default=0)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)
    p.add_argument("--rep_dim", type=int, default=32)
    p.add_argument("--z_dim", type=int, default=16)
    p.add_argument("--curvature", type=float, default=1.0)
    p.add_argument("--ae_n_epochs", type=int, default=150)
    p.add_argument("--svdd_n_epochs", type=int, default=100)
    p.add_argument("--ae_lr", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--objective", type=str, default="soft-boundary", choices=["one-class", "soft-boundary"])
    p.add_argument("--nu", type=float, default=0.1)
    p.add_argument("--lambda_svdd", type=float, default=5e-5)
    p.add_argument("--lambda_excl", type=float, default=1e-2, help="Weight of inter-class exclusion penalty.")
    p.add_argument("--margin_excl", type=float, default=0.1, help="Exclusion margin outside non-true spheres.")
    p.add_argument("--lambda_overlap", type=float, default=1e-2, help="Weight of sphere overlap penalty.")
    p.add_argument("--margin_overlap", type=float, default=0.05, help="Required extra separation between spheres.")
    p.add_argument("--warm_up_n_epochs", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true")
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

    # Required by CUDA/cuBLAS for deterministic GEMM paths on older PyTorch builds.
    if args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    os.makedirs(args.xp_path, exist_ok=True)
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    digits = list(range(10)) if args.digits == "all" else [int(x) for x in args.digits.split(",") if x.strip()]

    tr_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir, split="train", train_fraction=args.train_fraction, digits=digits, max_samples=args.max_train_samples
    )
    te_raw = MNISTDigitsProcessedRawDataset(
        root_dir=args.mnist_processed_dir, split="test", train_fraction=args.train_fraction, digits=digits, max_samples=args.max_test_samples
    )
    tr = ScaledDigitDataset(tr_raw)
    te = ScaledDigitDataset(te_raw)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    tr_loader = DataLoader(
        tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_jobs_dataloader,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )
    te_loader = DataLoader(
        te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_jobs_dataloader,
        worker_init_fn=_seed_worker,
        generator=loader_gen,
    )

    backbone = MNIST_LeNet_SVDDIAE(rep_dim=args.rep_dim)
    model = HyperbolicMultiSphereSVDD(backbone=backbone, rep_dim=args.rep_dim, z_dim=args.z_dim, n_digits=10, c=args.curvature).to(device)

    # Stage 1 AE pretrain
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
            for x_scaled, digits_b in tr_loader:
                x_scaled = x_scaled.to(device)
                _digits_b = digits_b.to(device)
                _rep, recon = model(x_scaled)
                loss = recon_mse_loss(recon, x_scaled)
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

    # Init centers and radii
    c_h = init_centers_h(model, tr_loader, device=device)
    R = torch.zeros((10,), device=device)

    # Stage 2 hyperbolic SVDD + recon
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_macro = -1e9
    best_epoch = -1
    best_per_digit = {}

    for ep in range(1, args.svdd_n_epochs + 1):
        model.train()
        losses = []
        rec_losses = []
        svdd_losses = []
        excl_losses = []
        ov_losses = []
        dist_sq_by_digit = [[] for _ in range(10)]
        for x_scaled, digits_b in tr_loader:
            x_scaled = x_scaled.to(device)
            digits_b = digits_b.to(device)
            rep, recon = model(x_scaled)
            z_h = model.project_self_h(rep, digits_b)
            c_b = c_h[digits_b]
            dist_sq = hyp_dist_sq_to_centers(z_h, c_b, c=args.curvature)
            z_all_h = model.project_all_h(rep)  # (B,K,z)
            # Use hyperbolic squared distance for exclusion, consistent with CIFAR trainer and paper objective.
            dist_sq_all = dist_sq_to_all_centers(z_all_h, c_h, curvature=args.curvature)
            rec = recon_mse_loss(recon, x_scaled)
            if args.objective == "soft-boundary":
                R_b = R[digits_b]
                sv = svdd_loss_soft_boundary(dist_sq, R_b, nu=args.nu)
            else:
                sv = svdd_loss_one_class(dist_sq)
            excl = inter_class_exclusion_loss(dist_sq_all, R, digits_b, margin=args.margin_excl)
            ov = sphere_overlap_penalty(c_h, R, curvature=args.curvature, margin=args.margin_overlap)
            loss = rec + args.lambda_svdd * sv + args.lambda_excl * excl + args.lambda_overlap * ov
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            rec_losses.append(float(rec.item()))
            svdd_losses.append(float(sv.item()))
            excl_losses.append(float(excl.item()))
            ov_losses.append(float(ov.item()))
            if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
                for k in range(10):
                    m = digits_b == k
                    if torch.any(m):
                        dist_sq_by_digit[k].append(dist_sq[m].detach().cpu())

        if args.objective == "soft-boundary" and ep > args.warm_up_n_epochs:
            R = update_radii(dist_sq_by_digit, nu=args.nu, device=device)
        print(
            f"[SVDD-H] {ep:03d}/{args.svdd_n_epochs} "
            f"loss={np.mean(losses):.6f} rec={np.mean(rec_losses):.6f} svdd={np.mean(svdd_losses):.6f} "
            f"excl={np.mean(excl_losses):.6f} overlap={np.mean(ov_losses):.6f} R_mean={float(R.mean().item()):.4f}"
        )

        if ep % args.eval_every == 0 or ep == args.svdd_n_epochs:
            macro, per_digit = evaluate(model, c_h, R, te_loader, device, args.objective)
            print(f"[EVAL] epoch={ep:03d} macro_auc={macro}")
            metric = macro if not np.isnan(macro) else -1e12
            if metric > best_macro:
                best_macro = macro
                best_epoch = ep
                best_per_digit = per_digit
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
                        "seed": args.seed,
                        "deterministic": args.deterministic,
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
                "seed": args.seed,
                "deterministic": args.deterministic,
            },
            os.path.join(args.xp_path, "checkpoint_latest.pth"),
        )

    out = {
        "best_epoch": int(best_epoch),
        "best_macro_auc": None if np.isnan(best_macro) else float(best_macro),
        "per_digit_best_auc": best_per_digit,
        "curvature": args.curvature,
        "objective": args.objective,
        "lambda_excl": args.lambda_excl,
        "margin_excl": args.margin_excl,
        "lambda_overlap": args.lambda_overlap,
        "margin_overlap": args.margin_overlap,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    with open(os.path.join(args.xp_path, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

