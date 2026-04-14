import torch
import torch.nn as nn

from hyperbolic_ops import expmap0, hyp_distance, proj_ball


class HyperbolicMultiSphereSVDD(nn.Module):
    """Shared AE backbone + per-class projection heads + hyperbolic embeddings."""

    def __init__(self, backbone: nn.Module, rep_dim: int = 32, z_dim: int = 16, n_digits: int = 10, c: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.rep_dim = rep_dim
        self.z_dim = z_dim
        self.n_digits = n_digits
        self.curvature = float(c)
        self.proj_heads = nn.ModuleList([nn.Linear(rep_dim, z_dim, bias=False) for _ in range(n_digits)])

    def forward(self, x_scaled: torch.Tensor):
        return self.backbone(x_scaled)

    def encode(self, x_scaled: torch.Tensor) -> torch.Tensor:
        rep, _ = self.backbone(x_scaled)
        return rep

    def project_self_h(self, rep: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        z = torch.empty((rep.size(0), self.z_dim), device=rep.device, dtype=rep.dtype)
        for k in range(self.n_digits):
            m = digits == k
            if torch.any(m):
                z[m] = self.proj_heads[k](rep[m])
        return expmap0(z, c=self.curvature)

    def project_all_h(self, rep: torch.Tensor) -> torch.Tensor:
        zs = [expmap0(self.proj_heads[k](rep), c=self.curvature) for k in range(self.n_digits)]
        return torch.stack(zs, dim=1)  # (B,K,z)


def hyp_dist_sq_to_centers(z_h: torch.Tensor, centers_h: torch.Tensor, c: float) -> torch.Tensor:
    # z_h: (B,z), centers_h: (B,z)
    return hyp_distance(z_h, centers_h, c=c) ** 2


def dist_sq_to_all_centers(z_all_h: torch.Tensor, c_h: torch.Tensor, curvature: float) -> torch.Tensor:
    # z_all_h: (B,K,z), c_h: (K,z)
    B, K, _ = z_all_h.shape
    out = torch.empty((B, K), device=z_all_h.device, dtype=z_all_h.dtype)
    for k in range(K):
        ck = c_h[k].unsqueeze(0).expand(B, -1)
        out[:, k] = hyp_distance(z_all_h[:, k, :], ck, c=curvature) ** 2
    return out


def svdd_loss_soft_boundary(dist_sq: torch.Tensor, R_d: torch.Tensor, nu: float) -> torch.Tensor:
    scores = dist_sq - (R_d ** 2)
    return torch.mean((R_d ** 2) + (1.0 / nu) * torch.relu(scores))


def svdd_loss_one_class(dist_sq: torch.Tensor) -> torch.Tensor:
    return torch.mean(dist_sq)


def update_radii(dist_sq_by_digit, nu: float, device: torch.device) -> torch.Tensor:
    r = torch.zeros((len(dist_sq_by_digit),), device=device, dtype=torch.float32)
    for k, chunks in enumerate(dist_sq_by_digit):
        if len(chunks) == 0:
            continue
        d = torch.cat(chunks, dim=0).numpy()
        # distance radius from sqrt(dist_sq) quantile
        q = float(torch.tensor(d).sqrt().quantile(1.0 - nu).item())
        r[k] = q
    return r


def update_radii_unsupervised(dist_sq_chunks_per_k: list, nu: float, device: torch.device) -> torch.Tensor:
    """Same rule as ``update_radii``: each list entry collects dist² to center k for points whose nearest sphere was k."""

    return update_radii(dist_sq_chunks_per_k, nu, device)


def init_centers_h(model: HyperbolicMultiSphereSVDD, train_loader, device: torch.device, eps: float = 1e-5):
    with torch.no_grad():
        c = torch.zeros((model.n_digits, model.z_dim), device=device)
        n = torch.zeros((model.n_digits,), device=device)
        for x_scaled, digits in train_loader:
            x_scaled = x_scaled.to(device)
            digits = digits.to(device)
            rep, _ = model(x_scaled)
            z_h = model.project_self_h(rep, digits)
            for k in range(model.n_digits):
                m = digits == k
                if torch.any(m):
                    c[k] += z_h[m].sum(dim=0)
                    n[k] += float(m.sum().item())
        n = n.clamp_min(1.0).view(-1, 1)
        c = c / n
        c = proj_ball(c, c=model.curvature, eps=eps)
    return c


def init_centers_h_unsupervised(
    model: HyperbolicMultiSphereSVDD, train_loader, device: torch.device, eps: float = 1e-5
):
    """Fréchet-style batch mean in the ball: average each head's hyperbolic embedding over all training points (no labels)."""

    with torch.no_grad():
        acc = torch.zeros((model.n_digits, model.z_dim), device=device)
        count = 0
        for batch in train_loader:
            x_scaled = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            rep, _ = model(x_scaled)
            z_all_h = model.project_all_h(rep)
            acc += z_all_h.sum(dim=0)
            count += int(x_scaled.size(0))
        c = acc / float(max(count, 1))
        c = proj_ball(c, c=model.curvature, eps=eps)
    return c


def soft_assignment_weights(dist_sq_all: torch.Tensor, beta: float) -> torch.Tensor:
    """Responsibilities q_{ik} ∝ exp(-beta * d_{ik}^2); rows sum to 1."""

    return torch.softmax(-float(beta) * dist_sq_all, dim=1)


def unsupervised_multi_sphere_loss(dist_sq_all: torch.Tensor, beta: float) -> torch.Tensor:
    """Expected distance under soft assignment (DAGMM-style): sum_k q_{ik} d_{ik}^2 averaged over batch."""

    w = soft_assignment_weights(dist_sq_all, beta)
    return torch.mean(torch.sum(w * dist_sq_all, dim=1))


def tune_gamma_unsup(
    model: HyperbolicMultiSphereSVDD,
    train_loader,
    criterion_rec,
    device: torch.device,
    beta: float,
    T: int = 5,
) -> float:
    """DASVDD-style ratio E[recon] / E[soft multi-sphere distance] over T random forward passes (no grad)."""

    model.eval()
    gammas = []
    for _ in range(T):
        tot_rec = 0.0
        tot_ms = 0.0
        n_batches = 0
        for batch in train_loader:
            x_scaled = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            rep, recon = model(x_scaled)
            rec = criterion_rec(recon, x_scaled)
            z_all = model.project_all_h(rep)
            with torch.no_grad():
                c0 = torch.zeros((model.n_digits, model.z_dim), device=device)
                for k in range(model.n_digits):
                    c0[k] = z_all[:, k, :].mean(dim=0)
                c0 = proj_ball(c0, c=model.curvature)
                dist_sq_all = dist_sq_to_all_centers(z_all, c0, curvature=model.curvature)
                ms = unsupervised_multi_sphere_loss(dist_sq_all, beta)
            tot_rec += float(rec.item())
            tot_ms += float(ms.item())
            n_batches += 1
        if n_batches == 0 or tot_ms <= 1e-12:
            continue
        gammas.append(tot_rec / n_batches / (tot_ms / n_batches))
    model.train()
    return float(sum(gammas) / max(len(gammas), 1)) if gammas else 1.0

