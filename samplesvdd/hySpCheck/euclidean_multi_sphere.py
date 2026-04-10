import torch
import torch.nn as nn


class EuclideanMultiSphereSVDD(nn.Module):
    """Shared AE backbone + per-cluster linear heads + Euclidean embeddings (R^{z_dim})."""

    def __init__(self, backbone: nn.Module, rep_dim: int = 32, z_dim: int = 16, n_digits: int = 10):
        super().__init__()
        self.backbone = backbone
        self.rep_dim = rep_dim
        self.z_dim = z_dim
        self.n_digits = n_digits
        self.proj_heads = nn.ModuleList([nn.Linear(rep_dim, z_dim, bias=False) for _ in range(n_digits)])

    def forward(self, x_scaled: torch.Tensor):
        return self.backbone(x_scaled)

    def encode(self, x_scaled: torch.Tensor) -> torch.Tensor:
        rep, _ = self.backbone(x_scaled)
        return rep

    def project_self(self, rep: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        z = torch.empty((rep.size(0), self.z_dim), device=rep.device, dtype=rep.dtype)
        for k in range(self.n_digits):
            m = digits == k
            if torch.any(m):
                z[m] = self.proj_heads[k](rep[m])
        return z

    def project_all(self, rep: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.proj_heads[k](rep) for k in range(self.n_digits)], dim=1)


def dist_sq_to_all_centers_e(z_all: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """z_all: (B,K,z), c: (K,z) -> (B,K) squared Euclidean distance per cluster head."""
    B, K, _ = z_all.shape
    out = torch.empty((B, K), device=z_all.device, dtype=z_all.dtype)
    for k in range(K):
        diff = z_all[:, k, :] - c[k].unsqueeze(0)
        out[:, k] = (diff ** 2).sum(dim=-1)
    return out


@torch.no_grad()
def init_centers_e(model: EuclideanMultiSphereSVDD, train_loader, device: torch.device):
    c = torch.zeros((model.n_digits, model.z_dim), device=device)
    n = torch.zeros((model.n_digits,), device=device)
    for x_scaled, digits in train_loader:
        x_scaled = x_scaled.to(device)
        digits = digits.to(device)
        rep, _ = model(x_scaled)
        for k in range(model.n_digits):
            m = digits == k
            if torch.any(m):
                z_k = model.proj_heads[k](rep[m])
                c[k] += z_k.sum(dim=0)
                n[k] += float(m.sum().item())
    n = n.clamp_min(1.0).view(-1, 1)
    return c / n
