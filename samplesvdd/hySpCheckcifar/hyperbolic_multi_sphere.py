import torch
import torch.nn as nn

from hyperbolic_ops import expmap0, hyp_distance, proj_ball


class HyperbolicMultiSphereSVDD(nn.Module):
    def __init__(self, backbone: nn.Module, rep_dim: int, z_dim: int = 32, n_classes: int = 10, c: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.rep_dim = rep_dim
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.curvature = float(c)
        self.proj_heads = nn.ModuleList([nn.Linear(rep_dim, z_dim, bias=False) for _ in range(n_classes)])

    def forward(self, x):
        return self.backbone(x)

    def project_self_h(self, rep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.empty((rep.size(0), self.z_dim), device=rep.device, dtype=rep.dtype)
        for k in range(self.n_classes):
            m = y == k
            if torch.any(m):
                z[m] = self.proj_heads[k](rep[m])
        return expmap0(z, c=self.curvature)

    def project_all_h(self, rep: torch.Tensor) -> torch.Tensor:
        zs = [expmap0(self.proj_heads[k](rep), c=self.curvature) for k in range(self.n_classes)]
        return torch.stack(zs, dim=1)  # (B,K,z)


def dist_sq_to_all_centers(z_all_h: torch.Tensor, c_h: torch.Tensor, curvature: float) -> torch.Tensor:
    # z_all_h: (B,K,z), c_h: (K,z)
    B, K, _ = z_all_h.shape
    out = torch.empty((B, K), device=z_all_h.device, dtype=z_all_h.dtype)
    for k in range(K):
        ck = c_h[k].unsqueeze(0).expand(B, -1)
        out[:, k] = hyp_distance(z_all_h[:, k, :], ck, c=curvature) ** 2
    return out


def svdd_loss_soft_boundary(dist_sq: torch.Tensor, r_y: torch.Tensor, nu: float) -> torch.Tensor:
    scores = dist_sq - (r_y ** 2)
    return torch.mean((r_y ** 2) + (1.0 / nu) * torch.relu(scores))


def svdd_loss_one_class(dist_sq: torch.Tensor) -> torch.Tensor:
    return torch.mean(dist_sq)


def update_radii(dist_sq_by_class, nu: float, device: torch.device) -> torch.Tensor:
    R = torch.zeros((len(dist_sq_by_class),), device=device, dtype=torch.float32)
    for k, chunks in enumerate(dist_sq_by_class):
        if len(chunks) == 0:
            continue
        d = torch.cat(chunks, dim=0)
        R[k] = torch.quantile(torch.sqrt(d), 1.0 - nu)
    return R


@torch.no_grad()
def init_centers_h(model: HyperbolicMultiSphereSVDD, loader, device: torch.device):
    c = torch.zeros((model.n_classes, model.z_dim), device=device)
    n = torch.zeros((model.n_classes,), device=device)
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        rep, _ = model(x)
        z = model.project_self_h(rep, y)
        for k in range(model.n_classes):
            m = y == k
            if torch.any(m):
                c[k] += z[m].sum(dim=0)
                n[k] += float(m.sum().item())
    c = c / n.clamp_min(1.0).view(-1, 1)
    c = proj_ball(c, c=model.curvature, eps=1e-5)
    return c

