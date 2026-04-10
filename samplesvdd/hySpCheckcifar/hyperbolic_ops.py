import torch


def proj_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    sqrt_c = c ** 0.5
    max_norm = (1.0 - eps) / sqrt_c
    norm = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return x * scale


def expmap0(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    sqrt_c = c ** 0.5
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return proj_ball(factor * v, c=c, eps=eps)


def acosh_safe(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x = torch.clamp(x, min=1.0 + eps)
    return torch.log(x + torch.sqrt(x * x - 1.0))


def hyp_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
    x2 = torch.sum(x * x, dim=-1)
    y2 = torch.sum(y * y, dim=-1)
    diff2 = torch.sum((x - y) ** 2, dim=-1)
    num = 2.0 * c * diff2
    den = (1.0 - c * x2).clamp_min(eps) * (1.0 - c * y2).clamp_min(eps)
    z = 1.0 + num / den
    return acosh_safe(z) / (c ** 0.5)

