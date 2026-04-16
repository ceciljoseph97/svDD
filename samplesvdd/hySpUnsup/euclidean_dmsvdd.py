"""
Euclidean Deep Multi-sphere SVDD (DMSVDD-style).

Structure follows Deep-Multi-Sphere-SVDD:
  - One encoder produces a single feature vector per sample.
  - z = W h in Euclidean latent space.
  - K learnable centers c_k in the same Euclidean latent space.
  - Nearest-sphere assignment: k* = argmin_k ||z - c_k||^2.
  - Hinge on assigned sphere: max(0, ||z-c_k*||^2 - R_k*^2), averaged with nu.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def dist_sq_z_to_centers(z: torch.Tensor, c_h: torch.Tensor, curvature: float | None = None) -> torch.Tensor:
    """z: (B, z_dim), c_h: (K, z_dim). Returns (B, K) squared Euclidean distance."""
    del curvature
    return ((z.unsqueeze(1) - c_h.unsqueeze(0)) ** 2).sum(dim=2)


class EuclideanDMSVDD(nn.Module):
    """Single shared Euclidean embedding + K Euclidean centers."""

    def __init__(self, backbone: nn.Module, rep_dim: int, z_dim: int, n_clusters: int, c: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.rep_dim = rep_dim
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.curvature = float(c)
        self._to_tangent = nn.Linear(rep_dim, z_dim, bias=True)
        self._c_raw = nn.Parameter(torch.randn(n_clusters, z_dim) * 0.02)

    @property
    def c_h(self) -> torch.Tensor:
        return self._c_raw

    def embed_h(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns z (B, z_dim) in Euclidean latent, rep, recon."""
        rep, recon = self.backbone(x)
        z = self._to_tangent(rep)
        return z, rep, recon

    def forward(self, x: torch.Tensor):
        rep, recon = self.backbone(x)
        z = self._to_tangent(rep)
        return z, recon


# Backward-compatible alias for existing imports.
HyperbolicDMSVDD = EuclideanDMSVDD
