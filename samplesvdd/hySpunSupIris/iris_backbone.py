"""Small MLP autoencoder for tabular Iris (4 features)."""
import torch
import torch.nn as nn


class IrisMLPSVDDIAE(nn.Module):
    def __init__(self, in_dim: int = 4, rep_dim: int = 32, hidden: int = 64):
        super().__init__()
        assert rep_dim > 0
        self.rep_dim = rep_dim
        self.in_dim = in_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, rep_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor):
        rep = self.encoder(x)
        recon = self.decoder(rep)
        return rep, recon


def recon_mse_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((recon - x) ** 2, dim=tuple(range(1, recon.dim()))))
