import torch
import torch.nn as nn


class CifarConvAE(nn.Module):
    def __init__(self, rep_dim: int = 128):
        super().__init__()
        self.rep_dim = rep_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc_enc = nn.Linear(128 * 4 * 4, rep_dim, bias=False)
        self.fc_dec = nn.Linear(rep_dim, 128 * 4 * 4, bias=False)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_enc(h.view(h.size(0), -1))

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 128, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return z, recon


def recon_mse_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((recon - x) ** 2, dim=(1, 2, 3)))

