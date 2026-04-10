import glob
import hashlib
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


MIN_MAX = [
    (-0.8826567065619495, 9.001545489292527),
    (-0.6661464580883915, 20.108062262467364),
    (-0.7820454743183202, 11.665100841080346),
    (-0.7645772083211267, 12.895051191467457),
    (-0.7253923114302238, 12.683235701611533),
    (-0.7698501867861425, 13.103278415430502),
    (-0.778418217980696, 10.457837397569108),
    (-0.7129780970522351, 12.057777597673047),
    (-0.8280402650205075, 10.581538445782988),
    (-0.7369959242164307, 10.697039838804978),
]


def global_contrast_normalization(x: torch.Tensor, scale: str = "l1") -> torch.Tensor:
    assert scale in ("l1", "l2")
    x = x.clone()
    n_features = int(np.prod(x.shape))
    x -= torch.mean(x)
    if scale == "l1":
        x_scale = torch.mean(torch.abs(x))
    else:
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x


def preprocess_batch_by_digit_minmax(x_raw: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
    x_gcn = global_contrast_normalization(x_raw.clone(), scale="l1")
    x_scaled = torch.empty_like(x_gcn)
    for k in range(10):
        mask = digits == k
        if torch.any(mask):
            vmin, vmax = MIN_MAX[k]
            x_scaled[mask] = (x_gcn[mask] - float(vmin)) / float(vmax - vmin)
    return x_scaled


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _in_split(path: str, train_fraction: float) -> bool:
    return ((_stable_hash_int(path) % 10000) / 10000.0) < train_fraction


class MNISTDigitsProcessedRawDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        train_fraction: float = 0.8,
        digits: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
    ):
        assert split in ("train", "test")
        self.root_dir = root_dir
        self.split = split
        self.train_fraction = float(train_fraction)
        digits = list(range(10)) if digits is None else [int(d) for d in digits]

        all_paths: List[Tuple[str, int]] = []
        for d in digits:
            cls_dir = os.path.join(self.root_dir, f"class_{d}")
            for p in sorted(glob.glob(os.path.join(cls_dir, "*.png"))):
                all_paths.append((p, d))

        if split == "train":
            self.samples = [(p, d) for (p, d) in all_paths if _in_split(p, self.train_fraction)]
        else:
            self.samples = [(p, d) for (p, d) in all_paths if not _in_split(p, self.train_fraction)]

        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
        if len(self.samples) == 0:
            raise RuntimeError("No samples found in MNIST_processed with current split/filter.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, digit = self.samples[idx]
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        x_raw = torch.from_numpy(arr).unsqueeze(0)
        return x_raw, int(digit), idx


class MNIST_LeNet_SVDDIAE(nn.Module):
    def __init__(self, rep_dim: int = 32):
        super().__init__()
        assert rep_dim % 16 == 0
        self.rep_dim = rep_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-4, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.deconv1 = nn.ConvTranspose2d(rep_dim // 16, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-4, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-4, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decode(self, rep: torch.Tensor) -> torch.Tensor:
        x = rep.view(rep.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        rep = self.encode(x)
        recon = self.decode(rep)
        return rep, recon


def recon_mse_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.sum((recon - x) ** 2, dim=tuple(range(1, recon.dim()))))

