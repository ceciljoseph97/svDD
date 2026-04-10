import os
import pickle
import tarfile
import urllib.request
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download_and_extract(root: str):
    os.makedirs(root, exist_ok=True)
    tar_path = os.path.join(root, "cifar-10-python.tar.gz")
    out_dir = os.path.join(root, "cifar-10-batches-py")
    if os.path.isdir(out_dir):
        return
    if not os.path.isfile(tar_path):
        urllib.request.urlretrieve(URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root)


def _load_batch(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    x = data["data"]  # (N,3072)
    y = np.array(data["labels"], dtype=np.int64)
    x = x.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    return x, y


class CIFAR10RawDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        digits: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        download: bool = True,
    ):
        assert split in ("train", "test")
        if download:
            _download_and_extract(root)
        base = os.path.join(root, "cifar-10-batches-py")
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Missing CIFAR folder: {base}")

        if split == "train":
            xs, ys = [], []
            for i in range(1, 6):
                x, y = _load_batch(os.path.join(base, f"data_batch_{i}"))
                xs.append(x)
                ys.append(y)
            x = np.concatenate(xs, axis=0)
            y = np.concatenate(ys, axis=0)
        else:
            x, y = _load_batch(os.path.join(base, "test_batch"))

        if digits is not None:
            digits = [int(d) for d in digits]
            m = np.isin(y, np.array(digits, dtype=np.int64))
            x = x[m]
            y = y[m]

        # Standard CIFAR10 normalization constants.
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
        x = (x - mean) / std

        if max_samples is not None:
            x = x[: int(max_samples)]
            y = y[: int(max_samples)]

        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx

