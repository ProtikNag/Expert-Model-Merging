"""RotatedMNIST data pipeline.

Each expert sees the *same* 10-class label space but with input images rotated
by a fixed angle. This means (i) all experts share the output head, which is
required for parameter-space merging without head re-alignment, and (ii) a
merged model can be evaluated on every rotation to measure capability retention.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, Subset


class RotatedMNIST(Dataset):
    """MNIST with every image rotated by a fixed angle ``theta_deg``.

    Parameters
    ----------
    root:
        Filesystem location where torchvision will cache MNIST.
    theta_deg:
        Rotation angle in degrees applied to every image.
    train:
        If True, use the training split; else the test split.
    subset:
        Optional upper bound on the number of samples (for CPU pilots).
    seed:
        Seed used for deterministic subsampling.
    """

    def __init__(self,
                 root: str,
                 theta_deg: float,
                 train: bool = True,
                 subset: Optional[int] = None,
                 seed: int = 0) -> None:
        self.theta_deg = theta_deg
        base = torchvision.datasets.MNIST(
            root=root, train=train, download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        if subset is not None and subset < len(base):
            gen = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(base), generator=gen)[:subset].tolist()
            self.base: Dataset = Subset(base, idx)
        else:
            self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]
        if self.theta_deg != 0.0:
            # ``rotate`` expects a tensor in [C,H,W] and fills with 0 (black).
            img = TF.rotate(img, self.theta_deg, fill=[0.0])
        return img, int(label)


def build_loader(root: str,
                 theta_deg: float,
                 train: bool,
                 batch_size: int,
                 subset: Optional[int] = None,
                 seed: int = 0,
                 shuffle: Optional[bool] = None) -> DataLoader:
    """Construct a DataLoader over a RotatedMNIST split."""
    ds = RotatedMNIST(root, theta_deg, train=train, subset=subset, seed=seed)
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_pretrain_loader(root: str,
                          batch_size: int,
                          subset: Optional[int] = None,
                          seed: int = 0) -> DataLoader:
    """Un-rotated MNIST loader used to train the shared initialization."""
    return build_loader(root, theta_deg=0.0, train=True,
                        batch_size=batch_size, subset=subset, seed=seed)
