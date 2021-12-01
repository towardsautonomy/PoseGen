import hashlib
from pathlib import Path

import torch
import torchvision


def get_md5(path: Path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class DeNormalize(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    Took from: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/6
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
