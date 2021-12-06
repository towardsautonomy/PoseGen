import hashlib
import logging

import numpy as np
import torch
from pathlib import Path
from typing import Union

import mmh3


def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    return torch.device("cuda" if cuda_available() else "cpu")


def binarize_pose(
    pose: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    # since masks are {0,1}^{w x h} the dataset mean is > 0
    # and the 1 and 0 values will be mapped to > 0 and < 0
    return pose >= 0


def get_md5(path: Path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def get_mmh3(s: str, seed: int) -> int:
    return mmh3.hash(s, seed=seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(name)
