from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
from PIL import Image

BinaryMask = Frame = NumpyNdArray = np.ndarray
PILImage = Image.Image


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"


@dataclass(frozen=False)
class Parts:
    car: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None
