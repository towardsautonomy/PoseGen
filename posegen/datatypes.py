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


class Architecture(Enum):
    un_conditional = "un_conditional"
    pose_only = "pose_only"
    pose_and_car = "pose_and_car"
    pose_car_background = "pose_car_background"


@dataclass(frozen=False)
class Parts:
    car: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None
