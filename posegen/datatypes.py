from enum import Enum
from typing import Optional, NamedTuple

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
    unconditional = "unconditional"
    pose_only = "pose_only"
    pose_and_car = "pose_and_car"
    pose_car_background = "pose_car_background"


class CarTensorData(NamedTuple):
    car: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None

    @property
    def has_background(self) -> bool:
        return self.background is not None

    def get_idx(self, idx: int) -> "CarTensorData":
        """
        Assumes that we have a batch of data so we can get individual items.
        """
        return CarTensorData(
            car=self.car[idx],
            pose=self.pose[idx],
            background=self.background[idx] if self.has_background else None,
        )

    def to(self, device: torch.device) -> "CarTensorData":
        return CarTensorData(
            car=self.car.to(device),
            pose=self.pose.to(device),
            background=self.background.to(device) if self.has_background else None,
        )
