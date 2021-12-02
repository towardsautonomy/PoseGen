from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image

from .datasets.utils import get_md5


BinaryMask = Frame = NumpyNdArray = np.ndarray
PILImage = Image.Image
TensorToPILFn = Callable[[torch.Tensor], PILImage]
PILToTensorFn = Callable[[PILImage], torch.Tensor]


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"


class Architecture(Enum):
    unconditional = "unconditional"
    pose_only = "pose_only"
    pose_and_car = "pose_and_car"
    pose_car_background = "pose_car_background"


@dataclass
class CarTensorData:
    car: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None


@dataclass
class CarWithMask:
    width: int
    height: int
    car_image_path: Path = None
    car_image_frame: PILImage = None
    mask_random: Optional[BinaryMask] = None
    category_idx: Optional[int] = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        tensor_to_image_fn: TensorToPILFn,
    ) -> "CarWithMask":
        _, w, h = tensor.shape
        image = tensor_to_image_fn(tensor)
        return CarWithMask(w, h, car_image_frame=image)

    @property
    def mask(self) -> BinaryMask:
        return self.mask_random if self.mask_random is not None else self.car_mask

    @property
    def mask_path(self) -> Optional[Path]:
        if self.car_image_path:
            md5 = get_md5(self.car_image_path)
            # TODO: add height and width to this
            # TODO: put in ~/.cache/ instead
            return Path("/tmp") / Path(f"car_mask_{md5}.npy")

    @property
    def car_mask(self) -> BinaryMask:
        from .instance_segmentation import get_mask

        path = self.mask_path
        if path is not None and path.exists():
            return np.load(open(path, "rb"), allow_pickle=True)
        mask = get_mask(image=self.car_image)
        if mask is None:
            mask = np.zeros((self.width, self.height), dtype=bool)
        if path:
            path.parent.mkdir(exist_ok=True)
            np.save(open(path, "wb"), mask)
        return mask

    @property
    def car_image(self) -> PILImage:
        return (
            self.car_image_frame
            if self.car_image_frame is not None
            else self._get_image_from_path()
        )

    def _get_image_from_path(self) -> PILImage:
        img = Image.open(self.car_image_path)
        return img.resize((self.width, self.height))


@dataclass
class CarTensorDataBatch:
    car: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None

    @classmethod
    def collate_fn(cls, batch: List[CarTensorData]) -> "CarTensorDataBatch":
        return cls(
            car=cls._get(batch, "car"),
            pose=cls._get(batch, "pose"),
            background=cls._get(batch, "background"),
        )

    @staticmethod
    def _get(batch: List[CarTensorData], name: str) -> Optional[torch.Tensor]:
        values = [getattr(item, name) for item in batch]
        return torch.stack(values) if values[0] is not None else None

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

    @property
    def has_background(self) -> bool:
        return self.background is not None

    def __len__(self) -> int:
        return len(self.car)
