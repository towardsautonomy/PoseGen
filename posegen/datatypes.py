from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image

from .utils import get_md5


BinaryMask = Frame = NumpyNdArray = np.ndarray
PILImage = Image.Image
TensorToPILFn = Callable[[torch.Tensor], PILImage]
PILToTensorFn = Callable[[PILImage], torch.Tensor]


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"


@dataclass
class Lambdas:
    gan: float
    full: float
    obj: float
    background: Optional[float] = None


@dataclass
class ObjectTensorData:
    object: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None


@dataclass
class ObjectWithMask:
    width: int
    height: int
    object_image_path: Path = None
    object_image_frame: PILImage = None
    mask_random: Optional[BinaryMask] = None

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        tensor_to_image_fn: TensorToPILFn,
    ) -> "ObjectWithMask":
        _, w, h = tensor.shape
        image = tensor_to_image_fn(tensor)
        return ObjectWithMask(w, h, object_image_frame=image)

    @property
    def mask(self) -> BinaryMask:
        return self.mask_random if self.mask_random is not None else self.object_mask

    @property
    def mask_path(self) -> Optional[Path]:
        if self.object_image_path:
            md5 = get_md5(self.object_image_path)
            # TODO: add height and width to this
            # TODO: put in ~/.cache/ instead
            # TODO: rename car => object
            return Path("/tmp") / Path(f"car_mask_{md5}.npy")

    @property
    def object_mask(self) -> BinaryMask:
        from .instance_segmentation import get_mask

        path = self.mask_path
        if path is not None and path.exists():
            return np.load(str(path), allow_pickle=True)
        mask = get_mask(image=self.object_image)
        if mask is None:
            mask = np.zeros((self.width, self.height), dtype=bool)
        if path:
            path.parent.mkdir(exist_ok=True)
            np.save(str(path), mask)
        return mask

    @property
    def object_image(self) -> PILImage:
        return (
            self.object_image_frame
            if self.object_image_frame is not None
            else self._get_image_from_path()
        )

    def _get_image_from_path(self) -> PILImage:
        img = Image.open(self.object_image_path)
        return img.resize((self.width, self.height))


@dataclass
class ObjectTensorDataBatch:
    object: torch.Tensor
    pose: torch.Tensor
    background: Optional[torch.Tensor] = None

    @classmethod
    def collate_fn(cls, batch: List[ObjectTensorData]) -> "ObjectTensorDataBatch":
        return cls(
            object=cls._get(batch, "object"),
            pose=cls._get(batch, "pose"),
            background=cls._get(batch, "background"),
        )

    @staticmethod
    def _get(batch: List[ObjectTensorData], name: str) -> Optional[torch.Tensor]:
        values = [getattr(item, name) for item in batch]
        return torch.stack(values) if values[0] is not None else None

    def get_idx(self, idx: int) -> "ObjectTensorData":
        """
        Assumes that we have a batch of data so we can get individual items.
        """
        return ObjectTensorData(
            object=self.object[idx],
            pose=self.pose[idx],
            background=self.background[idx] if self.has_background else None,
        )

    def to(self, device: torch.device) -> "ObjectTensorData":
        return ObjectTensorData(
            object=self.object.to(device),
            pose=self.pose.to(device),
            background=self.background.to(device) if self.has_background else None,
        )

    @property
    def has_background(self) -> bool:
        return self.background is not None

    def __len__(self) -> int:
        return len(self.object)
