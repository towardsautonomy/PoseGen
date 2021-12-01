import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import mmh3
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import get_md5
from ..datatypes import BinaryMask, NumpyNdArray, PILImage, Split
from ..instance_segmentation import get_mask


@dataclass(frozen=True)
class CarWithMask:
    width: int
    height: int
    car_image_path: Path = None
    car_image_frame: NumpyNdArray = None
    mask_random: Optional[BinaryMask] = None
    category_idx: Optional[int] = None

    @property
    def mask(self) -> BinaryMask:
        return self.mask_random if self.mask_random is not None else self.car_mask

    @property
    def mask_path(self) -> Optional[Path]:
        if self.car_image_path:
            md5 = get_md5(self.car_image_path)
            # TODO: add height and width to this
            return Path("/tmp") / Path(f"car_mask_{md5}.npy")

    @property
    @functools.lru_cache()
    def car_mask(self) -> BinaryMask:
        path = self.mask_path
        if path.exists():
            return np.load(open(path, "rb"), allow_pickle=True)
        mask = get_mask(image=self.car_image)
        if mask is None:
            mask = np.zeros((self.width, self.height), dtype=bool)
        if path:
            path.parent.mkdir(exist_ok=True)
            np.save(open(path, "wb"), mask)
        return mask

    @staticmethod
    def _frame_to_image(frame: NumpyNdArray) -> PILImage:
        # TODO: this will be a generated image. Is this the right way to read?
        return Image.fromarray(frame.astype("uint8"), "RGB")

    @property
    def car_image(self) -> PILImage:
        if self.car_image_frame:
            return self._frame_to_image(self.car_image_frame)
        img = Image.open(self.car_image_path)
        return img.resize((self.width, self.height))


@dataclass(frozen=False)
class CarDataset(Dataset):
    path: str
    random_pose: bool
    n_pose_pairs: int
    split: Split
    seed: int
    extension: str
    width: int
    height: int
    transforms_mean_cars: Tuple[float, ...]
    transforms_std_cars: Tuple[float, ...]
    transforms_mean_poses: Tuple[float, ...]
    transforms_std_poses: Tuple[float, ...]

    def __post_init__(self):
        if not self.random_pose and self.n_pose_pairs > 1:
            raise ValueError("each image has a unique pose")

        self._np_rand_state = np.random.RandomState(self.seed)
        self._df = self._split(self.path, self.extension, self.seed)
        self._df_split = self._df[self._df.split == self.split].sort_values(by="md5")
        self._cars = (
            CarWithMask(car_image_path=path, width=self.width, height=self.height)
            for path in self._df_split.path
        )
        # remove any images without a mask from the set
        self._cars = [car for car in self._cars if car.car_mask.sum() > 0]
        self._data = (
            self._construct_random_poses(self._cars, self.n_pose_pairs)
            if self.random_pose
            else self._cars
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self._data[idx]
        car = self.transform_fn_cars(item.car_image)
        mask_rgb = self._mask_to_rgb(item.mask)
        pose = self.transform_fn_poses(mask_rgb)
        return car, pose

    @staticmethod
    def _mask_to_rgb(mask: BinaryMask) -> PILImage:
        return Image.fromarray(mask).convert("RGB")

    @property
    def transform_fn_cars(self) -> Callable[[PILImage], torch.Tensor]:
        return self._transform_fn(self.transforms_mean_cars, self.transforms_std_cars)

    @property
    def transform_fn_poses(self) -> Callable[[PILImage], torch.Tensor]:
        return self._transform_fn(self.transforms_mean_poses, self.transforms_std_poses)

    @staticmethod
    def _transform_fn(
        mean: Tuple[float, ...], std: Tuple[float, ...]
    ) -> Callable[[PILImage], torch.Tensor]:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _construct_random_poses(
        self, cars: Sequence[CarWithMask], n_pose_pairs: int
    ) -> Sequence[CarWithMask]:
        # for each car randomly pair it with `n_pose_pairs` pose masks selected from the same set.
        random_poses = self._np_rand_state.choice(cars, len(cars) * n_pose_pairs)
        return [
            CarWithMask(
                car_image_path=car.car_image_path,
                mask_random=random_poses[idx * n_pose_pairs + idx_n].mask,
                width=self.width,
                height=self.height,
            )
            for idx, car in enumerate(cars)
            for idx_n in range(n_pose_pairs)
        ]

    @staticmethod
    def _get_mmh3(s: str, seed: int) -> str:
        return mmh3.hash(s, seed=seed)

    def _split(self, path_base: str, ext: str, seed: int) -> pd.DataFrame:
        # this could be done offline and stored so we don't have to do it on-the-fly
        # in general this is not very expensive though
        # the splitting is tied to the MD5 of the file and is invariant to the location on disk
        df = pd.DataFrame(
            {"path": path, "md5": get_md5(path)}
            for path in Path(path_base).rglob(f"*.{ext}")
        )
        # TODO: for stanford cars this should be at category level, so if it works we can claim generalization
        # TODO: use decimals (instead of the hardcoded 80/10/10) to make this more general
        splits = (
            df.md5.map(functools.partial(self._get_mmh3, seed=seed))
            .map(lambda x: x % 10)
            .map({8: Split.validation, 9: Split.test})
            .fillna(Split.train)
        )
        return df.assign(split=splits)
