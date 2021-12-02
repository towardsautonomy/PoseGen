import functools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from torch.utils.data import DataLoader

import mmh3
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..datatypes import (
    BinaryMask,
    CarTensorData,
    CarWithMask,
    CarTensorDataBatch,
    PILImage,
    Split,
    PILToTensorFn,
    TensorToPILFn,
)
from .utils import DeNormalize
from ..utils import get_md5


@dataclass
class CarDataset(Dataset):
    """
    TODO: generalize this to ObjectDataset
    """

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
        self.data = (
            self._construct_random_poses(self._cars, self.n_pose_pairs)
            if self.random_pose
            else self._cars
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> CarTensorData:
        item = self.data[idx]
        car = self.transform_fn_cars(item.car_image)
        mask_rgb = self._mask_to_rgb(item.mask)
        pose = self.transform_fn_poses(mask_rgb)
        return CarTensorData(car=car, pose=pose)

    def get_dataloader(
        self, batch_size: int, shuffle: bool, num_workers: int
    ) -> DataLoader:
        if shuffle and self.split != Split.train:
            raise ValueError("don't shuffle datasets other than train")
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=CarTensorDataBatch.collate_fn,
        )

    @staticmethod
    def _mask_to_rgb(mask: BinaryMask) -> PILImage:
        return Image.fromarray(mask).convert("RGB")

    @property
    def transform_fn_cars(self) -> PILToTensorFn:
        return self._transform_fn(self.transforms_mean_cars, self.transforms_std_cars)

    @property
    def transform_reverse_fn_cars(self) -> TensorToPILFn:
        return transforms.Compose(
            [
                DeNormalize(self.transforms_mean_cars, self.transforms_std_cars),
                transforms.ToPILImage(),
            ]
        )

    @property
    def transform_fn_poses(self) -> PILToTensorFn:
        return self._transform_fn(self.transforms_mean_poses, self.transforms_std_poses)

    @staticmethod
    def _transform_fn(mean: Tuple[float, ...], std: Tuple[float, ...]) -> PILToTensorFn:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def _construct_random_poses(
        self, cars: Sequence[CarWithMask], n_pose_pairs: int
    ) -> List[CarWithMask]:
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
