import functools
from typing import Tuple

import pandas as pd

from .dataset import CarDataset
from .. import config
from ..datatypes import Split
from ..utils import get_mmh3


class TeslaDataset(CarDataset):
    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        # this could be done offline and stored so we don't have to do it on-the-fly
        # in general this is not very expensive though
        # the splitting is tied to the MD5 of the file and is invariant to the location on disk

        # TODO: use decimals (instead of the hardcoded 80/10/10) to make this more general
        splits = (
            df.md5.map(functools.partial(get_mmh3, seed=self.seed))
            .map(lambda x: x % 10)
            .map({8: Split.validation, 9: Split.test})
            .fillna(Split.train)
        )
        return df.assign(split=splits)


def get_tesla_dataset(
    random_pose: bool,
    split: Split,
    path: str = config.tesla_path_dataset,
    seed: int = config.seed_data,
    n_pose_pairs: int = config.n_pose_pairs,
    extension: str = config.tesla_extension,
    width: int = config.width,
    height: int = config.height,
    transforms_mean_cars: Tuple[float, ...] = config.transforms_mean_cars_tesla,
    transforms_std_cars: Tuple[float, ...] = config.transforms_std_cars_tesla,
    transforms_mean_poses: Tuple[float, ...] = config.transforms_mean_poses_tesla,
    transforms_std_poses: Tuple[float, ...] = config.transforms_std_poses_tesla,
) -> CarDataset:
    return TeslaDataset(
        path=path,
        random_pose=random_pose,
        n_pose_pairs=n_pose_pairs,
        split=split,
        seed=seed,
        extension=extension,
        width=width,
        height=height,
        transforms_mean_cars=transforms_mean_cars,
        transforms_std_cars=transforms_std_cars,
        transforms_mean_poses=transforms_mean_poses,
        transforms_std_poses=transforms_std_poses,
    )
