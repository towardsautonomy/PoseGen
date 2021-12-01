from typing import Tuple

from .dataset import CarDataset
from ..datatypes import Split
from .. import config


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
    return CarDataset(
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
