from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from ..datatypes import (
    BinaryMask,
    ObjectTensorData,
    ObjectWithMask,
    ObjectTensorDataBatch,
    PILImage,
    Split,
    PILToTensorFn,
    TensorToPILFn,
)
from .utils import DeNormalize
from ..utils import get_md5


@dataclass
class ObjectDataset(Dataset):
    path: str
    random_pose: bool
    n_pose_pairs: int
    split: Split
    seed: int
    extension: str
    width: int
    height: int
    transforms_mean_objects: Tuple[float, ...]
    transforms_std_objects: Tuple[float, ...]
    transforms_mean_poses: Tuple[float, ...]
    transforms_std_poses: Tuple[float, ...]

    def __post_init__(self):
        if not self.random_pose and self.n_pose_pairs > 1:
            raise ValueError("each image has a unique pose")

        self._np_rand_state = np.random.RandomState(self.seed)
        self._df_path_md5 = self._get_df_path_md5(self.path, self.extension)
        self._df = self._split(self._df_path_md5)
        self._df_split = self._df[self._df.split == self.split].sort_values(by="md5")
        self._objects = (
            ObjectWithMask(object_image_path=path, width=self.width, height=self.height)
            for path in self._df_split.path
        )
        # remove any images without a mask from the set
        self._objects = [obj for obj in self._objects if obj.object_mask.sum() > 0]
        self.data = (
            self._construct_random_poses(self._objects, self.n_pose_pairs)
            if self.random_pose
            else self._objects
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> ObjectTensorData:
        item = self.data[idx]
        obj = self.transform_fn_objects(item.object_image)
        mask_rgb = self._mask_to_rgb(item.mask)
        pose = self.transform_fn_poses(mask_rgb)
        return ObjectTensorData(object=obj, pose=pose)

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
            collate_fn=ObjectTensorDataBatch.collate_fn,
        )

    @staticmethod
    def _mask_to_rgb(mask: BinaryMask) -> PILImage:
        return Image.fromarray(mask).convert("RGB")

    @property
    def transform_fn_objects(self) -> PILToTensorFn:
        return self._transform_fn(
            self.transforms_mean_objects, self.transforms_std_objects
        )

    @property
    def transform_reverse_fn_objects(self) -> TensorToPILFn:
        return transforms.Compose(
            [
                DeNormalize(self.transforms_mean_objects, self.transforms_std_objects),
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
        self, objects: Sequence[ObjectWithMask], n_pose_pairs: int
    ) -> List[ObjectWithMask]:
        # for each object randomly pair it with `n_pose_pairs` pose masks selected from the same set.
        random_poses = self._np_rand_state.choice(objects, len(objects) * n_pose_pairs)
        return [
            ObjectWithMask(
                object_image_path=obj.object_image_path,
                mask_random=random_poses[idx * n_pose_pairs + idx_n].mask,
                width=self.width,
                height=self.height,
            )
            for idx, obj in enumerate(objects)
            for idx_n in range(n_pose_pairs)
        ]

    @staticmethod
    def _get_df_path_md5(path_base: str, ext: str) -> pd.DataFrame:
        return pd.DataFrame(
            {"path": path, "md5": get_md5(path)}
            for path in Path(path_base).rglob(f"*.{ext}")
        )

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
