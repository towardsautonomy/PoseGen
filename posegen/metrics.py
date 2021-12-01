from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics import CosineSimilarity, FID

from .datatypes import BinaryMask
from .datasets.dataset import CarWithMask


@dataclass(frozen=False)
class IoU:
    real_poses_list: List[torch.Tensor]
    fake_images_list: List[torch.Tensor]

    def __post_init__(self):
        # TODO: this may cause OOM => do batches at a time
        self.real_poses = dim_zero_cat(self.real_poses_list)
        self.fake_images = dim_zero_cat(self.fake_images_list)
        self.n = len(self.real_poses)
        if self.n != len(self.fake_images):
            raise ValueError("unequal number of real poses and fake images")

    @staticmethod
    def iou(m1: BinaryMask, m2: BinaryMask) -> float:
        num = (m1 & m2).sum()
        den = (m1 | m2).sum()
        return num / den if den > 0 else 0

    def _compute_one(self, idx: int) -> float:
        # TODO: all three channels the same?
        pose = self.real_poses[idx].cpu().numpy()[0]
        fake = self.fake_images[idx].cpu().numpy()
        w, h = pose.shape
        car = CarWithMask(w, h, car_image_frame=fake)
        return self.iou(car.car_mask, pose)

    def compute(self) -> float:
        ious = [self._compute_one(idx) for idx in range(self.n)]
        return pd.Series(ious).mean()


class FIDBetter(FID):
    def sim(self) -> float:
        real, fake = dim_zero_cat(self.real_features), dim_zero_cat(self.fake_features)
        cos = CosineSimilarity(reduction="mean")
        return cos(real, fake).item()


iou = IoU.iou
