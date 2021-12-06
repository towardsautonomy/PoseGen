from dataclasses import dataclass, field
from typing import List

import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics import CosineSimilarity, FID, IS, KID

from .datatypes import (
    BinaryMask,
    ObjectTensorData,
    ObjectTensorDataBatch,
    ObjectWithMask,
    TensorToPILFn,
)
from .utils import binarize_pose


@dataclass(frozen=True)
class Metrics:
    inception_score: float
    fid: float
    kid: float
    inception_sim: float
    iou: float


@dataclass(frozen=False)
class IoU:
    tensor_to_image_fn: TensorToPILFn
    reals: List[ObjectTensorDataBatch] = field(default_factory=list)
    fakes: List[torch.Tensor] = field(default_factory=list)

    def update(self, real: ObjectTensorDataBatch, fake_object: torch.Tensor) -> None:
        if len(real.object) != len(fake_object):
            raise ValueError("unequal number of real and fake images")
        print(type(real), type(fake_object))
        self.reals.append(real)
        self.fakes.append(fake_object)

    def compute(self) -> float:
        ious = [
            self._compute_one(real_batch.get_idx(idx), fake_batch[idx])
            for real_batch, fake_batch in zip(self.reals, self.fakes)
            for idx in range(len(fake_batch))
        ]
        return pd.Series(ious).mean()

    @staticmethod
    def iou(m1: BinaryMask, m2: BinaryMask) -> float:
        num = (m1 & m2).sum()
        den = (m1 | m2).sum()
        return num / den if den > 0 else 0

    def _compute_one(self, real: ObjectTensorData, fake: torch.Tensor) -> float:
        # all three channels the same? looks like it
        real_pose = real.pose.cpu().numpy()[0]
        real_pose_binary = binarize_pose(real_pose)
        fake = ObjectWithMask.from_tensor(fake, self.tensor_to_image_fn)
        return self.iou(fake.object_mask, real_pose_binary)


class FIDBetter(FID):
    def sim(self) -> float:
        real, fake = dim_zero_cat(self.real_features), dim_zero_cat(self.fake_features)
        cos = CosineSimilarity(reduction="mean")
        return cos(real, fake).item()


@dataclass
class MetricCalculator:
    device: torch.device
    tensor_to_image_fn: TensorToPILFn

    def __post_init__(self):
        # TODO: what is this magic 32?
        self.fid_obj = FIDBetter().to(self.device)
        self.is_obj = IS().to(self.device)
        self.kid_obj = KID(subset_size=32).to(self.device)
        self.iou_obj = IoU(self.tensor_to_image_fn)

    def update(self, real: ObjectTensorDataBatch, fake_object: torch.Tensor) -> None:
        reals_inception = self.prepare_data_for_inception(real.object, self.device)
        fakes_inception = self.prepare_data_for_inception(fake_object, self.device)
        self.is_obj.update(fakes_inception)
        self.fid_obj.update(reals_inception, real=True)
        self.fid_obj.update(fakes_inception, real=False)
        self.kid_obj.update(reals_inception, real=True)
        self.kid_obj.update(fakes_inception, real=False)
        self.iou_obj.update(real, fake_object)

    def compute(self) -> Metrics:
        return Metrics(
            inception_score=self.is_obj.compute()[0].item(),
            fid=self.fid_obj.compute().item(),
            kid=self.kid_obj.compute()[0].item(),
            iou=self.iou_obj.compute(),
            inception_sim=self.fid_obj.sim(),
        )

    @staticmethod
    def prepare_data_for_inception(x, device):
        """
        Preprocess data to be feed into the Inception model.
        """

        x = F.interpolate(x, 299, mode="bicubic", align_corners=False)
        minv, maxv = float(x.min()), float(x.max())
        x.clamp_(min=minv, max=maxv).add_(-minv).div_(maxv - minv + 1e-5)
        x.mul_(255).add_(0.5).clamp_(0, 255)
        return x.to(device).to(torch.uint8)


iou = IoU.iou
