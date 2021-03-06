from dataclasses import dataclass
import functools
from typing import Dict, Optional

import numpy as np
from PIL import Image
import torch

from . import config
from .datasets import ObjectDataset, stanfordcars, tesla
from .datasets.utils import tensor_to_pil
from .datatypes import (
    BinaryMask,
    ObjectTensorData,
    ObjectTensorDataBatch,
    ObjectWithMask,
    PILImage,
    Split,
    TensorToPILFn,
)
from .metrics import MetricCalculator, Metrics, iou
from .utils import get_device


@dataclass
class Baseline:
    ds: ObjectDataset
    ds_train: Optional[ObjectDataset]
    batch_size: int
    num_workers: int
    seed: int

    def __post_init__(self):
        self.random_state = np.random.RandomState(seed=self.seed)

    @property
    def device(self) -> torch.device:
        return get_device()

    @property
    def tensor_to_pil_fn(self) -> TensorToPILFn:
        return self.ds.transform_reverse_fn_objects

    def _get_fakes(self, real: ObjectTensorDataBatch, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def compute(self) -> Metrics:
        metrics_calc = MetricCalculator(
            device=self.device,
            tensor_to_image_fn=self.tensor_to_pil_fn,
        )
        dl = self.ds.get_dataloader(
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        for batch_idx, real in enumerate(dl):
            fake_object = self._get_fakes(real, batch_idx)
            metrics_calc.update(real, fake_object)
        return metrics_calc.compute()


class Baseline1(Baseline):
    def _random_rgb_image(self, w: int, h: int) -> PILImage:
        np_array = self.random_state.rand(w, h, 3) * 255
        return Image.fromarray(np_array.astype("uint8")).convert("RGB")

    def _get_fakes(self, real: ObjectTensorDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Random RGBs.
        """
        b, _, w, h = real.object.shape
        return torch.stack(
            [
                self.ds.transform_fn_objects(self._random_rgb_image(w, h))
                for _ in range(b)
            ]
        )


class Baseline2(Baseline):
    def _get_fakes(self, real: ObjectTensorDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Random image from train.
        """
        n = len(real.object)
        idxs = self.random_state.choice(range(len(self.ds_train)), n, replace=True)
        return torch.stack([self.ds_train[idx].object for idx in idxs])


class Baseline3(Baseline):
    @property
    def train_masks(self) -> Dict[int, BinaryMask]:
        return {
            idx: self.ds_train.data[idx].object_mask
            for idx in range(len(self.ds_train))
        }

    def _get_one_fake(self, real: ObjectTensorData) -> torch.Tensor:
        data = ObjectWithMask.from_tensor(
            real.object, self.ds_train.transform_reverse_fn_objects
        )
        mask = data.object_mask
        best = max(
            (iou(mask, train_mask), idx) for idx, train_mask in self.train_masks.items()
        )
        best_train_idx = best[1]
        return self.ds_train[best_train_idx].object

    def _get_fakes(self, real: ObjectTensorDataBatch, batch_idx: int) -> torch.Tensor:
        """
        Nearest image in train via IoU.
        """
        return torch.stack(
            [self._get_one_fake(real.get_idx(idx)) for idx in range(len(real))]
        )


@dataclass
class FromDisk(Baseline):
    path: str
    tensor_to_pil_fn_provided: Optional[TensorToPILFn]

    @property
    def tensor_to_pil_fn(self) -> TensorToPILFn:
        return self.tensor_to_pil_fn_provided

    def _get_one_fake(self, idx: int, batch_idx: int) -> torch.Tensor:
        data_on_disk = np.load(self.path, allow_pickle=True)
        idx_ds = batch_idx * self.batch_size + idx
        data_path = self.ds.data[idx_ds].object_image_path
        loc, name = str(data_path).split("/")[-2:]
        key_to_find = f"{loc}/{name}"
        if key_to_find in data_on_disk:
            return torch.tensor(data_on_disk[key_to_find])
        else:
            print(f"missing {key_to_find}")
            return torch.rand(3, 256, 256)

    def _get_fakes(self, real: ObjectTensorDataBatch, batch_idx: int) -> torch.Tensor:
        return torch.stack(
            [self._get_one_fake(idx, batch_idx) for idx in range(len(real))]
        )


FromDiskShubham = functools.partial(
    FromDisk,
    tensor_to_pil_fn_provided=tensor_to_pil((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
)


@dataclass(frozen=True)
class BaselinesCars:
    ds: str
    batch_size: int = config.baselines_tesla_batch_size
    num_workers: int = config.baselines_tesla_num_workers
    seed: int = config.seed
    baseline1: bool = True
    baseline2: bool = True
    baseline3: bool = True

    @property
    def dataset_fn(self):
        return dict(
            tesla=tesla.get_tesla_dataset,
            stanford_cars=stanfordcars.get_stanford_cars_dataset,
        )

    def get_dataset(self, split: Split) -> ObjectDataset:
        fn = self.dataset_fn[self.ds]
        return fn(split)

    def _baselines(self, split: Split) -> Dict[str, Metrics]:
        ds_train = self.get_dataset(Split.train)
        ds = self.get_dataset(split)
        args = dict(
            ds=ds,
            ds_train=ds_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return {
            f"baseline_{b_nbr}": b_cls(**args).compute()
            for b_include, b_nbr, b_cls in (
                (self.baseline1, 1, Baseline1),
                (self.baseline2, 2, Baseline2),
                (self.baseline3, 3, Baseline3),
            )
            if b_include
        }

    def validation(self) -> Dict[str, Metrics]:
        return self._baselines(Split.validation)

    def test(self) -> Dict[str, Metrics]:
        return self._baselines(Split.test)

    def train(self) -> Dict[str, Metrics]:
        return self._baselines(Split.train)
