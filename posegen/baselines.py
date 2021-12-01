from dataclasses import dataclass
import functools
from typing import Dict

import numpy as np
from PIL import Image
import torch

from . import config
from .datasets import CarDataset
from .datatypes import CarTensorData, PILImage, Split
from .metrics import MetricCalculator, Metrics
from .datasets import tesla
from .utils import get_device


@dataclass
class Baseline:
    ds: CarDataset
    ds_train: CarDataset
    batch_size: int
    num_workers: int
    seed: int

    def __post_init__(self):
        self.random_state = np.random.RandomState(seed=self.seed)

    @property
    def device(self) -> torch.device:
        return get_device()

    def _get_fakes(self, real: CarTensorData) -> torch.Tensor:
        raise NotImplementedError

    def compute(self) -> Metrics:
        metrics_calc = MetricCalculator(self.device, self.ds.transform_reverse_fn_cars)
        dl = self.ds.get_dataloader(
            batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        for real in dl:
            fake_car = self._get_fakes(real)
            metrics_calc.update(real, fake_car)
        return metrics_calc.compute()


@dataclass
class Baseline1(Baseline):
    def _random_rgb_image(self, w: int, h: int) -> PILImage:
        np_array = self.random_state.rand(w, h, 3) * 255
        return Image.fromarray(np_array.astype("uint8")).convert("RGB")

    def _get_fakes(self, real: CarTensorData) -> torch.Tensor:
        """
        Random RGBs.
        """
        b, _, w, h = real.car.shape
        return torch.stack(
            [
                self.ds_train.transform_fn_cars(self._random_rgb_image(w, h))
                for _ in range(b)
            ]
        )


class Baseline2(Baseline):
    def _get_fakes(self, real: CarTensorData) -> torch.Tensor:
        """
        Random image from train.
        """
        n = len(real.car)
        idxs = self.random_state.choice(range(len(self.ds_train)), n, replace=True)
        return torch.stack([self.ds_train[idx].car for idx in idxs])


# class Baseline3(Baseline):
#     def _get_fakes(self, real: CarTensorData) -> torch.Tensor:
#         """
#         Nearest image in train. Almost doesn't make sense.
#         """
#         pass


@dataclass(frozen=True)
class BaselinesTesla:
    batch_size: int = config.baselines_tesla_batch_size
    num_workers: int = config.baselines_tesla_num_workers
    seed: int = config.seed

    def _baselines_tesla(self, split: Split) -> Dict[str, Metrics]:
        fn = functools.partial(tesla.get_tesla_dataset, random_pose=False)
        ds_train = fn(split=Split.train)
        ds = fn(split=split)
        args = dict(
            ds=ds,
            ds_train=ds_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            seed=self.seed,
        )
        return dict(
            baseline1=Baseline1(**args).compute(),
            baseline2=Baseline2(**args).compute(),
        )

    def validation(self) -> Dict[str, Metrics]:
        return self._baselines_tesla(Split.validation)

    def test(self) -> Dict[str, Metrics]:
        return self._baselines_tesla(Split.test)

    def train(self) -> Dict[str, Metrics]:
        return self._baselines_tesla(Split.train)
