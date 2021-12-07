from dataclasses import dataclass
import functools
from typing import Dict, Tuple

import pandas as pd
from scipy.io import loadmat

from .dataset import ObjectDataset
from .. import config
from ..datatypes import Split
from ..utils import get_mmh3


@dataclass
class StanfordCars(ObjectDataset):
    path_meta: str
    path_train: str
    path_test: str

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        lookup = self.lookup
        splits = (
            df.path.astype(str)
            .str.extract(r"_(?P<split>.+)/(?P<path>.+)$")
            .apply(lambda x: lookup[(x.split, x.path)], axis=1)
        )
        return df.assign(split=splits)

    @property
    def class_lookup(self) -> Dict[int, str]:
        return {
            idx + 1: cls.item()
            for idx, cls in enumerate(loadmat(self.path_meta)["class_names"][0])
        }

    def _get_data(self, path: str, partition: str) -> pd.DataFrame:
        data = loadmat(path)["annotations"][0]
        df = pd.DataFrame(
            {"name": d[-1].item(), "label_nbr": d[-2].item()} for d in data
        )
        df = df.assign(label=df.label_nbr.map(self.class_lookup), partition=partition)
        df = df.assign(make=df.label.str.split(" ").str[0])
        df = df.assign(year=df.label.str.split(" ").str[-1].astype(int))
        df = df.assign(type=df.label.str.split(" ").str[-2])
        df = df.assign(
            model=df.label.str.split(" ").str[1:-2].apply(lambda x: " ".join(x))
        )
        df = df.assign(
            key=df.apply(lambda x: f"{x['make']}-{x['model']}-{x['type']}", axis=1)
        )
        df = df.assign(
            murmur_hash=df.key.map(functools.partial(get_mmh3, seed=self.seed))
        )
        # TODO: make this more general
        df = df.assign(
            partition_new=(df.murmur_hash % 10)
            .map({8: Split.validation, 9: Split.test})
            .fillna(Split.train)
        )
        assert len(df) == df.label.count()
        return df

    @property
    def df_train(self) -> pd.DataFrame:
        return self._get_data(self.path_train, "train")

    @property
    def df_test(self) -> pd.DataFrame:
        return self._get_data(self.path_test, "test")

    @property
    def df(self) -> pd.DataFrame:
        return pd.concat([self.df_train, self.df_test]).reset_index(drop=True)

    @property
    def lookup(self) -> Dict[Tuple[str, str], str]:
        return {
            (row.partition, row["name"]): row.partition_new
            for _, row in self.df.iterrows()
        }


def get_stanford_cars_dataset(
    split: Split,
    random_pose: bool = False,
    path: str = config.stanford_cars_path_base,
    path_meta: str = config.stanford_cars_path_meta,
    path_train: str = config.stanford_cars_path_train,
    path_test: str = config.stanford_cars_path_test,
    seed: int = config.seed_data,
    n_pose_pairs: int = config.n_pose_pairs,
    extension: str = config.stanford_cars_extension,
    width: int = config.width,
    height: int = config.height,
    transforms_mean_cars: Tuple[float, ...] = config.transforms_mean_cars_stanford_cars,
    transforms_std_cars: Tuple[float, ...] = config.transforms_std_cars_stanford_cars,
    transforms_mean_poses: Tuple[
        float, ...
    ] = config.transforms_mean_poses_stanford_cars,
    transforms_std_poses: Tuple[float, ...] = config.transforms_std_poses_stanford_cars,
):
    return StanfordCars(
        path=path,
        path_meta=path_meta,
        path_train=path_train,
        path_test=path_test,
        random_pose=random_pose,
        split=split,
        seed=seed,
        n_pose_pairs=n_pose_pairs,
        extension=extension,
        width=width,
        height=height,
        transforms_mean_objects=transforms_mean_cars,
        transforms_std_objects=transforms_std_cars,
        transforms_mean_poses=transforms_mean_poses,
        transforms_std_poses=transforms_std_poses,
    )
