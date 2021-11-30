import functools
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import mmh3
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from ..datatype import BinaryMask, NumpyNdArray, Split


@dataclass(frozen=True)
class CarWithMask:
    car_image_path: Path
    width: int
    height: int
    mask_random: Optional[BinaryMask] = None
    category_idx: Optional[int] = None

    @property
    def mask(self) -> BinaryMask:
        return self.mask_random if self.mask_random is not None else self.car_mask

    @property
    @functools.lru_cache()
    def car_mask(self) -> BinaryMask:
        # TODO: cache this on disk if it becomes a bottleneck (probably for Stanford cars)
        # TODO: run instance segmentation
        raise NotImplementedError

    @property
    def car_image(self) -> NumpyNdArray:
        img = Image.open(self.car_image_path)
        return img.resize((self.width, self.height))


class CarDataset(Dataset):
    def __init__(
        self,
        path: str,
        random_pose: bool,
        n_pose_pairs: int,
        split: Split,
        seed: int,
        extension: str,
        width: int,
        height: int,
    ):
        if not random_pose and n_pose_pairs > 1:
            raise ValueError("each image has a unique pose")

        self.path = path
        self.seed = seed
        self.n_pose_pairs = n_pose_pairs
        self.extension = extension
        self.width = width
        self.height = height
        self.split = split
        self._np_rand_state = np.random.RandomState(self.seed)
        self._df = self._split(self.path, self.extension, self.seed)
        self._df_split = self._df[self._df.split == split].sort_values(by="path")
        self._cars = [
            CarWithMask(car_image_path=path, width=self.width, height=self.height)
            for path in self._df_split.path
        ]
        self._data = (
            self._construct_random_poses(self._cars, self.n_pose_pairs)
            if random_pose
            else self._cars
        )

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[NumpyNdArray, BinaryMask]:
        car = self._data[idx]
        return car.car_image, car.mask

    def _construct_random_poses(self, cars: Sequence[CarWithMask], n_pose_pairs: int) -> Sequence[CarWithMask]:
        # for each car randomly pair it with `n_pose_pairs` pose masks selected from the same set.
        random_poses = self._np_rand_state.choice(cars, len(cars) * n_pose_pairs)
        return [
            CarWithMask(
                car_image_path=car.car_image_path,
                mask_random=random_poses[idx * n_pose_pairs + idx_n].mask,
                width=self.width, height=self.height,
            )
            for idx, car in enumerate(cars)
            for idx_n in range(n_pose_pairs)
        ]

    @staticmethod
    def _get_mmh3(s: str, seed: int) -> str:
        return mmh3.hash(s, seed=seed)

    @staticmethod
    def _get_md5(path: Path) -> str:
        return hashlib.md5(open(path, 'rb').read()).hexdigest()

    def _split(self, path_base: str, ext: str, seed: int) -> pd.DataFrame:
        # this could be done offline and stored so we don't have to do it on-the-fly
        # in general this is not very expensive though
        # the splitting is tied to the MD5 of the file and is invariant to the location on disk
        df = pd.DataFrame(
            {'path': path, 'md5': self._get_md5(path)}
            for path in Path(path_base).rglob(f'*.{ext}')
        )
        # TODO: for stanford cars this should be at category level, so if it works we can claim generalization
        # TODO: use decimals (instead of the hardcoded 80/10/10) to make this more general
        splits = (
            df.md5
            .map(functools.partial(self._get_mmh3, seed=seed))
            .map(lambda x: x % 10)
            .map({8: Split.validation, 9: Split.test})
            .fillna(Split.train)
        )
        return df.assign(split=splits)


# # import packages
# import os
# import sys
#
# import numpy as np
# from PIL import Image, ImageOps
# from copy import deepcopy
# import torch
# import torchvision.transforms as transforms
#
# # add path to sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# UTILS_DIR = os.path.dirname(os.path.abspath('src/utils'))
# sys.path.append(BASE_DIR)
# sys.path.append(UTILS_DIR)
#
# # import utilities
# from utils import *
#
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
#
# # dataset class from which each dataset inherits
# class Dataset(torch.utils.data.Dataset):
#
#     def __init__(self):
#         pass
#
#     # method to get length of data
#     def __len__(self):
#         raise NotImplementedError
#
#     # method to get each item
#     def __getitem__(self, idx, object_id=None):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         sample = None
#         if object_id == None:
#             id_filename_pairs = self.get_id_filename_pairs()
#             assert(idx < len(id_filename_pairs))
#             id_filename_pair = id_filename_pairs[idx]
#             obj_type_id = list(id_filename_pair.keys())[0]
#             filename = list(id_filename_pair.values())[0]
#             # object image
#             obj_image = Image.open(filename).resize(self.get_resize_dim())
#             # check for grayscale image
#             if obj_image.mode == 'L':
#                 obj_image = ImageOps.colorize(obj_image, black ="blue", white ="white")
#             # perform transforms
#             if self.get_transforms() is not None:
#                 obj_image = self.get_transforms()(obj_image)
#
#             # background image
#             bgnd_img_index = np.random.randint(0, len(self.get_background_filenames()))
#             bgnd_image = Image.open(self.get_background_filenames()[bgnd_img_index]).resize(self.get_resize_dim())
#             # check for grayscale image
#             if bgnd_image.mode == 'L':
#                 bgnd_image = ImageOps.colorize(bgnd_image, black ="blue", white ="white")
#             # perform transforms
#             if self.get_transforms() is not None:
#                 bgnd_image = self.get_transforms()(bgnd_image)
#
#             # silhouette image
#             sil_img_index = np.random.randint(0, len(self.get_silhouette_filenames()))
#             sil_image = Image.open(self.get_silhouette_filenames()[sil_img_index]).resize(self.get_resize_dim())
#             # check for grayscale image
#             if bgnd_image.mode == 'L':
#                 sil_image = ImageOps.colorize(sil_image, black ="blue", white ="white")
#             # perform transforms
#             if self.get_transforms() is not None:
#                 sil_image = self.get_transforms()(sil_image)
#
#             sample = { 'obj_image': obj_image,
#                        'bgnd_image': bgnd_image,
#                        'sil_image': sil_image,
#                        'object_type_id': obj_type_id,
#                        'object_description': self.object_id_description_dict()[obj_type_id]
#                      }
#         else:
#             if object_id in self.get_object_type_ids():
#                 filenames = self.get_filenames(object_id)
#                 assert(idx < len(filenames))
#                 # object image
#                 obj_image = Image.open(filenames[idx]).resize(self.get_resize_dim())
#                 # check for grayscale image
#                 if obj_image.mode == 'L':
#                     obj_image = ImageOps.colorize(obj_image, black ="blue", white ="white")
#                 # perform transforms
#                 if self.get_transforms() is not None:
#                     obj_image = self.get_transforms()(obj_image)
#
#                 # background image
#                 bgnd_img_index = np.random.randint(0, len(self.get_background_filenames()))
#                 bgnd_image = Image.open(self.get_background_filenames()[bgnd_img_index]).resize(self.get_resize_dim())
#                 # check for grayscale image
#                 if bgnd_image.mode == 'L':
#                     bgnd_image = ImageOps.colorize(bgnd_image, black ="blue", white ="white")
#                 # perform transforms
#                 if self.get_transforms() is not None:
#                     bgnd_image = self.get_transforms()(bgnd_image)
#
#                 # silhouette image
#                 sil_img_index = np.random.randint(0, len(self.get_silhouette_filenames()))
#                 sil_image = Image.open(self.get_silhouette_filenames()[sil_img_index]).resize(self.get_resize_dim())
#                 # check for grayscale image
#                 if bgnd_image.mode == 'L':
#                     sil_image = ImageOps.colorize(sil_image, black ="blue", white ="white")
#                 # perform transforms
#                 if self.get_transforms() is not None:
#                     sil_image = self.get_transforms()(sil_image)
#
#                 sample = { 'obj_image': obj_image,
#                            'bgnd_image': bgnd_image,
#                            'sil_image': sil_image,
#                            'object_type_id': object_id,
#                            'object_description': self.object_id_description_dict()[object_id]
#                         }
#         return sample
#
#     # method to get resize shape
#     def get_resize_dim(self):
#         '''
#         Returns the dimensions of image size to be resized to (w, h)
#         '''
#         raise NotImplementedError
#
#     # method to get transforms functions
#     def get_transforms(self):
#         '''
#         Returns a list of transforms function
#         '''
#         raise NotImplementedError
#
#     # method to get a list of object type
#     def get_object_type_ids(self):
#         '''
#         Returns a numpy array of object type ids
#         '''
#         raise NotImplementedError
#
#     # method to get annotations
#     def get_annotations(self):
#         '''
#         Returns a dictionary of {object_type_id: [filenames]}
#         '''
#         raise NotImplementedError
#
#     # method to get a list of filenames corresponding to the object id
#     def get_filenames(self):
#         '''
#         Returns a list of filenames
#         '''
#         raise NotImplementedError
#
#     # method to get a list of background image filenames
#     def get_background_filenames(self):
#         raise NotImplementedError
#
#     # method to get a list of silhouette image filenames
#     def get_silhouette_filenames(self):
#         raise NotImplementedError
#
#     # method to get a list of [object type<->filename] pairs
#     def get_id_filename_pairs(self):
#         '''
#         Returns a list of [{object_type_id1: filename1}, {object_type_id2: filename2}, ...]
#         '''
#         raise NotImplementedError
#
#     # method to get a dictionary of {object type: object_description} pairs
#     def object_id_description_dict(self):
#         '''
#         Returns a dictionary of {object_type1: object_description1, object_type2: object_description2}
#         '''
#         raise NotImplementedError
