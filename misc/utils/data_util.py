import functools
from typing import Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch):
    return [(b["image"]) for b in batch]


def split(
    dataset: Dataset, eval_split: float, test_split: float, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    pass


def get_dataloaders(
    dataset,
    obj_data_dir,
    bgnd_data_dir,
    sil_data_dir,
    imsize,
    batch_size,
    eval_split=0.1,
    test_split=0.1,
    num_workers=16,
    seed=0,
):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = dataset(
        obj_dataroot=obj_data_dir,
        bgnd_dataroot=bgnd_data_dir,
        sil_dataroot=sil_data_dir,
        resize_dim=(imsize, imsize),
        transforms=transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.PILToTensor(),
                # transforms.ConvertImageDtype(torch.float),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        shuffle=True,
        verbose=True,
    )
    train, evaluation, test = split(dataset, eval_split, test_split, seed)
    dl_partial = functools.partial(
        torch.utils.data.DataLoader,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dl_train = dl_partial(dataset=train, shuffle=True)
    dl_eval = dl_partial(dataset=evaluation, shuffle=False)
    dl_test = dl_partial(dataset=test, shuffle=True)

    return dl_train, dl_eval, dl_test
