from numpy.random import shuffle
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def collate_fn(batch):
    return [(b['image']) for b in batch]

def get_dataloaders(dataset, obj_data_dir, bgnd_data_dir, sil_data_dir, imsize, batch_size, eval_size, num_workers=1):
    r"""
    Creates a dataloader from a directory containing image data.
    """

    dataset = dataset(
        obj_dataroot=obj_data_dir,
        bgnd_dataroot=bgnd_data_dir,
        sil_dataroot=sil_data_dir,
        resize_dim=(imsize,imsize),
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
    eval_dataset, train_dataset = torch.utils.data.random_split(
        dataset,
        [eval_size, len(dataset) - eval_size],
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_dataloader, eval_dataloader
