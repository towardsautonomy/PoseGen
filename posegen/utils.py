import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def get_device() -> torch.device:
    return torch.device("cuda" if cuda_available() else "cpu")
