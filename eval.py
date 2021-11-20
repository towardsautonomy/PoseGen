import os
import pprint
import argparse

import torch

from src.datasets import StanfordCarsDataset, PoseGenCarsDataset
from src.utils import get_dataloaders
from src.models import PoseGen_Generator, PoseGen_Discriminator, PoseGen
from src.trainer import evaluate


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default='StanfordCarsDataset',
        choices=['StanfordCarsDataset', 'PoseGenCarsDataset'],
        help="Dataset to use for training the model.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        required=True,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )

    return parser.parse_args()


def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    nz, eval_size, num_workers = (
        256,
        5000,
        4,
    )

    # Setup models
    net_g = PoseGen(nz=nz)
    net_d = PoseGen_Discriminator()

    # Loads checkpoint
    state_dict = torch.load(args.ckpt_path)
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configure dataloaders
    datasets = {
        "StanfordCarsDataset": StanfordCarsDataset,
        "PoseGenCarsDataset": PoseGenCarsDataset,
    }
    dataset = datasets[args.dataset]
    # Configures eval dataloader
    _, eval_dataloader = get_dataloaders(
        dataset, args.obj_data_dir, args.bgnd_data_dir, args.sil_data_dir, 
        args.im_size, args.batch_size, eval_size, num_workers
    )

    # Evaluate models
    metrics = evaluate(net_g, net_d, eval_dataloader, nz, args.device)
    pprint.pprint(metrics)


if __name__ == "__main__":
    eval(parse_args())
