import os
import pprint
import argparse
import glob
import numpy as np
from PIL import Image, ImageOps
import cv2

import torch
import torchvision.transforms as transforms
from src.models import PoseGen


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--obj_data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to object dataset directory.",
    )
    parser.add_argument(
        "--bgnd_data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to background dataset directory.",
    )
    parser.add_argument(
        "--sil_data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to silhouette dataset directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(root_dir, "out"),
        help=(
            "Path to output directory. "
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help=(
            "Name of the current experiment."
            "Checkpoints are stored in '{out_dir}/{name}/ckpt/'. "
        ),
    )

    parser.add_argument(
        "--im_size",
        type=int,
        default=256,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to train on.",
    )

    return parser.parse_args()

IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

def demo(args):
    r"""
    Configures and performs demo.
    """

    # Print command line arguments and architectures
    pprint.pprint(vars(args))

    # Sanity check
    if not os.path.exists(args.obj_data_dir):
        raise FileNotFoundError('Data directory {} is not found.'.format(args.obj_data_dir))

    if not os.path.exists(args.bgnd_data_dir):
        raise FileNotFoundError('Data directory {} is not found.'.format(args.bgnd_data_dir))

    if not os.path.exists(args.sil_data_dir):
        raise FileNotFoundError('Data directory {} is not found.'.format(args.sil_data_dir))

    # Check existing experiment
    exp_dir = os.path.join(args.out_dir, args.name)

    # Setup output directories
    ckpt_dir = os.path.join(exp_dir, "ckpt")
    for d in [args.out_dir, exp_dir, ckpt_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory '{d}' is not found.")

    # Setup models
    net_g = PoseGen(nz=256).to(args.device)

    # load weights
    ckpt_paths = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
    if ckpt_paths:  # Train from scratch if no checkpoints were found
        ckpt_path = sorted(ckpt_paths, key=lambda f: int(f[:-4]))[-1]
        ckpt_path = os.path.join(ckpt_dir, ckpt_path)
        state_dict = torch.load(ckpt_path)
        net_g.load_state_dict(state_dict["net_g"])
        print("Loaded checkpoint '{}'".format(ckpt_path))

    # get filenames
    obj_files = sorted(glob.glob(os.path.join(args.obj_data_dir, "*.JPEG")))
    bgnd_files = sorted(glob.glob(os.path.join(args.bgnd_data_dir, "*.JPEG")))
    sil_files = sorted(glob.glob(os.path.join(args.sil_data_dir, "*.png")))

    # check if number of files are greater than 0
    assert len(obj_files) > 0, "No object files found."
    assert len(bgnd_files) > 0, "No background files found."
    assert len(sil_files) > 0, "No silhouette files found."

    # compose transformations
    transforms_func=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            # transforms.PILToTensor(),
                            # transforms.ConvertImageDtype(torch.float),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]
                    )

    ## load images
    # select an object file and a background file randomly and keep it constant
    obj_file = np.random.choice(obj_files)
    bgnd_file = np.random.choice(bgnd_files)
    sil_file = np.random.choice(sil_files)

    # for obj_file in obj_files:
    # for bgnd_file in bgnd_files:
    for sil_file in sil_files:
        
        # object image
        obj_image = Image.open(obj_file).resize((args.im_size, args.im_size))
        obj_image_copy = np.array(obj_image).copy()
        # check for grayscale image
        if obj_image.mode == 'L':
            obj_image = ImageOps.colorize(obj_image, black ="blue", white ="white")
        obj_image = transforms_func(obj_image)
        obj_image = obj_image.unsqueeze(0)

        # background image
        bgnd_image = Image.open(bgnd_file).resize((args.im_size, args.im_size))
        bgnd_image_copy = np.array(bgnd_image).copy()
        # check for grayscale image
        if bgnd_image.mode == 'L':
            bgnd_image = ImageOps.colorize(bgnd_image, black ="blue", white ="white")
        bgnd_image = transforms_func(bgnd_image)
        bgnd_image = bgnd_image.unsqueeze(0)
        # silhouette image
        sil_image = Image.open(sil_file).resize((args.im_size, args.im_size))
        sil_image_copy = np.array(sil_image).copy()
        # check for grayscale image
        if bgnd_image.mode == 'L':
            sil_image = ImageOps.colorize(sil_image, black ="blue", white ="white")
        sil_image = transforms_func(sil_image)
        sil_image = sil_image.unsqueeze(0)

        # forward pass
        gen_im = net_g(obj_image.to(args.device), bgnd_image.to(args.device), sil_image.to(args.device))

        # visualize the result
        gen_im = denormalize(gen_im.detach())
        gen_im = gen_im.squeeze(0).cpu().numpy()
        gen_im = np.transpose(gen_im, (1, 2, 0))
        gen_im = (gen_im * 255).astype(np.uint8)

        # visualize the result
        obj_image_copy = cv2.cvtColor(obj_image_copy, cv2.COLOR_RGB2BGR)
        bgnd_image_copy = cv2.cvtColor(bgnd_image_copy, cv2.COLOR_RGB2BGR)
        gen_im = cv2.cvtColor(gen_im, cv2.COLOR_RGB2BGR)
        img_viz = cv2.hconcat([obj_image_copy, bgnd_image_copy, sil_image_copy, gen_im])
        cv2.imshow("Generated Image", img_viz)
        cv2.waitKey(0)

if __name__ == "__main__":
    demo(parse_args())