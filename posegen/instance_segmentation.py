from dataclasses import dataclass
from logging import root
from typing import List, Optional, Sequence

from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms

import config
from datatypes import BinaryMask, PILImage
from utils import get_device


@dataclass
class ObjectMask:
    object_name: str
    mask: BinaryMask


class InstanceSegmentation:
    def __init__(
        self,
        coco_categories: Sequence[str] = config.coco_instance_category_names,
        threshold: float = config.instance_segmentation_threshold,
    ):
        self.device = get_device()
        self.model = self._get_instance_segmentation_model()
        self.coco_categories = coco_categories
        self.threshold = threshold

    def _get_instance_segmentation_model(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model.to(self.device)

    def get_instance_segmentation(
        self,
        path: Optional[str] = None,
        image: Optional[PILImage] = None,
    ) -> List[ObjectMask]:
        """
        Takes either a path to an image on disk or a PIL Image object.
        Passes it through the model and returns object masks with score >= threshold.
        """
        img = np.array(Image.open(path) if path is not None else image)
        img = transforms.ToTensor()(img).unsqueeze(dim=0)
        predictions = self.model(img.to(self.device))[0]
        labels = predictions["labels"]
        scores = predictions["scores"]
        masks = predictions["masks"]
        return [
            ObjectMask(
                object_name=self.coco_categories[label.item()],
                mask=(mask >= self.threshold)
                .squeeze()
                .cpu()
                .detach()
                .numpy()
                .astype(bool),
            )
            for label, score, mask in zip(labels, scores, masks)
            if score.item() >= self.threshold
        ]


def get_mask(
    path: Optional[str] = None,
    image: Optional[PILImage] = None,
) -> Optional[BinaryMask]:
    # TODO: loads the model on each call => load once and run multiple times if this is a bottleneck
    res = InstanceSegmentation().get_instance_segmentation(path=path, image=image)
    masks = sorted(
        [x for x in res if x.object_name == "car"],
        key=lambda x: x.mask.sum(),
        reverse=True,
    )
    if len(masks) > 0:
        return masks[0].mask

# Use this to generate masks for datasets from 'TeslaPoseGen/{model}/{location}/image/{image_name}'
# and save them to 'TeslaPoseGen/{model}/{location}/mask/{image_name}'
def main():
    import glob
    import os
    import argparse
    
    # get root_path from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, 
                            help="Path to the root dir of the dataset")
    args = parser.parse_args()
    root_path = args.root_path
    
    # generate masks
    for location in os.listdir(root_path):
        for image_name in glob.glob(os.path.join(root_path, location, 'image', '*.JPEG')):
            mask = get_mask(image_name)
            if mask is not None:
                mask_path = os.path.join(root_path, location, 'mask', os.path.basename(image_name))
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                # save using pillow
                Image.fromarray((mask * 255.0).astype(np.uint8)).save(mask_path)
                print(f'Saved mask to: {mask_path}')
                    
# visualize samples as grid of images
def visualize_samples():
    import glob
    import os
    import argparse
    import math
    import torch
    import torchvision
    
    # visualize
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import Button
    
    # get root_path from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True, 
                            help="Path to the root dir of the dataset")
    args = parser.parse_args()
    root_path = args.root_path
                
    # load images and masks
    n_images = 64
    images = []
    background_images = []
    masks = []
    for location in os.listdir(root_path):
        image_names = glob.glob(os.path.join(root_path, location, 'image', '*.JPEG'))
        background_image_names = glob.glob(os.path.join(root_path, location, 'background', '*.JPEG'))
        # sample 'n_images' images randomly
        image_names = np.random.choice(image_names, size=64, replace=False)
        background_image_names = np.random.choice(background_image_names, size=64, replace=False)
        for i, image_name in enumerate(image_names):
            images.append(Image.open(image_name))
            background_images.append(Image.open(background_image_names[i]))
            mask_path = os.path.join(root_path, location, 'mask', os.path.basename(image_name))
            masks.append(Image.open(mask_path))
    
        # show grid of images of size 'n_images' and a grid of masks of size 'n_images'
        # side by side
        fig = plt.figure(figsize=(32, 16))
        plt.suptitle(f'{location}')
        # convert to tensors
        images = torchvision.utils.make_grid(torch.stack([transforms.ToTensor()(x) for x in images]), nrow=int(math.sqrt(n_images)), padding=5)
        masks = torchvision.utils.make_grid(torch.stack([transforms.ToTensor()(x) for x in masks]), nrow=int(math.sqrt(n_images)), padding=5)
        background_images = torchvision.utils.make_grid(torch.stack([transforms.ToTensor()(x) for x in background_images]), nrow=int(math.sqrt(n_images)), padding=5)
        # show images
        ax = fig.add_subplot(131)
        ax.imshow(background_images.permute(1, 2, 0))
        ax.set_title('Background')
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plt.tight_layout()
        ax = fig.add_subplot(132)
        ax.imshow(images.permute(1, 2, 0))
        ax.set_title('Images')
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plt.tight_layout()
        # show masks
        ax = fig.add_subplot(133)
        ax.imshow(masks.permute(1, 2, 0))
        ax.set_title('Masks')
        plt.axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.0, hspace=0.0)
        plt.tight_layout()
        
        # clear images and masks
        images = []
        masks = []
        background_images = []
        
    plt.show()

if __name__ == '__main__':
    main()
