from dataclasses import dataclass
from typing import List, Optional, Sequence

from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from . import config
from .datatype import BinaryMask, Frame


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
        self.model = self._get_instance_segmentation_model()
        self.coco_categories = coco_categories
        self.threshold = threshold

    @property
    def gpu(self) -> bool:
        return torch.cuda.is_available()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if self.gpu else "cpu")

    def _get_instance_segmentation_model(self):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        return model.to(self.device)

    def get_instance_segmentation(
        self,
        path: Optional[str] = None,
        image: Optional[Frame] = None,
    ) -> List[ObjectMask]:
        """
        Takes either a path to an image on disk or a PIL Image object.
        Passes it through the model and returns object masks with score >= threshold.
        """
        img = np.array(
            Image.open(path)
            if path is not None
            else image
        )
        img = transforms.ToTensor()(img).unsqueeze(dim=0)
        predictions = self.model(img.to(self.device))[0]
        labels = predictions["labels"]
        scores = predictions["scores"]
        masks = predictions["masks"]
        return [
            ObjectMask(
                object_name=self.coco_categories[label.item()],
                mask=(mask >= self.threshold).squeeze().cpu().detach().numpy().astype(bool),
            )
            for label, score, mask in zip(labels, scores, masks)
            if score.item() >= self.threshold
        ]


def get_mask(
    path: Optional[str] = None,
    image: Optional[Image] = None,
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
