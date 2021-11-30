from enum import Enum

import numpy as np
from PIL import Image

BinaryMask = Frame = NumpyNdArray = np.ndarray
PILImage = Image.Image


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"
