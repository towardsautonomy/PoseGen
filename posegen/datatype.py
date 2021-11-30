from enum import Enum

import numpy as np

BinaryMask = NumpyNdArray = np.ndarray


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"
