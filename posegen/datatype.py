from enum import Enum

import numpy as np

BinaryMask = Frame = NumpyNdArray = np.ndarray


class Split(Enum):
    train = "train"
    validation = "validation"
    test = "test"
