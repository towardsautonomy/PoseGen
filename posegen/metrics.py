from .datatype import BinaryMask


def iou(m1: BinaryMask, m2: BinaryMask) -> float:
    return (m1 & m2).sum() / (m1 | m2).sum()
