import numpy as np

from typing import Tuple
from dataclasses import dataclass


@dataclass
class DetectedObject:
    crop: np.ndarray
    bbox_tlbr: Tuple[int, int, int, int]
    class_name: str
    title: str

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
