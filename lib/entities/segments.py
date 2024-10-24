import numpy as np

from typing import Tuple
from dataclasses import dataclass


@dataclass
class SAMSegment:
    crop: np.ndarray
    bbox: Tuple[int, int, int, int]

    @property
    def center(self):
        x, y, w, h = self.bbox
        return (x + x + w) // 2, (y + y + h) // 2
