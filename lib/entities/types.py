import numpy as np
import torch

from typing import Union
from pathlib import Path
from PIL import Image


ImageType = Union[str, Path, np.ndarray, torch.Tensor, Image.Image]
