import torch
import numpy as np
import torchvision.transforms.v2 as transforms

from typing import Union, List
from pathlib import Path
from PIL import Image
from lib.entities.types import ImageType


class DINOModel:

    def __init__(self,
                 device: str = "cuda") -> None:
        self._device = device
        self._model = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16"
        )
        self._to_tensor = transforms.ToTensor()
        self._transforms = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._model.to(device)
        self._model = self._model.eval()

    def __call__(self, image: Union[ImageType, List[ImageType]]) -> torch.Tensor:
        if isinstance(image, list):
            image = torch.stack([self._load_image(e) for e in image], dim=0)
            single_input = False
        else:
            image = self._load_image(image).unsqueeze(0)
            single_input = True
        
        with torch.inference_mode():
            embeddings = self._model(image.to(self._device))

        if single_input:
            return embeddings.squeeze(0)
        return embeddings
 
    def _load_image(self, image: ImageType) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image))
            image = self._to_tensor(image)
        elif isinstance(image, np.ndarray) or isinstance(image, Image.Image):
            image = self._to_tensor(image)
        elif not isinstance(image, torch.Tensor):
            raise ValueError("Unknown type of the input image")
        return image
