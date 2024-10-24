import torch
import numpy as np

from typing import List, Union
from pathlib import Path
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator,
)
from PIL import Image
from lib.entities.types import ImageType
from lib.entities.segments import SAMSegment


class SAMModel:

    def __init__(self,
                 checkpoint_path: Union[str, Path],
                 backbone: str = "vit_h",
                 device: str = "cuda") -> None:
        """
        Initalizes the SAM class.

        """
        self._sam_checkpoint_path = checkpoint_path
        self._sam_backbone_type = backbone
        self._device = device

        self._sam = sam_model_registry[self._sam_backbone_type](
            checkpoint=self._sam_checkpoint_path
        ).to(device=self._device)

        self._mask_generator = SamAutomaticMaskGenerator(self._sam)
        self.image = None
        self.sam_result = None

    def __call__(self, image: Union[ImageType, List[ImageType]]) -> Union[List[SAMSegment], List[List[SAMSegment]]]:
        if isinstance(image, list):
            return [self._process_image(e) for e in image]
        return self._process_image(image)

    def _process_image(self, image: ImageType) -> SAMSegment:
        image = self._load_image(image)
        segments_output = self._mask_generator.generate(image)
        
        result = []
        for segment in segments_output:
            filtered_image = image * np.tile(segment["segmentation"][:, :, np.newaxis], (1, 1, 3))
            x, y, w, h = segment["bbox"]
            crop = filtered_image[y:y+h, x:x+w]
            result.append(SAMSegment(crop=crop.copy(), bbox=(x, y, w, h)))
        
        return result

    def _load_image(self, image: ImageType) -> np.ndarray:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image)).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.clone().detach().cpu().numpy()
        elif not isinstance(image, np.ndarray):
            raise ValueError("Unknown input image type")
        return image
