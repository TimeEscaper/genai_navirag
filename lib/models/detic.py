import inspect
import torch
import numpy as np
import detic
import PIL

from typing import List, Union, Set, Optional
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from lib.entities.types import ImageType
from lib.entities.detections import DetectedObject


class DeticModel:

    def __init__(self,
                 ignore_classes: Optional[Set[str]] = None):
        detic_root = Path(inspect.getfile(detic)).parent.parent
        detic_config = detic_root / "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(str(detic_config))
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
        # cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
        predictor = DefaultPredictor(cfg)

        BUILDIN_CLASSIFIER = {
            'lvis': str(detic_root / 'datasets/metadata/lvis_v1_clip_a+cname.npy'),
            'objects365': str(detic_root / 'datasets/metadata/o365_clip_a+cnamefix.npy'),
            'openimages': str(detic_root / 'datasets/metadata/oid_clip_a+cname.npy'),
            'coco': str(detic_root / 'datasets/metadata/coco_clip_a+cname.npy'),
        }

        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
            'objects365': 'objects365_v2_val',
            'openimages': 'oid_val_expanded',
            'coco': 'coco_2017_val',
        }

        vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
        metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
        classifier = BUILDIN_CLASSIFIER[vocabulary]
        num_classes = len(metadata.thing_classes)
        reset_cls_test(predictor.model, classifier, num_classes)

        self._predictor = predictor
        self._metadata = metadata
        
        self._ignore_classes = ignore_classes if ignore_classes is not None else set()

    def __call__(self, images: Union[ImageType, List[ImageType]]) -> Union[List[DetectedObject], List[List[DetectedObject]]]:
        if not isinstance(images, list):
            return self._process_image(images)
        return [self._process_image(e) for e in images]

    def _process_image(self, image: ImageType) -> List[DetectedObject]:
        image = self._load_image(image)
        with torch.inference_mode():
            outputs = self._predictor(image)
        counters = {}
        objects = []
        for i in range(len(outputs["instances"])):
            box = outputs["instances"].pred_boxes[i].tensor.clone().detach().cpu().numpy().round().astype(int).tolist()[0]
            box = (box[0], box[1], box[2], box[3])
            crop = image[box[1]:box[3], box[0]:box[2]].copy()
            class_name = self._metadata.thing_classes[outputs["instances"].pred_classes[i].item()]
            if class_name in self._ignore_classes:
                continue
            if class_name in counters:
                idx = counters[class_name] + 1
                counters[class_name] = idx
                object_name = f"{class_name}_{idx}"
            else:
                counters[class_name] = 1
                object_name = f"{class_name}_1"
            objects.append(DetectedObject(
                crop=crop,
                bbox_tlbr=box,
                class_name=class_name,
                title=object_name
            ))
        return objects

    def _load_image(self, image: ImageType) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = PIL.Image.open(str(image)).convert("RGB")
            image = np.array(image)
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.clone().detach().cpu().numpy()
        return image
