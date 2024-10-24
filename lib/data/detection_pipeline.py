import numpy as np

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm
from lib.models.llava import LLaVACaptioner
from lib.models.detic import DeticModel
from lib.data.models_registry import LocalModelsRegistry


def _batchify(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class DataPipelineStep(ABC):
    
    @abstractmethod
    def __call__(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class DataPipeLine:
    
    def __init__(self,
                 steps: List[DataPipelineStep]):
        self._steps = steps
        
    def __call__(self, *args, **kwargs):
        result = None
        for i, step in enumerate(self._steps):
            print(f"Doing step {i+1}")
            if i == 0:
                result = step(*args, **kwargs)
            else:
                result = step(result)
        return result


class TrajectoryImageCollector(DataPipelineStep):

    def __init__(self,
                 threshold_distance: float):
        super(TrajectoryImageCollector, self).__init__()
        self._threshold_distance = threshold_distance
        
    def __call__(self, data: Union[str, Path]):
        traj_dir = Path(data)
        odom = np.load(str(traj_dir / "trajectory.npy"))[:, :2]
        frames = sorted((traj_dir / "rgb_images").glob("*.jpg"))
        filtered_frames = [frames[0]]
        
        last_index = 0
        for i in range(1, len(frames)):
            distance = np.linalg.norm(odom[last_index] - odom[i])
            if distance > self._threshold_distance:
                filtered_frames.append(frames[i])
                last_index = i

        result = [{"id": e.stem, "path": str(e)} for e in filtered_frames]
        return result


class SceneCaptionStep(DataPipelineStep):
    
    def __init__(self,
                 models: LocalModelsRegistry,
                 batch_size: int = 1):
        super(SceneCaptionStep, self).__init__()
        self._models = models
        self._batch_size = batch_size

    def __call__(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        images = [e["path"] for e in data]
        captions = []
        for batch in _batchify(images, self._batch_size):
            batch_captions = self._models.llava_captioner(batch, mode="scene")
            captions = captions + batch_captions
        for i in range(len(data)):
            data[i]["caption"] = captions[i]
        return data


class DetectionStep(DataPipelineStep):
    
    def __init__(self,
                 models: LocalModelsRegistry):
        super(DetectionStep, self).__init__()
        self._models = models
    
    def __call__(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        images = [e["path"] for e in data]
        objects = self._models.detic(images)
        for i in range(len(data)):
            data[i]["objects"] = objects[i]
        return data
        

class DetectionReduceStep(DataPipelineStep):
    
    def __init__(self):
        super(DetectionReduceStep, self).__init__()
        
    def __call__(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for i in range(len(data)):
            data[i]["objects"] = {e.title: e.bbox_tlbr for e in data[i]["objects"]}
        return data


class GPTGraphGenerateStep(DataPipelineStep):
    
    def __init__(self, models: LocalModelsRegistry):
        super(GPTGraphGenerateStep, self).__init__()
        self._models = models
        
    def __call__(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        objects_lists = [e["objects"] for e in data]
        objects_graphs = self._models.gpt_graph_builder.query_batch(objects_lists)
        for i in range(len(data)):
            data[i]["graph"] = objects_graphs[i]
        return data
