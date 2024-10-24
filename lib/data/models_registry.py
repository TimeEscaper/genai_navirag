from typing import Optional
from lib.models.llava import LLaVACaptioner
from lib.models.detic import DeticModel
from lib.models.chat_gpt import ChatGPTGraphBuilder


class LocalModelsRegistry:
    
    def __init__(self,
                 llava_captioner: Optional[LLaVACaptioner],
                 detic: Optional[DeticModel],
                 gpt_graph_builder: Optional[ChatGPTGraphBuilder]):
        self._llava_captioner = llava_captioner
        self._detic = detic
        self._gpt_graph_builder = gpt_graph_builder
        
    @property
    def llava_captioner(self) -> Optional[LLaVACaptioner]:
        return self._llava_captioner
    
    @property
    def detic(self) -> Optional[DeticModel]:
        return self._detic

    @property
    def gpt_graph_builder(self) -> Optional[ChatGPTGraphBuilder]:
        return self._gpt_graph_builder
