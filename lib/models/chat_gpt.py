import os
import httpx

from typing import List
from openai import OpenAI
from lib.entities.detections import DetectedObject


class ChatGPTClient:
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 temperature: float = 0.):
        self._model = model
        self._temperature = temperature
        
        proxy_ip = os.environ["OPENAI_PROXY_IP"]
        proxy_user = os.environ["OPENAI_PROXY_USER"]
        proxy_password = os.environ["OPENAI_PROXY_PASSWORD"]
        proxy_port = os.environ["OPENAI_PROXY_PORT"]
        
        self._client = OpenAI(http_client=httpx.Client(proxies={
            "http://": f"socks5://{proxy_user}:{proxy_password}@{proxy_ip}:{proxy_port}",
            "https://": f"socks5://{proxy_user}:{proxy_password}@{proxy_ip}:{proxy_port}"
        }))
        
    def query(self, prompt: str, system_prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.
        )
        return completion.choices[0].message.content
    
    
class ChatGPTGraphBuilder:
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 temperature: float = 0.,
                 system_prompt: str = "You are expert for embodied robot navigation.",
                 prompt_template: str = "You are given with a/home/timur_akht/projects/multinav/lib/data JSON-like description of the scene. This description is a dictionary where the key is the class name and ID of the object, and value is the bounding box of the object in the source image. Based on this dictionary, you are asked to generate a graph-like description of the scene. You need to summarize spatial relationships between the objects on the scene. Be brief and precise, don't give introductory and conclusion words. Focus on the meaningful part of the response."):
        self._client = ChatGPTClient(model=model, 
                                     temperature=temperature)
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        
    def query(self, objects: List[DetectedObject]) -> str:
        objects = {e.title: e.bbox_tlbr for e in objects}
        prompt = f"{self._prompt_template}\n[BEGINNING OF THE JSON DICTIONARY]\n{objects}\n[END OF THE JSON DICTIONARY]"
        return self._client.query(prompt=prompt, 
                                  system_prompt=self._system_prompt)    
