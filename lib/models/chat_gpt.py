import os
import json
import tempfile
import time
import httpx

from typing import List
from pathlib import Path
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
    
    def query_batch(self, prompts: List[str], system_prompt: str) -> List[str]:
        batch_file = self._create_batch_file(prompts=prompts, 
                                             system_prompt=system_prompt)
        with open(batch_file, 'rb') as f:
            batch_input_file = self._client.files.create(
                file=f,
                purpose='batch'
            )
        
        batch_input_file_id = batch_input_file.id
        batch = self._client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "Batch job for processing prompts"
            }
        )
        
        batch_id = batch.id
        while True:
            batch_status = self._client.batches.retrieve(batch_id)
            status = batch_status.status
            if status == 'completed':
                break
            elif status in ['failed', 'cancelled']:
                raise RuntimeError(f"Batch status: {status}")
            else:
                time.sleep(60)
        
        output_file_id = batch_status.output_file_id
        output_content_response = self._client.files.content(output_file_id)
        output_content = output_content_response.text
        
        chunks = output_content.split("\n")
        messages = {}
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            chunk = json.loads(chunk)
            chunk_id = chunk["custom_id"]
            idx = int(chunk_id.split("-")[-1])
            error = chunk["error"]
            if error is None:
                message = chunk['response']['body']['choices'][0]['message']['content']
                messages[idx] = message
            else:
                pass
        
        results = []
        for i in range(len(prompts)):
            if i in messages:
                results.append(messages[i])
            else:
                results.append("")
      
        return results
        
    def _create_batch_file(self,
                           prompts: List[str], 
                           system_prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            for i in range(len(prompts)):
                request_line = self._create_batch_request(custom_id=f"request-{i}",
                                                          system_prompt=system_prompt,
                                                          user_prompt=prompts[i])
                json_line = json.dumps(request_line, ensure_ascii=False)
                f.write(json_line + "\n")
            output_file = f.name
        return output_file

    def _create_batch_request(self,
                              custom_id: str, 
                              system_prompt: str, 
                              user_prompt: str):
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self._model,
                "max_tokens": 1000,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            }
        }
        return request
    
    
class ChatGPTGraphBuilder:
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 temperature: float = 0.,
                 system_prompt: str = "You are expert for embodied robot navigation.",
                 prompt_template: str = "You are given with a JSON-like description of the scene. This description is a dictionary where the key is the class name and ID of the object, and value is the bounding box of the object in the source image. Based on this dictionary, you are asked to generate a graph-like description of the scene. You need to summarize spatial relationships between the objects on the scene. Be brief and precise, don't give introductory and conclusion words. Focus on the meaningful part of the response."):
        self._client = ChatGPTClient(model=model, 
                                     temperature=temperature)
        self._system_prompt = system_prompt
        self._prompt_template = prompt_template
        
    def query(self, objects: List[str]) -> str:
        # objects = {e.title: e.bbox_tlbr for e in objects}
        prompt = f"{self._prompt_template}\n[BEGINNING OF THE JSON DICTIONARY]\n{objects}\n[END OF THE JSON DICTIONARY]"
        return self._client.query(prompt=prompt, 
                                  system_prompt=self._system_prompt)
        
    def query_batch(self, objects: List[List[str]]) -> List[str]:
        prompts = [f"{self._prompt_template}\n[BEGINNING OF THE JSON DICTIONARY]\n{e}\n[END OF THE JSON DICTIONARY]" for e in objects]
        return self._client.query_batch(prompts, self._system_prompt)

