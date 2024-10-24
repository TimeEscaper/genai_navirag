import torch
import numpy as np
import re

from pathlib import Path
from typing import Tuple, List, Union
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from PIL import Image
from lib.entities.types import ImageType


class LLaVACaptioner:

    def __init__(self, 
                 model_path: str = "liuhaotian/llava-v1.6-vicuna-7b"):
        # Load the model, tokenizer, and image processor
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            self.model_name,
            use_flash_attn=False,
            device_map=None
        )
        self.model_config = self.model.config
        self.mm_use_im_start_end = self.model_config.mm_use_im_start_end

        # Move model to device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        if hasattr(self.model.get_model(), 'vision_tower'):
            self.model.get_model().vision_tower.to(device)

        # Determine conversation mode
        if "llama-2" in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "mistral" in self.model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif "v1.6-34b" in self.model_name.lower():
            self.conv_mode = "chatml_direct"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"

        self._scene_prompt = "Describe in detail what you see in front of you based on the given image. Identify key objects, their relative positions, any potential obstacles, and general surroundings."
        self._object_prompt = "Describe an object briefly, but still without a lack of details. Give only description, no introductory or concluding words."

    def _load_prompt_tensor(self, prompt: str) -> Tuple[torch.Tensor, str]:
        # Prepare the prompt by inserting image tokens
        qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        prompt_tensor = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return prompt_tensor, prompt_text

    def _load_image_tensor(self, images: Union[str, List[str]]) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        if not isinstance(images, list):
            images = [self._load_image(images)]
        images = [self._load_image(e) for e in images]
        image_sizes = [img.size for img in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model_config
        ).to(dtype=torch.float16)
        return images_tensor, image_sizes

    def __call__(self, images: Union[str, List[str]], mode: str, temperature: float = 0.) -> Union[str, List[str]]:
        if mode == "scene":
            prompt = self._scene_prompt
        elif mode == "object":
            prompt = self._object_prompt
        else:
            raise ValueError(f"Unknown mode {mode}")

        device = self.model.device  # Get the device of the model

        # Load prompt tensor
        prompt_tensor, prompt_text = self._load_prompt_tensor(prompt)
        prompt_tensor = prompt_tensor.to(device)

        # Load images
        image_tensor, image_sizes = self._load_image_tensor(images)
        image_tensor = image_tensor.to(device)

        if not isinstance(images, list):
            # Single image
            prompt_tensor = prompt_tensor.unsqueeze(0)  # Add batch dimension

            with torch.inference_mode():
                output_ids = self.model.generate(
                    prompt_tensor,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0. else False,
                    temperature=temperature if temperature > 0. else None,
                    top_p=0.9 if temperature > 0. else None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                )

            # Decode the output
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return output.strip()
        else:
            # Multiple images
            batch_size = len(images)
            prompt_tensor = prompt_tensor.unsqueeze(0).repeat(batch_size, 1)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    prompt_tensor,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=True if temperature > 0. else False,
                    temperature=temperature if temperature > 0. else None,
                    top_p=0.9 if temperature > 0. else None,
                    num_beams=1,
                    max_new_tokens=512,
                    use_cache=True,
                )

            # Decode the outputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs = [output.strip() for output in outputs]
            return outputs

    def _load_image(self, image: ImageType) -> torch.Tensor:
        if isinstance(image, str) or isinstance(image, Path):
            image = Image.open(str(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image
