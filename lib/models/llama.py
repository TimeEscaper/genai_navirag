import torch

from transformers import pipeline, Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class LlamaReaderQuantized:
    
    def __init__(self,
                 model_name: str = "meta-llama/Llama-2-70b-chat-hf"):
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._model = pipeline(
            model=model,
            tokenizer=self._tokenizer,
            task="text-generation",
            do_sample=False,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=500,
        )
        
    @property
    def model(self) -> AutoModelForCausalLM:
        return self._model
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer
