from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from transformers import (
    AutoModelForVision2Seq, 
    AutoTokenizer, 
    AutoImageProcessor,
    OwlViTProcessor, 
    OwlViTForObjectDetection,
    StoppingCriteria
)
import loguru
from PIL import Image
import loguru


@dataclass
class ModelConfig:
    name: str
    cache_dir: str


class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence: List[int] = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids


class VLMProcessor:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the VLM models and processors"""
        loguru.logger.info(f"Loading VLM from {self.model_config.name}")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_config.name,
            trust_remote_code=True,
            cache_dir=self.model_config.cache_dir
        ).cuda()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.name,
            trust_remote_code=True,
            use_fast=False,
            legacy=False
        )
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model_config.name,
            trust_remote_code=True
        )
        
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)

    def generate_captions(self, image: Image.Image, queries: List[str]) -> List[Dict[str, str]]:
        """Generate captions for the image based on queries"""
        results = []
        for query in queries:
            inputs = self.image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
            prompt = self.apply_prompt_template(query)
            language_inputs = self.tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
            
            generated_text = self.model.generate(
                **inputs,
                image_size=[image.size],
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=768,
                top_p=None,
                num_beams=1,
                stopping_criteria=[EosListStoppingCriteria()]
            )
            
            prediction = self.tokenizer.decode(
                generated_text[0],
                skip_special_tokens=True
            ).split("<|end|>")[0]
            
            results.append({
                "query": query,
                "response": prediction
            })
        
        return results

    @staticmethod
    def apply_prompt_template(prompt: str) -> str:
        return (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
        )
