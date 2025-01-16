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


@dataclass
class ModelConfig:
    name: str
    cache_dir: str

@dataclass
class DetectorConfig(ModelConfig):
    confidence_threshold: float
    target_classes: List[str]

class ObjectDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        loguru.logger.info(f"Loading object detector from {self.config.name}")
        self.processor = OwlViTProcessor.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir
        ).cuda()

    def detect_objects(self, image: Image.Image) -> Dict[str, Any]:
        """Detect objects in the image"""
        texts = [self.config.target_classes]
        texts = [[str(t) for t in texts[0]]]
        inputs = self.processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        target_sizes = torch.Tensor([image.size[::-1]]).cuda()
        
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=self.config.confidence_threshold,
            target_sizes=target_sizes
        )
        
        detections = []
        for box, score, label in zip(results[0]["boxes"], results[0]["scores"], results[0]["labels"]):
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "object": list(self.config.target_classes)[label.item()],
                "confidence": round(score.item(), 3),
                "bbox": box
            })
        
        return detections
