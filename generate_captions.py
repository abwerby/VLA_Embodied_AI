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
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import logging
import hydra
from omegaconf import DictConfig
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    cache_dir: str

@dataclass
class DetectorConfig(ModelConfig):
    confidence_threshold: float
    target_classes: List[str]

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence: List[int] = [32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

class ObjectDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self._initialize_model()

    def _initialize_model(self):
        logger.info(f"Loading object detector from {self.config.name}")
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

class VLMProcessor:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._initialize_models()

    def _initialize_models(self):
        """Initialize the VLM models and processors"""
        logger.info(f"Loading VLM from {self.model_config.name}")
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

class VisionAnalyzer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.vlm = VLMProcessor(ModelConfig(
            name=cfg.vlm_model.name,
            cache_dir=cfg.vlm_model.cache_dir
        ))
        self.detector = ObjectDetector(DetectorConfig(
            name=cfg.detector_model.name,
            cache_dir=cfg.detector_model.cache_dir,
            confidence_threshold=cfg.detector_model.confidence_threshold,
            target_classes=cfg.detector_model.target_classes
        ))
        self.output_dir = Path(cfg.output.dir) / Path(cfg.image_folder).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analyze a single image with both VLM and object detection"""
        image = Image.open(image_path).convert("RGB")
        
        # Get captions from VLM
        captions = self.vlm.generate_captions(image, self.cfg.queries)
        
        # Get object detections
        detections = self.detector.detect_objects(image)
        
        # Save visualization if enabled
        if self.cfg.output.save_visualizations:
            self.save_visualization(image, detections, image_path)
        
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "captions": captions,
            "detections": detections
        }

    def save_visualization(self, image: Image.Image, detections: List[Dict], image_path: Path):
        """Save visualization of detected objects"""
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        
        for det in detections:
            box = det["bbox"]
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none"
            )
            ax.add_patch(rect)
            ax.text(
                box[0],
                box[1],
                f"{det['object']}: {det['confidence']:.2f}",
                color="red",
                bbox=dict(facecolor="white", alpha=0.7)
            )
        
        plt.axis("off")
        vis_path = self.output_dir / "visualizations"
        vis_path.mkdir(exist_ok=True)
        plt.savefig(vis_path / f"{image_path.stem}_detection.jpg", bbox_inches="tight", dpi=300)
        plt.close()

    def save_results(self, results: Dict[str, Any]):
        """Save analysis results to JSON"""
        output_file = self.output_dir / f"{Path(results['image_name']).stem}_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

@hydra.main(config_path="config", config_name="generate_captions")
def main(cfg: DictConfig):
    # Get original working directory
    orig_cwd = hydra.utils.get_original_cwd()
    
    # Initialize analyzer
    analyzer = VisionAnalyzer(cfg)
    
    # Process images
    image_folder = Path(orig_cwd) / cfg.image_folder if not Path(cfg.image_folder).is_absolute() else Path(cfg.image_folder)
    image_paths = sorted(image_folder.glob("*.jpg"))
    
    if cfg.end_frame is not None:
        image_paths = image_paths[cfg.start_frame:cfg.end_frame]
    else:
        image_paths = image_paths[cfg.start_frame:]
    
    for image_path in image_paths:
        logger.info(f"Processing image: {image_path}")
        results = analyzer.analyze_image(image_path)
        analyzer.save_results(results)
  

if __name__ == "__main__":
    main()