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
from ov_object_detector import ObjectDetector, DetectorConfig
from vlm_processor import VLMProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    cache_dir: str

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