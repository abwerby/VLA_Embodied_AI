import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import loguru
import hydra
from omegaconf import DictConfig
import os
from pathlib import Path
import loguru
from llama_inference import LlamaInference
from generate_captions import VisionAnalyzer
from vlm_processor import VLMProcessor, ModelConfig
from generate_vqa import parse_frame_analysis
import cv2
import json

def merge_scene_descriptions(data):
    """
    Parse JSON string and merge all responses into a comprehensive scene description.
    
    Args:
        json_string (str): JSON string containing image information and captions
        
    Returns:
        str: Merged description of the scene
    """    
    # Extract all responses
    caption = ""
    for i in range(len(data)):    
        for query, response in data[i].items():
            caption += response + " "
    return caption


@hydra.main(config_path="config", config_name="pipeline")
def main(cfg: DictConfig):

    orig_cwd = hydra.utils.get_original_cwd()
    image_folder = Path(orig_cwd) / cfg.image_folder if not Path(cfg.image_folder).is_absolute() else Path(cfg.image_folder)
    image_paths = sorted(image_folder.glob("*.jpg"))
    
    if cfg.end_frame is not None:
        image_paths = image_paths[cfg.start_frame:cfg.end_frame]
    else:
        image_paths = image_paths[cfg.start_frame:]

    # Initialize analyzer
    # analyzer = VisionAnalyzer(cfg)
    vlm = VLMProcessor(ModelConfig(
            name=cfg.vlm_model.name,
            cache_dir=cfg.vlm_model.cache_dir
        ))

    llama_inference = LlamaInference(
        base_model_path=cfg.llama.base_model_path,
        new_model_path=cfg.llama.new_model_path,
        device="cuda" 
    )

    system_instruction = """You are an expert in construction site operations, specifically focusing on wheel loaders.
        Given the following scene description and object detections within the scene,
        Answer the following questions."""

    # loop through all images get the captions and detections then ask questions to the llama model
    for image_path in image_paths:
        loguru.logger.info(f"Processing image: {image_path}")
        img = (Image.open(image_path)).convert('RGB')
        # cv2.imshow('image', cv2.imread(str(image_path)))
        # cv2.waitKey(1)
        results = vlm.generate_captions(img, cfg.vlm_model.queries)
        caption = merge_scene_descriptions(results)
        print(caption)

if __name__ == "__main__":
    main()