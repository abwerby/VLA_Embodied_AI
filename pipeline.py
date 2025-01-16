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
import loguru
import hydra
from omegaconf import DictConfig
import os
from generate_captions import VLMProcessor, ObjectDetector 


