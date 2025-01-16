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
import imageio

def save_video(video, filename, save_dir, fps=10):
    os.makedirs(save_dir, exist_ok=True)
    wide_list = video

    # Prepare the video file path
    save_path = save_dir + '/' + filename

    # Create a writer object
    video_writer = imageio.get_writer(save_path, fps=fps)

    # Write frames to the video file
    for frame in wide_list[2:-1]:
        video_writer.append_data(frame)

    video_writer.close()

    print(f"Video saved to {save_path}")

def merge_scene_descriptions(data):
    """
    Parse list of scene descriptions and merge them into a single string
    Args:
        data (list): List of scene descriptions
    Returns:
        str: Merged description of the scene
    """    
    # Extract all responses
    caption = ""
    for i in range(len(data)):    
        caption += data[i]['response'] + " "
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

    # Initialize analyzer Couldn't fit in my GPU
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
    question = "What is the loader should do next?"
    imgs = []
    for image_path in image_paths:
        loguru.logger.info(f"Processing image: {image_path}")
        img1 = (Image.open(image_path)).convert('RGB')
        # write the question to the image with white color
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, question, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        results = vlm.generate_captions(img1, cfg.vlm_model.queries)
        caption = merge_scene_descriptions(results)
        response = llama_inference.generate_response(
            instruction=system_instruction,
            user_message= caption + " " + question,
            max_new_tokens=150
            )
        print("\n ------------------------------------------ \n")
        print(response)
        print("\n ------------------------------------------ \n")
        # write the response to the image with green color
        cv2.putText(img, response, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        imgs.append(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(1)
    cv2.destroyAllWindows()
    save_video(imgs, "demo.mp4", "output", fps=1)

if __name__ == "__main__":
    main()