# Visual Language Analysis for Construction Equipment

This project provides a pipeline to control wheel loader using visual language models and large language models.
```
Disclaimer: This solution is experimental and far from prefect. It is just a proof of concept.
```

## Main Steps in my Solution


1. Download the video from the online source (e.g., YouTube) using the `yt-dlp` tool.
2. Extract frames from the video using the `extract_video_frames.py` script, with a specified interval to avoid redundancy.
3. To create the VQA pairs dataset, First, generate captions and detect objects in the frames using the `generate_captions.py` script\
    (This script uses the visual language model to generate captions and open vocab object detection model to detect objects in the frames).
4. Given the captions and detected objects, generate Q&A pairs using the `generate_VQA.py` script.\
    (This script uses the Llama 3.1 8B model to generate Q&A pairs based on the scene analysis).
5. Now you have the Q&A pairs dataset, you can use it to fine-tune the Llama model using the `fine_tune_llama1b.py` script.\
    (This script fine-tunes the Llama 3.2 1B Instruct model on the Q&A pairs dataset to make it more suitable for the construction domain).
6. Finally, you can run the full pipeline using the `pipeline.py` script to combine all the components. \
    (This script reads the frames, generates captions and detects objects, prompts the fine-tuned Llama model to get the answers).

## Tips to Improve the Solution (Future Work)
<!-- 1. The open vocab object detection model is weak and not suitable for the construction domain. as it can't detect classes like `pile`, `pile of sand`, `wheel loader`, etc.\
    So, you can use a more powerful closed object detection model like YOLO, etc. and fine-tune it on the construction domain dataset.
2. The visual language model is not perfect and can't generate accurate captions.\
    I used the XGen-mini-MM-Phi3 model, and it couldn't detect and classify the ruuning operation of the wheel loader (e.g., `loading`, `unloading`, `moving`, etc.).
3. In general I didn't invset time in cleaning the data or making sure that it make sense to fine-tune to take informed decisions.
4. Extracting frames from the video is not the best way to get the frames, as it may cause redundancy.\
    You can use feature extraction models like `xFeat`, `SuperGlue`, etc. to make sure that you get frames with different features/actions.\
    Also, a good VLM (expensive) can help in where to extract the frames.
     -->
###  Object Detection Accuracy
The open vocab object detection model is weak and not suitable for the construction domain. as it can't detect classes like `pile`, `pile of sand`, `wheel loader`, etc. So, you can use a more powerful closed object detection model like YOLO, etc. and fine-tune it on the construction domain dataset.

### Better Captioning
The visual language model is not perfect and can't generate accurate captions.\
I used the XGen-mini-MM-Phi3 model, and it couldn't detect and classify the ruuning operation of the wheel loader (e.g., `loading`, `unloading`, `moving`, etc.).

### Data Quality & Consistency
In general I didn't invset time in cleaning the data or making sure that it make sense to fine-tune to take informed decisions.\
Garbage in, garbage out!

### LLM Fine-Tuning
Extracting frames from the video is not the best way to get the frames, as it may cause redundancy.\
You can use feature extraction models like `xFeat`, `SuperGlue`, etc. to make sure that you get frames with different features/actions.\
Also, a good VLM (expensive) can help in where to extract the frames.

### Performance & Real-Time Constraints
Running the pipeline in real-time on a job site requires hardware acceleration and quintzation.
Recommendation: use Quantized models, TensorRT.

## Quick Start

### 1. Configuration

All configurations are managed through YAML files in the `config` directory:

- `pipeline.yaml`: Main pipeline configuration
- `generate_captions.yaml`: Caption generation settings
- `generate_VQA.yaml`: Question-answer generation settings

Key paths to configure:
```yaml
# Example path configurations
image_folder: "/path/to/your/frames"
llama:
    base_model_path: "meta-llama/Llama-3.2-1B-Instruct"
    new_model_path: "/path/to/your/fine-tuned/model"
```

### 2. Usage

1. Extract frames from video:
```bash
python extract_video_frames.py <video_path> --interval 1.0
```

2. Generate captions and detect objects:
```bash
python generate_captions.py
```

3. Generate Q&A pairs:
```bash
python generate_VQA.py
```

4. Run the full pipeline:
```bash
python pipeline.py
```

## File Descriptions

- `extract_video_frames.py`: Extracts frames from video files at specified intervals
- `generate_captions.py`: Generates scene descriptions using visual language models
- `vlm_processor.py`: Handles visual language model processing
- `ov_object_detector.py`: Performs object detection on frames
- `generate_VQA.py`: Generates Q&A pairs based on scene analysis
- `llama_inference.py`: Handles LLM inference for Q&A generation
- `fine_tune_llama1b.py`: Fine-tunes Llama model for construction domain
- `pipeline.py`: Main pipeline combining all components

## Model Requirements

- Llama 3.2 1B Instruct model
- XGen-mini-MM-Phi3 for visual language processing
- OwlViT for object detection

## Output Structure

The pipeline generates:
- Frame extractions from videos
- Scene captions and object detections
- Construction-specific Q&A pairs
- Visualizations (optional)
