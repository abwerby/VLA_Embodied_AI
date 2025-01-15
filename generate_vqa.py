import torch
from transformers import pipeline
import json
import os
from pathlib import Path
from typing import Dict, List, Union
import logging
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_frame_analysis(json_path: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Parse a frame analysis JSON file and format it for LLM prompting.
    
    Args:
        json_path: Path to the JSON file containing frame analysis
        
    Returns:
        List of message dictionaries formatted for LLM prompting
    """
    try:
        # Read the JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Combine all captions and detections into a single description
        description_parts = []
        
        # Add captions
        for caption in data.get('captions', []):
            description_parts.append(caption['response'].strip())
        
        # Add detection information
        detections = data.get('detections', [])
        if detections:
            detection_desc = "Objects detected in the scene:\n"
            for det in detections:
                detection_desc += f"- {det['object']} (confidence: {det['confidence']:.2f}) at location {det['bbox']}\n"
            description_parts.append(detection_desc)
        
        # Combine all parts into a single description
        combined_description = " ".join(description_parts)
        
        # Create the system prompt
        system_prompt = """You are an expert in construction site operations, specifically focusing on wheel loaders. \
                         Given the following frame description and object detections, generate a set of 3-5 \
                         question-answer pairs related to what the loader might do, or \
                         next steps to take (load, unload, go to pile). Your answers should be concise, factual, and revolve around \
                         typical wheel-loader operations.\n \
                         Example planning questions based on the description:\n \
                         if there is a pile of material at loc [x, y] and the loader unloaded, what is the loader likely to do next? \
                              - go to the pile location [x, y]\n \
                        if the loader is loaded and there is a truck at loc [x, y], what is the loader likely to do next? \
                                - go to the truck location [x, y]\n \
                        \n \
                        step1: go to pile at loc [x, y], step2: load material, step3: go to truck at loc [x, y], step4: unload material\n \
                        decide the next step for the loader based on the current state of the loader and the scene\n \
                         \n\n \
                         output format: make the question-answer pairs in JSON format, with the question as the key and the answer as the value.\
                        Don't include the system prompt in the output, Don't write any Code, just write the question-answer pairs in JSON format.
                            """
        
        # Format messages for the LLM
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": combined_description
            }
        ]
        
        return messages
    
    except Exception as e:
        logger.error(f"Error parsing frame analysis file {json_path}: {str(e)}")
        return []

def batch_process_analysis_files(directory_path: Union[str, Path]) -> Dict[str, List[Dict[str, str]]]:
    """
    Process all analysis JSON files in a directory.
    
    Args:
        directory_path: Path to directory containing analysis JSON files
        
    Returns:
        Dictionary mapping frame names to their formatted messages
    """
    directory = Path(directory_path)
    results = {}
    
    for json_file in sorted(directory.glob("*_analysis.json")):
        frame_name = json_file.stem.replace('_analysis', '')
        messages = parse_frame_analysis(json_file)
        if messages:
            results[frame_name] = messages
            logger.info(f"Successfully processed frame {frame_name}")
    
    return results


@hydra.main(config_path="config", config_name="generate_VQA")
def main(cfg: DictConfig):
    # Initialize model pipeline with config parameters
    pipe = pipeline(
        "text-generation",
        model=cfg.model.name,
        torch_dtype=getattr(torch, cfg.model.dtype),
        device_map="auto",
        model_kwargs={"cache_dir": cfg.model.cache_dir}
    )
        
    # Process input directory from config
    all_frames = batch_process_analysis_files(cfg.data.input_dir)
    
    # Generate questions for each frame and save the results as JSON
    for frame_name, messages in all_frames.items():
        for message in messages:
            prompt = message['content']
            response = pipe(prompt, max_length=cfg.generation.max_length)[0]['generated_text']
            message['response'] = response
            
        # Save the generated questions
        output_file = Path(cfg.data.output_dir) / Path(cfg.data.input_dir).name
        os.makedirs(output_file, exist_ok=True)
        output_file /= f"{frame_name}_questions.json"
        # find the json part of the output response and write it to the output file
        with open(output_file, 'w') as f:
            json.dump(messages, f, indent=1)
        logger.info(f"Generated questions for frame {frame_name}. Saved to {output_file}")

if __name__ == "__main__":
    main()

