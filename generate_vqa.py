import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import os
from pathlib import Path
from typing import Dict, List, Union
import logging
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


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
        system_prompt = """
                            You are an expert in construction site operations, specifically focusing on wheel loaders. \
                            Given the following frame description and object detections, Given a scene caption of the operation \
                            generate 2-3 question-answer pairs related to what the loader should do, or \
                            next steps to take (load, unload, go to pile). Your answers should be concise, based only on the caption.\n\
                            Answer of the question should be based on the current state of the loader and the scene\n \
                            output format: make the question-answer pairs in JSON format, with the question as the key and the answer as the value.\
                            Don't include the system prompt in the output, Don't write any Code, just write the question-answer pairs in JSON format.\n \
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
    return results


@hydra.main(config_path="config", config_name="generate_VQA")
def main(cfg: DictConfig):
    # Initialize model pipeline with config parameters
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cfg.model.cache_dir
    )

    # Set pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Process input directory from config
    all_frames = batch_process_analysis_files(cfg.data.input_dir)
    
    # Generate questions for each frame and save the results as JSON
    # NOTE: This could be parallelized with batch processing
    for frame_name, messages in tqdm(all_frames.items()):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = pipe(prompt, max_length=cfg.generation.max_length, do_sample=True)
        response = response[0]["generated_text"].split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[1]
        # Save the generated questions
        output_file = Path(cfg.data.output_dir) / Path(cfg.data.input_dir).name
        os.makedirs(output_file, exist_ok=True)
        output_file /= f"{frame_name}_questions.json"
        res = {"frame_name": frame_name, "captions": messages[1]['content'], "questions": response}
        with open(output_file, 'w') as f:
            json.dump(res, f, indent=4)

if __name__ == "__main__":
    main()

