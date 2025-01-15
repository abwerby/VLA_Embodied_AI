import torch
from transformers import pipeline






import json
from pathlib import Path
from typing import Dict, List, Union
import logging

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
                         typical wheel-loader operations.
                         \n\n \
                         output format: make the question-answer pairs in JSON format, with the question as the key and the answer as the value."""
        
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

# Example usage
if __name__ == "__main__":
    
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"cache_dir": "model_cache"}
    )
        
    # Process a directory of files
    directory_path = "/export/home/werbya/VLA/vqa_dataset/clip3_frames"
    all_frames = batch_process_analysis_files(directory_path)
    
    # Generate questions for each frame and save the results as JSON
    for frame_name, messages in all_frames.items():
        for message in messages:
            prompt = message['content']
            response = pipe(prompt, max_length=1000)[0]['generated_text']
            message['response'] = response
            
        # Save the generated questions
        output_file = Path(directory_path) / f"{frame_name}_questions.json"
        with open(output_file, 'w') as f:
            json.dump(messages, f, indent=2)
            logger.info(f"Saved generated questions for frame {frame_name} to {output_file}")
            


