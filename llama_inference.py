from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional
from trl import setup_chat_format
from peft import (
    PeftModel,
)

class LlamaInference:
    def __init__(
        self,
        base_model_path: str,
        new_model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        system_instruction: Optional[str] = None,
        cache_dir: Optional[str] = "model_cache"
    ):
        """
        Initialize the LlamaInference class.
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to run the model on ("cuda" or "cpu")
            system_instruction: Optional system instruction to use for all prompts
        """
        self.device = device
        self.system_instruction = system_instruction or (
            "You are an expert in construction site operations, specifically focusing on wheel loaders. "
            "Given the following scene description and object detections within the scene, "
            "Answer the following questions."
        )
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        base_model_reload= AutoModelForCausalLM.from_pretrained(
            base_model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=cache_dir
        )
        # Merge adapter with base model
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            self.tokenizer.chat_template = None  # Reset the chat template
        base_model_reload, self.tokenizer = setup_chat_format(base_model_reload, self.tokenizer)
        model = PeftModel.from_pretrained(base_model_reload, new_model_path)

        self.model = model.merge_and_unload()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_response(
        self,
        instruction: str,
        user_message: str,
        max_new_tokens: int = 150,
    ) -> str:
        """
        Generate a response following the provided inference pattern.
        
        Args:
            instruction: System instruction/prompt
            user_message: The user's input message
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response string
        """
        # Format messages
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_message}
        ]
        
        # Apply chat template exactly as in the example
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize input as in the example
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1
            )
        
        # Decode and split response as in the example
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.split("assistant")[1].strip()
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response for a list of chat messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response string
        """
        # Ensure system message is first
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": self.system_instruction})
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()
    


# Example Usage
if __name__ == "__main__":
    # Initialize the inference class
    inference = LlamaInference(
        base_model_path="meta-llama/Llama-3.2-1B-Instruct",
        new_model_path="/export/home/werbya/VLA_Embodied_AI/llama-3.2-1b-sensmore-QA", 
        device="cuda" 
    )

    # Simple single message generation
    # Define your instruction
    instruction = """You are an expert in construction site operations, specifically focusing on wheel loaders.
        Given the following scene description and object detections within the scene,
        Answer the following questions."""

    # Get response
    response = inference.generate_response(
        instruction=instruction,
        user_message="In this scene, there's a wheel loader moving dirt. What should it do next?",
        max_new_tokens=150
    )
    print("\n ------------------------------------------ \n")
    print(response)

    # # For a multi-turn conversation
    # messages = [
    #     {"role": "user", "content": "There's a wheel loader with a full bucket. What should the operator do?"},
    #     {"role": "assistant", "content": "The operator should carefully transport the load to the designated dumping area..."},
    #     {"role": "user", "content": "What safety precautions should they take?"}
    # ]
    # response = inference.chat(messages)
    # print(response)