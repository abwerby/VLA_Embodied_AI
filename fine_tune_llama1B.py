from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import datasets
import bitsandbytes as bnb
import torch
import os, torch, wandb, sys
from trl import SFTTrainer, setup_chat_format, SFTConfig
import loguru
import json
import glob
import pandas as pd
import functools

# setup wandb
wandb.login()
run = wandb.init(
    project='Fine-tune Llama 3.2 for VQA', 
    job_type="training", 
    anonymous="allow"
)

sys_instruction = "You are an expert in construction site operations, specifically focusing on wheel loaders.\
                Given the following scene description and object detections within the scene, \
                Answer the following questions."

def find_all_linear_names(model, quantized=False):
    cls = bnb.nn.Linear4bit if quantized else torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16 bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_dataset(jsons_path):
    """
    Load the dataset from the JSON files.
    
    Args:
        jsons_path: Path to the directory containing the JSON files
        
    Returns:
        Dataset containing the JSON data
    """
    # all json files in the directory
    json_files = glob.glob(os.path.join(jsons_path, '*.json'))
    dataset = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        caption = data.get('captions', '')
        questions = data.get('questions', '{}')
        questions_list = list(questions.items())
        # generate example for each question
        for i, (question, answer) in enumerate(questions_list):
            caption_Q = f"{caption} {question}"
            example = {
                "caption": caption_Q,
                "answer": answer
            }
            dataset.append(example)
    df = pd.DataFrame(dataset)
    dataset = datasets.Dataset.from_pandas(df)
    return dataset

def format_chat_template(row, tokenizer):
    """
    Format the chat template for the given row.
    Args:
        row: Row of the dataset
        tokenizer: Tokenizer to use
    Returns:
        Row with the chat template applied
    """
    row_json = [
                    {"role": "system", "content": sys_instruction},
                    {"role": "user", "content": row["caption"]},
                    {"role": "assistant", "content": row["answer"]}
                ]
    
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row 


def main():
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    new_model = "llama-3.2-1b-sensmore-QA"
    jsons_path = "/export/home/werbya/VLA_Embodied_AI/VQA_dataset/clip3_frames"


    # load the raw dataset
    dataset = load_dataset(jsons_path)
    loguru.logger.info(f"Loaded raw dataset with {len(dataset)} examples")

    # set torch dtype and attn_implementation
    torch_dtype = torch.float16
    attn_implementation = "eager"

    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
        cache_dir="model_cache",
    )
    loguru.logger.info(f"Loaded model {base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    loguru.logger.info(f"Loaded tokenizer for {base_model}")

    # Prepare dataset
    dataset = dataset.map(
        functools.partial(format_chat_template, tokenizer=tokenizer),
        num_proc= 4,
    )
    loguru.logger.info(f"Prepared dataset with {len(dataset)} examples")

    # split the dataset into training and validation
    dataset = dataset.train_test_split(test_size=0.1)
    loguru.logger.info(f"Split dataset into training and validation")

    # find all linear names
    modules = find_all_linear_names(model, quantized=False)
    loguru.logger.info(f"linear modules: {modules}")

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=modules
    )
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        tokenizer.chat_template = None  # Reset the chat template
    model, tokenizer = setup_chat_format(model, tokenizer)
    model = get_peft_model(model, peft_config)
    loguru.logger.info(f"Loaded model with Peft")

    #Hyperparamter
    training_arguments = TrainingArguments(
        output_dir=new_model,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=0.2,
        logging_steps=1,
        warmup_steps=10,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        report_to="wandb"
    )
    loguru.logger.info(f"Loaded training arguments")

    # Setting sft parameters
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        max_seq_length= 512,
        dataset_text_field="text",
        args=training_arguments,
        packing= False,
    )

    # Train the model
    loguru.logger.info(f"Training the model..")
    trainer.train()
    loguru.logger.info(f"Training completed")

    # Save the model
    trainer.model.save_pretrained(new_model)
    loguru.logger.info(f"Model saved to {new_model}")

    wandb.finish()

if __name__ == "__main__":
    main()
