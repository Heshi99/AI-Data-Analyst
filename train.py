import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig # Import for QLoRA
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# --- Configuration ---
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# IMPORTANT: Adjust this path to your dataset's location on Google Drive
dataset_path = "/content/mistral_spider_train.jsonl"
MAX_SEQ_LENGTH = 384 # Increased max length for potentially longer SQL/schemas
TRAINING_SAMPLES = 1000 # Aim for more data, adjust based on resource limits and Colab limits

# --- 1. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Important for causal LMs

# --- 2. Load Model with QLoRA (Recommended for Free Tiers / Limited VRAM) ---
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for better precision if GPU supports (e.g., T4, A100)
    bnb_4bit_use_double_quant=True, # Double quantization for potentially better performance
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # Apply quantization config
    device_map="auto", # Automatically maps model to available devices (GPU if present)
    trust_remote_code=True,
    #attn_implementation="eager" # Eager is generally safer for debugging for some environments
)

# --- 3. Apply LoRA ---
lora_config = LoraConfig(
    r=32, # Increased rank for more capacity
    lora_alpha=64, # Alpha typically 2*r
    # Added more target modules common for Mistral instruction fine-tuning
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # Good to see what percentage of params are trainable

# --- 4. Load & Preprocess Dataset ---
dataset = load_dataset("json", data_files=dataset_path)["train"]
# Take a larger subset, adjust based on your machine's capability and desired training length
dataset = dataset.select(range(min(TRAINING_SAMPLES, len(dataset))))
print(f"Training on {len(dataset)} samples")

def tokenize_function(example):
    # Construct the full prompt as given to your agent for fine-tuning
    # Ensure example['prompt'] contains:
    # "You are a professional data analyst. Given the table below, write an SQL query to answer the
    # user's question.\n\nTable Schema:\n{table_description}\n\nUser Question:\n{question}\n\nReturn only the SQL query, no explanation."
    # And example['response'] contains the SQL.
    
    # Use Mistral's instruction format: <s>[INST] Instruction [/INST] Response</s>
    full_text = f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    
    tokenized_input = tokenizer(
        full_text,
        truncation=True,
        padding="max_length", # Or 'longest' if you want dynamic padding
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt" # KEEP this as "pt" so it returns tensors initially
    )
    
    # Labels should be the input_ids, with -100 for tokens not to compute loss on.
    # For full sequence training (simplest), just copy input_ids.
    labels = tokenized_input["input_ids"].clone()
    
    # Squeeze the 1-dim batch dimension for individual examples before the collator stacks them
    return {
        "input_ids": tokenized_input["input_ids"].squeeze(0),
        "attention_mask": tokenized_input["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0) # Also squeeze labels
    }

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False, # Process one by one to ensure .squeeze(0) works correctly per example
    remove_columns=dataset.column_names # Remove original columns to keep only 'input_ids', 'attention_mask', 'labels'
)

# Split into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42) # 10% for validation
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# --- 5. Configure Training Arguments ---
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3, # Train for more epochs
    logging_steps=50,
    learning_rate=2e-5, # Slightly lower learning rate for stability
    save_strategy="epoch", # Save model at end of each epoch
    save_total_limit=1, # Keep only the last checkpoint
    report_to="none", # Or "wandb" for better tracking
    dataloader_pin_memory=True, # Improves data loading speed on GPU
    dataloader_num_workers=2, # Use multiple workers for data loading
    remove_unused_columns=False, # Keep all columns for DataCollator
    # IMPORTANT: Match this to bnb_config's compute_dtype if using QLoRA
    bf16=True, # Enable bfloat16 mixed precision if GPU supports
    fp16=False, # Set to True if your GPU only supports float16, otherwise False
    max_steps=-1, # Don't limit by steps, let epochs control training length
    eval_strategy="epoch", # Evaluate at the end of each epoch (formerly evaluation_strategy)
    eval_steps=None, # Evaluate at end of epoch
    load_best_model_at_end=True, # Load the best model based on eval_loss
    metric_for_best_model="eval_loss",
)

# --- 6. Data Collator ---
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False, # Crucial for causal language modeling
    return_tensors="pt"
)

# --- 7. Initialize and Start Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # Use the split training set
    eval_dataset=eval_dataset, # Add the evaluation set
    data_collator=data_collator,
    # processing_class=tokenizer # This argument is not standard/needed here
)

print("Starting training...")
trainer.train()

# --- 8. Save Model ---
# IMPORTANT: Adjust this path to your desired save location on Google Drive
save_path = "/content/drive/MyDrive/Colab_LLM_FineTune/mistral-finetuned-sql-model"
model.save_pretrained(save_path) # Save the PEFT adapters
tokenizer.save_pretrained(save_path) # Save the tokenizer
print(f"Training completed and model saved to {save_path}!")