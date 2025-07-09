import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel # Import PeftModel for loading adapters

# --- Configuration ---

base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"

finetuned_model_path = "./mistral-finetuned-sql-model"

MAX_SEQ_LENGTH = 384 # Keep this consistent with your training

# --- 1. Load Tokenizer ---

print(f"Loading tokenizer from {finetuned_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# --- 2. Load Base Model with Quantization Config ---
print(f"Loading base model '{base_model_name}' with QLoRA configuration...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # Or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for better precision if GPU supports
    bnb_4bit_use_double_quant=True, # Double quantization for potentially better performance
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto", # Automatically maps model to available devices (GPU if present)
    trust_remote_code=True,
)

# --- 3. Load PEFT Adapters onto the Base Model ---

print(f"Loading PEFT adapters from {finetuned_model_path}...")
model = PeftModel.from_pretrained(base_model, finetuned_model_path)

print("Fine-tuned model loaded successfully!")

# Set the model to evaluation mode for inference
model.eval()

# --- 4. Define the Inference Function ---
def generate_sql(question, model, tokenizer, max_new_tokens=256, temperature=0.7, top_p=0.9):
    # Apply Mistral's instruction format: <s>[INST] Instruction [/INST]
    # Do not add </s> at the end of the prompt, as the model should generate it.
    prompt = f"<s>[INST] {question} [/INST]"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, truncation=True, max_length=MAX_SEQ_LENGTH)

    # Move inputs to the same device as the model (usually GPU if available)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # Generate the response
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id, # Stop generation at EOS token
            do_sample=True, # Use sampling for more diverse outputs
            temperature=temperature, # Control randomness
            top_p=top_p,       # Nucleus sampling
            num_return_sequences=1, # Generate a single sequence
        )

    # Decode the generated tokens.
    # The output includes the original prompt and the generated response.
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the generated SQL part (everything after [/INST] and before </s>)
    response_start_tag = "[/INST]"
    response_end_tag = "</s>"

    if response_start_tag in decoded_output:
        # Get everything after [/INST]
        generated_text = decoded_output.split(response_start_tag, 1)[1].strip()
        # Remove any trailing </s> if present
        if generated_text.endswith(response_end_tag):
            generated_text = generated_text[:-len(response_end_tag)].strip()
        return generated_text
    else:
        # Fallback if parsing fails (e.g., if model doesn't generate the tags)
        print(f"Warning: Could not parse response with tags. Full decoded output: {decoded_output}")
        return decoded_output


# --- 5. Example Usage ---
if __name__ == "__main__":
    print("\n--- Testing the Fine-tuned SQL Model ---")

    while True:
        user_question = input("Enter your question (or 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break

        print("\nGenerating SQL...")
        generated_sql = generate_sql(user_question, model, tokenizer)
        print(f"Question: {user_question}\nGenerated SQL:\n{generated_sql}\n")

    print("Exiting inference script.")