import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# -----------------------------
# Model
# -----------------------------
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -----------------------------
# QLoRA 4-bit config
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 compute on A40
)

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# 🔑 VERY IMPORTANT for QLoRA
model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA config
# -----------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Explicitly disable gradient checkpointing to avoid bitsandbytes errors
if hasattr(model, 'gradient_checkpointing_disable'):
    model.gradient_checkpointing_disable()

# -----------------------------
# Dataset
# -----------------------------
dataset = load_dataset("json", data_files="./slm1_train.jsonl")

# Load schema for context
with open("./schema.txt", "r") as f:
    schema_context = f.read()

# -----------------------------
# Tokenize dataset with Llama 3.2 Instruct format
# -----------------------------
def tokenize_function(example):
    # Use schema_context if context field doesn't exist
    context = example.get('context', schema_context)
    
    # Format according to Llama 3.2 Instruct template
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that converts natural language questions into SPARQL queries using the DBpedia ontology.<|eot_id|><|start_header_id|>user<|end_header_id|>

{example['instruction']}

Schema:
{context}

Question:
{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
    
    # Tokenize the text
    result = tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # Add labels for causal LM training
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize the dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names,
    batched=False
)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="outputs/Llama-3.2-3B-TEXT2SPARQL",
    per_device_train_batch_size=2,  # Reduced to avoid triggering gradient checkpointing
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 16
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,          # ✅ use bf16
    fp16=False,        # ❌ disable fp16 scaler
    logging_steps=50,
    save_steps=100,  # Save more frequently to avoid losing progress
    save_total_limit=3,  # Keep more checkpoints
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to="none",
    optim="adamw_torch",  # Use standard PyTorch optimizer instead of 8-bit
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    weight_decay=0.001,
    save_safetensors=True,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    args=training_args,
)

# -----------------------------
# Find latest checkpoint for resumption
# -----------------------------
output_dir = Path(training_args.output_dir)
last_checkpoint = None
if output_dir.exists():
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = str(max(checkpoints, key=lambda x: int(x.name.split("-")[-1])))
        print(f"\n🔄 Resuming training from checkpoint: {last_checkpoint}\n")
    else:
        print("\n🆕 Starting fresh training (no checkpoints found)\n")
else:
    print("\n🆕 Starting fresh training\n")

# -----------------------------
# Train
# -----------------------------
try:
    trainer.train(resume_from_checkpoint=last_checkpoint)
    print("\n✅ Training completed successfully!\n")
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user. Progress saved.\n")
except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("Progress saved to latest checkpoint.\n")
    raise

# -----------------------------
# Save adapters
# -----------------------------
trainer.save_model("outputs/Llama-3.2-3B-TEXT2SPARQL")