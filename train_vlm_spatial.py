#!/usr/bin/env python3
"""
VLM Spatial Training for Surgical Instrument Localization
Fine-tune Qwen2-VL on mmrech/pitvqa-spatial-vlm with coordinate pointing

Uses SFTTrainer with PEFT/LoRA for efficient VLM fine-tuning
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from PIL import Image

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # Start with base model
DATASET_ID = "mmrech/pitvqa-spatial-vlm"
OUTPUT_DIR = "pitvqa-qwen2vl-spatial"
HF_REPO = "mmrech/pitvqa-qwen2vl-spatial"

# Training hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 512

def format_sample(sample, processor):
    """Format sample for Qwen2-VL training."""
    image = sample['image']
    messages = json.loads(sample['messages'])

    # Build conversation
    user_content = messages[0]['content']
    assistant_content = messages[1]['content']

    # Qwen2-VL format with image placeholder
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_content}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_content}
            ]
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text, "images": [image]}


def main():
    print("=" * 60)
    print("VLM Spatial Training - Surgical Instrument Localization")
    print("=" * 60)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print(f"\nLoading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID)
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Val: {len(dataset['validation'])} samples")

    # Quantization config for efficient training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and processor
    print(f"\nLoading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config targeting VLM layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Preprocessing function
    def preprocess_function(examples):
        """Process batch of examples for training."""
        texts = []
        all_images = []

        for i in range(len(examples['image'])):
            image = examples['image'][i]
            messages = json.loads(examples['messages'][i])

            user_content = messages[0]['content']
            assistant_content = messages[1]['content']

            # Build conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_content}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": assistant_content}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            all_images.append([image])

        # Process with processor
        batch = processor(
            text=texts,
            images=all_images,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        # Set labels for causal LM
        batch["labels"] = batch["input_ids"].clone()

        return batch

    # Process datasets
    print("\nProcessing datasets...")
    train_dataset = dataset['train'].map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset['train'].column_names,
    )

    eval_dataset = dataset['validation'].map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset['validation'].column_names,
    )

    # Training config
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        hub_model_id=HF_REPO,
        push_to_hub=True,
        report_to="none",
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()

    # Save and push
    print("\nSaving model...")
    trainer.save_model()

    print(f"\nPushing to Hub: {HF_REPO}")
    trainer.push_to_hub()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: https://huggingface.co/{HF_REPO}")
    print("=" * 60)


if __name__ == "__main__":
    main()
