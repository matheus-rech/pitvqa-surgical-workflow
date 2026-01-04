#!/usr/bin/env python3
"""
Unified VLM Training for Multi-Task Surgical Understanding
Fine-tune Qwen2-VL on mmrech/pitvqa-unified-vlm for:
1. Phase Classification
2. Step Classification
3. Instrument Pointing
4. Anatomy Pointing

Uses SFTTrainer with PEFT/LoRA for efficient VLM fine-tuning
Fixed for Qwen2-VL's dynamic resolution image processing
"""

import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Configuration
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATASET_ID = "mmrech/pitvqa-unified-vlm"
OUTPUT_DIR = "pitvqa-qwen2vl-unified"
HF_REPO = "mmrech/pitvqa-qwen2vl-unified"

# Training hyperparameters (optimized for multi-task)
BATCH_SIZE = 1  # Reduced for variable-length pixel_values
GRADIENT_ACCUMULATION = 16  # Effective batch size: 16
LEARNING_RATE = 1e-4  # Lower for multi-task stability
NUM_EPOCHS = 3  # Adjust based on dataset size
MAX_SEQ_LENGTH = 512


@dataclass
class Qwen2VLDataCollator:
    """
    Custom data collator for Qwen2-VL that handles variable-length pixel_values.
    Processes images on-the-fly instead of pre-processing in map().
    """
    processor: Any

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Extract text and images from examples
        texts = []
        images = []

        for example in examples:
            # Get the formatted text (already processed)
            text = example.get('text', '')
            image = example.get('image')

            texts.append(text)
            images.append([image] if image else None)

        # Process with Qwen2-VL processor
        # This handles dynamic resolution per image
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

        # Create labels for causal LM training
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels (-100 = ignore in loss)
        batch["labels"][batch["labels"] == self.processor.tokenizer.pad_token_id] = -100

        return batch


def formatting_func(example):
    """Format a single example into conversation text."""
    messages_str = example['messages']
    image = example['image']

    # Parse messages
    messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str

    user_content = messages[0]['content']
    assistant_content = messages[1]['content']

    # Build conversation in Qwen2-VL format
    # Note: We don't include the image in the conversation dict here
    # The data collator will handle image processing
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
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

    return conversation


def main():
    print("=" * 70)
    print("UNIFIED VLM TRAINING - Multi-Task Surgical Understanding")
    print("=" * 70)
    print(f"Tasks: Phase Classification, Step Classification,")
    print(f"       Instrument Pointing, Anatomy Pointing")
    print("=" * 70)

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

    # Show task distribution
    print("\nTask distribution (train):")
    task_counts = {}
    for sample in dataset['train']:
        task = sample.get('task_type', 'unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    for task, count in sorted(task_counts.items()):
        pct = count / len(dataset['train']) * 100
        print(f"  {task}: {count} ({pct:.1f}%)")

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

    # LoRA config - higher rank for multi-task learning
    lora_config = LoraConfig(
        r=32,  # Increased for multi-task
        lora_alpha=64,
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

    # Format dataset: add formatted text column
    print("\nFormatting datasets...")

    def add_formatted_text(example):
        """Add formatted conversation text to example."""
        conversation = formatting_func(example)
        # Apply chat template to get the text
        text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    train_dataset = dataset['train'].map(
        add_formatted_text,
        num_proc=4,
        desc="Formatting train"
    )
    eval_dataset = dataset['validation'].map(
        add_formatted_text,
        num_proc=4,
        desc="Formatting validation"
    )

    print(f"  Formatted train: {len(train_dataset)} samples")
    print(f"  Formatted val: {len(eval_dataset)} samples")

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
        eval_steps=200,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        max_seq_length=MAX_SEQ_LENGTH,
        hub_model_id=HF_REPO,
        push_to_hub=True,
        report_to="none",
        # Important: Use dataset_text_field to specify which field has the text
        dataset_text_field="text",
        # Don't remove columns so we keep the image
        remove_unused_columns=False,
    )

    # Create custom data collator
    data_collator = Qwen2VLDataCollator(processor=processor)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  LoRA rank: 32")
    print("=" * 70)

    trainer.train()

    # Save and push
    print("\nSaving model...")
    trainer.save_model()

    print(f"\nPushing to Hub: {HF_REPO}")
    trainer.push_to_hub()

    print("\n" + "=" * 70)
    print("UNIFIED MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model: https://huggingface.co/{HF_REPO}")
    print("\nCapabilities:")
    print("  1. Surgical Phase Identification")
    print("  2. Surgical Step Identification")
    print("  3. Instrument Pointing (spatial localization)")
    print("  4. Anatomy Pointing (spatial localization)")
    print("=" * 70)


if __name__ == "__main__":
    main()
