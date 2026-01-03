"""
Agent 3: HuggingFace Skills Integration for SAGE/Molmo Fine-tuning

This module integrates with HuggingFace Skills Training infrastructure for:
- SFT (Supervised Fine-Tuning) with pointing annotations
- DPO (Direct Preference Optimization) for preference learning
- GRPO (Group Relative Policy Optimization) for RL-based training

Reference: https://huggingface.co/blog/hf-skills-training

Training is designed to work with:
- Claude Code + HF Skills plugin
- Codex + AGENTS.md
- Gemini CLI extensions
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class TrainingMethod(Enum):
    """Supported training methods."""
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


class ModelSize(Enum):
    """Supported SAGE/Molmo model sizes."""
    MOLMO_4B = "allenai/Molmo-7B-D-0924"  # 4B variant
    MOLMO_7B = "allenai/Molmo-7B-D-0924"  # 7B variant
    SAGE_MOLMO_8B_SFT = "allenai/SAGE-MM-Molmo2-8B-SFT"
    SAGE_MOLMO_8B_SFT_RL = "allenai/SAGE-MM-Molmo2-8B-SFT_RL"
    SAGE_QWEN_4B = "allenai/SAGE-MM-Qwen3-VL-4B-SFT_RL"
    SAGE_QWEN_8B = "allenai/SAGE-MM-Qwen3-VL-8B-SFT_RL"


class HardwareTier(Enum):
    """Available hardware tiers from HF Skills."""
    T4_SMALL = "t4-small"       # ~$0.75/hr, <1B models
    T4_MEDIUM = "t4-medium"     # ~$1.50/hr, 1-3B models
    A10G_SMALL = "a10g-small"   # ~$5/hr, 3-7B models
    A10G_LARGE = "a10g-large"   # ~$15/hr, 7B+ with LoRA


@dataclass
class TrainingConfig:
    """Configuration for HuggingFace Skills training."""

    # Model
    base_model: str
    output_model_name: str

    # Dataset
    dataset_id: str
    dataset_split: str = "train"

    # Training method
    method: TrainingMethod = TrainingMethod.SFT

    # Hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 2048

    # LoRA config (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # Hardware
    hardware_tier: HardwareTier = HardwareTier.A10G_LARGE

    # VLM-specific
    is_vision_model: bool = True
    image_column: str = "images"
    max_images_per_sample: int = 4

    # DPO-specific
    dpo_beta: float = 0.1

    # GRPO-specific
    grpo_num_generations: int = 4
    grpo_reward_functions: List[str] = field(default_factory=list)

    # Output
    push_to_hub: bool = True
    hub_token: Optional[str] = None

    def to_trl_config(self) -> Dict[str, Any]:
        """Convert to TRL trainer config format."""
        config = {
            "model_name_or_path": self.base_model,
            "dataset_name": self.dataset_id,
            "output_dir": f"./outputs/{self.output_model_name}",
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.output_model_name,
        }

        if self.use_lora:
            config["use_peft"] = True
            config["lora_r"] = self.lora_r
            config["lora_alpha"] = self.lora_alpha
            config["lora_dropout"] = self.lora_dropout
            config["lora_target_modules"] = self.lora_target_modules

        if self.method == TrainingMethod.DPO:
            config["beta"] = self.dpo_beta

        if self.method == TrainingMethod.GRPO:
            config["num_generations"] = self.grpo_num_generations

        return config


@dataclass
class GRPORewardConfig:
    """Configuration for GRPO reward functions."""

    # Pointing accuracy reward
    pointing_weight: float = 0.3
    pointing_threshold: float = 0.1  # 10% of image size

    # Phase classification reward
    phase_weight: float = 0.2

    # Step classification reward
    step_weight: float = 0.2

    # Instrument detection reward
    instrument_weight: float = 0.2

    # Format compliance reward
    format_weight: float = 0.1


class SurgicalRewardFunctions:
    """
    GRPO reward functions for surgical video understanding.

    These reward functions evaluate model outputs for:
    1. Pointing accuracy (spatial grounding)
    2. Phase/step classification
    3. Instrument detection
    4. Answer format compliance
    """

    def __init__(self, config: Optional[GRPORewardConfig] = None):
        self.config = config or GRPORewardConfig()

    def pointing_accuracy_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """
        Reward for correct spatial pointing.

        Checks if predicted point coordinates are within threshold
        of ground truth locations.
        """
        import re

        # Extract points from prediction
        point_pattern = r"<point x='([\d.]+)' y='([\d.]+)'>([^<]+)</point>"
        pred_points = re.findall(point_pattern, prediction)

        if not pred_points:
            return 0.0

        gt_points = ground_truth.get("points", [])
        if not gt_points:
            return 0.5  # Neutral if no ground truth

        # Calculate accuracy
        correct = 0
        for px, py, label in pred_points:
            px, py = float(px), float(py)

            for gt in gt_points:
                if gt["label"].lower() in label.lower():
                    dist = ((px - gt["x"])**2 + (py - gt["y"])**2)**0.5
                    if dist < self.config.pointing_threshold:
                        correct += 1
                        break

        return correct / max(len(pred_points), len(gt_points))

    def phase_classification_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Reward for correct surgical phase identification."""
        gt_phase = ground_truth.get("phase", "").lower()
        if not gt_phase:
            return 0.5

        # Check if phase is mentioned in prediction
        if gt_phase.replace("_", " ") in prediction.lower():
            return 1.0
        return 0.0

    def step_classification_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Reward for correct surgical step identification."""
        gt_step = ground_truth.get("step", "").lower()
        if not gt_step:
            return 0.5

        if gt_step.replace("_", " ") in prediction.lower():
            return 1.0
        return 0.0

    def instrument_detection_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Reward for correct instrument identification."""
        gt_instruments = ground_truth.get("instruments", [])
        if not gt_instruments:
            return 0.5

        prediction_lower = prediction.lower()
        detected = sum(
            1 for inst in gt_instruments
            if inst.lower().replace("_", " ") in prediction_lower
        )

        return detected / len(gt_instruments)

    def format_compliance_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Reward for proper response format."""
        rewards = []

        # Check for pointing format if expected
        if ground_truth.get("points"):
            has_points = "<point" in prediction and "</point>" in prediction
            rewards.append(1.0 if has_points else 0.0)

        # Check for reasonable length
        word_count = len(prediction.split())
        if 10 <= word_count <= 200:
            rewards.append(1.0)
        elif word_count < 5 or word_count > 500:
            rewards.append(0.0)
        else:
            rewards.append(0.5)

        return sum(rewards) / max(len(rewards), 1)

    def compute_reward(
        self,
        prediction: str,
        ground_truth: Dict[str, Any]
    ) -> float:
        """Compute total weighted reward."""
        rewards = {
            "pointing": self.pointing_accuracy_reward(prediction, ground_truth),
            "phase": self.phase_classification_reward(prediction, ground_truth),
            "step": self.step_classification_reward(prediction, ground_truth),
            "instrument": self.instrument_detection_reward(prediction, ground_truth),
            "format": self.format_compliance_reward(prediction, ground_truth),
        }

        weights = {
            "pointing": self.config.pointing_weight,
            "phase": self.config.phase_weight,
            "step": self.config.step_weight,
            "instrument": self.config.instrument_weight,
            "format": self.config.format_weight,
        }

        total = sum(rewards[k] * weights[k] for k in rewards)
        return total


class HFSkillsTrainer:
    """
    Trainer class that integrates with HuggingFace Skills infrastructure.

    This class generates training configurations and scripts compatible with:
    - HF Skills CLI commands
    - TRL (Transformer Reinforcement Learning) library
    - HuggingFace Jobs infrastructure
    """

    def __init__(
        self,
        config: TrainingConfig,
        reward_config: Optional[GRPORewardConfig] = None
    ):
        self.config = config
        self.reward_functions = SurgicalRewardFunctions(reward_config)

    def generate_training_script(self, output_path: str) -> str:
        """
        Generate a training script for HF Skills.

        The script can be executed via:
        - HF Skills CLI
        - Direct Python execution
        - HuggingFace Jobs
        """
        script_content = self._build_script()

        with open(output_path, 'w') as f:
            f.write(script_content)

        logger.info(f"Generated training script: {output_path}")
        return output_path

    def _build_script(self) -> str:
        """Build the training script content."""
        if self.config.method == TrainingMethod.SFT:
            return self._build_sft_script()
        elif self.config.method == TrainingMethod.DPO:
            return self._build_dpo_script()
        elif self.config.method == TrainingMethod.GRPO:
            return self._build_grpo_script()
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _build_sft_script(self) -> str:
        """Generate SFT training script."""
        return f'''#!/usr/bin/env python3
"""
PitVQA SAGE/Molmo SFT Training Script

Generated by Agent 3 for surgical video understanding fine-tuning.
Compatible with HuggingFace Skills Training.

Usage:
    python train_sft.py

Or via HF Skills:
    Fine-tune {self.config.base_model} on {self.config.dataset_id}
"""

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# Configuration
MODEL_NAME = "{self.config.base_model}"
DATASET_NAME = "{self.config.dataset_id}"
OUTPUT_NAME = "{self.config.output_model_name}"

# Load dataset
print(f"Loading dataset: {{DATASET_NAME}}")
dataset = load_dataset(DATASET_NAME, split="{self.config.dataset_split}")

# Load model and processor
print(f"Loading model: {{MODEL_NAME}}")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply LoRA
lora_config = LoraConfig(
    r={self.config.lora_r},
    lora_alpha={self.config.lora_alpha},
    lora_dropout={self.config.lora_dropout},
    target_modules={self.config.lora_target_modules},
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir=f"./outputs/{{OUTPUT_NAME}}",
    num_train_epochs={self.config.num_epochs},
    per_device_train_batch_size={self.config.batch_size},
    learning_rate={self.config.learning_rate},
    warmup_ratio={self.config.warmup_ratio},
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub={str(self.config.push_to_hub).lower()},
    hub_model_id=OUTPUT_NAME,
    report_to="tensorboard",
)

# Data collator for VLM
def collate_fn(examples):
    texts = []
    images = []

    for example in examples:
        messages = example["messages"]
        # Format as conversation
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{{role}}|>\\n{{content}}<|end|>\\n"
        texts.append(text)

        # Handle images
        if "images" in example and example["images"]:
            images.append(example["images"][0])
        else:
            images.append(None)

    # Process with tokenizer
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length={self.config.max_seq_length},
        return_tensors="pt"
    )

    batch["labels"] = batch["input_ids"].clone()
    return batch

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# Train
print("Starting training...")
trainer.train()

# Save and push
print("Saving model...")
trainer.save_model()
if {str(self.config.push_to_hub).lower()}:
    trainer.push_to_hub()

print(f"Training complete! Model saved to: {{OUTPUT_NAME}}")
'''

    def _build_dpo_script(self) -> str:
        """Generate DPO training script."""
        return f'''#!/usr/bin/env python3
"""
PitVQA SAGE/Molmo DPO Training Script

Direct Preference Optimization for surgical video understanding.
Trains model to prefer correct identifications over incorrect ones.

Usage:
    python train_dpo.py

Or via HF Skills:
    Run DPO on {self.config.dataset_id} with {self.config.base_model}
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig

# Configuration
MODEL_NAME = "{self.config.base_model}"
DATASET_NAME = "{self.config.dataset_id}"
OUTPUT_NAME = "{self.config.output_model_name}"

# Load dataset (must have 'chosen' and 'rejected' columns)
print(f"Loading DPO dataset: {{DATASET_NAME}}")
dataset = load_dataset(DATASET_NAME, split="{self.config.dataset_split}")

# Load model
print(f"Loading model: {{MODEL_NAME}}")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Reference model (frozen copy)
ref_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
peft_config = LoraConfig(
    r={self.config.lora_r},
    lora_alpha={self.config.lora_alpha},
    lora_dropout={self.config.lora_dropout},
    target_modules={self.config.lora_target_modules},
    bias="none",
    task_type="CAUSAL_LM"
)

# DPO training config
training_args = DPOConfig(
    output_dir=f"./outputs/{{OUTPUT_NAME}}",
    num_train_epochs={self.config.num_epochs},
    per_device_train_batch_size={self.config.batch_size},
    learning_rate={self.config.learning_rate},
    beta={self.config.dpo_beta},
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub={str(self.config.push_to_hub).lower()},
    hub_model_id=OUTPUT_NAME,
)

# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

# Train
print("Starting DPO training...")
trainer.train()

# Save
print("Saving model...")
trainer.save_model()
if {str(self.config.push_to_hub).lower()}:
    trainer.push_to_hub()

print(f"DPO training complete! Model: {{OUTPUT_NAME}}")
'''

    def _build_grpo_script(self) -> str:
        """Generate GRPO training script with surgical reward functions."""
        return f'''#!/usr/bin/env python3
"""
PitVQA SAGE/Molmo GRPO Training Script

Group Relative Policy Optimization for surgical skill recognition.
Uses programmatic rewards for verifiable surgical understanding tasks.

Reward functions:
- Pointing accuracy (spatial grounding)
- Phase classification
- Step identification
- Instrument detection
- Format compliance

Usage:
    python train_grpo.py

Or via HF Skills:
    Train with GRPO on {self.config.dataset_id} using {self.config.base_model}
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# Configuration
MODEL_NAME = "{self.config.base_model}"
DATASET_NAME = "{self.config.dataset_id}"
OUTPUT_NAME = "{self.config.output_model_name}"

# Reward weights
POINTING_WEIGHT = {self.reward_functions.config.pointing_weight}
PHASE_WEIGHT = {self.reward_functions.config.phase_weight}
STEP_WEIGHT = {self.reward_functions.config.step_weight}
INSTRUMENT_WEIGHT = {self.reward_functions.config.instrument_weight}
FORMAT_WEIGHT = {self.reward_functions.config.format_weight}
POINTING_THRESHOLD = {self.reward_functions.config.pointing_threshold}


def compute_surgical_reward(prediction: str, ground_truth: dict) -> float:
    """
    Compute reward for surgical video understanding prediction.

    Combines multiple reward signals:
    1. Pointing accuracy (are spatial coordinates correct?)
    2. Phase classification (is the surgical phase correct?)
    3. Step identification (is the surgical step correct?)
    4. Instrument detection (are instruments correctly identified?)
    5. Format compliance (is the output well-formed?)
    """
    rewards = {{}}

    # 1. Pointing accuracy reward
    point_pattern = r"<point x='([\\d.]+)' y='([\\d.]+)'>([^<]+)</point>"
    pred_points = re.findall(point_pattern, prediction)
    gt_points = ground_truth.get("points", [])

    if pred_points and gt_points:
        correct = 0
        for px, py, label in pred_points:
            px, py = float(px), float(py)
            for gt in gt_points:
                if gt["label"].lower() in label.lower():
                    dist = ((px - gt["x"])**2 + (py - gt["y"])**2)**0.5
                    if dist < POINTING_THRESHOLD:
                        correct += 1
                        break
        rewards["pointing"] = correct / max(len(pred_points), len(gt_points))
    else:
        rewards["pointing"] = 0.5 if not gt_points else 0.0

    # 2. Phase classification reward
    gt_phase = ground_truth.get("phase", "").lower()
    rewards["phase"] = 1.0 if gt_phase and gt_phase.replace("_", " ") in prediction.lower() else 0.0

    # 3. Step classification reward
    gt_step = ground_truth.get("step", "").lower()
    rewards["step"] = 1.0 if gt_step and gt_step.replace("_", " ") in prediction.lower() else 0.0

    # 4. Instrument detection reward
    gt_instruments = ground_truth.get("instruments", [])
    if gt_instruments:
        pred_lower = prediction.lower()
        detected = sum(1 for inst in gt_instruments if inst.lower().replace("_", " ") in pred_lower)
        rewards["instrument"] = detected / len(gt_instruments)
    else:
        rewards["instrument"] = 0.5

    # 5. Format compliance reward
    format_score = 0.5
    if gt_points:
        format_score = 1.0 if "<point" in prediction else 0.0
    word_count = len(prediction.split())
    if 10 <= word_count <= 200:
        format_score = (format_score + 1.0) / 2
    rewards["format"] = format_score

    # Weighted sum
    total = (
        rewards["pointing"] * POINTING_WEIGHT +
        rewards["phase"] * PHASE_WEIGHT +
        rewards["step"] * STEP_WEIGHT +
        rewards["instrument"] * INSTRUMENT_WEIGHT +
        rewards["format"] * FORMAT_WEIGHT
    )

    return total


def reward_fn(samples: list, outputs: list, ground_truths: list) -> list:
    """Batch reward function for GRPO trainer."""
    rewards = []
    for output, gt in zip(outputs, ground_truths):
        reward = compute_surgical_reward(output, gt)
        rewards.append(reward)
    return rewards


# Load dataset
print(f"Loading GRPO dataset: {{DATASET_NAME}}")
dataset = load_dataset(DATASET_NAME, split="{self.config.dataset_split}")

# Load model
print(f"Loading model: {{MODEL_NAME}}")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA config
peft_config = LoraConfig(
    r={self.config.lora_r},
    lora_alpha={self.config.lora_alpha},
    lora_dropout={self.config.lora_dropout},
    target_modules={self.config.lora_target_modules},
    bias="none",
    task_type="CAUSAL_LM"
)

# GRPO training config
training_args = GRPOConfig(
    output_dir=f"./outputs/{{OUTPUT_NAME}}",
    num_train_epochs={self.config.num_epochs},
    per_device_train_batch_size={self.config.batch_size},
    learning_rate={self.config.learning_rate},
    num_generations={self.config.grpo_num_generations},
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    push_to_hub={str(self.config.push_to_hub).lower()},
    hub_model_id=OUTPUT_NAME,
)

# Initialize GRPO trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[reward_fn],
    peft_config=peft_config,
)

# Train
print("Starting GRPO training with surgical reward functions...")
trainer.train()

# Save
print("Saving model...")
trainer.save_model()
if {str(self.config.push_to_hub).lower()}:
    trainer.push_to_hub()

print(f"GRPO training complete! Model: {{OUTPUT_NAME}}")
'''

    def generate_hf_skills_prompt(self) -> str:
        """
        Generate a prompt for HuggingFace Skills CLI.

        This prompt can be used with Claude Code, Codex, or Gemini CLI
        to trigger automated training.
        """
        method_str = {
            TrainingMethod.SFT: "Fine-tune",
            TrainingMethod.DPO: "Run DPO on",
            TrainingMethod.GRPO: "Train with GRPO on"
        }[self.config.method]

        prompt = f"""{method_str} {self.config.base_model} on {self.config.dataset_id}

Configuration:
- Output model: {self.config.output_model_name}
- Epochs: {self.config.num_epochs}
- Batch size: {self.config.batch_size}
- Learning rate: {self.config.learning_rate}
- Use LoRA: {self.config.use_lora} (r={self.config.lora_r})
- Hardware: {self.config.hardware_tier.value}
- This is a vision language model with images in '{self.config.image_column}' column
"""

        if self.config.method == TrainingMethod.DPO:
            prompt += f"- DPO beta: {self.config.dpo_beta}\n"
            prompt += "- Dataset has 'chosen' and 'rejected' columns for preferences\n"

        if self.config.method == TrainingMethod.GRPO:
            prompt += f"- GRPO generations: {self.config.grpo_num_generations}\n"
            prompt += "- Custom surgical reward functions for pointing, phase, step, instrument detection\n"

        return prompt


def create_surgical_vqa_trainer(
    base_model: str = "allenai/SAGE-MM-Molmo2-8B-SFT_RL",
    dataset_id: str = "matheus-rech/pitvqa-sage-sft",
    output_name: str = "matheus-rech/pitvqa-sage-surgical",
    method: str = "sft",
    num_epochs: int = 3,
    push_to_hub: bool = True
) -> HFSkillsTrainer:
    """
    Convenience function to create a trainer for surgical VQA.

    Args:
        base_model: HuggingFace model ID for SAGE/Molmo
        dataset_id: HuggingFace dataset with training data
        output_name: Name for the fine-tuned model
        method: Training method (sft, dpo, grpo)
        num_epochs: Number of training epochs
        push_to_hub: Whether to push to HuggingFace Hub

    Returns:
        Configured HFSkillsTrainer
    """
    config = TrainingConfig(
        base_model=base_model,
        output_model_name=output_name,
        dataset_id=dataset_id,
        method=TrainingMethod(method),
        num_epochs=num_epochs,
        push_to_hub=push_to_hub,
        hub_token=os.environ.get("HF_TOKEN"),
        is_vision_model=True
    )

    reward_config = GRPORewardConfig() if method == "grpo" else None

    return HFSkillsTrainer(config, reward_config)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HF Skills training configuration")
    parser.add_argument("--base-model", default="allenai/SAGE-MM-Molmo2-8B-SFT_RL")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset ID")
    parser.add_argument("--output-name", required=True, help="Output model name")
    parser.add_argument("--method", choices=["sft", "dpo", "grpo"], default="sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--generate-script", help="Path to save training script")
    parser.add_argument("--generate-prompt", action="store_true", help="Print HF Skills prompt")

    args = parser.parse_args()

    trainer = create_surgical_vqa_trainer(
        base_model=args.base_model,
        dataset_id=args.dataset,
        output_name=args.output_name,
        method=args.method,
        num_epochs=args.epochs
    )

    if args.generate_script:
        trainer.generate_training_script(args.generate_script)
        print(f"Training script saved to: {args.generate_script}")

    if args.generate_prompt:
        print("\n" + "="*60)
        print("HuggingFace Skills Prompt:")
        print("="*60)
        print(trainer.generate_hf_skills_prompt())
