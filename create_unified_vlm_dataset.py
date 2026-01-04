#!/usr/bin/env python3
"""
Create Unified VLM Dataset for Multi-Task Training

Combines:
1. Phase Classification (from pitvqa-sage-sft)
2. Step Classification (from pitvqa-sage-sft)
3. Instrument Pointing (from ground truth + pitvqa-spatial-vlm)
4. Anatomy Pointing (from Gemini annotations with confidence filtering)

Output: HuggingFace dataset mmrech/pitvqa-unified-vlm
"""

import json
import random
from collections import defaultdict
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, Image
from pathlib import Path

# Configuration
OUTPUT_DATASET = "mmrech/pitvqa-unified-vlm"
CONFIDENCE_THRESHOLD = 0.8  # For Gemini anatomy annotations

# Task type identifiers
TASK_PHASE = "phase_classification"
TASK_STEP = "step_classification"
TASK_INSTRUMENT = "instrument_pointing"
TASK_ANATOMY = "anatomy_pointing"


def load_sage_dataset():
    """Load classification data from pitvqa-sage-sft."""
    print("Loading pitvqa-sage-sft...")
    ds = load_dataset("mmrech/pitvqa-sage-sft")
    return ds


def load_spatial_dataset():
    """Load spatial pointing data from pitvqa-spatial-vlm."""
    print("Loading pitvqa-spatial-vlm...")
    ds = load_dataset("mmrech/pitvqa-spatial-vlm")
    return ds


def extract_phase_step_samples(sage_ds, split="train"):
    """Extract phase and step classification samples."""
    phase_samples = []
    step_samples = []

    for sample in sage_ds[split]:
        qa_type = sample.get('qa_type', '')
        messages = sample.get('messages', [])
        image = sample.get('image')

        if not messages or not image:
            continue

        # Handle both list and dict message formats
        if isinstance(messages, list):
            user_content = messages[0].get('content', '')
            assistant_content = messages[1].get('content', '')
        else:
            continue

        sample_data = {
            'image': image,
            'messages': json.dumps([
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ]),
            'video_id': sample.get('video_id', ''),
            'frame_id': sample.get('frame_id', ''),
            'task_type': '',
        }

        if qa_type == 'phase':
            sample_data['task_type'] = TASK_PHASE
            phase_samples.append(sample_data)
        elif qa_type == 'step':
            sample_data['task_type'] = TASK_STEP
            step_samples.append(sample_data)

    return phase_samples, step_samples


def extract_spatial_samples(spatial_ds, split="train"):
    """Extract instrument and anatomy pointing samples."""
    instrument_samples = []
    anatomy_samples = []

    for sample in spatial_ds[split]:
        ann_type = sample.get('annotation_type', '')
        source = sample.get('source', '')
        messages = sample.get('messages', '')
        image = sample.get('image')

        if not messages or not image:
            continue

        sample_data = {
            'image': image,
            'messages': messages if isinstance(messages, str) else json.dumps(messages),
            'video_id': sample.get('video_id', ''),
            'frame_id': sample.get('frame_id', ''),
            'task_type': '',
        }

        if 'instrument' in ann_type.lower():
            sample_data['task_type'] = TASK_INSTRUMENT
            instrument_samples.append(sample_data)
        elif 'anatomy' in ann_type.lower():
            sample_data['task_type'] = TASK_ANATOMY
            anatomy_samples.append(sample_data)

    return instrument_samples, anatomy_samples


def balance_dataset(samples_by_task, target_ratio=None):
    """Balance samples across task types.

    Args:
        samples_by_task: Dict of task_type -> list of samples
        target_ratio: Optional dict of task_type -> ratio (sums to 1.0)
    """
    if target_ratio is None:
        # Default: roughly equal distribution
        target_ratio = {
            TASK_PHASE: 0.25,
            TASK_STEP: 0.25,
            TASK_INSTRUMENT: 0.35,
            TASK_ANATOMY: 0.15,  # Less due to lower confidence
        }

    # Find the constraining factor (smallest available / target ratio)
    available = {task: len(samples) for task, samples in samples_by_task.items()}
    print(f"\nAvailable samples: {available}")

    # Calculate how many we can take from each
    total_possible = []
    for task, ratio in target_ratio.items():
        if task in available and ratio > 0:
            total_possible.append(available[task] / ratio)

    max_total = min(total_possible) if total_possible else 0
    print(f"Max balanced total: {int(max_total)}")

    # Sample from each task
    balanced = {}
    for task, ratio in target_ratio.items():
        if task not in samples_by_task:
            continue
        n_samples = int(max_total * ratio)
        samples = samples_by_task[task]
        if len(samples) > n_samples:
            balanced[task] = random.sample(samples, n_samples)
        else:
            balanced[task] = samples

    return balanced


def create_unified_dataset():
    """Create the unified multi-task dataset."""
    print("=" * 70)
    print("CREATING UNIFIED VLM DATASET")
    print("=" * 70)

    # Load source datasets
    sage_ds = load_sage_dataset()
    spatial_ds = load_spatial_dataset()

    # Extract samples by task type
    print("\nExtracting samples...")

    # From SAGE (classification)
    train_phase, train_step = extract_phase_step_samples(sage_ds, "train")
    val_phase, val_step = extract_phase_step_samples(sage_ds, "validation")

    # From Spatial (pointing)
    train_inst, train_anat = extract_spatial_samples(spatial_ds, "train")
    val_inst, val_anat = extract_spatial_samples(spatial_ds, "validation")

    print(f"\n📊 Raw sample counts:")
    print(f"  Phase (train/val): {len(train_phase)}/{len(val_phase)}")
    print(f"  Step (train/val): {len(train_step)}/{len(val_step)}")
    print(f"  Instrument (train/val): {len(train_inst)}/{len(val_inst)}")
    print(f"  Anatomy (train/val): {len(train_anat)}/{len(val_anat)}")

    # Combine for balancing
    train_by_task = {
        TASK_PHASE: train_phase,
        TASK_STEP: train_step,
        TASK_INSTRUMENT: train_inst,
        TASK_ANATOMY: train_anat,
    }

    val_by_task = {
        TASK_PHASE: val_phase,
        TASK_STEP: val_step,
        TASK_INSTRUMENT: val_inst,
        TASK_ANATOMY: val_anat,
    }

    # Balance datasets
    print("\n📊 Balancing datasets...")
    balanced_train = balance_dataset(train_by_task)
    balanced_val = balance_dataset(val_by_task)

    # Flatten and shuffle
    train_samples = []
    for task, samples in balanced_train.items():
        train_samples.extend(samples)
        print(f"  Train {task}: {len(samples)}")

    val_samples = []
    for task, samples in balanced_val.items():
        val_samples.extend(samples)
        print(f"  Val {task}: {len(samples)}")

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\n📊 Final dataset size:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Validation: {len(val_samples)}")

    # Task distribution
    print("\n📊 Task distribution (train):")
    task_counts = defaultdict(int)
    for s in train_samples:
        task_counts[s['task_type']] += 1
    for task, count in sorted(task_counts.items()):
        pct = count / len(train_samples) * 100
        print(f"  {task}: {count} ({pct:.1f}%)")

    # Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace format...")

    def samples_to_dict(samples):
        return {
            'image': [s['image'] for s in samples],
            'messages': [s['messages'] for s in samples],
            'video_id': [s['video_id'] for s in samples],
            'frame_id': [s['frame_id'] for s in samples],
            'task_type': [s['task_type'] for s in samples],
        }

    train_ds = Dataset.from_dict(samples_to_dict(train_samples))
    val_ds = Dataset.from_dict(samples_to_dict(val_samples))

    # Cast image column
    train_ds = train_ds.cast_column('image', Image())
    val_ds = val_ds.cast_column('image', Image())

    # Create DatasetDict
    dataset = DatasetDict({
        'train': train_ds,
        'validation': val_ds
    })

    print(f"\n📊 Dataset info:")
    print(dataset)

    # Push to hub
    print(f"\n🚀 Pushing to HuggingFace Hub: {OUTPUT_DATASET}")
    dataset.push_to_hub(OUTPUT_DATASET)

    print("\n" + "=" * 70)
    print("✅ UNIFIED DATASET CREATED")
    print("=" * 70)
    print(f"Dataset: https://huggingface.co/datasets/{OUTPUT_DATASET}")

    # Show sample from each task type
    print("\n📋 Sample from each task type:")
    for task_type in [TASK_PHASE, TASK_STEP, TASK_INSTRUMENT, TASK_ANATOMY]:
        for sample in train_samples:
            if sample['task_type'] == task_type:
                msgs = json.loads(sample['messages'])
                print(f"\n[{task_type}]")
                print(f"  User: {msgs[0]['content'][:60]}...")
                print(f"  Assistant: {msgs[1]['content'][:60]}...")
                break

    return dataset


if __name__ == "__main__":
    create_unified_dataset()
