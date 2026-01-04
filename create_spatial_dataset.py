#!/usr/bin/env python3
"""
Create Spatial Training Dataset for PitVQA

Converts ground truth CSVs to SAGE/Molmo-compatible format with coordinates.
Output: HuggingFace dataset ready for coordinate-aware training.

Usage:
    python create_spatial_dataset.py

    # Then push manually:
    huggingface-cli upload mmrech/pitvqa-spatial-sft ./spatial_dataset
"""

import json
import csv
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Configuration
GT_DIR = Path("ground_truth")
OUTPUT_DIR = Path("spatial_dataset")
GEMINI_ANNOTATIONS = Path("gemini_annotations/surgical_videopoint_molmo_format.json")

# Quadrant to coordinate mapping (PitVQA uses quadrants 1-5)
QUADRANT_CENTERS = {
    "1": (25.0, 25.0),   # Upper-left
    "2": (75.0, 25.0),   # Upper-right
    "3": (25.0, 75.0),   # Lower-left
    "4": (75.0, 75.0),   # Lower-right
    "5": (50.0, 50.0),   # Center
}

# Instrument name normalization
INSTRUMENT_NAMES = {
    "suction": "suction device",
    "freer_elevator": "Freer elevator",
    "ring_curette": "ring curette",
    "kerrisons": "Kerrison rongeur",
    "pituitary_rongeurs": "pituitary rongeurs",
    "bipolar": "bipolar forceps",
    "drill": "surgical drill",
    "cottonoid": "cottonoid",
    "haemostatic_foam": "hemostatic foam",
    "spatula_dissector": "spatula dissector",
    "cup_forceps": "cup forceps",
    "irrigation_syringe": "irrigation syringe",
    "scissors": "surgical scissors",
    "needle_holder": "needle holder",
}

# Question templates for diversity
POINT_QUESTIONS = [
    "Point to the {instrument} in this surgical frame.",
    "Locate the {instrument} in this image.",
    "Where is the {instrument} positioned?",
    "Identify the location of the {instrument}.",
    "Show me where the {instrument} is.",
]

DESCRIBE_QUESTIONS = [
    "What instrument is being used and where is it located?",
    "Describe the surgical instruments visible in this frame with their positions.",
    "Identify the primary instrument and its location in the surgical field.",
]


def quadrant_to_coordinates(quadrant: str, jitter: bool = True) -> Tuple[float, float]:
    """Convert PitVQA quadrant (1-5) to normalized coordinates (0-100)."""
    if quadrant not in QUADRANT_CENTERS:
        return (50.0, 50.0)

    x, y = QUADRANT_CENTERS[quadrant]

    if jitter:
        # Add small random offset for diversity (±10%)
        x += random.uniform(-10, 10)
        y += random.uniform(-10, 10)
        x = max(5, min(95, x))
        y = max(5, min(95, y))

    return (round(x, 1), round(y, 1))


def normalize_instrument_name(name: str) -> str:
    """Convert GT instrument name to natural language."""
    return INSTRUMENT_NAMES.get(name, name.replace("_", " "))


def create_pointing_sample(
    video_id: int,
    frame_time: int,
    instrument: str,
    quadrant: str
) -> Dict:
    """Create a single pointing training sample."""
    x, y = quadrant_to_coordinates(quadrant)
    instrument_name = normalize_instrument_name(instrument)

    # Random question template
    question = random.choice(POINT_QUESTIONS).format(instrument=instrument_name)

    # Structured answer with coordinates
    answer = f"<point x='{x}' y='{y}'>{instrument_name}</point>"

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer}
        ],
        "image": f"video_{video_id:02d}/frame_{frame_time:06d}.jpg",
        "metadata": {
            "video_id": video_id,
            "frame_time": frame_time,
            "instrument": instrument,
            "quadrant": quadrant,
            "coordinates": {"x": x, "y": y}
        }
    }


def create_description_sample(
    video_id: int,
    frame_time: int,
    instrument1: str,
    instrument2: str,
    quad1: str,
    quad2: str,
    step: str = None
) -> Dict:
    """Create a descriptive sample with multiple instruments."""
    x1, y1 = quadrant_to_coordinates(quad1)
    inst1_name = normalize_instrument_name(instrument1)

    question = random.choice(DESCRIBE_QUESTIONS)

    answer_parts = []
    if instrument1 and instrument1 not in ["no_visible_instrument", "out_of_patient"]:
        answer_parts.append(f"Primary instrument: <point x='{x1}' y='{y1}'>{inst1_name}</point>")

    if instrument2 and instrument2 not in ["no_secondary_instrument", "-2"]:
        x2, y2 = quadrant_to_coordinates(quad2)
        inst2_name = normalize_instrument_name(instrument2)
        answer_parts.append(f"Secondary: <point x='{x2}' y='{y2}'>{inst2_name}</point>")

    if step:
        answer_parts.append(f"Current step: {step.replace('_', ' ')}")

    if not answer_parts:
        return None  # Skip frames with no visible instruments

    answer = "\n".join(answer_parts)

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer}
        ],
        "image": f"video_{video_id:02d}/frame_{frame_time:06d}.jpg",
        "metadata": {
            "video_id": video_id,
            "frame_time": frame_time,
            "instruments": [instrument1, instrument2],
            "step": step
        }
    }


def load_ground_truth() -> Tuple[Dict, Dict]:
    """Load all ground truth CSV files."""
    instruments = {}
    steps = {}

    for i in range(1, 26):
        inst_file = GT_DIR / f"instruments_{i:02d}.csv"
        step_file = GT_DIR / f"steps_{i:02d}.csv"

        if inst_file.exists():
            with open(inst_file, 'r') as f:
                reader = csv.DictReader(f)
                instruments[i] = list(reader)

        if step_file.exists():
            with open(step_file, 'r') as f:
                reader = csv.DictReader(f)
                steps[i] = list(reader)

    return instruments, steps


def get_step_at_time(steps_data: List[Dict], frame_time: int) -> Optional[str]:
    """Get the surgical step at a given frame time."""
    current_step = None
    for row in steps_data:
        step_time = int(row.get('int_time', 0))
        if step_time <= frame_time:
            current_step = row.get('str_step')
        else:
            break
    return current_step


def create_dataset():
    """Create the full spatial training dataset."""
    print("Loading ground truth...")
    instruments, steps = load_ground_truth()

    samples = []
    stats = {
        "pointing_samples": 0,
        "description_samples": 0,
        "skipped_no_instrument": 0,
        "videos_processed": 0
    }

    print(f"Processing {len(instruments)} videos...")

    for video_id, inst_data in instruments.items():
        step_data = steps.get(video_id, [])

        # Sample frames (every 10th frame to avoid redundancy)
        frame_times = sorted(set(int(row.get('int_time', 0)) for row in inst_data))
        sampled_times = frame_times[::10]  # Every 10th frame

        for frame_time in sampled_times:
            # Find instrument data for this frame
            frame_rows = [r for r in inst_data if int(r.get('int_time', 0)) == frame_time]
            if not frame_rows:
                continue

            row = frame_rows[0]
            inst1 = row.get('str_instrument1', '')
            inst2 = row.get('str_instrument2', '')
            quad1 = row.get('pos_instrument1', '')
            quad2 = row.get('pos_instrument2', '')

            # Skip if no visible instrument
            if inst1 in ["no_visible_instrument", "out_of_patient", ""]:
                stats["skipped_no_instrument"] += 1
                continue

            # Get current surgical step
            current_step = get_step_at_time(step_data, frame_time)

            # Create pointing sample (50% chance)
            if random.random() < 0.5 and quad1:
                sample = create_pointing_sample(video_id, frame_time, inst1, quad1)
                samples.append(sample)
                stats["pointing_samples"] += 1

            # Create description sample (50% chance)
            if random.random() < 0.5:
                sample = create_description_sample(
                    video_id, frame_time, inst1, inst2, quad1, quad2, current_step
                )
                if sample:
                    samples.append(sample)
                    stats["description_samples"] += 1

        stats["videos_processed"] += 1
        print(f"  Video {video_id:02d}: {len([s for s in samples if s['metadata']['video_id'] == video_id])} samples")

    return samples, stats


def add_gemini_annotations(samples: List[Dict]) -> List[Dict]:
    """Add high-quality Gemini annotations to the dataset."""
    if not GEMINI_ANNOTATIONS.exists():
        print("Gemini annotations not found, skipping...")
        return samples

    print("Loading Gemini annotations...")
    with open(GEMINI_ANNOTATIONS, 'r') as f:
        gemini_data = json.load(f)

    # Filter to high-confidence instrument annotations
    gemini_samples = 0
    for ann in gemini_data:
        if ann.get('category') != 'instruments':
            continue
        if ann.get('confidence', 0) < 0.85:
            continue

        label = ann.get('label', '')
        points = ann.get('points', [[]])

        if not points or not points[0]:
            continue

        pt = points[0][0]
        x, y = pt.get('x', 50), pt.get('y', 50)

        question = f"Point to the {label} in this surgical frame."
        answer = f"<point x='{x}' y='{y}'>{label}</point>"

        sample = {
            "messages": [
                {"role": "user", "content": f"<image>\n{question}"},
                {"role": "assistant", "content": answer}
            ],
            "image": f"{ann.get('video_id', 'unknown')}/frame.jpg",
            "metadata": {
                "source": "gemini",
                "confidence": ann.get('confidence', 0),
                "label": label
            }
        }
        samples.append(sample)
        gemini_samples += 1

    print(f"Added {gemini_samples} Gemini annotation samples")
    return samples


def save_dataset(samples: List[Dict], output_dir: Path):
    """Save dataset in HuggingFace-compatible format."""
    output_dir.mkdir(exist_ok=True)

    # Split into train/val
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    # Save as JSONL
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "validation.jsonl"

    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nDataset saved to {output_dir}/")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")

    # Save dataset card
    readme = f"""# PitVQA Spatial Training Dataset

Coordinate-aware training dataset for surgical instrument localization.

## Dataset Structure

- **train.jsonl**: {len(train_samples)} training samples
- **validation.jsonl**: {len(val_samples)} validation samples

## Format

Each sample contains:
- `messages`: Chat format with `<point x='...' y='...'>label</point>` coordinates
- `image`: Path to surgical frame
- `metadata`: Source information (ground_truth or gemini)

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mmrech/pitvqa-spatial-sft")
```

## Training

Fine-tune Qwen2-VL or similar VLM for coordinate-aware outputs:

```
Fine-tune mmrech/pitvqa-qwen2vl-surgical on this dataset
for coordinate-aware surgical instrument localization
```
"""

    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="spatial_dataset", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("Creating Spatial Training Dataset")
    print("=" * 60)

    samples, stats = create_dataset()
    samples = add_gemini_annotations(samples)

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Pointing samples: {stats['pointing_samples']}")
    print(f"   Description samples: {stats['description_samples']}")
    print(f"   Skipped (no instrument): {stats['skipped_no_instrument']}")
    print(f"   Videos processed: {stats['videos_processed']}")

    save_dataset(samples, Path(args.output))

    print("\n✅ Done! To push to HuggingFace Hub, run:")
    print(f"   huggingface-cli upload mmrech/pitvqa-spatial-sft ./{args.output}")


if __name__ == "__main__":
    main()
