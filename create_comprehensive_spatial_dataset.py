#!/usr/bin/env python3
"""
Create Comprehensive Spatial Training Dataset for PitVQA

Combines:
1. Gemini annotations (instruments + anatomy with precise coordinates)
2. Ground truth data (temporal step information)
3. Multiple question types for diverse training

Output: HuggingFace dataset with instruments, anatomy, and temporal grounding.
"""

import json
import csv
import random
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import defaultdict

# Configuration
GEMINI_ANNOTATIONS = Path("gemini_annotations/surgical_videopoint_molmo_format.json")
GPT_ANNOTATIONS = Path("gpt_annotations/surgical_videopoint_molmo_format.json")
GT_DIR = Path("ground_truth")
OUTPUT_DIR = Path("comprehensive_spatial_dataset")

# Anatomy normalization (merge variants)
ANATOMY_NORMALIZE = {
    "mucosal_tissue": "mucosal tissue",
    "nasal_septum": "nasal septum",
    "nasal_cavity": "nasal cavity",
    "sphenoid_sinus": "sphenoid sinus",
    "tumor_tissue": "tumor tissue",
    "tumour_tissue": "tumor tissue",
    "blood_vessels": "blood vessels",
}

# Question templates by category
INSTRUMENT_POINT_QUESTIONS = [
    "Point to the {label} in this surgical frame.",
    "Locate the {label} in this image.",
    "Where is the {label} positioned?",
    "Identify the location of the {label}.",
    "Show me where the {label} is.",
]

ANATOMY_POINT_QUESTIONS = [
    "Point to the {label} in this surgical field.",
    "Where is the {label} visible?",
    "Locate the {label} anatomical structure.",
    "Identify where the {label} is in this frame.",
    "Show the position of the {label}.",
]

DESCRIBE_INSTRUMENT_QUESTIONS = [
    "What surgical instrument is visible and where is it located?",
    "Describe the instrument being used and its position.",
    "Identify the primary surgical tool in this frame with its location.",
]

DESCRIBE_ANATOMY_QUESTIONS = [
    "What anatomical structures are visible in this surgical view?",
    "Describe the visible anatomy and their positions.",
    "Identify the anatomical landmarks in this frame.",
]

COMBINED_QUESTIONS = [
    "Describe both the instruments and anatomy visible in this surgical frame.",
    "What is happening in this surgical view? Include instrument and anatomy locations.",
    "Provide a spatial description of this surgical field.",
]


def normalize_label(label: str, category: str) -> str:
    """Normalize labels for consistency."""
    label = label.lower().strip()
    if category == "anatomy":
        return ANATOMY_NORMALIZE.get(label, label)
    return label


def create_point_sample(
    video_id: str,
    timestamp: float,
    label: str,
    x: float,
    y: float,
    category: str,
    confidence: float
) -> Dict:
    """Create a pointing sample for instrument or anatomy."""

    if category == "instruments":
        question = random.choice(INSTRUMENT_POINT_QUESTIONS).format(label=label)
    else:
        question = random.choice(ANATOMY_POINT_QUESTIONS).format(label=label)

    answer = f"<point x='{x:.1f}' y='{y:.1f}'>{label}</point>"

    # Frame path based on timestamp (2 fps)
    frame_num = int(timestamp * 2)  # Convert to frame number at 2fps

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": answer}
        ],
        "video_id": video_id,
        "timestamp": timestamp,
        "frame_index": frame_num,
        "metadata": {
            "category": category,
            "label": label,
            "coordinates": {"x": x, "y": y},
            "confidence": confidence,
            "source": "gemini"
        }
    }


def create_multi_point_sample(
    video_id: str,
    timestamp: float,
    annotations: List[Dict]
) -> Optional[Dict]:
    """Create a sample with multiple points (instruments + anatomy)."""

    instruments = [a for a in annotations if a["category"] == "instruments"]
    anatomy = [a for a in annotations if a["category"] == "anatomy"]

    if not instruments and not anatomy:
        return None

    question = random.choice(COMBINED_QUESTIONS)

    answer_parts = []

    # Add instruments
    if instruments:
        answer_parts.append("**Instruments:**")
        for inst in instruments[:2]:  # Max 2 instruments
            x, y = inst["x"], inst["y"]
            label = inst["label"]
            answer_parts.append(f"- <point x='{x:.1f}' y='{y:.1f}'>{label}</point>")

    # Add anatomy
    if anatomy:
        answer_parts.append("**Anatomy:**")
        for anat in anatomy[:3]:  # Max 3 anatomy structures
            x, y = anat["x"], anat["y"]
            label = anat["label"]
            answer_parts.append(f"- <point x='{x:.1f}' y='{y:.1f}'>{label}</point>")

    frame_num = int(timestamp * 2)

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{question}"},
            {"role": "assistant", "content": "\n".join(answer_parts)}
        ],
        "video_id": video_id,
        "timestamp": timestamp,
        "frame_index": frame_num,
        "metadata": {
            "type": "multi_point",
            "num_instruments": len(instruments),
            "num_anatomy": len(anatomy),
            "source": "gemini"
        }
    }


def create_temporal_tracking_sample(
    video_id: str,
    label: str,
    timestamps: List[float],
    points: List[List[Dict]],
    category: str
) -> Optional[Dict]:
    """Create temporal tracking sample showing movement over time."""

    if len(timestamps) < 2:
        return None

    question = f"Track the {label} across multiple frames in this video sequence."

    answer_parts = [f"Tracking **{label}** over time:"]
    for i, (ts, pts) in enumerate(zip(timestamps, points)):
        if pts and len(pts) > 0:
            pt = pts[0]
            x, y = pt.get("x", 50), pt.get("y", 50)
            answer_parts.append(f"- Frame {i+1} (t={ts:.1f}s): <point x='{x:.1f}' y='{y:.1f}'>{label}</point>")

    return {
        "messages": [
            {"role": "user", "content": f"<video>\n{question}"},
            {"role": "assistant", "content": "\n".join(answer_parts)}
        ],
        "video_id": video_id,
        "timestamps": timestamps,
        "metadata": {
            "type": "temporal_tracking",
            "category": category,
            "label": label,
            "num_frames": len(timestamps),
            "source": "gemini"
        }
    }


def load_gemini_annotations() -> List[Dict]:
    """Load and process Gemini annotations."""
    if not GEMINI_ANNOTATIONS.exists():
        print("Gemini annotations not found!")
        return []

    with open(GEMINI_ANNOTATIONS, 'r') as f:
        data = json.load(f)

    return data


def load_ground_truth_steps() -> Dict[int, List[Dict]]:
    """Load surgical step information from ground truth."""
    steps = {}
    for i in range(1, 26):
        step_file = GT_DIR / f"steps_{i:02d}.csv"
        if step_file.exists():
            with open(step_file, 'r') as f:
                reader = csv.DictReader(f)
                steps[i] = list(reader)
    return steps


def get_step_at_time(steps_data: List[Dict], frame_time: float) -> Optional[str]:
    """Get the surgical step at a given time."""
    current_step = None
    for row in steps_data:
        step_time = int(row.get('int_time', 0))
        if step_time <= frame_time:
            current_step = row.get('str_step')
        else:
            break
    return current_step


def load_ground_truth_instruments() -> Dict[int, List[Dict]]:
    """Load instrument data from ground truth CSVs."""
    instruments = {}
    for i in range(1, 26):
        inst_file = GT_DIR / f"instruments_{i:02d}.csv"
        if inst_file.exists():
            with open(inst_file, 'r') as f:
                reader = csv.DictReader(f)
                instruments[i] = list(reader)
    return instruments


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


def quadrant_to_coordinates(quadrant: str, jitter: bool = True) -> Tuple[float, float]:
    """Convert PitVQA quadrant (1-5) to normalized coordinates (0-100)."""
    if quadrant not in QUADRANT_CENTERS:
        return (50.0, 50.0)

    x, y = QUADRANT_CENTERS[quadrant]

    if jitter:
        x += random.uniform(-10, 10)
        y += random.uniform(-10, 10)
        x = max(5, min(95, x))
        y = max(5, min(95, y))

    return (round(x, 1), round(y, 1))


def normalize_instrument_name(name: str) -> str:
    """Convert GT instrument name to natural language."""
    return INSTRUMENT_NAMES.get(name, name.replace("_", " "))


def create_gt_instrument_samples(instruments: Dict[int, List[Dict]]) -> Tuple[List[Dict], Dict]:
    """Create samples from ground truth instrument data."""
    samples = []
    stats = {"gt_instrument_pointing": 0, "gt_description": 0}

    for video_id, inst_data in instruments.items():
        # Sample every 10th frame to avoid redundancy
        frame_times = sorted(set(int(row.get('int_time', 0)) for row in inst_data))
        sampled_times = frame_times[::10]

        for frame_time in sampled_times:
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
                continue

            # Create pointing sample for primary instrument
            if quad1:
                x, y = quadrant_to_coordinates(quad1)
                inst_name = normalize_instrument_name(inst1)
                question = random.choice(INSTRUMENT_POINT_QUESTIONS).format(label=inst_name)
                answer = f"<point x='{x}' y='{y}'>{inst_name}</point>"

                sample = {
                    "messages": [
                        {"role": "user", "content": f"<image>\n{question}"},
                        {"role": "assistant", "content": answer}
                    ],
                    "video_id": f"video_{video_id:02d}",
                    "timestamp": frame_time / 2.0,  # Convert to seconds at 2fps
                    "frame_index": frame_time,
                    "metadata": {
                        "category": "instruments",
                        "label": inst_name,
                        "coordinates": {"x": x, "y": y},
                        "confidence": 1.0,
                        "source": "ground_truth",
                        "quadrant": quad1
                    }
                }
                samples.append(sample)
                stats["gt_instrument_pointing"] += 1

            # Create description sample with both instruments
            if inst2 and inst2 not in ["no_secondary_instrument", "-2", ""] and quad2:
                x1, y1 = quadrant_to_coordinates(quad1)
                x2, y2 = quadrant_to_coordinates(quad2)
                inst1_name = normalize_instrument_name(inst1)
                inst2_name = normalize_instrument_name(inst2)

                question = random.choice(DESCRIBE_INSTRUMENT_QUESTIONS)
                answer = f"**Primary instrument:** <point x='{x1}' y='{y1}'>{inst1_name}</point>\n**Secondary:** <point x='{x2}' y='{y2}'>{inst2_name}</point>"

                sample = {
                    "messages": [
                        {"role": "user", "content": f"<image>\n{question}"},
                        {"role": "assistant", "content": answer}
                    ],
                    "video_id": f"video_{video_id:02d}",
                    "timestamp": frame_time / 2.0,
                    "frame_index": frame_time,
                    "metadata": {
                        "type": "dual_instrument",
                        "instruments": [inst1_name, inst2_name],
                        "source": "ground_truth"
                    }
                }
                samples.append(sample)
                stats["gt_description"] += 1

    return samples, stats


def create_temporal_tracking_from_groups(grouped: Dict, min_frames: int = 3) -> Tuple[List[Dict], int]:
    """Create temporal tracking samples by grouping same objects across timestamps."""
    samples = []
    count = 0

    for video_id, ts_data in grouped.items():
        # Group by label across timestamps
        label_timeline = defaultdict(list)
        for ts, anns in sorted(ts_data.items()):
            for ann in anns:
                label_timeline[ann["label"]].append({
                    "timestamp": ts,
                    "x": ann["x"],
                    "y": ann["y"],
                    "category": ann["category"]
                })

        # Create tracking samples for labels with enough frames
        for label, timeline in label_timeline.items():
            if len(timeline) >= min_frames:
                # Sample up to 5 frames for the tracking sequence
                if len(timeline) > 5:
                    indices = sorted(random.sample(range(len(timeline)), 5))
                    timeline = [timeline[i] for i in indices]

                question = f"Track the {label} across multiple frames in this surgical video."
                answer_parts = [f"Tracking **{label}** movement:"]

                for i, frame in enumerate(timeline):
                    x, y = frame["x"], frame["y"]
                    ts = frame["timestamp"]
                    answer_parts.append(f"- t={ts:.1f}s: <point x='{x:.1f}' y='{y:.1f}'>{label}</point>")

                sample = {
                    "messages": [
                        {"role": "user", "content": f"<video>\n{question}"},
                        {"role": "assistant", "content": "\n".join(answer_parts)}
                    ],
                    "video_id": video_id,
                    "timestamps": [f["timestamp"] for f in timeline],
                    "metadata": {
                        "type": "temporal_tracking",
                        "category": timeline[0]["category"],
                        "label": label,
                        "num_frames": len(timeline),
                        "source": "gemini"
                    }
                }
                samples.append(sample)
                count += 1

    return samples, count


def create_dataset():
    """Create the comprehensive spatial training dataset."""
    print("Loading Gemini annotations...")
    annotations = load_gemini_annotations()

    print("Loading ground truth instruments...")
    gt_instruments = load_ground_truth_instruments()

    print("Loading ground truth steps...")
    steps = load_ground_truth_steps()

    samples = []
    stats = {
        "instrument_pointing": 0,
        "anatomy_pointing": 0,
        "multi_point": 0,
        "temporal_tracking": 0,
        "gt_instrument_pointing": 0,
        "gt_description": 0,
        "total_instruments": 0,
        "total_anatomy": 0,
    }

    # Group annotations by video and timestamp for multi-point samples
    grouped = defaultdict(lambda: defaultdict(list))

    print(f"Processing {len(annotations)} Gemini annotations...")

    for ann in annotations:
        video_id = ann.get("video_id", "unknown")
        label = ann.get("label", "")
        category = ann.get("category", "")
        confidence = ann.get("confidence", 0.8)
        timestamps = ann.get("two_fps_timestamps", [])
        points = ann.get("points", [[]])

        # Skip low confidence
        if confidence < 0.7:
            continue

        # Normalize label
        label = normalize_label(label, category)

        # Process each timestamp
        for i, ts in enumerate(timestamps):
            if i < len(points) and points[i]:
                pt = points[i][0] if isinstance(points[i], list) else points[i]
                x = pt.get("x", 50)
                y = pt.get("y", 50)

                # Create individual pointing sample
                sample = create_point_sample(
                    video_id, ts, label, x, y, category, confidence
                )
                samples.append(sample)

                if category == "instruments":
                    stats["instrument_pointing"] += 1
                    stats["total_instruments"] += 1
                else:
                    stats["anatomy_pointing"] += 1
                    stats["total_anatomy"] += 1

                # Group for multi-point and temporal samples
                grouped[video_id][ts].append({
                    "label": label,
                    "category": category,
                    "x": x,
                    "y": y,
                    "confidence": confidence
                })

    # Create multi-point samples (combining instruments + anatomy at same timestamp)
    print("Creating multi-point samples...")
    for video_id, ts_data in grouped.items():
        for ts, anns in ts_data.items():
            if len(anns) >= 2:
                multi_sample = create_multi_point_sample(video_id, ts, anns)
                if multi_sample:
                    samples.append(multi_sample)
                    stats["multi_point"] += 1

    # Create temporal tracking samples
    print("Creating temporal tracking samples...")
    tracking_samples, tracking_count = create_temporal_tracking_from_groups(grouped)
    samples.extend(tracking_samples)
    stats["temporal_tracking"] = tracking_count

    # Add ground truth instrument samples
    print(f"Processing ground truth from {len(gt_instruments)} videos...")
    gt_samples, gt_stats = create_gt_instrument_samples(gt_instruments)
    samples.extend(gt_samples)
    stats["gt_instrument_pointing"] = gt_stats["gt_instrument_pointing"]
    stats["gt_description"] = gt_stats["gt_description"]

    return samples, stats


def save_dataset(samples: List[Dict], output_dir: Path):
    """Save dataset in HuggingFace-compatible format."""
    output_dir.mkdir(exist_ok=True)

    # Shuffle and split
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

    # Create dataset card
    readme = f"""# PitVQA Comprehensive Spatial Training Dataset

Coordinate-aware training dataset for surgical instrument AND anatomical structure localization.

## Dataset Statistics

- **Total samples:** {len(samples)}
- **Training:** {len(train_samples)}
- **Validation:** {len(val_samples)}

## Categories

### Instruments
- Suction, ring curette, pituitary rongeurs, haemostatic foam, etc.
- Precise x,y coordinates (0-100 scale)

### Anatomy
- Mucosal tissue, sphenoid sinus, tumor tissue, nasal cavity, sella, etc.
- Spatial grounding for surgical navigation

### Sample Types
1. **Single-point**: Point to specific instrument or anatomy
2. **Multi-point**: Combined instrument + anatomy descriptions
3. **Temporal tracking**: Track objects across video frames

## Format

```json
{{
  "messages": [
    {{"role": "user", "content": "<image>\\nPoint to the suction..."}},
    {{"role": "assistant", "content": "<point x='48.5' y='80.2'>suction</point>"}}
  ],
  "video_id": "video_01",
  "timestamp": 0.5,
  "metadata": {{
    "category": "instruments",
    "label": "suction",
    "coordinates": {{"x": 48.5, "y": 80.2}},
    "confidence": 0.95
  }}
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("mmrech/pitvqa-comprehensive-spatial")
```

## Training Target

Fine-tune Qwen2-VL or similar VLM for:
- Surgical instrument localization
- Anatomical structure identification
- Temporal object tracking in surgical video
"""

    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)


def main():
    print("=" * 60)
    print("Creating Comprehensive Spatial Training Dataset")
    print("=" * 60)

    samples, stats = create_dataset()

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(samples)}")
    print(f"   Instrument pointing: {stats['instrument_pointing']}")
    print(f"   Anatomy pointing: {stats['anatomy_pointing']}")
    print(f"   Multi-point samples: {stats['multi_point']}")
    print(f"   Temporal tracking: {stats['temporal_tracking']}")
    print(f"   Total instruments: {stats['total_instruments']}")
    print(f"   Total anatomy: {stats['total_anatomy']}")

    save_dataset(samples, OUTPUT_DIR)

    print("\n✅ Done!")
    print(f"\nTo push to HuggingFace Hub:")
    print(f"   python -c \"from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='{OUTPUT_DIR}', repo_id='mmrech/pitvqa-comprehensive-spatial', repo_type='dataset')\"")


if __name__ == "__main__":
    main()
