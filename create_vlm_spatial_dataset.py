#!/usr/bin/env python3
"""
Create VLM Spatial Dataset with Actual Images

Merges:
- Images from mmrech/pitvqa-sage-sft
- Spatial annotations (instruments + anatomy with coordinates)

Output: HuggingFace dataset with images + pointing annotations
"""

import json
import csv
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset, Dataset, Features, Value, Image
from PIL import Image as PILImage
import io

# Configuration
GT_DIR = Path("ground_truth")
GEMINI_ANNOTATIONS = Path("gemini_annotations/surgical_videopoint_molmo_format.json")

# Quadrant to coordinate mapping
QUADRANT_CENTERS = {
    "1": (25.0, 25.0),
    "2": (75.0, 25.0),
    "3": (25.0, 75.0),
    "4": (75.0, 75.0),
    "5": (50.0, 50.0),
}

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

POINT_QUESTIONS = [
    "Point to the {label} in this surgical frame.",
    "Locate the {label} in this image.",
    "Where is the {label} positioned?",
    "Identify the location of the {label}.",
]

ANATOMY_QUESTIONS = [
    "Point to the {label} in this surgical field.",
    "Where is the {label} visible?",
    "Locate the {label} anatomical structure.",
]


def quadrant_to_coordinates(quadrant, jitter=True):
    if quadrant not in QUADRANT_CENTERS:
        return (50.0, 50.0)
    x, y = QUADRANT_CENTERS[quadrant]
    if jitter:
        x += random.uniform(-10, 10)
        y += random.uniform(-10, 10)
        x = max(5, min(95, x))
        y = max(5, min(95, y))
    return (round(x, 1), round(y, 1))


def normalize_instrument(name):
    return INSTRUMENT_NAMES.get(name, name.replace("_", " "))


def load_ground_truth():
    """Load ground truth instrument annotations indexed by (video_id, frame_id)."""
    annotations = {}

    for i in range(1, 26):
        inst_file = GT_DIR / f"instruments_{i:02d}.csv"
        if not inst_file.exists():
            continue

        video_id = f"video_{i:02d}"
        with open(inst_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_id = int(row.get('int_time', 0))
                key = (video_id, frame_id)

                inst1 = row.get('str_instrument1', '')
                inst2 = row.get('str_instrument2', '')
                quad1 = row.get('pos_instrument1', '')
                quad2 = row.get('pos_instrument2', '')

                if inst1 and inst1 not in ['no_visible_instrument', 'out_of_patient']:
                    annotations[key] = {
                        'instrument1': inst1,
                        'instrument2': inst2,
                        'quadrant1': quad1,
                        'quadrant2': quad2,
                    }

    return annotations


def load_gemini_annotations():
    """Load Gemini annotations indexed by (video_id, timestamp)."""
    if not GEMINI_ANNOTATIONS.exists():
        return {}

    with open(GEMINI_ANNOTATIONS, 'r') as f:
        data = json.load(f)

    annotations = defaultdict(list)
    for ann in data:
        if ann.get('confidence', 0) < 0.7:
            continue

        video_id = ann.get('video_id', '')
        timestamps = ann.get('two_fps_timestamps', [])
        points = ann.get('points', [[]])
        label = ann.get('label', '')
        category = ann.get('category', '')

        for i, ts in enumerate(timestamps):
            if i < len(points) and points[i]:
                pt = points[i][0] if isinstance(points[i], list) else points[i]
                # Convert timestamp to frame index (at 2fps, frame = timestamp * 2)
                # But ground truth uses different indexing - need to map
                frame_approx = int(ts * 2)  # Approximate

                annotations[(video_id, frame_approx)].append({
                    'label': label,
                    'category': category,
                    'x': pt.get('x', 50),
                    'y': pt.get('y', 50),
                })

    return dict(annotations)


def create_spatial_sample(image, video_id, frame_id, gt_ann, gemini_anns):
    """Create a spatial training sample with image."""
    samples = []

    # Ground truth instrument pointing
    if gt_ann:
        inst1 = gt_ann['instrument1']
        quad1 = gt_ann['quadrant1']

        if inst1 and quad1:
            x, y = quadrant_to_coordinates(quad1)
            inst_name = normalize_instrument(inst1)
            question = random.choice(POINT_QUESTIONS).format(label=inst_name)
            answer = f"<point x='{x}' y='{y}'>{inst_name}</point>"

            samples.append({
                'image': image,
                'messages': [
                    {'role': 'user', 'content': f'{question}'},
                    {'role': 'assistant', 'content': answer}
                ],
                'video_id': video_id,
                'frame_id': f"{frame_id:05d}.png",
                'annotation_type': 'instrument_pointing',
                'source': 'ground_truth'
            })

        # Dual instrument
        inst2 = gt_ann.get('instrument2', '')
        quad2 = gt_ann.get('quadrant2', '')
        if inst2 and inst2 not in ['no_secondary_instrument', '-2', ''] and quad2:
            x1, y1 = quadrant_to_coordinates(quad1)
            x2, y2 = quadrant_to_coordinates(quad2)
            inst1_name = normalize_instrument(inst1)
            inst2_name = normalize_instrument(inst2)

            question = "What surgical instruments are visible and where are they located?"
            answer = f"Primary: <point x='{x1}' y='{y1}'>{inst1_name}</point>\nSecondary: <point x='{x2}' y='{y2}'>{inst2_name}</point>"

            samples.append({
                'image': image,
                'messages': [
                    {'role': 'user', 'content': f'{question}'},
                    {'role': 'assistant', 'content': answer}
                ],
                'video_id': video_id,
                'frame_id': f"{frame_id:05d}.png",
                'annotation_type': 'dual_instrument',
                'source': 'ground_truth'
            })

    # Gemini anatomy/instrument annotations
    for ann in gemini_anns:
        label = ann['label']
        x, y = ann['x'], ann['y']
        category = ann['category']

        if category == 'anatomy':
            question = random.choice(ANATOMY_QUESTIONS).format(label=label)
        else:
            question = random.choice(POINT_QUESTIONS).format(label=label)

        answer = f"<point x='{x:.1f}' y='{y:.1f}'>{label}</point>"

        samples.append({
            'image': image,
            'messages': [
                {'role': 'user', 'content': f'{question}'},
                {'role': 'assistant', 'content': answer}
            ],
            'video_id': video_id,
            'frame_id': f"{frame_id:05d}.png",
            'annotation_type': f'{category}_pointing',
            'source': 'gemini'
        })

    return samples


def main():
    print("=" * 60)
    print("Creating VLM Spatial Dataset with Images")
    print("=" * 60)

    print("\nLoading ground truth annotations...")
    gt_annotations = load_ground_truth()
    print(f"  Loaded {len(gt_annotations)} frame annotations")

    print("\nLoading Gemini annotations...")
    gemini_annotations = load_gemini_annotations()
    print(f"  Loaded annotations for {len(gemini_annotations)} frames")

    print("\nLoading source images from mmrech/pitvqa-sage-sft...")
    source_ds = load_dataset("mmrech/pitvqa-sage-sft", split="train")
    print(f"  Loaded {len(source_ds)} images")

    # Index source images by (video_id, frame_id)
    print("\nIndexing images...")
    image_index = {}
    for sample in source_ds:
        video_id = sample['video_id']
        frame_id = int(sample['frame_id'].replace('.png', ''))
        image_index[(video_id, frame_id)] = sample['image']
    print(f"  Indexed {len(image_index)} unique frames")

    # Create spatial samples
    print("\nCreating spatial samples...")
    all_samples = []
    stats = defaultdict(int)

    for (video_id, frame_id), image in image_index.items():
        gt_ann = gt_annotations.get((video_id, frame_id))
        gemini_anns = gemini_annotations.get((video_id, frame_id), [])

        samples = create_spatial_sample(image, video_id, frame_id, gt_ann, gemini_anns)

        for s in samples:
            all_samples.append(s)
            stats[s['annotation_type']] += 1
            stats[s['source']] += 1

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(all_samples)}")
    for key, count in sorted(stats.items()):
        print(f"   {key}: {count}")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"\n   Train: {len(train_samples)}")
    print(f"   Validation: {len(val_samples)}")

    # Convert to HuggingFace Dataset
    print("\nCreating HuggingFace dataset...")

    def samples_to_dict(samples):
        return {
            'image': [s['image'] for s in samples],
            'messages': [json.dumps(s['messages']) for s in samples],
            'video_id': [s['video_id'] for s in samples],
            'frame_id': [s['frame_id'] for s in samples],
            'annotation_type': [s['annotation_type'] for s in samples],
            'source': [s['source'] for s in samples],
        }

    train_ds = Dataset.from_dict(samples_to_dict(train_samples))
    val_ds = Dataset.from_dict(samples_to_dict(val_samples))

    # Cast image column
    train_ds = train_ds.cast_column('image', Image())
    val_ds = val_ds.cast_column('image', Image())

    # Push to hub
    print("\nPushing to HuggingFace Hub...")
    train_ds.push_to_hub("mmrech/pitvqa-spatial-vlm", split="train")
    val_ds.push_to_hub("mmrech/pitvqa-spatial-vlm", split="validation")

    print("\n✅ Done!")
    print("Dataset: https://huggingface.co/datasets/mmrech/pitvqa-spatial-vlm")


if __name__ == "__main__":
    main()
