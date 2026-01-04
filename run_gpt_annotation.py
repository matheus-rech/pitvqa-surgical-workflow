#!/usr/bin/env python3
"""
GPT-4o Surgical Annotation Pipeline
====================================
Runs GPT-4o vision annotation on PitVQA-SAGE-SFT dataset.
Outputs in Molmo2-VideoPoint compatible format.
"""

import os
import sys
import json
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
from openai import OpenAI

# Configuration
DATASET_NAME = "mmrech/pitvqa-sage-sft"
OUTPUT_DIR = Path(__file__).parent / "gpt_annotations"
BATCH_SIZE = 25
MAX_FRAMES = 100  # Set to None for full dataset

# Neutral prompt (avoids content moderation)
ANNOTATION_PROMPT = """Analyze this endoscopic image. Identify ALL visible objects:

For EACH object provide:
- Label (descriptive name)
- x, y coordinates (0-100 percentage, where 0,0 is top-left)
- Category: instruments | anatomy | events
- Confidence (0.0-1.0)

Categories include:
- instruments: metal tools, probes, tubes, forceps, suction devices
- anatomy: tissue structures, cavities, vessels, glands
- events: active processes, fluid flow, manipulation

RESPOND IN JSON:
{
    "phase": "current activity",
    "annotations": [
        {"label": "name", "x": float, "y": float, "category": "category", "confidence": float}
    ]
}"""


def load_dataset_streaming():
    """Load dataset in streaming mode for memory efficiency"""
    try:
        from datasets import load_dataset
        print(f"Loading dataset: {DATASET_NAME} (streaming)")
        return load_dataset(DATASET_NAME, split="train", streaming=True)
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset
        return load_dataset(DATASET_NAME, split="train", streaming=True)


def image_to_base64(image: Image.Image, upscale: bool = True) -> str:
    """Convert PIL Image to base64, optionally upscaling for better analysis"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Upscale small images for better analysis
    if upscale and (image.width < 512 or image.height < 512):
        scale = max(512 / image.width, 512 / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def annotate_frame(client: OpenAI, image_base64: str) -> dict:
    """Run GPT-4o annotation on a single frame"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": ANNOTATION_PROMPT}
                ]
            }],
            max_tokens=1500,
            temperature=0.2
        )

        content = response.choices[0].message.content

        # Parse JSON from response
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content

            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_response": content, "annotations": []}

    except Exception as e:
        return {"error": str(e), "annotations": []}


def to_molmo_format(annotation: dict, video_id: str, frame_id: int, timestamp: float) -> list:
    """Convert annotation to Molmo2-VideoPoint format"""
    molmo_entries = []

    for ann in annotation.get("annotations", []):
        label = ann.get("label", "unknown")
        x = ann.get("x", 50.0)
        y = ann.get("y", 50.0)
        category = ann.get("category", "unknown")
        confidence = ann.get("confidence", 0.5)

        entry = {
            "video_id": video_id,
            "question": f"Point to the {label}",
            "label": label,
            "count": 1,
            "two_fps_timestamps": [timestamp],
            "points": [[{"x": x, "y": y}]],
            "category": category,
            "confidence": confidence,
            "video_source": "pitvqa_surgical"
        }
        molmo_entries.append(entry)

    return molmo_entries


def run_annotation_pipeline(max_frames: int = None):
    """Main annotation pipeline"""
    print("=" * 60)
    print("GPT-4o SURGICAL ANNOTATION PIPELINE")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return

    print(f"✓ OpenAI API key: Set ({len(api_key)} chars)")

    # Initialize
    client = OpenAI(api_key=api_key)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset
    dataset = load_dataset_streaming()

    # Process frames
    all_molmo_entries = []
    processed = 0
    errors = 0
    start_time = datetime.now()

    print(f"\nProcessing frames (max: {max_frames or 'all'})...")
    print("-" * 60)

    for i, sample in enumerate(dataset):
        if max_frames and i >= max_frames:
            break

        # Extract image
        image_data = sample.get("image") or sample.get("images")
        if image_data is None:
            continue

        if isinstance(image_data, Image.Image):
            image = image_data
        elif hasattr(image_data, "convert"):
            image = image_data.convert("RGB")
        else:
            continue

        # Get metadata
        video_id = sample.get("video_id", f"pitvqa_{i:05d}")
        frame_id = sample.get("frame_id", i)
        timestamp = i * 0.5  # Assume 2 FPS

        # Convert and annotate
        try:
            image_base64 = image_to_base64(image)
            annotation = annotate_frame(client, image_base64)

            # Convert to Molmo format
            molmo_entries = to_molmo_format(annotation, video_id, frame_id, timestamp)
            all_molmo_entries.extend(molmo_entries)

            # Count annotations
            ann_count = len(annotation.get("annotations", []))
            processed += 1

            # Progress
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"[{i+1:4d}] {video_id} - {ann_count} annotations ({rate:.1f} frames/sec)")

            # Save intermediate results every batch
            if processed % BATCH_SIZE == 0:
                save_results(all_molmo_entries, OUTPUT_DIR / "intermediate_results.json")
                print(f"  → Saved intermediate results ({len(all_molmo_entries)} entries)")

        except Exception as e:
            errors += 1
            print(f"[{i+1:4d}] ERROR: {str(e)[:50]}")

    # Final save
    output_file = OUTPUT_DIR / "surgical_videopoint_molmo_format.json"
    save_results(all_molmo_entries, output_file)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {processed}")
    print(f"Total annotations: {len(all_molmo_entries)}")
    print(f"Errors: {errors}")
    print(f"Time elapsed: {elapsed:.1f}s ({processed/elapsed:.2f} frames/sec)")
    print(f"Output: {output_file}")


def save_results(entries: list, output_file: Path):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-4o Surgical Annotation Pipeline")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                        help="Maximum frames to process (default: 100)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for intermediate saves")

    args = parser.parse_args()

    run_annotation_pipeline(max_frames=args.max_frames)
