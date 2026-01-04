#!/usr/bin/env python3
"""
Grok 4.1 Fast Surgical Annotation Pipeline
==========================================
Uses xAI's Grok-4.1-fast model for surgical image annotation.
Lower latency and less content moderation than GPT-4o.
"""

import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image
import httpx

# Configuration
DATASET_NAME = "mmrech/pitvqa-sage-sft"
OUTPUT_DIR = Path(__file__).parent / "grok_annotations"
BATCH_SIZE = 25
MAX_FRAMES = 500

# xAI Grok endpoint (OpenAI-compatible)
XAI_BASE_URL = "https://api.x.ai/v1"

# Surgical annotation prompt with PitVQA instrument vocabulary
ANNOTATION_PROMPT = """Analyze this endoscopic surgical image. Identify ALL visible objects.

INSTRUMENTS to look for (use exact names when possible):
- suction, suction_coagulator
- freer_elevator, spatula_dissector
- pituitary_rongeurs, kerrisons
- cottle, ring_curette
- bipolar, drill
- nasal_cutting_forceps
- micro_doppler, speculum
- knife, needle
- haemostatic_foam

ANATOMY to identify:
- nasal cavity, nasal septum
- sphenoid sinus, sella
- pituitary gland, tumor tissue
- blood vessels, mucosal tissue

EVENTS to note:
- bleeding, irrigation
- tissue manipulation, coagulation

For EACH object provide:
- label: specific name from lists above
- x, y: coordinates (0-100 percentage, 0,0 = top-left)
- category: instruments | anatomy | events
- confidence: 0.0-1.0

RESPOND IN JSON:
{
    "phase": "current surgical step",
    "annotations": [
        {"label": "name", "x": float, "y": float, "category": "category", "confidence": float}
    ]
}"""


def load_dataset_streaming():
    """Load dataset in streaming mode"""
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
    """Convert PIL Image to base64"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    if upscale and (image.width < 512 or image.height < 512):
        scale = max(512 / image.width, 512 / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def annotate_frame_grok(client: httpx.Client, image_base64: str, api_key: str) -> dict:
    """Run Grok-4.1-fast annotation on a single frame"""
    try:
        response = client.post(
            f"{XAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-4-latest",
                "messages": [{
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
                "max_tokens": 1500,
                "temperature": 0.2
            },
            timeout=60.0
        )

        result = response.json()

        if "error" in result:
            return {"error": result["error"], "annotations": []}

        content = result["choices"][0]["message"]["content"]

        # Parse JSON
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
            "video_source": "pitvqa_surgical",
            "annotator": "grok-4.1-fast"
        }
        molmo_entries.append(entry)

    return molmo_entries


def run_annotation_pipeline(max_frames: int = None):
    """Main annotation pipeline"""
    print("=" * 60)
    print("GROK 4.1 FAST SURGICAL ANNOTATION PIPELINE")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        print("ERROR: XAI_API_KEY not set")
        print("Set with: export XAI_API_KEY='your-key'")
        return

    print(f"✓ xAI API key: Set ({len(api_key)} chars)")

    # Initialize
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = httpx.Client()

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
            annotation = annotate_frame_grok(client, image_base64, api_key)

            # Convert to Molmo format
            molmo_entries = to_molmo_format(annotation, video_id, frame_id, timestamp)
            all_molmo_entries.extend(molmo_entries)

            ann_count = len(annotation.get("annotations", []))
            processed += 1

            elapsed = (datetime.now() - start_time).total_seconds()
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"[{i+1:4d}] {video_id} - {ann_count} annotations ({rate:.1f} frames/sec)")

            # Save intermediate
            if processed % BATCH_SIZE == 0:
                save_results(all_molmo_entries, OUTPUT_DIR / "intermediate_results.json")
                print(f"  → Saved intermediate ({len(all_molmo_entries)} entries)")

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

    client.close()


def save_results(entries: list, output_file: Path):
    """Save results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Grok 4.1 Surgical Annotation Pipeline")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    run_annotation_pipeline(max_frames=args.max_frames)
