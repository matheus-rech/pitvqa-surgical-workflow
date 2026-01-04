#!/usr/bin/env python3
"""
Gemini 2.5 Pro Surgical Annotation Pipeline
============================================
Uses Google's Gemini 2.5 Pro for surgical image annotation.
Supports both Google AI Studio and Vertex AI endpoints.
"""

import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

# Configuration
DATASET_NAME = "mmrech/pitvqa-sage-sft"
OUTPUT_DIR = Path(__file__).parent / "gemini_annotations"
BATCH_SIZE = 25
MAX_FRAMES = 500

# Model options
GEMINI_MODELS = [
    "gemini-3-flash-preview",  # Latest Gemini 3 Flash
    "gemini-2.5-flash-thinking-exp",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.0-flash-exp"
]

# Surgical annotation prompt with PitVQA vocabulary
ANNOTATION_PROMPT = """Analyze this endoscopic pituitary surgery image. Identify ALL visible objects.

INSTRUMENTS (use exact names):
- suction, suction_coagulator
- freer_elevator, spatula_dissector
- pituitary_rongeurs, kerrisons
- cottle, ring_curette
- bipolar, drill
- nasal_cutting_forceps
- micro_doppler, speculum
- knife, needle, haemostatic_foam

ANATOMY to identify:
- nasal cavity, nasal septum
- sphenoid sinus, sella
- pituitary gland, tumor tissue
- blood vessels, mucosal tissue

SURGICAL PHASES:
- nasal_corridor_creation
- anterior_sphenoidotomy
- septum_displacement
- sphenoid_sinus_clearance
- sellotomy, durotomy
- tumour_excision
- haemostasis
- closure (graft_placement, dural_sealant)

For EACH visible object:
- label: specific name from vocabulary
- x, y: coordinates (0-100 percentage)
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


def annotate_frame_gemini(image_base64: str, api_key: str, model: str = None) -> dict:
    """Run Gemini annotation on a single frame"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        # Try models in order
        models_to_try = [model] if model else GEMINI_MODELS

        for model_name in models_to_try:
            try:
                model_obj = genai.GenerativeModel(model_name)

                # Decode image for Gemini
                image_bytes = base64.b64decode(image_base64)

                response = model_obj.generate_content([
                    {"mime_type": "image/jpeg", "data": image_bytes},
                    ANNOTATION_PROMPT
                ], generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1500
                })

                content = response.text

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
                if "not found" in str(e).lower():
                    continue
                raise

        return {"error": "No working model found", "annotations": []}

    except ImportError:
        print("Installing google-generativeai...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "google-generativeai"], check=True)
        import google.generativeai as genai
        return annotate_frame_gemini(image_base64, api_key, model)
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
            "annotator": "gemini-2.5-pro"
        }
        molmo_entries.append(entry)

    return molmo_entries


def run_annotation_pipeline(max_frames: int = None, model: str = None):
    """Main annotation pipeline"""
    print("=" * 60)
    print("GEMINI 2.5 PRO SURGICAL ANNOTATION PIPELINE")
    print("=" * 60)

    # Check API key
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set")
        print("Get one at: https://aistudio.google.com/apikey")
        return

    print(f"✓ Google API key: Set ({len(api_key)} chars)")

    # Initialize
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
            annotation = annotate_frame_gemini(image_base64, api_key, model)

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


def save_results(entries: list, output_file: Path):
    """Save results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gemini Surgical Annotation Pipeline")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--model", type=str, default=None, help="Specific Gemini model to use")

    args = parser.parse_args()

    run_annotation_pipeline(max_frames=args.max_frames, model=args.model)
