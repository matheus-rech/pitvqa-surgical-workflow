#!/usr/bin/env python3
"""
Test Surgical Annotation Pipeline - Using OpenAI SDK
=====================================================
Uses official OpenAI Python library for proper image handling.
"""

import os
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent))

def check_openai_key():
    """Verify OpenAI API key is set"""
    key = os.environ.get('OPENAI_API_KEY', '')
    if key and len(key) > 10:
        print(f"✓ OPENAI_API_KEY: Set ({len(key)} chars)")
        return True
    print("✗ OPENAI_API_KEY: NOT SET")
    return False


def load_sample_frame():
    """Load a sample frame from the PitVQA dataset"""
    try:
        from datasets import load_dataset
        from PIL import Image

        print("\n" + "=" * 60)
        print("Loading Sample Frame from mmrech/pitvqa-sage-sft")
        print("=" * 60)

        dataset = load_dataset("mmrech/pitvqa-sage-sft", split="train", streaming=True)
        sample = next(iter(dataset))

        image_data = sample.get("image") or sample.get("images")

        if isinstance(image_data, Image.Image):
            image = image_data
        elif isinstance(image_data, dict) and "bytes" in image_data:
            image = Image.open(BytesIO(image_data["bytes"]))
        elif hasattr(image_data, "convert"):
            image = image_data.convert("RGB")
        else:
            print(f"Unknown image format: {type(image_data)}")
            return None, None

        video_id = sample.get("video_id", "pitvqa_test")
        qa_type = sample.get("qa_type", "unknown")
        messages = sample.get("messages", [])

        print(f"  Video ID: {video_id}")
        print(f"  QA Type: {qa_type}")
        print(f"  Image Size: {image.size}")

        if messages:
            print(f"  Original Q: {messages[0].get('content', '')[:80]}...")

        return image, {
            "video_id": video_id,
            "qa_type": qa_type,
            "frame_id": 0,
            "timestamp": 0.0
        }

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def prepare_image_for_api(image):
    """Prepare image for OpenAI API - upscale and encode"""
    from PIL import Image as PILImage

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Upscale small images
    min_size = 512
    if image.width < min_size or image.height < min_size:
        scale = max(min_size / image.width, min_size / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, PILImage.Resampling.LANCZOS)
        print(f"  Upscaled to: {image.size}")

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    img_bytes = buffered.getvalue()
    print(f"  Image size: {len(img_bytes) / 1024:.1f} KB")

    return base64.b64encode(img_bytes).decode('utf-8')


def test_gpt_annotation(image, metadata):
    """Test GPT-4o annotation using OpenAI SDK"""
    from openai import OpenAI

    print("\n" + "=" * 60)
    print("Testing GPT-4o Vision Annotation (OpenAI SDK)")
    print("=" * 60)

    client = OpenAI()
    base64_image = prepare_image_for_api(image)

    prompt = """You are analyzing a real surgical image from transsphenoidal pituitary surgery.

Look at THIS specific image and identify what you can actually see:
1. Visible surgical instruments
2. Anatomical structures you can identify
3. Current surgical activity

Be specific about what you observe in THIS image. Provide coordinates as percentages (0-100).

Respond in this exact JSON format:
{
    "phase": "what surgical phase this appears to be",
    "instruments": [
        {"label": "instrument_name", "x": 50.0, "y": 50.0, "confidence": 0.8, "reasoning": "why"}
    ],
    "anatomy": [
        {"label": "structure_name", "x": 50.0, "y": 50.0, "confidence": 0.8, "reasoning": "why"}
    ],
    "events": [
        {"label": "event_name", "confidence": 0.8, "reasoning": "why"}
    ],
    "safety_critical": ["list of critical structures visible"],
    "image_quality": "assessment of image quality for analysis"
}"""

    start_time = datetime.now()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.2
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        content = response.choices[0].message.content

        print(f"\nResponse Time: {elapsed:.1f}s")
        print(f"Tokens Used: {response.usage.total_tokens}")

        # Parse JSON
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0]
            else:
                json_str = content

            annotation = json.loads(json_str)
            print("\n✓ Successfully parsed JSON response")
        except json.JSONDecodeError:
            print("\n⚠ Could not parse JSON, showing raw response:")
            print(content[:500])
            annotation = {"raw_response": content}

        return annotation

    except Exception as e:
        print(f"Error: {e}")
        return None


def format_molmo_output(annotation, metadata):
    """Convert to Molmo2-VideoPoint format"""

    outputs = []

    for inst in annotation.get("instruments", []):
        outputs.append({
            "video_id": metadata["video_id"],
            "question": f"Point to the {inst['label'].replace('_', ' ')}",
            "label": inst["label"],
            "count": 1,
            "two_fps_timestamps": [metadata["timestamp"]],
            "points": [[{"x": inst.get("x", 50), "y": inst.get("y", 50)}]],
            "category": "instruments",
            "confidence": inst.get("confidence", 0.8),
            "reasoning": inst.get("reasoning", ""),
            "video_source": "pitvqa_surgical"
        })

    for anat in annotation.get("anatomy", []):
        outputs.append({
            "video_id": metadata["video_id"],
            "question": f"Point to the {anat['label'].replace('_', ' ')}",
            "label": anat["label"],
            "count": 1,
            "two_fps_timestamps": [metadata["timestamp"]],
            "points": [[{"x": anat.get("x", 50), "y": anat.get("y", 50)}]],
            "category": "anatomy",
            "confidence": anat.get("confidence", 0.8),
            "reasoning": anat.get("reasoning", ""),
            "video_source": "pitvqa_surgical"
        })

    return outputs


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("SURGICAL ANNOTATION PIPELINE - OpenAI SDK Test")
    print("=" * 60)

    if not check_openai_key():
        print("\nPlease set OPENAI_API_KEY environment variable")
        sys.exit(1)

    image, metadata = load_sample_frame()
    if image is None:
        print("Failed to load sample frame")
        sys.exit(1)

    annotation = test_gpt_annotation(image, metadata)

    if annotation is None:
        print("Annotation failed")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 60)
    print("ANNOTATION RESULTS")
    print("=" * 60)

    if "raw_response" in annotation:
        print("\nRaw Response (first 500 chars):")
        print(annotation["raw_response"][:500])
    else:
        print(f"\nSurgical Phase: {annotation.get('phase', 'Unknown')}")
        print(f"Image Quality: {annotation.get('image_quality', 'Unknown')}")

        print(f"\nInstruments ({len(annotation.get('instruments', []))}):")
        for inst in annotation.get("instruments", []):
            print(f"  • {inst['label']}: ({inst.get('x', 0):.1f}, {inst.get('y', 0):.1f}) "
                  f"conf={inst.get('confidence', 0):.2f}")
            if inst.get('reasoning'):
                print(f"    → {inst['reasoning'][:60]}...")

        print(f"\nAnatomy ({len(annotation.get('anatomy', []))}):")
        for anat in annotation.get("anatomy", []):
            print(f"  • {anat['label']}: ({anat.get('x', 0):.1f}, {anat.get('y', 0):.1f}) "
                  f"conf={anat.get('confidence', 0):.2f}")

        print(f"\nEvents ({len(annotation.get('events', []))}):")
        for event in annotation.get("events", []):
            print(f"  • {event['label']}: conf={event.get('confidence', 0):.2f}")

        safety = annotation.get("safety_critical", [])
        if safety:
            print(f"\n⚠️  Safety-Critical Structures: {', '.join(safety)}")

    # Convert to Molmo format
    molmo_output = format_molmo_output(annotation, metadata)

    print("\n" + "=" * 60)
    print("MOLMO2-VIDEOPOINT FORMAT (sample)")
    print("=" * 60)
    if molmo_output:
        print(json.dumps(molmo_output[0], indent=2))
    else:
        print("No structured output to convert")

    # Save results
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "gpt_sdk_annotation.json", 'w') as f:
        json.dump(annotation, f, indent=2)

    with open(output_dir / "gpt_sdk_molmo.json", 'w') as f:
        json.dump(molmo_output, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("✓ OpenAI SDK Test Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
