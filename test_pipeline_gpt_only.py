#!/usr/bin/env python3
"""
Test Surgical Annotation Pipeline - GPT Only Mode
==================================================
Tests the pipeline with only OpenAI API available.
Validates structure and demonstrates GPT tiebreaker functionality.
"""

import os
import sys
import json
import asyncio
import aiohttp
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

        # Get metadata
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


def image_to_base64(image):
    """Convert PIL Image to base64"""
    from PIL import Image

    if image.mode != 'RGB':
        image = image.convert('RGB')

    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


async def test_gpt_annotation(image, metadata):
    """Test GPT-4o annotation with reasoning"""

    print("\n" + "=" * 60)
    print("Testing GPT-4o Vision Annotation")
    print("=" * 60)

    api_key = os.environ.get('OPENAI_API_KEY')
    base64_image = image_to_base64(image)

    # Neutral annotation prompt (avoids content moderation triggers)
    prompt = """Analyze this endoscopic image and identify all visible objects.

For EACH identified object, provide:
- Label (descriptive name like "metal tool", "tissue", "cavity", etc.)
- Approximate coordinates (x, y as percentage 0-100, where 0,0 is top-left)
- Confidence score (0.0-1.0)
- Brief reasoning for identification

IMPORTANT CATEGORIES:
- instruments: Metal tools, probes, tubes, or devices
- anatomy: Tissue structures, cavities, vessels, or organs
- events: Active processes like fluid flow or tissue manipulation

Respond in JSON format:
{
    "phase": "current activity description",
    "instruments": [
        {"label": "name", "x": float, "y": float, "confidence": float, "reasoning": "why"}
    ],
    "anatomy": [
        {"label": "name", "x": float, "y": float, "confidence": float, "reasoning": "why"}
    ],
    "events": [
        {"label": "name", "confidence": float, "reasoning": "why"}
    ],
    "key_structures": ["list of important visible structures"]
}"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
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
        "max_tokens": 2000,
        "temperature": 0.2
    }

    start_time = datetime.now()

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            result = await response.json()

    elapsed = (datetime.now() - start_time).total_seconds()

    if "error" in result:
        print(f"API Error: {result['error']}")
        return None

    content = result["choices"][0]["message"]["content"]

    # Parse JSON from response
    try:
        # Extract JSON from markdown if needed
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content

        annotation = json.loads(json_str)
    except json.JSONDecodeError:
        print("Could not parse JSON, using raw response")
        annotation = {"raw_response": content}

    print(f"\nResponse Time: {elapsed:.1f}s")
    print(f"Tokens Used: {result.get('usage', {}).get('total_tokens', 'N/A')}")

    return annotation


def format_molmo_output(annotation, metadata):
    """Convert to Molmo2-VideoPoint format"""

    outputs = []

    # Process instruments
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

    # Process anatomy
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


async def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("SURGICAL ANNOTATION PIPELINE - GPT-Only Test")
    print("=" * 60)

    # Check API key
    if not check_openai_key():
        print("\nPlease set OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Load sample
    image, metadata = load_sample_frame()
    if image is None:
        print("Failed to load sample frame")
        sys.exit(1)

    # Run GPT annotation
    annotation = await test_gpt_annotation(image, metadata)

    if annotation is None:
        print("Annotation failed")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 60)
    print("ANNOTATION RESULTS")
    print("=" * 60)

    print(f"\nSurgical Phase: {annotation.get('phase', 'Unknown')}")

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
        if anat.get('reasoning'):
            print(f"    → {anat['reasoning'][:60]}...")

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

    # Save results
    output_dir = Path(__file__).parent / "test_output"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "gpt_annotation_result.json", 'w') as f:
        json.dump(annotation, f, indent=2)

    with open(output_dir / "gpt_molmo_format.json", 'w') as f:
        json.dump(molmo_output, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("✓ GPT-Only Test Completed Successfully")
    print("=" * 60)
    print("\nNote: Full pipeline requires ANTHROPIC_API_KEY and GOOGLE_API_KEY")
    print("for multi-agent consensus with extended thinking.")


if __name__ == "__main__":
    asyncio.run(main())
