#!/usr/bin/env python3
"""
Run Surgical Annotation Pipeline on PitVQA Dataset
===================================================
Loads frames from mmrech/pitvqa-sage-sft dataset and runs
multi-agent annotation to create surgical video pointing data.
"""

import subprocess
import sys
import json
import asyncio
from pathlib import Path
from datetime import datetime
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from surgical_annotation_pipeline import SurgicalAnnotationPipeline

# Configuration
DATASET_NAME = "mmrech/pitvqa-sage-sft"
OUTPUT_DIR = Path(__file__).parent / "surgical_pointing_annotations"
BATCH_SIZE = 50  # Process in batches to save intermediate results
MAX_FRAMES = None  # Set to None to process all, or a number to limit


def load_pitvqa_dataset(split: str = "train", max_samples: int = None):
    """Load the PitVQA dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets", "pillow"], check=True)
        from datasets import load_dataset

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Loaded {len(dataset)} samples")
    return dataset


def extract_image_from_sample(sample) -> Image.Image:
    """Extract PIL Image from dataset sample"""
    image_data = sample.get("image") or sample.get("images")

    if image_data is None:
        return None

    # Handle different image formats
    if isinstance(image_data, Image.Image):
        return image_data
    elif isinstance(image_data, dict) and "bytes" in image_data:
        return Image.open(io.BytesIO(image_data["bytes"]))
    elif isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    elif hasattr(image_data, "convert"):
        return image_data.convert("RGB")

    return None


def prepare_frames_for_annotation(dataset, start_idx: int = 0, batch_size: int = 50):
    """Prepare a batch of frames for annotation"""
    frames = []

    for i, sample in enumerate(dataset):
        if i < start_idx:
            continue
        if i >= start_idx + batch_size:
            break

        image = extract_image_from_sample(sample)
        if image is None:
            print(f"Warning: Could not extract image from sample {i}")
            continue

        frame_data = {
            "image": image,
            "video_id": sample.get("video_id", f"pitvqa_{i}"),
            "frame_id": sample.get("frame_id", i),
            "timestamp": i * 0.5,  # Assume 2 FPS
            "qa_type": sample.get("qa_type", "unknown"),
            "original_messages": sample.get("messages", "")
        }
        frames.append(frame_data)

    return frames


async def run_annotation_pipeline(
    dataset,
    output_dir: Path,
    batch_size: int = 50,
    max_frames: int = None
):
    """Run the full annotation pipeline on dataset"""

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = SurgicalAnnotationPipeline()

    # Tracking
    all_annotations = []
    total_frames = min(len(dataset), max_frames) if max_frames else len(dataset)
    start_time = datetime.now()

    print(f"\n{'='*60}")
    print(f"Starting Surgical Annotation Pipeline")
    print(f"{'='*60}")
    print(f"Total frames to process: {total_frames}")
    print(f"Batch size: {batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Process in batches
    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        print(f"\n--- Batch {batch_start//batch_size + 1}: frames {batch_start}-{batch_end} ---")

        # Prepare batch
        frames = prepare_frames_for_annotation(dataset, batch_start, batch_size)

        if not frames:
            print("No frames in this batch, skipping...")
            continue

        # Run annotation
        try:
            batch_results = await pipeline.annotate_batch(
                frames,
                output_path=str(output_dir / f"batch_{batch_start}_{batch_end}.json")
            )
            all_annotations.extend(batch_results)

        except Exception as e:
            print(f"Error processing batch: {e}")
            # Save what we have so far
            save_checkpoint(all_annotations, output_dir)
            continue

        # Progress report
        elapsed = (datetime.now() - start_time).total_seconds()
        frames_done = len(all_annotations)
        rate = frames_done / elapsed if elapsed > 0 else 0
        eta = (total_frames - frames_done) / rate if rate > 0 else 0

        print(f"\nProgress: {frames_done}/{total_frames} ({100*frames_done/total_frames:.1f}%)")
        print(f"Rate: {rate:.2f} frames/sec")
        print(f"ETA: {eta/60:.1f} minutes remaining")

    # Save final results
    save_final_results(all_annotations, output_dir, pipeline)

    print(f"\n{'='*60}")
    print(f"Annotation Complete!")
    print(f"{'='*60}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Total time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")

    return all_annotations


def save_checkpoint(annotations: list, output_dir: Path):
    """Save checkpoint of current annotations"""
    checkpoint_path = output_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved checkpoint: {checkpoint_path}")


def save_final_results(annotations: list, output_dir: Path, pipeline: SurgicalAnnotationPipeline):
    """Save final results in multiple formats"""

    # Raw annotations
    raw_path = output_dir / "surgical_annotations_raw.json"
    with open(raw_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved raw annotations: {raw_path}")

    # Molmo2-VideoPoint format
    molmo_format = pipeline.to_molmo_videopoint_format(annotations)
    molmo_path = output_dir / "surgical_videopoint_molmo_format.json"
    with open(molmo_path, 'w') as f:
        json.dump(molmo_format, f, indent=2)
    print(f"Saved Molmo2-VideoPoint format: {molmo_path}")

    # Statistics
    stats = compute_statistics(annotations)
    stats_path = output_dir / "annotation_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {stats_path}")

    # Generate summary report
    generate_report(annotations, stats, output_dir)


def compute_statistics(annotations: list) -> dict:
    """Compute annotation statistics"""
    if not annotations:
        return {"total": 0}

    # Count by category
    categories = {}
    labels = {}
    confidences = []
    consensus_methods = {}

    for ann in annotations:
        cat = ann.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

        label = ann.get("label", "unknown")
        labels[label] = labels.get(label, 0) + 1

        confidences.append(ann.get("confidence", 0))

        method = ann.get("consensus_method", "unknown")
        consensus_methods[method] = consensus_methods.get(method, 0) + 1

    return {
        "total_annotations": len(annotations),
        "categories": categories,
        "labels": dict(sorted(labels.items(), key=lambda x: -x[1])[:20]),  # Top 20
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
        "min_confidence": min(confidences) if confidences else 0,
        "max_confidence": max(confidences) if confidences else 0,
        "consensus_methods": consensus_methods,
        "points_per_frame": {
            "avg": sum(ann.get("count", 0) for ann in annotations) / len(annotations),
            "max": max(ann.get("count", 0) for ann in annotations),
            "min": min(ann.get("count", 0) for ann in annotations)
        }
    }


def generate_report(annotations: list, stats: dict, output_dir: Path):
    """Generate human-readable report"""
    report = f"""
# Surgical Video Pointing Annotation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total annotations: {stats['total_annotations']}
- Average confidence: {stats['avg_confidence']:.3f}
- Confidence range: {stats['min_confidence']:.3f} - {stats['max_confidence']:.3f}

## Categories
"""
    for cat, count in stats.get('categories', {}).items():
        report += f"- {cat}: {count} ({100*count/stats['total_annotations']:.1f}%)\n"

    report += """
## Top Labels
"""
    for label, count in list(stats.get('labels', {}).items())[:10]:
        report += f"- {label}: {count}\n"

    report += """
## Consensus Methods
"""
    for method, count in stats.get('consensus_methods', {}).items():
        report += f"- {method}: {count}\n"

    report += f"""
## Points per Frame
- Average: {stats['points_per_frame']['avg']:.2f}
- Range: {stats['points_per_frame']['min']} - {stats['points_per_frame']['max']}

## Output Files
- surgical_annotations_raw.json: Raw annotation data
- surgical_videopoint_molmo_format.json: Molmo2-VideoPoint compatible format
- annotation_statistics.json: Detailed statistics
"""

    report_path = output_dir / "ANNOTATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report: {report_path}")


async def test_single_frame():
    """Test annotation on a single frame"""
    print("Testing single frame annotation...")

    # Load a single sample
    dataset = load_pitvqa_dataset(max_samples=1)
    frames = prepare_frames_for_annotation(dataset, 0, 1)

    if not frames:
        print("No frames available for testing")
        return

    pipeline = SurgicalAnnotationPipeline()
    result = await pipeline.annotate_frame(
        frames[0]["image"],
        frames[0]["video_id"],
        frames[0]["frame_id"]
    )

    print("\n--- Test Result ---")
    print(json.dumps(result.__dict__ if hasattr(result, '__dict__') else result, indent=2))


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Surgical Video Annotation Pipeline")
    parser.add_argument("--test", action="store_true", help="Test on single frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    if args.test:
        asyncio.run(test_single_frame())
    else:
        output_dir = Path(args.output) if args.output else OUTPUT_DIR
        dataset = load_pitvqa_dataset(max_samples=args.max_frames)
        asyncio.run(run_annotation_pipeline(
            dataset,
            output_dir,
            batch_size=args.batch_size,
            max_frames=args.max_frames
        ))


if __name__ == "__main__":
    main()
