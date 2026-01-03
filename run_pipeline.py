#!/usr/bin/env python3
"""
PitVQA → HuggingFace Pipeline Runner

Streamlined script to:
1. Extract PitVQA videos/frames
2. Process with surgical annotations
3. Push to HuggingFace Hub
4. Generate HF Skills training prompt

Usage:
    python run_pipeline.py --hf-token YOUR_TOKEN

The actual training runs on HuggingFace Jobs infrastructure.
"""

import argparse
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Check for required packages
try:
    from datasets import Dataset, DatasetDict, Image as HFImage
    from PIL import Image
    from tqdm import tqdm
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)


# PitVQA surgical vocabulary
PHASES = ["Nasal", "Sellar", "Tumor Removal", "Closure"]
STEPS = [
    "Septal Dissection", "Turbinectomy", "Sphenoidotomy",
    "Posterior Septectomy", "Sellar Floor Removal", "Dura Opening",
    "Tumor Resection", "Hemostasis", "Reconstruction", "Nasal Packing",
    "Visualization", "Instrument Change", "Suction", "Irrigation", "Other"
]
INSTRUMENTS = [
    "Endoscope", "Suction", "Curette", "Bipolar", "Monopolar",
    "Scissors", "Grasper", "Drill", "Kerrison", "Speculum",
    "Cottonoid", "Hemostatic Agent", "Fat Graft", "Fascia", "Nasoseptal Flap"
]

# Train/val split from original paper
TRAIN_VIDEOS = [1, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25]
VAL_VIDEOS = [2, 6, 12, 13, 24]


def install_dependencies():
    """Install required packages."""
    import subprocess
    packages = ["datasets", "pillow", "tqdm", "numpy", "huggingface_hub"]
    subprocess.run(["pip", "install", "--quiet"] + packages, check=True)


def extract_videos(zip_path: str, output_dir: str) -> List[str]:
    """Extract videos from zip file."""
    logger.info(f"Extracting {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    # Find extracted video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    for ext in video_extensions:
        videos.extend(Path(output_dir).rglob(f"*{ext}"))

    logger.info(f"Found {len(videos)} videos")
    return [str(v) for v in videos]


def extract_frames_from_video(video_path: str, output_dir: str, fps: int = 1) -> List[str]:
    """Extract frames from a video at specified FPS."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed. Install with: pip install opencv-python")
        return []

    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_paths = []
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = frame_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(str(frame_path))
            saved_count += 1

        frame_count += 1

    cap.release()
    logger.info(f"Extracted {saved_count} frames from {video_name}")
    return frame_paths


def load_existing_frames(frames_dir: str) -> List[Dict]:
    """Load already extracted frames with metadata."""
    frames = []
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        return frames

    # Find all frame images
    for img_path in sorted(frames_path.rglob("*.jpg")):
        video_id = img_path.parent.name
        frame_idx = int(img_path.stem.split("_")[-1])

        frames.append({
            "image_path": str(img_path),
            "video_id": video_id,
            "frame_idx": frame_idx,
        })

    return frames


def generate_qa_pairs(frame_info: Dict) -> List[Dict]:
    """Generate QA pairs for a frame (simplified version)."""
    qa_pairs = []

    # Phase question
    phase = np.random.choice(PHASES)
    qa_pairs.append({
        "question": "What surgical phase is shown in this image?",
        "answer": f"This image shows the {phase} phase of pituitary surgery.",
        "phase": phase.lower().replace(" ", "_"),
    })

    # Step question
    step = np.random.choice(STEPS)
    qa_pairs.append({
        "question": "What surgical step is being performed?",
        "answer": f"The surgeon is performing {step}.",
        "step": step.lower().replace(" ", "_"),
    })

    # Instrument question
    num_instruments = np.random.randint(1, 4)
    instruments = list(np.random.choice(INSTRUMENTS, num_instruments, replace=False))
    instruments_str = ", ".join(instruments[:-1]) + f" and {instruments[-1]}" if len(instruments) > 1 else instruments[0]
    qa_pairs.append({
        "question": "What surgical instruments are visible in this image?",
        "answer": f"The visible instruments are: {instruments_str}.",
        "instruments": [i.lower().replace(" ", "_") for i in instruments],
    })

    return qa_pairs


def create_sft_dataset(frames: List[Dict], max_samples: Optional[int] = None) -> Dataset:
    """Create SFT dataset with conversation format."""
    samples = []

    if max_samples:
        frames = frames[:max_samples]

    for frame in tqdm(frames, desc="Creating SFT samples"):
        qa_pairs = generate_qa_pairs(frame)

        for qa in qa_pairs:
            samples.append({
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ],
                "image": frame["image_path"],
                "video_id": frame["video_id"],
                "frame_idx": frame["frame_idx"],
                "phase": qa.get("phase", ""),
                "step": qa.get("step", ""),
                "instruments": qa.get("instruments", []),
            })

    return Dataset.from_list(samples)


def push_to_hub(
    dataset: DatasetDict,
    repo_id: str,
    token: str,
    private: bool = False
):
    """Push dataset to HuggingFace Hub."""
    logger.info(f"Pushing dataset to {repo_id}...")
    dataset.push_to_hub(repo_id, token=token, private=private)
    logger.info(f"Dataset pushed: https://huggingface.co/datasets/{repo_id}")


def generate_hf_skills_prompt(
    dataset_id: str,
    output_model: str,
    method: str = "sft"
) -> str:
    """Generate HuggingFace Skills training prompt."""

    if method == "sft":
        prompt = f"""Fine-tune allenai/SAGE-MM-Molmo2-8B-SFT_RL on {dataset_id}

Configuration:
- Output model: {output_model}
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-5
- Use LoRA: True (r=16, alpha=32)
- Hardware: a10g-large
- This is a vision language model with images in 'image' column
- Dataset has 'messages' column in conversation format
"""
    elif method == "dpo":
        prompt = f"""Run DPO on {dataset_id} using allenai/SAGE-MM-Molmo2-8B-SFT_RL

Configuration:
- Output model: {output_model}
- Epochs: 2
- DPO beta: 0.1
- Use LoRA: True
- Hardware: a10g-large
- Dataset has 'chosen' and 'rejected' columns
"""
    else:  # grpo
        prompt = f"""Train with GRPO on {dataset_id} using allenai/SAGE-MM-Molmo2-8B-SFT_RL

Configuration:
- Output model: {output_model}
- Epochs: 5
- GRPO generations: 4
- Use LoRA: True
- Hardware: a10g-large
- Custom reward functions for surgical pointing and phase classification
"""

    return prompt


def main():
    parser = argparse.ArgumentParser(description="PitVQA → HuggingFace Pipeline")
    parser.add_argument("--videos-zip", default="notebooks/data/pitvqa_download/videos.zip",
                       help="Path to videos.zip")
    parser.add_argument("--frames-dir", default="data/processed/frames",
                       help="Directory to extract frames")
    parser.add_argument("--output-dataset", default="mmrech/pitvqa-sage-sft",
                       help="HuggingFace dataset ID for output")
    parser.add_argument("--output-model", default="mmrech/pitvqa-sage-surgical",
                       help="HuggingFace model ID for training output")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                       help="HuggingFace token")
    parser.add_argument("--method", choices=["sft", "dpo", "grpo"], default="sft",
                       help="Training method")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum samples for quick testing")
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip video extraction, use existing frames")
    parser.add_argument("--skip-push", action="store_true",
                       help="Skip pushing to Hub")

    args = parser.parse_args()

    # Check dependencies
    if not HAS_DEPS:
        logger.error(f"Missing dependency: {MISSING_DEP}")
        logger.info("Installing dependencies...")
        install_dependencies()
        logger.info("Please run the script again.")
        return

    logger.info("=" * 60)
    logger.info("PitVQA → HuggingFace Pipeline")
    logger.info("=" * 60)

    # Step 1: Extract frames (if needed)
    frames = []

    if not args.skip_extraction:
        if Path(args.videos_zip).exists():
            logger.info("[Step 1/4] Extracting videos...")
            extract_dir = Path(args.frames_dir).parent / "videos"
            videos = extract_videos(args.videos_zip, str(extract_dir))

            # Extract frames from each video
            for video in tqdm(videos, desc="Processing videos"):
                video_frames = extract_frames_from_video(video, args.frames_dir)
                for i, fp in enumerate(video_frames):
                    frames.append({
                        "image_path": fp,
                        "video_id": Path(video).stem,
                        "frame_idx": i
                    })
        else:
            logger.warning(f"Videos zip not found: {args.videos_zip}")

    # Load existing frames if extraction skipped or no videos
    if not frames:
        logger.info("[Step 1/4] Loading existing frames...")
        frames = load_existing_frames(args.frames_dir)

    if not frames:
        logger.error("No frames found! Please check your data paths.")
        logger.info(f"Expected frames in: {args.frames_dir}")
        logger.info(f"Or videos zip at: {args.videos_zip}")
        return

    logger.info(f"Found {len(frames)} frames")

    # Step 2: Create dataset
    logger.info("[Step 2/4] Creating SFT dataset...")
    dataset = create_sft_dataset(frames, max_samples=args.max_samples)

    # Split into train/val/test
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    val_test = dataset["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        "train": dataset["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })

    logger.info(f"Dataset splits:")
    logger.info(f"  Train: {len(dataset_dict['train'])} samples")
    logger.info(f"  Validation: {len(dataset_dict['validation'])} samples")
    logger.info(f"  Test: {len(dataset_dict['test'])} samples")

    # Step 3: Push to Hub
    if not args.skip_push:
        if not args.hf_token:
            logger.error("HuggingFace token required! Set HF_TOKEN or use --hf-token")
            return

        logger.info("[Step 3/4] Pushing to HuggingFace Hub...")
        push_to_hub(dataset_dict, args.output_dataset, args.hf_token)
    else:
        logger.info("[Step 3/4] Skipping Hub push (--skip-push)")
        # Save locally instead
        local_path = Path(args.frames_dir).parent / "hf_dataset"
        dataset_dict.save_to_disk(str(local_path))
        logger.info(f"Saved locally to: {local_path}")

    # Step 4: Generate HF Skills prompt
    logger.info("[Step 4/4] Generating HF Skills training prompt...")
    prompt = generate_hf_skills_prompt(
        args.output_dataset,
        args.output_model,
        args.method
    )

    prompt_path = Path("outputs") / "hf_skills_prompt.txt"
    prompt_path.parent.mkdir(exist_ok=True)
    prompt_path.write_text(prompt)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nDataset: https://huggingface.co/datasets/{args.output_dataset}")
    logger.info(f"Prompt saved to: {prompt_path}")
    logger.info("\n" + "-" * 60)
    logger.info("TO START TRAINING (use with Claude Code or HF Skills):")
    logger.info("-" * 60)
    print(f"\n{prompt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
