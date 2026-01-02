"""
PitVQA Dataset Builder Module

Creates HuggingFace Datasets from processed surgical video frames and QA pairs
for Visual Question Answering on pituitary surgery workflows.

Features:
- Video-level train/val/test splits (prevents data leakage)
- Efficient Parquet storage for metadata
- HuggingFace Hub integration
- Automatic dataset card generation with statistics
"""

import json
import logging
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

try:
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Value,
        ClassLabel,
        Image as ImageFeature,
    )
    from huggingface_hub import HfApi, create_repo
except ImportError:
    raise ImportError(
        "Please install the datasets and huggingface_hub libraries: "
        "pip install datasets huggingface_hub"
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Question type categories for PitVQA
QUESTION_TYPES = [
    "phase",           # Surgical phase identification
    "step",            # Surgical step within a phase
    "instrument",      # Instrument identification/counting
    "position",        # Anatomical position/location
    "yes_no",          # Binary questions
    "operation_note",  # Operative notes/descriptions
]


class DatasetBuilder:
    """
    Builds HuggingFace Datasets from processed frames and QA pairs.

    Handles:
    - Loading frames and QA annotations
    - Creating video-level splits (prevents data leakage)
    - Building efficient HuggingFace Dataset structures
    - Saving locally and pushing to HuggingFace Hub
    - Generating comprehensive dataset cards

    Example:
        >>> builder = DatasetBuilder()
        >>> dataset = builder.build_dataset("frames/", "qa_pairs.json")
        >>> splits = builder.create_splits(dataset)
        >>> builder.save_to_disk(splits, "output/pitvqa_dataset")
        >>> builder.push_to_hub(splits, "username/pitvqa-dataset", token="hf_xxx")
    """

    def __init__(
        self,
        question_types: Optional[List[str]] = None,
        random_seed: int = 42,
        image_column_mode: str = "path",  # "path" or "bytes"
    ):
        """
        Initialize the DatasetBuilder.

        Args:
            question_types: List of valid question types. Defaults to QUESTION_TYPES.
            random_seed: Random seed for reproducible splits.
            image_column_mode: How to store images - "path" for file paths (efficient),
                             "bytes" for embedded images (portable but larger).
        """
        self.question_types = question_types or QUESTION_TYPES
        self.random_seed = random_seed
        self.image_column_mode = image_column_mode
        random.seed(random_seed)

        logger.info(f"DatasetBuilder initialized with seed={random_seed}, "
                   f"image_mode={image_column_mode}")

    def build_dataset(
        self,
        frames_dir: Union[str, Path],
        qa_pairs_file: Union[str, Path],
        validate_images: bool = True,
    ) -> Dataset:
        """
        Build a HuggingFace Dataset from frames directory and QA pairs.

        Args:
            frames_dir: Directory containing extracted frames organized by video.
                       Expected structure: frames_dir/video_id/frame_xxx.png
            qa_pairs_file: JSON file containing QA pairs with frame references.
            validate_images: Whether to validate that referenced images exist.

        Returns:
            HuggingFace Dataset with columns: image, question, answer,
            question_type, video_id, frame_id

        Raises:
            FileNotFoundError: If frames_dir or qa_pairs_file don't exist.
            ValueError: If QA pairs file is malformed.
        """
        frames_dir = Path(frames_dir)
        qa_pairs_file = Path(qa_pairs_file)

        # Validate inputs
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        if not qa_pairs_file.exists():
            raise FileNotFoundError(f"QA pairs file not found: {qa_pairs_file}")

        logger.info(f"Loading QA pairs from {qa_pairs_file}")
        qa_pairs = self._load_qa_pairs(qa_pairs_file)
        logger.info(f"Loaded {len(qa_pairs)} QA pairs")

        # Process and validate data
        processed_data = self._process_qa_pairs(
            qa_pairs, frames_dir, validate_images
        )

        # Create dataset
        dataset = self._create_dataset(processed_data)

        logger.info(f"Built dataset with {len(dataset)} samples")
        self._log_dataset_stats(dataset)

        return dataset

    def _load_qa_pairs(self, qa_pairs_file: Path) -> List[Dict[str, Any]]:
        """Load QA pairs from JSON file."""
        with open(qa_pairs_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict with 'annotations' key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if "annotations" in data:
                return data["annotations"]
            elif "qa_pairs" in data:
                return data["qa_pairs"]
            elif "data" in data:
                return data["data"]
            else:
                raise ValueError(
                    "QA pairs JSON must be a list or dict with "
                    "'annotations', 'qa_pairs', or 'data' key"
                )
        else:
            raise ValueError(f"Unexpected QA pairs format: {type(data)}")

    def _process_qa_pairs(
        self,
        qa_pairs: List[Dict[str, Any]],
        frames_dir: Path,
        validate_images: bool,
    ) -> Dict[str, List]:
        """Process QA pairs and prepare data for dataset creation."""
        processed = {
            "image": [],
            "question": [],
            "answer": [],
            "question_type": [],
            "video_id": [],
            "frame_id": [],
        }

        missing_images = 0
        invalid_types = 0

        for idx, qa in enumerate(qa_pairs):
            # Extract fields with flexible key names
            video_id = qa.get("video_id", qa.get("video", "unknown"))
            frame_id = qa.get("frame_id", qa.get("frame", qa.get("image_id", "")))
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            q_type = qa.get("question_type", qa.get("type", "unknown"))

            # Normalize question type
            q_type = q_type.lower().replace(" ", "_").replace("-", "_")
            if q_type not in self.question_types:
                logger.debug(f"Unknown question type '{q_type}' at index {idx}")
                invalid_types += 1
                q_type = "unknown"

            # Resolve image path
            image_path = self._resolve_image_path(
                frames_dir, video_id, frame_id, qa
            )

            if validate_images and image_path and not Path(image_path).exists():
                logger.debug(f"Missing image: {image_path}")
                missing_images += 1
                continue

            # Store data
            processed["image"].append(str(image_path) if image_path else None)
            processed["question"].append(str(question))
            processed["answer"].append(str(answer))
            processed["question_type"].append(q_type)
            processed["video_id"].append(str(video_id))
            processed["frame_id"].append(str(frame_id))

        if missing_images > 0:
            logger.warning(f"Skipped {missing_images} samples with missing images")
        if invalid_types > 0:
            logger.warning(f"Found {invalid_types} samples with unknown question types")

        return processed

    def _resolve_image_path(
        self,
        frames_dir: Path,
        video_id: str,
        frame_id: str,
        qa: Dict[str, Any],
    ) -> Optional[Path]:
        """Resolve the full path to an image file."""
        # Try explicit image_path first
        if "image_path" in qa:
            explicit_path = Path(qa["image_path"])
            if explicit_path.is_absolute():
                return explicit_path
            return frames_dir / explicit_path

        # Try common naming patterns
        patterns = [
            frames_dir / video_id / frame_id,
            frames_dir / video_id / f"{frame_id}.png",
            frames_dir / video_id / f"{frame_id}.jpg",
            frames_dir / video_id / f"frame_{frame_id}.png",
            frames_dir / video_id / f"frame_{frame_id}.jpg",
            frames_dir / f"{video_id}_{frame_id}.png",
            frames_dir / f"{video_id}_{frame_id}.jpg",
        ]

        for pattern in patterns:
            if pattern.exists():
                return pattern

        # Return first pattern as default (even if doesn't exist)
        return patterns[0] if patterns else None

    def _create_dataset(self, processed_data: Dict[str, List]) -> Dataset:
        """Create HuggingFace Dataset from processed data."""
        # Define features
        if self.image_column_mode == "bytes":
            features = Features({
                "image": ImageFeature(),
                "question": Value("string"),
                "answer": Value("string"),
                "question_type": Value("string"),
                "video_id": Value("string"),
                "frame_id": Value("string"),
            })
        else:
            features = Features({
                "image": Value("string"),  # Store as path
                "question": Value("string"),
                "answer": Value("string"),
                "question_type": Value("string"),
                "video_id": Value("string"),
                "frame_id": Value("string"),
            })

        dataset = Dataset.from_dict(processed_data, features=features)
        return dataset

    def create_splits(
        self,
        dataset: Dataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: Optional[float] = None,
    ) -> DatasetDict:
        """
        Create train/val/test splits at the VIDEO level to prevent data leakage.

        Important: Splits are done by video_id, not by individual samples.
        This ensures frames from the same video don't appear in both
        training and test sets, preventing data leakage.

        Args:
            dataset: HuggingFace Dataset to split.
            train_ratio: Fraction of videos for training (default: 0.8).
            val_ratio: Fraction of videos for validation (default: 0.1).
            test_ratio: Fraction of videos for testing. If None, computed as
                       1 - train_ratio - val_ratio (default: 0.1).

        Returns:
            DatasetDict with 'train', 'validation', and 'test' splits.

        Raises:
            ValueError: If ratios don't sum to 1.0 or are invalid.
        """
        if test_ratio is None:
            test_ratio = 1.0 - train_ratio - val_ratio

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
            raise ValueError("Split ratios cannot be negative")

        # Get unique video IDs
        video_ids = list(set(dataset["video_id"]))
        random.shuffle(video_ids)

        n_videos = len(video_ids)
        n_train = int(n_videos * train_ratio)
        n_val = int(n_videos * val_ratio)

        train_videos = set(video_ids[:n_train])
        val_videos = set(video_ids[n_train:n_train + n_val])
        test_videos = set(video_ids[n_train + n_val:])

        logger.info(f"Split {n_videos} videos: {len(train_videos)} train, "
                   f"{len(val_videos)} val, {len(test_videos)} test")

        # Create split assignment
        def assign_split(example):
            video = example["video_id"]
            if video in train_videos:
                return {"split": "train"}
            elif video in val_videos:
                return {"split": "validation"}
            else:
                return {"split": "test"}

        # Add split column
        dataset = dataset.map(assign_split)

        # Create DatasetDict
        splits = DatasetDict({
            "train": dataset.filter(lambda x: x["split"] == "train"),
            "validation": dataset.filter(lambda x: x["split"] == "validation"),
            "test": dataset.filter(lambda x: x["split"] == "test"),
        })

        # Remove the temporary split column
        for split_name in splits:
            splits[split_name] = splits[split_name].remove_columns(["split"])

        logger.info(f"Created splits - train: {len(splits['train'])}, "
                   f"val: {len(splits['validation'])}, test: {len(splits['test'])}")

        return splits

    def save_to_disk(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_dir: Union[str, Path],
        save_format: str = "parquet",
    ) -> Path:
        """
        Save dataset to disk.

        Args:
            dataset: Dataset or DatasetDict to save.
            output_dir: Directory to save the dataset.
            save_format: Format to use - "parquet" (recommended) or "arrow".

        Returns:
            Path to the saved dataset directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving dataset to {output_dir}")

        if save_format == "parquet":
            if isinstance(dataset, DatasetDict):
                for split_name, split_data in dataset.items():
                    split_path = output_dir / f"{split_name}.parquet"
                    split_data.to_parquet(str(split_path))
                    logger.info(f"Saved {split_name} split to {split_path}")
            else:
                parquet_path = output_dir / "data.parquet"
                dataset.to_parquet(str(parquet_path))
                logger.info(f"Saved dataset to {parquet_path}")
        else:
            # Default Arrow format
            dataset.save_to_disk(str(output_dir))
            logger.info(f"Saved dataset in Arrow format to {output_dir}")

        # Generate and save dataset card
        dataset_card = self.generate_dataset_card(dataset)
        readme_path = output_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(dataset_card)
        logger.info(f"Generated dataset card at {readme_path}")

        return output_dir

    def push_to_hub(
        self,
        dataset: Union[Dataset, DatasetDict],
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> str:
        """
        Push dataset to HuggingFace Hub.

        Args:
            dataset: Dataset or DatasetDict to push.
            repo_id: Repository ID on HuggingFace Hub (e.g., "username/dataset-name").
            token: HuggingFace API token. If None, uses cached token.
            private: Whether the repository should be private.
            commit_message: Custom commit message.

        Returns:
            URL of the pushed dataset repository.
        """
        logger.info(f"Pushing dataset to HuggingFace Hub: {repo_id}")

        # Create repo if it doesn't exist
        api = HfApi()
        try:
            create_repo(
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
                private=private,
                exist_ok=True,
            )
        except Exception as e:
            logger.warning(f"Could not create repo (may already exist): {e}")

        # Generate dataset card
        dataset_card = self.generate_dataset_card(dataset)

        # Push dataset
        if commit_message is None:
            commit_message = f"Upload PitVQA dataset - {datetime.now().isoformat()}"

        dataset.push_to_hub(
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )

        # Upload README
        api.upload_file(
            path_or_fileobj=dataset_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Update dataset card",
        )

        hub_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Dataset pushed successfully: {hub_url}")

        return hub_url

    def generate_dataset_card(
        self,
        dataset: Union[Dataset, DatasetDict],
    ) -> str:
        """
        Generate a comprehensive dataset card (README.md) with statistics.

        Args:
            dataset: Dataset or DatasetDict to generate card for.

        Returns:
            Markdown string for the dataset card.
        """
        stats = self._compute_statistics(dataset)

        card = f"""---
language:
- en
license: cc-by-nc-4.0
task_categories:
- visual-question-answering
- image-classification
tags:
- medical
- surgical
- pituitary
- vqa
- surgery
- healthcare
pretty_name: PitVQA - Pituitary Surgery Visual Question Answering
size_categories:
- {self._get_size_category(stats['total_samples'])}
---

# PitVQA Dataset

## Dataset Description

Visual Question Answering dataset for pituitary surgery workflow analysis.
Contains surgical video frames paired with questions and answers about
surgical phases, steps, instruments, and anatomical positions.

### Dataset Summary

- **Total Samples**: {stats['total_samples']:,}
- **Total Videos**: {stats['total_videos']}
- **Question Types**: {len(stats['question_type_distribution'])}
- **Created**: {datetime.now().strftime('%Y-%m-%d')}

### Supported Tasks

- Visual Question Answering (VQA)
- Surgical Phase Recognition
- Instrument Detection
- Surgical Workflow Analysis

## Dataset Structure

### Data Instances

Each instance contains:
- `image`: Path to surgical frame image
- `question`: Natural language question about the frame
- `answer`: Ground truth answer
- `question_type`: Category of question (phase, step, instrument, position, yes_no, operation_note)
- `video_id`: Source video identifier
- `frame_id`: Frame identifier within the video

### Data Splits

| Split | Samples | Videos | Percentage |
|-------|---------|--------|------------|
"""

        if isinstance(dataset, DatasetDict):
            for split_name in ["train", "validation", "test"]:
                if split_name in dataset:
                    split_stats = self._compute_split_stats(dataset[split_name])
                    pct = (split_stats['samples'] / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
                    card += f"| {split_name} | {split_stats['samples']:,} | {split_stats['videos']} | {pct:.1f}% |\n"
        else:
            card += f"| all | {stats['total_samples']:,} | {stats['total_videos']} | 100% |\n"

        card += f"""
### Question Type Distribution

| Type | Count | Percentage |
|------|-------|------------|
"""

        for q_type, count in sorted(stats['question_type_distribution'].items(), key=lambda x: -x[1]):
            pct = (count / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0
            card += f"| {q_type} | {count:,} | {pct:.1f}% |\n"

        card += f"""
## Dataset Creation

### Curation Rationale

This dataset was created to support research in surgical AI, specifically
for understanding and analyzing pituitary surgery workflows through
visual question answering.

### Source Data

Frames extracted from pituitary surgery procedure videos with expert
annotations for questions and answers.

### Annotations

Questions and answers were created by medical experts with knowledge
of surgical procedures and anatomy.

## Considerations for Using the Data

### Social Impact

This dataset is intended for research purposes to improve surgical
AI systems that could assist in surgical training, procedure
documentation, and workflow analysis.

### Limitations

- Dataset may not cover all possible surgical scenarios
- Question types are limited to predefined categories
- Images are from a specific surgical context (pituitary surgery)

## Additional Information

### Dataset Curators

PitVQA Dataset Team

### Licensing Information

This dataset is released under the CC BY-NC 4.0 license.

### Citation Information

```bibtex
@dataset{{pitvqa_dataset,
    title = {{PitVQA: Pituitary Surgery Visual Question Answering Dataset}},
    year = {{{datetime.now().year}}},
    publisher = {{HuggingFace}},
}}
```

## Statistics Summary

- Total QA pairs: {stats['total_samples']:,}
- Unique videos: {stats['total_videos']}
- Unique frames: {stats['unique_frames']}
- Average questions per frame: {stats['avg_questions_per_frame']:.2f}
- Most common question type: {stats['most_common_type']}

---
*Dataset card generated automatically by PitVQA DatasetBuilder*
"""

        return card

    def _compute_statistics(
        self,
        dataset: Union[Dataset, DatasetDict],
    ) -> Dict[str, Any]:
        """Compute comprehensive statistics for a dataset."""
        if isinstance(dataset, DatasetDict):
            # Combine all splits for overall statistics
            all_data = {
                "video_id": [],
                "frame_id": [],
                "question_type": [],
            }
            for split in dataset.values():
                all_data["video_id"].extend(split["video_id"])
                all_data["frame_id"].extend(split["frame_id"])
                all_data["question_type"].extend(split["question_type"])
        else:
            all_data = {
                "video_id": dataset["video_id"],
                "frame_id": dataset["frame_id"],
                "question_type": dataset["question_type"],
            }

        total_samples = len(all_data["video_id"])
        unique_videos = set(all_data["video_id"])
        unique_frames = set(zip(all_data["video_id"], all_data["frame_id"]))
        q_type_dist = Counter(all_data["question_type"])

        return {
            "total_samples": total_samples,
            "total_videos": len(unique_videos),
            "unique_frames": len(unique_frames),
            "question_type_distribution": dict(q_type_dist),
            "avg_questions_per_frame": total_samples / len(unique_frames) if unique_frames else 0,
            "most_common_type": q_type_dist.most_common(1)[0][0] if q_type_dist else "N/A",
        }

    def _compute_split_stats(self, split: Dataset) -> Dict[str, int]:
        """Compute statistics for a single split."""
        return {
            "samples": len(split),
            "videos": len(set(split["video_id"])),
        }

    def _get_size_category(self, n: int) -> str:
        """Get HuggingFace size category string."""
        if n < 1000:
            return "n<1K"
        elif n < 10000:
            return "1K<n<10K"
        elif n < 100000:
            return "10K<n<100K"
        elif n < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"

    def _log_dataset_stats(self, dataset: Dataset) -> None:
        """Log dataset statistics."""
        stats = self._compute_statistics(dataset)
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Total samples: {stats['total_samples']:,}")
        logger.info(f"  Unique videos: {stats['total_videos']}")
        logger.info(f"  Unique frames: {stats['unique_frames']}")
        logger.info(f"  Question types: {list(stats['question_type_distribution'].keys())}")


def load_dataset_from_disk(
    dataset_dir: Union[str, Path],
    format: str = "auto",
) -> Union[Dataset, DatasetDict]:
    """
    Load a saved PitVQA dataset from disk.

    Args:
        dataset_dir: Directory containing the saved dataset.
        format: Format to load - "auto" (detect), "parquet", or "arrow".

    Returns:
        Loaded Dataset or DatasetDict.
    """
    from datasets import load_from_disk, load_dataset as hf_load_dataset

    dataset_dir = Path(dataset_dir)

    if format == "auto":
        # Check for Parquet files
        parquet_files = list(dataset_dir.glob("*.parquet"))
        if parquet_files:
            format = "parquet"
        else:
            format = "arrow"

    if format == "parquet":
        import pandas as pd
        splits = {}
        for parquet_file in dataset_dir.glob("*.parquet"):
            split_name = parquet_file.stem
            # Handle empty parquet files gracefully
            try:
                df = pd.read_parquet(str(parquet_file))
                if len(df) > 0:
                    splits[split_name] = Dataset.from_pandas(df)
                else:
                    logger.warning(f"Skipping empty split: {split_name}")
            except Exception as e:
                logger.warning(f"Could not load {parquet_file}: {e}")

        if len(splits) == 1 and "data" in splits:
            return splits["data"]
        return DatasetDict(splits)
    else:
        return load_from_disk(str(dataset_dir))


# Example usage and CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build HuggingFace Dataset from PitVQA frames and QA pairs"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing extracted frames",
    )
    parser.add_argument(
        "--qa-pairs",
        type=str,
        required=True,
        help="JSON file containing QA pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID to push to (e.g., 'username/dataset-name')",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "arrow"],
        default="parquet",
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    # Get token from args or environment
    token = args.hf_token or os.environ.get("HF_TOKEN")

    # Build dataset
    print("=" * 60)
    print("PitVQA Dataset Builder")
    print("=" * 60)

    builder = DatasetBuilder(random_seed=args.seed)

    print(f"\nBuilding dataset from:")
    print(f"  Frames: {args.frames_dir}")
    print(f"  QA Pairs: {args.qa_pairs}")

    dataset = builder.build_dataset(args.frames_dir, args.qa_pairs)

    print(f"\nCreating splits (train={args.train_ratio}, val={args.val_ratio})...")
    splits = builder.create_splits(
        dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(f"\nSaving to: {args.output_dir}")
    builder.save_to_disk(splits, args.output_dir, save_format=args.format)

    if args.push_to_hub:
        if not token:
            print("\nWarning: No HuggingFace token provided. Skipping Hub push.")
            print("Set HF_TOKEN environment variable or use --hf-token argument.")
        else:
            print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
            url = builder.push_to_hub(splits, args.push_to_hub, token=token)
            print(f"Dataset available at: {url}")

    print("\nDone!")
    print("=" * 60)
