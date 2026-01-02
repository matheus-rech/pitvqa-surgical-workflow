#!/usr/bin/env python3
"""
PitVQA Agent 1: Data Preparation Pipeline

Main orchestration script for the complete PitVQA data preparation workflow.
Coordinates frame extraction, QA generation, dataset building, and validation.

This is the main entry point for the Agent 1 data preparation pipeline.

Usage:
    python pitvqa_agent1_data_prep.py \
        --video-dir data/raw/pitvqa/videos \
        --annotation-dir data/raw/pitvqa/annotations \
        --output-dir data/processed \
        --fps 1 \
        --blur-threshold 100

Target outputs:
    - ~109,173 frames from 25 videos
    - ~884,242 QA pairs
    - HuggingFace dataset ready for training

Author: PitVQA Team
Version: 1.0.0
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from .frame_extractor import FrameExtractor
    from .qa_generator import QAGenerator
    from .dataset_builder import DatasetBuilder
    from .validators import DataValidator
except ImportError:
    # Allow running as script
    from frame_extractor import FrameExtractor
    from qa_generator import QAGenerator
    from dataset_builder import DatasetBuilder
    from validators import DataValidator


# Configure logging
def setup_logging(output_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        output_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("pitvqa_agent1")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = output_dir / f"agent1_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


@dataclass
class PipelineConfig:
    """Configuration for the data preparation pipeline."""

    # Input paths
    video_dir: Path
    annotation_dir: Path
    output_dir: Path

    # Frame extraction settings
    fps: float = 1.0
    blur_threshold: float = 100.0
    image_format: str = "png"

    # HuggingFace settings
    push_to_hub: Optional[str] = None
    hf_token: Optional[str] = None

    # Skip flags
    skip_extraction: bool = False
    skip_qa_generation: bool = False
    skip_validation: bool = False

    # Processing settings
    num_workers: int = 4
    batch_size: int = 100

    # Checkpoint settings
    checkpoint_dir: Optional[Path] = None
    resume_from_checkpoint: bool = True

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.video_dir = Path(self.video_dir)
        self.annotation_dir = Path(self.annotation_dir)
        self.output_dir = Path(self.output_dir)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output_dir / ".checkpoints"
        else:
            self.checkpoint_dir = Path(self.checkpoint_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "video_dir": str(self.video_dir),
            "annotation_dir": str(self.annotation_dir),
            "output_dir": str(self.output_dir),
            "fps": self.fps,
            "blur_threshold": self.blur_threshold,
            "image_format": self.image_format,
            "push_to_hub": self.push_to_hub,
            "skip_extraction": self.skip_extraction,
            "skip_qa_generation": self.skip_qa_generation,
            "skip_validation": self.skip_validation,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
        }


@dataclass
class PipelineStats:
    """Statistics tracking for the pipeline."""

    # Frame extraction stats
    total_videos: int = 0
    processed_videos: int = 0
    total_frames_extracted: int = 0
    blurry_frames_removed: int = 0
    final_frame_count: int = 0

    # QA generation stats
    total_annotations_loaded: int = 0
    qa_pairs_generated: int = 0
    qa_pairs_by_type: Dict[str, int] = field(default_factory=dict)

    # Dataset stats
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Validation stats
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Timing stats
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_durations: Dict[str, float] = field(default_factory=dict)

    def get_duration(self) -> Optional[timedelta]:
        """Get total pipeline duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for serialization."""
        return {
            "total_videos": self.total_videos,
            "processed_videos": self.processed_videos,
            "total_frames_extracted": self.total_frames_extracted,
            "blurry_frames_removed": self.blurry_frames_removed,
            "final_frame_count": self.final_frame_count,
            "total_annotations_loaded": self.total_annotations_loaded,
            "qa_pairs_generated": self.qa_pairs_generated,
            "qa_pairs_by_type": self.qa_pairs_by_type,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "step_durations": self.step_durations,
            "total_duration_seconds": self.get_duration().total_seconds() if self.get_duration() else None,
        }


@dataclass
class Checkpoint:
    """Checkpoint data for pipeline resumption."""

    step: str
    completed_videos: List[str] = field(default_factory=list)
    extracted_frames_dir: Optional[str] = None
    qa_pairs_file: Optional[str] = None
    dataset_dir: Optional[str] = None
    stats: Optional[Dict] = None
    timestamp: Optional[str] = None

    def save(self, checkpoint_dir: Path) -> None:
        """Save checkpoint to file."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "checkpoint.json"

        self.timestamp = datetime.now().isoformat()

        with open(checkpoint_file, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, checkpoint_dir: Path) -> Optional["Checkpoint"]:
        """Load checkpoint from file."""
        checkpoint_file = checkpoint_dir / "checkpoint.json"

        if not checkpoint_file.exists():
            return None

        with open(checkpoint_file, "r") as f:
            data = json.load(f)

        return cls(**data)


class PitVQADataPrep:
    """
    Main orchestration class for the PitVQA data preparation pipeline.

    Coordinates the entire workflow:
    1. Frame extraction from videos
    2. Blur detection and filtering
    3. Annotation loading and processing
    4. QA pair generation
    5. HuggingFace dataset building
    6. Output validation
    7. Dataset saving/pushing

    Supports checkpointing for resumable execution and comprehensive
    progress tracking with detailed logging.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the data preparation pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.stats = PipelineStats()
        self.checkpoint = None

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.config.output_dir)

        # Initialize components
        self.frame_extractor = None
        self.qa_generator = None
        self.dataset_builder = None
        self.validator = None

        self.logger.info("=" * 60)
        self.logger.info("PitVQA Agent 1: Data Preparation Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Video directory: {self.config.video_dir}")
        self.logger.info(f"Annotation directory: {self.config.annotation_dir}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"FPS: {self.config.fps}")
        self.logger.info(f"Blur threshold: {self.config.blur_threshold}")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")

        # Frame extractor - uses extraction_fps and blur_threshold parameters
        self.frame_extractor = FrameExtractor(
            extraction_fps=self.config.fps,
            blur_threshold=self.config.blur_threshold
        )

        # QA generator - uses qa_per_frame and balance_questions
        self.qa_generator = QAGenerator(
            qa_per_frame=8,  # Target 8 QA pairs per frame
            balance_questions=True,
            seed=42
        )

        # Dataset builder - uses random_seed and image_column_mode
        self.dataset_builder = DatasetBuilder(
            random_seed=42,
            image_column_mode="path"  # Store paths for efficiency
        )

        # Validator - uses specs and strict_mode
        self.validator = DataValidator(
            strict_mode=False,
            log_progress=True
        )

        # Store paths for use by components
        self.frames_dir = self.config.output_dir / "frames"
        self.qa_output_dir = self.config.output_dir / "qa_pairs"
        self.dataset_dir = self.config.output_dir / "dataset"

        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.qa_output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("All components initialized successfully")

    def _load_checkpoint(self) -> bool:
        """
        Load existing checkpoint if available.

        Returns:
            True if checkpoint was loaded, False otherwise
        """
        if not self.config.resume_from_checkpoint:
            return False

        self.checkpoint = Checkpoint.load(self.config.checkpoint_dir)

        if self.checkpoint:
            self.logger.info(f"Resuming from checkpoint: step '{self.checkpoint.step}'")
            self.logger.info(f"Checkpoint timestamp: {self.checkpoint.timestamp}")

            if self.checkpoint.stats:
                # Restore stats
                for key, value in self.checkpoint.stats.items():
                    if hasattr(self.stats, key):
                        setattr(self.stats, key, value)

            return True

        return False

    def _save_checkpoint(self, step: str, **kwargs) -> None:
        """
        Save checkpoint after a step completes.

        Args:
            step: Name of the completed step
            **kwargs: Additional checkpoint data
        """
        self.checkpoint = Checkpoint(
            step=step,
            stats=self.stats.to_dict(),
            **kwargs
        )
        self.checkpoint.save(self.config.checkpoint_dir)
        self.logger.debug(f"Checkpoint saved: {step}")

    def _step_timer(self, step_name: str):
        """
        Context manager for timing pipeline steps.

        Args:
            step_name: Name of the step being timed
        """
        class StepTimer:
            def __init__(self, pipeline, name):
                self.pipeline = pipeline
                self.name = name
                self.start = None

            def __enter__(self):
                self.start = time.time()
                self.pipeline.logger.info(f"Starting step: {self.name}")
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start
                self.pipeline.stats.step_durations[self.name] = duration
                self.pipeline.logger.info(
                    f"Completed step: {self.name} ({duration:.2f}s)"
                )
                return False

        return StepTimer(self, step_name)

    def step1_extract_frames(self) -> Tuple[int, List[str]]:
        """
        Step 1: Extract frames from all videos.

        Returns:
            Tuple of (frame count, list of frame paths)
        """
        if self.config.skip_extraction:
            self.logger.info("Skipping frame extraction (--skip-extraction flag)")

            # Load existing frames
            if self.frames_dir.exists():
                frame_paths = list(self.frames_dir.rglob("*.png"))
                frame_paths.extend(self.frames_dir.rglob("*.jpg"))
                self.stats.final_frame_count = len(frame_paths)
                return len(frame_paths), [str(p) for p in frame_paths]

            return 0, []

        with self._step_timer("frame_extraction"):
            # Get list of videos - use FrameExtractor's VIDEO_EXTENSIONS
            video_extensions = FrameExtractor.VIDEO_EXTENSIONS
            video_files = []

            if self.config.video_dir.exists():
                for ext in video_extensions:
                    video_files.extend(self.config.video_dir.glob(f"*{ext}"))
                    video_files.extend(self.config.video_dir.glob(f"*{ext.upper()}"))

            video_files = sorted(set(video_files))
            self.stats.total_videos = len(video_files)
            self.logger.info(f"Found {len(video_files)} video files")

            if not video_files:
                self.logger.warning("No video files found in video directory")
                return 0, []

            # Check checkpoint for already processed videos
            completed_videos = set()
            if self.checkpoint and self.checkpoint.step == "frame_extraction":
                completed_videos = set(self.checkpoint.completed_videos)
                self.logger.info(
                    f"Resuming: {len(completed_videos)} videos already processed"
                )

            all_frame_paths = []
            total_extracted = 0
            total_blurry = 0

            for i, video_path in enumerate(video_files, 1):
                if video_path.name in completed_videos:
                    self.logger.debug(f"Skipping already processed: {video_path.name}")
                    continue

                self.logger.info(
                    f"Processing video {i}/{len(video_files)}: {video_path.name}"
                )

                try:
                    # Extract frames using the FrameExtractor API
                    video_id = video_path.stem
                    video_output_dir = self.frames_dir / video_id

                    result = self.frame_extractor.extract_from_video(
                        video_path=video_path,
                        output_dir=video_output_dir,
                        filter_blur=True,
                        video_id=video_id,
                        show_progress=True
                    )

                    # Collect frame paths from output directory
                    video_frames = list(video_output_dir.glob("*.png"))
                    video_frames.extend(video_output_dir.glob("*.jpg"))
                    all_frame_paths.extend([str(p) for p in video_frames])

                    total_extracted += result['frames_sampled']
                    total_blurry += result['frames_blurry']

                    self.stats.processed_videos += 1
                    completed_videos.add(video_path.name)

                    self.logger.info(
                        f"  Sampled: {result['frames_sampled']}, "
                        f"Blurry removed: {result['frames_blurry']}, "
                        f"Saved: {result['frames_saved']}"
                    )

                    # Save checkpoint periodically
                    if self.stats.processed_videos % 5 == 0:
                        self._save_checkpoint(
                            "frame_extraction",
                            completed_videos=list(completed_videos),
                            extracted_frames_dir=str(self.frames_dir)
                        )

                except Exception as e:
                    self.logger.error(f"Error processing {video_path.name}: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    continue

            self.stats.total_frames_extracted = total_extracted
            self.stats.blurry_frames_removed = total_blurry
            self.stats.final_frame_count = len(all_frame_paths)

            # Save final checkpoint
            self._save_checkpoint(
                "frame_extraction_complete",
                completed_videos=list(completed_videos),
                extracted_frames_dir=str(self.frames_dir)
            )

            self.logger.info(f"Frame extraction complete:")
            self.logger.info(f"  Total sampled: {total_extracted}")
            self.logger.info(f"  Blurry removed: {total_blurry}")
            self.logger.info(f"  Final count: {len(all_frame_paths)}")

            return len(all_frame_paths), all_frame_paths

    def step2_load_annotations(self) -> Dict[str, Any]:
        """
        Step 2: Load and process annotations.

        Returns:
            Dictionary of loaded annotations
        """
        with self._step_timer("annotation_loading"):
            # Find annotation files
            annotation_files = list(self.config.annotation_dir.glob("*.json"))
            annotation_files.extend(self.config.annotation_dir.glob("*.csv"))

            self.logger.info(f"Found {len(annotation_files)} annotation files")

            all_annotations = {}
            total_loaded = 0

            for ann_file in annotation_files:
                self.logger.info(f"Loading: {ann_file.name}")

                try:
                    if ann_file.suffix == ".json":
                        with open(ann_file, "r") as f:
                            data = json.load(f)

                        if isinstance(data, list):
                            annotations = data
                        elif isinstance(data, dict):
                            annotations = data.get("annotations", data.get("data", [data]))
                        else:
                            annotations = [data]

                    elif ann_file.suffix == ".csv":
                        import pandas as pd
                        df = pd.read_csv(ann_file)
                        annotations = df.to_dict("records")

                    else:
                        self.logger.warning(f"Unsupported format: {ann_file.suffix}")
                        continue

                    all_annotations[ann_file.stem] = annotations
                    total_loaded += len(annotations)

                    self.logger.info(f"  Loaded {len(annotations)} annotations")

                except Exception as e:
                    self.logger.error(f"Error loading {ann_file.name}: {e}")

            self.stats.total_annotations_loaded = total_loaded
            self.logger.info(f"Total annotations loaded: {total_loaded}")

            return all_annotations

    def step3_generate_qa_pairs(self, annotations: Dict[str, Any]) -> List[Dict]:
        """
        Step 3: Generate QA pairs from annotations.

        Args:
            annotations: Loaded annotation data (frame_id -> annotation dict)

        Returns:
            List of generated QA pairs
        """
        if self.config.skip_qa_generation:
            self.logger.info("Skipping QA generation (--skip-qa-generation flag)")

            # Load existing QA pairs
            qa_file = self.qa_output_dir / "qa_pairs.json"
            if qa_file.exists():
                with open(qa_file, "r") as f:
                    qa_pairs = json.load(f)
                self.stats.qa_pairs_generated = len(qa_pairs)
                return qa_pairs

            return []

        with self._step_timer("qa_generation"):
            # Use the QAGenerator's generate_all_qa_pairs method
            qa_output_file = self.qa_output_dir / "qa_pairs.json"

            qa_pairs = self.qa_generator.generate_all_qa_pairs(
                frames_dir=self.frames_dir,
                annotation_dir=self.config.annotation_dir,
                output_file=qa_output_file
            )

            self.stats.qa_pairs_generated = len(qa_pairs)

            # Count by type
            type_counts = {}
            for qa in qa_pairs:
                qtype = qa.get("question_type", "unknown")
                type_counts[qtype] = type_counts.get(qtype, 0) + 1

            self.stats.qa_pairs_by_type = type_counts

            # Get statistics from generator
            gen_stats = self.qa_generator.get_statistics()

            self._save_checkpoint(
                "qa_generation_complete",
                qa_pairs_file=str(qa_output_file)
            )

            self.logger.info(f"Generated {len(qa_pairs)} QA pairs")
            self.logger.info(f"  Unique frames: {gen_stats.get('total_frames', 0)}")
            self.logger.info(f"  Avg QA per frame: {gen_stats.get('avg_qa_per_frame', 0):.2f}")

            for qtype, count in sorted(type_counts.items()):
                self.logger.info(f"  {qtype}: {count}")

            return qa_pairs

    def step4_build_dataset(
        self,
        frame_paths: List[str],
        qa_pairs: List[Dict]
    ) -> Any:
        """
        Step 4: Build HuggingFace dataset.

        Args:
            frame_paths: List of extracted frame paths
            qa_pairs: List of QA pairs

        Returns:
            HuggingFace DatasetDict
        """
        with self._step_timer("dataset_building"):
            # Save QA pairs to file for dataset builder
            qa_pairs_file = self.qa_output_dir / "qa_pairs.json"
            if not qa_pairs_file.exists() and qa_pairs:
                with open(qa_pairs_file, "w") as f:
                    json.dump(qa_pairs, f, indent=2)

            # Build dataset using DatasetBuilder API
            dataset = self.dataset_builder.build_dataset(
                frames_dir=self.frames_dir,
                qa_pairs_file=qa_pairs_file,
                validate_images=True
            )

            # Create splits
            splits = self.dataset_builder.create_splits(
                dataset,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1
            )

            # Update stats
            if hasattr(splits, "keys"):
                if "train" in splits:
                    self.stats.train_samples = len(splits["train"])
                if "validation" in splits:
                    self.stats.val_samples = len(splits["validation"])
                if "test" in splits:
                    self.stats.test_samples = len(splits["test"])

            self._save_checkpoint(
                "dataset_building_complete",
                dataset_dir=str(self.dataset_dir)
            )

            self.logger.info("Dataset built successfully:")
            self.logger.info(f"  Train: {self.stats.train_samples}")
            self.logger.info(f"  Validation: {self.stats.val_samples}")
            self.logger.info(f"  Test: {self.stats.test_samples}")

            return splits

    def step5_validate_outputs(self) -> Tuple[bool, Dict]:
        """
        Step 5: Validate all outputs.

        Returns:
            Tuple of (success, validation report)
        """
        if self.config.skip_validation:
            self.logger.info("Skipping validation (--skip-validation flag)")
            return True, {"passed": True}

        with self._step_timer("validation"):
            # Use DataValidator's validate_all method
            qa_pairs_file = self.qa_output_dir / "qa_pairs.json"

            validation_report = self.validator.validate_all(
                frames_dir=self.frames_dir,
                qa_pairs_file=qa_pairs_file if qa_pairs_file.exists() else None,
                dataset=None  # Dataset validation is done separately
            )

            self.stats.validation_passed = validation_report.overall_valid

            # Collect errors and warnings
            errors = []
            warnings = []

            for result in [validation_report.frames_result, validation_report.qa_pairs_result]:
                if result:
                    errors.extend([str(e) for e in result.errors])
                    warnings.extend([str(w) for w in result.warnings])

            self.stats.validation_errors = errors
            self.stats.validation_warnings = warnings

            if validation_report.overall_valid:
                self.logger.info("Validation PASSED")
            else:
                self.logger.warning("Validation FAILED")
                for error in errors[:10]:  # Show first 10 errors
                    self.logger.error(f"  {error}")

            for warning in warnings[:10]:  # Show first 10 warnings
                self.logger.warning(f"  {warning}")

            # Generate and log the full report
            report_text = self.validator.generate_report(validation_report)
            self.logger.debug(report_text)

            # Export JSON report
            report_json_path = self.config.output_dir / "validation_report.json"
            self.validator.export_report_json(validation_report, report_json_path)
            self.logger.info(f"Validation report saved to: {report_json_path}")

            return validation_report.overall_valid, {
                "passed": validation_report.overall_valid,
                "errors": errors,
                "warnings": warnings,
                "total_errors": validation_report.total_errors,
                "total_warnings": validation_report.total_warnings
            }

    def step6_save_dataset(self, dataset: Any) -> str:
        """
        Step 6: Save dataset and optionally push to Hub.

        Args:
            dataset: HuggingFace DatasetDict to save

        Returns:
            Path or URL where dataset was saved
        """
        with self._step_timer("dataset_saving"):
            # Save locally using DatasetBuilder API
            saved_path = self.dataset_builder.save_to_disk(
                dataset,
                output_dir=self.dataset_dir,
                save_format="parquet"
            )
            self.logger.info(f"Dataset saved to: {saved_path}")

            # Push to Hub if requested
            if self.config.push_to_hub:
                self.logger.info(f"Pushing to HuggingFace Hub: {self.config.push_to_hub}")

                try:
                    hub_url = self.dataset_builder.push_to_hub(
                        dataset,
                        repo_id=self.config.push_to_hub,
                        token=self.config.hf_token,
                        private=False
                    )
                    self.logger.info(f"Successfully pushed to: {hub_url}")
                    return hub_url

                except Exception as e:
                    self.logger.error(f"Failed to push to Hub: {e}")
                    import traceback
                    self.logger.debug(traceback.format_exc())
                    return str(saved_path)

            return str(saved_path)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate final pipeline report.

        Returns:
            Complete report dictionary
        """
        report = {
            "pipeline": "PitVQA Agent 1: Data Preparation",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "summary": {
                "videos_processed": f"{self.stats.processed_videos}/{self.stats.total_videos}",
                "frames_extracted": self.stats.total_frames_extracted,
                "frames_after_filtering": self.stats.final_frame_count,
                "qa_pairs_generated": self.stats.qa_pairs_generated,
                "dataset_samples": {
                    "train": self.stats.train_samples,
                    "validation": self.stats.val_samples,
                    "test": self.stats.test_samples,
                    "total": (
                        self.stats.train_samples +
                        self.stats.val_samples +
                        self.stats.test_samples
                    )
                },
                "validation_status": "PASSED" if self.stats.validation_passed else "FAILED",
                "total_duration": str(self.stats.get_duration()) if self.stats.get_duration() else None,
            }
        }

        # Save report
        report_file = self.config.output_dir / "pipeline_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_file}")

        return report

    def run(self) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline.

        Returns:
            Pipeline report dictionary
        """
        self.stats.start_time = datetime.now()

        try:
            # Initialize components
            self._initialize_components()

            # Load checkpoint if available
            self._load_checkpoint()

            # Step 1: Extract frames
            frame_count, frame_paths = self.step1_extract_frames()

            if frame_count == 0:
                self.logger.warning("No frames extracted. Check video directory.")

            # Step 2: Load annotations
            annotations = self.step2_load_annotations()

            if not annotations:
                self.logger.warning("No annotations loaded. Check annotation directory.")

            # Step 3: Generate QA pairs
            qa_pairs = self.step3_generate_qa_pairs(annotations)

            # Step 4: Build dataset
            dataset = self.step4_build_dataset(frame_paths, qa_pairs)

            # Step 5: Validate outputs
            validation_passed, validation_report = self.step5_validate_outputs()

            # Step 6: Save/push dataset
            if validation_passed or self.config.skip_validation:
                output_location = self.step6_save_dataset(dataset)
                self.logger.info(f"Dataset available at: {output_location}")
            else:
                self.logger.warning("Skipping save due to validation failures")

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

        finally:
            self.stats.end_time = datetime.now()

        # Generate and print report
        report = self.generate_report()

        self._print_summary(report)

        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print a formatted summary of the pipeline results."""
        summary = report["summary"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Videos processed: {summary['videos_processed']}")
        self.logger.info(f"Frames extracted: {summary['frames_extracted']}")
        self.logger.info(f"Frames after filtering: {summary['frames_after_filtering']}")
        self.logger.info(f"QA pairs generated: {summary['qa_pairs_generated']}")
        self.logger.info(f"Dataset samples:")
        self.logger.info(f"  Train: {summary['dataset_samples']['train']}")
        self.logger.info(f"  Validation: {summary['dataset_samples']['validation']}")
        self.logger.info(f"  Test: {summary['dataset_samples']['test']}")
        self.logger.info(f"  Total: {summary['dataset_samples']['total']}")
        self.logger.info(f"Validation: {summary['validation_status']}")
        self.logger.info(f"Duration: {summary['total_duration']}")
        self.logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PitVQA Agent 1: Data Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pitvqa_agent1_data_prep.py \\
      --video-dir data/raw/pitvqa/videos \\
      --annotation-dir data/raw/pitvqa/annotations \\
      --output-dir data/processed

  # With custom settings
  python pitvqa_agent1_data_prep.py \\
      --video-dir data/raw/pitvqa/videos \\
      --annotation-dir data/raw/pitvqa/annotations \\
      --output-dir data/processed \\
      --fps 2 \\
      --blur-threshold 150

  # Push to HuggingFace Hub
  python pitvqa_agent1_data_prep.py \\
      --video-dir data/raw/pitvqa/videos \\
      --annotation-dir data/raw/pitvqa/annotations \\
      --output-dir data/processed \\
      --push-to-hub username/pitvqa-dataset

  # Resume from checkpoint
  python pitvqa_agent1_data_prep.py \\
      --video-dir data/raw/pitvqa/videos \\
      --annotation-dir data/raw/pitvqa/annotations \\
      --output-dir data/processed \\
      --skip-extraction
        """
    )

    # Required arguments
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Path to directory containing video files"
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        required=True,
        help="Path to directory containing annotation files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for processed data"
    )

    # Frame extraction settings
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Blur detection threshold (default: 100)"
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format (default: png)"
    )

    # HuggingFace settings
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo ID to push dataset to (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)"
    )

    # Skip flags
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip frame extraction (use existing frames)"
    )
    parser.add_argument(
        "--skip-qa-generation",
        action="store_true",
        help="Skip QA pair generation (use existing pairs)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip output validation"
    )

    # Processing settings
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)"
    )

    # Checkpoint settings
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from checkpoint"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Get HuggingFace token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Create configuration
    config = PipelineConfig(
        video_dir=args.video_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        blur_threshold=args.blur_threshold,
        image_format=args.image_format,
        push_to_hub=args.push_to_hub,
        hf_token=hf_token,
        skip_extraction=args.skip_extraction,
        skip_qa_generation=args.skip_qa_generation,
        skip_validation=args.skip_validation,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        resume_from_checkpoint=not args.no_resume,
    )

    # Run pipeline
    pipeline = PitVQADataPrep(config)

    try:
        report = pipeline.run()

        # Exit code based on validation
        if report["summary"]["validation_status"] == "PASSED":
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
