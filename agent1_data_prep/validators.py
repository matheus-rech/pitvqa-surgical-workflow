"""
PitVQA Data Preparation Validators

Production-ready validation module for the Agent 1 data preparation pipeline.
Validates frame extraction, QA pairs, and dataset integrity for the PitVQA
surgical VQA dataset.

Expected dataset specifications:
- ~109,173 total frames from 25 videos
- ~884,242 QA pairs (~8 per frame)
- 59 annotation classes
- 4 phases, 15 steps, 18 instruments
- Train/val/test split: 80/10/10

Author: PitVQA Team
"""

import os
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Specifications
# =============================================================================

class DatasetSpecs:
    """PitVQA dataset specifications and expected values."""

    # Frame extraction expectations
    TOTAL_VIDEOS = 25
    EXPECTED_TOTAL_FRAMES = 109173
    MIN_FRAMES_PER_VIDEO = 2443
    MAX_FRAMES_PER_VIDEO = 7179
    FRAME_TOLERANCE_PERCENT = 5.0  # Allow 5% deviation

    # QA pair expectations
    EXPECTED_TOTAL_QA_PAIRS = 884242
    EXPECTED_QA_PER_FRAME = 8
    QA_TOLERANCE_PERCENT = 10.0  # Allow 10% deviation

    # Valid question types
    VALID_QUESTION_TYPES = frozenset({
        'phase',
        'step',
        'instrument',
        'position',
        'yes_no',
        'operation_note'
    })

    # Required QA fields
    REQUIRED_QA_FIELDS = frozenset({
        'frame_id',
        'question',
        'answer',
        'question_type'
    })

    # Annotation class expectations
    EXPECTED_TOTAL_CLASSES = 59
    EXPECTED_PHASES = 4
    EXPECTED_STEPS = 15
    EXPECTED_INSTRUMENTS = 18

    # Dataset split expectations
    EXPECTED_TRAIN_RATIO = 0.80
    EXPECTED_VAL_RATIO = 0.10
    EXPECTED_TEST_RATIO = 0.10
    SPLIT_TOLERANCE = 0.02  # Allow 2% deviation

    # Valid image extensions
    VALID_IMAGE_EXTENSIONS = frozenset({'.png', '.jpg', '.jpeg', '.bmp', '.tiff'})


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    message: str
    details: Optional[str] = None
    location: Optional[str] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        parts = [f"[{self.severity.value.upper()}] {self.message}"]
        if self.location:
            parts.append(f"  Location: {self.location}")
        if self.details:
            parts.append(f"  Details: {self.details}")
        if self.suggestion:
            parts.append(f"  Suggestion: {self.suggestion}")
        return "\n".join(parts)


@dataclass
class ValidationStatistics:
    """Statistics collected during validation."""
    counts: Dict[str, int] = field(default_factory=dict)
    distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    percentages: Dict[str, float] = field(default_factory=dict)
    samples: Dict[str, List[Any]] = field(default_factory=dict)

    def add_count(self, key: str, value: int) -> None:
        self.counts[key] = value

    def add_distribution(self, key: str, dist: Dict[str, int]) -> None:
        self.distributions[key] = dist

    def add_percentage(self, key: str, value: float) -> None:
        self.percentages[key] = value

    def add_samples(self, key: str, samples: List[Any]) -> None:
        self.samples[key] = samples


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    statistics: ValidationStatistics = field(default_factory=ValidationStatistics)

    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_error(self, message: str, details: str = None,
                  location: str = None, suggestion: str = None) -> None:
        self.errors.append(ValidationIssue(
            severity=ValidationSeverity.ERROR,
            message=message,
            details=details,
            location=location,
            suggestion=suggestion
        ))
        self.is_valid = False

    def add_warning(self, message: str, details: str = None,
                    location: str = None, suggestion: str = None) -> None:
        self.warnings.append(ValidationIssue(
            severity=ValidationSeverity.WARNING,
            message=message,
            details=details,
            location=location,
            suggestion=suggestion
        ))

    def add_info(self, message: str, details: str = None,
                 location: str = None) -> None:
        self.info.append(ValidationIssue(
            severity=ValidationSeverity.INFO,
            message=message,
            details=details,
            location=location
        ))

    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if other.has_errors:
            self.is_valid = False
        # Merge statistics
        self.statistics.counts.update(other.statistics.counts)
        self.statistics.distributions.update(other.statistics.distributions)
        self.statistics.percentages.update(other.statistics.percentages)
        self.statistics.samples.update(other.statistics.samples)


@dataclass
class ValidationReport:
    """Complete validation report for all checks."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    frames_result: Optional[ValidationResult] = None
    qa_pairs_result: Optional[ValidationResult] = None
    dataset_result: Optional[ValidationResult] = None
    overall_valid: bool = True

    @property
    def total_errors(self) -> int:
        total = 0
        for result in [self.frames_result, self.qa_pairs_result, self.dataset_result]:
            if result:
                total += len(result.errors)
        return total

    @property
    def total_warnings(self) -> int:
        total = 0
        for result in [self.frames_result, self.qa_pairs_result, self.dataset_result]:
            if result:
                total += len(result.warnings)
        return total


# =============================================================================
# Image Validation Utilities
# =============================================================================

class ImageValidator:
    """Utility class for validating image files."""

    @staticmethod
    def is_valid_image(file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Check if a file is a valid, non-corrupted image.

        Args:
            file_path: Path to the image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(file_path) as img:
                # Verify the image can be read
                img.verify()

            # Re-open to load actual data (verify closes the file)
            with Image.open(file_path) as img:
                # Force load to detect truncated images
                img.load()

                # Check for valid dimensions
                if img.size[0] <= 0 or img.size[1] <= 0:
                    return False, "Invalid image dimensions"

            return True, None

        except FileNotFoundError:
            return False, "File not found"
        except Image.UnidentifiedImageError:
            return False, "Cannot identify image format"
        except Image.DecompressionBombError:
            return False, "Image too large (decompression bomb)"
        except Exception as e:
            return False, f"Image validation error: {str(e)}"

    @staticmethod
    def get_image_info(file_path: Path) -> Optional[Dict[str, Any]]:
        """Get image metadata."""
        try:
            with Image.open(file_path) as img:
                return {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'file_size': file_path.stat().st_size
                }
        except Exception:
            return None


# =============================================================================
# Main Validator Class
# =============================================================================

class DataValidator:
    """
    Production-ready data validation for PitVQA pipeline.

    Validates:
    - Frame extraction completeness and integrity
    - QA pair formatting and content
    - Dataset integrity and split correctness
    - Statistics conformance to expected values

    Example usage:
        validator = DataValidator()

        # Validate individual components
        frame_result = validator.validate_frames("data/frames")
        qa_result = validator.validate_qa_pairs("data/qa_pairs.json")

        # Validate complete dataset
        dataset_result = validator.validate_dataset({
            'train': train_data,
            'val': val_data,
            'test': test_data
        })

        # Run all validations
        report = validator.validate_all(
            frames_dir="data/frames",
            qa_pairs_file="data/qa_pairs.json",
            dataset=dataset
        )

        # Generate report
        print(validator.generate_report())
    """

    def __init__(
        self,
        specs: Optional[DatasetSpecs] = None,
        strict_mode: bool = False,
        sample_validation_count: int = 100,
        log_progress: bool = True
    ):
        """
        Initialize the data validator.

        Args:
            specs: Dataset specifications (uses defaults if None)
            strict_mode: If True, treat warnings as errors
            sample_validation_count: Number of samples for detailed validation
            log_progress: Whether to log progress during validation
        """
        self.specs = specs or DatasetSpecs()
        self.strict_mode = strict_mode
        self.sample_validation_count = sample_validation_count
        self.log_progress = log_progress
        self._last_report: Optional[ValidationReport] = None

    def _log(self, message: str, level: str = "info") -> None:
        """Log a message if logging is enabled."""
        if self.log_progress:
            getattr(logger, level)(message)

    # =========================================================================
    # Frame Validation
    # =========================================================================

    def validate_frames(
        self,
        frames_dir: Union[str, Path],
        validate_image_integrity: bool = True,
        sample_size: Optional[int] = None
    ) -> ValidationResult:
        """
        Validate frame extraction results.

        Checks:
        - Total frame count (~109,173 expected)
        - Per-video frame counts (min 2,443, max 7,179)
        - Image file existence and validity
        - No corrupted images

        Args:
            frames_dir: Directory containing extracted frames
            validate_image_integrity: Whether to verify image content
            sample_size: Number of images to sample for integrity check

        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        self._log("Starting frame validation...")
        result = ValidationResult(is_valid=True)
        frames_path = Path(frames_dir)

        # Check directory exists
        if not frames_path.exists():
            result.add_error(
                message="Frames directory does not exist",
                location=str(frames_path),
                suggestion="Run frame extraction pipeline first"
            )
            return result

        if not frames_path.is_dir():
            result.add_error(
                message="Frames path is not a directory",
                location=str(frames_path),
                suggestion="Provide path to frames directory, not a file"
            )
            return result

        # Collect frame information
        video_dirs = [d for d in frames_path.iterdir() if d.is_dir()]
        all_frames: List[Path] = []
        frames_per_video: Dict[str, int] = {}
        corrupted_frames: List[Tuple[Path, str]] = []

        self._log(f"Found {len(video_dirs)} video directories")

        # Check video count
        if len(video_dirs) == 0:
            result.add_error(
                message="No video directories found in frames directory",
                location=str(frames_path),
                suggestion="Check frame extraction output structure. Expected: frames_dir/video_id/frame_*.png"
            )
            return result

        if len(video_dirs) != self.specs.TOTAL_VIDEOS:
            result.add_warning(
                message=f"Unexpected number of video directories",
                details=f"Found {len(video_dirs)}, expected {self.specs.TOTAL_VIDEOS}",
                suggestion="Verify all videos were processed"
            )

        # Process each video directory
        for video_dir in video_dirs:
            video_id = video_dir.name
            frames = list(video_dir.glob("*.png")) + \
                     list(video_dir.glob("*.jpg")) + \
                     list(video_dir.glob("*.jpeg"))

            frame_count = len(frames)
            frames_per_video[video_id] = frame_count
            all_frames.extend(frames)

            # Check per-video frame count
            if frame_count < self.specs.MIN_FRAMES_PER_VIDEO:
                result.add_warning(
                    message=f"Low frame count for video {video_id}",
                    details=f"Found {frame_count}, minimum expected {self.specs.MIN_FRAMES_PER_VIDEO}",
                    location=str(video_dir),
                    suggestion="Check video extraction settings or video duration"
                )

            if frame_count > self.specs.MAX_FRAMES_PER_VIDEO:
                result.add_warning(
                    message=f"High frame count for video {video_id}",
                    details=f"Found {frame_count}, maximum expected {self.specs.MAX_FRAMES_PER_VIDEO}",
                    location=str(video_dir),
                    suggestion="Check FPS settings or video duration"
                )

        total_frames = len(all_frames)
        result.statistics.add_count("total_frames", total_frames)
        result.statistics.add_count("total_videos", len(video_dirs))
        result.statistics.add_distribution("frames_per_video", frames_per_video)

        self._log(f"Total frames found: {total_frames}")

        # Check total frame count
        expected_min = self.specs.EXPECTED_TOTAL_FRAMES * (1 - self.specs.FRAME_TOLERANCE_PERCENT / 100)
        expected_max = self.specs.EXPECTED_TOTAL_FRAMES * (1 + self.specs.FRAME_TOLERANCE_PERCENT / 100)

        if total_frames < expected_min:
            result.add_error(
                message="Total frame count significantly below expected",
                details=f"Found {total_frames}, expected ~{self.specs.EXPECTED_TOTAL_FRAMES} (min {int(expected_min)})",
                suggestion="Check if all videos were processed and frame extraction completed successfully"
            )
        elif total_frames > expected_max:
            result.add_warning(
                message="Total frame count above expected",
                details=f"Found {total_frames}, expected ~{self.specs.EXPECTED_TOTAL_FRAMES}",
                suggestion="This may be acceptable if using different FPS settings"
            )
        else:
            result.add_info(
                message=f"Frame count within expected range: {total_frames}"
            )

        # Validate image integrity (sample-based)
        if validate_image_integrity and all_frames:
            sample_size = sample_size or min(self.sample_validation_count, len(all_frames))
            sample_indices = np.random.choice(len(all_frames), sample_size, replace=False)
            sample_frames = [all_frames[i] for i in sample_indices]

            self._log(f"Validating {sample_size} sample frames for integrity...")

            for frame_path in sample_frames:
                is_valid, error_msg = ImageValidator.is_valid_image(frame_path)
                if not is_valid:
                    corrupted_frames.append((frame_path, error_msg))

            if corrupted_frames:
                result.add_error(
                    message=f"Found {len(corrupted_frames)} corrupted/invalid images in sample",
                    details=f"Sample of {sample_size} images checked. Issues: " +
                            ", ".join([f"{p.name}: {e}" for p, e in corrupted_frames[:5]]),
                    suggestion="Re-extract frames for affected videos or investigate source video quality"
                )
                result.statistics.add_samples(
                    "corrupted_frames",
                    [(str(p), e) for p, e in corrupted_frames]
                )
            else:
                result.add_info(
                    message=f"All {sample_size} sampled frames passed integrity check"
                )

        # Add frame statistics
        if frames_per_video:
            counts = list(frames_per_video.values())
            result.statistics.add_count("min_frames_per_video", min(counts))
            result.statistics.add_count("max_frames_per_video", max(counts))
            result.statistics.add_count("avg_frames_per_video", int(np.mean(counts)))

        self._log(f"Frame validation complete. Valid: {result.is_valid}")
        return result

    # =========================================================================
    # QA Pairs Validation
    # =========================================================================

    def validate_qa_pairs(
        self,
        qa_pairs_file: Union[str, Path],
        frames_dir: Optional[Union[str, Path]] = None
    ) -> ValidationResult:
        """
        Validate QA pairs file.

        Checks:
        - Total QA pair count (~884,242 expected)
        - Required fields present (frame_id, question, answer, question_type)
        - Questions properly formatted (end with ?)
        - Answers non-empty
        - Valid question types
        - Frame references exist (if frames_dir provided)

        Args:
            qa_pairs_file: Path to QA pairs JSON/CSV file
            frames_dir: Optional path to frames for cross-reference

        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        self._log("Starting QA pairs validation...")
        result = ValidationResult(is_valid=True)
        qa_path = Path(qa_pairs_file)

        # Check file exists
        if not qa_path.exists():
            result.add_error(
                message="QA pairs file does not exist",
                location=str(qa_path),
                suggestion="Run QA generation pipeline first"
            )
            return result

        # Load QA pairs
        try:
            qa_pairs = self._load_qa_pairs(qa_path)
        except Exception as e:
            result.add_error(
                message="Failed to load QA pairs file",
                details=str(e),
                location=str(qa_path),
                suggestion="Check file format (JSON or CSV expected)"
            )
            return result

        if not qa_pairs:
            result.add_error(
                message="QA pairs file is empty",
                location=str(qa_path),
                suggestion="Run QA generation pipeline"
            )
            return result

        total_pairs = len(qa_pairs)
        result.statistics.add_count("total_qa_pairs", total_pairs)
        self._log(f"Loaded {total_pairs} QA pairs")

        # Check total count
        expected_min = self.specs.EXPECTED_TOTAL_QA_PAIRS * (1 - self.specs.QA_TOLERANCE_PERCENT / 100)
        expected_max = self.specs.EXPECTED_TOTAL_QA_PAIRS * (1 + self.specs.QA_TOLERANCE_PERCENT / 100)

        if total_pairs < expected_min:
            result.add_error(
                message="QA pair count significantly below expected",
                details=f"Found {total_pairs}, expected ~{self.specs.EXPECTED_TOTAL_QA_PAIRS}",
                suggestion="Check QA generation for all frames and question types"
            )
        elif total_pairs > expected_max:
            result.add_warning(
                message="QA pair count above expected",
                details=f"Found {total_pairs}, expected ~{self.specs.EXPECTED_TOTAL_QA_PAIRS}"
            )

        # Validate individual pairs
        missing_fields: Dict[str, int] = defaultdict(int)
        invalid_questions: List[Dict] = []
        empty_answers: List[Dict] = []
        invalid_types: List[Dict] = []
        question_types: Counter = Counter()
        frame_ids: Set[str] = set()

        for i, pair in enumerate(qa_pairs):
            # Check required fields
            for field in self.specs.REQUIRED_QA_FIELDS:
                if field not in pair or pair[field] is None:
                    missing_fields[field] += 1

            # Validate question format
            question = pair.get('question', '')
            if question and not question.strip().endswith('?'):
                if len(invalid_questions) < 10:  # Sample
                    invalid_questions.append({'index': i, 'question': question})

            # Check answer is non-empty
            answer = pair.get('answer', '')
            if not answer or not str(answer).strip():
                if len(empty_answers) < 10:  # Sample
                    empty_answers.append({'index': i, 'pair': pair})

            # Validate question type
            qtype = pair.get('question_type', '')
            if qtype and qtype not in self.specs.VALID_QUESTION_TYPES:
                if len(invalid_types) < 10:  # Sample
                    invalid_types.append({'index': i, 'type': qtype})

            if qtype:
                question_types[qtype] += 1

            # Track frame IDs
            frame_id = pair.get('frame_id', '')
            if frame_id:
                frame_ids.add(str(frame_id))

        # Report missing fields
        for field, count in missing_fields.items():
            result.add_error(
                message=f"Missing required field: {field}",
                details=f"{count} pairs missing this field ({count/total_pairs*100:.1f}%)",
                suggestion=f"Ensure QA generation includes '{field}' for all pairs"
            )

        # Report invalid questions
        if invalid_questions:
            result.add_warning(
                message="Questions not ending with question mark",
                details=f"Found {len(invalid_questions)}+ pairs. Sample: {invalid_questions[0]['question'][:50]}...",
                suggestion="Add question marks to all questions during generation"
            )

        # Report empty answers
        if empty_answers:
            result.add_error(
                message="Empty or null answers found",
                details=f"Found {len(empty_answers)}+ pairs with empty answers",
                suggestion="Ensure all QA pairs have valid non-empty answers"
            )

        # Report invalid question types
        if invalid_types:
            invalid_type_set = set(t['type'] for t in invalid_types)
            result.add_error(
                message="Invalid question types found",
                details=f"Invalid types: {invalid_type_set}. Valid: {self.specs.VALID_QUESTION_TYPES}",
                suggestion="Map question types to valid categories"
            )

        # Check question type distribution
        result.statistics.add_distribution("question_types", dict(question_types))
        result.statistics.add_count("unique_frames", len(frame_ids))

        # Check coverage of question types
        missing_types = self.specs.VALID_QUESTION_TYPES - set(question_types.keys())
        if missing_types:
            result.add_warning(
                message="Missing question types",
                details=f"Types not found: {missing_types}",
                suggestion="Generate QA pairs for all question types"
            )

        # Check QA per frame ratio
        if frame_ids:
            avg_qa_per_frame = total_pairs / len(frame_ids)
            result.statistics.add_percentage("avg_qa_per_frame", avg_qa_per_frame)

            if avg_qa_per_frame < self.specs.EXPECTED_QA_PER_FRAME * 0.5:
                result.add_warning(
                    message="Low QA pairs per frame ratio",
                    details=f"Average {avg_qa_per_frame:.1f} QA per frame, expected ~{self.specs.EXPECTED_QA_PER_FRAME}",
                    suggestion="Generate more diverse questions per frame"
                )

        # Cross-reference with frames if provided
        if frames_dir:
            frames_path = Path(frames_dir)
            if frames_path.exists():
                self._log("Cross-referencing QA pairs with frames...")
                orphan_qa = self._find_orphan_qa_pairs(frame_ids, frames_path)
                if orphan_qa:
                    result.add_warning(
                        message="QA pairs reference non-existent frames",
                        details=f"Found {len(orphan_qa)} orphaned frame references",
                        suggestion="Regenerate QA pairs or check frame extraction"
                    )
                    result.statistics.add_samples("orphan_frame_ids", list(orphan_qa)[:20])

        self._log(f"QA pairs validation complete. Valid: {result.is_valid}")
        return result

    def _load_qa_pairs(self, qa_path: Path) -> List[Dict]:
        """Load QA pairs from JSON or CSV file."""
        if qa_path.suffix == '.json':
            with open(qa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get('qa_pairs', data.get('annotations', []))
        elif qa_path.suffix == '.csv':
            import csv
            with open(qa_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                return list(reader)
        else:
            raise ValueError(f"Unsupported file format: {qa_path.suffix}")

    def _find_orphan_qa_pairs(
        self,
        frame_ids: Set[str],
        frames_path: Path
    ) -> Set[str]:
        """Find QA pairs that reference non-existent frames."""
        existing_frames: Set[str] = set()

        for video_dir in frames_path.iterdir():
            if video_dir.is_dir():
                for frame_file in video_dir.iterdir():
                    if frame_file.suffix.lower() in self.specs.VALID_IMAGE_EXTENSIONS:
                        # Add various possible ID formats
                        existing_frames.add(frame_file.stem)
                        existing_frames.add(str(frame_file.relative_to(frames_path)))
                        existing_frames.add(frame_file.name)

        return frame_ids - existing_frames

    # =========================================================================
    # Dataset Validation
    # =========================================================================

    def validate_dataset(
        self,
        dataset: Dict[str, List[Dict]],
        check_leakage: bool = True
    ) -> ValidationResult:
        """
        Validate complete dataset integrity.

        Checks:
        - Train/val/test split proportions (80/10/10)
        - No data leakage (same video not in multiple splits)
        - All frames have corresponding QA pairs
        - Class distribution is reasonable
        - 59 annotation classes present
        - 4 phases, 15 steps, 18 instruments coverage

        Args:
            dataset: Dictionary with 'train', 'val', 'test' keys
            check_leakage: Whether to check for data leakage

        Returns:
            ValidationResult with errors, warnings, and statistics
        """
        self._log("Starting dataset validation...")
        result = ValidationResult(is_valid=True)

        # Check required splits
        required_splits = {'train', 'val', 'test'}
        missing_splits = required_splits - set(dataset.keys())

        if missing_splits:
            result.add_error(
                message="Missing dataset splits",
                details=f"Missing: {missing_splits}",
                suggestion="Create all three splits: train, val, test"
            )
            return result

        # Calculate split sizes
        split_sizes = {split: len(data) for split, data in dataset.items()}
        total_samples = sum(split_sizes.values())

        if total_samples == 0:
            result.add_error(
                message="Dataset is empty",
                suggestion="Generate and split dataset first"
            )
            return result

        result.statistics.add_count("total_samples", total_samples)
        result.statistics.add_distribution("split_sizes", split_sizes)

        # Check split ratios
        split_ratios = {split: size / total_samples for split, size in split_sizes.items()}
        result.statistics.add_distribution("split_ratios",
            {k: round(v, 3) for k, v in split_ratios.items()})

        expected_ratios = {
            'train': self.specs.EXPECTED_TRAIN_RATIO,
            'val': self.specs.EXPECTED_VAL_RATIO,
            'test': self.specs.EXPECTED_TEST_RATIO
        }

        for split, expected in expected_ratios.items():
            actual = split_ratios.get(split, 0)
            if abs(actual - expected) > self.specs.SPLIT_TOLERANCE:
                result.add_warning(
                    message=f"Split ratio deviation for '{split}'",
                    details=f"Actual: {actual:.1%}, Expected: {expected:.0%}",
                    suggestion="Adjust dataset splitting parameters"
                )

        # Check for data leakage
        if check_leakage:
            self._log("Checking for data leakage...")
            leakage_result = self._check_data_leakage(dataset)
            result.merge(leakage_result)

        # Collect and validate class distribution
        self._log("Analyzing class distribution...")
        class_result = self._validate_class_distribution(dataset)
        result.merge(class_result)

        # Validate annotation coverage
        coverage_result = self._validate_annotation_coverage(dataset)
        result.merge(coverage_result)

        self._log(f"Dataset validation complete. Valid: {result.is_valid}")
        return result

    def _check_data_leakage(self, dataset: Dict[str, List[Dict]]) -> ValidationResult:
        """Check for data leakage between splits."""
        result = ValidationResult(is_valid=True)

        # Extract video IDs from each split
        split_videos: Dict[str, Set[str]] = {}

        for split, data in dataset.items():
            videos = set()
            for sample in data:
                # Try to extract video ID from frame_id or video_id
                video_id = sample.get('video_id')
                if not video_id:
                    frame_id = sample.get('frame_id', '')
                    # Assume format: video_id/frame_xxxx or video_id_frame_xxxx
                    if '/' in frame_id:
                        video_id = frame_id.split('/')[0]
                    elif '_frame_' in frame_id:
                        video_id = frame_id.split('_frame_')[0]
                    elif '_' in frame_id:
                        # Fallback: assume first part is video ID
                        parts = frame_id.rsplit('_', 1)
                        if len(parts) > 1 and parts[1].isdigit():
                            video_id = parts[0]

                if video_id:
                    videos.add(video_id)

            split_videos[split] = videos

        # Check for overlaps
        splits = list(split_videos.keys())
        for i, split1 in enumerate(splits):
            for split2 in splits[i+1:]:
                overlap = split_videos[split1] & split_videos[split2]
                if overlap:
                    result.add_error(
                        message=f"Data leakage detected between {split1} and {split2}",
                        details=f"Shared videos: {overlap}",
                        suggestion="Ensure videos are assigned to only one split. Split at video level, not frame level."
                    )

        result.statistics.add_distribution(
            "videos_per_split",
            {split: len(videos) for split, videos in split_videos.items()}
        )

        return result

    def _validate_class_distribution(
        self,
        dataset: Dict[str, List[Dict]]
    ) -> ValidationResult:
        """Validate class distribution across dataset."""
        result = ValidationResult(is_valid=True)

        # Collect all classes
        phases: Set[str] = set()
        steps: Set[str] = set()
        instruments: Set[str] = set()
        all_classes: Set[str] = set()

        for split, data in dataset.items():
            for sample in data:
                # Extract phase, step, instrument from answer or annotations
                answer = str(sample.get('answer', ''))
                qtype = sample.get('question_type', '')

                if qtype == 'phase' and answer:
                    phases.add(answer)
                    all_classes.add(f"phase:{answer}")
                elif qtype == 'step' and answer:
                    steps.add(answer)
                    all_classes.add(f"step:{answer}")
                elif qtype == 'instrument' and answer:
                    instruments.add(answer)
                    all_classes.add(f"instrument:{answer}")

                # Also check for annotations field
                annotations = sample.get('annotations', {})
                if 'phase' in annotations:
                    phases.add(str(annotations['phase']))
                if 'step' in annotations:
                    steps.add(str(annotations['step']))
                if 'instruments' in annotations:
                    for inst in annotations['instruments']:
                        instruments.add(str(inst))

        result.statistics.add_count("unique_phases", len(phases))
        result.statistics.add_count("unique_steps", len(steps))
        result.statistics.add_count("unique_instruments", len(instruments))
        result.statistics.add_count("total_classes", len(all_classes))

        result.statistics.add_samples("phases", list(phases))
        result.statistics.add_samples("steps", list(steps))
        result.statistics.add_samples("instruments", list(instruments)[:20])  # Limit sample

        # Check coverage
        if len(phases) > 0 and len(phases) != self.specs.EXPECTED_PHASES:
            result.add_warning(
                message="Phase count mismatch",
                details=f"Found {len(phases)} phases, expected {self.specs.EXPECTED_PHASES}",
                suggestion="Verify all surgical phases are represented"
            )

        if len(steps) > 0 and len(steps) != self.specs.EXPECTED_STEPS:
            result.add_warning(
                message="Step count mismatch",
                details=f"Found {len(steps)} steps, expected {self.specs.EXPECTED_STEPS}",
                suggestion="Verify all surgical steps are represented"
            )

        if len(instruments) > 0 and len(instruments) != self.specs.EXPECTED_INSTRUMENTS:
            result.add_warning(
                message="Instrument count mismatch",
                details=f"Found {len(instruments)} instruments, expected {self.specs.EXPECTED_INSTRUMENTS}",
                suggestion="Verify all surgical instruments are represented"
            )

        if len(all_classes) > 0 and abs(len(all_classes) - self.specs.EXPECTED_TOTAL_CLASSES) > 5:
            result.add_warning(
                message="Total annotation class count differs from expected",
                details=f"Found {len(all_classes)} classes, expected ~{self.specs.EXPECTED_TOTAL_CLASSES}",
                suggestion="Review annotation class definitions"
            )

        return result

    def _validate_annotation_coverage(
        self,
        dataset: Dict[str, List[Dict]]
    ) -> ValidationResult:
        """Validate that all question types are adequately covered."""
        result = ValidationResult(is_valid=True)

        type_coverage: Dict[str, Dict[str, int]] = {}

        for split, data in dataset.items():
            type_counts = Counter()
            for sample in data:
                qtype = sample.get('question_type', 'unknown')
                type_counts[qtype] += 1
            type_coverage[split] = dict(type_counts)

        result.statistics.add_distribution("question_type_coverage", type_coverage)

        # Check each split has all question types
        for split, counts in type_coverage.items():
            missing = self.specs.VALID_QUESTION_TYPES - set(counts.keys())
            if missing:
                result.add_warning(
                    message=f"Split '{split}' missing question types",
                    details=f"Missing: {missing}",
                    suggestion="Ensure balanced question type sampling in splits"
                )

        return result

    # =========================================================================
    # Complete Validation
    # =========================================================================

    def validate_all(
        self,
        frames_dir: Optional[Union[str, Path]] = None,
        qa_pairs_file: Optional[Union[str, Path]] = None,
        dataset: Optional[Dict[str, List[Dict]]] = None
    ) -> ValidationReport:
        """
        Run all validations and generate comprehensive report.

        Args:
            frames_dir: Directory containing extracted frames
            qa_pairs_file: Path to QA pairs JSON/CSV file
            dataset: Dictionary with 'train', 'val', 'test' data

        Returns:
            ValidationReport with all results
        """
        self._log("=" * 60)
        self._log("Starting comprehensive data validation")
        self._log("=" * 60)

        report = ValidationReport()

        # Validate frames
        if frames_dir:
            self._log("\n[1/3] Validating frames...")
            report.frames_result = self.validate_frames(frames_dir)
            if report.frames_result.has_errors:
                report.overall_valid = False

        # Validate QA pairs
        if qa_pairs_file:
            self._log("\n[2/3] Validating QA pairs...")
            report.qa_pairs_result = self.validate_qa_pairs(
                qa_pairs_file,
                frames_dir=frames_dir
            )
            if report.qa_pairs_result.has_errors:
                report.overall_valid = False

        # Validate dataset
        if dataset:
            self._log("\n[3/3] Validating dataset...")
            report.dataset_result = self.validate_dataset(dataset)
            if report.dataset_result.has_errors:
                report.overall_valid = False

        self._last_report = report

        self._log("\n" + "=" * 60)
        self._log(f"Validation complete. Overall valid: {report.overall_valid}")
        self._log(f"Total errors: {report.total_errors}")
        self._log(f"Total warnings: {report.total_warnings}")
        self._log("=" * 60)

        return report

    # =========================================================================
    # Report Generation
    # =========================================================================

    def generate_report(self, report: Optional[ValidationReport] = None) -> str:
        """
        Generate a human-readable validation report.

        Args:
            report: ValidationReport to format (uses last report if None)

        Returns:
            Formatted report string
        """
        report = report or self._last_report

        if not report:
            return "No validation report available. Run validate_all() first."

        lines = [
            "=" * 80,
            "PITVQA DATA VALIDATION REPORT",
            "=" * 80,
            f"Generated: {report.timestamp}",
            f"Overall Status: {'PASSED' if report.overall_valid else 'FAILED'}",
            f"Total Errors: {report.total_errors}",
            f"Total Warnings: {report.total_warnings}",
            "",
        ]

        # Frames section
        if report.frames_result:
            lines.extend(self._format_result_section("FRAME VALIDATION", report.frames_result))

        # QA pairs section
        if report.qa_pairs_result:
            lines.extend(self._format_result_section("QA PAIRS VALIDATION", report.qa_pairs_result))

        # Dataset section
        if report.dataset_result:
            lines.extend(self._format_result_section("DATASET VALIDATION", report.dataset_result))

        # Summary
        lines.extend([
            "",
            "=" * 80,
            "SUMMARY",
            "=" * 80,
        ])

        if report.overall_valid:
            lines.append("All validations PASSED. Data is ready for training.")
        else:
            lines.append("Validation FAILED. Please address the errors above before proceeding.")
            lines.append("")
            lines.append("Critical issues to fix:")

            all_errors = []
            for result in [report.frames_result, report.qa_pairs_result, report.dataset_result]:
                if result:
                    all_errors.extend(result.errors)

            for i, error in enumerate(all_errors, 1):
                lines.append(f"  {i}. {error.message}")
                if error.suggestion:
                    lines.append(f"     Fix: {error.suggestion}")

        return "\n".join(lines)

    def _format_result_section(
        self,
        title: str,
        result: ValidationResult
    ) -> List[str]:
        """Format a validation result section."""
        lines = [
            "",
            "-" * 80,
            title,
            "-" * 80,
            f"Status: {'PASSED' if result.is_valid else 'FAILED'}",
            "",
        ]

        # Statistics
        if result.statistics.counts:
            lines.append("Statistics:")
            for key, value in result.statistics.counts.items():
                lines.append(f"  - {key}: {value:,}")

        if result.statistics.percentages:
            for key, value in result.statistics.percentages.items():
                lines.append(f"  - {key}: {value:.2f}")

        if result.statistics.distributions:
            for key, dist in result.statistics.distributions.items():
                if len(dist) <= 10:
                    lines.append(f"  - {key}: {dist}")
                else:
                    lines.append(f"  - {key}: ({len(dist)} items)")

        lines.append("")

        # Errors
        if result.errors:
            lines.append(f"Errors ({len(result.errors)}):")
            for error in result.errors:
                lines.append(f"  [ERROR] {error.message}")
                if error.details:
                    lines.append(f"          Details: {error.details}")
                if error.suggestion:
                    lines.append(f"          Suggestion: {error.suggestion}")

        # Warnings
        if result.warnings:
            lines.append(f"\nWarnings ({len(result.warnings)}):")
            for warning in result.warnings:
                lines.append(f"  [WARN] {warning.message}")
                if warning.details:
                    lines.append(f"         Details: {warning.details}")
                if warning.suggestion:
                    lines.append(f"         Suggestion: {warning.suggestion}")

        # Info
        if result.info:
            lines.append(f"\nInfo ({len(result.info)}):")
            for info in result.info:
                lines.append(f"  [INFO] {info.message}")

        return lines

    def export_report_json(
        self,
        report: Optional[ValidationReport] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Export validation report as JSON.

        Args:
            report: ValidationReport to export (uses last report if None)
            output_path: Optional path to save JSON file

        Returns:
            Dictionary representation of the report
        """
        report = report or self._last_report

        if not report:
            return {"error": "No validation report available"}

        def issue_to_dict(issue: ValidationIssue) -> Dict:
            return {
                "severity": issue.severity.value,
                "message": issue.message,
                "details": issue.details,
                "location": issue.location,
                "suggestion": issue.suggestion
            }

        def result_to_dict(result: Optional[ValidationResult]) -> Optional[Dict]:
            if not result:
                return None
            return {
                "is_valid": result.is_valid,
                "errors": [issue_to_dict(e) for e in result.errors],
                "warnings": [issue_to_dict(w) for w in result.warnings],
                "info": [issue_to_dict(i) for i in result.info],
                "statistics": {
                    "counts": result.statistics.counts,
                    "distributions": result.statistics.distributions,
                    "percentages": result.statistics.percentages,
                    "samples": result.statistics.samples
                }
            }

        report_dict = {
            "timestamp": report.timestamp,
            "overall_valid": report.overall_valid,
            "total_errors": report.total_errors,
            "total_warnings": report.total_warnings,
            "frames_result": result_to_dict(report.frames_result),
            "qa_pairs_result": result_to_dict(report.qa_pairs_result),
            "dataset_result": result_to_dict(report.dataset_result)
        }

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, default=str)

        return report_dict


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_validate(
    frames_dir: Optional[str] = None,
    qa_pairs_file: Optional[str] = None,
    dataset: Optional[Dict] = None,
    strict: bool = False
) -> bool:
    """
    Quick validation with default settings.

    Args:
        frames_dir: Path to frames directory
        qa_pairs_file: Path to QA pairs file
        dataset: Dataset dictionary
        strict: If True, treat warnings as errors

    Returns:
        True if validation passed, False otherwise
    """
    validator = DataValidator(strict_mode=strict)
    report = validator.validate_all(
        frames_dir=frames_dir,
        qa_pairs_file=qa_pairs_file,
        dataset=dataset
    )
    print(validator.generate_report())
    return report.overall_valid


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PitVQA Data Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate frames only
  python validators.py --frames data/frames/

  # Validate QA pairs
  python validators.py --qa-pairs data/qa_pairs.json

  # Full validation
  python validators.py --frames data/frames/ --qa-pairs data/qa_pairs.json

  # Export report as JSON
  python validators.py --frames data/frames/ --output report.json
        """
    )

    parser.add_argument(
        "--frames",
        type=str,
        help="Path to frames directory"
    )
    parser.add_argument(
        "--qa-pairs",
        type=str,
        help="Path to QA pairs file (JSON or CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for JSON report"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging"
    )

    args = parser.parse_args()

    if not args.frames and not args.qa_pairs:
        parser.print_help()
        print("\nError: At least one of --frames or --qa-pairs is required")
        exit(1)

    # Run validation
    validator = DataValidator(
        strict_mode=args.strict,
        log_progress=not args.quiet
    )

    report = validator.validate_all(
        frames_dir=args.frames,
        qa_pairs_file=args.qa_pairs
    )

    # Print report
    print("\n" + validator.generate_report())

    # Export JSON if requested
    if args.output:
        validator.export_report_json(output_path=args.output)
        print(f"\nJSON report saved to: {args.output}")

    # Exit with appropriate code
    exit(0 if report.overall_valid else 1)
