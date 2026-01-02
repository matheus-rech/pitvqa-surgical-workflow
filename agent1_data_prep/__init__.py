"""
Agent 1: PitVQA Data Preparation Pipeline

This module handles the complete data preparation workflow for the PitVQA
surgical Visual Question Answering project.

Components:
- frame_extractor: Extract and filter frames from surgical videos
- qa_generator: Generate QA pairs from annotations
- dataset_builder: Create HuggingFace datasets
- validators: Validate pipeline outputs

Usage:
    # Frame extraction
    python -m agent1_data_prep.frame_extractor \
        data/raw/videos \
        data/processed/frames \
        --fps 1.0 \
        --blur-threshold 100

    # QA generation
    python -m agent1_data_prep.qa_generator \
        --annotation-dir data/annotations \
        --frames-dir data/processed \
        --output data/qa_pairs.json

    # Validation
    python -m agent1_data_prep.validators \
        --frames data/processed/frames \
        --qa-pairs data/qa_pairs.json

Target outputs:
    - ~109,173 frames from 25 videos
    - ~884,242 QA pairs
    - HuggingFace dataset ready for training
"""

from .frame_extractor import (
    FrameExtractor,
    ExtractionStats,
    BatchExtractionStats,
    setup_logging
)

from .qa_generator import (
    QAGenerator,
    QAPair,
    QuestionType,
    AnnotationSchema,
    QuestionTemplates,
    create_sample_annotations
)

from .validators import (
    DataValidator,
    ValidationResult,
    ValidationReport,
    ValidationIssue,
    ValidationSeverity,
    ValidationStatistics,
    DatasetSpecs,
    ImageValidator,
    quick_validate,
)

from .dataset_builder import (
    DatasetBuilder,
    load_dataset_from_disk,
)

__version__ = "1.0.0"
__author__ = "PitVQA Team"
__all__ = [
    # Frame extraction
    "FrameExtractor",
    "ExtractionStats",
    "BatchExtractionStats",
    "setup_logging",
    # QA generation
    "QAGenerator",
    "QAPair",
    "QuestionType",
    "AnnotationSchema",
    "QuestionTemplates",
    "create_sample_annotations",
    # Dataset builder
    "DatasetBuilder",
    "load_dataset_from_disk",
    # Validators
    "DataValidator",
    "ValidationResult",
    "ValidationReport",
    "ValidationIssue",
    "ValidationSeverity",
    "ValidationStatistics",
    "DatasetSpecs",
    "ImageValidator",
    "quick_validate",
]
