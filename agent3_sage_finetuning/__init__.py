"""
Agent 3: SAGE/Molmo Fine-tuning Pipeline for PitVQA Surgical Video Understanding

This module provides tools for fine-tuning SAGE and Molmo vision-language models
on pituitary surgery videos for:
- Spatial grounding (pointing to anatomical structures)
- Temporal grounding (surgical phase/step recognition)
- Instrument detection and tracking
- Visual question answering about surgical procedures

Components:
    - data_converter: Converts Agent 1/2 outputs to Molmo training format
    - pointing_annotator: Generates spatial pointing annotations
    - hf_skills_trainer: HuggingFace Skills integration for training
    - surgical_evaluator: Evaluation metrics for surgical VQA
    - pitvqa_agent3_sage_finetuning: Main pipeline orchestration

Usage:
    # Full pipeline
    python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \\
        --input-dataset mmrech/pitvqa-processed \\
        --output-dir outputs/pitvqa-sage \\
        --method sft \\
        --push-to-hub

    # Generate HF Skills training prompt
    python -m agent3_sage_finetuning.hf_skills_trainer \\
        --dataset mmrech/pitvqa-sage-sft \\
        --output-name mmrech/pitvqa-sage-surgical \\
        --method sft \\
        --generate-prompt

Reference:
    - SAGE: https://arxiv.org/abs/2512.13874
    - Molmo: https://allenai.org/blog/molmo2
    - HF Skills: https://huggingface.co/blog/hf-skills-training
"""

__version__ = "0.1.0"
__author__ = "PitVQA Team"

# Data conversion
from .data_converter import (
    MolmoDataConverter,
    TrainingMethod,
    PointAnnotation,
    BoundingBox,
    TemporalSegment,
    SurgicalVQASample,
    SurgicalAnatomyVocabulary,
    convert_pitvqa_to_molmo,
)

# Pointing annotation
from .pointing_annotator import (
    SurgicalPointingAnnotator,
    GroundingDINOAnnotator,
    VLMPseudoLabeler,
    DetectionResult,
    FrameAnnotation,
    generate_pointing_dataset,
)

# Training
from .hf_skills_trainer import (
    HFSkillsTrainer,
    TrainingConfig,
    GRPORewardConfig,
    SurgicalRewardFunctions,
    HardwareTier,
    ModelSize,
    create_surgical_vqa_trainer,
)

# Evaluation
from .surgical_evaluator import (
    SurgicalVQAEvaluator,
    PointingEvalResult,
    ClassificationEvalResult,
    VQAEvalResult,
    EvaluationReport,
    run_evaluation,
)

# Pipeline
from .pitvqa_agent3_sage_finetuning import (
    PitVQASAGEPipeline,
    PipelineConfig,
    PipelineStage,
)

__all__ = [
    # Version
    "__version__",

    # Data conversion
    "MolmoDataConverter",
    "TrainingMethod",
    "PointAnnotation",
    "BoundingBox",
    "TemporalSegment",
    "SurgicalVQASample",
    "SurgicalAnatomyVocabulary",
    "convert_pitvqa_to_molmo",

    # Pointing annotation
    "SurgicalPointingAnnotator",
    "GroundingDINOAnnotator",
    "VLMPseudoLabeler",
    "DetectionResult",
    "FrameAnnotation",
    "generate_pointing_dataset",

    # Training
    "HFSkillsTrainer",
    "TrainingConfig",
    "GRPORewardConfig",
    "SurgicalRewardFunctions",
    "HardwareTier",
    "ModelSize",
    "create_surgical_vqa_trainer",

    # Evaluation
    "SurgicalVQAEvaluator",
    "PointingEvalResult",
    "ClassificationEvalResult",
    "VQAEvalResult",
    "EvaluationReport",
    "run_evaluation",

    # Pipeline
    "PitVQASAGEPipeline",
    "PipelineConfig",
    "PipelineStage",
]
