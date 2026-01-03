"""
Agent 2: PitVQA Skill Extraction Pipeline

This module handles surgical skill extraction and embedding generation for the
PitVQA Visual Question Answering project. It processes frames from Agent 1 to
generate skill embeddings suitable for reinforcement learning training.

Components:
- VisionEncoder: Abstract base class for visual feature extraction
- CLIPVisionEncoder: CLIP-based vision encoder implementation
- TemporalEncoder: Transformer-based temporal context modeling
- SkillClassifier: Multi-task surgical skill classification
- MultiTaskHead: Neural network head for multi-task prediction
- SkillEmbeddingGenerator: Skill-aware embedding fusion
- SkillVocabulary: Surgical skill taxonomy

Usage:
    # Full pipeline execution
    python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
        --input-dataset mmrech/pitvqa-processed \
        --output-dir data/skill_embeddings \
        --vision-model clip-vit-l-14 \
        --batch-size 32 \
        --push-to-hub mmrech/pitvqa-skills

    # Programmatic usage
    from agent2_skill_extraction import (
        CLIPVisionEncoder,
        SkillClassifier,
        SkillEmbeddingGenerator,
        SkillVocabulary
    )

    # Initialize components
    vocabulary = SkillVocabulary()
    encoder = CLIPVisionEncoder(model_name="clip-vit-l-14")
    classifier = SkillClassifier(embedding_dim=768, vocabulary=vocabulary)
    generator = SkillEmbeddingGenerator(vision_dim=768, vocabulary=vocabulary)

    # Process frames
    visual_embeddings = encoder.encode(images)
    predictions = classifier.classify(visual_embeddings)
    skill_embeddings = generator.generate(visual_embeddings, predictions)

Target outputs:
    - 109k frame embeddings (512-dim skill embeddings)
    - Skill classifications per frame (phase, step, instruments, action)
    - HuggingFace dataset ready for RL training
"""

from .pitvqa_agent2_skill_extraction import (
    # Vision Encoders
    VisionEncoder,
    CLIPVisionEncoder,
    TemporalEncoder,

    # Skill Classification
    SkillClassifier,
    MultiTaskHead,

    # Embedding Generation
    SkillEmbeddingGenerator,
    SkillVocabulary,
    SkillCategory,

    # Pipeline
    SkillExtractionPipeline,
    PipelineConfig,
    PipelineStats,
    Checkpoint,

    # Utilities
    setup_logging,
    ImageFolderDataset,
)

__version__ = "1.0.0"
__author__ = "PitVQA Team"

__all__ = [
    # Vision Encoders
    "VisionEncoder",
    "CLIPVisionEncoder",
    "TemporalEncoder",

    # Skill Classification
    "SkillClassifier",
    "MultiTaskHead",

    # Embedding Generation
    "SkillEmbeddingGenerator",
    "SkillVocabulary",
    "SkillCategory",

    # Pipeline
    "SkillExtractionPipeline",
    "PipelineConfig",
    "PipelineStats",
    "Checkpoint",

    # Utilities
    "setup_logging",
    "ImageFolderDataset",
]
