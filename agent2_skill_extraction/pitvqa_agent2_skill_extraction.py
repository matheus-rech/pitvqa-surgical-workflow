#!/usr/bin/env python3
"""
PitVQA Agent 2: Skill Extraction Pipeline

Main orchestration script for surgical skill extraction and embedding generation.
Processes frames from Agent 1 output or HuggingFace datasets to generate skill
embeddings for reinforcement learning training.

Pipeline stages:
    1. Load frames from Agent 1 output or HuggingFace
    2. Extract visual features (batch processing)
    3. Classify skills (phase, step, instruments)
    4. Generate skill embeddings
    5. Validate and save results

Usage:
    python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
        --input-dataset mmrech/pitvqa-processed \
        --output-dir data/skill_embeddings \
        --vision-model clip-vit-l-14 \
        --batch-size 32 \
        --push-to-hub mmrech/pitvqa-skills

Target outputs:
    - 109k frame embeddings (768-dim)
    - Skill classifications per frame
    - HuggingFace dataset ready for RL training

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
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

# Conditional imports with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from datasets import Dataset, DatasetDict, load_dataset, Features, Value, Sequence
    from datasets import Image as HFImage
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    Dataset = None
    DatasetDict = None

try:
    from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
    from transformers import AutoProcessor, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Logging Configuration
# =============================================================================

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

    logger = logging.getLogger("pitvqa_agent2")
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
    log_file = output_dir / f"agent2_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# Skill Vocabulary and Classification
# =============================================================================

class SkillCategory(str, Enum):
    """Enumeration of surgical skill categories."""
    PHASE = "phase"
    STEP = "step"
    INSTRUMENT = "instrument"
    ACTION = "action"
    ANATOMY = "anatomy"


@dataclass
class SkillVocabulary:
    """
    Vocabulary of surgical skills for the PitVQA dataset.

    Contains the complete taxonomy of phases, steps, instruments, and actions
    used in pituitary surgery procedures.
    """

    # Surgical phases (4 classes)
    phases: List[str] = field(default_factory=lambda: [
        "nasal_phase",
        "sellar_phase",
        "tumor_removal_phase",
        "closure_phase"
    ])

    # Surgical steps (15 classes)
    steps: List[str] = field(default_factory=lambda: [
        "septal_dissection",
        "turbinectomy",
        "sphenoidotomy",
        "posterior_septectomy",
        "sellar_floor_removal",
        "dura_opening",
        "tumor_resection",
        "hemostasis",
        "reconstruction",
        "nasal_packing",
        "visualization",
        "instrument_change",
        "suction",
        "irrigation",
        "other"
    ])

    # Surgical instruments (18 classes)
    instruments: List[str] = field(default_factory=lambda: [
        "grasper",
        "scissors",
        "cautery",
        "suction",
        "curette",
        "drill",
        "endoscope",
        "bipolar",
        "monopolar",
        "retractor",
        "forceps",
        "needle_holder",
        "scalpel",
        "speculum",
        "irrigator",
        "cotton",
        "hemostatic_agent",
        "other"
    ])

    # Surgical actions (14 classes)
    actions: List[str] = field(default_factory=lambda: [
        "cutting",
        "grasping",
        "dissecting",
        "coagulating",
        "suctioning",
        "drilling",
        "irrigating",
        "retracting",
        "inspecting",
        "hemostasis",
        "packing",
        "inserting",
        "removing",
        "idle"
    ])

    # Anatomical landmarks (8 classes)
    anatomy: List[str] = field(default_factory=lambda: [
        "septum",
        "turbinate",
        "sphenoid_sinus",
        "sella",
        "dura",
        "tumor",
        "carotid",
        "optic_nerve"
    ])

    def get_vocabulary(self, category: SkillCategory) -> List[str]:
        """Get vocabulary for a specific category."""
        mapping = {
            SkillCategory.PHASE: self.phases,
            SkillCategory.STEP: self.steps,
            SkillCategory.INSTRUMENT: self.instruments,
            SkillCategory.ACTION: self.actions,
            SkillCategory.ANATOMY: self.anatomy
        }
        return mapping[category]

    def get_num_classes(self, category: SkillCategory) -> int:
        """Get number of classes for a category."""
        return len(self.get_vocabulary(category))

    def get_total_classes(self) -> int:
        """Get total number of classes across all categories."""
        total = 0
        total += len(self.phases)
        total += len(self.steps)
        total += len(self.instruments)
        total += len(self.actions)
        total += len(self.anatomy)
        return total

    def skill_to_index(self, skill: str, category: SkillCategory) -> int:
        """Convert skill name to index."""
        vocab = self.get_vocabulary(category)
        if skill in vocab:
            return vocab.index(skill)
        return len(vocab) - 1  # Return 'other' index

    def index_to_skill(self, index: int, category: SkillCategory) -> str:
        """Convert index to skill name."""
        vocab = self.get_vocabulary(category)
        if 0 <= index < len(vocab):
            return vocab[index]
        return "unknown"

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert vocabulary to dictionary."""
        return {
            "phases": self.phases,
            "steps": self.steps,
            "instruments": self.instruments,
            "actions": self.actions,
            "anatomy": self.anatomy
        }


# =============================================================================
# Vision Encoder
# =============================================================================

class VisionEncoder(ABC):
    """Abstract base class for vision encoders."""

    @abstractmethod
    def encode(self, images: List[Any]) -> np.ndarray:
        """Encode images to embeddings."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass


class CLIPVisionEncoder(VisionEncoder):
    """
    CLIP-based vision encoder for surgical frame feature extraction.

    Supports multiple CLIP variants:
    - clip-vit-b-32: ViT-B/32 (512-dim)
    - clip-vit-b-16: ViT-B/16 (512-dim)
    - clip-vit-l-14: ViT-L/14 (768-dim)
    - clip-vit-l-14-336: ViT-L/14@336px (768-dim)
    """

    MODEL_CONFIGS = {
        "clip-vit-b-32": ("openai/clip-vit-base-patch32", 512),
        "clip-vit-b-16": ("openai/clip-vit-base-patch16", 512),
        "clip-vit-l-14": ("openai/clip-vit-large-patch14", 768),
        "clip-vit-l-14-336": ("openai/clip-vit-large-patch14-336", 768),
    }

    def __init__(
        self,
        model_name: str = "clip-vit-l-14",
        device: str = "auto",
        use_fp16: bool = True
    ):
        """
        Initialize CLIP vision encoder.

        Args:
            model_name: CLIP model variant name
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            use_fp16: Whether to use FP16 for inference
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for CLIPVisionEncoder")
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for CLIPVisionEncoder")

        self.model_name = model_name
        self.use_fp16 = use_fp16

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Get model configuration
        if model_name in self.MODEL_CONFIGS:
            hf_model_id, self.embedding_dim = self.MODEL_CONFIGS[model_name]
        else:
            hf_model_id = model_name
            self.embedding_dim = 768  # Default

        # Load model and processor
        self.processor = CLIPProcessor.from_pretrained(hf_model_id)
        self.model = CLIPVisionModel.from_pretrained(hf_model_id)

        # Move to device and set precision
        self.model = self.model.to(self.device)
        if use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self.model.eval()

    def encode(self, images: List[Any]) -> np.ndarray:
        """
        Encode a batch of images to embeddings.

        Args:
            images: List of PIL Images or numpy arrays

        Returns:
            numpy array of shape (batch_size, embedding_dim)
        """
        if not images:
            return np.array([]).reshape(0, self.embedding_dim)

        # Convert to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.use_fp16 and self.device.type == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.pooler_output

        return embeddings.cpu().float().numpy()

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def encode_single(self, image: Any) -> np.ndarray:
        """Encode a single image."""
        return self.encode([image])[0]


class TemporalEncoder:
    """
    Temporal encoder for modeling sequential dependencies in surgical videos.

    Uses a Transformer encoder to capture temporal context across frames.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        device: str = "auto"
    ):
        """
        Initialize temporal encoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            device: Device to use
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TemporalEncoder")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

        # Move to device
        self.input_projection = self.input_projection.to(self.device)
        self.pos_encoding = self.pos_encoding.to(self.device)
        self.transformer = self.transformer.to(self.device)
        self.output_projection = self.output_projection.to(self.device)

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        return pe

    def encode(self, frame_embeddings: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Encode sequence of frame embeddings with temporal context.

        Args:
            frame_embeddings: Array of shape (seq_len, input_dim) or (batch, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Temporal embeddings of shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        """
        # Convert to tensor
        if frame_embeddings.ndim == 2:
            x = torch.tensor(frame_embeddings, device=self.device).unsqueeze(0)
            squeeze_batch = True
        else:
            x = torch.tensor(frame_embeddings, device=self.device)
            squeeze_batch = False

        batch_size, seq_len, _ = x.shape

        # Project input
        x = self.input_projection(x.float())

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        if mask is not None:
            mask = torch.tensor(mask, device=self.device)

        with torch.no_grad():
            x = self.transformer(x, src_key_padding_mask=mask)

        # Project output
        x = self.output_projection(x)

        result = x.cpu().numpy()

        if squeeze_batch:
            result = result.squeeze(0)

        return result


# =============================================================================
# Skill Classifier
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    Multi-task classification head for surgical skill prediction.

    Predicts multiple skill categories simultaneously:
    - Phase classification
    - Step classification
    - Instrument detection (multi-label)
    - Action recognition
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        vocabulary: Optional[SkillVocabulary] = None,
        dropout: float = 0.3
    ):
        """
        Initialize multi-task classification head.

        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            vocabulary: Skill vocabulary (default: standard PitVQA vocabulary)
            dropout: Dropout rate
        """
        super().__init__()

        self.vocabulary = vocabulary or SkillVocabulary()

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.phase_head = nn.Linear(hidden_dim, len(self.vocabulary.phases))
        self.step_head = nn.Linear(hidden_dim, len(self.vocabulary.steps))
        self.instrument_head = nn.Linear(hidden_dim, len(self.vocabulary.instruments))
        self.action_head = nn.Linear(hidden_dim, len(self.vocabulary.actions))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input embeddings of shape (batch_size, input_dim)

        Returns:
            Dictionary of logits for each task
        """
        shared_features = self.shared(x)

        return {
            "phase": self.phase_head(shared_features),
            "step": self.step_head(shared_features),
            "instrument": self.instrument_head(shared_features),
            "action": self.action_head(shared_features)
        }


class SkillClassifier:
    """
    Surgical skill classifier combining vision features with multi-task prediction.

    Uses pretrained vision embeddings to classify surgical skills across
    multiple categories (phase, step, instrument, action).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        vocabulary: Optional[SkillVocabulary] = None,
        device: str = "auto",
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize skill classifier.

        Args:
            embedding_dim: Input embedding dimension from vision encoder
            hidden_dim: Hidden dimension for classifier
            vocabulary: Skill vocabulary
            device: Device to use
            checkpoint_path: Path to pretrained classifier weights
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SkillClassifier")

        self.vocabulary = vocabulary or SkillVocabulary()

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Initialize multi-task head
        self.classifier = MultiTaskHead(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            vocabulary=self.vocabulary
        ).to(self.device)

        # Load checkpoint if provided
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)

        self.classifier.eval()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.classifier.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.classifier.load_state_dict(checkpoint)

    def classify(
        self,
        embeddings: np.ndarray,
        return_probs: bool = False
    ) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Classify skill embeddings.

        Args:
            embeddings: Frame embeddings of shape (batch_size, embedding_dim)
            return_probs: Whether to return probabilities in addition to predictions

        Returns:
            Dictionary with predictions for each skill category
        """
        x = torch.tensor(embeddings, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            logits = self.classifier(x)

        results = {}

        for task, task_logits in logits.items():
            # Single-label classification for phase, step, action
            if task in ["phase", "step", "action"]:
                probs = F.softmax(task_logits, dim=-1)
                preds = probs.argmax(dim=-1)

                results[task] = {
                    "predictions": preds.cpu().numpy(),
                    "labels": [
                        self.vocabulary.index_to_skill(idx.item(), SkillCategory(task))
                        for idx in preds
                    ]
                }

                if return_probs:
                    results[task]["probabilities"] = probs.cpu().numpy()

            # Multi-label classification for instruments
            elif task == "instrument":
                probs = torch.sigmoid(task_logits)
                preds = (probs > 0.5).int()

                results[task] = {
                    "predictions": preds.cpu().numpy(),
                    "labels": [
                        [
                            self.vocabulary.instruments[i]
                            for i, val in enumerate(pred)
                            if val == 1
                        ]
                        for pred in preds
                    ]
                }

                if return_probs:
                    results[task]["probabilities"] = probs.cpu().numpy()

        return results

    def classify_single(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Classify a single embedding."""
        result = self.classify(embedding.reshape(1, -1))
        return {
            task: {
                "prediction": data["predictions"][0],
                "label": data["labels"][0]
            }
            for task, data in result.items()
        }


# =============================================================================
# Skill Embedding Generator
# =============================================================================

class SkillEmbeddingGenerator:
    """
    Generates skill-aware embeddings by combining visual features with skill predictions.

    Creates a unified embedding that captures both visual content and predicted
    surgical skills, suitable for downstream RL training.
    """

    def __init__(
        self,
        vision_dim: int = 768,
        skill_embedding_dim: int = 128,
        output_dim: int = 512,
        vocabulary: Optional[SkillVocabulary] = None,
        device: str = "auto"
    ):
        """
        Initialize skill embedding generator.

        Args:
            vision_dim: Dimension of visual embeddings
            skill_embedding_dim: Dimension for skill category embeddings
            output_dim: Output embedding dimension
            vocabulary: Skill vocabulary
            device: Device to use
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SkillEmbeddingGenerator")

        self.vocabulary = vocabulary or SkillVocabulary()
        self.vision_dim = vision_dim
        self.skill_embedding_dim = skill_embedding_dim
        self.output_dim = output_dim

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Skill embedding layers
        self.phase_embedding = nn.Embedding(
            len(self.vocabulary.phases), skill_embedding_dim
        ).to(self.device)

        self.step_embedding = nn.Embedding(
            len(self.vocabulary.steps), skill_embedding_dim
        ).to(self.device)

        self.action_embedding = nn.Embedding(
            len(self.vocabulary.actions), skill_embedding_dim
        ).to(self.device)

        # Instrument embedding (multi-hot projection)
        self.instrument_projection = nn.Linear(
            len(self.vocabulary.instruments), skill_embedding_dim
        ).to(self.device)

        # Combined projection
        combined_dim = vision_dim + 4 * skill_embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        ).to(self.device)

    def generate(
        self,
        visual_embeddings: np.ndarray,
        skill_predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Generate skill-aware embeddings.

        Args:
            visual_embeddings: Visual features of shape (batch_size, vision_dim)
            skill_predictions: Dictionary with predictions from SkillClassifier

        Returns:
            Skill embeddings of shape (batch_size, output_dim)
        """
        batch_size = visual_embeddings.shape[0]

        # Convert visual embeddings
        visual = torch.tensor(visual_embeddings, device=self.device, dtype=torch.float32)

        # Get skill embeddings
        phase_idx = torch.tensor(
            skill_predictions["phase"]["predictions"], device=self.device, dtype=torch.long
        )
        phase_emb = self.phase_embedding(phase_idx)

        step_idx = torch.tensor(
            skill_predictions["step"]["predictions"], device=self.device, dtype=torch.long
        )
        step_emb = self.step_embedding(step_idx)

        action_idx = torch.tensor(
            skill_predictions["action"]["predictions"], device=self.device, dtype=torch.long
        )
        action_emb = self.action_embedding(action_idx)

        # Instrument multi-hot encoding
        instrument_preds = torch.tensor(
            skill_predictions["instrument"]["predictions"], device=self.device, dtype=torch.float32
        )
        instrument_emb = self.instrument_projection(instrument_preds)

        # Concatenate all embeddings
        combined = torch.cat([
            visual,
            phase_emb,
            step_emb,
            action_emb,
            instrument_emb
        ], dim=-1)

        # Fuse embeddings
        with torch.no_grad():
            output = self.fusion(combined)

        return output.cpu().numpy()

    def get_output_dim(self) -> int:
        """Get output embedding dimension."""
        return self.output_dim


# =============================================================================
# Pipeline Configuration and Statistics
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the skill extraction pipeline."""

    # Input settings
    input_dataset: Optional[str] = None
    input_dir: Optional[Path] = None
    input_split: str = "train"

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("data/skill_embeddings"))

    # Model settings
    vision_model: str = "clip-vit-l-14"
    classifier_checkpoint: Optional[str] = None
    use_temporal: bool = False

    # Processing settings
    batch_size: int = 32
    num_workers: int = 4
    use_fp16: bool = True
    device: str = "auto"

    # HuggingFace settings
    push_to_hub: Optional[str] = None
    hf_token: Optional[str] = None

    # Checkpoint settings
    checkpoint_interval: int = 1000
    resume_from_checkpoint: bool = True

    # Streaming settings
    streaming: bool = False
    max_samples: Optional[int] = None

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if self.input_dir is not None:
            self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "input_dataset": self.input_dataset,
            "input_dir": str(self.input_dir) if self.input_dir else None,
            "input_split": self.input_split,
            "output_dir": str(self.output_dir),
            "vision_model": self.vision_model,
            "classifier_checkpoint": self.classifier_checkpoint,
            "use_temporal": self.use_temporal,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "use_fp16": self.use_fp16,
            "device": self.device,
            "push_to_hub": self.push_to_hub,
            "streaming": self.streaming,
            "max_samples": self.max_samples
        }


@dataclass
class PipelineStats:
    """Statistics tracking for the pipeline."""

    # Processing stats
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0

    # Embedding stats
    embeddings_generated: int = 0
    embedding_dim: int = 0

    # Classification stats
    phase_distribution: Dict[str, int] = field(default_factory=dict)
    step_distribution: Dict[str, int] = field(default_factory=dict)
    instrument_distribution: Dict[str, int] = field(default_factory=dict)

    # Timing stats
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    step_durations: Dict[str, float] = field(default_factory=dict)

    # Memory stats
    peak_memory_gb: float = 0.0

    def get_duration(self) -> Optional[timedelta]:
        """Get total pipeline duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "failed_frames": self.failed_frames,
            "embeddings_generated": self.embeddings_generated,
            "embedding_dim": self.embedding_dim,
            "phase_distribution": self.phase_distribution,
            "step_distribution": self.step_distribution,
            "instrument_distribution": self.instrument_distribution,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "step_durations": self.step_durations,
            "peak_memory_gb": self.peak_memory_gb,
            "total_duration_seconds": self.get_duration().total_seconds() if self.get_duration() else None
        }


@dataclass
class Checkpoint:
    """Checkpoint data for pipeline resumption."""

    step: str
    processed_indices: List[int] = field(default_factory=list)
    embeddings_file: Optional[str] = None
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


# =============================================================================
# Image Dataset for Local Processing
# =============================================================================

class ImageFolderDataset(TorchDataset):
    """Dataset for loading images from a folder structure."""

    def __init__(self, root_dir: Path, transform=None):
        """
        Initialize image folder dataset.

        Args:
            root_dir: Root directory containing images
            transform: Optional image transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Find all images
        self.image_paths = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]:
            self.image_paths.extend(self.root_dir.rglob(ext))

        self.image_paths = sorted(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, str]:
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, str(image_path)
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            return None, str(image_path)


# =============================================================================
# Main Pipeline
# =============================================================================

class SkillExtractionPipeline:
    """
    Main orchestration class for the PitVQA skill extraction pipeline.

    Coordinates the workflow:
    1. Load frames from Agent 1 output or HuggingFace
    2. Extract visual features (batch processing)
    3. Classify skills (phase, step, instruments)
    4. Generate skill embeddings
    5. Validate and save results
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the skill extraction pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.stats = PipelineStats()
        self.checkpoint = None

        # Setup output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.config.output_dir / ".checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = setup_logging(self.config.output_dir)

        # Initialize components (lazy loading)
        self.vision_encoder: Optional[VisionEncoder] = None
        self.temporal_encoder: Optional[TemporalEncoder] = None
        self.skill_classifier: Optional[SkillClassifier] = None
        self.embedding_generator: Optional[SkillEmbeddingGenerator] = None
        self.vocabulary = SkillVocabulary()

        self.logger.info("=" * 60)
        self.logger.info("PitVQA Agent 2: Skill Extraction Pipeline")
        self.logger.info("=" * 60)
        self.logger.info(f"Input dataset: {self.config.input_dataset}")
        self.logger.info(f"Input directory: {self.config.input_dir}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        self.logger.info(f"Vision model: {self.config.vision_model}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Device: {self.config.device}")

    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing pipeline components...")

        # Vision encoder
        self.logger.info(f"Loading vision encoder: {self.config.vision_model}")
        self.vision_encoder = CLIPVisionEncoder(
            model_name=self.config.vision_model,
            device=self.config.device,
            use_fp16=self.config.use_fp16
        )
        self.stats.embedding_dim = self.vision_encoder.get_embedding_dim()

        # Temporal encoder (optional)
        if self.config.use_temporal:
            self.logger.info("Initializing temporal encoder...")
            self.temporal_encoder = TemporalEncoder(
                input_dim=self.vision_encoder.get_embedding_dim(),
                device=self.config.device
            )

        # Skill classifier
        self.logger.info("Initializing skill classifier...")
        self.skill_classifier = SkillClassifier(
            embedding_dim=self.vision_encoder.get_embedding_dim(),
            vocabulary=self.vocabulary,
            device=self.config.device,
            checkpoint_path=self.config.classifier_checkpoint
        )

        # Embedding generator
        self.logger.info("Initializing skill embedding generator...")
        self.embedding_generator = SkillEmbeddingGenerator(
            vision_dim=self.vision_encoder.get_embedding_dim(),
            vocabulary=self.vocabulary,
            device=self.config.device
        )

        self.logger.info("All components initialized successfully")

    def _load_checkpoint(self) -> bool:
        """Load existing checkpoint if available."""
        if not self.config.resume_from_checkpoint:
            return False

        self.checkpoint = Checkpoint.load(self.checkpoint_dir)

        if self.checkpoint:
            self.logger.info(f"Resuming from checkpoint: step '{self.checkpoint.step}'")
            self.logger.info(f"Processed indices: {len(self.checkpoint.processed_indices)}")

            if self.checkpoint.stats:
                for key, value in self.checkpoint.stats.items():
                    if hasattr(self.stats, key):
                        setattr(self.stats, key, value)

            return True

        return False

    def _save_checkpoint(self, step: str, **kwargs) -> None:
        """Save checkpoint after processing."""
        self.checkpoint = Checkpoint(
            step=step,
            stats=self.stats.to_dict(),
            **kwargs
        )
        self.checkpoint.save(self.checkpoint_dir)
        self.logger.debug(f"Checkpoint saved: {step}")

    def _step_timer(self, step_name: str):
        """Context manager for timing pipeline steps."""
        class StepTimer:
            def __init__(self_inner, pipeline, name):
                self_inner.pipeline = pipeline
                self_inner.name = name
                self_inner.start = None

            def __enter__(self_inner):
                self_inner.start = time.time()
                self_inner.pipeline.logger.info(f"Starting step: {self_inner.name}")
                return self_inner

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                duration = time.time() - self_inner.start
                self_inner.pipeline.stats.step_durations[self_inner.name] = duration
                self_inner.pipeline.logger.info(
                    f"Completed step: {self_inner.name} ({duration:.2f}s)"
                )
                return False

        return StepTimer(self, step_name)

    def stage1_load_data(self) -> Iterator[Tuple[int, Any, Dict]]:
        """
        Stage 1: Load frames from Agent 1 output or HuggingFace.

        Yields:
            Tuples of (index, image, metadata)
        """
        with self._step_timer("data_loading"):
            if self.config.input_dataset:
                # Load from HuggingFace
                self.logger.info(f"Loading dataset from HuggingFace: {self.config.input_dataset}")

                if not HF_DATASETS_AVAILABLE:
                    raise ImportError("datasets library required for HuggingFace loading")

                dataset = load_dataset(
                    self.config.input_dataset,
                    split=self.config.input_split,
                    streaming=self.config.streaming
                )

                # Get total count if not streaming
                if not self.config.streaming:
                    self.stats.total_frames = len(dataset)
                    self.logger.info(f"Dataset size: {self.stats.total_frames} samples")

                for idx, sample in enumerate(dataset):
                    if self.config.max_samples and idx >= self.config.max_samples:
                        break

                    # Skip if already processed (checkpoint recovery)
                    if self.checkpoint and idx in self.checkpoint.processed_indices:
                        continue

                    # Extract image and metadata
                    image = sample.get("image")
                    if isinstance(image, dict):
                        # Handle Image feature format
                        image = Image.open(image["path"])

                    metadata = {
                        k: v for k, v in sample.items()
                        if k != "image"
                    }

                    yield idx, image, metadata

            elif self.config.input_dir:
                # Load from local directory
                self.logger.info(f"Loading images from directory: {self.config.input_dir}")

                dataset = ImageFolderDataset(self.config.input_dir)
                self.stats.total_frames = len(dataset)
                self.logger.info(f"Found {self.stats.total_frames} images")

                for idx in range(len(dataset)):
                    if self.config.max_samples and idx >= self.config.max_samples:
                        break

                    if self.checkpoint and idx in self.checkpoint.processed_indices:
                        continue

                    image, path = dataset[idx]
                    if image is None:
                        continue

                    metadata = {"image_path": path, "frame_id": Path(path).stem}

                    yield idx, image, metadata

            else:
                raise ValueError("Either input_dataset or input_dir must be specified")

    def stage2_extract_features(
        self,
        images: List[Any]
    ) -> np.ndarray:
        """
        Stage 2: Extract visual features using the vision encoder.

        Args:
            images: List of PIL Images

        Returns:
            Visual embeddings of shape (batch_size, embedding_dim)
        """
        return self.vision_encoder.encode(images)

    def stage3_classify_skills(
        self,
        embeddings: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        """
        Stage 3: Classify skills from visual embeddings.

        Args:
            embeddings: Visual embeddings of shape (batch_size, embedding_dim)

        Returns:
            Skill predictions for each category
        """
        return self.skill_classifier.classify(embeddings, return_probs=True)

    def stage4_generate_embeddings(
        self,
        visual_embeddings: np.ndarray,
        skill_predictions: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """
        Stage 4: Generate skill-aware embeddings.

        Args:
            visual_embeddings: Visual features
            skill_predictions: Skill classification results

        Returns:
            Skill embeddings of shape (batch_size, output_dim)
        """
        return self.embedding_generator.generate(visual_embeddings, skill_predictions)

    def stage5_save_results(
        self,
        all_embeddings: np.ndarray,
        all_predictions: List[Dict],
        all_metadata: List[Dict]
    ) -> str:
        """
        Stage 5: Validate and save results to HuggingFace dataset format.

        Args:
            all_embeddings: All generated embeddings
            all_predictions: All skill predictions
            all_metadata: All frame metadata

        Returns:
            Path or URL where dataset was saved
        """
        with self._step_timer("saving_results"):
            self.logger.info(f"Saving {len(all_embeddings)} embeddings...")

            # Save embeddings as numpy file
            embeddings_file = self.config.output_dir / "skill_embeddings.npy"
            np.save(embeddings_file, all_embeddings)
            self.logger.info(f"Saved embeddings to: {embeddings_file}")

            # Create HuggingFace dataset
            if HF_DATASETS_AVAILABLE:
                # Prepare dataset dictionary
                dataset_dict = {
                    "embedding": all_embeddings.tolist(),
                    "phase": [p["phase"]["labels"][0] for p in all_predictions],
                    "phase_prob": [float(p["phase"]["probabilities"].max()) for p in all_predictions],
                    "step": [p["step"]["labels"][0] for p in all_predictions],
                    "step_prob": [float(p["step"]["probabilities"].max()) for p in all_predictions],
                    "instruments": [p["instrument"]["labels"][0] for p in all_predictions],
                    "action": [p["action"]["labels"][0] for p in all_predictions]
                }

                # Add metadata fields
                for key in all_metadata[0].keys():
                    if key not in dataset_dict:
                        dataset_dict[key] = [m.get(key) for m in all_metadata]

                # Create dataset
                dataset = Dataset.from_dict(dataset_dict)

                # Save to disk
                dataset_path = self.config.output_dir / "hf_dataset"
                dataset.save_to_disk(str(dataset_path))
                self.logger.info(f"Saved HuggingFace dataset to: {dataset_path}")

                # Push to Hub if requested
                if self.config.push_to_hub:
                    self.logger.info(f"Pushing to HuggingFace Hub: {self.config.push_to_hub}")

                    try:
                        dataset.push_to_hub(
                            self.config.push_to_hub,
                            token=self.config.hf_token,
                            private=False
                        )
                        hub_url = f"https://huggingface.co/datasets/{self.config.push_to_hub}"
                        self.logger.info(f"Successfully pushed to: {hub_url}")
                        return hub_url

                    except Exception as e:
                        self.logger.error(f"Failed to push to Hub: {e}")
                        return str(dataset_path)

                return str(dataset_path)

            return str(embeddings_file)

    def _update_stats(self, predictions: Dict) -> None:
        """Update classification statistics."""
        # Update phase distribution
        for label in predictions["phase"]["labels"]:
            self.stats.phase_distribution[label] = \
                self.stats.phase_distribution.get(label, 0) + 1

        # Update step distribution
        for label in predictions["step"]["labels"]:
            self.stats.step_distribution[label] = \
                self.stats.step_distribution.get(label, 0) + 1

        # Update instrument distribution
        for instruments in predictions["instrument"]["labels"]:
            for inst in instruments:
                self.stats.instrument_distribution[inst] = \
                    self.stats.instrument_distribution.get(inst, 0) + 1

    def run(self) -> Dict[str, Any]:
        """
        Run the complete skill extraction pipeline.

        Returns:
            Pipeline report dictionary
        """
        self.stats.start_time = datetime.now()

        try:
            # Initialize components
            self._initialize_components()

            # Load checkpoint if available
            self._load_checkpoint()

            # Collect all results
            all_embeddings = []
            all_predictions = []
            all_metadata = []
            processed_indices = self.checkpoint.processed_indices if self.checkpoint else []

            # Process in batches
            batch_images = []
            batch_metadata = []
            batch_indices = []

            with self._step_timer("batch_processing"):
                progress = tqdm(
                    self.stage1_load_data(),
                    total=self.stats.total_frames if self.stats.total_frames > 0 else None,
                    desc="Processing frames"
                )

                for idx, image, metadata in progress:
                    batch_images.append(image)
                    batch_metadata.append(metadata)
                    batch_indices.append(idx)

                    # Process batch when full
                    if len(batch_images) >= self.config.batch_size:
                        try:
                            # Stage 2: Extract features
                            visual_embeddings = self.stage2_extract_features(batch_images)

                            # Stage 3: Classify skills
                            predictions = self.stage3_classify_skills(visual_embeddings)

                            # Stage 4: Generate embeddings
                            skill_embeddings = self.stage4_generate_embeddings(
                                visual_embeddings, predictions
                            )

                            # Collect results
                            all_embeddings.append(skill_embeddings)
                            for i in range(len(batch_images)):
                                batch_pred = {
                                    task: {
                                        "labels": [data["labels"][i]],
                                        "probabilities": data["probabilities"][i:i+1]
                                    }
                                    for task, data in predictions.items()
                                }
                                all_predictions.append(batch_pred)
                            all_metadata.extend(batch_metadata)

                            # Update stats
                            self.stats.processed_frames += len(batch_images)
                            self._update_stats(predictions)
                            processed_indices.extend(batch_indices)

                            # Save checkpoint periodically
                            if self.stats.processed_frames % self.config.checkpoint_interval == 0:
                                self._save_checkpoint(
                                    "processing",
                                    processed_indices=processed_indices
                                )

                        except Exception as e:
                            self.logger.error(f"Error processing batch: {e}")
                            self.stats.failed_frames += len(batch_images)

                        # Clear batch
                        batch_images = []
                        batch_metadata = []
                        batch_indices = []

                # Process remaining images
                if batch_images:
                    try:
                        visual_embeddings = self.stage2_extract_features(batch_images)
                        predictions = self.stage3_classify_skills(visual_embeddings)
                        skill_embeddings = self.stage4_generate_embeddings(
                            visual_embeddings, predictions
                        )

                        all_embeddings.append(skill_embeddings)
                        for i in range(len(batch_images)):
                            batch_pred = {
                                task: {
                                    "labels": [data["labels"][i]],
                                    "probabilities": data["probabilities"][i:i+1]
                                }
                                for task, data in predictions.items()
                            }
                            all_predictions.append(batch_pred)
                        all_metadata.extend(batch_metadata)

                        self.stats.processed_frames += len(batch_images)
                        self._update_stats(predictions)

                    except Exception as e:
                        self.logger.error(f"Error processing final batch: {e}")
                        self.stats.failed_frames += len(batch_images)

            # Concatenate all embeddings
            if all_embeddings:
                all_embeddings = np.concatenate(all_embeddings, axis=0)
                self.stats.embeddings_generated = len(all_embeddings)
            else:
                all_embeddings = np.array([])

            # Stage 5: Save results
            output_location = self.stage5_save_results(
                all_embeddings, all_predictions, all_metadata
            )

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

        finally:
            self.stats.end_time = datetime.now()

        # Generate and save report
        report = self.generate_report()
        self._print_summary(report)

        return report

    def generate_report(self) -> Dict[str, Any]:
        """Generate final pipeline report."""
        report = {
            "pipeline": "PitVQA Agent 2: Skill Extraction",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "summary": {
                "total_frames": self.stats.total_frames,
                "processed_frames": self.stats.processed_frames,
                "failed_frames": self.stats.failed_frames,
                "embeddings_generated": self.stats.embeddings_generated,
                "embedding_dimension": self.stats.embedding_dim,
                "total_duration": str(self.stats.get_duration()) if self.stats.get_duration() else None
            }
        }

        # Save report
        report_file = self.config.output_dir / "pipeline_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_file}")

        return report

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print formatted summary."""
        summary = report["summary"]

        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Total frames: {summary['total_frames']}")
        self.logger.info(f"Processed frames: {summary['processed_frames']}")
        self.logger.info(f"Failed frames: {summary['failed_frames']}")
        self.logger.info(f"Embeddings generated: {summary['embeddings_generated']}")
        self.logger.info(f"Embedding dimension: {summary['embedding_dimension']}")
        self.logger.info(f"Duration: {summary['total_duration']}")
        self.logger.info("")
        self.logger.info("Phase distribution:")
        for phase, count in sorted(self.stats.phase_distribution.items()):
            self.logger.info(f"  {phase}: {count}")
        self.logger.info("")
        self.logger.info("Step distribution (top 5):")
        sorted_steps = sorted(
            self.stats.step_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for step, count in sorted_steps:
            self.logger.info(f"  {step}: {count}")
        self.logger.info("=" * 60)


# =============================================================================
# CLI Interface
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PitVQA Agent 2: Skill Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process from HuggingFace dataset
  python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \\
      --input-dataset mmrech/pitvqa-processed \\
      --output-dir data/skill_embeddings \\
      --vision-model clip-vit-l-14 \\
      --batch-size 32

  # Process from local directory
  python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \\
      --input-dir data/processed/frames \\
      --output-dir data/skill_embeddings \\
      --vision-model clip-vit-l-14

  # Push to HuggingFace Hub
  python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \\
      --input-dataset mmrech/pitvqa-processed \\
      --output-dir data/skill_embeddings \\
      --push-to-hub mmrech/pitvqa-skills

  # Use streaming for large datasets
  python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \\
      --input-dataset mmrech/pitvqa-processed \\
      --output-dir data/skill_embeddings \\
      --streaming \\
      --max-samples 10000
        """
    )

    # Input settings
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dataset",
        type=str,
        help="HuggingFace dataset ID (e.g., 'mmrech/pitvqa-processed')"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Path to directory containing frame images"
    )

    parser.add_argument(
        "--input-split",
        type=str,
        default="train",
        help="Dataset split to process (default: train)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for skill embeddings"
    )

    # Model settings
    parser.add_argument(
        "--vision-model",
        type=str,
        default="clip-vit-l-14",
        choices=["clip-vit-b-32", "clip-vit-b-16", "clip-vit-l-14", "clip-vit-l-14-336"],
        help="Vision encoder model (default: clip-vit-l-14)"
    )

    parser.add_argument(
        "--classifier-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained skill classifier checkpoint"
    )

    parser.add_argument(
        "--use-temporal",
        action="store_true",
        help="Enable temporal encoding for video sequences"
    )

    # Processing settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (default: auto)"
    )

    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 inference"
    )

    # HuggingFace settings
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo ID to push dataset to"
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (uses HF_TOKEN env var if not provided)"
    )

    # Streaming settings
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N samples (default: 1000)"
    )

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

    # Check required libraries
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    if not PIL_AVAILABLE:
        print("ERROR: Pillow is required. Install with: pip install pillow")
        sys.exit(1)

    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers is required. Install with: pip install transformers")
        sys.exit(1)

    # Get HuggingFace token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Create configuration
    config = PipelineConfig(
        input_dataset=args.input_dataset,
        input_dir=Path(args.input_dir) if args.input_dir else None,
        input_split=args.input_split,
        output_dir=Path(args.output_dir),
        vision_model=args.vision_model,
        classifier_checkpoint=args.classifier_checkpoint,
        use_temporal=args.use_temporal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_fp16=not args.no_fp16,
        device=args.device,
        push_to_hub=args.push_to_hub,
        hf_token=hf_token,
        checkpoint_interval=args.checkpoint_interval,
        resume_from_checkpoint=not args.no_resume,
        streaming=args.streaming,
        max_samples=args.max_samples
    )

    # Run pipeline
    pipeline = SkillExtractionPipeline(config)

    try:
        report = pipeline.run()

        # Exit with success
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
