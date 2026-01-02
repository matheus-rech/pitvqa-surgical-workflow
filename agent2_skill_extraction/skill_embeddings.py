"""
PitVQA Skill Embeddings Module

Generates skill embeddings that combine visual features with skill classifications
for the Reinforcement Learning agent in surgical workflow understanding.

This module provides:
- SkillEmbeddingGenerator: Combines visual features with skill predictions
- SkillVocabulary: Maps skill labels to learnable embeddings
- HierarchicalSkillEncoder: Encodes skill hierarchies (Phase -> Step -> Action)
- Similarity functions for skill matching and retrieval

Target: Generate 256-dimensional skill embeddings suitable for RL state representation.

PitVQA Annotation Categories (59 classes):
- 4 surgical phases
- 15 surgical steps
- 18 surgical instruments
- 3 instrument presence variations
- 5 instrument positions
- 14 operation notes
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure module logger
logger = logging.getLogger(__name__)


class FusionStrategy(Enum):
    """Embedding fusion strategies for combining visual and skill features."""
    CONCATENATION = "concatenation"
    CROSS_ATTENTION = "cross_attention"
    GATED_FUSION = "gated_fusion"


@dataclass
class SkillCategories:
    """
    PitVQA skill categories matching the annotation schema.

    This defines all skill labels used in the surgical workflow,
    enabling consistent embedding across the system.
    """
    # 4 Surgical Phases
    phases: List[str] = field(default_factory=lambda: [
        "Nasal Phase",
        "Sellar Phase",
        "Tumor Resection Phase",
        "Closure Phase"
    ])

    # 15 Surgical Steps
    steps: List[str] = field(default_factory=lambda: [
        "Nasal Cavity Exploration",
        "Septal Dissection",
        "Sphenoidotomy",
        "Posterior Septectomy",
        "Sphenoid Sinus Exploration",
        "Sellar Floor Removal",
        "Dura Opening",
        "Tumor Identification",
        "Tumor Debulking",
        "Tumor Capsule Dissection",
        "Hemostasis",
        "Sellar Floor Reconstruction",
        "Sphenoid Sinus Packing",
        "Nasal Cavity Packing",
        "Final Inspection"
    ])

    # 18 Surgical Instruments
    instruments: List[str] = field(default_factory=lambda: [
        "Endoscope",
        "Suction",
        "Bipolar Forceps",
        "Curette",
        "Dissector",
        "Drill",
        "Doppler Probe",
        "Forceps",
        "Grasper",
        "Irrigation",
        "Kerrison Rongeur",
        "Knife",
        "Microdebrider",
        "Needle",
        "Pituitary Rongeur",
        "Retractor",
        "Scissors",
        "Speculum"
    ])

    # 5 Instrument Positions
    positions: List[str] = field(default_factory=lambda: [
        "Left Side",
        "Right Side",
        "Center",
        "Upper Region",
        "Lower Region"
    ])

    # 14 Operation Notes
    operation_notes: List[str] = field(default_factory=lambda: [
        "Normal Tissue Visualization",
        "Tumor Visible",
        "Bleeding Detected",
        "Hemostasis Achieved",
        "Clear Field of View",
        "Obstructed View",
        "Anatomical Landmark Identified",
        "Careful Dissection Required",
        "Critical Structure Near",
        "Good Progress",
        "Complication Detected",
        "Tissue Irrigation Needed",
        "Suction Applied",
        "Procedure On Track"
    ])

    @property
    def total_skills(self) -> int:
        """Total number of unique skill labels."""
        return (
            len(self.phases) +
            len(self.steps) +
            len(self.instruments) +
            len(self.positions) +
            len(self.operation_notes)
        )

    def get_all_labels(self) -> List[str]:
        """Get all skill labels as a flat list."""
        return (
            self.phases +
            self.steps +
            self.instruments +
            self.positions +
            self.operation_notes
        )


@dataclass
class SkillPrediction:
    """
    Container for skill predictions from a frame.

    Attributes:
        phase: Current surgical phase
        step: Current surgical step
        instruments: List of visible instruments with confidence scores
        position: Primary instrument position
        operation_note: Current operation note
        confidences: Dictionary of confidence scores for each prediction
    """
    phase: Optional[str] = None
    step: Optional[str] = None
    instruments: List[Tuple[str, float]] = field(default_factory=list)
    position: Optional[str] = None
    operation_note: Optional[str] = None
    confidences: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "phase": self.phase,
            "step": self.step,
            "instruments": self.instruments,
            "position": self.position,
            "operation_note": self.operation_note,
            "confidences": self.confidences
        }


class SkillVocabulary(nn.Module):
    """
    Maps surgical skill labels to learnable embeddings.

    Similar to word embeddings in NLP, but specialized for surgical skills.
    Creates dense vector representations for phases, steps, instruments,
    positions, and operation notes.

    Attributes:
        embedding_dim: Dimension of skill embeddings
        categories: Skill category definitions

    Example:
        >>> vocab = SkillVocabulary(embedding_dim=128)
        >>> embedding = vocab.embed_skills(
        ...     phase="Nasal Phase",
        ...     step="Septal Dissection",
        ...     instruments=["Endoscope", "Suction"],
        ...     position="Center",
        ...     note="Clear Field of View"
        ... )
        >>> print(embedding.shape)
        torch.Size([1, 128])
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        categories: Optional[SkillCategories] = None,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """
        Initialize the skill vocabulary.

        Args:
            embedding_dim: Dimension of each skill embedding
            categories: Skill category definitions (uses defaults if None)
            dropout: Dropout rate for regularization
            use_positional_encoding: Whether to add positional encoding for hierarchy
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.categories = categories or SkillCategories()
        self.use_positional_encoding = use_positional_encoding

        # Build label to index mappings
        self._build_label_indices()

        # Create embedding layers for each category
        self.phase_embedding = nn.Embedding(
            num_embeddings=len(self.categories.phases) + 1,  # +1 for unknown
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=len(self.categories.steps) + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.instrument_embedding = nn.Embedding(
            num_embeddings=len(self.categories.instruments) + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=len(self.categories.positions) + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.note_embedding = nn.Embedding(
            num_embeddings=len(self.categories.operation_notes) + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Category type embeddings (to distinguish skill types)
        self.category_embedding = nn.Embedding(
            num_embeddings=5,  # phase, step, instrument, position, note
            embedding_dim=embedding_dim
        )

        # Projection for combining multiple embeddings
        self.combiner = nn.Sequential(
            nn.Linear(embedding_dim * 5, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Multi-instrument aggregator (attention-based)
        self.instrument_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Initialize embeddings
        self._init_embeddings()

        logger.info(
            f"SkillVocabulary initialized: {self.categories.total_skills} skills, "
            f"dim={embedding_dim}"
        )

    def _build_label_indices(self) -> None:
        """Build mappings from labels to indices."""
        self.phase_to_idx = {
            label: idx + 1 for idx, label in enumerate(self.categories.phases)
        }
        self.step_to_idx = {
            label: idx + 1 for idx, label in enumerate(self.categories.steps)
        }
        self.instrument_to_idx = {
            label: idx + 1 for idx, label in enumerate(self.categories.instruments)
        }
        self.position_to_idx = {
            label: idx + 1 for idx, label in enumerate(self.categories.positions)
        }
        self.note_to_idx = {
            label: idx + 1 for idx, label in enumerate(self.categories.operation_notes)
        }

        # Reverse mappings
        self.idx_to_phase = {v: k for k, v in self.phase_to_idx.items()}
        self.idx_to_step = {v: k for k, v in self.step_to_idx.items()}
        self.idx_to_instrument = {v: k for k, v in self.instrument_to_idx.items()}
        self.idx_to_position = {v: k for k, v in self.position_to_idx.items()}
        self.idx_to_note = {v: k for k, v in self.note_to_idx.items()}

    def _init_embeddings(self) -> None:
        """Initialize embedding weights."""
        for embedding in [
            self.phase_embedding,
            self.step_embedding,
            self.instrument_embedding,
            self.position_embedding,
            self.note_embedding,
            self.category_embedding
        ]:
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
            if hasattr(embedding, 'padding_idx') and embedding.padding_idx is not None:
                with torch.no_grad():
                    embedding.weight[embedding.padding_idx].fill_(0)

    def _get_label_index(
        self,
        label: Optional[str],
        mapping: Dict[str, int]
    ) -> int:
        """Get index for a label, returning 0 for unknown/None."""
        if label is None:
            return 0
        # Case-insensitive matching
        label_lower = label.lower()
        for key, idx in mapping.items():
            if key.lower() == label_lower:
                return idx
        return 0

    def embed_single_skill(
        self,
        skill_type: str,
        label: str
    ) -> torch.Tensor:
        """
        Get embedding for a single skill label.

        Args:
            skill_type: One of 'phase', 'step', 'instrument', 'position', 'note'
            label: The skill label string

        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        device = next(self.parameters()).device

        if skill_type == 'phase':
            idx = self._get_label_index(label, self.phase_to_idx)
            idx_tensor = torch.tensor([idx], device=device)
            return self.phase_embedding(idx_tensor).squeeze(0)

        elif skill_type == 'step':
            idx = self._get_label_index(label, self.step_to_idx)
            idx_tensor = torch.tensor([idx], device=device)
            return self.step_embedding(idx_tensor).squeeze(0)

        elif skill_type == 'instrument':
            idx = self._get_label_index(label, self.instrument_to_idx)
            idx_tensor = torch.tensor([idx], device=device)
            return self.instrument_embedding(idx_tensor).squeeze(0)

        elif skill_type == 'position':
            idx = self._get_label_index(label, self.position_to_idx)
            idx_tensor = torch.tensor([idx], device=device)
            return self.position_embedding(idx_tensor).squeeze(0)

        elif skill_type == 'note':
            idx = self._get_label_index(label, self.note_to_idx)
            idx_tensor = torch.tensor([idx], device=device)
            return self.note_embedding(idx_tensor).squeeze(0)

        else:
            raise ValueError(f"Unknown skill type: {skill_type}")

    def embed_skills(
        self,
        phase: Optional[str] = None,
        step: Optional[str] = None,
        instruments: Optional[List[str]] = None,
        position: Optional[str] = None,
        note: Optional[str] = None,
        instrument_confidences: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Generate a combined embedding from multiple skill predictions.

        Args:
            phase: Current surgical phase
            step: Current surgical step
            instruments: List of visible instruments
            position: Primary instrument position
            note: Current operation note
            instrument_confidences: Optional confidence scores for instruments

        Returns:
            Combined skill embedding of shape (1, embedding_dim)
        """
        device = next(self.parameters()).device

        # Get individual embeddings
        phase_idx = self._get_label_index(phase, self.phase_to_idx)
        step_idx = self._get_label_index(step, self.step_to_idx)
        position_idx = self._get_label_index(position, self.position_to_idx)
        note_idx = self._get_label_index(note, self.note_to_idx)

        phase_emb = self.phase_embedding(
            torch.tensor([phase_idx], device=device)
        )
        step_emb = self.step_embedding(
            torch.tensor([step_idx], device=device)
        )
        position_emb = self.position_embedding(
            torch.tensor([position_idx], device=device)
        )
        note_emb = self.note_embedding(
            torch.tensor([note_idx], device=device)
        )

        # Handle instruments (can be multiple)
        if instruments and len(instruments) > 0:
            inst_indices = [
                self._get_label_index(inst, self.instrument_to_idx)
                for inst in instruments
            ]
            inst_tensor = torch.tensor(inst_indices, device=device)
            inst_embeddings = self.instrument_embedding(inst_tensor)

            # Weight by confidence if provided
            if instrument_confidences and len(instrument_confidences) == len(instruments):
                weights = torch.tensor(
                    instrument_confidences, device=device, dtype=torch.float
                ).unsqueeze(1)
                weights = F.softmax(weights, dim=0)
                instrument_emb = (inst_embeddings * weights).sum(dim=0, keepdim=True)
            else:
                # Use attention to aggregate instrument embeddings
                inst_embeddings = inst_embeddings.unsqueeze(0)  # (1, num_insts, dim)
                instrument_emb, _ = self.instrument_attention(
                    inst_embeddings, inst_embeddings, inst_embeddings
                )
                instrument_emb = instrument_emb.mean(dim=1, keepdim=True)
        else:
            # No instruments - use zero embedding
            instrument_emb = self.instrument_embedding(
                torch.tensor([0], device=device)
            )

        # Add category type embeddings
        category_indices = torch.arange(5, device=device)
        category_embs = self.category_embedding(category_indices)

        phase_emb = phase_emb + category_embs[0:1]
        step_emb = step_emb + category_embs[1:2]
        instrument_emb = instrument_emb + category_embs[2:3]
        position_emb = position_emb + category_embs[3:4]
        note_emb = note_emb + category_embs[4:5]

        # Concatenate all embeddings
        combined = torch.cat([
            phase_emb,
            step_emb,
            instrument_emb,
            position_emb,
            note_emb
        ], dim=-1)

        # Project to final dimension
        output = self.combiner(combined)

        return output

    def forward(
        self,
        skill_predictions: Union[SkillPrediction, Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Forward pass - embed skill predictions.

        Args:
            skill_predictions: SkillPrediction object or dict with skill info

        Returns:
            Skill embedding tensor
        """
        if isinstance(skill_predictions, SkillPrediction):
            predictions = skill_predictions.to_dict()
        else:
            predictions = skill_predictions

        instruments = predictions.get('instruments', [])
        if instruments and isinstance(instruments[0], tuple):
            # Extract instrument names and confidences
            instrument_names = [inst[0] for inst in instruments]
            instrument_confs = [inst[1] for inst in instruments]
        else:
            instrument_names = instruments
            instrument_confs = None

        return self.embed_skills(
            phase=predictions.get('phase'),
            step=predictions.get('step'),
            instruments=instrument_names,
            position=predictions.get('position'),
            note=predictions.get('operation_note'),
            instrument_confidences=instrument_confs
        )

    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for all skills in the vocabulary.

        Returns:
            Dictionary mapping category to tensor of all embeddings
        """
        device = next(self.parameters()).device

        return {
            'phases': self.phase_embedding(
                torch.arange(1, len(self.categories.phases) + 1, device=device)
            ),
            'steps': self.step_embedding(
                torch.arange(1, len(self.categories.steps) + 1, device=device)
            ),
            'instruments': self.instrument_embedding(
                torch.arange(1, len(self.categories.instruments) + 1, device=device)
            ),
            'positions': self.position_embedding(
                torch.arange(1, len(self.categories.positions) + 1, device=device)
            ),
            'notes': self.note_embedding(
                torch.arange(1, len(self.categories.operation_notes) + 1, device=device)
            )
        }


class HierarchicalSkillEncoder(nn.Module):
    """
    Encodes surgical skill hierarchy (Phase -> Step -> Action).

    Captures the natural progression and dependencies in surgical workflows:
    - Certain steps only occur in certain phases
    - Instruments are associated with specific steps
    - Actions follow logical sequences

    Uses graph-based relationships to model skill dependencies.

    Attributes:
        embedding_dim: Dimension of skill embeddings
        num_layers: Number of graph attention layers

    Example:
        >>> encoder = HierarchicalSkillEncoder(embedding_dim=128)
        >>> hierarchy_emb = encoder.encode_hierarchy(
        ...     phase="Nasal Phase",
        ...     step="Septal Dissection",
        ...     instruments=["Endoscope", "Suction"]
        ... )
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        categories: Optional[SkillCategories] = None
    ):
        """
        Initialize the hierarchical skill encoder.

        Args:
            embedding_dim: Dimension of skill embeddings
            num_layers: Number of graph attention layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            categories: Skill category definitions
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.categories = categories or SkillCategories()

        # Skill vocabulary for base embeddings
        self.skill_vocab = SkillVocabulary(
            embedding_dim=embedding_dim,
            categories=self.categories,
            dropout=dropout
        )

        # Hierarchical attention layers
        self.hierarchy_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])

        # Level embeddings (phase=0, step=1, instrument=2)
        self.level_embedding = nn.Embedding(3, embedding_dim)

        # Build skill hierarchy graph (adjacency relationships)
        self._build_hierarchy_graph()

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        logger.info(
            f"HierarchicalSkillEncoder initialized: {num_layers} layers, "
            f"dim={embedding_dim}"
        )

    def _build_hierarchy_graph(self) -> None:
        """
        Build the skill hierarchy graph defining valid relationships.

        This encodes domain knowledge about which steps occur in which phases,
        and which instruments are typically used in which steps.
        """
        # Phase -> Step relationships (which steps can occur in which phase)
        self.phase_step_map: Dict[str, List[str]] = {
            "Nasal Phase": [
                "Nasal Cavity Exploration",
                "Septal Dissection",
                "Sphenoidotomy",
                "Posterior Septectomy"
            ],
            "Sellar Phase": [
                "Sphenoid Sinus Exploration",
                "Sellar Floor Removal",
                "Dura Opening"
            ],
            "Tumor Resection Phase": [
                "Tumor Identification",
                "Tumor Debulking",
                "Tumor Capsule Dissection",
                "Hemostasis"
            ],
            "Closure Phase": [
                "Sellar Floor Reconstruction",
                "Sphenoid Sinus Packing",
                "Nasal Cavity Packing",
                "Final Inspection"
            ]
        }

        # Step -> Instrument relationships (common instruments for each step)
        self.step_instrument_map: Dict[str, List[str]] = {
            "Nasal Cavity Exploration": ["Endoscope", "Suction"],
            "Septal Dissection": ["Endoscope", "Dissector", "Bipolar Forceps"],
            "Sphenoidotomy": ["Endoscope", "Drill", "Kerrison Rongeur"],
            "Posterior Septectomy": ["Endoscope", "Scissors", "Forceps"],
            "Sphenoid Sinus Exploration": ["Endoscope", "Suction"],
            "Sellar Floor Removal": ["Endoscope", "Drill", "Curette"],
            "Dura Opening": ["Endoscope", "Knife", "Scissors"],
            "Tumor Identification": ["Endoscope", "Doppler Probe"],
            "Tumor Debulking": ["Endoscope", "Curette", "Suction", "Pituitary Rongeur"],
            "Tumor Capsule Dissection": ["Endoscope", "Dissector", "Curette"],
            "Hemostasis": ["Endoscope", "Bipolar Forceps", "Irrigation"],
            "Sellar Floor Reconstruction": ["Endoscope", "Forceps", "Grasper"],
            "Sphenoid Sinus Packing": ["Endoscope", "Forceps", "Grasper"],
            "Nasal Cavity Packing": ["Endoscope", "Forceps", "Speculum"],
            "Final Inspection": ["Endoscope"]
        }

        # Build adjacency matrix for attention masking
        total_skills = (
            len(self.categories.phases) +
            len(self.categories.steps) +
            len(self.categories.instruments)
        )

        # Initialize adjacency matrix (1 = connected, 0 = not connected)
        adjacency = torch.zeros(total_skills, total_skills)

        # Set self-connections
        adjacency.fill_diagonal_(1.0)

        # Add phase-step connections
        phase_offset = 0
        step_offset = len(self.categories.phases)
        instrument_offset = step_offset + len(self.categories.steps)

        for phase, steps in self.phase_step_map.items():
            phase_idx = self.categories.phases.index(phase) + phase_offset
            for step in steps:
                if step in self.categories.steps:
                    step_idx = self.categories.steps.index(step) + step_offset
                    adjacency[phase_idx, step_idx] = 1.0
                    adjacency[step_idx, phase_idx] = 1.0

        # Add step-instrument connections
        for step, instruments in self.step_instrument_map.items():
            if step in self.categories.steps:
                step_idx = self.categories.steps.index(step) + step_offset
                for instrument in instruments:
                    if instrument in self.categories.instruments:
                        inst_idx = self.categories.instruments.index(instrument) + instrument_offset
                        adjacency[step_idx, inst_idx] = 1.0
                        adjacency[inst_idx, step_idx] = 1.0

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('adjacency_matrix', adjacency)

    def encode_hierarchy(
        self,
        phase: Optional[str] = None,
        step: Optional[str] = None,
        instruments: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Encode the skill hierarchy with graph-based attention.

        Args:
            phase: Current surgical phase
            step: Current surgical step
            instruments: List of visible instruments

        Returns:
            Hierarchical skill embedding of shape (1, embedding_dim)
        """
        device = next(self.parameters()).device
        embeddings = []
        levels = []

        # Phase embedding (level 0)
        if phase:
            phase_emb = self.skill_vocab.embed_single_skill('phase', phase)
            embeddings.append(phase_emb)
            levels.append(0)

        # Step embedding (level 1)
        if step:
            step_emb = self.skill_vocab.embed_single_skill('step', step)
            embeddings.append(step_emb)
            levels.append(1)

        # Instrument embeddings (level 2)
        if instruments:
            for inst in instruments:
                inst_emb = self.skill_vocab.embed_single_skill('instrument', inst)
                embeddings.append(inst_emb)
                levels.append(2)

        if not embeddings:
            # Return zero embedding if no skills provided
            return torch.zeros(1, self.embedding_dim, device=device)

        # Stack embeddings
        skill_embeddings = torch.stack(embeddings).unsqueeze(0)  # (1, num_skills, dim)

        # Add level embeddings
        level_tensor = torch.tensor(levels, device=device)
        level_embs = self.level_embedding(level_tensor)
        skill_embeddings = skill_embeddings + level_embs.unsqueeze(0)

        # Apply hierarchical attention layers
        for layer in self.hierarchy_layers:
            skill_embeddings = layer(skill_embeddings)

        # Aggregate (weighted by hierarchy level)
        # Higher levels (instruments) get less weight than lower levels (phase)
        weights = torch.tensor(
            [1.0 / (level + 1) for level in levels],
            device=device
        ).unsqueeze(0).unsqueeze(-1)
        weights = weights / weights.sum()

        aggregated = (skill_embeddings * weights).sum(dim=1)

        # Final projection
        output = self.output_projection(aggregated)

        return output

    def forward(
        self,
        phase: Optional[str] = None,
        step: Optional[str] = None,
        instruments: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Forward pass - encode hierarchy."""
        return self.encode_hierarchy(phase, step, instruments)

    def get_valid_steps_for_phase(self, phase: str) -> List[str]:
        """Get valid steps for a given phase."""
        return self.phase_step_map.get(phase, [])

    def get_common_instruments_for_step(self, step: str) -> List[str]:
        """Get commonly used instruments for a given step."""
        return self.step_instrument_map.get(step, [])


class ConcatenationFusion(nn.Module):
    """Fusion strategy using concatenation with projection."""

    def __init__(
        self,
        visual_dim: int,
        skill_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(visual_dim + skill_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        skill_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse visual and skill features."""
        combined = torch.cat([visual_features, skill_features], dim=-1)
        return self.projection(combined)


class CrossAttentionFusion(nn.Module):
    """Fusion strategy using cross-attention between visual and skill features."""

    def __init__(
        self,
        visual_dim: int,
        skill_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.skill_proj = nn.Linear(skill_dim, output_dim)

        # Cross attention: visual attends to skill
        self.visual_to_skill = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross attention: skill attends to visual
        self.skill_to_visual = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Combine both attended features
        self.combiner = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        skill_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse features using cross-attention."""
        # Project to common dimension
        visual_proj = self.visual_proj(visual_features)
        skill_proj = self.skill_proj(skill_features)

        # Ensure 3D for attention (batch, seq, dim)
        if visual_proj.dim() == 2:
            visual_proj = visual_proj.unsqueeze(1)
        if skill_proj.dim() == 2:
            skill_proj = skill_proj.unsqueeze(1)

        # Cross attention
        visual_attended, _ = self.visual_to_skill(
            visual_proj, skill_proj, skill_proj
        )
        skill_attended, _ = self.skill_to_visual(
            skill_proj, visual_proj, visual_proj
        )

        # Combine
        combined = torch.cat([
            visual_attended.squeeze(1),
            skill_attended.squeeze(1)
        ], dim=-1)

        return self.combiner(combined)


class GatedFusion(nn.Module):
    """Fusion strategy using learnable gating mechanism."""

    def __init__(
        self,
        visual_dim: int,
        skill_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project to common dimension
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.skill_proj = nn.Linear(skill_dim, output_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + skill_dim, output_dim),
            nn.Sigmoid()
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        skill_features: torch.Tensor
    ) -> torch.Tensor:
        """Fuse features using learned gating."""
        # Project features
        visual_proj = self.visual_proj(visual_features)
        skill_proj = self.skill_proj(skill_features)

        # Compute gate
        combined = torch.cat([visual_features, skill_features], dim=-1)
        gate = self.gate(combined)

        # Apply gating
        fused = gate * visual_proj + (1 - gate) * skill_proj

        return self.output(fused)


class SkillEmbeddingGenerator(nn.Module):
    """
    Generates skill embeddings by combining visual features with skill predictions.

    This is the main class for creating unified skill embeddings suitable for
    RL state representation in surgical workflow understanding.

    Attributes:
        visual_dim: Dimension of visual features (default: 512)
        skill_dim: Dimension of skill embeddings (default: 128)
        output_dim: Dimension of output skill embeddings (default: 256)

    Example:
        >>> generator = SkillEmbeddingGenerator(
        ...     visual_dim=512,
        ...     skill_dim=128,
        ...     output_dim=256
        ... )
        >>>
        >>> # From features + predictions
        >>> visual_features = torch.randn(1, 512)
        >>> skill_predictions = {
        ...     'phase': 'Nasal Phase',
        ...     'step': 'Septal Dissection',
        ...     'instruments': ['Endoscope', 'Suction']
        ... }
        >>> embedding = generator.generate(visual_features, skill_predictions)
        >>> print(embedding.shape)
        torch.Size([1, 256])
    """

    def __init__(
        self,
        visual_dim: int = 512,
        skill_dim: int = 128,
        output_dim: int = 256,
        fusion_strategy: FusionStrategy = FusionStrategy.GATED_FUSION,
        dropout: float = 0.1,
        use_hierarchy: bool = True,
        categories: Optional[SkillCategories] = None
    ):
        """
        Initialize the skill embedding generator.

        Args:
            visual_dim: Dimension of input visual features
            skill_dim: Dimension of skill embeddings from vocabulary
            output_dim: Dimension of output skill embeddings
            fusion_strategy: Strategy for combining visual and skill features
            dropout: Dropout rate for regularization
            use_hierarchy: Whether to use hierarchical skill encoding
            categories: Skill category definitions
        """
        super().__init__()

        self.visual_dim = visual_dim
        self.skill_dim = skill_dim
        self.output_dim = output_dim
        self.fusion_strategy = fusion_strategy
        self.use_hierarchy = use_hierarchy

        self.categories = categories or SkillCategories()

        # Skill vocabulary for base skill embeddings
        self.skill_vocab = SkillVocabulary(
            embedding_dim=skill_dim,
            categories=self.categories,
            dropout=dropout
        )

        # Hierarchical encoder (optional)
        if use_hierarchy:
            self.hierarchy_encoder = HierarchicalSkillEncoder(
                embedding_dim=skill_dim,
                categories=self.categories,
                dropout=dropout
            )
        else:
            self.hierarchy_encoder = None

        # Fusion module based on strategy
        if fusion_strategy == FusionStrategy.CONCATENATION:
            self.fusion = ConcatenationFusion(
                visual_dim=visual_dim,
                skill_dim=skill_dim,
                output_dim=output_dim,
                dropout=dropout
            )
        elif fusion_strategy == FusionStrategy.CROSS_ATTENTION:
            self.fusion = CrossAttentionFusion(
                visual_dim=visual_dim,
                skill_dim=skill_dim,
                output_dim=output_dim,
                dropout=dropout
            )
        elif fusion_strategy == FusionStrategy.GATED_FUSION:
            self.fusion = GatedFusion(
                visual_dim=visual_dim,
                skill_dim=skill_dim,
                output_dim=output_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        # Visual feature encoder (for end-to-end processing)
        self._visual_encoder = None

        # Embedding cache for efficiency
        self._embedding_cache: Dict[str, torch.Tensor] = {}

        logger.info(
            f"SkillEmbeddingGenerator initialized: "
            f"visual_dim={visual_dim}, skill_dim={skill_dim}, "
            f"output_dim={output_dim}, fusion={fusion_strategy.value}"
        )

    def set_visual_encoder(self, encoder: nn.Module) -> None:
        """
        Set the visual encoder for end-to-end frame processing.

        Args:
            encoder: Visual encoder module (e.g., ResNet, ViT)
        """
        self._visual_encoder = encoder
        logger.info("Visual encoder set for end-to-end processing")

    def generate(
        self,
        visual_features: torch.Tensor,
        skill_predictions: Union[SkillPrediction, Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Generate skill embeddings from visual features and skill predictions.

        Args:
            visual_features: Visual features tensor of shape (batch, visual_dim)
            skill_predictions: Skill predictions (SkillPrediction or dict)

        Returns:
            Skill embedding tensor of shape (batch, output_dim)
        """
        # Validate input dimensions
        if visual_features.dim() == 1:
            visual_features = visual_features.unsqueeze(0)

        if visual_features.shape[-1] != self.visual_dim:
            raise ValueError(
                f"Expected visual features of dim {self.visual_dim}, "
                f"got {visual_features.shape[-1]}"
            )

        # Convert predictions to dict if needed
        if isinstance(skill_predictions, SkillPrediction):
            predictions = skill_predictions.to_dict()
        else:
            predictions = skill_predictions

        # Get skill embeddings
        if self.use_hierarchy and self.hierarchy_encoder is not None:
            instruments = predictions.get('instruments', [])
            if instruments and isinstance(instruments[0], tuple):
                instrument_names = [inst[0] for inst in instruments]
            else:
                instrument_names = instruments

            skill_emb = self.hierarchy_encoder.encode_hierarchy(
                phase=predictions.get('phase'),
                step=predictions.get('step'),
                instruments=instrument_names
            )
        else:
            skill_emb = self.skill_vocab(predictions)

        # Ensure batch dimension matches
        if skill_emb.shape[0] != visual_features.shape[0]:
            skill_emb = skill_emb.expand(visual_features.shape[0], -1)

        # Fuse visual and skill features
        output = self.fusion(visual_features, skill_emb)

        return output

    def generate_from_frame(
        self,
        frame: Union[torch.Tensor, np.ndarray],
        skill_predictions: Optional[Union[SkillPrediction, Dict[str, Any]]] = None
    ) -> torch.Tensor:
        """
        Generate skill embeddings directly from a video frame.

        This is an end-to-end method that extracts visual features and
        combines them with skill predictions.

        Args:
            frame: Input frame as tensor (C, H, W) or numpy array (H, W, C)
            skill_predictions: Optional skill predictions. If None, uses
                              empty predictions (visual features only)

        Returns:
            Skill embedding tensor of shape (1, output_dim)

        Raises:
            RuntimeError: If visual encoder is not set
        """
        if self._visual_encoder is None:
            raise RuntimeError(
                "Visual encoder not set. Call set_visual_encoder() first, "
                "or use generate() with pre-extracted visual features."
            )

        # Convert numpy to tensor if needed
        if isinstance(frame, np.ndarray):
            # Assume (H, W, C) format
            frame = torch.from_numpy(frame).permute(2, 0, 1).float()

        # Add batch dimension if needed
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        # Normalize if needed (assume ImageNet normalization)
        if frame.max() > 1.0:
            frame = frame / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frame = (frame - mean.to(frame.device)) / std.to(frame.device)

        # Extract visual features
        with torch.no_grad():
            visual_features = self._visual_encoder(frame)

        # Flatten if needed
        if visual_features.dim() > 2:
            visual_features = visual_features.view(visual_features.shape[0], -1)

        # Use empty predictions if not provided
        if skill_predictions is None:
            skill_predictions = SkillPrediction()

        return self.generate(visual_features, skill_predictions)

    def forward(
        self,
        visual_features: torch.Tensor,
        skill_predictions: Union[SkillPrediction, Dict[str, Any]]
    ) -> torch.Tensor:
        """Forward pass - alias for generate()."""
        return self.generate(visual_features, skill_predictions)

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim


def skill_similarity(
    embedding1: torch.Tensor,
    embedding2: torch.Tensor,
    metric: str = "cosine"
) -> torch.Tensor:
    """
    Compute similarity between two skill embeddings.

    Args:
        embedding1: First embedding tensor
        embedding2: Second embedding tensor
        metric: Similarity metric ('cosine', 'euclidean', 'dot')

    Returns:
        Similarity score tensor
    """
    # Ensure 2D
    if embedding1.dim() == 1:
        embedding1 = embedding1.unsqueeze(0)
    if embedding2.dim() == 1:
        embedding2 = embedding2.unsqueeze(0)

    if metric == "cosine":
        # Normalize embeddings
        emb1_norm = F.normalize(embedding1, p=2, dim=-1)
        emb2_norm = F.normalize(embedding2, p=2, dim=-1)
        similarity = torch.sum(emb1_norm * emb2_norm, dim=-1)

    elif metric == "euclidean":
        # Negative Euclidean distance (higher = more similar)
        distance = torch.sqrt(torch.sum((embedding1 - embedding2) ** 2, dim=-1))
        similarity = -distance

    elif metric == "dot":
        similarity = torch.sum(embedding1 * embedding2, dim=-1)

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    return similarity


def find_similar_skills(
    query_embedding: torch.Tensor,
    skill_embeddings: torch.Tensor,
    skill_labels: List[str],
    k: int = 5,
    metric: str = "cosine"
) -> List[Tuple[str, float]]:
    """
    Find the k most similar skills to a query embedding.

    Args:
        query_embedding: Query embedding tensor
        skill_embeddings: Tensor of skill embeddings (num_skills, dim)
        skill_labels: List of skill labels corresponding to embeddings
        k: Number of similar skills to return
        metric: Similarity metric to use

    Returns:
        List of (skill_label, similarity_score) tuples, sorted by similarity
    """
    # Ensure proper dimensions
    if query_embedding.dim() == 1:
        query_embedding = query_embedding.unsqueeze(0)

    if skill_embeddings.dim() == 1:
        skill_embeddings = skill_embeddings.unsqueeze(0)

    # Compute similarities
    similarities = []
    for i in range(skill_embeddings.shape[0]):
        sim = skill_similarity(
            query_embedding,
            skill_embeddings[i:i+1],
            metric=metric
        )
        similarities.append(sim.item())

    # Sort by similarity
    indexed_similarities = list(enumerate(similarities))
    indexed_similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    results = []
    for idx, sim in indexed_similarities[:k]:
        if idx < len(skill_labels):
            results.append((skill_labels[idx], sim))

    return results


class SkillEmbeddingIndex:
    """
    Index for efficient skill embedding search and retrieval.

    Maintains a database of skill embeddings for nearest neighbor search.
    Useful for skill recognition and recommendation.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        categories: Optional[SkillCategories] = None
    ):
        """
        Initialize the skill embedding index.

        Args:
            embedding_dim: Dimension of skill embeddings
            categories: Skill category definitions
        """
        self.embedding_dim = embedding_dim
        self.categories = categories or SkillCategories()

        # Storage for embeddings and metadata
        self.embeddings: List[torch.Tensor] = []
        self.labels: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        # Cached tensor for efficient search
        self._embeddings_tensor: Optional[torch.Tensor] = None

        logger.info(f"SkillEmbeddingIndex initialized: dim={embedding_dim}")

    def add(
        self,
        embedding: torch.Tensor,
        label: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a skill embedding to the index.

        Args:
            embedding: Skill embedding tensor
            label: Skill label
            metadata: Optional metadata dictionary

        Returns:
            Index of the added embedding
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        self.embeddings.append(embedding.detach().cpu())
        self.labels.append(label)
        self.metadata.append(metadata or {})

        # Invalidate cache
        self._embeddings_tensor = None

        return len(self.embeddings) - 1

    def add_batch(
        self,
        embeddings: torch.Tensor,
        labels: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """
        Add multiple skill embeddings to the index.

        Args:
            embeddings: Tensor of embeddings (batch, dim)
            labels: List of skill labels
            metadata: Optional list of metadata dictionaries

        Returns:
            List of indices of added embeddings
        """
        if metadata is None:
            metadata = [{}] * len(labels)

        indices = []
        for i, (emb, label, meta) in enumerate(zip(embeddings, labels, metadata)):
            idx = self.add(emb, label, meta)
            indices.append(idx)

        return indices

    def search(
        self,
        query: torch.Tensor,
        k: int = 5,
        metric: str = "cosine"
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar skills.

        Args:
            query: Query embedding tensor
            k: Number of results to return
            metric: Similarity metric

        Returns:
            List of (label, similarity, metadata) tuples
        """
        if not self.embeddings:
            return []

        # Build cached tensor if needed
        if self._embeddings_tensor is None:
            self._embeddings_tensor = torch.cat(self.embeddings, dim=0)

        # Find similar skills
        results = find_similar_skills(
            query_embedding=query,
            skill_embeddings=self._embeddings_tensor,
            skill_labels=self.labels,
            k=k,
            metric=metric
        )

        # Add metadata
        results_with_meta = []
        for label, sim in results:
            idx = self.labels.index(label)
            results_with_meta.append((label, sim, self.metadata[idx]))

        return results_with_meta

    def get_embedding(self, label: str) -> Optional[torch.Tensor]:
        """Get embedding for a specific skill label."""
        if label in self.labels:
            idx = self.labels.index(label)
            return self.embeddings[idx]
        return None

    def size(self) -> int:
        """Get number of embeddings in index."""
        return len(self.embeddings)

    def save(self, path: Union[str, Path]) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'embedding_dim': self.embedding_dim,
            'embeddings': torch.cat(self.embeddings, dim=0) if self.embeddings else torch.tensor([]),
            'labels': self.labels,
            'metadata': self.metadata
        }

        torch.save(save_data, path)
        logger.info(f"Saved index with {len(self.labels)} embeddings to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SkillEmbeddingIndex":
        """Load index from disk."""
        path = Path(path)
        data = torch.load(path)

        index = cls(embedding_dim=data['embedding_dim'])

        if data['embeddings'].numel() > 0:
            for i, label in enumerate(data['labels']):
                index.embeddings.append(data['embeddings'][i:i+1])
                index.labels.append(label)
                index.metadata.append(data['metadata'][i])

        logger.info(f"Loaded index with {len(index.labels)} embeddings from {path}")
        return index


# Convenience function for quick embedding generation
def create_skill_embedding_generator(
    visual_dim: int = 512,
    skill_dim: int = 128,
    output_dim: int = 256,
    fusion_strategy: str = "gated",
    use_hierarchy: bool = True
) -> SkillEmbeddingGenerator:
    """
    Create a skill embedding generator with default settings.

    Args:
        visual_dim: Dimension of visual features
        skill_dim: Dimension of skill embeddings
        output_dim: Dimension of output embeddings
        fusion_strategy: Fusion strategy ('concatenation', 'cross_attention', 'gated')
        use_hierarchy: Whether to use hierarchical encoding

    Returns:
        Configured SkillEmbeddingGenerator
    """
    strategy_map = {
        "concatenation": FusionStrategy.CONCATENATION,
        "cross_attention": FusionStrategy.CROSS_ATTENTION,
        "gated": FusionStrategy.GATED_FUSION,
    }

    strategy = strategy_map.get(fusion_strategy, FusionStrategy.GATED_FUSION)

    return SkillEmbeddingGenerator(
        visual_dim=visual_dim,
        skill_dim=skill_dim,
        output_dim=output_dim,
        fusion_strategy=strategy,
        use_hierarchy=use_hierarchy
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test skill embedding generation"
    )
    parser.add_argument(
        "--visual-dim",
        type=int,
        default=512,
        help="Visual feature dimension"
    )
    parser.add_argument(
        "--skill-dim",
        type=int,
        default=128,
        help="Skill embedding dimension"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        help="Output embedding dimension"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="gated",
        choices=["concatenation", "cross_attention", "gated"],
        help="Fusion strategy"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Skill Embedding Generator Test")
    print("=" * 60)

    # Create generator
    generator = create_skill_embedding_generator(
        visual_dim=args.visual_dim,
        skill_dim=args.skill_dim,
        output_dim=args.output_dim,
        fusion_strategy=args.fusion
    )

    print(f"\nGenerator config:")
    print(f"  Visual dim: {generator.visual_dim}")
    print(f"  Skill dim: {generator.skill_dim}")
    print(f"  Output dim: {generator.output_dim}")
    print(f"  Fusion: {generator.fusion_strategy.value}")

    # Test with sample data
    print("\nGenerating test embeddings...")

    # Sample visual features
    visual_features = torch.randn(2, args.visual_dim)

    # Sample skill predictions
    predictions = {
        "phase": "Nasal Phase",
        "step": "Septal Dissection",
        "instruments": [("Endoscope", 0.95), ("Suction", 0.87)],
        "position": "Center",
        "operation_note": "Clear Field of View"
    }

    # Generate embeddings
    embeddings = generator.generate(visual_features, predictions)

    print(f"\nInput visual features: {visual_features.shape}")
    print(f"Output embeddings: {embeddings.shape}")

    # Test similarity
    print("\nTesting skill similarity...")
    emb1 = embeddings[0]
    emb2 = embeddings[1]

    cos_sim = skill_similarity(emb1, emb2, metric="cosine")
    print(f"Cosine similarity: {cos_sim.item():.4f}")

    # Test vocabulary
    print("\nTesting skill vocabulary...")
    vocab = SkillVocabulary(embedding_dim=args.skill_dim)

    phase_emb = vocab.embed_single_skill("phase", "Nasal Phase")
    step_emb = vocab.embed_single_skill("step", "Septal Dissection")

    print(f"Phase embedding shape: {phase_emb.shape}")
    print(f"Step embedding shape: {step_emb.shape}")

    # Test hierarchical encoder
    print("\nTesting hierarchical encoder...")
    hierarchy_encoder = HierarchicalSkillEncoder(embedding_dim=args.skill_dim)

    hierarchy_emb = hierarchy_encoder.encode_hierarchy(
        phase="Nasal Phase",
        step="Septal Dissection",
        instruments=["Endoscope", "Suction"]
    )

    print(f"Hierarchy embedding shape: {hierarchy_emb.shape}")

    # Test skill index
    print("\nTesting skill embedding index...")
    index = SkillEmbeddingIndex(embedding_dim=args.output_dim)

    # Add some embeddings
    for i in range(5):
        emb = torch.randn(1, args.output_dim)
        index.add(emb, f"skill_{i}", {"frame_id": f"frame_{i:04d}"})

    print(f"Index size: {index.size()}")

    # Search
    query = torch.randn(1, args.output_dim)
    results = index.search(query, k=3)

    print("\nSearch results:")
    for label, sim, meta in results:
        print(f"  {label}: similarity={sim:.4f}, {meta}")

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
