"""
PitVQA Skill Classifier Module

Multi-task classification for surgical skills extraction from visual features.
Classifies surgical phases, steps, instruments, positions, and operation notes
according to PitVQA's 59 annotation classes.

Architecture:
    - Shared representation backbone with task-specific heads
    - Uncertainty-based multi-task learning for loss weighting
    - Support for both single-label and multi-label classification
    - Focal loss for handling class imbalance

Target Performance: >85% accuracy on 512-dim visual features
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# PitVQA Annotation Schema (59 classes total)
# =============================================================================

class SkillCategory(Enum):
    """Enumeration of skill categories in PitVQA."""
    PHASE = "phase"
    STEP = "step"
    INSTRUMENT = "instrument"
    POSITION = "position"
    OPERATION_NOTE = "operation_note"


@dataclass
class PitVQASchema:
    """
    Complete schema of PitVQA annotation classes.

    Contains all 59 annotation classes organized by category:
    - 4 surgical phases
    - 15 surgical steps
    - 18 surgical instruments
    - 5 instrument positions
    - 14 operation notes
    - 3 presence variations (used for instrument status, not classified separately)
    """

    # 4 Surgical Phases (single-label classification)
    phases: List[str] = field(default_factory=lambda: [
        "Nasal Phase",
        "Sellar Phase",
        "Tumor Resection Phase",
        "Closure Phase"
    ])

    # 15 Surgical Steps (single-label classification)
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

    # 18 Surgical Instruments (multi-label classification)
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

    # 5 Instrument Positions (single-label classification)
    positions: List[str] = field(default_factory=lambda: [
        "Left Side",
        "Right Side",
        "Center",
        "Upper Region",
        "Lower Region"
    ])

    # 14 Operation Notes (single-label classification)
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

    # 3 Presence Variations (for reference, not a separate classification task)
    presence_variations: List[str] = field(default_factory=lambda: [
        "Present and Active",
        "Present and Inactive",
        "Not Visible"
    ])

    def get_num_classes(self, category: SkillCategory) -> int:
        """Get the number of classes for a given category."""
        mapping = {
            SkillCategory.PHASE: len(self.phases),
            SkillCategory.STEP: len(self.steps),
            SkillCategory.INSTRUMENT: len(self.instruments),
            SkillCategory.POSITION: len(self.positions),
            SkillCategory.OPERATION_NOTE: len(self.operation_notes),
        }
        return mapping[category]

    def get_class_names(self, category: SkillCategory) -> List[str]:
        """Get class names for a given category."""
        mapping = {
            SkillCategory.PHASE: self.phases,
            SkillCategory.STEP: self.steps,
            SkillCategory.INSTRUMENT: self.instruments,
            SkillCategory.POSITION: self.positions,
            SkillCategory.OPERATION_NOTE: self.operation_notes,
        }
        return mapping[category]


# Global schema instance
PITVQA_SCHEMA = PitVQASchema()


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Predicted logits of shape (N, C)
            targets: Ground truth labels of shape (N,) for single-label
                    or (N, C) for multi-label

        Returns:
            Computed focal loss
        """
        # Handle both single-label and multi-label cases
        if targets.dim() == 1:
            # Single-label classification
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            p_t = torch.exp(-ce_loss)
        else:
            # Multi-label classification (binary cross entropy)
            ce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets.float(), reduction="none"
            )
            p = torch.sigmoid(inputs)
            p_t = p * targets + (1 - p) * (1 - targets)
            ce_loss = ce_loss.sum(dim=-1)  # Sum over classes
            p_t = p_t.mean(dim=-1)  # Average probability

        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss specifically designed for multi-label classification.

    Applies focal loss to each class independently.

    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Compute multi-label focal loss.

        Args:
            inputs: Predicted logits of shape (N, C)
            targets: Binary targets of shape (N, C)

        Returns:
            Computed focal loss
        """
        targets = targets.float()
        p = torch.sigmoid(inputs)

        # Compute BCE for positive and negative samples
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )

        # Compute focal weight
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# Multi-Task Classification Head
# =============================================================================

class TaskHead(nn.Module):
    """
    Single task-specific classification head.

    Architecture:
        Input -> BatchNorm -> Linear -> ReLU -> Dropout -> Linear -> Output

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer
        num_classes: Number of output classes
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.head(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task classification head with shared representation.

    Features:
        - Shared feature transformation layer
        - Task-specific classification heads
        - Learnable uncertainty weights for multi-task learning

    Args:
        input_dim: Dimension of input visual features
        shared_dim: Dimension of shared representation
        task_configs: Dictionary mapping task names to number of classes
        hidden_dim: Hidden dimension for task heads
        dropout: Dropout probability
        use_uncertainty_weighting: Use learned uncertainty weights for losses
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        task_configs: Dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_uncertainty_weighting: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.task_configs = task_configs
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Shared feature transformation
        self.shared_layers = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task_name: TaskHead(
                input_dim=shared_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
            for task_name, num_classes in task_configs.items()
        })

        # Learnable uncertainty weights (log variance)
        # Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"
        if use_uncertainty_weighting:
            self.log_vars = nn.ParameterDict({
                task_name: nn.Parameter(torch.zeros(1))
                for task_name in task_configs.keys()
            })

        self._init_shared_weights()

        logger.info(
            f"MultiTaskHead initialized with {len(task_configs)} tasks: "
            f"{list(task_configs.keys())}"
        )

    def _init_shared_weights(self) -> None:
        """Initialize shared layer weights."""
        for module in self.shared_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass through all task heads.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Dictionary mapping task names to logits
        """
        # Shared representation
        shared_features = self.shared_layers(x)

        # Task-specific predictions
        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)

        return outputs

    def get_shared_features(self, x: Tensor) -> Tensor:
        """
        Get the shared representation features.

        Args:
            x: Input features of shape (batch_size, input_dim)

        Returns:
            Shared features of shape (batch_size, shared_dim)
        """
        return self.shared_layers(x)

    def get_uncertainty_weights(self) -> Dict[str, float]:
        """
        Get the current uncertainty-based task weights.

        Returns:
            Dictionary mapping task names to weights
        """
        if not self.use_uncertainty_weighting:
            return {task: 1.0 for task in self.task_configs.keys()}

        weights = {}
        for task_name, log_var in self.log_vars.items():
            # Weight = 1 / (2 * exp(log_var))
            weights[task_name] = 0.5 * torch.exp(-log_var).item()

        return weights


# =============================================================================
# Skill Classifier
# =============================================================================

@dataclass
class ClassificationResult:
    """Container for classification results with confidence scores."""

    task: str
    predicted_class: Union[int, List[int]]
    predicted_label: Union[str, List[str]]
    confidence: Union[float, List[float]]
    probabilities: Tensor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "task": self.task,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
        }


class SkillClassifier(nn.Module):
    """
    Multi-task surgical skill classifier for PitVQA dataset.

    Classifies surgical skills from visual features across multiple categories:
    - Phase (4 classes): Nasal, Sellar, Tumor Removal, Closure
    - Step (15 classes): Septal dissection, Sphenoidotomy, etc.
    - Instrument (18 classes, multi-label): Grasper, Scissors, etc.
    - Position (5 classes): Left, Right, Center, Upper, Lower
    - Operation Note (14 classes): Bleeding, Clear field, etc.

    Features:
        - Multi-task learning with shared representation
        - Uncertainty-based loss weighting
        - Support for focal loss to handle class imbalance
        - Pretrained weight loading for fine-tuning
        - Comprehensive prediction with confidence scores

    Args:
        input_dim: Dimension of input visual features (default: 512)
        num_phases: Number of phase classes (default: 4)
        num_steps: Number of step classes (default: 15)
        num_instruments: Number of instrument classes (default: 18)
        num_positions: Number of position classes (default: 5)
        num_operation_notes: Number of operation note classes (default: 14)
        shared_dim: Dimension of shared representation (default: 384)
        hidden_dim: Hidden dimension for task heads (default: 256)
        dropout: Dropout probability (default: 0.3)
        use_uncertainty_weighting: Use learned uncertainty weights (default: True)
        use_focal_loss: Use focal loss for class imbalance (default: True)
        focal_gamma: Focal loss gamma parameter (default: 2.0)
        focal_alpha: Focal loss alpha parameter (default: 0.25)
        schema: PitVQA schema for class names (default: None, uses global)

    Example:
        >>> classifier = SkillClassifier(input_dim=512)
        >>> features = torch.randn(16, 512)  # batch of 16
        >>> outputs = classifier(features)
        >>> predictions = classifier.predict(features)
        >>> skill_vector = classifier.get_skill_vector(features)
    """

    def __init__(
        self,
        input_dim: int = 512,
        num_phases: int = 4,
        num_steps: int = 15,
        num_instruments: int = 18,
        num_positions: int = 5,
        num_operation_notes: int = 14,
        shared_dim: int = 384,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_uncertainty_weighting: bool = True,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        schema: Optional[PitVQASchema] = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.hidden_dim = hidden_dim
        self.use_focal_loss = use_focal_loss
        self.schema = schema or PITVQA_SCHEMA

        # Task configurations
        self.task_configs = {
            "phase": num_phases,
            "step": num_steps,
            "instrument": num_instruments,
            "position": num_positions,
            "operation_note": num_operation_notes,
        }

        # Multi-label tasks (instruments can have multiple present)
        self.multi_label_tasks = {"instrument"}

        # Build multi-task head
        self.multi_task_head = MultiTaskHead(
            input_dim=input_dim,
            shared_dim=shared_dim,
            task_configs=self.task_configs,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_uncertainty_weighting=use_uncertainty_weighting,
        )

        # Loss functions
        self._setup_loss_functions(focal_gamma, focal_alpha)

        # Calculate total skill vector dimension
        self.skill_vector_dim = sum(self.task_configs.values())

        logger.info(
            f"SkillClassifier initialized: "
            f"input_dim={input_dim}, "
            f"total_classes={self.skill_vector_dim}, "
            f"tasks={list(self.task_configs.keys())}"
        )

    def _setup_loss_functions(
        self,
        focal_gamma: float,
        focal_alpha: float,
    ) -> None:
        """Set up loss functions for each task."""
        self.loss_functions = {}

        for task_name in self.task_configs.keys():
            if task_name in self.multi_label_tasks:
                # Multi-label task: use BCE or multi-label focal loss
                if self.use_focal_loss:
                    self.loss_functions[task_name] = MultiLabelFocalLoss(
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                else:
                    self.loss_functions[task_name] = nn.BCEWithLogitsLoss()
            else:
                # Single-label task: use CE or focal loss
                if self.use_focal_loss:
                    self.loss_functions[task_name] = FocalLoss(
                        alpha=focal_alpha,
                        gamma=focal_gamma,
                    )
                else:
                    self.loss_functions[task_name] = nn.CrossEntropyLoss()

    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass returning predictions for all tasks.

        Args:
            features: Visual features of shape (batch_size, input_dim)

        Returns:
            Dictionary mapping task names to logits:
            - 'phase': (batch_size, num_phases)
            - 'step': (batch_size, num_steps)
            - 'instrument': (batch_size, num_instruments)
            - 'position': (batch_size, num_positions)
            - 'operation_note': (batch_size, num_operation_notes)

        Raises:
            ValueError: If input dimension doesn't match expected
        """
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, input_dim), got {features.dim()}D"
            )

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {features.shape[-1]}"
            )

        return self.multi_task_head(features)

    def compute_loss(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        return_per_task: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        """
        Compute multi-task loss with uncertainty weighting.

        Args:
            predictions: Dictionary of task predictions (logits)
            targets: Dictionary of task targets
            return_per_task: Whether to return per-task losses

        Returns:
            Total weighted loss, and optionally per-task losses
        """
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        per_task_losses = {}

        # Get uncertainty weights
        weights = self.multi_task_head.get_uncertainty_weights()

        for task_name, pred in predictions.items():
            if task_name not in targets:
                continue

            target = targets[task_name]
            loss_fn = self.loss_functions[task_name]

            # Compute task loss
            task_loss = loss_fn(pred, target)
            per_task_losses[task_name] = task_loss

            # Apply uncertainty weighting
            weight = weights.get(task_name, 1.0)
            weighted_loss = weight * task_loss

            # Add regularization term for uncertainty (log_var)
            if self.multi_task_head.use_uncertainty_weighting:
                log_var = self.multi_task_head.log_vars[task_name]
                # Loss = 0.5 * exp(-log_var) * loss + 0.5 * log_var
                weighted_loss = weighted_loss + 0.5 * log_var.squeeze()

            total_loss = total_loss + weighted_loss

        if return_per_task:
            return total_loss, per_task_losses
        return total_loss

    def predict(
        self,
        features: Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, ClassificationResult]:
        """
        Make predictions with class labels and confidence scores.

        Args:
            features: Visual features of shape (batch_size, input_dim)
            threshold: Threshold for multi-label classification (default: 0.5)

        Returns:
            Dictionary mapping task names to ClassificationResult objects
        """
        was_training = self.training
        self.train(False)

        with torch.no_grad():
            outputs = self.forward(features)

        results = {}
        batch_size = features.shape[0]

        for task_name, logits in outputs.items():
            class_names = self._get_class_names(task_name)

            if task_name in self.multi_label_tasks:
                # Multi-label: apply sigmoid and threshold
                probs = torch.sigmoid(logits)
                predictions_tensor = (probs > threshold).int()

                # Get results for each sample in batch
                if batch_size == 1:
                    pred_indices = predictions_tensor[0].nonzero(as_tuple=True)[0].tolist()
                    pred_labels = [class_names[i] for i in pred_indices]
                    confidences = probs[0, pred_indices].tolist() if pred_indices else []
                else:
                    pred_indices = []
                    pred_labels = []
                    confidences = []
                    for i in range(batch_size):
                        sample_indices = predictions_tensor[i].nonzero(as_tuple=True)[0].tolist()
                        pred_indices.append(sample_indices)
                        pred_labels.append([class_names[j] for j in sample_indices])
                        confidences.append(probs[i, sample_indices].tolist())

                results[task_name] = ClassificationResult(
                    task=task_name,
                    predicted_class=pred_indices,
                    predicted_label=pred_labels,
                    confidence=confidences,
                    probabilities=probs,
                )
            else:
                # Single-label: apply softmax and argmax
                probs = F.softmax(logits, dim=-1)
                predictions_tensor = torch.argmax(probs, dim=-1)
                confidences = probs.gather(1, predictions_tensor.unsqueeze(-1)).squeeze(-1)

                if batch_size == 1:
                    pred_class = predictions_tensor[0].item()
                    pred_label = class_names[pred_class]
                    conf = confidences[0].item()
                else:
                    pred_class = predictions_tensor.tolist()
                    pred_label = [class_names[i] for i in pred_class]
                    conf = confidences.tolist()

                results[task_name] = ClassificationResult(
                    task=task_name,
                    predicted_class=pred_class,
                    predicted_label=pred_label,
                    confidence=conf,
                    probabilities=probs,
                )

        self.train(was_training)
        return results

    def get_skill_vector(
        self,
        features: Tensor,
        normalize: bool = True,
    ) -> Tensor:
        """
        Get concatenated skill representation vector.

        Concatenates probability distributions from all tasks into a single
        skill representation vector. Useful for downstream tasks or visualization.

        Args:
            features: Visual features of shape (batch_size, input_dim)
            normalize: Whether to L2-normalize the output vector

        Returns:
            Skill vector of shape (batch_size, skill_vector_dim)
            where skill_vector_dim = sum of all task class counts
        """
        was_training = self.training
        self.train(False)

        with torch.no_grad():
            outputs = self.forward(features)

        # Convert logits to probabilities
        prob_vectors = []

        for task_name in ["phase", "step", "instrument", "position", "operation_note"]:
            logits = outputs[task_name]

            if task_name in self.multi_label_tasks:
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=-1)

            prob_vectors.append(probs)

        # Concatenate all probability vectors
        skill_vector = torch.cat(prob_vectors, dim=-1)

        if normalize:
            skill_vector = F.normalize(skill_vector, p=2, dim=-1)

        self.train(was_training)
        return skill_vector

    def get_shared_representation(self, features: Tensor) -> Tensor:
        """
        Get the shared representation before task-specific heads.

        Args:
            features: Visual features of shape (batch_size, input_dim)

        Returns:
            Shared features of shape (batch_size, shared_dim)
        """
        return self.multi_task_head.get_shared_features(features)

    def _get_class_names(self, task_name: str) -> List[str]:
        """Get class names for a task."""
        mapping = {
            "phase": self.schema.phases,
            "step": self.schema.steps,
            "instrument": self.schema.instruments,
            "position": self.schema.positions,
            "operation_note": self.schema.operation_notes,
        }
        return mapping.get(task_name, [])

    def load_pretrained(
        self,
        checkpoint_path: Union[str, Path],
        strict: bool = False,
        map_location: Optional[str] = None,
    ) -> List[str]:
        """
        Load pretrained weights for fine-tuning.

        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce matching keys
            map_location: Device to load weights to

        Returns:
            List of missing or unexpected keys

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if map_location is None:
            map_location = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Load weights
        missing_keys, unexpected_keys = self.load_state_dict(
            state_dict, strict=strict
        )

        if missing_keys:
            logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        logger.info(f"Loaded pretrained weights from {checkpoint_path}")

        return missing_keys + unexpected_keys

    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            save_path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            metrics: Optional metrics dictionary
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "shared_dim": self.shared_dim,
                "hidden_dim": self.hidden_dim,
                "task_configs": self.task_configs,
                "use_focal_loss": self.use_focal_loss,
            },
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def freeze_shared_layers(self) -> None:
        """Freeze shared representation layers for fine-tuning task heads only."""
        for param in self.multi_task_head.shared_layers.parameters():
            param.requires_grad = False
        logger.info("Frozen shared layers")

    def unfreeze_shared_layers(self) -> None:
        """Unfreeze shared representation layers."""
        for param in self.multi_task_head.shared_layers.parameters():
            param.requires_grad = True
        logger.info("Unfrozen shared layers")

    def get_task_metrics(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each task.

        Args:
            predictions: Dictionary of task predictions (logits)
            targets: Dictionary of task targets
            threshold: Threshold for multi-label classification

        Returns:
            Dictionary of metrics per task (accuracy, precision, recall, f1)
        """
        metrics = {}

        for task_name, logits in predictions.items():
            if task_name not in targets:
                continue

            target = targets[task_name]

            if task_name in self.multi_label_tasks:
                # Multi-label metrics
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()

                # Per-sample metrics
                tp = (preds * target).sum(dim=-1)
                fp = (preds * (1 - target)).sum(dim=-1)
                fn = ((1 - preds) * target).sum(dim=-1)

                precision = (tp / (tp + fp + 1e-8)).mean().item()
                recall = (tp / (tp + fn + 1e-8)).mean().item()
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                # Exact match accuracy (all labels correct)
                accuracy = (preds == target).all(dim=-1).float().mean().item()
            else:
                # Single-label metrics
                preds = torch.argmax(logits, dim=-1)
                accuracy = (preds == target).float().mean().item()

                # Macro-averaged precision/recall
                num_classes = logits.shape[-1]
                precisions = []
                recalls = []

                for c in range(num_classes):
                    tp = ((preds == c) & (target == c)).sum().float()
                    fp = ((preds == c) & (target != c)).sum().float()
                    fn = ((preds != c) & (target == c)).sum().float()

                    if tp + fp > 0:
                        precisions.append((tp / (tp + fp)).item())
                    if tp + fn > 0:
                        recalls.append((tp / (tp + fn)).item())

                precision = sum(precisions) / len(precisions) if precisions else 0.0
                recall = sum(recalls) / len(recalls) if recalls else 0.0
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

            metrics[task_name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        return metrics

    def summary(self) -> str:
        """
        Get a summary of the model architecture.

        Returns:
            String summary of the model
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary_lines = [
            "=" * 60,
            "SkillClassifier Summary",
            "=" * 60,
            f"Input Dimension: {self.input_dim}",
            f"Shared Dimension: {self.shared_dim}",
            f"Hidden Dimension: {self.hidden_dim}",
            f"Skill Vector Dimension: {self.skill_vector_dim}",
            "-" * 60,
            "Task Configurations:",
        ]

        for task, num_classes in self.task_configs.items():
            multi_label = "(multi-label)" if task in self.multi_label_tasks else ""
            summary_lines.append(f"  - {task}: {num_classes} classes {multi_label}")

        summary_lines.extend([
            "-" * 60,
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Non-trainable Parameters: {total_params - trainable_params:,}",
            "-" * 60,
            f"Focal Loss: {'Enabled' if self.use_focal_loss else 'Disabled'}",
            f"Uncertainty Weighting: {'Enabled' if self.multi_task_head.use_uncertainty_weighting else 'Disabled'}",
            "=" * 60,
        ])

        return "\n".join(summary_lines)


# =============================================================================
# Factory Functions
# =============================================================================

def create_skill_classifier(
    input_dim: int = 512,
    pretrained_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> SkillClassifier:
    """
    Factory function to create a SkillClassifier.

    Args:
        input_dim: Dimension of input visual features
        pretrained_path: Optional path to pretrained weights
        device: Device to place model on
        **kwargs: Additional arguments for SkillClassifier

    Returns:
        Initialized SkillClassifier instance
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = SkillClassifier(input_dim=input_dim, **kwargs)

    if pretrained_path:
        classifier.load_pretrained(pretrained_path)

    classifier = classifier.to(device)

    logger.info(f"Created SkillClassifier on {device}")

    return classifier


def create_pitvqa_classifier(
    input_dim: int = 512,
    **kwargs,
) -> SkillClassifier:
    """
    Create a SkillClassifier with PitVQA default configuration.

    Uses the exact class counts from PitVQA dataset:
    - 4 phases
    - 15 steps
    - 18 instruments
    - 5 positions
    - 14 operation notes

    Args:
        input_dim: Dimension of input visual features
        **kwargs: Additional arguments for SkillClassifier

    Returns:
        Initialized SkillClassifier for PitVQA
    """
    return create_skill_classifier(
        input_dim=input_dim,
        num_phases=4,
        num_steps=15,
        num_instruments=18,
        num_positions=5,
        num_operation_notes=14,
        **kwargs,
    )


# =============================================================================
# Main / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PitVQA Skill Classifier Demo"
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=512,
        help="Input feature dimension (default: 512)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Demo batch size (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Set device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PitVQA Skill Classifier Demo")
    print("=" * 60)

    # Create classifier
    print("\n[1] Creating SkillClassifier...")
    classifier = create_pitvqa_classifier(
        input_dim=args.input_dim,
        device=device,
    )

    # Print summary
    print("\n[2] Model Summary:")
    print(classifier.summary())

    # Demo forward pass
    print(f"\n[3] Demo Forward Pass (batch_size={args.batch_size})...")
    demo_features = torch.randn(args.batch_size, args.input_dim).to(device)

    outputs = classifier(demo_features)
    print("Output shapes:")
    for task, logits in outputs.items():
        print(f"  - {task}: {logits.shape}")

    # Demo predictions
    print("\n[4] Demo Predictions (single sample)...")
    single_feature = torch.randn(1, args.input_dim).to(device)
    predictions = classifier.predict(single_feature)

    print("Predictions:")
    for task, result in predictions.items():
        print(f"  - {task}: {result.predicted_label} (conf: {result.confidence})")

    # Demo skill vector
    print("\n[5] Demo Skill Vector...")
    skill_vector = classifier.get_skill_vector(single_feature)
    print(f"Skill vector shape: {skill_vector.shape}")
    print(f"Skill vector (first 10 dims): {skill_vector[0, :10].tolist()}")

    # Demo loss computation
    print("\n[6] Demo Loss Computation...")
    # Create dummy targets
    targets = {
        "phase": torch.randint(0, 4, (args.batch_size,)).to(device),
        "step": torch.randint(0, 15, (args.batch_size,)).to(device),
        "instrument": torch.randint(0, 2, (args.batch_size, 18)).float().to(device),
        "position": torch.randint(0, 5, (args.batch_size,)).to(device),
        "operation_note": torch.randint(0, 14, (args.batch_size,)).to(device),
    }

    outputs = classifier(demo_features)
    total_loss, per_task_losses = classifier.compute_loss(
        outputs, targets, return_per_task=True
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    print("Per-task losses:")
    for task, loss in per_task_losses.items():
        print(f"  - {task}: {loss.item():.4f}")

    # Demo metrics
    print("\n[7] Demo Metrics...")
    metrics = classifier.get_task_metrics(outputs, targets)
    print("Metrics:")
    for task, task_metrics in metrics.items():
        print(f"  - {task}:")
        for metric_name, value in task_metrics.items():
            print(f"      {metric_name}: {value:.4f}")

    # Uncertainty weights
    print("\n[8] Uncertainty Weights...")
    weights = classifier.multi_task_head.get_uncertainty_weights()
    print("Task weights:")
    for task, weight in weights.items():
        print(f"  - {task}: {weight:.4f}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
