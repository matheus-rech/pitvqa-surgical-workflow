"""
Agent 3: Data Converter for SAGE/Molmo Fine-tuning

Converts PitVQA data (Agent 1/2 outputs) to Molmo training format with:
- Pointing annotations (x, y coordinates)
- Temporal grounding (timestamps)
- Conversation format for SFT
- Preference pairs for DPO

Reference: https://huggingface.co/blog/hf-skills-training
"""

import json
import logging
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

logger = logging.getLogger(__name__)


class TrainingMethod(Enum):
    """Supported training methods from HuggingFace Skills."""
    SFT = "sft"           # Supervised Fine-Tuning
    DPO = "dpo"           # Direct Preference Optimization
    GRPO = "grpo"         # Group Relative Policy Optimization


@dataclass
class PointAnnotation:
    """Represents a pointing annotation with coordinates."""
    x: float                     # Normalized x coordinate (0-1)
    y: float                     # Normalized y coordinate (0-1)
    label: str                   # What the point refers to
    confidence: float = 1.0      # Annotation confidence
    frame_idx: Optional[int] = None
    timestamp: Optional[float] = None

    def to_pixel(self, width: int, height: int) -> Tuple[int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        return (int(self.x * width), int(self.y * height))

    def to_molmo_format(self) -> str:
        """Format as Molmo pointing string: <point x='0.5' y='0.3'>label</point>"""
        return f"<point x='{self.x:.3f}' y='{self.y:.3f}'>{self.label}</point>"


@dataclass
class BoundingBox:
    """Represents a bounding box annotation."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str
    confidence: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    def to_point(self) -> PointAnnotation:
        """Convert bounding box to center point annotation."""
        cx, cy = self.center
        return PointAnnotation(x=cx, y=cy, label=self.label, confidence=self.confidence)


@dataclass
class TemporalSegment:
    """Represents a temporal segment in video."""
    start_time: float
    end_time: float
    label: str
    description: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_timestamp_str(self) -> str:
        """Format as HH:MM:SS - HH:MM:SS."""
        def format_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{format_time(self.start_time)} - {format_time(self.end_time)}"


@dataclass
class SurgicalVQASample:
    """A single VQA sample with pointing and temporal annotations."""
    video_id: str
    frame_idx: int
    timestamp: Optional[float]
    image_path: str
    question: str
    answer: str

    # Annotations
    points: List[PointAnnotation] = field(default_factory=list)
    boxes: List[BoundingBox] = field(default_factory=list)
    temporal_segments: List[TemporalSegment] = field(default_factory=list)

    # Metadata from Agent 2
    phase: Optional[str] = None
    step: Optional[str] = None
    instruments: List[str] = field(default_factory=list)
    action: Optional[str] = None
    skill_embedding: Optional[np.ndarray] = None

    def to_molmo_conversation(self) -> List[Dict[str, str]]:
        """Convert to Molmo conversation format for SFT."""
        # Build answer with pointing annotations
        answer_with_points = self.answer

        if self.points:
            point_strs = [p.to_molmo_format() for p in self.points]
            answer_with_points = f"{self.answer}\n\nLocations: {' '.join(point_strs)}"

        return [
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": answer_with_points}
        ]


class SurgicalAnatomyVocabulary:
    """Vocabulary of pituitary surgery anatomical structures and instruments."""

    ANATOMICAL_STRUCTURES = {
        # Nasal cavity
        "middle_turbinate": "Middle turbinate bone",
        "inferior_turbinate": "Inferior turbinate",
        "nasal_septum": "Nasal septum",
        "sphenoid_ostium": "Sphenoid sinus natural opening",

        # Sphenoid sinus
        "sphenoid_sinus": "Sphenoid sinus cavity",
        "sphenoid_rostrum": "Sphenoid rostrum",
        "sella_floor": "Sellar floor bone",
        "planum_sphenoidale": "Planum sphenoidale",
        "carotid_prominence": "Carotid artery prominence",
        "optic_prominence": "Optic nerve prominence",
        "clivus": "Clivus",

        # Sellar region
        "pituitary_gland": "Pituitary gland",
        "pituitary_tumor": "Pituitary adenoma/tumor",
        "dura_mater": "Dura mater",
        "diaphragma_sellae": "Diaphragma sellae",
        "cavernous_sinus": "Cavernous sinus",
        "internal_carotid": "Internal carotid artery",
        "optic_chiasm": "Optic chiasm",
        "optic_nerve": "Optic nerve",

        # Other
        "bleeding_site": "Active bleeding site",
        "tumor_capsule": "Tumor capsule",
        "normal_gland": "Normal pituitary tissue",
    }

    INSTRUMENTS = {
        "endoscope": "0° or 30° rigid endoscope",
        "suction": "Suction cannula",
        "curette": "Ring curette",
        "bipolar": "Bipolar forceps",
        "monopolar": "Monopolar cautery",
        "scissors": "Endoscopic scissors",
        "grasper": "Grasping forceps",
        "drill": "High-speed drill",
        "kerrison": "Kerrison rongeur",
        "speculum": "Nasal speculum",
        "cottonoid": "Cottonoid patty",
        "hemostatic_agent": "Hemostatic material (Surgicel/Floseal)",
        "fat_graft": "Abdominal fat graft",
        "fascia": "Fascia lata graft",
        "nasoseptal_flap": "Nasoseptal flap",
    }

    ACTIONS = {
        "dissecting": "Separating tissue planes",
        "resecting": "Removing tissue",
        "coagulating": "Cauterizing for hemostasis",
        "suctioning": "Aspirating blood/fluid",
        "drilling": "Removing bone with drill",
        "irrigating": "Washing with saline",
        "packing": "Placing hemostatic material",
        "reconstructing": "Rebuilding skull base",
        "inspecting": "Visual examination",
        "retracting": "Moving tissue aside",
    }

    @classmethod
    def get_all_labels(cls) -> List[str]:
        """Get all possible labels."""
        return (
            list(cls.ANATOMICAL_STRUCTURES.keys()) +
            list(cls.INSTRUMENTS.keys()) +
            list(cls.ACTIONS.keys())
        )


class MolmoDataConverter:
    """
    Converts PitVQA data to Molmo/SAGE training formats.

    Supports:
    - SFT: Conversation format with pointing
    - DPO: Preference pairs (correct vs incorrect identifications)
    - GRPO: Reward-based training for surgical skill recognition
    """

    def __init__(
        self,
        vocabulary: Optional[SurgicalAnatomyVocabulary] = None,
        include_pointing: bool = True,
        include_temporal: bool = True,
        max_points_per_sample: int = 5,
    ):
        self.vocabulary = vocabulary or SurgicalAnatomyVocabulary()
        self.include_pointing = include_pointing
        self.include_temporal = include_temporal
        self.max_points_per_sample = max_points_per_sample

    def load_agent1_output(
        self,
        dataset_path: str,
        split: str = "train"
    ) -> Dataset:
        """Load Agent 1 output (processed frames with QA pairs)."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required: pip install datasets")

        if dataset_path.startswith("matheus-rech/"):
            # Load from HuggingFace Hub
            return load_dataset(dataset_path, split=split)
        else:
            # Load from local disk
            return load_from_disk(dataset_path)

    def load_agent2_output(
        self,
        embeddings_path: str,
        metadata_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load Agent 2 output (skill embeddings and classifications)."""
        result = {}

        # Load embeddings
        if embeddings_path.endswith(".npy"):
            result["embeddings"] = np.load(embeddings_path)
        elif os.path.isdir(embeddings_path):
            # HuggingFace dataset format
            result["dataset"] = load_from_disk(embeddings_path)

        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                result["metadata"] = json.load(f)

        return result

    def generate_pointing_questions(
        self,
        sample: Dict[str, Any],
        anatomy_labels: Optional[List[str]] = None
    ) -> List[SurgicalVQASample]:
        """
        Generate pointing-based VQA samples from a frame.

        Creates questions like:
        - "Point to the pituitary gland in this image"
        - "Where is the suction instrument?"
        - "Identify all visible anatomical structures"
        """
        samples = []
        video_id = sample.get("video_id", "unknown")
        frame_idx = sample.get("frame_idx", 0)
        image_path = sample.get("image_path", "")

        # Get existing annotations
        instruments = sample.get("instruments", [])
        phase = sample.get("phase", "")
        step = sample.get("step", "")

        # Question templates for pointing
        POINTING_TEMPLATES = [
            ("Point to the {label} in this surgical image.",
             "The {label} is located here: {point}"),
            ("Where is the {label}?",
             "I can identify the {label} at: {point}"),
            ("Identify the location of the {label}.",
             "The {label} is visible at: {point}"),
            ("Can you show me where the {label} is?",
             "Here is the {label}: {point}"),
        ]

        # Generate for instruments if present
        for instrument in instruments:
            if instrument in self.vocabulary.INSTRUMENTS:
                template = random.choice(POINTING_TEMPLATES)

                # Generate pseudo-point (in real use, these would come from annotations)
                # For now, create placeholder that will be filled during annotation
                point = PointAnnotation(
                    x=0.5, y=0.5,  # Placeholder - needs real annotation
                    label=instrument,
                    frame_idx=frame_idx
                )

                samples.append(SurgicalVQASample(
                    video_id=video_id,
                    frame_idx=frame_idx,
                    timestamp=sample.get("timestamp"),
                    image_path=image_path,
                    question=template[0].format(label=instrument),
                    answer=template[1].format(
                        label=instrument,
                        point=point.to_molmo_format()
                    ),
                    points=[point],
                    phase=phase,
                    step=step,
                    instruments=instruments
                ))

        # Generate phase/step understanding questions
        if phase:
            samples.append(SurgicalVQASample(
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp=sample.get("timestamp"),
                image_path=image_path,
                question="What surgical phase is shown in this image?",
                answer=f"This image shows the {phase} phase of the pituitary surgery.",
                phase=phase,
                step=step,
                instruments=instruments
            ))

        if step:
            samples.append(SurgicalVQASample(
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp=sample.get("timestamp"),
                image_path=image_path,
                question="What surgical step is being performed?",
                answer=f"The surgeon is performing {step}.",
                phase=phase,
                step=step,
                instruments=instruments
            ))

        return samples

    def generate_temporal_questions(
        self,
        video_metadata: Dict[str, Any],
        phase_segments: List[TemporalSegment]
    ) -> List[Dict[str, Any]]:
        """
        Generate temporal grounding questions for video-level understanding.

        Creates questions like:
        - "When does tumor resection begin?"
        - "How long is the sellar phase?"
        - "What happens between 10:00 and 15:00?"
        """
        samples = []

        TEMPORAL_TEMPLATES = [
            ("When does {phase} begin in this surgery?",
             "The {phase} begins at {start_time}."),
            ("How long is the {phase} phase?",
             "The {phase} phase lasts {duration} ({start_time} to {end_time})."),
            ("What surgical phase occurs at {timestamp}?",
             "At {timestamp}, the surgery is in the {phase} phase, specifically {step}."),
        ]

        for segment in phase_segments:
            # Phase start question
            samples.append({
                "question": f"When does {segment.label} begin in this surgery?",
                "answer": f"The {segment.label} begins at {segment.to_timestamp_str().split(' - ')[0]}.",
                "temporal_segment": segment,
                "type": "temporal_grounding"
            })

            # Duration question
            duration_mins = segment.duration / 60
            samples.append({
                "question": f"How long is the {segment.label} phase?",
                "answer": f"The {segment.label} phase lasts approximately {duration_mins:.1f} minutes.",
                "temporal_segment": segment,
                "type": "temporal_duration"
            })

        return samples

    def convert_to_sft_format(
        self,
        samples: List[SurgicalVQASample]
    ) -> Dataset:
        """
        Convert samples to SFT training format for HuggingFace Skills.

        Output format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "images": ["path/to/image.jpg"]  # For VLM
        }
        """
        sft_data = []

        for sample in tqdm(samples, desc="Converting to SFT format"):
            sft_data.append({
                "messages": sample.to_molmo_conversation(),
                "images": [sample.image_path] if sample.image_path else [],
                "metadata": {
                    "video_id": sample.video_id,
                    "frame_idx": sample.frame_idx,
                    "phase": sample.phase,
                    "step": sample.step,
                    "instruments": sample.instruments,
                }
            })

        return Dataset.from_list(sft_data)

    def convert_to_dpo_format(
        self,
        samples: List[SurgicalVQASample],
        generate_negatives: bool = True
    ) -> Dataset:
        """
        Convert samples to DPO training format for preference learning.

        Output format:
        {
            "prompt": "...",
            "chosen": "correct response",
            "rejected": "incorrect response",
            "images": ["path/to/image.jpg"]
        }
        """
        dpo_data = []

        for sample in tqdm(samples, desc="Converting to DPO format"):
            # Chosen response is the correct answer
            chosen = sample.answer

            # Generate rejected response (incorrect identification)
            if generate_negatives:
                rejected = self._generate_negative_sample(sample)
            else:
                rejected = "I cannot identify the structures in this image."

            dpo_data.append({
                "prompt": sample.question,
                "chosen": chosen,
                "rejected": rejected,
                "images": [sample.image_path] if sample.image_path else [],
            })

        return Dataset.from_list(dpo_data)

    def _generate_negative_sample(self, sample: SurgicalVQASample) -> str:
        """Generate an incorrect response for DPO training."""
        # Strategy 1: Wrong structure identification
        all_structures = list(self.vocabulary.ANATOMICAL_STRUCTURES.keys())

        if sample.points:
            correct_label = sample.points[0].label
            wrong_labels = [s for s in all_structures if s != correct_label]
            if wrong_labels:
                wrong_label = random.choice(wrong_labels)
                return sample.answer.replace(correct_label, wrong_label)

        # Strategy 2: Wrong phase/step
        if sample.phase:
            phases = ["nasal_phase", "sellar_phase", "tumor_removal_phase", "closure_phase"]
            wrong_phases = [p for p in phases if p != sample.phase]
            if wrong_phases:
                return f"This image shows the {random.choice(wrong_phases)} phase."

        # Default: Generic wrong answer
        return "This image is too unclear to identify any structures."

    def convert_to_grpo_format(
        self,
        samples: List[SurgicalVQASample],
        reward_functions: Optional[Dict[str, callable]] = None
    ) -> Dataset:
        """
        Convert samples to GRPO training format for RL-based learning.

        GRPO is ideal for surgical skill recognition where we can
        programmatically verify correctness.

        Output format:
        {
            "prompt": "...",
            "images": ["..."],
            "ground_truth": {...}  # For reward computation
        }
        """
        grpo_data = []

        for sample in tqdm(samples, desc="Converting to GRPO format"):
            grpo_data.append({
                "prompt": sample.question,
                "images": [sample.image_path] if sample.image_path else [],
                "ground_truth": {
                    "answer": sample.answer,
                    "phase": sample.phase,
                    "step": sample.step,
                    "instruments": sample.instruments,
                    "points": [
                        {"x": p.x, "y": p.y, "label": p.label}
                        for p in sample.points
                    ]
                }
            })

        return Dataset.from_list(grpo_data)

    def create_surgical_vqa_dataset(
        self,
        agent1_path: str,
        agent2_path: Optional[str] = None,
        training_method: TrainingMethod = TrainingMethod.SFT,
        output_path: Optional[str] = None,
        push_to_hub: Optional[str] = None,
        hf_token: Optional[str] = None
    ) -> DatasetDict:
        """
        Main pipeline: Create complete training dataset from Agent 1/2 outputs.

        Args:
            agent1_path: Path to Agent 1 output (frames + QA)
            agent2_path: Path to Agent 2 output (skill embeddings)
            training_method: SFT, DPO, or GRPO
            output_path: Local save path
            push_to_hub: HuggingFace repo ID to push to
            hf_token: HuggingFace token

        Returns:
            DatasetDict with train/val/test splits
        """
        logger.info(f"Creating surgical VQA dataset for {training_method.value} training")

        # Load Agent 1 output
        agent1_data = self.load_agent1_output(agent1_path)
        logger.info(f"Loaded {len(agent1_data)} samples from Agent 1")

        # Load Agent 2 output if provided
        agent2_data = None
        if agent2_path:
            agent2_data = self.load_agent2_output(agent2_path)
            logger.info("Loaded skill embeddings from Agent 2")

        # Generate VQA samples with pointing
        all_samples = []
        for idx, sample in enumerate(tqdm(agent1_data, desc="Generating samples")):
            # Merge Agent 2 data if available
            if agent2_data and "dataset" in agent2_data:
                if idx < len(agent2_data["dataset"]):
                    a2_sample = agent2_data["dataset"][idx]
                    sample["phase"] = a2_sample.get("phase")
                    sample["step"] = a2_sample.get("step")
                    sample["instruments"] = a2_sample.get("instruments", [])

            # Generate pointing questions
            vqa_samples = self.generate_pointing_questions(sample)
            all_samples.extend(vqa_samples)

        logger.info(f"Generated {len(all_samples)} VQA samples")

        # Convert to training format
        if training_method == TrainingMethod.SFT:
            dataset = self.convert_to_sft_format(all_samples)
        elif training_method == TrainingMethod.DPO:
            dataset = self.convert_to_dpo_format(all_samples)
        elif training_method == TrainingMethod.GRPO:
            dataset = self.convert_to_grpo_format(all_samples)
        else:
            raise ValueError(f"Unknown training method: {training_method}")

        # Create train/val/test splits
        dataset = dataset.shuffle(seed=42)
        splits = dataset.train_test_split(test_size=0.2, seed=42)
        val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

        dataset_dict = DatasetDict({
            "train": splits["train"],
            "validation": val_test["train"],
            "test": val_test["test"]
        })

        # Save locally
        if output_path:
            dataset_dict.save_to_disk(output_path)
            logger.info(f"Saved dataset to {output_path}")

        # Push to Hub
        if push_to_hub:
            token = hf_token or os.environ.get("HF_TOKEN")
            dataset_dict.push_to_hub(push_to_hub, token=token)
            logger.info(f"Pushed dataset to {push_to_hub}")

        return dataset_dict


# Convenience functions for CLI usage
def convert_pitvqa_to_molmo(
    input_path: str,
    output_path: str,
    method: str = "sft",
    push_to_hub: Optional[str] = None
):
    """CLI-friendly conversion function."""
    converter = MolmoDataConverter()
    method_enum = TrainingMethod(method.lower())

    return converter.create_surgical_vqa_dataset(
        agent1_path=input_path,
        training_method=method_enum,
        output_path=output_path,
        push_to_hub=push_to_hub
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PitVQA to Molmo training format")
    parser.add_argument("--input", required=True, help="Agent 1 output path or HF dataset")
    parser.add_argument("--agent2-input", help="Agent 2 embeddings path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", choices=["sft", "dpo", "grpo"], default="sft")
    parser.add_argument("--push-to-hub", help="HuggingFace repo to push to")
    parser.add_argument("--hf-token", help="HuggingFace token")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    converter = MolmoDataConverter()
    dataset = converter.create_surgical_vqa_dataset(
        agent1_path=args.input,
        agent2_path=args.agent2_input,
        training_method=TrainingMethod(args.method),
        output_path=args.output,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token
    )

    print(f"\nDataset created:")
    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Validation: {len(dataset['validation'])} samples")
    print(f"  Test: {len(dataset['test'])} samples")
