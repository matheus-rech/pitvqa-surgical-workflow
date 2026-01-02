"""
PitVQA Question-Answer Generator Module

Generates Question-Answer pairs from PitVQA surgical video annotations.
The PitVQA dataset contains 59 annotation classes across multiple categories:
- 4 surgical phases
- 15 surgical steps
- 18 surgical instruments
- 3 instrument presence variations
- 5 instrument positions
- 14 operation notes

This module creates diverse QA pairs for training Visual Question Answering models
on pituitary surgery workflows.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Enumeration of question types for VQA."""
    PHASE = "phase"
    STEP = "step"
    INSTRUMENT_WHAT = "instrument_what"
    INSTRUMENT_COUNT = "instrument_count"
    INSTRUMENT_POSITION = "instrument_position"
    YES_NO_INSTRUMENT = "yes_no_instrument"
    YES_NO_PHASE = "yes_no_phase"
    YES_NO_STEP = "yes_no_step"
    OPERATION_NOTE = "operation_note"


@dataclass
class QAPair:
    """Data class representing a Question-Answer pair."""
    frame_id: str
    question: str
    answer: str
    question_type: str
    video_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "frame_id": self.frame_id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "video_id": self.video_id,
            "metadata": self.metadata
        }


@dataclass
class AnnotationSchema:
    """Schema defining the PitVQA annotation categories."""
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

    # 3 Instrument Presence Variations
    presence_variations: List[str] = field(default_factory=lambda: [
        "Present and Active",
        "Present and Inactive",
        "Not Visible"
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


class QuestionTemplates:
    """Templates for generating different types of questions."""

    # Phase questions
    PHASE_TEMPLATES = [
        "What surgical phase is this?",
        "Which phase of the surgery is currently being performed?",
        "What is the current surgical phase?",
        "Identify the surgical phase in this frame.",
        "What phase of the pituitary surgery is shown?"
    ]

    # Step questions
    STEP_TEMPLATES = [
        "What step is being performed?",
        "Which surgical step is currently in progress?",
        "What is the current surgical step?",
        "Identify the surgical step in this frame.",
        "What procedure step is being executed?"
    ]

    # Instrument identification questions
    INSTRUMENT_WHAT_TEMPLATES = [
        "What instruments are visible?",
        "Which surgical instruments can be seen?",
        "What tools are being used in this frame?",
        "Identify the visible surgical instruments.",
        "What surgical equipment is present?"
    ]

    # Instrument count questions
    INSTRUMENT_COUNT_TEMPLATES = [
        "How many instruments are visible?",
        "How many surgical tools can be seen?",
        "Count the number of visible instruments.",
        "What is the count of instruments in this frame?",
        "How many tools are present?"
    ]

    # Position questions
    POSITION_TEMPLATES = [
        "Where is the {instrument} positioned?",
        "What is the position of the {instrument}?",
        "In which region is the {instrument} located?",
        "Where is the {instrument} in the frame?",
        "Identify the position of the {instrument}."
    ]

    # Yes/No instrument questions
    YES_NO_INSTRUMENT_TEMPLATES = [
        "Is the {instrument} visible?",
        "Can you see the {instrument} in this frame?",
        "Is there a {instrument} present?",
        "Is the {instrument} being used?",
        "Do you observe a {instrument}?"
    ]

    # Yes/No phase questions
    YES_NO_PHASE_TEMPLATES = [
        "Is this the {phase}?",
        "Are we in the {phase}?",
        "Is the current phase the {phase}?",
        "Does this frame show the {phase}?",
        "Is the surgery in the {phase}?"
    ]

    # Yes/No step questions
    YES_NO_STEP_TEMPLATES = [
        "Is {step} being performed?",
        "Is the surgeon doing {step}?",
        "Is this the {step} step?",
        "Are we at the {step} stage?",
        "Is {step} currently in progress?"
    ]

    # Operation note questions
    OPERATION_NOTE_TEMPLATES = [
        "What is the operation note for this frame?",
        "What observation can be made about this scene?",
        "What is the current surgical status?",
        "Describe the current surgical situation.",
        "What notable condition is present?"
    ]


class QAGenerator:
    """
    Generator for Question-Answer pairs from PitVQA annotations.

    This class loads annotation files and generates diverse QA pairs
    for training VQA models on surgical procedures.
    """

    def __init__(
        self,
        schema: Optional[AnnotationSchema] = None,
        qa_per_frame: int = 8,
        balance_questions: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the QA Generator.

        Args:
            schema: Annotation schema defining categories. Uses default if None.
            qa_per_frame: Target number of QA pairs per frame.
            balance_questions: Whether to balance question types.
            seed: Random seed for reproducibility.
        """
        self.schema = schema or AnnotationSchema()
        self.qa_per_frame = qa_per_frame
        self.balance_questions = balance_questions
        self.templates = QuestionTemplates()

        if seed is not None:
            random.seed(seed)

        # Statistics tracking
        self._stats = defaultdict(int)
        self._annotations_cache: Dict[str, Dict] = {}

        logger.info(f"QAGenerator initialized with {qa_per_frame} QA pairs per frame target")

    def load_annotations(self, annotation_dir: Union[str, Path]) -> Dict[str, Dict]:
        """
        Load all annotation files from a directory.

        Args:
            annotation_dir: Path to directory containing annotation JSON files.

        Returns:
            Dictionary mapping frame_id to annotation data.

        Raises:
            FileNotFoundError: If annotation directory doesn't exist.
            ValueError: If no valid annotation files found.
        """
        annotation_path = Path(annotation_dir)

        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation directory not found: {annotation_path}")

        annotations = {}
        json_files = list(annotation_path.glob("*.json"))

        if not json_files:
            # Try to find annotations in subdirectories
            json_files = list(annotation_path.rglob("*.json"))

        if not json_files:
            logger.warning(f"No JSON annotation files found in {annotation_path}")
            return annotations

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different annotation formats
                if isinstance(data, dict):
                    if 'frames' in data:
                        # Format: {"frames": {"frame_id": {...}, ...}}
                        for frame_id, frame_data in data['frames'].items():
                            annotations[frame_id] = self._normalize_annotation(frame_data, frame_id)
                    elif 'annotations' in data:
                        # Format: {"annotations": [{"frame_id": ..., ...}, ...]}
                        for ann in data['annotations']:
                            frame_id = ann.get('frame_id', ann.get('id', str(len(annotations))))
                            annotations[frame_id] = self._normalize_annotation(ann, frame_id)
                    else:
                        # Assume direct frame_id -> annotation mapping
                        for frame_id, frame_data in data.items():
                            if isinstance(frame_data, dict):
                                annotations[frame_id] = self._normalize_annotation(frame_data, frame_id)
                elif isinstance(data, list):
                    # Format: [{"frame_id": ..., ...}, ...]
                    for ann in data:
                        frame_id = ann.get('frame_id', ann.get('id', str(len(annotations))))
                        annotations[frame_id] = self._normalize_annotation(ann, frame_id)

                logger.debug(f"Loaded annotations from {json_file.name}")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {json_file}: {e}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        self._annotations_cache = annotations
        self._stats['total_frames'] = len(annotations)
        logger.info(f"Loaded annotations for {len(annotations)} frames")

        return annotations

    def _normalize_annotation(self, data: Dict, frame_id: str) -> Dict:
        """
        Normalize annotation data to a consistent format.

        Args:
            data: Raw annotation data.
            frame_id: Frame identifier.

        Returns:
            Normalized annotation dictionary.
        """
        normalized = {
            'frame_id': frame_id,
            'phase': data.get('phase', data.get('surgical_phase', None)),
            'step': data.get('step', data.get('surgical_step', None)),
            'instruments': [],
            'operation_note': data.get('operation_note', data.get('note', None)),
            'video_id': data.get('video_id', data.get('video', self._extract_video_id(frame_id)))
        }

        # Handle instruments (can be list of strings or list of dicts)
        instruments_data = data.get('instruments', data.get('tools', []))

        if isinstance(instruments_data, list):
            for inst in instruments_data:
                if isinstance(inst, str):
                    normalized['instruments'].append({
                        'name': inst,
                        'presence': 'Present and Active',
                        'position': None
                    })
                elif isinstance(inst, dict):
                    normalized['instruments'].append({
                        'name': inst.get('name', inst.get('instrument', 'Unknown')),
                        'presence': inst.get('presence', inst.get('status', 'Present and Active')),
                        'position': inst.get('position', inst.get('location', None))
                    })
        elif isinstance(instruments_data, dict):
            for name, details in instruments_data.items():
                if isinstance(details, dict):
                    normalized['instruments'].append({
                        'name': name,
                        'presence': details.get('presence', 'Present and Active'),
                        'position': details.get('position', None)
                    })
                else:
                    normalized['instruments'].append({
                        'name': name,
                        'presence': str(details),
                        'position': None
                    })

        return normalized

    def _extract_video_id(self, frame_id: str) -> str:
        """Extract video ID from frame ID if possible."""
        # Common patterns: video01_frame001, v1_f001, etc.
        parts = frame_id.split('_')
        if len(parts) >= 2:
            return parts[0]
        return "unknown"

    def generate_qa_pairs(
        self,
        frame_id: str,
        annotations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs for a single frame.

        Args:
            frame_id: Unique identifier for the frame.
            annotations: Annotation data for this frame.

        Returns:
            List of QA pair dictionaries.
        """
        qa_pairs = []
        video_id = annotations.get('video_id', self._extract_video_id(frame_id))

        # Get available question generators
        generators = self._get_question_generators(annotations)

        if self.balance_questions:
            qa_pairs = self._generate_balanced_qa(frame_id, annotations, video_id, generators)
        else:
            qa_pairs = self._generate_all_qa(frame_id, annotations, video_id, generators)

        # Limit to target count
        if len(qa_pairs) > self.qa_per_frame:
            random.shuffle(qa_pairs)
            qa_pairs = qa_pairs[:self.qa_per_frame]

        # Update statistics
        for qa in qa_pairs:
            self._stats[f'qa_type_{qa["question_type"]}'] += 1
        self._stats['total_qa_pairs'] += len(qa_pairs)

        return qa_pairs

    def _get_question_generators(self, annotations: Dict) -> Dict[str, callable]:
        """Get available question generators based on annotations."""
        generators = {}

        if annotations.get('phase'):
            generators['phase'] = self._generate_phase_question
            generators['yes_no_phase'] = self._generate_yes_no_phase_question

        if annotations.get('step'):
            generators['step'] = self._generate_step_question
            generators['yes_no_step'] = self._generate_yes_no_step_question

        if annotations.get('instruments'):
            generators['instrument_what'] = self._generate_instrument_what_question
            generators['instrument_count'] = self._generate_instrument_count_question
            generators['yes_no_instrument'] = self._generate_yes_no_instrument_question

            # Position questions only if any instrument has position data
            if any(inst.get('position') for inst in annotations.get('instruments', [])):
                generators['instrument_position'] = self._generate_position_question

        if annotations.get('operation_note'):
            generators['operation_note'] = self._generate_operation_note_question

        return generators

    def _generate_balanced_qa(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str,
        generators: Dict[str, callable]
    ) -> List[Dict]:
        """Generate balanced QA pairs across question types."""
        qa_pairs = []

        if not generators:
            return qa_pairs

        # Calculate how many of each type to generate
        types_count = len(generators)
        base_per_type = self.qa_per_frame // types_count
        remainder = self.qa_per_frame % types_count

        type_counts = {}
        for i, q_type in enumerate(generators.keys()):
            type_counts[q_type] = base_per_type + (1 if i < remainder else 0)

        # Generate questions for each type
        for q_type, count in type_counts.items():
            generator = generators[q_type]
            for _ in range(count):
                qa = generator(frame_id, annotations, video_id)
                if qa:
                    qa_pairs.append(qa)

        return qa_pairs

    def _generate_all_qa(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str,
        generators: Dict[str, callable]
    ) -> List[Dict]:
        """Generate all possible QA pairs."""
        qa_pairs = []

        for generator in generators.values():
            qa = generator(frame_id, annotations, video_id)
            if qa:
                qa_pairs.append(qa)

        return qa_pairs

    def _generate_phase_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate a phase identification question."""
        phase = annotations.get('phase')
        if not phase:
            return None

        question = random.choice(self.templates.PHASE_TEMPLATES)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=phase,
            question_type=QuestionType.PHASE.value,
            video_id=video_id
        ).to_dict()

    def _generate_step_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate a step identification question."""
        step = annotations.get('step')
        if not step:
            return None

        question = random.choice(self.templates.STEP_TEMPLATES)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=step,
            question_type=QuestionType.STEP.value,
            video_id=video_id
        ).to_dict()

    def _generate_instrument_what_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate an instrument identification question."""
        instruments = annotations.get('instruments', [])
        if not instruments:
            return None

        # Get visible instruments
        visible_instruments = [
            inst['name'] for inst in instruments
            if inst.get('presence', '').lower() != 'not visible'
        ]

        if not visible_instruments:
            answer = "No instruments visible"
        elif len(visible_instruments) == 1:
            answer = visible_instruments[0]
        else:
            answer = ", ".join(visible_instruments[:-1]) + f" and {visible_instruments[-1]}"

        question = random.choice(self.templates.INSTRUMENT_WHAT_TEMPLATES)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=answer,
            question_type=QuestionType.INSTRUMENT_WHAT.value,
            video_id=video_id
        ).to_dict()

    def _generate_instrument_count_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate an instrument counting question."""
        instruments = annotations.get('instruments', [])

        # Count visible instruments
        visible_count = sum(
            1 for inst in instruments
            if inst.get('presence', '').lower() != 'not visible'
        )

        question = random.choice(self.templates.INSTRUMENT_COUNT_TEMPLATES)

        # Natural language answer
        if visible_count == 0:
            answer = "None"
        elif visible_count == 1:
            answer = "One"
        elif visible_count == 2:
            answer = "Two"
        elif visible_count == 3:
            answer = "Three"
        elif visible_count == 4:
            answer = "Four"
        elif visible_count == 5:
            answer = "Five"
        else:
            answer = str(visible_count)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=answer,
            question_type=QuestionType.INSTRUMENT_COUNT.value,
            video_id=video_id
        ).to_dict()

    def _generate_position_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate an instrument position question."""
        instruments = annotations.get('instruments', [])

        # Find instruments with position data
        positioned_instruments = [
            inst for inst in instruments
            if inst.get('position') and inst.get('presence', '').lower() != 'not visible'
        ]

        if not positioned_instruments:
            return None

        # Select a random instrument
        instrument = random.choice(positioned_instruments)

        template = random.choice(self.templates.POSITION_TEMPLATES)
        question = template.format(instrument=instrument['name'])

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=instrument['position'],
            question_type=QuestionType.INSTRUMENT_POSITION.value,
            video_id=video_id,
            metadata={'instrument': instrument['name']}
        ).to_dict()

    def _generate_yes_no_instrument_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate a yes/no question about instrument presence."""
        instruments = annotations.get('instruments', [])

        # Get set of visible instruments
        visible_names = {
            inst['name'].lower() for inst in instruments
            if inst.get('presence', '').lower() != 'not visible'
        }

        # Randomly choose to ask about a present or absent instrument
        if random.random() < 0.5 and visible_names:
            # Ask about a visible instrument (answer: Yes)
            instrument_name = random.choice(list(visible_names))
            answer = "Yes"
        else:
            # Ask about an instrument not visible (answer: No)
            all_instruments = set(i.lower() for i in self.schema.instruments)
            absent = all_instruments - visible_names
            if absent:
                instrument_name = random.choice(list(absent))
                answer = "No"
            elif visible_names:
                instrument_name = random.choice(list(visible_names))
                answer = "Yes"
            else:
                return None

        # Format instrument name properly
        instrument_name = instrument_name.title()

        template = random.choice(self.templates.YES_NO_INSTRUMENT_TEMPLATES)
        question = template.format(instrument=instrument_name)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=answer,
            question_type=QuestionType.YES_NO_INSTRUMENT.value,
            video_id=video_id,
            metadata={'instrument': instrument_name, 'expected': answer == "Yes"}
        ).to_dict()

    def _generate_yes_no_phase_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate a yes/no question about surgical phase."""
        current_phase = annotations.get('phase')
        if not current_phase:
            return None

        # Randomly choose to ask about current phase or different phase
        if random.random() < 0.5:
            # Ask about current phase (answer: Yes)
            phase = current_phase
            answer = "Yes"
        else:
            # Ask about a different phase (answer: No)
            other_phases = [p for p in self.schema.phases if p != current_phase]
            if other_phases:
                phase = random.choice(other_phases)
                answer = "No"
            else:
                phase = current_phase
                answer = "Yes"

        template = random.choice(self.templates.YES_NO_PHASE_TEMPLATES)
        question = template.format(phase=phase)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=answer,
            question_type=QuestionType.YES_NO_PHASE.value,
            video_id=video_id,
            metadata={'phase': phase, 'expected': answer == "Yes"}
        ).to_dict()

    def _generate_yes_no_step_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate a yes/no question about surgical step."""
        current_step = annotations.get('step')
        if not current_step:
            return None

        # Randomly choose to ask about current step or different step
        if random.random() < 0.5:
            # Ask about current step (answer: Yes)
            step = current_step
            answer = "Yes"
        else:
            # Ask about a different step (answer: No)
            other_steps = [s for s in self.schema.steps if s != current_step]
            if other_steps:
                step = random.choice(other_steps)
                answer = "No"
            else:
                step = current_step
                answer = "Yes"

        template = random.choice(self.templates.YES_NO_STEP_TEMPLATES)
        question = template.format(step=step)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=answer,
            question_type=QuestionType.YES_NO_STEP.value,
            video_id=video_id,
            metadata={'step': step, 'expected': answer == "Yes"}
        ).to_dict()

    def _generate_operation_note_question(
        self,
        frame_id: str,
        annotations: Dict,
        video_id: str
    ) -> Optional[Dict]:
        """Generate an operation note question."""
        note = annotations.get('operation_note')
        if not note:
            return None

        question = random.choice(self.templates.OPERATION_NOTE_TEMPLATES)

        return QAPair(
            frame_id=frame_id,
            question=question,
            answer=note,
            question_type=QuestionType.OPERATION_NOTE.value,
            video_id=video_id
        ).to_dict()

    def generate_all_qa_pairs(
        self,
        frames_dir: Union[str, Path],
        annotation_dir: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate QA pairs for all frames in a dataset.

        Args:
            frames_dir: Directory containing frame images.
            annotation_dir: Directory containing annotation files.
            output_file: Optional path to save generated QA pairs.

        Returns:
            List of all generated QA pairs.
        """
        frames_path = Path(frames_dir)

        # Load annotations
        annotations = self.load_annotations(annotation_dir)

        if not annotations:
            logger.warning("No annotations loaded. Returning empty list.")
            return []

        all_qa_pairs = []
        processed_frames = 0
        skipped_frames = 0

        for frame_id, frame_annotations in annotations.items():
            try:
                # Optionally verify frame exists
                if frames_path.exists():
                    frame_patterns = [
                        frames_path / f"{frame_id}.png",
                        frames_path / f"{frame_id}.jpg",
                        frames_path / frame_id,
                    ]
                    frame_exists = any(p.exists() for p in frame_patterns)

                    if not frame_exists:
                        # Check subdirectories
                        video_id = frame_annotations.get('video_id', '')
                        if video_id:
                            subdir_patterns = [
                                frames_path / video_id / f"{frame_id}.png",
                                frames_path / video_id / f"{frame_id}.jpg",
                            ]
                            frame_exists = any(p.exists() for p in subdir_patterns)

                        if not frame_exists:
                            logger.debug(f"Frame image not found for {frame_id}, generating QA anyway")

                # Generate QA pairs for this frame
                qa_pairs = self.generate_qa_pairs(frame_id, frame_annotations)
                all_qa_pairs.extend(qa_pairs)
                processed_frames += 1

            except Exception as e:
                logger.error(f"Error processing frame {frame_id}: {e}")
                skipped_frames += 1

        logger.info(f"Processed {processed_frames} frames, skipped {skipped_frames}")
        logger.info(f"Generated {len(all_qa_pairs)} total QA pairs")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved QA pairs to {output_path}")

        return all_qa_pairs

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about generated QA pairs.

        Returns:
            Dictionary containing generation statistics.
        """
        stats = dict(self._stats)

        # Calculate percentages
        total = stats.get('total_qa_pairs', 0)
        if total > 0:
            stats['distribution'] = {}
            for key, value in stats.items():
                if key.startswith('qa_type_'):
                    q_type = key.replace('qa_type_', '')
                    stats['distribution'][q_type] = {
                        'count': value,
                        'percentage': round(value / total * 100, 2)
                    }

        # Average QA per frame
        frames = stats.get('total_frames', 0)
        if frames > 0:
            stats['avg_qa_per_frame'] = round(total / frames, 2)

        return stats

    def reset_statistics(self) -> None:
        """Reset all tracked statistics."""
        self._stats.clear()
        logger.info("Statistics reset")


def create_sample_annotations(output_dir: Union[str, Path]) -> Path:
    """
    Create sample annotation files for testing.

    Args:
        output_dir: Directory to save sample annotations.

    Returns:
        Path to created sample file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    schema = AnnotationSchema()

    # Generate sample annotations
    sample_data = {
        "frames": {}
    }

    for i in range(10):
        frame_id = f"video01_frame{i:04d}"

        # Random annotations
        phase = random.choice(schema.phases)
        step = random.choice(schema.steps)

        # Random instruments (1-3)
        num_instruments = random.randint(1, 3)
        instruments = []
        for _ in range(num_instruments):
            instruments.append({
                "name": random.choice(schema.instruments),
                "presence": random.choice(schema.presence_variations),
                "position": random.choice(schema.positions) if random.random() > 0.3 else None
            })

        sample_data["frames"][frame_id] = {
            "phase": phase,
            "step": step,
            "instruments": instruments,
            "operation_note": random.choice(schema.operation_notes),
            "video_id": "video01"
        }

    sample_file = output_path / "sample_annotations.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)

    logger.info(f"Created sample annotations at {sample_file}")
    return sample_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA pairs from PitVQA annotations")
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default="data/annotations",
        help="Directory containing annotation files"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/processed",
        help="Directory containing frame images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/qa_pairs.json",
        help="Output file for generated QA pairs"
    )
    parser.add_argument(
        "--qa-per-frame",
        type=int,
        default=8,
        help="Number of QA pairs to generate per frame"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample annotations for testing"
    )

    args = parser.parse_args()

    if args.create_sample:
        sample_path = create_sample_annotations(args.annotation_dir)
        print(f"Created sample annotations at: {sample_path}")

    # Initialize generator
    generator = QAGenerator(
        qa_per_frame=args.qa_per_frame,
        balance_questions=True,
        seed=args.seed
    )

    # Generate QA pairs
    qa_pairs = generator.generate_all_qa_pairs(
        frames_dir=args.frames_dir,
        annotation_dir=args.annotation_dir,
        output_file=args.output
    )

    # Print statistics
    stats = generator.get_statistics()
    print("\n" + "=" * 50)
    print("QA Generation Statistics")
    print("=" * 50)
    print(f"Total frames processed: {stats.get('total_frames', 0)}")
    print(f"Total QA pairs generated: {stats.get('total_qa_pairs', 0)}")
    print(f"Average QA per frame: {stats.get('avg_qa_per_frame', 0)}")

    if 'distribution' in stats:
        print("\nQuestion Type Distribution:")
        for q_type, info in stats['distribution'].items():
            print(f"  {q_type}: {info['count']} ({info['percentage']}%)")
