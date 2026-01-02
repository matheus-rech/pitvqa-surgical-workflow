"""
PitVQA Preprocessing Module

Handles preprocessing of surgical videos, images, and text data
for Visual Question Answering pipelines.
"""

import os
import re
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms


class ImagePreprocessor:
    """
    Preprocessing pipeline for surgical images.

    Handles resizing, normalization, augmentation, and
    surgical-specific enhancements.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Initialize image preprocessor.

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to apply ImageNet normalization
            augment: Whether to apply data augmentation
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment

        # ImageNet normalization parameters
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Build transform pipeline
        self.transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build the transformation pipeline."""
        transform_list = [
            transforms.Resize(self.target_size),
        ]

        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05
                ),
            ])

        transform_list.append(transforms.ToTensor())

        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        return transforms.Compose(transform_list)

    def preprocess(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        """
        Preprocess a single image.

        Args:
            image: Image path or PIL Image

        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        return self.transform(image)

    def preprocess_batch(
        self,
        images: List[Union[str, Path, Image.Image]]
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of image paths or PIL Images

        Returns:
            Batch tensor of preprocessed images
        """
        processed = [self.preprocess(img) for img in images]
        return torch.stack(processed)

    def enhance_surgical_image(
        self,
        image: Image.Image,
        enhance_contrast: float = 1.2,
        enhance_sharpness: float = 1.1
    ) -> Image.Image:
        """
        Apply surgical-specific image enhancements.

        Improves visibility of surgical instruments and anatomy.

        Args:
            image: Input PIL Image
            enhance_contrast: Contrast enhancement factor
            enhance_sharpness: Sharpness enhancement factor

        Returns:
            Enhanced PIL Image
        """
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(enhance_contrast)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(enhance_sharpness)

        return image

    def remove_black_borders(
        self,
        image: np.ndarray,
        threshold: int = 10
    ) -> np.ndarray:
        """
        Remove black borders common in surgical endoscopic images.

        Args:
            image: Input image as numpy array
            threshold: Pixel intensity threshold for black detection

        Returns:
            Cropped image without black borders
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Find non-black pixels
        coords = np.column_stack(np.where(gray > threshold))

        if len(coords) == 0:
            return image

        # Get bounding box
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop image
        return image[y_min:y_max+1, x_min:x_max+1]


class VideoPreprocessor:
    """
    Preprocessing pipeline for surgical videos.

    Handles frame extraction, temporal sampling, and
    video-specific preprocessing.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        fps: int = 1,
        max_frames: Optional[int] = None
    ):
        """
        Initialize video preprocessor.

        Args:
            target_size: Target frame size (height, width)
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract
        """
        self.target_size = target_size
        self.fps = fps
        self.max_frames = max_frames
        self.image_preprocessor = ImagePreprocessor(target_size=target_size)

    def extract_frames(
        self,
        video_path: Union[str, Path],
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds (None for full video)

        Returns:
            List of extracted frames as numpy arrays
        """
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps == 0:
            video_fps = 30  # Default fallback

        # Calculate frame interval
        frame_interval = max(1, int(video_fps / self.fps))

        # Calculate start and end frames
        start_frame = int(start_time * video_fps)
        if end_time is not None:
            end_frame = min(int(end_time * video_fps), total_frames)
        else:
            end_frame = total_frames

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = start_frame
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_count - start_frame) % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

                if self.max_frames and len(frames) >= self.max_frames:
                    break

            frame_count += 1

        cap.release()
        return frames

    def sample_frames_uniformly(
        self,
        frames: List[np.ndarray],
        num_samples: int
    ) -> List[np.ndarray]:
        """
        Uniformly sample frames from a list.

        Args:
            frames: List of frames
            num_samples: Number of frames to sample

        Returns:
            Sampled frames
        """
        if len(frames) <= num_samples:
            return frames

        indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
        return [frames[i] for i in indices]

    def detect_scene_changes(
        self,
        frames: List[np.ndarray],
        threshold: float = 30.0
    ) -> List[int]:
        """
        Detect scene changes in video frames.

        Useful for identifying surgical phase transitions.

        Args:
            frames: List of video frames
            threshold: Difference threshold for scene change

        Returns:
            Indices of frames where scene changes occur
        """
        scene_changes = [0]  # First frame is always a scene start

        for i in range(1, len(frames)):
            # Calculate frame difference
            diff = cv2.absdiff(frames[i-1], frames[i])
            mean_diff = np.mean(diff)

            if mean_diff > threshold:
                scene_changes.append(i)

        return scene_changes


class TextPreprocessor:
    """
    Preprocessing pipeline for surgical VQA text data.

    Handles question and answer text normalization,
    vocabulary building, and tokenization helpers.
    """

    # Common surgical terms for vocabulary
    SURGICAL_TERMS = [
        "instrument", "forceps", "scissors", "grasper", "clip",
        "cautery", "suction", "needle", "suture", "retractor",
        "bleeding", "coagulation", "dissection", "cutting",
        "anatomy", "tissue", "organ", "vessel", "nerve",
        "phase", "step", "procedure", "action", "task"
    ]

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        max_length: int = 128
    ):
        """
        Initialize text preprocessor.

        Args:
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            max_length: Maximum sequence length
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.max_length = max_length

        # Vocabulary
        self.vocab = {}
        self.word_counts = Counter()

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text string

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)

        # Normalize whitespace
        text = ' '.join(text.split())

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        return text.strip()

    def preprocess_question(self, question: str) -> str:
        """
        Preprocess a VQA question.

        Args:
            question: Question text

        Returns:
            Preprocessed question
        """
        question = self.clean_text(question)

        # Ensure question ends with question mark
        if question and not question.endswith('?'):
            question += '?'

        return question

    def preprocess_answer(self, answer: str) -> str:
        """
        Preprocess a VQA answer.

        Args:
            answer: Answer text

        Returns:
            Preprocessed answer
        """
        return self.clean_text(answer)

    def build_vocabulary(
        self,
        texts: List[str],
        min_freq: int = 1
    ) -> Dict[str, int]:
        """
        Build vocabulary from a list of texts.

        Args:
            texts: List of text strings
            min_freq: Minimum word frequency to include

        Returns:
            Word to index mapping
        """
        # Count words
        for text in texts:
            words = self.clean_text(text).split()
            self.word_counts.update(words)

        # Build vocabulary with special tokens
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }

        idx = len(self.vocab)
        for word, count in self.word_counts.most_common():
            if count >= min_freq:
                self.vocab[word] = idx
                idx += 1

        return self.vocab

    def extract_question_type(self, question: str) -> str:
        """
        Extract the type of surgical VQA question.

        Args:
            question: Question text

        Returns:
            Question type category
        """
        question = question.lower()

        # Define question type patterns
        type_patterns = {
            'instrument': ['instrument', 'tool', 'device'],
            'phase': ['phase', 'step', 'stage'],
            'action': ['action', 'doing', 'performing', 'happening'],
            'anatomy': ['anatomy', 'organ', 'tissue', 'structure'],
            'yes_no': ['is there', 'is the', 'are there', 'can you see'],
            'count': ['how many', 'count', 'number of'],
            'location': ['where', 'location', 'position'],
            'what': ['what is', 'what are'],
        }

        for qtype, patterns in type_patterns.items():
            if any(p in question for p in patterns):
                return qtype

        return 'other'


class AnnotationPreprocessor:
    """
    Preprocessing for surgical VQA annotations.

    Handles cleaning, validation, and transformation
    of annotation data.
    """

    def __init__(self):
        """Initialize annotation preprocessor."""
        self.text_preprocessor = TextPreprocessor()

    def validate_annotation(self, annotation: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single annotation entry.

        Args:
            annotation: Annotation dictionary

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required fields
        required_fields = ['question', 'answer']
        for field in required_fields:
            if field not in annotation or not annotation[field]:
                issues.append(f"Missing or empty field: {field}")

        # Check image reference
        if 'image_path' not in annotation and 'frame_id' not in annotation:
            issues.append("Missing image reference (image_path or frame_id)")

        return len(issues) == 0, issues

    def clean_annotations(
        self,
        annotations: List[Dict],
        remove_invalid: bool = True
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Clean and validate a list of annotations.

        Args:
            annotations: List of annotation dictionaries
            remove_invalid: Whether to remove invalid entries

        Returns:
            Tuple of (valid annotations, invalid annotations)
        """
        valid = []
        invalid = []

        for ann in annotations:
            is_valid, issues = self.validate_annotation(ann)

            if is_valid:
                # Clean text fields
                cleaned = ann.copy()
                cleaned['question'] = self.text_preprocessor.preprocess_question(
                    ann['question']
                )
                cleaned['answer'] = self.text_preprocessor.preprocess_answer(
                    ann['answer']
                )
                cleaned['question_type'] = self.text_preprocessor.extract_question_type(
                    ann['question']
                )
                valid.append(cleaned)
            else:
                if not remove_invalid:
                    invalid.append({'annotation': ann, 'issues': issues})

        return valid, invalid

    def balance_by_question_type(
        self,
        annotations: List[Dict],
        max_per_type: Optional[int] = None
    ) -> List[Dict]:
        """
        Balance annotations by question type.

        Args:
            annotations: List of annotations
            max_per_type: Maximum samples per question type

        Returns:
            Balanced list of annotations
        """
        # Group by question type
        by_type = {}
        for ann in annotations:
            qtype = ann.get('question_type', 'other')
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(ann)

        # Determine max samples
        if max_per_type is None:
            max_per_type = min(len(v) for v in by_type.values())

        # Sample from each type
        balanced = []
        for qtype, anns in by_type.items():
            if len(anns) > max_per_type:
                indices = np.random.choice(
                    len(anns), max_per_type, replace=False
                )
                balanced.extend([anns[i] for i in indices])
            else:
                balanced.extend(anns)

        return balanced


def preprocess_dataset(
    data_dir: str,
    annotations_file: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    augment: bool = False
) -> Dict[str, any]:
    """
    Full preprocessing pipeline for surgical VQA dataset.

    Args:
        data_dir: Directory containing raw images/videos
        annotations_file: Path to annotations file
        output_dir: Directory to save processed data
        image_size: Target image size
        augment: Whether to apply augmentation

    Returns:
        Dictionary with preprocessing statistics
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize preprocessors
    image_proc = ImagePreprocessor(
        target_size=image_size,
        augment=augment
    )
    ann_proc = AnnotationPreprocessor()

    # Load annotations
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    if isinstance(annotations, dict):
        annotations = annotations.get('annotations', [])

    # Clean annotations
    valid_anns, invalid_anns = ann_proc.clean_annotations(annotations)

    # Save processed annotations
    processed_ann_path = output_path / 'annotations_processed.json'
    with open(processed_ann_path, 'w') as f:
        json.dump(valid_anns, f, indent=2)

    # Build vocabulary
    text_proc = TextPreprocessor()
    all_text = [a['question'] for a in valid_anns] + [a['answer'] for a in valid_anns]
    vocab = text_proc.build_vocabulary(all_text)

    vocab_path = output_path / 'vocabulary.json'
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    # Statistics
    stats = {
        'total_annotations': len(annotations),
        'valid_annotations': len(valid_anns),
        'invalid_annotations': len(invalid_anns),
        'vocabulary_size': len(vocab),
        'question_types': Counter(a['question_type'] for a in valid_anns),
        'output_dir': str(output_path)
    }

    print(f"Preprocessing complete!")
    print(f"  Valid annotations: {stats['valid_annotations']}")
    print(f"  Invalid annotations: {stats['invalid_annotations']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")

    return stats


if __name__ == "__main__":
    print("PitVQA Preprocessing Module")
    print("-" * 40)

    # Example usage
    # stats = preprocess_dataset(
    #     data_dir="data/raw",
    #     annotations_file="data/annotations/qa_pairs.json",
    #     output_dir="data/processed"
    # )

    print("Module loaded successfully!")
    print("Available classes:")
    print("  - ImagePreprocessor: Image preprocessing pipeline")
    print("  - VideoPreprocessor: Video frame extraction and processing")
    print("  - TextPreprocessor: Text normalization and vocabulary")
    print("  - AnnotationPreprocessor: Annotation validation and cleaning")
