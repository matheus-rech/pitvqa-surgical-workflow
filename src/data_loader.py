"""
PitVQA Data Loader Module

Handles loading surgical video/image data and QA annotation pairs
for Visual Question Answering on pituitary surgery workflows.
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class SurgicalVQADataset(Dataset):
    """
    PyTorch Dataset for Surgical Visual Question Answering.

    Loads image/video frames paired with questions and answers
    for training VQA models on surgical procedures.
    """

    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        transform: Optional[transforms.Compose] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_seq_length: int = 128,
        split: str = "train"
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to processed frames directory
            annotations_file: Path to QA annotations (JSON or CSV)
            transform: Image transformations to apply
            image_size: Target image size (height, width)
            max_seq_length: Maximum sequence length for text
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.split = split

        # Load annotations
        self.annotations = self._load_annotations(annotations_file)

        # Set up image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def _load_annotations(self, annotations_file: str) -> List[Dict]:
        """Load QA annotations from file."""
        annotations_path = Path(annotations_file)

        if annotations_path.suffix == '.json':
            with open(annotations_path, 'r') as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get('annotations', [])

        elif annotations_path.suffix == '.csv':
            df = pd.read_csv(annotations_path)
            return df.to_dict('records')

        else:
            raise ValueError(f"Unsupported annotation format: {annotations_path.suffix}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - image: Transformed image tensor
                - question: Question text
                - answer: Answer text
                - image_path: Path to original image
        """
        annotation = self.annotations[idx]

        # Load image
        image_path = self.data_dir / annotation.get('image_path', annotation.get('frame_id', ''))
        image = self._load_image(image_path)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'question': annotation.get('question', ''),
            'answer': annotation.get('answer', ''),
            'image_path': str(image_path),
            'question_type': annotation.get('question_type', 'unknown')
        }

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        if image_path.exists():
            return Image.open(image_path).convert('RGB')
        else:
            # Return placeholder if image not found
            return Image.new('RGB', self.image_size, color='gray')


class VideoFrameExtractor:
    """
    Extract frames from surgical videos for VQA processing.
    """

    def __init__(
        self,
        output_dir: str,
        fps: int = 1,
        image_format: str = 'png'
    ):
        """
        Initialize frame extractor.

        Args:
            output_dir: Directory to save extracted frames
            fps: Frames per second to extract
            image_format: Output image format ('png', 'jpg')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.image_format = image_format

    def extract_frames(
        self,
        video_path: str,
        video_id: Optional[str] = None
    ) -> List[str]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to video file
            video_id: Optional identifier for the video

        Returns:
            List of paths to extracted frames
        """
        video_path = Path(video_path)
        video_id = video_id or video_path.stem

        # Create output subdirectory for this video
        video_output_dir = self.output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.fps)

        frame_paths = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save frame
                frame_path = video_output_dir / f"frame_{saved_count:06d}.{self.image_format}"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                frame_paths.append(str(frame_path))
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extracted {saved_count} frames from {video_path.name}")

        return frame_paths


def create_dataloader(
    data_dir: str,
    annotations_file: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    split: str = "train",
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the surgical VQA dataset.

    Args:
        data_dir: Path to processed frames
        annotations_file: Path to annotations
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        split: Dataset split
        **kwargs: Additional arguments for SurgicalVQADataset

    Returns:
        PyTorch DataLoader
    """
    dataset = SurgicalVQADataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        split=split,
        **kwargs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_vqa_batch
    )


def collate_vqa_batch(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List[str]]]:
    """
    Custom collate function for VQA batches.

    Handles variable-length text data alongside image tensors.
    """
    images = torch.stack([item['image'] for item in batch])
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    question_types = [item['question_type'] for item in batch]

    return {
        'images': images,
        'questions': questions,
        'answers': answers,
        'image_paths': image_paths,
        'question_types': question_types
    }


if __name__ == "__main__":
    # Example usage
    print("PitVQA Data Loader Module")
    print("-" * 40)

    # Example: Extract frames from video
    # extractor = VideoFrameExtractor(output_dir="data/processed")
    # frames = extractor.extract_frames("data/raw/surgery_001.mp4")

    # Example: Create dataset and dataloader
    # dataloader = create_dataloader(
    #     data_dir="data/processed",
    #     annotations_file="data/annotations/qa_pairs.json",
    #     batch_size=16
    # )

    print("Module loaded successfully!")
    print("Use SurgicalVQADataset for loading data")
    print("Use VideoFrameExtractor for processing videos")
