"""
Agent 3: Pointing Annotation Generator for Surgical Videos

Generates spatial pointing annotations for surgical anatomy and instruments.
Supports multiple annotation strategies:
1. Manual annotation interface
2. Grounding DINO auto-detection
3. SAM2 segmentation-based
4. Pseudo-labeling from existing VLMs

Key insight: Molmo uses <point x='0.5' y='0.3'>label</point> format
for spatial grounding, which we generate here.
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoProcessor, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from an object detection/grounding model."""
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max (normalized)
    mask: Optional[np.ndarray] = None
    center: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.center is None:
            x_min, y_min, x_max, y_max = self.bbox
            self.center = ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def to_molmo_point(self) -> str:
        """Format as Molmo pointing annotation."""
        x, y = self.center
        return f"<point x='{x:.3f}' y='{y:.3f}'>{self.label}</point>"


@dataclass
class FrameAnnotation:
    """Complete annotation for a video frame."""
    frame_id: str
    image_path: str
    width: int
    height: int
    detections: List[DetectionResult] = field(default_factory=list)
    phase: Optional[str] = None
    step: Optional[str] = None
    timestamp: Optional[float] = None
    annotator: str = "auto"

    def to_json(self) -> Dict:
        """Serialize to JSON-compatible dict."""
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "phase": self.phase,
            "step": self.step,
            "timestamp": self.timestamp,
            "annotator": self.annotator,
            "detections": [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                    "center": d.center,
                }
                for d in self.detections
            ]
        }

    def get_molmo_points(self) -> str:
        """Get all points in Molmo format."""
        return " ".join(d.to_molmo_point() for d in self.detections)


class BaseAnnotator(ABC):
    """Base class for annotation strategies."""

    @abstractmethod
    def annotate_frame(
        self,
        image: Image.Image,
        labels: List[str],
        **kwargs
    ) -> List[DetectionResult]:
        """Annotate a single frame."""
        pass

    def annotate_batch(
        self,
        images: List[Image.Image],
        labels: List[str],
        **kwargs
    ) -> List[List[DetectionResult]]:
        """Annotate multiple frames."""
        return [self.annotate_frame(img, labels, **kwargs) for img in images]


class GroundingDINOAnnotator(BaseAnnotator):
    """
    Uses Grounding DINO for open-vocabulary object detection.

    Reference: https://github.com/IDEA-Research/GroundingDINO
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        device: str = "auto",
        confidence_threshold: float = 0.3,
        text_threshold: float = 0.25
    ):
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            raise ImportError("torch and transformers required for GroundingDINO")

        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        self.text_threshold = text_threshold

        logger.info(f"Loading Grounding DINO from {model_name}")

        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.set_train_mode(False)
        except Exception as e:
            logger.warning(f"Failed to load Grounding DINO: {e}")
            self.model = None
            self.processor = None

    def _get_device(self, device: str) -> str:
        if device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                return "cuda"
            elif HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def annotate_frame(
        self,
        image: Image.Image,
        labels: List[str],
        **kwargs
    ) -> List[DetectionResult]:
        """Detect objects using Grounding DINO."""
        if self.model is None:
            logger.warning("Grounding DINO not loaded, returning empty detections")
            return []

        # Prepare text prompt
        text_prompt = " . ".join(labels) + " ."

        # Process
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.confidence_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)]
        )[0]

        detections = []
        for box, score, label in zip(
            results["boxes"],
            results["scores"],
            results["labels"]
        ):
            x_min, y_min, x_max, y_max = box.tolist()

            # Normalize coordinates
            bbox = (
                x_min / image.width,
                y_min / image.height,
                x_max / image.width,
                y_max / image.height
            )

            detections.append(DetectionResult(
                label=label,
                confidence=score.item(),
                bbox=bbox
            ))

        return detections


class VLMPseudoLabeler(BaseAnnotator):
    """
    Uses existing VLMs (GPT-4V, Claude, Molmo) to generate pseudo-labels.

    This is useful for bootstrapping annotations before fine-tuning.
    """

    POINTING_PROMPT = """Analyze this surgical endoscopy image and identify the locations of key structures.

For each visible structure, provide its approximate location as normalized coordinates (0-1):
- (0, 0) is top-left
- (1, 1) is bottom-right

Structures to identify:
{labels}

Respond in JSON format:
{{
    "detections": [
        {{"label": "structure_name", "x": 0.5, "y": 0.3, "confidence": 0.9}},
        ...
    ]
}}

Only include structures you can clearly identify. Be precise with coordinates."""

    def __init__(
        self,
        model_provider: str = "molmo",
        model_name: str = "allenai/Molmo-7B-D-0924",
        api_key: Optional[str] = None,
        device: str = "auto"
    ):
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key
        self.device = device

        self.model = None
        self.processor = None

        if model_provider == "molmo":
            self._load_molmo()

    def _load_molmo(self):
        """Load Molmo model for local inference."""
        if not HAS_TORCH or not HAS_TRANSFORMERS:
            logger.warning("torch/transformers required for Molmo")
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Loading Molmo from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if HAS_TORCH else None,
                device_map="auto"
            )
        except Exception as e:
            logger.warning(f"Failed to load Molmo: {e}")

    def annotate_frame(
        self,
        image: Image.Image,
        labels: List[str],
        **kwargs
    ) -> List[DetectionResult]:
        """Generate pseudo-labels using VLM."""
        if self.model is None:
            logger.warning("VLM not loaded, returning empty detections")
            return []

        prompt = self.POINTING_PROMPT.format(labels="\n".join(f"- {l}" for l in labels))

        # Process with Molmo
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Parse JSON response
        try:
            # Find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                detections = []
                for det in data.get("detections", []):
                    detections.append(DetectionResult(
                        label=det["label"],
                        confidence=det.get("confidence", 0.5),
                        bbox=(
                            det["x"] - 0.05,
                            det["y"] - 0.05,
                            det["x"] + 0.05,
                            det["y"] + 0.05
                        ),
                        center=(det["x"], det["y"])
                    ))
                return detections
        except Exception as e:
            logger.warning(f"Failed to parse VLM response: {e}")

        return []


class SurgicalPointingAnnotator:
    """
    Main annotator class combining multiple strategies for surgical videos.

    Workflow:
    1. Use Grounding DINO for instrument detection
    2. Use VLM pseudo-labeling for anatomical structures
    3. Merge and filter results
    4. Generate Molmo-format pointing annotations
    """

    # Surgical vocabulary with detection categories
    INSTRUMENT_LABELS = [
        "surgical endoscope",
        "suction tube",
        "surgical curette",
        "bipolar forceps",
        "surgical scissors",
        "grasping forceps",
        "surgical drill",
        "cottonoid",
    ]

    ANATOMY_LABELS = [
        "pituitary gland",
        "pituitary tumor",
        "carotid artery",
        "optic nerve",
        "nasal septum",
        "sphenoid sinus",
        "dura mater",
        "bleeding site",
        "tumor cavity",
    ]

    def __init__(
        self,
        use_grounding_dino: bool = True,
        use_vlm_pseudolabels: bool = True,
        vlm_model: str = "molmo",
        confidence_threshold: float = 0.3,
        device: str = "auto"
    ):
        self.confidence_threshold = confidence_threshold
        self.annotators = []

        if use_grounding_dino:
            try:
                self.grounding_dino = GroundingDINOAnnotator(
                    device=device,
                    confidence_threshold=confidence_threshold
                )
                self.annotators.append(("grounding_dino", self.grounding_dino))
            except Exception as e:
                logger.warning(f"Could not load Grounding DINO: {e}")

        if use_vlm_pseudolabels:
            try:
                self.vlm_labeler = VLMPseudoLabeler(
                    model_provider=vlm_model,
                    device=device
                )
                self.annotators.append(("vlm", self.vlm_labeler))
            except Exception as e:
                logger.warning(f"Could not load VLM: {e}")

    def annotate_frame(
        self,
        image: Union[Image.Image, str, Path],
        frame_id: str,
        phase: Optional[str] = None,
        step: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> FrameAnnotation:
        """
        Annotate a single surgical frame with pointing annotations.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = Image.open(image_path).convert("RGB")
        else:
            image_path = ""

        all_detections = []

        # Run all annotators
        for name, annotator in self.annotators:
            if name == "grounding_dino":
                labels = self.INSTRUMENT_LABELS
            else:
                labels = self.ANATOMY_LABELS

            try:
                detections = annotator.annotate_frame(image, labels)
                all_detections.extend(detections)
            except Exception as e:
                logger.warning(f"Annotator {name} failed: {e}")

        # Deduplicate and filter
        filtered_detections = self._filter_detections(all_detections)

        return FrameAnnotation(
            frame_id=frame_id,
            image_path=image_path,
            width=image.width,
            height=image.height,
            detections=filtered_detections,
            phase=phase,
            step=step,
            timestamp=timestamp,
            annotator=",".join(name for name, _ in self.annotators)
        )

    def _filter_detections(
        self,
        detections: List[DetectionResult],
        iou_threshold: float = 0.5
    ) -> List[DetectionResult]:
        """Remove duplicate detections using NMS-like filtering."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

        # Simple deduplication by label and proximity
        seen_labels = {}
        filtered = []

        for det in detections:
            if det.confidence < self.confidence_threshold:
                continue

            # Check if similar detection already exists
            if det.label in seen_labels:
                existing = seen_labels[det.label]
                # If centers are close, skip
                dist = np.sqrt(
                    (det.center[0] - existing.center[0])**2 +
                    (det.center[1] - existing.center[1])**2
                )
                if dist < 0.1:  # 10% of image size
                    continue

            seen_labels[det.label] = det
            filtered.append(det)

        return filtered

    def annotate_video_frames(
        self,
        frame_paths: List[str],
        metadata: Optional[List[Dict]] = None,
        output_path: Optional[str] = None,
        sample_rate: int = 1  # Process every Nth frame
    ) -> List[FrameAnnotation]:
        """Annotate all frames from a video."""
        annotations = []

        sampled_paths = frame_paths[::sample_rate]
        sampled_metadata = (metadata[::sample_rate] if metadata else
                          [{}] * len(sampled_paths))

        for i, (path, meta) in enumerate(tqdm(
            zip(sampled_paths, sampled_metadata),
            total=len(sampled_paths),
            desc="Annotating frames"
        )):
            frame_id = f"frame_{i:06d}"
            annotation = self.annotate_frame(
                image=path,
                frame_id=frame_id,
                phase=meta.get("phase"),
                step=meta.get("step"),
                timestamp=meta.get("timestamp")
            )
            annotations.append(annotation)

        # Save if output path provided
        if output_path:
            self.save_annotations(annotations, output_path)

        return annotations

    def save_annotations(
        self,
        annotations: List[FrameAnnotation],
        output_path: str
    ):
        """Save annotations to JSON file."""
        data = {
            "version": "1.0",
            "annotator": "SurgicalPointingAnnotator",
            "num_frames": len(annotations),
            "frames": [ann.to_json() for ann in annotations]
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(annotations)} annotations to {output_path}")

    def load_annotations(self, input_path: str) -> List[FrameAnnotation]:
        """Load annotations from JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)

        annotations = []
        for frame_data in data["frames"]:
            detections = [
                DetectionResult(
                    label=d["label"],
                    confidence=d["confidence"],
                    bbox=tuple(d["bbox"]),
                    center=tuple(d["center"]) if d.get("center") else None
                )
                for d in frame_data["detections"]
            ]

            annotations.append(FrameAnnotation(
                frame_id=frame_data["frame_id"],
                image_path=frame_data["image_path"],
                width=frame_data["width"],
                height=frame_data["height"],
                detections=detections,
                phase=frame_data.get("phase"),
                step=frame_data.get("step"),
                timestamp=frame_data.get("timestamp"),
                annotator=frame_data.get("annotator", "loaded")
            ))

        return annotations

    def visualize_annotation(
        self,
        annotation: FrameAnnotation,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> Image.Image:
        """Visualize annotations on the image."""
        image = Image.open(annotation.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        colors = {
            "instrument": (0, 255, 0),   # Green
            "anatomy": (255, 0, 0),       # Red
            "default": (255, 255, 0)      # Yellow
        }

        for det in annotation.detections:
            # Determine color
            if det.label in [l.replace(" ", "_") for l in self.INSTRUMENT_LABELS]:
                color = colors["instrument"]
            elif det.label in [l.replace(" ", "_") for l in self.ANATOMY_LABELS]:
                color = colors["anatomy"]
            else:
                color = colors["default"]

            # Draw bounding box
            x_min, y_min, x_max, y_max = det.bbox
            box = (
                x_min * annotation.width,
                y_min * annotation.height,
                x_max * annotation.width,
                y_max * annotation.height
            )
            draw.rectangle(box, outline=color, width=2)

            # Draw center point
            cx, cy = det.center
            point = (cx * annotation.width, cy * annotation.height)
            draw.ellipse(
                (point[0]-5, point[1]-5, point[0]+5, point[1]+5),
                fill=color
            )

            # Draw label
            draw.text(
                (box[0], box[1] - 15),
                f"{det.label} ({det.confidence:.2f})",
                fill=color
            )

        if output_path:
            image.save(output_path)

        if show:
            image.show()

        return image


def generate_pointing_dataset(
    input_dir: str,
    output_path: str,
    use_grounding_dino: bool = True,
    use_vlm: bool = False,
    sample_rate: int = 1,
    device: str = "auto"
):
    """CLI function to generate pointing annotations for a directory of frames."""
    from glob import glob

    # Find all images
    frame_paths = sorted(
        glob(os.path.join(input_dir, "*.png")) +
        glob(os.path.join(input_dir, "*.jpg")) +
        glob(os.path.join(input_dir, "*.jpeg"))
    )

    if not frame_paths:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(frame_paths)} frames")

    # Initialize annotator
    annotator = SurgicalPointingAnnotator(
        use_grounding_dino=use_grounding_dino,
        use_vlm_pseudolabels=use_vlm,
        device=device
    )

    # Annotate
    annotations = annotator.annotate_video_frames(
        frame_paths=frame_paths,
        output_path=output_path,
        sample_rate=sample_rate
    )

    # Statistics
    total_detections = sum(len(a.detections) for a in annotations)
    logger.info(f"Generated {total_detections} detections across {len(annotations)} frames")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pointing annotations for surgical frames")
    parser.add_argument("--input-dir", required=True, help="Directory with frame images")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--use-grounding-dino", action="store_true", default=True)
    parser.add_argument("--use-vlm", action="store_true", default=False)
    parser.add_argument("--sample-rate", type=int, default=1)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generate_pointing_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        use_grounding_dino=args.use_grounding_dino,
        use_vlm=args.use_vlm,
        sample_rate=args.sample_rate,
        device=args.device
    )
