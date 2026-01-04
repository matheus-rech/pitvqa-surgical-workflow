#!/usr/bin/env python3
"""
Configuration for Surgical Annotation Pipeline
===============================================
Centralized configuration for API keys, models, and annotation parameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class APIConfig:
    """API configuration for multi-agent annotation"""

    # Claude (Primary Annotator)
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    claude_model: str = "claude-opus-4-5-20250514"

    # Gemini (Validator)
    google_api_key: str = field(default_factory=lambda: os.environ.get("GOOGLE_API_KEY", ""))
    gemini_model: str = "gemini-2.5-pro-preview-06-05"

    # OpenAI (Tiebreaker)
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    gpt_model: str = "gpt-4o"  # Or gpt-5.2 when available

    def validate(self) -> dict:
        """Check which APIs are available"""
        return {
            "claude": bool(self.anthropic_api_key),
            "gemini": bool(self.google_api_key),
            "gpt": bool(self.openai_api_key),
            "all_available": all([
                self.anthropic_api_key,
                self.google_api_key,
                self.openai_api_key
            ])
        }


@dataclass
class AnnotationConfig:
    """Configuration for annotation parameters"""

    # Consensus parameters
    agreement_threshold: float = 0.8  # Min agreement for consensus
    distance_threshold: float = 5.0   # Max distance for same point (% of image)
    confidence_threshold: float = 0.7  # Min confidence to include annotation

    # Processing parameters
    batch_size: int = 50
    max_retries: int = 3
    timeout_seconds: int = 60

    # Output format
    output_format: str = "molmo_videopoint"  # or "coco", "yolo", "custom"
    save_intermediate: bool = True

    # Image processing
    max_image_size: int = 1024  # Max dimension for API calls
    jpeg_quality: int = 95


@dataclass
class SurgicalCategories:
    """Surgical annotation categories for pituitary surgery"""

    instruments: List[str] = field(default_factory=lambda: [
        "pituitary_forceps",
        "suction_cannula",
        "curette",
        "ring_curette",
        "endoscope",
        "bipolar_cautery",
        "drill",
        "dissector",
        "scissors",
        "speculum",
        "doppler_probe",
        "micro_hook",
        "tumor_forceps",
        "irrigation_cannula",
        "cottonoid",
        "hemostatic_agent"
    ])

    anatomy: List[str] = field(default_factory=lambda: [
        "tumor",
        "pituitary_gland",
        "carotid_artery",
        "optic_nerve",
        "optic_chiasm",
        "sella_turcica",
        "sphenoid_sinus",
        "dura_mater",
        "diaphragma_sellae",
        "clivus",
        "posterior_clinoid",
        "anterior_clinoid",
        "suprasellar_cistern",
        "arachnoid",
        "tuberculum_sellae",
        "planum_sphenoidale",
        "medial_carotid_wall",
        "cavernous_sinus"
    ])

    events: List[str] = field(default_factory=lambda: [
        "active_bleeding",
        "tumor_removal",
        "cauterization",
        "irrigation",
        "dissection",
        "drilling",
        "hemostasis",
        "tissue_retraction",
        "dura_opening",
        "dura_closure",
        "fat_graft_placement",
        "nasoseptal_flap"
    ])

    surgical_phases: List[str] = field(default_factory=lambda: [
        "nasal_phase",
        "sphenoid_phase",
        "sellar_phase",
        "tumor_removal_phase",
        "reconstruction_phase",
        "closure_phase"
    ])

    def all_labels(self) -> List[str]:
        """Get all possible labels"""
        return self.instruments + self.anatomy + self.events

    def get_category(self, label: str) -> Optional[str]:
        """Get category for a label"""
        if label in self.instruments:
            return "instruments"
        elif label in self.anatomy:
            return "anatomy"
        elif label in self.events:
            return "events"
        return None


@dataclass
class DatasetConfig:
    """Dataset configuration"""

    # Source dataset
    dataset_name: str = "mmrech/pitvqa-sage-sft"
    split: str = "train"

    # Output dataset
    output_name: str = "mmrech/pitvqa-surgical-pointing"

    # Processing limits
    max_samples: Optional[int] = None
    skip_samples: int = 0

    # Column mappings
    image_column: str = "image"
    video_id_column: str = "video_id"
    frame_id_column: str = "frame_id"


@dataclass
class PipelineConfig:
    """Complete pipeline configuration"""

    api: APIConfig = field(default_factory=APIConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    categories: SurgicalCategories = field(default_factory=SurgicalCategories)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # Output directory
    output_dir: str = "./surgical_pointing_annotations"

    # Logging
    verbose: bool = True
    log_file: Optional[str] = None


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the default configuration"""
    return DEFAULT_CONFIG


def print_config_status():
    """Print configuration status"""
    config = get_config()
    api_status = config.api.validate()

    print("=" * 60)
    print("Surgical Annotation Pipeline Configuration")
    print("=" * 60)
    print("\nAPI Status:")
    print(f"  Claude (Opus 4.5):  {'✓' if api_status['claude'] else '✗'}")
    print(f"  Gemini (3 Pro):     {'✓' if api_status['gemini'] else '✗'}")
    print(f"  GPT (Tiebreaker):   {'✓' if api_status['gpt'] else '✗'}")
    print(f"\nAll APIs Ready: {'✓ YES' if api_status['all_available'] else '✗ NO'}")

    print("\nAnnotation Parameters:")
    print(f"  Agreement Threshold: {config.annotation.agreement_threshold}")
    print(f"  Distance Threshold:  {config.annotation.distance_threshold}%")
    print(f"  Confidence Threshold: {config.annotation.confidence_threshold}")

    print("\nCategories:")
    print(f"  Instruments: {len(config.categories.instruments)}")
    print(f"  Anatomy:     {len(config.categories.anatomy)}")
    print(f"  Events:      {len(config.categories.events)}")
    print(f"  Total:       {len(config.categories.all_labels())}")

    print("\nDataset:")
    print(f"  Source: {config.dataset.dataset_name}")
    print(f"  Output: {config.dataset.output_name}")
    print("=" * 60)


if __name__ == "__main__":
    print_config_status()
