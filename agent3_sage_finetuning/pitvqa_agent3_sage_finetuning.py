"""
Agent 3: PitVQA SAGE/Molmo Fine-tuning Pipeline

Main orchestration module for fine-tuning SAGE/Molmo on pituitary surgery videos.

Pipeline:
┌─────────────────────────────────────────────────────────────────────────────┐
│  Agent 1 Output          Agent 2 Output                                    │
│  (109k frames)           (skill embeddings)                                │
│       │                        │                                           │
│       └────────────┬───────────┘                                           │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 1: Pointing Annotation                                       │   │
│  │  - Grounding DINO for instruments                                   │   │
│  │  - VLM pseudo-labels for anatomy                                    │   │
│  │  - Output: (x, y) coordinates + labels                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 2: Data Conversion                                           │   │
│  │  - Convert to Molmo conversation format                             │   │
│  │  - Add <point> annotations                                          │   │
│  │  - Generate SFT/DPO/GRPO datasets                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 3: Training (via HF Skills)                                  │   │
│  │  - SFT: Supervised fine-tuning with pointing                        │   │
│  │  - DPO: Preference learning (correct vs incorrect)                  │   │
│  │  - GRPO: RL with surgical reward functions                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Stage 4: Evaluation                                                │   │
│  │  - Pointing accuracy                                                │   │
│  │  - Phase/step classification                                        │   │
│  │  - Instrument detection                                             │   │
│  │  - VQA quality (BLEU, ROUGE, BERTScore)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                    │                                                        │
│                    ▼                                                        │
│            PitVQA-SAGE Model                                               │
│     (matheus-rech/pitvqa-sage-surgical)                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \\
        --input-dataset matheus-rech/pitvqa-processed \\
        --agent2-embeddings data/skill_embeddings \\
        --output-dir outputs/pitvqa-sage \\
        --method sft \\
        --base-model allenai/SAGE-MM-Molmo2-8B-SFT_RL \\
        --push-to-hub matheus-rech/pitvqa-sage-surgical

Reference: https://huggingface.co/blog/hf-skills-training
"""

import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Local imports
from .data_converter import (
    MolmoDataConverter,
    TrainingMethod,
    SurgicalAnatomyVocabulary,
)
from .pointing_annotator import (
    SurgicalPointingAnnotator,
    FrameAnnotation,
    generate_pointing_dataset,
)
from .hf_skills_trainer import (
    HFSkillsTrainer,
    TrainingConfig,
    GRPORewardConfig,
    HardwareTier,
    ModelSize,
    create_surgical_vqa_trainer,
)
from .surgical_evaluator import (
    SurgicalVQAEvaluator,
    EvaluationReport,
)

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


class PipelineStage(Enum):
    """Pipeline execution stages."""
    ANNOTATION = "annotation"
    CONVERSION = "conversion"
    TRAINING = "training"
    EVALUATION = "evaluation"
    ALL = "all"


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""

    # Input sources
    agent1_dataset: str  # HF dataset ID or local path
    agent2_embeddings: Optional[str] = None  # Agent 2 output path

    # Base model
    base_model: str = "allenai/SAGE-MM-Molmo2-8B-SFT_RL"

    # Output
    output_dir: str = "outputs/pitvqa-sage"
    output_model_name: str = "matheus-rech/pitvqa-sage-surgical"

    # Training method
    method: TrainingMethod = TrainingMethod.SFT

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    use_lora: bool = True
    lora_r: int = 16

    # Hardware
    hardware_tier: str = "a10g-large"

    # Annotation settings
    use_grounding_dino: bool = True
    use_vlm_pseudolabels: bool = False
    annotation_sample_rate: int = 5  # Annotate every Nth frame

    # Push to Hub
    push_to_hub: bool = True
    hf_token: Optional[str] = None

    # Evaluation
    eval_split: str = "test"
    max_eval_samples: int = 500

    # Stages to run
    stages: List[PipelineStage] = None

    def __post_init__(self):
        if self.stages is None:
            self.stages = [PipelineStage.ALL]


class PitVQASAGEPipeline:
    """
    Main pipeline for fine-tuning SAGE/Molmo on PitVQA surgical videos.

    Orchestrates:
    1. Pointing annotation generation
    2. Data format conversion
    3. HF Skills training
    4. Model evaluation
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.annotator = None
        self.converter = None
        self.trainer = None
        self.evaluator = None

        # Paths for intermediate outputs
        self.annotations_path = self.output_dir / "annotations.json"
        self.sft_dataset_path = self.output_dir / "sft_dataset"
        self.dpo_dataset_path = self.output_dir / "dpo_dataset"
        self.grpo_dataset_path = self.output_dir / "grpo_dataset"
        self.training_script_path = self.output_dir / f"train_{config.method.value}.py"
        self.eval_report_path = self.output_dir / "evaluation_report.json"

        # Run log
        self.run_log = {
            "start_time": datetime.now().isoformat(),
            "config": self._config_to_dict(),
            "stages_completed": [],
            "errors": []
        }

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to serializable dict."""
        return {
            "agent1_dataset": self.config.agent1_dataset,
            "agent2_embeddings": self.config.agent2_embeddings,
            "base_model": self.config.base_model,
            "output_model_name": self.config.output_model_name,
            "method": self.config.method.value,
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "use_lora": self.config.use_lora,
            "hardware_tier": self.config.hardware_tier,
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Returns:
            Dictionary with pipeline results and paths to outputs.
        """
        logger.info("=" * 60)
        logger.info("PitVQA SAGE/Molmo Fine-tuning Pipeline")
        logger.info("=" * 60)
        logger.info(f"Base model: {self.config.base_model}")
        logger.info(f"Training method: {self.config.method.value}")
        logger.info(f"Output: {self.config.output_model_name}")
        logger.info("=" * 60)

        results = {}
        stages_to_run = self.config.stages

        if PipelineStage.ALL in stages_to_run:
            stages_to_run = [
                PipelineStage.ANNOTATION,
                PipelineStage.CONVERSION,
                PipelineStage.TRAINING,
                PipelineStage.EVALUATION
            ]

        try:
            # Stage 1: Pointing Annotation
            if PipelineStage.ANNOTATION in stages_to_run:
                results["annotation"] = self._run_annotation_stage()
                self.run_log["stages_completed"].append("annotation")

            # Stage 2: Data Conversion
            if PipelineStage.CONVERSION in stages_to_run:
                results["conversion"] = self._run_conversion_stage()
                self.run_log["stages_completed"].append("conversion")

            # Stage 3: Training
            if PipelineStage.TRAINING in stages_to_run:
                results["training"] = self._run_training_stage()
                self.run_log["stages_completed"].append("training")

            # Stage 4: Evaluation
            if PipelineStage.EVALUATION in stages_to_run:
                results["evaluation"] = self._run_evaluation_stage()
                self.run_log["stages_completed"].append("evaluation")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.run_log["errors"].append(str(e))
            raise

        finally:
            # Save run log
            self.run_log["end_time"] = datetime.now().isoformat()
            log_path = self.output_dir / "pipeline_run.json"
            with open(log_path, 'w') as f:
                json.dump(self.run_log, f, indent=2)

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Outputs saved to: {self.output_dir}")
        logger.info("=" * 60)

        return results

    def _run_annotation_stage(self) -> Dict[str, Any]:
        """Stage 1: Generate pointing annotations."""
        logger.info("\n[Stage 1/4] Pointing Annotation")
        logger.info("-" * 40)

        # Check if annotations already exist
        if self.annotations_path.exists():
            logger.info(f"Found existing annotations: {self.annotations_path}")
            with open(self.annotations_path, 'r') as f:
                data = json.load(f)
            return {"num_frames": data.get("num_frames", 0), "path": str(self.annotations_path)}

        # Initialize annotator
        self.annotator = SurgicalPointingAnnotator(
            use_grounding_dino=self.config.use_grounding_dino,
            use_vlm_pseudolabels=self.config.use_vlm_pseudolabels,
            device="auto"
        )

        # Load Agent 1 dataset to get frame paths
        logger.info(f"Loading frames from: {self.config.agent1_dataset}")

        if self.config.agent1_dataset.startswith("matheus-rech/"):
            dataset = load_dataset(self.config.agent1_dataset, split="train")
        else:
            dataset = load_from_disk(self.config.agent1_dataset)

        # Extract frame paths and metadata
        frame_paths = []
        metadata = []

        for sample in tqdm(dataset, desc="Preparing frames"):
            if "image_path" in sample:
                frame_paths.append(sample["image_path"])
            elif "image" in sample:
                # Save image to temp file if needed
                # For now, skip embedded images
                continue

            metadata.append({
                "phase": sample.get("phase"),
                "step": sample.get("step"),
                "timestamp": sample.get("timestamp"),
            })

        if not frame_paths:
            logger.warning("No frame paths found in dataset. Skipping annotation.")
            return {"num_frames": 0, "skipped": True}

        # Generate annotations
        annotations = self.annotator.annotate_video_frames(
            frame_paths=frame_paths,
            metadata=metadata,
            output_path=str(self.annotations_path),
            sample_rate=self.config.annotation_sample_rate
        )

        total_detections = sum(len(a.detections) for a in annotations)
        logger.info(f"Generated {total_detections} detections for {len(annotations)} frames")

        return {
            "num_frames": len(annotations),
            "total_detections": total_detections,
            "path": str(self.annotations_path)
        }

    def _run_conversion_stage(self) -> Dict[str, Any]:
        """Stage 2: Convert to Molmo training format."""
        logger.info("\n[Stage 2/4] Data Conversion")
        logger.info("-" * 40)

        # Initialize converter
        self.converter = MolmoDataConverter(
            vocabulary=SurgicalAnatomyVocabulary(),
            include_pointing=True,
            include_temporal=True
        )

        # Determine output path based on method
        if self.config.method == TrainingMethod.SFT:
            output_path = self.sft_dataset_path
        elif self.config.method == TrainingMethod.DPO:
            output_path = self.dpo_dataset_path
        else:
            output_path = self.grpo_dataset_path

        # Create dataset
        logger.info(f"Converting to {self.config.method.value.upper()} format")

        dataset = self.converter.create_surgical_vqa_dataset(
            agent1_path=self.config.agent1_dataset,
            agent2_path=self.config.agent2_embeddings,
            training_method=self.config.method,
            output_path=str(output_path),
            push_to_hub=f"{self.config.output_model_name}-{self.config.method.value}-data" if self.config.push_to_hub else None,
            hf_token=self.config.hf_token or os.environ.get("HF_TOKEN")
        )

        result = {
            "train_samples": len(dataset["train"]),
            "val_samples": len(dataset["validation"]),
            "test_samples": len(dataset["test"]),
            "path": str(output_path),
            "method": self.config.method.value
        }

        logger.info(f"Created dataset with {result['train_samples']} training samples")

        return result

    def _run_training_stage(self) -> Dict[str, Any]:
        """Stage 3: Generate training configuration and script."""
        logger.info("\n[Stage 3/4] Training Configuration")
        logger.info("-" * 40)

        # Determine dataset path
        if self.config.method == TrainingMethod.SFT:
            dataset_path = str(self.sft_dataset_path)
        elif self.config.method == TrainingMethod.DPO:
            dataset_path = str(self.dpo_dataset_path)
        else:
            dataset_path = str(self.grpo_dataset_path)

        # If push_to_hub was used, use the hub dataset
        hub_dataset = f"{self.config.output_model_name}-{self.config.method.value}-data"

        # Create training config
        training_config = TrainingConfig(
            base_model=self.config.base_model,
            output_model_name=self.config.output_model_name,
            dataset_id=hub_dataset if self.config.push_to_hub else dataset_path,
            method=self.config.method,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            use_lora=self.config.use_lora,
            lora_r=self.config.lora_r,
            hardware_tier=HardwareTier(self.config.hardware_tier),
            push_to_hub=self.config.push_to_hub,
            hub_token=self.config.hf_token or os.environ.get("HF_TOKEN"),
            is_vision_model=True
        )

        # Create trainer
        reward_config = GRPORewardConfig() if self.config.method == TrainingMethod.GRPO else None
        self.trainer = HFSkillsTrainer(training_config, reward_config)

        # Generate training script
        script_path = self.trainer.generate_training_script(str(self.training_script_path))

        # Generate HF Skills prompt
        hf_prompt = self.trainer.generate_hf_skills_prompt()
        prompt_path = self.output_dir / "hf_skills_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(hf_prompt)

        logger.info(f"Training script saved to: {script_path}")
        logger.info(f"HF Skills prompt saved to: {prompt_path}")

        logger.info("\n" + "="*50)
        logger.info("TO START TRAINING:")
        logger.info("="*50)
        logger.info("\nOption 1: Run locally")
        logger.info(f"  python {script_path}")
        logger.info("\nOption 2: Use HF Skills (Claude Code)")
        logger.info(f"  Copy the prompt from: {prompt_path}")
        logger.info("="*50)

        return {
            "script_path": script_path,
            "prompt_path": str(prompt_path),
            "config": training_config.to_trl_config()
        }

    def _run_evaluation_stage(self) -> Dict[str, Any]:
        """Stage 4: Evaluate the fine-tuned model."""
        logger.info("\n[Stage 4/4] Evaluation")
        logger.info("-" * 40)

        # Check if model exists (might not if training hasn't run yet)
        model_path = self.config.output_model_name

        try:
            # Try to load from Hub or local
            self.evaluator = SurgicalVQAEvaluator(
                model_path=model_path,
                device="auto"
            )

            # Load test dataset
            if self.config.method == TrainingMethod.SFT:
                dataset_path = self.sft_dataset_path
            elif self.config.method == TrainingMethod.DPO:
                dataset_path = self.dpo_dataset_path
            else:
                dataset_path = self.grpo_dataset_path

            if dataset_path.exists():
                test_dataset = load_from_disk(str(dataset_path))["test"]
            else:
                logger.warning("Test dataset not found. Skipping evaluation.")
                return {"skipped": True, "reason": "Test dataset not found"}

            # Run evaluation
            report = self.evaluator.evaluate_dataset(
                dataset=test_dataset,
                output_path=str(self.eval_report_path),
                max_samples=self.config.max_eval_samples
            )

            logger.info("\nEvaluation Results:")
            logger.info(report.to_markdown())

            return report.to_dict()

        except Exception as e:
            logger.warning(f"Could not run evaluation: {e}")
            logger.info("Evaluation will be available after training completes.")
            return {
                "skipped": True,
                "reason": str(e),
                "instruction": "Run evaluation after training with: python -m agent3_sage_finetuning.surgical_evaluator"
            }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PitVQA SAGE/Molmo Fine-tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with SFT
    python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \\
        --input-dataset matheus-rech/pitvqa-processed \\
        --output-dir outputs/pitvqa-sage \\
        --method sft

    # GRPO training with custom rewards
    python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \\
        --input-dataset matheus-rech/pitvqa-processed \\
        --output-dir outputs/pitvqa-sage-grpo \\
        --method grpo \\
        --epochs 5

    # Only generate training config (skip annotation/conversion)
    python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \\
        --stages training \\
        --input-dataset matheus-rech/pitvqa-processed \\
        --output-dir outputs/pitvqa-sage
        """
    )

    # Input/output
    parser.add_argument("--input-dataset", required=True,
                       help="Agent 1 output: HuggingFace dataset ID or local path")
    parser.add_argument("--agent2-embeddings",
                       help="Agent 2 output: skill embeddings path")
    parser.add_argument("--output-dir", default="outputs/pitvqa-sage",
                       help="Output directory for all artifacts")
    parser.add_argument("--output-model", default="matheus-rech/pitvqa-sage-surgical",
                       help="Name for the fine-tuned model on HuggingFace")

    # Model selection
    parser.add_argument("--base-model",
                       default="allenai/SAGE-MM-Molmo2-8B-SFT_RL",
                       help="Base SAGE/Molmo model to fine-tune")

    # Training method
    parser.add_argument("--method", choices=["sft", "dpo", "grpo"],
                       default="sft",
                       help="Training method: SFT, DPO, or GRPO")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--no-lora", action="store_true",
                       help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--hardware", default="a10g-large",
                       choices=["t4-small", "t4-medium", "a10g-small", "a10g-large"])

    # Annotation settings
    parser.add_argument("--use-grounding-dino", action="store_true", default=True)
    parser.add_argument("--use-vlm-labels", action="store_true",
                       help="Use VLM pseudo-labeling for anatomy")
    parser.add_argument("--annotation-sample-rate", type=int, default=5,
                       help="Annotate every Nth frame")

    # Hub settings
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--no-push", action="store_true",
                       help="Don't push to HuggingFace Hub")
    parser.add_argument("--hf-token", help="HuggingFace token")

    # Pipeline stages
    parser.add_argument("--stages", nargs="+",
                       choices=["annotation", "conversion", "training", "evaluation", "all"],
                       default=["all"],
                       help="Pipeline stages to run")

    # Evaluation
    parser.add_argument("--max-eval-samples", type=int, default=500)

    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Build config
    config = PipelineConfig(
        agent1_dataset=args.input_dataset,
        agent2_embeddings=args.agent2_embeddings,
        base_model=args.base_model,
        output_dir=args.output_dir,
        output_model_name=args.output_model,
        method=TrainingMethod(args.method),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lora=not args.no_lora,
        lora_r=args.lora_r,
        hardware_tier=args.hardware,
        use_grounding_dino=args.use_grounding_dino,
        use_vlm_pseudolabels=args.use_vlm_labels,
        annotation_sample_rate=args.annotation_sample_rate,
        push_to_hub=args.push_to_hub and not args.no_push,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        max_eval_samples=args.max_eval_samples,
        stages=[PipelineStage(s) for s in args.stages]
    )

    # Run pipeline
    pipeline = PitVQASAGEPipeline(config)
    results = pipeline.run()

    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    for stage, result in results.items():
        print(f"\n{stage.upper()}:")
        if isinstance(result, dict):
            for k, v in result.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {result}")
    print("="*60)


if __name__ == "__main__":
    main()
