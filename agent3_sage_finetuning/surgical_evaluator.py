"""
Agent 3: Surgical VQA Evaluation Pipeline

Evaluates fine-tuned SAGE/Molmo models on surgical video understanding tasks:
1. Pointing accuracy (spatial grounding)
2. Phase classification
3. Step classification
4. Instrument detection
5. VQA quality (BLEU, ROUGE, BERTScore)

Outputs evaluation reports compatible with MICCAI/medical AI benchmarks.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from datasets import Dataset, load_dataset, load_from_disk
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

logger = logging.getLogger(__name__)


@dataclass
class PointingEvalResult:
    """Results for pointing/spatial grounding evaluation."""
    num_samples: int = 0
    correct_points: int = 0
    total_pred_points: int = 0
    total_gt_points: int = 0

    # Per-class results
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)

    # Distance metrics
    mean_distance: float = 0.0
    median_distance: float = 0.0

    @property
    def precision(self) -> float:
        if self.total_pred_points == 0:
            return 0.0
        return self.correct_points / self.total_pred_points

    @property
    def recall(self) -> float:
        if self.total_gt_points == 0:
            return 0.0
        return self.correct_points / self.total_gt_points

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)


@dataclass
class ClassificationEvalResult:
    """Results for phase/step/instrument classification."""
    task_name: str
    num_classes: int
    num_samples: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0

    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class VQAEvalResult:
    """Results for VQA quality metrics."""
    num_samples: int = 0
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    rouge_l: float = 0.0
    meteor: float = 0.0
    bert_score: float = 0.0

    # Surgical-specific
    anatomical_accuracy: float = 0.0
    instrument_accuracy: float = 0.0


@dataclass
class EvaluationReport:
    """Complete evaluation report for surgical VQA model."""
    model_name: str
    dataset_name: str
    num_samples: int

    pointing: Optional[PointingEvalResult] = None
    phase_classification: Optional[ClassificationEvalResult] = None
    step_classification: Optional[ClassificationEvalResult] = None
    instrument_detection: Optional[ClassificationEvalResult] = None
    vqa_quality: Optional[VQAEvalResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
        }

        if self.pointing:
            result["pointing"] = {
                "precision": self.pointing.precision,
                "recall": self.pointing.recall,
                "f1": self.pointing.f1,
                "mean_distance": self.pointing.mean_distance,
                "per_class_accuracy": self.pointing.per_class_accuracy,
            }

        if self.phase_classification:
            result["phase_classification"] = {
                "accuracy": self.phase_classification.accuracy,
                "macro_f1": self.phase_classification.macro_f1,
                "weighted_f1": self.phase_classification.weighted_f1,
                "per_class_f1": self.phase_classification.per_class_f1,
            }

        if self.step_classification:
            result["step_classification"] = {
                "accuracy": self.step_classification.accuracy,
                "macro_f1": self.step_classification.macro_f1,
                "weighted_f1": self.step_classification.weighted_f1,
            }

        if self.instrument_detection:
            result["instrument_detection"] = {
                "accuracy": self.instrument_detection.accuracy,
                "macro_f1": self.instrument_detection.macro_f1,
            }

        if self.vqa_quality:
            result["vqa_quality"] = {
                "bleu_1": self.vqa_quality.bleu_1,
                "bleu_4": self.vqa_quality.bleu_4,
                "rouge_l": self.vqa_quality.rouge_l,
                "bert_score": self.vqa_quality.bert_score,
            }

        return result

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Surgical VQA Evaluation Report",
            f"",
            f"**Model:** {self.model_name}",
            f"**Dataset:** {self.dataset_name}",
            f"**Samples:** {self.num_samples}",
            f"",
        ]

        if self.pointing:
            lines.extend([
                "## Pointing/Spatial Grounding",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Precision | {self.pointing.precision:.3f} |",
                f"| Recall | {self.pointing.recall:.3f} |",
                f"| F1 | {self.pointing.f1:.3f} |",
                f"| Mean Distance | {self.pointing.mean_distance:.3f} |",
                f"",
            ])

        if self.phase_classification:
            lines.extend([
                "## Phase Classification",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Accuracy | {self.phase_classification.accuracy:.3f} |",
                f"| Macro F1 | {self.phase_classification.macro_f1:.3f} |",
                f"| Weighted F1 | {self.phase_classification.weighted_f1:.3f} |",
                f"",
            ])

        if self.step_classification:
            lines.extend([
                "## Step Classification",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Accuracy | {self.step_classification.accuracy:.3f} |",
                f"| Macro F1 | {self.step_classification.macro_f1:.3f} |",
                f"",
            ])

        if self.vqa_quality:
            lines.extend([
                "## VQA Quality",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| BLEU-1 | {self.vqa_quality.bleu_1:.3f} |",
                f"| BLEU-4 | {self.vqa_quality.bleu_4:.3f} |",
                f"| ROUGE-L | {self.vqa_quality.rouge_l:.3f} |",
                f"| BERTScore | {self.vqa_quality.bert_score:.3f} |",
                f"",
            ])

        return "\n".join(lines)


class SurgicalVQAEvaluator:
    """
    Evaluator for surgical video understanding models.

    Supports evaluation of:
    - Spatial grounding (pointing accuracy)
    - Temporal grounding (phase/step classification)
    - Instrument detection
    - Free-form VQA quality
    """

    PHASES = ["nasal_phase", "sellar_phase", "tumor_removal_phase", "closure_phase"]

    STEPS = [
        "septal_dissection", "turbinectomy", "sphenoidotomy",
        "posterior_septectomy", "sellar_floor_removal", "dura_opening",
        "tumor_resection", "hemostasis", "reconstruction", "nasal_packing",
        "visualization", "instrument_change", "suction", "irrigation", "other"
    ]

    INSTRUMENTS = [
        "endoscope", "suction", "curette", "bipolar", "monopolar",
        "scissors", "grasper", "drill", "kerrison", "speculum",
        "cottonoid", "hemostatic_agent", "fat_graft", "fascia",
        "nasoseptal_flap"
    ]

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        pointing_threshold: float = 0.1
    ):
        self.model_path = model_path
        self.pointing_threshold = pointing_threshold

        # Device setup
        if device == "auto":
            if HAS_TORCH and torch.cuda.is_available():
                self.device = "cuda"
            elif HAS_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.processor = None

    def load_model(self):
        """Load model for evaluation."""
        if not HAS_TORCH:
            raise ImportError("torch required for model evaluation")

        logger.info(f"Loading model from {self.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.set_train_mode(False)

    def generate_prediction(
        self,
        image: Any,
        question: str,
        max_tokens: int = 256
    ) -> str:
        """Generate model prediction for a single sample."""
        if self.model is None:
            self.load_model()

        inputs = self.processor(
            text=question,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def evaluate_pointing(
        self,
        predictions: List[str],
        ground_truths: List[Dict[str, Any]]
    ) -> PointingEvalResult:
        """Evaluate pointing/spatial grounding accuracy."""
        result = PointingEvalResult()

        distances = []
        per_class_correct = {}
        per_class_total = {}

        for pred, gt in zip(predictions, ground_truths):
            result.num_samples += 1

            # Extract predicted points
            point_pattern = r"<point x='([\d.]+)' y='([\d.]+)'>([^<]+)</point>"
            pred_points = re.findall(point_pattern, pred)
            gt_points = gt.get("points", [])

            result.total_pred_points += len(pred_points)
            result.total_gt_points += len(gt_points)

            # Match predictions to ground truth
            for px, py, label in pred_points:
                px, py = float(px), float(py)
                label_clean = label.strip().lower()

                for gt_point in gt_points:
                    gt_label = gt_point["label"].lower()

                    if gt_label in label_clean or label_clean in gt_label:
                        dist = ((px - gt_point["x"])**2 + (py - gt_point["y"])**2)**0.5
                        distances.append(dist)

                        # Per-class tracking
                        if gt_label not in per_class_total:
                            per_class_total[gt_label] = 0
                            per_class_correct[gt_label] = 0
                        per_class_total[gt_label] += 1

                        if dist < self.pointing_threshold:
                            result.correct_points += 1
                            per_class_correct[gt_label] += 1
                        break

        # Compute per-class accuracy
        for label in per_class_total:
            if per_class_total[label] > 0:
                result.per_class_accuracy[label] = (
                    per_class_correct.get(label, 0) / per_class_total[label]
                )

        # Distance metrics
        if distances:
            result.mean_distance = float(np.mean(distances))
            result.median_distance = float(np.median(distances))

        return result

    def evaluate_classification(
        self,
        predictions: List[str],
        ground_truths: List[str],
        classes: List[str],
        task_name: str
    ) -> ClassificationEvalResult:
        """Evaluate classification task (phase/step)."""
        if not HAS_SKLEARN:
            logger.warning("sklearn required for classification metrics")
            return ClassificationEvalResult(task_name=task_name, num_classes=len(classes))

        # Convert predictions to class labels
        pred_labels = []
        for pred in predictions:
            pred_lower = pred.lower()
            found = None
            for cls in classes:
                if cls.replace("_", " ") in pred_lower:
                    found = cls
                    break
            pred_labels.append(found or "unknown")

        # Filter to valid samples
        valid_pairs = [
            (p, g) for p, g in zip(pred_labels, ground_truths)
            if g in classes
        ]

        if not valid_pairs:
            return ClassificationEvalResult(task_name=task_name, num_classes=len(classes))

        y_pred, y_true = zip(*valid_pairs)

        # Compute metrics
        result = ClassificationEvalResult(
            task_name=task_name,
            num_classes=len(classes),
            num_samples=len(valid_pairs)
        )

        result.accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, average=None, zero_division=0
        )

        result.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        result.weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        for i, cls in enumerate(classes):
            result.per_class_precision[cls] = precision[i]
            result.per_class_recall[cls] = recall[i]
            result.per_class_f1[cls] = f1[i]

        result.confusion_matrix = confusion_matrix(y_true, y_pred, labels=classes)

        return result

    def evaluate_vqa_quality(
        self,
        predictions: List[str],
        references: List[str]
    ) -> VQAEvalResult:
        """Evaluate free-form VQA quality using NLG metrics."""
        result = VQAEvalResult(num_samples=len(predictions))

        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            from rouge_score import rouge_scorer

            smoother = SmoothingFunction()
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

            bleu_1_scores = []
            bleu_4_scores = []
            rouge_l_scores = []

            for pred, ref in zip(predictions, references):
                # BLEU
                ref_tokens = ref.lower().split()
                pred_tokens = pred.lower().split()

                bleu_1 = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(1.0, 0, 0, 0),
                    smoothing_function=smoother.method1
                )
                bleu_4 = sentence_bleu(
                    [ref_tokens], pred_tokens,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smoother.method1
                )

                bleu_1_scores.append(bleu_1)
                bleu_4_scores.append(bleu_4)

                # ROUGE-L
                rouge = scorer.score(ref, pred)
                rouge_l_scores.append(rouge['rougeL'].fmeasure)

            result.bleu_1 = float(np.mean(bleu_1_scores))
            result.bleu_4 = float(np.mean(bleu_4_scores))
            result.rouge_l = float(np.mean(rouge_l_scores))

        except ImportError:
            logger.warning("nltk and rouge_score required for VQA metrics")

        # BERTScore
        try:
            from bert_score import score as bert_score

            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            result.bert_score = float(F1.mean())

        except ImportError:
            logger.warning("bert_score not available")

        return result

    def evaluate_dataset(
        self,
        dataset: Any,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> EvaluationReport:
        """
        Run full evaluation on a dataset.

        Args:
            dataset: HuggingFace dataset or list of samples
            output_path: Path to save evaluation report
            max_samples: Maximum samples to evaluate

        Returns:
            EvaluationReport with all metrics
        """
        logger.info("Starting surgical VQA evaluation")

        # Load model if needed
        if self.model is None:
            self.load_model()

        # Prepare data
        if hasattr(dataset, '__len__'):
            samples = list(dataset)
        else:
            samples = dataset

        if max_samples:
            samples = samples[:max_samples]

        # Generate predictions
        predictions = []
        ground_truths = []
        phase_preds = []
        phase_gts = []
        step_preds = []
        step_gts = []
        vqa_refs = []

        for sample in tqdm(samples, desc="Generating predictions"):
            # Get image and question
            image = sample.get("image") or sample.get("images", [None])[0]
            question = sample.get("question") or sample["messages"][0]["content"]

            # Generate prediction
            pred = self.generate_prediction(image, question)
            predictions.append(pred)

            # Collect ground truth
            gt = sample.get("ground_truth", {})
            if not gt:
                gt = {
                    "phase": sample.get("phase"),
                    "step": sample.get("step"),
                    "points": sample.get("points", []),
                    "instruments": sample.get("instruments", []),
                }
            ground_truths.append(gt)

            # Phase/step labels
            if gt.get("phase"):
                phase_preds.append(pred)
                phase_gts.append(gt["phase"])
            if gt.get("step"):
                step_preds.append(pred)
                step_gts.append(gt["step"])

            # Reference answer for VQA
            if "answer" in sample:
                vqa_refs.append(sample["answer"])
            elif "messages" in sample and len(sample["messages"]) > 1:
                vqa_refs.append(sample["messages"][1]["content"])

        # Compute metrics
        report = EvaluationReport(
            model_name=self.model_path,
            dataset_name=str(dataset) if hasattr(dataset, '__str__') else "custom",
            num_samples=len(samples)
        )

        # Pointing evaluation
        has_points = any(gt.get("points") for gt in ground_truths)
        if has_points:
            report.pointing = self.evaluate_pointing(predictions, ground_truths)
            logger.info(f"Pointing F1: {report.pointing.f1:.3f}")

        # Phase classification
        if phase_preds:
            report.phase_classification = self.evaluate_classification(
                phase_preds, phase_gts, self.PHASES, "phase"
            )
            logger.info(f"Phase Accuracy: {report.phase_classification.accuracy:.3f}")

        # Step classification
        if step_preds:
            report.step_classification = self.evaluate_classification(
                step_preds, step_gts, self.STEPS, "step"
            )
            logger.info(f"Step Accuracy: {report.step_classification.accuracy:.3f}")

        # VQA quality
        if vqa_refs:
            report.vqa_quality = self.evaluate_vqa_quality(predictions, vqa_refs)
            logger.info(f"BLEU-4: {report.vqa_quality.bleu_4:.3f}")

        # Save report
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            # JSON report
            with open(output_path, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)

            # Markdown report
            md_path = output_path.replace(".json", ".md")
            with open(md_path, 'w') as f:
                f.write(report.to_markdown())

            logger.info(f"Evaluation report saved to {output_path}")

        return report


def run_evaluation(
    model_path: str,
    dataset_path: str,
    output_path: str,
    max_samples: Optional[int] = None
):
    """CLI-friendly evaluation function."""
    # Load dataset
    if dataset_path.startswith("matheus-rech/"):
        dataset = load_dataset(dataset_path, split="test")
    else:
        dataset = load_from_disk(dataset_path)

    # Run evaluation
    evaluator = SurgicalVQAEvaluator(model_path)
    report = evaluator.evaluate_dataset(
        dataset,
        output_path=output_path,
        max_samples=max_samples
    )

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate surgical VQA model")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--dataset", required=True, help="Dataset path or HF ID")
    parser.add_argument("--output", required=True, help="Output report path")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda/mps)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    report = run_evaluation(
        model_path=args.model,
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples
    )

    print("\n" + "="*60)
    print(report.to_markdown())
