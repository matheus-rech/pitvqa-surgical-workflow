#!/usr/bin/env python3
"""
Comprehensive Evaluation for Unified VLM Model
Publication-quality evaluation for MICCAI 2026 paper.

Evaluates all four task types:
1. Phase Classification
2. Step Classification
3. Instrument Pointing
4. Anatomy Pointing

Outputs:
- evaluation_results.json - Detailed metrics
- evaluation_report.md - Publication-ready markdown
- confusion_matrices/ - Per-task confusion matrices
- latex_tables.tex - Tables for paper
"""

import json
import re
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
)
from tqdm import tqdm

# Configuration
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
UNIFIED_DATASET = "mmrech/pitvqa-unified-vlm"
OUTPUT_DIR = "evaluation_results"

# Task types
TASK_PHASE = "phase_classification"
TASK_STEP = "step_classification"
TASK_INSTRUMENT = "instrument_pointing"
TASK_ANATOMY = "anatomy_pointing"

# Phase labels
PHASES = ["nasal", "sellar", "tumor_removal", "closure"]

# Step labels
STEPS = [
    "septal_dissection", "turbinectomy", "sphenoidotomy",
    "posterior_septectomy", "sellar_floor_removal", "dura_opening",
    "tumor_resection", "hemostasis", "reconstruction", "visualization"
]


def extract_coordinates(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract coordinates from point format."""
    match = re.search(r"<point x='([\d.]+)' y='([\d.]+)'>", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def has_point_format(text: str) -> bool:
    """Check if text contains point format."""
    return bool(re.search(r"<point x='[\d.]+' y='[\d.]+'>", text))


def euclidean_distance(pred: Tuple[float, float], gt: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return ((pred[0] - gt[0])**2 + (pred[1] - gt[1])**2)**0.5


def get_quadrant(x: float, y: float) -> str:
    """Get quadrant from coordinates (0-100 scale)."""
    if x < 50 and y < 50:
        return "TL"
    elif x >= 50 and y < 50:
        return "TR"
    elif x < 50 and y >= 50:
        return "BL"
    else:
        return "BR"


def extract_classification_label(text: str, labels: List[str]) -> str:
    """Extract classification label from text."""
    text_lower = text.lower().strip()

    for label in labels:
        label_variants = [
            label.lower(),
            label.replace("_", " ").lower(),
            label.replace("_", "").lower(),
        ]
        for variant in label_variants:
            if variant in text_lower:
                return label

    return "unknown"


class UnifiedVLMEvaluator:
    """Evaluator for unified multi-task VLM."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model and processor."""
        print(f"Loading model: {self.model_path}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)

        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.training = False

    def generate_prediction(self, image, question: str) -> str:
        """Generate model prediction."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        input_len = inputs['input_ids'].shape[1]
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def evaluate_classification(
        self,
        predictions: List[str],
        ground_truths: List[str],
        labels: List[str],
        task_name: str
    ) -> Dict:
        """Evaluate classification task."""
        pred_labels = [extract_classification_label(p, labels) for p in predictions]
        gt_labels = [extract_classification_label(g, labels) for g in ground_truths]

        # Filter valid samples
        valid_pairs = [
            (p, g) for p, g in zip(pred_labels, gt_labels)
            if g != "unknown"
        ]

        if not valid_pairs:
            return {"accuracy": 0, "f1_macro": 0, "f1_weighted": 0, "n_samples": 0}

        y_pred, y_true = zip(*valid_pairs)

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels + ["unknown"])

        per_class = {}
        for i, label in enumerate(labels):
            per_class[label] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]) if i < len(support) else 0,
            }

        return {
            "task": task_name,
            "n_samples": len(valid_pairs),
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "labels": labels + ["unknown"],
        }

    def evaluate_pointing(
        self,
        predictions: List[str],
        ground_truths: List[str],
        task_name: str
    ) -> Dict:
        """Evaluate pointing/localization task."""
        distances = []
        correct_10 = 0
        correct_15 = 0
        quadrant_matches = 0
        format_correct = 0
        total = 0

        for pred, gt in zip(predictions, ground_truths):
            pred_x, pred_y = extract_coordinates(pred)
            gt_x, gt_y = extract_coordinates(gt)

            if gt_x is None:
                continue

            total += 1

            if pred_x is not None:
                format_correct += 1
                dist = euclidean_distance((pred_x, pred_y), (gt_x, gt_y))
                distances.append(dist)

                if dist <= 10.0:
                    correct_10 += 1
                if dist <= 15.0:
                    correct_15 += 1

                if get_quadrant(pred_x, pred_y) == get_quadrant(gt_x, gt_y):
                    quadrant_matches += 1

        if total == 0:
            return {"accuracy_10": 0, "accuracy_15": 0, "n_samples": 0}

        return {
            "task": task_name,
            "n_samples": total,
            "format_rate": format_correct / total,
            "accuracy_10": correct_10 / total,
            "accuracy_15": correct_15 / total,
            "quadrant_accuracy": quadrant_matches / format_correct if format_correct > 0 else 0,
            "mean_distance": float(np.mean(distances)) if distances else 0,
            "median_distance": float(np.median(distances)) if distances else 0,
            "std_distance": float(np.std(distances)) if distances else 0,
        }

    def evaluate_dataset(
        self,
        dataset_name: str = UNIFIED_DATASET,
        split: str = "validation",
        max_samples: Optional[int] = None
    ) -> Dict:
        """Run full evaluation on dataset."""
        if self.model is None:
            self.load_model()

        print(f"\nLoading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Evaluating {len(dataset)} samples...")

        # Organize by task type
        samples_by_task = defaultdict(list)
        predictions_by_task = defaultdict(list)
        ground_truths_by_task = defaultdict(list)

        for sample in tqdm(dataset, desc="Generating predictions"):
            task_type = sample.get('task_type', 'unknown')
            image = sample['image']
            messages = json.loads(sample['messages']) if isinstance(sample['messages'], str) else sample['messages']

            question = messages[0]['content']
            gt_response = messages[1]['content']

            # Generate prediction
            pred = self.generate_prediction(image, question)

            samples_by_task[task_type].append(sample)
            predictions_by_task[task_type].append(pred)
            ground_truths_by_task[task_type].append(gt_response)

        # Evaluate each task type
        results = {
            "model": self.model_path,
            "dataset": dataset_name,
            "split": split,
            "total_samples": len(dataset),
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
        }

        # Phase classification
        if TASK_PHASE in predictions_by_task:
            results["tasks"][TASK_PHASE] = self.evaluate_classification(
                predictions_by_task[TASK_PHASE],
                ground_truths_by_task[TASK_PHASE],
                PHASES,
                TASK_PHASE
            )

        # Step classification
        if TASK_STEP in predictions_by_task:
            results["tasks"][TASK_STEP] = self.evaluate_classification(
                predictions_by_task[TASK_STEP],
                ground_truths_by_task[TASK_STEP],
                STEPS,
                TASK_STEP
            )

        # Instrument pointing
        if TASK_INSTRUMENT in predictions_by_task:
            results["tasks"][TASK_INSTRUMENT] = self.evaluate_pointing(
                predictions_by_task[TASK_INSTRUMENT],
                ground_truths_by_task[TASK_INSTRUMENT],
                TASK_INSTRUMENT
            )

        # Anatomy pointing
        if TASK_ANATOMY in predictions_by_task:
            results["tasks"][TASK_ANATOMY] = self.evaluate_pointing(
                predictions_by_task[TASK_ANATOMY],
                ground_truths_by_task[TASK_ANATOMY],
                TASK_ANATOMY
            )

        return results


def generate_latex_tables(results: Dict) -> str:
    """Generate LaTeX tables for paper."""
    lines = [
        "% Auto-generated evaluation tables",
        f"% Model: {results['model']}",
        f"% Generated: {results['timestamp']}",
        "",
    ]

    # Overall metrics table
    lines.extend([
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Multi-Task Model Evaluation Results}",
        "\\label{tab:evaluation}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Task & N & Accuracy & F1 (Macro) & F1 (Weighted) \\\\",
        "\\midrule",
    ])

    for task_name, task_results in results["tasks"].items():
        n = task_results.get("n_samples", 0)
        task_display = task_name.replace("_", " ").title()

        if "accuracy" in task_results:
            acc = task_results["accuracy"] * 100
            f1m = task_results.get("f1_macro", 0) * 100
            f1w = task_results.get("f1_weighted", 0) * 100
            lines.append(f"{task_display} & {n} & {acc:.1f}\\% & {f1m:.1f}\\% & {f1w:.1f}\\% \\\\")
        elif "accuracy_10" in task_results:
            acc10 = task_results["accuracy_10"] * 100
            acc15 = task_results["accuracy_15"] * 100
            quad = task_results.get("quadrant_accuracy", 0) * 100
            lines.append(f"{task_display} & {n} & {acc10:.1f}\\%@10 & {acc15:.1f}\\%@15 & {quad:.1f}\\% quad \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    # Pointing metrics table
    pointing_tasks = [TASK_INSTRUMENT, TASK_ANATOMY]
    has_pointing = any(t in results["tasks"] for t in pointing_tasks)

    if has_pointing:
        lines.extend([
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Spatial Localization Results}",
            "\\label{tab:pointing}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Task & Format\\% & Acc@10\\% & Acc@15\\% & Quadrant & Mean Dist \\\\",
            "\\midrule",
        ])

        for task in pointing_tasks:
            if task in results["tasks"]:
                r = results["tasks"][task]
                name = task.replace("_pointing", "").title()
                fmt = r.get("format_rate", 0) * 100
                a10 = r.get("accuracy_10", 0) * 100
                a15 = r.get("accuracy_15", 0) * 100
                quad = r.get("quadrant_accuracy", 0) * 100
                dist = r.get("mean_distance", 0)
                lines.append(f"{name} & {fmt:.1f} & {a10:.1f} & {a15:.1f} & {quad:.1f} & {dist:.2f} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])

    return "\n".join(lines)


def generate_markdown_report(results: Dict) -> str:
    """Generate markdown report."""
    lines = [
        "# Unified VLM Evaluation Report",
        "",
        f"**Model:** {results['model']}",
        f"**Dataset:** {results['dataset']}",
        f"**Split:** {results['split']}",
        f"**Total Samples:** {results['total_samples']}",
        f"**Timestamp:** {results['timestamp']}",
        "",
        "## Summary",
        "",
        "| Task | N | Primary Metric | Secondary Metric |",
        "|------|---|----------------|------------------|",
    ]

    for task_name, task_results in results["tasks"].items():
        n = task_results.get("n_samples", 0)
        task_display = task_name.replace("_", " ").title()

        if "accuracy" in task_results:
            primary = f"Accuracy: {task_results['accuracy']*100:.1f}%"
            secondary = f"F1: {task_results.get('f1_weighted', 0)*100:.1f}%"
        else:
            primary = f"Acc@15%: {task_results.get('accuracy_15', 0)*100:.1f}%"
            secondary = f"Mean Dist: {task_results.get('mean_distance', 0):.2f}"

        lines.append(f"| {task_display} | {n} | {primary} | {secondary} |")

    lines.append("")

    # Detailed results per task
    for task_name, task_results in results["tasks"].items():
        task_display = task_name.replace("_", " ").title()
        lines.extend([
            f"## {task_display}",
            "",
        ])

        if "per_class" in task_results:
            lines.extend([
                "### Per-Class Performance",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|-------|-----------|--------|-------|---------|",
            ])
            for cls, metrics in task_results["per_class"].items():
                lines.append(
                    f"| {cls} | {metrics['precision']*100:.1f}% | "
                    f"{metrics['recall']*100:.1f}% | {metrics['f1']*100:.1f}% | "
                    f"{metrics['support']} |"
                )
            lines.append("")

        if "accuracy_10" in task_results:
            lines.extend([
                "### Localization Metrics",
                "",
                f"- **Format Rate:** {task_results['format_rate']*100:.1f}%",
                f"- **Accuracy @10%:** {task_results['accuracy_10']*100:.1f}%",
                f"- **Accuracy @15%:** {task_results['accuracy_15']*100:.1f}%",
                f"- **Quadrant Accuracy:** {task_results['quadrant_accuracy']*100:.1f}%",
                f"- **Mean Distance:** {task_results['mean_distance']:.2f}",
                f"- **Median Distance:** {task_results['median_distance']:.2f}",
                f"- **Std Distance:** {task_results['std_distance']:.2f}",
                "",
            ])

    return "\n".join(lines)


def save_confusion_matrices(results: Dict, output_dir: Path):
    """Save confusion matrices as images."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for confusion matrices")
        return

    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    for task_name, task_results in results["tasks"].items():
        if "confusion_matrix" not in task_results:
            continue

        cm = np.array(task_results["confusion_matrix"])
        labels = task_results.get("labels", [])

        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix: {task_name.replace("_", " ").title()}')

        plt.tight_layout()
        plt.savefig(cm_dir / f"{task_name}_confusion.png", dpi=150)
        plt.close()

        print(f"  Saved: {cm_dir / f'{task_name}_confusion.png'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Unified VLM Model")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--dataset", default=UNIFIED_DATASET, help="Dataset to evaluate on")
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Max samples to evaluate")

    args = parser.parse_args()

    print("=" * 70)
    print("UNIFIED VLM EVALUATION")
    print("=" * 70)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run evaluation
    evaluator = UnifiedVLMEvaluator(args.model)
    results = evaluator.evaluate_dataset(
        dataset_name=args.dataset,
        split=args.split,
        max_samples=args.max_samples
    )

    # Save results
    print("\nSaving results...")

    # JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Markdown
    md_path = output_dir / "evaluation_report.md"
    with open(md_path, 'w') as f:
        f.write(generate_markdown_report(results))
    print(f"  Saved: {md_path}")

    # LaTeX
    tex_path = output_dir / "latex_tables.tex"
    with open(tex_path, 'w') as f:
        f.write(generate_latex_tables(results))
    print(f"  Saved: {tex_path}")

    # Confusion matrices
    save_confusion_matrices(results, output_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    for task_name, task_results in results["tasks"].items():
        task_display = task_name.replace("_", " ").title()
        n = task_results.get("n_samples", 0)

        if "accuracy" in task_results:
            acc = task_results["accuracy"] * 100
            print(f"{task_display}: {acc:.1f}% accuracy (n={n})")
        else:
            acc = task_results.get("accuracy_15", 0) * 100
            print(f"{task_display}: {acc:.1f}% @15% (n={n})")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
