"""
PitVQA Evaluation Module

Comprehensive evaluation tools for surgical Visual Question Answering models.
Includes metrics calculation, inference utilities, and result visualization.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

# Local imports
from data_loader import SurgicalVQADataset, create_dataloader, collate_vqa_batch


class VQAEvaluator:
    """
    Evaluator for Surgical VQA models.

    Handles inference, metrics calculation, and result analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        answer_vocab: Optional[Dict[str, int]] = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained VQA model
            device: Device for inference
            answer_vocab: Mapping from answers to indices
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.answer_vocab = answer_vocab or {}
        self.idx_to_answer = {v: k for k, v in self.answer_vocab.items()}

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        questions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for a batch.

        Args:
            images: Batch of image tensors
            questions: List of question strings

        Returns:
            Tuple of (predictions, probabilities)
        """
        images = images.to(self.device)

        # Forward pass
        logits = self.model(images, questions)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on entire dataset.

        Args:
            dataloader: DataLoader for evaluation data
            return_predictions: Whether to return individual predictions

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_questions = []
        all_answers = []
        all_question_types = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images']
            questions = batch['questions']
            answers = batch['answers']
            question_types = batch.get('question_types', ['unknown'] * len(questions))

            # Convert answers to indices
            labels = self._answers_to_indices(answers)

            # Get predictions
            predictions, probs = self.predict(images, questions)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
            all_questions.extend(questions)
            all_answers.extend(answers)
            all_question_types.extend(question_types)
            all_probs.append(probs.cpu().numpy())

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.concatenate(all_probs, axis=0)

        # Calculate metrics
        metrics = self._calculate_metrics(
            all_predictions,
            all_labels,
            all_question_types
        )

        # Add additional info
        metrics['num_samples'] = len(all_predictions)

        if return_predictions:
            metrics['predictions'] = {
                'predicted_indices': all_predictions.tolist(),
                'true_indices': all_labels.tolist(),
                'predicted_answers': [self.idx_to_answer.get(p, 'unknown') for p in all_predictions],
                'true_answers': all_answers,
                'questions': all_questions,
                'question_types': all_question_types,
                'probabilities': all_probs.tolist()
            }

        return metrics

    def _answers_to_indices(self, answers: List[str]) -> List[int]:
        """Convert answer strings to indices."""
        indices = []
        for answer in answers:
            if answer in self.answer_vocab:
                indices.append(self.answer_vocab[answer])
            else:
                # Handle unknown answers
                indices.append(-1)
        return indices

    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        question_types: List[str]
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        # Filter out invalid labels
        valid_mask = labels >= 0
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        question_types = [qt for qt, v in zip(question_types, valid_mask) if v]

        metrics = {}

        # Overall accuracy
        metrics['accuracy'] = float(accuracy_score(labels, predictions))

        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)

        metrics['per_class'] = {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'support': support_per_class.tolist()
        }

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()

        # Per question type metrics
        metrics['per_question_type'] = self._calculate_per_type_metrics(
            predictions, labels, question_types
        )

        return metrics

    def _calculate_per_type_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        question_types: List[str]
    ) -> Dict:
        """Calculate metrics for each question type."""
        type_metrics = {}

        unique_types = set(question_types)

        for qtype in unique_types:
            mask = np.array([qt == qtype for qt in question_types])

            if mask.sum() > 0:
                type_preds = predictions[mask]
                type_labels = labels[mask]

                accuracy = float(accuracy_score(type_labels, type_preds))
                precision, recall, f1, _ = precision_recall_fscore_support(
                    type_labels, type_preds, average='weighted', zero_division=0
                )

                type_metrics[qtype] = {
                    'accuracy': accuracy,
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'count': int(mask.sum())
                }

        return type_metrics


class InferenceEngine:
    """
    Inference engine for real-time predictions.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to saved model checkpoint
            config_path: Path to model configuration
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load answer vocabulary
        self.answer_vocab = self.config.get('answer_vocab', {})
        self.idx_to_answer = {v: k for k, v in self.answer_vocab.items()}

        # Set up image transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(self.config.get('image_size', (224, 224))),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _load_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        from train import VQAModel

        # Initialize model architecture
        model = VQAModel(
            num_classes=self.config.get('num_classes', 100),
            vision_encoder=self.config.get('vision_encoder', 'resnet50'),
            hidden_dim=self.config.get('hidden_dim', 512),
            num_attention_heads=self.config.get('num_attention_heads', 8),
            num_transformer_layers=self.config.get('num_transformer_layers', 2),
            dropout=self.config.get('dropout', 0.1)
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    @torch.no_grad()
    def predict_single(
        self,
        image,
        question: str,
        top_k: int = 5
    ) -> Dict:
        """
        Predict answer for a single image-question pair.

        Args:
            image: PIL Image or path to image
            question: Question string
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results
        """
        from PIL import Image as PILImage

        # Load image if path
        if isinstance(image, str):
            image = PILImage.open(image).convert('RGB')

        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get prediction
        logits = self.model(image_tensor, [question])
        probs = F.softmax(logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], min(top_k, len(probs[0])))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            answer = self.idx_to_answer.get(int(idx), f'class_{idx}')
            predictions.append({
                'answer': answer,
                'probability': float(prob),
                'class_index': int(idx)
            })

        return {
            'question': question,
            'predictions': predictions,
            'top_answer': predictions[0]['answer'] if predictions else 'unknown',
            'confidence': predictions[0]['probability'] if predictions else 0.0
        }

    @torch.no_grad()
    def predict_batch(
        self,
        images: List,
        questions: List[str]
    ) -> List[Dict]:
        """
        Predict answers for a batch of image-question pairs.

        Args:
            images: List of PIL Images or paths
            questions: List of questions

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image, question in zip(images, questions):
            result = self.predict_single(image, question)
            results.append(result)
        return results


class ResultsAnalyzer:
    """
    Analyze and visualize evaluation results.
    """

    def __init__(self, results: Dict, output_dir: str = "evaluation_results"):
        """
        Initialize results analyzer.

        Args:
            results: Evaluation results dictionary
            output_dir: Directory to save analysis outputs
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self) -> str:
        """Generate a text report of evaluation results."""
        lines = [
            "=" * 60,
            "SURGICAL VQA EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total Samples: {self.results.get('num_samples', 'N/A')}",
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Accuracy:  {self.results.get('accuracy', 0):.4f}",
            f"Precision: {self.results.get('precision', 0):.4f}",
            f"Recall:    {self.results.get('recall', 0):.4f}",
            f"F1 Score:  {self.results.get('f1_score', 0):.4f}",
            "",
        ]

        # Per question type metrics
        if 'per_question_type' in self.results:
            lines.extend([
                "METRICS BY QUESTION TYPE",
                "-" * 40
            ])

            for qtype, metrics in self.results['per_question_type'].items():
                lines.extend([
                    f"\n{qtype} (n={metrics['count']}):",
                    f"  Accuracy:  {metrics['accuracy']:.4f}",
                    f"  Precision: {metrics['precision']:.4f}",
                    f"  Recall:    {metrics['recall']:.4f}",
                    f"  F1 Score:  {metrics['f1_score']:.4f}"
                ])

        lines.extend(["", "=" * 60])

        report = "\n".join(lines)

        # Save report
        report_path = self.output_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        return report

    def save_results(self, filename: str = "evaluation_results.json"):
        """Save results to JSON file."""
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        results_json = self._convert_for_json(self.results)

        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"Results saved to {output_path}")

    def _convert_for_json(self, obj):
        """Recursively convert numpy arrays to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    def plot_confusion_matrix(
        self,
        class_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """Plot and save confusion matrix."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("matplotlib and seaborn required for plotting")
            return

        cm = np.array(self.results.get('confusion_matrix', []))

        if cm.size == 0:
            print("No confusion matrix data available")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=class_names or range(cm.shape[1]),
            yticklabels=class_names or range(cm.shape[0]),
            ax=ax
        )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix (Normalized)')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150)
        plt.close()

        print(f"Confusion matrix saved to {self.output_dir / 'confusion_matrix.png'}")

    def plot_per_type_accuracy(self, figsize: Tuple[int, int] = (12, 6)):
        """Plot accuracy by question type."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib required for plotting")
            return

        per_type = self.results.get('per_question_type', {})

        if not per_type:
            print("No per-type metrics available")
            return

        types = list(per_type.keys())
        accuracies = [per_type[t]['accuracy'] for t in types]
        counts = [per_type[t]['count'] for t in types]

        fig, ax1 = plt.subplots(figsize=figsize)

        # Bar chart for accuracy
        x = range(len(types))
        bars = ax1.bar(x, accuracies, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Question Type')
        ax1.set_ylabel('Accuracy', color='steelblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(types, rotation=45, ha='right')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim(0, 1)

        # Line for sample counts
        ax2 = ax1.twinx()
        ax2.plot(x, counts, 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Sample Count', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.annotate(
                f'{acc:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9
            )

        plt.title('Accuracy by Question Type')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'accuracy_by_type.png', dpi=150)
        plt.close()

        print(f"Accuracy plot saved to {self.output_dir / 'accuracy_by_type.png'}")


def evaluate_model(
    model_path: str,
    config_path: str,
    data_dir: str,
    annotations_file: str,
    output_dir: str = "evaluation_results",
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[str] = None
) -> Dict:
    """
    Main evaluation function.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to model config
        data_dir: Path to test data
        annotations_file: Path to test annotations
        output_dir: Directory for saving results
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        device: Device for evaluation

    Returns:
        Evaluation results dictionary
    """
    # Setup device
    device = torch.device(
        device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Using device: {device}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize inference engine
    engine = InferenceEngine(model_path, config_path, str(device))

    # Create test dataloader
    test_loader = create_dataloader(
        data_dir=data_dir,
        annotations_file=annotations_file,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        split="test"
    )

    # Initialize evaluator
    evaluator = VQAEvaluator(
        model=engine.model,
        device=device,
        answer_vocab=config.get('answer_vocab', {})
    )

    # Run evaluation
    print("Running evaluation...")
    results = evaluator.evaluate(test_loader, return_predictions=True)

    # Analyze results
    analyzer = ResultsAnalyzer(results, output_dir)

    # Generate outputs
    report = analyzer.generate_report()
    print("\n" + report)

    analyzer.save_results()

    # Generate plots if matplotlib available
    try:
        analyzer.plot_confusion_matrix()
        analyzer.plot_per_type_accuracy()
    except Exception as e:
        print(f"Could not generate plots: {e}")

    return results


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Surgical VQA Model"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to model configuration JSON'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to test data directory'
    )
    parser.add_argument(
        '--annotations_file',
        type=str,
        required=True,
        help='Path to test annotations'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for evaluation (cuda/cpu)'
    )

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        config_path=args.config_path,
        data_dir=args.data_dir,
        annotations_file=args.annotations_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")


if __name__ == "__main__":
    main()
