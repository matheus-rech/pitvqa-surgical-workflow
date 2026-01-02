"""
PitVQA Training Module

Training pipeline for Visual Question Answering models
on pituitary surgery workflows.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import numpy as np

# Local imports
from data_loader import create_dataloader, SurgicalVQADataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VQAModel(nn.Module):
    """
    Visual Question Answering model for surgical procedures.

    Architecture:
        - Vision Encoder: ResNet or ViT
        - Text Encoder: BERT-based
        - Fusion: Cross-attention
        - Decoder: Classification or generation head
    """

    def __init__(
        self,
        vision_model: str = "resnet50",
        text_model: str = "bert-base-uncased",
        hidden_dim: int = 768,
        num_classes: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Vision encoder
        self.vision_encoder = self._build_vision_encoder(vision_model)

        # Text encoder (placeholder - would use transformers library)
        self.text_encoder = nn.Sequential(
            nn.Embedding(30522, hidden_dim),  # BERT vocab size
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=4
            )
        )

        # Vision projection to match text dimension
        self.vision_proj = nn.Linear(2048, hidden_dim)

        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _build_vision_encoder(self, model_name: str) -> nn.Module:
        """Build vision encoder backbone."""
        try:
            from torchvision import models

            if model_name == "resnet50":
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # Remove final classification layer
                return nn.Sequential(*list(backbone.children())[:-1])
            elif model_name == "resnet101":
                backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
                return nn.Sequential(*list(backbone.children())[:-1])
            else:
                # Default to ResNet50
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                return nn.Sequential(*list(backbone.children())[:-1])
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
            # Fallback to simple CNN
            return nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 2048)
            )

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual features."""
        features = self.vision_encoder(images)
        features = features.flatten(1)
        features = self.vision_proj(features)
        return features.unsqueeze(1)  # [B, 1, D]

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode text features."""
        return self.text_encoder(input_ids)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Image tensor [B, C, H, W]
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]

        Returns:
            Logits [B, num_classes]
        """
        # Encode modalities
        vision_features = self.encode_vision(images)
        text_features = self.encode_text(input_ids)

        # Cross-attention: text attends to vision
        fused_features, _ = self.cross_attention(
            query=text_features,
            key=vision_features,
            value=vision_features
        )

        # Pool and classify
        pooled = fused_features.mean(dim=1)
        logits = self.classifier(pooled)

        return logits


class Trainer:
    """
    Training manager for VQA models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        output_dir: str = "outputs",
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimization
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.scheduler = scheduler

        # Mixed precision
        self.use_amp = use_amp and device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Logging
        self.writer = SummaryWriter(self.output_dir / "logs")

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=True
        )

        for batch_idx, batch in enumerate(progress):
            # Move data to device
            images = batch['images'].to(self.device)

            # Tokenize questions (simplified - would use tokenizer)
            questions = batch['questions']
            input_ids = self._tokenize_batch(questions).to(self.device)

            # Get target labels (simplified - would map answers to classes)
            targets = self._encode_answers(batch['answers']).to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits = self.model(images, input_ids)
                loss = self.criterion(logits, targets)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress.set_postfix({
                'loss': f"{total_loss / num_batches:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to tensorboard
            if self.global_step % 100 == 0:
                self.writer.add_scalar(
                    'train/loss',
                    loss.item() * self.gradient_accumulation_steps,
                    self.global_step
                )

        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)

        return avg_loss

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch['images'].to(self.device)
            input_ids = self._tokenize_batch(batch['questions']).to(self.device)
            targets = self._encode_answers(batch['answers']).to(self.device)

            logits = self.model(images, input_ids)
            loss = self.criterion(logits, targets)

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        self.history['val_loss'].append(avg_loss)
        self.history['val_accuracy'].append(accuracy)

        # Log to tensorboard
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/accuracy', accuracy, self.global_step)

        return avg_loss, accuracy

    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        early_stopping_patience: int = 5
    ):
        """
        Full training loop.

        Args:
            num_epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")

            # Validate
            val_loss, val_acc = self.validate()
            if self.val_loader:
                logger.info(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, val_loss)

            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        self.save_checkpoint(epoch, val_loss, is_final=True)

        # Save training history
        self._save_history()

        logger.info("Training complete!")

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'history': self.history
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Determine filename
        if is_best:
            filename = "best_model.pt"
        elif is_final:
            filename = "final_model.pt"
        else:
            filename = f"checkpoint_epoch_{epoch}.pt"

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.history = checkpoint.get('history', self.history)

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']

    def _tokenize_batch(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """
        Simple tokenization (placeholder for full tokenizer).
        In practice, use transformers AutoTokenizer.
        """
        # Simple character-level tokenization for demonstration
        batch_ids = []
        for text in texts:
            # Convert to character IDs (simplified)
            ids = [ord(c) % 30522 for c in text[:max_length]]
            # Pad to max_length
            ids = ids + [0] * (max_length - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def _encode_answers(self, answers: List[str]) -> torch.Tensor:
        """
        Encode answers to class indices (placeholder).
        In practice, use answer vocabulary mapping.
        """
        # Simple hash-based encoding for demonstration
        indices = [hash(ans) % self.model.num_classes for ans in answers]
        return torch.tensor(indices, dtype=torch.long)

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def create_model(config: Dict) -> VQAModel:
    """Create VQA model from config."""
    return VQAModel(
        vision_model=config.get('vision_model', 'resnet50'),
        text_model=config.get('text_model', 'bert-base-uncased'),
        hidden_dim=config.get('hidden_dim', 768),
        num_classes=config.get('num_classes', 1000),
        dropout=config.get('dropout', 0.1)
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train PitVQA Model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--train_annotations', type=str,
                        default='data/annotations/train.json',
                        help='Path to training annotations')
    parser.add_argument('--val_annotations', type=str,
                        default='data/annotations/val.json',
                        help='Path to validation annotations')

    # Model arguments
    parser.add_argument('--vision_model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'vit'],
                        help='Vision encoder architecture')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='Hidden dimension size')
    parser.add_argument('--num_classes', type=int, default=1000,
                        help='Number of answer classes')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loading workers')

    # System arguments
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    logger.info(f"Configuration: {config}")

    # Create data loaders
    logger.info("Creating data loaders...")

    train_loader = create_dataloader(
        data_dir=args.data_dir,
        annotations_file=args.train_annotations,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        split="train"
    )

    val_loader = None
    if Path(args.val_annotations).exists():
        val_loader = create_dataloader(
            data_dir=args.data_dir,
            annotations_file=args.val_annotations,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            split="val"
        )

    # Create model
    logger.info("Creating model...")
    model = create_model({
        'vision_model': args.vision_model,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes
    })

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=str(output_dir),
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        num_epochs=args.num_epochs,
        save_every=5,
        early_stopping_patience=10
    )


if __name__ == "__main__":
    main()
