"""
PitVQA Vision Encoder Module

Extracts visual features from surgical video frames using pretrained vision models.
Designed for efficient feature extraction in Visual Question Answering pipelines.

Features:
- Multiple backbone architectures (ResNet50, ViT, CLIP, DINOv2)
- Temporal encoding for video sequences
- GPU acceleration with automatic device handling
- Batch processing and feature caching
- Attention map visualization for interpretability

Supported Models:
- resnet50: Fast CNN baseline from torchvision
- vit-b-16: Vision Transformer from torchvision
- clip-vit-l-14: CLIP ViT-L/14 from OpenAI (best for VQA)
- clip-vit-b-32: CLIP ViT-B/32 from OpenAI (faster)
- dinov2-base: DINOv2 ViT-B/14 from Meta (surgical domain)
- dinov2-large: DINOv2 ViT-L/14 from Meta

Usage:
    # Single image encoding
    encoder = VisionEncoder(model_name="clip-vit-l-14", output_dim=512)
    features = encoder.encode(images)

    # Video sequence encoding with temporal modeling
    temporal = TemporalEncoder(input_dim=512, hidden_dim=256)
    video_features = temporal(frame_features)

    # CLI for testing
    python -m agent2_skill_extraction.vision_encoder \
        --image test.png --model clip-vit-l-14 --output features.npy
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Configure module logger
logger = logging.getLogger(__name__)

# Model registry with configurations
MODEL_CONFIGS: Dict[str, Dict] = {
    "resnet50": {
        "type": "torchvision",
        "module": "torchvision.models",
        "class": "resnet50",
        "weights": "ResNet50_Weights.IMAGENET1K_V2",
        "feature_dim": 2048,
        "input_size": 224,
        "description": "Fast CNN baseline, good for general features"
    },
    "vit-b-16": {
        "type": "torchvision",
        "module": "torchvision.models",
        "class": "vit_b_16",
        "weights": "ViT_B_16_Weights.IMAGENET1K_V1",
        "feature_dim": 768,
        "input_size": 224,
        "description": "Vision Transformer baseline from torchvision"
    },
    "clip-vit-l-14": {
        "type": "transformers",
        "model_id": "openai/clip-vit-large-patch14",
        "feature_dim": 768,
        "input_size": 224,
        "description": "CLIP ViT-L/14, best for VQA tasks (multimodal)"
    },
    "clip-vit-b-32": {
        "type": "transformers",
        "model_id": "openai/clip-vit-base-patch32",
        "feature_dim": 512,
        "input_size": 224,
        "description": "CLIP ViT-B/32, faster variant"
    },
    "dinov2-base": {
        "type": "transformers",
        "model_id": "facebook/dinov2-base",
        "feature_dim": 768,
        "input_size": 224,
        "description": "DINOv2 ViT-B/14, self-supervised (good for surgical)"
    },
    "dinov2-large": {
        "type": "transformers",
        "model_id": "facebook/dinov2-large",
        "feature_dim": 1024,
        "input_size": 224,
        "description": "DINOv2 ViT-L/14, larger variant"
    },
}


@dataclass
class EncodingStats:
    """Statistics from encoding operations."""

    num_images: int = 0
    feature_dim: int = 0
    encoding_time_seconds: float = 0.0
    images_per_second: float = 0.0
    device_used: str = "cpu"
    model_name: str = ""
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> Dict:
        """Convert stats to dictionary."""
        return {
            "num_images": self.num_images,
            "feature_dim": self.feature_dim,
            "encoding_time_seconds": round(self.encoding_time_seconds, 3),
            "images_per_second": round(self.images_per_second, 2),
            "device_used": self.device_used,
            "model_name": self.model_name,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }


class FeatureCache:
    """
    Simple LRU cache for image features.

    Uses image content hash as key to avoid recomputation.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize feature cache.

        Args:
            max_size: Maximum number of features to cache.
        """
        self.max_size = max_size
        self._cache: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _compute_hash(image: Union[torch.Tensor, np.ndarray, Image.Image]) -> str:
        """Compute hash of image content."""
        if isinstance(image, Image.Image):
            # Convert PIL to numpy
            arr = np.array(image)
        elif isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()
        else:
            arr = image

        # Use MD5 hash of image bytes
        return hashlib.md5(arr.tobytes()).hexdigest()

    def get(self, image: Union[torch.Tensor, np.ndarray, Image.Image]) -> Optional[torch.Tensor]:
        """
        Get cached features for an image.

        Args:
            image: Input image.

        Returns:
            Cached features if found, None otherwise.
        """
        key = self._compute_hash(image)

        if key in self._cache:
            self.hits += 1
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]

        self.misses += 1
        return None

    def put(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        features: torch.Tensor
    ) -> None:
        """
        Cache features for an image.

        Args:
            image: Input image.
            features: Computed features.
        """
        key = self._compute_hash(image)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = features.clone()
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached features."""
        self._cache.clear()
        self._access_order.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)


class VisionEncoder(nn.Module):
    """
    Vision encoder for surgical image feature extraction.

    Supports multiple backbone architectures and provides unified interface
    for feature extraction from images and video frames.

    Attributes:
        model_name: Name of the backbone model.
        output_dim: Dimension of output features.
        device: Device for computation (auto-detected if not specified).

    Example:
        >>> encoder = VisionEncoder("clip-vit-l-14", output_dim=512)
        >>> images = torch.randn(4, 3, 224, 224)
        >>> features = encoder.encode(images)
        >>> print(features.shape)  # (4, 512)

        >>> # From PIL images
        >>> pil_images = [Image.open(f) for f in image_files]
        >>> features = encoder.encode(pil_images)
    """

    def __init__(
        self,
        model_name: str = "clip-vit-l-14",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
        device: Optional[str] = None,
        cache_features: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize vision encoder.

        Args:
            model_name: Name of the backbone model. Options:
                - "resnet50": Fast CNN baseline
                - "vit-b-16": Vision Transformer
                - "clip-vit-l-14": CLIP ViT-L/14 (best for VQA)
                - "clip-vit-b-32": CLIP ViT-B/32
                - "dinov2-base": DINOv2 ViT-B/14
                - "dinov2-large": DINOv2 ViT-L/14
            pretrained: Whether to use pretrained weights.
            freeze_backbone: Whether to freeze backbone parameters.
            output_dim: Dimension of output features (projection applied).
            device: Device for computation. Auto-detected if None.
            cache_features: Whether to cache computed features.
            cache_size: Maximum number of features to cache.

        Raises:
            ValueError: If model_name is not supported.
        """
        super().__init__()

        if model_name not in MODEL_CONFIGS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Available models: {list(MODEL_CONFIGS.keys())}"
            )

        self.model_name = model_name
        self.config = MODEL_CONFIGS[model_name]
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize feature cache
        self.cache_features = cache_features
        self._cache = FeatureCache(max_size=cache_size) if cache_features else None

        # Load backbone model
        self._load_backbone()

        # Create projection layer if needed
        feature_dim = self.config["feature_dim"]
        if feature_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
        else:
            self.projection = nn.Identity()

        # Move to device
        self.to(self.device)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

        # Store attention maps for visualization
        self._attention_maps: Optional[torch.Tensor] = None
        self._register_attention_hooks()

        logger.info(
            f"VisionEncoder initialized: model={model_name}, "
            f"output_dim={output_dim}, device={self.device}, "
            f"frozen={freeze_backbone}"
        )

    def _load_backbone(self) -> None:
        """Load the backbone model based on type."""
        model_type = self.config["type"]

        if model_type == "torchvision":
            self._load_torchvision_backbone()
        elif model_type == "transformers":
            self._load_transformers_backbone()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_torchvision_backbone(self) -> None:
        """Load backbone from torchvision."""
        import torchvision.models as models
        from torchvision.models import get_weight

        model_class = getattr(models, self.config["class"])

        if self.pretrained:
            # Load with pretrained weights
            weights = get_weight(self.config["weights"])
            self.backbone = model_class(weights=weights)
            self.transforms = weights.transforms()
        else:
            self.backbone = model_class(weights=None)
            # Default transforms
            from torchvision import transforms
            self.transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(self.config["input_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        # Remove classification head for feature extraction
        if "resnet" in self.model_name:
            # For ResNet, remove fc layer
            self.backbone.fc = nn.Identity()
        elif "vit" in self.model_name:
            # For ViT, remove heads
            self.backbone.heads = nn.Identity()

        self._backbone_type = "torchvision"

    def _load_transformers_backbone(self) -> None:
        """Load backbone from HuggingFace transformers."""
        model_id = self.config["model_id"]

        if "clip" in self.model_name:
            from transformers import CLIPVisionModel, CLIPProcessor

            self.backbone = CLIPVisionModel.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self._backbone_type = "clip"

        elif "dinov2" in self.model_name:
            from transformers import Dinov2Model, AutoImageProcessor

            self.backbone = Dinov2Model.from_pretrained(
                model_id,
                torch_dtype=torch.float32
            )
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self._backbone_type = "dinov2"
        else:
            raise ValueError(f"Unknown transformers model: {model_id}")

    def _freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        logger.info(f"Froze {sum(1 for _ in self.backbone.parameters())} backbone parameters")

    def _register_attention_hooks(self) -> None:
        """Register hooks to capture attention maps."""
        self._attention_hooks = []

        if self._backbone_type in ["clip", "dinov2"]:
            # For transformer models, hook into attention layers
            def attention_hook(module, input, output):
                if hasattr(output, "attentions") and output.attentions is not None:
                    self._attention_maps = output.attentions

            # Register on the backbone
            if hasattr(self.backbone, "vision_model"):
                # CLIP structure
                handle = self.backbone.vision_model.register_forward_hook(attention_hook)
            else:
                handle = self.backbone.register_forward_hook(attention_hook)

            self._attention_hooks.append(handle)

    def preprocess(
        self,
        images: Union[torch.Tensor, np.ndarray, Image.Image, List]
    ) -> torch.Tensor:
        """
        Preprocess images for the model.

        Args:
            images: Input images. Can be:
                - torch.Tensor (B, C, H, W) or (C, H, W)
                - np.ndarray (B, H, W, C) or (H, W, C)
                - PIL.Image
                - List of any of the above

        Returns:
            Preprocessed tensor ready for encoding.
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = [Image.fromarray(images)]
        elif isinstance(images, torch.Tensor) and len(images.shape) == 3:
            images = [images]

        # Convert numpy arrays to PIL
        if isinstance(images, list):
            processed = []
            for img in images:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                processed.append(img)
            images = processed

        # Apply preprocessing
        if self._backbone_type == "torchvision":
            if isinstance(images, list):
                # Apply transforms to each image
                tensors = [self.transforms(img) for img in images]
                return torch.stack(tensors)
            else:
                return images  # Already a tensor

        elif self._backbone_type in ["clip", "dinov2"]:
            # Use HuggingFace processor
            inputs = self.processor(
                images=images,
                return_tensors="pt"
            )
            return inputs["pixel_values"]

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            images: Preprocessed image tensor (B, C, H, W).

        Returns:
            Feature tensor (B, output_dim).
        """
        images = images.to(self.device)

        if self._backbone_type == "torchvision":
            features = self.backbone(images)

        elif self._backbone_type == "clip":
            outputs = self.backbone(
                pixel_values=images,
                output_attentions=True
            )
            # Use pooled output
            features = outputs.pooler_output

        elif self._backbone_type == "dinov2":
            outputs = self.backbone(
                pixel_values=images,
                output_attentions=True
            )
            # Use CLS token
            features = outputs.last_hidden_state[:, 0]

        # Apply projection
        features = self.projection(features)

        return features

    @torch.no_grad()
    def encode(
        self,
        images: Union[torch.Tensor, np.ndarray, Image.Image, List],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Encode images to feature vectors.

        This is the main method for feature extraction. Handles preprocessing,
        batching, and optional caching.

        Args:
            images: Input images (single or batch).
            batch_size: Batch size for processing.
            show_progress: Show progress bar for large batches.
            use_cache: Whether to use feature cache.

        Returns:
            Feature tensor (N, output_dim) where N is number of images.

        Example:
            >>> encoder = VisionEncoder("clip-vit-l-14")
            >>> features = encoder.encode([img1, img2, img3])
            >>> print(features.shape)  # (3, 512)
        """
        self.train(False)

        # Convert to list for uniform handling
        if isinstance(images, (Image.Image, np.ndarray)):
            images = [images]
        elif isinstance(images, torch.Tensor):
            if len(images.shape) == 3:
                images = [images]
            else:
                # Already batched tensor
                images = [images[i] for i in range(images.shape[0])]

        all_features = []
        uncached_indices = []
        uncached_images = []

        # Check cache first
        if use_cache and self._cache is not None:
            for i, img in enumerate(images):
                cached = self._cache.get(img)
                if cached is not None:
                    all_features.append((i, cached))
                else:
                    uncached_indices.append(i)
                    uncached_images.append(img)
        else:
            uncached_indices = list(range(len(images)))
            uncached_images = images

        # Process uncached images
        if uncached_images:
            iterator = range(0, len(uncached_images), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding", unit="batch")

            for start_idx in iterator:
                end_idx = min(start_idx + batch_size, len(uncached_images))
                batch = uncached_images[start_idx:end_idx]

                # Preprocess
                batch_tensor = self.preprocess(batch)
                batch_tensor = batch_tensor.to(self.device)

                # Encode
                features = self.forward(batch_tensor)

                # Store features
                for j, feat in enumerate(features):
                    global_idx = uncached_indices[start_idx + j]
                    all_features.append((global_idx, feat.cpu()))

                    # Cache if enabled
                    if use_cache and self._cache is not None:
                        self._cache.put(batch[j], feat.cpu())

        # Sort by original index and stack
        all_features.sort(key=lambda x: x[0])
        result = torch.stack([f for _, f in all_features])

        return result

    @torch.no_grad()
    def encode_video(
        self,
        frames: Union[torch.Tensor, List[Image.Image], List[np.ndarray]],
        temporal_pooling: Literal["mean", "max", "first", "last", "none"] = "none",
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode a sequence of video frames.

        Args:
            frames: Sequence of video frames.
            temporal_pooling: How to aggregate temporal features:
                - "mean": Average pooling over time
                - "max": Max pooling over time
                - "first": Use only first frame
                - "last": Use only last frame
                - "none": Return all frame features (T, D)
            batch_size: Batch size for processing frames.
            show_progress: Show progress bar.

        Returns:
            If temporal_pooling is "none": (T, output_dim)
            Otherwise: (output_dim,)
        """
        # Encode all frames
        frame_features = self.encode(
            frames,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Apply temporal pooling
        if temporal_pooling == "mean":
            return frame_features.mean(dim=0)
        elif temporal_pooling == "max":
            return frame_features.max(dim=0)[0]
        elif temporal_pooling == "first":
            return frame_features[0]
        elif temporal_pooling == "last":
            return frame_features[-1]
        else:
            return frame_features

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """
        Get attention maps from the last forward pass.

        Only available for transformer-based models (ViT, CLIP, DINOv2).

        Returns:
            Attention maps tensor if available, None otherwise.
            Shape depends on model architecture.
        """
        return self._attention_maps

    def get_cache_stats(self) -> Dict:
        """Get feature cache statistics."""
        if self._cache is None:
            return {"caching_enabled": False}

        return {
            "caching_enabled": True,
            "cache_size": len(self._cache),
            "max_size": self._cache.max_size,
            "hits": self._cache.hits,
            "misses": self._cache.misses,
            "hit_rate": (
                self._cache.hits / (self._cache.hits + self._cache.misses)
                if (self._cache.hits + self._cache.misses) > 0
                else 0.0
            )
        }

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def feature_dim(self) -> int:
        """Original feature dimension before projection."""
        return self.config["feature_dim"]

    @property
    def input_size(self) -> int:
        """Expected input image size."""
        return self.config["input_size"]

    def __repr__(self) -> str:
        return (
            f"VisionEncoder(model_name='{self.model_name}', "
            f"output_dim={self.output_dim}, "
            f"device={self.device}, "
            f"frozen={self.freeze_backbone})"
        )


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for video frame sequences.

    Applies temporal modeling over frame-level features to capture
    surgical workflow dynamics.

    Supports multiple temporal modeling approaches:
    - LSTM: Recurrent modeling with long-term dependencies
    - Transformer: Self-attention for temporal relationships
    - Conv1D: Convolutional temporal modeling

    Example:
        >>> temporal = TemporalEncoder(
        ...     input_dim=512,
        ...     hidden_dim=256,
        ...     output_dim=512,
        ...     model_type="transformer"
        ... )
        >>> frame_features = torch.randn(16, 512)  # 16 frames
        >>> video_features = temporal(frame_features)
        >>> print(video_features.shape)  # (16, 512) or (512,) with pooling
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: Optional[int] = None,
        num_layers: int = 2,
        model_type: Literal["lstm", "transformer", "conv1d"] = "transformer",
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
        output_pooling: Literal["mean", "max", "last", "cls", "none"] = "none"
    ):
        """
        Initialize temporal encoder.

        Args:
            input_dim: Dimension of input frame features.
            hidden_dim: Hidden dimension for temporal model.
            output_dim: Output dimension. If None, uses input_dim.
            num_layers: Number of temporal layers.
            model_type: Type of temporal model ("lstm", "transformer", "conv1d").
            num_heads: Number of attention heads (for transformer).
            dropout: Dropout probability.
            bidirectional: Use bidirectional LSTM (for lstm type).
            output_pooling: How to pool temporal outputs:
                - "mean": Mean pooling
                - "max": Max pooling
                - "last": Last timestep
                - "cls": Learned CLS token (transformer only)
                - "none": Return all timesteps
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.model_type = model_type
        self.output_pooling = output_pooling

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Build temporal model
        if model_type == "lstm":
            self.temporal = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            temporal_output_dim = hidden_dim * (2 if bidirectional else 1)

        elif model_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
            self.temporal = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            temporal_output_dim = hidden_dim

            # Learnable CLS token for pooling
            if output_pooling == "cls":
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

            # Positional encoding
            self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_len=1000)

        elif model_type == "conv1d":
            layers = []
            for i in range(num_layers):
                in_ch = hidden_dim if i == 0 else hidden_dim
                out_ch = hidden_dim
                layers.extend([
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])
            self.temporal = nn.Sequential(*layers)
            temporal_output_dim = hidden_dim

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(temporal_output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim)
        )

        logger.info(
            f"TemporalEncoder initialized: type={model_type}, "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"output_dim={self.output_dim}, layers={num_layers}"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through temporal encoder.

        Args:
            x: Input features. Shape:
                - (T, D) for single sequence
                - (B, T, D) for batched sequences
            lengths: Optional sequence lengths for variable-length sequences.

        Returns:
            Temporal features. Shape depends on output_pooling:
                - "none": (B, T, output_dim) or (T, output_dim)
                - otherwise: (B, output_dim) or (output_dim,)
        """
        # Handle single sequence
        single_sequence = len(x.shape) == 2
        if single_sequence:
            x = x.unsqueeze(0)  # (1, T, D)

        B, T, D = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, T, hidden_dim)

        # Apply temporal model
        if self.model_type == "lstm":
            x, _ = self.temporal(x)  # (B, T, hidden_dim * 2)

        elif self.model_type == "transformer":
            # Add positional encoding
            x = self.pos_encoding(x)

            # Add CLS token if using cls pooling
            if self.output_pooling == "cls":
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_dim)

            x = self.temporal(x)  # (B, T(+1), hidden_dim)

        elif self.model_type == "conv1d":
            x = x.transpose(1, 2)  # (B, hidden_dim, T)
            x = self.temporal(x)
            x = x.transpose(1, 2)  # (B, T, hidden_dim)

        # Apply output pooling
        if self.output_pooling == "mean":
            x = x.mean(dim=1)  # (B, D)
        elif self.output_pooling == "max":
            x = x.max(dim=1)[0]  # (B, D)
        elif self.output_pooling == "last":
            x = x[:, -1]  # (B, D)
        elif self.output_pooling == "cls":
            x = x[:, 0]  # (B, D) - CLS token
        # else: keep all timesteps

        # Output projection
        x = self.output_proj(x)

        # Remove batch dimension if single sequence
        if single_sequence and len(x.shape) == 2:
            x = x.squeeze(0)

        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VideoFeatureExtractor:
    """
    High-level API for extracting features from video files.

    Combines VisionEncoder and TemporalEncoder for end-to-end
    video feature extraction.

    Example:
        >>> extractor = VideoFeatureExtractor(
        ...     vision_model="clip-vit-l-14",
        ...     temporal_model="transformer",
        ...     output_dim=512
        ... )
        >>> features = extractor.extract_from_video("surgery.mp4")
        >>> features = extractor.extract_from_frames(frame_dir)
    """

    def __init__(
        self,
        vision_model: str = "clip-vit-l-14",
        temporal_model: Optional[str] = "transformer",
        output_dim: int = 512,
        temporal_pooling: str = "none",
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize video feature extractor.

        Args:
            vision_model: Name of vision encoder model.
            temporal_model: Type of temporal encoder ("lstm", "transformer", "conv1d").
                          If None, no temporal modeling is applied.
            output_dim: Output feature dimension.
            temporal_pooling: Temporal pooling strategy.
            device: Device for computation.
            **kwargs: Additional arguments for encoders.
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize vision encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model,
            output_dim=output_dim,
            device=str(self.device),
            **kwargs.get("vision_kwargs", {})
        )

        # Initialize temporal encoder if specified
        self.temporal_encoder = None
        if temporal_model:
            self.temporal_encoder = TemporalEncoder(
                input_dim=output_dim,
                output_dim=output_dim,
                model_type=temporal_model,
                output_pooling=temporal_pooling,
                **kwargs.get("temporal_kwargs", {})
            ).to(self.device)

        self.output_dim = output_dim

    def extract_from_frames(
        self,
        frames: Union[List[Image.Image], List[np.ndarray], Path, str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Extract features from a sequence of frames.

        Args:
            frames: List of frames or path to directory containing frames.
            batch_size: Batch size for processing.
            show_progress: Show progress bar.

        Returns:
            Feature tensor of shape (T, output_dim) or (output_dim,) with pooling.
        """
        import cv2

        # Load frames from directory if path
        if isinstance(frames, (str, Path)):
            frame_dir = Path(frames)
            if not frame_dir.exists():
                raise FileNotFoundError(f"Frame directory not found: {frame_dir}")

            # Find all image files
            extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
            frame_files = sorted([
                f for f in frame_dir.iterdir()
                if f.suffix.lower() in extensions
            ])

            if not frame_files:
                raise ValueError(f"No image files found in {frame_dir}")

            frames = [Image.open(f).convert("RGB") for f in frame_files]

        # Extract frame features
        frame_features = self.vision_encoder.encode(
            frames,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Apply temporal modeling if available
        if self.temporal_encoder is not None:
            frame_features = frame_features.to(self.device)
            frame_features = self.temporal_encoder(frame_features)

        return frame_features

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        fps: float = 1.0,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Extract features directly from a video file.

        Args:
            video_path: Path to video file.
            fps: Frames per second to extract.
            batch_size: Batch size for processing.
            show_progress: Show progress bar.

        Returns:
            Feature tensor of shape (T, output_dim) or (output_dim,) with pooling.
        """
        import cv2

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if video_fps <= 0:
                video_fps = 30.0

            frame_interval = max(1, int(video_fps / fps))

            frames = []
            frame_count = 0

            pbar = None
            if show_progress:
                estimated = total_frames // frame_interval
                pbar = tqdm(total=estimated, desc="Extracting frames")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))

                    if pbar:
                        pbar.update(1)

                frame_count += 1

            if pbar:
                pbar.close()

        finally:
            cap.release()

        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")

        return self.extract_from_frames(
            frames,
            batch_size=batch_size,
            show_progress=show_progress
        )


def list_available_models() -> Dict[str, str]:
    """
    List all available vision encoder models.

    Returns:
        Dictionary mapping model names to descriptions.
    """
    return {name: config["description"] for name, config in MODEL_CONFIGS.items()}


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the vision encoder module.

    Args:
        level: Logging level.
        log_file: Optional file path for logs.
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def main():
    """CLI entry point for testing vision encoder."""
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Vision encoder for surgical image/video feature extraction"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image file"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        help="Path to directory containing frames"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="clip-vit-l-14",
        choices=list(MODEL_CONFIGS.keys()),
        help="Vision encoder model (default: clip-vit-l-14)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for features (.npy or .pt)"
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=512,
        help="Output feature dimension (default: 512)"
    )
    parser.add_argument(
        "--temporal",
        type=str,
        choices=["lstm", "transformer", "conv1d", "none"],
        default="none",
        help="Temporal modeling type (default: none)"
    )
    parser.add_argument(
        "--temporal-pooling",
        type=str,
        choices=["mean", "max", "last", "cls", "none"],
        default="none",
        help="Temporal pooling strategy (default: none)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second for video extraction (default: 1.0)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation (default: auto)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # List models and exit
    if args.list_models:
        print("\nAvailable Vision Encoder Models:")
        print("=" * 60)
        for name, desc in list_available_models().items():
            config = MODEL_CONFIGS[name]
            print(f"\n  {name}")
            print(f"    Description: {desc}")
            print(f"    Feature dim: {config['feature_dim']}")
            print(f"    Input size: {config['input_size']}x{config['input_size']}")
        print()
        return

    # Validate input
    if not any([args.image, args.video, args.frames_dir]):
        parser.error("One of --image, --video, or --frames-dir is required")

    # Process input
    start_time = time.time()

    if args.image:
        # Single image encoding
        print(f"\nEncoding image: {args.image}")
        print(f"Model: {args.model}")

        encoder = VisionEncoder(
            model_name=args.model,
            output_dim=args.output_dim,
            device=args.device
        )

        image = Image.open(args.image).convert("RGB")
        features = encoder.encode(image)

        print(f"\nFeature shape: {features.shape}")
        print(f"Feature dtype: {features.dtype}")
        print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")

    elif args.video or args.frames_dir:
        # Video/frames encoding
        temporal_model = args.temporal if args.temporal != "none" else None

        extractor = VideoFeatureExtractor(
            vision_model=args.model,
            temporal_model=temporal_model,
            output_dim=args.output_dim,
            temporal_pooling=args.temporal_pooling,
            device=args.device
        )

        if args.video:
            print(f"\nExtracting features from video: {args.video}")
            print(f"Model: {args.model}")
            print(f"FPS: {args.fps}")
            if temporal_model:
                print(f"Temporal model: {temporal_model}")

            features = extractor.extract_from_video(
                args.video,
                fps=args.fps,
                batch_size=args.batch_size
            )
        else:
            print(f"\nExtracting features from frames: {args.frames_dir}")
            print(f"Model: {args.model}")
            if temporal_model:
                print(f"Temporal model: {temporal_model}")

            features = extractor.extract_from_frames(
                args.frames_dir,
                batch_size=args.batch_size
            )

        print(f"\nFeature shape: {features.shape}")
        print(f"Feature dtype: {features.dtype}")

    elapsed = time.time() - start_time
    print(f"\nProcessing time: {elapsed:.2f}s")

    # Save features if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".npy":
            np.save(output_path, features.cpu().numpy())
        elif output_path.suffix == ".pt":
            torch.save(features, output_path)
        else:
            # Default to numpy
            np.save(output_path.with_suffix(".npy"), features.cpu().numpy())

        print(f"Features saved to: {output_path}")

    # Print cache stats
    if args.image:
        cache_stats = encoder.get_cache_stats()
        if cache_stats.get("caching_enabled"):
            print(f"\nCache stats: {cache_stats}")

    print("\nDone!")


if __name__ == "__main__":
    main()
