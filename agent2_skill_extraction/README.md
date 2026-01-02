# Agent 2: PitVQA Skill Extraction Pipeline

## Overview

Agent 2 handles surgical skill extraction and embedding generation for the PitVQA project. It processes frames from Agent 1 output (or HuggingFace datasets) to generate skill-aware embeddings suitable for reinforcement learning training.

## Pipeline Architecture

```
Agent 1 Output                 HuggingFace Dataset
(109k frames)                  (matheus-rech/pitvqa-processed)
        │                               │
        └───────────────┬───────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Stage 1: Data Loading                  │
│     (Streaming support, checkpoint recovery)        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 2: Visual Feature Extraction         │
│     CLIP ViT-L/14 (768-dim embeddings)             │
│     - Batch processing                              │
│     - GPU acceleration (CUDA/MPS)                   │
│     - FP16 inference                                │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│          Stage 3: Skill Classification              │
│     Multi-task prediction:                          │
│     - Phase (4 classes)                             │
│     - Step (15 classes)                             │
│     - Instruments (18 classes, multi-label)         │
│     - Action (14 classes)                           │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│         Stage 4: Skill Embedding Generation         │
│     Fusion of visual + skill embeddings             │
│     Output: 512-dim skill-aware embeddings          │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Stage 5: Save Results                  │
│     - NumPy arrays (.npy)                           │
│     - HuggingFace Dataset format                    │
│     - Push to Hub (optional)                        │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
            HuggingFace Dataset
            (matheus-rech/pitvqa-skills)
```

## Quick Start

### Prerequisites

```bash
# Required dependencies
pip install torch torchvision
pip install transformers
pip install datasets
pip install pillow
pip install tqdm
pip install numpy
```

### Basic Usage

```bash
# Process from HuggingFace dataset
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --vision-model clip-vit-l-14 \
    --batch-size 32

# Process from local directory (Agent 1 output)
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dir data/processed/frames \
    --output-dir data/skill_embeddings \
    --vision-model clip-vit-l-14 \
    --batch-size 32
```

### Push to HuggingFace Hub

```bash
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --push-to-hub matheus-rech/pitvqa-skills \
    --hf-token $HF_TOKEN
```

### Streaming Mode (Memory Efficient)

```bash
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --streaming \
    --max-samples 10000
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input-dataset` | HuggingFace dataset ID | None |
| `--input-dir` | Local directory with frames | None |
| `--input-split` | Dataset split to process | train |
| `--output-dir` | Output directory | Required |
| `--vision-model` | CLIP model variant | clip-vit-l-14 |
| `--classifier-checkpoint` | Pretrained classifier weights | None |
| `--use-temporal` | Enable temporal encoding | False |
| `--batch-size` | Processing batch size | 32 |
| `--num-workers` | Data loading workers | 4 |
| `--device` | Device (auto/cpu/cuda/mps) | auto |
| `--no-fp16` | Disable FP16 inference | False |
| `--push-to-hub` | HuggingFace repo ID | None |
| `--hf-token` | HuggingFace token | $HF_TOKEN |
| `--streaming` | Use streaming mode | False |
| `--max-samples` | Maximum samples to process | None |
| `--checkpoint-interval` | Save checkpoint every N samples | 1000 |
| `--no-resume` | Do not resume from checkpoint | False |
| `--log-level` | Logging level | INFO |

## Components

### 1. Vision Encoder (`CLIPVisionEncoder`)

Extracts visual features from surgical frames using CLIP.

```python
from agent2_skill_extraction import CLIPVisionEncoder

encoder = CLIPVisionEncoder(
    model_name="clip-vit-l-14",
    device="auto",
    use_fp16=True
)

# Encode batch of images
embeddings = encoder.encode(images)  # (batch_size, 768)
```

**Supported Models:**
| Model | Embedding Dim | Resolution |
|-------|---------------|------------|
| clip-vit-b-32 | 512 | 224px |
| clip-vit-b-16 | 512 | 224px |
| clip-vit-l-14 | 768 | 224px |
| clip-vit-l-14-336 | 768 | 336px |

### 2. Temporal Encoder (`TemporalEncoder`)

Models sequential dependencies across video frames.

```python
from agent2_skill_extraction import TemporalEncoder

temporal = TemporalEncoder(
    input_dim=768,
    hidden_dim=512,
    num_heads=8,
    num_layers=4
)

# Encode sequence with temporal context
temporal_embeddings = temporal.encode(frame_sequence)  # (seq_len, 512)
```

### 3. Skill Classifier (`SkillClassifier`)

Multi-task classification of surgical skills.

```python
from agent2_skill_extraction import SkillClassifier, SkillVocabulary

vocabulary = SkillVocabulary()
classifier = SkillClassifier(
    embedding_dim=768,
    vocabulary=vocabulary
)

# Classify skills
predictions = classifier.classify(embeddings, return_probs=True)
# predictions = {
#     "phase": {"predictions": [...], "labels": [...], "probabilities": [...]},
#     "step": {...},
#     "instrument": {...},
#     "action": {...}
# }
```

### 4. Skill Embedding Generator (`SkillEmbeddingGenerator`)

Fuses visual features with skill predictions.

```python
from agent2_skill_extraction import SkillEmbeddingGenerator

generator = SkillEmbeddingGenerator(
    vision_dim=768,
    skill_embedding_dim=128,
    output_dim=512
)

# Generate skill-aware embeddings
skill_embeddings = generator.generate(visual_embeddings, predictions)
# skill_embeddings: (batch_size, 512)
```

### 5. Skill Vocabulary (`SkillVocabulary`)

Complete taxonomy of surgical skills.

```python
from agent2_skill_extraction import SkillVocabulary, SkillCategory

vocab = SkillVocabulary()

# Access vocabularies
print(vocab.phases)      # 4 surgical phases
print(vocab.steps)       # 15 surgical steps
print(vocab.instruments) # 18 surgical instruments
print(vocab.actions)     # 14 surgical actions

# Convert between names and indices
idx = vocab.skill_to_index("nasal_phase", SkillCategory.PHASE)
name = vocab.index_to_skill(0, SkillCategory.PHASE)
```

## Skill Taxonomy

### Phases (4 classes)
| Phase | Description |
|-------|-------------|
| nasal_phase | Nasal corridor preparation |
| sellar_phase | Sellar floor exposure |
| tumor_removal_phase | Tumor resection |
| closure_phase | Reconstruction and closure |

### Steps (15 classes)
| Step | Description |
|------|-------------|
| septal_dissection | Nasal septum dissection |
| turbinectomy | Turbinate removal |
| sphenoidotomy | Sphenoid sinus opening |
| posterior_septectomy | Posterior septum removal |
| sellar_floor_removal | Sellar floor bone removal |
| dura_opening | Dural incision |
| tumor_resection | Tumor removal |
| hemostasis | Bleeding control |
| reconstruction | Skull base repair |
| nasal_packing | Nasal cavity packing |
| visualization | Camera adjustment |
| instrument_change | Tool switching |
| suction | Aspiration |
| irrigation | Wash/rinse |
| other | Miscellaneous |

### Instruments (18 classes, multi-label)
- grasper, scissors, cautery, suction, curette
- drill, endoscope, bipolar, monopolar, retractor
- forceps, needle_holder, scalpel, speculum, irrigator
- cotton, hemostatic_agent, other

### Actions (14 classes)
- cutting, grasping, dissecting, coagulating, suctioning
- drilling, irrigating, retracting, inspecting, hemostasis
- packing, inserting, removing, idle

## Output Format

### NumPy Embeddings
```python
import numpy as np
embeddings = np.load("data/skill_embeddings/skill_embeddings.npy")
# Shape: (N, 512) where N is number of frames
```

### HuggingFace Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("data/skill_embeddings/hf_dataset")
# Features:
# - embedding: List[float] (512 dims)
# - phase: str
# - phase_prob: float
# - step: str
# - step_prob: float
# - instruments: List[str]
# - action: str
# - (original metadata fields)
```

### Hub Dataset
```python
from datasets import load_dataset

dataset = load_dataset("matheus-rech/pitvqa-skills")
```

## Expected Outputs

| Metric | Target |
|--------|--------|
| Total frames processed | ~109,000 |
| Embedding dimension | 512 |
| Processing speed (GPU) | ~500 frames/sec |
| Processing speed (CPU) | ~50 frames/sec |
| Memory usage (batch=32) | ~4GB GPU RAM |

## Checkpointing

The pipeline supports automatic checkpointing for fault tolerance:

```bash
# Resume from last checkpoint (default)
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings

# Start fresh (ignore checkpoints)
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --no-resume
```

Checkpoints are saved to `.checkpoints/` in the output directory.

## Troubleshooting

### Out of GPU Memory
```bash
# Reduce batch size
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --batch-size 8

# Use CPU
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir data/skill_embeddings \
    --device cpu
```

### Missing Dependencies
```bash
# Install all required packages
pip install torch torchvision transformers datasets pillow tqdm numpy

# For Apple Silicon (M1/M2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### HuggingFace Authentication
```bash
# Login to HuggingFace
huggingface-cli login

# Or use token directly
export HF_TOKEN=your_token_here
python -m agent2_skill_extraction.pitvqa_agent2_skill_extraction \
    --push-to-hub matheus-rech/pitvqa-skills \
    --hf-token $HF_TOKEN
```

## Integration with Agent 3

Agent 2 outputs are designed for Agent 3 (RL training):

```python
from datasets import load_from_disk

# Load skill embeddings
skill_dataset = load_from_disk("data/skill_embeddings/hf_dataset")

# Use for RL training
for sample in skill_dataset:
    embedding = sample["embedding"]  # 512-dim state vector
    phase = sample["phase"]          # Current phase label
    step = sample["step"]            # Current step label
    instruments = sample["instruments"]  # Active instruments
    # ... use for reward shaping, policy training, etc.
```

## License

This pipeline is part of the PitVQA project, licensed under CC BY-NC-ND 4.0.

---

*Part of the PitVQA Surgical Workflow Understanding project - MICCAI 2026*
