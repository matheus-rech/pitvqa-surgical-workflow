# Agent 1: PitVQA Data Preparation Pipeline

## Overview

Agent 1 handles the complete data preparation workflow for the PitVQA surgical Visual Question Answering project. It transforms raw surgical videos and annotations into a training-ready HuggingFace dataset.

## Pipeline Stages

```
Raw Videos (25 .mp4)     Annotations (59 classes)
        │                        │
        ▼                        │
┌───────────────────┐            │
│ Frame Extraction  │            │
│ (1 fps, blur      │            │
│  filtering)       │            │
└───────────────────┘            │
        │                        │
        ▼                        ▼
┌───────────────────────────────────────┐
│         QA Pair Generation            │
│  (~8 questions per frame)             │
│  - Phase, Step, Instrument questions  │
│  - Position, Yes/No, Operation notes  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────┐
│ Dataset Builder   │
│ (Train/Val/Test   │
│  80/10/10 split)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Validation      │
│   & Statistics    │
└───────────────────┘
        │
        ▼
  HuggingFace Dataset
  (109k frames, 884k QA pairs)
```

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
brew install ffmpeg  # macOS
# or: apt-get install ffmpeg  # Linux
```

### Basic Usage

```bash
python -m agent1_data_prep.pitvqa_agent1_data_prep \
    --video-dir data/raw/pitvqa/videos \
    --annotation-dir data/raw/pitvqa/annotations \
    --output-dir data/processed \
    --fps 1 \
    --blur-threshold 100
```

### Push to HuggingFace Hub

```bash
python -m agent1_data_prep.pitvqa_agent1_data_prep \
    --video-dir data/raw/pitvqa/videos \
    --annotation-dir data/raw/pitvqa/annotations \
    --output-dir data/processed \
    --push-to-hub matheus-rech/pitvqa-processed \
    --hf-token $HF_TOKEN
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--video-dir` | Path to video files | Required |
| `--annotation-dir` | Path to annotation files | Required |
| `--output-dir` | Output directory | Required |
| `--fps` | Frames per second to extract | 1 |
| `--blur-threshold` | Laplacian variance threshold | 100 |
| `--push-to-hub` | HuggingFace repo ID | None |
| `--hf-token` | HuggingFace token | $HF_TOKEN |
| `--skip-extraction` | Skip frame extraction | False |
| `--skip-qa-generation` | Skip QA generation | False |

## Components

### 1. Frame Extractor (`frame_extractor.py`)

Extracts frames from surgical videos with blur detection.

```python
from agent1_data_prep import FrameExtractor

extractor = FrameExtractor(fps=1, blur_threshold=100)
stats = extractor.extract_from_directory(
    video_dir="data/raw/videos",
    output_dir="data/processed/frames"
)
print(f"Extracted {stats['valid_frames']} frames")
```

**Features:**
- Configurable FPS (default: 1)
- Laplacian variance blur detection
- Progress bars with tqdm
- Memory-efficient streaming

### 2. QA Generator (`qa_generator.py`)

Generates question-answer pairs from PitVQA annotations.

```python
from agent1_data_prep import QAGenerator

generator = QAGenerator()
generator.load_annotations("data/raw/annotations")
qa_pairs = generator.generate_all_qa_pairs(
    frames_dir="data/processed/frames",
    annotation_dir="data/raw/annotations"
)
print(f"Generated {len(qa_pairs)} QA pairs")
```

**Question Types:**
- Phase: "What surgical phase is this?"
- Step: "What step is being performed?"
- Instrument: "What instruments are visible?"
- Position: "Where is the instrument?"
- Yes/No: "Is the grasper visible?"
- Operation Note: "What is the operation note?"

### 3. Dataset Builder (`dataset_builder.py`)

Creates HuggingFace datasets with proper train/val/test splits.

```python
from agent1_data_prep import DatasetBuilder

builder = DatasetBuilder()
dataset = builder.build_dataset(
    frames_dir="data/processed/frames",
    qa_pairs_file="data/processed/qa_pairs.json"
)
dataset_dict = builder.create_splits(dataset)
builder.push_to_hub(dataset_dict, "username/pitvqa-processed")
```

**Features:**
- Video-level splits (no data leakage)
- 80/10/10 train/val/test ratio
- Parquet format for efficiency
- Automatic dataset card generation

### 4. Validator (`validators.py`)

Validates pipeline outputs against expected specifications.

```python
from agent1_data_prep import DataValidator

validator = DataValidator()
report = validator.validate_all(
    frames_dir="data/processed/frames",
    qa_pairs_file="data/processed/qa_pairs.json",
    dataset=dataset
)
print(report.summary())
```

**Validations:**
- Frame count (~109k expected)
- QA pair count (~884k expected)
- Class distribution (59 classes)
- Data integrity checks

## Expected Outputs

| Metric | Target |
|--------|--------|
| Videos processed | 25 |
| Total frames extracted | ~109,173 |
| Blurry frames filtered | ~10-15% |
| Valid frames | ~95,000-100,000 |
| QA pairs generated | ~884,242 |
| Questions per frame | ~8 |
| Annotation classes | 59 |

## Annotation Classes

| Category | Count | Examples |
|----------|-------|----------|
| Phases | 4 | Nasal, Sellar, Tumor Removal, Closure |
| Steps | 15 | Septal dissection, Sphenoidotomy, etc. |
| Instruments | 18 | Grasper, Scissors, Cautery, etc. |
| Presence | 3 | Single, Multiple, None |
| Positions | 5 | Left, Right, Center, etc. |
| Operation Notes | 14 | Bleeding, Clear field, etc. |

## Troubleshooting

### Out of Memory
```bash
# Process videos one at a time
python -m agent1_data_prep.pitvqa_agent1_data_prep \
    --video-dir data/raw/videos \
    --output-dir data/processed \
    --batch-size 1
```

### Resume After Interruption
```bash
# Skip already-extracted frames
python -m agent1_data_prep.pitvqa_agent1_data_prep \
    --video-dir data/raw/videos \
    --output-dir data/processed \
    --skip-extraction
```

### Validation Failed
Check the validation report for specific issues:
```python
validator = DataValidator()
report = validator.validate_all(...)
for error in report.errors:
    print(f"ERROR: {error}")
for warning in report.warnings:
    print(f"WARNING: {warning}")
```

## License

This pipeline is for the PitVQA dataset, which is licensed under CC BY-NC-ND 4.0.

---

*Part of the PitVQA Surgical Workflow Understanding project - MICCAI 2026*
