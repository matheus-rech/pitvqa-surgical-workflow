# PitVQA Surgical Workflow

**Unified Vision-Language Model for Multi-Task Surgical Understanding**

A comprehensive pipeline for training and evaluating a unified VLM that performs:
1. **Surgical Phase Classification** - Identify current surgical phase (nasal, sellar, tumor_removal, closure)
2. **Surgical Step Classification** - Identify surgical step (10 distinct steps)
3. **Instrument Pointing** - Spatial localization of surgical instruments
4. **Anatomy Pointing** - Spatial localization of anatomical structures

> Developed for MICCAI 2026 submission on pituitary surgery video understanding.

## Model & Dataset

| Resource | HuggingFace Link |
|----------|------------------|
| **Unified Model** | [mmrech/pitvqa-qwen2vl-unified](https://huggingface.co/mmrech/pitvqa-qwen2vl-unified) |
| **Unified Dataset** | [mmrech/pitvqa-unified-vlm](https://huggingface.co/datasets/mmrech/pitvqa-unified-vlm) |
| **Spatial Dataset** | [mmrech/pitvqa-spatial-vlm](https://huggingface.co/datasets/mmrech/pitvqa-spatial-vlm) |
| **Classification Dataset** | [mmrech/pitvqa-sage-sft](https://huggingface.co/datasets/mmrech/pitvqa-sage-sft) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qwen2-VL-2B-Instruct                         │
│                         (Base Model)                            │
├─────────────────────────────────────────────────────────────────┤
│                      QLoRA Adapter                              │
│              r=32, alpha=64, dropout=0.05                       │
│         Trainable: 36.9M params (1.64% of 2.25B)                │
├─────────────────────────────────────────────────────────────────┤
│                    Multi-Task Heads                             │
│  ┌─────────────┬─────────────┬──────────────┬────────────────┐  │
│  │   Phase     │    Step     │  Instrument  │    Anatomy     │  │
│  │Classification│Classification│  Pointing   │   Pointing    │  │
│  └─────────────┴─────────────┴──────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/matheus-rech/pitvqa-surgical-workflow.git
cd pitvqa-surgical-workflow

# Install dependencies
pip install -r requirements.txt
```

### Inference

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

# Load model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "mmrech/pitvqa-qwen2vl-unified")

# Example: Phase classification
conversation = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What surgical phase is shown in this image?"}
    ]}
]

text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[your_image], return_tensors="pt")

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=50)

response = processor.decode(outputs[0], skip_special_tokens=True)
# Output: "This is the sellar phase."

# Example: Instrument pointing
conversation = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Point to the suction device in this surgical image."}
    ]}
]
# Output: "<point x='32.5' y='45.2'>suction device</point>"
```

## Project Structure

```
pitvqa-surgical-workflow/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
│
├── # Dataset Creation
├── create_unified_vlm_dataset.py       # Merge datasets for multi-task training
├── create_spatial_dataset.py           # Create spatial pointing dataset
├── create_comprehensive_spatial_dataset.py
│
├── # Training
├── train_unified_vlm.py                # Main training script (QLoRA + SFTTrainer)
│
├── # Evaluation
├── evaluate_unified_vlm.py             # Comprehensive metrics for publication
├── demo_model_samples.py               # Visual demo with HTML report
├── test_baseline_performance.py        # Baseline model comparison
│
├── # Annotation Pipeline
├── surgical_annotation_pipeline.py     # Multi-model annotation orchestration
├── run_gemini_annotation.py            # Gemini 2.5 Pro annotation
├── run_gpt_annotation.py               # GPT-4o annotation
├── run_grok_annotation.py              # Grok annotation
├── annotation_config.py                # Shared configuration
│
├── # Inter-Rater Reliability
├── compute_irr_metrics.py              # Cohen's/Fleiss' Kappa, Krippendorff's Alpha
│
├── # Human Review
├── anatomy_review_tool.py              # CLI tool for annotation validation
│
├── # Data Directories
├── gemini_annotations/                 # Gemini annotation outputs
├── gpt_annotations/                    # GPT-4o annotation outputs
├── grok_annotations/                   # Grok annotation outputs
└── ground_truth/                       # Ground truth instrument data
```

## Pipeline Overview

### Phase 1: Annotation Generation

```bash
# Generate annotations using multiple LLMs
python run_gemini_annotation.py  # Gemini 2.5 Pro
python run_gpt_annotation.py     # GPT-4o
python run_grok_annotation.py    # Grok
```

### Phase 2: Inter-Rater Reliability Analysis

```bash
# Compute agreement metrics across annotators
python compute_irr_metrics.py

# Results saved to irr_analysis_report.json
# Key metrics: Cohen's Kappa, Fleiss' Kappa, Krippendorff's Alpha
```

### Phase 3: Human Annotation Review

```bash
# Review Gemini anatomy annotations with stratified sampling
python anatomy_review_tool.py           # Interactive review
python anatomy_review_tool.py --status  # Show progress
python anatomy_review_tool.py --export  # Export reviewed annotations
```

**Stratified Sampling:**
- High confidence (≥0.9): 10% sample
- Medium confidence (0.7-0.9): 30% sample
- Low confidence (<0.7): 100% review

### Phase 4: Dataset Creation

```bash
# Create unified multi-task dataset
python create_unified_vlm_dataset.py

# Pushes to: mmrech/pitvqa-unified-vlm
```

**Dataset Statistics:**
| Task | Train | Val | Ratio |
|------|-------|-----|-------|
| Phase Classification | 1,193 | 103 | 25% |
| Step Classification | 1,193 | 103 | 25% |
| Instrument Pointing | 1,670 | 144 | 35% |
| Anatomy Pointing | 716 | 62 | 15% |
| **Total** | **4,772** | **412** | 100% |

### Phase 5: Training

```bash
# Local training
python train_unified_vlm.py

# HuggingFace Jobs (A100 GPU)
# Configured via hf_jobs API
```

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen2-VL-2B-Instruct |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA Rank | 32 |
| LoRA Alpha | 64 |
| Batch Size | 1 (effective: 16 with gradient accumulation) |
| Learning Rate | 1e-4 |
| Epochs | 3 |
| Total Steps | 897 |

### Phase 6: Evaluation

```bash
# Run comprehensive evaluation
python evaluate_unified_vlm.py --model mmrech/pitvqa-qwen2vl-unified

# Quick demo
python demo_model_samples.py
```

**Outputs:**
- `evaluation_results/evaluation_results.json` - Detailed metrics
- `evaluation_results/evaluation_report.md` - Markdown report
- `evaluation_results/latex_tables.tex` - MICCAI paper tables
- `evaluation_results/confusion_matrices/*.png` - Visualizations
- `model_demo_report.html` - Visual demo

## Evaluation Metrics

### Classification Tasks
- **Accuracy**: Overall correctness
- **F1 Score**: Macro and weighted averages
- **Per-Class Metrics**: Precision, recall, F1, support
- **Confusion Matrix**: Full class-by-class breakdown

### Pointing Tasks
- **Accuracy @10%**: Point within 10% Euclidean distance
- **Accuracy @15%**: Point within 15% Euclidean distance
- **Quadrant Accuracy**: Correct image quadrant (TL/TR/BL/BR)
- **Mean/Median Distance**: Average localization error
- **Format Rate**: Valid `<point>` format in output

## Point Coordinate Format

All spatial annotations use normalized 0-100 coordinates:

```xml
<point x='32.5' y='45.2'>label_name</point>
```

Where:
- `x`: Horizontal position (0=left, 100=right)
- `y`: Vertical position (0=top, 100=bottom)

## Inter-Rater Reliability Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Krippendorff's Alpha | -0.008 | Low agreement (expected with spatial data) |
| Fleiss' Kappa | -0.10 | Below chance agreement |
| Gemini-GPT Spatial | 58.1% within 15% | Moderate spatial agreement |
| Quadrant Agreement | 56.9% | Better than chance (25%) |

## Citation

```bibtex
@inproceedings{pitvqa2026,
  title={Unified Vision-Language Model for Multi-Task Surgical Understanding},
  author={Rech, Matheus},
  booktitle={MICCAI 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PitVQA Dataset](https://github.com/surgical-vision/pitvqa) - Original pituitary surgery VQA dataset
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) - Base vision-language model
- [TRL](https://github.com/huggingface/trl) - Training library for LLMs
- [PEFT](https://github.com/huggingface/peft) - Parameter-efficient fine-tuning
