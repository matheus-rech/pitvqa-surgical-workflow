# Agent 3: SAGE/Molmo Fine-tuning Pipeline for Surgical Video Understanding

## Understanding SAGE vs Molmo

Before diving in, let's clarify the model architecture:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SAGE Agent System                              │
│         (Multi-turn reasoning, tool orchestration, temporal analysis)       │
│                                                                             │
│    ┌─────────────────────────────────────────────────────────────────────┐ │
│    │                          Molmo 2 (Base VLM)                         │ │
│    │                                                                     │ │
│    │   • Visual understanding (what's in the image)                      │ │
│    │   • Spatial pointing (pixel coordinates for objects)                │ │
│    │   • Object tracking (stable IDs across frames)                      │ │
│    │   • Temporal grounding (timestamps for events)                      │ │
│    │                                                                     │ │
│    │   ═══════════════════════════════════════════════════               │ │
│    │   THIS IS WHAT WE FINE-TUNE FOR SURGICAL UNDERSTANDING              │ │
│    │   ═══════════════════════════════════════════════════               │ │
│    └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | What it does | Our focus |
|-----------|--------------|-----------|
| **Molmo 2** | Base VLM with pointing and tracking | ✅ Fine-tune this for surgical vision |
| **SAGE** | Agentic reasoning layer | Uses fine-tuned Molmo as backbone |

When you fine-tune `allenai/SAGE-MM-Molmo2-8B-SFT_RL`, you're primarily adapting the Molmo vision components to understand surgical anatomy, instruments, and phases.

## Overview

Agent 3 fine-tunes SAGE/Molmo for pituitary endoscopy video understanding:

- **Spatial grounding**: Point to anatomical structures (sella, carotid, pituitary)
- **Instrument detection**: Identify and track surgical tools
- **Temporal understanding**: Recognize surgical phases and steps
- **Visual QA**: Answer questions about surgical procedures

## Pipeline Architecture

```
Agent 1 Output                    Agent 2 Output
(109k frames, QA pairs)           (512-dim skill embeddings)
         │                                │
         └────────────────┬───────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Pointing Annotation                                               │
│  • Grounding DINO → Instrument detection (endoscope, curette, suction)     │
│  • VLM pseudo-labels → Anatomical structures (pituitary, carotid, dura)    │
│  • Output: <point x='0.5' y='0.3'>pituitary_gland</point>                  │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2: Data Conversion                                                   │
│  • SFT: Conversation format with pointing annotations                       │
│  • DPO: Preference pairs (correct vs incorrect identifications)            │
│  • GRPO: RL format with surgical reward functions                           │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3: Training (via HuggingFace Skills)                                │
│  • Generates training scripts compatible with TRL                           │
│  • Submits to HuggingFace Jobs infrastructure                              │
│  • Supports A10G/T4 GPU tiers with LoRA                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 4: Evaluation                                                        │
│  • Pointing accuracy (spatial grounding F1)                                │
│  • Phase/step classification (accuracy, macro-F1)                          │
│  • Instrument detection (multi-label F1)                                   │
│  • VQA quality (BLEU-4, ROUGE-L, BERTScore)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
               PitVQA-SAGE Model
       (matheus-rech/pitvqa-sage-surgical)
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision
pip install transformers datasets
pip install trl peft
pip install pillow tqdm numpy
pip install scikit-learn  # For evaluation

# Optional for Grounding DINO
pip install groundingdino

# Optional for VQA metrics
pip install nltk rouge_score bert_score
```

### Full Pipeline

```bash
# Run complete pipeline with SFT training
python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \
    --input-dataset matheus-rech/pitvqa-processed \
    --agent2-embeddings data/skill_embeddings \
    --output-dir outputs/pitvqa-sage \
    --method sft \
    --base-model allenai/SAGE-MM-Molmo2-8B-SFT_RL \
    --push-to-hub
```

### Using HuggingFace Skills (Recommended)

After running the pipeline, use the generated prompt with Claude Code:

```bash
# 1. Generate training config
python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \
    --stages training \
    --input-dataset matheus-rech/pitvqa-processed \
    --output-dir outputs/pitvqa-sage

# 2. The pipeline outputs a prompt at: outputs/pitvqa-sage/hf_skills_prompt.txt
# 3. Use it with Claude Code to trigger HF Skills training
```

Example HF Skills prompt generated:

```
Fine-tune allenai/SAGE-MM-Molmo2-8B-SFT_RL on matheus-rech/pitvqa-sage-sft-data

Configuration:
- Output model: matheus-rech/pitvqa-sage-surgical
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-05
- Use LoRA: True (r=16)
- Hardware: a10g-large
- This is a vision language model with images in 'images' column
```

## Training Methods

### 1. SFT (Supervised Fine-Tuning)

Best for: Initial fine-tuning with high-quality surgical annotations

```bash
python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \
    --input-dataset matheus-rech/pitvqa-processed \
    --method sft \
    --epochs 3
```

Dataset format:
```json
{
    "messages": [
        {"role": "user", "content": "Point to the pituitary gland in this image."},
        {"role": "assistant", "content": "The pituitary gland is located here: <point x='0.52' y='0.38'>pituitary_gland</point>"}
    ],
    "images": ["path/to/frame.jpg"]
}
```

### 2. DPO (Direct Preference Optimization)

Best for: Aligning model to prefer correct identifications over incorrect ones

```bash
python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \
    --input-dataset matheus-rech/pitvqa-processed \
    --method dpo \
    --epochs 2
```

Dataset format:
```json
{
    "prompt": "What instrument is being used?",
    "chosen": "The surgeon is using a curette for tumor resection.",
    "rejected": "The surgeon is using scissors (incorrect).",
    "images": ["path/to/frame.jpg"]
}
```

### 3. GRPO (Group Relative Policy Optimization)

Best for: RL-based training with verifiable surgical rewards

```bash
python -m agent3_sage_finetuning.pitvqa_agent3_sage_finetuning \
    --input-dataset matheus-rech/pitvqa-processed \
    --method grpo \
    --epochs 5
```

Reward functions:
| Reward | Weight | Description |
|--------|--------|-------------|
| Pointing accuracy | 0.30 | Correct (x, y) coordinates within threshold |
| Phase classification | 0.20 | Correct surgical phase identification |
| Step classification | 0.20 | Correct surgical step identification |
| Instrument detection | 0.20 | Correct instruments mentioned |
| Format compliance | 0.10 | Proper <point> format, reasonable length |

## Components

### Data Converter

Converts Agent 1/2 outputs to Molmo training format:

```python
from agent3_sage_finetuning import MolmoDataConverter, TrainingMethod

converter = MolmoDataConverter()
dataset = converter.create_surgical_vqa_dataset(
    agent1_path="matheus-rech/pitvqa-processed",
    agent2_path="data/skill_embeddings",
    training_method=TrainingMethod.SFT,
    output_path="data/sft_dataset",
    push_to_hub="matheus-rech/pitvqa-sage-sft"
)
```

### Pointing Annotator

Generates spatial annotations for surgical structures:

```python
from agent3_sage_finetuning import SurgicalPointingAnnotator

annotator = SurgicalPointingAnnotator(
    use_grounding_dino=True,
    use_vlm_pseudolabels=True,
    device="cuda"
)

annotation = annotator.annotate_frame(
    image="frame_001.jpg",
    frame_id="frame_001",
    phase="tumor_removal_phase",
    step="tumor_resection"
)

# Output: <point x='0.45' y='0.52'>curette</point> <point x='0.62' y='0.38'>pituitary_tumor</point>
print(annotation.get_molmo_points())
```

### HF Skills Trainer

Generates training configurations for HuggingFace Skills:

```python
from agent3_sage_finetuning import create_surgical_vqa_trainer

trainer = create_surgical_vqa_trainer(
    base_model="allenai/SAGE-MM-Molmo2-8B-SFT_RL",
    dataset_id="matheus-rech/pitvqa-sage-sft",
    output_name="matheus-rech/pitvqa-sage-surgical",
    method="sft",
    num_epochs=3
)

# Generate training script
trainer.generate_training_script("train_sft.py")

# Generate HF Skills prompt
print(trainer.generate_hf_skills_prompt())
```

### Surgical Evaluator

Evaluates fine-tuned models:

```python
from agent3_sage_finetuning import SurgicalVQAEvaluator

evaluator = SurgicalVQAEvaluator(
    model_path="matheus-rech/pitvqa-sage-surgical",
    device="cuda"
)

report = evaluator.evaluate_dataset(
    dataset=test_dataset,
    output_path="evaluation_report.json"
)

print(report.to_markdown())
```

## Surgical Vocabulary

### Anatomical Structures

| Structure | Description | Detection Method |
|-----------|-------------|-----------------|
| pituitary_gland | Normal pituitary tissue | VLM |
| pituitary_tumor | Pituitary adenoma | VLM |
| carotid_prominence | Internal carotid artery | VLM |
| optic_prominence | Optic nerve prominence | VLM |
| sella_floor | Sellar floor bone | VLM |
| sphenoid_sinus | Sphenoid sinus cavity | VLM |
| dura_mater | Dura mater membrane | VLM |
| nasal_septum | Nasal septum | Grounding DINO |

### Instruments

| Instrument | Description | Detection Method |
|------------|-------------|-----------------|
| endoscope | 0°/30° rigid endoscope | Grounding DINO |
| suction | Suction cannula | Grounding DINO |
| curette | Ring curette | Grounding DINO |
| bipolar | Bipolar forceps | Grounding DINO |
| drill | High-speed drill | Grounding DINO |

### Surgical Phases

| Phase | Steps |
|-------|-------|
| nasal_phase | septal_dissection, turbinectomy, sphenoidotomy |
| sellar_phase | posterior_septectomy, sellar_floor_removal, dura_opening |
| tumor_removal_phase | tumor_resection, hemostasis |
| closure_phase | reconstruction, nasal_packing |

## Expected Outputs

| Metric | Target | Description |
|--------|--------|-------------|
| Pointing F1 | > 0.70 | Spatial grounding accuracy |
| Phase Accuracy | > 0.85 | Surgical phase classification |
| Step Accuracy | > 0.75 | Surgical step classification |
| Instrument F1 | > 0.80 | Multi-label instrument detection |
| BLEU-4 | > 0.40 | VQA answer quality |
| BERTScore | > 0.75 | Semantic similarity |

## Model Selection Guide

| Model | Size | Use Case | Training Time |
|-------|------|----------|---------------|
| SAGE-MM-Qwen3-VL-4B | 5B | Quick experiments | ~2 hours |
| SAGE-MM-Molmo2-8B-SFT | 9B | Best pointing | ~4 hours |
| SAGE-MM-Molmo2-8B-SFT_RL | 9B | Best reasoning | ~4 hours |
| SAGE-MM-Qwen3-VL-8B | 9B | Balanced | ~4 hours |

## Hardware Requirements

| Stage | Hardware | Memory | Time |
|-------|----------|--------|------|
| Annotation | CPU or GPU | 8GB RAM | ~1 hour |
| Conversion | CPU | 16GB RAM | ~30 min |
| Training (LoRA) | A10G-large | 24GB VRAM | ~4 hours |
| Evaluation | A10G-small | 16GB VRAM | ~1 hour |

## References

- **SAGE Paper**: [arXiv:2512.13874](https://arxiv.org/abs/2512.13874)
- **Molmo 2 Blog**: [Allen AI](https://allenai.org/blog/molmo2)
- **HF Skills Training**: [HuggingFace Blog](https://huggingface.co/blog/hf-skills-training)
- **TRL Library**: [HuggingFace TRL](https://huggingface.co/docs/trl)

## License

This pipeline is part of the PitVQA project, licensed under CC BY-NC-ND 4.0.

---

*Part of the PitVQA Surgical Workflow Understanding project - MICCAI 2026*
