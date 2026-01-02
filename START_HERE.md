# PitVQA Surgical Workflow

## Overview

This project focuses on Visual Question Answering (VQA) for pituitary surgery workflows. The goal is to develop AI models that can understand and answer questions about surgical procedures from video/image data.

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision transformers
pip install opencv-python pillow
pip install pandas numpy scikit-learn
pip install jupyter notebook
```

### 2. Project Structure

```
pitvqa-surgical-workflow/
├── data/
│   ├── raw/              # Raw surgical videos/images
│   ├── processed/        # Preprocessed frames
│   └── annotations/      # QA pairs and labels
├── models/
│   ├── vision/           # Visual encoders
│   └── vqa/              # VQA model architectures
├── notebooks/
│   └── exploration.ipynb # Data exploration
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── config.yaml
└── requirements.txt
```

### 3. Data Preparation

Place your surgical data in the `data/raw/` directory:
- Video files (.mp4, .avi)
- Image frames (.png, .jpg)
- Annotation files (.json, .csv)

### 4. Key Tasks

- [ ] Collect surgical video data
- [ ] Create QA annotation pairs
- [ ] Implement data preprocessing pipeline
- [ ] Train vision encoder (ResNet, ViT)
- [ ] Implement VQA model
- [ ] Evaluate model performance

## Resources

### Datasets
- [Cholec80](https://camma.unistra.fr/datasets) - Cholecystectomy surgical videos
- [M2CAI Challenge](http://camma.unistra.fr/m2cai2016/) - Surgical workflow datasets
- [CholecT45](https://github.com/CAMMA-public/cholect45) - Triplet annotations

### Papers
- "Surgical Visual Question Answering" - VQA for surgical procedures
- "Transformer-based models for surgical phase recognition"
- "Deep learning for surgical workflow analysis"

### Tools
- **CVAT** - Video annotation tool
- **Label Studio** - Data labeling platform
- **Weights & Biases** - Experiment tracking

## Model Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Visual Input   │     │  Text Question  │
│   (Video/Image) │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│  Vision Encoder │     │  Text Encoder   │
│  (ViT/ResNet)   │     │  (BERT/T5)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Fusion Module  │
            │  (Cross-Attn)   │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │  Answer Decoder │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │    Answer       │
            └─────────────────┘
```

## Example Questions

- "What surgical instrument is being used?"
- "What is the current surgical phase?"
- "Is the anatomy clearly visible?"
- "What action is the surgeon performing?"
- "Is there any bleeding visible?"

## Contact

For questions or collaboration, please reach out.

---

*Last updated: January 2025*
