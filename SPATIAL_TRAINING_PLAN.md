# Spatial Reasoning & Coordinate-Aware Training Plan

## Current State

### Trained Models Available
| Model | Base | Status | Best For |
|-------|------|--------|----------|
| [pitvqa-qwen2vl-surgical](https://hf.co/mmrech/pitvqa-qwen2vl-surgical) | Qwen2-VL-2B | Ready | Fast inference, spatial reasoning |
| [pitvqa-medgemma-surgical](https://hf.co/mmrech/pitvqa-medgemma-surgical) | MedGemma-4B | Ready | Medical domain knowledge |

### Validation Results
- **Taxonomy alignment:** ~17% (both models)
- **Confidence scores:** 0.87-0.88 average
- **Spatial coverage:** Good quadrant distribution
- **Key gap:** Generic labels vs domain-specific labels

---

## Recommended Model: Qwen2-VL-2B

### Why Qwen2-VL for Spatial Reasoning?

1. **Native spatial understanding:** Qwen2-VL was designed with coordinate-aware outputs
2. **Structured output capability:** Supports JSON with coordinates via prompting
3. **Efficient size:** 2B parameters = faster iteration, lower costs
4. **Already fine-tuned:** pitvqa-qwen2vl-surgical has surgical domain adaptation

### Spatial Capabilities (Built-in)
```python
# Qwen2-VL native pointing format
<ref>object name</ref><box>(x1, y1, x2, y2)</box>

# Or normalized coordinates (0-1000 scale)
"The instrument is at position (234, 567)"
```

---

## Training Strategy: Phase 2 - Spatial Enhancement

### Stage 1: Coordinate-Aware SFT (Recommended First)

**Objective:** Teach model to output precise coordinates for surgical instruments

**Dataset Format:**
```json
{
  "messages": [
    {"role": "user", "content": "<image>\nPoint to the suction device in this surgical frame."},
    {"role": "assistant", "content": "The suction device is located at <point x='67.5' y='45.2'>suction</point> in the lower right quadrant, actively removing blood from the surgical field."}
  ],
  "image": "frame_0123.jpg"
}
```

**Training Config:**
```python
# Qwen2-VL Spatial SFT
base_model = "mmrech/pitvqa-qwen2vl-surgical"  # Start from our trained model
dataset = "mmrech/pitvqa-spatial-sft"  # Need to create this
hardware = "t4-medium"  # 2B model is efficient

lora_config = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

training_args = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
}
```

### Stage 2: Grounded Reasoning (After Stage 1)

**Objective:** Connect spatial locations to surgical actions

**Example Training Pairs:**
```json
{
  "messages": [
    {"role": "user", "content": "<image>\nWhat surgical step is being performed and where is the primary instrument?"},
    {"role": "assistant", "content": "Current step: **Sphenoid sinus clearance**\n\nPrimary instrument: <point x='52.3' y='61.8'>ring curette</point>\n\nThe surgeon is using the ring curette to remove residual tissue from the sphenoid sinus. Secondary: <point x='78.1' y='34.5'>suction</point> maintaining field visibility."}
  ]
}
```

### Stage 3: GRPO with Spatial Rewards (Advanced)

**Objective:** Reinforce correct spatial predictions with programmatic rewards

**Reward Function:**
```python
def spatial_reward(prediction, ground_truth):
    rewards = {}

    # 1. Coordinate accuracy (IoU or distance-based)
    pred_coords = extract_coordinates(prediction)
    gt_coords = ground_truth["coordinates"]
    rewards["spatial"] = compute_distance_reward(pred_coords, gt_coords)

    # 2. Instrument identification
    pred_instrument = extract_instrument(prediction)
    rewards["instrument"] = 1.0 if pred_instrument == gt_coords["label"] else 0.0

    # 3. Quadrant accuracy (coarse spatial)
    rewards["quadrant"] = compute_quadrant_match(pred_coords, gt_coords)

    return 0.5 * rewards["spatial"] + 0.3 * rewards["instrument"] + 0.2 * rewards["quadrant"]
```

---

## Dataset Creation Pipeline

### Step 1: Convert Ground Truth to SAGE/Molmo Format

```python
# Convert PitVQA ground truth CSVs to spatial training format
def create_spatial_dataset():
    samples = []

    for video_id in range(1, 26):
        instruments_df = load_instruments(video_id)

        for frame_time in instruments_df['int_time'].unique():
            frame_data = instruments_df[instruments_df['int_time'] == frame_time]

            # Get instrument positions (quadrant 1-5 -> normalized coordinates)
            instrument = frame_data['str_instrument1'].iloc[0]
            quadrant = frame_data['pos_instrument1'].iloc[0]

            if instrument != 'no_visible_instrument':
                x, y = quadrant_to_coordinates(quadrant)

                sample = {
                    "messages": [
                        {"role": "user", "content": f"<image>\nPoint to the {instrument} in this frame."},
                        {"role": "assistant", "content": f"<point x='{x}' y='{y}'>{instrument}</point>"}
                    ],
                    "image": f"video_{video_id:02d}/frame_{frame_time:06d}.jpg"
                }
                samples.append(sample)

    return samples
```

### Step 2: Quadrant to Coordinate Mapping

```python
# PitVQA uses quadrants 1-5
# 1: Upper-left, 2: Upper-right, 3: Lower-left, 4: Lower-right, 5: Center

QUADRANT_CENTERS = {
    1: (25, 25),   # Upper-left
    2: (75, 25),   # Upper-right
    3: (25, 75),   # Lower-left
    4: (75, 75),   # Lower-right
    5: (50, 50),   # Center
}

def quadrant_to_coordinates(quadrant, jitter=True):
    if quadrant not in QUADRANT_CENTERS:
        return (50, 50)

    x, y = QUADRANT_CENTERS[quadrant]

    if jitter:
        # Add small random offset for diversity
        x += random.uniform(-10, 10)
        y += random.uniform(-10, 10)

    return (round(x, 1), round(y, 1))
```

---

## Recommended Execution Order

### Week 1: Dataset Preparation
1. [ ] Create `mmrech/pitvqa-spatial-sft` dataset
2. [ ] Convert ground truth CSVs to coordinate format
3. [ ] Add Gemini annotations (better taxonomy) as additional training data
4. [ ] Validate dataset format with Qwen2-VL processor

### Week 2: Stage 1 Training
1. [ ] Fine-tune pitvqa-qwen2vl-surgical on spatial dataset
2. [ ] Evaluate coordinate accuracy on held-out test set
3. [ ] Output: `mmrech/pitvqa-qwen2vl-spatial`

### Week 3: Evaluation & Iteration
1. [ ] Test on real surgical frames
2. [ ] Compare against baseline (non-spatial model)
3. [ ] Measure: coordinate MAE, quadrant accuracy, instrument F1

### Week 4 (Optional): GRPO Training
1. [ ] Implement spatial reward functions
2. [ ] Run GRPO on top of spatial SFT model
3. [ ] Output: `mmrech/pitvqa-qwen2vl-spatial-grpo`

---

## Expected Improvements

| Metric | Current (SFT only) | After Spatial Training |
|--------|-------------------|----------------------|
| Coordinate MAE | N/A (no coords) | < 15% of image size |
| Quadrant Accuracy | ~50% (random) | > 80% |
| Instrument F1 | ~0.65 | > 0.80 |
| Spatial Grounding | Generic | Precise pointing |

---

## Alternative: MedGemma for Medical Reasoning

If medical domain knowledge is more important than spatial precision:

```python
# MedGemma approach - medical terminology + spatial
base_model = "mmrech/pitvqa-medgemma-surgical"

# MedGemma excels at:
# - Medical terminology understanding
# - Clinical context awareness
# - Procedure recognition

# Less suited for:
# - Fine-grained coordinate outputs
# - Spatial grounding (not native capability)
```

**Recommendation:** Use Qwen2-VL for spatial reasoning, MedGemma for clinical understanding. Consider ensemble for production.

---

## Quick Start Commands

```bash
# 1. Create spatial dataset
python create_spatial_dataset.py --gt-dir ./ground_truth --output pitvqa-spatial-sft

# 2. Push to HuggingFace
huggingface-cli upload mmrech/pitvqa-spatial-sft ./pitvqa-spatial-sft

# 3. Submit training job (via Claude Code)
# Prompt: "Fine-tune mmrech/pitvqa-qwen2vl-surgical on mmrech/pitvqa-spatial-sft
#          for coordinate-aware surgical instrument localization"
```

---

## References

- [Qwen2-VL Technical Report](https://arxiv.org/abs/2409.12191) - Native spatial understanding
- [SAGE Paper](https://arxiv.org/abs/2312.00763) - Multi-stage video reasoning
- [PitVQA Dataset](https://github.com/surgical-vision/PitVQA) - Pituitary surgery benchmark
