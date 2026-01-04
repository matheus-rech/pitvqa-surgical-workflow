# Multi-Agent Surgical Annotation Pipeline

A consensus-based annotation system using Claude Opus 4.5, Gemini 3 Pro, and GPT-5.2
for creating high-quality surgical video pointing datasets.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              SURGICAL ANNOTATION CONSENSUS SYSTEM               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   CLAUDE    │    │   GEMINI    │    │   GPT-5.2   │        │
│  │  Opus 4.5   │    │   3 Pro     │    │ (Tiebreaker)│        │
│  │  (Primary)  │    │ (Validator) │    │             │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│    Annotate           Validate &         Resolve              │
│    Points &           Classify           Conflicts            │
│    BBoxes             Labels             (if needed)          │
│         │                  │                  │                │
│         └──────────────────┴──────────────────┘                │
│                            │                                   │
│                            ▼                                   │
│                   ┌─────────────────┐                         │
│                   │ CONSENSUS RESULT │                         │
│                   │ Molmo2-VideoPoint│                         │
│                   │ Format Output    │                         │
│                   └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install dependencies
pip install aiohttp pillow datasets huggingface_hub

# Set API keys
export ANTHROPIC_API_KEY="your-claude-key"
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
```

## Usage

### Test Single Frame
```bash
python run_surgical_annotation.py --test
```

### Annotate Full Dataset
```bash
# Process all frames
python run_surgical_annotation.py

# Process limited frames
python run_surgical_annotation.py --max-frames 100 --batch-size 25

# Custom output directory
python run_surgical_annotation.py --output ./my_annotations
```

### Python API
```python
import asyncio
from surgical_annotation_pipeline import SurgicalAnnotationPipeline
from PIL import Image

async def annotate():
    pipeline = SurgicalAnnotationPipeline()

    image = Image.open("surgical_frame.jpg")
    result = await pipeline.annotate_frame(
        image=image,
        video_id="surgery_001",
        frame_id=42,
        timestamp=21.0
    )

    print(f"Found {result.count} objects")
    for point in result.points:
        print(f"  - {point}")

asyncio.run(annotate())
```

## Output Format

### Molmo2-VideoPoint Compatible
```json
{
    "video_id": "pitvqa_001",
    "question": "Point to the pituitary forceps",
    "label": "pituitary_forceps",
    "count": 1,
    "two_fps_timestamps": [21.0],
    "points": [[{"x": 45.2, "y": 62.8}]],
    "category": "instruments",
    "video_source": "pitvqa_surgical"
}
```

## Surgical Categories

### Instruments (16 types)
- pituitary_forceps, suction_cannula, curette, ring_curette
- endoscope, bipolar_cautery, drill, dissector
- scissors, speculum, doppler_probe, micro_hook
- tumor_forceps, irrigation_cannula, cottonoid, hemostatic_agent

### Anatomy (18 types)
- tumor, pituitary_gland, carotid_artery, optic_nerve
- optic_chiasm, sella_turcica, sphenoid_sinus, dura_mater
- diaphragma_sellae, clivus, posterior/anterior_clinoid
- suprasellar_cistern, arachnoid, tuberculum_sellae
- planum_sphenoidale, medial_carotid_wall, cavernous_sinus

### Events (12 types)
- active_bleeding, tumor_removal, cauterization, irrigation
- dissection, drilling, hemostasis, tissue_retraction
- dura_opening, dura_closure, fat_graft_placement, nasoseptal_flap

## Consensus Algorithm

1. **Claude Opus 4.5** provides primary annotations with coordinates
2. **Gemini 3 Pro** validates and may correct/add annotations
3. **Agreement Score** calculated based on:
   - Point proximity (distance < 5% of image)
   - Label matching
4. If agreement < 80%, **GPT-5.2** resolves conflicts
5. Final consensus includes:
   - Agreed annotations (averaged coordinates)
   - High-confidence unique annotations (>90% confidence)

## Files

| File | Description |
|------|-------------|
| `surgical_annotation_pipeline.py` | Core annotation classes |
| `run_surgical_annotation.py` | Dataset processing runner |
| `annotation_config.py` | Configuration management |

## Training Integration

After annotation, use the output for Stage 2 training:

```python
from datasets import load_dataset

# Load the generated annotations
pointing_data = load_dataset("json", data_files="surgical_videopoint_molmo_format.json")

# Fine-tune on video pointing
# ... (see SAGE/Molmo2 training scripts)
```

## Related Resources

- [Molmo2-VideoPoint Dataset](https://huggingface.co/datasets/allenai/Molmo2-VideoPoint)
- [SAGE GitHub](https://github.com/allenai/SAGE)
- [molmo-utils](https://github.com/allenai/molmo-utils)
- [OLMo-core](https://olmo-core.readthedocs.io/)
