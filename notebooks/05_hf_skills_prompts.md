# HF Skills Training Prompts for PitVQA Surgical VLM

Use these prompts with Claude Code after installing HF Skills:

```bash
# Install HF Skills plugin
/plugin install hf-llm-trainer@huggingface-skills

# Configure MCP
claude mcp add --transport http hf-skills https://huggingface.co/mcp?bouquet=skills --header "Authorization: Bearer $HF_TOKEN"
```

---

## Scenario A: MedGemma-4B (Medical Specialist)

**Estimated cost:** ~$15-25 | **Time:** ~2-3 hours | **Risk:** Low

```
Fine-tune google/medgemma-4b-it on mmrech/pitvqa-sage-sft for surgical video understanding.

Configuration:
- Output model: mmrech/pitvqa-medgemma-surgical
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-5
- Use LoRA: True (r=16, alpha=32)
- Hardware: a10g-large
- Vision language model with 'image' and 'messages' columns

Training objective:
Fine-tune MedGemma for pituitary surgery understanding:
- Recognize surgical phases (Nasal, Sellar, Tumor Removal, Closure)
- Identify surgical steps (15 procedures including Septal Dissection, Sphenoidotomy, Dura Opening, etc.)
- Detect surgical instruments (18 tools including Endoscope, Suction, Curette, Bipolar, etc.)

This is a medical imaging task - MedGemma was pre-trained on medical images which should help with surgical frame understanding.
```

---

## Scenario B: Qwen2-VL-2B (Fast Baseline)

**Estimated cost:** ~$5-10 | **Time:** ~1-2 hours | **Risk:** Very Low

```
Fine-tune Qwen/Qwen2-VL-2B-Instruct on mmrech/pitvqa-sage-sft for surgical video understanding.

Configuration:
- Output model: mmrech/pitvqa-qwen2vl-surgical
- Epochs: 3
- Batch size: 8
- Learning rate: 2e-5
- Hardware: t4-medium
- Vision language model with 'image' and 'messages' columns

Training objective:
Fine-tune Qwen2-VL for pituitary surgery understanding:
- Recognize surgical phases (Nasal, Sellar, Tumor Removal, Closure)
- Identify surgical steps (15 procedures)
- Detect surgical instruments (18 tools)

This smaller model serves as a fast baseline to compare against larger models.
```

---

## Scenario C: SAGE-MM-Molmo2-8B (Original Goal - Let's Try!)

**Estimated cost:** ~$30-50 | **Time:** ~3-5 hours | **Risk:** Medium (might fail due to size)

```
Fine-tune allenai/SAGE-MM-Molmo2-8B-SFT_RL on mmrech/pitvqa-sage-sft for surgical video understanding.

Configuration:
- Output model: mmrech/pitvqa-sage-surgical
- Epochs: 3
- Batch size: 2
- Learning rate: 1e-5
- Use LoRA: True (r=8, alpha=16)
- Gradient checkpointing: True
- Hardware: a100-large
- Vision language model with 'image' and 'messages' columns

Training objective:
Fine-tune SAGE/Molmo for pituitary surgery understanding:
- Recognize surgical phases (Nasal, Sellar, Tumor Removal, Closure)
- Identify surgical steps (15 procedures including Septal Dissection, Sphenoidotomy, Dura Opening, etc.)
- Detect surgical instruments (18 tools including Endoscope, Suction, Curette, Bipolar, etc.)

Note: This is an 8B model which pushes HF Skills limits. Using aggressive LoRA settings and A100 hardware to maximize chances of success.
```

---

## Running the Jobs

1. Copy one prompt at a time into Claude Code
2. Claude will validate the dataset and show estimated cost
3. Approve to submit the job
4. Monitor progress with: "How's my training job doing?"
5. Jobs run asynchronously - you can submit all three, then check back

## Expected Output Models

| Scenario | Output Model | Size |
|----------|--------------|------|
| A | `mmrech/pitvqa-medgemma-surgical` | ~4B |
| B | `mmrech/pitvqa-qwen2vl-surgical` | ~2B |
| C | `mmrech/pitvqa-sage-surgical` | ~8B (LoRA adapter) |
| D | `mmrech/pitvqa-molmo-unsloth` | ~7B (LoRA adapter) |

## Comparison Metrics

After training, compare models on:
- Surgical phase accuracy
- Instrument detection F1
- Step identification accuracy
- Inference speed
- Model size / deployment feasibility
