#!/usr/bin/env python3
"""
Demo Model on Dataset Samples
Visual demonstration of trained VLM on unified dataset samples.

Creates an HTML report showing:
- Sample image
- Question asked
- Ground truth response
- Model prediction
- Correctness indicator
"""

import json
import base64
import re
from io import BytesIO
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# Configuration
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_MODEL = "mmrech/pitvqa-qwen2vl-spatial"  # Current spatial model
UNIFIED_DATASET = "mmrech/pitvqa-unified-vlm"
SAMPLES_PER_TASK = 3
OUTPUT_HTML = "model_demo_report.html"


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def extract_coordinates(text):
    """Extract coordinates from point format."""
    match = re.search(r"<point x='([\d.]+)' y='([\d.]+)'>", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None


def compute_point_distance(pred_text, gt_text):
    """Compute Euclidean distance between predicted and ground truth points."""
    pred_x, pred_y = extract_coordinates(pred_text)
    gt_x, gt_y = extract_coordinates(gt_text)

    if pred_x is None or gt_x is None:
        return None

    return ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5


def check_correctness(task_type, pred_text, gt_text):
    """Check if prediction is correct based on task type."""
    if 'pointing' in task_type:
        dist = compute_point_distance(pred_text, gt_text)
        if dist is None:
            return False, "No point format"
        return dist < 15.0, f"Distance: {dist:.1f}%"
    else:
        # Classification task
        gt_clean = gt_text.lower().strip()
        pred_clean = pred_text.lower().strip()
        is_correct = gt_clean in pred_clean or pred_clean in gt_clean
        return is_correct, "Match" if is_correct else "No match"


def generate_html_report(results, model_name):
    """Generate HTML report from results."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>PitVQA Model Demo Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .sample {{ border: 1px solid #ddd; border-radius: 8px; margin: 20px 0; padding: 20px; background: #f9f9f9; }}
        .sample-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .task-badge {{ padding: 5px 15px; border-radius: 15px; font-size: 12px; font-weight: bold; }}
        .phase_classification {{ background: #3498db; color: white; }}
        .step_classification {{ background: #9b59b6; color: white; }}
        .instrument_pointing {{ background: #2ecc71; color: white; }}
        .anatomy_pointing {{ background: #e67e22; color: white; }}
        .content {{ display: grid; grid-template-columns: 300px 1fr; gap: 20px; }}
        .image-container {{ text-align: center; }}
        .image-container img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; }}
        .qa-section {{ }}
        .qa-item {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
        .question {{ background: #ecf0f1; }}
        .ground-truth {{ background: #d5f5e3; }}
        .prediction {{ background: #fdebd0; }}
        .correct {{ border-left: 4px solid #27ae60; }}
        .incorrect {{ border-left: 4px solid #e74c3c; }}
        .label {{ font-weight: bold; color: #7f8c8d; font-size: 12px; margin-bottom: 5px; }}
        .status {{ margin-top: 10px; padding: 8px 15px; border-radius: 5px; display: inline-block; }}
        .status-correct {{ background: #27ae60; color: white; }}
        .status-incorrect {{ background: #e74c3c; color: white; }}
        .summary {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-top: 30px; }}
        .summary h2 {{ color: white; margin-top: 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .summary-item {{ text-align: center; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 5px; }}
        .summary-value {{ font-size: 24px; font-weight: bold; }}
        .summary-label {{ font-size: 12px; opacity: 0.8; }}
    </style>
</head>
<body>
    <h1>PitVQA Model Demo Report</h1>
    <p><strong>Model:</strong> {model_name}</p>
    <p><strong>Dataset:</strong> {UNIFIED_DATASET}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    # Group by task type
    by_task = {}
    for r in results:
        task = r['task_type']
        if task not in by_task:
            by_task[task] = []
        by_task[task].append(r)

    # Summary stats
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])

    task_stats = {}
    for task, samples in by_task.items():
        task_correct = sum(1 for s in samples if s['is_correct'])
        task_stats[task] = {'correct': task_correct, 'total': len(samples)}

    # Summary section
    html += f"""
    <div class="summary">
        <h2>Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-value">{correct}/{total}</div>
                <div class="summary-label">Overall Correct</div>
            </div>
"""

    for task, stats in task_stats.items():
        pct = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        task_name = task.replace('_', ' ').title()
        html += f"""
            <div class="summary-item">
                <div class="summary-value">{stats['correct']}/{stats['total']}</div>
                <div class="summary-label">{task_name}<br>({pct:.0f}%)</div>
            </div>
"""

    html += """
        </div>
    </div>
"""

    # Samples by task type
    for task_type in ['phase_classification', 'step_classification', 'instrument_pointing', 'anatomy_pointing']:
        if task_type not in by_task:
            continue

        task_name = task_type.replace('_', ' ').title()
        html += f"""
    <h2>{task_name}</h2>
"""

        for i, sample in enumerate(by_task[task_type]):
            status_class = "status-correct" if sample['is_correct'] else "status-incorrect"
            status_text = "Correct" if sample['is_correct'] else "Incorrect"
            correct_class = "correct" if sample['is_correct'] else "incorrect"

            html += f"""
    <div class="sample">
        <div class="sample-header">
            <span class="task-badge {task_type}">{task_name}</span>
            <span class="status {status_class}">{status_text} - {sample['reason']}</span>
        </div>
        <div class="content">
            <div class="image-container">
                <img src="data:image/png;base64,{sample['image_b64']}" alt="Sample image">
            </div>
            <div class="qa-section">
                <div class="qa-item question">
                    <div class="label">QUESTION</div>
                    {sample['question']}
                </div>
                <div class="qa-item ground-truth {correct_class}">
                    <div class="label">GROUND TRUTH</div>
                    {sample['ground_truth']}
                </div>
                <div class="qa-item prediction {correct_class}">
                    <div class="label">MODEL PREDICTION</div>
                    {sample['prediction']}
                </div>
            </div>
        </div>
    </div>
"""

    html += """
</body>
</html>
"""
    return html


def main():
    print("=" * 70)
    print("PITVQA MODEL DEMO - Testing on Dataset Samples")
    print("=" * 70)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter: {ADAPTER_MODEL}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    model.training = False

    # Load dataset
    print(f"\nLoading dataset: {UNIFIED_DATASET}")
    dataset = load_dataset(UNIFIED_DATASET, split="validation")
    print(f"  Total samples: {len(dataset)}")

    # Sample from each task type
    print("\nSampling from each task type...")
    samples_by_task = {
        'phase_classification': [],
        'step_classification': [],
        'instrument_pointing': [],
        'anatomy_pointing': [],
    }

    for sample in dataset:
        task = sample.get('task_type', '')
        if task in samples_by_task and len(samples_by_task[task]) < SAMPLES_PER_TASK:
            samples_by_task[task].append(sample)

    for task, samples in samples_by_task.items():
        print(f"  {task}: {len(samples)} samples")

    # Process samples
    print("\nGenerating predictions...")
    results = []

    for task_type, samples in samples_by_task.items():
        print(f"\n[{task_type}]")

        for sample in samples:
            image = sample['image']
            messages = json.loads(sample['messages']) if isinstance(sample['messages'], str) else sample['messages']

            user_msg = messages[0]['content']
            gt_response = messages[1]['content']

            # Build conversation for inference
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_msg}
                    ]
                }
            ]

            # Process
            text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                )

            # Decode
            input_len = inputs['input_ids'].shape[1]
            pred_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

            # Check correctness
            is_correct, reason = check_correctness(task_type, pred_text, gt_response)

            # Convert image to base64
            if isinstance(image, Image.Image):
                img_b64 = image_to_base64(image)
            else:
                img_b64 = image_to_base64(Image.open(image))

            results.append({
                'task_type': task_type,
                'question': user_msg,
                'ground_truth': gt_response,
                'prediction': pred_text,
                'is_correct': is_correct,
                'reason': reason,
                'image_b64': img_b64,
            })

            status = "OK" if is_correct else "X"
            print(f"  [{status}] Q: {user_msg[:50]}...")
            print(f"       GT: {gt_response[:50]}...")
            print(f"       Pred: {pred_text[:50]}...")

    # Generate HTML report
    print(f"\nGenerating HTML report: {OUTPUT_HTML}")
    html = generate_html_report(results, ADAPTER_MODEL)

    with open(OUTPUT_HTML, 'w') as f:
        f.write(html)

    # Summary
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    print(f"\nOverall: {correct}/{total} ({correct/total*100:.1f}%)")

    for task_type in samples_by_task.keys():
        task_results = [r for r in results if r['task_type'] == task_type]
        task_correct = sum(1 for r in task_results if r['is_correct'])
        if task_results:
            print(f"  {task_type}: {task_correct}/{len(task_results)}")

    print(f"\nReport saved to: {OUTPUT_HTML}")
    print("Open in browser to view visual results.")


if __name__ == "__main__":
    main()
