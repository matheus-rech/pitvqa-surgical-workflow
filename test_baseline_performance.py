#!/usr/bin/env python3
"""
Baseline Performance Test for PitVQA Spatial Model
Tests current model on multiple task types to establish baseline metrics
"""

import json
import torch
import random
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import re

# Models to test
BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_MODEL = "mmrech/pitvqa-qwen2vl-spatial"

# Datasets
SPATIAL_DATASET = "mmrech/pitvqa-spatial-vlm"
SAGE_DATASET = "mmrech/pitvqa-sage-sft"

def extract_coordinates(text):
    """Extract coordinates from point format."""
    match = re.search(r"<point x='([\d.]+)' y='([\d.]+)'>", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def compute_point_accuracy(pred_x, pred_y, gt_x, gt_y, threshold=15.0):
    """Check if prediction is within threshold % of ground truth."""
    if pred_x is None or gt_x is None:
        return False
    dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5
    return dist <= threshold

def test_model(model, processor, samples, task_type):
    """Test model on a set of samples."""
    results = []

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

        # Generate (inference mode)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Decode
        input_len = inputs['input_ids'].shape[1]
        pred_text = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Build result
        result = {
            'task_type': task_type,
            'question': user_msg[:80],
            'ground_truth': gt_response[:80],
            'prediction': pred_text[:80],
        }

        if task_type in ['instrument_pointing', 'anatomy_pointing', 'dual_instrument']:
            pred_x, pred_y = extract_coordinates(pred_text)
            gt_x, gt_y = extract_coordinates(gt_response)
            result['pred_coords'] = (pred_x, pred_y)
            result['gt_coords'] = (gt_x, gt_y)
            result['correct'] = compute_point_accuracy(pred_x, pred_y, gt_x, gt_y)
            result['has_point_format'] = pred_x is not None
        else:
            # Classification task - check if answer matches
            gt_clean = gt_response.lower().strip()
            pred_clean = pred_text.lower().strip()
            result['correct'] = gt_clean in pred_clean or pred_clean in gt_clean

        results.append(result)
        print(f"  {task_type}: Q='{user_msg[:40]}...' -> '{pred_text[:40]}...' | GT='{gt_response[:40]}...'")

    return results

def main():
    print("=" * 70)
    print("BASELINE PERFORMANCE TEST - PitVQA Spatial Model")
    print("=" * 70)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    # Set to inference mode
    model.training = False

    # Load datasets
    print(f"\nLoading datasets...")
    spatial_ds = load_dataset(SPATIAL_DATASET, split="validation")
    sage_ds = load_dataset(SAGE_DATASET, split="train")

    print(f"  Spatial: {len(spatial_ds)} samples")
    print(f"  SAGE: {len(sage_ds)} samples")

    # Prepare test samples
    print("\nPreparing test samples...")
    test_samples = {
        'instrument_pointing': [],
        'anatomy_pointing': [],
        'dual_instrument': [],
        'phase': [],
        'step': [],
    }

    # From spatial dataset
    for sample in spatial_ds:
        ann_type = sample.get('annotation_type', '')
        if 'instrument_pointing' in ann_type and len(test_samples['instrument_pointing']) < 10:
            test_samples['instrument_pointing'].append(sample)
        elif 'anatomy' in ann_type and len(test_samples['anatomy_pointing']) < 10:
            test_samples['anatomy_pointing'].append(sample)
        elif 'dual' in ann_type and len(test_samples['dual_instrument']) < 5:
            test_samples['dual_instrument'].append(sample)

    # From SAGE dataset (classification)
    for sample in sage_ds:
        qa_type = sample.get('qa_type', '')
        if qa_type == 'phase' and len(test_samples['phase']) < 10:
            test_samples['phase'].append(sample)
        elif qa_type == 'step' and len(test_samples['step']) < 10:
            test_samples['step'].append(sample)

    print("\nTest sample counts:")
    for task, samples in test_samples.items():
        print(f"  {task}: {len(samples)}")

    # Run tests
    print("\n" + "=" * 70)
    print("RUNNING TESTS")
    print("=" * 70)

    all_results = []
    for task_type, samples in test_samples.items():
        if not samples:
            print(f"\n[{task_type}] No samples available")
            continue

        print(f"\n[{task_type}] Testing {len(samples)} samples...")
        results = test_model(model, processor, samples, task_type)
        all_results.extend(results)

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    metrics_by_task = {}
    for task_type in test_samples.keys():
        task_results = [r for r in all_results if r['task_type'] == task_type]
        if not task_results:
            continue

        correct = sum(1 for r in task_results if r.get('correct', False))
        total = len(task_results)
        accuracy = correct / total if total > 0 else 0

        metrics_by_task[task_type] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }

        if task_type in ['instrument_pointing', 'anatomy_pointing', 'dual_instrument']:
            has_format = sum(1 for r in task_results if r.get('has_point_format', False))
            metrics_by_task[task_type]['format_rate'] = has_format / total if total > 0 else 0

    print("\nTask Performance:")
    print("-" * 50)
    for task, metrics in metrics_by_task.items():
        print(f"{task}:")
        print(f"  Accuracy: {metrics['correct']}/{metrics['total']} ({metrics['accuracy']*100:.1f}%)")
        if 'format_rate' in metrics:
            print(f"  Format Rate: {metrics['format_rate']*100:.1f}%")

    # Overall summary
    total_correct = sum(m['correct'] for m in metrics_by_task.values())
    total_samples = sum(m['total'] for m in metrics_by_task.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0

    print(f"\nOverall: {total_correct}/{total_samples} ({overall_accuracy*100:.1f}%)")

    # Save results
    output = {
        'metrics': metrics_by_task,
        'overall_accuracy': overall_accuracy,
        'results': all_results
    }

    with open('baseline_test_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n\nResults saved to baseline_test_results.json")

    print("\n" + "=" * 70)
    print("BASELINE ESTABLISHED")
    print("=" * 70)

if __name__ == "__main__":
    main()
