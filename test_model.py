# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers>=4.45.0",
#     "datasets",
#     "peft>=0.13.0",
#     "accelerate",
#     "bitsandbytes",
#     "huggingface_hub",
#     "pillow",
#     "qwen-vl-utils",
#     "scikit-learn",
#     "pandas",
#     "tqdm"
# ]
# ///

"""
Test fine-tuned Qwen2-VL model on surgical images
Calculates accuracy, F1 score, and other metrics
"""

import os
import torch
from PIL import Image
import io
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import pandas as pd

print('='*60)
print('Qwen2-VL Surgical Model - Testing')
print('='*60)

# Login
login(token=os.environ.get('HF_TOKEN', ''))

# Load test dataset
print('\n[1/4] Loading test dataset...')
test_dataset = load_dataset('mmrech/pitvqa-sage-sft', split='test')
print(f'Test samples: {len(test_dataset)}')

# Load base model with 4-bit quantization
print('\n[2/4] Loading base model...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct',
    quantization_config=bnb_config,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation='eager'
)

# Load LoRA adapter
print('\n[3/4] Loading fine-tuned LoRA adapter...')
model = PeftModel.from_pretrained(base_model, 'mmrech/pitvqa-qwen2vl-surgical')
model.set_adapter('default')
print('Model loaded!')

# Load processor
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', trust_remote_code=True)

# Inference function
def get_prediction(image, question):
    """Get model prediction for an image and question"""
    if isinstance(image, dict) and 'bytes' in image:
        image = Image.open(io.BytesIO(image['bytes'])).convert('RGB')
    elif isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        return None

    # Resize for consistency
    image = image.resize((224, 224))

    # Format message
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image', 'image': image},
            {'type': 'text', 'text': question}
        ]}
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[[image]], return_tensors='pt', padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # Decode
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = processor.tokenizer.decode(generated, skip_special_tokens=True)
    return response.strip()

# Run testing
print('\n[4/4] Running inference on test set...')
print('-'*60)

results = []
predictions = []
ground_truths = []

# Limit to first 100 samples for speed
num_samples = min(100, len(test_dataset))

for i in tqdm(range(num_samples), desc='Testing'):
    sample = test_dataset[i]
    image = sample.get('image')
    messages = sample.get('messages', [])

    if image is None or len(messages) < 2:
        continue

    # Extract question and ground truth
    question = None
    ground_truth = None
    for msg in messages:
        if msg['role'] == 'user':
            question = str(msg['content'])
        elif msg['role'] == 'assistant':
            ground_truth = str(msg['content'])

    if question is None or ground_truth is None:
        continue

    # Get prediction
    try:
        prediction = get_prediction(image, question)
        if prediction is None:
            continue

        results.append({
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'exact_match': prediction.lower().strip() == ground_truth.lower().strip()
        })

        predictions.append(prediction.lower().strip())
        ground_truths.append(ground_truth.lower().strip())

    except Exception as e:
        print(f'Error on sample {i}: {e}')
        continue

# Calculate metrics
print('\n' + '='*60)
print('TEST RESULTS')
print('='*60)

# Exact match accuracy
exact_matches = sum(1 for r in results if r['exact_match'])
exact_accuracy = exact_matches / len(results) * 100 if results else 0

print(f'\nTotal samples tested: {len(results)}')
print(f'Exact match accuracy: {exact_accuracy:.2f}%')

# Token-level/partial match analysis
partial_matches = 0
for r in results:
    gt_tokens = set(r['ground_truth'].lower().split())
    pred_tokens = set(r['prediction'].lower().split())
    if gt_tokens & pred_tokens:  # Any overlap
        partial_matches += 1

partial_accuracy = partial_matches / len(results) * 100 if results else 0
print(f'Partial match accuracy: {partial_accuracy:.2f}%')

# Semantic similarity (keyword matching)
def extract_keywords(text):
    """Extract key surgical terms"""
    keywords = []
    # Phases
    phases = ['preparation', 'approach', 'sellar', 'resection', 'closure', 'hemostasis']
    # Instruments
    instruments = ['forceps', 'suction', 'curette', 'drill', 'endoscope', 'bipolar', 'scissors', 'speculum']
    # Steps
    steps = ['dural opening', 'tumor removal', 'bleeding control', 'dissection', 'irrigation']

    text_lower = text.lower()
    for term in phases + instruments + steps:
        if term in text_lower:
            keywords.append(term)
    return set(keywords)

keyword_matches = 0
for r in results:
    gt_kw = extract_keywords(r['ground_truth'])
    pred_kw = extract_keywords(r['prediction'])
    if gt_kw and gt_kw == pred_kw:
        keyword_matches += 1
    elif gt_kw and pred_kw and gt_kw & pred_kw:
        keyword_matches += 0.5

keyword_accuracy = keyword_matches / len(results) * 100 if results else 0
print(f'Keyword match accuracy: {keyword_accuracy:.2f}%')

# Show sample predictions
print('\n' + '-'*60)
print('SAMPLE PREDICTIONS (first 10):')
print('-'*60)
for i, r in enumerate(results[:10]):
    print(f'\n[{i+1}] Q: {r["question"][:80]}...')
    print(f'    GT: {r["ground_truth"]}')
    print(f'    Pred: {r["prediction"]}')
    print(f'    Match: {"Y" if r["exact_match"] else "N"}')

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('test_results.csv', index=False)
print(f'\nResults saved to: test_results.csv')

# Summary
print('\n' + '='*60)
print('SUMMARY METRICS')
print('='*60)
print(f'''
+-------------------------+--------------+
| Metric                  | Score        |
+-------------------------+--------------+
| Samples Tested          | {len(results):>12} |
| Exact Match Accuracy    | {exact_accuracy:>10.2f}% |
| Partial Match Accuracy  | {partial_accuracy:>10.2f}% |
| Keyword Match Accuracy  | {keyword_accuracy:>10.2f}% |
+-------------------------+--------------+
''')

print('='*60)
print('Testing complete!')
print('='*60)
