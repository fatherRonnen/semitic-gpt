#!/usr/bin/env python3
"""
Paper 1 — mBERT / XLM-R Comparison on Sentiment Transfer.
Same task as Experiment H: train on Hebrew sentiment only, evaluate cross-lingually.
This provides a direct comparison: does HE→AR transfer also happen in mBERT/XLM-R?

Runs on CPU/GPU — these are small models (110M/278M), no OOM issues.
Requires: transformers, datasets, torch, scikit-learn
"""
import json, os, sys, random
sys.stdout.reconfigure(line_buffering=True)
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.metrics import accuracy_score

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = '/tmp/domain_experiments'  # Same data as Experiment H
RESULTS_DIR = '/tmp/experiments/paper1'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODELS = {
    'mbert': 'bert-base-multilingual-cased',
    'xlmr': 'xlm-roberta-base',
}

TRAIN_FILE = f'{DATA_DIR}/sentiment_train_H1_he_only.jsonl'
EVAL_FILES = {
    'he': f'{DATA_DIR}/sentiment_eval_he.jsonl',
    'ar': f'{DATA_DIR}/sentiment_eval_ar.jsonl',
    'fa': f'{DATA_DIR}/sentiment_eval_fa.jsonl',
    'en': f'{DATA_DIR}/sentiment_eval_en.jsonl',
}

MAX_EVAL = 200
EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================
# DATA
# ============================================================

class SentimentDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Map label strings to ints
        self.label_map = {}
        for s in samples:
            label = s['output'].strip().lower()
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        text = s['instruction']  # The text to classify
        label_str = s['output'].strip().lower()
        label = self.label_map.get(label_str, 0)
        
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_len, 
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_jsonl(path, max_samples=None):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    return {'accuracy': accuracy_score(labels, preds)}

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PAPER 1 — mBERT / XLM-R SENTIMENT COMPARISON")
    print("=" * 60)
    
    # Load Hebrew training data
    print("\nLoading Hebrew sentiment training data...")
    train_samples = load_jsonl(TRAIN_FILE)
    random.shuffle(train_samples)
    print(f"  {len(train_samples)} Hebrew training samples")
    
    # Determine label set
    labels = sorted(set(s['output'].strip().lower() for s in train_samples))
    label_map = {l: i for i, l in enumerate(labels)}
    num_labels = len(labels)
    print(f"  Labels: {labels} ({num_labels} classes)")
    
    all_results = {}
    
    # Also evaluate our 3B model results for reference
    all_results['SemiticGPT-3B'] = {
        'he': 0.185, 'ar': 0.490, 'fa': 0.015, 'en': 0.335,
        'note': 'From Experiment H (Hebrew-only training)'
    }
    
    for model_name, model_id in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'='*60}")
        
        # Load tokenizer and model
        print(f"  Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )
        
        # Create training dataset
        train_dataset = SentimentDataset(train_samples, tokenizer, MAX_LEN)
        
        # Training arguments
        output_dir = f'{RESULTS_DIR}/{model_name}_sentiment'
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            learning_rate=LR,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy='no',
            report_to='none',
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
        )
        
        # Train
        print(f"  Training on Hebrew sentiment ({EPOCHS} epochs)...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        
        # Evaluate on all languages
        print(f"\n  Evaluating cross-lingually...")
        all_results[model_name] = {}
        
        for lang, eval_file in EVAL_FILES.items():
            if not os.path.exists(eval_file):
                print(f"    {lang}: eval file not found, skipping")
                continue
            
            eval_samples = load_jsonl(eval_file, MAX_EVAL)
            
            # Classify each sample
            correct = 0
            total = 0
            model.eval()
            
            with torch.no_grad():
                for s in eval_samples:
                    text = s['instruction']
                    expected_label = s['output'].strip().lower()
                    
                    if expected_label not in label_map:
                        continue
                    
                    encoding = tokenizer(
                        text, truncation=True, max_length=MAX_LEN,
                        padding='max_length', return_tensors='pt'
                    ).to(device)
                    
                    outputs = model(**encoding)
                    pred_idx = outputs.logits.argmax(dim=-1).item()
                    pred_label = labels[pred_idx] if pred_idx < len(labels) else ''
                    
                    if pred_label == expected_label:
                        correct += 1
                    total += 1
            
            acc = correct / total if total > 0 else 0
            print(f"    {lang}: {acc*100:.1f}% ({total} samples)")
            all_results[model_name][lang] = {
                'accuracy': acc,
                'n': total
            }
        
        # Cleanup
        del model, tokenizer, trainer
        torch.cuda.empty_cache()
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'HE':>8} {'AR':>8} {'FA':>8} {'EN':>8} {'AR/HE ratio':>12}")
    print("-" * 68)
    for model_name in ['SemiticGPT-3B', 'mbert', 'xlmr']:
        r = all_results.get(model_name, {})
        he = r.get('he', r.get('he', {}).get('accuracy', 0))
        ar = r.get('ar', r.get('ar', {}).get('accuracy', 0))
        fa = r.get('fa', r.get('fa', {}).get('accuracy', 0))
        en = r.get('en', r.get('en', {}).get('accuracy', 0))
        if isinstance(he, dict): he = he.get('accuracy', 0)
        if isinstance(ar, dict): ar = ar.get('accuracy', 0)
        if isinstance(fa, dict): fa = fa.get('accuracy', 0)
        if isinstance(en, dict): en = en.get('accuracy', 0)
        ratio = ar / he if he > 0 else 0
        print(f"{model_name:<20} {he*100:>7.1f}% {ar*100:>7.1f}% {fa*100:>7.1f}% {en*100:>7.1f}% {ratio:>11.2f}x")
    
    # Save
    print(json.dumps(all_results, indent=2))
    with open(f'{RESULTS_DIR}/mbert_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    os.system(f"aws s3 cp {RESULTS_DIR}/mbert_comparison_results.json s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/paper1_mbert_results.json --quiet")
    print("\nResults uploaded to S3!")
