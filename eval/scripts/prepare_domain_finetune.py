#!/usr/bin/env python3
"""
Domain fine-tuning experiment: Multilingual news/topic classification.
Tests if the 3B model can be specialized for practical classification tasks.

Datasets:
- Hebrew: HebrewSentiment (HebArabNlpProject/HebrewSentiment) - 12.8K sentiment
- Arabic: SANAD/AJGT (Arabic news topics) or asas_dataset
- Farsi: PersiNLU / Persian sentiment
- English: AG News (baseline comparison)

Approach: Format as generative classification (instruction → label output)
"""
import json, os, sys, random, struct
sys.stdout.reconfigure(line_buffering=True)
from datasets import load_dataset
import sentencepiece as spm

HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')
OUTPUT_DIR = '/tmp/domain_finetune'
os.makedirs(OUTPUT_DIR, exist_ok=True)

sp = spm.SentencePieceProcessor('/tmp/eval/multilingual_32k.model')

samples = []

# ============================================================
# 1. Hebrew Sentiment (3-class: positive, negative, neutral)
# ============================================================
print("Loading Hebrew Sentiment...")
try:
    ds = load_dataset('HebArabNlpProject/HebrewSentiment', split='train', token=HF_TOKEN)
    label_map = {0: 'שלילי', 1: 'חיובי', 2: 'ניטרלי'}  # neg, pos, neutral in Hebrew
    label_map_en = {0: 'negative', 1: 'positive', 2: 'neutral'}
    count = 0
    for item in ds:
        text = item.get('text', item.get('sentence', ''))
        label = item.get('label', -1)
        if text and label in label_map:
            # Hebrew instruction
            samples.append({
                'instruction': f'סווג את הרגש של הטקסט הבא (חיובי/שלילי/ניטרלי):\n{text}',
                'output': label_map[label],
                'lang': 'he',
                'task': 'sentiment',
            })
            count += 1
    print(f"  ✅ Hebrew Sentiment: {count} samples")
except Exception as e:
    print(f"  ❌ Hebrew Sentiment: {e}")

# ============================================================
# 2. Arabic Sentiment (AJGT - Arabic Jordanian General Tweets)
# ============================================================
print("Loading Arabic Sentiment...")
try:
    ds = load_dataset('ajgt_twitter_ar', split='train', token=HF_TOKEN)
    count = 0
    for item in ds:
        text = item.get('text', '')
        label = item.get('label', -1)
        if text and label in [0, 1]:
            label_text = 'إيجابي' if label == 1 else 'سلبي'
            samples.append({
                'instruction': f'صنّف مشاعر النص التالي (إيجابي/سلبي):\n{text}',
                'output': label_text,
                'lang': 'ar',
                'task': 'sentiment',
            })
            count += 1
    print(f"  ✅ Arabic AJGT: {count} samples")
except Exception as e:
    print(f"  ❌ Arabic AJGT: {e}")
    # Fallback: try sanad
    try:
        ds = load_dataset('arabic_billion_words', split='train', streaming=True, token=HF_TOKEN)
        print("  Trying alternative Arabic dataset...")
    except:
        pass

# ============================================================
# 3. Persian Sentiment (PersiNLU or DeepSentiPers)
# ============================================================
print("Loading Persian Sentiment...")
try:
    ds = load_dataset('persiNLU', 'sentiment', split='train', token=HF_TOKEN)
    count = 0
    for item in ds:
        text = item.get('text', item.get('review', ''))
        label = item.get('label', -1)
        if text and label >= 0:
            labels_fa = {0: 'منفی', 1: 'خنثی', 2: 'مثبت'}
            if label in labels_fa:
                samples.append({
                    'instruction': f'احساسات متن زیر را طبقه‌بندی کنید (مثبت/منفی/خنثی):\n{text}',
                    'output': labels_fa[label],
                    'lang': 'fa',
                    'task': 'sentiment',
                })
                count += 1
    print(f"  ✅ Persian Sentiment: {count} samples")
except Exception as e:
    print(f"  ❌ Persian Sentiment: {e}")
    # Fallback: DeepSentiPers
    try:
        ds = load_dataset('sepidmnorozy/Persian_sentiment', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            text = item.get('text', '')
            label = item.get('label', -1)
            if text and label in [0, 1, 2]:
                labels_fa = {0: 'منفی', 1: 'خنثی', 2: 'مثبت'}
                samples.append({
                    'instruction': f'احساسات متن زیر را طبقه‌بندی کنید (مثبت/منفی/خنثی):\n{text}',
                    'output': labels_fa[label],
                    'lang': 'fa',
                    'task': 'sentiment',
                })
                count += 1
                if count >= 5000:
                    break
        print(f"  ✅ Persian Sentiment (fallback): {count} samples")
    except Exception as e2:
        print(f"  ❌ Persian Sentiment fallback: {e2}")

# ============================================================
# 4. English AG News (topic classification, 4 classes)
# ============================================================
print("Loading English AG News...")
try:
    ds = load_dataset('ag_news', split='train', token=HF_TOKEN)
    labels_en = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Technology'}
    count = 0
    for item in ds:
        text = item.get('text', '')
        label = item.get('label', -1)
        if text and label in labels_en and count < 5000:
            samples.append({
                'instruction': f'Classify the topic of this news article (World/Sports/Business/Technology):\n{text[:500]}',
                'output': labels_en[label],
                'lang': 'en',
                'task': 'topic',
            })
            count += 1
    print(f"  ✅ English AG News: {count} samples")
except Exception as e:
    print(f"  ❌ English AG News: {e}")

# ============================================================
# Summary and save
# ============================================================
print(f"\n=== Total domain samples: {len(samples)} ===")
lang_counts = {}
for s in samples:
    lang_counts[s['lang']] = lang_counts.get(s['lang'], 0) + 1
for lang, count in sorted(lang_counts.items()):
    print(f"  {lang}: {count}")

random.shuffle(samples)

# Save raw
with open(f'{OUTPUT_DIR}/domain_train.jsonl', 'w') as f:
    for s in samples:
        json.dump(s, f, ensure_ascii=False)
        f.write('\n')

# Tokenize for training
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"
EOS_ID = 3

n_val = int(len(samples) * 0.1)
val_samples = samples[:n_val]
train_samples = samples[n_val:]

def tokenize_save(samples, path):
    all_ids = []
    for s in samples:
        text = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}{s['output']}"
        ids = sp.encode(text)
        ids.append(EOS_ID)
        all_ids.extend(ids)
    with open(path, 'wb') as f:
        for tid in all_ids:
            f.write(struct.pack('<H', tid))
    return len(all_ids)

os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)
train_tok = tokenize_save(train_samples, f'{OUTPUT_DIR}/data/train_sft.bin')
val_tok = tokenize_save(val_samples, f'{OUTPUT_DIR}/data/val_sft.bin')

print(f"\nTokenized: {len(train_samples)} train ({train_tok} tokens), {len(val_samples)} val ({val_tok} tokens)")
print(f"Saved to {OUTPUT_DIR}/")

# Upload to S3
os.system(f"aws s3 sync {OUTPUT_DIR}/ s3://autoresearch-dashboard-196766918360/multilingual-7b/domain_finetune/ --quiet")
print("Uploaded to S3")
