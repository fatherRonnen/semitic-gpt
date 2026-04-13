#!/usr/bin/env python3
"""
Prepare clean v4 datasets with ALL bugs fixed:
1. Hebrew labels: use correct mapping (0=pos, 1=neg)
2. Arabic: shuffle before split
3. Strict train/eval separation (no leakage)
4. Translation: Helsinki-NLP/opus-100
"""
import json, os, sys, random
random.seed(42)
sys.stdout.reconfigure(line_buffering=True)
from datasets import load_dataset
from collections import Counter

OUTPUT_DIR = '/tmp/v4_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EVAL_SIZE = 500

# ============================================================
# SENTIMENT DATA
# ============================================================
print("=" * 60)
print("V4 DATA PREPARATION")
print("=" * 60)

# --- HEBREW ---
print("\n[1/4] Hebrew sentiment (mteb/hebrew_sentiment_analysis)")
print("  Dataset labels: 0=pos, 1=neg, 2=off-topic")
ds_train = load_dataset('mteb/hebrew_sentiment_analysis', split='train')
ds_test = load_dataset('mteb/hebrew_sentiment_analysis', split='test')
all_he = list(ds_train) + list(ds_test)
he_data = []
for item in all_he:
    if item['label'] == 2:
        continue
    he_data.append({
        'text': item['text'],
        'label_name': 'חיובי' if item['label'] == 0 else 'שלילי',
    })
random.shuffle(he_data)
he_labels = Counter(x['label_name'] for x in he_data)
print(f"  Total: {len(he_data)}, Labels: {dict(he_labels)}")

he_pos = [x for x in he_data if x['label_name'] == 'חיובי']
he_neg = [x for x in he_data if x['label_name'] == 'שלילי']
eval_per_class = EVAL_SIZE // 2
he_eval = he_pos[:eval_per_class] + he_neg[:eval_per_class]
he_train_pool = he_pos[eval_per_class:] + he_neg[eval_per_class:]
random.shuffle(he_eval)
random.shuffle(he_train_pool)
print(f"  Eval: {len(he_eval)} (balanced {eval_per_class}/{eval_per_class})")
print(f"  Train pool: {len(he_train_pool)}")

# --- ARABIC ---
print("\n[2/4] Arabic sentiment (arbml/Sentiment_Analysis_Tweets)")
print("  Dataset is SORTED by label - shuffling first!")
ds = load_dataset('arbml/Sentiment_Analysis_Tweets', split='train')
ar_data = [{'text': item['Tweet'], 'label_name': 'سلبي' if item['label'] == 0 else 'إيجابي'} for item in ds]
random.shuffle(ar_data)
ar_labels = Counter(x['label_name'] for x in ar_data)
print(f"  Total: {len(ar_data)}, Labels: {dict(ar_labels)}")

ar_pos = [x for x in ar_data if x['label_name'] == 'إيجابي']
ar_neg = [x for x in ar_data if x['label_name'] == 'سلبي']
ar_eval = ar_pos[:eval_per_class] + ar_neg[:eval_per_class]
ar_train_pool = ar_pos[eval_per_class:] + ar_neg[eval_per_class:]
random.shuffle(ar_eval)
random.shuffle(ar_train_pool)
print(f"  Eval: {len(ar_eval)} (balanced {eval_per_class}/{eval_per_class})")
print(f"  Train pool: {len(ar_train_pool)}")

# --- FARSI ---
print("\n[3/4] Farsi sentiment (sepidmnorozy/Persian_sentiment)")
ds = load_dataset('sepidmnorozy/Persian_sentiment', split='train')
fa_data = [{'text': item['text'], 'label_name': 'منفی' if item['label'] == 0 else 'مثبت'} for item in ds]
random.shuffle(fa_data)
fa_labels = Counter(x['label_name'] for x in fa_data)
print(f"  Total: {len(fa_data)}, Labels: {dict(fa_labels)}")

fa_pos = [x for x in fa_data if x['label_name'] == 'مثبت']
fa_neg = [x for x in fa_data if x['label_name'] == 'منفی']
fa_eval = fa_pos[:eval_per_class] + fa_neg[:eval_per_class]
fa_train_pool = fa_pos[eval_per_class:] + fa_neg[eval_per_class:]
random.shuffle(fa_eval)
random.shuffle(fa_train_pool)
print(f"  Eval: {len(fa_eval)} (balanced {eval_per_class}/{eval_per_class})")
print(f"  Train pool: {len(fa_train_pool)}")

# --- ENGLISH ---
print("\n[4/4] English sentiment (SST-2)")
ds = load_dataset('stanfordnlp/sst2', split='train')
en_data = [{'text': item['sentence'], 'label_name': 'negative' if item['label'] == 0 else 'positive'} for item in ds]
random.shuffle(en_data)
en_labels = Counter(x['label_name'] for x in en_data)
print(f"  Total: {len(en_data)}, Labels: {dict(en_labels)}")

en_pos = [x for x in en_data if x['label_name'] == 'positive']
en_neg = [x for x in en_data if x['label_name'] == 'negative']
en_eval = en_pos[:eval_per_class] + en_neg[:eval_per_class]
en_train_pool = en_pos[eval_per_class:] + en_neg[eval_per_class:]
random.shuffle(en_eval)
random.shuffle(en_train_pool)
print(f"  Eval: {len(en_eval)} (balanced {eval_per_class}/{eval_per_class})")
print(f"  Train pool: {len(en_train_pool)}")

# ============================================================
# FORMAT AND SAVE
# ============================================================
PROMPTS = {
    'he': 'סווג את הרגש של הטקסט הבא (חיובי/שלילי):\n',
    'ar': 'صنّف مشاعر النص التالي (إيجابي/سلبي):\n',
    'fa': 'احساسات متن زیر را طبقهبندی کنید (مثبت/منفی):\n',
    'en': 'Classify the sentiment (positive/negative):\n',
}

def format_sample(item, lang):
    return {
        'instruction': PROMPTS[lang] + item['text'][:500],
        'output': item['label_name'],
    }

all_eval = {'he': he_eval, 'ar': ar_eval, 'fa': fa_eval, 'en': en_eval}
all_train = {'he': he_train_pool, 'ar': ar_train_pool, 'fa': fa_train_pool, 'en': en_train_pool}

print("\n" + "=" * 60)
print("SAVING EVAL FILES (strict separation from train)")
print("=" * 60)

for lang, data in all_eval.items():
    formatted = [format_sample(item, lang) for item in data]
    path = f'{OUTPUT_DIR}/sentiment_eval_{lang}.jsonl'
    with open(path, 'w') as f:
        for s in formatted:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    labels = Counter(s['output'] for s in formatted)
    print(f"  {lang}: {len(formatted)} samples, labels={dict(labels)}")

print("\n" + "=" * 60)
print("SAVING TRAIN FILES")
print("=" * 60)

TRAIN_PER_CLASS = 2500
for lang, data in all_train.items():
    pos = [x for x in data if x['label_name'] in ['חיובי', 'إيجابي', 'مثبت', 'positive']]
    neg = [x for x in data if x['label_name'] in ['שלילי', 'سلبي', 'منفی', 'negative']]
    cap = min(TRAIN_PER_CLASS, len(pos), len(neg))
    balanced = pos[:cap] + neg[:cap]
    random.shuffle(balanced)
    formatted = [format_sample(item, lang) for item in balanced]
    path = f'{OUTPUT_DIR}/sentiment_train_{lang}.jsonl'
    with open(path, 'w') as f:
        for s in formatted:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    labels = Counter(s['output'] for s in formatted)
    print(f"  {lang}: {len(formatted)} samples, labels={dict(labels)}")

# ============================================================
# TRANSLATION DATA (Helsinki-NLP/opus-100)
# ============================================================
print("\n" + "=" * 60)
print("TRANSLATION DATA (Helsinki-NLP/opus-100)")
print("=" * 60)

TRANS_TRAIN = 3000
TRANS_EVAL = 200

for pair, lang1, lang2, target_name in [
    ('ar-en', 'ar', 'en', 'Arabic'),
    ('en-he', 'en', 'he', 'Hebrew'),
    ('en-fa', 'en', 'fa', 'Farsi'),
]:
    print(f"\n  Loading {pair}...")
    ds_test = load_dataset('Helsinki-NLP/opus-100', pair, split='test')
    ds_train = load_dataset('Helsinki-NLP/opus-100', pair, split=f'train[:{TRANS_TRAIN + TRANS_EVAL + 500}]')
    
    def make_pairs(dataset, max_n):
        pairs = []
        for item in dataset:
            t = item['translation']
            s1 = t[lang1].strip()
            s2 = t[lang2].strip()
            if s1 and s2 and 5 < len(s1) < 300 and 5 < len(s2) < 300:
                pairs.append((s1, s2))
            if len(pairs) >= max_n:
                break
        return pairs
    
    eval_pairs = make_pairs(ds_test, TRANS_EVAL)
    train_pairs = make_pairs(ds_train, TRANS_TRAIN)
    
    TRANS_PROMPTS = {
        'he': ('Translate to Hebrew: ', 'תרגם לאנגלית: '),
        'ar': ('Translate to Arabic: ', 'Translate to English: '),
        'fa': ('Translate to Farsi: ', 'Translate to English: '),
    }
    
    non_en = lang1 if lang1 != 'en' else lang2
    to_target, to_en = TRANS_PROMPTS[non_en]
    
    train_samples = []
    for s1, s2 in train_pairs:
        en_text = s1 if lang1 == 'en' else s2
        other_text = s2 if lang1 == 'en' else s1
        train_samples.append({'instruction': to_target + en_text, 'output': other_text})
        train_samples.append({'instruction': to_en + other_text, 'output': en_text})
    
    random.shuffle(train_samples)
    path = f'{OUTPUT_DIR}/translation_train_{non_en}.jsonl'
    with open(path, 'w') as f:
        for s in train_samples:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    eval_samples = []
    for s1, s2 in eval_pairs:
        en_text = s1 if lang1 == 'en' else s2
        other_text = s2 if lang1 == 'en' else s1
        eval_samples.append({'instruction': to_target + en_text, 'output': other_text})
    
    path = f'{OUTPUT_DIR}/translation_eval_{non_en}.jsonl'
    with open(path, 'w') as f:
        for s in eval_samples:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"  {pair}: {len(train_samples)} train, {len(eval_samples)} eval")

# ============================================================
# VALIDATION SAMPLES
# ============================================================
print("\n" + "=" * 60)
print("VALIDATION SAMPLES (check these!)")
print("=" * 60)

for lang in ['he', 'ar', 'fa', 'en']:
    print(f"\n--- {lang} sentiment eval (first 3) ---")
    with open(f'{OUTPUT_DIR}/sentiment_eval_{lang}.jsonl') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            s = json.loads(line)
            text = s['instruction'].split('\n', 1)[1][:80] if '\n' in s['instruction'] else '?'
            print(f"  [{s['output']}] {text}")

for lang in ['he', 'ar', 'fa']:
    print(f"\n--- {lang} translation eval (first 3) ---")
    path = f'{OUTPUT_DIR}/translation_eval_{lang}.jsonl'
    if os.path.exists(path):
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= 3: break
                s = json.loads(line)
                print(f"  IN:  {s['instruction'][:80]}")
                print(f"  OUT: {s['output'][:80]}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("DATA MANIFEST")
print("=" * 60)
import glob
for path in sorted(glob.glob(f'{OUTPUT_DIR}/*.jsonl')):
    name = os.path.basename(path)
    with open(path) as f:
        n = sum(1 for _ in f)
    size = os.path.getsize(path)
    print(f"  {name}: {n} samples ({size:,} bytes)")

print("\nDone! Files in:", OUTPUT_DIR)
