#!/usr/bin/env python3
"""
SFT Data Preparation v2 for Multilingual 3B GPT

Data sources:
1. HebrewGPT SFT v3 — 27K Hebrew instruction samples from our prior work (S3)
2. HebrewGPT individual datasets — alpaca_hebrew, chat, dolly, QA, summarization, etc. (S3)
3. Aya Dataset — human-annotated instructions (en, ar, fa)
4. arbml/alpaca_arabic — 52K Arabic alpaca-style instructions
5. FreedomIntelligence/alpaca-gpt4-arabic — 50K Arabic GPT-4 instructions
6. tatsu-lab/alpaca — 52K English instructions
7. databricks/dolly-15k — diverse English instructions

Output: tokenized binary data for SFT training.
"""

import os, sys, json, argparse, random
from collections import defaultdict
sys.stdout.reconfigure(line_buffering=True)

datasets_mod = None
spm = None
np = None

def ensure_imports():
    global datasets_mod, spm, np
    if datasets_mod is None:
        import datasets as _ds
        import sentencepiece as _spm
        import numpy as _np
        datasets_mod = _ds
        spm = _spm
        np = _np

# Chat format
USER_PREFIX = "### User:\n"
ASSISTANT_PREFIX = "### Assistant:\n"
TURN_END = "\n\n"

def format_instruction(instruction, response, input_text=None):
    if input_text and input_text.strip():
        user_text = f"{instruction}\n\n{input_text}"
    else:
        user_text = instruction
    return f"{USER_PREFIX}{user_text}{TURN_END}{ASSISTANT_PREFIX}{response}{TURN_END}"


def load_aya_multilingual(max_per_lang=5000):
    """Load Aya Dataset using correct language_code field."""
    ensure_imports()
    print("Loading Aya Dataset (using language_code field)...")
    
    code_map = {
        'eng': 'en',
        'arb': 'ar',    # Standard Arabic
        'ary': 'ar',    # Moroccan Arabic  
        'arz': 'ar',    # Egyptian Arabic
        'ars': 'ar',    # Najdi Arabic
        'apc': 'ar',    # South Levantine Arabic
        'pes': 'fa',    # Iranian Persian
    }
    
    ds = datasets_mod.load_dataset("CohereForAI/aya_dataset", split="train")
    
    # Group by our target language
    by_lang = defaultdict(list)
    for s in ds:
        code = s['language_code']
        target = code_map.get(code)
        if target:
            by_lang[target].append(s)
    
    all_samples = []
    for lang, samples in by_lang.items():
        random.shuffle(samples)
        selected = samples[:max_per_lang]
        for s in selected:
            all_samples.append({
                'text': format_instruction(s['inputs'], s['targets']),
                'lang': lang,
                'source': 'aya',
            })
        print(f"  Aya [{lang}]: {len(selected)} samples (from {len(samples)} available)")
    
    return all_samples


def load_arabic_alpaca(max_samples=5000):
    """Load arbml/alpaca_arabic — high-quality Arabic instructions."""
    ensure_imports()
    print("Loading arbml/alpaca_arabic...")
    
    try:
        ds = datasets_mod.load_dataset("arbml/alpaca_arabic", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        samples = []
        skipped = 0
        for i in indices:
            s = ds[i]
            instr = s.get('instruction', '').strip()
            out = s.get('output', '').strip()
            inp = s.get('input', '').strip()
            if not instr or not out:
                skipped += 1
                continue
            samples.append({
                'text': format_instruction(instr, out, inp),
                'lang': 'ar',
                'source': 'alpaca_arabic',
            })
        print(f"  alpaca_arabic: {len(samples)} samples (skipped {skipped} empty)")
        return samples
    except Exception as e:
        print(f"  Warning: Could not load alpaca_arabic: {e}")
        return []


def load_arabic_gpt4(max_samples=5000):
    """Load FreedomIntelligence/alpaca-gpt4-arabic — GPT-4 generated Arabic."""
    ensure_imports()
    print("Loading FreedomIntelligence/alpaca-gpt4-arabic...")
    
    try:
        ds = datasets_mod.load_dataset("FreedomIntelligence/alpaca-gpt4-arabic", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        samples = []
        skipped = 0
        for i in indices:
            s = ds[i]
            convs = s.get('conversations', [])
            if len(convs) < 2:
                skipped += 1
                continue
            # Find human/gpt pairs
            human = None
            for c in convs:
                if c['from'] == 'human':
                    human = c['value'].strip()
                elif c['from'] == 'gpt' and human:
                    gpt = c['value'].strip()
                    if human and gpt:
                        samples.append({
                            'text': format_instruction(human, gpt),
                            'lang': 'ar',
                            'source': 'alpaca_gpt4_arabic',
                        })
                    human = None
        print(f"  alpaca-gpt4-arabic: {len(samples)} samples (skipped {skipped} empty)")
        return samples[:max_samples]
    except Exception as e:
        print(f"  Warning: Could not load alpaca-gpt4-arabic: {e}")
        return []


def load_english_alpaca(max_samples=5000):
    """Load tatsu-lab/alpaca for English instruction data."""
    ensure_imports()
    print("Loading tatsu-lab/alpaca (English)...")
    
    try:
        ds = datasets_mod.load_dataset("tatsu-lab/alpaca", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        samples = []
        for i in indices:
            s = ds[i]
            instr = s.get('instruction', '').strip()
            out = s.get('output', '').strip()
            inp = s.get('input', '').strip()
            if not instr or not out:
                continue
            samples.append({
                'text': format_instruction(instr, out, inp),
                'lang': 'en',
                'source': 'alpaca_en',
            })
        print(f"  alpaca_en: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Warning: Could not load alpaca: {e}")
        return []


def load_hebrew_sft(data_dir, max_samples=10000):
    """Load Hebrew instruction data from S3 (HebrewGPT project)."""
    import json as _json
    print(f"Loading Hebrew SFT data from {data_dir}...")
    
    all_samples = []
    
    # Load all JSONL files
    for fname in os.listdir(data_dir):
        if not fname.endswith('.jsonl'):
            continue
        filepath = os.path.join(data_dir, fname)
        count = 0
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = _json.loads(line)
                except:
                    continue
                
                # Handle different formats
                if 'messages' in d:
                    # Chat format
                    msgs = d['messages']
                    if len(msgs) >= 2:
                        user_msg = msgs[0].get('content', '').strip()
                        asst_msg = msgs[1].get('content', '').strip()
                        if user_msg and asst_msg:
                            all_samples.append({
                                'text': format_instruction(user_msg, asst_msg),
                                'lang': 'he',
                                'source': f'hebrew_{fname.replace(".jsonl", "")}',
                            })
                            count += 1
                elif 'instruction' in d:
                    instr = d.get('instruction', '').strip()
                    inp = d.get('input', '').strip()
                    out = d.get('output', d.get('response', '')).strip()
                    if instr and out:
                        all_samples.append({
                            'text': format_instruction(instr, out, inp),
                            'lang': 'he',
                            'source': f'hebrew_{fname.replace(".jsonl", "")}',
                        })
                        count += 1
        
        if count > 0:
            print(f"  {fname}: {count} samples")
    
    # Shuffle and cap
    random.shuffle(all_samples)
    if max_samples and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]
    
    print(f"  Total Hebrew: {len(all_samples)} samples (capped from {len(all_samples)} if needed)")
    return all_samples


def load_dolly(max_samples=3000):
    """Load databricks/dolly-15k for diverse English instructions."""
    ensure_imports()
    print("Loading databricks/databricks-dolly-15k (English)...")
    
    try:
        ds = datasets_mod.load_dataset("databricks/databricks-dolly-15k", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        indices = indices[:max_samples]
        
        samples = []
        for i in indices:
            s = ds[i]
            instr = s.get('instruction', '').strip()
            resp = s.get('response', '').strip()
            ctx = s.get('context', '').strip()
            if not instr or not resp:
                continue
            samples.append({
                'text': format_instruction(instr, resp, ctx),
                'lang': 'en',
                'source': 'dolly',
            })
        print(f"  dolly: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"  Warning: Could not load dolly: {e}")
        return []


def tokenize_and_save(samples, tokenizer_path, output_dir, val_ratio=0.05):
    """Tokenize samples and save as binary files."""
    ensure_imports()
    
    sp = spm.SentencePieceProcessor(tokenizer_path)
    os.makedirs(output_dir, exist_ok=True)
    
    random.shuffle(samples)
    
    n_val = max(int(len(samples) * val_ratio), 100)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    # Stats
    source_counts = defaultdict(int)
    lang_counts = defaultdict(int)
    for s in samples:
        source_counts[s['source']] += 1
        lang_counts[s['lang']] += 1
    
    print(f"\n{'='*60}")
    print(f"DATASET VALIDATION")
    print(f"{'='*60}")
    print(f"Total samples: {len(samples)} ({len(train_samples)} train, {n_val} val)")
    print(f"\nBy source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} ({cnt*100/len(samples):.1f}%)")
    print(f"\nBy language:")
    for lang, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {cnt} ({cnt*100/len(samples):.1f}%)")
    
    # Validate samples
    print(f"\n--- Sample validation ---")
    empty_count = 0
    short_count = 0
    for s in samples:
        text = s['text']
        if not text.strip():
            empty_count += 1
        elif len(text) < 20:
            short_count += 1
    print(f"  Empty samples: {empty_count}")
    print(f"  Very short (<20 chars): {short_count}")
    
    # Show random samples per language
    print(f"\n--- Random samples per language ---")
    by_lang = defaultdict(list)
    for s in samples:
        by_lang[s['lang']].append(s)
    for lang in sorted(by_lang.keys()):
        s = random.choice(by_lang[lang])
        text = s['text'][:200].replace('\n', '\\n')
        print(f"\n  [{lang}] ({s['source']}): {text}...")
    
    # Tokenize
    print(f"\n--- Tokenization ---")
    total_tokens = 0
    for split_name, split_data in [('train', train_samples), ('val', val_samples)]:
        all_ids = []
        for s in split_data:
            ids = sp.encode(s['text'])
            ids.append(sp.eos_id())
            all_ids.extend(ids)
        
        arr = np.array(all_ids, dtype=np.uint16)
        filepath = os.path.join(output_dir, f'{split_name}_sft.bin')
        arr.tofile(filepath)
        total_tokens += len(arr)
        print(f"  {split_name}: {len(arr):,} tokens → {filepath}")
    
    # Token budget per language
    print(f"\n--- Token budget per language ---")
    for lang in sorted(by_lang.keys()):
        lang_tokens = 0
        for s in by_lang[lang]:
            lang_tokens += len(sp.encode(s['text'])) + 1
        print(f"  {lang}: {lang_tokens:,} tokens ({lang_tokens*100/total_tokens:.1f}%)")
    
    # Save metadata
    metadata = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': n_val,
        'total_tokens': total_tokens,
        'source_counts': dict(source_counts),
        'lang_counts': dict(lang_counts),
        'format': 'USER_PREFIX + instruction + ASSISTANT_PREFIX + response',
        'tokenizer': os.path.basename(tokenizer_path),
        'data_sources': [
            'CohereForAI/aya_dataset (en, ar dialects, fa)',
            'arbml/alpaca_arabic',
            'FreedomIntelligence/alpaca-gpt4-arabic',
            'tatsu-lab/alpaca (en)',
            'databricks/databricks-dolly-15k (en)',
        ],
        'notes': 'Hebrew data from HebrewGPT project (S3). Arabic from Aya + alpaca. Farsi from Aya. English from Aya + alpaca + dolly.',
    }
    with open(os.path.join(output_dir, 'sft_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata saved to {output_dir}/sft_metadata.json")
    
    print(f"\n{'='*60}")
    print(f"✅ SFT DATA PREPARATION COMPLETE")
    print(f"Total: {len(samples)} samples, {total_tokens:,} tokens")
    print(f"Languages: {dict(lang_counts)}")
    if 'he' not in dict(lang_counts):
        print(f"⚠️  No Hebrew instruction data — Hebrew relies on cross-lingual transfer")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', default='/tmp/sft_data_v2')
    parser.add_argument('--aya-per-lang', type=int, default=5000)
    parser.add_argument('--arabic-alpaca', type=int, default=5000)
    parser.add_argument('--arabic-gpt4', type=int, default=5000)
    parser.add_argument('--english-alpaca', type=int, default=5000)
    parser.add_argument('--dolly', type=int, default=3000)
    parser.add_argument('--hebrew-dir', default='/tmp/hebrew_sft', help='Dir with Hebrew JSONL files from S3')
    parser.add_argument('--hebrew-max', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Preparing multilingual SFT data v2")
    print(f"Output: {args.output}\n")
    
    all_samples = []
    
    # 1. Hebrew instruction data (from HebrewGPT project)
    if os.path.isdir(args.hebrew_dir):
        all_samples.extend(load_hebrew_sft(args.hebrew_dir, args.hebrew_max))
    else:
        print(f"⚠️  Hebrew dir not found: {args.hebrew_dir}")
    
    # 2. Aya (en + ar + fa)
    all_samples.extend(load_aya_multilingual(args.aya_per_lang))
    
    # 3. Arabic alpaca
    all_samples.extend(load_arabic_alpaca(args.arabic_alpaca))
    
    # 4. Arabic GPT-4 alpaca
    all_samples.extend(load_arabic_gpt4(args.arabic_gpt4))
    
    # 5. English alpaca
    all_samples.extend(load_english_alpaca(args.english_alpaca))
    
    # 6. English dolly
    all_samples.extend(load_dolly(args.dolly))
    
    if not all_samples:
        print("ERROR: No samples collected!")
        sys.exit(1)
    
    tokenize_and_save(all_samples, args.tokenizer, args.output)


if __name__ == '__main__':
    main()
