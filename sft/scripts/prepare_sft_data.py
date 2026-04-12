#!/usr/bin/env python3
"""
SFT Data Preparation for Multilingual 3B GPT

Downloads and processes multilingual instruction data from:
1. Aya Dataset (CohereForAI) — human-annotated, 65 languages
2. Bactrian-X — multilingual alpaca-style instructions
3. Custom templates for translation pairs

Outputs tokenized binary data for SFT training.

Usage:
    pip install datasets sentencepiece
    python prepare_sft_data.py --tokenizer /path/to/multilingual_32k.model --output /path/to/sft_data/
"""

import os, sys, json, argparse, random
from collections import defaultdict

# Lazy imports
datasets = None
spm = None
np = None


def ensure_imports():
    global datasets, spm, np
    if datasets is None:
        import datasets as _ds
        import sentencepiece as _spm
        import numpy as _np
        datasets = _ds
        spm = _spm
        np = _np


# ============ CHAT FORMAT ============
# Simple format that works well for small models:
# <|user|>\n{instruction}\n<|assistant|>\n{response}\n<|end|>
# We use special tokens within the existing vocab via reserved IDs
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
USER_PREFIX = "### User:\n"
ASSISTANT_PREFIX = "### Assistant:\n"
TURN_END = "\n\n"

def format_instruction(instruction, response, input_text=None):
    """Format a single instruction-response pair."""
    if input_text and input_text.strip():
        user_text = f"{instruction}\n\n{input_text}"
    else:
        user_text = instruction
    return f"{USER_PREFIX}{user_text}{TURN_END}{ASSISTANT_PREFIX}{response}{TURN_END}"


# ============ DATA SOURCES ============

def load_aya_dataset(langs, max_per_lang=5000):
    """Load Aya Dataset for specified languages.
    Aya has human-annotated instruction data in 65 languages."""
    ensure_imports()
    print("Loading Aya Dataset...")

    lang_map = {
        'en': 'English',
        'he': 'Hebrew',
        'ar': 'Arabic',
        'fa': 'Persian',
    }

    all_samples = []
    try:
        ds = datasets.load_dataset("CohereForAI/aya_dataset", split="train", trust_remote_code=True)
        for lang_code, lang_name in lang_map.items():
            if lang_code not in langs:
                continue
            lang_samples = [s for s in ds if s.get('language', '') == lang_name]
            random.shuffle(lang_samples)
            lang_samples = lang_samples[:max_per_lang]
            for s in lang_samples:
                all_samples.append({
                    'text': format_instruction(s['inputs'], s['targets']),
                    'lang': lang_code,
                    'source': 'aya',
                })
            print(f"  Aya [{lang_code}]: {len(lang_samples)} samples")
    except Exception as e:
        print(f"  Warning: Could not load Aya: {e}")

    return all_samples


def load_bactrian(langs, max_per_lang=3000):
    """Load Bactrian-X multilingual instruction data."""
    ensure_imports()
    print("Loading Bactrian-X...")

    lang_map = {'he': 'he', 'ar': 'ar', 'fa': 'fa', 'en': 'en'}
    all_samples = []

    for lang_code in langs:
        bx_code = lang_map.get(lang_code)
        if not bx_code:
            continue
        try:
            ds = datasets.load_dataset(f"MBZUAI/Bactrian-X", bx_code, split="train", trust_remote_code=True)
            indices = list(range(len(ds)))
            random.shuffle(indices)
            indices = indices[:max_per_lang]
            for i in indices:
                s = ds[i]
                inp = s.get('input', '')
                instr = s.get('instruction', '')
                out = s.get('output', '')
                if not out.strip():
                    continue
                all_samples.append({
                    'text': format_instruction(instr, out, inp),
                    'lang': lang_code,
                    'source': 'bactrian',
                })
            print(f"  Bactrian [{lang_code}]: {min(max_per_lang, len(ds))} samples")
        except Exception as e:
            print(f"  Warning: Could not load Bactrian {lang_code}: {e}")

    return all_samples


def generate_translation_pairs(langs, n_per_pair=500):
    """Generate translation instruction pairs from FLORES-200 parallel data."""
    ensure_imports()
    print("Loading FLORES-200 for translation pairs...")

    flores_map = {
        'en': 'eng_Latn',
        'he': 'heb_Hebr',
        'ar': 'arb_Arab',
        'fa': 'pes_Arab',
    }

    all_samples = []
    try:
        ds = datasets.load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)

        lang_pairs = [(a, b) for a in langs for b in langs if a != b]
        for src, tgt in lang_pairs:
            src_col = f"sentence_{flores_map[src]}"
            tgt_col = f"sentence_{flores_map[tgt]}"

            if src_col not in ds.column_names or tgt_col not in ds.column_names:
                print(f"  Warning: Missing column for {src}->{tgt}")
                continue

            indices = list(range(len(ds)))
            random.shuffle(indices)
            indices = indices[:n_per_pair]

            lang_names = {'en': 'English', 'he': 'Hebrew', 'ar': 'Arabic', 'fa': 'Farsi'}
            for i in indices:
                src_text = ds[i][src_col]
                tgt_text = ds[i][tgt_col]
                instruction = f"Translate the following {lang_names[src]} text to {lang_names[tgt]}:"
                all_samples.append({
                    'text': format_instruction(instruction, tgt_text, src_text),
                    'lang': f'{src}-{tgt}',
                    'source': 'flores_translation',
                })
            print(f"  Translation [{src}->{tgt}]: {len(indices)} pairs")

    except Exception as e:
        print(f"  Warning: Could not load FLORES: {e}")

    return all_samples


# ============ TOKENIZATION & OUTPUT ============

def tokenize_and_save(samples, tokenizer_path, output_dir, val_ratio=0.05):
    """Tokenize samples and save as binary files for training."""
    ensure_imports()

    sp = spm.SentencePieceProcessor(tokenizer_path)
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle all samples
    random.shuffle(samples)

    # Split train/val
    n_val = max(int(len(samples) * val_ratio), 100)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    # Stats
    source_counts = defaultdict(int)
    lang_counts = defaultdict(int)
    for s in samples:
        source_counts[s['source']] += 1
        lang_counts[s['lang']] += 1

    print(f"\nDataset statistics:")
    print(f"  Total: {len(samples)} ({len(train_samples)} train, {n_val} val)")
    print(f"  By source: {dict(source_counts)}")
    print(f"  By language: {dict(lang_counts)}")

    # Tokenize
    for split_name, split_data in [('train', train_samples), ('val', val_samples)]:
        all_ids = []
        for s in split_data:
            ids = sp.encode(s['text'])
            ids.append(sp.eos_id())  # Add EOS after each sample
            all_ids.extend(ids)

        arr = np.array(all_ids, dtype=np.uint16)
        filepath = os.path.join(output_dir, f'{split_name}_sft.bin')
        arr.tofile(filepath)
        print(f"  {split_name}: {len(arr)} tokens → {filepath}")

    # Save metadata
    metadata = {
        'total_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': n_val,
        'source_counts': dict(source_counts),
        'lang_counts': dict(lang_counts),
        'format': 'USER_PREFIX + instruction + ASSISTANT_PREFIX + response',
        'tokenizer': os.path.basename(tokenizer_path),
    }
    with open(os.path.join(output_dir, 'sft_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {output_dir}/sft_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, help='SentencePiece .model file')
    parser.add_argument('--output', default='/tmp/sft_data', help='Output directory')
    parser.add_argument('--langs', default='en,he,ar,fa', help='Language codes')
    parser.add_argument('--aya-per-lang', type=int, default=5000)
    parser.add_argument('--bactrian-per-lang', type=int, default=3000)
    parser.add_argument('--translation-per-pair', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    langs = args.langs.split(',')

    print(f"Preparing SFT data for languages: {langs}")
    print(f"Output: {args.output}\n")

    all_samples = []

    # 1. Aya Dataset (highest quality — human annotated)
    all_samples.extend(load_aya_dataset(langs, args.aya_per_lang))

    # 2. Bactrian-X (machine-translated alpaca)
    all_samples.extend(load_bactrian(langs, args.bactrian_per_lang))

    # 3. Translation pairs from FLORES
    all_samples.extend(generate_translation_pairs(langs, args.translation_per_pair))

    if not all_samples:
        print("ERROR: No samples collected!")
        sys.exit(1)

    # Tokenize and save
    tokenize_and_save(all_samples, args.tokenizer, args.output)

    print("\n✅ SFT data preparation complete!")
    print(f"Expected total: ~{len(all_samples)} samples")
    print(f"Run SFT with: python train_sft_3b.py --data {args.output} --checkpoint /path/to/best_model.pt")


if __name__ == '__main__':
    main()
