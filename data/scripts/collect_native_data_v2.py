#!/usr/bin/env python3
"""
Collect native Arabic and Farsi SFT data — fixed dataset names.

Arabic:
  - arbml/alpaca_arabic (52K native) 
  - FreedomIntelligence/Alpaca-Arabic-GPT4 (GPT-4 responses)
  - Yasbok/Alpaca_arabic_instruct (52K)
  - Aya Arabic (arb)

Farsi:
  - PNLPhub/FarsInstruct (9.3M samples — use a subset)
  - sinarashidi/alpaca-persian
  - Aya Farsi (pes)
"""

import os, sys, json, random
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = '/tmp/sft_v3_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_TOKEN = os.environ.get('HF_TOKEN', 'YOUR_HF_TOKEN')


def extract_instruction_output(item, source_name):
    """Extract instruction/output from various dataset formats."""
    instruction = ''
    output = ''
    
    # Standard alpaca format
    if 'instruction' in item:
        instruction = str(item.get('instruction', '')).strip()
        inp = str(item.get('input', '')).strip()
        output = str(item.get('output', '')).strip()
        if inp:
            instruction = instruction + '\n' + inp
    # FarsInstruct format
    elif 'inputs' in item:
        instruction = str(item.get('inputs', '')).strip()
        output = str(item.get('outputs', '')).strip()
    # Aya format
    elif 'inputs' in item and 'targets' in item:
        instruction = str(item.get('inputs', '')).strip()
        output = str(item.get('targets', '')).strip()
    
    return instruction, output


def collect_arabic():
    """Collect native Arabic instruction data."""
    from datasets import load_dataset
    
    all_samples = []
    
    # 1. arbml/alpaca_arabic
    print("Loading arbml/alpaca_arabic...")
    try:
        ds = load_dataset('arbml/alpaca_arabic', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            instruction, output = extract_instruction_output(item, 'arbml')
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'arbml/alpaca_arabic',
                    'lang': 'ar',
                })
                count += 1
        print(f"  ✅ arbml/alpaca_arabic: {count} samples")
    except Exception as e:
        print(f"  ❌ arbml/alpaca_arabic failed: {e}")
    
    # 2. FreedomIntelligence/Alpaca-Arabic-GPT4
    print("Loading FreedomIntelligence/Alpaca-Arabic-GPT4...")
    try:
        ds = load_dataset('FreedomIntelligence/Alpaca-Arabic-GPT4', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            instruction, output = extract_instruction_output(item, 'gpt4-arabic')
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'Alpaca-Arabic-GPT4',
                    'lang': 'ar',
                })
                count += 1
        print(f"  ✅ Alpaca-Arabic-GPT4: {count} samples")
    except Exception as e:
        print(f"  ❌ Alpaca-Arabic-GPT4 failed: {e}")
    
    # 3. Aya Arabic (arb)
    print("Loading Aya Arabic...")
    try:
        ds = load_dataset('CohereForAI/aya_dataset', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            if item.get('language_code') != 'arb':
                continue
            instruction = str(item.get('inputs', '')).strip()
            output = str(item.get('targets', '')).strip()
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'aya_arabic',
                    'lang': 'ar',
                })
                count += 1
        print(f"  ✅ Aya Arabic: {count} samples")
    except Exception as e:
        print(f"  ❌ Aya Arabic failed: {e}")
    
    # Deduplicate by instruction text
    seen = set()
    deduped = []
    for s in all_samples:
        key = s['instruction'][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    
    random.shuffle(deduped)
    print(f"  Total Arabic: {len(deduped)} (after dedup from {len(all_samples)})")
    
    # Save
    outfile = os.path.join(OUTPUT_DIR, 'arabic_native_all.jsonl')
    with open(outfile, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    for size in [1000, 3000, 5000, 10000]:
        subset = deduped[:size] if len(deduped) >= size else deduped
        outfile = os.path.join(OUTPUT_DIR, f'arabic_native_{size}.jsonl')
        with open(outfile, 'w') as f:
            for s in subset:
                json.dump(s, f, ensure_ascii=False)
                f.write('\n')
        print(f"  Saved arabic_native_{size}.jsonl ({len(subset)} samples)")
    
    return len(deduped)


def collect_farsi():
    """Collect native Farsi instruction data."""
    from datasets import load_dataset
    
    all_samples = []
    
    # 1. PNLPhub/FarsInstruct (9.3M — take first 20K)
    print("Loading PNLPhub/FarsInstruct (streaming, cap at 20K)...")
    try:
        ds = load_dataset('PNLPhub/FarsInstruct', split='train', streaming=True, token=HF_TOKEN)
        count = 0
        for item in ds:
            instruction = str(item.get('inputs', '')).strip()
            output = str(item.get('outputs', '')).strip()
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'FarsInstruct',
                    'lang': 'fa',
                })
                count += 1
            if count >= 20000:
                break
        print(f"  ✅ FarsInstruct: {count} samples")
    except Exception as e:
        print(f"  ❌ FarsInstruct failed: {e}")
    
    # 2. sinarashidi/alpaca-persian
    print("Loading sinarashidi/alpaca-persian...")
    try:
        ds = load_dataset('sinarashidi/alpaca-persian', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            instruction, output = extract_instruction_output(item, 'alpaca-persian')
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'alpaca-persian',
                    'lang': 'fa',
                })
                count += 1
        print(f"  ✅ alpaca-persian: {count} samples")
    except Exception as e:
        print(f"  ❌ alpaca-persian failed: {e}")
    
    # 3. Aya Farsi (pes)
    print("Loading Aya Farsi...")
    try:
        ds = load_dataset('CohereForAI/aya_dataset', split='train', token=HF_TOKEN)
        count = 0
        for item in ds:
            if item.get('language_code') != 'pes':
                continue
            instruction = str(item.get('inputs', '')).strip()
            output = str(item.get('targets', '')).strip()
            if instruction and output and len(output) > 10:
                all_samples.append({
                    'instruction': instruction,
                    'output': output,
                    'source': 'aya_farsi',
                    'lang': 'fa',
                })
                count += 1
        print(f"  ✅ Aya Farsi: {count} samples")
    except Exception as e:
        print(f"  ❌ Aya Farsi failed: {e}")
    
    # Deduplicate
    seen = set()
    deduped = []
    for s in all_samples:
        key = s['instruction'][:200]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    
    random.shuffle(deduped)
    print(f"  Total Farsi: {len(deduped)} (after dedup from {len(all_samples)})")
    
    outfile = os.path.join(OUTPUT_DIR, 'farsi_native_all.jsonl')
    with open(outfile, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    for size in [1000, 3000, 5000, 8000]:
        if len(deduped) >= size:
            subset = deduped[:size]
            outfile = os.path.join(OUTPUT_DIR, f'farsi_native_{size}.jsonl')
            with open(outfile, 'w') as f:
                for s in subset:
                    json.dump(s, f, ensure_ascii=False)
                    f.write('\n')
            print(f"  Saved farsi_native_{size}.jsonl ({len(subset)} samples)")
    
    return len(deduped)


def quality_check(filepath, n=3):
    """Print a few samples for quality inspection."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 10:
        print(f"  [empty or missing: {filepath}]")
        return
    print(f"\n--- Quality check: {os.path.basename(filepath)} ---")
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line)
            print(f"  [{item['source']}]")
            print(f"  Q: {item['instruction'][:150]}")
            print(f"  A: {item['output'][:150]}")
            print()


if __name__ == '__main__':
    print("="*60)
    print("Collecting native Arabic & Farsi SFT data")
    print("="*60)
    
    ar_count = collect_arabic()
    fa_count = collect_farsi()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Arabic={ar_count}, Farsi={fa_count}")
    print(f"{'='*60}")
    
    quality_check(os.path.join(OUTPUT_DIR, 'arabic_native_all.jsonl'))
    quality_check(os.path.join(OUTPUT_DIR, 'farsi_native_all.jsonl'))
    
    print("\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        lines = sum(1 for _ in open(fpath)) if size > 0 else 0
        print(f"  {f}: {size/1024/1024:.1f} MB, {lines} lines")
