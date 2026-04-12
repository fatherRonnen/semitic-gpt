#!/usr/bin/env python3
"""
Collect native Arabic and Farsi SFT data for experiments D/E/F.

Sources:
  Arabic:
    - OALL/Arabic-Alpaca-2.0 (~50K native)
    - Existing Aya Arabic (already have 5K)
  Farsi:
    - FarsInstruct (sajjadayobi67/FarsInstruct)
    - sinarashidi/alpaca-persian
    - Existing Aya Farsi (already have 1.6K)
"""

import os, sys, json, random
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = '/tmp/sft_v3_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def collect_arabic():
    """Collect native Arabic instruction data."""
    from datasets import load_dataset
    
    all_samples = []
    
    # 1. OALL/Arabic-Alpaca-2.0 — native Arabic
    print("Loading OALL/Arabic-Alpaca-2.0...")
    try:
        ds = load_dataset('OALL/Arabic-Alpaca-2.0', split='train')
        count = 0
        for item in ds:
            instruction = item.get('instruction', '').strip()
            inp = item.get('input', '').strip()
            output = item.get('output', '').strip()
            if not instruction or not output:
                continue
            if inp:
                instruction = instruction + '\n' + inp
            all_samples.append({
                'instruction': instruction,
                'output': output,
                'source': 'OALL/Arabic-Alpaca-2.0',
                'lang': 'ar',
            })
            count += 1
        print(f"  ✅ Arabic-Alpaca-2.0: {count} samples")
    except Exception as e:
        print(f"  ❌ Arabic-Alpaca-2.0 failed: {e}")
    
    # 2. FreedomIntelligence/alpaca-gpt4-arabic — GPT-4 Arabic responses
    print("Loading FreedomIntelligence/alpaca-gpt4-arabic...")
    try:
        ds = load_dataset('FreedomIntelligence/alpaca-gpt4-arabic', split='train')
        count = 0
        for item in ds:
            instruction = item.get('instruction', '').strip()
            inp = item.get('input', '').strip()
            output = item.get('output', '').strip()
            if not instruction or not output:
                continue
            if inp:
                instruction = instruction + '\n' + inp
            all_samples.append({
                'instruction': instruction,
                'output': output,
                'source': 'alpaca-gpt4-arabic',
                'lang': 'ar',
            })
            count += 1
        print(f"  ✅ alpaca-gpt4-arabic: {count} samples")
    except Exception as e:
        print(f"  ❌ alpaca-gpt4-arabic failed: {e}")
    
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
    
    # Save full set
    outfile = os.path.join(OUTPUT_DIR, 'arabic_native_all.jsonl')
    with open(outfile, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    # Save subsets for scaling experiment E
    for size in [1000, 3000, 5000, 10000]:
        subset = deduped[:size]
        outfile = os.path.join(OUTPUT_DIR, f'arabic_native_{size}.jsonl')
        with open(outfile, 'w') as f:
            for s in subset:
                json.dump(s, f, ensure_ascii=False)
                f.write('\n')
        print(f"  Saved arabic_native_{size}.jsonl")
    
    return len(deduped)


def collect_farsi():
    """Collect native Farsi instruction data."""
    from datasets import load_dataset
    
    all_samples = []
    
    # 1. FarsInstruct
    print("Loading FarsInstruct...")
    try:
        ds = load_dataset('sajjadayobi67/FarsInstruct', split='train')
        count = 0
        for item in ds:
            instruction = item.get('instruction', item.get('input', '')).strip()
            output = item.get('output', item.get('response', '')).strip()
            if not instruction or not output:
                continue
            all_samples.append({
                'instruction': instruction,
                'output': output,
                'source': 'FarsInstruct',
                'lang': 'fa',
            })
            count += 1
        print(f"  ✅ FarsInstruct: {count} samples")
    except Exception as e:
        print(f"  ❌ FarsInstruct failed: {e}")
    
    # 2. sinarashidi/alpaca-persian
    print("Loading sinarashidi/alpaca-persian...")
    try:
        ds = load_dataset('sinarashidi/alpaca-persian', split='train')
        count = 0
        for item in ds:
            instruction = item.get('instruction', '').strip()
            inp = item.get('input', '').strip()
            output = item.get('output', '').strip()
            if not instruction or not output:
                continue
            if inp:
                instruction = instruction + '\n' + inp
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
    
    # 3. Aya Farsi (pes = Western Farsi)
    print("Loading Aya Farsi...")
    try:
        ds = load_dataset('CohereForAI/aya_dataset', split='train')
        count = 0
        for item in ds:
            if item.get('language_code') != 'pes':
                continue
            instruction = item.get('inputs', '').strip()
            output = item.get('targets', '').strip()
            if not instruction or not output:
                continue
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
    
    # Save full set
    outfile = os.path.join(OUTPUT_DIR, 'farsi_native_all.jsonl')
    with open(outfile, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    # Save subsets
    for size in [1000, 3000, 5000]:
        if len(deduped) >= size:
            subset = deduped[:size]
            outfile = os.path.join(OUTPUT_DIR, f'farsi_native_{size}.jsonl')
            with open(outfile, 'w') as f:
                for s in subset:
                    json.dump(s, f, ensure_ascii=False)
                    f.write('\n')
            print(f"  Saved farsi_native_{size}.jsonl")
    
    return len(deduped)


def quality_check(filepath, lang, n=5):
    """Print a few samples for quality inspection."""
    print(f"\n--- Quality check: {filepath} ---")
    with open(filepath) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line)
            print(f"  [{item['source']}]")
            print(f"  Q: {item['instruction'][:120]}...")
            print(f"  A: {item['output'][:120]}...")
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
    
    # Quality checks
    quality_check(os.path.join(OUTPUT_DIR, 'arabic_native_all.jsonl'), 'ar')
    quality_check(os.path.join(OUTPUT_DIR, 'farsi_native_all.jsonl'), 'fa')
    
    print("\nFiles:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f}: {size/1024/1024:.1f} MB")
