#!/usr/bin/env python3
"""
Collect parallel translation data for Exp G.
Sources:
  - OPUS (via HuggingFace Helsinki-NLP datasets): Tatoeba, FLORES, UN Parallel
  - NLLB seed data
  - FLORES-200 devtest for evaluation

Language pairs (12 directions):
  HE↔EN, HE↔AR, HE↔FA
  AR↔EN, AR↔FA
  FA↔EN

Output: JSONL with instruction/output format for translation SFT.
"""

import os, sys, json, random
sys.stdout.reconfigure(line_buffering=True)

HF_TOKEN = os.environ['HF_TOKEN']  # Set via: export HF_TOKEN=your_token
OUTPUT_DIR = '/tmp/translation_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_NAMES = {
    'he': 'Hebrew', 'ar': 'Arabic', 'en': 'English', 'fa': 'Persian'
}

# FLORES-200 language codes
FLORES_CODES = {
    'he': 'heb_Hebr', 'ar': 'arb_Arab', 'en': 'eng_Latn', 'fa': 'pes_Arab'
}


def format_translation_sample(src_text, tgt_text, src_lang, tgt_lang, source_dataset):
    """Format as instruction-following translation sample."""
    src_name = LANG_NAMES[src_lang]
    tgt_name = LANG_NAMES[tgt_lang]
    
    # Vary the instruction format for diversity
    templates = [
        f"Translate the following {src_name} text to {tgt_name}:\n{src_text}",
        f"Translate from {src_name} to {tgt_name}:\n{src_text}",
        f"{src_name} → {tgt_name}:\n{src_text}",
        f"Please translate this to {tgt_name}:\n{src_text}",
    ]
    instruction = random.choice(templates)
    
    return {
        'instruction': instruction,
        'output': tgt_text,
        'source': source_dataset,
        'lang': f'{src_lang}-{tgt_lang}',
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
    }


def collect_opus_tatoeba():
    """Collect from Tatoeba (high quality, short sentences)."""
    from datasets import load_dataset
    
    pairs_collected = {}
    
    # Tatoeba pairs available on HF
    tatoeba_pairs = [
        ('ar', 'en'), ('he', 'en'), ('fa', 'en'),
        ('ar', 'he'),  # May not exist
    ]
    
    for src, tgt in tatoeba_pairs:
        pair_key = f'{src}-{tgt}'
        try:
            # Helsinki-NLP format
            ds_name = f"Helsinki-NLP/tatoeba_mt"
            print(f"  Trying Tatoeba {src}-{tgt}...")
            ds = load_dataset(ds_name, f"{src}-{tgt}", split='test', 
                            token=HF_TOKEN, trust_remote_code=True)
            samples = []
            for item in ds:
                src_text = item.get('sourceString', item.get('source', '')).strip()
                tgt_text = item.get('targetString', item.get('target', '')).strip()
                if src_text and tgt_text and len(src_text) > 5 and len(tgt_text) > 5:
                    # Both directions
                    samples.append(format_translation_sample(src_text, tgt_text, src, tgt, 'tatoeba'))
                    samples.append(format_translation_sample(tgt_text, src_text, tgt, src, 'tatoeba'))
            pairs_collected[pair_key] = samples
            print(f"    ✅ {pair_key}: {len(samples)} samples (both directions)")
        except Exception as e:
            print(f"    ❌ {pair_key}: {e}")
    
    return pairs_collected


def collect_opus_100():
    """Collect from opus-100 (diverse parallel data)."""
    from datasets import load_dataset
    
    pairs_collected = {}
    opus_pairs = [
        ('ar', 'en'), ('he', 'en'), ('fa', 'en'),
        ('ar', 'he'), ('ar', 'fa'),
    ]
    
    for src, tgt in opus_pairs:
        pair_key = f'{src}-{tgt}'
        try:
            print(f"  Trying opus-100 {src}-{tgt}...")
            ds = load_dataset("opus100", f"{src}-{tgt}", split='train',
                            token=HF_TOKEN, trust_remote_code=True)
            samples = []
            # Take up to 3000 per pair
            indices = list(range(len(ds)))
            random.shuffle(indices)
            for i in indices[:3000]:
                item = ds[i]
                trans = item.get('translation', item)
                src_text = trans.get(src, '').strip()
                tgt_text = trans.get(tgt, '').strip()
                if src_text and tgt_text and len(src_text) > 10 and len(tgt_text) > 10:
                    samples.append(format_translation_sample(src_text, tgt_text, src, tgt, 'opus100'))
                    samples.append(format_translation_sample(tgt_text, src_text, tgt, src, 'opus100'))
            pairs_collected[pair_key] = samples
            print(f"    ✅ {pair_key}: {len(samples)} samples (both directions)")
        except Exception as e:
            print(f"    ❌ {pair_key}: {e}")
    
    return pairs_collected


def collect_flores_dev():
    """Collect FLORES-200 devtest for evaluation (not training)."""
    from datasets import load_dataset
    
    print("  Loading FLORES-200 devtest...")
    try:
        ds = load_dataset("facebook/flores", "all", split='devtest', token=HF_TOKEN)
        
        flores_data = {lang: [] for lang in ['he', 'ar', 'en', 'fa']}
        for item in ds:
            for lang, code in FLORES_CODES.items():
                text = item.get(f'sentence_{code}', item.get(code, '')).strip()
                if text:
                    flores_data[lang].append(text)
        
        # Create eval pairs
        eval_pairs = []
        langs = ['he', 'ar', 'en', 'fa']
        for i, src in enumerate(langs):
            for j, tgt in enumerate(langs):
                if i == j:
                    continue
                pairs = []
                for k in range(min(len(flores_data[src]), len(flores_data[tgt]))):
                    if flores_data[src][k] and flores_data[tgt][k]:
                        pairs.append({
                            'src': flores_data[src][k],
                            'tgt': flores_data[tgt][k],
                            'src_lang': src,
                            'tgt_lang': tgt,
                        })
                eval_pairs.append((f'{src}-{tgt}', pairs))
                print(f"    FLORES eval {src}→{tgt}: {len(pairs)} pairs")
        
        return eval_pairs
    except Exception as e:
        print(f"    ❌ FLORES failed: {e}")
        # Try alternative name
        try:
            ds = load_dataset("openlanguagedata/flores_plus", split='devtest', token=HF_TOKEN)
            print(f"    Trying flores_plus... columns: {ds.column_names[:10]}")
            return []
        except Exception as e2:
            print(f"    ❌ flores_plus also failed: {e2}")
            return []


def collect_un_parallel():
    """UN Parallel corpus — good for AR-EN, AR-HE (via EN pivot)."""
    from datasets import load_dataset
    
    print("  Trying UN parallel corpus...")
    try:
        ds = load_dataset("Helsinki-NLP/opus-100", "ar-en", split='train',
                         token=HF_TOKEN, streaming=True)
        # Already covered in opus-100
        return {}
    except:
        return {}


def main():
    print("="*60)
    print("Collecting parallel translation data for Exp G")
    print("="*60)
    
    all_train_samples = []
    
    # 1. OPUS-100
    print("\n[1/3] OPUS-100 parallel data...")
    opus_data = collect_opus_100()
    for pair, samples in opus_data.items():
        all_train_samples.extend(samples)
    
    # 2. Tatoeba
    print("\n[2/3] Tatoeba parallel data...")
    tatoeba_data = collect_opus_tatoeba()
    for pair, samples in tatoeba_data.items():
        all_train_samples.extend(samples)
    
    # 3. FLORES eval
    print("\n[3/3] FLORES-200 evaluation data...")
    flores_eval = collect_flores_dev()
    
    # Deduplicate training data
    seen = set()
    deduped = []
    for s in all_train_samples:
        key = s['instruction'][:150]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    
    random.shuffle(deduped)
    
    # Stats
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(deduped)} training samples (from {len(all_train_samples)} raw)")
    
    # Per-direction breakdown
    direction_counts = {}
    for s in deduped:
        d = s['lang']
        direction_counts[d] = direction_counts.get(d, 0) + 1
    
    print("\nPer-direction:")
    for d, c in sorted(direction_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c}")
    
    # Save training data
    train_file = os.path.join(OUTPUT_DIR, 'translation_train_all.jsonl')
    with open(train_file, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    # Save subsets for experiments
    for size in [2000, 5000, 10000]:
        if len(deduped) >= size:
            subset = deduped[:size]
            outfile = os.path.join(OUTPUT_DIR, f'translation_train_{size}.jsonl')
            with open(outfile, 'w') as f:
                for s in subset:
                    json.dump(s, f, ensure_ascii=False)
                    f.write('\n')
            print(f"  Saved translation_train_{size}.jsonl")
    
    # Save FLORES eval
    if flores_eval:
        eval_file = os.path.join(OUTPUT_DIR, 'flores_eval.json')
        eval_data = {}
        for pair_name, pairs in flores_eval:
            eval_data[pair_name] = pairs
        with open(eval_file, 'w') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"  Saved flores_eval.json ({sum(len(v) for v in eval_data.values())} eval pairs)")
    
    # Quality check
    print(f"\n--- Quality check ---")
    for s in deduped[:5]:
        print(f"  [{s['lang']}] Q: {s['instruction'][:100]}")
        print(f"       A: {s['output'][:100]}")
        print()
    
    print(f"\nFiles in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f}: {size/1024/1024:.1f} MB")


if __name__ == '__main__':
    main()
