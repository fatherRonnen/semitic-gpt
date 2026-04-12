#!/usr/bin/env python3
"""
Collect parallel translation data v2 — use opus100 correct config names + direct downloads.
opus100 uses 'en-XX' format (alphabetical), not 'XX-en'.
Also download FLORES devtest directly from GitHub.
"""

import os, sys, json, random, urllib.request, zipfile, tempfile
sys.stdout.reconfigure(line_buffering=True)

HF_TOKEN = os.environ['HF_TOKEN']  # Set via: export HF_TOKEN=your_token
OUTPUT_DIR = '/tmp/translation_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANG_NAMES = {'he': 'Hebrew', 'ar': 'Arabic', 'en': 'English', 'fa': 'Persian'}


def format_translation_sample(src_text, tgt_text, src_lang, tgt_lang, source_dataset):
    templates = [
        f"Translate the following {LANG_NAMES[src_lang]} text to {LANG_NAMES[tgt_lang]}:\n{src_text}",
        f"Translate from {LANG_NAMES[src_lang]} to {LANG_NAMES[tgt_lang]}:\n{src_text}",
        f"{LANG_NAMES[src_lang]} → {LANG_NAMES[tgt_lang]}:\n{src_text}",
        f"Please translate this to {LANG_NAMES[tgt_lang]}:\n{src_text}",
    ]
    return {
        'instruction': random.choice(templates),
        'output': tgt_text,
        'source': source_dataset,
        'lang': f'{src_lang}-{tgt_lang}',
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
    }


def collect_opus100():
    """opus100 uses alphabetical order: ar-en, en-fa, en-he."""
    from datasets import load_dataset
    
    all_samples = []
    # Correct config names from the error message
    configs = [
        ('ar-en', 'ar', 'en'),
        ('en-fa', 'en', 'fa'),
        ('en-he', 'en', 'he'),
    ]
    
    for config, lang1, lang2 in configs:
        print(f"  Loading opus100 {config}...")
        try:
            ds = load_dataset("opus100", config, split='train', token=HF_TOKEN)
            count = 0
            indices = list(range(len(ds)))
            random.shuffle(indices)
            for i in indices[:3000]:
                item = ds[i]
                trans = item.get('translation', item)
                text1 = trans.get(lang1, '').strip()
                text2 = trans.get(lang2, '').strip()
                if text1 and text2 and len(text1) > 10 and len(text2) > 10:
                    # Both directions
                    all_samples.append(format_translation_sample(text1, text2, lang1, lang2, 'opus100'))
                    all_samples.append(format_translation_sample(text2, text1, lang2, lang1, 'opus100'))
                    count += 1
            print(f"    ✅ {config}: {count} pairs → {count*2} samples")
        except Exception as e:
            print(f"    ❌ {config}: {e}")
    
    return all_samples


def download_flores():
    """Download FLORES-200 devtest from GitHub release."""
    print("  Downloading FLORES-200 devtest...")
    
    flores_dir = '/tmp/flores200'
    os.makedirs(flores_dir, exist_ok=True)
    
    # FLORES-200 is available at this URL
    url = "https://tinyurl.com/flores200dataset"
    
    # Try the direct dataset approach
    try:
        from datasets import load_dataset
        # facebook/flores200 — try different names
        for name in ['Muennighoff/flores200', 'gsarti/flores_101']:
            try:
                print(f"    Trying {name}...")
                ds = load_dataset(name, split='devtest', token=HF_TOKEN)
                print(f"    Columns: {ds.column_names[:10]}")
                return ds
            except Exception as e:
                print(f"    {name} failed: {e}")
                continue
    except:
        pass
    
    return None


def collect_tatoeba_direct():
    """Download Tatoeba data directly from OPUS."""
    from datasets import load_dataset
    
    all_samples = []
    
    # Try Helsinki-NLP/tatoeba without trust_remote_code
    pairs = [
        ('heb', 'eng', 'he', 'en'),
        ('ara', 'eng', 'ar', 'en'),
        ('pes', 'eng', 'fa', 'en'),
        ('ara', 'heb', 'ar', 'he'),
    ]
    
    # Alternative: use opus_books or other available parallel datasets
    print("  Trying alternative parallel datasets...")
    
    # Helsinki-NLP translation pairs (these are pre-made)
    helsinki_pairs = [
        ('Helsinki-NLP/opus-mt-tc-big-he-en', 'he', 'en'),  # These are models not data
    ]
    
    # Try CCAligned (Facebook's parallel data)
    try:
        print("  Trying CCAligned he-en...")
        ds = load_dataset("yhavinga/ccmatrix", "en-he", split='train', 
                         streaming=True, token=HF_TOKEN)
        count = 0
        for item in ds:
            src = item.get('translation', {}).get('en', '').strip()
            tgt = item.get('translation', {}).get('he', '').strip()
            if src and tgt and 10 < len(src) < 500 and 10 < len(tgt) < 500:
                all_samples.append(format_translation_sample(src, tgt, 'en', 'he', 'ccmatrix'))
                all_samples.append(format_translation_sample(tgt, src, 'he', 'en', 'ccmatrix'))
                count += 1
            if count >= 2000:
                break
        print(f"    ✅ CCMatrix en-he: {count} pairs")
    except Exception as e:
        print(f"    ❌ CCMatrix en-he: {e}")
    
    # Try NLLB seed data
    try:
        print("  Trying allenai/nllb...")
        ds = load_dataset("allenai/nllb", "heb_Hebr-eng_Latn", split='train',
                         streaming=True, token=HF_TOKEN)
        count = 0
        for item in ds:
            src = item.get('translation', {}).get('heb_Hebr', '').strip()
            tgt = item.get('translation', {}).get('eng_Latn', '').strip()
            if src and tgt and len(src) > 10 and len(tgt) > 10:
                all_samples.append(format_translation_sample(src, tgt, 'he', 'en', 'nllb'))
                all_samples.append(format_translation_sample(tgt, src, 'en', 'he', 'nllb'))
                count += 1
            if count >= 2000:
                break
        print(f"    ✅ NLLB he-en: {count} pairs")
    except Exception as e:
        print(f"    ❌ NLLB: {e}")
    
    # Direct HE↔AR — very rare, use UN corpus or create from EN pivot
    # We'll generate HE↔AR and HE↔FA and AR↔FA from EN pivot using the existing data
    
    return all_samples


def create_pivot_pairs(all_samples):
    """Create HE↔AR, HE↔FA, AR↔FA pairs using English as pivot."""
    print("\n  Creating pivot-based pairs (via English)...")
    
    # Group by English source text
    en_to_he = {}
    en_to_ar = {}
    en_to_fa = {}
    
    for s in all_samples:
        if s['src_lang'] == 'en':
            en_text = s['instruction'].split('\n', 1)[-1].strip()  # Extract source text
            if s['tgt_lang'] == 'he':
                en_to_he[en_text[:100]] = s['output']
            elif s['tgt_lang'] == 'ar':
                en_to_ar[en_text[:100]] = s['output']
            elif s['tgt_lang'] == 'fa':
                en_to_fa[en_text[:100]] = s['output']
    
    pivot_samples = []
    
    # HE↔AR via English pivot
    common_he_ar = set(en_to_he.keys()) & set(en_to_ar.keys())
    for key in list(common_he_ar)[:1000]:
        he_text = en_to_he[key]
        ar_text = en_to_ar[key]
        pivot_samples.append(format_translation_sample(he_text, ar_text, 'he', 'ar', 'pivot'))
        pivot_samples.append(format_translation_sample(ar_text, he_text, 'ar', 'he', 'pivot'))
    print(f"    HE↔AR pivot pairs: {len(common_he_ar)} available, used {min(len(common_he_ar), 1000)}")
    
    # HE↔FA via English pivot
    common_he_fa = set(en_to_he.keys()) & set(en_to_fa.keys())
    for key in list(common_he_fa)[:1000]:
        he_text = en_to_he[key]
        fa_text = en_to_fa[key]
        pivot_samples.append(format_translation_sample(he_text, fa_text, 'he', 'fa', 'pivot'))
        pivot_samples.append(format_translation_sample(fa_text, he_text, 'fa', 'he', 'pivot'))
    print(f"    HE↔FA pivot pairs: {len(common_he_fa)} available, used {min(len(common_he_fa), 1000)}")
    
    # AR↔FA via English pivot
    common_ar_fa = set(en_to_ar.keys()) & set(en_to_fa.keys())
    for key in list(common_ar_fa)[:1000]:
        ar_text = en_to_ar[key]
        fa_text = en_to_fa[key]
        pivot_samples.append(format_translation_sample(ar_text, fa_text, 'ar', 'fa', 'pivot'))
        pivot_samples.append(format_translation_sample(fa_text, ar_text, 'fa', 'ar', 'pivot'))
    print(f"    AR↔FA pivot pairs: {len(common_ar_fa)} available, used {min(len(common_ar_fa), 1000)}")
    
    return pivot_samples


def main():
    print("="*60)
    print("Collecting parallel translation data for Exp G (v2)")
    print("="*60)
    
    all_samples = []
    
    # 1. OPUS-100 (ar-en, en-fa, en-he)
    print("\n[1/3] OPUS-100...")
    opus_samples = collect_opus100()
    all_samples.extend(opus_samples)
    print(f"  Total from OPUS-100: {len(opus_samples)}")
    
    # 2. Additional sources (CCMatrix, NLLB)
    print("\n[2/3] Additional parallel sources...")
    extra_samples = collect_tatoeba_direct()
    all_samples.extend(extra_samples)
    print(f"  Total from extra sources: {len(extra_samples)}")
    
    # 3. Pivot-based pairs
    print("\n[3/3] Pivot pairs...")
    pivot_samples = create_pivot_pairs(all_samples)
    all_samples.extend(pivot_samples)
    print(f"  Total from pivot: {len(pivot_samples)}")
    
    # Deduplicate
    seen = set()
    deduped = []
    for s in all_samples:
        key = s['instruction'][:150]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    
    random.shuffle(deduped)
    
    # Stats
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(deduped)} samples (from {len(all_samples)} raw)")
    
    direction_counts = {}
    for s in deduped:
        d = s['lang']
        direction_counts[d] = direction_counts.get(d, 0) + 1
    
    print("\nPer-direction:")
    for d, c in sorted(direction_counts.items(), key=lambda x: -x[1]):
        print(f"  {d}: {c}")
    
    # Save
    train_file = os.path.join(OUTPUT_DIR, 'translation_train_all.jsonl')
    with open(train_file, 'w') as f:
        for s in deduped:
            json.dump(s, f, ensure_ascii=False)
            f.write('\n')
    
    for size in [2000, 5000, 10000, 15000]:
        if len(deduped) >= size:
            subset = deduped[:size]
            outfile = os.path.join(OUTPUT_DIR, f'translation_train_{size}.jsonl')
            with open(outfile, 'w') as f:
                for s in subset:
                    json.dump(s, f, ensure_ascii=False)
                    f.write('\n')
            print(f"  Saved translation_train_{size}.jsonl")
    
    # Quality check
    print(f"\n--- Quality check (one per direction) ---")
    shown = set()
    for s in deduped:
        if s['lang'] not in shown:
            shown.add(s['lang'])
            print(f"  [{s['lang']}] Q: {s['instruction'][:120]}")
            print(f"       A: {s['output'][:120]}")
            print()
        if len(shown) >= 8:
            break
    
    print(f"\nFiles in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        lines = sum(1 for _ in open(fpath)) if 'jsonl' in f else 0
        print(f"  {f}: {size/1024/1024:.1f} MB, {lines} lines")


if __name__ == '__main__':
    main()
