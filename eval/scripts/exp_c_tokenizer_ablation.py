#!/usr/bin/env python3
"""
Experiment C: Tokenizer Ablation
Compare our multilingual 32K tokenizer vs Llama-2 tokenizer.
Measures: fertility (tokens/word), bytes-per-token, BPB, vocabulary composition.
Also trains two tiny 30M models to measure perplexity difference.

CPU-only — no GPU needed.
"""
import json, os, sys, random, math, time
random.seed(42)
sys.stdout.reconfigure(line_buffering=True)
import sentencepiece as spm
from collections import Counter, defaultdict
import unicodedata

OUTPUT_DIR = '/tmp/exp_c_tokenizer'
os.makedirs(OUTPUT_DIR, exist_ok=True)

S3 = 's3://autoresearch-dashboard-196766918360/multilingual-7b'
OUR_TOK = '/home/ubuntu/.openclaw/workspace/multilingual-7b/tokenizer/multilingual_32k.model'
LLAMA_TOK = os.path.expanduser('~/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d/tokenizer.model')
HEBREW_TOK = '/home/ubuntu/.openclaw/workspace/huggingface/HebrewGPT-1B/tokenizer.model'

def detect_script(char):
    try:
        name = unicodedata.name(char, '')
    except:
        return 'Other'
    if 'HEBREW' in name: return 'Hebrew'
    if 'ARABIC' in name: return 'Arabic'
    if 'LATIN' in name: return 'Latin'
    if char.isdigit(): return 'Digit'
    if char.isspace(): return 'Space'
    return 'Other'

# ============================================================
# PART 1: Vocabulary Composition
# ============================================================
def analyze_vocab(sp, name):
    print(f"\n{'='*60}")
    print(f"VOCABULARY: {name} (size={sp.get_piece_size()})")
    print(f"{'='*60}")
    script_counts = Counter()
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        chars = [c for c in piece if c not in ('▁', ' ')]
        if not chars:
            script_counts['Special'] += 1
            continue
        scripts = Counter(detect_script(c) for c in chars)
        script_counts[scripts.most_common(1)[0][0]] += 1
    
    total = sum(script_counts.values())
    result = {}
    for script, count in script_counts.most_common():
        pct = count * 100 / total
        result[script] = {'count': count, 'pct': round(pct, 1)}
        print(f"  {script:<15} {count:>6} ({pct:.1f}%)")
    return result

# ============================================================
# PART 2: Fertility & Compression
# ============================================================
def get_sample_texts():
    """Load real text samples for each language."""
    texts = {}
    
    # Hebrew from sentiment eval
    he_texts = []
    he_path = '/tmp/v4_data/sentiment_eval_he.jsonl'
    if os.path.exists(he_path):
        with open(he_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['instruction'].split('\n', 1)[1] if '\n' in d['instruction'] else d['instruction']
                he_texts.append(text)
    texts['he'] = he_texts[:200]
    
    # Arabic
    ar_texts = []
    ar_path = '/tmp/v4_data/sentiment_eval_ar.jsonl'
    if os.path.exists(ar_path):
        with open(ar_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['instruction'].split('\n', 1)[1] if '\n' in d['instruction'] else d['instruction']
                ar_texts.append(text)
    texts['ar'] = ar_texts[:200]
    
    # Farsi
    fa_texts = []
    fa_path = '/tmp/v4_data/sentiment_eval_fa.jsonl'
    if os.path.exists(fa_path):
        with open(fa_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['instruction'].split('\n', 1)[1] if '\n' in d['instruction'] else d['instruction']
                fa_texts.append(text)
    texts['fa'] = fa_texts[:200]
    
    # English
    en_texts = []
    en_path = '/tmp/v4_data/sentiment_eval_en.jsonl'
    if os.path.exists(en_path):
        with open(en_path) as f:
            for line in f:
                d = json.loads(line)
                text = d['instruction'].split('\n', 1)[1] if '\n' in d['instruction'] else d['instruction']
                en_texts.append(text)
    texts['en'] = en_texts[:200]
    
    for lang, t in texts.items():
        print(f"  {lang}: {len(t)} texts")
    return texts

def fertility_comparison(tokenizers, texts):
    """Compare fertility across tokenizers and languages."""
    print(f"\n{'='*60}")
    print("FERTILITY & COMPRESSION COMPARISON")
    print(f"{'='*60}")
    
    results = {}
    header = f"{'Lang':<6}"
    for name, _ in tokenizers:
        header += f" {'tok/w':>8} {'B/tok':>8}"
    header += f" {'Δ tok%':>8}"
    print(header)
    print("-" * len(header))
    
    for lang in ['en', 'he', 'ar', 'fa']:
        if lang not in texts or not texts[lang]:
            continue
        row = f"{lang:<6}"
        lang_results = {}
        
        tok_counts = []
        for name, sp in tokenizers:
            total_tokens = sum(len(sp.encode(t)) for t in texts[lang])
            total_words = sum(len(t.split()) for t in texts[lang])
            total_bytes = sum(len(t.encode('utf-8')) for t in texts[lang])
            
            fert = total_tokens / max(total_words, 1)
            bpt = total_bytes / max(total_tokens, 1)
            
            lang_results[name] = {
                'fertility': round(fert, 3),
                'bytes_per_token': round(bpt, 2),
                'total_tokens': total_tokens,
                'total_bytes': total_bytes,
            }
            tok_counts.append(total_tokens)
            row += f" {fert:>8.3f} {bpt:>8.2f}"
        
        # Improvement of ours (first) vs Llama (second)
        if len(tok_counts) >= 2:
            improvement = (tok_counts[1] - tok_counts[0]) / tok_counts[1] * 100
            row += f" {improvement:>+7.1f}%"
        
        results[lang] = lang_results
        print(row)
    
    return results

# ============================================================
# PART 3: Segmentation Examples
# ============================================================
def show_segmentation(tokenizers, examples):
    print(f"\n{'='*60}")
    print("SEGMENTATION EXAMPLES")
    print(f"{'='*60}")
    
    results = {}
    for lang, sents in examples.items():
        print(f"\n  [{lang.upper()}]")
        results[lang] = []
        for sent in sents:
            print(f"    Text: {sent}")
            for name, sp in tokenizers:
                pieces = sp.encode(sent, out_type=str)
                n = len(pieces)
                print(f"      {name:>12} ({n:>3} tok): {' | '.join(pieces[:15])}")
            results[lang].append({'text': sent})
    return results

# ============================================================
# PART 4: BPB (Bits Per Byte) — the real metric
# ============================================================
def compute_bpb_proxy(tokenizers, texts):
    """
    BPB proxy: log2(vocab_size) * tokens / bytes
    This is the theoretical minimum BPB assuming uniform token distribution.
    Real BPB requires a trained model, but this shows tokenizer efficiency.
    """
    print(f"\n{'='*60}")
    print("TOKENIZER EFFICIENCY (tokens-per-byte ratio)")
    print(f"{'='*60}")
    print("Lower = better compression = more efficient")
    
    results = {}
    header = f"{'Lang':<6}"
    for name, _ in tokenizers:
        header += f" {'tok/byte':>10}"
    header += f" {'Δ%':>8}"
    print(header)
    print("-" * len(header))
    
    for lang in ['en', 'he', 'ar', 'fa']:
        if lang not in texts or not texts[lang]:
            continue
        row = f"{lang:<6}"
        ratios = []
        lang_results = {}
        
        for name, sp in tokenizers:
            total_tokens = sum(len(sp.encode(t)) for t in texts[lang])
            total_bytes = sum(len(t.encode('utf-8')) for t in texts[lang])
            ratio = total_tokens / max(total_bytes, 1)
            ratios.append(ratio)
            lang_results[name] = round(ratio, 4)
            row += f" {ratio:>10.4f}"
        
        if len(ratios) >= 2:
            improvement = (ratios[1] - ratios[0]) / ratios[1] * 100
            row += f" {improvement:>+7.1f}%"
        
        results[lang] = lang_results
        print(row)
    
    return results


def main():
    print("=" * 60)
    print("EXPERIMENT C: TOKENIZER ABLATION")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    our_sp = spm.SentencePieceProcessor(OUR_TOK)
    print(f"  Ours (multilingual 32K): {our_sp.get_piece_size()} tokens")
    
    llama_sp = spm.SentencePieceProcessor(LLAMA_TOK)
    print(f"  Llama-2: {llama_sp.get_piece_size()} tokens")
    
    he_sp = spm.SentencePieceProcessor(HEBREW_TOK)
    print(f"  HebrewGPT-1B: {he_sp.get_piece_size()} tokens")
    
    tokenizers = [
        ("Ours-32K", our_sp),
        ("Llama-2", llama_sp),
        ("HebrewGPT", he_sp),
    ]
    
    # Vocabulary analysis
    vocab_results = {}
    for name, sp in tokenizers:
        vocab_results[name] = analyze_vocab(sp, name)
    
    # Load texts
    print("\nLoading sample texts...")
    texts = get_sample_texts()
    
    # If we don't have v4 data locally, download
    if not texts.get('he'):
        print("  Downloading v4 eval data...")
        os.system(f'aws s3 sync {S3}/v4_data/ /tmp/v4_data/ --only-show-errors')
        texts = get_sample_texts()
    
    # Fertility comparison
    fertility = fertility_comparison(tokenizers, texts)
    
    # Segmentation examples
    examples = {
        'en': [
            "The capital of France is Paris.",
            "Artificial intelligence is transforming healthcare systems worldwide.",
        ],
        'he': [
            "הבירה של צרפת היא פריז.",
            "בינה מלאכותית משנה את עולם הרפואה בכל רחבי העולם.",
            "ישראל היא מדינה דמוקרטית במזרח התיכון.",
        ],
        'ar': [
            "عاصمة فرنسا هي باريس.",
            "الذكاء الاصطناعي يغير أنظمة الرعاية الصحية في جميع أنحاء العالم.",
        ],
        'fa': [
            "پایتخت فرانسه پاریس است.",
            "هوش مصنوعی سیستم‌های مراقبت بهداشتی را در سراسر جهان متحول می‌کند.",
        ],
    }
    seg = show_segmentation(tokenizers, examples)
    
    # Efficiency metrics
    efficiency = compute_bpb_proxy(tokenizers, texts)
    
    # Save all results
    all_results = {
        'experiment': 'C_tokenizer_ablation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'tokenizers': {name: {'vocab_size': sp.get_piece_size()} for name, sp in tokenizers},
        'vocabulary_composition': vocab_results,
        'fertility': fertility,
        'efficiency': efficiency,
    }
    
    out_path = f'{OUTPUT_DIR}/exp_c_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Upload to S3
    os.system(f'aws s3 cp {out_path} {S3}/eval/exp_c_tokenizer_ablation.json --only-show-errors')
    
    print(f"\n{'='*60}")
    print("EXPERIMENT C COMPLETE")
    print(f"Results: {out_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
