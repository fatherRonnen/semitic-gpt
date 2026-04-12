#!/usr/bin/env python3
"""
Tokenizer Analysis for Multilingual 3B GPT Paper

Produces:
1. Fertility (tokens/word) per language
2. Compression ratio (bytes/token) per language  
3. Character-per-token ratios
4. Segmentation examples
5. Vocabulary composition by script
6. Comparison with Llama tokenizer as baseline
"""

import os, sys, json, re
from collections import Counter, defaultdict
import unicodedata

sys.stdout.reconfigure(line_buffering=True)
import sentencepiece as spm

def detect_script(char):
    """Detect the Unicode script block of a character."""
    try:
        name = unicodedata.name(char, '')
    except:
        return 'Other'
    if 'HEBREW' in name: return 'Hebrew'
    if 'ARABIC' in name: return 'Arabic'
    if 'LATIN' in name: return 'Latin'
    if 'CJK' in name: return 'CJK'
    if 'CYRILLIC' in name: return 'Cyrillic'
    if 'DIGIT' in name or char.isdigit(): return 'Digit'
    if char.isspace(): return 'Space'
    if 'PUNCTUATION' in name or 'MARK' in name: return 'Punctuation'
    return 'Other'


def analyze_vocabulary(sp):
    """Analyze vocabulary composition by script."""
    print("\n=== VOCABULARY COMPOSITION ===")
    script_counts = Counter()
    script_tokens = defaultdict(list)
    
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        # Determine dominant script
        chars = [c for c in piece if not c in ('▁', ' ')]
        if not chars:
            script_counts['Special'] += 1
            continue
        scripts = Counter(detect_script(c) for c in chars)
        dominant = scripts.most_common(1)[0][0]
        script_counts[dominant] += 1
        if len(script_tokens[dominant]) < 5:
            script_tokens[dominant].append(piece)
    
    total = sum(script_counts.values())
    print(f"\nVocab size: {sp.get_piece_size()}")
    print(f"\n{'Script':<15} {'Count':>6} {'Share':>7}")
    print("-" * 30)
    for script, count in script_counts.most_common():
        print(f"{script:<15} {count:>6} {count*100/total:>6.1f}%")
    
    print(f"\nSample tokens per script:")
    for script, tokens in sorted(script_tokens.items()):
        print(f"  {script}: {tokens}")
    
    return dict(script_counts)


def fertility_analysis(sp, texts_by_lang):
    """Compute fertility (tokens per word) for each language."""
    print("\n=== FERTILITY ANALYSIS (tokens/word) ===")
    results = {}
    
    for lang, texts in texts_by_lang.items():
        total_tokens = 0
        total_words = 0
        total_chars = 0
        total_bytes = 0
        
        for text in texts:
            ids = sp.encode(text)
            words = text.split()
            total_tokens += len(ids)
            total_words += len(words)
            total_chars += len(text)
            total_bytes += len(text.encode('utf-8'))
        
        fertility = total_tokens / max(total_words, 1)
        chars_per_token = total_chars / max(total_tokens, 1)
        bytes_per_token = total_bytes / max(total_tokens, 1)
        compression = total_bytes / max(total_tokens, 1)
        
        results[lang] = {
            'fertility': round(fertility, 3),
            'chars_per_token': round(chars_per_token, 3),
            'bytes_per_token': round(bytes_per_token, 3),
            'total_tokens': total_tokens,
            'total_words': total_words,
            'total_chars': total_chars,
            'total_bytes': total_bytes,
        }
        print(f"  {lang}: fertility={fertility:.3f} tok/word, "
              f"chars/tok={chars_per_token:.2f}, bytes/tok={bytes_per_token:.2f}")
    
    return results


def segmentation_examples(sp, texts_by_lang):
    """Show tokenization examples for each language."""
    print("\n=== SEGMENTATION EXAMPLES ===")
    
    examples = {
        'en': [
            "The capital of France is Paris.",
            "Artificial intelligence is transforming healthcare.",
            "The quick brown fox jumps over the lazy dog.",
        ],
        'he': [
            "הבירה של צרפת היא פריז.",
            "בינה מלאכותית משנה את עולם הרפואה.",
            "ישראל היא מדינה דמוקרטית במזרח התיכון.",
        ],
        'ar': [
            "عاصمة فرنسا هي باريس.",
            "الذكاء الاصطناعي يغير الرعاية الصحية.",
            "اللغة العربية من أقدم اللغات في العالم.",
        ],
        'fa': [
            "پایتخت فرانسه پاریس است.",
            "هوش مصنوعی مراقبت بهداشتی را متحول می‌کند.",
            "زبان فارسی یکی از زیباترین زبان‌های جهان است.",
        ],
    }
    
    results = {}
    for lang, sents in examples.items():
        results[lang] = []
        print(f"\n  [{lang.upper()}]")
        for sent in sents:
            pieces = sp.encode(sent, out_type=str)
            n_tokens = len(pieces)
            n_words = len(sent.split())
            fert = n_tokens / max(n_words, 1)
            segmented = ' | '.join(pieces)
            print(f"    Input ({n_words}w → {n_tokens}t, fert={fert:.1f}): {sent}")
            print(f"    Tokens: {segmented}")
            results[lang].append({
                'input': sent,
                'tokens': pieces,
                'n_tokens': n_tokens,
                'n_words': n_words,
                'fertility': round(fert, 2),
            })
    
    return results


def compare_tokenizers(our_sp, baseline_sp, texts_by_lang, baseline_name="Llama"):
    """Compare our tokenizer against a baseline."""
    print(f"\n=== TOKENIZER COMPARISON: Ours vs {baseline_name} ===")
    
    print(f"\n{'Language':<10} {'Ours tok/w':>10} {'Base tok/w':>10} {'Ours B/t':>10} {'Base B/t':>10} {'Improvement':>12}")
    print("-" * 65)
    
    results = {}
    for lang, texts in texts_by_lang.items():
        our_tokens = sum(len(our_sp.encode(t)) for t in texts)
        base_tokens = sum(len(baseline_sp.encode(t)) for t in texts)
        our_words = sum(len(t.split()) for t in texts)
        our_bytes = sum(len(t.encode('utf-8')) for t in texts)
        
        our_fert = our_tokens / max(our_words, 1)
        base_fert = base_tokens / max(our_words, 1)
        our_bpt = our_bytes / max(our_tokens, 1)
        base_bpt = our_bytes / max(base_tokens, 1)
        improvement = (base_tokens - our_tokens) / max(base_tokens, 1) * 100
        
        results[lang] = {
            'our_fertility': round(our_fert, 3),
            'baseline_fertility': round(base_fert, 3),
            'our_bytes_per_token': round(our_bpt, 2),
            'baseline_bytes_per_token': round(base_bpt, 2),
            'token_reduction_pct': round(improvement, 1),
        }
        print(f"{lang:<10} {our_fert:>10.3f} {base_fert:>10.3f} {our_bpt:>10.2f} {base_bpt:>10.2f} {improvement:>+11.1f}%")
    
    return results


def load_sample_texts(data_dir=None):
    """Load sample texts for each language from available data."""
    texts = defaultdict(list)
    
    # Use Hebrew SFT data if available
    hebrew_dir = '/tmp/hebrew_sft'
    if os.path.isdir(hebrew_dir):
        for fname in ['alpaca_hebrew.jsonl', 'dolly_hebrew.jsonl', 'heq_instruction.jsonl']:
            fpath = os.path.join(hebrew_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    for i, line in enumerate(f):
                        if i >= 500: break
                        try:
                            d = json.loads(line.strip())
                            text = d.get('instruction', '') + ' ' + d.get('input', '') + ' ' + d.get('output', d.get('response', ''))
                            texts['he'].append(text.strip())
                        except:
                            pass
    
    # Use Aya for other languages
    try:
        from datasets import load_dataset
        ds = load_dataset("CohereForAI/aya_dataset", split="train")
        code_map = {'eng': 'en', 'arb': 'ar', 'pes': 'fa'}
        for s in ds:
            code = s['language_code']
            lang = code_map.get(code)
            if lang and len(texts[lang]) < 500:
                texts[lang].append(s['inputs'] + ' ' + s['targets'])
    except:
        # Fallback: use simple test sentences
        texts['en'] = ["The capital of France is Paris. Artificial intelligence is transforming the world."] * 100
        texts['ar'] = ["عاصمة فرنسا هي باريس. الذكاء الاصطناعي يغير العالم."] * 100
        texts['fa'] = ["پایتخت فرانسه پاریس است. هوش مصنوعی جهان را تغییر می‌دهد."] * 100
    
    for lang in texts:
        print(f"  Loaded {len(texts[lang])} texts for {lang}")
    
    return dict(texts)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True, help='Our multilingual tokenizer')
    parser.add_argument('--baseline', default=None, help='Baseline tokenizer for comparison')
    parser.add_argument('--baseline-name', default='Baseline', help='Name for baseline tokenizer')
    parser.add_argument('--output', default='/tmp/tokenizer_analysis.json')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor(args.tokenizer)
    print(f"  Vocab size: {sp.get_piece_size()}")
    
    # 1. Vocabulary composition
    vocab_comp = analyze_vocabulary(sp)
    
    # 2. Load sample texts
    print("\nLoading sample texts...")
    texts = load_sample_texts()
    
    # 3. Fertility analysis
    fertility = fertility_analysis(sp, texts)
    
    # 4. Segmentation examples
    seg_examples = segmentation_examples(sp, texts)
    
    # 5. Baseline comparison (if provided)
    comparison = None
    if args.baseline:
        print(f"\nLoading baseline tokenizer: {args.baseline}")
        baseline_sp = spm.SentencePieceProcessor(args.baseline)
        print(f"  Baseline vocab size: {baseline_sp.get_piece_size()}")
        comparison = compare_tokenizers(sp, baseline_sp, texts, args.baseline_name)
    
    # Save results
    results = {
        'vocab_composition': vocab_comp,
        'fertility': fertility,
        'segmentation_examples': seg_examples,
        'comparison': comparison,
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
