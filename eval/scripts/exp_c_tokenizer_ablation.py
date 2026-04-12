#!/usr/bin/env python3
"""
Experiment C: Tokenizer Ablation Study
Compares our custom 32K multilingual tokenizer vs Llama-3 baseline tokenizer.

Metrics:
  1. Fertility (tokens per word) across HE, AR, EN, FA
  2. Bytes-per-token efficiency
  3. BPB with a small 30M model trained on each tokenizer (if checkpoints available)
     OR analytical comparison using existing Belebele text

Usage:
    python exp_c_tokenizer_ablation.py \
        --our-tokenizer /tmp/eval/multilingual_32k.model \
        --output /tmp/experiments/exp_c_results.json
"""

import os, sys, json, argparse, time, math, re
sys.stdout.reconfigure(line_buffering=True)

# ============ SAMPLE TEXTS FOR FERTILITY ANALYSIS ============
# Representative samples in each language — we'll also use Belebele if available

SAMPLE_TEXTS = {
    'he': [
        "ירושלים היא בירת ישראל ואחת הערים העתיקות בעולם. היא ממוקמת בהרי יהודה, בין הים התיכון לים המלח.",
        "הטכנולוגיה המודרנית משנה את פני החברה הישראלית בקצב מהיר. חברות הייטק רבות פועלות בתל אביב ובהרצליה.",
        "מערכת החינוך בישראל כוללת חינוך יסודי, חטיבת ביניים ותיכון. הלימודים חובה עד גיל שמונה עשרה.",
        "המטבח הישראלי מושפע ממסורות קולינריות מגוונות מהמזרח התיכון, צפון אפריקה ואירופה.",
        "ים המלח הוא הנקודה הנמוכה ביותר על פני כדור הארץ ומהווה אתר תיירותי פופולרי.",
    ],
    'ar': [
        "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدينة في العالم العربي. تقع على ضفاف نهر النيل.",
        "التعليم في الوطن العربي يواجه تحديات كثيرة تتعلق بالجودة والوصول والتمويل.",
        "الأدب العربي له تاريخ طويل يمتد لأكثر من ألف وخمسمائة عام من الشعر والنثر.",
        "التكنولوجيا الحديثة تغير حياة الملايين في منطقة الشرق الأوسط وشمال أفريقيا.",
        "اللغة العربية هي واحدة من أكثر اللغات انتشاراً في العالم ولغة القرآن الكريم.",
    ],
    'en': [
        "Jerusalem is one of the oldest cities in the world and holds significance for three major religions.",
        "Modern technology is transforming societies across the Middle East at an unprecedented pace.",
        "The education system faces challenges related to quality, access, and equitable funding worldwide.",
        "Artificial intelligence and machine learning are revolutionizing how we process and understand language.",
        "The Dead Sea is the lowest point on Earth's surface and attracts millions of tourists each year.",
    ],
    'fa': [
        "تهران پایتخت ایران و بزرگترین شهر این کشور است. این شهر در دامنه رشته‌کوه البرز قرار دارد.",
        "ادبیات فارسی یکی از غنی‌ترین ادبیات جهان است و شاعرانی مانند حافظ و فردوسی را به جهان معرفی کرده.",
        "آموزش و پرورش در ایران شامل دوره‌های ابتدایی، راهنمایی و دبیرستان می‌شود.",
        "فناوری اطلاعات در سال‌های اخیر رشد چشمگیری در ایران داشته است.",
        "زبان فارسی با الفبای عربی نوشته می‌شود اما ساختار دستوری متفاوتی دارد.",
    ],
}


def analyze_tokenizer(sp, texts, lang):
    """Compute fertility and efficiency metrics for a tokenizer on given texts."""
    total_tokens = 0
    total_words = 0
    total_chars = 0
    total_bytes = 0
    token_lengths = []

    for text in texts:
        ids = sp.encode(text)
        words = text.split()
        total_tokens += len(ids)
        total_words += len(words)
        total_chars += len(text)
        total_bytes += len(text.encode('utf-8'))

        # Token length distribution
        pieces = sp.encode(text, out_type=str)
        for piece in pieces:
            token_lengths.append(len(piece))

    fertility = total_tokens / total_words if total_words > 0 else 0
    bytes_per_token = total_bytes / total_tokens if total_tokens > 0 else 0
    chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0

    # Token length stats
    avg_token_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    single_char_pct = sum(1 for l in token_lengths if l <= 1) / len(token_lengths) * 100 if token_lengths else 0

    return {
        'fertility': round(fertility, 3),
        'bytes_per_token': round(bytes_per_token, 3),
        'chars_per_token': round(chars_per_token, 3),
        'total_tokens': total_tokens,
        'total_words': total_words,
        'total_bytes': total_bytes,
        'avg_token_length_chars': round(avg_token_len, 2),
        'single_char_token_pct': round(single_char_pct, 1),
    }


def load_belebele_texts(belebele_dir, lang, max_texts=100):
    """Load Belebele passages for a language."""
    filepath = os.path.join(belebele_dir, f'{lang}.jsonl')
    if not os.path.exists(filepath):
        return []
    texts = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            passage = item.get('flores_passage', item.get('passage', ''))
            if passage:
                texts.append(passage)
            if len(texts) >= max_texts:
                break
    return texts


def vocab_coverage_analysis(sp, name):
    """Analyze vocabulary composition."""
    total = sp.get_piece_size()

    # Count scripts in vocabulary
    script_counts = {'hebrew': 0, 'arabic': 0, 'latin': 0, 'other': 0}
    for i in range(total):
        piece = sp.id_to_piece(i)
        piece_clean = piece.replace('▁', '')
        if not piece_clean:
            continue
        # Check first real character
        for ch in piece_clean:
            if '\u0590' <= ch <= '\u05FF':
                script_counts['hebrew'] += 1
                break
            elif '\u0600' <= ch <= '\u06FF' or '\uFB50' <= ch <= '\uFDFF':
                script_counts['arabic'] += 1
                break
            elif ch.isascii() and ch.isalpha():
                script_counts['latin'] += 1
                break
            else:
                script_counts['other'] += 1
                break

    return {
        'name': name,
        'vocab_size': total,
        'script_distribution': script_counts,
        'script_pct': {k: round(v/total*100, 1) for k, v in script_counts.items()},
    }


def compare_tokenization_examples(sp_ours, sp_baseline, texts, lang):
    """Show side-by-side tokenization examples."""
    examples = []
    for text in texts[:2]:  # Just 2 examples per language
        ours_pieces = sp_ours.encode(text, out_type=str)
        baseline_pieces = sp_baseline.encode(text, out_type=str) if sp_baseline else ['N/A']
        examples.append({
            'text': text[:100] + ('...' if len(text) > 100 else ''),
            'ours_tokens': len(ours_pieces),
            'ours_pieces': ' '.join(ours_pieces[:20]) + ('...' if len(ours_pieces) > 20 else ''),
            'baseline_tokens': len(baseline_pieces),
            'baseline_pieces': ' '.join(baseline_pieces[:20]) + ('...' if len(baseline_pieces) > 20 else ''),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description='Experiment C: Tokenizer Ablation')
    parser.add_argument('--our-tokenizer', required=True, help='Path to our 32K multilingual tokenizer')
    parser.add_argument('--baseline-tokenizer', default=None, help='Path to Llama/baseline tokenizer (optional)')
    parser.add_argument('--belebele-dir', default='/tmp/eval/belebele', help='Belebele data directory')
    parser.add_argument('--output', default='/tmp/experiments/exp_c_results.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    import sentencepiece as spm

    print("Loading our tokenizer...")
    sp_ours = spm.SentencePieceProcessor(args.our_tokenizer)

    sp_baseline = None
    if args.baseline_tokenizer and os.path.exists(args.baseline_tokenizer):
        print("Loading baseline tokenizer...")
        sp_baseline = spm.SentencePieceProcessor(args.baseline_tokenizer)
    else:
        # Try to download Llama tokenizer
        print("Attempting to load Llama tokenizer via transformers...")
        try:
            from transformers import AutoTokenizer
            llama_tok = AutoTokenizer.from_pretrained(
                'meta-llama/Llama-3.2-1B',
                trust_remote_code=True,
            )
            print(f"  Loaded Llama tokenizer: vocab_size={llama_tok.vocab_size}")
        except Exception as e:
            print(f"  Could not load Llama tokenizer: {e}")
            print("  Will compare with analytical baseline instead.")
            llama_tok = None

    results = {
        'our_tokenizer': {},
        'baseline_tokenizer': {},
        'comparison': {},
        'vocab_analysis': {},
        'examples': {},
    }

    # Vocab analysis
    print("\n=== Vocabulary Analysis ===")
    results['vocab_analysis']['ours'] = vocab_coverage_analysis(sp_ours, 'multilingual_32k')
    print(f"  Our tokenizer: {results['vocab_analysis']['ours']['vocab_size']} tokens")
    print(f"  Script distribution: {results['vocab_analysis']['ours']['script_pct']}")

    if sp_baseline:
        results['vocab_analysis']['baseline'] = vocab_coverage_analysis(sp_baseline, 'baseline')
        print(f"  Baseline tokenizer: {results['vocab_analysis']['baseline']['vocab_size']} tokens")

    # Fertility analysis on sample texts
    print("\n=== Fertility Analysis (Sample Texts) ===")
    for lang in ['he', 'ar', 'en', 'fa']:
        texts = SAMPLE_TEXTS[lang]

        # Also add Belebele texts if available
        belebele_texts = load_belebele_texts(args.belebele_dir, lang)
        if belebele_texts:
            texts = texts + belebele_texts[:50]
            print(f"  {lang.upper()}: {len(SAMPLE_TEXTS[lang])} sample + {min(50, len(belebele_texts))} Belebele texts")
        else:
            print(f"  {lang.upper()}: {len(texts)} sample texts only")

        ours_metrics = analyze_tokenizer(sp_ours, texts, lang)
        results['our_tokenizer'][lang] = ours_metrics
        print(f"  Our tokenizer — fertility: {ours_metrics['fertility']:.3f}, bytes/token: {ours_metrics['bytes_per_token']:.3f}")

        if sp_baseline:
            baseline_metrics = analyze_tokenizer(sp_baseline, texts, lang)
            results['baseline_tokenizer'][lang] = baseline_metrics
            print(f"  Baseline — fertility: {baseline_metrics['fertility']:.3f}, bytes/token: {baseline_metrics['bytes_per_token']:.3f}")

            # Efficiency gain
            fert_ratio = baseline_metrics['fertility'] / ours_metrics['fertility'] if ours_metrics['fertility'] > 0 else 0
            results['comparison'][lang] = {
                'fertility_ratio': round(fert_ratio, 3),
                'fertility_improvement_pct': round((fert_ratio - 1) * 100, 1),
                'our_fertility': ours_metrics['fertility'],
                'baseline_fertility': baseline_metrics['fertility'],
            }
            print(f"  Ratio: {fert_ratio:.3f}x ({(fert_ratio-1)*100:.1f}% more efficient)")

        # Tokenization with Llama HF tokenizer (if no SPM baseline)
        if not sp_baseline and llama_tok is not None:
            llama_tokens = 0
            llama_words = 0
            for text in texts:
                ids = llama_tok.encode(text)
                llama_tokens += len(ids)
                llama_words += len(text.split())
            llama_fertility = llama_tokens / llama_words if llama_words > 0 else 0

            results['baseline_tokenizer'][lang] = {
                'fertility': round(llama_fertility, 3),
                'total_tokens': llama_tokens,
                'total_words': llama_words,
                'tokenizer': 'llama-3.2-1b',
            }

            fert_ratio = llama_fertility / ours_metrics['fertility'] if ours_metrics['fertility'] > 0 else 0
            results['comparison'][lang] = {
                'fertility_ratio': round(fert_ratio, 3),
                'fertility_improvement_pct': round((fert_ratio - 1) * 100, 1),
                'our_fertility': ours_metrics['fertility'],
                'baseline_fertility': round(llama_fertility, 3),
            }
            print(f"  Llama baseline — fertility: {llama_fertility:.3f}, ratio: {fert_ratio:.3f}x")

    # Tokenization examples
    print("\n=== Tokenization Examples ===")
    for lang in ['he', 'ar', 'en', 'fa']:
        texts = SAMPLE_TEXTS[lang]
        examples = compare_tokenization_examples(sp_ours, sp_baseline, texts, lang)
        results['examples'][lang] = examples
        for ex in examples:
            print(f"  [{lang.upper()}] '{ex['text']}'")
            print(f"    Ours: {ex['ours_tokens']} tokens — {ex['ours_pieces']}")
            if sp_baseline:
                print(f"    Base: {ex['baseline_tokens']} tokens — {ex['baseline_pieces']}")

    # Summary table
    print(f"\n{'='*60}")
    print("TOKENIZER ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Lang':<6} {'Ours Fert':>10} {'Base Fert':>10} {'Ratio':>8} {'Improvement':>12}")
    print(f"{'-'*46}")
    for lang in ['he', 'ar', 'en', 'fa']:
        ours_f = results['our_tokenizer'].get(lang, {}).get('fertility', 0)
        base_f = results['baseline_tokenizer'].get(lang, {}).get('fertility', 0)
        comp = results['comparison'].get(lang, {})
        ratio = comp.get('fertility_ratio', 0)
        improve = comp.get('fertility_improvement_pct', 0)
        print(f"{lang.upper():<6} {ours_f:>10.3f} {base_f:>10.3f} {ratio:>7.3f}x {improve:>+10.1f}%")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
