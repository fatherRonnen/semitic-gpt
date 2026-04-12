#!/usr/bin/env python3
"""
Standalone translation evaluation script.
Evaluates multiple models on held-out translation pairs using chrF.
"""
import json, os, sys, random
sys.stdout.reconfigure(line_buffering=True)
import torch
import sentencepiece as spm

sys.path.insert(0, '/tmp/eval')
from train_sft_3b import GPT, VOCAB_SIZE, DIM, DEPTH, N_HEADS, MAX_SEQ_LEN

sp = spm.SentencePieceProcessor('/tmp/eval/multilingual_32k.model')
device = 'cuda'
LANG_NAMES = {'he': 'Hebrew', 'ar': 'Arabic', 'en': 'English', 'fa': 'Persian'}

def load_model(path):
    model = GPT()
    state = torch.load(path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model

@torch.no_grad()
def generate(model, prompt, max_tokens=200):
    ids = sp.encode(prompt)
    ids = torch.tensor([ids], device=device)
    for _ in range(max_tokens):
        if ids.shape[1] >= MAX_SEQ_LEN:
            break
        logits = model(ids)[:, -1, :]
        next_id = logits.argmax(-1, keepdim=True)
        if next_id.item() == 3:
            break
        ids = torch.cat([ids, next_id], dim=1)
    return sp.decode(ids[0].tolist()[len(sp.encode(prompt)):])

def chrf_score(pred, ref):
    def char_ngrams(text, n=3):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    pred_ng = char_ngrams(pred)
    ref_ng = char_ngrams(ref)
    if not pred_ng or not ref_ng:
        return 0.0
    precision = len(pred_ng & ref_ng) / len(pred_ng) if pred_ng else 0
    recall = len(pred_ng & ref_ng) / len(ref_ng) if ref_ng else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def eval_translation(model, model_name, test_pairs, max_per_dir=50):
    results = {}
    for direction, pairs in test_pairs.items():
        if len(pairs) == 0:
            continue
        src_lang, tgt_lang = direction.split('-')
        eval_pairs = pairs[:max_per_dir]
        scores = []
        sample_pred = ''
        sample_ref = ''
        for i, pair in enumerate(eval_pairs):
            prompt = f"<|user|>\nTranslate from {LANG_NAMES.get(src_lang, src_lang)} to {LANG_NAMES.get(tgt_lang, tgt_lang)}:\n{pair['src']}\n<|assistant|>\n"
            pred = generate(model, prompt, max_tokens=200).strip()
            ref = pair['tgt'].strip()
            scores.append(chrf_score(pred, ref))
            if i == 0:
                sample_pred = pred
                sample_ref = ref
        avg = sum(scores) / len(scores) if scores else 0
        results[direction] = {
            'chrF': avg * 100,
            'n_eval': len(eval_pairs),
            'sample_pred': sample_pred[:200],
            'sample_ref': sample_ref[:200],
        }
        print(f"  {model_name} {direction}: chrF={avg*100:.1f}% ({len(eval_pairs)} pairs)")
    return results

# Load test pairs
print("Loading test data...")
test_pairs = {}

for fpath, n_tail in [
    ('/tmp/translation_data/liboaccn_ar_fa.jsonl', 200),
    ('/tmp/translation_data/direct_pairs_arhe_arfa.jsonl', 200),
    ('/tmp/translation_data/translation_train_all.jsonl', 300),
]:
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        items = [json.loads(line) for line in f]
    for item in items[-n_tail:]:
        direction = item['lang']
        if direction not in test_pairs:
            test_pairs[direction] = []
        src_text = item['instruction'].split('\n', 1)[-1].strip()
        test_pairs[direction].append({'src': src_text, 'tgt': item['output']})

print(f"Test directions: {[(d, len(p)) for d, p in sorted(test_pairs.items())]}")

# Find all available models
models = [
    ('D-baseline', '/tmp/sft_v3_runs/D/sft_model.pt'),
    ('G-trans', '/tmp/sft_v3_runs/G-trans/sft_model.pt'),
    ('G-mixed', '/tmp/sft_v3_runs/G-mixed/sft_model.pt'),
    ('G2-trans', '/tmp/sft_v3_runs/G2-trans/sft_model.pt'),
    ('G2-direct', '/tmp/sft_v3_runs/G2-direct/sft_model.pt'),
]

all_results = {}
for name, path in models:
    if not os.path.exists(path):
        print(f"\n⚠️ {name} not found, skipping")
        continue
    print(f"\n=== {name} ===")
    model = load_model(path)
    all_results[name] = eval_translation(model, name, test_pairs)
    del model
    torch.cuda.empty_cache()

with open('/tmp/experiments/exp_translation_eval.json', 'w') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n" + "="*80)
dirs = sorted(set(d for r in all_results.values() for d in r))
header = f"{'Direction':<12}"
for name in [n for n, _ in models if n in all_results]:
    header += f"{name:<15}"
print(header)
for d in dirs:
    row = f"{d:<12}"
    for name in [n for n, _ in models if n in all_results]:
        s = all_results.get(name, {}).get(d, {}).get('chrF', 0)
        row += f"{s:<15.1f}"
    print(row)
