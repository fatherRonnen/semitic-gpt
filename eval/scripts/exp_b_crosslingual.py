#!/usr/bin/env python3
"""
Experiment B: Cross-Lingual Transfer Analysis
Evaluates base vs SFT model on Belebele (multiple-choice reading comprehension)
across all 4 languages: Hebrew, Arabic, English, Farsi.

Also runs generation quality probes — short prompts in each language,
comparing fluency/coherence between base and SFT.

Usage:
    python exp_b_crosslingual.py \
        --base-checkpoint /tmp/eval/best_model.pt \
        --sft-checkpoint /tmp/eval/sft_model.pt \
        --tokenizer /tmp/eval/multilingual_32k.model \
        --output /tmp/experiments/exp_b_results.json
"""

import os, sys, json, argparse, time, random, gc, math
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ============ MODEL ARCHITECTURE ============
VOCAB_SIZE = 32000
DIM = 3072
DEPTH = 26
N_HEADS = 24
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, n_heads=N_HEADS):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        mlp_dim = ((int(2 * dim * 4 / 3) + 63) // 64) * 64
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_dim) for _ in range(depth)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        hd = dim // n_heads
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2).float() / hd))
        angles = torch.outer(torch.arange(MAX_SEQ_LEN).float(), freqs)
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.7, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= MAX_SEQ_LEN else idx[:, -MAX_SEQ_LEN:]
            logits = self(idx_cond)[:, -1, :]
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            if next_tok.item() == 3:
                break
        return idx


def load_model(checkpoint_path, device):
    model = GPT()
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    del ckpt, state_dict, clean_sd
    gc.collect()
    model = model.to(device).eval()
    return model


def generate_text(model, sp, prompt, device, max_tokens=100):
    ids = sp.encode(prompt)
    if len(ids) > MAX_SEQ_LEN - max_tokens:
        ids = ids[:MAX_SEQ_LEN - max_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens)
    return sp.decode(output_ids[0].tolist()[len(ids):])


# ============ BELEBELE EVALUATION ============

def download_belebele():
    """Download Belebele dataset using HF datasets library."""
    from datasets import load_dataset

    belebele_dir = '/tmp/eval/belebele'
    os.makedirs(belebele_dir, exist_ok=True)

    lang_map = {
        'heb_Hebr': 'he',
        'arb_Arab': 'ar',
        'eng_Latn': 'en',
        'pes_Arab': 'fa',
    }

    for hf_lang, short_lang in lang_map.items():
        outfile = os.path.join(belebele_dir, f'{short_lang}.jsonl')
        if os.path.exists(outfile) and os.path.getsize(outfile) > 100:
            print(f"  Belebele {short_lang} already exists ({os.path.getsize(outfile)} bytes)")
            continue

        print(f"  Downloading Belebele {hf_lang} → {short_lang}...")
        try:
            ds = load_dataset('facebook/belebele', hf_lang, split='test')
            with open(outfile, 'w') as f:
                for item in ds:
                    # Filter to only serializable fields
                    clean_item = {}
                    for k, v in item.items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            clean_item[k] = v
                        else:
                            clean_item[k] = str(v)
                    json.dump(clean_item, f, ensure_ascii=False)
                    f.write('\n')
            print(f"  ✅ {short_lang}: {len(ds)} samples")
        except Exception as e:
            print(f"  ❌ {short_lang} failed: {e}")

    return belebele_dir


def score_completion_ll(model, sp, context, completion, device):
    """Score P(completion | context) using average log-likelihood."""
    ctx_ids = sp.encode(context)
    comp_ids = sp.encode(completion)
    full_ids = ctx_ids + comp_ids

    if len(full_ids) > MAX_SEQ_LEN:
        overflow = len(full_ids) - MAX_SEQ_LEN
        ctx_ids = ctx_ids[overflow:]
        full_ids = full_ids[-MAX_SEQ_LEN:]

    if len(full_ids) < 2:
        return float('-inf')

    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)

    answer_start = max(len(ctx_ids) - 1, 0)
    answer_logits = logits[0, answer_start:]
    answer_targets = target_ids[0, answer_start:]

    if answer_logits.shape[0] == 0:
        return float('-inf')

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_lp = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    return token_lp.mean().item()


def eval_belebele(model, sp, data_file, lang, device, max_samples=None):
    """Evaluate on Belebele multiple-choice reading comprehension."""
    samples = []
    with open(data_file) as f:
        for line in f:
            samples.append(json.loads(line))

    if max_samples and len(samples) > max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]

    correct = 0
    total = 0

    for item in samples:
        passage = item.get('flores_passage', item.get('passage', ''))
        question = item.get('question', '')
        correct_idx = int(item.get('correct_answer_num', 1)) - 1

        choices = [
            item.get('mc_answer1', ''),
            item.get('mc_answer2', ''),
            item.get('mc_answer3', ''),
            item.get('mc_answer4', ''),
        ]

        context = f"{passage}\n\n{question}\n"

        scores = []
        for choice in choices:
            score = score_completion_ll(model, sp, context, choice, device)
            scores.append(score)

        pred = max(range(4), key=lambda j: scores[j])
        if pred == correct_idx:
            correct += 1
        total += 1

    acc = correct / total * 100 if total > 0 else 0
    random_baseline = 25.0
    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'random_baseline': random_baseline,
        'above_random': acc - random_baseline,
    }


# ============ PERPLEXITY / BPB ============

def compute_bpb(model, sp, texts, device, lang):
    """Compute bits-per-byte on a set of texts."""
    total_nll = 0
    total_tokens = 0
    total_bytes = 0

    for text in texts:
        text_bytes = len(text.encode('utf-8'))
        ids = sp.encode(text)
        if len(ids) < 2:
            continue
        if len(ids) > MAX_SEQ_LEN:
            ids = ids[:MAX_SEQ_LEN]

        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids)

        loss = F.cross_entropy(logits[0], target_ids[0], reduction='sum')
        total_nll += loss.item()
        total_tokens += len(ids) - 1
        total_bytes += text_bytes

    if total_bytes == 0:
        return {'bpb': float('inf'), 'perplexity': float('inf')}

    # BPB = total_nll / (total_bytes * ln(2))
    bpb = total_nll / (total_bytes * math.log(2))
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')

    return {
        'bpb': round(bpb, 4),
        'perplexity': round(ppl, 2),
        'total_tokens': total_tokens,
        'total_bytes': total_bytes,
    }


# ============ GENERATION PROBES ============

GENERATION_PROBES = {
    'he': [
        "ישראל היא מדינה",
        "הטכנולוגיה משנה את העולם",
        "ירושלים היא עיר",
    ],
    'ar': [
        "اللغة العربية هي",
        "التكنولوجيا تغير العالم",
        "القاهرة مدينة",
    ],
    'en': [
        "Artificial intelligence is",
        "The history of the Middle East",
        "Technology is changing",
    ],
    'fa': [
        "زبان فارسی یک",
        "تهران شهری است",
        "فناوری جهان را",
    ],
}


def run_generation_probes(model, sp, device):
    """Run generation probes in all 4 languages."""
    results = {}
    for lang, prompts in GENERATION_PROBES.items():
        lang_results = []
        for prompt in prompts:
            generated = generate_text(model, sp, prompt, device, max_tokens=80)
            lang_results.append({
                'prompt': prompt,
                'generated': generated.strip()[:300],
                'length_tokens': len(sp.encode(generated)),
            })
        results[lang] = lang_results
    return results


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Experiment B: Cross-Lingual Transfer Analysis')
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--sft-checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', default='/tmp/experiments/exp_b_results.json')
    parser.add_argument('--max-belebele', type=int, default=None, help='Max Belebele samples per lang (None=all)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sp = spm.SentencePieceProcessor(args.tokenizer)

    # Download Belebele
    print("Downloading Belebele dataset...")
    belebele_dir = download_belebele()

    # Get BPB texts from Belebele passages (language-specific text samples)
    bpb_texts = {}
    for lang in ['he', 'ar', 'en', 'fa']:
        belebele_file = os.path.join(belebele_dir, f'{lang}.jsonl')
        if os.path.exists(belebele_file):
            texts = []
            with open(belebele_file) as f:
                for line in f:
                    item = json.loads(line)
                    passage = item.get('flores_passage', item.get('passage', ''))
                    if passage:
                        texts.append(passage)
            bpb_texts[lang] = texts[:200]  # Cap at 200 passages for BPB

    results = {'base': {}, 'sft': {}, 'delta': {}}
    start_time = time.time()

    for model_name, ckpt_path in [('base', args.base_checkpoint), ('sft', args.sft_checkpoint)]:
        print(f"\n{'='*60}")
        print(f"Loading {model_name.upper()} model: {ckpt_path}")
        print(f"{'='*60}")
        model = load_model(ckpt_path, args.device)

        model_results = {'belebele': {}, 'bpb': {}, 'generation': {}}

        # Belebele evaluation
        for lang in ['he', 'ar', 'en', 'fa']:
            belebele_file = os.path.join(belebele_dir, f'{lang}.jsonl')
            if not os.path.exists(belebele_file):
                print(f"  [SKIP] Belebele {lang} — not found")
                continue

            print(f"\n  [{model_name.upper()}] Belebele {lang.upper()}...")
            task_start = time.time()
            result = eval_belebele(model, sp, belebele_file, lang, args.device, args.max_belebele)
            result['time_seconds'] = time.time() - task_start
            model_results['belebele'][lang] = result
            print(f"  ✅ Belebele {lang.upper()}: {result['accuracy']:.1f}% ({result['total']} samples, {result['time_seconds']:.0f}s, +{result['above_random']:.1f}% over random)")

        # BPB evaluation
        for lang, texts in bpb_texts.items():
            print(f"\n  [{model_name.upper()}] BPB {lang.upper()}...")
            bpb_result = compute_bpb(model, sp, texts, args.device, lang)
            model_results['bpb'][lang] = bpb_result
            print(f"  ✅ BPB {lang.upper()}: {bpb_result['bpb']:.4f} (PPL: {bpb_result['perplexity']:.1f})")

        # Generation probes
        print(f"\n  [{model_name.upper()}] Generation probes...")
        model_results['generation'] = run_generation_probes(model, sp, args.device)
        for lang, probes in model_results['generation'].items():
            for p in probes:
                print(f"    {lang.upper()}: '{p['prompt']}' → '{p['generated'][:80]}...'")

        results[model_name] = model_results

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute deltas
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY: Cross-Lingual Transfer")
    print(f"{'='*60}")

    print(f"\n--- Belebele Accuracy ---")
    print(f"{'Lang':<6} {'Base':>8} {'SFT':>8} {'Delta':>8}")
    print(f"{'-'*30}")
    for lang in ['he', 'ar', 'en', 'fa']:
        base_acc = results['base']['belebele'].get(lang, {}).get('accuracy', 0)
        sft_acc = results['sft']['belebele'].get(lang, {}).get('accuracy', 0)
        delta = sft_acc - base_acc
        results['delta'][f'belebele_{lang}'] = delta
        print(f"{lang.upper():<6} {base_acc:>7.1f}% {sft_acc:>7.1f}% {delta:>+7.1f}%")

    print(f"\n--- Bits Per Byte ---")
    print(f"{'Lang':<6} {'Base':>8} {'SFT':>8} {'Delta':>8}")
    print(f"{'-'*30}")
    for lang in ['he', 'ar', 'en', 'fa']:
        base_bpb = results['base']['bpb'].get(lang, {}).get('bpb', 0)
        sft_bpb = results['sft']['bpb'].get(lang, {}).get('bpb', 0)
        delta = sft_bpb - base_bpb
        results['delta'][f'bpb_{lang}'] = delta
        print(f"{lang.upper():<6} {base_bpb:>8.4f} {sft_bpb:>8.4f} {delta:>+8.4f}")

    results['total_time_seconds'] = time.time() - start_time

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {results['total_time_seconds']/60:.1f} minutes")


if __name__ == '__main__':
    main()
