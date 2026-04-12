#!/usr/bin/env python3
"""
Experiment A-fix: Hebrew Classification with Generation-Based Scoring
Re-runs sentiment and NLI using instruction-format generation + string matching.
Keeps original QA/trivia/translation/winograd results, replaces sentiment+NLI.

Usage:
    python exp_a_fix_classification.py \
        --base-checkpoint /tmp/eval/best_model.pt \
        --sft-checkpoint /tmp/eval/sft_model.pt \
        --tokenizer /tmp/eval/multilingual_32k.model \
        --data-dir /tmp/eval/hebrew_data \
        --output /tmp/experiments/exp_a_fix_results.json
"""

import os, sys, json, argparse, time, random, gc, re
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ============ MODEL ARCHITECTURE (same as exp_a) ============
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
    def generate(self, idx, max_new_tokens=50, temperature=0.3, top_k=20):
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
            if next_tok.item() == 3:  # EOS
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


def generate_text(model, sp, prompt, device, max_tokens=50):
    ids = sp.encode(prompt)
    if len(ids) > MAX_SEQ_LEN - max_tokens:
        ids = ids[:MAX_SEQ_LEN - max_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.3, top_k=20)
    return sp.decode(output_ids[0].tolist()[len(ids):])


def load_instruction_data(filepath, max_samples=300):
    samples = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            if 'messages' in item:
                msgs = item['messages']
                user_msg = next((m['content'] for m in msgs if m['role'] == 'user'), '')
                asst_msg = next((m['content'] for m in msgs if m['role'] == 'assistant'), '')
                samples.append({'input': user_msg, 'output': asst_msg})
            elif 'instruction' in item:
                inp = item.get('instruction', '')
                if item.get('input'):
                    inp += '\n' + item['input']
                samples.append({'input': inp, 'output': item.get('output', '')})
    random.shuffle(samples)
    return samples[:max_samples]


# ============ GENERATION-BASED CLASSIFICATION ============

SENTIMENT_LABELS = {
    'חיובי': 'positive', 'שלילי': 'negative', 'ניטרלי': 'neutral',
    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral',
    'חיוב': 'positive', 'שליל': 'negative',
}

NLI_LABELS = {
    'היסק': 'entailment', 'סתירה': 'contradiction', 'ניטרלי': 'neutral',
    'entailment': 'entailment', 'contradiction': 'contradiction', 'neutral': 'neutral',
    'נכון': 'entailment', 'שגוי': 'contradiction',
}


def classify_by_generation(model, sp, prompt, label_map, device):
    """Generate a response and extract classification label via string matching."""
    # Use instruction format matching the SFT training
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    generated = generate_text(model, sp, full_prompt, device, max_tokens=30)
    generated_lower = generated.strip().lower()

    # Also try without special tokens (for base model)
    if not generated.strip():
        full_prompt_alt = f"{prompt}\nתשובה: "
        generated = generate_text(model, sp, full_prompt_alt, device, max_tokens=30)
        generated_lower = generated.strip().lower()

    # Match against label map
    for label_text, label_class in label_map.items():
        if label_text in generated or label_text in generated_lower:
            return label_class, generated.strip()

    return None, generated.strip()


def extract_true_label(output_text, label_map):
    """Extract the true label from the gold output."""
    output_lower = output_text.strip().lower()
    for label_text, label_class in label_map.items():
        if label_text in output_text or label_text in output_lower:
            return label_class
    return None


def eval_classification_gen(model, sp, data_path, label_map, task_name, device, max_samples=200):
    """Evaluate classification using generation + string matching."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'accuracy': 0, 'total': 0, 'error': 'no data'}

    correct = 0
    total = 0
    predictions = []

    for i, s in enumerate(samples):
        true_label = extract_true_label(s['output'], label_map)
        if true_label is None:
            continue

        pred_label, gen_text = classify_by_generation(model, sp, s['input'], label_map, device)

        if pred_label == true_label:
            correct += 1
        total += 1
        predictions.append({
            'true': true_label,
            'pred': pred_label,
            'gen': gen_text[:100],
        })

        if total % 50 == 0:
            acc = correct / total * 100
            print(f"    {task_name}: {total} done, {correct}/{total} ({acc:.1f}%)")

    acc = correct / total * 100 if total > 0 else 0
    return {
        'accuracy': acc,
        'correct': correct,
        'total': total,
        'sample_predictions': predictions[:10],  # Save first 10 for inspection
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--sft-checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--data-dir', default='/tmp/eval/hebrew_data')
    parser.add_argument('--output', default='/tmp/experiments/exp_a_fix_results.json')
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sp = spm.SentencePieceProcessor(args.tokenizer)

    results = {'base': {}, 'sft': {}, 'delta': {}}
    start_time = time.time()

    tasks = {
        'sentiment': {
            'file': 'sentiment_instruction.jsonl',
            'labels': SENTIMENT_LABELS,
        },
        'nli': {
            'file': 'hebnli_instruction.jsonl',
            'labels': NLI_LABELS,
        },
    }

    for model_name, ckpt_path in [('base', args.base_checkpoint), ('sft', args.sft_checkpoint)]:
        print(f"\n{'='*60}")
        print(f"Loading {model_name.upper()} model: {ckpt_path}")
        print(f"{'='*60}")
        model = load_model(ckpt_path, args.device)

        for task_name, task_cfg in tasks.items():
            data_file = os.path.join(args.data_dir, task_cfg['file'])
            if not os.path.exists(data_file):
                print(f"  [SKIP] {task_name} — {data_file} not found")
                results[model_name][task_name] = {'error': 'file not found'}
                continue

            print(f"\n  [{model_name.upper()}] Task: {task_name} (generation-based)")
            task_start = time.time()
            result = eval_classification_gen(
                model, sp, data_file, task_cfg['labels'],
                task_name, args.device, args.max_samples
            )
            result['time_seconds'] = time.time() - task_start
            result['method'] = 'generation_string_match'
            results[model_name][task_name] = result
            print(f"  ✅ {task_name}: {result['accuracy']:.1f}% ({result['total']} samples, {result['time_seconds']:.0f}s)")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute deltas
    print(f"\n{'='*60}")
    print("RESULTS: Generation-Based Classification (Base vs SFT)")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Base':>8} {'SFT':>8} {'Delta':>8}")
    print(f"{'-'*39}")

    for task_name in tasks:
        base_val = results['base'].get(task_name, {}).get('accuracy', 0)
        sft_val = results['sft'].get(task_name, {}).get('accuracy', 0)
        delta = sft_val - base_val
        results['delta'][task_name] = delta
        print(f"{task_name:<15} {base_val:>7.1f}% {sft_val:>7.1f}% {delta:>+7.1f}%")

    results['total_time_seconds'] = time.time() - start_time

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {results['total_time_seconds']/60:.1f} minutes")


if __name__ == '__main__':
    main()
