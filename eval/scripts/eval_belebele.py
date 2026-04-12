#!/usr/bin/env python3
"""
Belebele Evaluation for Multilingual 3B GPT

Multiple-choice reading comprehension across 4 languages (en, he, ar, fa).
Uses log-likelihood scoring: for each question, compute P(answer|passage+question)
for all 4 choices, pick the highest.

No SFT needed — works on base models via perplexity ranking.

Usage:
    python eval_belebele.py --checkpoint /path/to/best_model.pt --tokenizer /path/to/multilingual_32k.model
"""

import os, sys, json, argparse, time
import gc
sys.stdout.reconfigure(line_buffering=True)  # Flush prints immediately
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm

# ============ MODEL (must match training) ============
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
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        mlp_dim = ((int(2 * DIM * 4 / 3) + 63) // 64) * 64
        self.blocks = nn.ModuleList([Block(DIM, N_HEADS, mlp_dim) for _ in range(DEPTH)])
        self.ln_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        hd = DIM // N_HEADS
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


# ============ BELEBELE LOADING ============
LANG_MAP = {
    'en': 'eng_Latn',
    'he': 'heb_Hebr',
    'ar': 'arb_Arab',
    'fa': 'pes_Arab',  # Western Farsi
}

def load_belebele(data_dir, lang_code):
    """Load Belebele JSONL file for a language."""
    filename = os.path.join(data_dir, f"{lang_code}.jsonl")
    samples = []
    with open(filename) as f:
        for line in f:
            item = json.loads(line)
            samples.append({
                'passage': item['flores_passage'],
                'question': item['question'],
                'choices': [
                    item['mc_answer1'],
                    item['mc_answer2'],
                    item['mc_answer3'],
                    item['mc_answer4'],
                ],
                'correct': int(item['correct_answer_num']) - 1,  # 0-indexed
            })
    return samples


def score_choice(model, tokenizer, passage, question, answer, device):
    """Score a single answer choice using log-likelihood of the answer tokens
    given the passage+question context."""
    # Format: "Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"
    context = f"Passage: {passage}\nQuestion: {question}\nAnswer: "
    full_text = context + answer

    context_ids = tokenizer.encode(context)
    full_ids = tokenizer.encode(full_text)

    # Truncate to MAX_SEQ_LEN if needed
    if len(full_ids) > MAX_SEQ_LEN:
        # Keep the end (answer portion), truncate context
        overflow = len(full_ids) - MAX_SEQ_LEN
        context_ids = context_ids[overflow:]
        full_ids = full_ids[-MAX_SEQ_LEN:]

    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids)

    # Only score the answer tokens (after context)
    answer_start = len(context_ids) - 1  # -1 because we shifted by 1
    if answer_start < 0:
        answer_start = 0

    answer_logits = logits[0, answer_start:]
    answer_targets = target_ids[0, answer_start:]

    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)

    # Average log-prob (length-normalized)
    return token_log_probs.mean().item()


def evaluate_language(model, tokenizer, samples, device, lang_name):
    """Evaluate all samples for one language."""
    correct = 0
    total = len(samples)

    for i, sample in enumerate(samples):
        scores = []
        for choice in sample['choices']:
            score = score_choice(
                model, tokenizer,
                sample['passage'], sample['question'], choice,
                device
            )
            scores.append(score)

        pred = max(range(4), key=lambda j: scores[j])
        if pred == sample['correct']:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{lang_name}] {i+1}/{total} — accuracy so far: {correct/(i+1)*100:.1f}%")

    accuracy = correct / total * 100
    return accuracy, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--tokenizer', required=True, help='Path to sentencepiece .model file')
    parser.add_argument('--data-dir', default='/tmp/belebele', help='Directory with Belebele JSONL files')
    parser.add_argument('--langs', default='en,he,ar,fa', help='Comma-separated language codes')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='belebele_results.json')
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor(args.tokenizer)

    # Load model
    print(f"Loading model: {args.checkpoint}")
    model = GPT()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    # Strip FSDP/DDP prefixes
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    del ckpt, state_dict, clean_sd  # Free CPU memory
    import gc; gc.collect()
    model = model.to(args.device).eval()
    # Skip torch.compile to save memory on small instances

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e9:.2f}B parameters on {args.device}")

    # Download Belebele if needed
    if not os.path.exists(args.data_dir):
        print("Downloading Belebele dataset...")
        os.makedirs(args.data_dir, exist_ok=True)
        import subprocess
        for lang in args.langs.split(','):
            belebele_code = LANG_MAP[lang]
            url = f"https://huggingface.co/datasets/facebook/belebele/resolve/main/{belebele_code}.jsonl"
            subprocess.run(['curl', '-sL', '-o', f'{args.data_dir}/{belebele_code}.jsonl', url], check=True)
            print(f"  Downloaded {belebele_code}")

    # Evaluate each language
    results = {}
    langs = args.langs.split(',')
    total_correct = 0
    total_samples = 0

    print(f"\n{'='*60}")
    print(f"BELEBELE EVALUATION — Multilingual 3B GPT")
    print(f"{'='*60}\n")

    for lang in langs:
        belebele_code = LANG_MAP[lang]
        filepath = os.path.join(args.data_dir, f"{belebele_code}.jsonl")
        if not os.path.exists(filepath):
            print(f"[SKIP] {lang} — file not found: {filepath}")
            continue

        print(f"\nEvaluating {lang.upper()} ({belebele_code})...")
        samples = load_belebele(args.data_dir, belebele_code)
        accuracy, correct, total = evaluate_language(model, sp, samples, args.device, lang.upper())

        results[lang] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'belebele_code': belebele_code,
        }
        total_correct += correct
        total_samples += total

        print(f"  ✅ {lang.upper()}: {accuracy:.1f}% ({correct}/{total})")

    # Overall
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0
    results['overall'] = {
        'accuracy': overall,
        'correct': total_correct,
        'total': total_samples,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL: {overall:.1f}% ({total_correct}/{total_samples})")
    print(f"Random baseline: 25.0%")
    print(f"{'='*60}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
