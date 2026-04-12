#!/usr/bin/env python3
"""
Experiment B: Cross-lingual Transfer Evaluation
Runs Belebele on base model AND SFT model, comparing transfer across languages.
Measures: base → multilingual-SFT for all 4 languages.
"""

import os, sys, json, argparse, time, gc
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def load_model(path, device):
    """Load model from checkpoint."""
    print(f"Loading model: {path}")
    model = GPT()
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    del ckpt, state_dict, clean_sd
    gc.collect()
    model = model.to(device).eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {param_count/1e9:.2f}B parameters on {device}")
    return model


LANG_MAP = {'en': 'eng_Latn', 'he': 'heb_Hebr', 'ar': 'arb_Arab', 'fa': 'pes_Arab'}


def load_belebele(data_dir, lang_code):
    """Load Belebele JSONL file."""
    filename = os.path.join(data_dir, f"{lang_code}.jsonl")
    samples = []
    with open(filename) as f:
        for line in f:
            item = json.loads(line)
            samples.append({
                'passage': item['flores_passage'],
                'question': item['question'],
                'choices': [item['mc_answer1'], item['mc_answer2'], item['mc_answer3'], item['mc_answer4']],
                'correct': int(item['correct_answer_num']) - 1,
            })
    return samples


def score_choice(model, tokenizer, passage, question, answer, device):
    """Score a single answer choice using log-likelihood."""
    context = f"Passage: {passage}\nQuestion: {question}\nAnswer: "
    full_text = context + answer
    context_ids = tokenizer.encode(context)
    full_ids = tokenizer.encode(full_text)
    if len(full_ids) > MAX_SEQ_LEN:
        overflow = len(full_ids) - MAX_SEQ_LEN
        context_ids = context_ids[overflow:]
        full_ids = full_ids[-MAX_SEQ_LEN:]
    input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids)
    answer_start = max(len(context_ids) - 1, 0)
    answer_logits = logits[0, answer_start:]
    answer_targets = target_ids[0, answer_start:]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)
    return token_log_probs.mean().item()


def evaluate_language(model, tokenizer, samples, device, lang_name):
    """Evaluate all samples for one language."""
    correct = 0
    total = len(samples)
    for i, sample in enumerate(samples):
        scores = [score_choice(model, tokenizer, sample['passage'], sample['question'], c, device)
                  for c in sample['choices']]
        pred = max(range(4), key=lambda j: scores[j])
        if pred == sample['correct']:
            correct += 1
        if (i + 1) % 100 == 0:
            print(f"  [{lang_name}] {i+1}/{total} — acc: {correct/(i+1)*100:.1f}%")
    accuracy = correct / total * 100
    return accuracy, correct, total


def generate_text(model, tokenizer, prompt, device, max_tokens=100):
    """Generate text from a prompt."""
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            if input_ids.shape[1] >= MAX_SEQ_LEN:
                break
            logits = model(input_ids)
            next_logits = logits[0, -1]
            # Temperature sampling with top-k
            top_k = 50
            temperature = 0.8
            next_logits = next_logits / temperature
            values, indices = torch.topk(next_logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_token = indices[torch.multinomial(probs, 1)]
            if next_token.item() == tokenizer.eos_id():
                break
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--sft-checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--data-dir', default='/tmp/belebele')
    parser.add_argument('--langs', default='en,he,ar,fa')
    parser.add_argument('--output', default='/tmp/results/exp_b_belebele_transfer.json')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(args.tokenizer)
    langs = args.langs.split(',')
    results = {'base': {}, 'sft': {}, 'generation': {'base': {}, 'sft': {}}}

    # Generation prompts per language
    gen_prompts = {
        'en': "The capital of France is",
        'he': "בירת צרפת היא",
        'ar': "عاصمة فرنسا هي",
        'fa': "پایتخت فرانسه"
    }

    # ---- PHASE 1: BASE MODEL ----
    print("\n" + "="*60)
    print("PHASE 1: BASE MODEL EVALUATION")
    print("="*60)
    model = load_model(args.base_checkpoint, args.device)

    # Belebele
    for lang in langs:
        belebele_code = LANG_MAP[lang]
        filepath = os.path.join(args.data_dir, f"{belebele_code}.jsonl")
        if not os.path.exists(filepath):
            print(f"  [SKIP] {lang} — missing {filepath}")
            continue
        print(f"\n  Evaluating BASE on {lang.upper()}...")
        samples = load_belebele(args.data_dir, belebele_code)
        acc, correct, total = evaluate_language(model, sp, samples, args.device, f"BASE-{lang.upper()}")
        results['base'][lang] = {'accuracy': acc, 'correct': correct, 'total': total}
        print(f"  ✅ BASE {lang.upper()}: {acc:.1f}% ({correct}/{total})")

    # Generation
    for lang, prompt in gen_prompts.items():
        print(f"\n  Generating BASE [{lang.upper()}]: {prompt}")
        text = generate_text(model, sp, prompt, args.device, max_tokens=80)
        results['generation']['base'][lang] = {'prompt': prompt, 'output': text}
        print(f"    → {text[:200]}")

    # Free base model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- PHASE 2: SFT MODEL ----
    print("\n" + "="*60)
    print("PHASE 2: SFT MODEL EVALUATION")
    print("="*60)
    model = load_model(args.sft_checkpoint, args.device)

    # Belebele
    for lang in langs:
        belebele_code = LANG_MAP[lang]
        filepath = os.path.join(args.data_dir, f"{belebele_code}.jsonl")
        if not os.path.exists(filepath):
            continue
        print(f"\n  Evaluating SFT on {lang.upper()}...")
        samples = load_belebele(args.data_dir, belebele_code)
        acc, correct, total = evaluate_language(model, sp, samples, args.device, f"SFT-{lang.upper()}")
        results['sft'][lang] = {'accuracy': acc, 'correct': correct, 'total': total}
        print(f"  ✅ SFT {lang.upper()}: {acc:.1f}% ({correct}/{total})")

    # Generation
    for lang, prompt in gen_prompts.items():
        print(f"\n  Generating SFT [{lang.upper()}]: {prompt}")
        text = generate_text(model, sp, prompt, args.device, max_tokens=80)
        results['generation']['sft'][lang] = {'prompt': prompt, 'output': text}
        print(f"    → {text[:200]}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- SUMMARY ----
    print("\n" + "="*60)
    print("EXPERIMENT B: CROSS-LINGUAL TRANSFER SUMMARY")
    print("="*60)
    print(f"\n{'Language':<8} {'Base %':>8} {'SFT %':>8} {'Delta':>8}")
    print("-" * 35)
    base_total_c, base_total_n = 0, 0
    sft_total_c, sft_total_n = 0, 0
    for lang in langs:
        base_acc = results['base'].get(lang, {}).get('accuracy', 0)
        sft_acc = results['sft'].get(lang, {}).get('accuracy', 0)
        delta = sft_acc - base_acc
        base_total_c += results['base'].get(lang, {}).get('correct', 0)
        base_total_n += results['base'].get(lang, {}).get('total', 0)
        sft_total_c += results['sft'].get(lang, {}).get('correct', 0)
        sft_total_n += results['sft'].get(lang, {}).get('total', 0)
        print(f"{lang.upper():<8} {base_acc:>7.1f}% {sft_acc:>7.1f}% {delta:>+7.1f}%")

    base_overall = base_total_c / max(base_total_n, 1) * 100
    sft_overall = sft_total_c / max(sft_total_n, 1) * 100
    print(f"{'OVERALL':<8} {base_overall:>7.1f}% {sft_overall:>7.1f}% {sft_overall-base_overall:>+7.1f}%")
    print(f"Random baseline: 25.0%")

    results['summary'] = {
        'base_overall': base_overall,
        'sft_overall': sft_overall,
        'delta': sft_overall - base_overall
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
