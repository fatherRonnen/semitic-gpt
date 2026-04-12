#!/usr/bin/env python3
"""
Experiment A: Hebrew Downstream Evaluation
Compares base model vs multilingual-SFT across 6 Hebrew tasks.

Tasks: Sentiment, NLI, QA, Winograd, Trivia, Translation
Method: Log-likelihood scoring for classification, generation for trivia/translation

Usage:
    python exp_a_hebrew_downstream.py \
        --base-checkpoint /tmp/eval/best_model.pt \
        --sft-checkpoint /tmp/eval/sft_model.pt \
        --tokenizer /tmp/eval/multilingual_32k.model \
        --data-dir /tmp/eval/hebrew_data \
        --output /tmp/experiments/exp_a_results.json
"""

import os, sys, json, argparse, time, random, gc
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
            if next_tok.item() == 3:  # EOS
                break
        return idx


def load_model(checkpoint_path, device):
    """Load model from checkpoint, return model on device."""
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


def compute_log_likelihood(model, sp, text, device):
    """Compute average log-likelihood of text."""
    ids = sp.encode(text)
    if len(ids) < 2:
        return float('-inf')
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(2, target_ids.unsqueeze(2)).squeeze(2)
    return token_lp.mean().item()


def score_completion(model, sp, context, completion, device):
    """Score P(completion | context) using log-likelihood."""
    ctx_ids = sp.encode(context)
    full_ids = sp.encode(context + completion)
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


def generate_text(model, sp, prompt, device, max_tokens=100):
    """Generate text given a prompt."""
    ids = sp.encode(prompt)
    if len(ids) > MAX_SEQ_LEN - max_tokens:
        ids = ids[:MAX_SEQ_LEN - max_tokens]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.7)
    return sp.decode(output_ids[0].tolist()[len(ids):])


# ============ TASK EVALUATIONS ============

def load_instruction_data(filepath, max_samples=300):
    """Load instruction JSONL, extract input/output pairs."""
    samples = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            # SFT format: {"instruction": ..., "input": ..., "output": ...}
            # or {"messages": [...]}
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


def eval_sentiment(model, sp, data_path, device, max_samples=300):
    """Sentiment classification: positive/negative/neutral via log-likelihood."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'accuracy': 0, 'total': 0, 'error': 'no data'}
    
    labels = ['חיובי', 'שלילי', 'ניטרלי']  # positive, negative, neutral
    correct = 0
    total = 0
    
    for s in samples:
        # Try to extract true label from output
        true_label = None
        output_lower = s['output'].strip()
        for i, label in enumerate(labels):
            if label in output_lower:
                true_label = i
                break
        if true_label is None:
            continue
        
        # Score each label
        scores = []
        for label in labels:
            score = score_completion(model, sp, s['input'] + '\nתשובה: ', label, device)
            scores.append(score)
        
        pred = max(range(len(labels)), key=lambda j: scores[j])
        if pred == true_label:
            correct += 1
        total += 1
    
    acc = correct / total * 100 if total > 0 else 0
    return {'accuracy': acc, 'correct': correct, 'total': total}


def eval_nli(model, sp, data_path, device, max_samples=300):
    """NLI: entailment/contradiction/neutral via log-likelihood."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'accuracy': 0, 'total': 0, 'error': 'no data'}
    
    labels = ['היסק', 'סתירה', 'ניטרלי']  # entailment, contradiction, neutral
    correct = 0
    total = 0
    
    for s in samples:
        true_label = None
        output = s['output'].strip()
        for i, label in enumerate(labels):
            if label in output:
                true_label = i
                break
        if true_label is None:
            continue
        
        scores = []
        for label in labels:
            score = score_completion(model, sp, s['input'] + '\nתשובה: ', label, device)
            scores.append(score)
        
        pred = max(range(len(labels)), key=lambda j: scores[j])
        if pred == true_label:
            correct += 1
        total += 1
    
    acc = correct / total * 100 if total > 0 else 0
    return {'accuracy': acc, 'correct': correct, 'total': total}


def eval_qa(model, sp, data_path, device, max_samples=300):
    """QA: generate answer, check if gold answer is contained."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'accuracy': 0, 'total': 0, 'error': 'no data'}
    
    correct = 0
    total = 0
    
    for s in samples:
        gold = s['output'].strip()
        if not gold:
            continue
        generated = generate_text(model, sp, s['input'] + '\nתשובה: ', device, max_tokens=80)
        # Check if gold answer (or significant part) appears in generation
        gold_words = [w for w in gold.split() if len(w) > 2]
        if gold_words:
            matches = sum(1 for w in gold_words if w in generated)
            if matches / len(gold_words) >= 0.3:  # 30% word overlap
                correct += 1
        total += 1
        
        if total % 50 == 0:
            print(f"    QA: {total} done, {correct}/{total} correct so far")
    
    acc = correct / total * 100 if total > 0 else 0
    return {'accuracy': acc, 'correct': correct, 'total': total}


def eval_winograd(model, sp, data_path, device, max_samples=300):
    """Winograd-style coreference: score two completions, pick lower perplexity."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'accuracy': 0, 'total': 0, 'error': 'no data'}
    
    correct = 0
    total = 0
    
    for s in samples:
        gold = s['output'].strip()
        if not gold:
            continue
        # Score gold answer vs random other answer
        score_gold = score_completion(model, sp, s['input'] + '\nתשובה: ', gold, device)
        # Use a dummy wrong answer
        wrong = 'לא ידוע'  # "unknown"
        score_wrong = score_completion(model, sp, s['input'] + '\nתשובה: ', wrong, device)
        
        if score_gold > score_wrong:
            correct += 1
        total += 1
    
    acc = correct / total * 100 if total > 0 else 0
    return {'accuracy': acc, 'correct': correct, 'total': total}


def eval_trivia(model, sp, data_path, device, max_samples=200):
    """Trivia: generate answer to question, check if correct answer appears."""
    # Reuse QA samples as trivia
    return eval_qa(model, sp, data_path, device, max_samples)


def eval_translation(model, sp, data_path, device, max_samples=200):
    """Translation HE↔EN: generate translation, compute word overlap (pseudo-BLEU)."""
    samples = load_instruction_data(data_path, max_samples)
    if not samples:
        return {'score': 0, 'total': 0, 'error': 'no data'}
    
    total_overlap = 0
    total = 0
    
    for s in samples:
        gold = s['output'].strip()
        if not gold:
            continue
        generated = generate_text(model, sp, s['input'] + '\nתרגום: ', device, max_tokens=120)
        
        # Simple word overlap as pseudo-BLEU
        gold_words = set(gold.lower().split())
        gen_words = set(generated.lower().split())
        if gold_words:
            overlap = len(gold_words & gen_words) / len(gold_words)
            total_overlap += overlap
        total += 1
        
        if total % 50 == 0:
            print(f"    Translation: {total} done, avg overlap: {total_overlap/total:.3f}")
    
    avg_overlap = total_overlap / total if total > 0 else 0
    return {'word_overlap': avg_overlap * 100, 'total': total}


# ============ MAIN ============

TASKS = {
    'sentiment': {
        'file': 'sentiment_instruction.jsonl',
        'func': eval_sentiment,
        'metric': 'accuracy',
    },
    'nli': {
        'file': 'hebnli_instruction.jsonl',
        'func': eval_nli,
        'metric': 'accuracy',
    },
    'qa': {
        'file': 'heq_instruction.jsonl',
        'func': eval_qa,
        'metric': 'accuracy',
    },
    'winograd': {
        'file': 'winograd_instruction.jsonl',
        'func': eval_winograd,
        'metric': 'accuracy',
    },
    'trivia': {
        'file': 'heq_instruction.jsonl',  # Reuse QA data
        'func': eval_trivia,
        'metric': 'accuracy',
    },
    'translation': {
        'file': 'translation_instruction.jsonl',
        'func': eval_translation,
        'metric': 'word_overlap',
    },
}


def main():
    parser = argparse.ArgumentParser(description='Experiment A: Hebrew Downstream Evaluation')
    parser.add_argument('--base-checkpoint', required=True)
    parser.add_argument('--sft-checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--data-dir', default='/tmp/eval/hebrew_data')
    parser.add_argument('--output', default='/tmp/experiments/exp_a_results.json')
    parser.add_argument('--max-samples', type=int, default=300)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sp = spm.SentencePieceProcessor(args.tokenizer)
    
    results = {'base': {}, 'sft': {}, 'delta': {}}
    start_time = time.time()

    for model_name, ckpt_path in [('base', args.base_checkpoint), ('sft', args.sft_checkpoint)]:
        print(f"\n{'='*60}")
        print(f"Loading {model_name.upper()} model: {ckpt_path}")
        print(f"{'='*60}")
        model = load_model(ckpt_path, args.device)
        
        for task_name, task_cfg in TASKS.items():
            data_file = os.path.join(args.data_dir, task_cfg['file'])
            if not os.path.exists(data_file):
                print(f"  [SKIP] {task_name} — {data_file} not found")
                results[model_name][task_name] = {'error': 'file not found'}
                continue
            
            print(f"\n  [{model_name.upper()}] Task: {task_name}")
            task_start = time.time()
            result = task_cfg['func'](model, sp, data_file, args.device, args.max_samples)
            result['time_seconds'] = time.time() - task_start
            results[model_name][task_name] = result
            
            metric = task_cfg['metric']
            val = result.get(metric, result.get('accuracy', 0))
            print(f"  ✅ {task_name}: {val:.1f}% ({result.get('total', 0)} samples, {result['time_seconds']:.0f}s)")
        
        # Free GPU memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Compute deltas
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY: Base vs SFT")
    print(f"{'='*60}")
    print(f"{'Task':<15} {'Base':>8} {'SFT':>8} {'Delta':>8}")
    print(f"{'-'*39}")
    
    for task_name, task_cfg in TASKS.items():
        metric = task_cfg['metric']
        base_val = results['base'].get(task_name, {}).get(metric, 0)
        sft_val = results['sft'].get(task_name, {}).get(metric, 0)
        delta = sft_val - base_val
        results['delta'][task_name] = {metric: delta}
        print(f"{task_name:<15} {base_val:>7.1f}% {sft_val:>7.1f}% {delta:>+7.1f}%")
    
    results['total_time_seconds'] = time.time() - start_time
    results['config'] = {
        'max_samples': args.max_samples,
        'base_checkpoint': args.base_checkpoint,
        'sft_checkpoint': args.sft_checkpoint,
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")
    print(f"Total time: {results['total_time_seconds']/60:.1f} minutes")


if __name__ == '__main__':
    main()
