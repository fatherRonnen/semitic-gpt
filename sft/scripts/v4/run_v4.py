#!/usr/bin/env python3
"""
SemiticGPT-3B: V4 Training with clean data
- Fixed Hebrew labels (0=pos, 1=neg)
- Balanced Arabic sentiment
- Real translation data (OPUS-100)
- No data leakage (strict train/eval split)
"""
import json, os, sys, random, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

DEVICE = 'cuda'
MODEL_DIR = '/tmp/model'
DATA_DIR = '/tmp/v4_data'
OUTPUT_DIR = '/tmp/improved_v4'
S3_BUCKET = 's3://autoresearch-dashboard-196766918360/multilingual-7b'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOCAB_SIZE = 32000
DIM = 3072
DEPTH = 26
N_HEADS = 24
HEAD_DIM = DIM // N_HEADS  # 128
ROPE_DIM = HEAD_DIM // 2   # 64
MAX_SEQ_LEN = 2048

USER_PREFIX = "<|user|> "
ASSISTANT_PREFIX = "<|assistant|> "
EOS_ID = 2

# ============================================================
# Model architecture (matches checkpoint format)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        q_rope = q[..., :ROPE_DIM]
        k_rope = k[..., :ROPE_DIM]
        cos = rope_cos[:T].unsqueeze(0).unsqueeze(0)
        sin = rope_sin[:T].unsqueeze(0).unsqueeze(0)
        
        def rotate_half(x):
            x1 = x[..., :x.shape[-1]//2]
            x2 = x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        
        q_rot = q_rope * cos + rotate_half(q_rope) * sin
        k_rot = k_rope * cos + rotate_half(k_rope) * sin
        q = torch.cat([q_rot, q[..., ROPE_DIM:]], dim=-1)
        k = torch.cat([k_rot, k[..., ROPE_DIM:]], dim=-1)
        
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = 8192
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = Attention(dim, n_heads)
        self.mlp = MLP(dim)
        self.ln1 = RMSNorm(dim)
        self.ln2 = RMSNorm(dim)
    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.blocks = nn.ModuleList([Block(DIM, N_HEADS) for _ in range(DEPTH)])
        self.ln_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.register_buffer('rope_cos', torch.zeros(MAX_SEQ_LEN, ROPE_DIM))
        self.register_buffer('rope_sin', torch.zeros(MAX_SEQ_LEN, ROPE_DIM))
    
    def forward(self, x):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)
        return self.head(self.ln_f(h))

# ============================================================
# Eval functions
# ============================================================
@torch.no_grad()
def score_sequence(model, sp, prompt_ids, completion_text):
    comp_ids = sp.encode(completion_text)
    if not comp_ids:
        return float('-inf')
    full_ids = prompt_ids + comp_ids
    if len(full_ids) > MAX_SEQ_LEN:
        full_ids = full_ids[:MAX_SEQ_LEN]
    x = torch.tensor([full_ids], device=DEVICE)
    logits = model(x)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    score = 0.0
    start_idx = len(prompt_ids) - 1
    n_scored = 0
    for i, tid in enumerate(comp_ids):
        pos = start_idx + i
        if pos < log_probs.size(0):
            score += log_probs[pos, tid].item()
            n_scored += 1
    return score / max(n_scored, 1)

def eval_sentiment(model, sp, eval_file, lang):
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f][:200]
    if not samples:
        return 0, 0, 0, 0
    
    unique_labels = list(set(s['output'].strip() for s in samples))
    print(f"    Labels in {lang}: {unique_labels}")
    
    # Logprob eval
    logprob_correct = 0
    for s in samples:
        prompt = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}"
        prompt_ids = sp.encode(prompt)
        scores = {label: score_sequence(model, sp, prompt_ids, label) for label in unique_labels}
        if max(scores, key=scores.get) == s['output'].strip():
            logprob_correct += 1
    
    # Generative eval (first 50)
    gen_correct = 0
    gen_total = min(50, len(samples))
    for s in samples[:gen_total]:
        prompt = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}"
        ids = sp.encode(prompt)
        x = torch.tensor([ids], device=DEVICE)
        with torch.no_grad():
            for _ in range(30):
                logits = model(x[:, -MAX_SEQ_LEN:])
                next_id = logits[0, -1].argmax().item()
                if next_id == EOS_ID:
                    break
                x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        gen = sp.decode(x[0, len(ids):].tolist()).strip()
        expected = s['output'].strip()
        if expected in gen or gen.startswith(expected):
            gen_correct += 1
    
    return logprob_correct / len(samples), len(samples), gen_correct / gen_total, gen_total


def main():
    print("=" * 60)
    print("SemiticGPT-3B: V4 Training (Clean Data)")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)
    
    # Download
    print("\n[1/6] Downloading model, tokenizer, and v4 data...")
    os.system(f'aws s3 cp {S3_BUCKET}/checkpoints/3b-v1-fsdp/sft_model.pt {MODEL_DIR}/model.pt --only-show-errors')
    os.system(f'aws s3 cp {S3_BUCKET}/tokenizer/multilingual_32k.model {MODEL_DIR}/tokenizer.model --only-show-errors')
    os.system(f'aws s3 sync {S3_BUCKET}/v4_data/ {DATA_DIR}/ --only-show-errors')
    print("  Done.")
    
    sp = spm.SentencePieceProcessor()
    sp.load(f'{MODEL_DIR}/tokenizer.model')
    
    # Load base model
    print("\n[2/6] Loading base model...")
    model = GPT()
    state = torch.load(f'{MODEL_DIR}/model.pt', map_location='cpu', weights_only=True)
    sd = state['model_state_dict']
    model.load_state_dict(sd)
    model = model.to(DEVICE).bfloat16()
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")
    
    # Baseline eval
    print("\n[3/6] Baseline eval (on CLEAN v4 data)...")
    baseline = {}
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/sentiment_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            lp_acc, lp_n, gen_acc, gen_n = eval_sentiment(model, sp, eval_file, lang)
            baseline[lang] = {'logprob': lp_acc, 'gen': gen_acc}
            print(f"    → {lang}: logprob={lp_acc*100:.1f}% ({lp_n}), gen={gen_acc*100:.1f}% ({gen_n})")
    
    # Prepare training data
    print("\n[4/6] Preparing training data...")
    all_data = []
    
    # Sentiment (train only — NO eval files!)
    sent_count = 0
    for lang in ['he', 'ar', 'fa', 'en']:
        path = f'{DATA_DIR}/sentiment_train_{lang}.jsonl'
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    all_data.append(json.loads(line))
                    sent_count += 1
    print(f"  Sentiment: {sent_count}")
    
    # Translation (train only)
    trans_count = 0
    for lang in ['he', 'ar', 'fa']:
        path = f'{DATA_DIR}/translation_train_{lang}.jsonl'
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    all_data.append(json.loads(line))
                    trans_count += 1
    print(f"  Translation: {trans_count}")
    print(f"  TOTAL: {len(all_data)}")
    
    random.shuffle(all_data)
    
    # Tokenize
    print("\n[5/6] Tokenizing...")
    all_ids = []
    for s in all_data:
        text = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}{s['output']}"
        ids = sp.encode(text)
        ids.append(EOS_ID)
        if len(ids) > 512:
            ids = ids[:512]
        all_ids.extend(ids)
    
    train_tensor = torch.tensor(all_ids, dtype=torch.long)
    split = int(len(train_tensor) * 0.95)
    train_t = train_tensor[:split]
    val_t = train_tensor[split:]
    print(f"  Tokens: {len(train_tensor):,} (train: {len(train_t):,}, val: {len(val_t):,})")
    
    # Train
    print("\n[6/6] Training v4 (8000 steps, AdamW, bfloat16, L40S 48GB)...")
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.95))
    opt_name = 'AdamW'
    warmup_steps = 200
    total_steps = 8000
    seq_len = 384
    best_val_loss = float('inf')
    
    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            lr = 2e-5 * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = 2e-5 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        start = random.randint(0, len(train_t) - seq_len - 1)
        x = train_t[start:start+seq_len].unsqueeze(0).to(DEVICE)
        y = train_t[start+1:start+seq_len+1].unsqueeze(0).to(DEVICE)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, VOCAB_SIZE), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 500 == 0 or step == 1:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, min(len(val_t) - seq_len - 1, 8000), seq_len):
                    vx = val_t[i:i+seq_len].unsqueeze(0).to(DEVICE)
                    vy = val_t[i+1:i+seq_len+1].unsqueeze(0).to(DEVICE)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        vlogits = model(vx)
                        vloss = F.cross_entropy(vlogits.float().view(-1, VOCAB_SIZE), vy.view(-1))
                    val_losses.append(vloss.item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            print(f"  Step {step}/{total_steps}: train={loss.item():.4f}, val={val_loss:.4f}, lr={lr:.2e}")
            sys.stdout.flush()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'val_loss': val_loss,
                    'config': {'vocab_size': VOCAB_SIZE, 'dim': DIM, 'depth': DEPTH, 'n_heads': N_HEADS, 'max_seq_len': MAX_SEQ_LEN},
                }, f'{OUTPUT_DIR}/sft_model_v4.pt')
            model.train()
    
    print(f"\n  Best val_loss: {best_val_loss:.4f}")
    
    # Final eval
    print("\n[EVAL] Final v4 evaluation on clean data...")
    model.eval()
    best = torch.load(f'{OUTPUT_DIR}/sft_model_v4.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    model = model.to(DEVICE).bfloat16()
    
    v4 = {}
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/sentiment_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            lp_acc, lp_n, gen_acc, gen_n = eval_sentiment(model, sp, eval_file, lang)
            v4[lang] = {'logprob': lp_acc, 'gen': gen_acc}
            b = baseline.get(lang, {})
            lp_delta = lp_acc - b.get('logprob', 0)
            gen_delta = gen_acc - b.get('gen', 0)
            lp_dir = "↑" if lp_delta > 0 else "↓" if lp_delta < 0 else "→"
            gen_dir = "↑" if gen_delta > 0 else "↓" if gen_delta < 0 else "→"
            print(f"    → {lang}: logprob={lp_acc*100:.1f}% ({lp_dir}{abs(lp_delta)*100:.1f}pp), gen={gen_acc*100:.1f}% ({gen_dir}{abs(gen_delta)*100:.1f}pp)")
    
    # Translation test
    print("\n  Translation samples:")
    tests = [
        ("Translate to Hebrew: The weather is nice today", ""),
        ("Translate to Arabic: I love reading books", ""),
        ("Translate to Farsi: Good morning, how are you?", ""),
        ("תרגם לאנגלית: אני אוהב לקרוא ספרים", ""),
        ("Translate to English: الطقس جميل اليوم", ""),
    ]
    for prompt_text, _ in tests:
        prompt = f"{USER_PREFIX}{prompt_text}\n{ASSISTANT_PREFIX}"
        ids = sp.encode(prompt)
        x = torch.tensor([ids], device=DEVICE)
        with torch.no_grad():
            for _ in range(60):
                logits = model(x[:, -MAX_SEQ_LEN:])
                next_id = logits[0, -1].argmax().item()
                if next_id == EOS_ID: break
                x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)
        gen = sp.decode(x[0, len(ids):].tolist()).strip()
        print(f"    {prompt_text[:50]}... → {gen[:60]}")
    
    # Save results
    results = {
        'version': 'v4',
        'training': {
            'steps': total_steps, 'best_val_loss': best_val_loss,
            'total_samples': len(all_data),
            'breakdown': {'sentiment': sent_count, 'translation': trans_count},
            'optimizer': 'AdamW', 'lr': '2e-5', 'seq_len': seq_len,
            'data_fixes': ['hebrew_labels_corrected', 'arabic_balanced', 'no_data_leakage', 'real_opus100_translation'],
        },
        'baseline': {k: v for k, v in baseline.items()},
        'v4': {k: v for k, v in v4.items()},
    }
    with open(f'{OUTPUT_DIR}/v4_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n  Uploading...")
    os.system(f'aws s3 cp {OUTPUT_DIR}/sft_model_v4.pt {S3_BUCKET}/checkpoints/improved-v4/sft_model.pt --only-show-errors')
    os.system(f'aws s3 cp {OUTPUT_DIR}/v4_results.json {S3_BUCKET}/eval/v4_results.json --only-show-errors')
    
    # Print summary table
    print("\n" + "=" * 60)
    print("V4 COMPLETE!")
    print(f"\n{'Lang':<5} {'Base LP':<10} {'V4 LP':<10} {'Base Gen':<10} {'V4 Gen':<10}")
    for lang in ['he', 'ar', 'fa', 'en']:
        b = baseline.get(lang, {})
        v = v4.get(lang, {})
        print(f"{lang:<5} {b.get('logprob',0)*100:<10.1f} {v.get('logprob',0)*100:<10.1f} {b.get('gen',0)*100:<10.1f} {v.get('gen',0)*100:<10.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
