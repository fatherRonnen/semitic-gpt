#!/usr/bin/env python3
"""
Experiments A + B for the paper (run on GPU):

A. Hebrew Downstream Eval
   - Sentiment classification (all 4 langs) at 2 stages: base, multilingual-SFT
   - Compare with HebrewGPT-1B baseline numbers

B. Cross-lingual Transfer  
   - Train EN-SFT only model (English sentiment + translation only)
   - Eval at 3 stages: base → EN-SFT → multilingual-SFT
   - Belebele at all 3 stages
   - Show transfer progression
"""
import json, os, sys, random, math, time
random.seed(42)
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

DEVICE = 'cuda'
MODEL_DIR = '/opt/dlami/nvme/model'
DATA_DIR = '/opt/dlami/nvme/v4_data'
OUTPUT_DIR = '/opt/dlami/nvme/exp_ab'
S3 = 's3://autoresearch-dashboard-196766918360/multilingual-7b'

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

VOCAB_SIZE, DIM, DEPTH, N_HEADS = 32000, 3072, 26, 24
HEAD_DIM = DIM // N_HEADS
ROPE_DIM = HEAD_DIM // 2
MAX_SEQ_LEN = 2048
EOS_ID = 2
USER_PREFIX = "<|user|> "
ASSISTANT_PREFIX = "<|assistant|> "

# ============================================================
# Model (same architecture as training)
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
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, rc, rs):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        def rot(x): return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], -1)
        cos, sin = rc[:T].unsqueeze(0).unsqueeze(0), rs[:T].unsqueeze(0).unsqueeze(0)
        qr, kr = q[..., :ROPE_DIM], k[..., :ROPE_DIM]
        q = torch.cat([qr*cos + rot(qr)*sin, q[..., ROPE_DIM:]], -1)
        k = torch.cat([kr*cos + rot(kr)*sin, k[..., ROPE_DIM:]], -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d,8192,bias=False), nn.Linear(d,8192,bias=False), nn.Linear(8192,d,bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn, self.mlp, self.ln1, self.ln2 = Attention(d,h), MLP(d), RMSNorm(d), RMSNorm(d)
    def forward(self, x, rc, rs):
        x = x + self.attn(self.ln1(x), rc, rs)
        return x + self.mlp(self.ln2(x))

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
        for b in self.blocks: h = b(h, self.rope_cos, self.rope_sin)
        return self.head(self.ln_f(h))

# ============================================================
# Eval functions
# ============================================================
@torch.no_grad()
def score_sequence(model, sp, prompt_ids, completion_text):
    comp_ids = sp.encode(completion_text)
    if not comp_ids: return float('-inf')
    full_ids = prompt_ids + comp_ids
    if len(full_ids) > MAX_SEQ_LEN: full_ids = full_ids[:MAX_SEQ_LEN]
    x = torch.tensor([full_ids], device=DEVICE)
    logits = model(x)
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    score = 0.0
    start_idx = len(prompt_ids) - 1
    n = 0
    for i, tid in enumerate(comp_ids):
        pos = start_idx + i
        if pos < log_probs.size(0):
            score += log_probs[pos, tid].item()
            n += 1
    return score / max(n, 1)

@torch.no_grad()
def generate(model, sp, prompt_text, max_tokens=60):
    prompt = f"{USER_PREFIX}{prompt_text}\n{ASSISTANT_PREFIX}"
    ids = sp.encode(prompt)
    x = torch.tensor([ids], device=DEVICE)
    for _ in range(max_tokens):
        logits = model(x[:, -MAX_SEQ_LEN:])
        next_id = logits[0, -1].float().argmax().item()
        if next_id == EOS_ID: break
        x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return sp.decode(x[0, len(ids):].tolist()).strip()

def eval_sentiment(model, sp, eval_file, n_samples=200, n_gen=50):
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f][:n_samples]
    if not samples: return {}
    
    unique_labels = list(set(s['output'].strip() for s in samples))
    
    # Logprob
    lp_correct = 0
    for s in samples:
        prompt = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}"
        prompt_ids = sp.encode(prompt)
        scores = {label: score_sequence(model, sp, prompt_ids, label) for label in unique_labels}
        if max(scores, key=scores.get) == s['output'].strip():
            lp_correct += 1
    
    # Generative
    gen_correct = 0
    for s in samples[:n_gen]:
        gen = generate(model, sp, s['instruction'])
        expected = s['output'].strip()
        if expected in gen or gen.startswith(expected):
            gen_correct += 1
    
    return {
        'logprob_acc': round(lp_correct / len(samples), 4),
        'logprob_n': len(samples),
        'gen_acc': round(gen_correct / n_gen, 4),
        'gen_n': n_gen,
    }

def eval_belebele(model, sp, lang, n_samples=100):
    """Evaluate on Belebele multiple-choice reading comprehension."""
    belebele_path = f'{DATA_DIR}/belebele/{lang}.jsonl'
    if not os.path.exists(belebele_path):
        print(f"    Belebele {lang} not found at {belebele_path}")
        return {}
    
    samples = []
    with open(belebele_path) as f:
        for line in f:
            samples.append(json.loads(line))
    samples = samples[:n_samples]
    
    correct = 0
    for s in samples:
        passage = s['flores_passage']
        question = s['question']
        choices = [s['mc_answer1'], s['mc_answer2'], s['mc_answer3'], s['mc_answer4']]
        correct_idx = int(s['correct_answer_num']) - 1
        
        prompt = f"Read the passage and answer the question.\n\nPassage: {passage[:500]}\n\nQuestion: {question}\n\nChoices:\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}\n\nAnswer:"
        prompt_ids = sp.encode(f"{USER_PREFIX}{prompt}\n{ASSISTANT_PREFIX}")
        
        labels = ['A', 'B', 'C', 'D']
        scores = {i: score_sequence(model, sp, prompt_ids, labels[i]) for i in range(4)}
        pred = max(scores, key=scores.get)
        if pred == correct_idx:
            correct += 1
    
    return {
        'accuracy': round(correct / len(samples), 4),
        'n': len(samples),
    }

# ============================================================
# EN-SFT Training (for Experiment B)
# ============================================================
def train_en_sft(base_state_dict, sp):
    """Train EN-only SFT model (English sentiment + English translation only)."""
    print("\n[B] Training EN-SFT model (English-only fine-tuning)...")
    
    model = GPT()
    model.load_state_dict(base_state_dict)
    model = model.to(DEVICE).bfloat16()
    model.train()
    
    # Load English-only data
    all_data = []
    en_sent_path = f'{DATA_DIR}/sentiment_train_en.jsonl'
    if os.path.exists(en_sent_path):
        with open(en_sent_path) as f:
            for line in f:
                all_data.append(json.loads(line))
    print(f"  EN sentiment: {len(all_data)}")
    
    # Add English side of translation (EN→target direction only)
    for lang in ['he', 'ar', 'fa']:
        path = f'{DATA_DIR}/translation_train_{lang}.jsonl'
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    # Only keep English instructions (Translate to X: ...)
                    if d['instruction'].startswith('Translate to'):
                        all_data.append(d)
    print(f"  Total EN-SFT data: {len(all_data)}")
    
    random.shuffle(all_data)
    
    # Tokenize
    all_ids = []
    for s in all_data:
        text = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}{s['output']}"
        ids = sp.encode(text)
        ids.append(EOS_ID)
        if len(ids) > 512: ids = ids[:512]
        all_ids.extend(ids)
    
    train_t = torch.tensor(all_ids, dtype=torch.long)
    print(f"  Tokens: {len(train_t):,}")
    
    # Quick training: 2000 steps
    # A10G 24GB: use SGD (no optimizer states) to fit in VRAM
    # SGD works for short fine-tuning with higher LR
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
    total_steps = 2000
    warmup = 100
    seq_len = 256  # shorter to save memory
    
    for step in range(1, total_steps + 1):
        if step <= warmup:
            lr = 5e-4 * step / warmup
        else:
            lr = 5e-4 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup)))
        for pg in optimizer.param_groups: pg['lr'] = lr
        
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
            print(f"    Step {step}/{total_steps}: loss={loss.item():.4f}, lr={lr:.2e}")
            sys.stdout.flush()
    
    # Save
    torch.save({'model_state_dict': model.state_dict()}, f'{OUTPUT_DIR}/en_sft_model.pt')
    print("  EN-SFT model saved.")
    
    model.eval()
    return model


def main():
    print("=" * 60)
    print("EXPERIMENTS A + B")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 60)
    
    # Download everything
    print("\n[SETUP] Downloading models and data...")
    os.system(f'aws s3 cp {S3}/checkpoints/3b-v1-fsdp/sft_model.pt {MODEL_DIR}/base_model.pt --only-show-errors')
    os.system(f'aws s3 cp {S3}/checkpoints/improved-v4/sft_model.pt {MODEL_DIR}/v4_model.pt --only-show-errors')
    os.system(f'aws s3 cp {S3}/tokenizer/multilingual_32k.model {MODEL_DIR}/tokenizer.model --only-show-errors')
    os.system(f'aws s3 sync {S3}/v4_data/ {DATA_DIR}/ --only-show-errors')
    os.system(f'mkdir -p {DATA_DIR}/belebele && aws s3 sync {S3}/belebele/ {DATA_DIR}/belebele/ --only-show-errors')
    
    # Install sentencepiece if needed
    try:
        import sentencepiece
    except ImportError:
        os.system(f'aws s3 cp {S3}/wheels/ /tmp/wheels/ --recursive --only-show-errors')
        os.system('pip install --no-deps /tmp/wheels/sentencepiece*.whl 2>/dev/null')
    
    sp = spm.SentencePieceProcessor()
    sp.load(f'{MODEL_DIR}/tokenizer.model')
    
    # ============================================================
    # Stage 1: BASE model eval
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 1: BASE MODEL (pre-trained, no SFT)")
    print("=" * 60)
    
    model = GPT()
    base_state = torch.load(f'{MODEL_DIR}/base_model.pt', map_location='cpu', weights_only=True)
    base_sd = base_state['model_state_dict']
    model.load_state_dict(base_sd)
    model = model.to(DEVICE).bfloat16()
    model.eval()
    
    base_results = {'sentiment': {}, 'belebele': {}}
    
    print("\n  Sentiment eval...")
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/sentiment_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            r = eval_sentiment(model, sp, eval_file)
            base_results['sentiment'][lang] = r
            print(f"    {lang}: logprob={r['logprob_acc']*100:.1f}%, gen={r['gen_acc']*100:.1f}%")
            sys.stdout.flush()
    
    print("\n  Belebele eval...")
    for lang in ['he', 'ar', 'fa', 'en']:
        r = eval_belebele(model, sp, lang)
        base_results['belebele'][lang] = r
        if r:
            print(f"    {lang}: {r['accuracy']*100:.1f}%")
        sys.stdout.flush()
    
    # ============================================================
    # Stage 2: EN-SFT model (Experiment B)
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 2: EN-SFT MODEL (English-only fine-tuning)")
    print("=" * 60)
    
    # Need to reload base for training
    model_en = train_en_sft(base_sd, sp)
    
    en_sft_results = {'sentiment': {}, 'belebele': {}}
    
    print("\n  Sentiment eval (EN-SFT)...")
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/sentiment_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            r = eval_sentiment(model_en, sp, eval_file)
            en_sft_results['sentiment'][lang] = r
            print(f"    {lang}: logprob={r['logprob_acc']*100:.1f}%, gen={r['gen_acc']*100:.1f}%")
            sys.stdout.flush()
    
    print("\n  Belebele eval (EN-SFT)...")
    for lang in ['he', 'ar', 'fa', 'en']:
        r = eval_belebele(model_en, sp, lang)
        en_sft_results['belebele'][lang] = r
        if r:
            print(f"    {lang}: {r['accuracy']*100:.1f}%")
        sys.stdout.flush()
    
    del model_en
    torch.cuda.empty_cache()
    
    # ============================================================
    # Stage 3: Multilingual-SFT (v4)
    # ============================================================
    print("\n" + "=" * 60)
    print("STAGE 3: MULTILINGUAL-SFT (v4, all 4 languages)")
    print("=" * 60)
    
    model_v4 = GPT()
    v4_state = torch.load(f'{MODEL_DIR}/v4_model.pt', map_location='cpu', weights_only=True)
    model_v4.load_state_dict(v4_state['model_state_dict'])
    model_v4 = model_v4.to(DEVICE).bfloat16()
    model_v4.eval()
    del v4_state
    
    v4_results = {'sentiment': {}, 'belebele': {}}
    
    print("\n  Sentiment eval (v4)...")
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/sentiment_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            r = eval_sentiment(model_v4, sp, eval_file)
            v4_results['sentiment'][lang] = r
            print(f"    {lang}: logprob={r['logprob_acc']*100:.1f}%, gen={r['gen_acc']*100:.1f}%")
            sys.stdout.flush()
    
    print("\n  Belebele eval (v4)...")
    for lang in ['he', 'ar', 'fa', 'en']:
        r = eval_belebele(model_v4, sp, lang)
        v4_results['belebele'][lang] = r
        if r:
            print(f"    {lang}: {r['accuracy']*100:.1f}%")
        sys.stdout.flush()
    
    # Generation samples
    print("\n  Generation samples (v4):")
    gen_prompts = [
        "Write a short sentence in Hebrew about the weather.",
        "اكتب جملة قصيرة عن الطقس بالعربية.",
        "یک جمله کوتاه درباره آب و هوا به فارسی بنویسید.",
        "What is the capital of Israel?",
    ]
    gen_samples = []
    for p in gen_prompts:
        result = generate(model_v4, sp, p)
        print(f"    Q: {p[:60]}")
        print(f"    A: {result[:80]}")
        gen_samples.append({'prompt': p, 'response': result})
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n--- Sentiment Classification (Logprob Accuracy) ---")
    print(f"{'Lang':<6} {'Base':>8} {'EN-SFT':>8} {'Multi-SFT':>10} {'Δ Base→Multi':>14}")
    print("-" * 50)
    for lang in ['he', 'ar', 'fa', 'en']:
        b = base_results['sentiment'].get(lang, {}).get('logprob_acc', 0)
        e = en_sft_results['sentiment'].get(lang, {}).get('logprob_acc', 0)
        m = v4_results['sentiment'].get(lang, {}).get('logprob_acc', 0)
        delta = m - b
        print(f"{lang:<6} {b*100:>7.1f}% {e*100:>7.1f}% {m*100:>9.1f}% {delta*100:>+13.1f}pp")
    
    print("\n--- Sentiment Classification (Generative Accuracy) ---")
    print(f"{'Lang':<6} {'Base':>8} {'EN-SFT':>8} {'Multi-SFT':>10}")
    print("-" * 35)
    for lang in ['he', 'ar', 'fa', 'en']:
        b = base_results['sentiment'].get(lang, {}).get('gen_acc', 0)
        e = en_sft_results['sentiment'].get(lang, {}).get('gen_acc', 0)
        m = v4_results['sentiment'].get(lang, {}).get('gen_acc', 0)
        print(f"{lang:<6} {b*100:>7.1f}% {e*100:>7.1f}% {m*100:>9.1f}%")
    
    print("\n--- Belebele Reading Comprehension ---")
    print(f"{'Lang':<6} {'Base':>8} {'EN-SFT':>8} {'Multi-SFT':>10}")
    print("-" * 35)
    for lang in ['he', 'ar', 'fa', 'en']:
        b = base_results['belebele'].get(lang, {}).get('accuracy', 0)
        e = en_sft_results['belebele'].get(lang, {}).get('accuracy', 0)
        m = v4_results['belebele'].get(lang, {}).get('accuracy', 0)
        print(f"{lang:<6} {b*100:>7.1f}% {e*100:>7.1f}% {m*100:>9.1f}%")
    
    # Save
    all_results = {
        'experiment': 'A_B_downstream_crosslingual',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'base': base_results,
        'en_sft': en_sft_results,
        'multilingual_sft': v4_results,
        'generation_samples': gen_samples,
    }
    
    out_path = f'{OUTPUT_DIR}/exp_ab_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    os.system(f'aws s3 cp {out_path} {S3}/eval/exp_ab_results.json --only-show-errors')
    os.system(f'aws s3 cp {OUTPUT_DIR}/en_sft_model.pt {S3}/checkpoints/en-sft/sft_model.pt --only-show-errors')
    
    print(f"\nResults saved to {out_path} and S3")
    print("EXPERIMENTS A + B COMPLETE!")


if __name__ == '__main__':
    main()
