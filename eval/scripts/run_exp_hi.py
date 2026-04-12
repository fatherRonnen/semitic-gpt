#!/usr/bin/env python3
"""
Run Experiments H & I on GPU.
Fine-tune D-SFT model on domain tasks, evaluate cross-lingually.
"""
import json, os, sys, random, struct, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, '/tmp/eval')
from train_sft_3b import GPT, VOCAB_SIZE, DIM, DEPTH, N_HEADS, MAX_SEQ_LEN

sp = spm.SentencePieceProcessor('/tmp/eval/multilingual_32k.model')
device = 'cuda'
EOS_ID = 3
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"

DATA_DIR = '/tmp/domain_experiments'
RESULTS_DIR = '/tmp/experiments'
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_model(path, half=False):
    model = GPT()
    state = torch.load(path, map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    if half:
        model = model.half()
    model = model.to(device)
    return model

def tokenize_samples(samples, max_len=512):
    """Tokenize instruction/output pairs into token ids."""
    all_ids = []
    for s in samples:
        text = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}{s['output']}"
        ids = sp.encode(text)
        ids.append(EOS_ID)
        if len(ids) > max_len:
            ids = ids[:max_len]
        all_ids.extend(ids)
    return all_ids

def save_bin(ids, path):
    with open(path, 'wb') as f:
        for tid in ids:
            f.write(struct.pack('<H', tid))
    return len(ids)

def train_sft(model, train_file, val_file, output_path, steps=1000, lr=1e-5, batch_tokens=2048):
    """SFT training loop — model in fp16, SGD no momentum (minimal memory)."""
    model.train()
    # SGD without momentum — no extra state buffers at all
    # Model is already fp16, gradients will be fp16 too
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * 100, momentum=0, weight_decay=0)
    
    # Load data
    with open(train_file, 'rb') as f:
        train_data = f.read()
    train_ids = list(struct.iter_unpack('<H', train_data))
    train_ids = [x[0] for x in train_ids]
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    
    with open(val_file, 'rb') as f:
        val_data = f.read()
    val_ids = list(struct.iter_unpack('<H', val_data))
    val_ids = [x[0] for x in val_ids]
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    
    seq_len = min(MAX_SEQ_LEN, 256)
    best_val_loss = float('inf')
    
    for step in range(1, steps + 1):
        start = random.randint(0, len(train_tensor) - seq_len - 1)
        x = train_tensor[start:start+seq_len].unsqueeze(0).to(device)
        y = train_tensor[start+1:start+seq_len+1].unsqueeze(0).to(device)
        
        logits = model(x)
        # Compute loss in fp32 for stability
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 100 == 0 or step == 1:
            model.eval()
            val_losses = []
            for i in range(0, min(len(val_tensor) - seq_len - 1, 5000), seq_len):
                vx = val_tensor[i:i+seq_len].unsqueeze(0).to(device)
                vy = val_tensor[i+1:i+seq_len+1].unsqueeze(0).to(device)
                with torch.no_grad():
                    vlogits = model(vx)
                    vloss = F.cross_entropy(vlogits.float().view(-1, vlogits.size(-1)), vy.view(-1))
                val_losses.append(vloss.item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
            print(f"  Step {step}/{steps}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, output_path)
            
            model.train()
    
    print(f"  Best val_loss: {best_val_loss:.4f}")
    return best_val_loss

@torch.no_grad()
def eval_classification(model, eval_file, max_new=20):
    """Evaluate classification accuracy by generating and matching expected output."""
    model.eval()
    
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f]
    
    correct = 0
    total = 0
    
    for s in samples[:200]:  # Cap at 200 for speed
        prompt = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}"
        ids = sp.encode(prompt)
        if len(ids) > MAX_SEQ_LEN - max_new:
            ids = ids[:MAX_SEQ_LEN - max_new]
        
        x = torch.tensor([ids], device=device)
        
        # Greedy generate
        for _ in range(max_new):
            logits = model(x[:, -MAX_SEQ_LEN:])
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS_ID:
                break
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        
        generated_ids = x[0, len(ids):].tolist()
        generated = sp.decode(generated_ids).strip()
        expected = s['output'].strip()
        
        # Fuzzy match: check if expected label appears in generated text
        if expected.lower() in generated.lower() or generated.lower() in expected.lower():
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0
    return acc, total

# ============================================================
# MAIN
# ============================================================
all_results = {}

# Load base D-SFT model
print("Loading D-SFT model...")
base_model_path = '/tmp/sft_v3_runs/D/sft_model.pt'

# ============================================================
# EXP H: SENTIMENT
# ============================================================
print("\n" + "="*60)
print("EXP H: Cross-lingual Sentiment Transfer")
print("="*60)

configs = [
    ('H1_he_only', 'sentiment_train_H1_he_only.jsonl'),
    ('H2_all_langs', 'sentiment_train_H2_all_langs.jsonl'),
    ('H3_ar_fa_only', 'sentiment_train_H3_ar_fa_only.jsonl'),
]

eval_files = {
    'he': f'{DATA_DIR}/sentiment_eval_he.jsonl',
    'ar': f'{DATA_DIR}/sentiment_eval_ar.jsonl',
    'fa': f'{DATA_DIR}/sentiment_eval_fa.jsonl',
    'en': f'{DATA_DIR}/sentiment_eval_en.jsonl',
}

# First, evaluate D-baseline (no sentiment training) on all eval sets
print("\n--- D-baseline (no sentiment training) ---")
model = load_model(base_model_path, half=True)
all_results['D-baseline'] = {}
for lang, eval_file in eval_files.items():
    acc, n = eval_classification(model, eval_file)
    print(f"  {lang}: {acc*100:.1f}% ({n} samples)")
    all_results['D-baseline'][lang] = {'accuracy': acc, 'n': n}
del model
torch.cuda.empty_cache()

for config_name, train_file in configs:
    print(f"\n--- {config_name} ---")
    
    # Prepare binary train/val data
    with open(f'{DATA_DIR}/{train_file}') as f:
        samples = [json.loads(line) for line in f]
    
    random.shuffle(samples)
    n_val = max(200, int(len(samples) * 0.1))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    
    train_ids = tokenize_samples(train_samples)
    val_ids = tokenize_samples(val_samples)
    
    train_bin = f'/tmp/{config_name}_train.bin'
    val_bin = f'/tmp/{config_name}_val.bin'
    save_bin(train_ids, train_bin)
    save_bin(val_ids, val_bin)
    print(f"  Data: {len(train_samples)} train ({len(train_ids)} tokens), {len(val_samples)} val ({len(val_ids)} tokens)")
    
    # Load fresh D-SFT in fp16 and fine-tune
    model = load_model(base_model_path, half=True)
    output_path = f'/tmp/{config_name}_model.pt'
    
    best_val = train_sft(model, train_bin, val_bin, output_path, steps=500, lr=2e-5)
    
    # Reload best and evaluate
    del model
    torch.cuda.empty_cache()
    model = load_model(output_path, half=True)
    all_results[config_name] = {'val_loss': best_val}
    
    for lang, eval_file in eval_files.items():
        acc, n = eval_classification(model, eval_file)
        print(f"  Eval {lang}: {acc*100:.1f}% ({n} samples)")
        all_results[config_name][lang] = {'accuracy': acc, 'n': n}
    
    del model
    torch.cuda.empty_cache()

# ============================================================
# EXP I: NEWS CLASSIFICATION
# ============================================================
print("\n" + "="*60)
print("EXP I: News Topic Classification")
print("="*60)

# I1: English-only news training
print("\n--- I1: EN-only news ---")
with open(f'{DATA_DIR}/news_train_I1_en_only.jsonl') as f:
    news_samples = [json.loads(line) for line in f]

random.shuffle(news_samples)
n_val = 500
news_val = news_samples[:n_val]
news_train = news_samples[n_val:]

train_ids = tokenize_samples(news_train)
val_ids = tokenize_samples(news_val)
save_bin(train_ids, '/tmp/I1_train.bin')
save_bin(val_ids, '/tmp/I1_val.bin')
print(f"  Data: {len(news_train)} train ({len(train_ids)} tokens), {len(news_val)} val ({len(val_ids)} tokens)")

# D-baseline on news
print("\n  D-baseline on news:")
model = load_model(base_model_path, half=True)
acc, n = eval_classification(model, f'{DATA_DIR}/news_eval_en.jsonl')
print(f"    EN news: {acc*100:.1f}% ({n})")
all_results['D-baseline']['en_news'] = {'accuracy': acc, 'n': n}
del model
torch.cuda.empty_cache()

# Train I1
model = load_model(base_model_path, half=True)
best_val = train_sft(model, '/tmp/I1_train.bin', '/tmp/I1_val.bin', '/tmp/I1_model.pt', steps=500, lr=2e-5)
del model
torch.cuda.empty_cache()
model = load_model('/tmp/I1_model.pt', half=True)
acc, n = eval_classification(model, f'{DATA_DIR}/news_eval_en.jsonl')
print(f"  I1 EN news: {acc*100:.1f}% ({n})")
all_results['I1_en_only'] = {
    'val_loss': best_val,
    'en_news': {'accuracy': acc, 'n': n},
}
del model
torch.cuda.empty_cache()

# ============================================================
# Save all results
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

with open(f'{RESULTS_DIR}/exp_hi_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(json.dumps(all_results, indent=2))

# Upload
os.system(f"aws s3 cp {RESULTS_DIR}/exp_hi_results.json s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/exp_hi_results.json --quiet")
print("\nResults uploaded to S3!")
