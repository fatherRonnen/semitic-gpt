#!/usr/bin/env python3
"""
Paper 1 — QA Cross-Lingual Transfer Experiment.
Fine-tune 3B model on Hebrew QA (TyDiQA-GoldP), evaluate zero-shot on AR, FA, EN.

Conditions:
  - Baseline: D-SFT model with no QA training
  - HE-only: Fine-tuned on Hebrew QA only
  - All-langs: Fine-tuned on HE+AR+FA+EN QA (upper bound)

Requires: datasets, sentencepiece, torch
GPU: L40S 48GB (fp16 + SGD no momentum)
"""
import json, os, sys, random, struct, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
import sentencepiece as spm
from datasets import load_dataset

sys.path.insert(0, '/tmp/eval')
from train_sft_3b import GPT, VOCAB_SIZE, DIM, DEPTH, N_HEADS, MAX_SEQ_LEN

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = '/tmp/sft_v3_runs/D/sft_model.pt'
TOKENIZER_PATH = '/tmp/eval/multilingual_32k.model'
RESULTS_DIR = '/tmp/experiments/paper1'
DATA_DIR = '/tmp/paper1_qa_data'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# TyDiQA language codes mapped to our codes
TYDIQA_LANG_MAP = {
    'hebrew': 'he',
    'arabic': 'ar',
    'persian': 'fa', 
    'english': 'en',
}

TRAIN_STEPS = 800
LR = 2e-5
SEQ_LEN = 256
MAX_TRAIN_SAMPLES = 3000
MAX_EVAL_SAMPLES = 200

device = 'cuda'
EOS_ID = 3
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"

sp = spm.SentencePieceProcessor(TOKENIZER_PATH)

# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_qa_data():
    """Download TyDiQA-GoldP and format as instruction-output pairs."""
    print("Downloading TyDiQA-GoldP...")
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", trust_remote_code=True)
    
    lang_data = {lang: [] for lang in ['he', 'ar', 'fa', 'en']}
    
    for split in ['train', 'validation']:
        for example in ds[split]:
            lang_id = example.get('id', '').split('-')[0] if 'id' in example else ''
            # TyDiQA uses language in the id field or we detect from the text
            # Try to map by language field if available
            detected_lang = None
            for tydi_lang, our_lang in TYDIQA_LANG_MAP.items():
                if tydi_lang in example.get('id', '').lower():
                    detected_lang = our_lang
                    break
            
            if detected_lang is None:
                continue
            
            context = example['context']
            question = example['question']
            # Get answer - TyDiQA has answers field
            answers = example.get('answers', {})
            if not answers or not answers.get('text'):
                continue
            answer = answers['text'][0]
            
            if not answer.strip():
                continue
                
            # Format as instruction
            instruction = f"Read the following passage and answer the question.\n\nPassage: {context[:500]}\n\nQuestion: {question}"
            output = answer
            
            lang_data[detected_lang].append({
                'instruction': instruction,
                'output': output,
                'split': split
            })
    
    # Save per-language data
    for lang in ['he', 'ar', 'fa', 'en']:
        samples = lang_data[lang]
        print(f"  {lang}: {len(samples)} total samples")
        
        # Split into train/eval
        random.shuffle(samples)
        n_eval = min(MAX_EVAL_SAMPLES, len(samples) // 5)
        eval_samples = samples[:n_eval]
        train_samples = samples[n_eval:n_eval + MAX_TRAIN_SAMPLES]
        
        with open(f'{DATA_DIR}/qa_train_{lang}.jsonl', 'w') as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        with open(f'{DATA_DIR}/qa_eval_{lang}.jsonl', 'w') as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    return lang_data

# ============================================================
# MODEL UTILITIES (same as run_exp_hi.py)
# ============================================================

def load_model(path, half=True):
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

def train_sft(model, train_file, val_file, output_path, steps=800, lr=2e-5):
    """SFT training — fp16 model, SGD no momentum."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * 100, momentum=0, weight_decay=0)
    
    with open(train_file, 'rb') as f:
        train_data = f.read()
    train_ids = [x[0] for x in struct.iter_unpack('<H', train_data)]
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    
    with open(val_file, 'rb') as f:
        val_data = f.read()
    val_ids = [x[0] for x in struct.iter_unpack('<H', val_data)]
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    
    seq_len = min(MAX_SEQ_LEN, SEQ_LEN)
    best_val_loss = float('inf')
    
    for step in range(1, steps + 1):
        start = random.randint(0, len(train_tensor) - seq_len - 1)
        x = train_tensor[start:start+seq_len].unsqueeze(0).to(device)
        y = train_tensor[start+1:start+seq_len+1].unsqueeze(0).to(device)
        
        logits = model(x)
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
    
    return best_val_loss

@torch.no_grad()
def eval_qa(model, eval_file, max_new=50):
    """Evaluate QA by generating answer and computing token-level F1."""
    model.eval()
    
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f]
    
    exact_match = 0
    f1_scores = []
    total = 0
    
    for s in samples[:MAX_EVAL_SAMPLES]:
        prompt = f"{USER_PREFIX}{s['instruction']}\n{ASSISTANT_PREFIX}"
        ids = sp.encode(prompt)
        if len(ids) > MAX_SEQ_LEN - max_new:
            ids = ids[:MAX_SEQ_LEN - max_new]
        
        x = torch.tensor([ids], device=device)
        
        for _ in range(max_new):
            logits = model(x[:, -MAX_SEQ_LEN:])
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS_ID:
                break
            x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        
        generated_ids = x[0, len(ids):].tolist()
        generated = sp.decode(generated_ids).strip().lower()
        expected = s['output'].strip().lower()
        
        # Exact match
        if generated == expected or expected in generated:
            exact_match += 1
        
        # Token F1
        gen_tokens = set(generated.split())
        exp_tokens = set(expected.split())
        if gen_tokens and exp_tokens:
            precision = len(gen_tokens & exp_tokens) / len(gen_tokens)
            recall = len(gen_tokens & exp_tokens) / len(exp_tokens)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            f1 = 1.0 if generated == expected else 0.0
        f1_scores.append(f1)
        total += 1
    
    em = exact_match / total if total > 0 else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    return {'exact_match': em, 'f1': avg_f1, 'n': total}

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PAPER 1 — QA CROSS-LINGUAL TRANSFER")
    print("=" * 60)
    
    # Step 1: Prepare data
    print("\n--- Preparing QA Data ---")
    prepare_qa_data()
    
    all_results = {}
    
    # Step 2: Baseline (no QA training)
    print("\n--- Baseline (D-SFT, no QA training) ---")
    model = load_model(MODEL_PATH, half=True)
    all_results['baseline'] = {}
    for lang in ['he', 'ar', 'fa', 'en']:
        eval_file = f'{DATA_DIR}/qa_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            result = eval_qa(model, eval_file)
            print(f"  {lang}: EM={result['exact_match']*100:.1f}%, F1={result['f1']*100:.1f}% (n={result['n']})")
            all_results['baseline'][lang] = result
    del model
    torch.cuda.empty_cache()
    
    # Step 3: Hebrew-only QA training
    print("\n--- Hebrew-only QA Training ---")
    train_file = f'{DATA_DIR}/qa_train_he.jsonl'
    if os.path.exists(train_file):
        with open(train_file) as f:
            he_samples = [json.loads(line) for line in f]
        
        if len(he_samples) > 50:
            random.shuffle(he_samples)
            n_val = max(100, len(he_samples) // 10)
            val_samples = he_samples[:n_val]
            train_samples = he_samples[n_val:]
            
            train_ids = tokenize_samples(train_samples)
            val_ids = tokenize_samples(val_samples)
            save_bin(train_ids, f'{DATA_DIR}/he_train.bin')
            save_bin(val_ids, f'{DATA_DIR}/he_val.bin')
            print(f"  Data: {len(train_samples)} train ({len(train_ids)} tokens), {len(val_samples)} val")
            
            model = load_model(MODEL_PATH, half=True)
            best_val = train_sft(model, f'{DATA_DIR}/he_train.bin', f'{DATA_DIR}/he_val.bin', 
                               f'{RESULTS_DIR}/qa_he_only_model.pt', steps=TRAIN_STEPS)
            del model
            torch.cuda.empty_cache()
            
            # Evaluate
            model = load_model(f'{RESULTS_DIR}/qa_he_only_model.pt', half=True)
            all_results['he_only'] = {'val_loss': best_val}
            for lang in ['he', 'ar', 'fa', 'en']:
                eval_file = f'{DATA_DIR}/qa_eval_{lang}.jsonl'
                if os.path.exists(eval_file):
                    result = eval_qa(model, eval_file)
                    print(f"  {lang}: EM={result['exact_match']*100:.1f}%, F1={result['f1']*100:.1f}%")
                    all_results['he_only'][lang] = result
            del model
            torch.cuda.empty_cache()
        else:
            print("  Not enough Hebrew QA data, skipping")
    
    # Step 4: All-langs QA training
    print("\n--- All-langs QA Training ---")
    all_train_samples = []
    for lang in ['he', 'ar', 'fa', 'en']:
        tf = f'{DATA_DIR}/qa_train_{lang}.jsonl'
        if os.path.exists(tf):
            with open(tf) as f:
                samples = [json.loads(line) for line in f]
            all_train_samples.extend(samples[:MAX_TRAIN_SAMPLES // 4])
    
    if len(all_train_samples) > 100:
        random.shuffle(all_train_samples)
        n_val = max(100, len(all_train_samples) // 10)
        val_s = all_train_samples[:n_val]
        train_s = all_train_samples[n_val:]
        
        train_ids = tokenize_samples(train_s)
        val_ids = tokenize_samples(val_s)
        save_bin(train_ids, f'{DATA_DIR}/all_train.bin')
        save_bin(val_ids, f'{DATA_DIR}/all_val.bin')
        print(f"  Data: {len(train_s)} train ({len(train_ids)} tokens), {len(val_s)} val")
        
        model = load_model(MODEL_PATH, half=True)
        best_val = train_sft(model, f'{DATA_DIR}/all_train.bin', f'{DATA_DIR}/all_val.bin',
                           f'{RESULTS_DIR}/qa_all_langs_model.pt', steps=TRAIN_STEPS)
        del model
        torch.cuda.empty_cache()
        
        model = load_model(f'{RESULTS_DIR}/qa_all_langs_model.pt', half=True)
        all_results['all_langs'] = {'val_loss': best_val}
        for lang in ['he', 'ar', 'fa', 'en']:
            eval_file = f'{DATA_DIR}/qa_eval_{lang}.jsonl'
            if os.path.exists(eval_file):
                result = eval_qa(model, eval_file)
                print(f"  {lang}: EM={result['exact_match']*100:.1f}%, F1={result['f1']*100:.1f}%")
                all_results['all_langs'][lang] = result
        del model
        torch.cuda.empty_cache()
    
    # Save results
    print("\n" + "=" * 60)
    print("FINAL QA RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))
    
    with open(f'{RESULTS_DIR}/qa_transfer_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    os.system(f"aws s3 cp {RESULTS_DIR}/qa_transfer_results.json s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/paper1_qa_results.json --quiet")
    print("\nResults uploaded to S3!")
