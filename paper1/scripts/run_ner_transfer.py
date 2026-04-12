#!/usr/bin/env python3
"""
Paper 1 — NER Cross-Lingual Transfer Experiment.
Fine-tune 3B model on Hebrew WikiANN NER, evaluate zero-shot on AR, FA, EN.

Conditions:
  - Baseline: D-SFT model with no NER training
  - HE-only: Fine-tuned on Hebrew NER only
  - All-langs: Fine-tuned on HE+AR+FA+EN NER (upper bound)

Requires: datasets, sentencepiece, torch
GPU: L40S 48GB (fp16 + SGD no momentum)
"""
import json, os, sys, random, struct, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, '/tmp/eval')
from train_sft_3b import GPT, VOCAB_SIZE, DIM, DEPTH, N_HEADS, MAX_SEQ_LEN

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = '/tmp/sft_v3_runs/D/sft_model.pt'
TOKENIZER_PATH = '/tmp/eval/multilingual_32k.model'
RESULTS_DIR = '/tmp/experiments/paper1'
DATA_DIR = '/tmp/paper1_ner_data'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

LANGS = ['he', 'ar', 'fa', 'en']
NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}

TRAIN_STEPS = 800
LR = 2e-5
BATCH_TOKENS = 2048
SEQ_LEN = 256
MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES = 500

device = 'cuda'
EOS_ID = 3
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"

sp = spm.SentencePieceProcessor(TOKENIZER_PATH)

# ============================================================
# DATA PREPARATION
# ============================================================

def download_wikiann():
    """Download WikiANN NER dataset for all 4 languages."""
    from datasets import load_dataset
    
    data = {}
    for lang in LANGS:
        print(f"  Downloading WikiANN/{lang}...")
        try:
            ds = load_dataset('wikiann', lang, trust_remote_code=True)
            data[lang] = ds
            print(f"    train: {len(ds['train'])}, val: {len(ds.get('validation', []))}, test: {len(ds['test'])}")
        except Exception as e:
            print(f"    WARNING: Failed to load {lang}: {e}")
            data[lang] = None
    return data


def format_ner_sample(tokens, ner_tags):
    """
    Format a NER sample as instruction/output for our model.
    Input: list of tokens and list of NER tag ids (WikiANN uses 0=O, 1=B-PER, 2=I-PER, 3=B-ORG, 4=I-ORG, 5=B-LOC, 6=I-LOC)
    Output: instruction-response format for token classification.
    """
    # WikiANN tag mapping
    wikiann_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    
    text = ' '.join(tokens)
    # Create labeled output: only mention non-O entities
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, tag_id in zip(tokens, ner_tags):
        tag = wikiann_labels[tag_id]
        if tag.startswith('B-'):
            if current_entity:
                entities.append(f"{' '.join(current_tokens)} [{current_entity}]")
            current_entity = tag[2:]
            current_tokens = [token]
        elif tag.startswith('I-') and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append(f"{' '.join(current_tokens)} [{current_entity}]")
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append(f"{' '.join(current_tokens)} [{current_entity}]")
    
    if not entities:
        output = "No named entities found."
    else:
        output = '; '.join(entities)
    
    instruction = f"Identify all named entities (PER, ORG, LOC) in the following text:\n{text}"
    
    return {'instruction': instruction, 'output': output}


def prepare_ner_data(wikiann_data):
    """Prepare NER samples as instruction/output JSONL files."""
    for lang in LANGS:
        if wikiann_data[lang] is None:
            continue
        
        ds = wikiann_data[lang]
        
        # Train split
        train_samples = []
        train_ds = ds['train']
        indices = list(range(len(train_ds)))
        random.shuffle(indices)
        for idx in indices[:MAX_TRAIN_SAMPLES]:
            item = train_ds[idx]
            sample = format_ner_sample(item['tokens'], item['ner_tags'])
            train_samples.append(sample)
        
        train_path = f'{DATA_DIR}/ner_train_{lang}.jsonl'
        with open(train_path, 'w') as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"  {lang} train: {len(train_samples)} samples → {train_path}")
        
        # Eval split (test set)
        eval_samples = []
        test_ds = ds['test']
        indices = list(range(len(test_ds)))
        random.shuffle(indices)
        for idx in indices[:MAX_EVAL_SAMPLES]:
            item = test_ds[idx]
            sample = format_ner_sample(item['tokens'], item['ner_tags'])
            eval_samples.append(sample)
        
        eval_path = f'{DATA_DIR}/ner_eval_{lang}.jsonl'
        with open(eval_path, 'w') as f:
            for s in eval_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        print(f"  {lang} eval: {len(eval_samples)} samples → {eval_path}")


# ============================================================
# MODEL UTILITIES (same pattern as run_exp_hi.py)
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


def train_sft(model, train_file, val_file, output_path, steps=TRAIN_STEPS, lr=LR):
    """SFT training loop — model in fp16, SGD no momentum."""
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
            print(f"    Step {step}/{steps}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict()}, output_path)
            
            model.train()
    
    print(f"    Best val_loss: {best_val_loss:.4f}")
    return best_val_loss


# ============================================================
# NER EVALUATION
# ============================================================

def compute_entity_f1(predicted_text, expected_text):
    """
    Compute entity-level F1 between predicted and expected NER output.
    Both are semicolon-separated strings like "token [TYPE]; token2 [TYPE2]"
    """
    def parse_entities(text):
        entities = set()
        if 'no named entities' in text.lower():
            return entities
        parts = text.split(';')
        for part in parts:
            part = part.strip()
            if '[' in part and ']' in part:
                entities.add(part.lower().strip())
        return entities
    
    pred_entities = parse_entities(predicted_text)
    gold_entities = parse_entities(expected_text)
    
    if not gold_entities and not pred_entities:
        return 1.0, 1.0, 1.0  # Both empty = perfect
    if not gold_entities:
        return 0.0, 1.0, 0.0 if not pred_entities else (0.0, 0.0, 0.0)
    if not pred_entities:
        return 0.0, 0.0, 0.0
    
    tp = len(pred_entities & gold_entities)
    precision = tp / len(pred_entities) if pred_entities else 0
    recall = tp / len(gold_entities) if gold_entities else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


@torch.no_grad()
def eval_ner(model, eval_file, max_new=80):
    """Evaluate NER by generating and computing entity F1."""
    model.eval()
    
    with open(eval_file) as f:
        samples = [json.loads(line) for line in f]
    
    all_f1 = []
    all_precision = []
    all_recall = []
    
    for s in samples[:MAX_EVAL_SAMPLES]:
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
        
        precision, recall, f1 = compute_entity_f1(generated, expected)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
    
    n = len(all_f1)
    avg_p = sum(all_precision) / n if n > 0 else 0
    avg_r = sum(all_recall) / n if n > 0 else 0
    avg_f1 = sum(all_f1) / n if n > 0 else 0
    
    return {'precision': avg_p, 'recall': avg_r, 'f1': avg_f1, 'n': n}


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Paper 1: NER Cross-Lingual Transfer Experiment")
    print("=" * 60)
    
    all_results = {}
    
    # Step 1: Download and prepare data
    print("\n[1/4] Downloading WikiANN NER data...")
    wikiann_data = download_wikiann()
    
    print("\n[2/4] Preparing NER instruction data...")
    prepare_ner_data(wikiann_data)
    
    # Step 2: Baseline evaluation (D-SFT, no NER training)
    print("\n[3/4] Evaluating conditions...")
    print("\n--- Baseline: D-SFT (no NER training) ---")
    model = load_model(MODEL_PATH, half=True)
    all_results['baseline'] = {}
    for lang in LANGS:
        eval_file = f'{DATA_DIR}/ner_eval_{lang}.jsonl'
        if os.path.exists(eval_file):
            metrics = eval_ner(model, eval_file)
            print(f"  {lang}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} (n={metrics['n']})")
            all_results['baseline'][lang] = metrics
    del model
    torch.cuda.empty_cache()
    
    # Step 3: HE-only NER training
    print("\n--- HE-only: Fine-tuned on Hebrew NER only ---")
    train_file_he = f'{DATA_DIR}/ner_train_he.jsonl'
    if os.path.exists(train_file_he):
        with open(train_file_he) as f:
            he_samples = [json.loads(line) for line in f]
        
        random.shuffle(he_samples)
        n_val = max(200, int(len(he_samples) * 0.1))
        val_samples = he_samples[:n_val]
        train_samples = he_samples[n_val:]
        
        train_ids = tokenize_samples(train_samples)
        val_ids = tokenize_samples(val_samples)
        
        train_bin = f'{DATA_DIR}/ner_he_train.bin'
        val_bin = f'{DATA_DIR}/ner_he_val.bin'
        save_bin(train_ids, train_bin)
        save_bin(val_ids, val_bin)
        print(f"  Data: {len(train_samples)} train ({len(train_ids)} tokens), {n_val} val ({len(val_ids)} tokens)")
        
        model = load_model(MODEL_PATH, half=True)
        he_model_path = f'{DATA_DIR}/ner_he_only_model.pt'
        best_val = train_sft(model, train_bin, val_bin, he_model_path, steps=TRAIN_STEPS, lr=LR)
        
        del model
        torch.cuda.empty_cache()
        model = load_model(he_model_path, half=True)
        
        all_results['he_only'] = {'val_loss': best_val}
        for lang in LANGS:
            eval_file = f'{DATA_DIR}/ner_eval_{lang}.jsonl'
            if os.path.exists(eval_file):
                metrics = eval_ner(model, eval_file)
                print(f"  {lang}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} (n={metrics['n']})")
                all_results['he_only'][lang] = metrics
        
        del model
        torch.cuda.empty_cache()
    
    # Step 4: All-langs NER training (upper bound)
    print("\n--- All-langs: Fine-tuned on HE+AR+FA+EN NER ---")
    all_train_samples = []
    for lang in LANGS:
        train_file = f'{DATA_DIR}/ner_train_{lang}.jsonl'
        if os.path.exists(train_file):
            with open(train_file) as f:
                samples = [json.loads(line) for line in f]
            all_train_samples.extend(samples)
    
    if all_train_samples:
        random.shuffle(all_train_samples)
        n_val = max(400, int(len(all_train_samples) * 0.05))
        val_samples = all_train_samples[:n_val]
        train_samples = all_train_samples[n_val:]
        
        train_ids = tokenize_samples(train_samples)
        val_ids = tokenize_samples(val_samples)
        
        train_bin = f'{DATA_DIR}/ner_all_train.bin'
        val_bin = f'{DATA_DIR}/ner_all_val.bin'
        save_bin(train_ids, train_bin)
        save_bin(val_ids, val_bin)
        print(f"  Data: {len(train_samples)} train ({len(train_ids)} tokens), {n_val} val ({len(val_ids)} tokens)")
        
        model = load_model(MODEL_PATH, half=True)
        all_model_path = f'{DATA_DIR}/ner_all_langs_model.pt'
        best_val = train_sft(model, train_bin, val_bin, all_model_path, steps=TRAIN_STEPS, lr=LR)
        
        del model
        torch.cuda.empty_cache()
        model = load_model(all_model_path, half=True)
        
        all_results['all_langs'] = {'val_loss': best_val}
        for lang in LANGS:
            eval_file = f'{DATA_DIR}/ner_eval_{lang}.jsonl'
            if os.path.exists(eval_file):
                metrics = eval_ner(model, eval_file)
                print(f"  {lang}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} (n={metrics['n']})")
                all_results['all_langs'][lang] = metrics
        
        del model
        torch.cuda.empty_cache()
    
    # Step 5: Save results
    print("\n[4/4] Saving results...")
    results_path = f'{RESULTS_DIR}/ner_transfer_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 60)
    print("FINAL NER TRANSFER RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))
    
    # Upload to S3
    os.system(f"aws s3 cp {results_path} s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/paper1/ner_transfer_results.json --quiet")
    print("\nResults uploaded to S3!")


if __name__ == '__main__':
    main()
