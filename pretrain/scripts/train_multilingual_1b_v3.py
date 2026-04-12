#!/usr/bin/env python3
"""
Multilingual 1B GPT Training — v3 (Arabic-Rebalanced)

Identical architecture and hyperparams to v2 (which achieved BPB 3.77).
Only change: training data uses training-data-v2 with expanded Arabic
(~1B AR tokens, up from 497M).

Expected: Arabic BPB should improve from 4.70 significantly.
"""

import os
import sys
import json
import math
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# ============ MODEL CONFIG ============
VOCAB_SIZE = 32000
DIM = 1536
DEPTH = 16
N_HEADS = 12
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000
DROPOUT = 0.1

# ============ TRAINING CONFIG (identical to v2) ============
TOTAL_STEPS = 5000
WARMUP_STEPS = 200
STABLE_END = 3500
MIN_LR_RATIO = 0.03

BATCH_PER_GPU = 8
GRAD_ACCUM = 4

ADAMW_LR = 5e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_WD = 0.02
ADAMW_EPS = 1e-8

LABEL_SMOOTHING = 0.06
GRAD_CLIP = 1.0

SWA_START_FRAC = 0.40
SWA_FREQ = 20

EVAL_EVERY = 250
SAVE_EVERY = 1000

DATA_DIR = "/tmp/training-data"
CKPT_DIR = "/tmp/checkpoints"
LOG_FILE = "/tmp/training.log"
EVAL_FILE = "/tmp/eval_results.json"

S3_BUCKET = "autoresearch-dashboard-196766918360"
S3_PREFIX = "multilingual-7b"
VERSION = "v3"

# ============ MODEL (identical to v2) ============
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
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = dropout
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, n_heads=N_HEADS,
                 max_seq_len=MAX_SEQ_LEN, rope_theta=ROPE_THETA, dropout=DROPOUT):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        mlp_dim = ((int(2 * dim * 4 / 3) + 63) // 64) * 64
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_dim, dropout) for _ in range(depth)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        hd = dim // n_heads
        freqs = 1.0 / (rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

# ============ WSD LINEAR SCHEDULE ============
def wsd_lr_linear(step, total_steps, warmup_steps, stable_end, min_lr_ratio, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        return base_lr
    else:
        progress = (step - stable_end) / max(total_steps - stable_end, 1)
        return base_lr * (1.0 - progress * (1.0 - min_lr_ratio))

# ============ DATA LOADING ============
class BinaryDataset:
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
    def get_batch(self, batch_size, device, rng):
        ix = torch.from_numpy(rng.integers(0, self.n_tokens - self.seq_len - 1, size=(batch_size,)))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

def load_val_data(path, seq_len, max_batches=20, batch_size=8):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    n_tokens = len(data)
    batches = []
    stride = seq_len + 1
    all_starts = list(range(0, n_tokens - stride, stride))
    max_samples = max_batches * batch_size
    if len(all_starts) > max_samples:
        step_size = len(all_starts) // max_samples
        all_starts = all_starts[::step_size][:max_samples]
    for i in range(0, len(all_starts), batch_size):
        batch_starts = all_starts[i:i+batch_size]
        if len(batch_starts) < batch_size:
            break
        x = torch.stack([torch.from_numpy(data[s:s+seq_len].astype(np.int64)) for s in batch_starts])
        y = torch.stack([torch.from_numpy(data[s+1:s+1+seq_len].astype(np.int64)) for s in batch_starts])
        batches.append((x, y))
    return batches

@torch.no_grad()
def evaluate(model, val_batches, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_batches:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return avg_loss / math.log(2)

class Logger:
    def __init__(self, log_file, rank):
        self.rank = rank
        if rank == 0:
            self.f = open(log_file, 'w')
    def log(self, msg):
        if self.rank == 0:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            self.f.write(line + '\n')
            self.f.flush()
    def close(self):
        if self.rank == 0:
            self.f.close()

class SWAState:
    def __init__(self):
        self.avg_state = None
        self.n_averaged = 0
    def update(self, model):
        state = {k: v.cpu().float().clone() for k, v in model.module.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
            self.n_averaged = 1
        else:
            n = self.n_averaged
            for k in self.avg_state:
                self.avg_state[k] = (self.avg_state[k] * n + state[k]) / (n + 1)
            self.n_averaged += 1

# ============ GENERATION (new in v3) ============
@torch.no_grad()
def generate_samples(model, prompts, tokenizer_path, device, max_tokens=100):
    """Generate text samples for qualitative evaluation."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    
    results = []
    model.eval()
    for prompt_text in prompts:
        ids = sp.Encode(prompt_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            if input_ids.shape[1] >= MAX_SEQ_LEN:
                break
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
            next_logit = logits[0, -1]
            next_id = torch.multinomial(F.softmax(next_logit / 0.8, dim=-1), 1)
            if next_id.item() == sp.eos_id():
                break
            input_ids = torch.cat([input_ids, next_id.unsqueeze(0)], dim=1)
        
        output = sp.Decode(input_ids[0].tolist())
        results.append({"prompt": prompt_text, "output": output, "tokens_generated": input_ids.shape[1] - len(ids)})
    
    model.train()
    return results

# ============ MAIN ============
def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    effective_batch = BATCH_PER_GPU * GRAD_ACCUM * world_size
    tokens_per_step = effective_batch * MAX_SEQ_LEN

    logger = Logger(LOG_FILE, rank)
    logger.log(f"=== Multilingual 1B Training v3 (Arabic-Rebalanced) ===")
    logger.log(f"World size: {world_size}, Batch/GPU: {BATCH_PER_GPU}, Grad accum: {GRAD_ACCUM}")
    logger.log(f"Effective batch: {effective_batch} seqs = {tokens_per_step:,} tokens/step")
    logger.log(f"Total steps: {TOTAL_STEPS} = {TOTAL_STEPS * tokens_per_step:,} tokens")
    logger.log(f"Schedule: WSD-LINEAR | warmup={WARMUP_STEPS} | stable_end={STABLE_END} | total={TOTAL_STEPS}")
    logger.log(f"AdamW LR={ADAMW_LR}, betas={ADAMW_BETAS}, WD={ADAMW_WD}")
    logger.log(f"Label smoothing={LABEL_SMOOTHING}, min_lr={MIN_LR_RATIO}, grad_clip={GRAD_CLIP}")
    logger.log(f"SWA: start={int(TOTAL_STEPS*SWA_START_FRAC)}, freq={SWA_FREQ}")
    logger.log(f"Data: training-data-v2 (expanded Arabic)")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Data
    logger.log("Loading training data...")
    train_ds = BinaryDataset(f"{DATA_DIR}/train.bin", MAX_SEQ_LEN)
    logger.log(f"Train tokens: {train_ds.n_tokens:,}")

    logger.log("Loading validation data...")
    val_batches = load_val_data(f"{DATA_DIR}/val.bin", MAX_SEQ_LEN)
    val_lang_batches = {}
    for lang in ['en', 'ar', 'he', 'fa']:
        vpath = f"{DATA_DIR}/val_{lang}.bin"
        if os.path.exists(vpath):
            val_lang_batches[lang] = load_val_data(vpath, MAX_SEQ_LEN)
            logger.log(f"  val_{lang}: {len(val_lang_batches[lang])} batches")

    # Model
    logger.log("Creating model...")
    torch.manual_seed(42)
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_emb = n_params - model.tok_emb.weight.numel()
    logger.log(f"Model params: {n_params:,} (non-embedding: {n_params_no_emb:,})")

    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WD,
                                   betas=ADAMW_BETAS, eps=ADAMW_EPS)
    logger.log(f"Optimizer: AdamW, beta2={ADAMW_BETAS[1]}")

    swa = SWAState()
    swa_start_step = int(TOTAL_STEPS * SWA_START_FRAC)
    rng = np.random.default_rng(42 + rank)
    scaler = torch.amp.GradScaler('cuda')
    best_val_bpb = float('inf')
    eval_results = []
    tokens_processed = 0
    start_time = time.time()

    logger.log(f"Starting training...")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        lr = wsd_lr_linear(step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for micro in range(GRAD_ACCUM):
            x, y = train_ds.get_batch(BATCH_PER_GPU, device, rng)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                       label_smoothing=LABEL_SMOOTHING) / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        tokens_processed += tokens_per_step

        if step >= swa_start_step and step % SWA_FREQ == 0 and rank == 0:
            swa.update(model)

        if step % 50 == 0 and rank == 0:
            elapsed = time.time() - start_time
            tps = tokens_processed / elapsed
            bpb = accum_loss / math.log(2)
            phase = "warmup" if step < WARMUP_STEPS else ("stable" if step < STABLE_END else "decay")
            logger.log(f"Step {step}/{TOTAL_STEPS} [{phase}] | Loss: {accum_loss:.4f} | "
                       f"BPB: {bpb:.4f} | LR: {lr:.6f} | Tokens: {tokens_processed:,} | "
                       f"TPS: {tps:,.0f} | SWA: {swa.n_averaged} | {elapsed/60:.1f}min")

        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
            if rank == 0:
                logger.log(f"--- Evaluation at step {step} ---")
                combined_bpb = evaluate(model.module, val_batches, device)
                logger.log(f"  Combined val BPB: {combined_bpb:.4f}")
                result = {"step": step, "tokens": tokens_processed, "combined_bpb": combined_bpb}
                for lang, batches in val_lang_batches.items():
                    lang_bpb = evaluate(model.module, batches, device)
                    result[f"{lang}_bpb"] = lang_bpb
                    logger.log(f"  {lang} val BPB: {lang_bpb:.4f}")
                eval_results.append(result)
                with open(EVAL_FILE, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                if combined_bpb < best_val_bpb:
                    best_val_bpb = combined_bpb
                    torch.save(model.module.state_dict(), f"{CKPT_DIR}/best_model.pt")
                    logger.log(f"  New best! BPB: {combined_bpb:.4f}")
            dist.barrier()

        if step % SAVE_EVERY == 0 and rank == 0:
            ckpt = {
                'step': step, 'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                'best_val_bpb': best_val_bpb, 'tokens_processed': tokens_processed,
                'eval_results': eval_results, 'swa_n': swa.n_averaged,
            }
            torch.save(ckpt, f"{CKPT_DIR}/ckpt_step_{step}.pt")
            logger.log(f"Saved checkpoint at step {step}")
            os.system(f"aws s3 cp {CKPT_DIR}/best_model.pt s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_best_model.pt --quiet &")
            os.system(f"aws s3 cp {CKPT_DIR}/ckpt_step_{step}.pt s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_ckpt_step_{step}.pt --quiet &")
            os.system(f"aws s3 cp {EVAL_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_eval_results.json --quiet &")

    # Finalize
    if rank == 0:
        torch.save(model.module.state_dict(), f"{CKPT_DIR}/final_model.pt")
        logger.log("Saved final model")

        if swa.avg_state is not None and swa.n_averaged > 0:
            logger.log(f"Evaluating SWA model ({swa.n_averaged} checkpoints)...")
            swa_model = GPT().to(device)
            swa_load = {k: v.float().to(device) for k, v in swa.avg_state.items()}
            swa_model.load_state_dict(swa_load)
            swa_bpb = evaluate(swa_model, val_batches, device)
            logger.log(f"SWA model combined BPB: {swa_bpb:.4f} (vs best raw: {best_val_bpb:.4f})")
            swa_result = {"step": "swa", "combined_bpb": swa_bpb, "n_averaged": swa.n_averaged}
            for lang, batches in val_lang_batches.items():
                lang_bpb = evaluate(swa_model, batches, device)
                swa_result[f"{lang}_bpb"] = lang_bpb
                logger.log(f"  SWA {lang} BPB: {lang_bpb:.4f}")
            eval_results.append(swa_result)
            if swa_bpb < best_val_bpb:
                logger.log(f"SWA model is BETTER — saving as best")
                best_val_bpb = swa_bpb
            else:
                logger.log(f"Raw best model is better — keeping it")
            torch.save(swa_load, f"{CKPT_DIR}/swa_model.pt")
            with open(EVAL_FILE, 'w') as f:
                json.dump(eval_results, f, indent=2)
            del swa_model

        # Generate samples for qualitative comparison
        logger.log("Generating text samples...")
        prompts = [
            "The history of artificial intelligence began",
            "براشית ברא אלוהים",
            "المدع والطكنولوجيا بإسرائيل",
            "في بداية القرن العشرين",
            "اللغة العربية هي من",
            "تاريخ الحضارة الإسلامية",
            "تاریخ ایران باستان",
            "زبان فارسی یکی از",
        ]
        try:
            samples = generate_samples(model.module, prompts, "/tmp/tokenizer.model", device)
            with open(f"{CKPT_DIR}/generation_samples.json", 'w') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            logger.log(f"Generated {len(samples)} samples")
        except Exception as e:
            logger.log(f"Generation failed: {e}")

        # Final upload
        os.system(f"aws s3 sync {CKPT_DIR}/ s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}/")
        os.system(f"aws s3 cp {LOG_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_training.log")
        os.system(f"aws s3 cp {EVAL_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_eval_results.json")

        elapsed = time.time() - start_time
        logger.log(f"=== Training complete! Total time: {elapsed/3600:.2f}h ===")
        logger.log(f"Best combined BPB: {best_val_bpb:.4f}")
        logger.log(f"SWA checkpoints: {swa.n_averaged}")
        logger.log(f"Total tokens: {tokens_processed:,}")

    logger.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
