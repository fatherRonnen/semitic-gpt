# EXPERIMENT: Baseline — cosine warm-restart (3 cycles) matching failed 1B config
"""
Proxy model (~150M params) to test training schedules.
This baseline replicates the failed 1B run's schedule at smaller scale.
"""
import os
import sys
import math
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DEVICE_BATCH_SIZE, DATA_DIR

# ============ PROXY MODEL CONFIG ============
DIM = 768
DEPTH = 12
N_HEADS = 12
ROPE_THETA = 10000
DROPOUT = 0.1

# ============ TRAINING CONFIG ============
TOTAL_STEPS = 200
WARMUP_STEPS = 20
NUM_CYCLES = 3
BATCH_SIZE = 8
GRAD_ACCUM = 1  # no accumulation for speed

MUON_LR = 0.05
MUON_MOMENTUM = 0.95
ADAMW_LR = 3e-4
ADAMW_WD = 0.01

DEVICE = "cuda"

# ============ MODEL ============
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
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

# ============ MUON OPTIMIZER ============
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.ndim >= 2:
                    g = self._newton_schulz(g)
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                p.add_(buf, alpha=-lr)
    def _newton_schulz(self, G, steps=5):
        a, b, c = (3.4445, -4.7750, 2.0315)
        if G.ndim > 2:
            shape = G.shape
            G = G.reshape(G.shape[0], -1)
            reshaped = True
        else:
            reshaped = False
        X = G / (G.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))
        if reshaped:
            X = X.reshape(shape)
        return X

# ============ LR SCHEDULE ============
def cosine_warmrestart_lr(step, total_steps, warmup_steps, num_cycles, base_lr):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cycle_progress = (progress * num_cycles) % 1.0
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

# ============ MAIN ============
def main():
    device = torch.device(DEVICE)
    print(f"=== Multilingual Proxy Training (Baseline) ===")

    # Model
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # NO torch.compile — takes 5+ min on L40S, exceeds training budget

    # Data
    train_loader = DataLoader(os.path.join(DATA_DIR, "train.bin"), MAX_SEQ_LEN, BATCH_SIZE)

    # Optimizers
    muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    opt_muon = Muon(muon_params, lr=MUON_LR, momentum=MUON_MOMENTUM)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=ADAMW_LR, weight_decay=ADAMW_WD)

    scaler = torch.amp.GradScaler('cuda')
    best_val_bpb = float('inf')
    best_state = None

    eval_points = {
        int(TOTAL_STEPS * 0.25): "25pct",
        int(TOTAL_STEPS * 0.50): "50pct",
        int(TOTAL_STEPS * 0.75): "75pct",
    }
    eval_results = {}
    start_time = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        muon_lr = cosine_warmrestart_lr(step, TOTAL_STEPS, WARMUP_STEPS, NUM_CYCLES, MUON_LR)
        adamw_lr = cosine_warmrestart_lr(step, TOTAL_STEPS, WARMUP_STEPS, NUM_CYCLES, ADAMW_LR)
        for g in opt_muon.param_groups: g['lr'] = muon_lr
        for g in opt_adamw.param_groups: g['lr'] = adamw_lr

        opt_muon.zero_grad()
        opt_adamw.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(opt_muon)
        scaler.unscale_(opt_adamw)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt_muon)
        scaler.step(opt_adamw)
        scaler.update()

        if step % 200 == 0:
            elapsed = time.time() - start_time
            bpb = accum_loss / math.log(2)
            print(f"Step {step}/{TOTAL_STEPS} | Loss: {accum_loss:.4f} | BPB: {bpb:.4f} | "
                  f"LR: {muon_lr:.6f} | {elapsed:.0f}s")

        # Eval at checkpoints
        if step in eval_points:
            val_bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"), device=DEVICE)
            label = eval_points[step]
            eval_results[label] = val_bpb
            print(f"eval_{label}_bpb={val_bpb:.4f}")
            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    # Final eval
    elapsed = time.time() - start_time

    # Restore best if we have one
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    val_bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"), device=DEVICE)
    if val_bpb < best_val_bpb:
        best_val_bpb = val_bpb

    val_en = evaluate_bpb(model, os.path.join(DATA_DIR, "val_en.bin"), device=DEVICE)
    val_ar = evaluate_bpb(model, os.path.join(DATA_DIR, "val_ar.bin"), device=DEVICE)
    val_he = evaluate_bpb(model, os.path.join(DATA_DIR, "val_he.bin"), device=DEVICE)
    val_fa = evaluate_bpb(model, os.path.join(DATA_DIR, "val_fa.bin"), device=DEVICE)

    print(f"\n=== RESULTS ===")
    for label, bpb in eval_results.items():
        print(f"eval_{label}_bpb={bpb:.4f}")
    print(f"val_en_bpb={val_en:.4f}")
    print(f"val_ar_bpb={val_ar:.4f}")
    print(f"val_he_bpb={val_he:.4f}")
    print(f"val_fa_bpb={val_fa:.4f}")
    print(f"training_seconds={elapsed:.1f}")
    print(f"total_steps={TOTAL_STEPS}")
    print(f"val_bpb={best_val_bpb:.4f}")

if __name__ == "__main__":
    main()
