# EXPERIMENT: batch=16, 1000 steps, LR=5e-4, WSD (stable→800, decay→1000) — double tokens vs exp_7 by 2× batch

"""
Analysis of exp_7 (val_bpb=43.3470, 1500 steps, LR=5e-4, batch=8):
- Best so far. Tokens: 1500 × 8 × 2048 = 24.6M
- exp_8 (LR=8e-4) was worse → LR=5e-4 is the sweet spot
- exp_9 (WD=0.05 + 2000 steps) → FAILED (timeout or divergence from WD=0.05)
- Key bottleneck: only 24.6M tokens — trivially small for 150M params

Strategy for exp_10:
1. batch_size = 16 (double from 8) → 2× tokens per step
   Timing: 1000 × 340ms = 340s + ~70s eval ≈ 410s < 480s budget
   Tokens: 1000 × 16 × 2048 = 32.8M (33% MORE than exp_7's 24.6M)
2. TOTAL_STEPS = 1000 — compensate for slower steps
3. LR = 5e-4 (proven best — do NOT chase 8e-4 which failed)
   Linear scaling rule suggests 2× batch → √2× LR ≈ 7e-4, but exp_8 showed 8e-4 hurts
   Keep 5e-4 — safer, already well-tuned
4. WSD schedule:
   - Warmup: 0→100 steps (same fraction as before)
   - Stable: 100→800 steps (80% of total at peak LR)
   - Decay: 800→1000 cosine to 5% of peak
5. SWA: start at step 800 (80%), freq=20
   - 200 decay steps / 20 = 10 checkpoints
6. Keep: label_smoothing=0.1, WD=0.01, clip=1.0, bf16 autocast
7. Mid-evals: only at 50% (step 500) and 100% (step 1000) — minimal eval overhead
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DEVICE_BATCH_SIZE, DATA_DIR

# ============ PROXY MODEL CONFIG (FIXED — do not change) ============
DIM        = 768
DEPTH      = 12
N_HEADS    = 12
ROPE_THETA = 10000
DROPOUT    = 0.1

# ============ TRAINING CONFIG ============
TOTAL_STEPS  = 1000
WARMUP_STEPS = 100       # 10% warmup
STABLE_END   = 800       # stable phase through 80% of training
MIN_LR_RATIO = 0.05      # decay to 5% of peak

BATCH_SIZE = 16          # KEY CHANGE: doubled from 8 → more tokens/step
GRAD_ACCUM = 1

ADAMW_LR    = 5e-4       # proven best (exp_7); keep stable
ADAMW_WD    = 0.01       # proven stable
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS   = 1e-8

LABEL_SMOOTHING = 0.1
GRAD_CLIP       = 1.0

# SWA config
SWA_START_FRAC = 0.80    # step 800
SWA_FREQ       = 20

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
        self.up   = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim,  bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.qkv  = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim,   bias=False)
        self.attn_dropout = dropout
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.ln1  = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2  = RMSNorm(dim)
        self.mlp  = SwiGLU(dim, mlp_dim)
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
        self.head.weight = self.tok_emb.weight  # weight tying

        hd     = dim // n_heads
        freqs  = 1.0 / (rope_theta ** (torch.arange(0, hd, 2).float() / hd))
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
        x   = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

# ============ WSD LR SCHEDULE ============
def wsd_lr(step, total_steps, warmup_steps, stable_end, min_lr_ratio, base_lr):
    """
    Warmup-Stable-Decay (WSD):
    - [0, warmup_steps): linear warmup 0 → base_lr
    - [warmup_steps, stable_end): constant base_lr
    - [stable_end, total_steps]: cosine decay base_lr → min_lr_ratio * base_lr
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        return base_lr
    else:
        progress = (step - stable_end) / max(total_steps - stable_end, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

# ============ SWA ============
class SWAModel:
    """Manual Stochastic Weight Averaging — running mean of model weights."""
    def __init__(self):
        self.avg_state  = None
        self.n_averaged = 0

    def update(self, model):
        state = {k: v.cpu().float().clone() for k, v in model.state_dict().items()}
        if self.avg_state is None:
            self.avg_state = state
            self.n_averaged = 1
        else:
            n = self.n_averaged
            for k in self.avg_state:
                self.avg_state[k] = (self.avg_state[k] * n + state[k]) / (n + 1)
            self.n_averaged += 1

    def apply_to(self, model, device):
        """Load averaged weights into model in-place."""
        if self.avg_state is None:
            return
        model.load_state_dict({k: v.to(device) for k, v in self.avg_state.items()})

# ============ MAIN ============
def main():
    device = torch.device(DEVICE)
    print(f"=== Multilingual Proxy Training — WSD + SWA, {TOTAL_STEPS} steps, batch={BATCH_SIZE}, LR={ADAMW_LR} ===")

    torch.manual_seed(42)
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"Schedule: WSD | warmup={WARMUP_STEPS} | stable_end={STABLE_END} | total={TOTAL_STEPS}")
    print(f"AdamW LR={ADAMW_LR}, WD={ADAMW_WD}, betas={ADAMW_BETAS}")
    print(f"Label smoothing={LABEL_SMOOTHING}")

    swa_start_step = int(TOTAL_STEPS * SWA_START_FRAC)
    print(f"SWA start: step {swa_start_step}, freq={SWA_FREQ}")

    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    total_tokens    = TOTAL_STEPS * tokens_per_step
    print(f"Tokens/step: {tokens_per_step:,} | Total: {total_tokens:,}")
    print(f"(exp_7 had 24.6M tokens; this experiment: {total_tokens/1e6:.1f}M = {total_tokens/24576000*100:.0f}% of exp_7)")

    # Sanity check
    with torch.no_grad():
        dummy  = torch.zeros(1, 8, dtype=torch.long, device=device)
        logits = model(dummy)
        init_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), dummy.view(-1))
        init_bpb  = init_loss.item() / math.log(2)
        expected  = math.log2(VOCAB_SIZE)
        print(f"Sanity — init BPB={init_bpb:.3f} (random ≈{expected:.3f})")

    # Data
    train_loader = DataLoader(
        os.path.join(DATA_DIR, "train.bin"),
        MAX_SEQ_LEN,
        BATCH_SIZE
    )

    # AdamW for all parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ADAMW_LR,
        weight_decay=ADAMW_WD,
        betas=ADAMW_BETAS,
        eps=ADAMW_EPS
    )
    print(f"AdamW trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    swa = SWAModel()

    best_val_bpb = float('inf')
    best_step    = -1
    best_state   = None

    # Only eval at 50% and 100% during training to save time
    eval_points = {
        max(1, int(TOTAL_STEPS * 0.50)): "50pct",
        TOTAL_STEPS:                      "100pct",
    }
    eval_results = {"25pct": 0.0, "50pct": 0.0, "75pct": 0.0}

    loss_history = []
    start_time   = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        model.train()

        # WSD LR schedule
        lr = wsd_lr(step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss   = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                label_smoothing=LABEL_SMOOTHING
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Log raw NLL (no label smoothing) for comparability
        with torch.no_grad():
            raw_loss  = F.cross_entropy(logits.view(-1, logits.size(-1)).detach(), y.view(-1))
            train_bpb = raw_loss.item() / math.log(2)
        loss_history.append(train_bpb)

        # SWA accumulation
        if step >= swa_start_step and step % SWA_FREQ == 0:
            swa.update(model)

        # Progress log every 100 steps
        if step % 100 == 0:
            elapsed = time.time() - start_time
            avg_bpb = sum(loss_history[-100:]) / min(100, len(loss_history))
            phase   = ("warmup" if step < WARMUP_STEPS
                       else ("stable" if step < STABLE_END else "decay"))
            print(f"Step {step:4d}/{TOTAL_STEPS} [{phase}] | "
                  f"TrainBPB≈{train_bpb:.4f} (avg100={avg_bpb:.4f}) | "
                  f"LR={lr:.6f} | gnorm={grad_norm:.3f} | {elapsed:.0f}s")

        # Mid-training evals
        if step in eval_points:
            model.eval()
            with torch.no_grad():
                val_bpb = evaluate_bpb(
                    model,
                    os.path.join(DATA_DIR, "val.bin"),
                    device=DEVICE
                )
            model.train()

            label   = eval_points[step]
            eval_results[label] = val_bpb
            elapsed = time.time() - start_time
            swa_info = (f" [SWA: {swa.n_averaged} ckpts]"
                        if swa.n_averaged > 0 else "")
            print(f"  EVAL {label}: val_bpb={val_bpb:.4f}  (step={step}, {elapsed:.0f}s){swa_info}")

            if val_bpb < best_val_bpb:
                best_val_bpb = val_bpb
                best_step    = step
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val_bpb={val_bpb:.4f} at step {step}")

    elapsed = time.time() - start_time
    print(f"\n=== Training complete: {elapsed:.1f}s, {TOTAL_STEPS} steps ===")
    print(f"SWA: {swa.n_averaged} checkpoints (from step {swa_start_step})")

    # ===== Final Model Selection =====
    if swa.n_averaged > 0:
        print(f"Applying SWA ({swa.n_averaged} averaged checkpoints) to model...")
        swa.apply_to(model, device)
        print(f"SWA applied — using SWA model for final eval")
    elif best_state is not None:
        print(f"Loading best raw checkpoint (step {best_step}, bpb={best_val_bpb:.4f})")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ===== Final Evaluation =====
    print(f"\n=== Final Evaluation ===")
    model.eval()
    with torch.no_grad():
        val_bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"),    device=DEVICE)
        val_en  = evaluate_bpb(model, os.path.join(DATA_DIR, "val_en.bin"), device=DEVICE)
        val_ar  = evaluate_bpb(model, os.path.join(DATA_DIR, "val_ar.bin"), device=DEVICE)
        val_he  = evaluate_bpb(model, os.path.join(DATA_DIR, "val_he.bin"), device=DEVICE)
        val_fa  = evaluate_bpb(model, os.path.join(DATA_DIR, "val_fa.bin"), device=DEVICE)

    total_elapsed = time.time() - start_time

    print(f"\n=== RESULTS ===")
    print(f"eval_25pct_bpb={eval_results.get('25pct', 0.0):.4f}")
    print(f"eval_50pct_bpb={eval_results.get('50pct', 0.0):.4f}")
    print(f"eval_75pct_bpb={eval_results.get('75pct', 0.0):.4f}")
    print(f"val_en_bpb={val_en:.4f}")
    print(f"val_ar_bpb={val_ar:.4f}")
    print(f"val_he_bpb={val_he:.4f}")
    print(f"val_fa_bpb={val_fa:.4f}")
    print(f"training_seconds={total_elapsed:.1f}")
    print(f"total_steps={TOTAL_STEPS}")
    print(f"best_step={best_step}")
    print(f"swa_checkpoints={swa.n_averaged}")
    print(f"val_bpb={val_bpb:.4f}")

if __name__ == "__main__":
    main()