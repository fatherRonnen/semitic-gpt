# EXPERIMENT: WSD linear-decay + SWA 40%/freq=5 (denser) + 1700 steps + min_lr=0.03 + label_smooth=0.04

"""
Building on exp_19 (best, val_bpb=41.7454):
- WSD linear-decay, SWA from 40%/freq=8, 1700 steps, LR=5e-4, WD=0.02
- label_smooth=0.06, min_lr_ratio=0.03

Changes for exp_20 (two minimal, targeted tweaks):

1. label_smoothing: 0.06 → 0.04
   - Consistent improvement trend across experiments:
       exp_8:  label_smooth=0.10, val_bpb=43.82
       exp_15: label_smooth=0.08, val_bpb=41.80 (−2.02)
       exp_19: label_smooth=0.06, val_bpb=41.75 (−0.05)
   - Each reduction has improved BPB; continuing to 0.04
   - Risk: potential overfitting, but SWA+WD=0.02 provide regularization
   - At 27M training tokens, model is not overfit — direct gradient signal helps
   - 0.04 is still non-zero, so maintains token-distribution robustness

2. SWA_FREQ: 8 → 5 (denser weight averaging)
   - SWA start = step 680 (40% of 1700), runs to step 1700
   - With freq=8: (1700-680)//8 = 127 checkpoints
   - With freq=5: (1700-680)//5 = 204 checkpoints (~61% more)
   - More checkpoints = smoother average trajectory in weight space
   - Overhead: ~77 extra dict copies of CPU floats (~150M params × 4 bytes × 77 = ~46MB, negligible)
   - No additional forward passes, no timing impact
   - Why freq=5 not freq=3: freq=3 gives 340 checkpoints, possibly too correlated (too frequent)
     freq=5 = sweet spot between diversity and density

3. Everything else IDENTICAL to exp_19:
   - TOTAL_STEPS=1700, WARMUP_STEPS=100, STABLE_END=1190 (70%)
   - ADAMW_LR=5e-4, WD=0.02, betas=(0.9,0.95), eps=1e-8
   - MIN_LR_RATIO=0.03
   - SWA_START_FRAC=0.40 (step 680)
   - BATCH_SIZE=8, GRAD_ACCUM=1
   - Evals at 50% and 100% only

Timing: 1700 × ~170ms = 289s training + ~90s eval = ~379s < 480s budget
Hypothesis: label_smooth=0.04 + denser SWA → ~0.1-0.3 BPB improvement
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
TOTAL_STEPS  = 1700
WARMUP_STEPS = 100       # linear warmup to peak LR
STABLE_END   = 1190      # 70% of 1700 — proven optimal ratio
MIN_LR_RATIO = 0.03      # proven: deeper decay → flatter minima for SWA

BATCH_SIZE = 8
GRAD_ACCUM = 1

ADAMW_LR    = 5e-4       # proven optimal
ADAMW_WD    = 0.02       # proven optimal
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS   = 1e-8

LABEL_SMOOTHING = 0.04   # 0.06→0.04: continue consistent improvement trend
GRAD_CLIP       = 1.0

# SWA config
SWA_START_FRAC = 0.40   # step 680 — proven optimal
SWA_FREQ       = 5      # 8→5: denser averaging (~204 vs 127 checkpoints)

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

# ============ WSD LR SCHEDULE — LINEAR DECAY ============
def wsd_lr_linear(step, total_steps, warmup_steps, stable_end, min_lr_ratio, base_lr):
    """
    WSD with LINEAR decay phase:
    - [0, warmup_steps): linear warmup 0 → base_lr
    - [warmup_steps, stable_end): constant base_lr
    - [stable_end, total_steps]: LINEAR decay base_lr → min_lr_ratio * base_lr
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        return base_lr
    else:
        progress = (step - stable_end) / max(total_steps - stable_end, 1)
        return base_lr * (1.0 - progress * (1.0 - min_lr_ratio))

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
        if self.avg_state is None:
            return
        model.load_state_dict({k: v.to(device) for k, v in self.avg_state.items()})

# ============ MAIN ============
def main():
    device = torch.device(DEVICE)
    print(f"=== Exp 20: WSD linear + SWA 40%/freq=5 (denser) + 1700 steps + min_lr=0.03 + label_smooth=0.04 ===")

    torch.manual_seed(42)
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    swa_start_step = int(TOTAL_STEPS * SWA_START_FRAC)
    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    expected_swa_ckpts = (TOTAL_STEPS - swa_start_step) // SWA_FREQ

    print(f"Schedule: WSD-LINEAR | warmup={WARMUP_STEPS} | stable_end={STABLE_END} | total={TOTAL_STEPS}")
    print(f"AdamW LR={ADAMW_LR}, WD={ADAMW_WD}, betas={ADAMW_BETAS}")
    print(f"Label smoothing={LABEL_SMOOTHING}, min_lr_ratio={MIN_LR_RATIO}")
    print(f"SWA start: step {swa_start_step} ({SWA_START_FRAC*100:.0f}%), freq={SWA_FREQ}")
    print(f"Expected SWA checkpoints: {expected_swa_ckpts}")
    print(f"Tokens/step: {tokens_per_step:,} | Total: {TOTAL_STEPS * tokens_per_step:,}")

    # Sanity check
    with torch.no_grad():
        dummy  = torch.zeros(1, 8, dtype=torch.long, device=device)
        logits = model(dummy)
        init_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), dummy.view(-1))
        init_bpb  = init_loss.item() / math.log(2)
        expected_bpb = math.log2(VOCAB_SIZE)
        print(f"Sanity — init BPB={init_bpb:.3f} (random ≈{expected_bpb:.3f})")

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
    print(f"AdamW params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    swa = SWAModel()

    best_val_bpb = float('inf')
    best_step    = -1
    best_state   = None

    # Eval at 50% and 100% only (saves budget)
    eval_points = {
        max(1, int(TOTAL_STEPS * 0.50)): "50pct",
        TOTAL_STEPS:                      "100pct",
    }
    eval_results = {"25pct": 0.0, "50pct": 0.0, "75pct": 0.0}

    loss_history = []
    start_time   = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        model.train()

        # WSD-LINEAR LR schedule
        lr = wsd_lr_linear(step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
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

        # Log raw NLL (not smoothed loss) for comparability
        with torch.no_grad():
            raw_loss  = F.cross_entropy(logits.view(-1, logits.size(-1)).detach(), y.view(-1))
            train_bpb = raw_loss.item() / math.log(2)
        loss_history.append(train_bpb)

        # SWA accumulation — from 40% of training, every SWA_FREQ steps
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
                  f"LR={lr:.6f} | gnorm={grad_norm:.3f} | "
                  f"SWA_ckpts={swa.n_averaged} | {elapsed:.0f}s")

        # Mid-training evals
        if step in eval_points:
            model.eval()
            with torch.no_grad():
                val_bpb_mid = evaluate_bpb(
                    model,
                    os.path.join(DATA_DIR, "val.bin"),
                    device=DEVICE
                )
            model.train()

            label = eval_points[step]
            eval_results[label] = val_bpb_mid
            elapsed = time.time() - start_time
            print(f"  EVAL {label}: val_bpb={val_bpb_mid:.4f}  "
                  f"(step={step}, {elapsed:.0f}s, SWA={swa.n_averaged})")

            if val_bpb_mid < best_val_bpb:
                best_val_bpb = val_bpb_mid
                best_step    = step
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✓ New best val_bpb={val_bpb_mid:.4f} at step {step}")

    elapsed = time.time() - start_time
    print(f"\n=== Training complete: {elapsed:.1f}s, {TOTAL_STEPS} steps ===")
    print(f"SWA: {swa.n_averaged} checkpoints averaged (start={swa_start_step}, freq={SWA_FREQ})")

    # ===== Final Model Selection: compare SWA vs best raw checkpoint =====
    raw_model_bpb = best_val_bpb

    if swa.n_averaged > 0:
        print(f"\nEvaluating SWA model ({swa.n_averaged} checkpoints)...")
        original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        swa.apply_to(model, device)
        model.eval()
        with torch.no_grad():
            swa_bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"), device=DEVICE)
        print(f"SWA val_bpb={swa_bpb:.4f} vs best raw={raw_model_bpb:.4f}")

        if swa_bpb <= raw_model_bpb:
            print(f"→ Using SWA model (better by {raw_model_bpb - swa_bpb:.4f})")
            # model already has SWA weights applied
        else:
            print(f"→ Using best raw checkpoint (better by {swa_bpb - raw_model_bpb:.4f})")
            model.load_state_dict({k: v.to(device) for k, v in original_state.items()})
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    elif best_state is not None:
        print(f"No SWA collected — loading best raw checkpoint (step {best_step})")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ===== Final Evaluation on all val sets =====
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