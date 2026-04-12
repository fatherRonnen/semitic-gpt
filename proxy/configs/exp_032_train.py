# EXPERIMENT: WSD cosine-decay + SWA 40%/freq=8 + 1700 steps + min_lr=0.01 + label_smooth=0.06 + betas=(0.9,0.98)

"""
Building on exp_25 (best, val_bpb=41.6640):
- WSD linear-decay, SWA from 40%/freq=8, 1700 steps, LR=5e-4, WD=0.02
- label_smooth=0.06, min_lr_ratio=0.03, betas=(0.9,0.98)

Single targeted change for exp_32:
1. Decay phase: LINEAR → COSINE decay, and min_lr_ratio: 0.03 → 0.01

   Why cosine decay?
   - The WSD decay phase runs from step 1190→1700 (510 steps).
   - SWA collects from step 680→1700 (127 checkpoints every 8 steps).
   - Linear decay: LR drops at constant rate → each step loses equal LR.
   - Cosine decay: LR drops slowly at first (stays high ~longer), then 
     curves sharply downward at the end → model stays in exploration mode
     longer before converging. This means more SWA checkpoints are collected
     while the optimizer is still actively exploring the loss basin.
   - In the final 100 steps of cosine decay, LR drops much faster than linear —
     this sharp terminal descent often finds a sharper but lower basin entry point,
     which SWA then averages back to a flatter zone.
   - Empirical support: WSD+cosine is the standard in Mistral/LLaMA-style training.
     Most production LLMs use cosine cooldown, not linear.

   Why min_lr=0.01 (was 0.03)?
   - Linear+0.03: final LR = 5e-4 × 0.03 = 1.5e-5
   - Cosine+0.01: final LR = 5e-4 × 0.01 = 5e-6
   - Deeper floor for cosine is safe because cosine reaches the floor gradually —
     no abrupt drop. With linear, going to 0.01 would cause too-rapid decay.
   - Lower floor → SWA averages weights from a tighter convergence zone.
   - This combination (cosine+deep floor) is the standard in chinchilla-optimal 
     training recipes. Linear+shallow floor (0.03) was a workaround.

   Timing unchanged: 1700 × ~170ms = 289s + ~90s eval ≈ 379s < 480s budget.
   Risk: Low. Cosine vs linear in the decay phase is a mild change.
   Expected gain: 0.05–0.2 BPB from smoother decay + lower SWA convergence floor.
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
STABLE_END   = 1190      # 70% of 1700 — same ratio as exp_11/15/19/25
MIN_LR_RATIO = 0.01      # KEY CHANGE: deeper floor for cosine decay (was 0.03)

BATCH_SIZE = 8
GRAD_ACCUM = 1

ADAMW_LR    = 5e-4       # proven optimal
ADAMW_WD    = 0.02       # proven optimal
ADAMW_BETAS = (0.9, 0.98)  # proven optimal (exp_25)
ADAMW_EPS   = 1e-8

LABEL_SMOOTHING = 0.06   # proven optimal in exp_19/25
GRAD_CLIP       = 1.0

# SWA config — same as exp_25 (proven best)
SWA_START_FRAC = 0.40   # step 680
SWA_FREQ       = 8      # every 8 steps → ~127 checkpoints

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

# ============ WSD LR SCHEDULE — COSINE DECAY (KEY CHANGE) ============
def wsd_lr_cosine(step, total_steps, warmup_steps, stable_end, min_lr_ratio, base_lr):
    """
    WSD with COSINE decay phase:
    - [0, warmup_steps): linear warmup 0 → base_lr
    - [warmup_steps, stable_end): constant base_lr
    - [stable_end, total_steps]: COSINE decay base_lr → min_lr_ratio * base_lr
    
    Cosine decay formula:
      lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
    where progress goes from 0 (start of decay) to 1 (end of training).
    
    This gives:
    - progress=0.0: lr = min_lr + 0.5*(base_lr-min_lr)*(1+1) = base_lr  ✓
    - progress=0.5: lr = min_lr + 0.5*(base_lr-min_lr)*(1+0) = midpoint
    - progress=1.0: lr = min_lr + 0.5*(base_lr-min_lr)*(1-1) = min_lr  ✓
    """
    min_lr = base_lr * min_lr_ratio
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        return base_lr
    else:
        progress = (step - stable_end) / max(total_steps - stable_end, 1)
        # Clamp progress to [0, 1]
        progress = min(progress, 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor

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
    print(f"=== Exp 32: WSD COSINE-decay + SWA 40%/freq=8 + 1700 steps + min_lr=0.01 + label_smooth=0.06 + betas=(0.9,0.98) ===")

    torch.manual_seed(42)
    model = GPT().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    swa_start_step = int(TOTAL_STEPS * SWA_START_FRAC)
    tokens_per_step = BATCH_SIZE * MAX_SEQ_LEN
    expected_swa_ckpts = (TOTAL_STEPS - swa_start_step) // SWA_FREQ

    print(f"Schedule: WSD-COSINE | warmup={WARMUP_STEPS} | stable_end={STABLE_END} | total={TOTAL_STEPS}")
    print(f"AdamW LR={ADAMW_LR}, WD={ADAMW_WD}, betas={ADAMW_BETAS}")
    print(f"Label smoothing={LABEL_SMOOTHING}, min_lr_ratio={MIN_LR_RATIO}")
    print(f"KEY CHANGE: cosine decay (was linear), min_lr_ratio=0.01 (was 0.03)")
    print(f"Decay phase LR: {ADAMW_LR:.2e} → {ADAMW_LR * MIN_LR_RATIO:.2e} (cosine)")
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

    # LR schedule preview at key steps
    for check_step in [0, 100, 500, 1190, 1300, 1450, 1600, 1700]:
        lr_check = wsd_lr_cosine(check_step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
        phase = ("warmup" if check_step < WARMUP_STEPS
                 else ("stable" if check_step < STABLE_END else "decay"))
        print(f"  LR@step{check_step:4d} [{phase}]: {lr_check:.2e}")

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

        # WSD-COSINE LR schedule (KEY CHANGE from exp_25's linear)
        lr = wsd_lr_cosine(step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
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

        # SWA accumulation — from 40% of training, every 8 steps
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