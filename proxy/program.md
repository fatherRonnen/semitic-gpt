# Multilingual 1B GPT AutoResearch — Training Schedule Optimization

You are an autonomous AI researcher optimizing the **training schedule** for a multilingual 1B-parameter language model (Hebrew, Arabic, Farsi, English).

## Context

A full 1B training run peaked at step 2000/15000 (val BPB 12.35) then degraded continuously to 13.5+ by step 3500. The cosine warm-restart LR schedule decayed too aggressively — LR hit near-zero by step 4000, wasting 73% of planned compute.

**Your job:** Find the optimal training schedule through small-scale proxy experiments (150M params, ~2 min each), then the winner will be scaled to 1B.

## Architecture (FIXED — do NOT change)

```python
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000
DROPOUT = 0.1
```

For proxy experiments, use a scaled-down version:
```python
# Proxy model (~150M params)
DIM = 768
DEPTH = 12
N_HEADS = 12
```

The full 1B model uses DIM=1536, DEPTH=16, N_HEADS=12, but we test with the proxy.

## What We Know from the Failed Run

- **Optimizer**: Muon (2D+ params) + AdamW (1D params) — this worked well, keep it
- **Muon LR peak**: 0.05, AdamW LR peak: 3e-4
- **Schedule**: Cosine with 3 warm restarts over 15K steps → peaked at step 2000, then degraded
- **Batch**: 4×GPU with grad_accum=8, effective batch=256 sequences × 2048 tokens = 524K tokens/step
- **SWA**: Set to start at 80% but never reached (training stopped early)
- **Data**: ~4B tokens (EN 37.5%, HE 25%, FA 25%, AR 12.4%), shuffled with 32K SentencePiece vocab

## What to Explore (in order of expected impact)

### 1. LR Schedule Type
- **Warmup-Stable-Decay (WSD)**: warmup → constant LR → linear/cosine decay in final 20%
- **Linear decay**: simple, often competitive
- **Cosine (no restarts)**: single cycle, gentler than multi-restart
- **Trapezoidal**: warmup → plateau → linear ramp-down
- vs the failed **cosine with 3 warm restarts**

### 2. Total Training Steps / Tokens
- Original plan: 15K steps (~7.9B tokens = ~2 epochs over 4B data)
- The model peaked at 2K steps (~1B tokens = 0.25 epochs)
- Test: 2000 / 3000 / 5000 / 7500 steps
- Maybe 1 epoch (7600 steps) is the sweet spot

### 3. Peak LR
- Original: Muon 0.05, AdamW 3e-4
- Test: Muon 0.01–0.08, AdamW 1e-4–5e-4

### 4. Warmup Length
- Original: 500 steps (3.3% of 15K)
- Test: 100–1000 steps

### 5. Decay Schedule Parameters
- For WSD: stable fraction (50%–80%), decay type (linear vs cosine)
- For cosine: min LR ratio (0 vs 0.01 vs 0.1)
- For warm restarts: fewer cycles (1–2) vs 3

### 6. Weight Decay
- Original: 0.01 (AdamW only — Muon has no WD)
- Test: 0.01–0.1

### 7. SWA Timing
- Original: last 20% — but training died before then
- Test: last 10%, 20%, 30% of actual training steps
- Or disable SWA and rely on best-checkpoint

### 8. Batch Size / Grad Accumulation
- Original: effective 256 sequences
- Test: 128, 256, 512

## Data Paths

Training data is pre-tokenized binary (uint16, vocab 32000):
```
data/train.bin          — ~4B tokens, shuffled
data/val.bin            — combined validation (20M tokens)
data/val_en.bin         — English validation
data/val_ar.bin         — Arabic validation
data/val_he.bin         — Hebrew validation
data/val_fa.bin         — Farsi validation
```

## Hardware

Single L40S GPU (48GB VRAM). Each experiment must complete in **2 minutes**.
With 150M proxy model on single GPU, you can fit batch_size=16 with seq_len=2048.

## Evaluation

At the end of training, evaluate on ALL val sets and print:
```
val_bpb=<combined>
val_en_bpb=<english>
val_ar_bpb=<arabic>
val_he_bpb=<hebrew>
val_fa_bpb=<farsi>
training_seconds=<time>
total_steps=<steps>
best_step=<step_with_best_val>
```

**The primary metric is `val_bpb` (combined).** Lower is better.

Also evaluate at 25%, 50%, 75%, 100% of training to track the loss curve — print these as:
```
eval_25pct_bpb=<val>
eval_50pct_bpb=<val>
eval_75pct_bpb=<val>
```

## Rules

1. **Only modify train.py** — prepare.py is read-only
2. **Output complete train.py** in a ```python block
3. **Start with a comment:** `# EXPERIMENT: <description>`
4. **Stay within 48GB VRAM**
5. **Must complete in 2 minutes** (proxy model is small)
6. **DO NOT change model architecture** (DIM/DEPTH/N_HEADS are fixed for proxy)
7. **Focus ONLY on training schedule/optimizer hyperparameters**
8. **Keep Muon + AdamW split** (2D+ params → Muon, 1D → AdamW)
9. **Print val_bpb= as the LAST output line** matching the regex `val_bpb=([0-9.]+)`
10. **Use data/ paths** — data files are pre-downloaded
11. **DO NOT use DDP/torchrun** — single GPU experiments
12. **DO NOT download any data** — it's already in data/
13. **torch.compile may fail** — wrap in try/except

## CRITICAL: Performance Budget

On this L40S GPU with the proxy model (dim=768, depth=12, batch=8, seq=2048):
- Forward+backward per step: ~170ms with bf16 autocast
- With grad_accum=2: ~340ms per optimizer step
- With Muon: add ~20% overhead → ~400ms per step

**Safe step budget: 500 steps maximum.** This takes ~3.5 min with eval overhead.
Do NOT exceed 500 steps unless you reduce the model or batch size.

Use `GRAD_ACCUM = 1` (no accumulation) for faster iteration — saves 50% time.
With batch=8 × seq=2048 = 16K tokens/step × 500 steps = 8M tokens total.

## ABSOLUTE BAN: torch.compile

**DO NOT use torch.compile() under any circumstances.** It takes 5+ minutes to compile on this system, which exceeds the entire training budget. This has caused every experiment to time out so far.

Just run the model in eager mode. The L40S is fast enough.

## Data File Paths (MANDATORY)

The data files are:
```
data/train.bin          — 4B tokens (NOT train_morphology.bin)
data/val.bin            — combined validation  
data/val_en.bin         — English validation
data/val_ar.bin         — Arabic validation
data/val_he.bin         — Hebrew validation
data/val_fa.bin         — Farsi validation
```

**Use prepare.py's DATA_DIR and DataLoader to load data.** The prepare.py already handles everything.
Example:
```python
from prepare import DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DATA_DIR
train_dl = DataLoader(os.path.join(DATA_DIR, "train.bin"), MAX_SEQ_LEN, 8)
val_bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"), device="cuda")
```

## CRITICAL CONSTRAINTS (READ THIS FIRST)

### Performance Budget
- Each Muon step takes ~1.25 seconds (Newton-Schulz is expensive on L40S)
- **Maximum 200 training steps** (250s training + 100s eval = 350s < 480s timeout)
- Do NOT use torch.compile() — compilation alone takes 5+ minutes, exceeds budget
- Do NOT use gradient accumulation (GRAD_ACCUM=1) unless you reduce steps proportionally

### Data Files
Files are at `data/train.bin`, `data/val.bin`, `data/val_en.bin`, etc. (NOT morphology names)
Use prepare.py:
```python
from prepare import DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DATA_DIR
import os
train_dl = DataLoader(os.path.join(DATA_DIR, "train.bin"), MAX_SEQ_LEN, 8)
bpb = evaluate_bpb(model, os.path.join(DATA_DIR, "val.bin"), device="cuda")
```

### What NOT to Do
- NO torch.compile()
- NO more than 200 steps
- NO GRAD_ACCUM > 1 (without reducing steps)
- NO downloading data (it's already there)
- NO importing from prepare.py things that don't exist (only: DataLoader, evaluate_bpb, VOCAB_SIZE, MAX_SEQ_LEN, DEVICE_BATCH_SIZE, DATA_DIR)
