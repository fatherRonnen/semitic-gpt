# Proxy Experiment Recipe Selection

## Summary

We conducted **32 proxy experiments** (small-scale, ~58M parameter models trained on the same data split for 500-2000 steps) plus **2 Muon architecture experiments** to identify the optimal training recipe for the full 3B model.

## Key Findings

### 1. AdamW > Muon at Scale

The Muon optimizer showed promise at high learning rates (LR=0.05 achieved BPB=9.39 at 58M scale) but proved unstable:
- Muon at LR=0.02: BPB 13.31 (reasonable)
- Muon at LR=0.05: BPB 9.39 (best Muon result)  
- However, Muon caused divergence in multiple proxy experiments (exp_004: BPB=65.43)
- AdamW proved more stable and predictable across configurations

**Decision:** Use AdamW for the 3B run for stability at scale.

### 2. WSD Schedule with Linear Decay

The best proxy experiments used Warmup-Stable-Decay (WSD) schedule with linear decay:
- **exp_025** (best overall): val_bpb = **41.66** — WSD linear-decay + SWA 40%/freq=8 + 1700 steps + min_lr=0.03 + label_smooth=0.06 + betas=(0.9, 0.98)
- **exp_019**: val_bpb = 41.75 — WSD linear-decay + SWA 40%/freq=8 + 1700 steps + min_lr=0.03 + label_smooth=0.06
- **exp_015**: val_bpb = 41.80 — WSD linear-decay + SWA from 40% freq=8 + 1700 steps + label_smooth=0.08

However, for the 3B model, **cosine decay** was chosen for stability at longer training runs (20K steps), with the understanding that WSD might perform slightly better (confirmed in post-hoc analysis).

### 3. Stochastic Weight Averaging (SWA) Helps

Experiments with SWA (averaging weights from 40% of training, frequency=8) consistently outperformed those without:
- Without SWA (exp_003): 55.29
- With SWA (exp_011): 42.84
- Best with SWA (exp_025): 41.66

### 4. Optimal Hyperparameters Identified

From the top-5 proxy experiments:

| Rank | Exp | BPB   | Key Config |
|------|-----|-------|------------|
| 1    | 025 | 41.66 | WSD linear, SWA@40%/8, betas=(0.9,0.98), label_smooth=0.06 |
| 2    | 019 | 41.75 | WSD linear, SWA@40%/8, label_smooth=0.06 |
| 3    | 015 | 41.80 | WSD linear, SWA@40%/8, label_smooth=0.08 |
| 4    | 011 | 42.84 | WSD, SWA@step750/freq=10, 1500 steps |
| 5    | 007 | 43.35 | WSD, 1500 steps, LR=5e-4, label_smooth=0.1 |

### 5. Failure Modes

11 out of 32 experiments failed (val_bpb=99.0, indicating divergence):
- Most failures involved: too-dense SWA (freq<6), EMA tracking, aggressive min_lr ratios (<0.02), or longer training without proper LR decay
- GradScaler with bf16 caused failures (exp_004 onwards removed it)
- Muon optimizer caused instability at conservative LRs

### 6. Final Recipe for 3B Training

Based on proxy results, the following recipe was selected:
- **Optimizer:** AdamW (lr=3e-4, betas=(0.9, 0.95), wd=0.1)
- **Schedule:** Cosine decay with 2000-step warmup
- **Batch size:** 512K tokens (gradient accumulation)
- **Precision:** bf16 (no GradScaler needed)
- **Warmup:** 2000 steps
- **No label smoothing** at scale (helps in proxy but adds complexity)
- **No SWA** at scale (checkpoint best model by validation loss)

### Muon Architecture Experiments (2 additional)

Separate from the 32 proxy runs, 25 Muon configurations were tested at 58M scale:
- Best Muon: LR=0.05, BPB=9.39 (500 steps)
- Best AdamW (matched scale): BPB=11.23 at LR=5e-4
- Muon wins at small scale but at the cost of stability—rejected for 3B run

## Experiment Timeline

All 32 proxy experiments ran on 2026-04-02, completing in ~5 hours total (each experiment takes ~10 minutes at 58M parameters). The Muon architecture experiments ran on 2026-04-01.

## Files

- `results/results.tsv` — Full results table for all 32 proxy experiments
- `results/muon_arch_results.log` — Muon architecture experiment logs
- `results/tokenizer_fertility_results.json` — Tokenizer fertility analysis from proxy phase
- `results/tokenizer_training_results.json` — Tokenizer training experiment results
- `configs/exp_XXX_train.py` — Representative training scripts (snapshots)
