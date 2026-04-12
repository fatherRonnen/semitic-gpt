# Pretraining

3B parameter decoder-only transformer trained from scratch on ~20B multilingual tokens.

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 3.04B |
| Layers | 36 |
| Hidden dim | 2560 |
| Attention heads | 20 |
| Head dim | 128 |
| Vocab size | 32,768 |
| Sequence length | 2,048 |
| Position encoding | RoPE |
| Activation | SwiGLU |
| Normalization | RMSNorm (pre-norm) |

## Training Configuration

- **Optimizer:** AdamW (lr=3e-4, β₁=0.9, β₂=0.95, wd=0.1)
- **Schedule:** Cosine decay, 2000-step warmup
- **Batch size:** 512K tokens
- **Precision:** bf16 with FSDP
- **Hardware:** 8× NVIDIA A10G (g6e.48xlarge)

## Directory Structure

```
pretrain/
├── scripts/
│   ├── train_multilingual_3b.py       # Single-GPU training script
│   ├── train_multilingual_3b_fsdp.py  # FSDP multi-GPU training script
│   ├── train_multilingual_1b_v3.py    # 1B model script (earlier iteration)
│   └── launch_v3_training.sh          # Training launch script
├── config/
│   └── training_config.json           # Full hyperparameter config
└── logs/
    ├── 3b_v1_training.log             # Training output log
    ├── 3b_v1_p5_training.log          # Phase 5 training log
    ├── 3b_v1_fsdp_eval_results.json   # FSDP checkpoint evaluation
    └── 3b_v1_eval_results.json        # Early checkpoint evaluation
```

## Checkpoints (not in repo)

Model checkpoints (~12.5GB each) are stored on S3:
- `s3://autoresearch-dashboard-196766918360/multilingual-7b/checkpoints/3b-v1-fsdp/`
- Best model: `best_model.pt`
