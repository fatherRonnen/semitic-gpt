# Supervised Fine-Tuning (SFT)

Instruction tuning experiments testing data composition and scaling effects.

## Experiments

| Config | Steps | Languages | Purpose |
|--------|-------|-----------|---------|
| D-baseline | 5K | HE, AR, EN, FA | Full baseline SFT |
| E-1K | 1K | HE, AR, EN, FA | SFT scaling: minimal |
| E-3K | 3K | HE, AR, EN, FA | SFT scaling: medium |
| E-5K | 5K | HE, AR, EN, FA | SFT scaling: full |
| F-no-arfa | 5K | HE, EN | Ablation: remove AR/FA |
| F-only-arfa | 5K | AR, FA | Ablation: only AR/FA |

## Key Findings

1. **SFT improves BPB** monotonically across all languages
2. **Diminishing returns** after 3K steps
3. **Removing languages** from SFT degrades their BPB
4. D-baseline achieves best overall balance

## Directory Structure

```
sft/
├── scripts/
│   ├── train_sft_3b.py          # SFT training script
│   ├── prepare_sft_data.py      # Data preparation (v1)
│   └── prepare_sft_data_v2.py   # Data preparation (v2)
├── data_recipes/
│   └── sft_v2_metadata.json     # SFT data statistics
├── config/
│   ├── D-baseline.json          # Config for each variant
│   ├── E-1K.json
│   ├── E-3K.json
│   ├── E-5K.json
│   ├── F-no-arfa.json
│   └── F-only-arfa.json
└── logs/
    ├── D_training.log
    ├── E-1K_training.log
    ├── E-3K_training.log
    ├── E-5K_training.log
    ├── F-no-arfa_training.log
    └── F-only-arfa_training.log
```

## Checkpoints (not in repo)

SFT checkpoints are stored on S3 alongside the pretrained checkpoints.
