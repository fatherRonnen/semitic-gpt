# Tokenizer

Custom SentencePiece BPE tokenizer for Hebrew, Arabic, Farsi, and English.

## Configuration

- **Vocab size:** 32,768 tokens
- **Algorithm:** BPE (SentencePiece)
- **Language sampling:** Equal 25% per language during training
- **Character coverage:** 99.95% with byte fallback

## Fertility Rates

| Language | Tokens/Word | Notes |
|----------|------------|-------|
| English  | 1.2        | Most efficient (Latin script well-represented in BPE) |
| Hebrew   | 1.4        | Good efficiency for a non-Latin script |
| Arabic   | 1.5        | Slightly higher due to morphological complexity |
| Farsi    | 1.6        | Highest, partly due to complex morphology |

## Directory Structure

```
tokenizer/
├── scripts/
│   └── train_tokenizer.py      # Tokenizer training script
├── config/
│   └── tokenizer_config.json   # Training configuration
└── artifacts/
    ├── fertility_report.json   # Per-language fertility analysis
    ├── vocabulary_stats.json   # Vocabulary composition stats
    └── README.md               # How to obtain the .model file
```

## Large Files (not in repo)

The trained `.model` file (~900KB) is stored on S3. See `artifacts/README.md` for download instructions.
