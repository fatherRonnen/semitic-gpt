# Data Collection & Preparation

Scripts and manifests for the multilingual pretraining corpus.

## Data Mixture

| Language | Share | Tokens | Role |
|----------|-------|--------|------|
| Hebrew   | 40%   | ~8B    | Anchor language (overrepresented for transfer) |
| Arabic   | 25%   | ~5B    | Primary transfer target |
| English  | 20%   | ~4B    | Quality anchor + benchmarking |
| Farsi    | 15%   | ~3B    | Control (Indo-European, not Semitic) |

**Total: ~20B tokens**

## Directory Structure

```
data/
├── scripts/
│   ├── data_pipeline.py              # Main data processing pipeline
│   ├── collect_native_data.py        # Native text collection (v1)
│   ├── collect_native_data_v2.py     # Native text collection (v2)
│   ├── collect_translation_data.py   # Parallel data collection (v1)
│   ├── collect_translation_data_v2.py# Parallel data collection (v2)
│   ├── arabic_collect.py             # Arabic-specific collection
│   ├── arabic_collect_c4.py          # Arabic C4 collection
│   ├── arabic_collect_s3.py          # Arabic collection (standalone version)
│   ├── farsi_collect.py              # Farsi-specific collection
│   └── retokenize_arabic.py          # Arabic retokenization
└── manifests/
    ├── source_manifest.json          # All data sources with URLs
    ├── mixture_spec.json             # Training mixture specification
    └── training_data_metadata.json   # Actual training data stats
```

## Large Files (not in repo)

Tokenized training data (`.jsonl` files, ~40GB total) is stored on S3:
- `s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data-v2/`
