# SemiticGPT

A 3-billion parameter multilingual foundation model trained from scratch for Hebrew, Arabic, Farsi, and English — a Semitic-centered language cluster.

## Paper

**SemiticGPT: A Low-Cost Recipe for Multilingual Foundation Models in an Under-Resourced Semitic-Centered Language Cluster**

*Ronnen Slasky, Independent Researcher*

> We present a practical blueprint for building a multilingual foundation model from scratch for an under-resourced language family. Our central finding is that multilingual pretraining with a linguistically-motivated language cluster produces meaningful cross-lingual semantic transfer between related languages: fine-tuning on Hebrew sentiment data alone improves Arabic sentiment accuracy from 5.5% to 49% with zero Arabic task data. Equally instructive: English-mediated parallel data does not enable direct translation between non-English pairs, and script similarity alone does not predict semantic transfer.

## Repository Structure

```
semitic-gpt/
├── tokenizer/          # Step 1: Tokenizer design and training
│   ├── scripts/        # Training scripts
│   ├── config/         # Tokenizer config (vocab size, sampling)
│   └── artifacts/      # Fertility reports, vocabulary stats
├── data/               # Step 2: Data collection and curation
│   ├── scripts/        # Collection, cleaning, dedup scripts
│   └── manifests/      # Source lists, mixture specs, metadata
├── proxy/              # Step 3: Proxy-scale experiments
│   ├── configs/        # Experiment configs
│   ├── results/        # Sweep results (TSV/JSON)
│   └── rationale/      # Recipe selection rationale
├── pretrain/           # Step 4: Full-scale pretraining
│   ├── config/         # Training hyperparameters
│   ├── scripts/        # Launch scripts, FSDP config
│   └── logs/           # Training metric summaries
├── sft/                # Step 5: Supervised fine-tuning
│   ├── config/         # SFT hyperparameters per config
│   ├── scripts/        # Training scripts
│   └── data_recipes/   # Dataset composition specs
├── eval/               # Step 6: Evaluation
│   ├── scripts/        # All benchmark scripts
│   ├── prompts/        # Exact prompts used per task
│   ├── results/        # Score files (JSON)
│   └── predictions/    # Raw model outputs (sample)
├── paper/              # Paper artifacts
│   ├── tex/            # LaTeX source
│   ├── figures/        # Generated figures
│   └── tables/         # Generated tables
└── paper1/             # Cross-lingual transfer paper (companion)
    ├── scripts/        # NER, QA, mBERT comparison scripts
    └── results/        # Transfer experiment results
```

## Key Results

| Evaluation | Finding |
|---|---|
| **Sentiment Transfer** | Hebrew-only training → 49% Arabic accuracy (9× over baseline) |
| **Cross-lingual Retrieval** | 90% EN↔HE accuracy (9× chance) |
| **Translation** | 18.7% chrF with direct parallel data; 0% without |
| **BPB** | Hebrew 0.876, Arabic 0.726, Farsi 0.657, English 0.964 |
| **Belebele** | Near chance at 3B scale (expected) |

## Model Weights

Model weights and tokenizer are available on Hugging Face: *(link TBD)*

Checkpoints are stored on S3: `s3://autoresearch-dashboard-196766918360/multilingual-7b/`

## Reproducing Results

Each directory contains the scripts and configs needed to reproduce that step. See the paper's practitioner guide (Section 7) for the full recipe.

```bash
# Example: Run sentiment transfer evaluation
cd eval/
python scripts/run_domain_experiments.py --config configs/sentiment_transfer.json
```

## Cost

Total pretraining: ~$1,456 on AWS spot instances (L40S + H100).
Total evaluation: ~$50 on L40S spot instances.

## Citation

```bibtex
@article{slasky2026semiticgpt,
  title={SemiticGPT: A Low-Cost Recipe for Multilingual Foundation Models in an Under-Resourced Semitic-Centered Language Cluster},
  author={Slasky, Ronnen},
  year={2026}
}
```

## License

Code: MIT. Model weights: see Hugging Face model card.
