# Evaluation

Comprehensive evaluation suite covering language modeling, reading comprehension, translation, embeddings, sentiment transfer, and news classification.

## Evaluation Tasks

| Task | Metric | Scripts |
|------|--------|---------|
| Language Modeling | BPB | `exp_b_crosslingual.py` |
| Reading Comprehension | Belebele accuracy | `eval_belebele.py` |
| Translation | chrF | `eval_translation.py` |
| Cross-lingual Embeddings | Retrieval accuracy | `eval_embeddings.py` |
| Sentiment (domain transfer) | Classification accuracy | `exp_a_fix_classification.py` |
| News Classification | 4-class accuracy | `exp_a_fix_classification.py` |
| Tokenizer Analysis | Fertility metrics | `tokenizer_analysis.py`, `exp_c_tokenizer_ablation.py` |

## Headline Results

- **Cross-lingual sentiment transfer:** Hebrew-only training improves Arabic 5.5% → 49% (9× gain, zero Arabic task data)
- **Embedding alignment:** EN↔HE retrieval at 90% (vs 10% chance)
- **Translation:** Direct parallel data achieves 18.7% chrF for AR→FA
- **News classification:** 4% → 79% with domain fine-tuning

## Directory Structure

```
eval/
├── scripts/
│   ├── eval_belebele.py               # Belebele benchmark
│   ├── eval_belebele_transfer.py      # Belebele transfer experiments
│   ├── exp_a_hebrew_downstream.py     # Hebrew downstream tasks
│   ├── exp_a_fix_classification.py    # Sentiment + news classification
│   ├── exp_b_crosslingual.py          # Cross-lingual BPB + generation
│   ├── exp_c_tokenizer_ablation.py    # Tokenizer ablation study
│   ├── tokenizer_analysis.py          # Tokenizer analysis tools
│   ├── model_arch.py                  # Model architecture utilities
│   ├── eval_embeddings.py             # Embedding quality evaluation
│   ├── eval_translation.py            # Translation evaluation
│   ├── prepare_domain_finetune.py     # Domain fine-tuning preparation
│   └── run_exp_hi.py                  # Domain fine-tuning experiments
├── results/
│   ├── exp_a_results.json             # Hebrew downstream (initial)
│   ├── exp_a_fix_results.json         # Hebrew downstream (fixed)
│   ├── exp_b_results.json             # Cross-lingual evaluation
│   ├── exp_c_results.json             # Tokenizer ablation
│   ├── exp_D_results.json             # D-baseline SFT evaluation
│   ├── exp_E-1K_results.json          # E-1K SFT evaluation
│   ├── exp_E-3K_results.json          # E-3K SFT evaluation
│   ├── exp_E-5K_results.json          # E-5K SFT evaluation
│   ├── exp_F-no-arfa_results.json     # F-no-arfa evaluation
│   ├── exp_F-only-arfa_results.json   # F-only-arfa evaluation
│   ├── exp_hi_results.json            # Domain fine-tuning results
│   ├── belebele_3b_results.json       # Belebele 3B results
│   ├── embedding_eval.log             # Embedding evaluation log
│   └── translation_eval.log           # Translation evaluation log
└── prompts/
    └── prompts.md                     # Exact prompts for all tasks
```
