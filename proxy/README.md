# Proxy Experiments

Automated hyperparameter search using 32 small-scale proxy experiments (58M parameters) to find the optimal training recipe before committing to the full 3B run.

## Results Summary

- **32 proxy experiments** + **2 Muon architecture experiments**
- Best proxy result: **exp_025** with val_bpb = 41.66
- Key finding: **AdamW > Muon** for stability at scale
- Full rationale: see `rationale/recipe_selection.md`

## Directory Structure

```
proxy/
├── run_agent.py           # Automated experiment agent
├── prepare.py             # Data preparation for proxy runs
├── setup_and_run.sh       # Environment setup script
├── train.py               # Base training script
├── program.md             # Agent program specification
├── results/
│   ├── results.tsv                      # All 32 experiment results
│   ├── muon_arch_results.log            # Muon architecture experiments
│   ├── tokenizer_fertility_results.json # Tokenizer experiments
│   └── tokenizer_training_results.json  # Tokenizer training results
├── rationale/
│   └── recipe_selection.md              # Decision rationale
└── configs/
    ├── exp_000_train.py   # Snapshot: initial experiment
    ├── exp_010_train.py   # Snapshot: mid-point
    ├── exp_020_train.py   # Snapshot: later iteration
    └── exp_032_train.py   # Snapshot: final experiment
```
