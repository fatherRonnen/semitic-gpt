#!/bin/bash
# Paper 1 — Run all experiments on GPU instance
# Deploy: scp to GPU, then run
# Expected runtime: ~4h total on L40S 48GB
set -e

echo "============================================================"
echo "PAPER 1 — FULL EXPERIMENT SUITE"
echo "Started: $(date -u)"
echo "============================================================"

# Install dependencies
pip install datasets transformers scikit-learn accelerate --quiet 2>/dev/null

# Ensure model files exist
if [ ! -f /tmp/sft_v3_runs/D/sft_model.pt ]; then
    echo "ERROR: Base model not found at /tmp/sft_v3_runs/D/sft_model.pt"
    echo "Download with: aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/3b-v1-fsdp/sft_model.pt /tmp/sft_v3_runs/D/sft_model.pt"
    exit 1
fi

if [ ! -f /tmp/eval/multilingual_32k.model ]; then
    echo "ERROR: Tokenizer not found"
    exit 1
fi

mkdir -p /tmp/experiments/paper1

# ============================================================
# Experiment 1: NER Transfer
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 1: NER TRANSFER ($(date -u))"
echo "============================================================"
python3 /tmp/paper1_scripts/run_ner_transfer.py 2>&1 | tee /tmp/experiments/paper1/ner_log.txt

# ============================================================
# Experiment 2: QA Transfer
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 2: QA TRANSFER ($(date -u))"
echo "============================================================"
python3 /tmp/paper1_scripts/run_qa_transfer.py 2>&1 | tee /tmp/experiments/paper1/qa_log.txt

# ============================================================
# Experiment 3: mBERT / XLM-R Comparison
# ============================================================
echo ""
echo "============================================================"
echo "EXPERIMENT 3: mBERT / XLM-R COMPARISON ($(date -u))"
echo "============================================================"
python3 /tmp/paper1_scripts/run_mbert_comparison.py 2>&1 | tee /tmp/experiments/paper1/mbert_log.txt

# ============================================================
# Upload all results
# ============================================================
echo ""
echo "============================================================"
echo "UPLOADING RESULTS ($(date -u))"
echo "============================================================"
aws s3 sync /tmp/experiments/paper1/ s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/paper1/ --quiet
echo "All results uploaded to S3!"

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE: $(date -u)"
echo "============================================================"

# Auto-shutdown
echo "Shutting down in 60 seconds..."
sleep 60
sudo shutdown -h now
