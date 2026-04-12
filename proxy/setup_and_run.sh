#!/bin/bash
# Setup script for multilingual autoresearch on g6e.xlarge (single L40S)
set -euo pipefail

echo "=== Multilingual AutoResearch Setup ==="

# Install dependencies
pip install -q boto3 numpy sentencepiece 2>/dev/null || true

# Download data from S3
DATA_DIR="/home/ubuntu/autoresearch/data"
mkdir -p "$DATA_DIR"

echo "Downloading training data from S3..."
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/train.bin "$DATA_DIR/train.bin" &
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/val.bin "$DATA_DIR/val.bin" &
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/val_en.bin "$DATA_DIR/val_en.bin" &
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/val_ar.bin "$DATA_DIR/val_ar.bin" &
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/val_he.bin "$DATA_DIR/val_he.bin" &
aws s3 cp s3://autoresearch-dashboard-196766918360/multilingual-7b/training-data/val_fa.bin "$DATA_DIR/val_fa.bin" &
wait

echo "Data downloaded:"
ls -lh "$DATA_DIR/"

# Download autoresearch scripts
WORK_DIR="/home/ubuntu/autoresearch"
aws s3 sync s3://autoresearch-dashboard-196766918360/multilingual-7b/autoresearch/ "$WORK_DIR/" --exclude "data/*" --exclude "snapshots/*" 2>/dev/null || true

echo "=== Setup complete. Starting autoresearch agent... ==="
cd "$WORK_DIR"
python3 -u run_agent.py 2>&1 | tee /tmp/autoresearch.log

# Upload results when done
echo "Uploading results to S3..."
aws s3 cp results.tsv s3://autoresearch-dashboard-196766918360/multilingual-7b/autoresearch/results.tsv
aws s3 sync snapshots/ s3://autoresearch-dashboard-196766918360/multilingual-7b/autoresearch/snapshots/
echo "=== All done! ==="
