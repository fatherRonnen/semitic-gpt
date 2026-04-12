#!/usr/bin/env python3
"""
Supplemental Arabic collection from allenai/c4 (replaces deprecated mc4).
Run AFTER arabic_collect.py finishes.
"""
import os
import sys
import json
import hashlib
import time
import logging
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path("/tmp/arabic_data/c4")
DATA_DIR.mkdir(parents=True, exist_ok=True)
S3_BUCKET = "autoresearch-dashboard-196766918360"
S3_PREFIX = "multilingual-7b/data/arabic/c4"

seen_hashes = set()

def text_hash(text):
    return hashlib.md5(text.strip()[:500].encode()).hexdigest()

def arabic_quality_filter(text):
    if not text or len(text.strip()) < 50:
        return False
    text = text.strip()
    url_chars = text.count('http') + text.count('www.') + text.count('<')
    if url_chars > len(text) * 0.3:
        return False
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or 
                       '\u0750' <= c <= '\u077F' or
                       '\uFB50' <= c <= '\uFDFF' or
                       '\uFE70' <= c <= '\uFEFF')
    if arabic_chars < len(text) * 0.3:
        return False
    if len(text.split()) < 10:
        return False
    h = text_hash(text)
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    return True

def save_and_upload(batch, file_idx):
    if not batch:
        return
    local_path = DATA_DIR / f"c4_ar_{file_idx:04d}.jsonl"
    with open(local_path, 'w', encoding='utf-8') as f:
        for rec in batch:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    s3 = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/c4_ar_{file_idx:04d}.jsonl"
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log.info(f"  Uploaded {s3_key} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
    local_path.unlink()

def main():
    target_bytes = 2_000_000_000  # 2GB
    log.info(f"=== Collecting allenai/c4 Arabic (target: {target_bytes/1e9:.0f} GB) ===")
    
    # Try multilingual c4
    try:
        ds = load_dataset("allenai/c4", "ar", split="train", streaming=True)
    except Exception as e:
        log.warning(f"allenai/c4 ar failed: {e}, trying multilingual variant")
        ds = load_dataset("allenai/c4", "multilingual-ar", split="train", streaming=True)
    
    batch = []
    total_docs = 0
    total_bytes = 0
    filtered = 0
    file_idx = 0
    
    for example in tqdm(ds, desc="C4 Arabic"):
        text = example.get("text", "")
        if arabic_quality_filter(text):
            batch.append({"text": text, "source": "c4", "lang": "ar"})
            total_bytes += len(text.encode('utf-8'))
            total_docs += 1
        else:
            filtered += 1
        
        if len(batch) >= 10000:
            save_and_upload(batch, file_idx)
            file_idx += 1
            batch = []
            log.info(f"  C4: {total_docs:,} docs, {total_bytes/1e9:.2f} GB, filtered {filtered:,}")
        
        if total_bytes >= target_bytes:
            break
    
    save_and_upload(batch, file_idx)
    
    # Update manifest
    manifest_path = DATA_DIR.parent / "manifest_c4.json"
    manifest = {
        "source": "allenai/c4",
        "language": "ar",
        "total_docs": total_docs,
        "total_bytes": total_bytes,
        "estimated_tokens": int(total_bytes / 3.5),
        "collection_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    s3 = boto3.client('s3')
    s3.upload_file(str(manifest_path), S3_BUCKET, "multilingual-7b/data/arabic/manifest_c4.json")
    
    log.info(f"\nC4 DONE: {total_docs:,} docs, {total_bytes/1e9:.2f} GB, ~{int(total_bytes/3.5):,} tokens")

if __name__ == "__main__":
    main()
