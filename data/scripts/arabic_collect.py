#!/usr/bin/env python3
"""
Arabic Data Collection Pipeline for Multilingual 1B Model
Collects, filters, and saves Arabic text data from multiple sources.
Target: ~1B tokens (~500M additional tokens over current 497M)

Sources:
1. mC4 Arabic (streaming) - ~500M tokens
2. Arabic Wikipedia - ~200M tokens  
3. CC100 Arabic - ~300M tokens
4. OPUS parallel (EN↔AR) - bonus parallel data

Output: JSONL files to S3 under multilingual-7b/data/arabic/
"""
import os
import sys
import json
import hashlib
import time
import logging
import subprocess
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm
import boto3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path("/tmp/arabic_data")
S3_BUCKET = "autoresearch-dashboard-196766918360"
S3_PREFIX = "multilingual-7b/data/arabic"

# Dedup tracking
seen_hashes = set()

# Stats
stats = defaultdict(lambda: {"docs": 0, "bytes": 0, "filtered": 0})


def setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["mc4", "cc100", "wikipedia", "parallel", "misc"]:
        (DATA_DIR / sub).mkdir(exist_ok=True)


def text_hash(text):
    return hashlib.md5(text.strip()[:500].encode()).hexdigest()


def arabic_quality_filter(text):
    """Quality filtering for Arabic text."""
    if not text or len(text.strip()) < 50:
        return False
    text = text.strip()
    
    # Skip if mostly URLs/HTML
    url_chars = text.count('http') + text.count('www.') + text.count('<')
    if url_chars > len(text) * 0.3:
        return False
    
    # Check for Arabic characters (Arabic Unicode block)
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or 
                       '\u0750' <= c <= '\u077F' or  # Arabic Supplement
                       '\uFB50' <= c <= '\uFDFF' or  # Arabic Presentation Forms-A
                       '\uFE70' <= c <= '\uFEFF')     # Arabic Presentation Forms-B
    if arabic_chars < len(text) * 0.3:
        return False
    
    # Skip very short after stripping
    if len(text.split()) < 10:
        return False
    
    # Dedup check
    h = text_hash(text)
    if h in seen_hashes:
        return False
    seen_hashes.add(h)
    
    return True


def save_jsonl(records, path):
    """Write records to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def upload_to_s3(local_path, s3_key):
    """Upload a file to S3."""
    s3 = boto3.client('s3')
    s3.upload_file(str(local_path), S3_BUCKET, s3_key)
    log.info(f"  Uploaded {s3_key} ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")


def flush_batch(batch, source, file_idx):
    """Save batch to local file and upload to S3."""
    if not batch:
        return
    local_path = DATA_DIR / source / f"{source}_ar_{file_idx:04d}.jsonl"
    save_jsonl(batch, local_path)
    s3_key = f"{S3_PREFIX}/{source}/{source}_ar_{file_idx:04d}.jsonl"
    upload_to_s3(local_path, s3_key)
    # Clean local file to save disk
    local_path.unlink()


def collect_mc4(target_bytes=2_000_000_000):
    """Collect mC4 Arabic - largest source."""
    log.info("=== Collecting mC4 Arabic ===")
    source = "mc4"
    
    try:
        ds = load_dataset("mc4", "ar", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        filtered = 0
        file_idx = 0
        
        for example in tqdm(ds, desc="mC4 Arabic"):
            text = example.get("text", "")
            
            if arabic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "mc4",
                    "lang": "ar",
                    "url": example.get("url", ""),
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
            else:
                filtered += 1
            
            if len(batch) >= 10000:
                flush_batch(batch, source, file_idx)
                file_idx += 1
                batch = []
                log.info(f"  mC4: {total_docs:,} docs, {total_bytes/1e9:.2f} GB, filtered {filtered:,}")
            
            if total_bytes >= target_bytes:
                log.info(f"  mC4 target reached: {total_bytes/1e9:.2f} GB")
                break
        
        # Flush remaining
        flush_batch(batch, source, file_idx)
        
        stats[source] = {"docs": total_docs, "bytes": total_bytes, "filtered": filtered}
        log.info(f"  mC4 DONE: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
        
    except Exception as e:
        log.error(f"mC4 collection failed: {e}")
        import traceback
        traceback.print_exc()


def collect_wikipedia(target_bytes=800_000_000):
    """Collect Arabic Wikipedia."""
    log.info("=== Collecting Arabic Wikipedia ===")
    source = "wikipedia"
    
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        filtered = 0
        file_idx = 0
        
        for example in tqdm(ds, desc="Wiki Arabic"):
            text = example.get("text", "")
            title = example.get("title", "")
            
            # Prepend title
            if title:
                text = f"{title}\n\n{text}"
            
            if arabic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "wikipedia",
                    "lang": "ar",
                    "title": title,
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
            else:
                filtered += 1
            
            if len(batch) >= 10000:
                flush_batch(batch, source, file_idx)
                file_idx += 1
                batch = []
                log.info(f"  Wiki: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
            
            if total_bytes >= target_bytes:
                break
        
        flush_batch(batch, source, file_idx)
        stats[source] = {"docs": total_docs, "bytes": total_bytes, "filtered": filtered}
        log.info(f"  Wikipedia DONE: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
        
    except Exception as e:
        log.error(f"Wikipedia collection failed: {e}")
        import traceback
        traceback.print_exc()


def collect_cc100(target_bytes=1_200_000_000):
    """Collect CC100 Arabic."""
    log.info("=== Collecting CC100 Arabic ===")
    source = "cc100"
    
    try:
        ds = load_dataset("cc100", lang="ar", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        filtered = 0
        file_idx = 0
        
        for example in tqdm(ds, desc="CC100 Arabic"):
            text = example.get("text", "")
            
            if arabic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "cc100",
                    "lang": "ar",
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
            else:
                filtered += 1
            
            if len(batch) >= 10000:
                flush_batch(batch, source, file_idx)
                file_idx += 1
                batch = []
                log.info(f"  CC100: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
            
            if total_bytes >= target_bytes:
                break
        
        flush_batch(batch, source, file_idx)
        stats[source] = {"docs": total_docs, "bytes": total_bytes, "filtered": filtered}
        log.info(f"  CC100 DONE: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
        
    except Exception as e:
        log.error(f"CC100 collection failed: {e}")
        import traceback
        traceback.print_exc()


def collect_opus_parallel(target_bytes=400_000_000):
    """Collect OPUS-100 EN↔AR parallel data."""
    log.info("=== Collecting OPUS-100 EN↔AR ===")
    source = "parallel"
    
    try:
        ds = load_dataset("opus100", "ar-en", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        file_idx = 0
        
        for example in tqdm(ds, desc="OPUS EN↔AR"):
            translation = example.get("translation", {})
            ar_text = translation.get("ar", "")
            en_text = translation.get("en", "")
            
            if ar_text and en_text and len(ar_text) > 20:
                rec = {
                    "text": ar_text,
                    "source": "opus100",
                    "lang": "ar",
                    "parallel_en": en_text,
                }
                batch.append(rec)
                total_bytes += len(ar_text.encode('utf-8'))
                total_docs += 1
            
            if len(batch) >= 10000:
                flush_batch(batch, source, file_idx)
                file_idx += 1
                batch = []
            
            if total_bytes >= target_bytes:
                break
        
        flush_batch(batch, source, file_idx)
        stats[source] = {"docs": total_docs, "bytes": total_bytes, "filtered": 0}
        log.info(f"  OPUS DONE: {total_docs:,} docs, {total_bytes/1e9:.2f} GB")
        
    except Exception as e:
        log.error(f"OPUS collection failed: {e}")
        import traceback
        traceback.print_exc()


def save_manifest():
    """Save collection manifest to S3."""
    manifest = {
        "language": "ar",
        "collection_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {},
        "total_docs": 0,
        "total_bytes": 0,
        "estimated_tokens": 0,
    }
    
    for source, s in stats.items():
        manifest["sources"][source] = s
        manifest["total_docs"] += s["docs"]
        manifest["total_bytes"] += s["bytes"]
    
    # Rough estimate: ~3.5 bytes per token for Arabic with our tokenizer
    manifest["estimated_tokens"] = int(manifest["total_bytes"] / 3.5)
    
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    upload_to_s3(manifest_path, f"{S3_PREFIX}/manifest.json")
    
    log.info(f"\n{'='*60}")
    log.info(f"COLLECTION COMPLETE")
    log.info(f"  Total docs: {manifest['total_docs']:,}")
    log.info(f"  Total bytes: {manifest['total_bytes']/1e9:.2f} GB")
    log.info(f"  Est tokens: {manifest['estimated_tokens']:,}")
    log.info(f"{'='*60}")
    
    return manifest


def main():
    setup()
    
    log.info("Starting Arabic data collection pipeline")
    log.info(f"Target: ~4 GB raw text (~1B tokens)")
    log.info(f"Output: s3://{S3_BUCKET}/{S3_PREFIX}/")
    
    # Collect from each source
    # mC4 is biggest, get ~2GB from it
    collect_mc4(target_bytes=2_000_000_000)
    
    # Wikipedia - high quality, grab all of it (usually ~800MB)
    collect_wikipedia(target_bytes=800_000_000)
    
    # CC100 - fill remaining gap
    collect_cc100(target_bytes=1_200_000_000)
    
    # OPUS parallel - bonus parallel data
    collect_opus_parallel(target_bytes=400_000_000)
    
    # Save manifest
    manifest = save_manifest()
    
    return manifest


if __name__ == "__main__":
    main()
