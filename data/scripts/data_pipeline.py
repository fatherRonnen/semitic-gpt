#!/usr/bin/env python3
"""
Multilingual Data Pipeline - Tokenize data for 1B model training
Target: ~10B tokens across 4 languages (en, ar, he, fa)
"""

import os
import sys
import json
import time
import struct
import hashlib
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Install dependencies first
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "sentencepiece", "datasets", "boto3", "tqdm", "zstandard"])

import sentencepiece as spm
from tqdm import tqdm
import boto3

# === Configuration ===
TOKENIZER_PATH = "/tmp/tokenizer.model"
OUTPUT_DIR = Path("/tmp/tokenized")
OUTPUT_DIR.mkdir(exist_ok=True)

S3_BUCKET = "autoresearch-dashboard-196766918360"

# Token targets per language
TARGETS = {
    "en": 5_000_000_000,   # 5B
    "ar": 2_000_000_000,   # 2B
    "he": 1_500_000_000,   # 1.5B
    "fa": 1_500_000_000,   # 1.5B
}

CHUNK_SIZE = 100_000_000  # 100M tokens per chunk file

# === Step 0: Download tokenizer ===
print("=== Downloading tokenizer ===")
subprocess.check_call(["aws", "s3", "cp",
    f"s3://{S3_BUCKET}/multilingual-7b/tokenizer/multilingual_32k.model",
    TOKENIZER_PATH])

sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)
VOCAB_SIZE = sp.GetPieceSize()
BOS_ID = sp.bos_id()
EOS_ID = sp.eos_id()
print(f"Tokenizer loaded: vocab={VOCAB_SIZE}, BOS={BOS_ID}, EOS={EOS_ID}")
assert VOCAB_SIZE <= 65535, "Vocab too large for uint16"


def tokenize_and_save(lang, text_iter, target_tokens, desc=""):
    """Tokenize texts from an iterator and save as uint16 chunks."""
    lang_dir = OUTPUT_DIR / lang
    lang_dir.mkdir(exist_ok=True)

    total_tokens = 0
    chunk_idx = 0
    buffer = []
    buffer_tokens = 0
    doc_count = 0

    print(f"\n=== Tokenizing {lang} ({desc}) - target: {target_tokens:,} tokens ===")

    for text in text_iter:
        if not text or len(text.strip()) < 50:
            continue

        ids = sp.Encode(text)
        if len(ids) < 10:
            continue

        # Add BOS + tokens + EOS
        doc_tokens = [BOS_ID] + ids + [EOS_ID]
        buffer.extend(doc_tokens)
        buffer_tokens += len(doc_tokens)
        doc_count += 1
        total_tokens += len(doc_tokens)

        # Save chunk when buffer is large enough
        if buffer_tokens >= CHUNK_SIZE:
            arr = np.array(buffer[:CHUNK_SIZE], dtype=np.uint16)
            chunk_path = lang_dir / f"chunk_{chunk_idx:04d}.bin"
            arr.tofile(str(chunk_path))
            print(f"  [{lang}] Saved chunk {chunk_idx}: {CHUNK_SIZE:,} tokens "
                  f"(total: {total_tokens:,}/{target_tokens:,}, docs: {doc_count:,})")
            buffer = buffer[CHUNK_SIZE:]
            buffer_tokens = len(buffer)
            chunk_idx += 1

        # Check if we've reached target
        if total_tokens >= target_tokens:
            break

    # Save remaining buffer
    if buffer:
        arr = np.array(buffer, dtype=np.uint16)
        chunk_path = lang_dir / f"chunk_{chunk_idx:04d}.bin"
        arr.tofile(str(chunk_path))
        print(f"  [{lang}] Saved final chunk {chunk_idx}: {len(buffer):,} tokens")
        chunk_idx += 1

    print(f"  [{lang}] DONE: {total_tokens:,} tokens in {chunk_idx} chunks, {doc_count:,} docs")
    return total_tokens, doc_count


# === Step 1: English from C4 ===
def english_iter():
    from datasets import load_dataset
    print("Loading C4 English (streaming)...")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    for example in ds:
        yield example.get("text", "")

en_tokens, en_docs = tokenize_and_save("en", english_iter(), TARGETS["en"], "C4-en")


# === Step 2: Arabic from CC100 ===
def arabic_iter():
    from datasets import load_dataset
    print("Loading CC100 Arabic (streaming)...")
    try:
        ds = load_dataset("cc100", lang="ar", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            yield example.get("text", "")
    except Exception as e:
        print(f"CC100-ar failed: {e}, falling back to Wikipedia Arabic")
        ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            yield example.get("text", "")

ar_tokens, ar_docs = tokenize_and_save("ar", arabic_iter(), TARGETS["ar"], "CC100-ar")


# === Step 3: Hebrew from S3 datasets + HuggingFace ===
def hebrew_iter():
    from datasets import load_dataset
    import json as jlib

    # First try our S3 data
    s3 = boto3.client('s3')
    hebrew_prefix = "datasets/hebrew/"
    print(f"Listing Hebrew data from s3://{S3_BUCKET}/{hebrew_prefix}...")

    paginator = s3.get_paginator('list_objects_v2')
    files_found = 0
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=hebrew_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if 'tokenized' in key:
                continue
            if not (key.endswith('.jsonl') or key.endswith('.jsonl.zst') or
                    key.endswith('.txt') or key.endswith('.json')):
                continue
            files_found += 1

            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                body = resp['Body'].read()

                if key.endswith('.zst'):
                    import zstandard
                    dctx = zstandard.ZstdDecompressor()
                    body = dctx.decompress(body, max_output_size=500*1024*1024)

                content = body.decode('utf-8', errors='ignore')

                if key.endswith('.jsonl') or key.endswith('.jsonl.zst'):
                    for line in content.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            doc = jlib.loads(line)
                            text = doc.get('text', doc.get('content', ''))
                            if text:
                                yield text
                        except:
                            continue
                elif key.endswith('.txt'):
                    # Split on double newlines for documents
                    for doc in content.split('\n\n'):
                        if doc.strip():
                            yield doc.strip()
                elif key.endswith('.json'):
                    try:
                        data = jlib.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    yield item.get('text', item.get('content', ''))
                                elif isinstance(item, str):
                                    yield item
                    except:
                        pass
            except Exception as e:
                print(f"  Error reading {key}: {e}")
                continue

    print(f"  Found {files_found} Hebrew files on S3")

    # Supplement with HuggingFace Wikipedia Hebrew
    print("  Supplementing with Wikipedia Hebrew...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.he", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            yield example.get("text", "")
    except Exception as e:
        print(f"  Wikipedia-he failed: {e}")

    # CC100 Hebrew as additional source
    print("  Supplementing with CC100 Hebrew...")
    try:
        ds = load_dataset("cc100", lang="he", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            yield example.get("text", "")
    except Exception as e:
        print(f"  CC100-he failed: {e}")

he_tokens, he_docs = tokenize_and_save("he", hebrew_iter(), TARGETS["he"], "S3+Wiki+CC100")


# === Step 4: Farsi from S3 ===
def farsi_iter():
    import json as jlib

    s3 = boto3.client('s3')
    farsi_prefix = "multilingual-7b/data/farsi/misc/"
    print(f"Listing Farsi data from s3://{S3_BUCKET}/{farsi_prefix}...")

    paginator = s3.get_paginator('list_objects_v2')
    files = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=farsi_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.jsonl') or key.endswith('.jsonl.zst'):
                files.append(key)

    print(f"  Found {len(files)} Farsi files")
    # Only process enough files for ~1.5B tokens
    # Estimate ~2-3M tokens per 10MB file, so ~500-750 files
    # Process up to 800 files to be safe
    files = sorted(files)[:800]

    for key in files:
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
            body = resp['Body'].read()

            if key.endswith('.zst'):
                import zstandard
                dctx = zstandard.ZstdDecompressor()
                body = dctx.decompress(body, max_output_size=500*1024*1024)

            content = body.decode('utf-8', errors='ignore')
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = jlib.loads(line)
                    text = doc.get('text', doc.get('content', ''))
                    if text:
                        yield text
                except:
                    # Maybe plain text
                    if len(line) > 50:
                        yield line
        except Exception as e:
            print(f"  Error reading {key}: {e}")
            continue

    # Fallback: supplement with CC100 Farsi
    print("  Supplementing with CC100 Farsi if needed...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cc100", lang="fa", split="train", streaming=True, trust_remote_code=True)
        for example in ds:
            yield example.get("text", "")
    except Exception as e:
        print(f"  CC100-fa failed: {e}")

fa_tokens, fa_docs = tokenize_and_save("fa", farsi_iter(), TARGETS["fa"], "S3-farsi+CC100")


# === Step 5: Create train/val splits ===
print("\n=== Creating train/val splits ===")

stats = {
    "en": {"tokens": en_tokens, "docs": en_docs},
    "ar": {"tokens": ar_tokens, "docs": ar_docs},
    "he": {"tokens": he_tokens, "docs": he_docs},
    "fa": {"tokens": fa_tokens, "docs": fa_docs},
}

VAL_TOKENS_PER_LANG = 5_000_000  # 5M tokens per language for val (equal)

total_train = 0
total_val = 0

for lang in ["en", "ar", "he", "fa"]:
    lang_dir = OUTPUT_DIR / lang
    chunks = sorted(lang_dir.glob("chunk_*.bin"))
    if not chunks:
        print(f"  WARNING: No chunks for {lang}")
        continue

    # Load all tokens for this language
    print(f"  Loading {lang} chunks...")
    all_tokens = []
    for chunk_path in chunks:
        arr = np.fromfile(str(chunk_path), dtype=np.uint16)
        all_tokens.append(arr)
    all_tokens = np.concatenate(all_tokens)
    print(f"  {lang}: {len(all_tokens):,} tokens loaded")

    # Split: last VAL_TOKENS_PER_LANG for val, rest for train
    val_size = min(VAL_TOKENS_PER_LANG, len(all_tokens) // 50)  # at most 2%
    train_arr = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_arr = all_tokens[-val_size:] if val_size > 0 else np.array([], dtype=np.uint16)

    # Save per-language files
    train_path = OUTPUT_DIR / f"train_{lang}.bin"
    val_path = OUTPUT_DIR / f"val_{lang}.bin"
    train_arr.tofile(str(train_path))
    val_arr.tofile(str(val_path))

    stats[lang]["train_tokens"] = len(train_arr)
    stats[lang]["val_tokens"] = len(val_arr)
    total_train += len(train_arr)
    total_val += len(val_arr)

    print(f"  {lang}: train={len(train_arr):,}, val={len(val_arr):,}")

    # Clean up chunks to save disk
    for chunk_path in chunks:
        chunk_path.unlink()

# Concatenate all train and val
print("\n  Concatenating all languages...")
train_parts = []
val_parts = []
for lang in ["en", "ar", "he", "fa"]:
    train_path = OUTPUT_DIR / f"train_{lang}.bin"
    val_path = OUTPUT_DIR / f"val_{lang}.bin"
    if train_path.exists():
        train_parts.append(np.fromfile(str(train_path), dtype=np.uint16))
    if val_path.exists():
        val_parts.append(np.fromfile(str(val_path), dtype=np.uint16))

# Shuffle at a coarse level - split into ~1M token blocks and shuffle
print("  Shuffling training data (block shuffle)...")
BLOCK_SIZE = 1_000_000
all_train = np.concatenate(train_parts)
n_blocks = len(all_train) // BLOCK_SIZE
if n_blocks > 0:
    # Truncate to exact blocks for clean shuffle
    trimmed = all_train[:n_blocks * BLOCK_SIZE].reshape(n_blocks, BLOCK_SIZE)
    remainder = all_train[n_blocks * BLOCK_SIZE:]
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_blocks)
    shuffled = trimmed[indices].reshape(-1)
    all_train = np.concatenate([shuffled, remainder])

all_val = np.concatenate(val_parts)

# Save final files
print("  Saving final train.bin and val.bin...")
FINAL_TRAIN = "/tmp/multilingual_train.bin"
FINAL_VAL = "/tmp/multilingual_val.bin"
all_train.tofile(FINAL_TRAIN)
all_val.tofile(FINAL_VAL)

print(f"  Train: {len(all_train):,} tokens ({os.path.getsize(FINAL_TRAIN) / 1e9:.2f} GB)")
print(f"  Val: {len(all_val):,} tokens ({os.path.getsize(FINAL_VAL) / 1e9:.2f} GB)")

# === Step 6: Create metadata ===
metadata = {
    "total_train_tokens": int(len(all_train)),
    "total_val_tokens": int(len(all_val)),
    "vocab_size": VOCAB_SIZE,
    "dtype": "uint16",
    "block_shuffle_seed": 42,
    "block_shuffle_size": BLOCK_SIZE,
    "languages": {},
    "tokenizer": "multilingual_32k.model",
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

total_all = int(len(all_train)) + int(len(all_val))
for lang in ["en", "ar", "he", "fa"]:
    lang_total = stats[lang].get("train_tokens", 0) + stats[lang].get("val_tokens", 0)
    metadata["languages"][lang] = {
        "total_tokens": int(lang_total),
        "train_tokens": int(stats[lang].get("train_tokens", 0)),
        "val_tokens": int(stats[lang].get("val_tokens", 0)),
        "docs": int(stats[lang].get("docs", 0)),
        "pct": round(lang_total / total_all * 100, 1) if total_all > 0 else 0,
    }

META_PATH = "/tmp/data_metadata.json"
with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n=== Metadata ===")
print(json.dumps(metadata, indent=2))

# === Step 7: Upload to S3 ===
print("\n=== Uploading to S3 ===")
s3_prefix = f"s3://{S3_BUCKET}/multilingual-7b/training-data"

uploads = [
    (FINAL_TRAIN, f"{s3_prefix}/train.bin"),
    (FINAL_VAL, f"{s3_prefix}/val.bin"),
    (META_PATH, f"{s3_prefix}/metadata.json"),
]

# Per-language val sets
for lang in ["en", "ar", "he", "fa"]:
    val_path = OUTPUT_DIR / f"val_{lang}.bin"
    if val_path.exists() and val_path.stat().st_size > 0:
        uploads.append((str(val_path), f"{s3_prefix}/val_{lang}.bin"))

for local, remote in uploads:
    size_mb = os.path.getsize(local) / 1e6
    print(f"  Uploading {local} ({size_mb:.1f} MB) -> {remote}")
    subprocess.check_call(["aws", "s3", "cp", local, remote])

print("\n=== ALL DONE ===")
print(f"Total train: {len(all_train):,} tokens")
print(f"Total val: {len(all_val):,} tokens")
print(f"Uploaded to: {s3_prefix}/")
