#!/usr/bin/env python3
"""
Arabic Retokenizer v2 — Memory-efficient version.
Writes tokens to disk in chunks instead of holding everything in RAM.
"""
import os, sys, json, time, hashlib, struct
import numpy as np
from pathlib import Path
import sentencepiece as spm
import boto3

S3_BUCKET = "autoresearch-dashboard-196766918360"
S3_PREFIX = "multilingual-7b"
TOKENIZER_PATH = "/tmp/tokenizer.model"
WORK_DIR = Path("/tmp/retokenize")
WORK_DIR.mkdir(parents=True, exist_ok=True)

AR_TOKEN_TARGET = 503_000_000
AR_VAL_TOKENS = 5_000_000
CHUNK_SIZE = 50_000_000  # 50M tokens per chunk file (~100MB)
BLOCK_SHUFFLE_SIZE = 1_000_000
BLOCK_SHUFFLE_SEED = 43

s3 = boto3.client('s3')

# Download tokenizer
print("=== Downloading tokenizer ===", flush=True)
os.system(f"aws s3 cp s3://{S3_BUCKET}/{S3_PREFIX}/tokenizer/multilingual_32k.model {TOKENIZER_PATH} 2>&1")
sp = spm.SentencePieceProcessor()
sp.Load(TOKENIZER_PATH)
BOS_ID, EOS_ID = sp.bos_id(), sp.eos_id()
print(f"Tokenizer loaded: vocab={sp.GetPieceSize()}, BOS={BOS_ID}, EOS={EOS_ID}", flush=True)

def iter_s3_jsonl(prefix):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not key.endswith('.jsonl'):
                continue
            tmp = WORK_DIR / "temp_dl.jsonl"
            s3.download_file(S3_BUCKET, key, str(tmp))
            with open(tmp, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        text = rec.get('text', '')
                        if text and len(text.strip()) >= 50:
                            yield text
                    except:
                        continue
            tmp.unlink()

# === Step 1: Tokenize Arabic data to disk chunks ===
print("\n=== Step 1: Tokenize Arabic data to disk chunks ===", flush=True)

sources = [
    f"{S3_PREFIX}/data/arabic/c4/",
    f"{S3_PREFIX}/data/arabic/wikipedia/",
    f"{S3_PREFIX}/data/arabic/parallel/",
]

chunk_dir = WORK_DIR / "ar_chunks"
chunk_dir.mkdir(exist_ok=True)

buffer = []
total_tokens = 0
total_docs = 0
chunk_idx = 0
target_total = AR_TOKEN_TARGET + AR_VAL_TOKENS
done = False

for source in sources:
    if done:
        break
    print(f"  Processing {source}...", flush=True)
    for text in iter_s3_jsonl(source):
        ids = sp.Encode(text)
        if len(ids) < 10:
            continue
        doc_tokens = [BOS_ID] + ids + [EOS_ID]
        buffer.extend(doc_tokens)
        total_tokens += len(doc_tokens)
        total_docs += 1

        # Flush buffer to disk when it gets big
        if len(buffer) >= CHUNK_SIZE:
            arr = np.array(buffer[:CHUNK_SIZE], dtype=np.uint16)
            arr.tofile(str(chunk_dir / f"chunk_{chunk_idx:04d}.bin"))
            buffer = buffer[CHUNK_SIZE:]
            chunk_idx += 1
            print(f"    {total_docs:,} docs, {total_tokens:,} tokens, {chunk_idx} chunks saved", flush=True)

        if total_tokens >= target_total:
            print(f"  Target reached: {total_tokens:,} tokens", flush=True)
            done = True
            break

# Flush remaining buffer
if buffer:
    arr = np.array(buffer, dtype=np.uint16)
    arr.tofile(str(chunk_dir / f"chunk_{chunk_idx:04d}.bin"))
    chunk_idx += 1
    del buffer

print(f"  Total: {total_docs:,} docs, {total_tokens:,} tokens in {chunk_idx} chunks", flush=True)

# === Step 2: Split val and train from chunks ===
print("\n=== Step 2: Split val/train from chunks ===", flush=True)

# Read first chunk for val tokens
first_chunk = np.fromfile(str(chunk_dir / "chunk_0000.bin"), dtype=np.uint16)
val_arr = first_chunk[:AR_VAL_TOKENS]
val_path = WORK_DIR / "val_ar.bin"
val_arr.tofile(str(val_path))
print(f"  Val: {len(val_arr):,} tokens", flush=True)

# Write trimmed first chunk back
remaining = first_chunk[AR_VAL_TOKENS:]
remaining.tofile(str(chunk_dir / "chunk_0000.bin"))
del first_chunk, val_arr, remaining

# === Step 3: Download old train.bin ===
print("\n=== Step 3: Download old train.bin ===", flush=True)
old_train_path = WORK_DIR / "old_train.bin"
os.system(f"aws s3 cp s3://{S3_BUCKET}/{S3_PREFIX}/training-data/train.bin {old_train_path} 2>&1")
old_size = os.path.getsize(old_train_path)
old_tokens = old_size // 2
print(f"  Old train.bin: {old_tokens:,} tokens ({old_size / 1e9:.2f} GB)", flush=True)

# === Step 4: Concatenate old + new Arabic chunks ===
print("\n=== Step 4: Concatenate ===", flush=True)
combined_path = WORK_DIR / "train_combined.bin"

with open(combined_path, 'wb') as fout:
    # Copy old train.bin
    print("  Copying old train.bin...", flush=True)
    with open(old_train_path, 'rb') as fin:
        while True:
            data = fin.read(100 * 1024 * 1024)
            if not data:
                break
            fout.write(data)
    
    # Append Arabic chunks
    print("  Appending Arabic chunks...", flush=True)
    ar_train_tokens = 0
    chunk_files = sorted(chunk_dir.glob("chunk_*.bin"))
    for cf in chunk_files:
        chunk_data = np.fromfile(str(cf), dtype=np.uint16)
        # Limit to target
        if ar_train_tokens + len(chunk_data) > AR_TOKEN_TARGET:
            chunk_data = chunk_data[:AR_TOKEN_TARGET - ar_train_tokens]
        chunk_data.tofile(fout)
        ar_train_tokens += len(chunk_data)
        cf.unlink()  # Free disk
        if ar_train_tokens >= AR_TOKEN_TARGET:
            break

# Clean up
old_train_path.unlink()
import shutil
shutil.rmtree(chunk_dir, ignore_errors=True)

combined_size = os.path.getsize(combined_path)
total_train_tokens = combined_size // 2
print(f"  Combined: {total_train_tokens:,} tokens ({combined_size / 1e9:.2f} GB)", flush=True)
print(f"  Added Arabic: {ar_train_tokens:,} tokens", flush=True)

# === Step 5: Block shuffle ===
print(f"\n=== Step 5: Block shuffle (seed={BLOCK_SHUFFLE_SEED}) ===", flush=True)

data = np.memmap(str(combined_path), dtype=np.uint16, mode='r')
n_blocks = len(data) // BLOCK_SHUFFLE_SIZE
rng = np.random.RandomState(BLOCK_SHUFFLE_SEED)
block_order = rng.permutation(n_blocks)

final_path = WORK_DIR / "train.bin"
final_data = np.memmap(str(final_path), dtype=np.uint16, mode='w+',
                        shape=(n_blocks * BLOCK_SHUFFLE_SIZE,))

print(f"  Shuffling {n_blocks:,} blocks...", flush=True)
for i, old_idx in enumerate(block_order):
    s_old = old_idx * BLOCK_SHUFFLE_SIZE
    s_new = i * BLOCK_SHUFFLE_SIZE
    final_data[s_new:s_new + BLOCK_SHUFFLE_SIZE] = data[s_old:s_old + BLOCK_SHUFFLE_SIZE]
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{n_blocks} blocks shuffled", flush=True)

remainder_start = n_blocks * BLOCK_SHUFFLE_SIZE
if remainder_start < len(data):
    remainder = np.array(data[remainder_start:])
    final_data.flush()
    del final_data
    with open(final_path, 'ab') as f:
        remainder.tofile(f)
    total_final = n_blocks * BLOCK_SHUFFLE_SIZE + len(remainder)
else:
    final_data.flush()
    del final_data
    total_final = n_blocks * BLOCK_SHUFFLE_SIZE

del data
combined_path.unlink()
print(f"  Final: {total_final:,} tokens", flush=True)

# === Step 6: Upload to S3 ===
print("\n=== Step 6: Upload to S3 ===", flush=True)
s3_out = f"s3://{S3_BUCKET}/{S3_PREFIX}/training-data-v2"

os.system(f"aws s3 cp {final_path} {s3_out}/train.bin 2>&1")
os.system(f"aws s3 cp {val_path} {s3_out}/val_ar.bin 2>&1")

for vf in ["val.bin", "val_en.bin", "val_he.bin", "val_fa.bin"]:
    os.system(f"aws s3 cp s3://{S3_BUCKET}/{S3_PREFIX}/training-data/{vf} {s3_out}/{vf} 2>&1")

metadata = {
    "total_train_tokens": total_final,
    "total_val_tokens": 25_000_000,
    "vocab_size": sp.GetPieceSize(),
    "dtype": "uint16",
    "block_shuffle_seed": BLOCK_SHUFFLE_SEED,
    "block_shuffle_size": BLOCK_SHUFFLE_SIZE,
    "languages": {
        "en": {"train_tokens": 1_495_000_931, "val_tokens": 5_000_000,
               "pct": round(1_495_000_931 / total_final * 100, 1)},
        "ar": {"train_tokens": 497_000_000 + ar_train_tokens, "val_tokens": AR_VAL_TOKENS,
               "pct": round((497_000_000 + ar_train_tokens) / total_final * 100, 1),
               "note": f"Original 497M + {ar_train_tokens:,} new from c4/wiki/opus"},
        "he": {"train_tokens": 995_001_913, "val_tokens": 5_000_000,
               "pct": round(995_001_913 / total_final * 100, 1)},
        "fa": {"train_tokens": 995_000_027, "val_tokens": 5_000_000,
               "pct": round(995_000_027 / total_final * 100, 1)},
    },
    "tokenizer": "multilingual_32k.model",
    "bos_id": BOS_ID, "eos_id": EOS_ID,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
meta_path = WORK_DIR / "metadata.json"
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)
os.system(f"aws s3 cp {meta_path} {s3_out}/metadata.json 2>&1")

print(f"\n{'='*60}", flush=True)
print(f"RETOKENIZATION COMPLETE", flush=True)
print(f"  Old: {old_tokens:,} tokens", flush=True)
print(f"  New Arabic: {ar_train_tokens:,} tokens", flush=True)
print(f"  Total: {total_final:,} tokens", flush=True)
print(f"  Output: {s3_out}/", flush=True)
for lang, info in metadata["languages"].items():
    print(f"  {lang}: {info['train_tokens']:,} ({info['pct']}%)", flush=True)
print(f"{'='*60}", flush=True)
