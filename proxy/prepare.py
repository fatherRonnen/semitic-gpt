#!/usr/bin/env python3
"""
Prepare module for multilingual autoresearch experiments.
Provides data loading and evaluation utilities.

Data: pre-tokenized binary files (uint16, vocab=32000)
"""

import os
import math
import numpy as np
import torch
import torch.nn.functional as F

# Constants
VOCAB_SIZE = 32000
MAX_SEQ_LEN = 2048
DEVICE_BATCH_SIZE = 16

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class DataLoader:
    """Streams token chunks from a binary file."""

    def __init__(self, filename, seq_len=MAX_SEQ_LEN, batch_size=DEVICE_BATCH_SIZE):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)
        self.chunk = seq_len * batch_size
        self.pos = 0
        print(f"DataLoader: {filename}, {self.num_tokens:,} tokens")

    def next_batch(self):
        buf = self.data[self.pos:self.pos + self.seq_len * self.batch_size + 1]
        if len(buf) < self.seq_len * self.batch_size + 1:
            self.pos = 0
            buf = self.data[self.pos:self.pos + self.seq_len * self.batch_size + 1]
        buf = torch.from_numpy(buf.astype(np.int64))
        x = buf[:-1].view(self.batch_size, self.seq_len)
        y = buf[1:].view(self.batch_size, self.seq_len)
        self.pos += self.chunk
        if self.pos + self.chunk + 1 > self.num_tokens:
            self.pos = 0
        return x, y


@torch.no_grad()
def evaluate_bpb(model, val_path, seq_len=MAX_SEQ_LEN, device="cuda",
                 max_batches=20, batch_size=8):
    """Evaluate model on val set, return bits-per-byte (BPB).
    
    For multilingual tokenizer with vocab=32K, we use a fixed
    bytes_per_token estimate based on the tokenizer fertility report.
    """
    model.eval()
    data = np.memmap(val_path, dtype=np.uint16, mode='r')
    
    # bytes_per_token for this tokenizer (from fertility_report.json)
    # EN=3.81, AR=6.50, HE=5.83, FA=6.97
    # Weighted by data mix (37.5% EN, 12.4% AR, 25% HE, 25% FA): ~5.39
    BPT = 5.39

    total_loss = 0.0
    total_count = 0
    pos = 0

    for _ in range(max_batches):
        end = pos + seq_len * batch_size + 1
        if end > len(data):
            break
        buf = torch.from_numpy(data[pos:end].astype(np.int64))
        x = buf[:-1].view(batch_size, seq_len).to(device)
        y = buf[1:].view(batch_size, seq_len).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)),
                               y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_count += y.numel()
        pos += seq_len * batch_size

    avg_ce = total_loss / total_count  # nats per token
    bpb = avg_ce * BPT / math.log(2)
    model.train()
    return bpb


if __name__ == "__main__":
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")
    print(f"MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    for f in ['train.bin', 'val.bin', 'val_en.bin', 'val_ar.bin', 'val_he.bin', 'val_fa.bin']:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            data = np.memmap(path, dtype=np.uint16, mode='r')
            print(f"  {f}: {len(data):,} tokens")
