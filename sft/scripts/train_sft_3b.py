#!/usr/bin/env python3
"""
Multilingual 3B GPT — SFT Training

Fine-tunes the base model on instruction data (Aya + Bactrian-X + FLORES translations).
Uses the same architecture as pretraining with LoRA-free full fine-tuning
(model is 3B params, fits in 24GB A10G in bf16).

Usage:
    python train_sft_3b.py --checkpoint /path/to/best_model.pt \
        --tokenizer /path/to/multilingual_32k.model \
        --data-dir /path/to/sft_data/ \
        --output /path/to/sft_model.pt
"""

import os, sys, json, math, time, random, argparse
sys.stdout.reconfigure(line_buffering=True)
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ============ MODEL (must match training) ============
VOCAB_SIZE = 32000
DIM = 3072
DEPTH = 26
N_HEADS = 24
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1*cos - x2*sin, x1*sin + x2*cos), dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        mlp_dim = ((int(2 * DIM * 4 / 3) + 63) // 64) * 64
        self.blocks = nn.ModuleList([Block(DIM, N_HEADS, mlp_dim) for _ in range(DEPTH)])
        self.ln_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        hd = DIM // N_HEADS
        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, hd, 2).float() / hd))
        angles = torch.outer(torch.arange(MAX_SEQ_LEN).float(), freqs)
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new=200, temp=0.7, top_k=40, rep_penalty=1.2):
        for _ in range(max_new):
            idx_c = idx[:, -MAX_SEQ_LEN:]
            logits = self(idx_c)[:, -1, :]
            if rep_penalty > 1.0:
                for token_id in set(idx[0].tolist()[-50:]):
                    logits[0, token_id] /= rep_penalty
            logits = logits / temp
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            nx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nx], dim=1)
            if nx.item() == 2:
                break
        return idx


# ============ DATA LOADING ============
USER_PREFIX = "### User:\n"
ASSISTANT_PREFIX = "### Assistant:\n"
TURN_END = "\n\n"

def load_sft_data(data_dir, split='train'):
    """Load tokenized SFT data."""
    filepath = os.path.join(data_dir, f'{split}_sft.bin')
    data = np.fromfile(filepath, dtype=np.uint16)
    return torch.from_numpy(data.astype(np.int64))

def get_batch(data, batch_size, seq_len, device):
    """Get a random batch of sequences."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y


# ============ TRAINING ============
def train(args):
    device = args.device
    print(f"Device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    sp = spm.SentencePieceProcessor(args.tokenizer)
    
    # Load model
    print(f"Loading base model: {args.checkpoint}")
    model = GPT()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    clean_sd = {}
    for k, v in state_dict.items():
        k = k.replace('_orig_mod.', '').replace('module.', '')
        clean_sd[k] = v
    model.load_state_dict(clean_sd, strict=False)
    del ckpt, state_dict, clean_sd
    gc.collect()
    
    model = model.to(device).train()
    # Use bf16 for memory efficiency
    model = model.to(torch.bfloat16)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e9:.2f}B parameters")
    
    # Load data
    print(f"Loading SFT data from: {args.data_dir}")
    train_data = load_sft_data(args.data_dir, 'train')
    val_data = load_sft_data(args.data_dir, 'val')
    print(f"Train: {len(train_data)} tokens, Val: {len(val_data)} tokens")
    
    # Optimizer — 8-bit Adam for memory efficiency (halves optimizer states)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        print("Using 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        print("Using standard AdamW")
    
    # Cosine schedule with warmup
    def get_lr(step):
        if step < args.warmup_steps:
            return args.lr * step / args.warmup_steps
        decay_ratio = (step - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        return args.lr * 0.1 + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * decay_ratio))
    
    # Enable gradient checkpointing to save VRAM
    for block in model.blocks:
        block._gradient_checkpointing = True
    original_block_forward = Block.forward
    def checkpointed_forward(self, x, cos, sin):
        if self.training and hasattr(self, '_gradient_checkpointing') and self._gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(original_block_forward, self, x, cos, sin, use_reentrant=False)
        return original_block_forward(self, x, cos, sin)
    Block.forward = checkpointed_forward
    
    # Training loop
    best_val_loss = float('inf')
    grad_accum = args.grad_accum
    print(f"\nStarting SFT training for {args.max_steps} steps...")
    print(f"Batch size: {args.batch_size} x {grad_accum} accum = {args.batch_size * grad_accum} effective, Seq len: {MAX_SEQ_LEN}, LR: {args.lr}")
    
    t0 = time.time()
    for step in range(1, args.max_steps + 1):
        # LR schedule
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for micro in range(grad_accum):
            x, y = get_batch(train_data, args.batch_size, MAX_SEQ_LEN, device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1)) / grad_accum
            loss.backward()
            accum_loss += loss.item()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss = type('obj', (object,), {'item': lambda self: accum_loss})()  # For logging
        
        # Logging
        if step % 10 == 0:
            elapsed = time.time() - t0
            tps = step * args.batch_size * grad_accum * MAX_SEQ_LEN / elapsed
            print(f"Step {step}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {lr:.6f} | TPS: {tps:.0f} | {elapsed:.0f}s")
        
        # Eval
        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            val_losses = []
            for _ in range(20):
                x, y = get_batch(val_data, args.batch_size, MAX_SEQ_LEN, device)
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    val_loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
                val_losses.append(val_loss.item())
            avg_val = sum(val_losses) / len(val_losses)
            print(f"  📊 Val loss: {avg_val:.4f} {'(NEW BEST!)' if avg_val < best_val_loss else ''}")
            
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'step': step,
                    'val_loss': avg_val,
                    'config': {
                        'vocab_size': VOCAB_SIZE, 'dim': DIM, 'depth': DEPTH,
                        'n_heads': N_HEADS, 'max_seq_len': MAX_SEQ_LEN,
                    }
                }, args.output)
                print(f"  💾 Best model saved to {args.output}")
            
            model.train()
        
        # Generate samples periodically
        if step % args.sample_every == 0 or step == args.max_steps:
            model.eval()
            prompts = [
                ("EN", "### User:\nWhat is the capital of France?\n\n### Assistant:\n"),
                ("HE", "### User:\nמה בירת צרפת?\n\n### Assistant:\n"),
                ("AR", "### User:\nما هي عاصمة فرنسا؟\n\n### Assistant:\n"),
                ("FA", "### User:\nپایتخت فرانسه کجاست؟\n\n### Assistant:\n"),
                ("TRANSLATE", "### User:\nTranslate the following Hebrew text to English:\nשלום עולם, איך אתה היום?\n\n### Assistant:\n"),
            ]
            print(f"\n  🔤 Generation samples (step {step}):")
            for label, prompt in prompts:
                ids = torch.tensor([sp.encode(prompt)], device=device, dtype=torch.long)
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = model.generate(ids, max_new=100, temp=0.7, top_k=40)
                text = sp.decode(out[0].tolist())
                # Just show the assistant response
                if "### Assistant:" in text:
                    response = text.split("### Assistant:")[-1].strip()[:200]
                else:
                    response = text[len(prompt):].strip()[:200]
                print(f"    [{label}] {response}")
            print()
            model.train()
    
    # Final save
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SFT TRAINING COMPLETE")
    print(f"Steps: {args.max_steps}, Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"{'='*60}")
    
    # Upload to S3
    print("Uploading to S3...")
    os.system(f"aws s3 cp {args.output} s3://autoresearch-dashboard-196766918360/multilingual-7b/checkpoints/3b-v1-fsdp/sft_model.pt --quiet")
    os.system(f"aws s3 cp /tmp/sft/sft.log s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/sft_3b.log --quiet 2>/dev/null")
    print("Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', default='/tmp/sft/sft_model.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)  # 1 for 24GB GPU
    parser.add_argument('--grad-accum', type=int, default=4)  # Effective batch = 4
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-steps', type=int, default=2000)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=200)
    parser.add_argument('--sample-every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    train(args)


if __name__ == '__main__':
    main()
