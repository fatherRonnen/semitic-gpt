#!/usr/bin/env python3
"""
Multilingual 3.14B GPT Training — FSDP Version (Arabic-Rebalanced Data)

Converted from DDP script. Key changes:
- FullyShardedDataParallel (FSDP) with FULL_SHARD strategy
- Mixed precision: bf16 compute, fp32 reduce
- transformer_auto_wrap_policy wrapping each Block
- Activation checkpointing via FSDP's apply_activation_checkpointing
- FULL_STATE_DICT for checkpoint saving
- SWA simplified (gather full state dict on rank 0)

Architecture: dim=3072, depth=26, heads=24, ~3.14B params
Data: training-data-v2 (4.48B tokens, multi-epoch)
Schedule: WSD-LINEAR
LR: 3e-4

Run with:
  /opt/pytorch/bin/torchrun --nproc_per_node=8 --master_port=29500 train_multilingual_3b_fsdp.py
"""

import os, sys, json, math, time, copy, functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# Activation checkpointing for FSDP
try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )
    HAS_FSDP_AC = True
except ImportError:
    HAS_FSDP_AC = False

# ============ MODEL CONFIG ============
VOCAB_SIZE = 32000
DIM = 3072
DEPTH = 26
N_HEADS = 24
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000
DROPOUT = 0.05

# ============ TRAINING CONFIG ============
TOTAL_STEPS = 20000
WARMUP_STEPS = 600
STABLE_END = 14000
MIN_LR_RATIO = 0.03

BATCH_PER_GPU = 4          # FSDP uses less memory per GPU → can increase from 2 to 4
GRAD_ACCUM = 8             # With 8 GPUs: 8*4*8 = 256 seqs = 524K tokens/step
# Total: 20000 * 524288 = 10.49B tokens

ADAMW_LR = 3e-4
ADAMW_BETAS = (0.9, 0.98)
ADAMW_WD = 0.02
ADAMW_EPS = 1e-8

LABEL_SMOOTHING = 0.06
GRAD_CLIP = 1.0

SWA_START_FRAC = 0.40
SWA_FREQ = 40

EVAL_EVERY = 500
SAVE_EVERY = 2000
LOG_EVERY = 50

DATA_DIR = "/tmp/training-data"
CKPT_DIR = "/tmp/checkpoints"
LOG_FILE = "/tmp/training.log"
EVAL_FILE = "/tmp/eval_results.json"

S3_BUCKET = "autoresearch-dashboard-196766918360"
S3_PREFIX = "multilingual-7b"
VERSION = "3b-v1-fsdp"

# ============ MODEL ============
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
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3*dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = dropout
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, dropout)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x, cos, sin):
        x = x + self.drop(self.attn(self.ln1(x), cos, sin))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=DIM, depth=DEPTH, n_heads=N_HEADS,
                 max_seq_len=MAX_SEQ_LEN, rope_theta=ROPE_THETA, dropout=DROPOUT):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        mlp_dim = ((int(2 * dim * 4 / 3) + 63) // 64) * 64  # = 8192 for dim=3072
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_dim, dropout) for _ in range(depth)])
        self.ln_f = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        hd = dim // n_heads
        freqs = 1.0 / (rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        angles = torch.outer(torch.arange(max_seq_len).float(), freqs)
        self.register_buffer('rope_cos', angles.cos())
        self.register_buffer('rope_sin', angles.sin())
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T][None, None]
        sin = self.rope_sin[:T][None, None]
        for block in self.blocks:
            # NOTE: With FSDP activation checkpointing applied externally via
            # apply_activation_checkpointing, we do NOT need manual torch_checkpoint here.
            # FSDP's checkpoint wrapper handles it.
            x = block(x, cos, sin)
        return self.head(self.ln_f(x))

# ============ WSD LINEAR SCHEDULE ============
def wsd_lr_linear(step, total_steps, warmup_steps, stable_end, min_lr_ratio, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    elif step < stable_end:
        return base_lr
    else:
        progress = (step - stable_end) / max(total_steps - stable_end, 1)
        return base_lr * (1.0 - progress * (1.0 - min_lr_ratio))

# ============ DATA LOADING ============
class BinaryDataset:
    def __init__(self, path, seq_len):
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len
        self.n_tokens = len(self.data)
    def get_batch(self, batch_size, device, rng):
        ix = torch.from_numpy(rng.integers(0, self.n_tokens - self.seq_len - 1, size=(batch_size,)))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.seq_len].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.seq_len].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

def load_val_data(path, seq_len, max_batches=20, batch_size=8):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    n_tokens = len(data)
    batches = []
    stride = seq_len + 1
    all_starts = list(range(0, n_tokens - stride, stride))
    max_samples = max_batches * batch_size
    if len(all_starts) > max_samples:
        step_size = len(all_starts) // max_samples
        all_starts = all_starts[::step_size][:max_samples]
    for i in range(0, len(all_starts), batch_size):
        batch_starts = all_starts[i:i+batch_size]
        if len(batch_starts) < batch_size:
            break
        x = torch.stack([torch.from_numpy(data[s:s+seq_len].astype(np.int64)) for s in batch_starts])
        y = torch.stack([torch.from_numpy(data[s+1:s+1+seq_len].astype(np.int64)) for s in batch_starts])
        batches.append((x, y))
    return batches

@torch.no_grad()
def evaluate(model, val_batches, device):
    """Evaluate model. Works with both FSDP-wrapped and unwrapped models."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in val_batches:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += y.numel()
    model.train()
    return (total_loss / total_tokens) / math.log(2) if total_tokens > 0 else float('inf')

class Logger:
    def __init__(self, log_file, rank):
        self.rank = rank
        if rank == 0:
            self.f = open(log_file, 'w')
    def log(self, msg):
        if self.rank == 0:
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            self.f.write(line + '\n')
            self.f.flush()
    def close(self):
        if self.rank == 0:
            self.f.close()

class SWAState:
    """SWA for FSDP: gathers full state dict on rank 0 before averaging."""
    def __init__(self):
        self.avg_state = None
        self.n_averaged = 0

    def update(self, model):
        """Gather full state dict from FSDP model and update running average on rank 0."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state = model.state_dict()

        # Only rank 0 gets the full state dict with rank0_only=True
        if dist.get_rank() != 0:
            return

        if self.avg_state is None:
            self.avg_state = {k: v.float().clone() for k, v in state.items()}
            self.n_averaged = 1
        else:
            n = self.n_averaged
            for k in self.avg_state:
                self.avg_state[k] = (self.avg_state[k] * n + state[k].float()) / (n + 1)
            self.n_averaged += 1

# ============ FSDP HELPERS ============
def get_fsdp_wrap_policy():
    """Wrap each Block as a separate FSDP unit."""
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )

def get_mixed_precision():
    """bf16 for compute, fp32 for reduce (gradient all-reduce in fp32 for stability)."""
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )

def save_fsdp_checkpoint(model, optimizer, scaler, step, best_val_bpb,
                         tokens_processed, eval_results, swa_n, ckpt_dir, logger):
    """Save full state dict checkpoint from FSDP model (rank 0 only)."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()

    if dist.get_rank() == 0:
        ckpt = {
            'step': step,
            'model': model_state,
            'best_val_bpb': best_val_bpb,
            'tokens_processed': tokens_processed,
            'eval_results': eval_results,
            'swa_n': swa_n,
            # NOTE: We don't save optimizer/scaler state for simplicity with FSDP.
            # For full resumability, use FSDP's ShardedStateDictConfig instead.
        }
        torch.save(ckpt, f"{ckpt_dir}/ckpt_step_{step}.pt")
        logger.log(f"Saved checkpoint at step {step}")
    dist.barrier()

def save_fsdp_model(model, path, logger):
    """Save just the model state dict (rank 0 only)."""
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state, path)
        logger.log(f"Saved model to {path}")
    dist.barrier()

# ============ MAIN ============
def main():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    effective_batch = BATCH_PER_GPU * GRAD_ACCUM * world_size
    tokens_per_step = effective_batch * MAX_SEQ_LEN

    logger = Logger(LOG_FILE, rank)
    logger.log(f"=== Multilingual 3.14B Training — FSDP (Arabic-Rebalanced) ===")
    logger.log(f"World size: {world_size}, Batch/GPU: {BATCH_PER_GPU}, Grad accum: {GRAD_ACCUM}")
    logger.log(f"Effective batch: {effective_batch} seqs = {tokens_per_step:,} tokens/step")
    logger.log(f"Total steps: {TOTAL_STEPS} = {TOTAL_STEPS * tokens_per_step:,} tokens")
    logger.log(f"Schedule: WSD-LINEAR | warmup={WARMUP_STEPS} | stable_end={STABLE_END} | total={TOTAL_STEPS}")
    logger.log(f"AdamW LR={ADAMW_LR}, betas={ADAMW_BETAS}, WD={ADAMW_WD}")
    logger.log(f"Label smoothing={LABEL_SMOOTHING}, min_lr={MIN_LR_RATIO}, grad_clip={GRAD_CLIP}")
    logger.log(f"SWA: start={int(TOTAL_STEPS*SWA_START_FRAC)}, freq={SWA_FREQ}")
    logger.log(f"Model: dim={DIM}, depth={DEPTH}, heads={N_HEADS}, dropout={DROPOUT}")
    logger.log(f"FSDP: FULL_SHARD, MixedPrecision(bf16/fp32), Block-level wrapping")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Data
    logger.log("Loading training data...")
    train_ds = BinaryDataset(f"{DATA_DIR}/train.bin", MAX_SEQ_LEN)
    logger.log(f"Train tokens: {train_ds.n_tokens:,}")

    logger.log("Loading validation data...")
    val_batches = load_val_data(f"{DATA_DIR}/val.bin", MAX_SEQ_LEN)
    val_lang_batches = {}
    for lang in ['en', 'ar', 'he', 'fa']:
        vpath = f"{DATA_DIR}/val_{lang}.bin"
        if os.path.exists(vpath):
            val_lang_batches[lang] = load_val_data(vpath, MAX_SEQ_LEN)
            logger.log(f"  val_{lang}: {len(val_lang_batches[lang])} batches")

    # Model — create on CPU first, then FSDP wraps and shards to GPUs
    logger.log("Creating model...")
    torch.manual_seed(42)
    model = GPT()
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_emb = n_params - model.tok_emb.weight.numel()
    logger.log(f"Model params: {n_params:,} (non-embedding: {n_params_no_emb:,})")

    # Wrap with FSDP
    logger.log("Wrapping model with FSDP...")
    wrap_policy = get_fsdp_wrap_policy()
    mixed_precision = get_mixed_precision()

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        limit_all_gathers=True,
        use_orig_params=True,  # Required for weight decay on specific params
    )

    logger.log(f"FSDP wrapped. GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # Apply activation checkpointing to each Block within FSDP
    if HAS_FSDP_AC:
        check_fn = lambda submodule: isinstance(submodule, Block)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper,
            check_fn=check_fn,
        )
        logger.log("Applied FSDP activation checkpointing to Block layers")
    else:
        logger.log("WARNING: FSDP activation checkpointing not available, using manual checkpointing")

    # Optimizer — use standard AdamW (bitsandbytes may not work well with FSDP sharded params)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WD,
                                         betas=ADAMW_BETAS, eps=ADAMW_EPS)
        logger.log("Using 8-bit AdamW (bitsandbytes)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=ADAMW_LR, weight_decay=ADAMW_WD,
                                       betas=ADAMW_BETAS, eps=ADAMW_EPS)
        logger.log("Using standard AdamW (bitsandbytes not available)")

    swa = SWAState()
    swa_start_step = int(TOTAL_STEPS * SWA_START_FRAC)
    rng = np.random.default_rng(42 + rank)

    # Use ShardedGradScaler for FSDP (handles sharded gradients correctly)
    scaler = ShardedGradScaler()

    best_val_bpb = float('inf')
    eval_results = []
    tokens_processed = 0
    start_time = time.time()

    logger.log("Starting training...")

    for step in range(1, TOTAL_STEPS + 1):
        model.train()
        lr = wsd_lr_linear(step, TOTAL_STEPS, WARMUP_STEPS, STABLE_END, MIN_LR_RATIO, ADAMW_LR)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro in range(GRAD_ACCUM):
            x, y = train_ds.get_batch(BATCH_PER_GPU, device, rng)

            # For FSDP with gradient accumulation, we need to use no_sync() for all
            # micro-steps except the last one to avoid unnecessary all-reduce
            ctx = model.no_sync() if micro < GRAD_ACCUM - 1 else nullcontext()
            with ctx:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                                           label_smoothing=LABEL_SMOOTHING) / GRAD_ACCUM
                scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        # FSDP clip_grad_norm_ works on the sharded params
        model.clip_grad_norm_(GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        tokens_processed += tokens_per_step

        # SWA: gather full state dict and average on rank 0
        if step >= swa_start_step and step % SWA_FREQ == 0:
            swa.update(model)

        if step % LOG_EVERY == 0 and rank == 0:
            elapsed = time.time() - start_time
            tps = tokens_processed / elapsed
            bpb = accum_loss / math.log(2)
            phase = "warmup" if step < WARMUP_STEPS else ("stable" if step < STABLE_END else "decay")
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            logger.log(f"Step {step}/{TOTAL_STEPS} [{phase}] | Loss: {accum_loss:.4f} | "
                       f"BPB: {bpb:.4f} | LR: {lr:.6f} | Tokens: {tokens_processed:,} | "
                       f"TPS: {tps:,.0f} | SWA: {swa.n_averaged} | Mem: {mem:.1f}GB | {elapsed/60:.1f}min")

        if step % EVAL_EVERY == 0 or step == TOTAL_STEPS:
            # All ranks participate in eval (FSDP needs all ranks for forward pass)
            combined_bpb = evaluate(model, val_batches, device)

            if rank == 0:
                logger.log(f"--- Evaluation at step {step} ---")
                logger.log(f"  Combined val BPB: {combined_bpb:.4f}")
                result = {"step": step, "tokens": tokens_processed, "combined_bpb": combined_bpb}

            for lang, batches in val_lang_batches.items():
                lang_bpb = evaluate(model, batches, device)
                if rank == 0:
                    result[f"{lang}_bpb"] = lang_bpb
                    logger.log(f"  {lang} val BPB: {lang_bpb:.4f}")

            if rank == 0:
                eval_results.append(result)
                with open(EVAL_FILE, 'w') as f:
                    json.dump(eval_results, f, indent=2)
                if combined_bpb < best_val_bpb:
                    best_val_bpb = combined_bpb
                    save_fsdp_model(model, f"{CKPT_DIR}/best_model.pt", logger)
                    logger.log(f"  New best! BPB: {combined_bpb:.4f}")
                else:
                    pass  # no new best
            dist.barrier()

        if step % SAVE_EVERY == 0:
            save_fsdp_checkpoint(
                model, optimizer, scaler, step, best_val_bpb,
                tokens_processed, eval_results, swa.n_averaged, CKPT_DIR, logger
            )
            if rank == 0:
                os.system(f"aws s3 cp {CKPT_DIR}/best_model.pt s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_best_model.pt --quiet &")
                os.system(f"aws s3 cp {EVAL_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_eval_results.json --quiet &")
                os.system(f"aws s3 cp {LOG_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_training.log --quiet &")

    # Finalize
    save_fsdp_model(model, f"{CKPT_DIR}/final_model.pt", logger)

    if rank == 0:
        # SWA evaluation
        if swa.avg_state is not None and swa.n_averaged > 0:
            logger.log(f"Evaluating SWA model ({swa.n_averaged} checkpoints)...")
            swa_model = GPT().to(device)
            swa_load = {k: v.float().to(device) for k, v in swa.avg_state.items()}
            swa_model.load_state_dict(swa_load)
            swa_bpb = evaluate(swa_model, val_batches, device)
            logger.log(f"SWA model combined BPB: {swa_bpb:.4f} (vs best raw: {best_val_bpb:.4f})")
            swa_result = {"step": "swa", "combined_bpb": swa_bpb, "n_averaged": swa.n_averaged}
            for lang, batches in val_lang_batches.items():
                lang_bpb = evaluate(swa_model, batches, device)
                swa_result[f"{lang}_bpb"] = lang_bpb
                logger.log(f"  SWA {lang} BPB: {lang_bpb:.4f}")
            eval_results.append(swa_result)
            torch.save(swa_load, f"{CKPT_DIR}/swa_model.pt")
            with open(EVAL_FILE, 'w') as f:
                json.dump(eval_results, f, indent=2)
            del swa_model

        # Final S3 upload
        logger.log("Uploading all artifacts to S3...")
        os.system(f"aws s3 sync {CKPT_DIR}/ s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}/")
        os.system(f"aws s3 cp {LOG_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_training.log")
        os.system(f"aws s3 cp {EVAL_FILE} s3://{S3_BUCKET}/{S3_PREFIX}/checkpoints/{VERSION}_eval_results.json")

        elapsed = time.time() - start_time
        logger.log(f"=== Training complete! Total time: {elapsed/3600:.2f}h ===")
        logger.log(f"Best combined BPB: {best_val_bpb:.4f}")
        logger.log(f"Total tokens: {tokens_processed:,}")

    logger.close()
    dist.destroy_process_group()

# Need nullcontext for no_sync
from contextlib import nullcontext

if __name__ == '__main__':
    main()
