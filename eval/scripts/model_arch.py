"""Shared model architecture for multilingual 3B GPT — must match training exactly."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

VOCAB_SIZE = 32000
DIM = 3072
DEPTH = 26
N_HEADS = 24
HEAD_DIM = DIM // N_HEADS  # 128
MAX_SEQ_LEN = 2048
ROPE_THETA = 10000.0
HIDDEN_DIM = ((int(2 * DIM * 4 / 3) + 63) // 64) * 64  # SwiGLU hidden


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=ROPE_THETA):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis):
    # x: (B, n_heads, S, head_dim)
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs_cis[:S].unsqueeze(0).unsqueeze(1)  # (1, 1, S, D//2)
    x_rot = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rot.type_as(x)


class FusedAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = FusedAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, hidden_dim)

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MultilingualGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([
            TransformerBlock(DIM, N_HEADS, HIDDEN_DIM) for _ in range(DEPTH)
        ])
        self.norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        # Tied embeddings
        self.head.weight = self.tok_emb.weight
        # Precompute RoPE
        self.register_buffer('freqs_cis', precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN))

    def forward(self, tokens, targets=None):
        B, S = tokens.shape
        x = self.tok_emb(tokens)
        mask = torch.triu(torch.full((S, S), float('-inf'), device=tokens.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
        for layer in self.layers:
            x = layer(x, self.freqs_cis, mask)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss


def load_model(path, device='cuda'):
    """Load model from checkpoint, stripping prefixes."""
    model = MultilingualGPT()
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    # Strip prefixes
    cleaned = {}
    for k, v in state.items():
        new_k = k
        for prefix in ['_orig_mod.', 'module.']:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    # Handle tied weights - remove head.weight if present (will be tied)
    if 'head.weight' in cleaned and 'tok_emb.weight' in cleaned:
        if torch.equal(cleaned['head.weight'], cleaned['tok_emb.weight']):
            del cleaned['head.weight']
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device).eval()
    return model


def load_tokenizer(path):
    """Load SentencePiece tokenizer."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(path)
    return sp
