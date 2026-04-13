#!/usr/bin/env python3
"""Translation eval on GPU - no sentencepiece dep, download via S3"""
import json, os, sys, torch, time
sys.stdout.reconfigure(line_buffering=True)
import torch.nn as nn
import torch.nn.functional as F

DEVICE = 'cuda'
MODEL_DIR = '/tmp/model'
DATA_DIR = '/tmp/v4_data'
S3 = 's3://autoresearch-dashboard-196766918360/multilingual-7b'

VOCAB_SIZE, DIM, DEPTH, N_HEADS = 32000, 3072, 26, 24
HEAD_DIM = DIM // N_HEADS
ROPE_DIM = HEAD_DIM // 2
MAX_SEQ_LEN = 2048
EOS_ID = 2
USER_PREFIX = "<|user|> "
ASSISTANT_PREFIX = "<|assistant|> "

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).type_as(x) * self.weight

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
    def forward(self, x, rc, rs):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        def rot(x): return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], -1)
        cos, sin = rc[:T].unsqueeze(0).unsqueeze(0), rs[:T].unsqueeze(0).unsqueeze(0)
        qr, kr = q[..., :ROPE_DIM], k[..., :ROPE_DIM]
        q = torch.cat([qr*cos + rot(qr)*sin, q[..., ROPE_DIM:]], -1)
        k = torch.cat([kr*cos + rot(kr)*sin, k[..., ROPE_DIM:]], -1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d,8192,bias=False), nn.Linear(d,8192,bias=False), nn.Linear(8192,d,bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn, self.mlp, self.ln1, self.ln2 = Attention(d,h), MLP(d), RMSNorm(d), RMSNorm(d)
    def forward(self, x, rc, rs):
        x = x + self.attn(self.ln1(x), rc, rs)
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, DIM)
        self.blocks = nn.ModuleList([Block(DIM, N_HEADS) for _ in range(DEPTH)])
        self.ln_f = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.register_buffer('rope_cos', torch.zeros(MAX_SEQ_LEN, ROPE_DIM))
        self.register_buffer('rope_sin', torch.zeros(MAX_SEQ_LEN, ROPE_DIM))
    def forward(self, x):
        h = self.tok_emb(x)
        for b in self.blocks: h = b(h, self.rope_cos, self.rope_sin)
        return self.head(self.ln_f(h))

# Use sentencepiece C++ binary directly via subprocess if available, 
# or install from the /opt/pytorch venv's existing packages
# Actually, let's just install spm from the wheel on S3
os.makedirs(MODEL_DIR, exist_ok=True)

# First, upload sentencepiece wheel to S3 from local, then download on GPU
# Actually simpler: just use ctypes to load the .so directly
# Simplest: check if spm is in /opt/pytorch
print("Checking for sentencepiece...")
import subprocess
result = subprocess.run(['/opt/pytorch/bin/python', '-c', 'import sentencepiece; print("OK")'], 
                       capture_output=True, text=True)
if result.stdout.strip() == 'OK':
    print("Found in /opt/pytorch!")
    # Add the site-packages to our path
    sp_path = subprocess.run(['/opt/pytorch/bin/python', '-c', 
        'import sentencepiece, os; print(os.path.dirname(os.path.dirname(sentencepiece.__file__)))'],
        capture_output=True, text=True).stdout.strip()
    sys.path.insert(0, sp_path)
else:
    # Try pip install from S3 wheel
    print("Not found, trying pip install from local cache...")
    os.system('pip3 install sentencepiece 2>/dev/null || /opt/pytorch/bin/pip install sentencepiece 2>/dev/null')

import sentencepiece as spm

print("Downloading model + data...")
os.system(f'aws s3 cp {S3}/checkpoints/improved-v4/sft_model.pt {MODEL_DIR}/v4_model.pt --only-show-errors')
os.system(f'aws s3 cp {S3}/tokenizer/multilingual_32k.model {MODEL_DIR}/tokenizer.model --only-show-errors')
os.system(f'aws s3 sync {S3}/v4_data/ {DATA_DIR}/ --only-show-errors')

sp = spm.SentencePieceProcessor()
sp.load(f'{MODEL_DIR}/tokenizer.model')

print("Loading model (GPU, bfloat16, inference only)...")
model = GPT()
state = torch.load(f'{MODEL_DIR}/v4_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(state['model_state_dict'])
model = model.to(DEVICE).bfloat16()
model.eval()
del state
torch.cuda.empty_cache()
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

@torch.no_grad()
def generate(prompt_text, max_tokens=80):
    prompt = f"{USER_PREFIX}{prompt_text}\n{ASSISTANT_PREFIX}"
    ids = sp.encode(prompt)
    x = torch.tensor([ids], device=DEVICE)
    for _ in range(max_tokens):
        logits = model(x[:, -MAX_SEQ_LEN:])
        next_id = logits[0, -1].float().argmax().item()
        if next_id == EOS_ID: break
        x = torch.cat([x, torch.tensor([[next_id]], device=DEVICE)], dim=1)
    return sp.decode(x[0, len(ids):].tolist()).strip()

print("\n" + "=" * 60)
print("TRANSLATION EVAL — Custom Prompts")
print("=" * 60)

tests = [
    ("Translate to Hebrew: The weather is beautiful today and I want to go outside", "EN→HE"),
    ("Translate to Hebrew: I love reading books about history", "EN→HE"),
    ("Translate to Hebrew: Good morning, how are you feeling today?", "EN→HE"),
    ("Translate to Arabic: Good morning, how are you?", "EN→AR"),
    ("Translate to Arabic: I love reading books about history", "EN→AR"),
    ("Translate to Arabic: The weather is beautiful today", "EN→AR"),
    ("Translate to Farsi: The weather is beautiful today", "EN→FA"),
    ("Translate to Farsi: I love reading books", "EN→FA"),
    ("Translate to Farsi: Good morning, how are you?", "EN→FA"),
    ("תרגם לאנגלית: אני אוהב לקרוא ספרים על היסטוריה", "HE→EN"),
    ("תרגם לאנגלית: מזג האוויר יפה היום", "HE→EN"),
    ("תרגם לאנגלית: בוקר טוב, מה שלומך?", "HE→EN"),
    ("Translate to English: الطقس جميل اليوم", "AR→EN"),
    ("Translate to English: أحب قراءة الكتب عن التاريخ", "AR→EN"),
    ("Translate to English: هوا امروز زیباست", "FA→EN"),
    ("Translate to English: من عاشق خواندن کتاب هستم", "FA→EN"),
]

for prompt, direction in tests:
    result = generate(prompt)
    print(f"\n[{direction}]")
    print(f"  IN:  {prompt}")
    print(f"  OUT: {result[:150]}")
    sys.stdout.flush()

print("\n" + "=" * 60)
print("OPUS-100 TEST SET (with reference translations)")
print("=" * 60)

for lang in ['he', 'ar', 'fa']:
    path = f'{DATA_DIR}/translation_eval_{lang}.jsonl'
    if not os.path.exists(path): continue
    print(f"\n--- {lang.upper()} (first 5) ---")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= 5: break
            s = json.loads(line)
            result = generate(s['instruction'])
            print(f"\n  IN:       {s['instruction'][:80]}")
            print(f"  Expected: {s['output'][:80]}")
            print(f"  Got:      {result[:80]}")
            sys.stdout.flush()

print("\n\nDone!")
