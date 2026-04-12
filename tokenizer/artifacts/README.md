# Tokenizer Artifacts

## Files in this directory

- `fertility_report.json` — Per-language fertility analysis (tokens/word ratio)
- `vocabulary_stats.json` — Vocabulary composition and statistics

## Large artifacts (not in git)

The trained SentencePiece model file is too large for the repository:

- **`multilingual_32k.model`** (~900KB) — The trained BPE tokenizer model
  - Location: `s3://autoresearch-dashboard-196766918360/multilingual-7b/tokenizer/multilingual_32k.model`
  - Also available locally at the training machine under `/tmp/experiments/tokenizer/multilingual_32k.model`
  
- **`multilingual_32k.vocab`** (~550KB) — Human-readable vocabulary file
  - Location: `s3://autoresearch-dashboard-196766918360/multilingual-7b/tokenizer/multilingual_32k.vocab`

## Usage

```python
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('multilingual_32k.model')

# Encode
tokens = sp.encode("שלום עולם", out_type=int)

# Decode
text = sp.decode(tokens)
```
