# Evaluation Prompts

This document records the exact prompt formats used for each evaluation task in the SemiticGPT experiments.

---

## 1. Sentiment Classification (Binary: Positive/Negative)

Used in: `exp_a_hebrew_downstream.py`, `exp_a_fix_classification.py`

### SFT-format prompt (primary):
```
<|user|>
{instruction_text}
<|assistant|>
```

### Fallback prompt (base model):
```
{instruction_text}
תשובה: 
```

**Label mapping (Hebrew):**
- חיובי / חיוב / positive → positive
- שלילי / שליל / negative → negative
- ניטרלי / neutral → neutral

**Evaluation method:** Generation-based classification. The model generates free text, which is matched against the label map via string containment. Max generation tokens: 30.

---

## 2. News Topic Classification (4-class)

Used in: `exp_a_fix_classification.py`

**Classes:** World, Sports, Business, Science/Technology  
**Chance level:** 25%

Same prompt format as sentiment (SFT-format with `<|user|>` / `<|assistant|>` tags). The instruction contains the news article text and asks for categorization.

---

## 3. Reading Comprehension (Belebele)

Used in: `eval_belebele.py`

### Log-likelihood scoring prompt:
```
Passage: {flores_passage}
Question: {question}
Answer: {answer_choice}
```

**Method:** For each of the 4 answer choices, compute P(answer | passage + question) using log-likelihood scoring. The choice with highest log-likelihood is selected.

**Languages:** Hebrew (heb_Hebr), Arabic (arb_Arab), English (eng_Latn), Farsi (pes_Arab)

---

## 4. Translation

Used in: `eval_translation.py`

### Translation prompt:
```
<|user|>
Translate from {source_language_name} to {target_language_name}:
{source_text}
<|assistant|>
```

**Language names used:**
- Hebrew, Arabic, English, Farsi

**Evaluation metric:** chrF (character n-gram F-score)
**Max generation tokens:** 200

**Directions evaluated:** All 10 pairwise directions across 4 languages (AR→FA, FA→AR, HE→EN, EN→HE, AR→EN, EN→AR, HE→AR, AR→HE, EN→FA, FA→EN)

---

## 5. Cross-lingual Embedding Extraction

Used in: `eval_embeddings.py`

### Method:
No explicit prompt—raw sentences are fed to the model and embeddings are extracted from the **last hidden layer** using **mean pooling** across all token positions.

```python
def get_embedding(model, text, pool='mean'):
    """Get embedding from model's last hidden layer."""
    # Encode text → feed through model → extract last layer hidden states → mean pool
```

**Evaluation:** 10-way cross-lingual retrieval. Given a sentence in language A, retrieve its translation from 10 candidates in language B using cosine similarity. Chance = 10%.

**Test sentences:** 10 parallel sentence pairs across all 4 languages, covering diverse topics.

---

## 6. Generation Quality Probes (Cross-lingual)

Used in: `exp_b_crosslingual.py`

### Prompt prefixes per language:

**Hebrew:**
- ישראל היא מדינה
- הטכנולוגיה משנה את העולם
- ירושלים היא עיר

**Arabic:**
- اللغة العربية هي
- التكنولوجيا تغير العالم
- القاهرة مدينة

**English:**
- Artificial intelligence is
- The history of the Middle East
- Technology is changing

**Farsi:**
- زبان فارسی یک
- تهران شهری است
- فناوری جهان را

**Method:** Open-ended generation from prefix. Max tokens: 80. Evaluates fluency and coherence qualitatively.

---

## 7. BPB (Bits Per Byte) Evaluation

Used in: `exp_b_crosslingual.py`, multiple SFT evaluation scripts

### Method:
No prompt—compute cross-entropy loss on held-out test data, converted to bits-per-byte:

```
BPB = (total_loss * total_tokens) / (total_bytes * ln(2))
```

Evaluated on 10K held-out tokens per language.

---

## Notes

- All generative evaluations use greedy decoding (temperature=0) with the specified `max_tokens`
- The SFT instruction format (`<|user|>\n...\n<|assistant|>\n`) matches the format used during SFT training
- Base (pretrained) model evaluations use the fallback prompt format without special tokens
- Evaluation sample sizes are deliberately small (200 per task) due to generation-based evaluation being compute-intensive
