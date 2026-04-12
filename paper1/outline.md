# Paper 1 Outline — "Cross-Lingual Transfer in Semitic Languages: Evidence from a 3B Multilingual Model"

**Target venue:** EMNLP / ACL Findings
**Key claim:** Linguistic family relatedness, not script similarity, enables zero-shot cross-lingual transfer in multilingual LMs.

---

## Abstract (~200 words)
- Hebrew-only task training → 9x Arabic improvement (zero Arabic data)
- Farsi (same script) gets zero transfer (controls for script)
- Multiple tasks: sentiment, NER, QA
- Comparison against mBERT/XLM-R shows the effect is amplified in custom models
- Morphological analysis explains mechanism

## 1. Introduction
- Cross-lingual transfer is well-studied (mBERT, XLM-R) but mostly for European languages
- Semitic family has unique morphological properties (triconsonantal roots)
- Research question: Does linguistic family membership predict transfer better than script overlap?
- Our setup: Hebrew/Arabic (same family, different script) vs Arabic/Farsi (different family, same script)
- Preview results: family > script, with specific morphological explanation

## 2. Background & Related Work
- 2.1 Cross-lingual transfer in multilingual models (Pires et al. 2019, Wu & Dredze 2019)
- 2.2 Semitic language processing (DictaBERT, Jais, AraGPT2)
- 2.3 Typological predictors of transfer (Lauscher et al. 2020, Gerz et al. 2018)
- 2.4 Script vs family effects (Muller et al. 2021, Fujinuma et al. 2022)
- 2.5 Our model: SemiticGPT (cite Paper 2)

## 3. Experimental Setup
- 3.1 Model: SemiticGPT-3B (3.04B params, 4 languages, custom tokenizer)
- 3.2 Languages and their relationships:
  - Hebrew (Semitic, Hebrew script)
  - Arabic (Semitic, Arabic script) ← same family as Hebrew
  - Farsi (Indo-European, Arabic script) ← same script as Arabic, different family
  - English (Indo-European, Latin script) ← control
- 3.3 Transfer protocol: train on Language A only, evaluate on B, C, D

## 4. Experiments

### 4.1 Experiment 1: Sentiment Classification
- Data: HE sentiment (mteb/hebrew_sentiment_analysis), AR (arbml/Sentiment_Analysis_Tweets), FA (sepidmnorozy/Persian_sentiment), EN (SST-2 subset)
- Configs: HE-only, All-langs, AR+FA only
- **Results:** HE-only → AR 49% (from 5.5%), FA 1.5% (from 0.5%)
- Already complete ✓

### 4.2 Experiment 2: Named Entity Recognition
- Data: WikiANN (HE, AR, FA, EN)
- Configs: HE-only, All-langs (upper bound), Baseline
- Hypothesis: NER for person/place names should transfer (cognate root patterns)
- Script: `run_ner_transfer.py`

### 4.3 Experiment 3: Question Answering
- Data: TyDiQA-GoldP (HE, AR, FA, EN splits)
- Configs: HE-only, All-langs, Baseline
- Hypothesis: Transfer depends on task structure, not just content
- Script: `run_qa_transfer.py`

### 4.4 Experiment 4: Comparison with mBERT / XLM-R
- Same sentiment task as 4.1, using:
  - bert-base-multilingual-cased (mBERT)
  - xlm-roberta-base (XLM-R)
  - SemiticGPT-3B (ours)
- All trained on Hebrew-only, evaluated on all 4 languages
- Key question: Is the HE→AR transfer an artifact of our model, or does it appear in all multilingual models?
- Script: `run_mbert_comparison.py`

## 5. Linguistic Analysis
- 5.1 Shared Semitic morphological system (triconsonantal roots, binyanim/awzān)
- 5.2 Structural parallels enabling transfer (verb patterns, construct state, agreement)
- 5.3 Why Farsi is excluded (agglutinative vs templatic, ezafe vs construct state)
- 5.4 Cognates and subword overlap
- Source: `morphological_analysis.md`

## 6. Results & Discussion
- 6.1 Summary table (all tasks × all configs × all languages)
- 6.2 The family > script pattern is consistent across tasks
- 6.3 Transfer magnitude varies by task (sentiment > NER > QA?)
- 6.4 mBERT/XLM-R comparison: does our custom model amplify transfer?
- 6.5 Bidirectional transfer (AR→HE also works, 19%→23%)
- 6.6 Focused > diluted (H3 beats H2 for Arabic)

## 7. Implications
- For multilingual NLP: group languages by family, not geography/script
- For Hebrew/Arabic NLP: build one good Hebrew system, get Arabic partially free
- For transfer research: morphological similarity is a better predictor than shared script

## 8. Limitations
- Small evaluation sets (200 samples)
- Single model (no confidence intervals at 3B scale)
- Limited to 4 languages (more Semitic languages like Amharic would strengthen claims)
- Sentiment data quality varies across languages

## 9. Conclusion
- Linguistic family > script for cross-lingual transfer (across 3 tasks)
- Practical implication: annotate in one Semitic language, deploy across family
- Custom multilingual model may amplify family-based transfer vs generic multilingual models

---

## Experiments Status

| Experiment | Script | Status | Needs GPU? |
|-----------|--------|--------|-----------|
| Sentiment (Exp H) | run_exp_hi.py | ✅ COMPLETE | — |
| NER Transfer | run_ner_transfer.py | 📝 Ready | Yes |
| QA Transfer | run_qa_transfer.py | 📝 Ready | Yes |
| mBERT Comparison | run_mbert_comparison.py | 📝 Ready | Yes (light) |
| Morphology Analysis | morphological_analysis.md | ✅ COMPLETE | No |

## Estimated GPU Time
- NER: ~1.5h (3 configs × 800 steps + eval)
- QA: ~1.5h (3 configs × 800 steps + eval)
- mBERT: ~1h (2 models × 3 epochs, small models)
- **Total: ~4h on g6e.xlarge ($1.20/hr) = ~$5**
