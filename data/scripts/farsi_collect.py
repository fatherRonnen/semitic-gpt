#!/usr/bin/env python3
"""
Farsi Data Collection Pipeline for Multilingual 7B Model
Collects, filters, and saves Farsi text data from multiple sources.
Outputs JSONL files to /data/farsi_data/
"""
import os
import json
import hashlib
import time
import sys
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DATA_DIR = Path("/data/farsi_data")
MANIFEST = {"sources": {}, "total_documents": 0, "total_bytes": 0, "estimated_tokens": 0}

def setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["mc4", "cc100", "wikipedia", "oscar", "parallel", "twitter", "telegram", "news", "literature", "misc"]:
        (DATA_DIR / sub).mkdir(exist_ok=True)

def save_jsonl(records, path):
    """Append records to a JSONL file."""
    with open(path, 'a', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def basic_quality_filter(text):
    """Basic quality filtering for Persian text."""
    if not text or len(text.strip()) < 50:
        return False
    text = text.strip()
    # Skip if mostly URLs/HTML
    url_chars = text.count('http') + text.count('www.') + text.count('<')
    if url_chars > len(text) * 0.3:
        return False
    # Check for some Persian characters (Unicode range for Arabic script used by Persian)
    persian_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
    if persian_chars < len(text) * 0.3:
        return False
    return True

def has_zwnj(text):
    """Check for ZWNJ - quality signal for Persian."""
    return '\u200c' in text

def collect_mc4():
    """Collect mC4 Persian - our biggest source (~3-5B tokens)."""
    log.info("=== Collecting mC4 Persian ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("mc4", "fa", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        file_idx = 0
        
        for i, example in enumerate(ds):
            text = example.get("text", "")
            if basic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "mc4",
                    "lang": "fa",
                    "has_zwnj": has_zwnj(text),
                    "url": example.get("url", ""),
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
            
            if len(batch) >= 10000:
                outpath = DATA_DIR / "mc4" / f"mc4_fa_{file_idx:04d}.jsonl"
                save_jsonl(batch, outpath)
                log.info(f"mc4: saved {total_docs} docs, {total_bytes/1e9:.2f} GB so far")
                batch = []
                file_idx += 1
            
            # Safety: log progress every 100k
            if (i + 1) % 100000 == 0:
                log.info(f"mc4: processed {i+1} raw examples, kept {total_docs}")
        
        # Save remaining
        if batch:
            save_jsonl(batch, DATA_DIR / "mc4" / f"mc4_fa_{file_idx:04d}.jsonl")
        
        MANIFEST["sources"]["mc4"] = {
            "docs": total_docs,
            "bytes": total_bytes,
            "quality": "medium",
            "files": file_idx + 1
        }
        log.info(f"mc4 DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB")
    except Exception as e:
        log.error(f"mc4 failed: {e}")

def collect_cc100():
    """Collect CC-100 Persian."""
    log.info("=== Collecting CC-100 Persian ===")
    try:
        from datasets import load_dataset
        ds = load_dataset("cc100", lang="fa", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        file_idx = 0
        
        for i, example in enumerate(ds):
            text = example.get("text", "")
            if basic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "cc100",
                    "lang": "fa",
                    "has_zwnj": has_zwnj(text),
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
            
            if len(batch) >= 10000:
                outpath = DATA_DIR / "cc100" / f"cc100_fa_{file_idx:04d}.jsonl"
                save_jsonl(batch, outpath)
                log.info(f"cc100: saved {total_docs} docs, {total_bytes/1e9:.2f} GB so far")
                batch = []
                file_idx += 1
        
        if batch:
            save_jsonl(batch, DATA_DIR / "cc100" / f"cc100_fa_{file_idx:04d}.jsonl")
        
        MANIFEST["sources"]["cc100"] = {
            "docs": total_docs,
            "bytes": total_bytes,
            "quality": "medium",
            "files": file_idx + 1
        }
        log.info(f"cc100 DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB")
    except Exception as e:
        log.error(f"cc100 failed: {e}")

def collect_wikipedia():
    """Collect Persian Wikipedia via HuggingFace."""
    log.info("=== Collecting Persian Wikipedia ===")
    try:
        from datasets import load_dataset
        # Try pre-processed HF version first
        ds = load_dataset("codersan/Persian-Wikipedia-Corpus", split="train", streaming=True)
        
        batch = []
        total_docs = 0
        total_bytes = 0
        
        for example in ds:
            text = example.get("text", "") or example.get("content", "")
            if basic_quality_filter(text):
                rec = {
                    "text": text,
                    "source": "wikipedia",
                    "lang": "fa",
                    "has_zwnj": has_zwnj(text),
                    "title": example.get("title", ""),
                }
                batch.append(rec)
                total_bytes += len(text.encode('utf-8'))
                total_docs += 1
        
        save_jsonl(batch, DATA_DIR / "wikipedia" / "wiki_fa.jsonl")
        
        MANIFEST["sources"]["wikipedia"] = {
            "docs": total_docs,
            "bytes": total_bytes,
            "quality": "high"
        }
        log.info(f"Wikipedia DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB")
    except Exception as e:
        log.error(f"Wikipedia HF failed: {e}, trying dump...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikipedia", "20231101.fa", split="train", streaming=True)
            batch = []
            total_docs = 0
            total_bytes = 0
            for example in ds:
                text = example.get("text", "")
                if basic_quality_filter(text):
                    rec = {"text": text, "source": "wikipedia", "lang": "fa", "title": example.get("title", "")}
                    batch.append(rec)
                    total_bytes += len(text.encode('utf-8'))
                    total_docs += 1
            save_jsonl(batch, DATA_DIR / "wikipedia" / "wiki_fa.jsonl")
            MANIFEST["sources"]["wikipedia"] = {"docs": total_docs, "bytes": total_bytes, "quality": "high"}
            log.info(f"Wikipedia DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB")
        except Exception as e2:
            log.error(f"Wikipedia fallback also failed: {e2}")

def collect_oscar():
    """Collect OSCAR Persian."""
    log.info("=== Collecting OSCAR Persian ===")
    try:
        from datasets import load_dataset
        # Try multiple OSCAR versions
        for name in ["oscar-corpus/OSCAR-2301", "oscar-corpus/OSCAR-2201", "oscar"]:
            try:
                if name == "oscar":
                    ds = load_dataset(name, "unshuffled_deduplicated_fa", split="train", streaming=True)
                else:
                    ds = load_dataset(name, language="fa", split="train", streaming=True)
                
                batch = []
                total_docs = 0
                total_bytes = 0
                file_idx = 0
                
                for i, example in enumerate(ds):
                    text = example.get("text", "")
                    if basic_quality_filter(text):
                        rec = {"text": text, "source": "oscar", "lang": "fa", "has_zwnj": has_zwnj(text)}
                        batch.append(rec)
                        total_bytes += len(text.encode('utf-8'))
                        total_docs += 1
                    
                    if len(batch) >= 10000:
                        save_jsonl(batch, DATA_DIR / "oscar" / f"oscar_fa_{file_idx:04d}.jsonl")
                        log.info(f"oscar: saved {total_docs} docs, {total_bytes/1e9:.2f} GB")
                        batch = []
                        file_idx += 1
                
                if batch:
                    save_jsonl(batch, DATA_DIR / "oscar" / f"oscar_fa_{file_idx:04d}.jsonl")
                
                MANIFEST["sources"]["oscar"] = {"docs": total_docs, "bytes": total_bytes, "quality": "medium"}
                log.info(f"OSCAR DONE: {total_docs} docs, {total_bytes/1e9:.2f} GB")
                return
            except Exception as inner_e:
                log.warning(f"OSCAR {name} failed: {inner_e}, trying next...")
                continue
        log.error("All OSCAR versions failed")
    except Exception as e:
        log.error(f"OSCAR failed: {e}")

def collect_parallel():
    """Collect parallel/bilingual corpora - CRITICAL for multilingual training."""
    log.info("=== Collecting Parallel Corpora ===")
    from datasets import load_dataset
    
    parallel_sources = [
        # EN-FA parallel
        ("opus100", {"path": "opus100", "name": "en-fa"}, "en", "fa"),
        ("opus_books", {"path": "opus_books", "name": "en-fa"}, "en", "fa"),
        # Try TED talks
        ("ted_talks_iwslt", {"path": "ted_talks_iwslt", "name": "en-fa"}, "en", "fa"),
    ]
    
    total_pairs = 0
    total_bytes = 0
    
    for src_name, load_args, lang1, lang2 in parallel_sources:
        try:
            log.info(f"Loading parallel: {src_name}")
            ds = load_dataset(streaming=True, split="train", **load_args)
            batch = []
            pairs = 0
            
            for example in ds:
                # OPUS format typically has translation dict
                trans = example.get("translation", example)
                text_1 = trans.get(lang1, "")
                text_2 = trans.get(lang2, "")
                
                if text_1 and text_2 and len(text_2) > 10:
                    rec = {
                        "text_fa": text_2,
                        "text_en": text_1 if lang1 == "en" else "",
                        "text_ar": text_1 if lang1 == "ar" else "",
                        "source": src_name,
                        "type": "parallel",
                        "lang_pair": f"{lang1}-{lang2}",
                    }
                    batch.append(rec)
                    pairs += 1
                    total_bytes += len(text_2.encode('utf-8'))
            
            if batch:
                save_jsonl(batch, DATA_DIR / "parallel" / f"{src_name}_{lang1}_{lang2}.jsonl")
                total_pairs += pairs
                log.info(f"{src_name}: {pairs} pairs saved")
        except Exception as e:
            log.warning(f"Parallel {src_name} failed: {e}")
    
    # Persian-Arabic parallel from Telegram (HuggingFace)
    try:
        log.info("Loading Persian-Arabic Telegram pairs...")
        ds = load_dataset("AbdulazizAlshamsi/persian_arabic_pairs_telegram", split="train", streaming=True)
        batch = []
        pairs = 0
        for example in ds:
            text_fa = example.get("persian", example.get("fa", ""))
            text_ar = example.get("arabic", example.get("ar", ""))
            if text_fa and text_ar:
                rec = {
                    "text_fa": text_fa,
                    "text_ar": text_ar,
                    "source": "persian_arabic_telegram",
                    "type": "parallel",
                    "lang_pair": "ar-fa",
                }
                batch.append(rec)
                pairs += 1
                total_bytes += len(text_fa.encode('utf-8'))
        if batch:
            save_jsonl(batch, DATA_DIR / "parallel" / "persian_arabic_telegram.jsonl")
            total_pairs += pairs
            log.info(f"Persian-Arabic Telegram: {pairs} pairs saved")
    except Exception as e:
        log.warning(f"Persian-Arabic Telegram failed: {e}")
    
    MANIFEST["sources"]["parallel_en_fa"] = {"pairs": total_pairs, "bytes": total_bytes, "quality": "high"}
    log.info(f"Parallel DONE: {total_pairs} total pairs, {total_bytes/1e9:.2f} GB")

def collect_twitter():
    """Collect Persian Twitter datasets."""
    log.info("=== Collecting Twitter/Social Media ===")
    from datasets import load_dataset
    
    twitter_datasets = [
        "moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis",
    ]
    
    total_docs = 0
    total_bytes = 0
    
    for ds_name in twitter_datasets:
        try:
            log.info(f"Loading {ds_name}")
            ds = load_dataset(ds_name, split="train", streaming=True)
            batch = []
            for example in ds:
                text = example.get("text", example.get("tweet", ""))
                if text and len(text) > 20:
                    rec = {"text": text, "source": "twitter", "lang": "fa", "dataset": ds_name}
                    batch.append(rec)
                    total_docs += 1
                    total_bytes += len(text.encode('utf-8'))
            if batch:
                safe_name = ds_name.replace("/", "_")
                save_jsonl(batch, DATA_DIR / "twitter" / f"{safe_name}.jsonl")
                log.info(f"{ds_name}: {len(batch)} tweets")
        except Exception as e:
            log.warning(f"{ds_name} failed: {e}")
    
    MANIFEST["sources"]["twitter"] = {"docs": total_docs, "bytes": total_bytes, "quality": "medium"}

def collect_telegram():
    """Collect Persian Telegram datasets."""
    log.info("=== Collecting Telegram Data ===")
    from datasets import load_dataset
    
    telegram_datasets = [
        "mshojaei77/PersianTelegramChannels",
        "mshojaei77/alpaca_persian_telegram",
    ]
    
    total_docs = 0
    total_bytes = 0
    
    for ds_name in telegram_datasets:
        try:
            log.info(f"Loading {ds_name}")
            ds = load_dataset(ds_name, split="train", streaming=True)
            batch = []
            for example in ds:
                text = example.get("text", example.get("message", example.get("content", "")))
                if text and basic_quality_filter(text):
                    rec = {"text": text, "source": "telegram", "lang": "fa", "dataset": ds_name}
                    batch.append(rec)
                    total_docs += 1
                    total_bytes += len(text.encode('utf-8'))
            if batch:
                safe_name = ds_name.replace("/", "_")
                save_jsonl(batch, DATA_DIR / "telegram" / f"{safe_name}.jsonl")
                log.info(f"{ds_name}: {len(batch)} messages")
        except Exception as e:
            log.warning(f"{ds_name} failed: {e}")
    
    MANIFEST["sources"]["telegram"] = {"docs": total_docs, "bytes": total_bytes, "quality": "varies"}

def collect_news():
    """Collect Persian news corpora."""
    log.info("=== Collecting News Corpora ===")
    from datasets import load_dataset
    
    news_datasets = [
        "RohanAiLab/persian_daily_news",
        "RohanAiLab/persian_news_dataset",
    ]
    
    total_docs = 0
    total_bytes = 0
    
    for ds_name in news_datasets:
        try:
            log.info(f"Loading {ds_name}")
            ds = load_dataset(ds_name, split="train", streaming=True)
            batch = []
            file_idx = 0
            for example in ds:
                text = example.get("text", example.get("content", example.get("article", "")))
                title = example.get("title", example.get("headline", ""))
                if title and text:
                    full_text = f"{title}\n\n{text}"
                elif text:
                    full_text = text
                else:
                    continue
                
                if basic_quality_filter(full_text):
                    rec = {"text": full_text, "source": "news", "lang": "fa", "dataset": ds_name}
                    batch.append(rec)
                    total_docs += 1
                    total_bytes += len(full_text.encode('utf-8'))
                
                if len(batch) >= 10000:
                    safe_name = ds_name.replace("/", "_")
                    save_jsonl(batch, DATA_DIR / "news" / f"{safe_name}_{file_idx:04d}.jsonl")
                    batch = []
                    file_idx += 1
            
            if batch:
                safe_name = ds_name.replace("/", "_")
                save_jsonl(batch, DATA_DIR / "news" / f"{safe_name}_{file_idx:04d}.jsonl")
            log.info(f"{ds_name}: {total_docs} articles so far")
        except Exception as e:
            log.warning(f"{ds_name} failed: {e}")
    
    MANIFEST["sources"]["news"] = {"docs": total_docs, "bytes": total_bytes, "quality": "high"}

def collect_misc_corpora():
    """Collect miscellaneous Persian corpora from HuggingFace."""
    log.info("=== Collecting Misc Corpora ===")
    from datasets import load_dataset
    
    misc_datasets = [
        "ysn-rfd/Fibonacci-Pre_Train-Persian-Corpus-Raw-Texts-Dataset",
        "mshojaei77/PersianCorpus_merged",
        "ali619/corpus-dataset-normalized-for-persian-farsi",
        "mshojaei77/persian-document-corpus",
        "Deep-Research-Team/Pre-Training-Persian-Corpus-Raw-Texts-Dataset",
    ]
    
    total_docs = 0
    total_bytes = 0
    
    for ds_name in misc_datasets:
        try:
            log.info(f"Loading {ds_name}")
            ds = load_dataset(ds_name, split="train", streaming=True)
            batch = []
            file_idx = 0
            doc_count = 0
            for example in ds:
                text = example.get("text", example.get("content", ""))
                if basic_quality_filter(text):
                    rec = {"text": text, "source": "misc_corpus", "lang": "fa", "dataset": ds_name}
                    batch.append(rec)
                    doc_count += 1
                    total_bytes += len(text.encode('utf-8'))
                
                if len(batch) >= 10000:
                    safe_name = ds_name.replace("/", "_")
                    save_jsonl(batch, DATA_DIR / "misc" / f"{safe_name}_{file_idx:04d}.jsonl")
                    batch = []
                    file_idx += 1
            
            if batch:
                safe_name = ds_name.replace("/", "_")
                save_jsonl(batch, DATA_DIR / "misc" / f"{safe_name}_{file_idx:04d}.jsonl")
            
            total_docs += doc_count
            log.info(f"{ds_name}: {doc_count} docs")
        except Exception as e:
            log.warning(f"{ds_name} failed: {e}")
    
    MANIFEST["sources"]["misc_corpus"] = {"docs": total_docs, "bytes": total_bytes, "quality": "medium-high"}

def collect_literature():
    """Collect Persian literature datasets."""
    log.info("=== Collecting Literature ===")
    from datasets import load_dataset
    
    lit_datasets = [
        "RohanAiLab/persian_blog",  # 400k blog posts
    ]
    
    total_docs = 0
    total_bytes = 0
    
    for ds_name in lit_datasets:
        try:
            log.info(f"Loading {ds_name}")
            ds = load_dataset(ds_name, split="train", streaming=True)
            batch = []
            file_idx = 0
            for example in ds:
                text = example.get("text", example.get("content", ""))
                if basic_quality_filter(text):
                    rec = {"text": text, "source": "literature", "lang": "fa", "dataset": ds_name}
                    batch.append(rec)
                    total_docs += 1
                    total_bytes += len(text.encode('utf-8'))
                
                if len(batch) >= 10000:
                    safe_name = ds_name.replace("/", "_")
                    save_jsonl(batch, DATA_DIR / "literature" / f"{safe_name}_{file_idx:04d}.jsonl")
                    batch = []
                    file_idx += 1
            
            if batch:
                safe_name = ds_name.replace("/", "_")
                save_jsonl(batch, DATA_DIR / "literature" / f"{safe_name}_{file_idx:04d}.jsonl")
            log.info(f"{ds_name}: {total_docs} docs")
        except Exception as e:
            log.warning(f"{ds_name} failed: {e}")
    
    MANIFEST["sources"]["literature"] = {"docs": total_docs, "bytes": total_bytes, "quality": "high"}

def compute_final_stats():
    """Compute final statistics and save manifest."""
    total_docs = 0
    total_bytes = 0
    
    for src, stats in MANIFEST["sources"].items():
        total_docs += stats.get("docs", 0) + stats.get("pairs", 0)
        total_bytes += stats.get("bytes", 0)
    
    MANIFEST["total_documents"] = total_docs
    MANIFEST["total_bytes"] = total_bytes
    # Rough estimate: 1 token ≈ 3-4 bytes for Persian (UTF-8)
    MANIFEST["estimated_tokens"] = int(total_bytes / 3.5)
    
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(MANIFEST, f, indent=2, ensure_ascii=False)
    
    log.info(f"\n{'='*60}")
    log.info(f"COLLECTION COMPLETE")
    log.info(f"Total documents: {total_docs:,}")
    log.info(f"Total bytes: {total_bytes:,} ({total_bytes/1e9:.2f} GB)")
    log.info(f"Estimated tokens: {MANIFEST['estimated_tokens']:,} ({MANIFEST['estimated_tokens']/1e9:.2f}B)")
    log.info(f"Manifest saved to: {manifest_path}")
    log.info(f"{'='*60}")

def main():
    log.info("Starting Farsi data collection pipeline")
    setup()
    
    # Priority order as specified
    # 1. Biggest sources first
    collect_mc4()
    collect_cc100()
    
    # 2. High quality
    collect_wikipedia()
    collect_oscar()
    
    # 3. Critical parallel data
    collect_parallel()
    
    # 4-5. Social media
    collect_twitter()
    collect_telegram()
    
    # 6-7. News and literature
    collect_news()
    collect_literature()
    
    # 8. Misc corpora
    collect_misc_corpora()
    
    # Final stats
    compute_final_stats()

if __name__ == "__main__":
    main()
