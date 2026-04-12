"""
Train a SentencePiece BPE tokenizer for multilingual (HE/AR/FA/EN) use.

Configuration:
- Vocab size: 32,768
- Algorithm: BPE via SentencePiece
- Language sampling: Equal 25% per language
- Character coverage: 99.95%
- Byte fallback: enabled
"""

import sentencepiece as spm
import os
import random
from pathlib import Path


def sample_balanced_corpus(data_dir: str, output_path: str, 
                           target_mb_per_lang: int = 50,
                           languages: list = None):
    """
    Create a balanced training corpus by sampling equal amounts from each language.
    
    Args:
        data_dir: Directory containing language subdirectories (he/, ar/, fa/, en/)
        output_path: Path to write the combined training corpus
        target_mb_per_lang: Target size in MB per language (default: 50MB)
        languages: List of language codes (default: he, ar, fa, en)
    """
    if languages is None:
        languages = ['he', 'ar', 'fa', 'en']
    
    target_bytes = target_mb_per_lang * 1024 * 1024
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for lang in languages:
            lang_dir = Path(data_dir) / lang
            if not lang_dir.exists():
                print(f"Warning: {lang_dir} not found, skipping")
                continue
            
            # Collect all text files for this language
            files = list(lang_dir.glob('*.txt'))
            random.shuffle(files)
            
            bytes_written = 0
            for f in files:
                if bytes_written >= target_bytes:
                    break
                with open(f, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        line = line.strip()
                        if len(line) > 10:  # Skip very short lines
                            out_f.write(line + '\n')
                            bytes_written += len(line.encode('utf-8'))
                            if bytes_written >= target_bytes:
                                break
            
            print(f"  [{lang}] Sampled {bytes_written / 1024 / 1024:.1f} MB")
    
    print(f"Combined corpus written to {output_path}")


def train_tokenizer(corpus_path: str, model_prefix: str = 'multilingual_32k',
                    vocab_size: int = 32768):
    """
    Train a SentencePiece BPE tokenizer.
    
    Args:
        corpus_path: Path to the balanced training corpus
        model_prefix: Output model file prefix
        vocab_size: Target vocabulary size
    """
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name='nfkc',
        max_sentence_length=4096,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count(),
        # Special tokens
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=['<|user|>', '<|assistant|>'],
    )
    print(f"Tokenizer trained: {model_prefix}.model ({vocab_size} tokens)")


def compute_fertility(model_path: str, test_texts: dict) -> dict:
    """
    Compute fertility (tokens per word) for each language.
    
    Args:
        model_path: Path to trained .model file
        test_texts: Dict mapping language code to list of test sentences
    
    Returns:
        Dict with fertility rates per language
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    fertility = {}
    for lang, texts in test_texts.items():
        total_tokens = 0
        total_words = 0
        for text in texts:
            tokens = sp.encode(text, out_type=int)
            words = text.split()
            total_tokens += len(tokens)
            total_words += len(words)
        fertility[lang] = round(total_tokens / max(total_words, 1), 2)
    
    return fertility


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train multilingual BPE tokenizer')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with language subdirectories')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for model files')
    parser.add_argument('--vocab-size', type=int, default=32768)
    parser.add_argument('--mb-per-lang', type=int, default=50,
                        help='MB to sample per language')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    corpus_path = os.path.join(args.output_dir, 'tokenizer_train_corpus.txt')
    model_prefix = os.path.join(args.output_dir, 'multilingual_32k')
    
    print("Step 1: Sampling balanced corpus...")
    sample_balanced_corpus(args.data_dir, corpus_path, args.mb_per_lang)
    
    print("\nStep 2: Training tokenizer...")
    train_tokenizer(corpus_path, model_prefix, args.vocab_size)
    
    print("\nDone! Files created:")
    print(f"  {model_prefix}.model")
    print(f"  {model_prefix}.vocab")
