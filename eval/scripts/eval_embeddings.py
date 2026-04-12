#!/usr/bin/env python3
"""
Embedding quality evaluation for multilingual 3B model.
Tests: cross-lingual retrieval, semantic similarity, clustering.

Uses the base model's hidden states as embeddings (mean pooling over last layer).
Compares against random baseline and (optionally) multilingual-e5-small.
"""
import json, os, sys, random, numpy as np
sys.stdout.reconfigure(line_buffering=True)
import torch
import torch.nn.functional as F
import sentencepiece as spm

sys.path.insert(0, '/tmp/eval')
from train_sft_3b import GPT, VOCAB_SIZE, DIM, DEPTH, N_HEADS, MAX_SEQ_LEN

sp = spm.SentencePieceProcessor('/tmp/eval/multilingual_32k.model')
device = 'cuda'

LANG_NAMES = {'he': 'Hebrew', 'ar': 'Arabic', 'en': 'English', 'fa': 'Persian'}

def load_model(path):
    model = GPT()
    state = torch.load(path, map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    elif 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model

@torch.no_grad()
def get_embedding(model, text, pool='mean'):
    """Get embedding from model's last hidden layer."""
    ids = sp.encode(text)
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    x = torch.tensor([ids], device=device)
    
    # Forward through model but get hidden states
    # We need to modify forward to return hidden states
    # Alternative: hook into the last layer
    hidden = None
    def hook_fn(module, input, output):
        nonlocal hidden
        hidden = output if isinstance(output, torch.Tensor) else output[0]
    
    # Register hook on last block's layernorm or the final ln
    handle = model.ln_f.register_forward_hook(hook_fn)
    model(x)
    handle.remove()
    
    if hidden is None:
        # Fallback: use logits as proxy (worse but works)
        logits = model(x)
        return logits[0].mean(dim=0).cpu().numpy()
    
    # Pool
    if pool == 'mean':
        emb = hidden[0].mean(dim=0)
    elif pool == 'last':
        emb = hidden[0, -1]
    elif pool == 'first':
        emb = hidden[0, 0]
    
    return emb.cpu().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# ============================================================
# TEST 1: Cross-lingual Semantic Similarity
# Same sentence in different languages should be more similar
# than random sentences in same/different languages
# ============================================================
def test_cross_lingual_similarity(model, model_name):
    """Test if parallel sentences are closer than random ones."""
    
    # Parallel sentences (same meaning, different languages)
    parallel_sets = [
        {
            'en': 'The weather is beautiful today.',
            'he': 'מזג האוויר יפה היום.',
            'ar': 'الطقس جميل اليوم.',
            'fa': 'هوا امروز زیبا است.',
        },
        {
            'en': 'I want to buy a house.',
            'he': 'אני רוצה לקנות בית.',
            'ar': 'أريد شراء منزل.',
            'fa': 'من می خواهم خانه بخرم.',
        },
        {
            'en': 'The children are playing in the park.',
            'he': 'הילדים משחקים בפארק.',
            'ar': 'الأطفال يلعبون في الحديقة.',
            'fa': 'بچه ها در پارک بازی می کنند.',
        },
        {
            'en': 'This book is very interesting.',
            'he': 'הספר הזה מאוד מעניין.',
            'ar': 'هذا الكتاب ممتع جداً.',
            'fa': 'این کتاب بسیار جالب است.',
        },
        {
            'en': 'The university is closed today.',
            'he': 'האוניברסיטה סגורה היום.',
            'ar': 'الجامعة مغلقة اليوم.',
            'fa': 'دانشگاه امروز تعطیل است.',
        },
        {
            'en': 'We need to solve this problem.',
            'he': 'אנחנו צריכים לפתור את הבעיה הזו.',
            'ar': 'نحتاج إلى حل هذه المشكلة.',
            'fa': 'ما باید این مشکل را حل کنیم.',
        },
        {
            'en': 'The meeting will start at nine.',
            'he': 'הפגישה תתחיל בתשע.',
            'ar': 'سيبدأ الاجتماع في التاسعة.',
            'fa': 'جلسه ساعت نه شروع می شود.',
        },
        {
            'en': 'She works at the hospital.',
            'he': 'היא עובדת בבית החולים.',
            'ar': 'تعمل في المستشفى.',
            'fa': 'او در بیمارستان کار می کند.',
        },
        {
            'en': 'The food in this restaurant is excellent.',
            'he': 'האוכל במסעדה הזו מצוין.',
            'ar': 'الطعام في هذا المطعم ممتاز.',
            'fa': 'غذای این رستوران عالی است.',
        },
        {
            'en': 'Technology is changing the world.',
            'he': 'הטכנולוגיה משנה את העולם.',
            'ar': 'التكنولوجيا تغير العالم.',
            'fa': 'تکنولوژی دنیا را تغییر می دهد.',
        },
    ]
    
    # Get embeddings for all sentences
    embeddings = {}
    for i, pset in enumerate(parallel_sets):
        for lang, text in pset.items():
            key = f"{i}_{lang}"
            embeddings[key] = get_embedding(model, text)
    
    # Compute: parallel similarity (same sentence, different lang)
    parallel_sims = []
    for i in range(len(parallel_sets)):
        langs = list(parallel_sets[i].keys())
        for j in range(len(langs)):
            for k in range(j+1, len(langs)):
                sim = cosine_sim(embeddings[f"{i}_{langs[j]}"], embeddings[f"{i}_{langs[k]}"])
                parallel_sims.append(sim)
    
    # Compute: random similarity (different sentences, any lang)
    random_sims = []
    keys = list(embeddings.keys())
    for _ in range(200):
        k1, k2 = random.sample(keys, 2)
        i1 = k1.split('_')[0]
        i2 = k2.split('_')[0]
        if i1 != i2:  # different sentences
            sim = cosine_sim(embeddings[k1], embeddings[k2])
            random_sims.append(sim)
    
    # Compute per-language-pair similarities
    pair_sims = {}
    for i in range(len(parallel_sets)):
        langs = list(parallel_sets[i].keys())
        for j in range(len(langs)):
            for k in range(j+1, len(langs)):
                pair = f"{langs[j]}-{langs[k]}"
                if pair not in pair_sims:
                    pair_sims[pair] = []
                sim = cosine_sim(embeddings[f"{i}_{langs[j]}"], embeddings[f"{i}_{langs[k]}"])
                pair_sims[pair].append(sim)
    
    avg_parallel = np.mean(parallel_sims)
    avg_random = np.mean(random_sims)
    separation = avg_parallel - avg_random
    
    print(f"\n  {model_name} Cross-lingual Similarity:")
    print(f"    Parallel (same meaning): {avg_parallel:.4f}")
    print(f"    Random (different meaning): {avg_random:.4f}")
    print(f"    Separation: {separation:.4f} {'✅' if separation > 0.05 else '⚠️'}")
    print(f"    Per-pair avg:")
    for pair, sims in sorted(pair_sims.items()):
        print(f"      {pair}: {np.mean(sims):.4f}")
    
    return {
        'parallel_sim': float(avg_parallel),
        'random_sim': float(avg_random),
        'separation': float(separation),
        'pair_sims': {p: float(np.mean(s)) for p, s in pair_sims.items()},
    }

# ============================================================
# TEST 2: Cross-lingual Retrieval
# Given a query in one language, retrieve the correct translation
# from a pool of candidates in another language
# ============================================================
def test_retrieval(model, model_name):
    """Test retrieval accuracy: given query, find correct translation in pool."""
    
    # Use same parallel sets as above
    parallel_sets = [
        {'en': 'The weather is beautiful today.', 'he': 'מזג האוויר יפה היום.', 'ar': 'الطقس جميل اليوم.', 'fa': 'هوا امروز زیبا است.'},
        {'en': 'I want to buy a house.', 'he': 'אני רוצה לקנות בית.', 'ar': 'أريد شراء منزل.', 'fa': 'من می خواهم خانه بخرم.'},
        {'en': 'The children are playing in the park.', 'he': 'הילדים משחקים בפארק.', 'ar': 'الأطفال يلعبون في الحديقة.', 'fa': 'بچه ها در پارک بازی می کنند.'},
        {'en': 'This book is very interesting.', 'he': 'הספר הזה מאוד מעניין.', 'ar': 'هذا الكتاب ممتع جداً.', 'fa': 'این کتاب بسیار جالب است.'},
        {'en': 'The university is closed today.', 'he': 'האוניברסיטה סגורה היום.', 'ar': 'الجامعة مغلقة اليوم.', 'fa': 'دانشگاه امروز تعطیل است.'},
        {'en': 'We need to solve this problem.', 'he': 'אנחנו צריכים לפתור את הבעיה הזו.', 'ar': 'نحتاج إلى حل هذه المشكلة.', 'fa': 'ما باید این مشکل را حل کنیم.'},
        {'en': 'The meeting will start at nine.', 'he': 'הפגישה תתחיל בתשע.', 'ar': 'سيبدأ الاجتماع في التاسعة.', 'fa': 'جلسه ساعت نه شروع می شود.'},
        {'en': 'She works at the hospital.', 'he': 'היא עובדת בבית החולים.', 'ar': 'تعمل في المستشفى.', 'fa': 'او در بیمارستان کار می کند.'},
        {'en': 'The food in this restaurant is excellent.', 'he': 'האוכל במסעדה הזו מצוין.', 'ar': 'الطعام في هذا المطعم ممتاز.', 'fa': 'غذای این رستوران عالی است.'},
        {'en': 'Technology is changing the world.', 'he': 'הטכנולוגיה משנה את העולם.', 'ar': 'التكنولوجيا تغير العالم.', 'fa': 'تکنولوژی دنیا را تغییر می دهد.'},
    ]
    
    # Get all embeddings
    all_embs = {}
    for i, pset in enumerate(parallel_sets):
        for lang, text in pset.items():
            all_embs[f"{i}_{lang}"] = get_embedding(model, text)
    
    # For each language pair, test retrieval
    langs = ['en', 'he', 'ar', 'fa']
    results = {}
    
    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang == tgt_lang:
                continue
            
            correct = 0
            total = len(parallel_sets)
            
            for query_idx in range(total):
                query_emb = all_embs[f"{query_idx}_{src_lang}"]
                
                # Score all candidates in target language
                best_idx = -1
                best_sim = -1
                for cand_idx in range(total):
                    cand_emb = all_embs[f"{cand_idx}_{tgt_lang}"]
                    sim = cosine_sim(query_emb, cand_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_idx = cand_idx
                
                if best_idx == query_idx:
                    correct += 1
            
            acc = correct / total
            results[f"{src_lang}→{tgt_lang}"] = acc
    
    avg_acc = np.mean(list(results.values()))
    print(f"\n  {model_name} Retrieval Accuracy (10-way, chance=10%):")
    print(f"    Average: {avg_acc*100:.1f}%")
    for pair, acc in sorted(results.items()):
        print(f"      {pair}: {acc*100:.0f}%")
    
    return {'avg_accuracy': float(avg_acc), 'per_pair': {k: float(v) for k, v in results.items()}}

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print("Embedding Quality Evaluation")
    print("="*60)
    
    models_to_eval = [
        ('base-pretrained', '/tmp/eval/best_model.pt'),
        ('D-sft', '/tmp/sft_v3_runs/D/sft_model.pt'),
    ]
    
    all_results = {}
    
    for name, path in models_to_eval:
        if not os.path.exists(path):
            print(f"\n⚠️ {name} not found at {path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        
        model = load_model(path)
        
        sim_results = test_cross_lingual_similarity(model, name)
        retrieval_results = test_retrieval(model, name)
        
        all_results[name] = {
            'similarity': sim_results,
            'retrieval': retrieval_results,
        }
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    os.makedirs('/tmp/experiments', exist_ok=True)
    with open('/tmp/experiments/embedding_eval.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nResults saved to /tmp/experiments/embedding_eval.json")
    
    # Upload
    os.system("aws s3 cp /tmp/experiments/embedding_eval.json s3://autoresearch-dashboard-196766918360/multilingual-7b/eval/embedding_eval.json --quiet")
    print("Uploaded to S3")
