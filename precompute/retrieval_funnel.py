"""
Phase 4 — Retrieval Funnel
Reads : candidates.jsonl, job_description.md
Writes: precompute/cache/funnel_shortlist.json

Strategy:
  - Sparse leg  : BM25 over headline + summary + career descriptions + skill names
  - Dense leg   : all-MiniLM-L6-v2 with CONTRASTIVE archetype scoring
                  score = max_sim(doc, positive_archetypes) - sim(doc, negative_archetype)
                  (directly targets JD's stated trap: single-vector rewards vocabulary mirroring)
  - Fusion      : Reciprocal Rank Fusion (RRF, k=60)
  - Funnel size : 4000 (sized for token-budget check in Phase 5)
  - No title filter — real fits hide in generic-titled candidates (README Section 4 note)
"""

import json
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH      = "../data/candidates.jsonl"
JD_PATH        = "data/job_description.md"
OUTPUT_PATH    = "precompute/cache/funnel_shortlist.json"
MODEL_NAME     = "all-MiniLM-L6-v2"   # CPU-friendly, 22M params, 384-dim
FUNNEL_SIZE    = 4000
RRF_K          = 60

# ─── Archetype texts (contrastive — from JD Section 5) ───────────────────────

POSITIVE_ARCHETYPES = [
    # (a) The "shipper" archetype
    """
    Shipped end-to-end production ranking search recommendation system to real users at meaningful scale.
    Fast pragmatic product engineering build deploy iterate. Applied ML engineer at product company startup.
    Own the intelligence layer ranking retrieval matching systems. Ship working ranker in a week.
    Scrappy product-engineering attitude technical depth modern ML systems.
    """,

    # (b) The "pre-LLM-era ML engineer" archetype
    """
    Embeddings based retrieval sentence transformers BGE E5 word2vec BERT dense retrieval.
    Vector database Pinecone Weaviate Qdrant Milvus Elasticsearch FAISS hybrid search ANN index.
    BM25 information retrieval NDCG MRR MAP evaluation framework offline online A/B ranking metrics.
    Python PyTorch production ML system drift index refresh retrieval quality regression.
    Learning to rank LambdaMART fine-tuning LoRA PEFT recommendation system.
    """,

    # (c) JD "how to read between the lines" ideal candidate paragraph
    """
    6 to 8 years total experience 4 to 5 years applied ML AI roles product companies not pure services.
    Shipped at least one end-to-end ranking search recommendation system real users meaningful scale.
    Strong defensible opinions retrieval hybrid dense evaluation offline online LLM integration fine-tune prompt.
    Located willing to relocate Noida Pune Hyderabad Mumbai Delhi. Active job market open to work.
    Senior AI engineer applied scientist ML engineer NLP engineer recommendation systems engineer.
    """,
]

NEGATIVE_ARCHETYPE = """
LangChain tutorial OpenAI API demo proof of concept chatbot wrapper GPT demo framework enthusiast.
TCS Infosys Wipro Accenture Cognizant Capgemini consulting IT services outsourcing staff augmentation body shopping.
Computer vision object detection image classification YOLO CNN convolutional OpenCV image segmentation.
Speech recognition ASR text to speech TTS speaker diarization robotics autonomous LIDAR point cloud SLAM.
Pure research academic lab PhD researcher postdoc paper publication university no production deployment.
Senior Staff Principal title hopping company switching every year career optimization ladder climbing.
Marketing manager HR manager content writer business analyst non-technical role.
"""


# ─── Text building ────────────────────────────────────────────────────────────

def build_candidate_text(c: dict) -> str:
    """Concatenate all searchable fields per README Phase 4 spec."""
    parts = []
    profile = c.get("profile", {})
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")
    for role in c.get("career_history", []):
        parts.append(role.get("description", "") or "")
    for skill in c.get("skills", []):
        parts.append(skill.get("name", "") or "")
    return " ".join(p for p in parts if p).strip()


def tokenize(text: str):
    """Simple whitespace+punctuation tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ─── RRF fusion ───────────────────────────────────────────────────────────────

def rrf_score(rank: int, k: int = RRF_K) -> float:
    return 1.0 / (k + rank + 1)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> dict:
    """
    rankings: list of candidate_id lists, each ordered best→worst.
    Returns dict {candidate_id: rrf_score}.
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] += rrf_score(rank, k)
    return scores


# ─── Main ────────────────────────────────────────────────────────────────────

def run():
    # 1. Load JD
    print("[*] Loading job description...")
    with open(JD_PATH, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    # 2. Stream candidates, build corpus
    print("[*] Streaming candidates, building corpus...")
    t0 = time.time()
    candidate_ids = []
    corpus_texts  = []

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate_ids.append(c.get("candidate_id", ""))
            corpus_texts.append(build_candidate_text(c))
            if (i + 1) % 20000 == 0:
                print(f"  ... loaded {i+1:,} candidates")

    print(f"[+] Loaded {len(candidate_ids):,} candidates in {time.time()-t0:.1f}s")

    # ── SPARSE LEG: BM25 ──────────────────────────────────────────────────────
    print("[*] Building BM25 index...")
    t0 = time.time()
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[+] BM25 index built in {time.time()-t0:.1f}s")

    print("[*] BM25 query...")
    jd_tokens = tokenize(jd_text)
    bm25_scores = bm25.get_scores(jd_tokens)  # array of length N
    bm25_ranking = list(np.argsort(bm25_scores)[::-1])  # indices, best first
    bm25_id_ranking = [candidate_ids[i] for i in bm25_ranking]
    print(f"[+] BM25 top score: {bm25_scores[bm25_ranking[0]]:.3f}")

    # ── DENSE LEG: sentence-transformer with contrastive archetypes ───────────
    print(f"[*] Loading sentence-transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("[+] Model loaded.")

    # Encode archetypes
    print("[*] Encoding archetypes...")
    pos_embeddings = model.encode(
        POSITIVE_ARCHETYPES, normalize_embeddings=True, show_progress_bar=False
    )  # shape (3, 384)
    neg_embedding = model.encode(
        [NEGATIVE_ARCHETYPE], normalize_embeddings=True, show_progress_bar=False
    )[0]  # shape (384,)

    # Encode all candidates in batches
    print("[*] Encoding 100K candidate texts (batched, CPU)...")
    t0 = time.time()
    # Truncate each text to 512 chars to keep encoding fast and memory sane
    truncated_texts = [t[:512] if len(t) > 512 else t for t in corpus_texts]

    BATCH = 512
    all_embeddings = []
    for start in range(0, len(truncated_texts), BATCH):
        batch = truncated_texts[start:start + BATCH]
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs)
        if (start // BATCH + 1) % 20 == 0:
            done = min(start + BATCH, len(truncated_texts))
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(truncated_texts) - done) / rate if rate > 0 else 0
            print(f"  ... encoded {done:,}/{len(truncated_texts):,}  ({rate:.0f}/s, ~{remaining:.0f}s left)")

    doc_embeddings = np.vstack(all_embeddings)  # (N, 384)
    print(f"[+] All candidates encoded in {time.time()-t0:.1f}s  shape={doc_embeddings.shape}")

    # Contrastive scoring: max positive similarity - negative similarity
    print("[*] Computing contrastive similarity scores...")
    # Positive: max cosine sim across 3 positive archetypes
    pos_sims = doc_embeddings @ pos_embeddings.T   # (N, 3)
    max_pos_sim = pos_sims.max(axis=1)             # (N,)
    # Negative: cosine sim to negative archetype
    neg_sim = doc_embeddings @ neg_embedding       # (N,)
    # Contrastive score
    contrastive_scores = max_pos_sim - neg_sim     # (N,)

    dense_ranking = list(np.argsort(contrastive_scores)[::-1])
    dense_id_ranking = [candidate_ids[i] for i in dense_ranking]
    print(f"[+] Dense contrastive top score: {contrastive_scores[dense_ranking[0]]:.4f}")
    print(f"[+] Dense contrastive score range: [{contrastive_scores.min():.4f}, {contrastive_scores.max():.4f}]")

    # ── RRF FUSION ────────────────────────────────────────────────────────────
    print("[*] Fusing BM25 + dense via RRF...")
    rrf_scores = reciprocal_rank_fusion([bm25_id_ranking, dense_id_ranking], k=RRF_K)
    fused_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Take top FUNNEL_SIZE
    shortlist_ids = [cid for cid, _ in fused_ranking[:FUNNEL_SIZE]]
    print(f"[+] Funnel size: {len(shortlist_ids):,}")

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "funnel_size": len(shortlist_ids),
        "model": MODEL_NAME,
        "rrf_k": RRF_K,
        "shortlist": shortlist_ids,
        "rrf_scores": {cid: round(score, 8) for cid, score in fused_ranking[:FUNNEL_SIZE]},
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"[+] Funnel shortlist written to {OUTPUT_PATH}")

    # ── Acceptance test ───────────────────────────────────────────────────────
    print()
    print("=== ACCEPTANCE TEST: TOP 30 OF FUNNEL ===")
    print("(Checking: NOT all AI-titled; must include generic-titled candidates)")
    print()

    # Load candidate details for top 30
    top30_ids = set(shortlist_ids[:30])
    top30_lookup = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                if c.get("candidate_id") in top30_ids:
                    top30_lookup[c["candidate_id"]] = c
            except:
                continue

    ai_title_kw = {"ai", "ml", "machine learning", "data scientist", "nlp",
                   "deep learning", "applied scientist", "recommendation"}
    generic_count = 0
    for rank_pos, cid in enumerate(shortlist_ids[:30], 1):
        c = top30_lookup.get(cid, {})
        profile = c.get("profile", {})
        title = profile.get("current_title", "N/A")
        company = profile.get("current_company", "N/A")
        yoe = profile.get("years_of_experience", "N/A")
        loc = profile.get("location", "N/A")
        is_generic = not any(k in title.lower() for k in ai_title_kw)
        if is_generic:
            generic_count += 1
        marker = "  <-- generic title" if is_generic else ""
        print(f"  #{rank_pos:2d}  {cid}  '{title}' @ {company}  yoe={yoe}  {loc}{marker}")

    print()
    if generic_count >= 1:
        print(f"[PASS] {generic_count} generic-titled candidates in top 30 — funnel is NOT title-filtering")
    else:
        print("[WARN] All top 30 have AI-adjacent titles — may be over-fitting to vocabulary")


if __name__ == "__main__":
    run()
