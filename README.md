# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Team:** team_solo | **Author:** Lakavath Peer Singh  
**Submission file:** `team_solo.csv` | **Validated:** ✅ `Submission is valid.`

---

## Overview

A fully interpretable, multi-stage candidate ranking pipeline for the Redrob Hackathon. The system ranks 100,000 candidates for a Senior AI Engineer role, producing a top-100 CSV that:

- Runs in **< 2 minutes**, **CPU-only**, **zero network calls** during ranking
- Hard-excludes all **44 honeypot candidates** (0 in top-100)
- Generates **grounded, candidate-specific reasoning** (no templates, no hallucination)
- Achieves **100/100 LLM judgment coverage** for the final top-100

---

## Architecture

```
candidates.jsonl (100K)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ PRECOMPUTE (run once, results cached to disk)               │
│                                                             │
│  Phase 1: Corpus Miner                                      │
│    → global_fact_sheet.json (63 companies, pool medians)    │
│                                                             │
│  Phase 2: Company Tier Classifier                           │
│    → company_tiers.json (A=elite AI, B=unicorn,            │
│        C=consulting-disqualifier, D=filler)                 │
│                                                             │
│  Phase 3: Rule Scorer + Honeypot Ensemble                   │
│    → rule_scores.parquet (100K scored, 44 honeypots)        │
│                                                             │
│  Phase 4: Retrieval Funnel (BM25 + Dense + RRF)             │
│    → funnel_shortlist.json (top-4000 for LLM review)        │
│                                                             │
│  Phase 5: LLM Judge (Groq API, precompute only)             │
│    → llm_judgments.jsonl (549 judgments, top-100 covered)   │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ RANKING (rank.py — CPU, <2 min, zero network)               │
│                                                             │
│  Phase 6: Composite scorer                                  │
│    final_score = rule_score + bounded_llm_adjustment        │
│                                                             │
│  Phase 7: Hard exclusions + sort                            │
│    Honeypots hard-excluded; top-100 selected                │
│                                                             │
│  Phase 8: Grounded reasoning generation                     │
│    Per-candidate justification from actual JSON fields      │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
   team_solo.csv (100 rows, validated ✅)
```

---

## Scoring Formula

```
final_score = rule_score + llm_modifier

rule_score = (
    0.40 × ml_fit_score          # Production embeddings, vector DB, eval frameworks
  + 0.20 × company_tier_score    # Tier A-D based on JD-grounded company quality
  + 0.20 × behavioral_score      # 23 Redrob signals (recency, response rate, etc.)
  + 0.10 × location_score        # Pune/Noida/NCR preferred; outside India penalized
  + 0.05 × notice_score          # ≤30 days preferred; >90 days penalized
  + 0.05 × bonus_score           # GitHub activity, open-source, assessment scores
  - penalties                    # Consulting-only, CV-primary, pure-research, title-chaser
)

llm_modifier = {A: +0.15, B: +0.05, C: 0.0, D: -0.15}  # Bounded; rule_score stays dominant
```

---

## Honeypot Detection (4 checks, 44 flagged)

| Check | Description |
|---|---|
| **Expert-zero** | Skill listed as "expert" but `duration_months = 0` |
| **Timeline impossibility** | Career span months >> `years_of_experience × 12 + 24` |
| **Role predates company** | `role.duration_months > company_age_months` (uses corpus earliest-start date) |
| **YOE-career gap** | `career_span_years - stated_yoe > 3` years |

All 44 flagged candidates are **hard-excluded** before top-100 selection.

---

## Company Tier System

| Tier | Companies | Score modifier |
|---|---|---|
| **A** — Elite AI/Global tech | Google, Meta, Amazon, Microsoft, Apple, Sarvam AI, Krutrim, Genpact AI, Observe.ai, Mad Street Den, Niramai, Wysa, Verloop.io, Rephrase.ai, Haptik, Yellow.ai, Glance | +0.20 |
| **B** — Indian unicorns | Swiggy, Zomato, Razorpay, Flipkart, CRED, Freshworks, PhonePe, Paytm, InMobi, Dream11, Zoho, Nykaa | +0.10 |
| **C** — IT consulting (penalized if only experience) | TCS, Infosys, Wipro, Accenture, Capgemini, Cognizant, HCL, Mphasis | −0.20 |
| **D** — Neutral/filler | Pied Piper, Hooli, Stark Industries, Initech, Acme Corp… | 0.00 |

---

## Retrieval Funnel (Phase 4)

**RRF fusion (k=60)** of two retrieval signals:

1. **BM25 sparse** — TF-IDF keyword match against JD terms
2. **Dense semantic** — `all-MiniLM-L6-v2` embeddings with 3 positive archetypes:
   - *"Senior ML Engineer with production retrieval systems at product companies"*
   - *"Applied scientist ranking recommendation systems, shipped to real users"*
   - *"NLP engineer hybrid search vector databases evaluation NDCG"*
   
   And 1 negative archetype (consulting-only, research-only, keywords-no-production).

No title filtering — funnel is recall-optimized.

---

## LLM Judge (Phase 5 — Precompute Only)

- **Model:** `llama-3.1-8b-instant` via Groq API
- **Scope:** Top-549 candidates by rule_score from the 4000-candidate funnel
- **Output per candidate:** `relevance_tier` (A/B/C/D), matched/missing requirements, honeypot flag, draft justification
- **Zero LLM calls during `rank.py`** — all judgments cached in `llm_judgments.jsonl`
- **Top-100 coverage:** 100/100

---

## Reproduction

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1 — Precompute (run once, ~8 hours for full LLM pass)

```bash
# Phase 1: Corpus mining
python precompute/corpus_miner.py

# Phase 2: Company tiers
python precompute/company_tiers.py

# Phase 3: Rule scoring (scores all 100K candidates)
python precompute/rule_scorer.py

# Phase 4: Retrieval funnel (produces 4000-candidate shortlist)
python precompute/retrieval_funnel.py

# Phase 5: LLM judgment (Groq API key required, precompute only)
GROQ_API_KEY=your_key_here python precompute/llm_judge.py --top-n 4000
```

> All precompute outputs are already cached in `precompute/cache/` — you can skip to Step 2.

### Step 2 — Rank (< 2 minutes, CPU-only, zero network)

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./team_solo.csv
```

### Step 3 — Validate

```bash
python data/validate_submission.py team_solo.csv
# → Submission is valid.
```

---

## File Structure

```
redrob-ranker/
├── rank.py                          # Final ranker (Phase 6-8) — CPU, <2min, zero network
├── requirements.txt                 # All dependencies + versions
├── submission_metadata.yaml         # Submission metadata (team, compute, AI tools)
├── team_solo.csv                    # SUBMISSION FILE — top-100 ranked candidates
│
├── precompute/
│   ├── corpus_miner.py              # Phase 1: Extract company stats from 100K candidates
│   ├── company_tiers.py             # Phase 2: Classify companies A-D
│   ├── rule_scorer.py               # Phase 3: Score all 100K + honeypot detection
│   ├── retrieval_funnel.py          # Phase 4: BM25 + Dense + RRF → 4000 shortlist
│   ├── llm_judge.py                 # Phase 5: LLM judgment (Groq, precompute only)
│   └── cache/
│       ├── global_fact_sheet.json   # Phase 1 output: company stats, pool medians
│       ├── company_tiers.json       # Phase 2 output: tier classifications
│       ├── rule_scores.parquet      # Phase 3 output: 100K scored candidates
│       ├── funnel_shortlist.json    # Phase 4 output: 4000-candidate shortlist
│       └── llm_judgments.jsonl      # Phase 5 output: LLM judgment cache
│
└── data/
    ├── job_description.md           # The target JD
    ├── candidate_schema.json        # Candidate data schema reference
    ├── sample_candidates.json       # 10 sample candidates for testing
    ├── sample_submission.csv        # Format reference only (not a real ranking)
    ├── validate_submission.py       # Official validator
    └── submission_metadata_template.yaml
```

---

## Key Design Decisions

### Why rule-based scoring dominates (not LLM-first)?
The spec explicitly warns against "finding candidates whose skills section contains the most AI keywords." A pure embedding or LLM approach falls into exactly this trap. The rule scorer reads *career history*, *actual production deployment signals*, and *behavioral engagement* — not just skills lists.

### Why 40% weight on ML fit?
The JD has hard requirements (production embeddings, vector DB, eval frameworks). These are binary — either you've shipped them or you haven't. The rule scorer checks `career_history[].description` and `skills[].duration_months`, not just presence.

### Why the retrieval funnel before LLM?
100K × LLM = impossible within any budget. The RRF funnel reduces to 4000 high-recall candidates. The LLM then acts as a precision layer on top.

### Why bound the LLM modifier (±0.15)?
To prevent LLM hallucination from dominating. A Tier A candidate who was correctly identified by rules doesn't need a +0.15 boost to make top-100. The bound ensures rule quality stays the primary signal.

---

## Results

| Metric | Value |
|---|---|
| Honeypots in top-100 | **0** (44 flagged, all excluded) |
| LLM coverage (top-100) | **100/100** |
| Scores non-increasing | **PASS** |
| Runtime (rank.py) | **< 2 minutes** |
| Network calls (rank.py) | **0** |

**Top 5:**
| Rank | Candidate | Title | Company |
|---|---|---|---|
| 1 | CAND_0046525 | Senior ML Engineer | Genpact AI |
| 2 | CAND_0041669 | Recommendation Systems Engineer | CRED |
| 3 | CAND_0018499 | Senior ML Engineer | Zomato |
| 4 | CAND_0086022 | Senior Applied Scientist | Sarvam AI |
| 5 | CAND_0039754 | Senior Applied Scientist | Meta |

---

## AI Tools Declaration

- **Antigravity (Google DeepMind)** — AI pair-programming assistant used for system design, implementation, and debugging
- **Groq API (llama-3.1-8b-instant)** — Used exclusively in Phase 5 (precompute/llm_judge.py) for candidate evaluation; zero LLM calls in rank.py
