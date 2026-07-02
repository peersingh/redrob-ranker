"""
Phase 3 — Rule-based Structural Scorer + Honeypot Ensemble
Reads: candidates.jsonl, company_tiers.json, global_fact_sheet.json
Writes: precompute/cache/rule_scores.parquet

Score weights (all traceable to JD Section 5):
  core_ml_score   0.40  — "things you absolutely need": embeddings, vector DB, eval frameworks
  company_tier    0.20  — "applied ML at product companies (not pure services)"
  behavioral      0.20  — "perfect-on-paper but inactive is not actually available"
  location        0.10  — "Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi NCR acceptable"
  notice          0.05  — "sub-30-day strongly preferred"
  bonus           0.05  — "nice to have" items (fine-tuning, L2R, open-source)

Penalties reduce the total score; they are traceable to JD disqualifier text.
"""

import json
import os
import re
from datetime import date, datetime
from collections import defaultdict

import pandas as pd

# ─── Constants ────────────────────────────────────────────────────────────────

TODAY = date(2026, 7, 2)

DATA_PATH      = "../data/candidates.jsonl"
TIERS_PATH     = "precompute/cache/company_tiers.json"
FACTSHEET_PATH = "precompute/cache/global_fact_sheet.json"
OUTPUT_PATH    = "precompute/cache/rule_scores.parquet"

# Score weights (sum = 1.0, traceable to JD)
W_CORE_ML    = 0.40
W_COMPANY    = 0.20
W_BEHAVIORAL = 0.20
W_LOCATION   = 0.10
W_NOTICE     = 0.05
W_BONUS      = 0.05

# ─── Keyword sets (all lowercase) ────────────────────────────────────────────

# "Things you absolutely need" — core positive signals
KW_EMBEDDINGS = {
    "embedding", "embeddings", "sentence-transformer", "sentence transformer",
    "word2vec", "bert", "bge", "e5", "semantic search", "dense retrieval",
    "openai embedding", "vector embedding", "ada-002", "text-embedding"
}
KW_VECTOR_DB = {
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "vector database", "vector store", "vector index", "hybrid search",
    "ann search", "approximate nearest neighbor", "hnsw"
}
KW_RANKING_SEARCH = {
    "ranking", "search", "retrieval", "recommendation", "recommender",
    "bm25", "tf-idf", "tfidf", "information retrieval", "learning to rank",
    "reranking", "re-ranking", "rrf", "reciprocal rank"
}
KW_EVAL_FRAMEWORK = {
    "ndcg", "mrr", "map@", "mean average precision", "precision@",
    "recall@", "evaluation framework", "offline eval", "online eval",
    "a/b test", "a/b testing", "ab test", "offline-to-online", "ranking eval"
}
KW_PRODUCTION = {
    "production", "deployed", "real users", "at scale", "serving",
    "inference", "latency", "throughput", "sla", "live system", "prod"
}
KW_PYTHON = {
    "python", "pytorch", "tensorflow", "numpy", "pandas", "sklearn",
    "scikit-learn", "transformers", "hugging face", "huggingface"
}

# Nice-to-have signals (bonus points)
KW_BONUS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "learning-to-rank", "lambdamart", "xgboost rank", "lightgbm rank",
    "open source", "github", "arxiv", "paper", "conference", "talk", "blog post"
}

# Negative / disqualifier signals
KW_CV_SPEECH_ROBOTICS = {
    "computer vision", "object detection", "image classification",
    "image segmentation", "yolo", "cnn", "convolutional", "opencv",
    "speech recognition", "asr", "text-to-speech", "tts", "speaker",
    "robotics", "autonomous", "lidar", "point cloud", "slam"
}
KW_RESEARCH_ONLY = {
    "research lab", "research intern", "phd researcher", "postdoc",
    "research scientist", "ai researcher", "ml researcher",
    "paper published", "academic research", "university research"
}
KW_FRAMEWORK_TOURIST = {
    "langchain tutorial", "langchain demo", "openai api",
    "chatgpt wrapper", "rag demo", "gpt-4 demo"
}
KW_CONSULTING_ROLE = {
    "it services", "outsourcing", "staff augmentation", "body shopping"
}

# Preferred locations (JD: Pune/Noida → best; Hyd/Pune/Mumbai/Delhi NCR → good)
BEST_LOCATIONS = {"pune", "noida"}
GOOD_LOCATIONS = {"hyderabad", "mumbai", "delhi", "gurugram", "gurgaon", "bengaluru", "bangalore"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def months_between(d1: date, d2: date) -> float:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def keyword_hit_count(text: str, keyword_set: set) -> int:
    """Count how many distinct keywords from the set appear in text."""
    count = 0
    for kw in keyword_set:
        if kw in text:
            count += 1
    return count


def build_full_text(candidate: dict) -> str:
    """Concatenate all searchable text fields, lowercased."""
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("headline", "") or "")
    parts.append(profile.get("summary", "") or "")
    for role in candidate.get("career_history", []):
        parts.append(role.get("description", "") or "")
        parts.append(role.get("title", "") or "")
    for skill in candidate.get("skills", []):
        parts.append(skill.get("name", "") or "")
    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", "") or "")
    return " ".join(parts).lower()


# ─── Honeypot checks ─────────────────────────────────────────────────────────

def honeypot_check_1(candidate: dict):
    """Expert skill with zero usage duration."""
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") == "expert" and skill.get("duration_months") == 0:
            return True, f"expert skill '{skill.get('name')}' has duration_months=0"
    return False, ""


def honeypot_check_2(candidate: dict):
    """Career total duration >> stated years_of_experience."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience")
    if yoe is None:
        return False, ""
    career_months = sum(
        r.get("duration_months", 0) or 0
        for r in candidate.get("career_history", [])
    )
    threshold = float(yoe) * 12 + 24
    if career_months > threshold:
        return True, (
            f"career total {career_months:.0f}mo > "
            f"stated_yoe({yoe}yr)*12+24 = {threshold:.0f}mo"
        )
    return False, ""


def honeypot_check_3(candidate: dict, company_earliest: dict):
    """Role tenure exceeds plausible company age (company earliest seen date)."""
    for role in candidate.get("career_history", []):
        company = (role.get("company") or "").strip().lower()
        if company not in company_earliest:
            continue
        earliest_str = company_earliest[company]
        if earliest_str == "unknown":
            continue
        earliest_dt = parse_date(earliest_str)
        if not earliest_dt:
            continue
        company_age_months = months_between(earliest_dt, TODAY)
        role_duration = role.get("duration_months") or 0
        if role_duration > company_age_months:
            return True, (
                f"role at '{company}' lasted {role_duration}mo but company "
                f"earliest seen {earliest_str} → only {company_age_months:.0f}mo old"
            )
    return False, ""


def honeypot_check_4(candidate: dict):
    """Stated experience much shorter than career span."""
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience")
    if yoe is None:
        return False, ""
    earliest_start = None
    for role in candidate.get("career_history", []):
        sd = parse_date(role.get("start_date"))
        if sd and (earliest_start is None or sd < earliest_start):
            earliest_start = sd
    if not earliest_start:
        return False, ""
    career_span_years = (TODAY - earliest_start).days / 365.25
    gap = career_span_years - float(yoe)
    if gap > 3:
        return True, (
            f"career starts {earliest_start} → span {career_span_years:.1f}yr "
            f"but stated yoe={yoe} → gap={gap:.1f}yr > 3yr"
        )
    return False, ""


def honeypot_check_5(candidate: dict):
    """Expert proficiency + very low skill assessment score (<30). Verify count before using."""
    signals = candidate.get("redrob_signals", {})
    assessment = signals.get("skill_assessment_scores") or {}
    for skill in candidate.get("skills", []):
        if skill.get("proficiency") != "expert":
            continue
        skill_name = (skill.get("name") or "").strip()
        # Try exact match, then lowercase match
        score = assessment.get(skill_name)
        if score is None:
            score = assessment.get(skill_name.lower())
        if score is not None and score < 30:
            return True, (
                f"expert in '{skill_name}' but assessment score={score:.0f} < 30"
            )
    return False, ""


# ─── Structural scoring ───────────────────────────────────────────────────────

def score_core_ml(candidate: dict, full_text: str) -> float:
    """Score 0–1 for core ML/AI fit to JD requirements."""
    score = 0.0

    # 1. Embeddings/retrieval experience (JD: "things you absolutely need")
    emb_hits = keyword_hit_count(full_text, KW_EMBEDDINGS)
    score += min(emb_hits * 0.08, 0.20)

    # 2. Vector DB / hybrid search infra (JD: "things you absolutely need")
    vdb_hits = keyword_hit_count(full_text, KW_VECTOR_DB)
    score += min(vdb_hits * 0.08, 0.20)

    # 3. Ranking / search / recommendation shipped to real users
    rank_hits = keyword_hit_count(full_text, KW_RANKING_SEARCH)
    score += min(rank_hits * 0.05, 0.15)

    # 4. Evaluation framework experience (NDCG, MRR, MAP, A/B)
    eval_hits = keyword_hit_count(full_text, KW_EVAL_FRAMEWORK)
    score += min(eval_hits * 0.07, 0.20)

    # 5. Production deployment signal
    prod_hits = keyword_hit_count(full_text, KW_PRODUCTION)
    score += min(prod_hits * 0.03, 0.10)

    # 6. Python / ML stack
    py_hits = keyword_hit_count(full_text, KW_PYTHON)
    score += min(py_hits * 0.02, 0.10)

    # 7. Skill-level signals: advanced/expert AI skills with endorsements
    ai_skill_score = 0.0
    for skill in candidate.get("skills", []):
        sk_name = (skill.get("name") or "").lower()
        prof = skill.get("proficiency", "")
        endorse = skill.get("endorsements", 0) or 0
        sk_text = sk_name
        hits = keyword_hit_count(sk_text, KW_EMBEDDINGS | KW_VECTOR_DB | KW_RANKING_SEARCH | KW_EVAL_FRAMEWORK)
        if hits > 0:
            level_mult = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.4, "beginner": 0.1}.get(prof, 0.2)
            endorse_mult = min(1.0 + endorse * 0.02, 1.5)
            ai_skill_score += level_mult * endorse_mult * 0.03
    score += min(ai_skill_score, 0.20)

    # Cap at 1.0 before penalties
    return min(score, 1.0)


def score_company_tier(candidate: dict, tier_map: dict) -> float:
    """Score 0–1 for company tier composition of career history.
    JD: 'applied ML at product companies (not pure services)'; 4-5 yrs at product co."""
    history = candidate.get("career_history", [])
    if not history:
        return 0.0

    total_months = 0
    weighted_months = 0.0
    for role in history:
        company = (role.get("company") or "").strip().lower()
        dur = role.get("duration_months") or 0
        tier_entry = tier_map.get(company, {"modifier": 0.0})
        modifier = tier_entry.get("modifier", 0.0)
        # Map modifier to 0-1 contribution: -0.4→0, 0→0.5, 0.2→0.7, 0.4→1.0
        tier_score = (modifier + 0.4) / 0.8  # linear map [-0.4, 0.4] → [0, 1]
        weighted_months += dur * tier_score
        total_months += dur

    if total_months == 0:
        return 0.3  # neutral default
    return min(weighted_months / total_months, 1.0)


def score_behavioral(candidate: dict) -> float:
    """Score 0–1 for platform engagement / availability signals.
    JD: 'perfect-on-paper but inactive is not actually available — down-weight accordingly'."""
    signals = candidate.get("redrob_signals", {})
    score = 0.5  # start neutral

    # open_to_work: strong signal
    if signals.get("open_to_work_flag"):
        score += 0.15
    else:
        score -= 0.15

    # last_active_date recency
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_ago = (TODAY - last_active).days
        if days_ago <= 30:
            score += 0.15
        elif days_ago <= 90:
            score += 0.05
        elif days_ago <= 180:
            score -= 0.05
        else:
            score -= 0.20

    # recruiter_response_rate (0-1)
    rrr = signals.get("recruiter_response_rate")
    if rrr is not None:
        score += (float(rrr) - 0.5) * 0.20  # ±0.10

    # interview_completion_rate (0-1)
    icr = signals.get("interview_completion_rate")
    if icr is not None:
        score += (float(icr) - 0.5) * 0.10  # ±0.05

    # applications submitted recently = active job seeker
    apps = signals.get("applications_submitted_30d") or 0
    if apps > 0:
        score += min(apps * 0.01, 0.05)

    return max(0.0, min(score, 1.0))


def score_location(candidate: dict) -> float:
    """Score 0–1 for location fit.
    JD: 'Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi NCR acceptable; willing to relocate.'"""
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()

    # Check best locations first
    for loc in BEST_LOCATIONS:
        if loc in location:
            return 1.0

    # Good locations
    for loc in GOOD_LOCATIONS:
        if loc in location:
            return 0.70

    # Willing to relocate
    if signals.get("willing_to_relocate"):
        return 0.55

    # India, but not preferred city
    country = (profile.get("country") or "").lower()
    if "india" in country or "in" == country:
        return 0.30

    return 0.10


def score_notice(candidate: dict) -> float:
    """Score 0–1 for notice period fit.
    JD: 'sub-30-day strongly preferred (can buy out up to 30 days); 30+ day bar is higher'."""
    signals = candidate.get("redrob_signals", {})
    np_days = signals.get("notice_period_days")
    if np_days is None:
        return 0.40  # unknown = neutral-ish
    np_days = int(np_days)
    if np_days == 0:
        return 1.0
    elif np_days <= 30:
        return 0.85
    elif np_days <= 60:
        return 0.55
    elif np_days <= 90:
        return 0.35
    else:
        return 0.15


def score_bonus(candidate: dict, full_text: str) -> float:
    """Score 0–1 for 'nice to have' signals (fine-tuning, L2R, open-source)."""
    score = 0.0
    bonus_hits = keyword_hit_count(full_text, KW_BONUS)
    score += min(bonus_hits * 0.08, 0.50)

    signals = candidate.get("redrob_signals", {})
    gh = signals.get("github_activity_score", -1)
    if gh is not None and gh != -1:
        score += min(float(gh) / 100.0 * 0.40, 0.40)

    return min(score, 1.0)


# ─── Disqualifier penalties ───────────────────────────────────────────────────

def compute_penalties(candidate: dict, full_text: str, tier_map: dict) -> tuple:
    """Returns (total_penalty: float, reasons: list[str]).
    Penalties are negative, applied after weighted sum."""
    penalties = []
    reasons = []

    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # 1. 100% Tier-C consulting career (JD: "people who have only worked at consulting firms")
    if history:
        all_c = all(
            tier_map.get((r.get("company") or "").strip().lower(), {}).get("tier", "").startswith("C")
            for r in history
            if r.get("company")
        )
        if all_c and len(history) >= 2:
            penalties.append(-0.30)
            reasons.append("entire career at Tier-C consulting firms")

    # 2. Primary expertise CV/speech/robotics without NLP/IR
    cv_hits = keyword_hit_count(full_text, KW_CV_SPEECH_ROBOTICS)
    ir_hits = keyword_hit_count(full_text, KW_EMBEDDINGS | KW_VECTOR_DB | KW_RANKING_SEARCH)
    if cv_hits >= 3 and ir_hits == 0:
        penalties.append(-0.25)
        reasons.append(f"primary expertise CV/speech/robotics ({cv_hits} hits), zero NLP/IR signals")

    # 3. Pure research only — no production signals (JD: "pure research... no production deployment")
    research_hits = keyword_hit_count(full_text, KW_RESEARCH_ONLY)
    prod_hits_pen = keyword_hit_count(full_text, KW_PRODUCTION)
    if research_hits >= 2 and prod_hits_pen == 0:
        penalties.append(-0.20)
        reasons.append(f"pure research signals ({research_hits}), no production deployment signals")

    # 4. Title-chaser pattern: ≥3 job changes each <20 months AND title escalation
    if len(history) >= 3:
        sorted_roles = sorted(
            [r for r in history if r.get("start_date")],
            key=lambda r: r.get("start_date", "")
        )
        short_hops = sum(1 for r in sorted_roles if (r.get("duration_months") or 0) < 20)
        senior_titles = {"senior", "staff", "principal", "lead", "director"}
        title_progression = [
            any(t in (r.get("title") or "").lower() for t in senior_titles)
            for r in sorted_roles
        ]
        if short_hops >= 3 and sum(title_progression) >= 2:
            penalties.append(-0.15)
            reasons.append(f"title-chaser pattern: {short_hops} roles <20mo with title escalation")

    # 5. Framework tourist / LangChain-only signal
    tourist_hits = keyword_hit_count(full_text, KW_FRAMEWORK_TOURIST)
    pre_llm_hits = keyword_hit_count(full_text, KW_EMBEDDINGS | KW_VECTOR_DB | KW_RANKING_SEARCH)
    if tourist_hits >= 2 and pre_llm_hits == 0:
        penalties.append(-0.15)
        reasons.append(f"framework tourist signal ({tourist_hits} hits), no pre-LLM ML experience")

    # 6. Zero GitHub AND no open-source/paper signals for 5+ yrs of career
    signals = candidate.get("redrob_signals", {})
    gh = signals.get("github_activity_score", -1)
    yoe = profile.get("years_of_experience") or 0
    if gh == -1 and yoe >= 5 and keyword_hit_count(full_text, KW_BONUS) == 0:
        penalties.append(-0.10)
        reasons.append("5+ yrs experience, no GitHub, no open-source/paper signals")

    total_penalty = sum(penalties)
    return total_penalty, reasons


# ─── Main scoring function ────────────────────────────────────────────────────

def score_candidate(candidate: dict, tier_map: dict, company_earliest: dict):
    full_text = build_full_text(candidate)

    # Component scores
    core_ml   = score_core_ml(candidate, full_text)
    company   = score_company_tier(candidate, tier_map)
    behavioral= score_behavioral(candidate)
    location  = score_location(candidate)
    notice    = score_notice(candidate)
    bonus     = score_bonus(candidate, full_text)

    # Weighted sum
    raw = (
        W_CORE_ML    * core_ml +
        W_COMPANY    * company +
        W_BEHAVIORAL * behavioral +
        W_LOCATION   * location +
        W_NOTICE     * notice +
        W_BONUS      * bonus
    )

    # Penalties
    penalty, penalty_reasons = compute_penalties(candidate, full_text, tier_map)
    final_score = max(0.0, raw + penalty)

    # Honeypot ensemble
    hp_flags = []
    hp_reasons = []
    for check_fn, args in [
        (honeypot_check_1, (candidate,)),
        (honeypot_check_2, (candidate,)),
        (honeypot_check_3, (candidate, company_earliest)),
        (honeypot_check_4, (candidate,)),
    ]:
        flagged, reason = check_fn(*args)
        if flagged:
            hp_flags.append(True)
            hp_reasons.append(reason)

    honeypot_flag = len(hp_flags) > 0
    honeypot_reason = "; ".join(hp_reasons) if hp_reasons else ""

    return {
        "candidate_id":   candidate.get("candidate_id", ""),
        "rule_score":     round(final_score, 6),
        "raw_score":      round(raw, 6),
        "penalty":        round(penalty, 6),
        "core_ml":        round(core_ml, 4),
        "company_tier":   round(company, 4),
        "behavioral":     round(behavioral, 4),
        "location":       round(location, 4),
        "notice":         round(notice, 4),
        "bonus":          round(bonus, 4),
        "honeypot_flag":  honeypot_flag,
        "honeypot_reason": honeypot_reason,
        "penalty_reasons": "; ".join(penalty_reasons),
    }


# ─── Entry point ─────────────────────────────────────────────────────────────

def run():
    print("[*] Loading company tiers...")
    with open(TIERS_PATH, 'r', encoding='utf-8') as f:
        tier_map = json.load(f)

    print("[*] Loading global fact sheet...")
    with open(FACTSHEET_PATH, 'r', encoding='utf-8') as f:
        fact_sheet = json.load(f)
    company_earliest = {
        co: data["earliest_known_start"]
        for co, data in fact_sheet["companies"].items()
    }

    # ── First pass: check honeypot_5 count, decide whether to include ─────────
    print("[*] Pre-scan: counting honeypot check 5 (expert + low assessment score)...")
    hp5_candidates = []
    total_scanned = 0
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                total_scanned += 1
                flagged, reason = honeypot_check_5(c)
                if flagged:
                    hp5_candidates.append(c.get("candidate_id", ""))
            except json.JSONDecodeError:
                continue

    hp5_count = len(hp5_candidates)
    use_hp5 = hp5_count <= 100
    print(f"[*] Honeypot check 5 count: {hp5_count}  ->  {'INCLUDED' if use_hp5 else 'REJECTED (>100, likely noise)'}")

    # ── Main scoring pass ─────────────────────────────────────────────────────
    print("[*] Scoring all candidates...")
    rows = []
    honeypot_ids = set()

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = score_candidate(candidate, tier_map, company_earliest)

            # Optionally add check 5 result
            if use_hp5:
                flagged5, reason5 = honeypot_check_5(candidate)
                if flagged5 and not row["honeypot_flag"]:
                    row["honeypot_flag"] = True
                    row["honeypot_reason"] = (row["honeypot_reason"] + "; " + reason5).lstrip("; ")

            if row["honeypot_flag"]:
                honeypot_ids.add(row["candidate_id"])

            rows.append(row)

            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1:,} candidates scored")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\n[+] Scored {len(df):,} candidates -> {OUTPUT_PATH}")
    print(f"[+] Total honeypot flags: {df['honeypot_flag'].sum()}")
    print()

    # ── Acceptance test output ────────────────────────────────────────────────
    print("=== ACCEPTANCE TEST ===")
    print(f"Total honeypot flags (all checks): {df['honeypot_flag'].sum()}")
    print()

    # Top 10 scorers
    top10 = df.nlargest(10, "rule_score")
    print("TOP 10 SCORERS:")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        cand_lookup = {}
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                if c.get("candidate_id") in set(top10["candidate_id"]):
                    cand_lookup[c["candidate_id"]] = c
            except:
                continue

    for _, row in top10.iterrows():
        cid = row["candidate_id"]
        c = cand_lookup.get(cid, {})
        title = c.get("profile", {}).get("current_title", "N/A")
        company = c.get("profile", {}).get("current_company", "N/A")
        yoe = c.get("profile", {}).get("years_of_experience", "N/A")
        loc = c.get("profile", {}).get("location", "N/A")
        print(f"  {cid}  score={row['rule_score']:.4f}  '{title}' @ {company}  yoe={yoe}  loc={loc}")

    print()
    print("5 FLAGGED HONEYPOT CANDIDATES:")
    flagged_df = df[df["honeypot_flag"]].head(5)
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        flagged_lookup = {}
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                if c.get("candidate_id") in set(flagged_df["candidate_id"]):
                    flagged_lookup[c["candidate_id"]] = c
            except:
                continue

    for _, row in flagged_df.iterrows():
        cid = row["candidate_id"]
        c = flagged_lookup.get(cid, {})
        title = c.get("profile", {}).get("current_title", "N/A")
        print(f"  {cid}  '{title}'  REASON: {row['honeypot_reason']}")


if __name__ == "__main__":
    run()
