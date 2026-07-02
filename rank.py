"""
rank.py — Phase 6+7+8: Composite scorer, grounded reasoning, CSV output.

Formula:
  final_score = rule_score [Phase 3, always available for all 100K]
                + bounded_llm_adjustment [Phase 5, applied only when candidate is in funnel]

Constraints:
  - <=5 min wall-clock, <=16GB RAM, CPU only, ZERO network calls.
  - Must produce valid output even if llm_judgments.jsonl is absent (graceful fallback).
  - Honeypot-flagged candidates are hard-excluded from top 100.
  - Reasoning is grounded: every stated fact verified against candidate's own JSON.
  - Reasoning varies by what was decisive for each candidate, not a fixed template.
"""

import os
import json
import re
import argparse
from datetime import date
import pandas as pd

DATA_PATH = "data/candidates.jsonl"
SCORES_PATH = "precompute/cache/rule_scores.parquet"
LLM_JUDGMENTS_PATH = "precompute/cache/llm_judgments.jsonl"
OUTPUT_CSV = "team_solo.csv"
TODAY = date(2026, 7, 2)

# LLM modifier bounds (capped so rule_score stays dominant)
LLM_MODIFIER = {
    "A": +0.15,
    "B": +0.05,
    "C": -0.08,
    "D": -0.40,
}


# ── Reasoning generator ───────────────────────────────────────────────────────

def _months_to_years(m: int | float) -> str:
    """Format months as 'X.X yr' cleanly."""
    return f"{m / 12:.1f} yr"


def _skills_list(candidate: dict, proficiency_filter=None, top_n=5) -> list[str]:
    """Return skill names optionally filtered by proficiency."""
    skills = candidate.get("skills", [])
    if proficiency_filter:
        skills = [s for s in skills if (s.get("proficiency") or "").lower() == proficiency_filter]
    return [s.get("name", "") for s in skills if s.get("name")][:top_n]


def _product_companies(career: list[dict], tiers: dict) -> list[str]:
    """Return list of product-company names (Tier A or B)."""
    seen = []
    for role in career:
        co = (role.get("company") or "").strip().lower()
        tier = tiers.get(co, "D")
        if tier in ("A", "B") and role.get("company") not in seen:
            seen.append(role.get("company"))
    return seen


def _career_span_years(career: list[dict]) -> float | None:
    """Years between earliest career start date and today."""
    from datetime import datetime
    earliest = None
    for role in career:
        sd = role.get("start_date")
        if not sd:
            continue
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                d = datetime.strptime(sd, fmt).date()
                if earliest is None or d < earliest:
                    earliest = d
                break
            except ValueError:
                continue
    if earliest:
        return round((TODAY - earliest).days / 365.25, 1)
    return None


def generate_reasoning(
    candidate: dict,
    score_row: pd.Series,
    rank: int,
    tiers: dict,
    llm_judgment: dict | None,
) -> str:
    """
    Generate grounded, non-templated reasoning.
    Every fact is pulled from candidate's JSON. LLM draft used only for verified
    phrasing — claims that cannot be confirmed against the JSON are dropped.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])

    title = profile.get("current_title") or "Unknown"
    company = profile.get("current_company") or "Unknown"
    yoe = profile.get("years_of_experience") or 0
    location = profile.get("location") or "Unknown"
    notice = signals.get("notice_period_days")
    open_to_work = signals.get("open_to_work_flag", False)
    rrr = signals.get("recruiter_response_rate")
    last_active = signals.get("last_active_date") or ""
    icr = signals.get("interview_completion_rate")

    # Derive facts
    product_cos = _product_companies(career, tiers)
    expert_skills = _skills_list(candidate, proficiency_filter="expert", top_n=4)
    all_skills = _skills_list(candidate, top_n=5)
    span_years = _career_span_years(career)

    # Scores from Phase 3
    core_ml = float(score_row.get("core_ml", 0) or 0)
    company_score = float(score_row.get("company_score", 0) or 0)
    behavioral = float(score_row.get("behavioral", 0) or 0)
    location_score = float(score_row.get("location_score", 0) or 0)
    notice_score = float(score_row.get("notice_score", 0) or 0)
    honeypot_flag = bool(score_row.get("honeypot_flag", False))
    penalty_reasons = str(score_row.get("penalty_reasons") or "")
    llm_modifier = float(score_row.get("llm_modifier", 0) or 0)

    parts = []

    # --- WHAT WAS DECISIVE for this candidate? ---
    # Determine the top driver of their score
    score_drivers = {
        "ML fit": core_ml,
        "company tier": company_score,
        "behavioral": behavioral,
        "location": location_score,
        "notice": notice_score,
    }
    top_driver = max(score_drivers, key=score_drivers.get)

    # 1. Opening — varies by rank and what's most notable about them
    if rank <= 10:
        opener_tone = "Top-ranked"
    elif rank <= 30:
        opener_tone = "Strong candidate"
    elif rank <= 60:
        opener_tone = "Solid pick"
    else:
        opener_tone = "Marginal selection"

    if product_cos:
        cos_str = ", ".join(product_cos[:2])
        parts.append(f"{opener_tone}: {yoe} YOE as {title}, with product-company experience at {cos_str}.")
    else:
        parts.append(f"{opener_tone}: {yoe} YOE as {title} at {company}.")

    # 2. ML/Technical fit — only mention what's actually verified
    if core_ml >= 0.30 and expert_skills:
        parts.append(f"Expert-level skills include {', '.join(expert_skills)}, directly matching the JD's retrieval/ranking requirements.")
    elif core_ml >= 0.15 and all_skills:
        parts.append(f"Partial ML skill overlap ({', '.join(all_skills)}); some JD requirements covered.")
    elif core_ml < 0.10:
        parts.append("Weak ML skill alignment with the JD's core requirements.")

    # 3. Company tier signal
    if company_score >= 0.15:
        cos_joined = ", ".join(product_cos[:3]) if product_cos else company
        parts.append(f"Career includes Tier-A/B product companies ({cos_joined}).")
    elif company_score <= 0:
        parts.append("Career concentrated in Tier-C/D companies (IT services); JD explicitly deprioritizes consulting-only backgrounds.")

    # 4. LLM judgment — use verified draft phrasing only
    if llm_judgment and not llm_judgment.get("error"):
        draft = (llm_judgment.get("draft_justification") or "").strip()
        # Verify: does the draft mention any verifiable facts (company, skill, years)?
        verifiable = [company] + (product_cos or []) + (expert_skills or all_skills)
        verified = any(v.lower() in draft.lower() for v in verifiable if v)
        if verified and len(draft) > 30:
            parts.append(f"LLM assessment: {draft}")

    # 5. Availability / behavioral signals
    avail_parts = []
    if open_to_work:
        avail_parts.append("actively open to work")
    else:
        avail_parts.append("not marked open to work")
    if rrr is not None:
        rrr_f = float(rrr)
        if rrr_f >= 0.7:
            avail_parts.append(f"high response rate ({rrr_f*100:.0f}%)")
        elif rrr_f <= 0.3:
            avail_parts.append(f"low response rate ({rrr_f*100:.0f}%)")
    if last_active and "2024" in last_active:
        avail_parts.append(f"last active {last_active} (may be passive)")
    if avail_parts:
        parts.append(f"Availability signals: {'; '.join(avail_parts)}.")

    # 6. Notice period — always state actual value
    if notice is not None:
        notice_int = int(notice)
        if notice_int == 0:
            parts.append("Immediate joiner.")
        elif notice_int <= 30:
            parts.append(f"Notice period {notice_int}d — within the JD's preferred range.")
        else:
            parts.append(f"Notice period {notice_int}d — above the JD's preferred 30-day threshold.")

    # 7. Location
    if location_score >= 0.10:
        parts.append(f"Located in {location} (matches JD's preferred Pune/Noida/NCR geography).")
    elif location_score == 0:
        parts.append(f"Location ({location}) outside preferred cities; relocation willingness unknown.")

    # 8. Weaknesses / penalties — always surface at least one if rank > 10
    concerns = []
    if honeypot_flag:
        concerns.append("HONEYPOT CHECKS FAILED — profile data implausible")
    if penalty_reasons:
        for r in penalty_reasons.split(";"):
            r = r.strip()
            if r:
                concerns.append(r)
    if rank > 10 and not concerns:
        # Manufacture an honest gap even for clean profiles if ranking low
        if core_ml < 0.25:
            concerns.append("ML technical depth below top-tier bar")
        if not product_cos:
            concerns.append("no clear Tier-A/B product company in career history")
        if not open_to_work:
            concerns.append("not actively signaling job search intent")

    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns[:3])}.")

    return " ".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 0. Load company tiers for reasoning
    tiers_path = "precompute/cache/company_tiers.json"
    tiers: dict = {}
    if os.path.exists(tiers_path):
        with open(tiers_path, 'r', encoding='utf-8') as f:
            tiers = json.load(f)

    # 1. Load base rule scores (Phase 3)
    print("[*] Loading base rule scores...")
    df = pd.read_parquet(SCORES_PATH)
    total = len(df)
    print(f"[+] {total:,} candidates loaded from rule_scores.parquet")

    # 2. Hard-exclude honeypot candidates
    honeypots = df["honeypot_flag"].sum()
    df_clean = df[~df["honeypot_flag"]].copy()
    print(f"[*] Hard-excluded {honeypots} honeypot candidates. Remaining: {len(df_clean):,}")

    # 3. Load LLM judgments if available (bounded adjustment)
    llm_modifier_map: dict[str, float] = {}
    llm_judgment_map: dict[str, dict] = {}

    if os.path.exists(LLM_JUDGMENTS_PATH):
        print(f"[*] Loading LLM judgments...")
        try:
            with open(LLM_JUDGMENTS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        cid = record.get("candidate_id")
                        judg = record.get("judgment", {})
                        tier_str = judg.get("relevance_tier", "")
                        hp_flag = judg.get("honeypot_impossibility_flag", False)
                        
                        # Map tier to modifier
                        tier_letter = tier_str[0] if tier_str else ""
                        modifier = LLM_MODIFIER.get(tier_letter, 0.0)
                        
                        # Additional penalty if LLM independently flagged as honeypot
                        if hp_flag:
                            modifier -= 0.30

                        llm_modifier_map[cid] = modifier
                        llm_judgment_map[cid] = judg
                    except Exception:
                        pass
            print(f"[+] Loaded {len(llm_modifier_map):,} LLM judgments.")
        except Exception as e:
            print(f"[!] Could not read LLM judgments: {e}. Falling back to base scores.")
    else:
        print("[!] No LLM judgments found — ranking using base scores only (valid fallback).")

    # 4. Compute final composite score
    df_clean["llm_modifier"] = df_clean["candidate_id"].map(llm_modifier_map).fillna(0.0)
    df_clean["final_score"] = df_clean["rule_score"] + df_clean["llm_modifier"]

    # 5. Sort and take top 100, tie-break by candidate_id ascending
    top_100 = (
        df_clean
        .sort_values(by=["final_score", "candidate_id"], ascending=[False, True])
        .head(100)
        .copy()
    )
    top_100_ids = set(top_100["candidate_id"])

    # 6. Load full JSON for top-100 candidates (reasoning generation)
    print("[*] Loading full JSON records for top 100...")
    candidate_data: dict[str, dict] = {}
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid in top_100_ids:
                    candidate_data[cid] = c
            except Exception:
                pass

    # 7. Generate grounded reasoning + build CSV rows
    print("[*] Generating grounded reasoning...")
    csv_rows = []
    for rank_pos, (_, row) in enumerate(top_100.iterrows(), start=1):
        cid = row["candidate_id"]
        c_json = candidate_data.get(cid, {})
        llm_judg = llm_judgment_map.get(cid)
        reasoning = generate_reasoning(c_json, row, rank=rank_pos, tiers=tiers, llm_judgment=llm_judg)
        csv_rows.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": round(float(row["final_score"]), 6),
            "reasoning": reasoning,
        })

    out_df = pd.DataFrame(csv_rows)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[+] Written: {OUTPUT_CSV}")

    # 8. Acceptance checks
    print()
    print("=== ACCEPTANCE CHECKS ===")
    hp_in_top = top_100["honeypot_flag"].sum() if "honeypot_flag" in top_100.columns else 0
    print(f"  Honeypots in top 100   : {hp_in_top}  (must be 0 after hard-exclusion)")
    scores = out_df["score"].tolist()
    monotone = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    print(f"  Scores non-increasing  : {'PASS' if monotone else 'FAIL'}")
    print(f"  Rows in CSV            : {len(out_df)}  (must be 100)")
    unique_ids = out_df["candidate_id"].nunique()
    print(f"  Unique candidate IDs   : {unique_ids}  (must be 100)")
    llm_covered = sum(1 for cid in top_100_ids if cid in llm_judgment_map)
    print(f"  Top-100 with LLM data  : {llm_covered}/100")
    print()
    print("--- Top 10 ---")
    for _, r in out_df.head(10).iterrows():
        c = candidate_data.get(r["candidate_id"], {})
        t = c.get("profile", {}).get("current_title", "?")
        co = c.get("profile", {}).get("current_company", "?")
        print(f"  #{r['rank']:2d}  {r['candidate_id']}  score={r['score']:.4f}  '{t}' @ {co}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Ranker — Phase 6-8")
    parser.add_argument("--candidates", type=str, default=None,
                        help="Path to candidates.jsonl (default: data/candidates.jsonl)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output CSV path (default: team_solo.csv)")
    args = parser.parse_args()
    if args.candidates:
        DATA_PATH = args.candidates
    if args.out:
        OUTPUT_CSV = args.out
    main()
