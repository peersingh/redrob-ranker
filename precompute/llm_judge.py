"""
Phase 5 — LLM judgment pass (Groq, async + token-bucket rate limiter)

ROOT CAUSE of slow serial: Groq API latency = 6-8s per call.
Serial wastes all that wait time. Async fills it with parallel calls.
BUT raw async burst → rate limit → backoff → slow again.

FIX: Token-bucket rate limiter serializes REQUEST LAUNCH at 30 RPM
while letting 8 workers execute concurrently. This way:
  - 8 requests in-flight simultaneously (each taking ~6-8s)
  - New request allowed every 2s (30 RPM cap respected)
  - Throughput: 30 RPM = 5x faster than serial's 6.6 RPM
  - ETA 4000 candidates: ~135 min at 30 RPM
"""

import os, json, asyncio, time, argparse
import pandas as pd
from pathlib import Path
from groq import AsyncGroq

API_KEY   = os.environ.get("GROQ_API_KEY", "")
MODEL     = "llama-3.1-8b-instant"
WORKERS   = 8          # concurrent in-flight requests
RPM_LIMIT = 28         # slightly under 30 to be safe
TOP_N     = 4000

DATA_PATH   = "data/candidates.jsonl"
SCORES_PATH = "precompute/cache/rule_scores.parquet"
FUNNEL_PATH = "precompute/cache/funnel_shortlist.json"
OUTPUT_PATH = "precompute/cache/llm_judgments.jsonl"

SYSTEM = """Senior AI Engineer ranker. Evaluate this candidate for a role requiring:
- Production embeddings/vector-DB/hybrid-search shipped to real users
- Ranking/recommendation/search systems at product companies
- 5-9 YOE (4-5 at product co, not consulting), eval frameworks (NDCG/MRR/MAP)

Tiers: A=strong match, B=possible, C=weak, D=disqualified
Tier A companies: google/meta/amazon/microsoft/apple/sarvam ai/krutrim/niramai/haptik/yellow.ai/genpact ai/observe.ai/mad street den/wysa/verloop.io
Tier B companies: swiggy/zomato/razorpay/flipkart/cred/freshworks/phonepe/paytm/zoho/inmobi/dream11
Tier C (consulting, penalize if only career): tcs/infosys/wipro/accenture/capgemini/cognizant/hcl/mphasis
Tier D (filler/neutral): pied piper/hooli/dunder mifflin/stark industries/initech/acme/wayne

Disqualify (D): pure academic research only, AI exp=only LangChain <12mo, no prod code 18+mo as senior, 100% consulting career.
Honeypot signals: expert skill + duration_months=0, career_months >> yoe*12+24, role months > company age.

Respond ONLY with valid JSON (no markdown):
{"relevance_tier":"A (Strong Match)|B (Possible Match)|C (Weak Match)|D (Disqualified)","matched_requirements":["..."],"missing_violated_requirements":["..."],"honeypot_impossibility_flag":false,"honeypot_reason":"","draft_justification":"1-2 sentences citing company/skill names and one weakness."}"""


def compact(c: dict) -> str:
    sig = c.get("redrob_signals", {})
    return json.dumps({
        "id": c.get("candidate_id"),
        "title": c.get("profile", {}).get("current_title"),
        "co": c.get("profile", {}).get("current_company"),
        "yoe": c.get("profile", {}).get("years_of_experience"),
        "loc": c.get("profile", {}).get("location"),
        "summary": (c.get("profile", {}).get("summary") or "")[:150],
        "career": [
            {"co": r.get("company"), "title": r.get("title"),
             "months": r.get("duration_months"), "start": r.get("start_date"),
             "desc": (r.get("description") or "")[:100]}
            for r in c.get("career_history", [])
        ],
        "skills": [
            {"n": s.get("name"), "lvl": s.get("proficiency"), "mo": s.get("duration_months")}
            for s in c.get("skills", [])
        ],
        "sig": {
            "open": sig.get("open_to_work_flag"),
            "last": sig.get("last_active_date"),
            "rrr": sig.get("recruiter_response_rate"),
            "np": sig.get("notice_period_days"),
        },
    }, separators=(",", ":"))


class TokenBucket:
    """Allows at most `rate_per_min` request launches per minute.
    CORRECT: lock held only to reserve a slot, released BEFORE sleeping.
    Workers sleep concurrently, achieving true parallel throughput."""
    def __init__(self, rate_per_min: float):
        self.interval = 60.0 / rate_per_min
        self._next_slot = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            # Reserve the next available slot (lock released after this block)
            self._next_slot = max(self._next_slot, now) + self.interval
            wake_at = self._next_slot
        # Sleep WITHOUT holding the lock — all workers sleep concurrently
        wait = wake_at - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)


async def call_api(client: AsyncGroq, text: str, retries: int = 6) -> dict:
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": text},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=300,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err or "rate limit" in err:
                wait = min(4 * 2 ** attempt, 60)
                await asyncio.sleep(wait)
            else:
                await asyncio.sleep(min(2 ** attempt, 20))
    return {"error": "max retries exceeded"}


async def main(limit=None, top_n=TOP_N):
    if not API_KEY:
        raise ValueError("Set GROQ_API_KEY")

    client = AsyncGroq(api_key=API_KEY)

    # 1. Build target list (funnel ∩ top_n by rule_score)
    df = pd.read_parquet(SCORES_PATH)
    with open(FUNNEL_PATH) as f:
        funnel_ids = set(json.load(f)["shortlist"])
    df_f = df[df["candidate_id"].isin(funnel_ids)].sort_values("rule_score", ascending=False)
    target_ids = list(df_f["candidate_id"].head(top_n))
    if limit:
        target_ids = target_ids[:limit]

    # 2. Checkpoint
    completed: set = set()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    r = json.loads(line)
                    if not r.get("judgment", {}).get("error"):
                        completed.add(r["candidate_id"])
                except Exception:
                    pass

    remaining = [cid for cid in target_ids if cid not in completed]
    print(f"[*] Model  : {MODEL}  Workers={WORKERS}  Rate={RPM_LIMIT} RPM")
    print(f"[*] Target : {len(target_ids)} | Done: {len(completed)} | Remaining: {len(remaining)}")
    if not remaining:
        print("[+] All done!")
        return

    # 3. Load candidate data
    remaining_set = set(remaining)
    candidates: dict = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                c = json.loads(line)
                cid = c.get("candidate_id")
                if cid in remaining_set:
                    candidates[cid] = compact(c)
            except Exception:
                pass

    eta_min = len(remaining) / RPM_LIMIT
    print(f"[*] Loaded {len(candidates)} records | ETA ~{eta_min:.0f} min at {RPM_LIMIT} RPM")
    print(f"[*] Start: {time.strftime('%H:%M:%S UTC')}")
    print()

    # 4. Async worker pool with token-bucket rate limiter
    bucket = TokenBucket(RPM_LIMIT)
    work_queue: asyncio.Queue = asyncio.Queue()
    for cid in remaining:
        await work_queue.put(cid)

    out_lock = asyncio.Lock()
    done_count = 0
    t_start = time.time()

    async def worker():
        nonlocal done_count
        while True:
            try:
                cid = work_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if cid not in candidates:
                work_queue.task_done()
                continue

            # Acquire rate limit token before firing
            await bucket.acquire()

            judgment = await call_api(client, candidates[cid])

            async with out_lock:
                with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"candidate_id": cid, "judgment": judgment}) + "\n")
                done_count += 1
                total = len(completed) + done_count
                pct = total / len(target_ids) * 100
                elapsed = time.time() - t_start
                rate = done_count / elapsed * 60 if elapsed > 0 else 0
                tier = judgment.get("relevance_tier", judgment.get("error", "?"))[:15]
                hp   = judgment.get("honeypot_impossibility_flag", False)
                if done_count % 50 == 0 or done_count <= 5 or hp:
                    eta_r = (len(remaining) - done_count) / (rate / 60) if rate > 0 else 0
                    print(f"  [{total}/{len(target_ids)} {pct:.1f}%] {cid} | {tier} hp={hp} | {rate:.1f} RPM | ETA {eta_r/60:.1f}h")

            work_queue.task_done()

    # Launch worker pool
    workers = [asyncio.create_task(worker()) for _ in range(WORKERS)]
    await asyncio.gather(*workers)

    elapsed = time.time() - t_start
    final_rate = done_count / (elapsed / 60) if elapsed > 0 else 0
    print(f"\n[+] Done! {done_count} judged in {elapsed/60:.1f} min ({final_rate:.1f} RPM avg)")
    print(f"    Total valid in cache: {len(completed) + done_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, top_n=args.top_n))
