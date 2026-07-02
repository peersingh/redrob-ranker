import json
import os
from datetime import datetime, timezone
from collections import defaultdict

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def mine_corpus(input_path, output_path):
    print(f"[*] Commencing corpus mining on {input_path}...")

    company_data = defaultdict(lambda: {"earliest_date": None, "sizes": set(), "industries": set(), "headcount": 0})

    total_candidates = 0
    experience_distribution = []
    github_scores = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
                total_candidates += 1

                profile = candidate.get("profile", {})
                yoe = profile.get("years_of_experience")
                if yoe is not None:
                    experience_distribution.append(float(yoe))

                signals = candidate.get("redrob_signals", {})
                gh_score = signals.get("github_activity_score")
                if gh_score is not None and gh_score != -1:
                    github_scores.append(float(gh_score))

                history = candidate.get("career_history", [])
                for role in history:
                    company_raw = role.get("company")
                    if not company_raw:
                        continue

                    # Normalize company name (case/whitespace) so lookups match downstream
                    company_name = company_raw.strip().lower()
                    start_dt = role.get("start_date")
                    comp_size = role.get("company_size")
                    industry = role.get("industry")

                    company_data[company_name]["headcount"] += 1

                    parsed_start = parse_date(start_dt)
                    if parsed_start:
                        current_earliest = company_data[company_name]["earliest_date"]
                        if not current_earliest or parsed_start < current_earliest:
                            company_data[company_name]["earliest_date"] = parsed_start

                    if comp_size:
                        company_data[company_name]["sizes"].add(comp_size)
                    if industry:
                        company_data[company_name]["industries"].add(industry)

            except json.JSONDecodeError:
                continue

    if total_candidates == 0:
        print("[!] Error: No valid candidate records processed.")
        return

    experience_distribution.sort()
    github_scores.sort()

    median_yoe = experience_distribution[len(experience_distribution) // 2] if experience_distribution else 0
    median_gh = github_scores[len(github_scores) // 2] if github_scores else 0

    fact_sheet = {
        "metadata": {
            "total_pool_candidates": total_candidates,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "global_median_years_experience": median_yoe,
            "global_median_github_score": median_gh
        },
        "companies": {}
    }

    for comp, metrics in company_data.items():
        earliest_str = metrics["earliest_date"].strftime("%Y-%m-%d") if metrics["earliest_date"] else "unknown"
        fact_sheet["companies"][comp] = {
            "earliest_known_start": earliest_str,
            "headcount": metrics["headcount"],
            "observed_sizes": list(metrics["sizes"]),
            "primary_industries": list(metrics["industries"])
        }

    with open(output_path, 'w', encoding='utf-8') as out_f:
        json.dump(fact_sheet, out_f, indent=2)

    print(f"[+] Global Fact Sheet successfully serialized to {output_path}")
    print(f"[+] {total_candidates} candidates processed, {len(company_data)} distinct companies found.")

if __name__ == "__main__":
    DATA_PATH = "../data/candidates.jsonl"
    OUTPUT_PATH = "precompute/cache/global_fact_sheet.json"
    if os.path.exists(DATA_PATH):
        mine_corpus(DATA_PATH, OUTPUT_PATH)
    else:
        print(f"[!] Target file '{DATA_PATH}' not found.")
