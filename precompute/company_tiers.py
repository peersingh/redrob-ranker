import json
import os

# ── Tier definitions (exact from README) ──────────────────────────────────────
# All company names are normalized: strip + lower (matching corpus_miner.py)

TIER_D = {
    "modifier": 0.0,
    "label": "D - filler/neutral",
    "companies": {
        "pied piper", "hooli", "dunder mifflin", "stark industries",
        "initech", "acme corp", "wayne enterprises", "globex inc"
    }
}

TIER_C = {
    "modifier": -0.4,
    "label": "C - JD-named disqualifier (services/consulting)",
    "companies": {
        "tcs", "infosys", "wipro",
        "accenture", "capgemini", "cognizant",
        "tech mahindra", "hcl", "mphasis", "mindtree"
    }
}

TIER_B = {
    "modifier": 0.2,
    "label": "B - product/unicorn company",
    "companies": {
        "swiggy", "razorpay", "zomato", "flipkart", "cred",
        "freshworks", "vedantu", "ola", "nykaa", "phonepe",
        "meesho", "upgrad", "unacademy", "byju's", "paytm",
        "inmobi", "dream11", "pharmeasy", "zoho", "policybazaar"
    }
}

TIER_A = {
    "modifier": 0.4,
    "label": "A - rare high-signal (applied-AI / global tech)",
    "companies": {
        # Global tech giants
        "google", "meta", "microsoft", "amazon", "netflix",
        "apple", "salesforce", "linkedin", "uber", "adobe",
        # Indian applied-AI / NLP / voice / vision startups
        "sarvam ai", "krutrim", "observe.ai", "mad street den",
        "niramai", "wysa", "verloop.io", "rephrase.ai", "aganitha",
        "saarthi.ai", "haptik", "yellow.ai", "glance", "locobuzz",
        "genpact ai"
    }
}

ALL_TIERS = [TIER_A, TIER_B, TIER_C, TIER_D]


def build_tier_map(fact_sheet_path, output_path):
    print(f"[*] Loading fact sheet from {fact_sheet_path}...")
    with open(fact_sheet_path, 'r', encoding='utf-8') as f:
        fact_sheet = json.load(f)

    companies_in_pool = set(fact_sheet["companies"].keys())
    print(f"[*] {len(companies_in_pool)} companies in pool to classify.")

    tier_map = {}
    unassigned = []

    for company in companies_in_pool:
        assigned = False
        for tier in ALL_TIERS:
            if company in tier["companies"]:
                tier_map[company] = {
                    "tier": tier["label"],
                    "modifier": tier["modifier"]
                }
                assigned = True
                break
        if not assigned:
            unassigned.append(company)

    if unassigned:
        print(f"[!] WARNING — {len(unassigned)} companies NOT assigned to any tier:")
        for c in sorted(unassigned):
            print(f"    - {c!r}")
    else:
        print(f"[+] All {len(tier_map)} companies successfully assigned to a tier.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tier_map, f, indent=2)

    print(f"[+] Company tiers written to {output_path}")
    return tier_map


def print_spot_checks(tier_map):
    print()
    print("=== SPOT CHECK (5 entries) ===")
    spot = ["tcs", "google", "swiggy", "pied piper", "krutrim"]
    for co in spot:
        entry = tier_map.get(co, "NOT FOUND")
        print(f"  [{co}] -> {entry}")


def print_tier_summary(tier_map):
    print()
    print("=== TIER DISTRIBUTION SUMMARY ===")
    from collections import Counter
    counts = Counter(v["tier"] for v in tier_map.values())
    for tier_label, count in sorted(counts.items()):
        print(f"  {tier_label}: {count} companies")


if __name__ == "__main__":
    FACT_SHEET_PATH = "precompute/cache/global_fact_sheet.json"
    OUTPUT_PATH = "precompute/cache/company_tiers.json"

    tier_map = build_tier_map(FACT_SHEET_PATH, OUTPUT_PATH)
    print_spot_checks(tier_map)
    print_tier_summary(tier_map)
