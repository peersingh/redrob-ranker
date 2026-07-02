import json

with open('precompute/cache/llm_judgments.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]

print(f"Total records: {len(lines)}")
ok, errors = 0, 0
for r in lines:
    j = r.get('judgment', {})
    if j.get('error'):
        errors += 1
        print(f"  ERROR: {r['candidate_id']} -> {j['error']}")
    else:
        ok += 1

print(f"\nOK: {ok}, Errors: {errors}")
if ok > 0:
    sample = next(r for r in lines if not r.get('judgment', {}).get('error'))
    print("\nSample judgment:")
    print(json.dumps(sample['judgment'], indent=2))
