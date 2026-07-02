import json

with open('precompute/cache/global_fact_sheet.json') as f:
    fs = json.load(f)

print('=== PHASE 1 ACCEPTANCE TEST ===')
print(f'Total candidates processed : {fs["metadata"]["total_pool_candidates"]}')
print(f'Distinct companies found   : {len(fs["companies"])}')
print()

for co in ['tcs', 'krutrim', 'google']:
    d = fs['companies'].get(co)
    if d:
        print(f'[{co}] headcount={d["headcount"]}  earliest={d["earliest_known_start"]}')
    else:
        print(f'[{co}] NOT FOUND')

print()
print('Expected:')
print('  [tcs]     headcount=23483  earliest=2011-04-18')
print('  [krutrim] headcount=64     earliest=2018-11-05')
print('  [google]  headcount=14     earliest=2019-01-04')
