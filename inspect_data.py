import json, collections, io, csv, sys
p = "data/shopmate_products.json"
raw = json.loads(open(p, encoding="utf-8").read())
print("JSON top-level type:", type(raw).__name__)
if isinstance(raw, dict):
    print("keys:", list(raw.keys())[:10])
    for k, v in raw.items():
        if isinstance(v, list):
            print(f"  list key '{k}' len={len(v)}")
            recs = v
            break
else:
    recs = raw
print("records:", len(recs))
print("\n--- first record keys ---")
print(json.dumps(recs[0], ensure_ascii=False, indent=2)[:2500])
print("\n--- field presence ---")
keys = collections.Counter()
for r in recs:
    for k in r:
        keys[k] += 1
for k, c in keys.most_common():
    print(f"  {k}: {c}")
