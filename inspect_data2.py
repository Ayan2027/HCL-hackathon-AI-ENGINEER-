import json, collections, csv, statistics
recs = json.loads(open("data/shopmate_products.json", encoding="utf-8").read())["products"]
ids = [r["product_id"] for r in recs]
print("count:", len(recs), "| unique ids:", len(set(ids)))
print("id pattern sample:", ids[:3], "...", ids[-3:])
print("\ncategory distribution:")
for c, n in collections.Counter(r["category"] for r in recs).most_common(): print(f"  {c}: {n}")
print("\nlanguage:", dict(collections.Counter(r["description_language"] for r in recs)))
print("no rating:", sum(1 for r in recs if r.get("rating") is None))
print("no brand:", sum(1 for r in recs if not (r.get("brand") or "").strip()))
print("zero stock:", sum(1 for r in recs if (r.get("stock_count") or 0) == 0))
print("no image:", sum(1 for r in recs if not (r.get("image_url") or "").strip()))
print("no description:", sum(1 for r in recs if not (r.get("description") or "").strip()))
print("empty embedding_text:", sum(1 for r in recs if not (r.get("embedding_text") or "").strip()))
print("stock_is_simulated true:", sum(1 for r in recs if r.get("stock_is_simulated") is True))
print("retail_price < price:", sum(1 for r in recs if r.get("retail_price") and r.get("price") and r["retail_price"] < r["price"]))
print("price missing/None:", sum(1 for r in recs if r.get("price") is None))
print("\nsubcategory samples:", [r["subcategory"] for r in recs[:3]])
print("\nquality_flags combos:", collections.Counter(tuple(r.get("quality_flags") or []) for r in recs).most_common(8))
print("\nrating present sample:", [(r["product_id"], r["rating"]) for r in recs if r.get("rating") is not None][:5])
print("original_values with rating:", sum(1 for r in recs if isinstance(r.get("original_values"), dict) and "rating" in r["original_values"]))

# CSV check
with open("data/shopmate_products.csv", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
print("\nCSV rows:", len(rows))
print("CSV columns:", list(rows[0].keys()))
csv_ids = [r.get("product_id") for r in rows]
print("CSV unique ids:", len(set(csv_ids)), "| json-csv id diff:", len(set(ids) ^ set(csv_ids)))
