"""
Generate shoppers and past orders with an LLM (Member 4) -> data/raw/customers.json, orders.json

    python -m scripts.generate_shoppers              # Groq (needs GROQ_API_KEY), 30 shoppers
    python -m scripts.generate_shoppers --offline    # deterministic fallback, no API

The brief asks for LLM-generated shopper data, with the exact prompts and the model saved.
  * Prompt template: scripts/prompts/shopper_prompt.md (read by this script)
  * Every filled prompt + model + raw reply: data/raw/generation_log.json
Then code cleans the LLM output: unknown product IDs are dropped, the required edge cases
are enforced, and shoppers left with too few orders are topped up from their categories.
If Groq is unavailable, a seeded fallback produces the same shape (logged as "fallback").
"""
import argparse
import json
import random
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.config import BASE_DIR, LEGACY_CATEGORIES as CATEGORIES, reload_settings  # legacy generator
from app.schemas import Customer, Order

RAW = BASE_DIR / "data" / "raw"
PROMPT_FILE = BASE_DIR / "scripts" / "prompts" / "shopper_prompt.md"
TOTAL = 30
BATCH = 10
FIRST_NAMES = ["Priya", "Rahul", "Ananya", "Arjun", "Sneha", "Vikram", "Kavya", "Rohan", "Isha", "Aditya",
               "Meera", "Karan", "Pooja", "Siddharth", "Neha", "Amit", "Ritu", "Manish", "Divya", "Saurabh",
               "Anjali", "Nikhil", "Shreya", "Harsh", "Tanvi", "Deepak", "Ayesha", "Rajat", "Nandini", "Varun"]
LAST_NAMES = ["Sharma", "Verma", "Gupta", "Singh", "Iyer", "Reddy", "Nair", "Patel", "Das", "Mehta",
              "Kulkarni", "Chatterjee", "Yadav", "Khan", "Joshi", "Mishra", "Bose", "Rao", "Kapoor", "Pandey"]
CITIES = ["Prayagraj", "Lucknow", "Patna", "Delhi", "Mumbai", "Bengaluru", "Pune", "Kolkata", "Jaipur",
          "Hyderabad", "Chennai", "Bhopal", "Indore", "Varanasi", "Kanpur", "Ranchi"]


def _sections(text: str) -> Dict[str, str]:
    parts = re.split(r"^## ", text, flags=re.M)
    out = {}
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out[title.strip()] = body.strip()
    return out


def load_prompts() -> Dict[str, str]:
    s = _sections(PROMPT_FILE.read_text(encoding="utf-8"))
    return {"system": s["System"], "user": s["User (per batch)"],
            "edge": s["Edge-case instructions (added to the first batch only)"]}


def _cid(n: int) -> str:
    return f"C-{n:03d}"


def _fill(template: str, **values) -> str:
    # str.format would choke on JSON braces in the template; replace known keys only
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------
def generate_with_llm(products: List[Dict[str, Any]], oos_ids: List[str], log: List[Dict]) -> List[Dict[str, Any]]:
    settings = reload_settings()
    if not settings.groq_api_key or settings.groq_api_key.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY not set")
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key, timeout=120, max_retries=2)
    prompts = load_prompts()
    brands = sorted({p["brand"] for p in products if p.get("brand")})
    product_lines = "\n".join(
        f"{p['product_id']} | {p['name']} | {p['category']} | {p['price']} | {'yes' if p['stock_qty'] > 0 else 'NO'}"
        for p in products if p.get("price"))
    shoppers: List[Dict[str, Any]] = []
    for start in range(1, TOTAL + 1, BATCH):
        end = min(start + BATCH - 1, TOTAL)
        edge = ""
        if start == 1:
            edge = _fill(prompts["edge"], first_id=_cid(1), second_id=_cid(2), new_id=_cid(3),
                         contradict_id=_cid(4), oos_id=_cid(5), oos_ids=", ".join(oos_ids[:6]))
        user = _fill(prompts["user"], n=end - start + 1, id_from=_cid(start), id_to=_cid(end),
                     edge_cases=edge, categories=json.dumps(CATEGORIES), brands=json.dumps(brands),
                     products=product_lines)
        kwargs = dict(model=settings.groq_model, temperature=0.8, max_tokens=8000,
                      messages=[{"role": "system", "content": prompts["system"]}, {"role": "user", "content": user}],
                      response_format={"type": "json_object"})
        if settings.groq_model.startswith("openai/gpt-oss"):
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        t0 = time.perf_counter()
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        log.append({"model": settings.groq_model, "batch": f"{_cid(start)}..{_cid(end)}",
                    "system_prompt": prompts["system"], "user_prompt": user, "raw_reply": content,
                    "seconds": round(time.perf_counter() - t0, 1)})
        batch = json.loads(content).get("shoppers", [])
        print(f"  batch {_cid(start)}..{_cid(end)}: {len(batch)} shoppers in {log[-1]['seconds']}s")
        shoppers.extend(batch)
    return shoppers


# ---------------------------------------------------------------------------
# Deterministic fallback (same output shape)
# ---------------------------------------------------------------------------
def generate_fallback(products: List[Dict[str, Any]], oos_ids: List[str], seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    brands_by_cat: Dict[str, List[str]] = {}
    for p in products:
        if p.get("brand"):
            brands_by_cat.setdefault(p["category"], [])
            if p["brand"] not in brands_by_cat[p["category"]]:
                brands_by_cat[p["category"]].append(p["brand"])
    shoppers = []
    for n in range(1, TOTAL + 1):
        name = (f"{FIRST_NAMES[n - 1]} {LAST_NAMES[(n * 7) % len(LAST_NAMES)]}" if n > 2
                else ["Priya Sharma", "Rahul Verma"][n - 1])
        cats = rng.sample(CATEGORIES, rng.randint(1, 3))
        budget_max = rng.choice([1500, 2000, 3000, 5000, 8000, 15000, 30000, 60000])
        shoppers.append({
            "customer_id": _cid(n), "name": name, "city": rng.choice(CITIES),
            "preferred_language": rng.choices(["en", "hinglish", "hi"], [0.5, 0.33, 0.17])[0],
            "budget_min": rng.choice([None, 200, 500]), "budget_max": budget_max,
            "preferred_brands": rng.sample(brands_by_cat[cats[0]], min(2, len(brands_by_cat[cats[0]]))),
            "preferred_categories": cats,
            "sizes": {"shoe": rng.choice([6, 7, 8, 9, 10]), "apparel": rng.choice(["S", "M", "L", "XL"])},
            "persona_note": f"Shops mostly for {cats[0].lower()} and watches the budget closely.",
            "orders": [],
        })
    return shoppers


# ---------------------------------------------------------------------------
# Clean-up shared by both paths (code, not LLM)
# ---------------------------------------------------------------------------
def clean(shoppers: List[Dict[str, Any]], products: List[Dict[str, Any]], oos_ids: List[str],
          seed: int) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    rng = random.Random(seed + 1)
    by_id = {p["product_id"]: p for p in products}
    brands = {p["brand"] for p in products if p.get("brand")}
    report = {"dropped_invalid_order_refs": 0, "topped_up_shoppers": 0, "input_shoppers": len(shoppers)}

    # one record per expected ID, in order
    seen: Dict[str, Dict[str, Any]] = {}
    for s in shoppers:
        cid = str(s.get("customer_id", "")).upper()
        if re.fullmatch(r"C-\d{3}", cid) and cid not in seen:
            seen[cid] = s
    fallback = {s["customer_id"]: s for s in generate_fallback(products, oos_ids, seed)}
    customers, orders = [], []
    start_day = date(2025, 11, 1)

    for n in range(1, TOTAL + 1):
        cid = _cid(n)
        s = dict(seen.get(cid) or fallback[cid])
        s["customer_id"] = cid
        if n == 1:
            s["name"] = "Priya Sharma"
        if n == 2:
            s["name"] = "Rahul Verma"
        s["preferred_brands"] = [b for b in (s.get("preferred_brands") or []) if b in brands][:3]
        s["preferred_categories"] = [c for c in (s.get("preferred_categories") or []) if c in CATEGORIES][:3] \
            or [rng.choice(CATEGORIES)]
        if s.get("preferred_language") not in {"en", "hi", "hinglish"}:
            s["preferred_language"] = "en"
        if not isinstance(s.get("sizes"), dict):
            s["sizes"] = {}

        raw_orders = s.pop("orders", []) or []
        valid = []
        for o in raw_orders:
            pid = str(o.get("product_id", "")).upper()
            if pid in by_id and by_id[pid].get("price"):
                valid.append({"product_id": pid, "quantity": max(1, min(3, int(o.get("quantity") or 1))),
                              "order_date": str(o.get("order_date") or "")})
            else:
                report["dropped_invalid_order_refs"] += 1

        # ---- enforce edge cases ----
        if n == 3:                        # brand-new shopper
            valid = []
            s["persona_note"] = "Brand-new shopper with no order history."
        if n == 4:                        # contradictory tastes
            s["budget_max"] = 1500
            s["preferred_brands"] = [b for b in ["Sony", "Nike", "Columbia"] if b in brands]
            s["persona_note"] = "Says they hate sports and want cheap items, yet likes premium sports brands."
            sports = [p for p in products if p["category"] == "Sports & Outdoors" and p.get("price")]
            if not any(by_id[o["product_id"]]["category"] == "Sports & Outdoors" for o in valid):
                valid.append({"product_id": rng.choice(sports)["product_id"], "quantity": 1, "order_date": ""})
        if n == 5:                        # favourites now out of stock
            have = {o["product_id"] for o in valid}
            for pid in [x for x in oos_ids if by_id[x].get("price")][:3]:
                if pid not in have:
                    valid.append({"product_id": pid, "quantity": 1, "order_date": ""})
            s["persona_note"] = "Keeps re-buying favourite products that are now out of stock."

        # most shoppers need 3+ orders
        if n != 3 and len(valid) < 3:
            report["topped_up_shoppers"] += 1
            pool = [p for p in products if p["category"] in s["preferred_categories"] and p.get("price")]
            for p in rng.sample(pool, min(3 - len(valid), len(pool))):
                valid.append({"product_id": p["product_id"], "quantity": 1, "order_date": ""})

        for o in valid:
            try:
                day = date.fromisoformat(o["order_date"])
                if not date(2025, 11, 1) <= day <= date(2026, 11, 15):
                    raise ValueError
            except ValueError:
                day = start_day + timedelta(days=rng.randint(0, 379))
            orders.append({"order_id": f"O-{len(orders) + 1:05d}", "customer_id": cid, "product_id": o["product_id"],
                           "quantity": o["quantity"], "price_paid": by_id[o["product_id"]]["price"] * o["quantity"],
                           "order_date": day.isoformat()})
        customers.append(Customer(**s).model_dump())
    for o in orders:
        Order(**o)
    return customers, orders, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip the LLM, use the seeded fallback")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    products = json.loads((RAW / "products.json").read_text(encoding="utf-8"))
    oos_ids = [p["product_id"] for p in products if p["stock_qty"] == 0]
    log: List[Dict[str, Any]] = []
    mode = "fallback"
    if not args.offline:
        try:
            print("Generating shoppers with Groq ...")
            shoppers = generate_with_llm(products, oos_ids, log)
            mode = "llm"
        except Exception as exc:  # noqa: BLE001
            print(f"LLM generation failed ({type(exc).__name__}: {exc}); using offline fallback.")
            log.append({"error": f"{type(exc).__name__}: {exc}"})
            shoppers = generate_fallback(products, oos_ids, args.seed)
    else:
        shoppers = generate_fallback(products, oos_ids, args.seed)

    customers, orders, report = clean(shoppers, products, oos_ids, args.seed)
    (RAW / "customers.json").write_text(json.dumps(customers, ensure_ascii=False, indent=1), encoding="utf-8")
    (RAW / "orders.json").write_text(json.dumps(orders, ensure_ascii=False, indent=1), encoding="utf-8")
    (RAW / "generation_log.json").write_text(json.dumps(
        {"mode": mode, "model": reload_settings().groq_model if mode == "llm" else None,
         "prompt_file": "scripts/prompts/shopper_prompt.md", "cleanup_report": report, "calls": log},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Mode: {mode}. Wrote {len(customers)} shoppers and {len(orders)} orders. Clean-up: {report}")
    print("Next: python -m scripts.validate_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
