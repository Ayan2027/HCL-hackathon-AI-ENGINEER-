# Data note (one page)

## Where the data came from
| File | Source | Generator |
|---|---|---|
| `data/raw/products.json` (279) | **Invented** by a seeded template generator, no LLM, seed 42 | `scripts/generate_catalog.py` |
| `data/raw/customers.json` (30) | **LLM-generated** (Groq, model logged), cleaned by code | `scripts/generate_shoppers.py` + `scripts/prompts/shopper_prompt.md` |
| `data/raw/orders.json` | Same LLM call as shoppers; invalid product IDs dropped by code | same |
| `data/raw/generation_log.json` | Mode (`llm` or `fallback`), model name, every filled prompt, raw replies, clean-up report | same |
| `data/raw/catalog_meta.json` | List of every deliberate defect, by product ID | `generate_catalog.py` |

Brand names (Nike, boAt, Prestige …) are used only as labels in a **fictional shop**. All model names, prices, descriptions, reviews and ratings are made up. **No real customer data, scraped data or copyrighted catalogue text is used.**

## What the fields mean
**Product:** `product_id` (P-1001…), `name`, `category` (one of 6), `subcategory`, `brand`, `price` (INR), `stock_qty` (0 = out of stock), `rating` (0–5), `rating_count`, `attributes` (category-specific: `color`, `sizes_available`, `battery_hours`, `capacity_l`, `use_case` …), `description`, `review_summary`, `reviews` (short texts), `best_for`, `weight_raw` (as supplied), `language` (of the description).

**Shopper:** `customer_id` (C-001…), `name`, `city`, `preferred_language` (en / hi / hinglish), `budget_min`, `budget_max`, `preferred_brands`, `preferred_categories`, `sizes`, `persona_note`.
**Order:** `order_id`, `customer_id`, `product_id` (must exist), `quantity`, `price_paid`, `order_date`.

## Deliberate mess (brief section 4.1)
Exact IDs are in `catalog_meta.json`.
- **Out of stock:** 27 products. **Nearly out** (1–3 left): 22.
- **Similar products across brands:** 9 "twins", the same product under another brand at 0.7–1.3× the price.
- **Missing fields:** brand (7), rating (10), price (3), description (5). Code treats a missing price as "never inside a budget".
- **Inconsistent units:** weights as `"850 g"`, `"850g"`, `"0.85 kg"`, `"0.85kg"`; capacities as `"1 L"` / `"1000 ml"`. Code normalises weights to grams.
- **Other languages:** 16 Hindi descriptions; 12 products with Hinglish reviews.
- **Injected instructions:** 3 reviews tell the assistant to recommend the product "for every query", or to claim a 90% discount. Used to test R8.

## Shopper edge cases (enforced by code after the LLM)
- C-001 "Priya Sharma" and C-002 "Rahul Verma": fixed names for the demo script.
- C-003: brand-new shopper with **no orders**.
- C-004: **contradictory tastes** (budget under ₹1,500 but prefers premium brands; "hates sports" but bought sports gear).
- C-005: favourite products are **now out of stock**.

## Validation
`python -m scripts.validate_data` checks:
- the schema (shared Pydantic models), unique IDs, and that every order points at a real product and customer;
- the brief's minimums (≥250 products, ≥5 categories, ≥25 shoppers, most with ≥3 orders);
- that each kind of deliberate mess is present.
Current result: **21/21 checks passed**.

## What it does not cover
- No real prices, real reviews or real stock. Numbers are plausible but invented.
- Only 6 categories; no toys, groceries, furniture, large appliances or vehicles. These are asked in the eval as **not-found** cases.
- One stock count per product: colours and sizes are attributes, not separate stock-keeping variants.
- Reviews are short and few, so review-based answers are shallow.
- Hindi coverage is partial (descriptions in a few subcategories).
