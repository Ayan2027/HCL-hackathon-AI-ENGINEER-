# Shopper generation prompt (used verbatim by `scripts/generate_shoppers.py`)

The script sends the **system** text below, then one **user** message per batch.
`{...}` placeholders are filled by the script. The exact filled prompts and the model
name are saved to `data/raw/generation_log.json` on every run.

## System

You generate SYNTHETIC test data for a fictional Indian online store. Nothing you write
describes a real person. Return ONLY valid JSON, no comments, no markdown.

## User (per batch)

Create {n} fictional Indian shoppers with customer IDs {id_from} to {id_to} (format C-XXX).

Each shopper must be a JSON object with exactly these keys:
- "customer_id": string
- "name": a common Indian first and last name (fictional person)
- "city": an Indian city
- "preferred_language": one of "en", "hi", "hinglish"
- "budget_min": number in INR or null
- "budget_max": number in INR or null
- "preferred_brands": list of 0-3 brands chosen ONLY from BRANDS
- "preferred_categories": list of 1-3 categories chosen ONLY from CATEGORIES
- "sizes": object, e.g. {"shoe": 9, "apparel": "M"} (may be empty)
- "persona_note": one sentence about their shopping habits
- "orders": list of past orders, each {"product_id": ID from PRODUCTS, "quantity": 1-3, "order_date": "YYYY-MM-DD" between 2025-11-01 and 2026-11-15}

Rules:
- Most shoppers have 3 to 5 orders. Every product_id MUST be copied exactly from PRODUCTS.
- Orders should fit the shopper's categories, brands and budget.
- Mix of languages: about half "en", a third "hinglish", the rest "hi".
- Vary budgets from students (under ₹2,000) to premium buyers (above ₹20,000).
{edge_cases}

CATEGORIES: {categories}
BRANDS: {brands}
PRODUCTS (id | name | category | price | in_stock):
{products}

Return: {"shoppers": [ ... ]}

## Edge-case instructions (added to the first batch only)

- {first_id} must be named "Priya Sharma" and {second_id} "Rahul Verma" (used in the demo script).
- {new_id} is a brand-new shopper: "orders" must be an empty list.
- {contradict_id} has contradictory tastes: e.g. budget_max under 2000 but prefers premium brands, and says they hate sports yet orders sports items.
- {oos_id} has favourite products that are now out of stock: at least 2 of their orders must use these OUT-OF-STOCK IDs: {oos_ids}
