# Setup

## 0. ShopMate catalogue on Windows (PowerShell) — current setup

**Python version:** use **Python 3.12** for the full app. `faiss-cpu` has no wheel for Python 3.15,
so a 3.15 environment can only run keyword-only search (it still passes the test suite; the
FAISS test is skipped there). The steps below were run on Windows 11 with CPython 3.12.10.

```powershell
# 1. Install Python 3.12 next to any other version (official Python install manager, per-user)
py install 3.12
py -V:3.12 --version                      # Python 3.12.10

# 2. Separate virtual environment (does not touch an existing .venv)
py -V:3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1           # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch first
pip install -r requirements.txt
python -c "import faiss, torch, sentence_transformers; print(faiss.__version__, torch.__version__)"

# 3. Settings (never commit .env; never paste the key into docs or screenshots)
Copy-Item .env.example .env               # only if you do not have a .env yet
#   GROQ_API_KEY=<your key>   MOCK_LLM=false (real Groq) or true (offline rules)
#   USE_VECTOR_SEARCH=true    only after step 6 reports OK

# 4. Catalogue files: put them here (JSON = source of truth, CSV = cross-check)
#   data\shopmate_products.json
#   data\shopmate_products.csv

# 5. Import the catalogue (safe, repeatable)
python -m app.db.catalog_import --dry-run   # validate files + show the plan; adds missing columns only
python -m app.db.catalog_import             # backs up the DB to data\backups\, then imports in one transaction

# 6. Embedding model (~470 MB, once) and the FAISS index
python -m scripts.download_model
python -m scripts.build_index               # must end with "OK: 300 vectors ..."

# 7. Run
uvicorn app.main:app --port 8000            # terminal 1   -> http://localhost:8000/docs
streamlit run ui/streamlit_app.py           # terminal 2   -> http://localhost:8501
python -m pytest                            # tests (mock LLM, temp databases)
```

`GET /health` reports what is actually running: `products`, `vector_index`, `retrieval_mode`
(`hybrid` or `keyword`), `vector_status` (why vectors are off), `llm_mode`, `archived_products`
and `inventory_simulated`.

### What the catalogue import does
- Validates every JSON record (required fields, unique IDs, types). One bad record aborts the
  whole import and the database is not changed.
- Cross-checks the CSV field by field against the JSON and reports differences (JSON wins).
- Upserts the products; unchanged rows are skipped, so running it twice changes nothing.
- Products from the earlier sample catalogue that **orders still reference** are kept as
  **archived** history rows (`is_active = 0`): never searched, listed, indexed or recommended,
  and shown in "my orders" as *no longer in our catalogue*. Unreferenced old products are removed.
  No order or customer is modified, and no old product ID is mapped to an unrelated new one.
- Rebuilds the FTS5 keyword index from the active catalogue and writes
  `data\catalog_import_report.json` (counts, category distribution, missing fields,
  zero-stock count, CSV check, unresolved order references, backup path).
- Missing ratings and brands stay NULL. `original_values` (e.g. a rating removed from the source)
  is kept for provenance only and is never shown as a fact.

To undo an import, stop the API and copy the backup over the database:
`Copy-Item data\backups\shopsense-<timestamp>.db data\shopsense.db` (also delete
`data\shopsense.db-wal` and `-shm` if present), then run `python -m scripts.build_index`.

### Rebuilding the FAISS index
Run `python -m scripts.build_index` after every catalogue import. The API also refuses a saved
index whose catalogue fingerprint, model or size no longer matches, and rebuilds it at start-up.
Files in `data\index\`: `products.faiss`, `id_map.json` (vector position -> product_id) and
`meta.json` (model, fingerprint, dimension, count, library versions, build time).

### Hindi text in PowerShell
The Windows console can garble Devanagari, so a Hindi reply that looks broken in PowerShell is
often a terminal problem, not an application problem. Test Hindi in the browser UI, or send it
from Python (`requests.post(..., json={...})`) rather than through `curl` in PowerShell.

### Known limitations
- 268 of 300 products have no rating and 21 have no brand; these show as
  "Not available in catalogue" and are never filled in.
- Stock counts are **simulated demo inventory**; the UI and answers say so.
- Shoppers and orders are **synthetic** and were generated for the earlier sample catalogue, so
  all historical order lines point at archived products, and shopper brand/category preferences
  refer to the old catalogue.
- The Groq free tier rate-limits (HTTP 429); the client then waits about 14 s and retries once,
  so some replies are slow. `MOCK_LLM=true` avoids this for offline demos.
- `eval/eval_set.json` and `scripts/run_eval.py` were built for the earlier sample catalogue.
  Deterministic checks for the ShopMate catalogue live in `tests/test_shopmate.py`.

---

# Team setup (every member, 0:00–0:20)

## 1. Repo owner only: create the repo and push Phase 1

```bash
# Create an EMPTY repo on GitHub named shopsense-ai (no README), then:
cd shopsense-ai
git init -b main
git add .
git commit -m "Phase 1: shared foundation (config, schemas, SQL schema, sample data)"
git remote add origin https://github.com/<OWNER>/shopsense-ai.git
git push -u origin main

# Create the four feature branches
for b in member1/ai-engine member2/backend member3/frontend member4/data-eval; do
  git branch "$b" && git push -u origin "$b"
done
```
Windows PowerShell (instead of the `for` loop):
```powershell
"member1/ai-engine","member2/backend","member3/frontend","member4/data-eval" | % { git branch $_; git push -u origin $_ }
```
Then on GitHub, go to **Settings → Collaborators** and add the 3 teammates.

## 2. Every member

```bash
git clone https://github.com/<OWNER>/shopsense-ai.git
cd shopsense-ai
git checkout member1/ai-engine        # your own branch

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac/Linux:    source .venv/bin/activate

# CPU-only PyTorch FIRST (avoids a ~2 GB CUDA download on Linux)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

copy .env.example .env               # Mac/Linux: cp .env.example .env
# put your Groq key in .env (each member can use their own free key)

python -m scripts.download_model     # ~470 MB, do it NOW
python -m scripts.check_setup        # all FAIL items must be fixed
pytest                               # Phase 1 tests must pass
```

## 3. Daily git loop (each member, on your branch)

```bash
git add .
git commit -m "M1: hybrid retrieval with FTS5 + FAISS"
git push
# Pick up teammates' merged work:
git fetch origin && git merge origin/main
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: app` | Run commands from the project root; use `python -m scripts.xxx` |
| Model download slow or failing | Set `USE_VECTOR_SEARCH=false` in `.env` (keyword-only search) |
| Groq 401 / model not found | Run `python -m scripts.check_setup`; it lists the models your key can use |
| No internet at demo | `MOCK_LLM=true`; the whole pipeline still runs |
