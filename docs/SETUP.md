# Setup (every member, 0:00–0:20)

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
