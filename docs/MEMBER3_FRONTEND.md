# Member 3: Streamlit frontend (`ui/`)

| File | What it does |
|---|---|
| `ui/streamlit_app.py` | Page layout: sidebar (API status, shopper login, new chat, demo prompts, product lookup, live catalogue update) and the chat |
| `ui/components.py` | Answer-type badges, product cards, comparison table, follow-up chips, trace viewer |
| `ui/api_client.py` | The only place that calls the API. Timeouts plus friendly errors; Indian ₹ formatting |

## Run it

```bash
uvicorn app.main:app --port 8000          # terminal 1 (API)
streamlit run ui/streamlit_app.py         # terminal 2 -> http://localhost:8501
```
If the API runs elsewhere, set `API_URL=http://<host>:8000` in `.env`.

## How each requirement shows up on screen

| Requirement | On screen |
|---|---|
| R4 reason + ID | Every card shows the **product ID**, price from the DB, a stock badge (green / orange "only N left" / red) and 💡 the reason |
| R2 clarification | Orange "❓ Quick question" badge plus the question |
| R5 memory | Follow-up chips ("Only Nike", "Show cheaper ones", "Compare the first two") send a new message in the **same session** |
| Comparison | Side-by-side table built from the API's `comparison` field; missing values show "not listed" |
| R6 not found / refusal | Grey "🔍 Not in catalogue" and red "🛡️ Can't help with that" badges |
| R9 privacy | Changing the shopper in the sidebar **always starts a new chat**. Identity comes from the login picker only |
| R10 trace | "🔎 Trace" expander under each answer loads `GET /trace/{id}`: intent and its source, the tools run with timings, and retrieved candidates |
| R11 live updates | Sidebar "Update catalogue": change stock or price through `POST /ingest`, then ask again |

## Design choices to explain to the judges
- **The UI adds no facts.** It only displays the API response, so what the judges see equals what `/chat` returns.
- **Chips appear only on the newest answer.** Old chips can't be clicked out of context.
- **Graceful failure.** If the API is down, the sidebar says how to start it and the chat box is disabled. Chat timeouts show a message, not a crash.
- **Demo buttons** cover every judged scenario (discovery, budget, vague, Hinglish, Hindi, follow-up, compare, stock, not found, privacy, injection) for a smooth 10-minute demo.

## Demo script (10 minutes)
1. Log in as **Rahul (C-002)** → "Budget" demo (trekking gift) → open its **Trace**.
2. Click the chip **Compare the first two**.
3. **New chat** → "Vague: I need a laptop" → answer "around 50k for study" → honest not-found.
4. Log in as **Priya (C-001)** → "Discovery: running shoes for flat feet" → "only Nike" → "Is the blue one in size 9?".
5. "Hinglish" demo → reply comes in Hinglish.
6. "Not found" → "Privacy" → "Injection": all refused or declined.
7. Sidebar **Update catalogue**: set P-1004 stock to 0 → ask "running shoes" again → it disappears.
