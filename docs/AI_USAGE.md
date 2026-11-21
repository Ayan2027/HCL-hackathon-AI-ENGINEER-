# AI tool usage declaration

The brief (section 10) allows AI coding assistants if we say where we used them and can explain every line.

## Tools used
| Tool | Used for |
|---|---|
| **Claude (Anthropic), via chat** | Architecture proposal; first drafts of all modules (config, schemas, AI engine, backend, UI, data and eval scripts, Docker, docs); test cases |
| **Groq LLM** (`openai/gpt-oss-120b`) | At runtime: intent understanding and answer wording. At data time: generating synthetic shoppers and orders (prompt in `scripts/prompts/shopper_prompt.md`, log in `data/raw/generation_log.json`) |

## What the team did
- Defined the problem understanding, architecture choices and the 4-way split (PPT).
- Copied the generated code into their own branches, **read every file**, ran the tests, and fixed integration issues.
- Ran the data generation, validation and evaluation **on our own machines**. The numbers in `docs/EVAL_REPORT.md` come from those runs only.
- Prepared and rehearsed the demo; each member can explain their module and the overall pipeline.

## Where AI was not used
- Evaluation results: produced by running `scripts/run_eval.py`, never written by hand or by an AI.
- No real customer data was given to any AI tool.

_(Edit this file to reflect exactly what your team did before submitting.)_
