# Agent benchmark: pyursa tools vs pandas code

`agent_bench.py` asks one question: **with the same LLM, does an agent answer data-analysis
questions better when it drives pyursa's tool interface than when it writes pandas code?**

Each run gives the model one natural-language question and a turn budget. The model replies with one JSON
action per turn until it answers. Two conditions are compared:

| condition    | how the model acts | what it sees back |
|--------------|--------------------|-------------------|
| `pyursa` | `{"action": "<tool>", "args": {...}}` calling `pyursa.tools.ToolSession` (`load_file`, `describe`, `transform`, `preview`, `count_rows`, `explain`, `history`, `undo`, `lineage`, `list_frames`, `help`) | the tool's JSON result, truncated to ~3000 chars |
| `pandas`     | `{"action": "python", "code": "..."}`, run in a fresh `python -I` subprocess (cwd = data dir, 30 s timeout, `pd` and `DATA_DIR` predefined, no state between calls) | stdout + stderr, truncated to ~2000 chars (keeps the end of a traceback) |

Both finish with `{"action": "answer", "value": ...}`. Both get the file names but **not** the column names,
so each agent has to inspect the data first. A reply that does not parse as a JSON action counts as an
error turn, and the model gets a corrective message.

## Data and tasks

The data is synthetic and deterministic (numpy seed, default 50,000 orders). It is written to a temp dir,
and each run gets a fresh copy:

- `orders.parquet`: order_id, customer_id, qty (int, ~3% null, ~4% <= 0), price, status
  (paid / shipped / cancelled / refunded), day (2026-01-01 .. 2026-09-27)
- `customers.csv`: customer_id, segment (enterprise / smb / consumer), region (east / west / north / south)

Ground truth is computed independently with pandas. "Valid order" means qty > 0 (a null qty is not valid)
and status not cancelled. Revenue is qty * price.

| id  | question (short) | answer type |
|-----|------------------|-------------|
| t1  | total revenue of valid orders | number, 2 dp |
| t2  | % of all orders that are refunded | number, 2 dp |
| t3  | distinct customers with an order 2026-04-01..2026-06-30 | int |
| t4  | mean price of orders by enterprise customers (join) | number, 2 dp |
| t5  | region with the highest valid revenue (join + group) | string |
| t6  | top 3 customer_ids by valid revenue, ordered | JSON list of ints |
| t7  | orders with missing or non-positive qty (null handling) | int |
| t8  | largest single valid-order revenue in region west (join) | number, 2 dp |
| t9  | valid smb orders in August 2026 (join + date range) | int |
| t10 | regions ordered by number of shipped orders, ascending | JSON list of strings |

The tasks only use operations that both conditions support: filter, arithmetic columns, join, group-by,
sort, head, sum/mean/min/max/count/n_unique/len, and date-range comparisons. The ground-truth code refuses
ties at ranking cut-offs, so ranked answers are never ambiguous.

**Scoring:** a number counts as correct within a relative tolerance of 1e-3 or an absolute tolerance of 0.01
(`"4.87%"` and `"1,234.5"` are accepted). Integers must match exactly. Strings are compared trimmed and
case-insensitive. Lists must match exactly, in order.

## Running

Run from the repository root. `--allow-code-exec` is required for the `pandas` condition, because that
condition **executes model-generated Python on your machine**. The subprocess runs with API-key-like
environment variables removed, but it is not a sandbox, so use a VM or container if that matters to you.
Without the flag, the pandas condition is skipped with a message.

bash / zsh:

```bash
export GEMINI_API_KEY=...            # or OPENAI_API_KEY / ANTHROPIC_API_KEY ...
PYTHONPATH=src python benchmarks/agent_bench.py --llm gemini:gemini-3.8-flash \
    --allow-code-exec --trials 3 --out results.json
```

PowerShell:

```powershell
$env:GEMINI_API_KEY = "..."
$env:PYTHONPATH = "src"
python benchmarks/agent_bench.py --llm gemini:gemini-3.8-flash --allow-code-exec --trials 3 --out results.json
```

Other specs: `openai:<model>`, `anthropic:claude-haiku-5-5`, `ollama:llama3.2` (local, no key).

Demo without any API key (a scripted fake model that solves every task):

```bash
PYTHONPATH=src python benchmarks/agent_bench.py --fake --allow-code-exec
```

Options: `--conditions pyursa,pandas`, `--tasks t1,t3|all`, `--trials N`, `--rows N`, `--seed N`,
`--max-turns N` (default 10), `--out results.json` (all run records, including full transcripts).

Tests: `PYTHONPATH=src python -m pytest -q tests/test_bench_harness.py`

## Reading the results

The script prints one line per run, then a per-task table, then a summary for each condition:

- **success**: share of runs where the final answer matched the ground truth. This is the headline number.
- **turns**: model calls used, including the answer turn. A run that hits `--max-turns` without answering fails.
- **errors**: turns whose action failed: unparseable reply, unknown action, a tool returning `ok: false`,
  or code that exits non-zero or times out. Fewer errors suggests an interface that is easier to use correctly.
- **tokens**: approximate prompt + completion tokens (characters / 4) summed over the run. Each turn
  resends the full transcript, so tokens grow faster than linearly with turns.
- **seconds**: wall time, including model latency and code or tool execution.

`--out` saves every transcript. Read the failures before drawing conclusions, because they show
*why* a condition lost: a wrong definition of "valid", a broken join, bad syntax, or running out of turns.

The fake-mode numbers only show that the harness works. They say nothing about either interface.
The fake model follows fixed scripts (e.g. its pyursa script loads, describes, transforms, previews, then answers).

## Caveats

- **One model per run.** Results can differ a lot between models, so test the models you actually use and
  run several `--trials`, since LLM outputs vary from run to run.
- **Synthetic data and a small task set (10 questions).** These are clean, well-specified
  aggregation questions. They are not representative of messy real-world analysis, and the differences
  will be noisy. With 10 tasks × 3 trials, a gap of a few percentage points is within noise.
- **Prompt asymmetry is unavoidable.** The pyursa system prompt has to describe the tool and transform
  formats, so it is longer (more input tokens per turn). The pandas prompt relies on the model already knowing
  pandas. Both get the same file names and no column names.
- **Token counts are approximate** (chars / 4) and do not match any provider's tokenizer.
- **Single-JSON-action protocol.** The harness does not use native function-calling, so models that are weak at
  strict JSON are penalised equally in both conditions.
- Report the numbers as they are, including runs where pyursa loses, together with the model, trial count,
  row count and seed.
