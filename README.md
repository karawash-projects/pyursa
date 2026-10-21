# pyursa

**Engine-agnostic dataframes for the agentic AI era.** Named for *Ursa*, the bear, in the
tradition of pandas and Polars. One strict, lazy API runs on
Polars or DuckDB with identical results. The plan layer adds what AI agents need:
structured errors with fixes, undo, lineage, dry runs, token-budgeted summaries,
LLM-powered operators, and an MCP server.

```python
import pyursa as pu
from pyursa import col

orders = pu.read_parquet("orders/*.parquet")        # lazy
print(orders.describe_for_llm(tokens=500))           # fits an LLM context window

report = (orders
    .filter(col("qty") > 0, col("status") != "cancelled", intent="valid orders only")
    .with_columns(revenue=col("qty") * col("price"))
    .group_by("region")
    .agg(revenue=col("revenue").sum(), orders=pu.len(), p90=col("price").quantile(0.9))
    .sort("revenue", descending=True))

report.explain()            # dry run: schema, size bounds, compiled query
report.collect("duckdb")    # pyarrow.Table, identical on "polars"
report.undo(2); report.lineage("revenue"); report.write_parquet("report.parquet")
```

```python
orders.filter(col("quantity") > "5")
# [COLUMN_NOT_FOUND] Column 'quantity' not found in the input of this step.
#   suggestion: Did you mean 'qty'?  ->  col("qty")
```

## Install

```bash
pip install "pyursa[all]"     # Polars + DuckDB backends
pip install "pyursa[mcp]"     # + MCP server (pyursa-mcp)
```

Python 3.10+. From a source checkout: `pip install -e ".[dev]"`.

## Highlights

- **Same answers on every engine.** One semantics contract (nulls, NaN, sort stability,
  rounding, decimals, casts, overflow, quantiles, window ties) enforced by conformance tests
  on every backend. Float sums may differ in the last digits; everything else is identical.
- **Strict types, no silent corruption.** Exact `Decimal(p,s)` math, time-zone-aware
  `Datetime(tz)`, `List`/`Struct` pass-through, strict casts, integer overflow errors;
  lossy inputs raise errors instead of corrupting data.
- **Complete toolkit.** Filters, joins (inner, left, full, semi, anti), group-by with
  median/quantile/std, window functions (rank, row_number, running totals, lag), `when/then`,
  string and date functions, distinct, concat, Parquet/CSV/JSON in and out.
- **Built for agents.** Errors are JSON with paste-ready fixes, raised before execution.
  Immutable frames with `history()`/`undo()`, column `lineage()`, `explain()` dry runs, and
  `describe_for_llm(tokens=N)` that always fits and flags prompt-injection text.
- **LLM operators.** `llm_filter`, `llm_map`, `llm_extract` with **any major LLM**: 53
  OpenAI-compatible providers (OpenAI, Gemini, Mistral, Cohere, xAI, DeepSeek, Qwen, Kimi, GLM,
  Groq, Together, Fireworks, Azure, Vertex, Databricks, Snowflake, Cloudflare, OpenRouter, LiteLLM,
  Ollama, vLLM, ...) plus native Claude and Amazon Bedrock (Nova, Claude on AWS). Dedup, batching,
  caching, row/token budgets, cost reports and PII redaction included. See [providers](docs/providers.md).
- **MCP server.** Handles instead of data, sandboxed files, a code-free expression parser,
  per-client session isolation.
- **Fast.** About 3–4× pandas on a typical analysis at 2M rows (see `scripts/e2e.py`).

## Documentation

[Guide](docs/guide.md) · [Types](docs/types.md) · [Agents and MCP](docs/agents.md) ·
[LLM operators](docs/llm.md) · [LLM providers](docs/providers.md) · [Semantics](docs/semantics.md) · [API reference](docs/api.md) ·
[Security](SECURITY.md) · [Stability policy](docs/stability.md) · [Changelog](CHANGELOG.md)

## Testing

```bash
pytest                                    # unit, conformance, provider and security tests
python scripts/e2e.py                     # end-to-end checks, including pandas comparison
python scripts/e2e.py --llm gemini:<model>   # live LLM check with any provider
python benchmarks/agent_bench.py --llm gemini:<model> --allow-code-exec   # agent benchmark
```

## License

Apache-2.0. See [LICENSE](LICENSE).
