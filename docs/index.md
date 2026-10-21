# pyursa

Engine-agnostic dataframes for the agentic AI era. One strict, lazy API runs on
Polars or DuckDB with identical results (float sums can differ in the last digits), and the plan layer adds what agents need:
structured errors with fixes, undo, lineage, dry runs, token-budgeted summaries,
LLM-powered operators, and an MCP server.

## Install

```bash
pip install "pyursa[all]"        # Polars + DuckDB backends
pip install "pyursa[mcp]"        # + the MCP server
pip install "pyursa[llm]"        # + the Anthropic SDK (other providers need nothing extra)
```

Requires Python 3.10+ and PyArrow. Install at least one backend.

## Five-minute tour

```python
import pyursa as pu
from pyursa import col

orders = pu.read_parquet("orders/*.parquet")        # lazy; nothing is read yet
customers = pu.read_csv("customers.csv")

print(orders.describe_for_llm(tokens=500))           # what an agent sees first

report = (
    orders
    .filter(col("qty") > 0, col("status") != "cancelled", intent="valid orders only")
    .with_columns(revenue=col("qty") * col("price"))
    .join(customers, on="customer_id")
    .group_by("segment")
    .agg(revenue=col("revenue").sum(), orders=pu.len(), median_price=col("price").median())
    .sort("revenue", descending=True)
)

report.explain()                 # dry run: output schema, size bounds, compiled query
report.collect("duckdb")         # -> pyarrow.Table, identical on "polars"
report.history()                 # every step and its intent
report.undo(2)                   # the frame two steps ago, instantly
report.lineage("revenue")        # which source columns, filters and joins produced it
report.write_parquet("report.parquet")
```

Mistakes fail fast with a fix an agent can apply:

```python
orders.filter(col("quantity") > "5")
# [COLUMN_NOT_FOUND] Column 'quantity' not found in the input of this step.
#   suggestion: Did you mean 'qty'?  ->  col("qty")
```

## Where to next

- [User guide](guide.md): loading, expressions, joins, windows, writing.
- [Types](types.md): the strict type system and what it refuses.
- [Agents and MCP](agents.md): tools, the MCP server, error handling.
- [LLM operators](llm.md): `llm_filter`, `llm_map`, `llm_extract`, providers, cost controls.
- [Semantics](semantics.md): the rules every backend follows.
- [API reference](api.md): every public function and method.
- [Security](security.md) and [stability policy](stability.md).
