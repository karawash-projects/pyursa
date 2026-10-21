"""Run with:  PYTHONPATH=src python examples/quickstart.py"""

import pyursa as pu
from pyursa import col

orders = pu.from_dict(
    {
        "order_id": [1, 2, 3, 4, 5, 6],
        "region": ["east", "west", "east", None, "west", "east"],
        "qty": [10, 20, None, 5, 20, 10],
        "price": [1.5, 2.0, 3.0, None, 1.0, 4.0],
    },
    name="orders",
)

# 1. An agent looks at the data without loading it into its context.
print(orders.describe_for_llm(tokens=300))

# 2. It makes a mistake; the error is structured and suggests the fix.
try:
    orders.filter(col("quantity") > "5")
except pu.PyursaError as e:
    print("\nError for the agent:", e.to_json(indent=2))

# 3. It builds a pipeline, recording why each step exists.
report = (
    orders.filter(col("qty") > 5, intent="ignore sample orders")
    .with_columns(revenue=col("qty") * col("price"))
    .group_by("region")
    .agg(col("revenue").sum().alias("revenue"), pu.len().alias("orders"))
)

# 4. Dry run before executing.
plan = report.explain(backend="duckdb")
print("\nEstimated max rows:", plan["estimated_rows_max"])

# 5. Same plan, any engine, identical Arrow result.
print("\npolars:", report.to_dicts("polars"))
print("duckdb:", report.to_dicts("duckdb"))

# 6. Audit and undo.
print("\nHistory:", [(s["op"], s["intent"]) for s in report.history()])
print("Lineage of revenue:", report.lineage("revenue"))
print("Undo 2 steps ->", report.undo(2).columns)
