# User guide

## Frames are lazy and immutable

Every method returns a new `Frame` describing a plan. Nothing runs until you call
`collect()`, `to_dicts()`, `to_pandas()`, `to_polars()`, `count()` or a `write_*`
method. The original frame never changes, so `undo()` and branching are free.

## Loading data

| Source | Function | Notes |
|---|---|---|
| Parquet file, glob, directory or list | `pu.read_parquet("data/*.parquet")` | Scanned lazily by the engine. Files must share a schema. |
| CSV | `pu.read_csv(path, schema={"amount": "Decimal(12,2)"}, delimiter=",")` | Type inference is done once by PyArrow, so every engine sees the same types. Empty fields become null (`empty_as_null=False` to keep `""`). |
| Newline-delimited JSON | `pu.read_json(path)` | One object per line. |
| In memory | `pu.from_dict`, `pu.from_arrow`, `pu.from_pandas`, `pu.from_polars` | Converted once to canonical Arrow types. |

## Expressions

```python
from pyursa import col, lit, when

col("price") * col("qty")                         # arithmetic
(col("qty") > 0) & col("region").is_in(["east", "west"])
col("discount").fill_null(0)
col("score").is_between(10, 20)
col("amount").round(2)                            # half away from zero, every engine
when(col("score") >= 90).then("A").when(col("score") >= 80).then("B").otherwise("C")
pu.coalesce(col("phone"), col("mobile"), lit("n/a"))
```

Use `&`, `|` and `~` to combine conditions (Python's `and`/`or` raise a helpful error).
Compare to missing values with `.is_null()`, not `== None`.

### Strings and dates

```python
col("name").str.strip().str.lower()
col("email").str.ends_with("@example.com")
col("sku").str.contains_regex(r"^[A-Z]{3}-\d+$")    # portable syntax: lookaround/backreferences are rejected
col("note").str.replace("\n", " ").str.slice(0, 80)  # 0-based
col("day").dt.year(), col("day").dt.weekday()        # ISO: Monday = 1
col("ts").dt.truncate("month")                       # year, month, week, day, hour
```

### Aggregations

`sum`, `mean`, `min`, `max`, `count` (non-null), `n_unique` (non-null), `median`,
`quantile(q)`, `std`, `var` (sample), `any`, `all`, and `pu.len()` (rows).

```python
orders.group_by("region", "segment").agg(revenue=col("revenue").sum(), p90=col("price").quantile(0.9))
orders.agg(n=pu.len(), customers=col("customer_id").n_unique())
```

## Window functions

Keep one row per input row while computing per-group values:

```python
orders.with_columns(
    region_total=col("revenue").sum().over("region"),
    share=col("revenue") / col("revenue").sum().over("region"),
    rank_in_region=pu.rank().over("region", order_by="revenue", descending=True),
    running=col("revenue").cum_sum().over("customer_id", order_by="day"),
    previous=col("revenue").shift(1).over("customer_id", order_by="day"),
)
```

Top N per group:

```python
(orders.with_columns(rn=pu.row_number().over("region", order_by="revenue", descending=True))
       .filter(col("rn") <= 3))
```

Order-dependent functions (`row_number`, `rank`, `cum_sum`, `shift`) require
`order_by=`. Ties keep input order, so results are deterministic.

## Combining frames

```python
a.join(b, on="customer_id")                  # inner (default)
a.join(b, on=["k1", "k2"], how="left")
a.join(b, on="id", how="full")               # keys are coalesced
a.join(b, on="id", how="semi")               # rows of a that have a match
a.join(b, on="id", how="anti")               # rows of a without a match
pu.concat([jan, feb, mar])                   # same columns and types; matched by name
orders.distinct("customer_id")               # first occurrence of each customer
```

Join keys must have the same type on both sides; a mismatch is an error with a cast suggestion.

## Running and writing

```python
frame.collect()                  # pyarrow.Table on the default backend
frame.collect("duckdb")          # choose an engine per call
pu.set_default_backend("duckdb")
frame.write_parquet("out.parquet")
frame.write_csv("out.csv")
frame.write_json("out.jsonl")    # newline-delimited
```

Writers refuse to overwrite existing files unless `overwrite=True`.

## Choosing a backend

Both backends return identical results (enforced by the conformance suite).
Polars is usually fastest for in-memory work; DuckDB is a strong choice for
larger-than-memory Parquet scans. You can add your own backend: see
[CONTRIBUTING](https://github.com/) and `pyursa.backends.Backend`.
