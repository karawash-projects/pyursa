"""Journey: a pandas analyst moves a sales report to pyursa.

Data comes the way analysts have it: Parquet written by pandas (category columns, int16/int32, float32,
NaN, nanosecond timestamps that are whole seconds, a time-zone column, missing values) and a CSV export
of customers that uses "NA" as a country code and empty cells for missing values. Every result is
checked against pandas on both engines.
"""

from __future__ import annotations

import math

from _harness import check, equal

N = 60_000


def _make_data(j):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(7)
    ts = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 3600, N), unit="s")
    orders = pd.DataFrame({
        "order_id": np.arange(1, N + 1, dtype="int64"),
        "customer_id": rng.integers(1, 2001, N).astype("int32"),
        "region": pd.Categorical(rng.choice(["east", "west", "north", "south"], N)),
        "product": pd.Categorical(rng.choice([f"P{i:02d}" for i in range(30)], N)),
        "qty": rng.integers(0, 12, N).astype("int16"),
        "unit_price": rng.choice([4.99, 9.5, 19.99, 49.0, 120.25], N).astype("float32"),
        "discount": np.where(rng.random(N) < 0.1, np.nan, rng.choice([0.0, 0.05, 0.1], N)),
        "status": rng.choice(["paid", "paid", "paid", "refunded", "cancelled", None], N),
        "ordered_at": ts,  # datetime64[ns], whole seconds
        "ordered_at_local": ts.tz_localize("UTC").tz_convert("America/Toronto"),
    })
    orders.to_parquet(j.dir / "orders.parquet", index=False, row_group_size=10_000)
    countries = ["CA", "US", "NA", "MX", "", "DE"]  # NA = Namibia; "" = missing
    with open(j.dir / "customers.csv", "w", encoding="utf-8") as fh:
        fh.write("customer_id,name,country,segment\n")
        for i in range(1, 2001):
            seg = "" if i % 17 == 0 else ("enterprise" if i % 5 == 0 else "smb")
            fh.write(f"{i},Customer {i},{countries[i % len(countries)]},{seg}\n")
    return orders, pd.read_csv(j.dir / "customers.csv", keep_default_na=False, na_values=[""])


def _close(a, b, rel=1e-9):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return (a is None or (isinstance(a, float) and math.isnan(a))) == (b is None or (isinstance(b, float) and math.isnan(b)))
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(a)), abs(float(b)))
    return a == b


def _rows_close(got, want, keys, what):
    equal(len(got), len(want), f"{what}: row count")
    for g, w in zip(got, want):
        for k in keys:
            if not _close(g[k], w[k]):
                raise AssertionError(f"{what}: column {k}: got {g[k]!r}, pandas {w[k]!r} in row {w}")


def run(j):
    import pandas as pd

    import pyursa as pu
    from pyursa import col

    backends = pu.available_backends()
    with j.step("environment has both engines", critical=True):
        check(set(backends) >= {"polars", "duckdb"}, f"backends: {backends}")

    with j.step("create pandas-written Parquet and a CSV export", critical=True):
        pdo, pdc = _make_data(j)

    with j.step("read both files; types are what an analyst expects", critical=True):
        orders = pu.read_parquet(str(j.dir / "orders.parquet"))
        customers = pu.read_csv(str(j.dir / "customers.csv"))
        s = orders.schema
        equal({k: str(v) for k, v in s.items()}, {
            "order_id": "Int64", "customer_id": "Int64", "region": "String", "product": "String",
            "qty": "Int64", "unit_price": "Float64", "discount": "Float64", "status": "String",
            "ordered_at": "Datetime", "ordered_at_local": "Datetime(America/Toronto)"}, "orders schema")
        equal(str(customers.schema["country"]), "String")

    with j.step("CSV keeps 'NA' (Namibia) and treats only empty cells as missing"):
        c = customers.group_by("country").agg(n=pu.len()).sort("country").to_dicts()
        got = {r["country"]: r["n"] for r in c}
        want = pdc.country.where(pdc.country.notna(), None).value_counts(dropna=False).to_dict()
        want = {(None if k is None or k != k else k): v for k, v in want.items()}
        equal(got, want, "country counts")
        check(got.get("NA", 0) > 0, "'NA' kept as a country code")
        equal(customers.filter(col("segment").is_null()).count(), 117, "missing segments")

    with j.step("row counts and basic stats match pandas on every engine"):
        for b in backends:
            equal(orders.count(b), N, f"{b} count")
            st = orders.select(col("qty").sum().alias("q"), col("discount").mean().alias("d"),
                               col("status").null_count().alias("ns") if hasattr(col("status"), "null_count")
                               else col("status").is_null().cast("Int64").sum().alias("ns")).to_dicts(b)[0]
            equal(st["q"], int(pdo.qty.sum()), f"{b} qty sum")
            check(_close(st["d"], float(pdo.discount.mean())), f"{b} discount mean {st['d']} vs {pdo.discount.mean()}")
            equal(st["ns"], int(pdo.status.isna().sum()), f"{b} missing status")

    with j.step("describe_for_llm fits its budget and shows the data honestly"):
        text = orders.describe_for_llm(tokens=400)
        check(len(text) / 4 <= 400 * 1.1, f"summary too long: ~{len(text) / 4:.0f} tokens")
        check("60,000 rows" in text and "region" in text and "⚠" not in text, text[:500])

    def report(b):
        valid = orders.filter(col("status") != "cancelled", col("status").is_not_null(), col("qty") > 0,
                              intent="valid orders")
        enriched = (valid.join(customers, on="customer_id", how="left")
                    .with_columns(revenue=col("qty") * col("unit_price") * (1 - col("discount").fill_null(0.0))))
        return (enriched.group_by("region", "country")
                .agg(revenue=col("revenue").sum(), orders=pu.len(), buyers=col("customer_id").n_unique(),
                     median_qty=col("qty").median())
                .sort("region", "country").to_dicts(b))

    with j.step("revenue report by region and country matches pandas on every engine"):
        v = pdo[(pdo.status.notna()) & (pdo.status != "cancelled") & (pdo.qty > 0)].copy()
        v = v.merge(pdc, on="customer_id", how="left")
        v["revenue"] = v.qty.astype("int64") * v.unit_price.astype("float64") * (1 - v.discount.fillna(0.0))
        v["region"] = v.region.astype(str)
        want = (v.groupby(["region", "country"], dropna=False)
                .agg(revenue=("revenue", "sum"), orders=("order_id", "size"), buyers=("customer_id", "nunique"),
                     median_qty=("qty", "median")).reset_index())
        want["country"] = want.country.where(want.country.notna(), None)
        want = want.sort_values(["region", "country"], na_position="last").to_dict("records")
        results = {b: report(b) for b in backends}
        for b, got in results.items():
            _rows_close(got, want, ["region", "country", "revenue", "orders", "buyers", "median_qty"], b)

    with j.step("top 3 orders per customer (row_number window) match pandas"):
        top = (orders.filter(col("status") == "paid")
               .with_columns(value=col("qty") * col("unit_price"))
               .with_columns(rn=pu.row_number().over("customer_id", order_by=["value", "order_id"], descending=True))
               .filter(col("rn") <= 3).select("customer_id", "order_id", "rn").sort("customer_id", "rn"))
        p = pdo[pdo.status == "paid"].copy()
        p["value"] = p.qty.astype("float64") * p.unit_price.astype("float64")
        p = p.sort_values(["customer_id", "value", "order_id"], ascending=[True, False, False])
        p["rn"] = p.groupby("customer_id").cumcount() + 1
        want = p[p.rn <= 3][["customer_id", "order_id", "rn"]].sort_values(["customer_id", "rn"]).to_dict("records")
        for b in backends:
            got = top.to_dicts(b)
            equal(len(got), len(want), f"{b} rows")
            equal([(r["customer_id"], r["order_id"]) for r in got[:500]],
                  [(int(r["customer_id"]), int(r["order_id"])) for r in want[:500]], f"{b} first rows")

    with j.step("running total per customer matches pandas cumsum"):
        rt = (orders.select("customer_id", "order_id", "qty")
              .with_columns(running=col("qty").cum_sum().over("customer_id", order_by="order_id"))
              .sort("customer_id", "order_id"))
        p = pdo.sort_values(["customer_id", "order_id"])
        want = p.groupby("customer_id").qty.cumsum().astype("int64").tolist()
        for b in backends:
            equal([r["running"] for r in rt.to_dicts(b)], want, f"{b} running totals")

    with j.step("monthly orders (date truncation, both time zones) match pandas"):
        for column, pandas_col in (("ordered_at", pdo.ordered_at), ("ordered_at_local", pdo.ordered_at_local)):
            m = (orders.with_columns(month=col(column).dt.truncate("month"))
                 .group_by("month").agg(n=pu.len()).sort("month"))
            want = pandas_col.dt.tz_localize(None).dt.to_period("M").value_counts().sort_index().tolist()
            for b in backends:
                equal([r["n"] for r in m.to_dicts(b)], want, f"{b} {column}")

    with j.step("Polars and DuckDB return byte-identical Arrow tables for the report"):
        valid = orders.filter(col("status") == "paid").join(customers, on="customer_id")
        q = valid.group_by("segment").agg(n=pu.len(), q=col("qty").sum()).sort("segment")
        tabs = [q.collect(b) for b in backends]
        check(all(t.equals(tabs[0]) for t in tabs), "tables differ between engines")

    with j.step("interop: to_pandas and to_polars round-trip the numbers"):
        df = orders.head(1000).to_pandas()
        check(len(df) == 1000 and df.qty.sum() == pdo.head(1000).qty.sum(), "to_pandas")
        pl_df = orders.head(1000).to_polars()
        check(pl_df.height == 1000, "to_polars")

    with j.step("export to Parquet, CSV and JSON, read back the same rows"):
        part = orders.filter(col("region") == "east").select("order_id", "region", "qty", "discount", "ordered_at").sort("order_id")
        expected = part.to_dicts()
        part.write_parquet(str(j.dir / "east.parquet"))
        equal(pu.read_parquet(str(j.dir / "east.parquet")).sort("order_id").to_dicts(), expected, "parquet roundtrip")
        part.write_csv(str(j.dir / "east.csv"))
        back = pu.read_csv(str(j.dir / "east.csv"), schema={"ordered_at": "Datetime"}).sort("order_id").to_dicts()
        equal(len(back), len(expected), "csv rows")
        equal([r["qty"] for r in back], [r["qty"] for r in expected], "csv qty")
        part.write_json(str(j.dir / "east.json"))
        import json
        rows = [json.loads(line) for line in open(j.dir / "east.json", encoding="utf-8")]
        equal(len(rows), len(expected), "json rows")
        check(all(r["discount"] is None or isinstance(r["discount"], float) for r in rows), "json discount values")

    with j.step("history and lineage explain how revenue was computed"):
        rep = (orders.filter(col("qty") > 0, intent="drop empty orders")
               .with_columns(revenue=col("qty") * col("unit_price"))
               .group_by("region").agg(revenue=col("revenue").sum()))
        lin = rep.lineage("revenue")
        check(set(lin["sources"]) >= {"orders.qty", "orders.unit_price"}, f"lineage {lin}")
        check(any(h.get("intent") == "drop empty orders" for h in rep.history()), "intent recorded")
        equal(rep.undo(1).schema.get("revenue") is not None, True, "undo back to with_columns")

    _ = pd  # pandas is the reference throughout
