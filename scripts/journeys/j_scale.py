"""Journey: the same report at production size (3 million rows), against pandas.

Checks that results stay identical across engines and pandas at scale, that pyursa is faster than
pandas on a typical pipeline, that lazy planning and summaries stay fast on a big file, and that peak
memory stays reasonable. Set PYURSA_JOURNEY_ROWS to change the size.
"""

from __future__ import annotations

import os
import time

from _harness import check, equal

ROWS = int(os.environ.get("PYURSA_JOURNEY_ROWS", "3000000"))


def run(j):
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    import pyursa as pu
    from pyursa import col

    backends = pu.available_backends()
    with j.step(f"write a {ROWS:,}-row orders file and a customer table", critical=True):
        rng = np.random.default_rng(1)
        orders = pa.table({
            "order_id": np.arange(ROWS, dtype="int64"),
            "customer_id": rng.integers(0, 50_000, ROWS),
            "region": rng.choice(np.array(["east", "west", "north", "south", "central"]), ROWS),
            "qty": rng.integers(0, 20, ROWS),
            "price": np.round(rng.random(ROWS) * 200, 2),
            "status": rng.choice(np.array(["paid", "paid", "paid", "refunded", "cancelled"]), ROWS),
        })
        pq.write_table(orders, j.dir / "orders.parquet", row_group_size=250_000)
        cust = pa.table({"customer_id": np.arange(50_000), "segment": np.array(["smb", "mid", "ent"])[np.arange(50_000) % 3]})
        pq.write_table(cust, j.dir / "customers.parquet")

    def pyursa_report(b):
        o = pu.read_parquet(str(j.dir / "orders.parquet"))
        c = pu.read_parquet(str(j.dir / "customers.parquet"))
        return (o.filter(col("status") == "paid", col("qty") > 0).join(c, on="customer_id")
                .with_columns(revenue=col("qty") * col("price"))
                .group_by("region", "segment").agg(revenue=col("revenue").sum(), n=pu.len(),
                                                   buyers=col("customer_id").n_unique())
                .sort("region", "segment").to_dicts(b))

    def pandas_report():
        o = pd.read_parquet(j.dir / "orders.parquet")
        c = pd.read_parquet(j.dir / "customers.parquet")
        o = o[(o.status == "paid") & (o.qty > 0)].merge(c, on="customer_id")
        o["revenue"] = o.qty * o.price
        r = o.groupby(["region", "segment"]).agg(revenue=("revenue", "sum"), n=("order_id", "size"),
                                                 buyers=("customer_id", "nunique")).reset_index()
        return r.sort_values(["region", "segment"]).to_dict("records")

    times = {}
    with j.step("report gives the same numbers on every engine and in pandas"):
        t = time.perf_counter()
        want = pandas_report()
        times["pandas"] = time.perf_counter() - t
        for b in backends:
            pyursa_report(b)  # warm up
            t = time.perf_counter()
            got = pyursa_report(b)
            times[b] = time.perf_counter() - t
            equal(len(got), len(want), f"{b} groups")
            for g, w in zip(got, want):
                equal((g["region"], g["segment"], g["n"], g["buyers"]), (w["region"], w["segment"], w["n"], w["buyers"]), b)
                check(abs(g["revenue"] - w["revenue"]) <= 1e-9 * abs(w["revenue"]), f"{b} revenue {g} vs {w}")
        j.note("time: " + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in times.items())
               + " | speedup: " + ", ".join(f"{b} {times['pandas'] / times[b]:.1f}x" for b in backends))

    with j.step("pyursa is faster than pandas on this pipeline on every engine"):
        for b in backends:
            check(times[b] < times["pandas"], f"{b} {times[b]:.2f}s vs pandas {times['pandas']:.2f}s")

    with j.step("planning and explain() on the big file are instant (no data read)"):
        t = time.perf_counter()
        f = pu.read_parquet(str(j.dir / "orders.parquet")).filter(col("qty") > 3).group_by("region").agg(n=pu.len())
        info = f.explain()
        dt_ = time.perf_counter() - t
        check(dt_ < 0.5, f"explain took {dt_:.2f}s")
        equal(info["output_schema"], {"region": "String", "n": "Int64"}, "schema")

    with j.step("describe_for_llm on 3M rows stays fast and within budget"):
        t = time.perf_counter()
        text = pu.read_parquet(str(j.dir / "orders.parquet")).describe_for_llm(tokens=500)
        dt_ = time.perf_counter() - t
        check(dt_ < 5, f"describe took {dt_:.2f}s")
        check(len(text) / 4 <= 550, f"~{len(text) / 4:.0f} tokens")
        j.note(f"describe_for_llm: {dt_ * 1000:.0f} ms")

    with j.step("window functions at scale agree across engines"):
        o = pu.read_parquet(str(j.dir / "orders.parquet"))
        q = (o.filter(col("customer_id") < 2000)
             .with_columns(rn=pu.row_number().over("customer_id", order_by="order_id"),
                           running=col("qty").cum_sum().over("customer_id", order_by="order_id"))
             .filter(col("rn") <= 5).select("customer_id", "order_id", "rn", "running").sort("customer_id", "rn"))
        tabs = [q.collect(b) for b in backends]
        check(all(t.equals(tabs[0]) for t in tabs), "window results differ")

    with j.step("peak memory stays reasonable"):
        try:
            import resource
        except ImportError:
            from _harness import Skip
            raise Skip("resource module not available on this OS") from None
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if os.uname().sysname != "Darwin" else 1024 * 1024)
        j.note(f"peak RSS {peak_mb:,.0f} MB")
        check(peak_mb < 4000, f"peak memory {peak_mb:.0f} MB")
