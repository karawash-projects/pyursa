"""Journey: an agent recovers from its own mistakes using pyursa's structured errors.

The "agent" writes plausible but wrong code, reads each error as JSON, applies the first suggested fix
(parsed with pyursa's safe expression parser, as an MCP client would) and must then succeed. Errors must
be raised while the plan is built (before any data is read), be JSON-serializable, carry a stable code,
and their fixes must work when pasted.
"""

from __future__ import annotations

import json
import time

from _harness import check, equal


def run(j):
    import pyarrow as pa
    import pyarrow.parquet as pq

    import pyursa as pu
    from pyursa import col
    from pyursa.parse import parse_expr

    pq.write_table(pa.table({
        "order_id": [1, 2, 3, 4], "qty": [2, 0, 5, 1], "price": [9.5, 3.0, 2.5, 100.0],
        "status": ["paid", None, "paid", "refunded"], "region": ["east", "west", "east", "north"],
        "customer_id": [10, 11, 10, 12]}), j.dir / "orders.parquet")
    pq.write_table(pa.table({"customer_id": ["10", "11", "12"], "name": ["Ann", "Bo", "Cy"]}), j.dir / "customers.parquet")
    orders = pu.read_parquet(str(j.dir / "orders.parquet"))
    customers = pu.read_parquet(str(j.dir / "customers.parquet"))

    def error_of(fn):
        t = time.perf_counter()
        try:
            fn()
        except pu.PyursaError as e:
            d = e.to_dict()
            json.dumps(d)  # must be serializable for an agent
            check(isinstance(d.get("error"), str) and d["error"].isupper(), f"bad code {d.get('error')}")
            check(isinstance(d.get("message"), str) and d["message"], "empty message")
            return d, time.perf_counter() - t
        raise AssertionError("expected a pyursa error, got none")

    def fix_expr(d):
        fixes = [s["fix"] for s in d.get("suggestions", []) if s.get("fix")]
        check(fixes, f"no paste-ready fix in {d}")
        return parse_expr(fixes[0])

    with j.step("typo in a column name: COLUMN_NOT_FOUND with the right column as the fix"):
        d, dt_ = error_of(lambda: orders.filter(col("quantity") > 0))
        equal(d["error"], "COLUMN_NOT_FOUND")
        check("qty" in d["details"]["available_columns"], "available columns listed")
        fixed = orders.filter(fix_expr(d) > 0)
        equal(fixed.count(), 3, "after fix")
        check(dt_ < 0.05, "error should come from plan building, not execution")

    with j.step("comparing a number with text: TYPE_MISMATCH with a numeric fix"):
        d, _ = error_of(lambda: orders.filter(col("qty") > "1"))
        equal(d["error"], "TYPE_MISMATCH")
        equal(orders.filter(fix_expr(d)).count(), 2, "after fix")

    with j.step("== None: the error explains is_null()"):
        d, _ = error_of(lambda: orders.filter(col("status") == None))  # noqa: E711
        equal(orders.filter(fix_expr(d)).count(), 1, "after fix")

    with j.step("join keys of different types are caught before running, with a cast as fix"):
        d, dt_ = error_of(lambda: orders.join(customers, on="customer_id"))
        equal(d["error"], "TYPE_MISMATCH")
        check(dt_ < 0.05, "caught at plan time")
        fixed_customers = customers.with_columns(customer_id=col("customer_id").cast("Int64"))
        equal(orders.join(fixed_customers, on="customer_id").count(), 4, "after cast")

    with j.step("aggregation inside filter is rejected with guidance"):
        d, _ = error_of(lambda: orders.filter(col("qty").sum() > 3))
        check(d["error"] in ("INVALID_OPERATION", "TYPE_MISMATCH"), d["error"])
        check(d.get("suggestions"), "has suggestions")

    with j.step("a failing cast reports CAST_FAILED at run time on every engine"):
        bad = pu.from_dict({"x": ["1", "two", "3"]}).with_columns(n=col("x").cast("Int64"))
        for b in pu.available_backends():
            d, _ = error_of(lambda b=b: bad.collect(b))
            equal(d["error"], "CAST_FAILED", b)

    with j.step("unknown backend and bad arguments are structured"):
        d, _ = error_of(lambda: orders.collect("spark"))
        equal(d["error"], "UNKNOWN_BACKEND")
        check("duckdb" in d["details"]["available"], "lists available engines")
        d, _ = error_of(lambda: orders.filter("qty > 1"))
        equal(d["error"], "BAD_ARGUMENTS")
        d, _ = error_of(lambda: orders.head(-1))
        check(d["error"] in ("BAD_ARGUMENTS", "INVALID_OPERATION"), d["error"])
        d, _ = error_of(lambda: pu.read_parquet([]))
        check(d["error"] in ("FILE_NOT_FOUND", "BAD_ARGUMENTS"), d["error"])
        d, _ = error_of(lambda: pu.read_parquet(str(j.dir / "missing.parquet")))
        equal(d["error"], "FILE_NOT_FOUND")

    with j.step("an agent loop: four mistakes in a row, each fixed from the error alone"):
        attempts = [
            lambda: orders.select("ordr_id", "qty"),
            lambda: orders.filter(col("region") == "east").select("order_id", "qyt"),
            lambda: orders.group_by("regoin").agg(n=pu.len()),
            lambda: orders.sort("pricee"),
        ]
        for attempt in attempts:
            d, _ = error_of(attempt)
            equal(d["error"], "COLUMN_NOT_FOUND")
            fixed = d["suggestions"][0]["fix"]
            check(parse_expr(fixed).name in orders.schema, f"fix {fixed} names a real column")

    with j.step("a column name with quotes and backslashes still gets a working fix"):
        weird = pu.from_dict({'we"ird\\col': [1, 2]})
        d, _ = error_of(lambda: weird.select('we"ird\\co'))
        equal(weird.select(fix_expr(d)).count(), 2, "fix with quotes")

    with j.step("errors never leak file contents or Python tracebacks"):
        d, _ = error_of(lambda: orders.filter(col("secret_column") > 0))
        text = json.dumps(d)
        check("Traceback" not in text and "paid" not in text, text[:300])

    with j.step("explain() catches a bad plan as a dry run without reading data"):
        t = time.perf_counter()
        info = orders.filter(col("qty") > 0).with_columns(v=col("qty") * col("price")).explain()
        check(time.perf_counter() - t < 0.5, "explain too slow")
        check(info["output_schema"]["v"] == "Float64", info["output_schema"])
        check("compiled" in info or "plan" in info, list(info))
