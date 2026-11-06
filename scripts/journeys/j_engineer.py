"""Journey: a data engineer builds a daily pipeline over a Parquet "lake".

Files arrive from three different writers (PyArrow, Polars and DuckDB), each with its own storage types:
dictionary/category strings, int32 keys, float32 measures, millisecond and zoned timestamps, decimals
for money. The engineer scans the folder lazily, deduplicates, joins reference data, keeps an exact
money ledger, checks overflow protection, exports the result and verifies it reads back identically
on every engine.
"""

from __future__ import annotations

import datetime as dt
import decimal
import time

from _harness import check, equal, same_table

D = decimal.Decimal


def _write_lake(j):
    import duckdb
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    lake = j.dir / "lake" / "payments"
    lake.mkdir(parents=True)
    base = dt.datetime(2025, 3, 1, tzinfo=dt.timezone.utc)
    rows = []
    for i in range(30_000):
        rows.append({
            "payment_id": i // 1 if i % 50 else i - 1,  # every 50th row duplicates the previous id
            "account": f"A{(i * 7) % 997:04d}",
            "amount": D(((i * 37) % 100_000)) / 100 - D(250),
            "fx_rate": [1.0, 1.25, 0.8, float("nan")][i % 4] if i % 1000 else None,
            "paid_at": base + dt.timedelta(minutes=i),
            "channel": ["card", "wire", "ach", "card"][i % 4],
        })
    chunks = [rows[:10_000], rows[10_000:20_000], rows[20_000:]]
    schema = pa.schema([("payment_id", pa.int32()), ("account", pa.dictionary(pa.int16(), pa.string())),
                        ("amount", pa.decimal128(12, 2)), ("fx_rate", pa.float32()),
                        ("paid_at", pa.timestamp("ms", tz="UTC")), ("channel", pa.string())])
    # writer 1: PyArrow, small row groups, dictionary + int32 + float32 + ms timestamps
    t1 = pa.Table.from_pylist(chunks[0], schema=pa.schema([f if f.name != "account" else pa.field("account", pa.string()) for f in schema]))
    t1 = t1.set_column(1, "account", t1.column("account").dictionary_encode().cast(pa.dictionary(pa.int16(), pa.string())))
    pq.write_table(t1.cast(schema), lake / "part-0.parquet", row_group_size=2_000)
    # writer 2: Polars, Categorical strings
    df2 = pl.from_dicts(chunks[1]).with_columns(pl.col("account").cast(pl.Categorical),
                                                pl.col("payment_id").cast(pl.Int32),
                                                pl.col("fx_rate").cast(pl.Float32),
                                                pl.col("amount").cast(pl.Decimal(12, 2)))
    df2.write_parquet(lake / "part-1.parquet", row_group_size=2_500)
    # writer 3: DuckDB, its own decimal width and TIMESTAMPTZ
    t3 = pa.Table.from_pylist(chunks[2], schema=pa.schema([f if f.name != "account" else pa.field("account", pa.string()) for f in schema]))
    con = duckdb.connect()
    con.register("t3", t3)
    con.execute(f"COPY (SELECT payment_id, account, CAST(amount AS DECIMAL(12,2)) AS amount, fx_rate, paid_at, channel FROM t3) "
                f"TO '{(lake / 'part-2.parquet').as_posix()}' (FORMAT parquet, ROW_GROUP_SIZE 3000)")
    con.close()
    accounts = pa.table({"account": [f"A{i:04d}" for i in range(997)],
                         "owner": [f"Owner {i % 120}" for i in range(997)],
                         "tier": [["gold", "silver", "bronze"][i % 3] for i in range(997)]})
    pq.write_table(accounts, j.dir / "lake" / "accounts.parquet")
    return rows


def run(j):
    import pyursa as pu
    from pyursa import col

    backends = pu.available_backends()

    with j.step("write a Parquet lake from PyArrow, Polars and DuckDB", critical=True):
        rows = _write_lake(j)

    with j.step("scan the folder and a glob lazily; building plans reads no data"):
        t = time.perf_counter()
        lake = pu.read_parquet(str(j.dir / "lake" / "payments"))
        glob = pu.read_parquet(str(j.dir / "lake" / "payments" / "part-*.parquet"))
        plan = lake.filter(col("amount") > 0).group_by("channel").agg(total=col("amount").sum())
        info = plan.explain()
        check(time.perf_counter() - t < 1.0, "plan building took too long")
        equal(info["output_schema"], {"channel": "String", "total": "Decimal(38,2)"}, "explain schema")
        equal(lake.schema, glob.schema, "folder and glob schemas")

    with j.step("files from three writers read as one strict schema"):
        equal({k: str(v) for k, v in lake.schema.items()},
              {"payment_id": "Int64", "account": "String", "amount": "Decimal(12,2)", "fx_rate": "Float64",
               "paid_at": "Datetime(UTC)", "channel": "String"}, "lake schema")
        for b in backends:
            equal(lake.count(b), len(rows), f"{b} rows")

    with j.step("every engine reads the lake identically (whole table)"):
        tabs = [lake.sort("paid_at", "payment_id").collect(b) for b in backends]
        check(all(same_table(t, tabs[0]) for t in tabs), "engines disagree on the raw lake")

    with j.step("deduplicate on payment_id keeping the first payment"):
        dedup = (lake.with_columns(rn=pu.row_number().over("payment_id", order_by="paid_at"))
                 .filter(col("rn") == 1).drop("rn"))
        want = len({r["payment_id"] for r in rows})
        for b in backends:
            equal(dedup.count(b), want, f"{b} unique payments")
        dup_ids = lake.group_by("payment_id").agg(n=pu.len()).filter(col("n") > 1)
        equal(dup_ids.count(), len(rows) - want, "duplicate ids")

    with j.step("exact money ledger per account and tier equals Python Decimal sums"):
        accounts = pu.read_parquet(str(j.dir / "lake" / "accounts.parquet"))
        ledger = (lake.join(accounts, on="account", how="left")
                  .group_by("tier").agg(balance=col("amount").sum(), n=pu.len()).sort("tier"))
        tiers = {f"A{i:04d}": ["gold", "silver", "bronze"][i % 3] for i in range(997)}
        want: dict[str, D] = {}
        for r in rows:
            want[tiers[r["account"]]] = want.get(tiers[r["account"]], D(0)) + r["amount"]
        for b in backends:
            got = {r["tier"]: r["balance"] for r in ledger.to_dicts(b)}
            equal(got, want, f"{b} balances")
            check(all(isinstance(v, D) for v in got.values()), "balances must be exact Decimals")

    with j.step("fees: decimal products and comparisons are exact"):
        fees = lake.with_columns(fee=col("amount") * D("0.0275")).filter(col("fee") > D("10.000001"))
        want = [r["amount"] * D("0.0275") for r in rows if r["amount"] * D("0.0275") > D("10.000001")]
        for b in backends:
            got = sorted(r["fee"] for r in fees.select("fee").to_dicts(b))
            equal(got, sorted(want), f"{b} fees")

    with j.step("NaN and null FX rates: filters, max and counts agree with Python"):
        rates = [r["fx_rate"] for r in rows]
        n_nan = sum(1 for v in rates if v is not None and v != v)
        n_null = sum(1 for v in rates if v is None)
        for b in backends:
            out = lake.select(col("fx_rate").is_null().cast("Int64").sum().alias("nulls"),
                              (col("fx_rate") == float("nan")).cast("Int64").sum().alias("nans"),
                              col("fx_rate").max().alias("mx")).to_dicts(b)[0]
            equal((out["nulls"], out["nans"]), (n_null, n_nan), f"{b} null/NaN counts")
            check(out["mx"] != out["mx"], f"{b}: max with NaN present should be NaN, got {out['mx']}")
            equal(lake.filter(col("fx_rate") > 1.1).count(b), sum(1 for v in rates if v is not None and (v != v or v > 1.1)),
                  f"{b} fx_rate > 1.1 (NaN counts as largest)")

    with j.step("overflow protection: a ledger that can't fit 38 digits raises, never wraps"):
        huge = pu.from_dict({"x": [D("9" * 36 + ".99")]}).with_columns(x=col("x").cast("Decimal(38,2)"))  # the largest Decimal(38,2)
        many = pu.concat([huge] * 50)
        for b in backends:
            try:
                many.select(col("x").sum()).collect(b)
                raise AssertionError(f"{b}: no error for a sum that needs 39 digits")
            except pu.PyursaError as e:
                equal(e.code, "DECIMAL_OVERFLOW", f"{b} code")
        big = pu.from_dict({"n": [2**62, 2**62]})
        for b in backends:
            try:
                big.select(col("n").sum()).collect(b)
                raise AssertionError(f"{b}: Int64 overflow not detected")
            except pu.PyursaError as e:
                equal(e.code, "INTEGER_OVERFLOW", f"{b} code")

    with j.step("daily summary with time zones and windows matches Python"):
        daily = (lake.with_columns(day=col("paid_at").dt.truncate("day"))
                 .group_by("day").agg(n=pu.len(), total=col("amount").sum()).sort("day")
                 .with_columns(running=col("total").cum_sum().over(order_by="day")))
        days: dict = {}
        for r in rows:
            k = r["paid_at"].replace(hour=0, minute=0)
            days[k] = days.get(k, D(0)) + r["amount"]
        running, acc = [], D(0)
        for k in sorted(days):
            acc += days[k]
            running.append(acc)
        for b in backends:
            got = daily.to_dicts(b)
            equal([r["day"] for r in got], sorted(days), f"{b} days")
            equal([r["running"] for r in got], running, f"{b} running balance")

    with j.step("export the curated table and read it back identically on every engine"):
        curated = (lake.join(pu.read_parquet(str(j.dir / "lake" / "accounts.parquet")), on="account")
                   .select("payment_id", "account", "owner", "tier", "amount", "fx_rate", "paid_at")
                   .sort("payment_id", "paid_at"))
        curated.write_parquet(str(j.dir / "curated.parquet"))
        back = pu.read_parquet(str(j.dir / "curated.parquet")).sort("payment_id", "paid_at")
        equal(back.schema, curated.schema, "schema after export")
        ref = curated.collect(backends[0])
        for b in backends:
            check(same_table(back.collect(b), ref), f"{b}: re-read differs from what was written")

    with j.step("anti join finds payments for unknown accounts (none expected), semi join keeps known"):
        accounts = pu.read_parquet(str(j.dir / "lake" / "accounts.parquet"))
        for b in backends:
            equal(lake.join(accounts, on="account", how="anti").count(b), 0, f"{b} anti")
            equal(lake.join(accounts, on="account", how="semi").count(b), len(rows), f"{b} semi")

    with j.step("rewriting a source file is picked up by the same lazy frame"):
        import pyarrow as pa
        import pyarrow.parquet as pq
        p = j.dir / "small.parquet"
        pq.write_table(pa.table({"v": [1, 2]}), p)
        f = pu.read_parquet(str(p)).select(col("v").sum().alias("s"))
        equal(f.to_dicts()[0]["s"], 3, "first read")
        time.sleep(0.01)
        pq.write_table(pa.table({"v": [10, 20]}), p)
        for b in backends:
            equal(f.to_dicts(b)[0]["s"], 30, f"{b} after rewrite")
