"""Conformance on Parquet FILES, not just in-memory tables.

Files keep their own storage types (dictionary/category, int8-32, uint8-32, float32, s/ms timestamps,
time zones, large strings) and engines read them with their own readers, statistics and pushdown. Every
operation below must give the same answer on every engine, and the same answer as the in-memory frame
built from the same Arrow table."""

import datetime as dt
import decimal
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyursa as pu
from pyursa import col

BACKENDS = pu.available_backends()
N = 40  # rows; files are written with small row groups so statistics and pruning come into play


def _ts(unit, tz=None):
    base = dt.datetime(2024, 3, 10, 1, 30, tzinfo=dt.timezone.utc if tz else None)
    vals = [None if i % 7 == 3 else base + dt.timedelta(hours=i) for i in range(N)]
    return pa.array(vals, pa.timestamp(unit, tz=tz))


def _with_nan(t):
    vals = [None if i % 9 == 4 else (float("nan") if i % 5 == 0 else i * 1.25 - 10) for i in range(N)]
    return pa.array(vals, t)


CASES = {
    "dictionary": pa.array([["a", "b", "c", None][i % 4] for i in range(N)]).dictionary_encode(),
    "large_string": pa.array([f"s{i % 6}" if i % 8 else None for i in range(N)], pa.large_string()),
    "int8": pa.array([(i % 11) - 5 if i % 6 else None for i in range(N)], pa.int8()),
    "int16": pa.array([i * 100 if i % 6 else None for i in range(N)], pa.int16()),
    "int32": pa.array([i * 1000 - 7 if i % 6 else None for i in range(N)], pa.int32()),
    "uint8": pa.array([i % 200 for i in range(N)], pa.uint8()),
    "uint32": pa.array([i * 100_000 for i in range(N)], pa.uint32()),
    "float32_nan": _with_nan(pa.float32()),
    "float64_nan": _with_nan(pa.float64()),
    "ts_s": _ts("s"),
    "ts_ms": _ts("ms"),
    "ts_ms_utc": _ts("ms", "UTC"),
    "ts_us_toronto": _ts("us", "America/Toronto"),
    "date32": pa.array([dt.date(2024, 1, 1) + dt.timedelta(days=i) if i % 5 else None for i in range(N)], pa.date32()),
    "decimal": pa.array([decimal.Decimal(i * 37) / 100 if i % 6 else None for i in range(N)], pa.decimal128(10, 2)),
    "bool": pa.array([None if i % 7 == 0 else i % 3 == 0 for i in range(N)]),
    "list_int32": pa.array([[i, i + 1] if i % 4 else None for i in range(N)], pa.list_(pa.int32())),
}

GROUP = pa.array([["x", "y", "z"][i % 3] for i in range(N)]).dictionary_encode()


def _table(name):
    return pa.table({"g": GROUP, "k": pa.array(range(N), pa.int32()), "v": CASES[name]})


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request, tmp_path_factory):
    name = request.param
    t = _table(name)
    path = tmp_path_factory.mktemp(name) / "t.parquet"
    pq.write_table(t, path, row_group_size=8)
    return name, pu.read_parquet(str(path)), pu.from_arrow(t)


def _ops(name, f):
    v = f.schema["v"]
    yield "identity", f
    yield "is_null", f.filter(col("v").is_null())
    yield "concat_memory", pu.concat([f, pu.from_arrow(f.head(3).collect("polars"))])
    yield "join_memory", f.join(pu.from_dict({"k": [1, 2, 3, 30], "w": ["a", "b", "c", "d"]}), on="k", how="left")
    yield "sort_by_group", f.sort("g", "k")
    if v.kind not in ("List", "Struct"):
        yield "sort", f.sort("v", "k")
        yield "sort_desc", f.sort("v", "k", descending=[True, False])
        yield "group_counts", f.group_by("g").agg(n=pu.len(), u=col("v").n_unique())
        yield "distinct", f.select("v").distinct()
        yield "group_by_value", f.group_by("v").agg(n=pu.len())
    if v.kind in ("Int64", "Float64", "Decimal"):
        yield "filter_gt", f.filter(col("v") > 5)
        yield "filter_lt", f.filter(col("v") < 5)
        yield "minmax", f.group_by("g").agg(lo=col("v").min(), hi=col("v").max())
        yield "arith", f.with_columns(w=col("v") * 3 + 1)
    if v.kind in ("Int64", "Float64"):
        yield "sum_mean", f.group_by("g").agg(s=col("v").sum(), m=col("v").mean())
    if v.kind == "Float64":
        yield "eq_nan", f.filter(col("v") == float("nan"))
        yield "gt_big", f.filter(col("v") > 1e6)
    if v.kind == "String":
        yield "upper", f.with_columns(u=col("v").str.upper())
        yield "eq_str", f.filter(col("v") == "b")
        yield "contains", f.filter(col("v").str.contains("1"))
    if v.kind in ("Datetime", "Date"):
        yield "dt_year", f.with_columns(y=col("v").dt.year())
        yield "minmax_dt", f.group_by("g").agg(lo=col("v").min(), hi=col("v").max())
    if v.kind == "Boolean":
        yield "filter_bool", f.filter(col("v"))


def _norm(x):
    if isinstance(x, float):
        return ("nan",) if math.isnan(x) else round(x, 9)
    if isinstance(x, list):
        return [_norm(i) for i in x]
    if isinstance(x, dict):
        return {k: _norm(i) for k, i in x.items()}
    return x


def test_file_and_memory_agree_on_every_engine(case):
    name, file_frame, mem_frame = case
    assert file_frame.schema == mem_frame.schema, name
    for op, fq in _ops(name, file_frame):
        mq = dict(_ops(name, mem_frame))[op]
        expected = None
        for b in BACKENDS:
            for label, q in (("file", fq), ("memory", mq)):
                t = q.collect(b)
                assert pu.types.schema_from_arrow(t.schema) == q.schema, (name, op, b, label, t.schema)
                rows = _norm(t.to_pylist())
                if expected is None:
                    expected = rows
                assert rows == expected, (name, op, b, label)


def test_nan_parquet_row_groups_not_pruned(tmp_path):
    # min/max statistics ignore NaN; an engine that prunes on them drops NaN rows
    p = tmp_path / "n.parquet"
    pq.write_table(pa.table({"x": [float("nan"), 50.0]}), p)
    f = pu.read_parquet(str(p))
    for b in BACKENDS:
        assert f.filter(col("x") > 100).collect(b).num_rows == 1, b
        assert f.filter(col("x") == float("nan")).collect(b).num_rows == 1, b
        assert f.select(col("x").max().alias("m")).collect(b).column(0)[0].as_py() != 50.0, b
