"""Every backend must return identical results under the semantics contract
(pu.SEMANTICS). Each case states the expected output explicitly, so a new
backend is correct when it passes this file unchanged."""

import math

import pytest

import pyursa as pu
from pyursa import col, lit

from .conftest import BACKENDS


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12) or a == b
    return a == b


def assert_rows(actual, expected):
    assert len(actual) == len(expected), f"{actual} != {expected}"
    for ra, re in zip(actual, expected):
        assert ra.keys() == re.keys(), f"{list(ra)} != {list(re)}"
        for k in ra:
            assert same(ra[k], re[k]), f"column {k}: {ra[k]!r} != {re[k]!r}\nfull: {actual}"


CASES = {
    "filter_drops_null_conditions": (
        lambda o, c: o.filter(col("sales") > 5).select("id"),
        [{"id": 1}, {"id": 2}, {"id": 5}, {"id": 6}],
    ),
    "division_is_float_ieee": (
        lambda o, c: o.select("id", (col("sales") / col("price")).alias("r")).filter(col("id").is_in([1, 6])),
        [{"id": 1, "r": 10 / 1.5}, {"id": 6, "r": float("inf")}],
    ),
    "sum_of_only_nulls_is_null": (
        lambda o, c: o.filter(col("id") == 3).agg(col("sales").sum().alias("s")),
        [{"s": None}],
    ),
    "sum_over_no_rows_is_null_len_zero": (
        lambda o, c: o.filter(col("id") > 100).agg(col("sales").sum().alias("s"), pu.len().alias("n")),
        [{"s": None, "n": 0}],
    ),
    "n_unique_ignores_null": (
        lambda o, c: o.agg(col("region").n_unique().alias("u"), col("sales").n_unique().alias("v")),
        [{"u": 2, "v": 3}],
    ),
    "count_vs_len": (
        lambda o, c: o.agg(col("sales").count().alias("c"), pu.len().alias("n")),
        [{"c": 5, "n": 6}],
    ),
    "sort_stable_nulls_last_asc": (
        lambda o, c: o.sort("sales").select("id"),
        [{"id": 4}, {"id": 1}, {"id": 6}, {"id": 2}, {"id": 5}, {"id": 3}],
    ),
    "sort_stable_nulls_last_desc": (
        lambda o, c: o.sort("sales", descending=True).select("id"),
        [{"id": 2}, {"id": 5}, {"id": 1}, {"id": 6}, {"id": 4}, {"id": 3}],
    ),
    "group_by_sorted_by_key_nulls_last": (
        lambda o, c: o.group_by("region").agg(pu.len().alias("n"), col("sales").sum().alias("s"),
                                             col("price").mean().alias("p")),
        [{"region": "east", "n": 3, "s": 20, "p": 1.5},
         {"region": "west", "n": 2, "s": 40, "p": 1.5},
         {"region": None, "n": 1, "s": 5, "p": None}],
    ),
    "inner_join_keeps_left_then_right_order": (
        lambda o, c: o.join(c, on="region").select("id", "manager"),
        [{"id": 1, "manager": "Ana"}, {"id": 2, "manager": "Bo"}, {"id": 2, "manager": "Cy"},
         {"id": 3, "manager": "Ana"}, {"id": 5, "manager": "Bo"}, {"id": 5, "manager": "Cy"},
         {"id": 6, "manager": "Ana"}],
    ),
    "left_join_null_keys_never_match": (
        lambda o, c: o.filter(col("id").is_in([3, 4])).join(c, on="region", how="left").select("id", "region", "manager"),
        [{"id": 3, "region": "east", "manager": "Ana"}, {"id": 4, "region": None, "manager": None}],
    ),
    "join_suffix_on_collision": (
        lambda o, c: o.head(1).join(o.select("id", "sales"), on="id").select("id", "sales", "sales_right"),
        [{"id": 1, "sales": 10, "sales_right": 10}],
    ),
    "with_columns_replace_and_add": (
        lambda o, c: o.with_columns(col("sales") * 2, flag=col("price").is_null()).select("id", "sales", "flag").head(4),
        [{"id": 1, "sales": 20, "flag": False}, {"id": 2, "sales": 40, "flag": False},
         {"id": 3, "sales": None, "flag": False}, {"id": 4, "sales": 10, "flag": True}],
    ),
    "int_plus_float_is_float": (
        lambda o, c: o.select((col("sales") + col("price")).alias("x")).head(1),
        [{"x": 11.5}],
    ),
    "string_ops": (
        lambda o, c: o.filter(col("region").str.contains("as")).select(col("region").str.upper()).head(1),
        [{"region": "EAST"}],
    ),
    "dates_compare_and_minmax": (
        lambda o, c: o.filter(col("day") >= lit(__import__("datetime").date(2026, 1, 4)))
                      .agg(col("day").min().alias("lo"), pu.len().alias("n")),
        [{"lo": __import__("datetime").date(2026, 1, 4), "n": 3}],
    ),
    "rename_drop": (
        lambda o, c: o.rename({"sales": "amount"}).drop("price", "day", "region").head(1),
        [{"id": 1, "amount": 10}],
    ),
    "cast_int_to_string": (
        lambda o, c: o.select(col("id").cast("String")).head(2),
        [{"id": "1"}, {"id": "2"}],
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_case(case, backend, orders, customers):
    build, expected = CASES[case]
    assert_rows(build(orders, customers).to_dicts(backend), expected)


@pytest.mark.parametrize("case", sorted(CASES))
def test_identical_arrow_across_backends(case, orders, customers):
    frame = CASES[case][0](orders, customers)
    tables = [frame.collect(b) for b in BACKENDS]
    for t in tables[1:]:
        assert t.schema.equals(tables[0].schema)
        assert_rows(t.to_pylist(), tables[0].to_pylist())


def test_parquet_scan(tmp_path, backend):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table({"a": pa.array([3, 1, 2], pa.int32()), "b": ["x", "y", "z"]}), path)
    df = pu.read_parquet(str(path))
    assert df.schema == {"a": pu.Int64, "b": pu.String}
    assert df.sort("a").to_dicts(backend) == [{"a": 1, "b": "y"}, {"a": 2, "b": "z"}, {"a": 3, "b": "x"}]


def test_engine_error_is_structured(backend):
    df = pu.from_dict({"s": ["1", "x"]})
    with pytest.raises(pu.BackendError) as ei:
        df.select(col("s").cast("Int64")).collect(backend)
    d = ei.value.to_dict()
    assert d["error"] == "CAST_FAILED" and d["details"]["backend"] == backend
    assert d["details"]["compiled"]
