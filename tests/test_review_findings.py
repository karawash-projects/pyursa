"""Regression tests for the independent pre-release review (1.0.0rc1). Each test is a
reported reproduction; every backend must give the stated (identical) behavior."""

import datetime as dt
import decimal as D
import math
import os
import zoneinfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyursa as pu
from pyursa import col, lit
from pyursa.tools import ToolSession

BACKENDS = pu.available_backends()
NAN = float("nan")


def both(frame):
    out = [frame.to_dicts(b) for b in BACKENDS]
    for o in out[1:]:
        assert _same(o, out[0]), (o, out[0])
    return out[0]


def _same(a, b):
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12) or a == b
    return a == b


def err_code(fn):
    with pytest.raises(pu.PyursaError) as ei:
        fn()
    return ei.value.code


# 1. names differing only by case
def test_case_colliding_names_are_rejected():
    assert err_code(lambda: pu.from_arrow(pa.table({"A": [1], "a": [2]}))) == "DUPLICATE_COLUMN"
    df = pu.from_dict({"A": [1]})
    assert err_code(lambda: df.with_columns(a=col("A"))) == "DUPLICATE_COLUMN"
    assert err_code(lambda: df.rename({"A": "a"}).with_columns(A=col("a"))) == "DUPLICATE_COLUMN"


# 2. glob characters in sandboxed paths
def test_bracket_paths_are_literal(tmp_path):
    root = tmp_path / "data[1]"
    root.mkdir()
    (tmp_path / "data1").mkdir()
    pq.write_table(pa.table({"ok": [1]}), root / "a.parquet")
    pq.write_table(pa.table({"SECRET": [999]}), tmp_path / "data1" / "a.parquet")
    r = ToolSession(roots=[root]).load_file("a.parquet")
    assert r["ok"] and r["schema"] == {"ok": "Int64"}
    (tmp_path / "x1.parquet").write_bytes((tmp_path / "data1" / "a.parquet").read_bytes())
    pq.write_table(pa.table({"literal": [1]}), tmp_path / "x[1].parquet")
    assert pu.read_parquet(str(tmp_path / "x[1].parquet")).columns == ["literal"]


@pytest.mark.skipif(os.name == "nt", reason="symlinks need admin rights on Windows")
def test_directory_scan_with_brackets_cannot_follow_outside(tmp_path):
    root = tmp_path / "root"
    (root / "l[n]k").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    pq.write_table(pa.table({"s": ["secret"]}), outside / "s.parquet")
    os.symlink(outside, root / "lnk")
    pq.write_table(pa.table({"v": [1]}), root / "l[n]k" / "ok.parquet")
    r = ToolSession(roots=[root]).load_file(".")
    assert r["ok"] is False or r["schema"] == {"v": "Int64"}


# 3. engine panics never escape ToolSession; bad decimal literals rejected at build time
def test_tool_session_never_raises(tmp_path):
    pq.write_table(pa.table({"x": [D.Decimal("1.5")]}, schema=pa.schema([("x", pa.decimal128(10, 1))])), tmp_path / "d.parquet")
    s = ToolSession(roots=[tmp_path])
    h = s.load_file("d.parquet")["handle"]
    for cond in ('col("x") == decimal("NaN")', 'col("x") == decimal("Infinity")', 'col("x") == decimal(1e400)'):
        r = s.transform(h, [{"op": "filter", "condition": cond}])
        assert r["ok"] is False, cond
    r = s.transform(h, [{"op": "rename", "mapping": {"x": "a\x00b"}}])
    assert r["ok"] is False


# 4. integer overflow is an error on every engine
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("build", [
    lambda f: f.agg(s=col("x").sum()),
    lambda f: f.select(y=col("x") + 1),
    lambda f: f.select(y=col("x") * 2),
    lambda f: f.select(y=-col("m")),
    lambda f: f.select(y=col("m").abs()),
    lambda f: f.select(y=col("x").cum_sum().over(order_by="x")),
    lambda f: f.select(y=col("x").sum().over()),
])
def test_integer_overflow_raises(backend, build):
    f = pu.from_dict({"x": [2 ** 63 - 1, 1], "m": [-(2 ** 63), 0]})
    with pytest.raises(pu.PyursaError) as ei:
        build(f).collect(backend)
    assert ei.value.code == "INTEGER_OVERFLOW"


def test_large_but_valid_integer_math():
    f = pu.from_dict({"x": [2 ** 62, 2 ** 62 - 1]})
    assert both(f.agg(s=col("x").sum())) == [{"s": 2 ** 63 - 1}]


# 5. decimal literals with exponents
def test_decimal_exponent_literals():
    f = pu.from_dict({"x": [1000, 1500]})
    assert both(f.filter(col("x") == lit(D.Decimal("1E+3")))) == [{"x": 1000}]
    assert both(f.select(y=col("x") * D.Decimal("1E+2"))) == [{"y": D.Decimal("100000")}, {"y": D.Decimal("150000")}]


# 6. casts: one strict contract
def test_float_to_int_requires_whole_numbers():
    f = pu.from_dict({"x": [2.0, -3.0, None]})
    assert both(f.select(y=col("x").cast("Int64"))) == [{"y": 2}, {"y": -3}, {"y": None}]
    for b in BACKENDS:
        for bad in ([2.7], [NAN], [float("inf")]):
            assert err_code(lambda b=b, bad=bad: pu.from_dict({"x": bad}).select(col("x").cast("Int64")).collect(b)) == "CAST_FAILED"
    assert both(pu.from_dict({"x": [2.7, -2.7]}).select(y=col("x").round(0).cast("Int64"))) == [{"y": 3}, {"y": -3}]


def test_decimal_to_int_requires_whole_numbers():
    f = pu.from_arrow(pa.table({"d": pa.array([D.Decimal("3.00"), D.Decimal("-100.00")], pa.decimal128(10, 2))}))
    assert both(f.select(i=col("d").cast("Int64"))) == [{"i": 3}, {"i": -100}]
    g = pu.from_arrow(pa.table({"d": pa.array([D.Decimal("2.70")], pa.decimal128(10, 2))}))
    for b in BACKENDS:
        assert err_code(lambda b=b: g.select(col("d").cast("Int64")).collect(b)) == "CAST_FAILED"


def test_text_parsing_casts():
    f = pu.from_dict({"s": [" 2", "+3", "-4", None]})
    assert both(f.select(i=col("s").cast("Int64"), x=col("s").cast("Float64"))) == [
        {"i": 2, "x": 2.0}, {"i": 3, "x": 3.0}, {"i": -4, "x": -4.0}, {"i": None, "x": None}]
    g = pu.from_dict({"s": ["1e3", ".5", "inf", "-Infinity", "NaN"]})
    out = both(g.select(x=col("s").cast("Float64")))
    assert [r["x"] for r in out][:4] == [1000.0, 0.5, float("inf"), float("-inf")] and math.isnan(out[4]["x"])
    assert both(pu.from_dict({"s": ["TRUE", "false", "1", "0"]}).select(b=col("s").cast("Boolean"))) == [
        {"b": True}, {"b": False}, {"b": True}, {"b": False}]
    assert both(pu.from_dict({"s": ["2024-01-05", " 2024-12-31 "]}).select(d=col("s").cast("Date"))) == [
        {"d": dt.date(2024, 1, 5)}, {"d": dt.date(2024, 12, 31)}]
    assert both(pu.from_dict({"s": ["2024-01-05T10:11", "2024-01-05 10:11:12.5"]}).select(t=col("s").cast("Datetime"))) == [
        {"t": dt.datetime(2024, 1, 5, 10, 11)}, {"t": dt.datetime(2024, 1, 5, 10, 11, 12, 500000)}]
    assert both(pu.from_dict({"s": ["1.255", "3"]}).select(d=col("s").cast("Decimal(10,2)"))) == [
        {"d": D.Decimal("1.26")}, {"d": D.Decimal("3.00")}]
    for bad, to in [("1.5", "Int64"), ("1e3", "Int64"), ("abc", "Float64"), ("yes", "Boolean"),
                    ("2024-1-5", "Date"), ("2024-02-30", "Date"), ("12/01/2024", "Date"), ("x", "Decimal(10,2)")]:
        for b in BACKENDS:
            code = err_code(lambda b=b, bad=bad, to=to: pu.from_dict({"s": [bad]}).select(col("s").cast(to)).collect(b))
            assert code in ("CAST_FAILED", "BACKEND_ERROR"), (bad, to, b, code)


def test_to_text_casts_match():
    f = pu.from_dict({"x": [1.5e-7, 1e20, 0.1, NAN, float("-inf"), 1e-5], "b": [True, False, None, True, False, None]})
    assert both(f.select(t=col("b").cast("String")))[:2] == [{"t": "true"}, {"t": "false"}]
    for b in BACKENDS:  # contract: float text round-trips exactly (notation may differ between engines)
        rt = f.select("x", back=col("x").cast("String").cast("Float64")).to_dicts(b)
        assert all(_same(r["x"], r["back"]) for r in rt), (b, rt)
        texts = [r["s"] for r in f.select(s=col("x").cast("String")).to_dicts(b)]
        assert texts[3] == "NaN" and texts[4] == "-inf", (b, texts)
    g = pu.from_dict({"t": [dt.datetime(2024, 1, 5, 10, 11, 12, 500000), dt.datetime(2024, 1, 5, 10, 11, 12)]})
    assert both(g.select(s=col("t").cast("String"))) == [{"s": "2024-01-05 10:11:12.500000"}, {"s": "2024-01-05 10:11:12"}]
    m = pu.from_arrow(pa.table({"d": pa.array([D.Decimal("1.255")], pa.decimal128(10, 3))}))
    assert both(m.select(r=col("d").cast("Decimal(10,2)"), f=col("d").cast("Float64").cast("Decimal(10,1)"))) == [
        {"r": D.Decimal("1.26"), "f": D.Decimal("1.3")}]


def test_disallowed_casts_fail_at_build_time():
    f = pu.from_dict({"x": [1.5], "d": [dt.date(2024, 1, 1)]})
    for bad in (lambda: f.select(col("x").cast("Boolean")), lambda: f.select(col("d").cast("Int64"))):
        assert err_code(bad) == "INVALID_OPERATION"
    z = pu.from_arrow(pa.table({"t": pa.array([dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)], pa.timestamp("us", "UTC"))}))
    assert err_code(lambda: z.select(col("t").cast("String"))) == "INVALID_OPERATION"


# 7. NaN: one total order everywhere (NaN is the largest value and equals itself)
def test_nan_semantics_identical(tmp_path):
    data = {"x": [1.0, NAN, None, -2.0], "g": ["a", "a", "b", "b"]}
    mem = pu.from_dict(data)
    pq.write_table(pa.table(data), tmp_path / "n.parquet")
    disk = pu.read_parquet(str(tmp_path / "n.parquet"))
    for f in (mem, disk):
        out = both(f.filter(col("x") > 0).select("x"))
        assert out[0]["x"] == 1.0 and math.isnan(out[1]["x"]) and len(out) == 2
        agg = both(f.agg(mn=col("x").min(), mx=col("x").max()))[0]
        assert agg["mn"] == -2.0 and math.isnan(agg["mx"])
        assert math.isnan(both(f.group_by("g").agg(mx=col("x").max()))[0]["mx"])
        assert math.isnan(both(f.select(m=col("x").max().over("g")))[0]["m"])
    assert both(pu.from_dict({"x": [1.0, float("inf")]}).agg(s=col("x").std())) [0]["s"] != 0
    assert mem.describe_for_llm(500, backend="polars") == mem.describe_for_llm(500, backend="duckdb")


# 8. DST: truncating to the hour across a fall-back
def test_hour_truncation_across_dst():
    ny = zoneinfo.ZoneInfo("America/New_York")
    instants = [dt.datetime(2024, 11, 3, h, 30, tzinfo=dt.timezone.utc) for h in (4, 5, 6)]  # 00:30, 01:30 EDT, 01:30 EST
    f = pu.from_arrow(pa.table({"t": pa.array(instants, pa.timestamp("us", "America/New_York"))}))
    out = both(f.select(h=col("t").dt.truncate("hour")))
    assert [r["h"].astimezone(dt.timezone.utc).hour for r in out] == [4, 5, 6]
    assert len(both(f.group_by(f.select(h=col("t").dt.truncate("hour")).columns[0]).agg(n=pu.len()) if False else
                    f.with_columns(h=col("t").dt.truncate("hour")).group_by("h").agg(n=pu.len()))) == 3
    assert ny  # zone used above


# 9. reserved internal names
def test_reserved_names_rejected():
    for name in ("__pu_rn", "__pu_idx", "__pu_l"):
        assert err_code(lambda name=name: pu.from_dict({name: [1]})) == "INVALID_OPERATION"
        assert err_code(lambda name=name: pu.from_dict({"x": [1]}).with_columns(**{name: col("x")})) == "INVALID_OPERATION"


# 10. bad LLM answers become null failures, never crash
@pytest.mark.parametrize("dtype, answer", [("Decimal(10,2)", "abc"), ("Int64", "inf"), ("Int64", "1e400"),
                                           ("Int64", 10 ** 30), ("Decimal(10,2)", "123456789012"), ("Float64", "nan")])
def test_bad_llm_answers_are_failures(dtype, answer):
    from pyursa import semantic as S
    saved = S.CONFIG
    S.CONFIG = S.LLMConfig(client=pu.FunctionClient(lambda r: answer))
    try:
        out = pu.from_dict({"t": ["a"]}).llm_map("x", output="y", dtype=dtype).to_dicts()
        assert out == [{"t": "a", "y": None}] and pu.llm_reports()[0]["failed_items"] == 1
    finally:
        S.CONFIG = saved


def test_large_llm_integers_are_exact():
    from pyursa import semantic as S
    saved = S.CONFIG
    S.CONFIG = S.LLMConfig(client=pu.FunctionClient(lambda r: "9007199254740993"))
    try:
        assert pu.from_dict({"t": ["a"]}).llm_map("x", output="y", dtype="Int64").to_dicts()[0]["y"] == 9007199254740993
    finally:
        S.CONFIG = saved


# 11. is_in
def test_is_in_consistent():
    f = pu.from_dict({"x": [1, 2, None]})
    assert both(f.select(y=col("x").is_in([1.0, 2.5]))) == [{"y": True}, {"y": False}, {"y": None}]
    assert both(f.select(y=col("x").is_in([]))) == [{"y": False}, {"y": False}, {"y": None}]


# 12. Null-typed columns
def test_null_columns_on_every_backend():
    f = pu.from_dict({"x": [1, 2]})
    out = both(f.with_columns(n=lit(None)))
    assert out == [{"x": 1, "n": None}, {"x": 2, "n": None}]
    assert both(pu.from_arrow(pa.table({"n": pa.nulls(2)})).agg(m=col("n").max())) == [{"m": None}]


# 13. error fixes that actually work
def test_suggested_fixes_work():
    from pyursa.parse import parse_expr
    f = pu.from_dict({"qty": [2, 3], "price": [1.5, 2.0], "k": ["a", "b"], "d": ["2024-01-01", "2024-02-01"]})
    with pytest.raises(pu.PyursaError) as ei:
        f.group_by("k").agg(revenue=col("qty") * col("price"))
    fix = ei.value.suggestions[0]["fix"]
    assert both(f.group_by("k").agg(parse_expr(fix).alias("revenue"))) == [{"k": "a", "revenue": 3.0}, {"k": "b", "revenue": 6.0}]
    with pytest.raises(pu.PyursaError) as ei:
        f.select(col("d").dt.year())
    both(f.select(parse_expr(ei.value.suggestions[0]["fix"]).dt.year()))
    with pytest.raises(pu.PyursaError) as ei:
        f.group_by("k").agg(col("k").count())
    parse_expr(ei.value.suggestions[0]["fix"])


# 14/15. input validation in tools and frames
def test_tool_session_bad_arguments(tmp_path):
    pq.write_table(pa.table({"x": [1]}), tmp_path / "a.parquet")
    s = ToolSession(roots=[tmp_path])
    h = s.load_file("a.parquet")["handle"]
    for r in (s.transform(h, "not a list"), s.transform(h, [5]), s.transform(h, [{"op": "filter", "condition": 5}]),
              s.transform(h, [{"op": "with_columns", "columns": ["x"]}]), s.preview(h, rows=float("inf")),
              s.lineage(h, column=5), s.set_backend(None), s.describe(h, tokens=-1)):
        assert r["ok"] is False and r["error"]["error"] in ("BAD_ARGUMENTS", "INVALID_OPERATION"), r
    f = pu.from_dict({"x": [1]})
    for bad in (lambda: f.head(10 ** 30), lambda: f.rename({"x": 1}), lambda: f.checkout(""),
                lambda: f.describe_for_llm(tokens=-1), lambda: pu.set_default_backend(None)):
        with pytest.raises(pu.PyursaError):
            bad()


# 16/17. time-zone literals
def test_fixed_offset_and_nonexistent_literals():
    utc5 = dt.timezone(dt.timedelta(hours=5))
    f = pu.from_arrow(pa.table({"t": pa.array([dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)], pa.timestamp("us", "+05:00"))}))
    assert f.schema["t"] == pu.Datetime("Etc/GMT-5")
    assert both(f.select(h=col("t").dt.hour())) == [{"h": 5}]
    assert both(f.filter(col("t") == lit(dt.datetime(2026, 1, 1, 5, tzinfo=utc5)))) != []
    ny = zoneinfo.ZoneInfo("America/New_York")
    g = pu.from_arrow(pa.table({"t": pa.array([dt.datetime(2024, 3, 10, 7, tzinfo=dt.timezone.utc)], pa.timestamp("us", "America/New_York"))}))
    both(g.filter(col("t") >= lit(dt.datetime(2024, 3, 10, 2, 0, tzinfo=ny))))  # nonexistent local time: same instant everywhere


# --- second review round ------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_engines_read_bracket_paths_literally(tmp_path, backend):
    (tmp_path / "data[1]").mkdir()
    (tmp_path / "data1").mkdir()
    pq.write_table(pa.table({"v": ["public"]}), tmp_path / "data[1]" / "a.parquet")
    pq.write_table(pa.table({"v": ["SECRET"]}), tmp_path / "data1" / "a.parquet")
    s = ToolSession(roots=[tmp_path / "data[1]"], backend=backend)
    for target in ("a.parquet", "."):
        h = s.load_file(target)["handle"]
        assert s.preview(h)["rows"] == [{"v": "public"}], target
    assert pu.read_parquet(str(tmp_path / "data[1]" / "a.parquet")).to_dicts(backend) == [{"v": "public"}]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("build", [
    lambda f: f.filter(col("x") + 1 > 0),
    lambda f: f.with_columns(t=(col("x") + 1) > 0),
    lambda f: f.with_columns(t=pu.when(col("x") + 1 > 0).then(1).otherwise(0)),
    lambda f: f.filter(-col("m") > 0),
])
def test_overflow_inside_comparisons_raises(backend, build):
    f = pu.from_dict({"x": [2 ** 63 - 1, 1], "m": [-(2 ** 63), 0]})
    with pytest.raises(pu.PyursaError) as ei:
        build(f).collect(backend)
    assert ei.value.code == "INTEGER_OVERFLOW"


def test_std_with_nan_matches():
    out = both(pu.from_dict({"g": [1, 1, 2, 2], "x": [NAN, 1.0, 2.0, 3.0]}).group_by("g").agg(s=col("x").std()))
    assert math.isnan(out[0]["s"]) and math.isclose(out[1]["s"], 0.7071067811865476)


@pytest.mark.parametrize("bad", ["2024-01-05 24:00:00", "2024-01-05 23:59:60", "2024-01-05 23:59:60.5", "0000-01-01 00:00"])
def test_datetime_text_edge_cases_rejected(bad):
    for b in BACKENDS:
        assert err_code(lambda b=b: pu.from_dict({"s": [bad]}).select(col("s").cast("Datetime")).collect(b)) == "CAST_FAILED"
    for b in BACKENDS:
        assert err_code(lambda b=b: pu.from_dict({"s": ["0000-01-01"]}).select(col("s").cast("Date")).collect(b)) == "CAST_FAILED"


def test_decimal_text_rounding_carry():
    out = both(pu.from_dict({"s": ["99999999999999999999.5", "-0.5"]}).select(d=col("s").cast("Decimal(38,0)")))
    assert out == [{"d": D.Decimal("100000000000000000000")}, {"d": D.Decimal("-1")}]


def test_error_classification_ignores_column_names():
    f = pu.from_dict({"overflow_count": ["ab"]})
    with pytest.raises(pu.InvalidOperationError):  # non-portable regex is rejected when the plan is built
        f.filter(col("overflow_count").str.contains_regex("a(?=b)"))


def test_huge_decimal_literals_rejected_cleanly():
    from pyursa.parse import parse_expr
    for src in ('col("x") == decimal("1e40")', 'col("x") == decimal("1e39")'):
        with pytest.raises(pu.PyursaError):
            parse_expr(src)
    with pytest.raises(pu.UnsupportedTypeError):
        lit(D.Decimal("1e40"))


def test_tool_session_prefers_literal_names(tmp_path):
    pq.write_table(pa.table({"v": ["bracket file"]}), tmp_path / "x[1].parquet")
    pq.write_table(pa.table({"v": ["other file"]}), tmp_path / "x1.parquet")
    s = ToolSession(roots=[tmp_path])
    assert s.preview(s.load_file("x[1].parquet")["handle"])["rows"] == [{"v": "bracket file"}]
    assert s.load_file("x*.parquet")["files"] == 2
