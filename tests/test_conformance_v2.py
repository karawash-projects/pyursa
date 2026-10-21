"""Conformance for types and operations added in 0.5: every case states the
expected output, and every backend must produce it (and identical Arrow)."""

import datetime as dt
import decimal as D
import math
import zoneinfo

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyursa as pu
from pyursa import col, lit, when

UTC = dt.timezone.utc
TOR = zoneinfo.ZoneInfo("America/Toronto")


def same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    return a == b


def check(frame, expected, backend):
    rows = frame.to_dicts(backend)
    assert len(rows) == len(expected), rows
    for r, e in zip(rows, expected):
        assert list(r) == list(e), (r, e)
        for k in r:
            assert same(r[k], e[k]), f"{k}: {r[k]!r} != {e[k]!r} (all: {rows})"


@pytest.fixture
def money():
    return pu.from_arrow(pa.table({
        "id": [1, 2, 3],
        "a": pa.array([D.Decimal("1.15"), D.Decimal("2.25"), None], pa.decimal128(10, 2)),
        "b": pa.array([D.Decimal("1.155"), D.Decimal("3.000"), D.Decimal("0.001")], pa.decimal128(10, 3)),
    }))


@pytest.fixture
def people():
    return pu.from_dict({
        "id": [1, 2, 3, 4, 5, 6],
        "team": ["x", "x", "y", "x", "y", None],
        "score": [10, 30, 20, 30, None, 5],
        "name": ["  Ana ", "bo", "Cyd", "héllo", None, "Eve"],
        "ok": [True, False, None, True, None, False],
        "day": [dt.date(2026, 3, 4), dt.date(2026, 1, 1), dt.date(2025, 12, 31), dt.date(2026, 3, 9), None,
                dt.date(2026, 2, 28)],
    })


# --- types ------------------------------------------------------------------

def test_decimal_multiplication_is_exact(money, backend):
    f = money.select("id", m=col("a") * col("a"), p=col("a") * col("b"))
    assert f.schema["m"] == pu.Decimal(38, 4) and f.schema["p"] == pu.Decimal(38, 5)
    check(f, [{"id": 1, "m": D.Decimal("1.3225"), "p": D.Decimal("1.32825")},
              {"id": 2, "m": D.Decimal("5.0625"), "p": D.Decimal("6.75000")},
              {"id": 3, "m": None, "p": None}], backend)


def test_decimal_add_int_sum_mean(money, backend):
    f = money.select(s=col("a") + col("b"), i=col("a") + 1, h=col("a") / 2)
    assert f.schema["s"] == pu.Decimal(38, 3) and f.schema["i"] == pu.Decimal(38, 2) and f.schema["h"] == pu.Float64
    check(f.head(1), [{"s": D.Decimal("2.305"), "i": D.Decimal("2.15"), "h": 0.575}], backend)
    agg = money.agg(s=col("a").sum(), m=col("a").mean(), mx=col("b").max())
    assert agg.schema["s"] == pu.Decimal(38, 2)
    check(agg, [{"s": D.Decimal("3.40"), "m": 1.7, "mx": D.Decimal("3.000")}], backend)


def test_decimal_compare_round_and_literal(money, backend):
    check(money.filter(col("a") > 2).select("id"), [{"id": 2}], backend)
    check(money.filter(col("b") == lit(D.Decimal("0.001"))).select("id"), [{"id": 3}], backend)
    check(money.select(r=col("b").round(2)).head(2), [{"r": D.Decimal("1.16")}, {"r": D.Decimal("3.00")}], backend)


def test_decimal_scale_overflow_rejected():
    df = pu.from_arrow(pa.table({"a": pa.array([D.Decimal("1")], pa.decimal128(38, 20))}))
    with pytest.raises(pu.TypeMismatchError):
        df.select(col("a") * col("a"))


def test_timezone_preserved_and_local_parts(backend):
    df = pu.from_arrow(pa.table({"ts": pa.array([dt.datetime(2026, 1, 1, 3, 30, tzinfo=UTC)],
                                               pa.timestamp("us", tz="America/Toronto"))}))
    assert df.schema["ts"] == pu.Datetime("America/Toronto")
    out = df.select("ts", h=col("ts").dt.hour(), d=col("ts").dt.date(), y=col("ts").dt.year()).collect(backend)
    assert out.schema.field("ts").type == pa.timestamp("us", tz="America/Toronto")
    row = out.to_pylist()[0]
    assert row["h"] == 22 and row["d"] == dt.date(2025, 12, 31) and row["y"] == 2025  # local time in Toronto
    check(df.filter(col("ts") > lit(dt.datetime(2025, 12, 31, 22, 0, tzinfo=TOR))).select(n=col("h") if False else col("ts").dt.minute()),
          [{"n": 30}], backend)


def test_timezone_mismatch_is_an_error():
    df = pu.from_arrow(pa.table({"ts": pa.array([dt.datetime(2026, 1, 1, tzinfo=UTC)], pa.timestamp("us", tz="UTC"))}))
    with pytest.raises(pu.TypeMismatchError) as ei:
        df.filter(col("ts") > lit(dt.datetime(2026, 1, 1)))
    assert "time zone" in str(ei.value)


def test_nested_columns_pass_through(backend):
    df = pu.from_arrow(pa.table({
        "id": [2, 1, 3],
        "tags": pa.array([["a", "b"], None, ["c"]], pa.list_(pa.string())),
        "meta": pa.array([{"k": 1, "v": "x"}, {"k": 2, "v": "y"}, None]),
    }))
    assert df.schema["tags"] == pu.List(pu.String)
    out = df.filter(col("id") > 1).sort("id").select("id", "tags", "meta", n=col("tags").is_null())
    check(out, [{"id": 2, "tags": ["a", "b"], "meta": {"k": 1, "v": "x"}, "n": False},
                {"id": 3, "tags": ["c"], "meta": None, "n": False}], backend)
    check(df.agg(c=col("tags").count()), [{"c": 2}], backend)


def test_nested_misuse_is_rejected():
    df = pu.from_arrow(pa.table({"l": pa.array([[1]], pa.list_(pa.int64())), "k": [1]}))
    for bad in (lambda: df.sort("l"), lambda: df.group_by("l").agg(pu.len()), lambda: df.distinct(),
                lambda: df.agg(col("l").n_unique()), lambda: df.filter(col("l") == 1)):
        with pytest.raises(pu.PyursaError):
            bad()


def test_dictionary_columns_decode_to_string(backend):
    df = pu.from_arrow(pa.table({"c": pa.array(["x", "y", "x"]).dictionary_encode()}))
    assert df.schema["c"] == pu.String
    check(df.group_by("c").agg(n=pu.len()), [{"c": "x", "n": 2}, {"c": "y", "n": 1}], backend)


def test_lossy_inputs_are_refused():
    with pytest.raises(pu.UnsupportedTypeError):  # nanoseconds would be truncated
        pu.from_arrow(pa.table({"t": pa.array([1_000_000_001], pa.timestamp("ns"))}))
    pu.from_arrow(pa.table({"t": pa.array([1_000_000_000], pa.timestamp("ns"))}))  # whole microseconds: fine
    with pytest.raises(pu.UnsupportedTypeError):  # uint64 overflow
        pu.from_arrow(pa.table({"u": pa.array([2 ** 64 - 1], pa.uint64())}))
    with pytest.raises(pu.UnsupportedTypeError):
        pu.from_arrow(pa.table({"d": pa.array([D.Decimal("1")], pa.decimal256(50, 2))}))
    with pytest.raises(pu.UnsupportedTypeError):
        pu.from_arrow(pa.table({"b": pa.array([b"x"], pa.binary())}))


def test_type_strings_round_trip():
    for spec in ["Int64", "Decimal(12,2)", "Datetime(UTC)", "Datetime(America/Toronto)", "List(Int64)",
                 "Struct(a: Int64, b: List(String))"]:
        assert str(pu.DType(spec)) == spec
    assert pu.DType("Decimal(12,2)") == pu.Decimal(12, 2) and pu.Datetime("UTC") != pu.Datetime


# --- expressions ----------------------------------------------------------------

def test_when_then_otherwise(people, backend):
    f = people.select("id", band=when(col("score") >= 30).then("high").when(col("score") >= 10).then("mid").otherwise("low"),
                      flag=when(col("ok")).then(1))
    check(f, [{"id": 1, "band": "mid", "flag": 1}, {"id": 2, "band": "high", "flag": None},
              {"id": 3, "band": "mid", "flag": None}, {"id": 4, "band": "high", "flag": 1},
              {"id": 5, "band": "low", "flag": None}, {"id": 6, "band": "low", "flag": None}], backend)


def test_when_type_mismatch(people):
    with pytest.raises(pu.TypeMismatchError):
        people.select(when(col("ok")).then(1).otherwise("x"))


def test_fill_null_coalesce_between_abs(people, backend):
    f = people.select(s=col("score").fill_null(0), t=pu.coalesce(col("team"), lit("?")),
                      b=col("score").is_between(10, 20), a=(-col("score")).abs())
    rows = f.to_dicts(backend)
    assert [r["s"] for r in rows] == [10, 30, 20, 30, 0, 5]
    assert [r["t"] for r in rows] == ["x", "x", "y", "x", "y", "?"]
    assert [r["b"] for r in rows] == [True, False, True, False, None, False]
    assert [r["a"] for r in rows] == [10, 30, 20, 30, None, 5]


def test_round_half_away_from_zero(backend):
    df = pu.from_dict({"f": [2.5, -2.5, 0.5, 1.25, -0.125]})
    check(df.select(r=col("f").round(0)), [{"r": 3.0}, {"r": -3.0}, {"r": 1.0}, {"r": 1.0}, {"r": -0.0}], backend)
    check(df.select(r=col("f").round(1)).filter(col("r") != 0), [{"r": 2.5}, {"r": -2.5}, {"r": 0.5}, {"r": 1.3}, {"r": -0.1}], backend)


def test_string_functions(people, backend):
    f = people.select(
        st=col("name").str.strip(), ln=col("name").str.len(), up=col("name").str.upper(),
        rx=col("name").str.contains_regex(r"^[A-Z]"), ew=col("name").str.ends_with("e"),
        rp=col("name").str.replace("l", "L"), sl=col("name").str.slice(1, 2),
    )
    rows = f.to_dicts(backend)
    assert [r["st"] for r in rows] == ["Ana", "bo", "Cyd", "héllo", None, "Eve"]
    assert [r["ln"] for r in rows] == [6, 2, 3, 5, None, 3]
    assert [r["rx"] for r in rows] == [False, False, True, False, None, True]
    assert [r["ew"] for r in rows] == [False, False, False, False, None, True]
    assert rows[3]["rp"] == "héLLo" and rows[3]["up"] == "HÉLLO"
    assert [r["sl"] for r in rows] == [" A", "o", "yd", "él", None, "ve"]


def test_invalid_regex_rejected(people):
    with pytest.raises(pu.InvalidOperationError):
        people.filter(col("name").str.contains_regex("(unclosed"))


def test_date_functions(people, backend):
    f = people.select("id", y=col("day").dt.year(), m=col("day").dt.month(), d=col("day").dt.day(),
                      w=col("day").dt.weekday(), wk=col("day").dt.truncate("week"), mo=col("day").dt.truncate("month"))
    rows = {r["id"]: r for r in f.to_dicts(backend)}
    assert rows[1] == {"id": 1, "y": 2026, "m": 3, "d": 4, "w": 3, "wk": dt.date(2026, 3, 2), "mo": dt.date(2026, 3, 1)}
    assert rows[3]["w"] == 3 and rows[3]["wk"] == dt.date(2025, 12, 29)   # ISO week starts Monday
    assert rows[4]["w"] == 1 and rows[5]["y"] is None


def test_datetime_truncate_hour(backend):
    df = pu.from_dict({"t": [dt.datetime(2026, 5, 6, 7, 59, 30)]})
    check(df.select(h=col("t").dt.truncate("hour"), s=col("t").dt.second()),
          [{"h": dt.datetime(2026, 5, 6, 7, 0), "s": 30}], backend)


def test_new_aggregations(people, backend):
    f = people.group_by("team").agg(med=col("score").median(), q=col("score").quantile(0.25),
                                    sd=col("score").std(), v=col("score").var(),
                                    an=col("ok").any(), al=col("ok").all())
    x = [{"team": "x", "med": 30.0, "q": 20.0, "sd": math.sqrt(400 / 3), "v": 400 / 3, "an": True, "al": False},
         {"team": "y", "med": 20.0, "q": 20.0, "sd": None, "v": None, "an": None, "al": None},
         {"team": None, "med": 5.0, "q": 5.0, "sd": None, "v": None, "an": False, "al": False}]
    check(f, x, backend)


# --- plan operations ----------------------------------------------------------------

def test_distinct(backend):
    df = pu.from_dict({"k": ["b", None, "b", "a", None], "v": [1, 2, 3, 4, 5]})
    check(df.select("k").distinct(), [{"k": "b"}, {"k": None}, {"k": "a"}], backend)
    check(df.distinct("k"), [{"k": "b", "v": 1}, {"k": None, "v": 2}, {"k": "a", "v": 4}], backend)
    check(df.distinct("k").agg(n=pu.len()), [{"n": 3}], backend)


def test_concat(backend):
    a = pu.from_dict({"x": [3, 1], "y": ["a", "b"]})
    b = pu.from_dict({"y": ["c"], "x": [2]})
    check(pu.concat([a, b]), [{"x": 3, "y": "a"}, {"x": 1, "y": "b"}, {"x": 2, "y": "c"}], backend)
    check(a.concat(b).sort("x").head(1), [{"x": 1, "y": "b"}], backend)
    with pytest.raises(pu.TypeMismatchError):
        a.concat(pu.from_dict({"x": [1.5], "y": ["z"]}))


@pytest.fixture
def lr():
    left = pu.from_dict({"k": [1, 2, None, 4], "l": ["a", "b", "c", "d"]})
    right = pu.from_dict({"k": [4, 5, 1, None, 1], "r": ["x", "y", "z", "n", "w"]})
    return left, right


def test_full_join(lr, backend):
    left, right = lr
    check(left.join(right, on="k", how="full"),
          [{"k": 1, "l": "a", "r": "z"}, {"k": 1, "l": "a", "r": "w"}, {"k": 2, "l": "b", "r": None},
           {"k": None, "l": "c", "r": None}, {"k": 4, "l": "d", "r": "x"},
           {"k": 5, "l": None, "r": "y"}, {"k": None, "l": None, "r": "n"}], backend)


def test_semi_anti_join(lr, backend):
    left, right = lr
    check(left.join(right, on="k", how="semi"), [{"k": 1, "l": "a"}, {"k": 4, "l": "d"}], backend)
    check(left.join(right, on="k", how="anti"), [{"k": 2, "l": "b"}, {"k": None, "l": "c"}], backend)


def test_window_aggregates_keep_row_order(people, backend):
    f = people.select("id", tot=col("score").sum().over("team"), n=pu.len().over("team"),
                      all_max=col("score").max().over())
    check(f, [{"id": 1, "tot": 70, "n": 3, "all_max": 30}, {"id": 2, "tot": 70, "n": 3, "all_max": 30},
              {"id": 3, "tot": 20, "n": 2, "all_max": 30}, {"id": 4, "tot": 70, "n": 3, "all_max": 30},
              {"id": 5, "tot": 20, "n": 2, "all_max": 30}, {"id": 6, "tot": 5, "n": 1, "all_max": 30}], backend)


def test_row_number_rank_dense(people, backend):
    f = people.with_columns(
        rn=pu.row_number().over("team", order_by="score", descending=True),
        rk=pu.rank().over("team", order_by="score", descending=True),
        dr=pu.rank(dense=True).over("team", order_by="score"),
    ).select("id", "rn", "rk", "dr")
    check(f, [{"id": 1, "rn": 3, "rk": 3, "dr": 1}, {"id": 2, "rn": 1, "rk": 1, "dr": 2},
              {"id": 3, "rn": 1, "rk": 1, "dr": 1}, {"id": 4, "rn": 2, "rk": 1, "dr": 2},
              {"id": 5, "rn": 2, "rk": 2, "dr": 2}, {"id": 6, "rn": 1, "rk": 1, "dr": 1}], backend)


def test_cum_sum_and_shift(people, backend):
    f = people.select("id", cs=col("score").cum_sum().over("team", order_by="id"),
                      prev=col("score").shift(1).over("team", order_by="id"),
                      nxt=col("id").shift(-1).over(order_by="id"))
    check(f, [{"id": 1, "cs": 10, "prev": None, "nxt": 2}, {"id": 2, "cs": 40, "prev": 10, "nxt": 3},
              {"id": 3, "cs": 20, "prev": None, "nxt": 4}, {"id": 4, "cs": 70, "prev": 30, "nxt": 5},
              {"id": 5, "cs": 20, "prev": 20, "nxt": 6}, {"id": 6, "cs": 5, "prev": None, "nxt": None}], backend)


def test_window_validation(people):
    for bad in (lambda: people.with_columns(pu.row_number().over("team")),            # needs order_by
                lambda: people.filter(pu.row_number().over("team", order_by="id") == 1),  # window in filter
                lambda: people.with_columns(col("score").sum()),                       # bare aggregation
                lambda: people.group_by("team").agg(col("score").sum().over("team"))):
        with pytest.raises(pu.PyursaError):
            bad()


def test_top_n_per_group_pattern(people, backend):
    best = (people.with_columns(rn=pu.row_number().over("team", order_by="score", descending=True))
                  .filter(col("rn") == 1).select("team", "id"))
    check(best, [{"team": "x", "id": 2}, {"team": "y", "id": 3}, {"team": None, "id": 6}], backend)


# --- I/O ---------------------------------------------------------------------------------

def test_write_and_read_roundtrip(tmp_path, people, backend):
    for ext, reader in (("parquet", pu.read_parquet), ("csv", pu.read_csv), ("json", pu.read_json)):
        path = str(tmp_path / f"p.{ext}")
        n = getattr(people.select("id", "team", "score"), f"write_{ext}")(path, backend=backend)
        assert n == 6
        back = reader(path).sort("id").to_dicts(backend)
        assert [r["team"] for r in back] == ["x", "x", "y", "x", "y", None]
    with pytest.raises(pu.PyursaError):
        people.write_parquet(str(tmp_path / "p.parquet"))  # exists, no overwrite


def test_glob_and_directory_parquet(tmp_path, backend):
    for i in range(3):
        pq.write_table(pa.table({"part": [i] * 2, "v": [i, i + 10]}), tmp_path / f"part-{i}.parquet")
    df = pu.read_parquet(str(tmp_path / "*.parquet"))
    assert df.plan.rows_bound() == 6
    check(df.group_by("part").agg(s=col("v").sum()), [{"part": 0, "s": 10}, {"part": 1, "s": 12}, {"part": 2, "s": 14}], backend)
    assert pu.read_parquet(str(tmp_path)).count(backend) == 6


# --- cross-backend identity for everything above ---------------------------------------

def test_order_elision_does_not_change_results(people, backend):
    a = people.sort("score", descending=True).join(people.select("id", t2=col("team")), on="id").group_by("team").agg(n=pu.len())
    check(a, [{"team": "x", "n": 3}, {"team": "y", "n": 2}, {"team": None, "n": 1}], backend)
