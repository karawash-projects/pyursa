import json

import pytest

import pyursa as pu
from pyursa import col


def err(fn):
    with pytest.raises(pu.PyursaError) as ei:
        fn()
    d = ei.value.to_dict()
    json.dumps(d)  # always serializable
    return d


def test_errors_raise_at_build_time_not_collect(orders):
    # no backend is touched: the mistake is caught while building the plan
    d = err(lambda: orders.filter(col("salez") > 1))
    assert d["error"] == "COLUMN_NOT_FOUND"
    assert d["suggestions"][0]["fix"] == 'col("sales")'
    assert "sales" in d["details"]["available_columns"]


def test_case_insensitive_suggestion(orders):
    assert err(lambda: orders.select("Region"))["suggestions"][0]["fix"] == 'col("region")'


def test_type_mismatch_suggests_numeric_literal(orders):
    d = err(lambda: orders.filter(col("sales") > "5"))
    assert d["error"] == "TYPE_MISMATCH"
    assert d["suggestions"][0]["fix"] == 'col("sales") > 5'


def test_compare_to_none(orders):
    d = err(lambda: orders.filter(col("region") != None))  # noqa: E711
    assert d["suggestions"][0]["fix"] == 'col("region").is_not_null()'


def test_python_and_is_rejected(orders):
    d = err(lambda: orders.filter((col("sales") > 1) and (col("price") > 1)))
    assert "&" in d["suggestions"][0]["description"]


def test_non_boolean_filter(orders):
    assert err(lambda: orders.filter(col("sales")))["error"] == "TYPE_MISMATCH"


def test_mixed_agg_select(orders):
    assert err(lambda: orders.select(col("sales").sum(), "region"))["error"] == "INVALID_OPERATION"


def test_bare_column_in_agg(orders):
    d = err(lambda: orders.group_by("region").agg(col("sales")))
    assert d["suggestions"][0]["fix"] == 'col("sales").sum()'


def test_mean_of_string(orders):
    assert err(lambda: orders.agg(col("region").mean()))["error"] == "TYPE_MISMATCH"


def test_duplicate_output_names(orders):
    d = err(lambda: orders.select(col("sales"), col("sales") * 2))
    assert d["error"] == "DUPLICATE_COLUMN" and "alias" in d["suggestions"][0]["fix"]


def test_join_key_type_mismatch(orders):
    other = pu.from_dict({"id": ["1"], "x": [1]})
    d = err(lambda: orders.join(other, on="id"))
    assert d["error"] == "TYPE_MISMATCH" and "cast" in d["suggestions"][0]["fix"]


def test_unknown_backend(orders):
    d = err(lambda: orders.collect("spark"))
    assert d["error"] == "UNKNOWN_BACKEND" and "duckdb" in d["details"]["available"]


def test_str_on_number(orders):
    assert err(lambda: orders.filter(col("sales").str.contains("1")))["error"] == "TYPE_MISMATCH"


def test_abbreviation_suggestion():
    df = pu.from_dict({"qty": [1], "price": [1.0]})
    assert err(lambda: df.select("quantity"))["suggestions"][0]["fix"] == 'col("qty")'
