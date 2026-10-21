"""DuckDB 2.0 previews removed Arrow fetching from connections; pyursa must still return results."""

import pytest

duckdb = pytest.importorskip("duckdb")

import pyursa as pu  # noqa: E402
from pyursa.backends import duckdb_backend as D  # noqa: E402


class _NoArrowConnection:
    def __init__(self, con):
        self._con = con

    def __getattr__(self, name):
        if name in ("to_arrow_table", "fetch_arrow_table"):
            raise AttributeError(name)
        return getattr(self._con, name)


def test_results_without_connection_arrow_fetch(monkeypatch):
    real = duckdb.connect
    monkeypatch.setattr(D.duckdb, "connect", lambda *a, **k: _NoArrowConnection(real(*a, **k)))
    df = pu.from_dict({"r": ["e", "w", "e"], "x": [1, 2, 3]})
    out = df.filter(pu.col("x") > 1).group_by("r").agg(pu.col("x").sum().alias("t")).sort("r")
    assert out.to_dicts(backend="duckdb") == [{"r": "e", "t": 3}, {"r": "w", "t": 2}]
