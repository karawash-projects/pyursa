from .base import (
    SEMANTICS,
    Backend,
    available_backends,
    default_backend,
    get_backend,
    register_backend,
    set_default_backend,
)


def _polars():
    from .polars_backend import PolarsBackend
    return PolarsBackend()


def _duckdb():
    from .duckdb_backend import DuckDBBackend
    return DuckDBBackend()


register_backend("polars", _polars)
register_backend("duckdb", _duckdb)

__all__ = [
    "SEMANTICS", "Backend", "available_backends", "default_backend",
    "get_backend", "register_backend", "set_default_backend",
]
