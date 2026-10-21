"""Backend protocol and registry.

A backend compiles a logical plan to its engine and returns Arrow. The
library casts every result to the plan's declared schema, so all backends
return identically-typed data. Behavioral rules every backend must follow
live in SEMANTICS and are enforced by tests/test_conformance.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import pyarrow as pa

from ..errors import BackendError, UnknownBackendError
from ..plan import Node
from ..types import cast_column, to_arrow_schema

SEMANTICS = {
    "null_comparison": "Comparisons with null yield null; filter() drops rows where the condition is null.",
    "division": "'/' always returns Float64 and follows IEEE 754 (x/0 = inf, 0/0 = NaN).",
    "sum_of_nulls": "sum() over only nulls (or no rows) returns null, not 0.",
    "n_unique": "n_unique() does not count null as a value.",
    "count": "count() counts non-null values; pu.len() counts rows.",
    "sort": "sort() is stable and places nulls last in both directions.",
    "group_by_order": "group_by().agg() output is sorted by the keys ascending, nulls last.",
    "join_order": "join() keeps left row order, then right row order within matches; full joins put unmatched right rows last; null keys never match (anti joins keep null-key rows).",
    "distinct": "distinct() keeps the first occurrence of each row (or subset) in input order; nulls compare equal.",
    "concat": "concat() returns all rows of the first frame, then the second, and so on.",
    "rounding": "round() rounds half away from zero (2.5 -> 3, -2.5 -> -3).",
    "decimals": "Decimal +,-,* are exact (scale: max for +/-, sum for *); / returns Float64; sum keeps the scale; mean/median/std return Float64.",
    "quantiles": "median()/quantile(q) use linear interpolation; std()/var() are sample statistics (ddof=1).",
    "windows": "Window results keep input row order; ties in order_by keep input order; rank() is SQL RANK with nulls last; cum_sum skips nulls.",
    "time_zones": "Datetime(tz) keeps its zone; .dt parts are computed in that zone; comparing different zones is an error.",
    "strings": "str.len counts characters; str.strip removes spaces, tabs and newlines; str.slice is 0-based.",
    "row_order": "filter/select/with_columns/limit preserve input row order.",
    "nan": "NaN is the largest float value and equals itself (in comparisons, sorting, grouping and min/max); max() returns NaN when one is present; std/var with an infinite value return NaN.",
    "integer_overflow": "Int64 arithmetic, sums and running totals that leave the Int64 range raise INTEGER_OVERFLOW; they never wrap around.",
    "casts": "Casts are strict and identical everywhere: text is trimmed of spaces/tabs/newlines and must then be a valid literal (integers like '+12', floats like '1e3', 'inf' or 'nan', decimals with up to 20 integer digits and 17 places, booleans 'true'/'false'/'1'/'0', dates 'YYYY-MM-DD', datetimes 'YYYY-MM-DD HH:MM[:SS[.ffffff]]' with ' ' or 'T', years 0001-9999, hours 00-23, no leap seconds); anything else raises CAST_FAILED. Float or Decimal to Int64 requires a whole number. Conversions to Decimal round half away from zero. Floats print as text that converts back to the exact same value ('NaN', 'inf'; notation such as '0.00001' vs '1e-5' can differ between engines); datetimes print as 'YYYY-MM-DD HH:MM:SS[.ffffff]'. Casts that have no single meaning (dates to numbers, floats to Boolean, changing a time zone) are rejected when the plan is built.",
    "names": "Column names must differ by more than letter case, must not contain NUL, and must not start with the reserved prefix '__pu_'.",
    "float_precision": "Float sums and means can differ in the last digits (about 1e-15 relative) between engines because they add in different orders; every other result is identical.",
    "known_differences": "Rare cases engines still differ on, by design of the engines: upper()/lower() of a few special characters (e.g. 'ß', 'İ'), and DuckDB may return -0.0 as 0.0 after sorting.",
}


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def _run(self, node: Node) -> pa.Table: ...

    @abstractmethod
    def compile(self, node: Node) -> str:
        """Human-readable compiled form (SQL text, engine plan, ...) for explain()."""

    def execute(self, node: Node) -> pa.Table:
        try:
            table = self._run(node)
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:  # engine failures (including Rust panics) become structured errors
            raise engine_error(self, node, exc) from None
        return conform(table, node, self.name)


def engine_error(backend: "Backend", node: Node, exc: BaseException) -> BackendError:
    from .castrules import SENTINEL_PREFIX
    text = f"{type(exc).__name__}: {exc}"
    try:
        query = backend.compile(node)
    except BaseException:
        query = None
    details = {"backend": backend.name, "engine_error": text[:2000], "compiled": query}
    # classify on the first line only: later lines echo expressions and column names
    low = text.splitlines()[0].lower() if text else ""
    if SENTINEL_PREFIX.lower() in low:
        target = low.split(SENTINEL_PREFIX.lower(), 1)[1].split("'")[0].split('"')[0].split("]")[0].split()[0]
        return BackendError(
            f"Some values can't be converted to {target} under the strict cast rules.",
            details=details, code="CAST_FAILED",
            suggestions=[{"description": "Text must match the target format exactly after trimming spaces "
                                         "(see pu.SEMANTICS['casts']). Filter or clean invalid values first, e.g. with "
                                         ".str.contains_regex(...) and when/then."}],
        )
    if ("f64" in low and "i64" in low) or "type double with value" in low:
        return BackendError(
            "A floating-point value isn't a whole number (or is NaN/inf), so it can't become Int64.",
            details=details, code="CAST_FAILED",
            suggestions=[{"description": "Round first.", "fix": 'col("x").round(0).cast("Int64")'}],
        )
    if ("overflow" in low or ("i128" in low and "i64" in low)
            or ("out of range" in low and any(k in low for k in ("int64", "bigint", "integer", "hugeint")))):
        return BackendError(
            "An integer result is outside the Int64 range.", details=details, code="INTEGER_OVERFLOW",
            suggestions=[{"description": "Cast to Decimal(38,0) or Float64 before the calculation.",
                          "fix": 'col("x").cast("Decimal(38,0)")'}],
        )
    if any(k in low for k in ("conversion error", "could not convert", "conversion from", "invalid date",
                              "cannot parse", "could not parse", "failed to parse")) or (
            "out of range" in low and any(k in low for k in ("date", "time", "decimal"))):
        return BackendError(
            "Some values can't be converted to the target type.", details=details, code="CAST_FAILED",
            suggestions=[{"description": "Check the values (e.g. impossible dates such as 2024-02-30) and clean "
                                         "or filter them before casting."}],
        )
    return BackendError(
        f"The {backend.name} engine failed while executing the plan: {exc}",
        details=details,
        suggestions=[{"description": "Check casts and value ranges; frame.explain() shows the compiled query."}],
    )


def conform(table: pa.Table, node: Node, backend: str) -> pa.Table:
    target = to_arrow_schema(node.schema)
    if table.column_names != list(node.schema):
        raise BackendError(
            f"Backend '{backend}' returned columns {table.column_names}, expected {list(node.schema)}.",
            details={"backend": backend}, code="BACKEND_CONTRACT_VIOLATION",
        )
    cols = []
    for field, column in zip(target, table.columns):
        if pa.types.is_null(field.type):  # engines may type all-null results as int32 etc.
            if column.null_count != len(column):
                raise BackendError(f"Backend '{backend}' returned values for Null column '{field.name}'.",
                                   code="BACKEND_CONTRACT_VIOLATION")
            cols.append(pa.nulls(len(column)))
        else:
            cols.append(cast_column(column, field.type, field.name))
    return pa.Table.from_arrays(cols, schema=target)


_REGISTRY: dict[str, Callable[[], Backend]] = {}
_INSTANCES: dict[str, Backend] = {}
_DEFAULT = ["polars"]


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    _REGISTRY[name] = factory
    _INSTANCES.pop(name, None)


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str | None = None) -> Backend:
    name = name or _DEFAULT[0]
    if name not in _REGISTRY:
        raise UnknownBackendError(
            f"Unknown backend '{name}'.",
            details={"requested": name, "available": available_backends()},
            suggestions=[{"description": f"Use one of: {', '.join(available_backends())}."}],
        )
    if name not in _INSTANCES:
        _INSTANCES[name] = _REGISTRY[name]()
    return _INSTANCES[name]


def set_default_backend(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise UnknownBackendError("set_default_backend() needs a backend name.",
                                  details={"available": available_backends()})
    get_backend(name)
    _DEFAULT[0] = name


def default_backend() -> str:
    return _DEFAULT[0]
