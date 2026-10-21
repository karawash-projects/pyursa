"""Compile the IR to DuckDB SQL.

Ordering is expensive in SQL (row numbers + ORDER BY), so the compiler tracks
whether the parent needs rows in order (`ordered`). A group-by, for example,
doesn't care about its input order, so joins and sorts below it skip that work.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal

import pyarrow as pa

from .. import expr as E
from .. import plan as P
from ..types import DType, common_type, literal_dtype
from .base import Backend

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None

from .castrules import BOOL_FALSE, BOOL_TRUE, DATE_RE, DATETIME_RE, DECIMAL_RE, FLOAT_RE, INT_RE, sentinel

RN = "__pu_rn"
_WS = "' ' || chr(9) || chr(10) || chr(13)"


def sql_type(t: DType) -> str:
    k = t.kind
    if k == "Int64":
        return "BIGINT"
    if k == "Float64":
        return "DOUBLE"
    if k == "String":
        return "VARCHAR"
    if k == "Boolean":
        return "BOOLEAN"
    if k == "Date":
        return "DATE"
    if k == "Datetime":
        return "TIMESTAMPTZ" if t.tz else "TIMESTAMP"
    if k == "Decimal":
        return f"DECIMAL({t.precision},{t.scale})"
    if k == "List":
        return f"{sql_type(t.inner)}[]"
    if k == "Struct":
        return "STRUCT(" + ", ".join(f"{q(n)} {sql_type(ft)}" for n, ft in t.fields) + ")"
    raise TypeError(f"No SQL type for {t}")


def checked_int(h: str) -> str:
    """Narrow a HUGEINT expression to BIGINT, raising on overflow. An explicit range check, because
    DuckDB's optimizer can drop a plain narrowing CAST when the result only feeds a comparison."""
    return (f"(CASE WHEN ({h}) IS NULL THEN NULL "
            f"WHEN ({h}) BETWEEN -9223372036854775808 AND 9223372036854775807 THEN CAST(({h}) AS BIGINT) "
            f"ELSE error('Overflow: integer result outside the Int64 range') END)")


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def s(value: str) -> str:
    """Inline SQL string literal (display only; execution binds parameters). NUL becomes chr(0)."""
    parts = ["'" + part.replace("'", "''") + "'" for part in value.split("\x00")]
    return parts[0] if len(parts) == 1 else "(" + " || chr(0) || ".join(parts) + ")"


def _lit(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return f"CAST({value} AS BIGINT)"
    if isinstance(value, float):
        return f"CAST({s(repr(value))} AS DOUBLE)"
    if isinstance(value, _decimal.Decimal):
        scale = max(0, -value.as_tuple().exponent) if isinstance(value.as_tuple().exponent, int) else 0
        return f"CAST({s(str(value))} AS DECIMAL(38,{min(scale, 38)}))"
    if isinstance(value, str):
        return s(value)
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            return f"CAST({s(value.isoformat(sep=' '))} AS TIMESTAMPTZ)"
        return f"CAST({s(value.isoformat(sep=' '))} AS TIMESTAMP)"
    if isinstance(value, _dt.date):
        return f"CAST({s(value.isoformat())} AS DATE)"
    raise TypeError(f"Unsupported literal {value!r}")


class _Compiler:
    """params=None renders literals inline (for explain output); a dict binds every
    user-supplied value as a named parameter, so no data is ever spliced into SQL text."""

    def __init__(self, params: dict | None = None) -> None:
        self.tables: dict[str, pa.Table] = {}
        self.n = 0
        self.params = params

    def lit(self, value) -> str:
        if isinstance(value, _dt.datetime) and value.tzinfo is not None:
            value = value.astimezone(_dt.timezone.utc)  # the exact instant, even for nonexistent local times
        if self.params is None or value is None or isinstance(value, bool):
            return _lit(value)
        name = f"p{len(self.params)}"
        self.params[name] = value
        if isinstance(value, int):
            return f"CAST(${name} AS BIGINT)"
        if isinstance(value, float):
            return f"CAST(${name} AS DOUBLE)"
        if isinstance(value, _decimal.Decimal):
            exp = value.as_tuple().exponent
            scale = max(0, -exp) if isinstance(exp, int) else 0
            return f"CAST(${name} AS DECIMAL(38,{min(scale, 38)}))"
        if isinstance(value, str):
            return f"CAST(${name} AS VARCHAR)"
        if isinstance(value, _dt.datetime):
            return f"CAST(${name} AS {'TIMESTAMPTZ' if value.tzinfo else 'TIMESTAMP'})"
        if isinstance(value, _dt.date):
            return f"CAST(${name} AS DATE)"
        return f"${name}"

    def alias(self) -> str:
        self.n += 1
        return f"q{self.n}"

    def cols(self, schema, prefix: str = "") -> str:
        return ", ".join(f"{prefix}{q(c)}" for c in schema)

    # --- plans ---------------------------------------------------------------
    def node(self, node: P.Node, ordered: bool = True) -> str:
        if isinstance(node, P.Source):
            cols = self.cols(node.schema)
            if node.paths:
                if any(ch in p for p in node.paths for ch in "[]*?{}"):
                    # read_parquet always expands globs; a PyArrow dataset reads exactly these files
                    import pyarrow.dataset as pads
                    name = f"src_{node.uid}"
                    self.tables[name] = pads.dataset(list(node.paths), format="parquet")
                    return f"SELECT {cols} FROM {name}"
                return f"SELECT {cols} FROM read_parquet([{', '.join(self.lit(p) for p in node.paths)}])"
            name = f"src_{node.uid}"
            self.tables[name] = node.table
            return f"SELECT {cols} FROM {name}"
        if isinstance(node, P.Filter):
            child = self.node(node.child, ordered)
            return f"SELECT * FROM ({child}) AS {self.alias()} WHERE {self.expr(node.predicate, node.child.schema)}"
        if isinstance(node, (P.Project, P.WithColumns)):
            sch = node.child.schema
            child = self.node(node.child, ordered or node.needs_row_index)
            if node.needs_row_index:
                child = f"SELECT *, row_number() OVER () AS {RN} FROM ({child}) AS {self.alias()}"
            if isinstance(node, P.Project):
                parts = [self.out(n, e, t, sch) for n, e, t in node.named]
            else:
                new = {n: (e, t) for n, e, t in node.named}
                parts = [self.out(c, *new[c], sch) if c in new else q(c) for c in sch]
                parts += [self.out(n, e, t, sch) for n, e, t in node.named if n not in sch]
            if not node.needs_row_index:
                return f"SELECT {', '.join(parts)} FROM ({child}) AS {self.alias()}"
            # window operators may reorder rows: carry the row number and restore input order
            inner = f"SELECT {', '.join(parts)}, {RN} FROM ({child}) AS {self.alias()}"
            out_cols = self.cols(node.schema)
            return f"SELECT {out_cols} FROM ({inner}) AS {self.alias()}" + (f" ORDER BY {RN}" if ordered else "")
        if isinstance(node, P.Aggregate):
            aggs = [self.out(n, e, t, node.child.schema) for n, e, t in node.named]
            child = f"({self.node(node.child, False)}) AS {self.alias()}"
            if not node.keys:
                return f"SELECT {', '.join(aggs)} FROM {child}"
            keys = self.cols(node.keys)
            sql = f"SELECT {keys}, {', '.join(aggs)} FROM {child} GROUP BY {keys}"
            if ordered:
                sql += " ORDER BY " + ", ".join(f"{q(k)} ASC NULLS LAST" for k in node.keys)
            return sql
        if isinstance(node, P.Sort):
            if not ordered:  # nobody downstream depends on this order
                return self.node(node.child, False)
            order = ", ".join(f"{q(c)} {'DESC' if d else 'ASC'} NULLS LAST" for c, d in zip(node.by, node.descending))
            inner = f"SELECT *, row_number() OVER () AS {RN} FROM ({self.node(node.child, True)}) AS {self.alias()}"
            return f"SELECT {self.cols(node.schema)} FROM ({inner}) AS {self.alias()} ORDER BY {order}, {RN}"
        if isinstance(node, P.Limit):
            return f"SELECT * FROM ({self.node(node.child, True)}) AS {self.alias()} LIMIT {node.n}"
        if isinstance(node, P.Distinct):
            if node.subset is None and not ordered:
                return f"SELECT DISTINCT * FROM ({self.node(node.child, False)}) AS {self.alias()}"
            inner = f"SELECT *, row_number() OVER () AS {RN} FROM ({self.node(node.child, True)}) AS {self.alias()}"
            part = self.cols(node.keys)
            sql = (f"SELECT {self.cols(node.schema)} FROM ({inner}) AS {self.alias()} "
                   f"QUALIFY row_number() OVER (PARTITION BY {part} ORDER BY {RN}) = 1")
            return sql + (f" ORDER BY {RN}" if ordered else "")
        if isinstance(node, P.Concat):
            cols = self.cols(node.schema)
            if not ordered:
                return " UNION ALL ".join(
                    f"SELECT {cols} FROM ({self.node(c, False)}) AS {self.alias()}" for c in node.inputs)
            parts = [f"SELECT {cols}, {i} AS __pu_src, row_number() OVER () AS {RN} "
                     f"FROM ({self.node(c, True)}) AS {self.alias()}" for i, c in enumerate(node.inputs)]
            return f"SELECT {cols} FROM ({' UNION ALL '.join(parts)}) AS {self.alias()} ORDER BY __pu_src, {RN}"
        if isinstance(node, P.Join):
            return self.join(node, ordered)
        if isinstance(node, P.Rename):
            cols = ", ".join(f"{q(c)} AS {q(node.mapping.get(c, c))}" for c in node.child.schema)
            return f"SELECT {cols} FROM ({self.node(node.child, ordered)}) AS {self.alias()}"
        if isinstance(node, P.Drop):
            return f"SELECT {self.cols(node.schema)} FROM ({self.node(node.child, ordered)}) AS {self.alias()}"
        raise NotImplementedError(f"duckdb backend: {node.op}")

    def join(self, node: P.Join, ordered: bool) -> str:
        l, r = self.alias(), self.alias()
        left_sql, right_sql = self.node(node.left, ordered), self.node(node.right, ordered)
        if ordered:
            left_sql = f"SELECT *, row_number() OVER () AS __pu_l FROM ({left_sql}) AS {self.alias()}"
            right_sql = f"SELECT *, row_number() OVER () AS __pu_r FROM ({right_sql}) AS {self.alias()}"
        cond = " AND ".join(f"{l}.{q(k)} = {r}.{q(k)}" for k in node.on)
        if node.how in ("semi", "anti"):
            kind = "SEMI JOIN" if node.how == "semi" else "ANTI JOIN"
            sql = f"SELECT {self.cols(node.left.schema, l + '.')} FROM ({left_sql}) AS {l} {kind} ({right_sql}) AS {r} ON {cond}"
            return sql + (f" ORDER BY {l}.__pu_l" if ordered else "")
        if node.how == "full":
            cols = [f"COALESCE({l}.{q(c)}, {r}.{q(c)}) AS {q(c)}" if c in node.on else f"{l}.{q(c)}"
                    for c in node.left.schema]
        else:
            cols = [f"{l}.{q(c)}" for c in node.left.schema]
        cols += [f"{r}.{q(src)} AS {q(dst)}" for src, dst in node.right_out]
        how = {"inner": "INNER JOIN", "left": "LEFT JOIN", "full": "FULL OUTER JOIN"}[node.how]
        sql = f"SELECT {', '.join(cols)} FROM ({left_sql}) AS {l} {how} ({right_sql}) AS {r} ON {cond}"
        if ordered:
            sql += f" ORDER BY {l}.__pu_l NULLS LAST, {r}.__pu_r NULLS LAST"
        return sql

    def out(self, name: str, e: E.Expr, t: DType, schema) -> str:
        inner = E.unalias(e)
        sql = self.expr(inner, schema)
        if t != DType.NULL and not (t.is_nested and isinstance(inner, E.Col)):
            sql = f"CAST({sql} AS {sql_type(t)})"
        return f"{sql} AS {q(name)}"

    # --- expressions -----------------------------------------------------------
    def expr(self, e: E.Expr, schema) -> str:
        x = lambda sub: self.expr(sub, schema)  # noqa: E731
        if isinstance(e, E.Col):
            return q(e.name)
        if isinstance(e, E.Lit):
            return self.lit(e.value)
        if isinstance(e, E.Alias):
            return x(e.operand)
        if isinstance(e, E.BinOp):
            l, r = x(e.left), x(e.right)
            if e.op == "/":
                return f"(CAST({l} AS DOUBLE) / CAST({r} AS DOUBLE))"
            if e.op in E.ARITH:
                out_t = e.dtype(schema)
                if out_t == DType.INT64:  # compute in HUGEINT and narrow with an explicit check
                    return checked_int(f"CAST({l} AS HUGEINT) {e.op} CAST({r} AS HUGEINT)")
                if out_t.is_decimal:
                    lt, rt = e.left.dtype(schema), e.right.dtype(schema)
                    # operands at their own scale; DuckDB computes decimal +,-,* exactly
                    if lt.is_integer:
                        l = f"CAST({l} AS DECIMAL(38,0))"
                    if rt.is_integer:
                        r = f"CAST({r} AS DECIMAL(38,0))"
                return f"({l} {e.op} {r})"
            if e.op in E.COMPARE:
                lt, rt = e.left.dtype(schema), e.right.dtype(schema)
                if lt != rt and lt != DType.NULL and rt != DType.NULL:
                    ct = common_type(lt, rt)
                    if ct is not None and ct.is_numeric:
                        l, r = f"CAST({l} AS {sql_type(ct)})", f"CAST({r} AS {sql_type(ct)})"
            op = {"==": "=", "!=": "<>", "&": "AND", "|": "OR"}.get(e.op, e.op)
            return f"({l} {op} {r})"
        if isinstance(e, E.Unary):
            v = x(e.operand)
            if e.op in ("neg", "abs") and e.operand.dtype(schema) == DType.INT64:
                inner = f"(-CAST({v} AS HUGEINT))" if e.op == "neg" else f"abs(CAST({v} AS HUGEINT))"
                return checked_int(inner)
            return {"not": f"(NOT {v})", "neg": f"(-{v})", "abs": f"abs({v})",
                    "is_null": f"({v} IS NULL)", "is_not_null": f"({v} IS NOT NULL)"}[e.op]
        if isinstance(e, E.Round):
            t = e.operand.dtype(schema)
            v = x(e.operand)
            return v if t.is_integer else f"round({v}, {e.decimals})"
        if isinstance(e, E.IsIn):
            v = x(e.operand)
            if not e.values:
                return f"(CASE WHEN {v} IS NULL THEN NULL ELSE FALSE END)"
            t = e.operand.dtype(schema)
            ct = t
            for val in e.values:
                ct = common_type(ct, literal_dtype(val)) or ct
            if ct != t:
                v = f"CAST({v} AS {sql_type(ct)})"
            items = [self.lit(val) if ct == literal_dtype(val) else f"CAST({self.lit(val)} AS {sql_type(ct)})" for val in e.values]
            return f"({v} IN ({', '.join(items)}))"
        if isinstance(e, E.Cast):
            return self.cast(x(e.operand), e.operand.dtype(schema), e.to)
        if isinstance(e, E.Coalesce):
            t = e.dtype(schema)
            items = [x(i) if t == DType.NULL else f"CAST({x(i)} AS {sql_type(t)})" for i in e.items]
            return f"COALESCE({', '.join(items)})"
        if isinstance(e, E.Case):
            t = e.dtype(schema)
            cast = (lambda v: f"CAST({v} AS {sql_type(t)})") if t != DType.NULL else (lambda v: v)
            parts = " ".join(f"WHEN {x(c)} THEN {cast(x(v))}" for c, v in e.branches)
            default = cast(x(e.default)) if e.default is not None else "NULL"
            return f"(CASE {parts} ELSE {default} END)"
        if isinstance(e, E.StrOp):
            v = x(e.operand)
            return {
                "contains": lambda: f"contains({v}, {self.lit(e.arg)})",
                "contains_regex": lambda: f"regexp_matches({v}, {self.lit(e.arg)})",
                "starts_with": lambda: f"starts_with({v}, {self.lit(e.arg)})",
                "ends_with": lambda: f"ends_with({v}, {self.lit(e.arg)})",
                "lower": lambda: f"lower({v})",
                "upper": lambda: f"upper({v})",
                "strip": lambda: f"trim({v}, {_WS})",
                "len": lambda: f"length({v})",
                "replace": lambda: f"replace({v}, {self.lit(e.arg)}, {self.lit(e.arg2)})",
                "slice": lambda: (f"substring({v}, {e.arg + 1}, {e.arg2})" if e.arg2 is not None
                                  else f"substring({v}, {e.arg + 1})"),
            }[e.op]()
        if isinstance(e, E.DtOp):
            t = e.operand.dtype(schema)
            v = x(e.operand)
            local = f"timezone({self.lit(t.tz)}, {v})" if t.kind == "Datetime" and t.tz else v
            if e.op == "date":
                return f"CAST({local} AS DATE)"
            if e.op == "truncate":
                tr = f"date_trunc({s(e.unit)}, {local})"
                if t.kind == "Datetime" and t.tz:
                    if e.unit == "hour":  # subtract the offset within the hour: exact across DST changes
                        return f"({v} - ({local} - {tr}))"
                    return f"timezone({self.lit(t.tz)}, {tr})"
                return tr
            fn = {"weekday": "isodow"}.get(e.op, e.op)
            return f"{fn}({local})"
        if isinstance(e, E.Agg):
            return self.agg(e, schema)
        if isinstance(e, E.Window):
            return self.window(e, schema)
        raise NotImplementedError(f"duckdb backend: {type(e).__name__}")

    def cast(self, v: str, src: DType, to: DType) -> str:
        """The strict cast contract (SEMANTICS['casts']); mirrors the Polars backend."""
        T = sql_type(to) if to != DType.NULL else None
        if src == to or T is None:
            return v
        if src == DType.NULL:
            return f"CAST({v} AS {T})"
        if src == DType.STRING:
            tr = f"trim({v}, {_WS})"
            low = f"lower({tr})"
            bad = s(sentinel(to))

            def m(pattern):
                return f"regexp_full_match({tr}, {s(pattern)})"
            if to == DType.INT64:
                return f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {m(INT_RE)} THEN ltrim({tr}, '+') ELSE {bad} END AS BIGINT)"
            if to == DType.FLOAT64:
                return (f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {m(FLOAT_RE)} THEN ltrim({tr}, '+') "
                        f"WHEN {low} IN ('inf', '+inf', 'infinity', '+infinity') THEN 'inf' "
                        f"WHEN {low} IN ('-inf', '-infinity') THEN '-inf' "
                        f"WHEN {low} IN ('nan', '+nan', '-nan') THEN 'nan' ELSE {bad} END AS DOUBLE)")
            if to.is_decimal:
                wide = f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {m(DECIMAL_RE)} THEN ltrim({tr}, '+') ELSE {bad} END AS DECIMAL(38,17))"
                return f"CAST(round({wide}, {to.scale}) AS {T})"
            if to == DType.BOOLEAN:
                t_vals = ", ".join(s(x) for x in BOOL_TRUE)
                f_vals = ", ".join(s(x) for x in BOOL_FALSE)
                return (f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {low} IN ({t_vals}) THEN 'true' "
                        f"WHEN {low} IN ({f_vals}) THEN 'false' ELSE {bad} END AS BOOLEAN)")
            if to == DType.DATE:
                return f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {m(DATE_RE)} THEN {tr} ELSE {bad} END AS DATE)"
            if to.kind == "Datetime":
                norm = f"replace({tr}, 'T', ' ')"
                norm = f"(CASE WHEN length({norm}) = 16 THEN {norm} || ':00' ELSE {norm} END)"
                return f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {m(DATETIME_RE)} THEN {norm} ELSE {bad} END AS TIMESTAMP)"
        if to == DType.INT64 and src.is_float:  # must be whole; NaN/inf fail
            return f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN {v} = trunc({v}) THEN {v} ELSE CAST('nan' AS DOUBLE) END AS BIGINT)"
        if to == DType.INT64 and src.is_decimal:
            txt = f"regexp_replace(CAST({v} AS VARCHAR), '\\.0*$', '')"
            return (f"CAST(CASE WHEN {v} IS NULL THEN NULL WHEN regexp_full_match({txt}, {s(INT_RE)}) THEN {txt} "
                    f"ELSE {s(sentinel(to))} END AS BIGINT)")
        if to.is_decimal and (src.is_float or src.is_decimal):
            return f"CAST(round({v}, {to.scale}) AS {T})"
        if to == DType.STRING and src.kind == "Datetime":
            return (f"(CASE WHEN epoch_us({v}) % 1000000 = 0 THEN strftime({v}, '%Y-%m-%d %H:%M:%S') "
                    f"ELSE strftime({v}, '%Y-%m-%d %H:%M:%S.%f') END)")
        if to == DType.STRING and src.is_float:
            return (f"(CASE WHEN isnan({v}) THEN 'NaN' ELSE "
                    f"regexp_replace(CAST({v} AS VARCHAR), 'e([+-])0+([0-9])', 'e\\1\\2') END)")
        if to == DType.STRING and src == DType.BOOLEAN:
            return f"(CASE WHEN {v} THEN 'true' WHEN NOT {v} THEN 'false' END)"
        return f"CAST({v} AS {T})"

    def agg(self, e: E.Agg, schema) -> str:
        if e.operand is None:
            return "COUNT(*)"
        v = self.expr(e.operand, schema)
        fn = e.fn
        if fn in ("mean", "median", "quantile", "std", "var"):
            f = f"CAST({v} AS DOUBLE)"
            if fn == "quantile":
                return f"quantile_cont({f}, {float(e.param)!r})"
            if fn in ("std", "var"):
                finite = f"(CASE WHEN isinf({f}) OR isnan({f}) THEN NULL ELSE {f} END)"  # avoid DuckDB's range error
                core = f"stddev_samp({finite})" if fn == "std" else f"var_samp({finite})"
                return f"(CASE WHEN bool_or(isinf({f}) OR isnan({f})) THEN CAST('nan' AS DOUBLE) ELSE {core} END)"
            return {"mean": f"AVG({f})", "median": f"median({f})", "std": f"stddev_samp({f})", "var": f"var_samp({f})"}[fn]
        if fn == "sum" and e.operand.dtype(schema) == DType.INT64:
            return checked_int(f"SUM(CAST({v} AS HUGEINT))")
        if fn == "max" and e.operand.dtype(schema).is_float:
            # NaN is the largest value; also stops DuckDB answering from Parquet statistics, which skip NaN
            return f"(CASE WHEN bool_or(isnan({v})) THEN CAST('nan' AS DOUBLE) ELSE MAX({v}) END)"
        return {"sum": f"SUM({v})", "min": f"MIN({v})", "max": f"MAX({v})", "count": f"COUNT({v})",
                "n_unique": f"COUNT(DISTINCT {v})", "any": f"bool_or({v})", "all": f"bool_and({v})"}[fn]

    def window(self, w: E.Window, schema) -> str:
        part = f"PARTITION BY {self.cols(w.partition_by)} " if w.partition_by else ""
        inner = w.inner
        if isinstance(inner, E.Agg):
            over = f"OVER ({part.strip()})"
            if inner.fn == "sum" and inner.operand is not None and inner.operand.dtype(schema) == DType.INT64:
                v = self.expr(inner.operand, schema)
                return checked_int(f"SUM(CAST({v} AS HUGEINT)) {over}")
            if inner.fn == "max" and inner.operand is not None and inner.operand.dtype(schema).is_float:
                v = self.expr(inner.operand, schema)
                return f"(CASE WHEN bool_or(isnan({v})) {over} THEN CAST('nan' AS DOUBLE) ELSE MAX({v}) {over} END)"
            return f"{self.agg(inner, schema)} {over}"
        direction = "DESC" if w.descending else "ASC"
        order = ", ".join(f"{q(c)} {direction} NULLS LAST" for c in w.order_by)
        fn = inner.fn
        if fn in ("rank", "dense_rank"):
            return f"{fn.upper()}() OVER ({part}ORDER BY {order})"
        stable = f"{order}, {RN} ASC"
        if fn == "row_number":
            return f"ROW_NUMBER() OVER ({part}ORDER BY {stable})"
        v = self.expr(inner.operand, schema)
        if fn == "cum_sum":
            frame = f"OVER ({part}ORDER BY {stable} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
            if inner.operand.dtype(schema) == DType.INT64:
                return checked_int(f"SUM(CAST({v} AS HUGEINT)) {frame}")
            return f"SUM({v}) {frame}"
        if fn == "shift":
            n = inner.param
            if n == 0:
                return v
            f = "LAG" if n > 0 else "LEAD"
            return f"{f}({v}, {abs(n)}) OVER ({part}ORDER BY {stable})"
        raise NotImplementedError(fn)  # pragma: no cover


class DuckDBBackend(Backend):
    name = "duckdb"

    def __init__(self) -> None:
        if duckdb is None:
            raise ImportError("The duckdb backend needs `pip install duckdb`.")

    def compile(self, node: P.Node) -> str:
        return _Compiler().node(node)

    def _run(self, node: P.Node) -> pa.Table:
        params: dict = {}
        c = _Compiler(params)
        sql = c.node(node)
        con = duckdb.connect()
        try:
            con.execute("SET preserve_insertion_order = true")
            con.execute("SET TimeZone = 'UTC'")
            if any(pa.types.is_floating(f.type) for t in c.tables.values() for f in t.schema):
                # DuckDB's filter pushdown into Arrow scans compares NaN by IEEE rules, which contradicts
                # the NaN-is-largest rule it uses everywhere else; keep filters in DuckDB itself.
                con.execute("SET disabled_optimizers = 'filter_pushdown'")
            for name, table in c.tables.items():
                con.register(name, table)
            for method in ("to_arrow_table", "fetch_arrow_table"):  # duckdb >= 1.5 / older 1.x
                if hasattr(con, method):
                    res = con.execute(sql, params) if params else con.execute(sql)
                    return getattr(res, method)()
            # duckdb 2.0 previews removed Arrow fetching from the connection; relations still have it
            rel = con.sql(sql, params=params) if params else con.sql(sql)
            return rel.to_arrow_table()
        finally:
            con.close()
