"""Compile the IR to Polars lazy frames."""

from __future__ import annotations

import pyarrow as pa

from .. import expr as E
from .. import plan as P
from ..types import DType, common_type
from ..types import literal_dtype as _lit_type
from .base import Backend

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None

from .castrules import BOOL_FALSE, BOOL_TRUE, DATE_RE, DATETIME_RE, DECIMAL_RE, FLOAT_RE, INT_RE, sentinel

IDX = "__pu_idx"
_TRUNC = {"year": "1y", "month": "1mo", "week": "1w", "day": "1d", "hour": "1h"}
_WS = " \t\n\r"


def _pl_dtype(t: DType):
    k = t.kind
    if k == "Int64":
        return pl.Int64
    if k == "Float64":
        return pl.Float64
    if k == "String":
        return pl.String
    if k == "Boolean":
        return pl.Boolean
    if k == "Date":
        return pl.Date
    if k == "Datetime":
        return pl.Datetime("us", t.tz)
    if k == "Decimal":
        return pl.Decimal(t.precision, t.scale)
    if k == "List":
        return pl.List(_pl_dtype(t.inner))
    if k == "Struct":
        return pl.Struct({n: _pl_dtype(ft) for n, ft in t.fields})
    return pl.Null


class PolarsBackend(Backend):
    name = "polars"

    def __init__(self) -> None:
        if pl is None:
            raise ImportError("The polars backend needs `pip install polars`.")

    def _run(self, node: P.Node) -> pa.Table:
        return self._lazy(node).collect().to_arrow()

    def compile(self, node: P.Node) -> str:
        return self._lazy(node).explain()

    # --- plans -------------------------------------------------------------
    def _lazy(self, node: P.Node):
        if isinstance(node, P.Source):
            if node.paths:  # paths are already expanded and sandbox-checked: read them literally
                return pl.scan_parquet(node.paths if len(node.paths) > 1 else node.paths[0], glob=False)
            return pl.from_arrow(node.table).lazy()
        if isinstance(node, P.Filter):
            return self._lazy(node.child).filter(self._expr(node.predicate, node.child.schema))
        if isinstance(node, (P.Project, P.WithColumns)):
            lf = self._lazy(node.child)
            if node.needs_row_index:
                lf = lf.with_row_index(IDX)
            outs = [self._out(n, e, t, node.child.schema) for n, e, t in node.named]
            if isinstance(node, P.Project):
                return lf.select(outs)
            lf = lf.with_columns(outs)
            return lf.drop(IDX) if node.needs_row_index else lf
        if isinstance(node, P.Aggregate):
            aggs = [self._out(n, e, t, node.child.schema) for n, e, t in node.named]
            lf = self._lazy(node.child)
            if not node.keys:
                return lf.select(aggs)
            return lf.group_by(node.keys).agg(aggs).sort(node.keys, nulls_last=True, maintain_order=True)
        if isinstance(node, P.Sort):
            return self._lazy(node.child).sort(node.by, descending=node.descending, nulls_last=True, maintain_order=True)
        if isinstance(node, P.Limit):
            return self._lazy(node.child).head(node.n)
        if isinstance(node, P.Distinct):
            return self._lazy(node.child).unique(subset=node.keys, keep="first", maintain_order=True)
        if isinstance(node, P.Concat):
            cols = list(node.schema)
            return pl.concat([self._lazy(c).select(cols) for c in node.inputs], how="vertical")
        if isinstance(node, P.Join):
            return self._join(node)
        if isinstance(node, P.Rename):
            return self._lazy(node.child).rename(node.mapping)
        if isinstance(node, P.Drop):
            return self._lazy(node.child).drop(node.columns)
        raise NotImplementedError(f"polars backend: {node.op}")

    def _join(self, node: P.Join):
        left, right = self._lazy(node.left), self._lazy(node.right)
        if node.how in ("semi", "anti"):
            return left.join(right.select(node.on), on=node.on, how=node.how, maintain_order="left", nulls_equal=False)
        renames = {src: dst for src, dst in node.right_out if src != dst}
        if renames:
            right = right.rename(renames)
        if node.how == "full":
            out = (left.with_row_index("__pu_l")
                   .join(right.with_row_index("__pu_r"), on=node.on, how="full", coalesce=True, nulls_equal=False)
                   .sort(["__pu_l", "__pu_r"], nulls_last=True, maintain_order=True)
                   .drop(["__pu_l", "__pu_r"]))
            return out.select(list(node.schema))
        return left.join(right, on=node.on, how=node.how, maintain_order="left_right", nulls_equal=False, coalesce=True)

    def _out(self, name: str, e: E.Expr, t: DType, schema):
        inner = E.unalias(e)
        out = self._expr(inner, schema)
        if t != DType.NULL and not (t.is_nested and isinstance(inner, E.Col)):
            out = out.cast(_pl_dtype(t))
        return out.alias(name)

    # --- expressions -------------------------------------------------------
    def _expr(self, e: E.Expr, schema):
        x = lambda sub: self._expr(sub, schema)  # noqa: E731
        if isinstance(e, E.Col):
            return pl.col(e.name)
        if isinstance(e, E.Lit):
            if e.value is None:
                return pl.lit(None)
            if e._dtype.kind == "Datetime" and e._dtype.tz:
                import datetime as _dt
                utc = e.value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
                return pl.lit(utc, dtype=pl.Datetime("us")).dt.replace_time_zone("UTC").dt.convert_time_zone(e._dtype.tz)
            return pl.lit(e.value, dtype=_pl_dtype(e._dtype))
        if isinstance(e, E.Alias):
            return x(e.operand).alias(e.name)
        if isinstance(e, E.BinOp):
            l, r = x(e.left), x(e.right)
            lt, rt = e.left.dtype(schema), e.right.dtype(schema)
            if e.op == "/":
                return l.cast(pl.Float64) / r.cast(pl.Float64)
            if e.op in E.ARITH:
                out_t = e.dtype(schema)
                if out_t == DType.INT64:  # detect overflow instead of wrapping around
                    l, r = l.cast(pl.Int128), r.cast(pl.Int128)
                    res = {"+": lambda: l + r, "-": lambda: l - r, "*": lambda: l * r}[e.op]()
                    return res.cast(pl.Int64, strict=True)
                if out_t.is_decimal or (lt != rt and out_t != DType.NULL):
                    pt = _pl_dtype(out_t)  # exact decimal math: compute at the result scale
                    l, r = l.cast(pt), r.cast(pt)
                return {"+": lambda: l + r, "-": lambda: l - r, "*": lambda: l * r}[e.op]()
            if e.op in E.COMPARE and lt != rt and lt != DType.NULL and rt != DType.NULL:
                ct = common_type(lt, rt)
                if ct is not None:
                    l, r = l.cast(_pl_dtype(ct)), r.cast(_pl_dtype(ct))
            return {
                "==": lambda: l == r, "!=": lambda: l != r, "<": lambda: l < r,
                "<=": lambda: l <= r, ">": lambda: l > r, ">=": lambda: l >= r,
                "&": lambda: l & r, "|": lambda: l | r,
            }[e.op]()
        if isinstance(e, E.Unary):
            v = x(e.operand)
            if e.op in ("neg", "abs") and e.operand.dtype(schema) == DType.INT64:
                w = v.cast(pl.Int128)
                zero = pl.lit(0, dtype=pl.Int128)
                flipped = zero - w  # Int128 has no unary negation in Polars
                out = flipped if e.op == "neg" else pl.when(w < zero).then(flipped).otherwise(w)
                return out.cast(pl.Int64, strict=True)
            return {"not": lambda: ~v, "neg": lambda: -v, "abs": v.abs,
                    "is_null": v.is_null, "is_not_null": v.is_not_null}[e.op]()
        if isinstance(e, E.Round):
            t = e.operand.dtype(schema)
            v = x(e.operand)
            return v if t.is_integer else v.round(e.decimals, mode="half_away_from_zero")
        if isinstance(e, E.IsIn):
            t = e.operand.dtype(schema)
            ct = t
            for v in e.values:
                ct = common_type(ct, _lit_type(v)) or ct
            v = x(e.operand)
            if ct != t:
                v = v.cast(_pl_dtype(ct))
            values = [float(val) for val in e.values] if ct.is_float else list(e.values)
            return v.is_in(pl.Series(values, dtype=_pl_dtype(ct)).implode())
        if isinstance(e, E.Cast):
            return self._cast(x(e.operand), e.operand.dtype(schema), e.to)
        if isinstance(e, E.Coalesce):
            t = e.dtype(schema)
            items = [x(i) for i in e.items]
            if t != DType.NULL:
                items = [i.cast(_pl_dtype(t)) for i in items]
            return pl.coalesce(items)
        if isinstance(e, E.Case):
            t = e.dtype(schema)
            cast = (lambda v: v.cast(_pl_dtype(t))) if t != DType.NULL else (lambda v: v)
            chain = None
            for cond, value in e.branches:
                chain = (pl.when(x(cond)) if chain is None else chain.when(x(cond))).then(cast(x(value)))
            default = cast(x(e.default)) if e.default is not None else pl.lit(None)
            return chain.otherwise(default)  # type: ignore[union-attr]
        if isinstance(e, E.StrOp):
            v = x(e.operand)
            op = e.op
            if op == "contains":
                return v.str.contains(e.arg, literal=True)
            if op == "contains_regex":
                return v.str.contains(e.arg, literal=False)
            if op == "starts_with":
                return v.str.starts_with(e.arg)
            if op == "ends_with":
                return v.str.ends_with(e.arg)
            if op == "lower":
                return v.str.to_lowercase()
            if op == "upper":
                return v.str.to_uppercase()
            if op == "strip":
                return v.str.strip_chars(_WS)
            if op == "len":
                return v.str.len_chars()
            if op == "replace":
                return v.str.replace_all(e.arg, e.arg2, literal=True)
            if op == "slice":
                return v.str.slice(e.arg, e.arg2)
        if isinstance(e, E.DtOp):
            v = x(e.operand)
            if e.op == "date":
                return v.dt.date()
            if e.op == "truncate":
                return v.dt.truncate(_TRUNC[e.unit])
            return getattr(v.dt, e.op)()
        if isinstance(e, E.Agg):
            return self._agg(e, schema)
        if isinstance(e, E.Window):
            return self._window(e, schema)
        raise NotImplementedError(f"polars backend: {type(e).__name__}")

    def _cast(self, v, src: DType, to: DType):
        """The strict cast contract (SEMANTICS['casts']); mirrored in the DuckDB backend."""
        if src == to:
            return v
        if src == DType.NULL:
            return v.cast(_pl_dtype(to))
        if src == DType.STRING:
            s = v.str.strip_chars(" \t\n\r")
            low = s.str.to_lowercase()
            if to == DType.INT64:
                ok = s.str.contains(INT_RE)
                return pl.when(v.is_null()).then(None).when(ok).then(s.str.replace(r"^\+", "")) \
                    .otherwise(pl.lit(sentinel(to))).cast(pl.Int64, strict=True)
            if to == DType.FLOAT64:
                txt = (pl.when(v.is_null()).then(None)
                       .when(s.str.contains(FLOAT_RE)).then(s.str.replace(r"^\+", ""))
                       .when(low.is_in(["inf", "+inf", "infinity", "+infinity"])).then(pl.lit("inf"))
                       .when(low.is_in(["-inf", "-infinity"])).then(pl.lit("-inf"))
                       .when(low.is_in(["nan", "+nan", "-nan"])).then(pl.lit("nan"))
                       .otherwise(pl.lit(sentinel(to))))
                return txt.cast(pl.Float64, strict=True)
            if to.is_decimal:
                txt = pl.when(v.is_null()).then(None).when(s.str.contains(DECIMAL_RE)) \
                    .then(s.str.replace(r"^\+", "")).otherwise(pl.lit(sentinel(to)))
                wide = txt.cast(pl.Decimal(38, 17), strict=True)
                return wide.round(to.scale, mode="half_away_from_zero").cast(_pl_dtype(to), strict=True)
            if to == DType.BOOLEAN:
                txt = pl.when(v.is_null()).then(None).when(low.is_in(BOOL_TRUE)).then(pl.lit("1")) \
                    .when(low.is_in(BOOL_FALSE)).then(pl.lit("0")).otherwise(pl.lit(sentinel(to)))
                return txt.cast(pl.Int8, strict=True).cast(pl.Boolean)
            if to == DType.DATE:
                txt = pl.when(v.is_null()).then(None).when(s.str.contains(DATE_RE)).then(s) \
                    .otherwise(pl.lit(sentinel(to)))
                return txt.str.to_date("%Y-%m-%d", strict=True)
            if to.kind == "Datetime":
                norm = s.str.replace("T", " ")
                norm = pl.when(norm.str.len_chars() == 16).then(norm + ":00").otherwise(norm)
                txt = pl.when(v.is_null()).then(None).when(s.str.contains(DATETIME_RE)).then(norm) \
                    .otherwise(pl.lit(sentinel(to)))
                return txt.str.to_datetime("%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=True)
        if to == DType.INT64 and src.is_float:  # must be whole; NaN/inf fail
            return pl.when(v.is_null()).then(None).when(v == v.floor()).then(v) \
                .otherwise(pl.lit(float("nan"))).cast(pl.Int64, strict=True)
        if to == DType.INT64 and src.is_decimal:  # must be whole: go through exact text
            txt = v.cast(pl.String).str.replace(r"\.0*$", "")
            ok = txt.str.contains(INT_RE)
            return pl.when(v.is_null()).then(None).when(ok).then(txt).otherwise(pl.lit(sentinel(to))) \
                .cast(pl.Int64, strict=True)
        if to.is_decimal and (src.is_float or src.is_decimal):  # round half away from zero to the scale
            return v.round(to.scale, mode="half_away_from_zero").cast(_pl_dtype(to), strict=True)
        if to == DType.STRING and src.kind == "Datetime":
            fmt_s, fmt_us = "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%.6f"
            return pl.when(v.dt.microsecond() == 0).then(v.dt.strftime(fmt_s)).otherwise(v.dt.strftime(fmt_us))
        if to == DType.STRING and src.is_float:
            return pl.when(v.is_nan()).then(pl.lit("NaN")).otherwise(v.cast(pl.String))
        return v.cast(_pl_dtype(to), strict=True)

    def _agg(self, e: E.Agg, schema):
        if e.operand is None:
            return pl.len()
        v = self._expr(e.operand, schema)
        t = e.operand.dtype(schema)
        fn = e.fn
        if t == DType.NULL and fn not in ("count", "n_unique"):  # older Polars can't aggregate Null columns
            return pl.lit(None).first()
        if fn == "sum":  # contract: sum of only nulls is null; integer overflow is an error
            total = v.cast(pl.Int128).sum().cast(pl.Int64, strict=True) if t == DType.INT64 else v.sum()
            return pl.when(v.count() == 0).then(None).otherwise(total)
        if fn == "max" and t.is_float:  # contract: NaN is the largest value
            return pl.when(v.is_nan().any()).then(pl.lit(float("nan"))).otherwise(v.max())
        if fn == "n_unique":  # contract: nulls are not counted
            return v.drop_nulls().n_unique()
        if fn in ("mean", "median", "quantile", "std", "var"):
            f = v.cast(pl.Float64) if t.is_decimal else v
            if fn == "quantile":
                return f.quantile(e.param, interpolation="linear")
            if fn in ("std", "var"):
                return getattr(f, fn)(ddof=1)
            return getattr(f, fn)()
        if fn in ("any", "all"):  # contract: null when there are no non-null values
            return pl.when(v.count() == 0).then(None).otherwise(getattr(v, fn)())
        return getattr(v, fn)()

    def _window(self, w: E.Window, schema):
        pb = w.partition_by or None
        inner = w.inner
        if isinstance(inner, E.Agg):
            a = self._agg(inner, schema)
            return a.over(pb) if pb else a
        order = [pl.col(c) for c in w.order_by]
        tiebreak = [(-pl.col(IDX).cast(pl.Int64)) if w.descending else pl.col(IDX)]
        fn = inner.fn
        if fn in ("rank", "dense_rank"):
            o = pl.col(w.order_by[0])
            r = o.rank("min" if fn == "rank" else "dense", descending=w.descending)
            if fn == "rank":
                nulls_at = o.count() + 1
            else:
                nulls_at = r.max().fill_null(0) + 1
            if pb:
                r, nulls_at = r.over(pb), nulls_at.over(pb)
            return pl.when(o.is_null()).then(nulls_at).otherwise(r)
        if fn == "row_number":
            base = pl.int_range(pl.len()) + 1
        elif fn == "cum_sum":
            v = self._expr(inner.operand, schema)
            t = inner.operand.dtype(schema)
            zero = pl.lit(0).cast(_pl_dtype(t))
            filled = v.fill_null(zero)
            running = filled.cast(pl.Int128).cum_sum().cast(pl.Int64, strict=True) if t == DType.INT64 else filled.cum_sum()
            base = pl.when(v.cum_count() == 0).then(None).otherwise(running)
        elif fn == "shift":
            base = self._expr(inner.operand, schema).shift(inner.param)
        else:  # pragma: no cover
            raise NotImplementedError(fn)
        return base.over(pb, order_by=order + tiebreak, descending=w.descending, nulls_last=True)
