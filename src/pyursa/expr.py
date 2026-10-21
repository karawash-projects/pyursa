"""Backend-neutral expression IR with static type checking."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .errors import InvalidOperationError, TypeMismatchError, column_not_found
from .types import DType, arithmetic_type, common_type, literal_dtype

ARITH = {"+", "-", "*", "/"}
COMPARE = {"==", "!=", "<", "<=", ">", ">="}
LOGIC = {"&", "|"}
AGGS = {"sum", "mean", "min", "max", "count", "n_unique", "len", "median", "quantile", "std", "var", "any", "all"}
WINDOW_ONLY = {"row_number", "rank", "dense_rank", "cum_sum", "shift"}
DT_PARTS = {"year", "month", "day", "weekday", "hour", "minute", "second"}
TRUNC_UNITS = {"year", "month", "week", "day", "hour"}


class Expr:
    """Base class. Expressions are immutable trees; nothing is evaluated here."""

    __hash__ = object.__hash__

    # --- operators -------------------------------------------------------
    def _bin(self, op: str, other: Any, reverse: bool = False) -> "Expr":
        other = wrap(other)
        return BinOp(op, other, self) if reverse else BinOp(op, self, other)

    def __add__(self, o): return self._bin("+", o)
    def __radd__(self, o): return self._bin("+", o, True)
    def __sub__(self, o): return self._bin("-", o)
    def __rsub__(self, o): return self._bin("-", o, True)
    def __mul__(self, o): return self._bin("*", o)
    def __rmul__(self, o): return self._bin("*", o, True)
    def __truediv__(self, o): return self._bin("/", o)
    def __rtruediv__(self, o): return self._bin("/", o, True)
    def __eq__(self, o): return self._bin("==", o)  # type: ignore[override]
    def __ne__(self, o): return self._bin("!=", o)  # type: ignore[override]
    def __lt__(self, o): return self._bin("<", o)
    def __le__(self, o): return self._bin("<=", o)
    def __gt__(self, o): return self._bin(">", o)
    def __ge__(self, o): return self._bin(">=", o)
    def __and__(self, o): return self._bin("&", o)
    def __rand__(self, o): return self._bin("&", o, True)
    def __or__(self, o): return self._bin("|", o)
    def __ror__(self, o): return self._bin("|", o, True)
    def __invert__(self): return Unary("not", self)
    def __neg__(self): return Unary("neg", self)

    def __bool__(self):
        raise InvalidOperationError(
            "An expression has no truth value; it is evaluated by the engine, not Python.",
            suggestions=[{"description": "Combine conditions with & and | instead of 'and'/'or'.",
                          "fix": '(col("a") > 1) & (col("b") < 5)'}],
        )

    # --- row-level methods -----------------------------------------------
    def alias(self, name: str) -> "Expr": return Alias(self, name)
    def is_null(self) -> "Expr": return Unary("is_null", self)
    def is_not_null(self) -> "Expr": return Unary("is_not_null", self)
    def is_in(self, values: Iterable[Any]) -> "Expr": return IsIn(self, list(values))
    def is_between(self, lower: Any, upper: Any) -> "Expr":
        """Inclusive on both ends."""
        return (self >= lower) & (self <= upper)
    def cast(self, dtype: DType | str) -> "Expr": return Cast(self, DType(dtype))
    def fill_null(self, value: Any) -> "Expr": return Coalesce([self, wrap(value)])
    def abs(self) -> "Expr": return Unary("abs", self)
    def round(self, decimals: int = 0) -> "Expr":
        """Round half away from zero (2.5 -> 3, -2.5 -> -3) on every engine."""
        return Round(self, decimals)

    # --- aggregations ----------------------------------------------------
    def sum(self) -> "Expr": return Agg("sum", self)
    def mean(self) -> "Expr": return Agg("mean", self)
    def min(self) -> "Expr": return Agg("min", self)
    def max(self) -> "Expr": return Agg("max", self)
    def count(self) -> "Expr": return Agg("count", self)
    def n_unique(self) -> "Expr": return Agg("n_unique", self)
    def median(self) -> "Expr": return Agg("median", self)
    def quantile(self, q: float) -> "Expr":
        """Continuous quantile with linear interpolation, q in [0, 1]."""
        return Agg("quantile", self, q)
    def std(self) -> "Expr":
        """Sample standard deviation (ddof=1)."""
        return Agg("std", self)
    def var(self) -> "Expr":
        """Sample variance (ddof=1)."""
        return Agg("var", self)
    def any(self) -> "Expr": return Agg("any", self)
    def all(self) -> "Expr": return Agg("all", self)

    # --- window-only functions --------------------------------------------
    def cum_sum(self) -> "Expr":
        """Running total; needs .over(..., order_by=...). Nulls are skipped (SQL semantics)."""
        return WinFn("cum_sum", self)
    def shift(self, n: int = 1) -> "Expr":
        """Value n rows earlier (n<0: later) within the window order; needs .over(order_by=...)."""
        return WinFn("shift", self, n)

    def over(self, partition_by: str | Sequence[str] | None = None, *, order_by: str | Sequence[str] | None = None,
             descending: bool = False) -> "Expr":
        """Evaluate per partition without collapsing rows, e.g. col("x").sum().over("region")."""
        pb = [] if partition_by is None else ([partition_by] if isinstance(partition_by, str) else list(partition_by))
        ob = [] if order_by is None else ([order_by] if isinstance(order_by, str) else list(order_by))
        return Window(self, pb, ob, descending)

    @property
    def str(self) -> "_StrNamespace":
        return _StrNamespace(self)

    @property
    def dt(self) -> "_DtNamespace":
        return _DtNamespace(self)

    # --- analysis (overridden) -------------------------------------------
    def dtype(self, schema: dict[str, DType]) -> DType: raise NotImplementedError
    def children(self) -> list["Expr"]: return []
    def output_name(self) -> str: raise NotImplementedError

    def columns(self) -> set[str]:
        out: set[str] = set()
        for c in self.children():
            out |= c.columns()
        return out

    def has_agg(self) -> bool:
        return any(c.has_agg() for c in self.children())

    def has_window(self) -> bool:
        return any(c.has_window() for c in self.children())

    def needs_row_index(self) -> bool:
        return any(c.needs_row_index() for c in self.children())

    def cols_outside_agg(self) -> set[str]:
        out: set[str] = set()
        for c in self.children():
            out |= c.cols_outside_agg()
        return out


class _StrNamespace:
    def __init__(self, e: Expr): self._e = e
    def contains(self, s: str) -> Expr: return StrOp("contains", self._e, s)
    def contains_regex(self, pattern: str) -> Expr:
        """Regex search. Use common syntax (RE2/Rust); lookarounds and backreferences are not portable."""
        return StrOp("contains_regex", self._e, pattern)
    def starts_with(self, s: str) -> Expr: return StrOp("starts_with", self._e, s)
    def ends_with(self, s: str) -> Expr: return StrOp("ends_with", self._e, s)
    def lower(self) -> Expr: return StrOp("lower", self._e)
    def upper(self) -> Expr: return StrOp("upper", self._e)
    def strip(self) -> Expr: return StrOp("strip", self._e)
    def len(self) -> Expr:
        """Length in characters."""
        return StrOp("len", self._e)
    def replace(self, old: str, new: str) -> Expr:
        """Replace every literal occurrence of `old`."""
        return StrOp("replace", self._e, old, new)
    def slice(self, start: int, length: int | None = None) -> Expr:
        """0-based substring."""
        return StrOp("slice", self._e, start, length)


class _DtNamespace:
    def __init__(self, e: Expr): self._e = e
    def year(self) -> Expr: return DtOp("year", self._e)
    def month(self) -> Expr: return DtOp("month", self._e)
    def day(self) -> Expr: return DtOp("day", self._e)
    def weekday(self) -> Expr:
        """ISO weekday: Monday=1 ... Sunday=7."""
        return DtOp("weekday", self._e)
    def hour(self) -> Expr: return DtOp("hour", self._e)
    def minute(self) -> Expr: return DtOp("minute", self._e)
    def second(self) -> Expr: return DtOp("second", self._e)
    def date(self) -> Expr: return DtOp("date", self._e)
    def truncate(self, unit: str) -> Expr:
        """Start of the year/month/week (ISO, Monday)/day/hour."""
        return DtOp("truncate", self._e, unit)


def wrap(value: Any) -> Expr:
    return value if isinstance(value, Expr) else Lit(value)


def _mismatch(e, msg, lt, rt, desc, fix=None) -> TypeMismatchError:
    s = {"description": desc}
    if fix:
        s["fix"] = fix
    return TypeMismatchError(msg, details={"expression": repr(e), "left_type": str(lt), "right_type": str(rt)},
                             suggestions=[s])


class Col(Expr):
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise InvalidOperationError(f"Column name must be a string, got {type(name).__name__}.")
        self.name = name

    def dtype(self, schema):
        if self.name not in schema:
            raise column_not_found(self.name, schema, "the input of this step")
        return schema[self.name]

    def output_name(self): return self.name
    def columns(self): return {self.name}
    def cols_outside_agg(self): return {self.name}
    def __repr__(self): return f'col("{self.name}")'


class Lit(Expr):
    def __init__(self, value: Any):
        import decimal as _decimal
        if isinstance(value, _decimal.Decimal) and value.is_finite():
            exp = value.as_tuple().exponent
            if isinstance(exp, int) and exp > 0:  # 1E+3 -> 1000 (engines mis-bind exponent form)
                if len(value.as_tuple().digits) + exp > 38:
                    from .errors import UnsupportedTypeError
                    raise UnsupportedTypeError(f"Decimal literal {value} needs more than 38 digits.")
                value = value.quantize(_decimal.Decimal(1), context=_decimal.Context(prec=38))
        self.value = value
        self._dtype = literal_dtype(value)

    def dtype(self, schema): return self._dtype
    def output_name(self): return "literal"
    def __repr__(self): return f"lit({_lit_repr(self.value)})"


def _lit_repr(v: Any) -> str:
    """Literal text that parse.py accepts back, so error fixes can be pasted."""
    import datetime as _dt
    import decimal as _decimal
    if isinstance(v, _decimal.Decimal):
        return f"decimal({str(v)!r})"
    if isinstance(v, _dt.datetime) and v.tzinfo is None:
        return f"datetime({v.year}, {v.month}, {v.day}, {v.hour}, {v.minute}, {v.second}, {v.microsecond})"
    if isinstance(v, _dt.date) and not isinstance(v, _dt.datetime):
        return f"date({v.year}, {v.month}, {v.day})"
    return repr(v)


class BinOp(Expr):
    def __init__(self, op: str, left: Expr, right: Expr):
        self.op, self.left, self.right = op, left, right

    def children(self): return [self.left, self.right]
    def output_name(self):
        return self.left.output_name() if not isinstance(self.left, Lit) else self.right.output_name()

    def dtype(self, schema):
        if self.op in COMPARE and (_is_none(self.left) or _is_none(self.right)):
            target = self.right if _is_none(self.left) else self.left
            method = "is_null" if self.op == "==" else "is_not_null"
            raise InvalidOperationError(
                f"Comparing to None with '{self.op}' always yields null, so it filters out every row.",
                details={"expression": repr(self)},
                suggestions=[{"description": f"Use .{method}() to test for missing values.",
                              "fix": f"{target!r}.{method}()"}],
            )
        lt, rt = self.left.dtype(schema), self.right.dtype(schema)
        if self.op in LOGIC:
            for side, t in ((self.left, lt), (self.right, rt)):
                if t not in (DType.BOOLEAN, DType.NULL):
                    raise _mismatch(self, f"'{self.op}' needs Boolean operands, but {side!r} is {t}.",
                                    lt, rt, "Wrap values in a comparison first, e.g. col(\"x\") > 0.")
            return DType.BOOLEAN
        if self.op in ARITH:
            for side, t in ((self.left, lt), (self.right, rt)):
                if not (t.is_numeric or t == DType.NULL):
                    hint = f'{side!r}.cast("Float64")' if t == DType.STRING else None
                    raise _mismatch(self, f"Arithmetic '{self.op}' needs numeric operands, but {side!r} is {t}.",
                                    lt, rt, "Cast the column to a numeric type.", hint)
            out = arithmetic_type(self.op, lt, rt)
            if out is None:
                raise _mismatch(self, f"Decimal result of {self!r} would need more than 38 digits of scale.",
                                lt, rt, "Round one side first or cast to Float64.", f'{self.left!r}.round(2)')
            return out
        if self.op in COMPARE:
            ct = common_type(lt, rt)
            if ct is None:
                fix = None
                if isinstance(self.right, Lit) and lt == DType.STRING:
                    fix = f"{self.left!r} {self.op} {str(self.right.value)!r}"
                elif isinstance(self.right, Lit) and lt.is_numeric and rt == DType.STRING:
                    try:
                        num = int(self.right.value) if lt.is_integer else float(self.right.value)
                        fix = f"{self.left!r} {self.op} {num!r}"
                    except ValueError:
                        fix = f'{self.left!r}.cast("String") {self.op} {self.right.value!r}'
                desc = "Make both sides the same type with .cast()."
                if lt.kind == rt.kind == "Datetime":
                    desc = "Both datetimes must use the same time zone; compare against a value in that zone."
                raise _mismatch(self, f"Cannot compare {lt} with {rt}.", lt, rt, desc, fix)
            if ct.is_nested:
                raise _mismatch(self, f"Cannot compare nested values ({ct}).", lt, rt, "Compare individual fields instead.")
            if self.op not in ("==", "!=") and not ct.is_orderable:
                raise _mismatch(self, f"'{self.op}' is not defined for {lt}.", lt, rt, "Use == or != instead.")
            return DType.BOOLEAN
        raise InvalidOperationError(f"Unknown operator '{self.op}'.")

    def __repr__(self): return f"({self.left!r} {self.op} {self.right!r})"


def _is_none(e: Expr) -> bool:
    return isinstance(e, Lit) and e.value is None


class Unary(Expr):
    def __init__(self, op: str, operand: Expr):
        self.op, self.operand = op, operand

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if self.op in ("is_null", "is_not_null"):
            return DType.BOOLEAN
        if self.op == "not":
            if t not in (DType.BOOLEAN, DType.NULL):
                raise TypeMismatchError(f"'~' needs a Boolean operand, got {t}.", details={"expression": repr(self)})
            return DType.BOOLEAN
        if self.op in ("neg", "abs"):
            if not t.is_numeric:
                raise TypeMismatchError(f"{self.op} needs a numeric operand, got {t}.", details={"expression": repr(self)})
            return t
        raise InvalidOperationError(f"Unknown unary op '{self.op}'.")

    def __repr__(self):
        if self.op == "not":
            return f"~{self.operand!r}"
        if self.op == "neg":
            return f"-{self.operand!r}"
        return f"{self.operand!r}.{self.op}()"


class Round(Expr):
    def __init__(self, operand: Expr, decimals: int):
        if not isinstance(decimals, int) or isinstance(decimals, bool) or decimals < 0:
            raise InvalidOperationError("round() needs a non-negative integer number of decimals.")
        self.operand, self.decimals = operand, decimals

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if not t.is_numeric:
            raise TypeMismatchError(f"round() needs a numeric operand, got {t}.", details={"expression": repr(self)})
        if t.is_decimal:
            return DType("Decimal", precision=38, scale=min(t.scale, self.decimals))  # type: ignore[type-var]
        return t

    def __repr__(self): return f"{self.operand!r}.round({self.decimals})"


class IsIn(Expr):
    def __init__(self, operand: Expr, values: list[Any]):
        self.operand, self.values = operand, values

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if t.is_nested:
            raise TypeMismatchError(f"is_in() is not defined for {t}.")
        for v in self.values:
            if v is None:
                raise InvalidOperationError("is_in() values may not contain None.",
                                            suggestions=[{"description": "Combine with .is_null() instead.",
                                                          "fix": f"{self.operand!r}.is_in([...]) | {self.operand!r}.is_null()"}])
            if common_type(t, literal_dtype(v)) is None:
                raise TypeMismatchError(f"is_in() value {v!r} ({literal_dtype(v)}) does not match column type {t}.",
                                        details={"expression": repr(self)})
        return DType.BOOLEAN

    def __repr__(self): return f"{self.operand!r}.is_in({self.values!r})"


class Cast(Expr):
    def __init__(self, operand: Expr, to: DType):
        self.operand, self.to = operand, to

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if not cast_allowed(t, self.to):
            hint = CAST_HINTS.get((t.kind, self.to.kind), "Convert explicitly, e.g. with a comparison or .dt / .str functions.")
            raise InvalidOperationError(
                f"Cannot cast {t} to {self.to}.",
                details={"expression": repr(self), "from": str(t), "to": str(self.to)},
                suggestions=[{"description": hint}],
            )
        return self.to

    def __repr__(self): return f'{self.operand!r}.cast("{self.to}")'


_NUMERIC = {"Int64", "Float64", "Decimal"}
_CASTS = {
    "Int64": _NUMERIC | {"String", "Boolean"},
    "Float64": _NUMERIC | {"String"},
    "Decimal": _NUMERIC | {"String"},
    "Boolean": {"Boolean", "Int64", "Float64", "String"},
    "String": _NUMERIC | {"String", "Boolean", "Date", "Datetime"},
    "Date": {"Date", "String", "Datetime"},
    "Datetime": {"Datetime", "Date", "String"},
}
CAST_HINTS = {
    ("Float64", "Boolean"): "Compare instead: col(\"x\") != 0.",
    ("Decimal", "Boolean"): "Compare instead: col(\"x\") != 0.",
    ("Date", "Int64"): "Use .dt.year(), .dt.month() or .dt.day().",
    ("Datetime", "Int64"): "Use the .dt functions.",
}


def cast_allowed(src: DType, to: DType) -> bool:
    """The strict cast matrix shared by every engine (see SEMANTICS['casts'])."""
    if to == DType.NULL or to.is_nested or src.is_nested:
        return False
    if src == DType.NULL or src == to:
        return True
    if to.kind not in _CASTS.get(src.kind, set()):
        return False
    if src.kind == "Datetime" and src.tz:  # zoned values only keep their zone
        return to == src
    if to.kind == "Datetime" and to.tz:  # can't invent a zone for naive values or text
        return False
    return True


class Coalesce(Expr):
    def __init__(self, items: list[Expr]):
        self.items = items

    def children(self): return list(self.items)
    def output_name(self): return self.items[0].output_name()

    def dtype(self, schema):
        out = DType.NULL
        for e in self.items:
            t = e.dtype(schema)
            ct = common_type(out, t)
            if ct is None:
                raise TypeMismatchError(f"fill_null value of type {t} does not match {out}.",
                                        details={"expression": repr(self)},
                                        suggestions=[{"description": "Use a fill value of the same type, or cast."}])
            out = ct
        return out

    def __repr__(self):
        if len(self.items) == 2:
            return f"{self.items[0]!r}.fill_null({self.items[1]!r})"
        return f"coalesce({', '.join(repr(i) for i in self.items)})"


class Case(Expr):
    """when(cond).then(value)[.when(...).then(...)].otherwise(default)"""

    def __init__(self, branches: list[tuple[Expr, Expr]], default: Expr | None):
        self.branches, self.default = branches, default

    def children(self):
        out: list[Expr] = []
        for c, v in self.branches:
            out += [c, v]
        return out + ([self.default] if self.default is not None else [])

    def output_name(self): return self.branches[0][1].output_name()

    def dtype(self, schema):
        out = DType.NULL
        for cond, _value in self.branches:
            ct = cond.dtype(schema)
            if ct not in (DType.BOOLEAN, DType.NULL):
                raise TypeMismatchError(f"when() needs a Boolean condition, got {ct}.", details={"condition": repr(cond)})
        values = [v for _, v in self.branches] + ([self.default] if self.default is not None else [])
        for v in values:
            t = v.dtype(schema)
            merged = common_type(out, t)
            if merged is None:
                raise TypeMismatchError(
                    f"when/then branches have incompatible types ({out} and {t}).",
                    details={"expression": repr(self)},
                    suggestions=[{"description": "Make every then()/otherwise() value the same type, e.g. with .cast()."}],
                )
            out = merged
        return out

    def when(self, condition: Any) -> "_When":
        return _When(self.branches, wrap(condition))

    def otherwise(self, value: Any) -> "Case":
        return Case(self.branches, wrap(value))

    def __repr__(self):
        s = "".join(f"when({c!r}).then({v!r})." for c, v in self.branches)
        return s + (f"otherwise({self.default!r})" if self.default is not None else "otherwise(None)")


class _When:
    def __init__(self, branches: list[tuple[Expr, Expr]], condition: Expr):
        self._branches, self._condition = branches, condition

    def then(self, value: Any) -> Case:
        return Case(self._branches + [(self._condition, wrap(value))], None)


def when(condition: Any) -> _When:
    """Conditional expression: pu.when(col("x") > 0).then("pos").otherwise("non-pos")."""
    return _When([], wrap(condition))


class StrOp(Expr):
    def __init__(self, op: str, operand: Expr, arg: Any = None, arg2: Any = None):
        self.op, self.operand, self.arg, self.arg2 = op, operand, arg, arg2

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if t not in (DType.STRING, DType.NULL):
            raise TypeMismatchError(
                f".str.{self.op}() needs a String column, got {t}.",
                details={"expression": repr(self)},
                suggestions=[{"description": "Cast to String first.",
                              "fix": f'{self.operand!r}.cast("String").str.{self.op}(...)'}],
            )
        if self.op in ("contains", "contains_regex", "starts_with", "ends_with", "replace"):
            if not isinstance(self.arg, str):
                raise TypeMismatchError(f".str.{self.op}() argument must be a string.")
        if self.op == "replace" and not isinstance(self.arg2, str):
            raise TypeMismatchError(".str.replace() needs (old: str, new: str).")
        if self.op == "replace" and self.arg == "":
            raise InvalidOperationError(".str.replace() needs a non-empty `old` string.")
        if self.op == "slice":
            if not isinstance(self.arg, int) or self.arg < 0 or (self.arg2 is not None and (not isinstance(self.arg2, int) or self.arg2 < 0)):
                raise InvalidOperationError(".str.slice(start, length) needs non-negative integers.")
        if self.op == "contains_regex":
            import re as _re
            try:
                _re.compile(self.arg)
            except _re.error as e:
                raise InvalidOperationError(f"Invalid regex {self.arg!r}: {e}.") from None
            if _re.search(r"\(\?<?[=!]|\(\?P=|\\[1-9]", self.arg):
                raise InvalidOperationError(
                    f"Regex {self.arg!r} uses lookaround or backreferences, which not every engine supports.",
                    suggestions=[{"description": "Rewrite without (?=...), (?!...), (?<=...), (?<!...) or \\1; "
                                                 "combine several contains_regex() conditions with & and ~ instead."}],
                )
        if self.op in ("contains", "contains_regex", "starts_with", "ends_with"):
            return DType.BOOLEAN
        if self.op == "len":
            return DType.INT64
        return DType.STRING

    def __repr__(self):
        args = [repr(a) for a in (self.arg, self.arg2) if a is not None]
        return f"{self.operand!r}.str.{self.op}({', '.join(args)})"


class DtOp(Expr):
    def __init__(self, op: str, operand: Expr, unit: str | None = None):
        self.op, self.operand, self.unit = op, operand, unit

    def children(self): return [self.operand]
    def output_name(self): return self.operand.output_name()

    def dtype(self, schema):
        t = self.operand.dtype(schema)
        if not t.is_temporal:
            raise TypeMismatchError(
                f".dt.{self.op}() needs a Date or Datetime column, got {t}.",
                details={"expression": repr(self)},
                suggestions=[{"description": "Cast ISO-formatted text first.", "fix": f'{self.operand!r}.cast("Date")'}],
            )
        if self.op in ("hour", "minute", "second") and t == DType.DATE:
            raise TypeMismatchError(f".dt.{self.op}() needs a Datetime column; {self.operand!r} is a Date.")
        if self.op in DT_PARTS:
            return DType.INT64
        if self.op == "date":
            return DType.DATE
        if self.op == "truncate":
            if self.unit not in TRUNC_UNITS:
                raise InvalidOperationError(f"truncate() unit must be one of {sorted(TRUNC_UNITS)}, got {self.unit!r}.")
            if self.unit == "hour" and t == DType.DATE:
                raise InvalidOperationError("Cannot truncate a Date to the hour.")
            return t
        raise InvalidOperationError(f"Unknown dt op '{self.op}'.")

    def __repr__(self):
        return f"{self.operand!r}.dt.{self.op}({'' if self.unit is None else repr(self.unit)})"


class Agg(Expr):
    def __init__(self, fn: str, operand: Expr | None, param: Any = None):
        self.fn, self.operand, self.param = fn, operand, param

    def children(self): return [self.operand] if self.operand is not None else []
    def has_agg(self): return True
    def cols_outside_agg(self): return set()
    def output_name(self): return "len" if self.operand is None else self.operand.output_name()

    def dtype(self, schema):
        if self.operand is None:
            return DType.INT64
        if self.operand.has_agg():
            raise InvalidOperationError(f"Nested aggregation in {self!r} is not allowed.", details={"expression": repr(self)})
        if self.operand.has_window():
            raise InvalidOperationError(f"Cannot aggregate a window expression in {self!r}.",
                                        suggestions=[{"description": "Compute the window column with with_columns() first."}])
        t = self.operand.dtype(schema)
        if self.fn == "count":
            return DType.INT64
        if t.is_nested:
            raise TypeMismatchError(f"{self.fn}() is not defined for nested type {t}; only count() is.",
                                    details={"expression": repr(self)})
        if self.fn == "n_unique":
            return DType.INT64
        if self.fn in ("sum", "mean", "median", "quantile", "std", "var"):
            if not (t.is_numeric or t == DType.NULL):
                raise TypeMismatchError(
                    f"{self.fn}() needs a numeric column, but {self.operand!r} is {t}.",
                    details={"expression": repr(self), "type": str(t)},
                    suggestions=[{"description": "Use count() or n_unique() for non-numeric columns, or cast first.",
                                  "fix": f"{self.operand!r}.n_unique()"}],
                )
            if self.fn == "quantile" and not (isinstance(self.param, (int, float)) and 0 <= self.param <= 1):
                raise InvalidOperationError("quantile(q) needs 0 <= q <= 1.")
            if self.fn == "sum":
                if t.is_decimal:
                    return DType("Decimal", precision=38, scale=t.scale)
                return t if t != DType.NULL else DType.INT64
            return DType.FLOAT64
        if self.fn in ("min", "max"):
            if not (t.is_orderable or t == DType.NULL):
                raise TypeMismatchError(f"{self.fn}() is not defined for {t}.")
            return t
        if self.fn in ("any", "all"):
            if t not in (DType.BOOLEAN, DType.NULL):
                raise TypeMismatchError(f"{self.fn}() needs a Boolean column, got {t}.",
                                        suggestions=[{"description": "Aggregate a condition instead.",
                                                      "fix": f"({self.operand!r} > 0).{self.fn}()"}])
            return DType.BOOLEAN
        raise InvalidOperationError(f"Unknown aggregation '{self.fn}'.")

    def __repr__(self):
        if self.operand is None:
            return "len()"
        p = "" if self.param is None else repr(self.param)
        return f"{self.operand!r}.{self.fn}({p})"


class WinFn(Expr):
    """Order-dependent window functions: row_number, rank, dense_rank, cum_sum, shift."""

    def __init__(self, fn: str, operand: Expr | None, param: Any = None):
        self.fn, self.operand, self.param = fn, operand, param

    def children(self): return [self.operand] if self.operand is not None else []
    def output_name(self): return self.fn if self.operand is None else self.operand.output_name()

    def dtype(self, schema):
        if self.fn in ("row_number", "rank", "dense_rank"):
            return DType.INT64
        t = self.operand.dtype(schema)  # type: ignore[union-attr]
        if self.fn == "cum_sum":
            if not t.is_numeric:
                raise TypeMismatchError(f"cum_sum() needs a numeric column, got {t}.")
            return DType("Decimal", precision=38, scale=t.scale) if t.is_decimal else t
        if self.fn == "shift":
            if not isinstance(self.param, int) or isinstance(self.param, bool):
                raise InvalidOperationError("shift(n) needs an integer.")
            return t
        raise InvalidOperationError(f"Unknown window function '{self.fn}'.")

    def __repr__(self):
        if self.operand is None:
            return f"{self.fn}()"
        return f"{self.operand!r}.{self.fn}({'' if self.param is None else self.param!r})"


class Window(Expr):
    def __init__(self, inner: Expr, partition_by: list[str], order_by: list[str], descending: bool):
        self.inner, self.partition_by, self.order_by, self.descending = inner, partition_by, order_by, descending

    def children(self): return [self.inner]
    def output_name(self): return self.inner.output_name()
    def has_agg(self): return False
    def has_window(self): return True
    def needs_row_index(self): return True
    def cols_outside_agg(self): return set(self.partition_by) | set(self.order_by) | self.inner.columns()
    def columns(self): return set(self.partition_by) | set(self.order_by) | self.inner.columns()

    def dtype(self, schema):
        for c in self.partition_by + self.order_by:
            if c not in schema:
                raise column_not_found(c, schema, "over()")
            if not schema[c].is_hashable:
                raise TypeMismatchError(f"over() cannot partition or order by nested column '{c}'.")
        for c in self.order_by:
            if not schema[c].is_orderable:
                raise TypeMismatchError(f"over() cannot order by {schema[c]} column '{c}'.")
        inner = self.inner
        if isinstance(inner, Alias):
            raise InvalidOperationError("Put .alias() after .over(), not inside it.",
                                        suggestions=[{"description": "Alias the window result.",
                                                      "fix": 'col("x").sum().over("k").alias("x_total")'}])
        if isinstance(inner, WinFn):
            if not self.order_by:
                raise InvalidOperationError(
                    f"{inner.fn}() depends on row order, so over() needs order_by=.",
                    suggestions=[{"description": "Specify how rows are ordered within each partition.",
                                  "fix": f'{inner!r}.over("<partition>", order_by="<column>")'}],
                )
            if inner.fn in ("rank", "dense_rank") and len(self.order_by) != 1:
                raise InvalidOperationError(f"{inner.fn}() needs exactly one order_by column.")
            return inner.dtype(schema)
        if isinstance(inner, Agg):
            if self.order_by:
                raise InvalidOperationError(
                    "Aggregations over a window ignore order; drop order_by, or use cum_sum() for running totals.")
            if inner.fn not in ("sum", "mean", "min", "max", "count", "len", "n_unique"):
                raise InvalidOperationError(f"{inner.fn}() is not supported over a window yet.")
            return inner.dtype(schema)
        raise InvalidOperationError(
            "over() applies to an aggregation (sum, mean, min, max, count, len, n_unique) "
            "or a window function (row_number, rank, dense_rank, cum_sum, shift).",
            details={"expression": repr(inner)},
        )

    def __repr__(self):
        parts = []
        if self.partition_by:
            parts.append(repr(self.partition_by[0] if len(self.partition_by) == 1 else self.partition_by))
        if self.order_by:
            parts.append(f"order_by={self.order_by!r}")
        if self.descending:
            parts.append("descending=True")
        return f"{self.inner!r}.over({', '.join(parts)})"


class Alias(Expr):
    def __init__(self, operand: Expr, name: str):
        if not isinstance(name, str) or not name:
            raise InvalidOperationError("alias() needs a non-empty string name.")
        self.operand, self.name = operand, name

    def children(self): return [self.operand]
    def output_name(self): return self.name
    def dtype(self, schema): return self.operand.dtype(schema)
    def __repr__(self): return f'{self.operand!r}.alias("{self.name}")'


def col(name: str) -> Col:
    """Reference a column."""
    return Col(name)


def lit(value: Any) -> Lit:
    """A literal value."""
    return Lit(value)


def len_() -> Agg:
    """Row count aggregation (counts every row, including nulls)."""
    return Agg("len", None)


def row_number() -> WinFn:
    """1, 2, 3... within each partition; use with .over(..., order_by=...). Ties keep input order."""
    return WinFn("row_number", None)


def rank(dense: bool = False) -> WinFn:
    """SQL RANK (or DENSE_RANK) by the window's single order_by column; ties share a rank, nulls rank last.
    Use as pu.rank().over("team", order_by="score", descending=True)."""
    return WinFn("dense_rank" if dense else "rank", None)


def coalesce(*exprs: Any) -> Coalesce:
    """First non-null value among the arguments."""
    if len(exprs) < 2:
        raise InvalidOperationError("coalesce() needs at least two arguments.")
    return Coalesce([wrap(e) for e in exprs])


def unalias(e: Expr) -> Expr:
    return e.operand if isinstance(e, Alias) else e
