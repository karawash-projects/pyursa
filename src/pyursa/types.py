"""A small, strict, engine-neutral type system.

Every backend maps to and from these types, and every result is cast back to
them, so the same plan returns identically-typed Arrow data on any engine.
Nothing is converted silently: unsupported or lossy inputs raise an error.

Types: Int64, Float64, String, Boolean, Date, Datetime / Datetime("UTC"),
Decimal(precision, scale), List(inner), Struct({"field": type}), Null.
"""

from __future__ import annotations

import datetime as _dt
import decimal as _decimal
import re
from typing import Any

import pyarrow as pa

from .errors import UnsupportedTypeError

_SIMPLE = ("Int64", "Float64", "String", "Boolean", "Date", "Null")
MAX_DECIMAL_PRECISION = 38


class DType:
    """An immutable engine-neutral data type."""

    __slots__ = ("kind", "tz", "precision", "scale", "inner", "fields")

    # filled in after the class body
    INT64: "DType"
    FLOAT64: "DType"
    STRING: "DType"
    BOOLEAN: "DType"
    DATE: "DType"
    DATETIME: "DType"
    NULL: "DType"

    def __new__(cls, spec: "str | DType", *, tz: str | None = None, precision: int | None = None,
                scale: int | None = None, inner: "DType | None" = None,
                fields: "tuple[tuple[str, DType], ...] | None" = None) -> "DType":
        if isinstance(spec, DType):
            return spec
        if not isinstance(spec, str):
            raise UnsupportedTypeError(f"Not a type: {spec!r}.",
                                       suggestions=[{"description": "Use a type name such as \"Int64\" or pu.Decimal(10, 2)."}])
        if any(ch in spec for ch in "(["):
            return _parse(spec)
        kind = spec.strip()
        if kind not in _SIMPLE and kind not in ("Datetime", "Decimal", "List", "Struct"):
            raise UnsupportedTypeError(
                f"Unknown type '{spec}'.",
                details={"known": list(_SIMPLE) + ["Datetime", "Datetime(tz)", "Decimal(p,s)", "List(T)", "Struct(...)"]},
                suggestions=[{"description": "Use one of the supported type names."}],
            )
        self = object.__new__(cls)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "tz", tz)
        if kind == "Decimal":
            precision = MAX_DECIMAL_PRECISION if precision is None else precision
            scale = 0 if scale is None else scale
            if not (1 <= precision <= MAX_DECIMAL_PRECISION and 0 <= scale <= precision):
                raise UnsupportedTypeError(f"Invalid Decimal({precision}, {scale}); need 1 <= precision <= 38 and 0 <= scale <= precision.")
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "scale", scale)
        if kind == "List" and inner is None:
            raise UnsupportedTypeError("List needs an inner type, e.g. List(Int64).")
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "fields", tuple(fields) if fields is not None else None)
        return self

    def __setattr__(self, *_):
        raise AttributeError("DType is immutable")

    def __call__(self, *args: Any) -> "DType":
        """pu.Datetime("UTC") -> timezone-aware datetime type."""
        if self.kind == "Datetime" and self.tz is None and len(args) == 1:
            return DType("Datetime", tz=args[0])
        raise TypeError(f"{self} is not callable with {args!r}")

    def _key(self):
        return (self.kind, self.tz, self.precision, self.scale, self.inner, self.fields)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            try:
                other = DType(other)
            except Exception:
                return False
        return isinstance(other, DType) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __str__(self) -> str:
        if self.kind == "Datetime" and self.tz:
            return f"Datetime({self.tz})"
        if self.kind == "Decimal":
            return f"Decimal({self.precision},{self.scale})"
        if self.kind == "List":
            return f"List({self.inner})"
        if self.kind == "Struct":
            return "Struct(" + ", ".join(f"{n}: {t}" for n, t in self.fields or ()) + ")"
        return self.kind

    __repr__ = __str__

    @property
    def value(self) -> str:  # backwards compatibility with the former Enum
        return str(self)

    # --- classification ---------------------------------------------------
    @property
    def is_integer(self) -> bool: return self.kind == "Int64"
    @property
    def is_float(self) -> bool: return self.kind == "Float64"
    @property
    def is_decimal(self) -> bool: return self.kind == "Decimal"
    @property
    def is_numeric(self) -> bool: return self.kind in ("Int64", "Float64", "Decimal")
    @property
    def is_temporal(self) -> bool: return self.kind in ("Date", "Datetime")
    @property
    def is_nested(self) -> bool: return self.kind in ("List", "Struct")
    @property
    def is_orderable(self) -> bool:
        return self.is_numeric or self.is_temporal or self.kind in ("String", "Boolean")
    @property
    def is_hashable(self) -> bool:  # usable as group/join/distinct key
        return not self.is_nested and self.kind != "Null"

    def to_arrow(self) -> pa.DataType:
        k = self.kind
        if k == "Int64":
            return pa.int64()
        if k == "Float64":
            return pa.float64()
        if k == "String":
            return pa.string()
        if k == "Boolean":
            return pa.bool_()
        if k == "Date":
            return pa.date32()
        if k == "Datetime":
            return pa.timestamp("us", tz=self.tz)
        if k == "Decimal":
            return pa.decimal128(self.precision, self.scale)
        if k == "List":
            return pa.list_(self.inner.to_arrow())  # type: ignore[union-attr]
        if k == "Struct":
            return pa.struct([(n, t.to_arrow()) for n, t in self.fields or ()])
        return pa.null()


DType.INT64 = DType("Int64")
DType.FLOAT64 = DType("Float64")
DType.STRING = DType("String")
DType.BOOLEAN = DType("Boolean")
DType.DATE = DType("Date")
DType.DATETIME = DType("Datetime")
DType.NULL = DType("Null")

Int64, Float64, String, Boolean, Date, Datetime, Null = (
    DType.INT64, DType.FLOAT64, DType.STRING, DType.BOOLEAN, DType.DATE, DType.DATETIME, DType.NULL,
)


def Decimal(precision: int = MAX_DECIMAL_PRECISION, scale: int = 0) -> DType:  # noqa: N802
    """Exact decimal type, e.g. Decimal(12, 2) for money."""
    return DType("Decimal", precision=precision, scale=scale)


def List(inner: "DType | str") -> DType:  # noqa: N802
    return DType("List", inner=DType(inner))


def Struct(fields: "dict[str, DType | str]") -> DType:  # noqa: N802
    return DType("Struct", fields=tuple((n, DType(t)) for n, t in fields.items()))


_TOKEN = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_/+\-]*|\d+|[(),:\[\]])")


def _parse(spec: str) -> DType:
    toks = _TOKEN.findall(spec)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def take(expected=None):
        nonlocal pos
        tok = peek()
        if tok is None or (expected and tok != expected):
            raise UnsupportedTypeError(f"Cannot parse type '{spec}'.")
        pos += 1
        return tok

    def parse_type() -> DType:
        name = take()
        if name == "Datetime" and peek() == "(":
            take("(")
            parts = []
            while peek() not in (")", None):
                parts.append(take())
            take(")")
            return DType("Datetime", tz="".join(parts))
        if name == "Decimal" and peek() == "(":
            take("(")
            p = int(take())
            s = 0
            if peek() == ",":
                take(",")
                s = int(take())
            take(")")
            return Decimal(p, s)
        if name == "List" and peek() in ("(", "["):
            close = ")" if take() == "(" else "]"
            inner = parse_type()
            take(close)
            return List(inner)
        if name == "Struct" and peek() == "(":
            take("(")
            fields = {}
            while peek() != ")":
                fname = take()
                take(":")
                fields[fname] = parse_type()
                if peek() == ",":
                    take(",")
            take(")")
            return Struct(fields)
        return DType(name)

    t = parse_type()
    if pos != len(toks):
        raise UnsupportedTypeError(f"Cannot parse type '{spec}'.")
    return t


Schema = dict  # ordered mapping: column name -> DType


def from_arrow(t: pa.DataType) -> DType:
    if pa.types.is_dictionary(t):
        return from_arrow(t.value_type)  # categorical -> its values (lossless decode)
    if pa.types.is_boolean(t):
        return DType.BOOLEAN
    if pa.types.is_integer(t):
        return DType.INT64
    if pa.types.is_floating(t):
        return DType.FLOAT64
    if pa.types.is_decimal(t):
        if t.precision > MAX_DECIMAL_PRECISION:
            raise UnsupportedTypeError(
                f"Decimal precision {t.precision} exceeds the supported maximum of 38.",
                details={"arrow_type": str(t)},
                suggestions=[{"description": "Cast to decimal128(38, s) or float64 before loading, accepting the loss explicitly."}],
            )
        return Decimal(t.precision, t.scale)
    if pa.types.is_string(t) or pa.types.is_large_string(t) or getattr(pa.types, "is_string_view", lambda _: False)(t):
        return DType.STRING
    if pa.types.is_date(t):
        return DType.DATE
    if pa.types.is_timestamp(t):
        return DType("Datetime", tz=normalize_tz(t.tz))
    if pa.types.is_null(t):
        return DType.NULL
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return List(from_arrow(t.value_type))
    if pa.types.is_struct(t):
        return Struct({t.field(i).name: from_arrow(t.field(i).type) for i in range(t.num_fields)})
    raise UnsupportedTypeError(
        f"Arrow type '{t}' is not supported.",
        details={"arrow_type": str(t)},
        suggestions=[{"description": "Cast the column to a supported type before loading, "
                                     "e.g. binary -> string, time -> string, duration -> int64."}],
    )


RESERVED_PREFIX = "__pu_"


def validate_names(names: Any, where: str = "the result") -> None:
    """Column names must be unique ignoring case (DuckDB resolves identifiers case-insensitively),
    contain no NUL character, and not use the reserved internal prefix."""
    from .errors import DuplicateColumnError, InvalidOperationError
    seen: dict[str, str] = {}
    for n in names:
        if not isinstance(n, str) or n == "":
            raise InvalidOperationError(f"Column names must be non-empty strings, got {n!r}.")
        if "\x00" in n:
            raise InvalidOperationError(f"Column name {n!r} contains a NUL character.")
        if n.startswith(RESERVED_PREFIX):
            raise InvalidOperationError(f"Column name '{n}' uses the reserved prefix '{RESERVED_PREFIX}'.",
                                        suggestions=[{"description": "Rename the column."}])
        key = n.casefold()
        if key in seen:
            if seen[key] == n:
                raise DuplicateColumnError(f"{where} has two columns named '{n}'.",
                                           suggestions=[{"description": "Rename or alias one of them."}])
            raise DuplicateColumnError(
                f"{where} has columns '{seen[key]}' and '{n}' that differ only by case, which not every engine "
                "can tell apart.", details={"columns": [seen[key], n]},
                suggestions=[{"description": "Rename one of them so the names differ by more than case."}])
        seen[key] = n


def schema_from_arrow(schema: pa.Schema) -> Schema:
    validate_names(schema.names, "The input")
    return {f.name: from_arrow(f.type) for f in schema}


def to_arrow_schema(schema: Schema) -> pa.Schema:
    return pa.schema([(n, t.to_arrow()) for n, t in schema.items()])


def cast_column(column: pa.ChunkedArray | pa.Array, target: pa.DataType, name: str = ""):
    """Safe cast: raises instead of truncating, overflowing or losing precision."""
    if column.type.equals(target):
        return column
    try:
        if pa.types.is_dictionary(column.type):
            column = column.cast(column.type.value_type)
        return column.cast(target, safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
        raise UnsupportedTypeError(
            f"Column '{name}' cannot be converted from {column.type} to {target} without losing data: {e}",
            details={"column": name, "from": str(column.type), "to": str(target)},
            suggestions=[{"description": "Cast it yourself first (e.g. to float64 or string) to accept the conversion explicitly."}],
        ) from None


def normalize_table(table: pa.Table) -> pa.Table:
    """Cast an Arrow table to canonical types, refusing lossy conversions."""
    target = to_arrow_schema(schema_from_arrow(table.schema))
    if table.schema.equals(target):
        return table
    cols = [cast_column(c, f.type, f.name) for f, c in zip(target, table.columns)]
    return pa.Table.from_arrays(cols, schema=target)


def literal_dtype(value: Any) -> DType:
    if value is None:
        return DType.NULL
    if isinstance(value, bool):
        return DType.BOOLEAN
    if isinstance(value, int):
        if not -(2 ** 63) <= value < 2 ** 63:
            raise UnsupportedTypeError(f"Integer literal {value} is outside the Int64 range.",
                                       suggestions=[{"description": "Use a Decimal literal for larger values."}])
        return DType.INT64
    if isinstance(value, float):
        return DType.FLOAT64
    if isinstance(value, _decimal.Decimal):
        if not value.is_finite():
            raise UnsupportedTypeError(f"Decimal literal {value} is not a finite number.",
                                       suggestions=[{"description": "Use a float such as float('nan') for non-finite values."}])
        exp = value.as_tuple().exponent
        scale = max(0, -exp) if isinstance(exp, int) else 0
        if scale > MAX_DECIMAL_PRECISION or len(value.as_tuple().digits) + max(0, exp) > MAX_DECIMAL_PRECISION:
            raise UnsupportedTypeError(f"Decimal literal {value} needs more than 38 digits.")
        return Decimal(MAX_DECIMAL_PRECISION, scale)
    if isinstance(value, str):
        return DType.STRING
    if isinstance(value, _dt.datetime):
        return DType("Datetime", tz=tz_name(value.tzinfo)) if value.tzinfo else DType.DATETIME
    if isinstance(value, _dt.date):
        return DType.DATE
    raise UnsupportedTypeError(
        f"Literal of Python type '{type(value).__name__}' is not supported.",
        details={"value": repr(value)},
    )


_OFFSET = re.compile(r"^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$")


def normalize_tz(tz: str | None) -> str | None:
    """Map fixed offsets ("+05:00", "UTC+05:00") to IANA names every engine understands."""
    if tz is None:
        return None
    m = _OFFSET.match(tz.strip())
    if not m:
        return tz
    sign, hours, minutes = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if minutes:
        raise UnsupportedTypeError(
            f"Fixed UTC offset '{tz}' is not supported by every engine.",
            suggestions=[{"description": "Use an IANA zone name instead, e.g. 'Asia/Kolkata' for +05:30."}],
        )
    if hours == 0:
        return "UTC"
    return f"Etc/GMT{'-' if sign == '+' else '+'}{hours}"  # Etc/GMT signs are inverted by convention


def tz_name(tzinfo: Any) -> str | None:
    if tzinfo is None:
        return None
    key = getattr(tzinfo, "key", None) or getattr(tzinfo, "zone", None)
    if key:
        return key
    off = tzinfo.utcoffset(None)
    if off == _dt.timedelta(0):
        return "UTC"
    if off is not None:
        total = int(off.total_seconds())
        sign = "+" if total >= 0 else "-"
        h, rem = divmod(abs(total), 3600)
        return normalize_tz(f"{sign}{h:02d}:{rem // 60:02d}")
    name = tzinfo.tzname(None)
    return name or str(tzinfo)


def common_type(a: DType, b: DType) -> DType | None:
    """Type both sides can be compared/combined as, or None if incompatible."""
    if a == b:
        return a
    if a == DType.NULL:
        return b
    if b == DType.NULL:
        return a
    if a.is_numeric and b.is_numeric:
        if DType.FLOAT64 in (a, b):
            return DType.FLOAT64
        if a.is_decimal and b.is_decimal:
            return Decimal(MAX_DECIMAL_PRECISION, max(a.scale, b.scale))  # type: ignore[arg-type,type-var]
        dec = a if a.is_decimal else b
        return Decimal(MAX_DECIMAL_PRECISION, dec.scale)  # type: ignore[arg-type]
    return None


def arithmetic_type(op: str, a: DType, b: DType) -> DType | None:
    """Result type of a +,-,* b for numeric operands (division is always Float64)."""
    if op == "/":
        return DType.FLOAT64
    if a == DType.NULL and b == DType.NULL:
        return DType.NULL
    if a == DType.NULL or b == DType.NULL:
        return a if b == DType.NULL else b
    if op == "*" and a.is_decimal and b.is_decimal:
        scale = a.scale + b.scale  # type: ignore[operator]
        if scale > MAX_DECIMAL_PRECISION:
            return None
        return Decimal(MAX_DECIMAL_PRECISION, scale)
    return common_type(a, b)
