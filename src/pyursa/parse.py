"""Safe parser for expression strings sent by agents over a tool interface.

Agents write the same syntax as the Python API, e.g.
    (col("qty") > 5) & col("region").is_in(["east", "west"])
The string is parsed with `ast` against a strict whitelist and converted to
IR nodes. Nothing is ever passed to eval/exec, so no arbitrary code can run.
"""

from __future__ import annotations

import ast
import datetime as _dt
import decimal as _decimal
from typing import Any

from .errors import InvalidOperationError, PyursaError
from .expr import Case, Expr, _When, coalesce, col, len_, lit, rank, row_number, when

MAX_LENGTH = 2000
MAX_NODES = 400

_FUNCS = {"col": col, "lit": lit, "len": len_, "when": when, "coalesce": coalesce,
          "row_number": row_number, "rank": rank}
_VALUE_FUNCS = {"date": _dt.date, "datetime": _dt.datetime, "decimal": lambda v: _decimal.Decimal(str(v))}
_METHODS = {
    "alias", "is_null", "is_not_null", "is_in", "is_between", "cast", "fill_null", "abs", "round",
    "sum", "mean", "min", "max", "count", "n_unique", "median", "quantile", "std", "var", "any", "all",
    "cum_sum", "shift", "over", "then", "when", "otherwise",
}
_STR_METHODS = {"contains", "contains_regex", "starts_with", "ends_with", "lower", "upper", "strip", "len",
                "replace", "slice"}
_DT_METHODS = {"year", "month", "day", "weekday", "hour", "minute", "second", "date", "truncate"}
_KEYWORDS = {"over": {"order_by", "descending"}, "rank": {"dense"}, "date": None, "datetime": None}
_BINOPS = {ast.Add: "__add__", ast.Sub: "__sub__", ast.Mult: "__mul__", ast.Div: "__truediv__",
           ast.BitAnd: "__and__", ast.BitOr: "__or__"}
_CMPOPS = {ast.Eq: "__eq__", ast.NotEq: "__ne__", ast.Lt: "__lt__", ast.LtE: "__le__",
           ast.Gt: "__gt__", ast.GtE: "__ge__"}


def _fail(msg: str, source: str, node: ast.AST | None = None, fix: str | None = None) -> InvalidOperationError:
    details: dict[str, Any] = {"expression": source}
    if node is not None and hasattr(node, "col_offset"):
        details["position"] = node.col_offset
    s = {"description": "Use the Python API syntax: col(), lit(), pyursa methods and operators & | ~."}
    if fix:
        s["fix"] = fix
    return InvalidOperationError(msg, details=details, suggestions=[s], code="EXPRESSION_SYNTAX")


def parse_expr(source: str) -> Expr:
    if not isinstance(source, str):
        raise _fail("Expression must be a string.", repr(source))
    if len(source) > MAX_LENGTH:
        raise _fail(f"Expression is longer than {MAX_LENGTH} characters.", source[:80] + "…")
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as e:
        raise _fail(f"Invalid syntax: {e.msg}.", source) from None
    except (RecursionError, MemoryError, ValueError):
        raise _fail("Expression is too deeply nested.", source) from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise _fail("Expression is too complex.", source)
    try:
        out = _Converter(source).visit(tree.body)
    except PyursaError as e:
        if e.code == "EXPRESSION_SYNTAX":
            raise
        raise _fail(str(e.message), source) from None
    except (TypeError, ValueError, AttributeError, RecursionError, MemoryError, OverflowError) as e:
        raise _fail(f"Invalid expression: {type(e).__name__}: {e}", source) from None
    if isinstance(out, (list, tuple, dict)):
        raise _fail("An expression must be a column, value or computation, not a list.", source)
    if not isinstance(out, Expr):
        # a bare column name like "qty" is a common agent shorthand
        if isinstance(out, str):
            return col(out)
        return lit(out)
    return out


class _Converter:
    def __init__(self, source: str) -> None:
        self.source = source

    def visit(self, node: ast.AST) -> Any:
        method = getattr(self, "v_" + type(node).__name__, None)
        if method is None:
            raise _fail(f"'{type(node).__name__}' is not allowed in expressions.", self.source, node)
        return method(node)

    def v_Constant(self, node: ast.Constant) -> Any:
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise _fail(f"Literal {node.value!r} is not allowed.", self.source, node)

    def v_List(self, node: ast.List) -> list[Any]:
        return [self._value(e) for e in node.elts]

    v_Tuple = v_List

    def v_Name(self, node: ast.Name) -> Any:
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise _fail(f"Unknown name '{node.id}'. Refer to columns with col(\"{node.id}\").",
                    self.source, node, fix=f'col("{node.id}")')

    def v_UnaryOp(self, node: ast.UnaryOp) -> Any:
        v = self.visit(node.operand)
        if isinstance(node.op, (ast.Invert, ast.Not)):
            return ~self._expr(v)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v
        raise _fail("Unsupported unary operator.", self.source, node)

    def v_BinOp(self, node: ast.BinOp) -> Expr:
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise _fail(f"Operator '{type(node.op).__name__}' is not supported.", self.source, node)
        left, right = self.visit(node.left), self.visit(node.right)
        if not isinstance(left, Expr):
            left = lit(left)
        return getattr(left, op)(right)

    def v_BoolOp(self, node: ast.BoolOp) -> Expr:
        # agents often write `and` / `or`; treat them as & / |
        vals = [self._expr(self.visit(v)) for v in node.values]
        out = vals[0]
        for v in vals[1:]:
            out = (out & v) if isinstance(node.op, ast.And) else (out | v)
        return out

    def v_Compare(self, node: ast.Compare) -> Expr:
        parts = [self.visit(node.left)] + [self.visit(c) for c in node.comparators]
        result = None
        for op, a, b in zip(node.ops, parts, parts[1:]):
            name = _CMPOPS.get(type(op))
            if name is None:
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(b, list):
                    e = self._expr(a).is_in(b)
                    e = ~e if isinstance(op, ast.NotIn) else e
                elif isinstance(op, (ast.Is, ast.IsNot)) and b is None:
                    e = self._expr(a).is_null() if isinstance(op, ast.Is) else self._expr(a).is_not_null()
                else:
                    raise _fail(f"Comparison '{type(op).__name__}' is not supported.", self.source, node)
            else:
                e = getattr(self._expr(a), name)(b)
            result = e if result is None else (result & e)
        return result  # type: ignore[return-value]

    def v_Call(self, node: ast.Call) -> Any:
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        if node.keywords:
            allowed = _KEYWORDS.get(fname or "", set())
            for k in node.keywords:
                if k.arg is None or (allowed is not None and k.arg not in allowed):
                    raise _fail(f"Keyword argument '{k.arg}' is not supported for {fname}().", self.source, node)
        args = [self._value(a) for a in node.args]
        kwargs = {k.arg: self._value(k.value) for k in node.keywords}
        if isinstance(f, ast.Name):
            if f.id in _FUNCS:
                return self._call(_FUNCS[f.id], args, kwargs, f.id, node)
            if f.id in _VALUE_FUNCS:
                try:
                    return _VALUE_FUNCS[f.id](*args, **kwargs)
                except (TypeError, ValueError, _decimal.InvalidOperation) as e:
                    raise _fail(f"Invalid {f.id}(): {e}.", self.source, node) from None
            raise _fail(f"Function '{f.id}' is not allowed. Allowed: {', '.join(sorted({**_FUNCS, **_VALUE_FUNCS}))}.",
                        self.source, node)
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Attribute) and f.value.attr in ("str", "dt"):
                ns = f.value.attr
                allowed_ns = _STR_METHODS if ns == "str" else _DT_METHODS
                if f.attr not in allowed_ns:
                    raise _fail(f"{ns}.{f.attr}() is not available. Allowed: {sorted(allowed_ns)}.", self.source, node)
                target = self._expr(self.visit(f.value.value))
                return self._call(getattr(getattr(target, ns), f.attr), args, kwargs, f"{ns}.{f.attr}", node)
            if f.attr not in _METHODS:
                raise _fail(f"Method .{f.attr}() is not allowed. Allowed: {sorted(_METHODS)}.", self.source, node)
            target = self.visit(f.value)
            if not isinstance(target, (Expr, _When)):
                target = lit(target)
            if isinstance(target, _When) and f.attr != "then":
                raise _fail("when(...) must be followed by .then(value).", self.source, node)
            if f.attr in ("when", "otherwise") and not isinstance(target, Case):
                raise _fail(f".{f.attr}() is only valid after when(...).then(...).", self.source, node)
            if not hasattr(target, f.attr):
                raise _fail(f".{f.attr}() is not valid here.", self.source, node)
            return self._call(getattr(target, f.attr), args, kwargs, f.attr, node)
        raise _fail("Only allowed functions and expression methods may be called.", self.source, node)

    def _call(self, fn, args, kwargs, name, node):
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            raise _fail(f"Wrong arguments for {name}(): {e}.", self.source, node) from None

    def v_Attribute(self, node: ast.Attribute) -> Any:
        raise _fail(f"Attribute access '.{node.attr}' is only allowed as a method call.", self.source, node)

    def _value(self, node: ast.AST) -> Any:
        return self.visit(node)

    def _expr(self, v: Any) -> Expr:
        return v if isinstance(v, Expr) else lit(v)
