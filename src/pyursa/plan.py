"""Logical plan nodes. Each node computes and validates its output schema
when it is constructed, so mistakes surface immediately, before execution."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .errors import (
    DuplicateColumnError,
    InvalidOperationError,
    TypeMismatchError,
    column_not_found,
)
from .expr import Col, Expr, unalias
from .types import DType, Schema, normalize_table, schema_from_arrow, validate_names


class Node:
    op = "node"

    def __init__(self) -> None:
        self.schema: Schema = self._compute_schema()
        validate_names(self.schema, f"The output of {self.op}()")

    def _compute_schema(self) -> Schema: raise NotImplementedError
    def children(self) -> list["Node"]: return []
    def params(self) -> str: return ""
    def rows_bound(self) -> int | None: raise NotImplementedError

    def describe(self) -> str:
        p = self.params()
        return f"{self.op}({p})" if p else self.op

    def fingerprint(self) -> str:
        h = hashlib.sha1(self.describe().encode())
        for c in self.children():
            h.update(c.fingerprint().encode())
        return h.hexdigest()[:10]

    def tree(self, indent: int = 0) -> str:
        line = "  " * indent + self.describe()
        return "\n".join([line] + [c.tree(indent + 1) for c in self.children()])

    def _require(self, names, where: str) -> None:
        for n in names:
            if n not in self.children()[0].schema:
                raise column_not_found(n, self.children()[0].schema, where)


class Source(Node):
    op = "source"

    def __init__(self, *, table: pa.Table | None = None, path: str | list[str] | None = None,
                 fmt: str | None = None, name: str | None = None) -> None:
        if (table is None) == (path is None):
            raise InvalidOperationError("Source needs exactly one of table or path.")
        self.table = normalize_table(table) if table is not None else None
        self.fmt = fmt
        self.name = name or "data"
        self.uid = uuid.uuid4().hex[:8]
        self.paths: list[str] = []
        if path is not None:
            self.paths = _expand_paths(path)
            self._rows = sum(pq.read_metadata(p).num_rows for p in self.paths)
            self._arrow_schema = pq.read_schema(self.paths[0])
            for p in self.paths[1:]:
                other = pq.read_schema(p)
                if not other.equals(self._arrow_schema, check_metadata=False):
                    raise TypeMismatchError(
                        f"Parquet files have different schemas: {self.paths[0]} vs {p}.",
                        details={"first": str(self._arrow_schema), "other": str(other), "file": p},
                        suggestions=[{"description": "Load the files separately, align types with cast(), then concat()."}],
                    )
        else:
            self._rows = self.table.num_rows  # type: ignore[union-attr]
            self._arrow_schema = self.table.schema  # type: ignore[union-attr]
        super().__init__()

    @property
    def path(self) -> str | None:
        return self.paths[0] if len(self.paths) == 1 else (None if not self.paths else self.paths[0])

    def _compute_schema(self): return schema_from_arrow(self._arrow_schema)
    def params(self):
        if self.paths:
            where = f"path={self.paths[0]!r}" if len(self.paths) == 1 else f"files={len(self.paths)}"
        else:
            where = "in-memory"
        return f"{self.name}, {where}, rows={self._rows}, id={self.uid}"
    def rows_bound(self): return self._rows


def _expand_paths(path: str | list[str]) -> list[str]:
    """A string may be a file, a directory or a glob; a list is taken literally (no glob expansion),
    so resolved paths passed by the sandbox can never match other files."""
    import glob
    import os
    if not isinstance(path, (str, os.PathLike)):
        out = [str(p) for p in path]
        for p in out:
            if not os.path.isfile(p):
                raise InvalidOperationError(f"File '{p}' does not exist.", code="FILE_NOT_FOUND")
        if not out:
            raise InvalidOperationError("No files given.", code="FILE_NOT_FOUND")
        return out
    item = str(path)
    if os.path.isfile(item):
        return [item]
    if os.path.isdir(item):
        matches = sorted(glob.glob(os.path.join(glob.escape(item), "**", "*.parquet"), recursive=True))
        if not matches:
            raise InvalidOperationError(f"No .parquet files in directory '{item}'.", code="FILE_NOT_FOUND")
        return matches
    if any(ch in item for ch in "*?["):
        matches = sorted(glob.glob(item, recursive=True))
        if not matches:
            raise InvalidOperationError(f"No files match '{item}'.", code="FILE_NOT_FOUND")
        return matches
    raise InvalidOperationError(f"File '{item}' does not exist.", code="FILE_NOT_FOUND")


class Filter(Node):
    op = "filter"

    def __init__(self, child: Node, predicate: Expr) -> None:
        self.child, self.predicate = child, predicate
        super().__init__()

    def _compute_schema(self):
        if self.predicate.has_agg():
            raise InvalidOperationError(
                "filter() cannot contain an aggregation.",
                details={"predicate": repr(self.predicate)},
                suggestions=[{"description": "Aggregate first with group_by().agg(), then filter the result."}],
            )
        if self.predicate.has_window():
            raise InvalidOperationError(
                "filter() cannot contain a window expression.",
                details={"predicate": repr(self.predicate)},
                suggestions=[{"description": "Compute the window value with with_columns(), then filter on the new column: "
                                             "frame.with_columns(w=<window expression>).filter(<condition on col(\"w\")>)."}],
            )
        t = self.predicate.dtype(self.child.schema)
        if t != DType.BOOLEAN:
            raise TypeMismatchError(
                f"filter() needs a Boolean condition, got {t}.",
                details={"predicate": repr(self.predicate), "type": str(t)},
                suggestions=[{"description": "Compare the value to something.", "fix": f"{self.predicate!r} > 0"}],
            )
        return dict(self.child.schema)

    def children(self): return [self.child]
    def params(self): return repr(self.predicate)
    def rows_bound(self): return self.child.rows_bound()


def _named(exprs: list[Expr], schema: Schema, where: str) -> list[tuple[str, Expr, DType]]:
    out, seen = [], {}
    for e in exprs:
        name = e.output_name()
        if name in seen:
            raise DuplicateColumnError(
                f"Two expressions in {where} produce the column '{name}'.",
                details={"column": name, "expressions": [repr(seen[name]), repr(e)]},
                suggestions=[{"description": "Give one of them a distinct name with .alias().",
                              "fix": f'{e!r}.alias("{name}_2")'}],
            )
        seen[name] = e
        out.append((name, e, e.dtype(schema)))
    return out


class Project(Node):
    op = "select"

    def __init__(self, child: Node, exprs: list[Expr]) -> None:
        self.child, self.exprs = child, exprs
        super().__init__()

    def _compute_schema(self):
        if not self.exprs:
            raise InvalidOperationError("select() needs at least one column.")
        for e in self.exprs:
            if e.has_agg():
                raise InvalidOperationError(
                    "select() mixes aggregations with row-level columns.",
                    details={"expression": repr(e)},
                    suggestions=[{"description": "Group first, or select only aggregations for a one-row summary.",
                                  "fix": f'frame.group_by("<key>").agg({e!r})'}],
                )
        self.named = _named(self.exprs, self.child.schema, "select()")
        self.needs_row_index = any(e.needs_row_index() for e in self.exprs)
        return {n: t for n, _, t in self.named}

    def children(self): return [self.child]
    def params(self): return ", ".join(repr(e) for e in self.exprs)
    def rows_bound(self): return self.child.rows_bound()


class WithColumns(Node):
    op = "with_columns"

    def __init__(self, child: Node, exprs: list[Expr]) -> None:
        self.child, self.exprs = child, exprs
        super().__init__()

    def _compute_schema(self):
        for e in self.exprs:
            if e.has_agg():
                raise InvalidOperationError(
                    "with_columns() cannot contain a bare aggregation.",
                    details={"expression": repr(e)},
                    suggestions=[{"description": "Use a window to keep one value per row.",
                                  "fix": f'{e!r}.over("<group column>")'}],
                )
        self.named = _named(self.exprs, self.child.schema, "with_columns()")
        self.needs_row_index = any(e.needs_row_index() for e in self.exprs)
        schema = dict(self.child.schema)
        for n, _, t in self.named:
            schema[n] = t
        return schema

    def children(self): return [self.child]
    def params(self): return ", ".join(repr(e) for e in self.exprs)
    def rows_bound(self): return self.child.rows_bound()


class Aggregate(Node):
    """group_by(keys).agg(aggs). With no keys it is a one-row global aggregation.
    Contract: output is sorted by keys ascending, nulls last."""

    op = "aggregate"

    def __init__(self, child: Node, keys: list[str], aggs: list[Expr]) -> None:
        self.child, self.keys, self.aggs = child, keys, aggs
        super().__init__()

    def _compute_schema(self):
        self._require(self.keys, "group_by()")
        if len(set(self.keys)) != len(self.keys):
            raise DuplicateColumnError("group_by() keys contain duplicates.", details={"keys": self.keys})
        if not self.aggs:
            raise InvalidOperationError("agg() needs at least one aggregation.",
                                        suggestions=[{"description": "Count rows per group.", "fix": "pu.len()"}])
        for k in self.keys:
            if not self.child.schema[k].is_hashable:
                raise TypeMismatchError(f"Cannot group by {self.child.schema[k]} column '{k}'.")
        for e in self.aggs:
            if e.has_window():
                raise InvalidOperationError("agg() cannot contain a window expression.",
                                            details={"expression": repr(e)})
            outside = unalias(e).cols_outside_agg()
            if not e.has_agg() or outside:
                bad = sorted(outside) or [repr(e)]
                inner = unalias(e)
                fix = (f'col("{bad[0]}").sum()' if isinstance(inner, Col)
                       else f"{inner!r}.sum()") if outside else "len()"
                raise InvalidOperationError(
                    f"agg() expression {e!r} uses {bad} outside an aggregation.",
                    details={"expression": repr(e)},
                    suggestions=[{"description": "Aggregate the whole expression, e.g. with .sum(), .mean() or .max().",
                                  "fix": fix}],
                )
        named = _named(self.aggs, self.child.schema, "agg()")
        for n, _, _ in named:
            if n in self.keys:
                expr = next(e for name, e, _ in named if name == n)
                raise DuplicateColumnError(
                    f"Aggregation output '{n}' collides with a group key.",
                    suggestions=[{"description": "Alias the aggregation.", "fix": f'{unalias(expr)!r}.alias("{n}_agg")'}],
                )
        self.named = named
        schema = {k: self.child.schema[k] for k in self.keys}
        schema.update({n: t for n, _, t in named})
        return schema

    def children(self): return [self.child]
    def params(self): return f"keys={self.keys}, aggs=[{', '.join(repr(a) for a in self.aggs)}]"
    def rows_bound(self): return 1 if not self.keys else self.child.rows_bound()


class Sort(Node):
    """Contract: stable sort, nulls last regardless of direction."""

    op = "sort"

    def __init__(self, child: Node, by: list[str], descending: list[bool]) -> None:
        self.child, self.by, self.descending = child, by, descending
        super().__init__()

    def _compute_schema(self):
        if not self.by:
            raise InvalidOperationError("sort() needs at least one column.")
        self._require(self.by, "sort()")
        if len(self.descending) != len(self.by):
            raise InvalidOperationError("descending must be a bool or one bool per sort column.")
        for c in self.by:
            if not self.child.schema[c].is_orderable:
                raise TypeMismatchError(f"Cannot sort by {self.child.schema[c]} column '{c}'.")
        return dict(self.child.schema)

    def children(self): return [self.child]
    def params(self): return f"by={self.by}, descending={self.descending}"
    def rows_bound(self): return self.child.rows_bound()


class Limit(Node):
    op = "limit"

    def __init__(self, child: Node, n: int) -> None:
        self.child, self.n = child, n
        super().__init__()

    def _compute_schema(self):
        if not isinstance(self.n, int) or isinstance(self.n, bool) or not 0 <= self.n < 2 ** 63:
            raise InvalidOperationError(f"limit/head needs a non-negative integer, got {self.n!r}.")
        return dict(self.child.schema)

    def children(self): return [self.child]
    def params(self): return str(self.n)
    def rows_bound(self):
        b = self.child.rows_bound()
        return self.n if b is None else min(b, self.n)


class Join(Node):
    """Contract: output keeps left row order, then right row order within matches;
    for full joins, unmatched right rows follow in right order. Null keys never match.
    Right-side non-key columns that collide get `suffix`. semi/anti return left columns only."""

    op = "join"
    HOWS = ("inner", "left", "full", "semi", "anti")

    def __init__(self, left: Node, right: Node, on: list[str], how: str, suffix: str) -> None:
        self.left, self.right, self.on, self.how, self.suffix = left, right, on, how, suffix
        super().__init__()

    def _compute_schema(self):
        if self.how not in self.HOWS:
            raise InvalidOperationError(f"Join type '{self.how}' is not supported.",
                                        suggestions=[{"description": f"Use one of: {', '.join(self.HOWS)}."}])
        if not self.on:
            raise InvalidOperationError("join() needs at least one key in `on`.")
        if len(set(self.on)) != len(self.on):
            raise DuplicateColumnError("join() keys contain duplicates.")
        for k in self.on:
            if k not in self.left.schema:
                raise column_not_found(k, self.left.schema, "the left side of join()")
            if k not in self.right.schema:
                raise column_not_found(k, self.right.schema, "the right side of join()")
            lt, rt = self.left.schema[k], self.right.schema[k]
            if lt != rt:
                raise TypeMismatchError(
                    f"Join key '{k}' is {lt} on the left but {rt} on the right.",
                    details={"key": k, "left_type": str(lt), "right_type": str(rt)},
                    suggestions=[{"description": "Cast one side so both keys have the same type.",
                                  "fix": f'right.with_columns(col("{k}").cast("{lt}"))'}],
                )
            if not lt.is_hashable:
                raise TypeMismatchError(f"Cannot join on {lt} column '{k}'.")
        schema = dict(self.left.schema)
        self.right_out: list[tuple[str, str]] = []  # (source name, output name)
        if self.how in ("semi", "anti"):
            return schema
        for n, t in self.right.schema.items():
            if n in self.on:
                continue
            out = n + self.suffix if n in schema else n
            if out in schema:
                raise DuplicateColumnError(f"Join output column '{out}' would be duplicated.",
                                           suggestions=[{"description": "Pick a different suffix."}])
            schema[out] = t
            self.right_out.append((n, out))
        return schema

    def children(self): return [self.left, self.right]
    def params(self): return f"on={self.on}, how={self.how!r}"
    def rows_bound(self):
        if self.how in ("semi", "anti"):
            return self.left.rows_bound()
        return None  # can grow up to left x right


class Distinct(Node):
    """Contract: keeps the first occurrence of each distinct row (or subset), in input order."""

    op = "distinct"

    def __init__(self, child: Node, subset: list[str] | None) -> None:
        self.child, self.subset = child, subset
        super().__init__()

    def _compute_schema(self):
        cols = self.subset if self.subset is not None else list(self.child.schema)
        if not cols:
            raise InvalidOperationError("distinct() needs at least one column.")
        self._require(cols, "distinct()")
        for c in cols:
            if not self.child.schema[c].is_hashable:
                raise TypeMismatchError(f"distinct() cannot compare {self.child.schema[c]} column '{c}'.",
                                        suggestions=[{"description": "Pass subset= with non-nested columns."}])
        self.keys = cols
        return dict(self.child.schema)

    def children(self): return [self.child]
    def params(self): return "" if self.subset is None else f"subset={self.subset}"
    def rows_bound(self): return self.child.rows_bound()


class Concat(Node):
    """Vertical union: rows of each input in order. Columns are matched by name and must have equal types."""

    op = "concat"

    def __init__(self, inputs: list[Node]) -> None:
        self.inputs = inputs
        super().__init__()

    def _compute_schema(self):
        if len(self.inputs) < 2:
            raise InvalidOperationError("concat() needs at least two frames.")
        first = self.inputs[0].schema
        for i, other in enumerate(self.inputs[1:], start=1):
            if set(other.schema) != set(first):
                missing = sorted(set(first) - set(other.schema))
                extra = sorted(set(other.schema) - set(first))
                raise TypeMismatchError(
                    f"concat() frame {i} has different columns.",
                    details={"missing": missing, "extra": extra},
                    suggestions=[{"description": "select()/rename() so every frame has the same column names."}],
                )
            for n, t in first.items():
                if other.schema[n] != t:
                    raise TypeMismatchError(
                        f"concat() column '{n}' is {t} in frame 0 but {other.schema[n]} in frame {i}.",
                        suggestions=[{"description": "Cast so the types match.", "fix": f'col("{n}").cast("{t}")'}],
                    )
        return dict(first)

    def children(self): return list(self.inputs)
    def params(self): return f"{len(self.inputs)} frames"
    def rows_bound(self):
        bounds = [c.rows_bound() for c in self.inputs]
        return None if any(b is None for b in bounds) else sum(bounds)  # type: ignore[misc]


class Rename(Node):
    op = "rename"

    def __init__(self, child: Node, mapping: dict[str, str]) -> None:
        self.child, self.mapping = child, mapping
        super().__init__()

    def _compute_schema(self):
        self._require(self.mapping, "rename()")
        for old, new in self.mapping.items():
            if not isinstance(new, str) or not new:
                raise InvalidOperationError(f"rename() needs string names; '{old}' -> {new!r} is not valid.")
        out: Schema = {}
        for n, t in self.child.schema.items():
            new = self.mapping.get(n, n)
            if new in out:
                raise DuplicateColumnError(f"rename() would create two columns named '{new}'.")
            out[new] = t
        return out

    def children(self): return [self.child]
    def params(self): return repr(self.mapping)
    def rows_bound(self): return self.child.rows_bound()


class Drop(Node):
    op = "drop"

    def __init__(self, child: Node, columns: list[str]) -> None:
        self.child, self.columns = child, columns
        super().__init__()

    def _compute_schema(self):
        self._require(self.columns, "drop()")
        out = {n: t for n, t in self.child.schema.items() if n not in self.columns}
        if not out:
            raise InvalidOperationError("drop() would remove every column.")
        return out

    def children(self): return [self.child]
    def params(self): return repr(self.columns)
    def rows_bound(self): return self.child.rows_bound()


def sources(node: Node) -> list[Source]:
    if isinstance(node, Source):
        return [node]
    out: list[Source] = []
    for c in node.children():
        for s in sources(c):
            if s not in out:
                out.append(s)
    return out


# ---------------------------------------------------------------------------
# Column-level lineage, computed statically from the plan.
# ---------------------------------------------------------------------------

def lineage(node: Node) -> dict[str, dict[str, Any]]:
    """For every output column: the expression that produced it, the source
    columns it derives from, and the filters/joins/groupings applied upstream."""

    def entry(expr, srcs, filters, ops):
        return {"expr": expr, "sources": sorted(srcs), "filters": list(filters), "operations": list(ops)}

    if isinstance(node, Source):
        return {c: entry(f"{node.name}.{c}", {f"{node.name}.{c}"}, [], []) for c in node.schema}

    if getattr(node, "semantic", False):
        lin = lineage(node.child)
        tag = f"{node.op}({node.instruction!r})"
        if node.kind == "filter":
            for v in lin.values():
                v["filters"].append(tag)
            return lin
        srcs: set[str] = set()
        filters: list[str] = []
        for i in node.inputs:
            srcs |= set(lin[i]["sources"])
            filters += [f for f in lin[i]["filters"] if f not in filters]
        for name in node.outputs:
            lin[name] = entry(f"LLM: {node.instruction}", srcs, filters, [tag])
        return lin

    if isinstance(node, Filter):
        lin = lineage(node.child)
        pred = repr(node.predicate)
        for v in lin.values():
            v["filters"].append(pred)
        return lin

    def derived(child_lin, e, op):
        inputs = sorted(e.columns())
        srcs: set[str] = set()
        filters: list[str] = []
        ops: list[str] = []
        for i in inputs:
            srcs |= set(child_lin[i]["sources"])
            filters += [f for f in child_lin[i]["filters"] if f not in filters]
            ops += [o for o in child_lin[i]["operations"] if o not in ops]
        is_passthrough = isinstance(e, Col)
        expr = child_lin[e.name]["expr"] if is_passthrough else repr(unalias(e))
        return entry(expr, srcs, filters, ops if is_passthrough else ops + [op])

    if isinstance(node, (Project, WithColumns)):
        child_lin = lineage(node.child)
        out = dict(child_lin) if isinstance(node, WithColumns) else {}
        for name, e, _ in node.named:
            out[name] = derived(child_lin, e, node.op)
        return {c: out[c] for c in node.schema}

    if isinstance(node, Aggregate):
        child_lin = lineage(node.child)
        out = {k: child_lin[k] for k in node.keys}
        for name, e, _ in node.named:
            d = derived(child_lin, e, f"group_by({node.keys})" if node.keys else "global aggregate")
            out[name] = d
        return out

    if isinstance(node, Join):
        ll, rl = lineage(node.left), lineage(node.right)
        tag = f"join(on={node.on}, how={node.how!r})"
        out = {}
        if node.how in ("semi", "anti"):
            for c in node.left.schema:
                out[c] = dict(ll[c], filters=ll[c]["filters"] + [f"{node.how} join on {node.on}"])
            return out
        for c in node.left.schema:
            out[c] = dict(ll[c], operations=ll[c]["operations"] + [tag])
            if c in node.on:
                out[c]["sources"] = sorted(set(ll[c]["sources"]) | set(rl[c]["sources"]))
        for src, dst in node.right_out:
            out[dst] = dict(rl[src], operations=rl[src]["operations"] + [tag])
        return out

    if isinstance(node, Distinct):
        lin = lineage(node.child)
        for v in lin.values():
            v["operations"] = v["operations"] + [f"distinct({node.keys})"]
        return lin

    if isinstance(node, Concat):
        lins = [lineage(c) for c in node.inputs]
        out = {}
        for c in node.schema:
            srcs: set[str] = set()
            filters: list[str] = []
            for lin in lins:
                srcs |= set(lin[c]["sources"])
                filters += [f for f in lin[c]["filters"] if f not in filters]
            out[c] = entry(f"concat of {len(lins)} frames", srcs, filters, ["concat"])
        return out

    if isinstance(node, Rename):
        child_lin = lineage(node.child)
        return {node.mapping.get(c, c): v for c, v in child_lin.items()}

    if isinstance(node, Drop):
        child_lin = lineage(node.child)
        return {c: child_lin[c] for c in node.schema}

    # Sort / Limit pass lineage through unchanged.
    return lineage(node.children()[0])
