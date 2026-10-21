"""The user-facing Frame: immutable, lazy, versioned, engine-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pyarrow as pa

from . import plan as P
from .backends import get_backend
from .errors import HistoryError, InvalidOperationError
from .expr import Col, Expr, col
from .types import DType


@dataclass(frozen=True)
class Step:
    id: str
    op: str
    args: str
    intent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "op": self.op, "args": self.args, "intent": self.intent}
        d.update(self.extra)
        return d


def _as_expr(x: Any) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, str):
        return col(x)
    raise InvalidOperationError(
        f"Expected a column name or expression, got {type(x).__name__}.",
        suggestions=[{"description": "Pass a column name string or an expression.", "fix": 'col("name")'}],
    )


def _as_names(items: Sequence[Any], where: str) -> list[str]:
    out = []
    for i in items:
        if isinstance(i, Col):
            out.append(i.name)
        elif isinstance(i, str):
            out.append(i)
        else:
            raise InvalidOperationError(f"{where} takes column names, got {i!r}.",
                                        suggestions=[{"description": "Create the column with with_columns() first."}])
    return out


def _opt_names(items: Sequence[Any] | None) -> list[str] | None:
    if items is None:
        return None
    return _as_names([items] if isinstance(items, str) else list(items), "inputs")


def _flatten(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for i in items:
        if isinstance(i, (list, tuple)):
            out.extend(i)
        else:
            out.append(i)
    return out


class Frame:
    """An immutable, lazy table. Every method returns a new Frame and records a
    step in its history; nothing runs until collect()/to_*()."""

    __slots__ = ("_node", "_parent", "_step")

    def __init__(self, node: P.Node, parent: "Frame | None" = None, step: Step | None = None) -> None:
        self._node = node
        self._parent = parent
        self._step = step or Step(id=node.fingerprint()[:8], op=node.op, args=node.params())

    # --- introspection ----------------------------------------------------
    @property
    def schema(self) -> dict[str, DType]:
        """Column name -> type of the frame's output. Known without executing anything."""
        return dict(self._node.schema)

    @property
    def columns(self) -> list[str]:
        """Output column names, in order."""
        return list(self._node.schema)

    @property
    def plan(self) -> P.Node:
        """The logical plan (IR) behind this frame."""
        return self._node

    def __repr__(self) -> str:
        cols = ", ".join(f"{n}: {t}" for n, t in list(self._node.schema.items())[:8])
        more = "" if len(self._node.schema) <= 8 else f", ... (+{len(self._node.schema) - 8})"
        return f"<pyursa.Frame [{cols}{more}] steps={len(self.history())} lazy>"

    # --- transformations --------------------------------------------------
    def _derive(self, node: P.Node, intent: str | None, **extra: Any) -> "Frame":
        step = Step(id=node.fingerprint()[:8], op=node.op, args=node.params(), intent=intent, extra=extra)
        return Frame(node, parent=self, step=step)

    def filter(self, *predicates: Expr, intent: str | None = None) -> "Frame":
        """Keep rows where every condition is true (null counts as false). Conditions are combined with &."""
        if not predicates:
            raise InvalidOperationError("filter() needs at least one condition.")
        pred = predicates[0]
        for p in predicates[1:]:
            pred = pred & p
        return self._derive(P.Filter(self._node, pred), intent)

    def select(self, *exprs: Any, intent: str | None = None, **named: Any) -> "Frame":
        """Choose or compute columns. Keyword arguments name new columns: select("id", total=col("a") + col("b")). Selecting only aggregations gives a one-row summary."""
        es = [_as_expr(e) for e in _flatten(exprs)] + [_as_expr(e).alias(n) for n, e in named.items()]
        if es and all(e.has_agg() for e in es):
            return self._derive(P.Aggregate(self._node, [], es), intent)
        return self._derive(P.Project(self._node, es), intent)

    def with_columns(self, *exprs: Expr, intent: str | None = None, **named: Expr) -> "Frame":
        """Add or replace columns, keeping the others: with_columns(revenue=col("qty") * col("price")). Window expressions are allowed."""
        es = [_as_expr(e) for e in _flatten(exprs)] + [_as_expr(e).alias(n) for n, e in named.items()]
        if not es:
            raise InvalidOperationError("with_columns() needs at least one expression.")
        return self._derive(P.WithColumns(self._node, es), intent)

    def group_by(self, *keys: Any) -> "GroupBy":
        """Start a grouped aggregation; finish it with .agg(...). Output is sorted by the keys."""
        return GroupBy(self, _as_names(_flatten(keys), "group_by()"))

    def agg(self, *aggs: Expr, intent: str | None = None, **named: Expr) -> "Frame":
        """Aggregate the whole frame into one row: agg(total=col("x").sum(), n=pu.len())."""
        es = [_as_expr(a) for a in _flatten(aggs)] + [_as_expr(e).alias(n) for n, e in named.items()]
        return self._derive(P.Aggregate(self._node, [], es), intent)

    def sort(self, *by: Any, descending: bool | Sequence[bool] = False, intent: str | None = None) -> "Frame":
        """Stable sort by one or more columns; nulls always last."""
        names = _as_names(_flatten(by), "sort()")
        desc = [descending] * len(names) if isinstance(descending, bool) else list(descending)
        return self._derive(P.Sort(self._node, names, desc), intent)

    def head(self, n: int = 5, intent: str | None = None) -> "Frame":
        """First n rows (in the current row order)."""
        return self._derive(P.Limit(self._node, n), intent)

    limit = head

    def join(self, other: "Frame", on: str | Sequence[str], how: str = "inner",
             suffix: str = "_right", intent: str | None = None) -> "Frame":
        """how: "inner", "left", "full", "semi" (left rows with a match) or "anti" (left rows without one)."""
        if not isinstance(other, Frame):
            raise InvalidOperationError("join() needs another Frame.")
        keys = [on] if isinstance(on, str) else list(on)
        return self._derive(P.Join(self._node, other._node, keys, how, suffix), intent,
                            right_step=other._step.id)

    def rename(self, mapping: dict[str, str], intent: str | None = None) -> "Frame":
        """Rename columns: rename({"old": "new"})."""
        return self._derive(P.Rename(self._node, dict(mapping)), intent)

    def drop(self, *columns: Any, intent: str | None = None) -> "Frame":
        """Remove columns."""
        return self._derive(P.Drop(self._node, _as_names(_flatten(columns), "drop()")), intent)

    def distinct(self, subset: str | Sequence[str] | None = None, intent: str | None = None) -> "Frame":
        """Remove duplicate rows (or rows duplicated on `subset`), keeping the first occurrence in order."""
        cols = None if subset is None else _as_names([subset] if isinstance(subset, str) else list(subset), "distinct()")
        return self._derive(P.Distinct(self._node, cols), intent)

    unique = distinct

    def concat(self, *others: "Frame", intent: str | None = None) -> "Frame":
        """Append the rows of other frames (same column names and types) after this one."""
        frames = [self, *_flatten(others)]
        for f in frames:
            if not isinstance(f, Frame):
                raise InvalidOperationError("concat() takes Frames.")
        return self._derive(P.Concat([f._node for f in frames]), intent, others=[f._step.id for f in frames[1:]])

    # --- writing ------------------------------------------------------------
    def write_parquet(self, path: str, backend: str | None = None, overwrite: bool = False) -> int:
        """Execute and write a Parquet file. Returns the number of rows written."""
        import pyarrow.parquet as pq
        table = self._for_write(path, overwrite, backend)
        pq.write_table(table, path)
        return table.num_rows

    def write_csv(self, path: str, backend: str | None = None, overwrite: bool = False) -> int:
        """Execute and write a CSV file (nested columns are not supported in CSV)."""
        import pyarrow.csv as pcsv
        if any(t.is_nested for t in self.schema.values()):
            raise InvalidOperationError("CSV cannot hold List/Struct columns.",
                                        suggestions=[{"description": "Drop them or use write_parquet()/write_json()."}])
        table = self._for_write(path, overwrite, backend)
        pcsv.write_csv(table, path)
        return table.num_rows

    def write_json(self, path: str, backend: str | None = None, overwrite: bool = False) -> int:
        """Execute and write newline-delimited JSON (one object per row)."""
        import json as _json
        table = self._for_write(path, overwrite, backend)
        with open(path, "w", encoding="utf-8") as fh:
            for row in table.to_pylist():
                fh.write(_json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")
        return table.num_rows

    def _for_write(self, path: str, overwrite: bool, backend: str | None) -> pa.Table:
        import os
        if os.path.exists(path) and not overwrite:
            raise InvalidOperationError(f"'{path}' already exists.", code="FILE_EXISTS",
                                        suggestions=[{"description": "Pass overwrite=True or choose another path."}])
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent):
            raise InvalidOperationError(f"Directory '{parent}' does not exist.", code="DIRECTORY_NOT_FOUND")
        return self.collect(backend)

    # --- semantic (LLM) operators ----------------------------------------
    def llm_filter(self, condition: str, *, inputs: Sequence[str] | None = None, client: Any = None,
                   max_rows: int | None = None, batch_size: int | None = None, intent: str | None = None,
                   redact: bool | None = None, config: Any = None) -> "Frame":
        """Keep rows the model judges to satisfy `condition`, e.g. "the review complains about delivery"."""
        from .semantic import SemanticOp
        return self._derive(SemanticOp(self._node, "filter", condition, _opt_names(inputs), {}, client=client,
                                       max_rows=max_rows, batch_size=batch_size, config=config, redact=redact), intent)

    def llm_map(self, instruction: str, *, output: str, dtype: DType | str = "String",
                inputs: Sequence[str] | None = None, client: Any = None, max_rows: int | None = None,
                batch_size: int | None = None, intent: str | None = None, redact: bool | None = None,
                config: Any = None) -> "Frame":
        """Add one typed column computed by the model, e.g. a sentiment label or a 1-5 score."""
        from .semantic import SemanticOp
        return self._derive(SemanticOp(self._node, "map", instruction, _opt_names(inputs), {output: DType(dtype)},
                                       client=client, max_rows=max_rows, batch_size=batch_size, config=config,
                                       redact=redact), intent)

    def llm_extract(self, instruction: str, *, fields: dict[str, DType | str], inputs: Sequence[str] | None = None,
                    client: Any = None, max_rows: int | None = None, batch_size: int | None = None,
                    intent: str | None = None, redact: bool | None = None, config: Any = None) -> "Frame":
        """Add several typed columns extracted from text, e.g. fields={"company": "String", "amount": "Float64"}."""
        from .semantic import SemanticOp
        if not fields:
            raise InvalidOperationError("llm_extract() needs at least one field.")
        return self._derive(SemanticOp(self._node, "extract", instruction, _opt_names(inputs),
                                       {n: DType(t) for n, t in fields.items()}, client=client,
                                       max_rows=max_rows, batch_size=batch_size, config=config, redact=redact), intent)

    # --- versioning -------------------------------------------------------
    def _chain(self) -> list["Frame"]:
        chain, f = [], self
        while f is not None:
            chain.append(f)
            f = f._parent
        return chain[::-1]

    def history(self) -> list[dict[str, Any]]:
        """Every step from the source to this frame, oldest first."""
        return [f._step.to_dict() for f in self._chain()]

    def undo(self, steps: int = 1) -> "Frame":
        """Return the frame as it was `steps` transformations ago. Instant: no data is reloaded."""
        chain = self._chain()
        if not isinstance(steps, int) or steps < 0:
            raise HistoryError("undo() needs a non-negative integer.")
        if steps >= len(chain):
            raise HistoryError(
                f"Cannot undo {steps} step(s); this frame has only {len(chain) - 1} after its source.",
                details={"history": [s["id"] for s in self.history()]},
                suggestions=[{"description": "Use frame.history() to see available steps, then checkout(id)."}],
            )
        return chain[-1 - steps]

    def checkout(self, step_id: str) -> "Frame":
        """Return the frame at a given step id from history()."""
        if not isinstance(step_id, str) or len(step_id) < 4:
            raise HistoryError("checkout() needs a step id (at least 4 characters) from history().")
        matches = [f for f in self._chain() if f._step.id == step_id or f._step.id.startswith(step_id)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise HistoryError(f"Step id '{step_id}' is ambiguous; use more characters.")
        raise HistoryError(f"No step '{step_id}' in this frame's history.",
                           details={"available": [s["id"] for s in self.history()]})

    # --- provenance & planning --------------------------------------------
    def lineage(self, column: str | None = None) -> dict[str, Any]:
        """Column-level provenance: for each column, the source columns, filters, joins and groupings behind it."""
        lin = P.lineage(self._node)
        if column is None:
            return lin
        if column not in lin:
            from .errors import column_not_found
            raise column_not_found(column, lin, "this frame")
        return lin[column]

    def explain(self, backend: str | None = None) -> dict[str, Any]:
        """Dry run: what would execute, the output schema and a size estimate. Runs nothing."""
        from .semantic import CONFIG, has_semantic, placeholder
        b = get_backend(backend)
        rows = self._node.rows_bound()
        widths = {"Int64": 8, "Float64": 8, "Boolean": 1, "Date": 4, "Datetime": 8, "String": 32, "Null": 0,
                  "Decimal": 16, "List": 64, "Struct": 64}
        width = sum(widths[t.kind] for t in self._node.schema.values())
        return {
            "backend": b.name,
            "output_schema": {n: str(t) for n, t in self._node.schema.items()},
            "estimated_rows_max": rows,
            "estimated_bytes_max": None if rows is None else rows * width,
            "sources": [s.describe() for s in P.sources(self._node)],
            "logical_plan": self._node.tree(),
            "compiled": b.compile(placeholder(self._node)),
            **({"llm_steps": _llm_estimates(self._node, CONFIG)} if has_semantic(self._node) else {}),
        }

    # --- execution --------------------------------------------------------
    def collect(self, backend: str | None = None) -> pa.Table:
        """Execute on a backend (default: pu.default_backend()) and return Arrow."""
        from .semantic import execute
        return execute(self._node, get_backend(backend))

    def to_arrow(self, backend: str | None = None) -> pa.Table:
        """Same as collect()."""
        return self.collect(backend)

    def to_polars(self, backend: str | None = None):
        """Execute and return a polars.DataFrame."""
        import polars as pl
        return pl.from_arrow(self.collect(backend))

    def to_pandas(self, backend: str | None = None):
        """Execute and return a pandas.DataFrame."""
        return self.collect(backend).to_pandas()

    def to_dicts(self, backend: str | None = None) -> list[dict[str, Any]]:
        """Execute and return a list of row dictionaries."""
        return self.collect(backend).to_pylist()

    def count(self, backend: str | None = None) -> int:
        """Execute and return the number of rows."""
        from .expr import len_
        return self.agg(len_().alias("n")).collect(backend).column(0)[0].as_py()

    # --- LLM-facing summaries ---------------------------------------------
    def profile(self, backend: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Per-column statistics as a dict (row count, nulls, distinct, min/max/mean, top values, sample rows), computed by the engine."""
        from .summarize import profile
        return profile(self, backend=backend, **kwargs)

    def describe_for_llm(self, tokens: int = 800, backend: str | None = None, **kwargs: Any) -> str:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 20:
            raise InvalidOperationError("describe_for_llm() needs tokens >= 20.")
        """Markdown summary that fits within `tokens`, for putting a dataset in an LLM's context. Pass redact=True to mask PII."""
        from .summarize import describe_for_llm
        return describe_for_llm(self, tokens=tokens, backend=backend, **kwargs)


def _llm_estimates(node: P.Node, config: Any) -> list[dict[str, Any]]:
    out = []
    for c in node.children():
        out += _llm_estimates(c, config)
    if getattr(node, "semantic", False):
        rows = node.child.rows_bound()
        size = node.batch_size or config.batch_size
        out.append({
            "op": node.op, "instruction": node.instruction,
            "max_rows_to_model": rows, "row_budget": node.max_rows or config.max_rows,
            "max_llm_calls": None if rows is None else -(-rows // size),
            "note": "Upper bound before dedup and cache hits; nothing has been sent.",
        })
    return out


class GroupBy:
    def __init__(self, frame: Frame, keys: list[str]) -> None:
        self._frame, self._keys = frame, keys

    def agg(self, *aggs: Expr, intent: str | None = None, **named: Expr) -> Frame:
        """Aggregations per group: agg(total=col("x").sum(), n=pu.len()). Output is sorted by the group keys, nulls last."""
        es = [_as_expr(a) for a in _flatten(aggs)] + [_as_expr(e).alias(n) for n, e in named.items()]
        return self._frame._derive(P.Aggregate(self._frame._node, self._keys, es), intent)

    def __repr__(self) -> str:
        return f"<GroupBy keys={self._keys}; call .agg(...)>"


# --- constructors -----------------------------------------------------------

def from_arrow(table: pa.Table, name: str = "data") -> Frame:
    return Frame(P.Source(table=table, name=name))


def from_dict(data: dict[str, list[Any]], name: str = "data") -> Frame:
    return from_arrow(pa.table(data), name)


def from_pandas(df, name: str = "data") -> Frame:
    return from_arrow(pa.Table.from_pandas(df, preserve_index=False), name)


def from_polars(df, name: str = "data") -> Frame:
    return from_arrow(df.to_arrow(), name)


def read_parquet(path: str | Sequence[str], name: str | None = None) -> Frame:
    """Lazily scan Parquet: a file, a list of files, a glob ("data/*.parquet") or a directory."""
    import os
    first = (path if isinstance(path, str) else str(list(path)[0])).rstrip("/\\")
    stem = os.path.splitext(os.path.basename(first))[0]
    if any(ch in stem for ch in "*?[") or not stem:  # a glob: name it after its folder
        stem = os.path.basename(os.path.dirname(first)) or "data"
    default = stem
    return Frame(P.Source(path=path if isinstance(path, str) else [str(p) for p in path], fmt="parquet",
                          name=name or default))


def read_json(path: str, name: str | None = None) -> Frame:
    """Read newline-delimited JSON (one object per line) with engine-independent type inference."""
    import os

    import pyarrow.json as pjson
    return from_arrow(pjson.read_json(path), name or os.path.splitext(os.path.basename(path))[0])


def concat(frames: Sequence[Frame], intent: str | None = None) -> Frame:
    """Stack frames vertically (same column names and types)."""
    frames = list(frames)
    if len(frames) < 2:
        raise InvalidOperationError("concat() needs at least two frames.")
    return frames[0].concat(*frames[1:], intent=intent)


def _json_default(v: Any) -> Any:
    import datetime as _dt
    import decimal as _decimal
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    if isinstance(v, _decimal.Decimal):
        return str(v)
    return str(v)


def read_csv(path: str, name: str | None = None, *, empty_as_null: bool = True,
             schema: dict[str, DType | str] | None = None, delimiter: str = ",") -> Frame:
    """Read a CSV with a single, engine-independent type inference (PyArrow).

    Empty fields become null (like pandas); pass empty_as_null=False to keep them as "".
    Pass schema={"col": "Decimal(12,2)", ...} to fix column types instead of inferring them."""
    import os

    import pyarrow.csv as pcsv
    opts = pcsv.ConvertOptions(
        strings_can_be_null=empty_as_null,
        column_types={n: DType(t).to_arrow() for n, t in (schema or {}).items()} or None,
    )
    parse = pcsv.ParseOptions(delimiter=delimiter)
    return from_arrow(pcsv.read_csv(path, parse_options=parse, convert_options=opts),
                      name or os.path.splitext(os.path.basename(path))[0])
