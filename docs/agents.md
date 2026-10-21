# Agents and MCP

## What makes it agent-friendly

- **Errors before execution.** Plans are type-checked as they are built. Errors carry a
  stable `code`, `details` and `suggestions` with paste-ready `fix` snippets:
  ```python
  try:
      frame.filter(col("quantity") > "5")
  except pu.PyursaError as e:
      e.to_dict()
  # {"error": "COLUMN_NOT_FOUND", "message": "...", "details": {"available_columns": [...]},
  #  "suggestions": [{"description": "Did you mean 'qty'?", "fix": "col(\"qty\")"}]}
  ```
- **Context-sized summaries.** `describe_for_llm(tokens=N)` always fits the budget,
  is identical on every backend, and flags instruction-like text found anywhere in a column.
- **Free exploration.** Frames are immutable; `history()`, `undo(n)` and `checkout(id)`
  cost nothing. Every step accepts `intent="why"` for the audit trail.
- **Dry runs.** `explain()` returns the output schema, row/byte upper bounds, LLM call
  estimates and the compiled engine query without executing anything.
- **Provenance.** `lineage(col)` shows the source columns, filters, joins and groupings
  behind any column.

## Tool layer: `ToolSession`

`pyursa.tools.ToolSession` exposes the library as JSON-in/JSON-out functions for any
function-calling framework (LangGraph, CrewAI, Pydantic AI, plain tool use):

```python
from pyursa.tools import ToolSession
s = ToolSession(roots=["./data"], backend="duckdb", redact_pii=False)
s.load_file("orders.parquet")          # {"ok": true, "handle": "f1", "schema": {...}, "rows": 200000}
s.transform("f1", [
    {"op": "filter", "condition": 'col("qty") > 0 and col("status") != "cancelled"'},
    {"op": "group_by", "keys": ["region"], "aggs": {"revenue": 'col("qty") * col("price")'}},
])                                      # structured error: the agg isn't an aggregation, with a fix
```

Design rules:

- Agents pass **handles**, never data.
- Every call returns `{"ok": true, ...}` or `{"ok": false, "error": {...}}`; nothing raises.
  Transform errors include `step_index`.
- Expressions are parsed by a whitelisted parser; no code is executed.
- Files are confined to `roots` (symlinks and glob matches are checked individually).
- `preview` caps at 50 rows and returns exact numbers plus JSON rows.

Call `s.help()` for the full expression syntax and step formats.

## MCP server

```bash
pyursa-mcp --root ./data                                   # stdio, one client
pyursa-mcp --root ./data --transport streamable-http --port 8000   # many clients
pyursa-mcp --root ./data --llm gemini:gemini-3.8-flash --llm-price 0.10,0.50 --redact-pii
```

Client configuration (Claude Desktop, Claude Code, or any MCP client):

```json
{"mcpServers": {"pyursa": {"command": "pyursa-mcp", "args": ["--root", "/path/to/data"]}}}
```

Tools: `load_file`, `describe`, `transform`, `preview`, `count_rows`, `explain`, `history`,
`undo`, `lineage`, `export`, `drop_frame`, `set_backend`, `list_frames`, `llm_usage`,
`pyursa_help`.

**Isolation.** Each connected client gets its own session: frames, LLM cache and usage
reports are never shared (verified by a two-client HTTP test). Only server-wide settings
and an explicit `--llm-cache` file are shared.

**Security.** The HTTP transport has no authentication and binds to `127.0.0.1` by
default. Put it behind an authenticating proxy before exposing it. See [Security](security.md).
