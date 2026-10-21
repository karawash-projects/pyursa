# Security policy

## Reporting a vulnerability

Please report security issues privately, not in public issues. Use GitHub's
"Report a vulnerability" (Security → Advisories) on the repository, or email the
maintainers. Include a minimal reproduction and the affected version. We aim to
acknowledge reports within 3 working days and to ship a fix or mitigation for
confirmed high-severity issues within 30 days.

Supported versions: the latest minor release receives security fixes.

## Threat model

pyursa is designed to be driven by AI agents, so it assumes **the agent's
input is untrusted**: expressions, transform steps, file paths and the data itself.

What the library defends against:

| Risk | Defense |
|---|---|
| Code execution through expressions | Expressions are parsed with a whitelisted `ast` walker (`parse.py`); `eval`/`exec` are never used. Fuzz tests (`tests/test_security.py`) check that no dangerous audit events fire. |
| SQL injection (DuckDB backend) | Identifiers are quoted; **every user value is bound as a named parameter**, never spliced into SQL text. |
| File access outside allowed folders | `ToolSession` / MCP server resolve every path (including each glob match and symlink target) and refuse anything outside `--root`. |
| Prompt injection via data | Summaries and previews mark cell values as untrusted; summaries flag instruction-like text found anywhere in a column; LLM steps place row values in a JSON data block the model is told never to follow. |
| Runaway cost | LLM steps have row budgets, optional token budgets and cost reports; previews are capped at 50 rows. |
| Cross-client leakage (MCP over HTTP) | Each client connection gets its own session: frames, LLM cache and usage reports. |
| Secrets in errors | Provider errors never include API keys (tested). |
| Silent data corruption | Strict types and casts; lossy conversions (nanosecond truncation, Int64 overflow, decimal precision loss, non-whole float to integer) raise errors on every engine. |

What it does **not** do:

- **Authentication.** The MCP HTTP transport has none. It binds to `127.0.0.1` by
  default; expose it only behind an authenticating proxy.
- **Resource isolation.** Queries run in-process. A huge join can exhaust memory.
  Use `explain()` to check bounds, and run untrusted workloads in a container with
  memory limits.
- **Complete PII removal.** `redact_pii` is regex-based and will miss names,
  addresses and unusual formats. Don't rely on it for regulated data.
- **Guaranteed prompt-injection immunity.** Defenses reduce risk; models can still
  be influenced by data. Review LLM-derived columns before acting on them.
- **Time-of-check races on files.** A file swapped for a symlink after the sandbox
  check could be read. Don't give untrusted parties write access to `--root`.

## Data sent to LLM providers

Only `llm_filter`, `llm_map` and `llm_extract` send data off the machine, and only
the columns listed in `inputs` (default: all columns of the frame). Each step's
report states the model, the number of rows sent and whether redaction was on.
