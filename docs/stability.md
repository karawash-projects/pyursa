# Stability policy

pyursa follows [Semantic Versioning](https://semver.org/).

## Before 1.0 (0.x)

The API may still change between minor versions (0.5 → 0.6). Every breaking change is
listed in the changelog with a migration note, and we deprecate before removing where
practical.

## From 1.0

**Stable** (no breaking changes within a major version):

- Everything in `pyursa.__all__` and the methods of `Frame`, `GroupBy` and expressions.
- Error classes and their `code` strings; the shape of `to_dict()` (`error`, `message`,
  `details`, `suggestions`). New codes and new keys in `details` may be added.
- The semantics contract (`pu.SEMANTICS`). A change in results for the same input is a
  breaking change, except bug fixes where an engine violated the contract.
- `ToolSession` method names, arguments and the `{"ok": ...}` result shape; MCP tool names.

**Not covered:**

- Text of error messages and `describe_for_llm` output (they may be improved for agents).
- `explain()["compiled"]` and `logical_plan` text.
- Modules and names starting with `_`, and the internal plan/IR classes.

## Deprecation

Deprecated features emit `DeprecationWarning` for at least one minor release before
removal, and the changelog names the replacement.

## Supported versions

- Python: the versions tested in CI (currently 3.10–3.14).
- Dependencies: the lower bounds in `pyproject.toml` are tested in CI on every change.
- Security fixes go to the latest minor release.
