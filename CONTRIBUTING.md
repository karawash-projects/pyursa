# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest                       # unit, conformance, provider and security tests
python scripts/e2e.py        # end-to-end checks
ruff check src tests scripts benchmarks
```

## Ground rules

1. **Every behavior is the same on every backend.** If you add an operation, add a
   case with explicit expected output to `tests/test_conformance*.py`. It must pass on
   all backends unchanged. If engines disagree, pick a rule, document it in
   `SEMANTICS` (`backends/base.py`) and make each backend follow it.
2. **No silent coercion.** New types or casts must refuse lossy conversions.
3. **Errors are for agents.** Raise an `PyursaError` subclass with a stable `code`,
   machine-readable `details`, and at least one `suggestion` (with a `fix` snippet
   when one exists). Catch mistakes when the plan is built, not at execution.
4. **Untrusted input.** Anything reachable from `parse.py` or `tools.py` must be safe
   for hostile input. Add a case to `tests/test_security.py`.
5. **Public API changes** follow the stability policy in `docs/stability.md` and get a
   `CHANGELOG.md` entry.

## Adding a backend

Subclass `pyursa.backends.Backend`, implement `compile(node)` and
`_run(node) -> pyarrow.Table`, register it with `pu.register_backend`, and add its
name to `BACKENDS` in `tests/conftest.py`. It is done when the conformance suites pass.
