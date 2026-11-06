#!/usr/bin/env python3
"""Run the end-to-end user journeys against the INSTALLED pyursa package.

    pip install "pyursa[all,mcp]"            # or: pip install dist/pyursa-*.whl[all,mcp]
    python scripts/journeys/run.py           # every journey
    python scripts/journeys/run.py analyst mcp_stdio -v

Each journey is a realistic multi-step workflow (an analyst, a data engineer, an AI agent over MCP, LLM
pipelines, error recovery, scale, packaging). Exit code 1 if any step fails.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _harness import Journey, StopJourney  # noqa: E402


def main() -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to a code page without ✗
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("journeys", nargs="*")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    import pyursa
    src = Path(pyursa.__file__).resolve()
    import os
    if (HERE.parent.parent / "src") in src.parents and not os.environ.get("PYURSA_JOURNEYS_ALLOW_SRC"):
        print(f"refusing to run: pyursa is imported from the source tree ({src}); install the package first")
        return 2
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for mod in ("polars", "duckdb", "pyarrow", "pandas", "mcp"):
        try:
            versions[mod] = version(mod)
        except PackageNotFoundError:
            versions[mod] = "-"
    print(f"pyursa {pyursa.__version__} from {src.parent}")
    print(f"python {sys.version.split()[0]} | " + " | ".join(f"{k} {v}" for k, v in versions.items()) + "\n")
    names = args.journeys or sorted(p.stem[2:] for p in HERE.glob("j_*.py"))
    all_results = {}
    t0 = time.perf_counter()
    for name in names:
        print(f"== {name} " + "=" * (60 - len(name)), flush=True)
        j = Journey(name, args.verbose)
        try:
            importlib.import_module(f"j_{name}").run(j)
        except StopJourney as e:
            print(f"    (stopped: critical step failed: {e})")
        except Exception as e:  # noqa: BLE001
            j.results.append(("journey setup", "FAIL", 0.0, f"{type(e).__name__}: {e}"))
            print(f"    FAIL journey crashed: {type(e).__name__}: {e}")
        all_results[name] = j.results
    print("\n" + "=" * 64)
    fails = 0
    for name, res in all_results.items():
        p = sum(r[1] == "PASS" for r in res)
        f = sum(r[1] == "FAIL" for r in res)
        s = sum(r[1] == "SKIP" for r in res)
        fails += f
        print(f"  {'ok  ' if not f else 'FAIL'} {name:<16} {p} passed, {f} failed, {s} skipped")
        for r in res:
            if r[1] == "FAIL":
                print(f"         ✗ {r[0]}: {r[3].splitlines()[0][:200]}")
    total = sum(len(r) for r in all_results.values())
    print(f"\n{total - fails} of {total} steps ok in {time.perf_counter() - t0:.1f}s")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
