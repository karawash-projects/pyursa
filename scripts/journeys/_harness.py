"""Tiny harness for end-to-end user journeys.

A journey is a function `run(j: Journey)` in a module named j_<name>.py. It works the way a real user or
agent would: through the installed `pyursa` package only (never the source tree), in a fresh temporary
folder (`j.dir`). Each `with j.step("...")` block is one checked step; a failing step is recorded and the
journey continues with the next step unless it is marked `critical=True`.
"""

from __future__ import annotations

import contextlib
import tempfile
import time
import traceback
from pathlib import Path


class StopJourney(Exception):
    pass


class Journey:
    def __init__(self, name: str, verbose: bool = False) -> None:
        self.name = name
        self.verbose = verbose
        self.dir = Path(tempfile.mkdtemp(prefix=f"pyursa-journey-{name}-"))
        self.results: list[tuple[str, str, float, str]] = []  # (step, status, seconds, detail)

    @contextlib.contextmanager
    def step(self, desc: str, critical: bool = False):
        t = time.perf_counter()
        try:
            yield
        except Skip as e:
            self.results.append((desc, "SKIP", time.perf_counter() - t, str(e)))
            print(f"    SKIP {desc}: {e}", flush=True)
        except Exception as e:  # noqa: BLE001 - every failure is recorded
            detail = f"{type(e).__name__}: {e}"
            if self.verbose:
                detail += "\n" + traceback.format_exc()
            self.results.append((desc, "FAIL", time.perf_counter() - t, detail))
            print(f"    FAIL {desc}\n         {detail.splitlines()[0][:300]}", flush=True)
            if critical:
                raise StopJourney(desc) from e
        else:
            dt = time.perf_counter() - t
            self.results.append((desc, "PASS", dt, ""))
            print(f"    PASS {desc} ({dt * 1000:.0f} ms)", flush=True)

    def note(self, text: str) -> None:
        print(f"         {text}", flush=True)


class Skip(Exception):
    """Raise inside a step when it can't run here (missing optional dependency)."""


def check(cond: bool, msg: str = "check failed") -> None:
    if not cond:
        raise AssertionError(msg)


def equal(a, b, msg: str = "") -> None:
    if a != b:
        raise AssertionError(f"{msg} expected {b!r}, got {a!r}"[:2000])


def same_table(a, b) -> bool:
    """Arrow tables with the same schema and values, where NaN equals NaN (pyursa's semantics)."""
    if not a.schema.equals(b.schema) or a.num_rows != b.num_rows:
        return False
    norm = lambda v: ("NaN",) if isinstance(v, float) and v != v else v  # noqa: E731
    return all([norm(v) for v in a.column(i).to_pylist()] == [norm(v) for v in b.column(i).to_pylist()]
               for i in range(a.num_columns))


def console_script(name: str):
    """Path of an installed console script for this interpreter (bin/ on Unix, Scripts\\ on Windows)."""
    import sys
    import sysconfig
    from pathlib import Path
    dirs = [Path(sysconfig.get_path("scripts")), Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"]
    try:
        dirs.append(Path(sysconfig.get_path("scripts", f"{__import__('os').name}_user")))
    except KeyError:
        pass
    for d in dirs:
        for cand in (d / name, d / f"{name}.exe"):
            if cand.exists():
                return cand
    raise AssertionError(f"{name} console script not found in {', '.join(map(str, dirs))}; is pyursa[mcp] installed?")
