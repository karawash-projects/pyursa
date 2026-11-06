"""Journey: someone installs pyursa from PyPI and starts using it.

Checks the installed distribution itself: metadata and version agree, the console script works and
enforces the safe defaults, importing is quick and doesn't load engines eagerly, the package carries its
type marker and licence, and (when PYURSA_WHEEL points at a built wheel) a bare install without engines
gives a helpful error and adding an engine makes it work.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from _harness import Skip, check, equal


def _script(name: str) -> Path:
    from _harness import console_script
    return console_script(name)


def run(j):
    from importlib import metadata

    import pyursa as pu

    with j.step("installed metadata matches pyursa.__version__"):
        equal(metadata.version("pyursa"), pu.__version__, "version")
        md = metadata.metadata("pyursa")
        equal(md["License-Expression"] or md["License"], "Apache-2.0", "licence")
        check(md["Requires-Python"] == ">=3.10", md["Requires-Python"])
        classifiers = md.get_all("Classifier") or []
        check(any("Beta" in c for c in classifiers), classifiers)

    with j.step("package files: py.typed, licence; no tests or caches shipped"):
        files = [str(f) for f in (metadata.files("pyursa") or [])]
        check(any(f.endswith("pyursa/py.typed") for f in files), "py.typed missing")
        check(any("LICENSE" in f for f in files), "LICENSE missing")
        check(not any("/tests/" in f or f.startswith("tests/") or ".pyc" in f and "__pycache__" not in f for f in files),
              [f for f in files if "test" in f][:5])

    with j.step("import is quick and loads no engine until it's needed"):
        code = ("import sys, time; t=time.perf_counter(); import pyursa; dt=time.perf_counter()-t; "
                "print(dt, 'polars' in sys.modules, 'duckdb' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, cwd=j.dir)
        dt_, pl_loaded, dd_loaded = out.stdout.split()
        j.note(f"import pyursa: {float(dt_) * 1000:.0f} ms (polars loaded: {pl_loaded}, duckdb loaded: {dd_loaded})")
        check(float(dt_) < 3, f"import took {dt_}s")
        check(dd_loaded == "False", "duckdb imported eagerly")

    with j.step("pyursa-mcp --help works and documents the safe defaults"):
        out = subprocess.run([str(_script("pyursa-mcp")), "--help"], capture_output=True, text=True, timeout=60)
        equal(out.returncode, 0, out.stderr)
        for flag in ("--root", "--allow-write", "--max-load-mb", "--transport"):
            check(flag in out.stdout, f"{flag} missing from --help")

    with j.step("pyursa-mcp refuses to start without --root"):
        out = subprocess.run([str(_script("pyursa-mcp"))], capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        check(out.returncode != 0 and "--root" in out.stderr, out.stderr[-300:])

    with j.step("python -m pyursa.mcp_server is equivalent"):
        out = subprocess.run([sys.executable, "-m", "pyursa.mcp_server", "--help"], capture_output=True, text=True, timeout=60)
        equal(out.returncode, 0, out.stderr[-300:])

    with j.step("the semantics contract and providers are inspectable"):
        check({"null_comparison", "decimals", "files", "regex", "known_differences"} <= set(pu.SEMANTICS), list(pu.SEMANTICS))
        names = [p["provider"] for p in pu.list_providers()]
        check(len(names) >= 50 and "gemini" in names and "bedrock" in names, names[:10])

    with j.step("a bare install (no engines) explains what to install; adding one makes it work"):
        wheel = os.environ.get("PYURSA_WHEEL")
        if not wheel:
            raise Skip("set PYURSA_WHEEL=<path to the built wheel> to test a clean install")
        venv = j.dir / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, timeout=300)
        py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        t = time.perf_counter()
        r = subprocess.run([str(py), "-m", "pip", "install", "-q", wheel], capture_output=True, text=True, timeout=600)
        equal(r.returncode, 0, r.stderr[-500:])
        code = ("import pyursa as pu\n"
                "f = pu.from_dict({'x': [1, 2]})\n"
                "try:\n    f.collect()\n    print('NO-ERROR')\n"
                "except pu.PyursaError as e:\n    print(e.code, '|', e.message, '|', e.suggestions)\n")
        out = subprocess.run([str(py), "-c", code], capture_output=True, text=True, timeout=120)
        check("pip install" in out.stdout and "NO-ERROR" not in out.stdout, out.stdout + out.stderr[-300:])
        r = subprocess.run([str(py), "-m", "pip", "install", "-q", "duckdb"], capture_output=True, text=True, timeout=600)
        equal(r.returncode, 0, r.stderr[-500:])
        out = subprocess.run([str(py), "-c", "import pyursa as pu; print(pu.from_dict({'x':[1,2]}).select(pu.col('x').sum()).to_dicts())"],
                             capture_output=True, text=True, timeout=120)
        check("[{'x': 3}]" in out.stdout, out.stdout + out.stderr[-300:])
        j.note(f"clean install + checks: {time.perf_counter() - t:.0f} s")
