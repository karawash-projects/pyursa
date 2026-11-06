"""Journey: several AI agents share one pyursa MCP server over streamable HTTP.

Two clients connect at the same time and must never see or use each other's frames; one client's
concurrent calls must not corrupt its handles; a client that disconnects leaves nothing behind for the
next one; the server shuts down cleanly.
"""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from _harness import Journey, check, equal
from j_mcp_stdio import Tools, import_client, import_http_client, server_command, server_env


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited early with code {proc.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
            return
        except OSError:
            time.sleep(0.1)
    raise AssertionError(f"server did not accept connections on port {port} within {timeout:.0f}s")


def run(j: Journey) -> None:
    with j.step("mcp HTTP client and installed pyursa-mcp script are available", critical=True):
        ClientSession = import_client()[0]
        http_client = import_http_client()
        cmd = server_command()

    root = j.dir / "data"
    root.mkdir()
    pq.write_table(pa.table({"order_id": list(range(1, 101)), "amount": [float(i) for i in range(100)]}),
                   root / "orders.parquet")
    pq.write_table(pa.table({"customer": ["a", "b", "c"], "tier": ["gold", "silver", "gold"]}),
                   root / "customers.parquet")

    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    proc = None
    try:
        with j.step("start the server over streamable HTTP on 127.0.0.1", critical=True):
            log = open(j.dir / "server.log", "w")  # a file, not a pipe: request logs can't fill a buffer
            proc = subprocess.Popen([cmd, "--root", str(root), "--transport", "streamable-http",
                                     "--host", "127.0.0.1", "--port", str(port)],
                                    env=server_env(), stdin=subprocess.DEVNULL, stdout=log, stderr=log)
            _wait_for_port(port, proc)

        async def connect(stack):
            streams = await stack.enter_async_context(http_client(url))
            cs = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
            await cs.initialize()
            return Tools(cs)

        async def main() -> None:
            from contextlib import AsyncExitStack
            async with AsyncExitStack() as stack_a:
                a = await connect(stack_a)
                async with AsyncExitStack() as stack_b:
                    b = await connect(stack_b)

                    with j.step("two clients load different data at the same time"):
                        ra, rb = await asyncio.gather(a.ok("load_file", path="orders.parquet"),
                                                      b.ok("load_file", path="customers.parquet"))
                        equal(ra["handle"], "f1")
                        equal(rb["handle"], "f1", "each session numbers its own handles:")
                        check(ra["schema"] != rb["schema"], "the two f1 handles must be different frames")
                        ta = await a.ok("transform", handle="f1", steps=[{"op": "filter", "condition": 'col("amount") > 49'}])
                        equal(ta["handle"], "f2")

                    with j.step("isolation: list_frames differ; B can't see or use A's f2; f1 means each one's own"):
                        fa = (await a.ok("list_frames"))["frames"]
                        fb = (await b.ok("list_frames"))["frames"]
                        equal([f["handle"] for f in fa], ["f1", "f2"])
                        equal([f["handle"] for f in fb], ["f1"])
                        check("tier" in fb[0]["schema"] and "amount" in fa[0]["schema"], (fa, fb))
                        for name, args in (("describe", {"handle": "f2"}), ("count_rows", {"handle": "f2"}),
                                           ("preview", {"handle": "f2"}), ("lineage", {"handle": "f2"})):
                            r = await b(name, **args)
                            check(not r["ok"] and r["error"]["error"] == "HANDLE_NOT_FOUND", f"B {name}(f2): {r}")
                            check("amount" not in json.dumps(r), f"B learned about A's data: {r}")
                        equal((await a.ok("count_rows", handle="f2"))["rows"], 50)
                        equal((await b.ok("count_rows", handle="f1"))["rows"], 3)

                    with j.step("B dropping its f1 doesn't touch A's f1"):
                        await b.ok("drop_frame", handle="f1")
                        equal((await a.ok("count_rows", handle="f1"))["rows"], 100)

                    with j.step("per-client settings: B switching backend doesn't change A's"):
                        await b.ok("set_backend", backend="duckdb")
                        await a.ok("set_backend", backend="polars")
                        pa_ = await a.ok("explain", handle="f2")
                        await b.ok("load_file", path="customers.parquet")
                        pb_ = await b.ok("explain", handle="f2")
                        equal(pa_["backend"], "polars")
                        equal(pb_["backend"], "duckdb")

                    with j.step("LLM usage reports are per client"):
                        ua, ub = await a.ok("llm_usage"), await b.ok("llm_usage")
                        check(isinstance(ua["reports"], list) and isinstance(ub["reports"], list), (ua, ub))

                with j.step("one client's 25 concurrent transforms: all succeed with distinct, correct handles"):
                    results = await asyncio.gather(*[
                        a.ok("transform", handle="f1", steps=[{"op": "filter", "condition": f'col("order_id") <= {n}'}])
                        for n in range(1, 26)])
                    handles = [r["handle"] for r in results]
                    equal(len(set(handles)), 25, "duplicate handles from concurrent calls:")
                    counts = await asyncio.gather(*[a.ok("count_rows", handle=h) for h in handles])
                    equal([c["rows"] for c in counts], list(range(1, 26)), "each handle keeps its own pipeline:")
                    equal(len((await a.ok("list_frames"))["frames"]), 27)

                with j.step("A still works after B disconnected"):
                    equal((await a.ok("count_rows", handle="f2"))["rows"], 50)

            async with AsyncExitStack() as stack_c:
                c = await connect(stack_c)
                with j.step("a new client after A and B disconnected gets a fresh, empty session"):
                    equal((await c.ok("list_frames"))["frames"], [])
                    r = await c("count_rows", handle="f2")
                    check(not r["ok"] and r["error"]["error"] == "HANDLE_NOT_FOUND", r)
                    equal((await c.ok("load_file", path="orders.parquet"))["handle"], "f1")

            with j.step("sandbox and read-only rules hold over HTTP too"):
                async with AsyncExitStack() as stack_d:
                    d = await connect(stack_d)
                    for path in ("../x.parquet", "/etc/passwd", "/**/*"):
                        r = await d("load_file", path=path)
                        check(not r["ok"] and r["error"]["error"] in ("PATH_NOT_ALLOWED", "FILE_NOT_FOUND"), (path, r))
                    h = (await d.ok("load_file", path="orders.parquet"))["handle"]
                    r = await d("export", handle=h, path="out.parquet")
                    check(not r["ok"] and r["error"]["error"] == "READ_ONLY", r)

        asyncio.run(asyncio.wait_for(main(), timeout=120))

        with j.step("server shuts down cleanly on terminate"):
            proc.terminate()
            proc.wait(timeout=15)
            check(proc.returncode is not None, "server did not exit")
            err = (j.dir / "server.log").read_text(errors="replace")
            # mcp 1.x logs ClosedResourceError tracebacks from its own HTTP transport when a client
            # disconnects; only tracebacks that go through pyursa's code count as failures
            ours = [b for b in err.split("Traceback (most recent call last):")[1:]
                    if "pyursa" in b.split("\n\n")[0] or "pyursa" in "\n".join(b.splitlines()[:40])]
            check(not ours, f"server printed a traceback from pyursa:\n{ours[0][-1500:] if ours else ''}")
            check(not Path(root / "out.parquet").exists(), "read-only server wrote a file")
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
