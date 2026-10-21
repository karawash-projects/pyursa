import asyncio
import json
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pyursa.tools import ToolSession


@pytest.fixture
def session(tmp_path):
    pq.write_table(pa.table({
        "order_id": [1, 2, 3, 4, 5],
        "region": ["east", "west", "east", None, "west"],
        "qty": [10, 20, None, 5, 20],
    }), tmp_path / "orders.parquet")
    (tmp_path / "managers.csv").write_text("region,manager\neast,Ana\nwest,Bo\n")
    return ToolSession(roots=[tmp_path])


def test_agent_workflow(session, tmp_path):
    r = session.load_file("orders.parquet")
    assert r["ok"] and r["handle"] == "f1" and r["rows"] == 5
    m = session.load_file("managers.csv")["handle"]

    assert "5 rows" in session.describe("f1", tokens=300)["summary"]

    t = session.transform("f1", [
        {"op": "filter", "condition": 'col("qty") > 5'},
        {"op": "join", "right": m, "on": ["region"]},
        {"op": "group_by", "keys": ["manager"], "aggs": {"total": 'col("qty").sum()', "n": "len()"}},
    ], intent="volume per manager")
    assert t["ok"] and t["schema"] == {"manager": "String", "total": "Int64", "n": "Int64"}

    rows = session.preview(t["handle"])
    assert "| Ana | 10 | 1 |" in rows["table"].replace('"', "")

    assert session.history(t["handle"])["history"][1]["intent"] == "volume per manager"
    assert session.undo(t["handle"], 3)["handle"] == "f1"  # back to the original handle

    out = session.export(t["handle"], "out/result.parquet")
    assert not out["ok"]  # parent dir missing -> structured error, not a crash
    (tmp_path / "out").mkdir()
    assert session.export(t["handle"], "out/result.parquet")["rows"] == 2

    assert session.set_backend("duckdb")["ok"]
    assert session.preview(t["handle"])["table"] == rows["table"]


def test_errors_are_returned_not_raised(session):
    session.load_file("orders.parquet")
    r = session.transform("f1", [{"op": "filter", "condition": 'col("quantity") > 5'}])
    assert r["ok"] is False
    assert r["error"]["error"] == "COLUMN_NOT_FOUND"
    assert r["error"]["suggestions"][0]["fix"] == 'col("qty")'
    assert r["error"]["details"]["step_index"] == 0
    json.dumps(r)

    assert session.transform("f1", [{"op": "pivot"}])["error"]["error"] == "INVALID_OPERATION"
    assert session.describe("f99")["error"]["error"] == "HANDLE_NOT_FOUND"
    assert session.transform("f1", [{"op": "filter"}])["error"]["error"] == "BAD_ARGUMENTS"


def test_sandbox(session):
    assert session.load_file("/etc/passwd")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert session.load_file("../../etc/passwd")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert session.load_file("nope.parquet")["error"]["error"] == "FILE_NOT_FOUND"
    r = session.transform(session.load_file("orders.parquet")["handle"],
                          [{"op": "filter", "condition": '__import__("os").system("id")'}])
    assert r["error"]["error"] == "EXPRESSION_SYNTAX"


def test_preview_is_capped(session):
    session.load_file("orders.parquet")
    assert session.preview("f1", rows=10_000)["max_rows"] == 50


def test_mcp_server_over_stdio(session, tmp_path):
    """End-to-end: launch the real server as a subprocess and call it as an MCP client."""
    pytest.importorskip("mcp")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "pyursa.mcp_server", "--root", str(tmp_path)],
            env={"PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1] / "src")},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = {t.name for t in (await client.list_tools()).tools}
                assert {"load_file", "describe", "transform", "undo", "lineage", "explain"} <= tools

                def payload(res):
                    if getattr(res, "structuredContent", None):
                        sc = res.structuredContent
                        return sc.get("result", sc)
                    return json.loads(res.content[0].text)

                r = payload(await client.call_tool("load_file", {"path": "orders.parquet"}))
                assert r["ok"] and r["handle"] == "f1"
                r = payload(await client.call_tool("transform", {
                    "handle": "f1", "steps": [{"op": "filter", "condition": 'col("qty") >= 20'}]}))
                assert r["handle"] == "f2"
                r = payload(await client.call_tool("count_rows", {"handle": "f2"}))
                assert r["rows"] == 2
                r = payload(await client.call_tool("transform", {
                    "handle": "f1", "steps": [{"op": "filter", "condition": 'col("qtty") > 1'}]}))
                assert r["ok"] is False and r["error"]["error"] == "COLUMN_NOT_FOUND"

    asyncio.run(run())


def test_mcp_http_sessions_are_isolated(session, tmp_path):
    """Two clients connected at the same time over HTTP never see each other's frames."""
    pytest.importorskip("mcp")
    import socket
    import subprocess
    import time

    from mcp import ClientSession
    try:
        from mcp.client.streamable_http import streamable_http_client as http_client
    except ImportError:  # mcp 1.x
        from mcp.client.streamable_http import streamablehttp_client as http_client

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = {**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[1] / "src")}
    proc = subprocess.Popen([sys.executable, "-m", "pyursa.mcp_server", "--root", str(tmp_path),
                             "--transport", "streamable-http", "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        url = f"http://127.0.0.1:{port}/mcp"

        def payload(res):
            sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
            return sc.get("result", sc) if sc else json.loads(res.content[0].text)

        async def run():
            async with http_client(url) as streams_a, http_client(url) as streams_b:
                async with ClientSession(streams_a[0], streams_a[1]) as a, ClientSession(streams_b[0], streams_b[1]) as b:
                    await a.initialize()
                    await b.initialize()
                    ra = payload(await a.call_tool("load_file", {"path": "orders.parquet"}))
                    assert ra["handle"] == "f1"
                    # client B has its own empty session
                    assert payload(await b.call_tool("list_frames", {}))["frames"] == []
                    rb = payload(await b.call_tool("describe", {"handle": "f1"}))
                    assert rb["ok"] is False and rb["error"]["error"] == "HANDLE_NOT_FOUND"
                    # client A still sees its frame
                    assert payload(await a.call_tool("count_rows", {"handle": "f1"}))["rows"] == 5

        asyncio.run(asyncio.wait_for(run(), timeout=60))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
