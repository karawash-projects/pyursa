"""Journey: an AI agent works with pyursa through the MCP server over stdio, like a desktop client.

The INSTALLED `pyursa-mcp` script is started with --root, and the official `mcp` client library drives a
realistic session: explore files, summarize, build a pipeline from expression strings, check it, recover
from its own mistake using the fix in the error, probe the sandbox (everything must be refused without
revealing anything), and finally export from a second server started with --allow-write. The results
are compared with the same pipeline computed in-process.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from _harness import Journey, Skip, check, equal

# ---------------------------------------------------------------------------------------------------------
# helpers shared with j_mcp_http.py


def server_command() -> str:
    """The installed `pyursa-mcp` console script for this interpreter (as a desktop client runs it)."""
    from _harness import console_script
    return str(console_script("pyursa-mcp"))


def import_client():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise Skip(f"mcp client library not importable: {e}") from None
    return ClientSession, StdioServerParameters, stdio_client


def import_http_client():
    try:
        from mcp.client.streamable_http import streamable_http_client as http_client
    except ImportError:
        try:
            from mcp.client.streamable_http import streamablehttp_client as http_client
        except ImportError as e:
            raise Skip(f"streamable HTTP client not available in this mcp version: {e}") from None
    return http_client


def payload(res) -> dict:
    """The JSON body of a tool result. Every pyursa tool must answer {"ok": true|false, ...}."""
    check(not getattr(res, "isError", False) and not getattr(res, "is_error", False),
          f"tool raised instead of returning a result: {getattr(res, 'content', res)}"[:600])
    sc = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if sc:
        body = sc.get("result", sc) if isinstance(sc, dict) else sc
    else:
        body = json.loads(res.content[0].text)  # must be valid JSON
    check(isinstance(body, dict) and isinstance(body.get("ok"), bool), f"result is not {{ok: bool, ...}}: {body!r}"[:500])
    json.dumps(body)  # and serializable back to JSON
    return body


class Tools:
    """await tools(name, **args) -> body, every result checked for the {ok: ...} shape."""

    def __init__(self, session) -> None:
        self.s = session

    async def __call__(self, name: str, **args) -> dict:
        return payload(await self.s.call_tool(name, args))

    async def ok(self, name: str, **args) -> dict:
        body = await self(name, **args)
        check(body["ok"], f"{name}({args}) failed: {json.dumps(body)[:600]}")
        return body


def jsonable(v):
    """The encoding the preview tool uses for exact JSON rows."""
    import datetime as dt
    import decimal
    if isinstance(v, decimal.Decimal):
        return str(v)
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    if isinstance(v, float) and math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    return v


def server_env() -> dict:
    """The server must import the INSTALLED pyursa, never a source tree from PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ---------------------------------------------------------------------------------------------------------
# data

SECRET = "TOPSECRET-7f3a91"
N_ORDERS = 2000
COUNTRIES = ["CA", "US", "NA", "FR"]  # "NA" is Namibia, not a missing value


def make_data(base: Path) -> tuple[Path, Path]:
    root, outside = base / "data", base / "zz_outside"
    (root / "sales" / "2024").mkdir(parents=True)
    (root / "sales" / ".ipynb_checkpoints").mkdir(parents=True)
    outside.mkdir()
    regions = ["east", "west", "north", "south"]
    price = [round(5 + (i * 37 % 1000) / 10, 2) for i in range(N_ORDERS)]
    price[17] = float("nan")
    pq.write_table(pa.table({
        "order_id": pa.array(range(1, N_ORDERS + 1), pa.int32()),
        "region": pa.array([regions[i % 4] for i in range(N_ORDERS)]).dictionary_encode(),  # category
        "qty": pa.array([i % 10 for i in range(N_ORDERS)], pa.int32()),
        "price": pa.array(price, pa.float32()),
        "customer_id": pa.array([1 + i % 50 for i in range(N_ORDERS)], pa.int64()),
    }), root / "orders.parquet", row_group_size=256)
    lines = ["customer_id,name,country"] + [f"{i},Customer {i},{COUNTRIES[i % 4]}" for i in range(1, 51)]
    (root / "customers.csv").write_text("\n".join(lines) + "\n")
    for q, start in (("q1", 0), ("q2", 100)):
        pq.write_table(pa.table({"day": pa.array(range(start, start + 100), pa.int32()),
                                 "amount": pa.array([float(x) for x in range(100)])}),
                       root / "sales" / "2024" / f"{q}.parquet")
    pq.write_table(pa.table({"day": [999], "amount": [1e9]}), root / "sales" / ".ipynb_checkpoints" / "x.parquet")
    (root / ".env").write_text(f"API_KEY={SECRET}\n")
    pq.write_table(pa.table({"secret": [SECRET]}), outside / "secret.parquet")
    if os.name != "nt":
        os.symlink(outside, root / "link_out")
    return root, outside


def pipeline_steps(customers: str) -> list[dict]:
    return [
        {"op": "filter", "condition": 'col("qty") > 0'},
        {"op": "join", "right": customers, "on": "customer_id", "how": "left"},
        {"op": "with_columns", "columns": {"rev": 'col("qty") * col("price")'}},
        {"op": "group_by", "keys": ["country"],
         "aggs": {"revenue": 'col("rev").sum()', "orders": "len()", "avg_qty": 'col("qty").mean()'}},
        {"op": "sort", "by": ["country"]},
    ]


def in_process(root: Path, backend: str) -> list[dict]:
    import pyursa as pu
    from pyursa import col
    orders = pu.read_parquet(str(root / "orders.parquet"))
    customers = pu.read_csv(str(root / "customers.csv"))
    out = (orders.filter(col("qty") > 0).join(customers, on="customer_id", how="left")
           .with_columns(rev=col("qty") * col("price"))
           .group_by("country").agg(revenue=col("rev").sum(), orders=pu.len(), avg_qty=col("qty").mean())
           .sort("country"))
    return [jsonable(r) for r in out.to_dicts(backend=backend)]


# ---------------------------------------------------------------------------------------------------------


def run(j: Journey) -> None:
    ClientSession, StdioServerParameters, stdio_client = None, None, None
    with j.step("mcp client library and installed pyursa-mcp script are available", critical=True):
        ClientSession, StdioServerParameters, stdio_client = import_client()
        cmd = server_command()
        j.note(cmd)
    with j.step("write realistic data (category/int32/float32+NaN parquet, CSV with 'NA', nested folder, "
                "hidden folder, .env, outside symlink)", critical=True):
        root, outside = make_data(j.dir)

    async def session(extra_args: list[str], body) -> None:
        params = StdioServerParameters(command=cmd, args=["--root", str(root), *extra_args], env=server_env())
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as cs:
                await cs.initialize()
                await body(cs, Tools(cs))

    async def agent(cs, t: Tools) -> None:
        with j.step("list tools: the documented tool set is offered, each with a description", critical=True):
            tools = {tool.name: tool for tool in (await cs.list_tools()).tools}
            need = {"pyursa_help", "load_file", "list_frames", "describe", "transform", "preview", "count_rows",
                    "explain", "history", "undo", "lineage", "export", "drop_frame", "set_backend", "llm_usage"}
            check(need <= set(tools), f"missing tools: {sorted(need - set(tools))}")
            check(all(tools[n].description for n in need), "a tool has no description")

        with j.step("pyursa_help explains expressions, ops, semantics and backends"):
            h = await t.ok("pyursa_help")
            check("filter" in json.dumps(h["transform_ops"]) and h["backends"], json.dumps(h)[:300])

        handles: dict[str, str] = {}
        with j.step("load_file: orders.parquet (lazy), customers.csv, folder 'sales', glob 'sales/**/*.parquet'",
                    critical=True):
            o = await t.ok("load_file", path="orders.parquet")
            equal(o["schema"], {"order_id": "Int64", "region": "String", "qty": "Int64", "price": "Float64",
                                "customer_id": "Int64"}, "orders schema (file types conformed):")
            handles["orders"] = o["handle"]
            c = await t.ok("load_file", path="customers.csv")
            equal(c["schema"]["country"], "String", "country type:")
            handles["customers"] = c["handle"]
            f = await t.ok("load_file", path="sales")
            equal(f["files"], 2, "folder load must skip the hidden .ipynb_checkpoints folder:")
            g = await t.ok("load_file", path="sales/**/*.parquet")
            equal(g["files"], 2, "glob must skip hidden folders:")
            handles["sales"] = f["handle"]

        with j.step("count_rows on loaded frames; hidden checkpoint rows are not included"):
            equal((await t.ok("count_rows", handle=handles["orders"]))["rows"], N_ORDERS)
            equal((await t.ok("count_rows", handle=handles["customers"]))["rows"], 50)
            equal((await t.ok("count_rows", handle=handles["sales"]))["rows"], 200)

        with j.step("list_frames shows the session's handles with schemas"):
            fr = (await t.ok("list_frames"))["frames"]
            check({x["handle"] for x in fr} >= set(handles.values()), json.dumps(fr)[:300])

        with j.step("describe customers: fits the token budget, keeps 'NA' as a country, no warnings"):
            d = (await t.ok("describe", handle=handles["customers"], tokens=300))["summary"]
            check(math.ceil(len(d) / 4) <= 300, f"summary too long: {len(d)} chars")
            check('"NA"' in d, f"'NA' should be a real value in the summary:\n{d}")
            check("⚠" not in d, f"unexpected warning in summary:\n{d}")
            check("50 rows" in d, d[:200])

        with j.step("describe orders (category/float32/NaN file) fits small budgets"):
            for budget in (60, 200, 800):
                d = (await t.ok("describe", handle=handles["orders"], tokens=budget))["summary"]
                check(math.ceil(len(d) / 4) <= budget, f"{budget}-token summary is {math.ceil(len(d) / 4)} tokens")

        pipe = None
        with j.step("transform: filter -> join -> with_columns -> group_by -> sort, with an intent", critical=True):
            pipe = await t.ok("transform", handle=handles["orders"], steps=pipeline_steps(handles["customers"]),
                              intent="revenue by country for the weekly report")
            equal(pipe["schema"], {"country": "String", "revenue": "Float64", "orders": "Int64", "avg_qty": "Float64"})
            check(pipe["handle"] not in handles.values(), "transform must return a NEW handle")

        expected = {b: in_process(root, b) for b in ("polars", "duckdb")}
        with j.step("in-process pipeline gives identical rows on polars and duckdb"):
            equal(expected["duckdb"], expected["polars"])

        with j.step("preview + count: rows match the in-process computation exactly, on both engines"):
            for backend in ("polars", "duckdb"):
                await t.ok("set_backend", backend=backend)
                p = await t.ok("preview", handle=pipe["handle"], rows=50)
                equal(p["rows"], expected[backend], f"[{backend}] preview rows:")
                equal((await t.ok("count_rows", handle=pipe["handle"]))["rows"], len(COUNTRIES))
                check({r["country"] for r in p["rows"]} == set(COUNTRIES), "'NA' must be its own country group")
                check(p["table"].startswith("| country |"), p["table"][:100])
            nan_rows = [r for r in p["rows"] if r["revenue"] == "NaN"]
            equal(len(nan_rows), 1, "exactly one country has the NaN price in its revenue:")

        with j.step("preview of the original orders handle is unchanged (immutability)"):
            p0 = await t.ok("preview", handle=handles["orders"], rows=3)
            equal([r["order_id"] for r in p0["rows"]], [1, 2, 3])

        with j.step("preview caps rows at 50 and accepts silly sizes"):
            equal((await t.ok("preview", handle=handles["orders"], rows=10**9))["rows_shown"], 50)
            equal((await t.ok("preview", handle=handles["orders"], rows=-5))["rows_shown"], 0)

        with j.step("explain is a dry run with schema, bounds and the compiled query"):
            e = await t.ok("explain", handle=pipe["handle"])
            equal(list(e["output_schema"]), ["country", "revenue", "orders", "avg_qty"])
            check("estimated_rows_max" in e and e.get("compiled"), json.dumps(e)[:300])
            check(len(e["sources"]) == 2, json.dumps(e["sources"])[:300])

        with j.step("lineage traces revenue to qty and price, the filter, the join and the grouping"):
            lin = (await t.ok("lineage", handle=pipe["handle"], column="revenue"))["lineage"]
            blob = json.dumps(lin)
            check("price" in blob and "qty" in blob, blob[:400])
            check(lin.get("filters"), f"filter missing from lineage: {blob[:400]}")
            check(any("join" in str(o) for o in lin.get("operations", [])) or "join" in blob, blob[:400])
            check("country" in blob, blob[:400])

        with j.step("history records every step and the intent"):
            hist = (await t.ok("history", handle=pipe["handle"]))["history"]
            ops = [h_["op"] for h_ in hist]
            check(ops[0] == "source" and len(ops) >= 6, ops)
            check(any(h_.get("intent") == "revenue by country for the weekly report" for h_ in hist), hist)

        with j.step("undo 2 steps returns the frame before group_by/sort (with the derived column)"):
            u = await t.ok("undo", handle=pipe["handle"], steps=2)
            check("rev" in u["schema"] and "country" in u["schema"], u["schema"])
            equal((await t.ok("count_rows", handle=u["handle"]))["rows"], N_ORDERS - N_ORDERS // 10)
            bad = await t("undo", handle=pipe["handle"], steps=99)
            check(not bad["ok"], "undo beyond the source must be an error")

        with j.step("agent mistake: typo column -> structured error with a fix -> pasted fix succeeds"):
            wrong = 'col("quantity") > 5'
            err = await t("transform", handle=handles["orders"], steps=[{"op": "filter", "condition": wrong}])
            check(not err["ok"], "typo must fail")
            equal(err["error"]["error"], "COLUMN_NOT_FOUND")
            fix = err["error"]["suggestions"][0]["fix"]
            equal(fix, 'col("qty")')
            check(err["error"]["details"].get("step_index") == 0, err["error"]["details"])
            fixed = await t.ok("transform", handle=handles["orders"],
                               steps=[{"op": "filter", "condition": wrong.replace('col("quantity")', fix)}])
            equal((await t.ok("count_rows", handle=fixed["handle"]))["rows"], N_ORDERS * 4 // 10)

        with j.step("more agent mistakes are structured errors, never crashes"):
            cases = [
                ("transform", {"handle": handles["orders"], "steps": [{"op": "filter", "condition": 'col("qty") > "5"'}]}, "TYPE_MISMATCH"),
                ("transform", {"handle": handles["orders"], "steps": [{"op": "filter", "condition": '__import__("os").system("id")'}]}, None),
                ("transform", {"handle": handles["orders"], "steps": [{"op": "explode"}]}, "INVALID_OPERATION"),
                ("transform", {"handle": handles["orders"], "steps": []}, "BAD_ARGUMENTS"),
                ("describe", {"handle": "f999"}, "HANDLE_NOT_FOUND"),
                ("describe", {"handle": handles["orders"], "tokens": 5}, "BAD_ARGUMENTS"),
                ("set_backend", {"backend": "spark"}, "UNKNOWN_BACKEND"),
                ("lineage", {"handle": pipe["handle"], "column": "nope"}, None),
            ]
            for name, args, code in cases:
                r = await t(name, **args)
                check(not r["ok"], f"{name}({args}) should fail")
                check(r["error"].get("error") and r["error"].get("message"), r)
                if code:
                    equal(r["error"]["error"], code, f"{name}({args}):")
            await t.ok("set_backend", backend="polars")

        with j.step("sandbox: every escape attempt is refused and reveals nothing"):
            attacks = {
                "../zz_outside/secret.parquet": [SECRET],
                "/etc/passwd": ["root:"],
                "/etc/pass*": ["passwd", "root:"],
                "/**/*": ["/usr", "/bin", "/etc", SECRET],
                "**/../../zz_outside/*.parquet": [SECRET],
                ".env": [SECRET, "API_KEY"],
                "./.env": [SECRET, "API_KEY"],
                "sales/.ipynb_checkpoints/x.parquet": ["1000000000", "999"],
                "sales/.ipynb_checkpoints": ["1000000000"],
                str(outside / "secret.parquet"): [SECRET],
            }
            if os.name != "nt":
                attacks.update({"link_out/secret.parquet": [SECRET], "link_out/*.parquet": [SECRET],
                                "link_out": [SECRET], "**/secret.parquet": [SECRET]})
            for path, markers in attacks.items():
                r = await t("load_file", path=path)
                check(not r["ok"], f"load_file({path!r}) must be refused: {json.dumps(r)[:300]}")
                code = r["error"]["error"]
                check(code in ("PATH_NOT_ALLOWED", "FILE_NOT_FOUND"), f"{path!r}: unexpected code {code}")
                blob = json.dumps(r)
                leaked = [m for m in markers if m in blob]
                check(not leaked, f"{path!r} revealed {leaked}: {blob[:400]}")

        with j.step("export is refused: the server runs read-only without --allow-write"):
            r = await t("export", handle=pipe["handle"], path="report.parquet")
            check(not r["ok"] and r["error"]["error"] == "READ_ONLY", json.dumps(r)[:300])
            check(not (root / "report.parquet").exists(), "a read-only server wrote a file")

        with j.step("drop_frame frees a handle; using it afterwards is a structured error"):
            await t.ok("drop_frame", handle=fixed["handle"])
            r = await t("count_rows", handle=fixed["handle"])
            check(not r["ok"] and r["error"]["error"] == "HANDLE_NOT_FOUND", r)

        with j.step("llm_usage answers (no LLM configured: empty report list)"):
            u = await t.ok("llm_usage")
            check(isinstance(u["reports"], list), u)

    async def writer(cs, t: Tools) -> None:
        o = await t.ok("load_file", path="orders.parquet")
        c = await t.ok("load_file", path="customers.csv")
        pipe = await t.ok("transform", handle=o["handle"], steps=pipeline_steps(c["handle"]))
        rows = (await t.ok("preview", handle=pipe["handle"], rows=50))["rows"]

        with j.step("--allow-write: export parquet, csv and json into the root"):
            for name in ("report.parquet", "report.csv", "report.json"):
                r = await t.ok("export", handle=pipe["handle"], path=name)
                equal(r["rows"], len(COUNTRIES), f"{name} rows:")
                check(Path(r["path"]).resolve().parent == root.resolve(), f"{name} written outside root: {r['path']}")
            r = await t("export", handle=pipe["handle"], path="report.parquet")
            check(not r["ok"] and r["error"]["error"] == "FILE_EXISTS", "existing file must not be overwritten silently")
            await t.ok("export", handle=pipe["handle"], path="report.parquet", overwrite=True)

        with j.step("--allow-write still refuses bad targets (.py, hidden names, outside, missing folder)"):
            bad = {"evil.py": "INVALID_OPERATION", ".hidden.parquet": "PATH_NOT_ALLOWED",
                   "sales/.ipynb_checkpoints/y.parquet": "PATH_NOT_ALLOWED", "../escape.parquet": "PATH_NOT_ALLOWED",
                   str(outside / "x.parquet"): "PATH_NOT_ALLOWED", "nope/x.parquet": "DIRECTORY_NOT_FOUND"}
            if os.name != "nt":
                bad["link_out/x.parquet"] = "PATH_NOT_ALLOWED"
            for path, code in bad.items():
                r = await t("export", handle=pipe["handle"], path=path)
                check(not r["ok"], f"export to {path!r} must be refused")
                equal(r["error"]["error"], code, f"export {path!r}:")
            check(not (j.dir / "escape.parquet").exists() and not (outside / "x.parquet").exists(), "file escaped root")
            check(not (root / "evil.py").exists() and not (root / ".hidden.parquet").exists(), "bad file was written")

        with j.step("exported files reread with pyursa give the same rows"):
            import pyursa as pu
            equal([jsonable(r) for r in pu.read_parquet(str(root / "report.parquet")).to_dicts()], rows, "parquet:")
            csv_rows = pu.read_csv(str(root / "report.csv")).sort("country").to_dicts()
            equal([r["country"] for r in csv_rows], [r["country"] for r in rows], "csv countries ('NA' kept):")
            equal([r["orders"] for r in csv_rows], [r["orders"] for r in rows], "csv orders:")
            for line, want in zip((root / "report.json").read_text().splitlines(), rows):
                got = json.loads(line, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"non-strict JSON {c}")))
                equal(got["country"], want["country"])
                equal(got["revenue"], None if want["revenue"] == "NaN" else want["revenue"], "json revenue:")

        with j.step("the new files are loadable by the agent in the same session"):
            r = await t.ok("load_file", path="report.json")
            equal((await t.ok("count_rows", handle=r["handle"]))["rows"], len(COUNTRIES))

    async def main() -> None:
        await asyncio.wait_for(session([], agent), timeout=120)
        with j.step("start a second server with --allow-write", critical=True):
            pass
        await asyncio.wait_for(session(["--allow-write"], writer), timeout=120)

    with j.step("server without --root refuses to start"):
        import subprocess
        p = subprocess.run([cmd], capture_output=True, text=True, timeout=60, env=server_env(), stdin=subprocess.DEVNULL)
        check(p.returncode != 0 and "--root" in (p.stderr + p.stdout), f"rc={p.returncode} {p.stderr[-300:]}")

    asyncio.run(main())
