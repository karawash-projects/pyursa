"""Security tests: parser fuzzing, hostile identifiers/values, sandbox escapes."""

import os
import random
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import pyursa as pu
from pyursa import col, lit
from pyursa.parse import parse_expr
from pyursa.tools import ToolSession

# --- parser fuzzing ------------------------------------------------------------

_DANGEROUS = {"os.system", "os.exec", "os.posix_spawn", "subprocess.Popen", "exec", "socket.connect",
              "os.remove", "shutil.rmtree", "ctypes.dlopen"}
_EVENTS: list[str] = []
_WATCH = [False]


def _hook(event, args):
    if _WATCH[0] and (event in _DANGEROUS or event.startswith("os.system") or event == "builtins.input"):
        _EVENTS.append(event)


sys.addaudithook(_hook)

PIECES = ['col("a")', 'col("b")', "lit(1)", "1", "2.5", '"x"', "None", "True", "(", ")", "[", "]", ",", ".",
          "+", "-", "*", "/", "&", "|", "~", "==", "<", ">=", " and ", " or ", " not ", " in ", " is ",
          "sum()", "over(", "when(", "then(", "otherwise(", "str", "dt", "contains(", "len()", "__import__",
          "__class__", "__subclasses__()", "open(", "eval(", "exec(", "lambda x: x", "os", "sys", "getattr(",
          "[x for x in ()]", "{}", "{'a': 1}", "f'{1}'", "b'x'", "@", ":=", "**", "%", "//", "<<", "...", "\\",
          "order_by=", "descending=True", "date(2026,1,1)", "decimal('1.5')", "row_number()", "rank(dense=True)"]


@pytest.mark.parametrize("seed", range(4))
def test_parser_fuzz_never_executes_or_crashes(seed):
    rng = random.Random(seed)
    _WATCH[0] = True
    try:
        for _ in range(1500):
            src = "".join(rng.choice(PIECES) for _ in range(rng.randint(1, 14)))
            try:
                parse_expr(src)
            except pu.PyursaError:
                pass  # rejections must be structured errors, nothing else
    finally:
        _WATCH[0] = False
    assert _EVENTS == [], f"dangerous audit events during parsing: {_EVENTS}"


@pytest.mark.parametrize("src", ["-" * 1990 + "1", "~" * 1990 + "col('a')", "not " * 490 + "True",
                                 "[" * 150 + "]" * 150, "col('a')" + ".abs()" * 250, "1" + "+1" * 900])
def test_pathological_inputs_are_rejected_cleanly(src):
    try:
        parse_expr(src)
    except pu.PyursaError as e:
        assert e.code == "EXPRESSION_SYNTAX"


@pytest.mark.parametrize("payload", [
    '__import__("os").system("echo pwned")', 'col("a").__class__.__base__.__subclasses__()',
    'getattr(col("a"), "__class__")', '(lambda: 1)()', 'col("a").str.__class__', 'col("a").alias.__globals__',
    'col("a").over.__func__', "[c for c in ().__class__.__mro__]", 'col("a").cast("Int64").__reduce__()',
    'open("/etc/passwd").read()', 'col(open("/etc/passwd").read())', "breakpoint()", "globals()",
    'col("a").sum().over("k", order_by=__import__("os"))',
])
def test_known_escape_payloads_rejected(payload):
    with pytest.raises(pu.PyursaError) as ei:
        parse_expr(payload)
    assert ei.value.code == "EXPRESSION_SYNTAX"


# --- hostile identifiers and values reach both engines safely --------------------------

HOSTILE_NAMES = ['a"b', "select", "x y", "ünïcødé", "'; DROP TABLE t; --", "1col", "Group", 'q"";--', "\\n"]
HOSTILE_VALUES = ["it's", 'say "hi"', "back\\slash", "new\nline", "'; DROP TABLE t; --", "%_", "ünï", "", "\x00z"]


@pytest.mark.parametrize("backend", pu.available_backends())
def test_hostile_identifiers_and_values(backend):
    data = {n: HOSTILE_VALUES[: len(HOSTILE_VALUES)] for n in HOSTILE_NAMES}
    df = pu.from_dict(data)
    first = HOSTILE_NAMES[0]
    out = (df.filter(col(first) != lit("it's"))
             .with_columns(**{"new\"col": col(HOSTILE_NAMES[3]).str.replace("'", "''")})
             .group_by(HOSTILE_NAMES[1]).agg(**{"n'": pu.len()})
             .sort("n'"))
    rows = out.to_dicts(backend)
    assert len(rows) == len(HOSTILE_VALUES) - 1 and all(r["n'"] == 1 for r in rows)
    for v in HOSTILE_VALUES:
        assert df.filter(col(first) == v).count(backend) == 1, repr(v)
        assert df.filter(col(first).str.contains(v)).count(backend) >= 1, repr(v)


@pytest.mark.parametrize("backend", pu.available_backends())
def test_hostile_file_paths(tmp_path, backend):
    d = tmp_path / "it's a dir"
    d.mkdir()
    path = d / "x'; DROP.parquet"
    pq.write_table(pa.table({"v": [1, 2]}), path)
    assert pu.read_parquet(str(path)).count(backend) == 2


# --- sandbox -------------------------------------------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.mkdir()
    pq.write_table(pa.table({"s": ["top secret"]}), outside / "s.parquet")
    pq.write_table(pa.table({"v": [1]}), root / "ok.parquet")
    return root, outside


@pytest.mark.skipif(os.name == "nt", reason="symlinks need admin rights on Windows")
def test_symlink_escape_blocked(sandbox):
    root, outside = sandbox
    os.symlink(outside / "s.parquet", root / "link.parquet")
    os.symlink(outside, root / "linkdir")
    s = ToolSession(roots=[root])
    assert s.load_file("link.parquet")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert s.load_file("linkdir")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert s.load_file("linkdir/*.parquet")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert s.load_file("*.parquet")["error"]["error"] == "PATH_NOT_ALLOWED"  # glob includes the symlink
    assert s.load_file("ok.parquet")["ok"]


def test_traversal_and_absolute_paths_blocked(sandbox):
    root, outside = sandbox
    s = ToolSession(roots=[root])
    for bad in ("../secret/s.parquet", str(outside / "s.parquet"), "../secret/*.parquet", str(outside)):
        r = s.load_file(bad)
        assert r["ok"] is False and r["error"]["error"] in ("PATH_NOT_ALLOWED", "FILE_NOT_FOUND"), (bad, r)
    h = s.load_file("ok.parquet")["handle"]
    assert s.export(h, "../secret/out.parquet")["error"]["error"] == "PATH_NOT_ALLOWED"
    assert not (outside / "out.parquet").exists()


def test_errors_never_leak_api_keys(monkeypatch):
    from pyursa.providers import OpenAICompatibleClient
    c = OpenAICompatibleClient("m", base_url="http://127.0.0.1:9/v1", api_key="sk-SECRET-123", max_attempts=1, timeout=1)
    pu.set_llm(c)
    try:
        with pytest.raises(pu.PyursaError) as ei:
            pu.from_dict({"t": ["a"]}).llm_map("x", output="y").collect()
        assert "sk-SECRET-123" not in ei.value.to_json() and "sk-SECRET-123" not in str(ei.value)
    finally:
        from pyursa import semantic as S
        S.CONFIG.client = None
