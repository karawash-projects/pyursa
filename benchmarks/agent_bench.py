"""Agent benchmark: does an LLM agent analyse data better through pyursa's
tool interface than by writing pandas code? Same model, same tasks, same data.

    PYTHONPATH=src python benchmarks/agent_bench.py --llm gemini:gemini-3.8-flash --allow-code-exec
    PYTHONPATH=src python benchmarks/agent_bench.py --fake --allow-code-exec      # no API key needed

Conditions
  pyursa  the model drives pyursa.tools.ToolSession via JSON actions
  pandas      the model writes Python/pandas code, run in a fresh subprocess
              (needs --allow-code-exec, because it executes model-generated code)

Headline metrics per run: correct, turns, error turns, approx tokens in/out
(chars / 4) and wall time. See benchmarks/README.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

try:  # allow running without PYTHONPATH=src from a source checkout
    import pyursa  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CONDITIONS = ("pyursa", "pandas")
TOOL_RESULT_LIMIT = 3000
CODE_OUTPUT_LIMIT = 2000
CODE_TIMEOUT_S = 30
ORDERS_FILE = "orders.parquet"
CUSTOMERS_FILE = "customers.csv"
DATA_FILES = (ORDERS_FILE, CUSTOMERS_FILE)

# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------


def generate_data(out_dir: str | os.PathLike, rows: int = 50_000, seed: int = 0) -> Path:
    """Write orders.parquet and customers.csv deterministically; return the directory."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n_cust = max(100, rows // 25)
    cust_ids = np.arange(1001, 1001 + n_cust, dtype=np.int64)
    segment = rng.choice(["enterprise", "smb", "consumer"], size=n_cust, p=[0.2, 0.35, 0.45])
    region = rng.choice(["east", "west", "north", "south"], size=n_cust, p=[0.4, 0.3, 0.2, 0.1])
    pd.DataFrame({"customer_id": cust_ids, "segment": segment, "region": region}).to_csv(
        out / CUSTOMERS_FILE, index=False)

    weights = rng.pareto(1.5, n_cust) + 1.0  # a few heavy customers, so "top 3" is clear
    weights /= weights.sum()
    customer_id = rng.choice(cust_ids, size=rows, p=weights)
    qty = rng.integers(1, 11, size=rows).astype(np.int64)
    bad = rng.random(rows) < 0.04
    qty[bad] = rng.choice([0, -1, -2], size=int(bad.sum()))
    qty_null = rng.random(rows) < 0.03
    price = np.clip(np.round(rng.lognormal(3.5, 0.8, size=rows), 2), 0.5, 2000.0)
    status = rng.choice(["paid", "shipped", "cancelled", "refunded"], size=rows, p=[0.55, 0.3, 0.1, 0.05])
    start = dt.date(2026, 1, 1)
    offsets = rng.integers(0, 270, size=rows)  # 2026-01-01 .. 2026-09-27
    day = [start + dt.timedelta(days=int(o)) for o in offsets]

    table = pa.table({
        "order_id": pa.array(np.arange(1, rows + 1, dtype=np.int64)),
        "customer_id": pa.array(customer_id, pa.int64()),
        "qty": pa.array(qty, pa.int64(), mask=qty_null),
        "price": pa.array(price, pa.float64()),
        "status": pa.array(status.tolist(), pa.string()),
        "day": pa.array(day, pa.date32()),
    })
    pq.write_table(table, out / ORDERS_FILE)
    return out


# --------------------------------------------------------------------------------------
# Tasks (ground truth computed independently with pandas)
# --------------------------------------------------------------------------------------

VALID_DEF = ("A *valid* order has qty > 0 (a missing qty is not valid) and status other than "
             "'cancelled'. The revenue of an order is qty * price.")


@dataclass(frozen=True)
class Task:
    id: str
    question: str
    kind: str  # "float" (tolerance), "int" (exact), "str", "list"
    truth: Callable[[Any, Any], Any]  # (orders_df, customers_df) -> answer


def _load_pandas(data_dir: Path):
    import pandas as pd
    o = pd.read_parquet(data_dir / ORDERS_FILE)
    c = pd.read_csv(data_dir / CUSTOMERS_FILE)
    o["day"] = pd.to_datetime(o["day"])
    return o, c


def _valid(o):
    return o[(o["qty"] > 0) & (o["status"] != "cancelled")]


def _strict_top(s, n: int, ascending: bool = False) -> list:
    """Sorted index labels of a Series; refuses ties at the cut so answers are unambiguous."""
    s = s.sort_values(ascending=ascending)
    vals = s.tolist()
    if len(set(vals[: n + 1])) < min(n + 1, len(vals)):
        raise ValueError(f"Ambiguous ranking (tie) in ground truth: {s.head(n + 1).to_dict()}")
    return s.index[:n].tolist()


def _t1(o, c):
    v = _valid(o)
    return round(float((v["qty"] * v["price"]).sum()), 2)


def _t2(o, c):
    return round(100.0 * float((o["status"] == "refunded").mean()), 2)


def _t3(o, c):
    m = (o["day"] >= "2026-04-01") & (o["day"] <= "2026-06-30")
    return int(o.loc[m, "customer_id"].nunique())


def _t4(o, c):
    ent = c.loc[c["segment"] == "enterprise", "customer_id"]
    return round(float(o.loc[o["customer_id"].isin(ent), "price"].mean()), 2)


def _t5(o, c):
    v = _valid(o).merge(c, on="customer_id")
    rev = (v["qty"] * v["price"]).groupby(v["region"]).sum()
    return _strict_top(rev, 1)[0]


def _t6(o, c):
    v = _valid(o)
    rev = (v["qty"] * v["price"]).groupby(v["customer_id"]).sum()
    return [int(x) for x in _strict_top(rev, 3)]


def _t7(o, c):
    return int((o["qty"].isna() | (o["qty"] <= 0)).sum())


def _t8(o, c):
    v = _valid(o).merge(c, on="customer_id")
    v = v[v["region"] == "west"]
    return round(float((v["qty"] * v["price"]).max()), 2)


def _t9(o, c):
    v = _valid(o).merge(c, on="customer_id")
    m = (v["segment"] == "smb") & (v["day"] >= "2026-08-01") & (v["day"] <= "2026-08-31")
    return int(m.sum())


def _t10(o, c):
    s = o[o["status"] == "shipped"].merge(c, on="customer_id")
    counts = s.groupby("region").size()
    return [str(x) for x in _strict_top(counts, len(counts), ascending=True)]


TASKS: list[Task] = [
    Task("t1", f"{VALID_DEF} What is the total revenue of all valid orders? Answer as a number rounded to 2 decimals.",
         "float", _t1),
    Task("t2", "What percentage of all orders (every row in the orders file) have status 'refunded'? "
               "Answer as a percentage number between 0 and 100, rounded to 2 decimals (e.g. 4.87).", "float", _t2),
    Task("t3", "How many distinct customers placed at least one order (any status) with an order date between "
               "2026-04-01 and 2026-06-30 inclusive? Answer as an integer.", "int", _t3),
    Task("t4", "What is the average order price (the price column, not qty * price) over all orders (any status) "
               "placed by customers in the 'enterprise' segment? Answer as a number rounded to 2 decimals.",
         "float", _t4),
    Task("t5", f"{VALID_DEF} Which customer region has the highest total revenue from valid orders? "
               "Answer with the region name as a string.", "str", _t5),
    Task("t6", f"{VALID_DEF} Which 3 customers have the highest total revenue from valid orders? Answer as a JSON "
               "list of the 3 integer customer_ids, highest revenue first.", "list", _t6),
    Task("t7", "How many orders have a missing qty or a qty less than or equal to 0? Answer as an integer.",
         "int", _t7),
    Task("t8", f"{VALID_DEF} What is the largest revenue of a single valid order placed by a customer in the "
               "'west' region? Answer as a number rounded to 2 decimals.", "float", _t8),
    Task("t9", f"{VALID_DEF} How many valid orders were placed by customers in the 'smb' segment with an order "
               "date in August 2026 (2026-08-01 to 2026-08-31 inclusive)? Answer as an integer.", "int", _t9),
    Task("t10", "Count the orders with status 'shipped' per customer region. Answer as a JSON list of all region "
                "names ordered from the fewest shipped orders to the most.", "list", _t10),
]
TASKS_BY_ID = {t.id: t for t in TASKS}


def ground_truth(data_dir: Path) -> dict[str, Any]:
    o, c = _load_pandas(Path(data_dir))
    return {t.id: t.truth(o, c) for t in TASKS}


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------

REL_TOL = 1e-3
ABS_TOL = 0.01


def _to_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").rstrip("%").strip()
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _norm_str(v: Any) -> str:
    return str(v).strip().strip("'\"").strip().lower()


def _scalar_equal(a: Any, truth: Any) -> bool:
    if isinstance(truth, (int, float)) and not isinstance(truth, bool):
        x = _to_number(a)
        return x is not None and float(x) == float(truth)
    return _norm_str(a) == _norm_str(truth)


def score(answer: Any, truth: Any, kind: str) -> bool:
    if answer is None:
        return False
    if kind == "float":
        x = _to_number(answer)
        if x is None or math.isnan(x):
            return False
        t = float(truth)
        return abs(x - t) <= ABS_TOL or abs(x - t) <= REL_TOL * abs(t)
    if kind == "int":
        x = _to_number(answer)
        return x is not None and x == float(truth)
    if kind == "str":
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        return _norm_str(answer) == _norm_str(truth)
    if kind == "list":
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError:
                return False
        if not isinstance(answer, (list, tuple)) or len(answer) != len(truth):
            return False
        return all(_scalar_equal(a, t) for a, t in zip(answer, truth))
    raise ValueError(f"unknown task kind {kind!r}")


# --------------------------------------------------------------------------------------
# Reply parsing
# --------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)```", re.DOTALL)


def parse_reply(text: str) -> dict[str, Any] | None:
    """Return the first JSON object with an "action" key, or None.

    Accepts a bare object, an object inside a ```json fence, or an object embedded in prose.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    dec = json.JSONDecoder()
    for chunk in candidates:
        chunk = chunk.strip()
        for i, ch in enumerate(chunk):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(chunk, i)
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("action"), str):
                return obj
    return None


def _truncate(s: str, limit: int, keep_tail: bool = False) -> str:
    if len(s) <= limit:
        return s
    cut = len(s) - limit
    if keep_tail:  # keep the end (where the exception message is), plus a little head
        head = limit // 4
        return s[:head] + f"\n...[{cut} chars truncated]...\n" + s[-(limit - head):]
    return s[:limit] + f"...[{cut} chars truncated]"


# --------------------------------------------------------------------------------------
# Conditions (environments)
# --------------------------------------------------------------------------------------

AF_ACTIONS = ("help", "load_file", "list_frames", "describe", "transform", "preview", "count_rows",
              "explain", "history", "undo", "lineage")

PROTOCOL = (
    "Reply with exactly ONE JSON object per turn and nothing else (a ```json fence is fine). "
    "You will get the result of your action, then reply with your next action. "
    'When you know the final answer, reply {{"action": "answer", "value": <answer>}}, where <answer> is '
    "a JSON number, string or list in the format the question asks for. You have at most {max_turns} turns."
)


def _af_system(max_turns: int) -> str:
    from pyursa.tools import OPS_HELP
    ops = {k: v for k, v in OPS_HELP.items() if not k.startswith("llm_")}
    return (
        "You are a data analyst. You answer questions by calling dataframe tools.\n"
        f"Data files (in the working directory): {ORDERS_FILE}, {CUSTOMERS_FILE}. "
        "Column names and types are not given; inspect the data to find them.\n\n"
        "Actions: {\"action\": \"<tool>\", \"args\": {...}}. Tools:\n"
        "- load_file(path, name=None) -> handle like \"f1\"\n"
        "- list_frames()\n"
        "- describe(handle, tokens=800): schema, stats, sample rows\n"
        "- transform(handle, steps, intent=None): apply steps, returns a NEW handle\n"
        "- preview(handle, rows=10): show rows\n"
        "- count_rows(handle)\n"
        "- explain(handle), history(handle), undo(handle, steps=1), lineage(handle, column=None)\n"
        "- help(): full syntax reference\n"
        f"Transform step formats ({{\"op\": name, ...}}): {json.dumps(ops)}\n"
        'Expressions are strings: col("qty") > 0, (col("a") > 1) & (col("b") != "x"), col("x").is_null(), '
        'col("x").is_in([...]), col("d") >= date(2026, 3, 1), col("a") * col("b"). '
        'Aggregations: col("x").sum(), .mean(), .min(), .max(), .count(), .n_unique(), len().\n\n'
        + PROTOCOL.format(max_turns=max_turns)
    )


def _pandas_system(max_turns: int) -> str:
    return (
        "You are a data analyst. You answer questions by writing Python code with pandas.\n"
        f"Data files (in DATA_DIR, also the working directory): {ORDERS_FILE}, {CUSTOMERS_FILE}. "
        "Column names and types are not given; inspect the data to find them.\n\n"
        'Actions: {"action": "python", "code": "<python source>"}. Each call runs in a FRESH Python process '
        "(no state is kept between calls) with `import pandas as pd` and `DATA_DIR` (a pathlib.Path) already "
        "defined. You see stdout and stderr (truncated), so print() what you need. Timeout: "
        f"{CODE_TIMEOUT_S}s.\n\n" + PROTOCOL.format(max_turns=max_turns)
    )


class PyursaEnv:
    name = "pyursa"

    def __init__(self, data_dir: Path) -> None:
        from pyursa.tools import ToolSession
        self.session = ToolSession(roots=[str(data_dir)])

    def system_prompt(self, max_turns: int) -> str:
        return _af_system(max_turns)

    def step(self, action: dict[str, Any]) -> tuple[str, bool]:
        """Run one non-answer action; return (observation text, is_error)."""
        name = action.get("action")
        if name not in AF_ACTIONS:
            return (f"Unknown action {name!r}. Use one of: {', '.join(AF_ACTIONS)}, answer."), True
        args = action.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return 'ERROR: "args" must be a JSON object.', True
        try:
            result = getattr(self.session, name)(**args)
        except Exception as e:  # ToolSession catches most errors itself; be defensive anyway
            result = {"ok": False, "error": {"error": "HARNESS_ERROR", "message": f"{type(e).__name__}: {e}"}}
        text = json.dumps(result, default=str, ensure_ascii=False)
        return _truncate(text, TOOL_RESULT_LIMIT), not (isinstance(result, dict) and result.get("ok"))

    def close(self) -> None:
        pass


_SECRET_ENV = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)


class PandasEnv:
    name = "pandas"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.script_dir = Path(tempfile.mkdtemp(prefix="agent_bench_code_"))
        self.n = 0
        self.env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
        self.env["PYTHONIOENCODING"] = "utf-8"

    def system_prompt(self, max_turns: int) -> str:
        return _pandas_system(max_turns)

    def step(self, action: dict[str, Any]) -> tuple[str, bool]:
        if action.get("action") != "python":
            return f"Unknown action {action.get('action')!r}. Use python or answer.", True
        code = action.get("code")
        if not isinstance(code, str) or not code.strip():
            return 'ERROR: "code" must be a non-empty string of Python source.', True
        self.n += 1
        script = self.script_dir / f"step_{self.n}.py"
        preamble = ("import pandas as pd\nfrom pathlib import Path\n"
                    f"DATA_DIR = Path({str(self.data_dir)!r})\n# --- agent code ---\n")
        script.write_text(preamble + code, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], cwd=str(self.data_dir), env=self.env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace",
                                  timeout=CODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"ERROR: code timed out after {CODE_TIMEOUT_S}s.", True
        out = (proc.stdout or "") + (proc.stderr or "")
        failed = proc.returncode != 0
        if not out.strip():
            out = "(no output; use print() to see values)"
        if failed:
            out = f"[exit code {proc.returncode}]\n" + out
        return _truncate(out, CODE_OUTPUT_LIMIT, keep_tail=failed), failed

    def close(self) -> None:
        shutil.rmtree(self.script_dir, ignore_errors=True)


def make_env(condition: str, data_dir: Path):
    if condition == "pyursa":
        return PyursaEnv(data_dir)
    if condition == "pandas":
        return PandasEnv(data_dir)
    raise ValueError(f"Unknown condition {condition!r}; use one of {CONDITIONS}.")


# --------------------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------------------

CORRECTIVE = ('Your reply was not a valid action. Reply with exactly one JSON object such as '
              '{"action": "...", ...} or {"action": "answer", "value": ...}.')


@dataclass
class RunRecord:
    task: str
    condition: str
    trial: int
    correct: bool
    turns: int
    error_turns: int
    approx_tokens_in: int
    approx_tokens_out: int
    seconds: float
    answer: Any
    truth: Any
    finished: bool
    error: str | None = None
    transcript: list[dict[str, str]] = field(default_factory=list)


def build_user(question: str, history: list[tuple[str, str]]) -> str:
    parts = [f"QUESTION: {question}"]
    for i, (reply, obs) in enumerate(history, 1):
        parts.append(f"--- turn {i} ---\nYOUR ACTION:\n{reply.strip()}\n\nRESULT:\n{obs}")
    parts.append("Reply with your next action as one JSON object.")
    return "\n\n".join(parts)


def run_episode(client: Any, task: Task, condition: str, data_dir: Path, truth: Any,
                max_turns: int = 10, trial: int = 0, keep_transcript: bool = False) -> RunRecord:
    env = make_env(condition, data_dir)
    system = env.system_prompt(max_turns)
    history: list[tuple[str, str]] = []
    tok_in = tok_out = errors = turns = 0
    answer: Any = None
    finished = False
    err: str | None = None
    t0 = time.perf_counter()
    try:
        while turns < max_turns:
            user = build_user(task.question, history)
            tok_in += (len(system) + len(user)) // 4
            try:
                reply = client.complete(system, user)
            except Exception as e:  # provider failure: stop this run, keep the benchmark going
                err = f"{type(e).__name__}: {e}"
                break
            reply = reply if isinstance(reply, str) else str(reply)
            turns += 1
            tok_out += len(reply) // 4
            action = parse_reply(reply)
            if action is None:
                errors += 1
                history.append((reply, CORRECTIVE))
                continue
            if action["action"] == "answer":
                if "value" not in action:
                    errors += 1
                    history.append((reply, 'ERROR: "answer" needs a "value" field.'))
                    continue
                answer, finished = action["value"], True
                break
            obs, is_err = env.step(action)
            errors += int(is_err)
            history.append((reply, obs))
    finally:
        env.close()
    seconds = time.perf_counter() - t0
    transcript = [{"reply": r, "result": o} for r, o in history] if keep_transcript else []
    return RunRecord(task=task.id, condition=condition, trial=trial, correct=finished and score(answer, truth, task.kind),
                     turns=turns, error_turns=errors, approx_tokens_in=tok_in, approx_tokens_out=tok_out,
                     seconds=round(seconds, 3), answer=answer, truth=truth, finished=finished, error=err,
                     transcript=transcript)


# --------------------------------------------------------------------------------------
# Fake scripted client (demo / tests, no API key)
# --------------------------------------------------------------------------------------

_VALID_F = '(col("qty") > 0) & (col("status") != "cancelled")'
_REV = 'col("qty") * col("price")'

# per task: (needs customers join, transform steps applied to orders, column to read, read mode)
FAKE_AF: dict[str, tuple[bool, list[dict[str, Any]], str, str]] = {
    "t1": (False, [{"op": "filter", "condition": _VALID_F},
                   {"op": "aggregate", "aggs": {"rev": f"({_REV}).sum()"}}], "rev", "scalar"),
    "t2": (False, [{"op": "aggregate", "aggs": {"n": "len()",
                                                "r": '(col("status") == "refunded").cast("Int64").sum()'}},
                   {"op": "with_columns", "columns": {"pct": 'col("r") * 100.0 / col("n")'}}], "pct", "scalar"),
    "t3": (False, [{"op": "filter", "condition": '(col("day") >= date(2026, 4, 1)) & (col("day") <= date(2026, 6, 30))'},
                   {"op": "aggregate", "aggs": {"n": 'col("customer_id").n_unique()'}}], "n", "scalar"),
    "t4": (True, [{"op": "filter", "condition": 'col("segment") == "enterprise"'},
                  {"op": "aggregate", "aggs": {"avg": 'col("price").mean()'}}], "avg", "scalar"),
    "t5": (True, [{"op": "filter", "condition": _VALID_F},
                  {"op": "with_columns", "columns": {"rev": _REV}},
                  {"op": "group_by", "keys": ["region"], "aggs": {"total": 'col("rev").sum()'}},
                  {"op": "sort", "by": ["total"], "descending": True}, {"op": "head", "n": 1}], "region", "scalar"),
    "t6": (False, [{"op": "filter", "condition": _VALID_F},
                   {"op": "with_columns", "columns": {"rev": _REV}},
                   {"op": "group_by", "keys": ["customer_id"], "aggs": {"total": 'col("rev").sum()'}},
                   {"op": "sort", "by": ["total"], "descending": True}, {"op": "head", "n": 3}],
           "customer_id", "list"),
    "t7": (False, [{"op": "filter", "condition": 'col("qty").is_null() | (col("qty") <= 0)'},
                   {"op": "aggregate", "aggs": {"n": "len()"}}], "n", "scalar"),
    "t8": (True, [{"op": "filter", "condition": f'{_VALID_F} & (col("region") == "west")'},
                  {"op": "aggregate", "aggs": {"mx": f"({_REV}).max()"}}], "mx", "scalar"),
    "t9": (True, [{"op": "filter", "condition": f'{_VALID_F} & (col("segment") == "smb") & '
                                                '(col("day") >= date(2026, 8, 1)) & (col("day") <= date(2026, 8, 31))'},
                  {"op": "aggregate", "aggs": {"n": "len()"}}], "n", "scalar"),
    "t10": (True, [{"op": "filter", "condition": 'col("status") == "shipped"'},
                   {"op": "group_by", "keys": ["region"], "aggs": {"n": "len()"}},
                   {"op": "sort", "by": ["n"], "descending": False}], "region", "list"),
}

_PD_PRE = ("import json\no = pd.read_parquet(DATA_DIR / 'orders.parquet')\n"
           "c = pd.read_csv(DATA_DIR / 'customers.csv')\no['day'] = pd.to_datetime(o['day'])\n"
           "v = o[(o.qty > 0) & (o.status != 'cancelled')].copy()\nv['rev'] = v.qty * v.price\n"
           "vc = v.merge(c, on='customer_id')\n")
FAKE_PD: dict[str, str] = {
    "t1": "ans = round(float(v.rev.sum()), 2)",
    "t2": "ans = round(float((o.status == 'refunded').mean() * 100), 2)",
    "t3": "ans = int(o[o.day.between('2026-04-01', '2026-06-30')].customer_id.nunique())",
    "t4": "ans = round(float(o.merge(c, on='customer_id').query(\"segment == 'enterprise'\").price.mean()), 2)",
    "t5": "ans = vc.groupby('region').rev.sum().idxmax()",
    "t6": "ans = [int(x) for x in v.groupby('customer_id').rev.sum().nlargest(3).index]",
    "t7": "ans = int((o.qty.isna() | (o.qty <= 0)).sum())",
    "t8": "ans = round(float(vc[vc.region == 'west'].rev.max()), 2)",
    "t9": "ans = int(((vc.segment == 'smb') & vc.day.between('2026-08-01', '2026-08-31')).sum())",
    "t10": "ans = list(o[o.status == 'shipped'].merge(c, on='customer_id').groupby('region').size()"
           ".sort_values().index)",
}


def _parse_cell(s: str) -> Any:
    s = s.strip()
    if s == "null":
        return None
    if len(s) >= 2 and s[0] == s[-1] == '"':
        return s[1:-1]
    try:
        f = float(s.replace(",", ""))
        return int(f) if f.is_integer() and "e" not in s.lower() and "." not in s else f
    except ValueError:
        return s


def parse_preview_table(table: str) -> list[dict[str, Any]]:
    lines = [ln for ln in table.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [_parse_cell(x) for x in ln.strip().strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
    return rows


class FakeClient:
    """Scripted 'model' that solves every task correctly, for demos and tests.

    It reads the condition from the system prompt, the task from the question, and the turn
    number from the transcript, and parses real tool/code output to produce its answer.
    """

    model_id = "fake:scripted"

    def complete(self, system: str, user: str) -> str:
        m = re.search(r"QUESTION: (.*?)\n\n", user + "\n\n", re.DOTALL)
        question = m.group(1) if m else ""
        task = next((t for t in TASKS if t.question == question), None)
        if task is None:
            return "I don't know this task."
        results = re.findall(r"RESULT:\n(.*?)(?=\n\n--- turn |\n\nReply with your next action)", user, re.DOTALL)
        turn = len(results)
        if "dataframe tools" in system:
            return self._pyursa(task, turn, results)
        return self._pandas(task, turn, results)

    @staticmethod
    def _act(obj: dict[str, Any]) -> str:
        return "```json\n" + json.dumps(obj) + "\n```"

    def _pyursa(self, task: Task, turn: int, results: list[str]) -> str:
        join, steps, column, mode = FAKE_AF[task.id]
        script: list[dict[str, Any]] = [{"action": "load_file", "args": {"path": ORDERS_FILE}},
                                        {"action": "describe", "args": {"handle": "f1", "tokens": 400}}]
        if join:
            script.append({"action": "load_file", "args": {"path": CUSTOMERS_FILE}})
            steps = [{"op": "join", "right": "f2", "on": ["customer_id"], "how": "inner"}] + steps
        if mode == "scalar" and column != "region":
            # preview prints floats with 4 significant digits, so read float results as fixed-point ints (x 1e4)
            steps = steps + [{"op": "with_columns",
                              "columns": {"fixed": f'(col("{column}") * 10000).round(0).cast("Int64")'}}]
        script.append({"action": "transform", "args": {"handle": "f1", "steps": steps, "intent": task.id}})
        if turn < len(script):
            return self._act(script[turn])
        if turn == len(script):
            handle = json.loads(results[-1])["handle"]
            return self._act({"action": "preview", "args": {"handle": handle, "rows": 10}})
        rows = parse_preview_table(json.loads(results[-1])["table"])
        if mode == "scalar" and column != "region":
            value: Any = round(rows[0]["fixed"] / 10000, 2)
            return self._act({"action": "answer", "value": int(value) if task.kind == "int" else value})
        values = [r[column] for r in rows]
        return self._act({"action": "answer", "value": values if mode == "list" else values[0]})

    def _pandas(self, task: Task, turn: int, results: list[str]) -> str:
        if turn == 0:
            code = ("print(pd.read_parquet(DATA_DIR / 'orders.parquet').head(3))\n"
                    "print(pd.read_csv(DATA_DIR / 'customers.csv').head(3))")
            return self._act({"action": "python", "code": code})
        if turn == 1:
            return self._act({"action": "python", "code": _PD_PRE + FAKE_PD[task.id] + "\nprint(json.dumps(ans))"})
        return self._act({"action": "answer", "value": json.loads(results[-1].strip().splitlines()[-1])})


# --------------------------------------------------------------------------------------
# Benchmark driver and reporting
# --------------------------------------------------------------------------------------

CODE_EXEC_SKIP_MSG = ("Skipping the 'pandas' condition: it executes model-generated Python on this machine. "
                      "Re-run with --allow-code-exec to include it.")


def resolve_conditions(conditions: list[str], allow_code_exec: bool) -> tuple[list[str], list[str]]:
    out, msgs = [], []
    for c in conditions:
        if c not in CONDITIONS:
            raise ValueError(f"Unknown condition {c!r}; choose from {', '.join(CONDITIONS)}.")
        if c == "pandas" and not allow_code_exec:
            msgs.append(CODE_EXEC_SKIP_MSG)
            continue
        if c not in out:
            out.append(c)
    return out, msgs


def resolve_tasks(spec: str) -> list[Task]:
    if spec.strip().lower() == "all":
        return list(TASKS)
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [i for i in ids if i not in TASKS_BY_ID]
    if unknown:
        raise ValueError(f"Unknown task id(s) {unknown}; choose from {', '.join(TASKS_BY_ID)} or 'all'.")
    return [TASKS_BY_ID[i] for i in ids]


def run_benchmark(client: Any, tasks: list[Task], conditions: list[str], trials: int = 1, rows: int = 50_000,
                  max_turns: int = 10, seed: int = 0, keep_transcripts: bool = False,
                  log: Callable[[str], None] = lambda m: print(m, flush=True)) -> list[RunRecord]:
    records: list[RunRecord] = []
    base = Path(tempfile.mkdtemp(prefix="agent_bench_"))
    try:
        master = generate_data(base / "master", rows=rows, seed=seed)
        truth = ground_truth(master)
        n = 0
        total = trials * len(tasks) * len(conditions)
        started = time.time()
        for trial in range(trials):
            for task in tasks:
                for cond in conditions:
                    n += 1
                    elapsed = time.time() - started
                    eta = "" if n == 1 else f", about {elapsed / (n - 1) * (total - n + 1) / 60:.0f} min left"
                    log(f"... run {n}/{total}: trial {trial + 1} {task.id} {cond} (elapsed {elapsed / 60:.1f} min{eta})")
                    run_dir = base / f"run_{n}"  # fresh copy per run, so an agent cannot corrupt later runs
                    run_dir.mkdir()
                    for f in DATA_FILES:
                        shutil.copy2(master / f, run_dir / f)
                    rec = run_episode(client, task, cond, run_dir, truth[task.id], max_turns=max_turns,
                                      trial=trial, keep_transcript=keep_transcripts)
                    records.append(rec)
                    mark = "PASS" if rec.correct else "FAIL"
                    log(f"[{mark}] {n}/{total} trial {trial + 1} {task.id:<4} {cond:<10} turns={rec.turns} "
                        f"err={rec.error_turns} {rec.seconds:.1f}s answer={json.dumps(rec.answer, default=str)}"
                        + (f" error={rec.error}" if rec.error else ""))
                    shutil.rmtree(run_dir, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return records


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize(records: list[RunRecord]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for cond in dict.fromkeys(r.condition for r in records):
        rs = [r for r in records if r.condition == cond]
        out[cond] = {
            "runs": len(rs),
            "success_rate": _mean([float(r.correct) for r in rs]),
            "mean_turns": _mean([r.turns for r in rs]),
            "mean_error_turns": _mean([r.error_turns for r in rs]),
            "mean_tokens": _mean([r.approx_tokens_in + r.approx_tokens_out for r in rs]),
            "mean_seconds": _mean([r.seconds for r in rs]),
            "provider_errors": sum(1 for r in rs if r.error),
        }
    return out


def format_report(records: list[RunRecord]) -> str:
    lines = ["", "Per task (success = passed/runs; other columns are means):",
             f"{'task':<5} {'condition':<11} {'success':>8} {'turns':>6} {'errors':>7} {'tokens':>8} {'seconds':>8}"]
    keys = list(dict.fromkeys((r.task, r.condition) for r in records))
    for task, cond in keys:
        rs = [r for r in records if r.task == task and r.condition == cond]
        lines.append(f"{task:<5} {cond:<11} {sum(r.correct for r in rs):>4}/{len(rs):<3} "
                     f"{_mean([r.turns for r in rs]):>6.1f} {_mean([r.error_turns for r in rs]):>7.1f} "
                     f"{_mean([r.approx_tokens_in + r.approx_tokens_out for r in rs]):>8.0f} "
                     f"{_mean([r.seconds for r in rs]):>8.2f}")
    lines += ["", "Summary per condition:",
              f"{'condition':<11} {'runs':>5} {'success':>8} {'turns':>6} {'errors':>7} {'tokens':>8} {'seconds':>8}"]
    for cond, s in summarize(records).items():
        lines.append(f"{cond:<11} {s['runs']:>5} {s['success_rate']:>7.0%} {s['mean_turns']:>6.2f} "
                     f"{s['mean_error_turns']:>7.2f} {s['mean_tokens']:>8.0f} {s['mean_seconds']:>8.2f}"
                     + (f"   ({s['provider_errors']} runs hit provider errors)" if s["provider_errors"] else ""))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark an LLM agent on data tasks: pyursa tools vs pandas code.")
    p.add_argument("--llm", help='Model spec for pyursa.providers.llm_client, e.g. "gemini:gemini-3.8-flash".')
    p.add_argument("--fake", action="store_true", help="Use a scripted fake client (no API key needed).")
    p.add_argument("--conditions", default="pyursa,pandas", help="Comma-separated: pyursa,pandas.")
    p.add_argument("--tasks", default="all", help="Comma-separated task ids (t1,t3) or 'all'.")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--rows", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--allow-code-exec", action="store_true",
                   help="Allow the pandas condition, which runs model-generated Python in a subprocess.")
    p.add_argument("--out", help="Write all run records (with transcripts) to this JSON file.")
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if not args.fake and not args.llm:
        build_parser().error("pass --llm SPEC or --fake")
    conditions, msgs = resolve_conditions([c.strip() for c in args.conditions.split(",") if c.strip()],
                                          args.allow_code_exec)
    for m in msgs:
        print(m)
    if not conditions:
        print("No conditions left to run.")
        return {"records": [], "summary": {}, "skipped": msgs}
    tasks = resolve_tasks(args.tasks)
    if args.fake:
        client: Any = FakeClient()
    else:
        from pyursa.providers import llm_client
        client = llm_client(args.llm)
    model = getattr(client, "model_id", args.llm or "unknown")
    print(f"Model: {model} | conditions: {', '.join(conditions)} | tasks: {len(tasks)} | trials: {args.trials} "
          f"| rows: {args.rows} | max turns: {args.max_turns}")
    records = run_benchmark(client, tasks, conditions, trials=args.trials, rows=args.rows,
                            max_turns=args.max_turns, seed=args.seed, keep_transcripts=bool(args.out))
    print(format_report(records))
    summary = summarize(records)
    result = {"model": model, "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "args": vars(args), "summary": summary, "skipped": msgs, "records": [asdict(r) for r in records]}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {len(records)} run records to {args.out}")
    return result


if __name__ == "__main__":
    main()
