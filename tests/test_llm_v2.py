import datetime as dt
import json
import re
import threading

import pytest

import pyursa as pu
from pyursa import semantic as S
from pyursa.redact import redact_text


@pytest.fixture(autouse=True)
def fresh():
    saved = S.CONFIG
    S.CONFIG = S.LLMConfig()
    yield
    S.CONFIG = saved


class TextModel(pu.PromptClient):
    """Fake text model that reports real-looking token usage."""

    model_id = "fake"

    def __init__(self, answer=lambda d: "positive"):
        self.answer, self.prompts = answer, []

    def complete_with_usage(self, system, user):
        self.prompts.append(user)
        items = json.loads(re.search(r"<data>\n(.*)\n</data>", user, re.S).group(1))
        field = re.search(r'Output fields and types: \{"(\w+)"', user).group(1)
        out = {"results": [{"id": it["id"], field: self.answer(it["data"])} for it in items]}
        return json.dumps(out), {"input_tokens": 100, "output_tokens": 10}

    def complete(self, system, user):
        return self.complete_with_usage(system, user)[0]


def test_token_usage_and_cost():
    pu.set_llm(TextModel(), batch_size=2, price_per_mtok=(1.0, 5.0))
    pu.from_dict({"t": ["a", "b", "c"]}).llm_map("x", output="y").collect()
    r = pu.llm_reports()[0]
    assert r["llm_calls"] == 2 and r["input_tokens"] == 200 and r["output_tokens"] == 20
    assert r["tokens_estimated"] is False and r["cost_usd"] == pytest.approx(200 / 1e6 + 20 * 5 / 1e6)


def test_tokens_are_estimated_when_provider_gives_none():
    class Plain(TextModel):
        def complete_with_usage(self, system, user):
            return super().complete_with_usage(system, user)[0], None
    pu.set_llm(Plain())
    pu.from_dict({"t": ["a"]}).llm_map("x", output="y").collect()
    r = pu.llm_reports()[0]
    assert r["tokens_estimated"] is True and r["input_tokens"] > 0


def test_token_budget_refuses_before_calling():
    m = TextModel()
    pu.set_llm(m, max_input_tokens=50)
    with pytest.raises(pu.LLMBudgetError) as ei:
        pu.from_dict({"t": ["x" * 400]}).llm_map("x", output="y").collect()
    assert m.prompts == [] and ei.value.details["estimated_input_tokens"] > 50


def test_redaction_masks_pii_before_sending():
    m = TextModel()
    pu.set_llm(m, redact_pii=True)
    df = pu.from_dict({"t": ["mail ana@example.com or call +1 416-555-0199, card 4111 1111 1111 1111, ssn 123-45-6789"]})
    df.llm_map("x", output="y").collect()
    sent = m.prompts[0]
    for secret in ("ana@example.com", "416-555-0199", "4111 1111 1111 1111", "123-45-6789"):
        assert secret not in sent
    assert "[EMAIL]" in sent and "[CARD]" in sent and "[SSN]" in sent and "[PHONE]" in sent
    assert pu.llm_reports()[0]["redacted"] is True


@pytest.mark.parametrize("text, expected", [
    ("order 12345 shipped 2026-01-05", "order 12345 shipped 2026-01-05"),   # ids and dates survive
    ("price 1,234.50", "price 1,234.50"),
    ("4111 1111 1111 1112 is not a valid card", "4111 1111 1111 1112 is not a valid card"),  # Luhn fails
    ("ip 10.0.0.1", "ip [IP]"),
])
def test_redaction_precision(text, expected):
    assert redact_text(text) == expected


def test_reports_are_per_thread():
    pu.set_llm(pu.FunctionClient(lambda r: "ok"))
    seen = {}

    def worker(name, n):
        pu.from_dict({"t": [str(i) for i in range(n)]}).llm_map("x", output="y").collect()
        seen[name] = pu.llm_reports()[0]["rows"]

    threads = [threading.Thread(target=worker, args=(f"t{n}", n)) for n in (3, 7)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"t3": 3, "t7": 7}


def test_persistent_cache_restores_typed_values(tmp_path):
    path = str(tmp_path / "c.sqlite")
    pu.set_llm(pu.FunctionClient(lambda r: "2026-03-04", model_id="stub"), cache_path=path)
    df = pu.from_dict({"t": ["a"]})
    df.llm_map("date", output="d", dtype="Date").collect()
    pu.set_llm(pu.FunctionClient(lambda r: (_ for _ in ()).throw(AssertionError("should be cached")), model_id="stub"),
               cache_path=path)
    assert df.llm_map("date", output="d", dtype="Date").to_dicts()[0]["d"] == dt.date(2026, 3, 4)


def test_decimal_llm_output():
    import decimal
    pu.set_llm(pu.FunctionClient(lambda r: "19.99"))
    out = pu.from_dict({"t": ["a"]}).llm_map("price", output="p", dtype="Decimal(10,2)").to_dicts()
    assert out[0]["p"] == decimal.Decimal("19.99")


def test_tool_sessions_have_isolated_llm_settings(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pyursa.tools import ToolSession
    pq.write_table(pa.table({"t": ["a", "b"]}), tmp_path / "r.parquet")
    calls_a, calls_b = [], []
    a = ToolSession(roots=[tmp_path], llm=pu.FunctionClient(lambda r: calls_a.append(r) or "A"))
    b = ToolSession(roots=[tmp_path], llm=pu.FunctionClient(lambda r: calls_b.append(r) or "B"))
    for s, label in ((a, "A"), (b, "B")):
        h = s.load_file("r.parquet")["handle"]
        t = s.transform(h, [{"op": "llm_map", "instruction": "x", "output": "y"}])
        assert {r["y"] for r in s.preview(t["handle"])["rows"]} == {label}
    assert len(calls_a) == 2 and len(calls_b) == 2  # separate caches, separate clients
    assert a.llm_usage()["reports"][0]["model"].startswith("function:")
    assert S.CONFIG.client is None  # the global default was never touched


def test_tool_session_redacts_previews_and_summaries(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pyursa.tools import ToolSession
    pq.write_table(pa.table({"email": ["ana@example.com", "bo@example.org"]}), tmp_path / "p.parquet")
    s = ToolSession(roots=[tmp_path], redact_pii=True)
    h = s.load_file("p.parquet")["handle"]
    assert "example.com" not in json.dumps(s.preview(h)) and "[EMAIL]" in s.preview(h)["table"]
    assert "example.com" not in s.describe(h, tokens=2000)["summary"]


def test_different_functions_never_share_cache():
    pu.set_llm(pu.FunctionClient(lambda r: 5))
    pu.from_dict({"t": ["a"]}).llm_map("x", output="y", dtype="Int64").collect()
    pu.set_llm(pu.FunctionClient(lambda r: 7))
    assert pu.from_dict({"t": ["a"]}).llm_map("x", output="y", dtype="Int64").to_dicts()[0]["y"] == 7
