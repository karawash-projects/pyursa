"""Tests for benchmarks/agent_bench.py, using the scripted fake client (no API key, no network)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "agent_bench.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("agent_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent_bench"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fake_records(bench):
    return bench.run_benchmark(bench.FakeClient(), bench.TASKS, ["pyursa", "pandas"],
                               trials=1, rows=2000, max_turns=10, log=lambda _: None)


def test_fake_client_solves_every_task_in_both_conditions(bench, fake_records):
    assert len(fake_records) == 2 * len(bench.TASKS)
    failed = [(r.task, r.condition, r.answer, r.truth, r.error) for r in fake_records if not r.correct]
    assert not failed
    for r in fake_records:
        assert r.finished and r.error_turns == 0 and 1 < r.turns <= 10
        assert r.approx_tokens_in > 0 and r.approx_tokens_out > 0
    summary = bench.summarize(fake_records)
    assert summary["pyursa"]["success_rate"] == 1.0
    assert summary["pandas"]["success_rate"] == 1.0
    assert "Summary per condition" in bench.format_report(fake_records)


def test_data_generation_is_deterministic(bench, tmp_path):
    a = bench.ground_truth(bench.generate_data(tmp_path / "a", rows=2000, seed=0))
    b = bench.ground_truth(bench.generate_data(tmp_path / "b", rows=2000, seed=0))
    assert a == b
    assert set(a) == {t.id for t in bench.TASKS}


@pytest.mark.parametrize("text", [
    '{"action": "answer", "value": 3}',
    '```json\n{"action": "answer", "value": 3}\n```',
    'Sure, here it is:\n```\n{"action": "answer", "value": 3}\n```\nDone.',
    'I will answer now {"action": "answer", "value": 3} thanks',
])
def test_parse_reply_accepts_json_bare_fenced_or_embedded(bench, text):
    assert bench.parse_reply(text) == {"action": "answer", "value": 3}


def test_parse_reply_nested_args(bench):
    text = '```json\n{"action": "transform", "args": {"handle": "f1", "steps": [{"op": "head", "n": 2}]}}\n```'
    assert bench.parse_reply(text)["args"]["steps"] == [{"op": "head", "n": 2}]


@pytest.mark.parametrize("text", ["", "hello", "{not json}", '{"value": 3}', "```json\n[1, 2]\n```"])
def test_parse_reply_rejects_garbage(bench, text):
    assert bench.parse_reply(text) is None


class _GarbageThenAnswer:
    model_id = "fake:garbage"

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def complete(self, system, user):
        self.calls += 1
        return "let me think..." if self.calls == 1 else json.dumps({"action": "answer", "value": self.answer})


def test_garbage_reply_is_an_error_turn_with_corrective_message(bench, tmp_path):
    task = bench.TASKS_BY_ID["t7"]
    rec = bench.run_episode(_GarbageThenAnswer(42), task, "pyursa", tmp_path, truth=42, keep_transcript=True)
    assert rec.turns == 2 and rec.error_turns == 1 and rec.correct
    assert rec.transcript[0]["result"] == bench.CORRECTIVE


def test_max_turns_stops_unfinished_run(bench, tmp_path):
    class Babbler:
        def complete(self, system, user):
            return "no json here"
    rec = bench.run_episode(Babbler(), bench.TASKS[0], "pyursa", tmp_path, truth=1.0, max_turns=3)
    assert rec.turns == 3 and rec.error_turns == 3 and not rec.finished and not rec.correct


def test_tool_errors_count_as_error_turns(bench, tmp_path):
    env = bench.PyursaEnv(tmp_path)
    obs, err = env.step({"action": "preview", "args": {"handle": "nope"}})
    assert err and "HANDLE_NOT_FOUND" in obs
    obs, err = env.step({"action": "rm_rf", "args": {}})
    assert err and "Unknown action" in obs


def test_scoring_tolerance(bench):
    s = bench.score
    assert s(1000.5, 1000.0, "float")          # within relative 1e-3
    assert not s(1002.0, 1000.0, "float")
    assert s(0.009, 0.0, "float")              # within absolute 0.01
    assert not s(0.02, 0.0, "float")
    assert s("4.87%", 4.87, "float") and s("1,234.50", 1234.5, "float")
    assert not s(None, 1.0, "float") and not s("abc", 1.0, "float") and not s(True, 1.0, "float")
    assert s(12, 12, "int") and s(12.0, 12, "int") and not s(13, 12, "int")
    assert s("  East ", "east", "str") and s(["east"], "east", "str") and not s("west", "east", "str")
    assert s([3, 1, 2], [3, 1, 2], "list") and s("[3, 1, 2]", [3, 1, 2], "list")
    assert not s([1, 2, 3], [3, 1, 2], "list")  # order matters
    assert s(["South", "north"], ["south", "north"], "list")
    assert not s([3, 1], [3, 1, 2], "list")


def test_pandas_condition_skipped_without_allow_code_exec(bench, capsys):
    conds, msgs = bench.resolve_conditions(["pyursa", "pandas"], allow_code_exec=False)
    assert conds == ["pyursa"] and msgs == [bench.CODE_EXEC_SKIP_MSG]
    assert bench.resolve_conditions(["pyursa", "pandas"], allow_code_exec=True) == (["pyursa", "pandas"], [])

    result = bench.main(["--fake", "--conditions", "pandas", "--rows", "500"])
    out = capsys.readouterr().out
    assert "--allow-code-exec" in out and result["records"] == []


def test_cli_fake_run_writes_results(bench, tmp_path, capsys):
    out_file = tmp_path / "results.json"
    result = bench.main(["--fake", "--tasks", "t1,t6", "--conditions", "pyursa", "--rows", "1000",
                         "--out", str(out_file)])
    assert result["summary"]["pyursa"]["success_rate"] == 1.0
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert [r["task"] for r in saved["records"]] == ["t1", "t6"]
    assert saved["records"][0]["transcript"]


def test_resolve_tasks(bench):
    assert len(bench.resolve_tasks("all")) == len(bench.TASKS)
    assert [t.id for t in bench.resolve_tasks("t3, t1")] == ["t3", "t1"]
    with pytest.raises(ValueError):
        bench.resolve_tasks("t99")
