"""Provider tests against a real local HTTP server speaking the OpenAI chat API."""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

import pyursa as pu
from pyursa import semantic as S
from pyursa.providers import OpenAICompatibleClient


@pytest.fixture(autouse=True)
def fresh_llm_config():
    saved = (S.CONFIG.client, S.CONFIG.cache)
    S.CONFIG.cache = S.LLMCache()
    yield
    S.CONFIG.client, S.CONFIG.cache = saved


class FakeOpenAI(BaseHTTPRequestHandler):
    """Answers sentiment tasks; can be told to fail N times first."""

    state = {"requests": [], "fail_first": 0, "status": 429}

    def log_message(self, *a):  # keep test output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.state["fail_first"] > 0:
            self.state["fail_first"] -= 1
            self.send_response(self.state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(json.dumps({"error": {"message": "slow down"}}).encode())
            return
        user = body["messages"][1]["content"]
        items = json.loads(re.search(r"<data>\n(.*)\n</data>", user, re.S).group(1))
        field = re.search(r'Output fields and types: \{"(\w+)"', user).group(1)
        results = [{"id": it["id"], field: "negative" if "late" in it["data"]["text"].lower() else "positive"}
                   for it in items]
        reply = {"choices": [{"message": {"role": "assistant", "content": json.dumps({"results": results})}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())


@pytest.fixture
def server():
    FakeOpenAI.state = {"requests": [], "fail_first": 0, "status": 429}
    httpd = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", FakeOpenAI.state
    httpd.shutdown()


@pytest.fixture
def reviews():
    return pu.from_dict({"text": ["Arrived late", "Love it", "late again", "Love it"]})


def test_openai_compatible_end_to_end(server, reviews, backend):
    url, state = server
    pu.set_llm(OpenAICompatibleClient("test-model", base_url=url, api_key="sk-test"))
    out = reviews.llm_map("sentiment", output="s").group_by("s").agg(n=pu.len()).to_dicts(backend)
    assert out == [{"s": "negative", "n": 2}, {"s": "positive", "n": 2}]
    req = state["requests"][0]
    assert req["path"] == "/v1/chat/completions" and req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "test-model"
    assert req["body"]["messages"][0]["role"] == "system" and "untrusted" in req["body"]["messages"][0]["content"]
    assert len(state["requests"]) == 1  # 4 rows, 3 unique, one batch


def test_retries_on_rate_limit(server, reviews):
    url, state = server
    state["fail_first"] = 2
    pu.set_llm(OpenAICompatibleClient("m", base_url=url))
    assert reviews.llm_map("sentiment", output="s").count() == 4
    assert len(state["requests"]) == 3
    assert state["requests"][0]["auth"] is None  # local servers need no key


def test_auth_error_is_structured_and_not_retried(server, reviews):
    url, state = server
    state["fail_first"], state["status"] = 5, 401
    pu.set_llm(OpenAICompatibleClient("m", base_url=url, api_key="bad"))
    with pytest.raises(pu.LLMProviderError) as ei:
        reviews.llm_map("sentiment", output="s").collect()
    d = ei.value.to_dict()
    assert d["details"]["status"] == 401 and "slow down" in d["message"]
    assert "bad" not in json.dumps(d)  # the key never leaks into errors
    assert len(state["requests"]) == 1


def test_unreachable_server(reviews):
    pu.set_llm(OpenAICompatibleClient("m", base_url="http://127.0.0.1:9/v1", max_attempts=1, timeout=2))
    with pytest.raises(pu.LLMProviderError):
        reviews.llm_map("x", output="y").collect()


def test_json_mode_flag(server, reviews):
    url, state = server
    pu.set_llm(OpenAICompatibleClient("m", base_url=url, json_mode=True))
    reviews.head(1).llm_map("sentiment", output="s").collect()
    assert state["requests"][0]["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("spec, base, env", [
    ("openai:gpt-test", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    ("gemini:gemini-3.8-flash", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    ("groq:llama-test", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    ("openrouter:amazon/nova-lite-v1", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
])
def test_presets_read_their_env_var(monkeypatch, spec, base, env):
    monkeypatch.setenv(env, "k")
    c = pu.llm_client(spec)
    assert c.base_url == base and c.api_key == "k" and c.model_id == spec


def test_missing_key_names_the_env_var(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("gemini:gemini-3.8-flash")
    assert "GEMINI_API_KEY" in str(ei.value)


def test_local_providers_need_no_key():
    c = pu.llm_client("ollama:llama3.2")
    assert c.base_url == "http://localhost:11434/v1" and c.api_key is None


def test_unknown_spec():
    with pytest.raises(pu.LLMNotConfiguredError):
        pu.llm_client("nope:model")


def test_bedrock_converse_for_nova():
    sent = {}

    def converse(**kw):
        sent.update(kw)
        return {"output": {"message": {"content": [{"text": '{"results": [{"id": 0, "keep": true}]}'}]}}}

    pu.set_llm(pu.llm_client("bedrock:amazon.nova-lite-v1:0", client=SimpleNamespace(converse=converse)))
    assert pu.from_dict({"t": ["hi"]}).llm_filter("is a greeting").count() == 1
    assert sent["modelId"] == "amazon.nova-lite-v1:0"  # model id keeps its own colon
    assert sent["messages"][0]["content"][0]["text"].startswith("Instruction:")


def test_different_models_do_not_share_cache(server, reviews):
    url, state = server
    pu.set_llm(OpenAICompatibleClient("model-a", base_url=url))
    reviews.llm_map("sentiment", output="s").collect()
    pu.set_llm(OpenAICompatibleClient("model-b", base_url=url))
    reviews.llm_map("sentiment", output="s").collect()
    assert len(state["requests"]) == 2
