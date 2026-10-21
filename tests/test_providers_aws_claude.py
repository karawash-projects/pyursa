"""Claude and Amazon Nova integration paths, tested against local servers that speak the
real wire formats (OpenAI chat completions and Bedrock Converse)."""

import builtins
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pyursa as pu
from pyursa import semantic as S
from pyursa.providers import BedrockClient, OpenAICompatibleClient


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    saved = S.CONFIG
    S.CONFIG = S.LLMConfig()
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_API_KEY",
                "ANTHROPIC_WORKSPACE_ID", "LITELLM_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                "AWS_SESSION_TOKEN", "AWS_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    yield
    S.CONFIG = saved


def _answer(prompt: str) -> dict:
    items = json.loads(re.search(r"<data>\n(.*)\n</data>", prompt, re.S).group(1))
    field = re.search(r'Output fields and types: \{"(\w+)"', prompt).group(1)
    def value(text):
        late = "late" in text
        return late if field == "keep" else ("negative" if late else "positive")
    return {"results": [{"id": it["id"], field: value(it["data"]["t"])} for it in items]}


class Converse(BaseHTTPRequestHandler):
    """Speaks Bedrock's Converse REST API."""

    state: dict = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["requests"].append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        if self.state["fail"]:
            status, msg = self.state["fail"].pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(json.dumps({"message": msg}).encode())
            return
        prompt = body["messages"][0]["content"][0]["text"]
        reply = {"output": {"message": {"role": "assistant", "content": [{"text": json.dumps(_answer(prompt))}]}},
                 "stopReason": "end_turn", "usage": {"inputTokens": 321, "outputTokens": 12, "totalTokens": 333}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())


@pytest.fixture
def converse():
    Converse.state = {"requests": [], "fail": []}
    httpd = HTTPServer(("127.0.0.1", 0), Converse)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", Converse.state
    httpd.shutdown()


DATA = {"t": ["arrived late", "love it", "late again", "love it"]}


@pytest.mark.parametrize("model", ["us.amazon.nova-2-lite-v1:0", "us.anthropic.claude-haiku-5-5"])
def test_bedrock_converse_with_api_key_no_sdk(converse, monkeypatch, model, backend):
    url, state = converse
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-key")
    pu.set_llm(BedrockClient(model, endpoint_url=url))
    out = pu.from_dict(DATA).llm_map("sentiment", output="s").group_by("s").agg(n=pu.len()).to_dicts(backend)
    assert out == [{"s": "negative", "n": 2}, {"s": "positive", "n": 2}]
    req = state["requests"][0]
    assert req["path"] == "/model/" + model.replace(":", "%3A") + "/converse"
    assert req["auth"] == "Bearer bedrock-key"
    assert req["body"]["system"][0]["text"].startswith("You are a data-processing function")
    assert req["body"]["inferenceConfig"]["maxTokens"] == 4096
    rep = pu.llm_reports()[0]
    assert rep["model"] == f"bedrock:{model}" and rep["input_tokens"] == 321 and rep["tokens_estimated"] is False


def test_bedrock_default_endpoint_uses_region(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    monkeypatch.setenv("AWS_REGION", "ca-central-1")
    c = pu.llm_client("bedrock:us.amazon.nova-2-lite-v1:0")
    assert isinstance(c, BedrockClient) and c.endpoint_url == "https://bedrock-runtime.ca-central-1.amazonaws.com"
    assert BedrockClient("m", api_key="k").endpoint_url == "https://bedrock-runtime.ca-central-1.amazonaws.com"
    assert BedrockClient("m", api_key="k", region="eu-west-1").region == "eu-west-1"


def test_bedrock_retries_throttling_then_succeeds(converse, monkeypatch):
    url, state = converse
    state["fail"] = [(429, "Too many requests"), (503, "ServiceUnavailable")]
    pu.set_llm(BedrockClient("us.amazon.nova-2-lite-v1:0", endpoint_url=url, api_key="k"))
    assert pu.from_dict(DATA).llm_map("x", output="s").count() == 4
    assert len(state["requests"]) == 3


def test_bedrock_inference_profile_hint_and_no_key_leak(converse):
    url, state = converse
    state["fail"] = [(400, "Invocation of model ID amazon.nova-2-lite-v1:0 with on-demand throughput isn't supported. "
                           "Retry your request with the ID or ARN of an inference profile")]
    pu.set_llm(BedrockClient("amazon.nova-2-lite-v1:0", endpoint_url=url, api_key="SECRET-KEY"))
    with pytest.raises(pu.LLMProviderError) as ei:
        pu.from_dict(DATA).llm_map("x", output="s").collect()
    text = ei.value.to_json()
    assert "inference profile" in text and "us.amazon.nova-2-lite-v1:0" in text and "SECRET-KEY" not in text
    assert ei.value.details["status"] == 400 and len(state["requests"]) == 1  # 400 is not retried


def test_bedrock_without_key_or_boto3_explains_both_options(monkeypatch):
    real_import = builtins.__import__

    def no_boto3(name, *a, **k):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_boto3)
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("bedrock:us.amazon.nova-2-lite-v1:0")
    text = json.dumps(ei.value.to_dict())
    assert "AWS_BEARER_TOKEN_BEDROCK" in text and "boto3" in text and "$env:" in text


def test_bedrock_iam_path_with_boto3_style_client():
    sent = {}

    def converse(**kw):
        sent.update(kw)
        return {"output": {"message": {"content": [{"text": '{"results": [{"id": 0, "keep": true}]}'}]}},
                "usage": {"inputTokens": 5, "outputTokens": 2}}

    pu.set_llm(BedrockClient("us.amazon.nova-micro-v1:0", client=type("C", (), {"converse": staticmethod(converse)})()))
    assert pu.from_dict({"t": ["hi"]}).llm_filter("is a greeting").count() == 1
    assert sent["modelId"] == "us.amazon.nova-micro-v1:0" and sent["inferenceConfig"] == {"maxTokens": 4096}


# --- Claude through OpenAI-compatible endpoints -------------------------------------------

def test_claude_openai_compatible_preset(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_1")
    c = pu.llm_client("claude:claude-haiku-5-5")
    assert isinstance(c, OpenAICompatibleClient)
    assert c.base_url == "https://api.anthropic.com/v1" and c.api_key == "sk-ant"
    assert c.headers["anthropic-workspace-id"] == "wrkspc_1" and c.model_id == "claude:claude-haiku-5-5"


def test_anthropic_falls_back_to_openai_compatible_without_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")

    def no_sdk(*a, **k):
        raise ImportError("AnthropicClient needs `pip install anthropic`.")

    monkeypatch.setattr("pyursa.providers.AnthropicClient", no_sdk)
    c = pu.llm_client("anthropic:claude-haiku-5-5")
    assert isinstance(c, OpenAICompatibleClient) and c.base_url == "https://api.anthropic.com/v1"


class OpenAIish(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["requests"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
        reply = {"choices": [{"message": {"content": json.dumps(_answer(body["messages"][1]["content"]))}}],
                 "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "amazon/nova-lite-v1", "anthropic/claude-haiku-5-5"])
def test_any_model_through_an_openai_compatible_gateway(model):
    """Claude direct (Anthropic's compat layer) and Nova/Claude via gateways all use the same wire format."""
    OpenAIish.state = {"requests": []}
    httpd = HTTPServer(("127.0.0.1", 0), OpenAIish)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        pu.set_llm(OpenAICompatibleClient(model, base_url=f"http://127.0.0.1:{httpd.server_port}/v1", api_key="k",
                                          headers={"anthropic-workspace-id": "w"}))
        assert pu.from_dict(DATA).llm_filter("is negative").count() == 2
        req = OpenAIish.state["requests"][0]
        assert req["path"] == "/v1/chat/completions" and req["body"]["model"] == model
        assert req["headers"]["anthropic-workspace-id"] == "w" and "temperature" not in req["body"]
        assert "response_format" not in req["body"]  # ignored by Claude's compat layer; prompts don't rely on it
    finally:
        httpd.shutdown()


# --- presets and guards -----------------------------------------------------------------------

def test_bedrock_openai_endpoint_preset_and_guard(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "k")
    c = pu.llm_client("bedrock-openai:openai.gpt-oss-120b-1:0", region="us-west-2")
    assert c.base_url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"
    for model in ("us.amazon.nova-2-lite-v1:0", "us.anthropic.claude-haiku-5-5", "meta.llama3-70b-instruct-v1:0"):
        with pytest.raises(pu.LLMNotConfiguredError) as ei:
            pu.llm_client(f"bedrock-openai:{model}")
        assert ei.value.suggestions[0]["fix"] == f'pu.llm_client("bedrock:{model}")'


def test_litellm_gateway_key_is_optional(monkeypatch):
    c = pu.llm_client("litellm:bedrock-nova-lite")
    assert c.base_url == "http://localhost:4000/v1" and c.api_key is None
    monkeypatch.setenv("LITELLM_API_KEY", "sk-gw")
    assert pu.llm_client("litellm:bedrock-nova-lite").api_key == "sk-gw"


def test_list_providers_covers_claude_and_nova_paths():
    rows = {r["provider"]: r for r in pu.list_providers()}
    for name in ("anthropic", "claude", "bedrock", "bedrock-openai", "openrouter", "litellm", "openai", "gemini"):
        assert name in rows
    assert "Nova" in rows["bedrock"]["about"] and "AWS_BEARER_TOKEN_BEDROCK" in rows["bedrock"]["about"]
    assert "NOT serve Nova" in rows["bedrock-openai"]["about"]


def test_spec_without_colon_is_rejected():
    with pytest.raises(pu.LLMNotConfiguredError):
        pu.llm_client("claude-haiku-5-5")
