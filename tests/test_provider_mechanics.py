"""Generic mechanics shared by every OpenAI-compatible provider: auth styles, URL placeholders,
alternative key variables, query parameters, token-limit field, and the catch-all provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pyursa as pu
from pyursa import providers as P
from pyursa.providers import OpenAICompatibleClient, Provider


class Server(BaseHTTPRequestHandler):
    state: dict = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.state["requests"].append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()},
                                       "body": body})
        reject = self.state.get("reject_field")
        if reject and reject in body:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            msg = (f"Unsupported parameter: '{reject}' is not supported with this model. "
                   f"Use '{self.state['want_field']}' instead.")
            self.wfile.write(json.dumps({"error": {"message": msg}}).encode())
            return
        reply = {"choices": [{"message": {"content": json.dumps({"results": [{"id": 0, "keep": True}]})}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())


@pytest.fixture
def server():
    Server.state = {"requests": []}
    httpd = HTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}", Server.state
    httpd.shutdown()


@pytest.fixture
def preset(monkeypatch):
    def add(name, **kw):
        monkeypatch.setitem(P.PROVIDERS, name, Provider(**kw))
    return add


def ask(client):
    return client.complete("system text", "user text")


def test_placeholders_from_kwargs_then_env_then_defaults(preset, monkeypatch, server):
    url, state = server
    preset("acme", base_url=url + "/acct/{account_id}/{zone}/v1", key_env="ACME_KEY",
           params=(("account_id", "ACME_ACCOUNT"), ("zone", "ACME_ZONE")), defaults=(("zone", "main"),))
    monkeypatch.setenv("ACME_KEY", "k")
    monkeypatch.setenv("ACME_ACCOUNT", "env-acct")
    ask(pu.llm_client("acme:m"))
    ask(pu.llm_client("acme:m", account_id="kw acct/1", zone="z2"))
    paths = [r["path"] for r in state["requests"]]
    assert paths == ["/acct/env-acct/main/v1/chat/completions", "/acct/kw%20acct%2F1/z2/v1/chat/completions"]


def test_missing_placeholder_names_the_env_var(preset, monkeypatch):
    preset("acme", base_url="https://x/{account_id}/v1", key_env=None, params=(("account_id", "ACME_ACCOUNT"),))
    monkeypatch.delenv("ACME_ACCOUNT", raising=False)
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("acme:m")
    d = ei.value.to_dict()
    assert d["details"]["missing"] == ["account_id"] and "ACME_ACCOUNT" in json.dumps(d)


def test_api_key_header_auth_and_query(preset, monkeypatch, server):
    url, state = server
    preset("azureish", base_url=url + "/openai/v1", key_env="AZ_KEY", auth="header:api-key",
           query=(("api-version", "preview"),))
    monkeypatch.setenv("AZ_KEY", "secret")
    ask(pu.llm_client("azureish:deploy"))
    req = state["requests"][0]
    assert req["path"] == "/openai/v1/chat/completions?api-version=preview"
    assert req["headers"]["api-key"] == "secret" and "authorization" not in req["headers"]


def test_alternative_key_variables(preset, monkeypatch):
    preset("acme", base_url="https://x/v1", key_env="ACME_KEY", alt_key_envs=("ACME_TOKEN",))
    monkeypatch.delenv("ACME_KEY", raising=False)
    monkeypatch.setenv("ACME_TOKEN", "t")
    assert pu.llm_client("acme:m").api_key == "t"
    monkeypatch.delenv("ACME_TOKEN")
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("acme:m")
    assert "ACME_TOKEN" in str(ei.value)


def test_no_auth_provider_sends_no_header(preset, server):
    url, state = server
    preset("local", base_url=url + "/v1", key_env=None, auth="none")
    ask(pu.llm_client("local:m", api_key="ignored"))
    assert "authorization" not in state["requests"][0]["headers"]


@pytest.mark.parametrize("start, reject, want", [("max_tokens", "max_tokens", "max_completion_tokens"),
                                                  ("max_completion_tokens", "max_completion_tokens", "max_tokens")])
def test_token_limit_field_is_negotiated(server, start, reject, want):
    url, state = server
    state["reject_field"], state["want_field"] = reject, want
    c = OpenAICompatibleClient("m", base_url=url + "/v1", token_param=start)
    if start == "max_completion_tokens":  # make the rejection message name the unknown field
        state["want_field"] = "max_tokens (unknown field max_completion_tokens)"
    ask(c)
    bodies = [r["body"] for r in state["requests"]]
    assert reject in bodies[0] and want in bodies[1] and reject not in bodies[1]
    ask(c)  # remembered: no extra failed request
    assert len(state["requests"]) == 3 and want in state["requests"][2]["body"]


def test_token_param_can_be_omitted(server):
    url, state = server
    ask(OpenAICompatibleClient("m", base_url=url + "/v1", token_param=""))
    body = state["requests"][0]["body"]
    assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_catch_all_openai_compatible(monkeypatch, server):
    url, state = server
    monkeypatch.setenv("OPENAI_BASE_URL", url + "/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-any")
    c = pu.llm_client("openai-compatible:my-model")
    assert c.model_id == "openai-compatible:my-model"
    ask(c)
    assert state["requests"][0]["headers"]["authorization"] == "Bearer sk-any"
    monkeypatch.delenv("OPENAI_BASE_URL")
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("openai-compatible:m")
    assert "OPENAI_BASE_URL" in str(ei.value.to_dict())


@pytest.mark.parametrize("body, expected", [
    ('[{"error": {"code": 400, "message": "API key not valid.", "status": "INVALID_ARGUMENT"}}]', "API key not valid."),
    ('{"error": {"message": "Incorrect API key provided"}}', "Incorrect API key provided"),
    ('{"message": "The security token included in the request is invalid."}', "The security token included"),
    ('{"detail": "Not authenticated"}', "Not authenticated"),
    ('{"errors": [{"code": 10000, "message": "Authentication error"}], "success": false}', "Authentication error"),
])
def test_provider_error_messages_are_extracted(server, body, expected):
    url, state = server

    class Err(Server):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(400)
            self.end_headers()
            self.wfile.write(body.encode())

    httpd = HTTPServer(("127.0.0.1", 0), Err)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        c = OpenAICompatibleClient("m", base_url=f"http://127.0.0.1:{httpd.server_port}/v1", api_key="k", max_attempts=1)
        with pytest.raises(pu.LLMProviderError) as ei:
            ask(c)
        assert expected in ei.value.message
        if "key" in expected.lower() or "authent" in expected.lower() or "token" in expected.lower():
            assert "key" in ei.value.suggestions[0]["description"].lower()
    finally:
        httpd.shutdown()
