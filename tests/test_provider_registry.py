"""Every provider in the registry builds a valid client and sends a correctly shaped request."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pyursa as pu
from pyursa import providers as P

ALL = sorted(P.PROVIDERS)


class Echo(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Echo.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        out = {"choices": [{"message": {"content": "{\"results\": []}"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


@pytest.fixture(scope="module")
def echo():
    httpd = HTTPServer(("127.0.0.1", 0), Echo)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def configure(monkeypatch, name):
    p = P.PROVIDERS[name]
    for var in p.key_vars:
        monkeypatch.delenv(var, raising=False)
    if p.key_env:
        monkeypatch.setenv(p.key_env, f"key-for-{name}")
    for _, env in p.params:
        monkeypatch.setenv(env, "my-value")
    if name == "azure":
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    if name == "vertex":
        return {"api_key": "ya29.token"}
    return {}


@pytest.mark.parametrize("name", ALL)
def test_every_provider_builds_and_sends_a_valid_request(name, monkeypatch, echo):
    kw = configure(monkeypatch, name)
    model = "openai.gpt-oss-120b-1:0" if name == "bedrock-openai" else "some-model"
    c = pu.llm_client(f"{name}:{model}", **kw)
    assert "{" not in c.base_url and c.base_url.startswith("http"), c.base_url
    expected_url = c.base_url
    p = P.PROVIDERS[name]
    c.base_url = echo + "/v1"  # send the exact same request to a local server
    Echo.seen.clear()
    c.complete("system prompt", "user prompt")
    req = Echo.seen[0]
    assert req["path"].startswith("/v1/chat/completions")
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]
    assert req["body"][p.token_param or "max_tokens"] == 4096 or not p.token_param
    other = "max_tokens" if p.token_param == "max_completion_tokens" else "max_completion_tokens"
    assert other not in req["body"]  # never both (Cerebras and Fireworks reject that)
    key = kw.get("api_key") or (f"key-for-{name}" if p.key_env else None)
    if p.auth.startswith("header:"):
        assert req["headers"][p.auth.split(":", 1)[1]] == key and "authorization" not in req["headers"]
    elif p.auth == "none" or not key:
        assert "authorization" not in req["headers"]
    else:
        assert req["headers"]["authorization"] == f"Bearer {key}"
    assert expected_url.startswith("https://") or p.key_optional or p.auth == "none", expected_url


@pytest.mark.parametrize("name", [n for n in ALL if P.PROVIDERS[n].key_env and not P.PROVIDERS[n].key_optional])
def test_missing_key_message_names_the_variable(name, monkeypatch):
    p = P.PROVIDERS[name]
    for var in (*p.key_vars, "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    for _, env in p.params:
        monkeypatch.setenv(env, "v")
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client(f"{name}:m")
    assert p.key_env in str(ei.value)


def test_every_provider_documents_a_source():
    for name, p in P.PROVIDERS.items():
        assert p.docs.startswith("https://"), name


def test_url_examples(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_RESOURCE", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://contoso.openai.azure.com/")
    assert pu.llm_client("azure:my-deployment").base_url == "https://contoso.openai.azure.com/openai/v1"
    assert pu.llm_client("azure:d", resource="fabrikam").base_url == "https://fabrikam.openai.azure.com/openai/v1"

    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi")
    monkeypatch.setenv("DATABRICKS_HOST", "https://adb-123.4.azuredatabricks.net/")
    assert pu.llm_client("databricks:databricks-gpt-oss-120b").base_url == \
        "https://adb-123.4.azuredatabricks.net/ai-gateway/mlflow/v1"

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    v = pu.llm_client("vertex:gemini-3.8-flash", api_key="tok")
    assert v.base_url == "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/global/endpoints/openapi"
    assert v.model == "google/gemini-3.8-flash"
    v2 = pu.llm_client("vertex:google/gemini-3.8-flash", api_key="tok", location="us-central1")
    assert v2.base_url.startswith("https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/")

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
    assert pu.llm_client("cloudflare:@cf/meta/llama-3.1-8b-instruct").base_url == \
        "https://api.cloudflare.com/client/v4/accounts/acc123/ai/v1"
    assert pu.llm_client("cloudflare-gateway:openai/gpt-x").base_url == \
        "https://gateway.ai.cloudflare.com/v1/acc123/default/compat"

    monkeypatch.setenv("SNOWFLAKE_PAT", "pat")
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "myorg-acct")
    sf = pu.llm_client("snowflake:claude-sonnet-4-6")
    assert sf.base_url == "https://myorg-acct.snowflakecomputing.com/api/v2/cortex/v1"
    assert sf.token_param == "max_completion_tokens"

    monkeypatch.setenv("OCI_GENAI_API_KEY", "o")
    monkeypatch.delenv("OCI_REGION", raising=False)
    assert pu.llm_client("oci:xai.grok-3").base_url == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "d")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-9")
    monkeypatch.delenv("DASHSCOPE_REGION", raising=False)
    assert pu.llm_client("dashscope-workspace:qwen-plus").base_url == \
        "https://ws-9.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


def test_gemini_accepts_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    assert pu.llm_client("gemini:gemini-3.8-flash").api_key == "g"


def test_vertex_without_token_or_google_auth_explains(monkeypatch):
    import builtins
    real = builtins.__import__

    def block(name, *a, **k):
        if name.startswith("google"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client("vertex:gemini-3.8-flash")
    assert "gcloud auth" in str(ei.value.to_dict())


def test_vertex_refreshes_google_tokens(monkeypatch, echo):
    calls = []

    def fake_source():
        def token():
            calls.append(1)
            return f"tok{len(calls)}"
        return token

    monkeypatch.setattr(P, "_google_token_source", fake_source)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    c = pu.llm_client("vertex:gemini-3.8-flash")
    c.base_url = echo + "/v1"
    Echo.seen.clear()
    c.complete("s", "u")
    c.complete("s", "u")
    assert [r["headers"]["authorization"] for r in Echo.seen] == ["Bearer tok1", "Bearer tok2"]


@pytest.mark.parametrize("name", ["github", "llama", "hyperbolic", "lambda", "writer", "watsonx", "kong"])
def test_retired_or_incompatible_services_explain(name):
    with pytest.raises(pu.LLMNotConfiguredError) as ei:
        pu.llm_client(f"{name}:m")
    assert "isn't available" in str(ei.value)


def test_print_providers(capsys):
    pu.print_providers()
    out = capsys.readouterr().out
    assert "bedrock" in out and "azure" in out and "AZURE_OPENAI_RESOURCE" in out
