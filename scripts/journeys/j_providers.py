"""Journey: real provider client code paths against local fake servers (no internet, no real keys).

Covers the OpenAI-compatible client (custom base_url and a preset provider pointed at a local URL),
token-limit field negotiation, usage accounting, rate limits (Google retryDelay, daily quota), bad keys
with Gemini's list-wrapped error bodies, redirects (the key must not follow), transient 500s, timeouts,
the Claude fallback, the native Anthropic wrapper, and Bedrock Converse with an API key and SigV4.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from _harness import Journey, Skip, check, equal

SECRET = "sk-journey-SECRET-123"


def _answer_prompt(prompt: str) -> str:
    """Answer the pyursa prompt like a model would: one result per item in the <data> block."""
    m = re.search(r"<data>\n(.*)\n</data>\s*$", prompt, re.S)
    f = re.search(r"Output fields and types: \{(.*?)\}\.", prompt)
    if not m or not f:  # not a pyursa prompt (a direct .complete() call)
        return "ok"
    items = json.loads(m.group(1))
    names = re.findall(r'"([^"]+)":', f.group(1))
    out = []
    for it in items:
        text = " ".join(str(v) for v in it["data"].values()).lower()
        row = {"id": it["id"]}
        for n in names:
            row[n] = "negative" if any(w in text for w in ("broke", "late", "bad")) else "positive"
        out.append(row)
    return json.dumps({"results": out})


class FakeServer:
    """A local HTTP server. `script` is a list of callables (handler -> True if it answered); each request
    consumes the first one; when the list is empty, `default` answers. Every request is recorded."""

    def __init__(self, default=None):
        self.requests: list[dict] = []
        self.script: list = []
        self.default = default
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                outer.requests.append({"method": self.command, "path": self.path,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body,
                                       "t": time.monotonic()})
                fn = outer.script.pop(0) if outer.script else outer.default
                if fn is None:
                    self.send_json(404, {"error": {"message": "no handler"}})
                else:
                    fn(self, body)

            def send_json(self, status, obj, headers=None):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            do_POST = _handle
            do_GET = _handle

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def openai_ok(h, body):
    prompt = body["messages"][-1]["content"]
    reply = _answer_prompt(prompt)
    h.send_json(200, {"choices": [{"message": {"role": "assistant", "content": reply}}],
                      "usage": {"prompt_tokens": 111, "completion_tokens": 22}})


def bedrock_ok(h, body):
    prompt = body["messages"][0]["content"][0]["text"]
    h.send_json(200, {"output": {"message": {"role": "assistant", "content": [{"text": _answer_prompt(prompt)}]}},
                      "usage": {"inputTokens": 77, "outputTokens": 9}, "stopReason": "end_turn"})


def status(code, obj, headers=None):
    return lambda h, body: h.send_json(code, obj, headers)


_ENV_PATTERN = re.compile(r"(API_KEY|_TOKEN|ACCESS_KEY|SECRET|^AWS_|BASE_URL|^GOOGLE_|^OPENAI_|^ANTHROPIC_)")


def run(j: Journey) -> None:
    import pyursa as pu
    from pyursa import semantic as S

    saved_env = {k: v for k, v in os.environ.items() if _ENV_PATTERN.search(k)}
    for k in saved_env:
        del os.environ[k]
    saved_config = S.CONFIG
    reviews = pu.from_dict({"id": [1, 2, 3, 4], "text": ["Great blender", "It broke after a day",
                                                        "Arrived late", "Great blender"]})
    api = FakeServer(default=openai_ok)
    other = FakeServer(default=openai_ok)
    try:
        def sentiment(client, frame=reviews):
            S.CONFIG = S.LLMConfig(client=client, batch_size=2)
            return frame.llm_map("sentiment", output="s", inputs=["text"]).to_dicts()

        with j.step("openai-compatible: full llm_map through a local server", critical=True):
            api.requests.clear()
            c = pu.llm_client("openai-compatible:local-model", base_url=api.url + "/v1", api_key=SECRET)
            rows = sentiment(c)
            equal([r["s"] for r in rows], ["positive", "negative", "negative", "positive"])
            equal(len(api.requests), 2, "3 unique rows / batch 2 -> 2 calls;")
            r0 = api.requests[0]
            equal(r0["path"], "/v1/chat/completions")
            equal(r0["headers"].get("authorization"), f"Bearer {SECRET}")
            equal(r0["body"]["model"], "local-model")
            equal([m["role"] for m in r0["body"]["messages"]], ["system", "user"])
            check("max_tokens" in r0["body"], str(r0["body"].keys()))

        with j.step("usage tokens reported by the server reach llm_reports"):
            rep = pu.llm_reports()[0]
            equal((rep["input_tokens"], rep["output_tokens"], rep["tokens_estimated"]), (222, 44, False), "tokens:")
            equal(rep["model"], "openai-compatible:local-model")

        with j.step("a preset provider (openai) pointed at a local base_url sends max_completion_tokens"):
            api.requests.clear()
            c = pu.llm_client("openai:gpt-local", base_url=api.url + "/v1", api_key=SECRET)
            sentiment(c, reviews.head(1))
            check("max_completion_tokens" in api.requests[0]["body"], str(api.requests[0]["body"].keys()))
            check("max_tokens" not in api.requests[0]["body"])
            equal(c.model_id, "openai:gpt-local")

        with j.step("token-limit field negotiation: a server that rejects max_tokens gets max_completion_tokens"):
            api.requests.clear()
            api.script = [status(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported "
                                                            "with this model. Use 'max_completion_tokens' instead.",
                                                 "type": "invalid_request_error"}})]
            c = pu.OpenAICompatibleClient("o-local", base_url=api.url + "/v1", api_key=SECRET, max_attempts=2)
            sentiment(c, reviews.head(1))
            equal(len(api.requests), 2)
            check("max_tokens" in api.requests[0]["body"] and "max_completion_tokens" in api.requests[1]["body"])
            api.requests.clear()
            sentiment(c, pu.from_dict({"id": [9], "text": ["bad"]}))
            check("max_completion_tokens" in api.requests[0]["body"], "the choice should be remembered")
            equal(len(api.requests), 1)

        with j.step("429 with Google-style retryDelay: waits as told, then succeeds"):
            api.requests.clear()
            body = [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                               "message": "You exceeded your current quota. Please retry in 1.2s.",
                               "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                                            "violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
                                           {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"}]}}]
            api.script = [status(429, body)]
            c = pu.llm_client("gemini:gemini-local", base_url=api.url + "/v1beta/openai", api_key=SECRET, max_attempts=3)
            rows = sentiment(c, reviews.head(1))
            equal(rows[0]["s"], "positive")
            equal(len(api.requests), 2)
            gap = api.requests[1]["t"] - api.requests[0]["t"]
            check(0.9 <= gap < 2.5, f"waited {gap:.2f}s, the server asked for 1s")

        with j.step("429 daily quota fails fast with a clear quota hint"):
            api.requests.clear()
            body = [{"error": {"code": 429, "message": "You exceeded your current quota, please check your plan and "
                                                       "billing details.",
                               "details": [{"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}]
            api.script = [status(429, body)] * 4
            c = pu.llm_client("gemini:gemini-local", base_url=api.url + "/v1beta/openai", api_key=SECRET, max_attempts=4)
            t = time.monotonic()
            try:
                sentiment(c, reviews.head(1))
                raise AssertionError("expected LLMProviderError")
            except pu.LLMProviderError as e:
                equal(e.details.get("status"), 429)
                check("exceeded your current quota" in e.message, e.message)
                check("not a pyursa problem" in e.suggestions[0]["description"], str(e.suggestions))
            check(time.monotonic() - t < 1.5, "should not retry a daily quota")
            equal(len(api.requests), 1, "requests sent:")
            api.script.clear()

        with j.step("401 with a list-wrapped Gemini error: real message and an API-key hint"):
            api.requests.clear()
            api.script = [status(400, [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                  "message": "API key not valid. Please pass a valid API key."}}]),
                          status(401, [{"error": {"code": 401, "status": "UNAUTHENTICATED",
                                                  "message": "API key not valid. Please pass a valid API key."}}])]
            for expect_status in (400, 401):
                c = pu.llm_client("gemini:gemini-local", base_url=api.url + "/v1beta/openai", api_key="bad", max_attempts=1)
                try:
                    c.complete("s", "hi")
                    raise AssertionError("expected LLMProviderError")
                except pu.LLMProviderError as e:
                    equal(e.details.get("status"), expect_status)
                    check("API key not valid" in e.message, e.message)
                    check("Bad Request" not in e.message and "Unauthorized" not in e.message, e.message)
                    hint = e.suggestions[0]["description"]
                    check("API key" in hint and "GEMINI_API_KEY" in hint, hint)
                    check("bad" not in json.dumps(e.to_dict()).replace("Bad", ""), "the key must not appear in errors")

        with j.step("302 redirect: the key is not sent to the new host and a structured error comes back"):
            api.requests.clear()
            other.requests.clear()
            api.script = [lambda h, b: (h.send_response(302), h.send_header("Location", other.url + "/steal"),
                                        h.send_header("Content-Length", "0"), h.end_headers())]
            c = pu.OpenAICompatibleClient("m", base_url=api.url + "/v1", api_key=SECRET, max_attempts=1)
            S.CONFIG = S.LLMConfig(client=c)
            try:
                reviews.head(1).llm_map("sentiment", output="s", inputs=["text"]).collect()
                raise AssertionError("expected LLMProviderError")
            except pu.LLMProviderError as e:
                equal(e.details.get("status"), 302)
                check("redirect" in e.message.lower(), e.message)
            equal(other.requests, [], "the redirect target was contacted:")
            check(SECRET not in json.dumps(other.requests))

        with j.step("transient 500 is retried, then the step succeeds"):
            api.requests.clear()
            api.script = [status(500, {"error": {"message": "internal error, try again"}})]
            c = pu.OpenAICompatibleClient("m", base_url=api.url + "/v1", api_key=SECRET, max_attempts=2)
            rows = sentiment(c, reviews.head(2))
            equal([r["s"] for r in rows], ["positive", "negative"])
            equal(len(api.requests), 2)

        with j.step("a server that never answers times out with a structured error"):
            slow = FakeServer(default=lambda h, b: time.sleep(3))
            try:
                c = pu.OpenAICompatibleClient("m", base_url=slow.url + "/v1", api_key=SECRET, max_attempts=1, timeout=0.5)
                t = time.monotonic()
                try:
                    c.complete("s", "hi")
                    raise AssertionError("expected LLMProviderError")
                except pu.LLMProviderError as e:
                    check("Could not reach" in e.message or "timed out" in e.message.lower(), e.message)
                check(time.monotonic() - t < 2.5, "timeout not honored")
            finally:
                slow.close()

        with j.step("claude via Anthropic's OpenAI-compatible endpoint (preset 'claude', local base_url)"):
            api.requests.clear()
            c = pu.llm_client("claude:claude-local", base_url=api.url + "/v1", api_key=SECRET)
            rows = sentiment(c, reviews.head(2))
            equal([r["s"] for r in rows], ["positive", "negative"])
            equal(api.requests[0]["headers"].get("authorization"), f"Bearer {SECRET}")
            equal(c.model_id, "claude:claude-local")

        with j.step("anthropic: without the SDK, llm_client falls back to the OpenAI-compatible client"):
            try:
                import anthropic  # noqa: F401
                raise Skip("the anthropic SDK is installed here; the fallback path isn't used")
            except ImportError:
                pass
            c = pu.llm_client("anthropic:claude-local", api_key=SECRET)
            check(isinstance(c, pu.OpenAICompatibleClient), type(c).__name__)
            equal(c.base_url, "https://api.anthropic.com/v1")

        with j.step("anthropic: client options such as timeout/base_url give a client or a structured error"):
            try:
                pu.llm_client("anthropic:claude-local", api_key=SECRET, base_url=api.url + "/v1", timeout=5)
            except pu.PyursaError:
                pass  # a structured "not supported" error would also be fine
            # a raw TypeError is the failure this step guards against

        with j.step("native Anthropic wrapper: llm_map through AnthropicClient with an SDK-shaped client"):
            from types import SimpleNamespace

            from pyursa.semantic import AnthropicClient
            seen = []

            class Messages:
                def create(self, *, model, max_tokens, system, messages):
                    seen.append({"model": model, "system": system, "messages": messages})
                    return SimpleNamespace(content=[SimpleNamespace(type="text", text=_answer_prompt(messages[0]["content"]))],
                                           usage=SimpleNamespace(input_tokens=50, output_tokens=5))
            c = AnthropicClient(model="claude-local", client=SimpleNamespace(messages=Messages()))
            rows = sentiment(c, reviews.head(2))
            equal([r["s"] for r in rows], ["positive", "negative"])
            equal(seen[0]["model"], "claude-local")
            check("untrusted data" in seen[0]["system"])
            rep = pu.llm_reports()[0]
            equal((rep["input_tokens"], rep["output_tokens"], rep["model"]), (50, 5, "anthropic:claude-local"))

        bed = FakeServer(default=bedrock_ok)
        try:
            with j.step("Bedrock Converse with a Bedrock API key (bearer) through llm_map"):
                bed.requests.clear()
                c = pu.BedrockClient("us.amazon.nova-2-lite-v1:0", api_key="bedrock-key-123", endpoint_url=bed.url,
                                     region="us-east-1")
                equal(c.auth_method, "api-key")
                rows = sentiment(c, reviews.head(2))
                equal([r["s"] for r in rows], ["positive", "negative"])
                r0 = bed.requests[0]
                equal(r0["path"], "/model/us.amazon.nova-2-lite-v1%3A0/converse")
                equal(r0["headers"].get("authorization"), "Bearer bedrock-key-123")
                check(r0["body"]["inferenceConfig"]["maxTokens"] > 0 and r0["body"]["system"][0]["text"])
                rep = pu.llm_reports()[0]
                equal((rep["input_tokens"], rep["output_tokens"]), (77, 9))

            with j.step("Bedrock Converse signed with AWS access keys (SigV4)"):
                bed.requests.clear()
                c = pu.BedrockClient("us.anthropic.claude-haiku-5-5", endpoint_url=bed.url, region="eu-west-1",
                                     aws_access_key_id="AKIDJOURNEYEXAMPLE", aws_secret_access_key="journey/secret/key",
                                     aws_session_token="journey-session-token")
                equal(c.auth_method, "access-keys")
                rows = sentiment(c, reviews.head(1))
                equal(rows[0]["s"], "positive")
                h = bed.requests[0]["headers"]
                auth = h.get("authorization", "")
                m = re.fullmatch(r"AWS4-HMAC-SHA256 Credential=AKIDJOURNEYEXAMPLE/(\d{8})/eu-west-1/bedrock/aws4_request, "
                                 r"SignedHeaders=([a-z0-9;-]+), Signature=([0-9a-f]{64})", auth)
                check(m is not None, f"Authorization header: {auth!r}")
                check(h.get("x-amz-date", "").startswith(m.group(1)), "x-amz-date must match the credential date")
                equal(h.get("x-amz-security-token"), "journey-session-token")
                check("host" in m.group(2).split(";") and "x-amz-date" in m.group(2).split(";"), m.group(2))
                check("journey/secret/key" not in json.dumps(h), "the secret key must never be sent")

            with j.step("Bedrock retries re-sign each attempt, and errors carry Bedrock hints"):
                bed.requests.clear()
                bed.script = [status(503, {"message": "ServiceUnavailable"})]
                c = pu.BedrockClient("us.amazon.nova-2-lite-v1:0", endpoint_url=bed.url, region="us-east-1",
                                     aws_access_key_id="AKIDJOURNEYEXAMPLE", aws_secret_access_key="k", max_attempts=2)
                c.complete("s", "hi")
                equal(len(bed.requests), 2)
                check(bed.requests[1]["headers"]["authorization"].startswith("AWS4-HMAC-SHA256"))
                bed.script = [status(403, {"message": "The security token included in the request is invalid."})]
                c = pu.BedrockClient("us.amazon.nova-2-lite-v1:0", endpoint_url=bed.url, region="us-east-1",
                                     api_key="bad", max_attempts=1)
                try:
                    c.complete("s", "hi")
                    raise AssertionError("expected LLMProviderError")
                except pu.LLMProviderError as e:
                    equal(e.details.get("status"), 403)
                    check("security token" in e.message, e.message)
                    check(any("AWS_BEARER_TOKEN_BEDROCK" in s["description"] for s in e.suggestions), str(e.suggestions))
        finally:
            bed.close()
    finally:
        api.close()
        other.close()
        S.CONFIG = saved_config
        for k in [k for k in os.environ if _ENV_PATTERN.search(k)]:
            del os.environ[k]
        os.environ.update(saved_env)
