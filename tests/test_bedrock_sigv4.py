"""AWS access-key authentication for Bedrock: Signature V4 computed with the standard library,
checked byte-for-byte against botocore (AWS's own signer) and end to end against a local server."""

import datetime as dt
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import pyursa as pu
from pyursa import semantic as S
from pyursa.providers import AwsCredentials, BedrockClient, sigv4_headers

AKID, SECRET = "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
NOW = dt.datetime(2026, 10, 20, 14, 5, 9, tzinfo=dt.timezone.utc)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    saved = S.CONFIG
    S.CONFIG = S.LLMConfig()
    for v in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
              "AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(v, raising=False)
    yield
    S.CONFIG = saved


def botocore_signature(url, body, creds, region, service="bedrock"):
    """Sign with botocore's SigV4 building blocks at the same fixed timestamp."""
    pytest.importorskip("botocore")
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    stamp = NOW.strftime("%Y%m%dT%H%M%SZ")
    req = AWSRequest(method="POST", url=url, data=body, headers={"Content-Type": "application/json"})
    req.context["timestamp"] = stamp
    req.headers["X-Amz-Date"] = stamp
    signer = SigV4Auth(Credentials(creds.access_key_id, creds.secret_access_key, creds.session_token),
                       service, region)
    if creds.session_token:
        req.headers["X-Amz-Security-Token"] = creds.session_token
    canonical = signer.canonical_request(req)
    to_sign = signer.string_to_sign(req, canonical)
    signature = signer.signature(to_sign, req)
    scope = f"{NOW.strftime('%Y%m%d')}/{region}/{service}/aws4_request"
    signed = signer.signed_headers(signer.headers_to_sign(req))
    return {"Authorization": f"AWS4-HMAC-SHA256 Credential={creds.access_key_id}/{scope}, SignedHeaders={signed}, "
                             f"Signature={signature}", "X-Amz-Date": stamp,
            "X-Amz-Security-Token": creds.session_token}


@pytest.mark.parametrize("model", ["us.amazon.nova-2-lite-v1:0", "us.anthropic.claude-haiku-5-5",
                                   "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0"])
@pytest.mark.parametrize("token", [None, "IQoJb3JpZ2luX2VjEXAMPLE//token=="])
def test_signature_matches_botocore(model, token):
    import urllib.parse
    creds = AwsCredentials(AKID, SECRET, token)
    url = f"https://bedrock-runtime.ca-central-1.amazonaws.com/model/{urllib.parse.quote(model, safe='')}/converse"
    body = json.dumps({"messages": [{"role": "user", "content": [{"text": "héllo"}]}]}).encode()
    ours = sigv4_headers("POST", url, body, {"Content-Type": "application/json"}, creds, "ca-central-1", now=NOW)
    theirs = botocore_signature(url, body, creds, "ca-central-1")
    assert ours["Authorization"] == theirs["Authorization"]
    assert ours["x-amz-date"] == theirs["X-Amz-Date"]
    if token:
        assert ours["x-amz-security-token"] == theirs["X-Amz-Security-Token"]


@pytest.mark.parametrize("query", ["b=2&a=x%20y", "api-version=preview", "a=x+y&c=%2F&b"])
def test_signature_with_query_matches_botocore(query):
    creds = AwsCredentials(AKID, SECRET)
    url = "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions?" + query
    body = b'{"model":"openai.gpt-oss-120b-1:0"}'
    ours = sigv4_headers("POST", url, body, {"Content-Type": "application/json"}, creds, "us-east-1", now=NOW)
    assert ours["Authorization"] == botocore_signature(url, body, creds, "us-east-1")["Authorization"]


class Signed(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Signed.seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
        prompt = json.loads(body)["messages"][0]["content"][0]["text"]
        n = len(json.loads(re.search(r"<data>\n(.*)\n</data>", prompt, re.S).group(1)))
        text = json.dumps({"results": [{"id": i, "s": "ok"} for i in range(n)]})
        out = {"output": {"message": {"content": [{"text": text}]}}, "usage": {"inputTokens": 9, "outputTokens": 3}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode())


@pytest.fixture
def server():
    Signed.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), Signed)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


def test_access_keys_from_environment_sign_requests(monkeypatch, server):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", AKID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "tok")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    c = pu.llm_client("bedrock:eu.amazon.nova-2-lite-v1:0", endpoint_url=server)
    assert c.auth_method == "access-keys" and SECRET not in repr(c._creds)
    pu.set_llm(c)
    assert pu.from_dict({"t": ["a", "b"]}).llm_map("x", output="s").count() == 2
    h = Signed.seen[0]["headers"]
    assert h["authorization"].startswith(f"AWS4-HMAC-SHA256 Credential={AKID}/") and "/eu-west-1/bedrock/aws4_request" in h["authorization"]
    assert h["x-amz-security-token"] == "tok" and re.fullmatch(r"\d{8}T\d{6}Z", h["x-amz-date"])
    assert "SignedHeaders=content-type;host;x-amz-date;x-amz-security-token" in h["authorization"]


def test_explicit_keys_and_priority(monkeypatch, server):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "api-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ENVKEY")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "envsecret")
    assert BedrockClient("m").auth_method == "api-key"  # a Bedrock API key wins over environment access keys
    c = BedrockClient("m", aws_access_key_id="EXPLICIT", aws_secret_access_key="s", endpoint_url=server)
    assert c.auth_method == "access-keys" and c._creds.access_key_id == "EXPLICIT"  # explicit keys win over all
    with pytest.raises(pu.LLMNotConfiguredError):
        BedrockClient("m", aws_access_key_id="only-half")


def test_bedrock_openai_endpoint_with_access_keys(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", AKID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    c = pu.llm_client("bedrock-openai:openai.gpt-oss-120b-1:0", region="us-west-2")
    assert c._aws is not None and c.api_key is None
    assert c.base_url == "https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1"


def test_errors_never_contain_secret(monkeypatch):
    c = BedrockClient("m", aws_access_key_id=AKID, aws_secret_access_key=SECRET,
                      endpoint_url="http://127.0.0.1:9", max_attempts=1, timeout=2)
    pu.set_llm(c)
    with pytest.raises(pu.AgentFrameError if hasattr(pu, "AgentFrameError") else pu.PyursaError) as ei:
        pu.from_dict({"t": ["a"]}).llm_map("x", output="s").collect()
    assert SECRET not in ei.value.to_json() and SECRET not in str(ei.value)
