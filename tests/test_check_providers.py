"""The provider-check script's verdicts, including the cases seen on a real machine."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_providers as C  # noqa: E402


@pytest.mark.parametrize("status, message, faked, expected", [
    (401, "Invalid key", [], "endpoint OK"),
    (403, "Forbidden", [], "endpoint OK"),
    (400, "Incorrect API key provided. You can obtain an API key from https://console.x.ai.", [], "endpoint OK"),  # xAI
    (400, "API key not valid. Please pass a valid API key.", [], "endpoint OK"),                                   # Gemini
    (404, "Model not found, inaccessible, and/or not deployed", [], "endpoint OK"),                                 # Fireworks
    (404, "The model `pyursa-probe-model` does not exist or you do not have access to it.", [], "endpoint OK"),    # SambaNova
    (410, "The model 'meta/llama-3.1-8b-instruct' has reached its end of life", [], "endpoint OK"),               # NVIDIA
    (404, "", ["CLOUDFLARE_ACCOUNT_ID"], "needs settings"),                                                      # Cloudflare
    (400, "Workspace endpoint is invalid.", ["DASHSCOPE_WORKSPACE_ID"], "needs settings"),                       # DashScope
    (401, "bad key", ["SNOWFLAKE_ACCOUNT"], "endpoint OK"),
    (404, "Not Found", [], "CHECK"),
    (500, "boom", [], "CHECK"),
    (None, "Could not reach", [], "unreachable"),
])
def test_classify(status, message, faked, expected):
    assert C.classify(status, message, faked=faked, local=False)[0] == expected


class Handler(BaseHTTPRequestHandler):
    mode = "openai"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if Handler.mode == "openai":
            body, code = json.dumps({"object": "list", "data": [{"id": "llama3.2"}]}), 200
        elif Handler.mode == "auth":
            body, code = "{}", 401
        else:
            body, code = "<html>my web app</html>", 404
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body.encode())


@pytest.mark.parametrize("mode, expected", [("openai", "running"), ("auth", "running"), ("other", "port busy")])
def test_local_server_check(mode, expected):
    Handler.mode = mode
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        verdict, info = C.local_server_check(f"http://127.0.0.1:{httpd.server_port}/v1")
        assert verdict == expected, info
        if mode == "openai":
            assert "llama3.2" in info
    finally:
        httpd.shutdown()


def test_local_server_not_running():
    assert C.local_server_check("http://127.0.0.1:9/v1")[0] == "not running"
