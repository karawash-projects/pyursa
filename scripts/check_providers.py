#!/usr/bin/env python3
"""Check every LLM provider pyursa knows about, from your own machine.

    python scripts/check_providers.py              # all providers
    python scripts/check_providers.py groq azure   # just these
    python scripts/check_providers.py --model groq=llama-3.3-70b-versatile --model openai=<model>

For each provider:
- If its key (and any URL settings) are set in your environment, it makes a REAL call: a tiny
  llm_map over two rows. Costs a fraction of a cent. Pass --model provider=<id> to choose the
  model; otherwise only the endpoint check runs.
- Otherwise it sends a request with a fake key. "endpoint OK" means the service answered
  "unauthorized", which proves the address and request format are right.
Local servers (Ollama, LM Studio, ...) are checked only if they're running.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyursa as pu  # noqa: E402
from pyursa import providers as P  # noqa: E402
from pyursa import semantic as S  # noqa: E402

# Real, publicly listed model ids (from each provider's docs) so the server checks the key, not the model name.
PROBE_MODEL = {
    "bedrock-openai": "openai.gpt-oss-120b-1:0",
    "fireworks": "accounts/fireworks/models/glm-5p3-flash",
    "sambanova": "Meta-Llama-3.3-70B-Instruct",
    "nvidia": "meta/llama-3.3-70b-instruct",
    "snowflake": "llama3.1-70b",
    "cloudflare": "@cf/meta/llama-3.1-8b-instruct",
    "together": "Qwen/Qwen3.5-9B",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.8-flash",
    "vertex": "google/gemini-3.8-flash",
    "mistral": "mistral-large-latest",
    "openai": "gpt-6-astra",
}


def configured(name: str) -> bool:
    p = P.PROVIDERS[name]
    has_key = any(os.environ.get(v) for v in p.key_vars) or not p.key_vars or p.key_optional
    has_params = all(os.environ.get(env) or dict(p.defaults).get(k) for k, env in p.params)
    return has_key and has_params and not p.key_optional


def real_call(name: str, model: str) -> tuple[str, str]:
    client = pu.llm_client(f"{name}:{model}", max_attempts=2, timeout=60)
    S.CONFIG = S.LLMConfig(client=client)
    frame = pu.from_dict({"text": ["I love this product", "It arrived broken and late"]})
    rows = frame.llm_map("Sentiment as one lowercase word: positive or negative", output="s", inputs=["text"]).to_dicts()
    rep = pu.llm_reports()[0]
    labels = [r["s"] for r in rows]
    ok = labels == ["positive", "negative"] and rep["failed_items"] == 0
    return ("WORKS" if ok else "ANSWERED"), f"{labels} tokens in/out {rep['input_tokens']}/{rep['output_tokens']}"


def classify(status: int | None, message: str, *, faked: list[str], local: bool) -> tuple[str, str]:
    """Turn a probe result into a verdict. Kept separate so it can be unit-tested."""
    low = message.lower()
    if status in (401, 403) or (status and P._looks_like_auth_error(message)):
        return "endpoint OK", f"HTTP {status}: rejected the fake key, as expected"
    if faked:
        return "needs settings", "set " + ", ".join(faked)
    if status in (400, 404, 410, 422) and "model" in low:  # 410: model retired
        return "endpoint OK", f"HTTP {status}: server answered (the test model isn't available on it)"
    if status:
        return "CHECK", f"HTTP {status}: {message[:80]}"
    if local:
        return "not running", "local server not started"
    return "unreachable", message[:80]


def local_server_check(base_url: str) -> tuple[str, str]:
    """Is the thing on this port really an OpenAI-compatible server? Ask it for its model list."""
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=3) as r:
            data = json.loads(r.read().decode() or "{}")
        ids = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)] if isinstance(data, dict) else []
        if isinstance(data, dict) and "data" in data:
            return "running", "OpenAI-compatible; models: " + (", ".join(map(str, ids[:3])) or "none loaded")
        return "port busy", "something else is answering on this port"
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return "running", "OpenAI-compatible server that needs a key"
        return "port busy", f"another program answers on this port (HTTP {e.code}), not this server"
    except (urllib.error.URLError, OSError, ValueError):
        return "not running", "local server not started"


def endpoint_check(name: str) -> tuple[str, str]:
    p = P.PROVIDERS[name]
    if p.base_url.startswith("http://"):  # local server: check what is really listening
        return local_server_check(p.base_url)
    kw, faked = {}, []
    for k, env in p.params:
        value = os.environ.get(env) or dict(p.defaults).get(k)
        if not value:
            value, _ = "pyursa-probe", faked.append(env)
        kw[k] = value
    if name == "vertex" and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        kw["project"], _ = "pyursa-probe", faked.append("GOOGLE_CLOUD_PROJECT")
    if name == "azure" and os.environ.get("AZURE_OPENAI_ENDPOINT") and "AZURE_OPENAI_RESOURCE" in faked:
        faked.remove("AZURE_OPENAI_RESOURCE")
        kw.pop("resource", None)
    try:
        c = P.OpenAICompatibleClient(PROBE_MODEL.get(name, "pyursa-probe-model"), provider=name,
                                     api_key="invalid-pyursa-probe-key", max_attempts=1, timeout=10, **kw)
        c.complete("s", "hi")
        return "ANSWERED", "accepted a fake key"
    except pu.PyursaError as e:
        return classify(e.details.get("status"), e.message, faked=faked, local=False)


def check(name: str, models: dict[str, str]) -> tuple[str, str, str]:
    try:
        if configured(name) and name in models:
            status, info = real_call(name, models[name])
        else:
            status, info = endpoint_check(name)
    except pu.PyursaError as e:
        status, info = "FAILED", str(e).splitlines()[0][:100]
    except Exception as e:  # pragma: no cover
        status, info = "FAILED", f"{type(e).__name__}: {e}"[:100]
    return name, status, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("providers", nargs="*", help="provider names (default: all)")
    ap.add_argument("--model", action="append", default=[], metavar="PROVIDER=MODEL",
                    help="model id for a real call, e.g. groq=llama-3.3-70b-versatile")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.model)
    names = args.providers or sorted(P.PROVIDERS)
    unknown = [n for n in names if n not in P.PROVIDERS]
    if unknown:
        print(f"Unknown providers: {unknown}. Known: {', '.join(sorted(P.PROVIDERS))}")
        return 2
    print(f"pyursa {pu.__version__}: checking {len(names)} providers...\n")
    with cf.ThreadPoolExecutor(12) as ex:
        results = list(ex.map(lambda n: check(n, models), names))
    order = {"WORKS": 0, "endpoint OK": 1, "running": 2, "needs settings": 3, "not running": 4}
    for name, status, info in sorted(results, key=lambda r: (order.get(r[1], 9), r[0])):
        print(f"  {status:<15} {name:<20} {info}")
    bad = [r for r in results if r[1] in ("FAILED", "CHECK")]
    if not models:
        print("\nTip: add --model <provider>=<model id> to make a real call with your key, "
              "e.g. --model groq=llama-3.3-70b-versatile")
    print(f"\n{sum(r[1] == 'WORKS' for r in results)} working with your keys, "
          f"{sum(r[1] == 'endpoint OK' for r in results)} endpoints confirmed, {len(bad)} to look at.")
    print("Bedrock (Nova, Claude on AWS) uses its own API; with a Bedrock API key or AWS access keys set, run: "
          "python scripts/e2e.py --only llm --llm bedrock:us.amazon.nova-2-lite-v1:0")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
