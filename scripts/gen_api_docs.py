#!/usr/bin/env python3
"""Generate docs/api.md from the public API's signatures and docstrings.

    python scripts/gen_api_docs.py          # writes docs/api.md
    python scripts/gen_api_docs.py --check  # exit 1 if docs/api.md is out of date (used in CI)
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyursa as pu  # noqa: E402
from pyursa import expr as E  # noqa: E402
from pyursa.frame import GroupBy  # noqa: E402
from pyursa.tools import ToolSession  # noqa: E402


def sig(obj, name: str) -> str:
    try:
        s = str(inspect.signature(obj))
    except (TypeError, ValueError):
        s = "(...)"
    s = s.replace("'", "").replace("pyursa.types.", "").replace("pyursa.frame.", "")
    s = s.replace("pyursa.expr.", "").replace("collections.abc.", "").replace("typing.", "")
    return f"{name}{s}"


def doc(obj) -> str:
    d = inspect.getdoc(obj) or ""
    return d.strip()


def section(title: str, items: list[tuple[str, object]], prefix: str = "") -> list[str]:
    out = [f"## {title}", ""]
    for name, obj in items:
        out.append(f"### `{prefix}{sig(obj, name)}`")
        out.append("")
        d = doc(obj)
        if d:
            out.append(d)
            out.append("")
    return out


def public_methods(cls, skip=()) -> list[tuple[str, object]]:
    out = []
    for name, obj in cls.__dict__.items():
        if name.startswith("_") or name in skip:
            continue
        if isinstance(obj, property):
            out.append((name, obj.fget))
        elif callable(obj):
            out.append((name, obj))
    return out


def build() -> str:
    lines = [
        "# API reference",
        "",
        "_Generated from the code by `scripts/gen_api_docs.py`; do not edit by hand._",
        "",
        f"Version {pu.__version__}.",
        "",
    ]
    readers = [(n, getattr(pu, n)) for n in ("read_parquet", "read_csv", "read_json", "from_dict", "from_arrow",
                                             "from_pandas", "from_polars", "concat")]
    lines += section("Creating frames", readers, "pu.")
    lines += section("Expressions", [(n, getattr(pu, n)) for n in ("col", "lit", "len", "when", "coalesce",
                                                                   "row_number", "rank")], "pu.")
    lines += section("Frame", public_methods(pu.Frame), "Frame.")
    lines += section("GroupBy", public_methods(GroupBy), "GroupBy.")
    expr_methods = [(n, o) for n, o in public_methods(E.Expr, skip=("dtype", "children", "output_name", "columns",
                                                                    "has_agg", "has_window", "needs_row_index",
                                                                    "cols_outside_agg"))]
    lines += section("Expression methods", expr_methods, "Expr.")
    lines += section("String methods (`col(...).str`)", public_methods(E._StrNamespace), "str.")
    lines += section("Date and time methods (`col(...).dt`)", public_methods(E._DtNamespace), "dt.")
    lines += section("Types", [(n, getattr(pu, n)) for n in ("Decimal", "List", "Struct")], "pu.")
    lines += ["Constants: `pu.Int64`, `pu.Float64`, `pu.String`, `pu.Boolean`, `pu.Date`, `pu.Datetime` "
              "(call it for a zone: `pu.Datetime(\"UTC\")`), `pu.Null`. Strings work too: `\"Decimal(12,2)\"`.", ""]
    lines += section("Backends", [(n, getattr(pu, n)) for n in ("available_backends", "default_backend",
                                                                "set_default_backend", "register_backend",
                                                                "get_backend")], "pu.")
    lines += section("LLM operators", [(n, getattr(pu, n)) for n in ("set_llm", "llm_client", "llm_reports",
                                                                     "list_providers", "print_providers")], "pu.")
    lines += section("LLM clients", [("OpenAICompatibleClient", pu.OpenAICompatibleClient),
                                     ("AnthropicClient", pu.AnthropicClient), ("BedrockClient", pu.BedrockClient),
                                     ("FunctionClient", pu.FunctionClient), ("PromptClient", pu.PromptClient)], "pu.")
    tool_methods = [(n, o) for n, o in public_methods(ToolSession)]
    lines += section("Agent tools (`pyursa.tools.ToolSession`)", [("ToolSession", ToolSession)] + tool_methods)
    lines += ["## Errors", "",
              "Every error is an `pu.PyursaError` with `.code`, `.message`, `.details`, `.suggestions`, "
              "`.to_dict()` and `.to_json()`.", ""]
    for name in sorted(n for n in pu.__all__ if n.endswith("Error")):
        cls = getattr(pu, name)
        lines.append(f"- `pu.{name}` — code `{getattr(cls, 'code', '')}`")
    lines.append("")
    lines += ["## Semantics", "", "The rules every backend follows (`pu.SEMANTICS`):", ""]
    for key, rule in pu.SEMANTICS.items():
        lines.append(f"- **{key}**: {rule}")
    lines.append("")
    return "\n".join(lines)


GROUPS = [
    ("Model vendors", ["openai", "claude", "gemini", "mistral", "cohere", "meta", "xai", "deepseek", "dashscope",
                       "dashscope-workspace", "moonshot", "zai", "minimax", "perplexity", "ai21", "reka", "upstage"]),
    ("Inference hosts and routers", ["groq", "together", "fireworks", "cerebras", "sambanova", "deepinfra", "nebius",
                                     "novita", "baseten", "openrouter", "huggingface", "nvidia", "scaleway", "ovhcloud"]),
    ("Cloud platforms", ["azure", "vertex", "bedrock-openai", "databricks", "snowflake", "cloudflare", "oci"]),
    ("Gateways", ["litellm", "portkey", "vercel", "helicone", "cloudflare-gateway"]),
    ("Local and self-hosted", ["ollama", "lmstudio", "vllm", "llamacpp", "localai", "jan", "tgi", "sglang",
                               "koboldcpp", "gpt4all"]),
]


def build_providers() -> str:
    from pyursa import providers as P
    listed = {n for _, names in GROUPS for n in names}
    missing = set(P.PROVIDERS) - listed
    assert not missing, f"add these providers to GROUPS: {sorted(missing)}"
    lines = [
        "# LLM providers",
        "",
        "_Generated from `src/pyursa/providers.py` by `scripts/gen_api_docs.py`; do not edit by hand._",
        "",
        "Use any of these with `pu.llm_client(\"<provider>:<model>\")`. Every entry was checked against the",
        "provider's official documentation on 2026-10-20. Pass the model id your provider lists.",
        "Run `python scripts/check_providers.py` to test them from your machine.",
        "",
        "Set keys in PowerShell with `$env:NAME=\"...\"`, or `export NAME=...` on macOS/Linux.",
        "",
        "## Native APIs",
        "",
        "| Provider | What it covers |",
        "|---|---|",
    ]
    for name, about in P.NATIVE_PROVIDERS.items():
        lines.append(f"| `{name}` | {about} |")
    lines.append("")
    for title, names in GROUPS:
        lines += [f"## {title}", "", "| Provider | Key variable | URL settings | Models | Notes |", "|---|---|---|---|---|"]
        for n in names:
            p = P.PROVIDERS[n]
            key = " or ".join(f"`{k}`" for k in p.key_vars) or "none"
            if p.key_optional and p.key_vars:
                key += " (optional)"
            settings = ", ".join(f"`{env}`" + (f" (default `{dict(p.defaults)[k]}`)" if k in dict(p.defaults) else "")
                                 for k, env in p.params) or "-"
            notes = p.note
            if p.token_param == "max_completion_tokens":
                notes = (notes + "; " if notes else "") + "sends `max_completion_tokens`"
            if p.auth.startswith("header:"):
                notes = (notes + "; " if notes else "") + f"key sent in `{p.auth.split(':', 1)[1]}` header"
            lines.append(f"| [`{n}`]({p.docs}) | {key} | {settings} | {p.models} | {notes} |")
        lines.append("")
    lines += ["## Retired or not OpenAI-compatible", "", "| Name | Status |", "|---|---|"]
    seen = set()
    for n, why in P.UNAVAILABLE.items():
        if why in seen:
            continue
        seen.add(why)
        lines.append(f"| `{n}` | {why} |")
    lines += ["", "## Anything else", "",
              "Any other OpenAI-compatible server works too:", "",
              "```python",
              "pu.llm_client(\"openai-compatible:<model>\")   # base URL from OPENAI_BASE_URL, key from OPENAI_API_KEY",
              "pu.OpenAICompatibleClient(\"<model>\", base_url=\"https://host/v1\", api_key=\"...\", headers={...})",
              "```", ""]
    return "\n".join(lines)


def main() -> int:
    targets = {ROOT / "docs" / "api.md": build(), ROOT / "docs" / "providers.md": build_providers()}
    if "--check" in sys.argv:
        stale = [t.name for t, text in targets.items()
                 if not t.exists() or t.read_text(encoding="utf-8") != text]
        if stale:
            print(f"{', '.join(stale)} out of date; run python scripts/gen_api_docs.py")
            return 1
        return 0
    for target, text in targets.items():
        target.parent.mkdir(exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
