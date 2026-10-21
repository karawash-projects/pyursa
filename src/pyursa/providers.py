"""LLM providers for llm_filter / llm_map / llm_extract.

Almost every model is reachable through one dependency-free OpenAI-compatible client
(POST {base_url}/chat/completions): OpenAI GPT, Claude (Anthropic's OpenAI-compatible
endpoint), Gemini, Llama, Mistral, DeepSeek, Qwen, Grok, OpenRouter, LiteLLM and other
gateways, Bedrock's own OpenAI endpoint, and local servers (Ollama, LM Studio, vLLM).

The exception is AWS: Bedrock's OpenAI-compatible endpoint does not serve Amazon Nova or
Claude, so `bedrock:` uses Bedrock's native Converse API. It works with just a Bedrock API
key (AWS_BEARER_TOKEN_BEDROCK, no SDK needed), or with IAM credentials through boto3.

    pu.set_llm(pu.llm_client("anthropic:claude-haiku-5-5"))        # Claude (native SDK, or OpenAI-compatible fallback)
    pu.set_llm(pu.llm_client("claude:claude-haiku-5-5"))           # Claude through Anthropic's OpenAI-compatible endpoint
    pu.set_llm(pu.llm_client("bedrock:us.amazon.nova-2-lite-v1:0"))  # Nova on AWS (Bedrock API key or IAM)
    pu.set_llm(pu.llm_client("bedrock:us.anthropic.claude-haiku-5-5")) # Claude on AWS
    pu.set_llm(pu.llm_client("openrouter:amazon/nova-lite-v1"))    # Nova in OpenAI format through a gateway
    pu.set_llm(pu.llm_client("gemini:gemini-3.8-flash"))           # GEMINI_API_KEY
    pu.set_llm(pu.llm_client("ollama:llama3.2"))                   # local, no key
    pu.set_llm(pu.OpenAICompatibleClient("my-model", base_url="http://my-server:8000/v1"))

Model names change often, so pass the id your provider lists; only the Anthropic
default is built in. `pu.list_providers()` prints every provider and its key variable.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .errors import PyursaError
from .semantic import AnthropicClient, LLMNotConfiguredError, PromptClient


class LLMProviderError(PyursaError):
    code = "LLM_PROVIDER_ERROR"


@dataclass(frozen=True)
class Provider:
    """How to reach one OpenAI-compatible service.

    base_url may contain {placeholders}; each is filled from a keyword argument of the same name,
    then from the environment variable named in `params`, then from `defaults`."""

    base_url: str
    key_env: str | None                    # main variable holding the key (None = no key)
    note: str = ""
    key_optional: bool = False             # send the key if set, but don't require it
    models: str = ""                       # what the provider is typically used for
    alt_key_envs: tuple[str, ...] = ()     # other variable names accepted for the key
    auth: str = "bearer"                   # "bearer", "header:<name>" or "none"
    params: tuple[tuple[str, str], ...] = ()     # (placeholder, environment variable)
    defaults: tuple[tuple[str, str], ...] = ()   # (placeholder, default value)
    query: tuple[tuple[str, str], ...] = ()      # fixed query parameters
    token_param: str = "max_tokens"        # "max_tokens", "max_completion_tokens" or "" to omit
    docs: str = ""                         # official documentation URL

    @property
    def key_vars(self) -> tuple[str, ...]:
        return ((self.key_env,) if self.key_env else ()) + self.alt_key_envs


def _p(base_url, key_env=None, note="", models="", docs="", **kw) -> Provider:
    return Provider(base_url, key_env, note=note, models=models, docs=docs, **kw)


_MCT = "max_completion_tokens"
_LOCAL = {"key_optional": True}

# Every entry was checked against the provider's official documentation on 2026-10-20 (see `docs`).
PROVIDERS: dict[str, Provider] = {
    # --- model vendors ---------------------------------------------------------------------------
    "openai": _p("https://api.openai.com/v1", "OPENAI_API_KEY", models="GPT, o-series", token_param=_MCT,
                 docs="https://developers.openai.com/api/reference/resources/chat"),
    "claude": _p("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",
                 "Claude through Anthropic's OpenAI-compatible endpoint (native SDK: anthropic:)", models="Claude",
                 docs="https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk"),
    "gemini": _p("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", models="Gemini",
                 alt_key_envs=("GOOGLE_API_KEY",), docs="https://ai.google.dev/gemini-api/docs/openai"),
    "mistral": _p("https://api.mistral.ai/v1", "MISTRAL_API_KEY", models="Mistral, Codestral",
                  docs="https://docs.mistral.ai/api/"),
    "cohere": _p("https://api.cohere.ai/compatibility/v1", "COHERE_API_KEY", models="Command",
                 alt_key_envs=("CO_API_KEY",), docs="https://docs.cohere.com/docs/compatibility-api"),
    "meta": _p("https://api.meta.ai/v1", "MODEL_API_KEY", "Meta Model API (successor of the Llama API)",
               models="Muse", token_param=_MCT, docs="https://dev.meta.ai/docs/protocols/chat-completions"),
    "xai": _p("https://api.x.ai/v1", "XAI_API_KEY", "Chat Completions is a legacy endpoint at xAI", models="Grok",
              docs="https://docs.x.ai/docs/guides/chat-completions"),
    "deepseek": _p("https://api.deepseek.com", "DEEPSEEK_API_KEY", models="DeepSeek",
                   docs="https://api-docs.deepseek.com/"),
    "dashscope": _p("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
                    "Alibaba Model Studio, international (legacy shared domain); keys are region-bound",
                    models="Qwen", docs="https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope"),
    "dashscope-workspace": _p("https://{workspace_id}.{region}.maas.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
                              "Alibaba Model Studio workspace domain (recommended)", models="Qwen",
                              params=(("workspace_id", "DASHSCOPE_WORKSPACE_ID"), ("region", "DASHSCOPE_REGION")),
                              defaults=(("region", "ap-southeast-1"),),
                              docs="https://www.alibabacloud.com/help/model-studio/base-url"),
    "moonshot": _p("https://api.moonshot.ai/v1", "MOONSHOT_API_KEY", models="Kimi", token_param=_MCT,
                   docs="https://platform.kimi.ai/docs/api/chat"),
    "zai": _p("https://api.z.ai/api/paas/v4", "ZAI_API_KEY", models="GLM",
              docs="https://docs.z.ai/guides/develop/openai/python"),
    "minimax": _p("https://api.minimax.io/v1", "MINIMAX_API_KEY", models="MiniMax", token_param=_MCT,
                  docs="https://platform.minimax.io/docs/api-reference/text-openai-api"),
    "perplexity": _p("https://api.perplexity.ai/router/v1", "PERPLEXITY_API_KEY",
                     "Perplexity's OpenAI-format router (open-weight models); Sonar moved to /v1/sonar",
                     models="hosted open-weight models", docs="https://docs.perplexity.ai/guides/chat-completions-guide"),
    "ai21": _p("https://api.ai21.com/studio/v1", "AI21_API_KEY", "OpenAI-shaped API; max 4096 output tokens",
               models="Jamba", docs="https://docs.ai21.com/reference/jamba-1-6-api-ref"),
    "reka": _p("https://api.reka.ai/v1", "REKA_API_KEY", models="Reka", docs="https://docs.reka.ai/quick-start"),
    "upstage": _p("https://api.upstage.ai/v1", "UPSTAGE_API_KEY", models="Solar",
                  docs="https://console.upstage.ai/docs/capabilities/chat"),
    # --- inference hosts and routers ------------------------------------------------------------------
    "groq": _p("https://api.groq.com/openai/v1", "GROQ_API_KEY", models="Llama, Qwen, gpt-oss", token_param=_MCT,
               docs="https://console.groq.com/docs/openai"),
    "together": _p("https://api.together.ai/v1", "TOGETHER_API_KEY", models="Llama, Qwen, DeepSeek, ...",
                   docs="https://docs.together.ai/docs/openai-api-compatibility"),
    "fireworks": _p("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY",
                    "Model ids are full paths: accounts/fireworks/models/...", models="open models",
                    docs="https://docs.fireworks.ai/tools-sdks/openai-compatibility"),
    "cerebras": _p("https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", models="Llama, Qwen, gpt-oss",
                   docs="https://inference-docs.cerebras.ai/resources/openai"),
    "sambanova": _p("https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY", models="Llama, DeepSeek, gpt-oss",
                    docs="https://docs.sambanova.ai/cloud/docs/capabilities/openai-compatibility"),
    "deepinfra": _p("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY", models="open models",
                    docs="https://docs.deepinfra.com/chat/overview"),
    "nebius": _p("https://api.tokenfactory.nebius.com/v1", "NEBIUS_API_KEY", "Nebius Token Factory",
                 models="open models", docs="https://docs.tokenfactory.nebius.com/quickstart.md"),
    "novita": _p("https://api.novita.ai/openai/v1", "NOVITA_API_KEY", models="open models",
                 docs="https://docs.novita.ai/guides/llm-api"),
    "baseten": _p("https://inference.baseten.co/v1", "BASETEN_API_KEY", models="open models",
                  docs="https://docs.baseten.co/development/model-apis/overview"),
    "openrouter": _p("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "One key for many vendors",
                     models="Nova, Claude, GPT, Gemini, Llama, ...", docs="https://openrouter.ai/docs/quickstart"),
    "huggingface": _p("https://router.huggingface.co/v1", "HF_TOKEN", "Hugging Face Inference Providers router",
                      models="open models (suffix :fastest, :cheapest or :<provider>)",
                      docs="https://huggingface.co/docs/inference-providers/index"),
    "nvidia": _p("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", "NVIDIA API catalog (hosted NIM)",
                 models="Llama, Nemotron, ...", docs="https://docs.api.nvidia.com/nim/reference/llm-apis"),
    "scaleway": _p("https://api.scaleway.ai/v1", "SCW_SECRET_KEY", models="open models",
                   docs="https://www.scaleway.com/en/docs/generative-apis/reference-content/openai-compatibility.md"),
    "ovhcloud": _p("https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "OVH_AI_ENDPOINTS_ACCESS_TOKEN",
                   models="open models", alt_key_envs=("AI_ENDPOINT_API_KEY",),
                   docs="https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-getting-started"),
    # --- cloud platforms --------------------------------------------------------------------------------
    "azure": _p("https://{resource}.openai.azure.com/openai/v1", "AZURE_OPENAI_API_KEY",
                "Azure OpenAI / Microsoft Foundry v1 API; the model is your DEPLOYMENT name; resource from "
                "AZURE_OPENAI_RESOURCE or AZURE_OPENAI_ENDPOINT", models="GPT, DeepSeek, ... (your deployments)",
                auth="header:api-key", params=(("resource", "AZURE_OPENAI_RESOURCE"),), token_param=_MCT,
                docs="https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle"),
    "vertex": _p("https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi", None,
                 "Google Vertex AI; OAuth token from Application Default Credentials (pip install google-auth) "
                 "or api_key=<access token>; models are google/<name>", models="Gemini, Model Garden",
                 auth="google", params=(("project", "GOOGLE_CLOUD_PROJECT"), ("location", "GOOGLE_CLOUD_LOCATION")),
                 defaults=(("location", "global"),),
                 docs="https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.endpoints.chat/completions"),
    "bedrock-openai": _p("https://bedrock-runtime.{region}.amazonaws.com/openai/v1", "AWS_BEARER_TOKEN_BEDROCK",
                         "Bedrock's OpenAI-compatible endpoint; it does NOT serve Nova or Claude (use bedrock:)",
                         models="gpt-oss, Qwen3, DeepSeek V3.x, newer Mistral",
                         params=(("region", "AWS_REGION"),), defaults=(("region", "us-east-1"),),
                         docs="https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html"),
    "databricks": _p("https://{workspace_host}/ai-gateway/mlflow/v1", "DATABRICKS_TOKEN",
                     "Databricks Foundation Model APIs; model = serving endpoint name", models="Claude, Llama, gpt-oss, ...",
                     params=(("workspace_host", "DATABRICKS_HOST"),),
                     docs="https://docs.databricks.com/aws/en/machine-learning/model-serving/query-chat-models"),
    "snowflake": _p("https://{account}.snowflakecomputing.com/api/v2/cortex/v1", "SNOWFLAKE_PAT",
                    "Snowflake Cortex (programmatic access token)", models="Claude, GPT, Llama, DeepSeek",
                    params=(("account", "SNOWFLAKE_ACCOUNT"),), token_param=_MCT,
                    docs="https://docs.snowflake.com/en/user-guide/snowflake-cortex/open_ai_sdk"),
    "cloudflare": _p("https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "CLOUDFLARE_API_TOKEN",
                     "Cloudflare Workers AI; models start with @cf/", models="Llama, Mistral, Qwen, ...",
                     alt_key_envs=("CLOUDFLARE_API_KEY",), params=(("account_id", "CLOUDFLARE_ACCOUNT_ID"),),
                     docs="https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/"),
    "oci": _p("https://inference.generativeai.{region}.oci.oraclecloud.com/openai/v1", "OCI_GENAI_API_KEY",
              "Oracle OCI Generative AI (API-key auth); Cohere models aren't served here", models="Grok, Llama, gpt-oss",
              params=(("region", "OCI_REGION"),), defaults=(("region", "us-chicago-1"),),
              docs="https://docs.oracle.com/en-us/iaas/Content/generative-ai/openai-compatible-api.htm"),
    # --- gateways ----------------------------------------------------------------------------------------
    "litellm": _p("http://localhost:4000/v1", "LITELLM_API_KEY",
                  "LiteLLM proxy: any model, including Bedrock Nova and Claude, in OpenAI format; model = proxy alias",
                  models="anything the gateway is configured for", **_LOCAL,
                  docs="https://docs.litellm.ai/docs/proxy/quick_start"),
    "portkey": _p("https://api.portkey.ai/v1", "PORTKEY_API_KEY", "Portkey gateway; model as @provider-slug/model",
                  models="anything in your Portkey catalog", auth="header:x-portkey-api-key",
                  docs="https://portkey.ai/docs/api-reference/inference-api/chat"),
    "vercel": _p("https://ai-gateway.vercel.sh/v1", "AI_GATEWAY_API_KEY", "Vercel AI Gateway; model as provider/model",
                 models="anything on the gateway", alt_key_envs=("VERCEL_OIDC_TOKEN",),
                 docs="https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions"),
    "helicone": _p("https://ai-gateway.helicone.ai", "HELICONE_API_KEY", "Helicone AI Gateway",
                   models="anything on the gateway", docs="https://docs.helicone.ai/gateway/overview"),
    "cloudflare-gateway": _p("https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat", "CLOUDFLARE_API_TOKEN",
                             "Cloudflare AI Gateway unified endpoint; model as provider/model; key = the upstream "
                             "provider's key unless stored in the gateway", models="anything on the gateway",
                             key_optional=True, params=(("account_id", "CLOUDFLARE_ACCOUNT_ID"),
                                                        ("gateway_id", "CLOUDFLARE_GATEWAY_ID")),
                             defaults=(("gateway_id", "default"),),
                             docs="https://developers.cloudflare.com/ai-gateway/usage/chat-completion/"),
    # --- local and self-hosted -------------------------------------------------------------------------------
    "ollama": _p("http://localhost:11434/v1", "OLLAMA_API_KEY", "Local Ollama (key only for Ollama Cloud)",
                 models="Llama, Qwen, Gemma, gpt-oss, ...", **_LOCAL, docs="https://docs.ollama.com/api/openai-compatibility"),
    "lmstudio": _p("http://localhost:1234/v1", "LM_API_TOKEN", "LM Studio local server", models="local models",
                   **_LOCAL, docs="https://lmstudio.ai/docs/app/api/endpoints/openai"),
    "vllm": _p("http://localhost:8000/v1", "VLLM_API_KEY", "vLLM server; model = served model name",
               models="open models", **_LOCAL,
               docs="https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/"),
    "llamacpp": _p("http://127.0.0.1:8080/v1", "LLAMA_API_KEY", "llama.cpp llama-server", models="GGUF models",
                   **_LOCAL, docs="https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md"),
    "localai": _p("http://localhost:8080/v1", "LOCALAI_API_KEY", "LocalAI", models="local models", **_LOCAL,
                  docs="https://localai.io/docs/features/text-generation/"),
    "jan": _p("http://127.0.0.1:1337/v1", "JAN_API_KEY", "Jan local API server", models="local models", **_LOCAL,
              docs="https://docs.jan.ai/desktop/api-server"),
    "tgi": _p("http://localhost:3000/v1", "HF_TOKEN", "Hugging Face TGI Messages API; model is usually \"tgi\"",
              models="the served model", **_LOCAL, docs="https://huggingface.co/docs/text-generation-inference/messages_api"),
    "sglang": _p("http://127.0.0.1:30000/v1", "SGLANG_API_KEY", "SGLang server", models="open models", **_LOCAL,
                 docs="https://docs.sglang.io/basic_usage/openai_api_completions.html"),
    "koboldcpp": _p("http://localhost:5001/v1", "KOBOLDCPP_PASSWORD", "KoboldCpp", models="GGUF models", **_LOCAL,
                    docs="https://github.com/LostRuins/koboldcpp/wiki"),
    "gpt4all": _p("http://localhost:4891/v1", None, "GPT4All (enable the Local API Server in settings)",
                  models="GPT4All models", auth="none", docs="https://docs.gpt4all.io/gpt4all_api_server/home.html"),
}

# Retired or not OpenAI-compatible (checked 2026-10-20): explain instead of failing with a vague error.
UNAVAILABLE = {
    "github": "GitHub Models was retired on 2026-07-30; GitHub suggests Microsoft Foundry (use azure:).",
    "github-models": "GitHub Models was retired on 2026-07-30; GitHub suggests Microsoft Foundry (use azure:).",
    "llama": "Meta's Llama API was replaced by the Meta Model API (use meta:), or run Llama via groq:, together:, "
             "cerebras:, ollama: and others.",
    "hyperbolic": "Hyperbolic retired its serverless inference API.",
    "lambda": "Lambda's Inference API was discontinued.",
    "writer": "Writer's chat API isn't OpenAI-compatible (/v1/chat); reach Palmyra through a gateway (litellm:) "
              "or a custom PromptClient.",
    "watsonx": "IBM watsonx.ai needs IAM token exchange and its OpenAI-style route is in preview; use a gateway "
               "(litellm:) for now.",
    "kong": "Kong AI Gateway routes are user-defined: use OpenAICompatibleClient(model, base_url=<your route>).",
}

NATIVE_PROVIDERS = {
    "openai-compatible": "Any other OpenAI-compatible server: base URL from OPENAI_BASE_URL (or base_url=), "
                         "key from OPENAI_API_KEY if the server needs one",
    "anthropic": "Claude through the native Anthropic SDK (falls back to the OpenAI-compatible endpoint if the SDK "
                 "isn't installed); key ANTHROPIC_API_KEY",
    "bedrock": "Any Bedrock model, including Amazon Nova and Claude, through the Converse API. Auth: Bedrock API key "
               "(AWS_BEARER_TOKEN_BEDROCK) or AWS access keys (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, optional "
               "AWS_SESSION_TOKEN), both without any SDK; profiles, SSO and roles with boto3. Region AWS_REGION",
}

_RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}
_NOT_ON_BEDROCK_OPENAI = ("nova", "anthropic", "claude", "llama", "titan")


def list_providers() -> list[dict[str, Any]]:
    """Every provider llm_client() accepts, with its endpoint, key variable and typical models."""
    out = [{"provider": name, "api": "native", "about": about} for name, about in NATIVE_PROVIDERS.items()]
    for name, p in PROVIDERS.items():
        out.append({"provider": name, "api": "openai-compatible", "base_url": p.base_url,
                    "key_env": p.key_env, "alt_key_envs": list(p.alt_key_envs),
                    "key_required": bool(p.key_env) and not p.key_optional,
                    "url_settings": {k: v for k, v in p.params}, "models": p.models, "about": p.note, "docs": p.docs})
    return out


def _region(region: str | None) -> str:
    return region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"


def _missing_key(provider: str, env: str, alternatives: tuple[str, ...] = ()) -> LLMNotConfiguredError:
    alt = f" (or {', '.join(alternatives)})" if alternatives else ""
    return LLMNotConfiguredError(
        f"{provider} needs an API key in the {env} environment variable{alt}.",
        details={"provider": provider, "env_var": env, "alternatives": list(alternatives)},
        suggestions=[{"description": f"Set it: PowerShell  $env:{env}=\"...\"  |  macOS/Linux  "
                                     f"export {env}=...  |  or pass api_key=."}],
    )


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], *, timeout: float, max_attempts: int,
               provider: str, model: str, hints, signer=None) -> dict[str, Any]:
    """POST JSON with retries on rate limits and transient errors. Errors never include keys.
    `signer(url, data, headers)` returns extra headers (e.g. AWS Signature V4), recomputed per attempt."""
    data = json.dumps(body).encode()
    last = ""
    for attempt in range(max_attempts):
        all_headers = {"Content-Type": "application/json", **headers}
        if signer is not None:
            all_headers.update(signer(url, data, all_headers))
        req = urllib.request.Request(url, data=data, headers=all_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = _error_text(e)
            if e.code not in _RETRY_STATUSES or attempt == max_attempts - 1 or _daily_quota_exhausted(e):
                raise LLMProviderError(
                    f"{provider} returned HTTP {e.code}: {detail}",
                    details={"provider": provider, "model": model, "status": e.code, "url": url.split("?")[0]},
                    suggestions=hints(e.code, detail),
                ) from None
            last = f"HTTP {e.code}"
            ra = _retry_after(e)
            wait = ra if ra is not None else 2 ** attempt + random.random()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == max_attempts - 1:
                raise LLMProviderError(
                    f"Could not reach {provider} at {url.split('?')[0]}: {getattr(e, 'reason', e)}",
                    details={"provider": provider, "url": url.split("?")[0]},
                    suggestions=[{"description": "Check the URL, region and network; for local servers make sure they are running."}],
                ) from None
            last = str(e)
            wait = 2 ** attempt + random.random()
        time.sleep(min(wait, 30))
    raise LLMProviderError(f"{provider} failed after retries: {last}")  # pragma: no cover


@dataclass(frozen=True)
class AwsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str | None = None

    def __repr__(self) -> str:  # never print secrets
        return f"AwsCredentials(access_key_id='{self.access_key_id[:4]}...', secret_access_key='***')"


def _aws_credentials(access_key_id: str | None = None, secret_access_key: str | None = None,
                     session_token: str | None = None) -> AwsCredentials | None:
    """Explicit keys, else the standard AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN."""
    if access_key_id or secret_access_key:
        if not (access_key_id and secret_access_key):
            raise LLMNotConfiguredError("Pass both aws_access_key_id and aws_secret_access_key.")
        return AwsCredentials(access_key_id, secret_access_key, session_token)
    akid, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if akid and secret:
        return AwsCredentials(akid, secret, session_token or os.environ.get("AWS_SESSION_TOKEN"))
    return None


def _uri_encode(text: str, keep_slash: bool) -> str:
    return urllib.parse.quote(text, safe="/-_.~" if keep_slash else "-_.~")


def sigv4_headers(method: str, url: str, body: bytes, headers: dict[str, str], creds: AwsCredentials,
                  region: str, service: str = "bedrock", now: Any = None) -> dict[str, str]:
    """AWS Signature Version 4 headers for one request (standard library only).
    https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv-create-signed-request.html"""
    import datetime as _dt
    import hashlib
    import hmac

    now = now or _dt.datetime.now(_dt.timezone.utc)
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    parts = urllib.parse.urlsplit(url)
    host = parts.netloc
    # every path segment is URI-encoded twice for non-S3 services: the client already encoded it once
    canonical_uri = _uri_encode(parts.path or "/", keep_slash=True)
    # the query string is already URI-encoded by the caller: sort the pairs as-is (same as botocore)
    pairs = [p.split("=", 1) if "=" in p else [p, ""] for p in parts.query.split("&") if p]
    canonical_query = "&".join(f"{k}={v}" for k, v in sorted(pairs))
    signed = {"host": host, "x-amz-date": amz_date}
    if creds.session_token:
        signed["x-amz-security-token"] = creds.session_token
    for k, v in headers.items():
        if k.lower() == "content-type":
            signed["content-type"] = v
    names = sorted(signed)
    canonical_headers = "".join(f"{n}:{' '.join(str(signed[n]).split())}\n" for n in names)
    signed_headers = ";".join(names)
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical_request = "\n".join([method, canonical_uri, canonical_query, canonical_headers, signed_headers,
                                    payload_hash])
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest()])

    def _hmac(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode(), hashlib.sha256).digest()

    k = _hmac(("AWS4" + creds.secret_access_key).encode(), day)
    for piece in (region, service, "aws4_request"):
        k = _hmac(k, piece)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    out = {"x-amz-date": amz_date, "Authorization": (
        f"AWS4-HMAC-SHA256 Credential={creds.access_key_id}/{scope}, SignedHeaders={signed_headers}, "
        f"Signature={signature}")}
    if creds.session_token:
        out["x-amz-security-token"] = creds.session_token
    return out


class OpenAICompatibleClient(PromptClient):
    """Any server implementing POST {base_url}/chat/completions.

    Use a preset (provider="groq") or any URL (base_url="https://host/v1"). URL placeholders such as
    {account_id} or {resource} can be passed as keyword arguments or come from the environment."""

    def __init__(self, model: str, *, provider: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 4096, json_mode: bool = False,
                 timeout: float = 120.0, max_attempts: int = 4, extra_body: dict[str, Any] | None = None,
                 headers: dict[str, str] | None = None, region: str | None = None,
                 token_param: str | None = None, query: dict[str, str] | None = None, **params: str) -> None:
        if not model:
            raise LLMNotConfiguredError("OpenAICompatibleClient needs a model name.",
                                        suggestions=[{"description": "Pass the provider's model id.",
                                                      "fix": 'pu.llm_client("gemini:gemini-3.8-flash")'}])
        preset = PROVIDERS.get(provider or "")
        if provider and preset is None:
            raise LLMNotConfiguredError(
                f"Unknown provider '{provider}'.",
                details={"known": sorted(PROVIDERS)},
                suggestions=[{"description": "Use a known provider or pass base_url= for any OpenAI-compatible server."}],
            )
        if provider == "bedrock-openai" and any(k in model.lower() for k in _NOT_ON_BEDROCK_OPENAI):
            raise LLMNotConfiguredError(
                f"Bedrock's OpenAI-compatible endpoint doesn't serve '{model}' (Nova, Claude, Llama and Titan "
                "are Converse/Invoke only).",
                details={"model": model},
                suggestions=[{"description": "Use the Converse-based provider instead (same Bedrock API key).",
                              "fix": f'pu.llm_client("bedrock:{model}")'}],
            )
        if region is not None:
            params.setdefault("region", region)
        name = provider or "custom"
        if not base_url:
            base_url, model = _prepare(name, params, model)
        self.base_url = _fill_url(base_url or (preset.base_url if preset else ""), preset, params, name)
        if not self.base_url:
            raise LLMNotConfiguredError("Pass provider= or base_url=.")
        self._token_fn = None
        if preset and preset.auth == "google" and api_key is None:
            self._token_fn = _google_token_source()
        self._aws = None
        if api_key is None and preset and preset.key_vars:
            api_key = next((os.environ[v] for v in preset.key_vars if os.environ.get(v)), None)
            if not api_key and provider == "bedrock-openai":
                self._aws = _aws_credentials(params.pop("aws_access_key_id", None),
                                             params.pop("aws_secret_access_key", None),
                                             params.pop("aws_session_token", None))
            if not api_key and not self._aws and not preset.key_optional:
                raise _missing_key(name, preset.key_env or preset.key_vars[0], preset.alt_key_envs)
        self.auth = preset.auth if preset else "bearer"
        self.headers = dict(headers or {})
        if provider == "claude" and os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            self.headers.setdefault("anthropic-workspace-id", os.environ["ANTHROPIC_WORKSPACE_ID"])
        q = dict(preset.query) if preset else {}
        q.update(query or {})
        self.query = q
        self.token_param = token_param if token_param is not None else (preset.token_param if preset else "max_tokens")
        self.api_key, self.model, self.max_tokens = api_key, model, max_tokens
        self.json_mode, self.timeout, self.max_attempts = json_mode, timeout, max_attempts
        self.extra_body = extra_body or {}
        self.provider = name
        self.model_id = f"{self.provider}:{model}"

    def _hints(self, status: int, detail: str) -> list[dict[str, str]]:
        env = PROVIDERS[self.provider].key_env if self.provider in PROVIDERS else None
        return _hints(status, env, detail)

    def _auth_headers(self) -> dict[str, str]:
        if self._token_fn is not None:
            return {"Authorization": f"Bearer {self._token_fn()}"}
        if not self.api_key or self.auth == "none":
            return {}
        if self.auth.startswith("header:"):
            return {self.auth.split(":", 1)[1]: self.api_key}
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(self, system: str, user: str) -> str:
        return self.complete_with_usage(system, user)[0]

    def _body(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if self.token_param:
            body[self.token_param] = self.max_tokens
        body.update(self.extra_body)
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete_with_usage(self, system: str, user: str):
        url = f"{self.base_url}/chat/completions"
        if self.query:
            url += "?" + urllib.parse.urlencode(self.query)
        headers = {**self.headers, **self._auth_headers()}
        signer = None
        if self._aws is not None:
            creds, region = self._aws, self.base_url.split("bedrock-runtime.", 1)[-1].split(".", 1)[0]
            signer = lambda u, data, h: sigv4_headers("POST", u, data, h, creds, region)  # noqa: E731
        try:
            out = _post_json(url, self._body(system, user), headers, timeout=self.timeout,
                             max_attempts=self.max_attempts, provider=self.provider, model=self.model,
                             hints=self._hints, signer=signer)
        except LLMProviderError as e:
            swap = _token_param_swap(e, self.token_param)
            if swap is None:
                raise
            self.token_param = swap  # e.g. newer OpenAI models want max_completion_tokens; remember it
            out = _post_json(url, self._body(system, user), headers, timeout=self.timeout,
                             max_attempts=self.max_attempts, provider=self.provider, model=self.model,
                             hints=self._hints, signer=signer)
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise LLMProviderError(f"Unexpected response shape from {self.provider}.",
                                   details={"response_keys": list(out) if isinstance(out, dict) else None}) from None
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        u = out.get("usage") or {}
        used = None
        if u.get("prompt_tokens") is not None:
            used = {"input_tokens": u.get("prompt_tokens") or 0, "output_tokens": u.get("completion_tokens") or 0}
        return content or "", used


def _prepare(provider: str, params: dict[str, str], model: str) -> tuple[str | None, str]:
    """Provider-specific URL and model conventions. Returns (explicit base_url or None, model)."""
    if provider == "azure" and not params.get("resource") and not os.environ.get("AZURE_OPENAI_RESOURCE"):
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        if endpoint:  # the Azure SDK convention: https://<resource>.openai.azure.com/
            endpoint = endpoint.rstrip("/")
            return (endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1"), model
    if provider == "databricks":
        host = params.get("workspace_host") or os.environ.get("DATABRICKS_HOST", "")
        if host:
            params["workspace_host"] = host.split("://", 1)[-1].strip("/")
    if provider == "vertex":
        location = params.get("location") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        params["location"] = location
        params.setdefault("host", "aiplatform.googleapis.com" if location == "global"
                          else f"{location}-aiplatform.googleapis.com")
        if "/" not in model:  # Vertex model ids carry the publisher: google/gemini-...
            model = f"google/{model}"
    return None, model


def _google_token_source():
    """Application Default Credentials (gcloud auth application-default login, a service account, ...).
    Tokens last about an hour; this refreshes them as needed."""
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError:
        raise LLMNotConfiguredError(
            "Vertex AI needs an OAuth access token.",
            suggestions=[{"description": "pip install google-auth and run `gcloud auth application-default login` "
                                         "(tokens are then refreshed automatically), or pass "
                                         "api_key=<output of `gcloud auth print-access-token`>."}],
        ) from None
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except Exception as e:
        raise LLMNotConfiguredError(f"No Google credentials found: {e}",
                                    suggestions=[{"description": "Run `gcloud auth application-default login`."}]) from None
    import threading as _threading
    lock = _threading.Lock()

    def token() -> str:
        with lock:
            if not creds.valid:
                creds.refresh(google.auth.transport.requests.Request())
            return creds.token

    return token


def _token_param_swap(e: LLMProviderError, current: str) -> str | None:
    """If a 400 says the token-limit field is wrong, return the field to use instead."""
    if e.details.get("status") != 400:
        return None
    msg = e.message.lower()
    if current == "max_tokens" and "max_completion_tokens" in msg:
        return "max_completion_tokens"
    if current == "max_completion_tokens" and "max_completion_tokens" in msg and (
            "unrecognized" in msg or "unknown" in msg or "not supported" in msg or "extra" in msg):
        return "max_tokens"
    return None


def _fill_url(url: str, preset: Provider | None, params: dict[str, str], provider: str) -> str:
    """Fill {placeholders} from keyword arguments, then environment variables, then defaults."""
    import re as _re
    names = _re.findall(r"\{(\w+)\}", url)
    if not names:
        return url.rstrip("/")
    env_for = dict(preset.params) if preset else {}
    defaults = dict(preset.defaults) if preset else {}
    if "region" in names:
        env_for.setdefault("region", "AWS_REGION")
    values, missing = {}, []
    for n in names:
        v = params.get(n) or (os.environ.get(env_for[n]) if n in env_for else None)
        if n == "region" and not v:
            v = os.environ.get("AWS_DEFAULT_REGION") if env_for.get("region") == "AWS_REGION" else None
        v = v or defaults.get(n)
        if not v:
            missing.append(n)
        else:
            values[n] = v
    if missing:
        sets = "  ".join(f'$env:{env_for[m]}="..."' for m in missing if m in env_for)
        raise LLMNotConfiguredError(
            f"{provider} needs {', '.join(missing)} for its URL.",
            details={"provider": provider, "missing": missing,
                     "env_vars": {m: env_for.get(m) for m in missing}, "url_template": url},
            suggestions=[{"description": "Pass them as keyword arguments, e.g. "
                                         + ", ".join(f"{m}='...'" for m in missing)
                                         + (f", or set the environment variables (PowerShell: {sets})." if sets else ".")}],
        )
    for n, v in values.items():
        url = url.replace("{" + n + "}", urllib.parse.quote(str(v), safe="-._~"))
    return url.rstrip("/")


class BedrockClient(PromptClient):
    """Amazon Bedrock Converse API: Amazon Nova, Claude, Llama, Mistral and every other Bedrock text model.

    Authentication, first match wins (no SDK needed for 1-3):
    1. api_key= or AWS_BEARER_TOKEN_BEDROCK: a Bedrock API key;
    2. aws_access_key_id= / aws_secret_access_key= (and aws_session_token=);
    3. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN) environment variables;
    4. boto3, if installed: profiles (AWS_PROFILE), SSO, instance and container roles.
    Requests for 2-3 are signed with AWS Signature V4 using the standard library.
    Region: region= or AWS_REGION / AWS_DEFAULT_REGION (default us-east-1).
    Newer models need a cross-region inference profile id such as "us.amazon.nova-2-lite-v1:0"
    or "us.anthropic.claude-haiku-5-5"; enable model access in the Bedrock console first."""

    def __init__(self, model: str, *, region: str | None = None, max_tokens: int = 4096, api_key: str | None = None,
                 aws_access_key_id: str | None = None, aws_secret_access_key: str | None = None,
                 aws_session_token: str | None = None, profile: str | None = None,
                 client: Any = None, endpoint_url: str | None = None, timeout: float = 120.0,
                 max_attempts: int = 4) -> None:
        if not model:
            raise LLMNotConfiguredError("BedrockClient needs a model id, e.g. us.amazon.nova-2-lite-v1:0.")
        self.model, self.max_tokens, self.timeout, self.max_attempts = model, max_tokens, timeout, max_attempts
        self.region = _region(region)
        self.model_id = f"bedrock:{model}"
        self._client = client
        self._key = None
        self._creds: AwsCredentials | None = None
        self.auth_method = "client" if client is not None else None
        if client is None:
            explicit_keys = aws_access_key_id or aws_secret_access_key
            if api_key or (not explicit_keys and not profile and os.environ.get("AWS_BEARER_TOKEN_BEDROCK")):
                self._key = api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
                self.auth_method = "api-key"
            elif not profile and (creds := _aws_credentials(aws_access_key_id, aws_secret_access_key, aws_session_token)):
                self._creds = creds
                self.auth_method = "access-keys"
            else:
                try:
                    import boto3
                except ImportError:
                    raise LLMNotConfiguredError(
                        "Bedrock needs a Bedrock API key, AWS access keys, or boto3 for profiles and roles.",
                        details={"model": model, "region": self.region},
                        suggestions=[
                            {"description": "Bedrock API key: PowerShell  $env:AWS_BEARER_TOKEN_BEDROCK=\"...\"  |  "
                                            "macOS/Linux  export AWS_BEARER_TOKEN_BEDROCK=..."},
                            {"description": "Access keys: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                                            "(plus AWS_SESSION_TOKEN for temporary keys)."},
                            {"description": "Profiles, SSO or roles: pip install boto3 and set AWS_PROFILE."},
                        ],
                    ) from None
                session = boto3.Session(profile_name=profile) if profile else boto3.Session()
                self._client = session.client("bedrock-runtime", region_name=self.region)
                self.auth_method = "boto3"
        self.endpoint_url = (endpoint_url or f"https://bedrock-runtime.{self.region}.amazonaws.com").rstrip("/")

    def _request(self, system: str, user: str) -> dict[str, Any]:
        system_blocks = [{"text": system}]
        messages = [{"role": "user", "content": [{"text": user}]}]
        config = {"maxTokens": self.max_tokens}
        if self._client is not None:  # boto3 or an injected client
            try:
                return self._client.converse(modelId=self.model, system=system_blocks, messages=messages,
                                             inferenceConfig=config)
            except Exception as e:
                raise LLMProviderError(
                    f"Bedrock call failed: {type(e).__name__}: {e}",
                    details={"provider": "bedrock", "model": self.model, "region": self.region},
                    suggestions=_bedrock_hints(0, str(e)),
                ) from None
        url = f"{self.endpoint_url}/model/{urllib.parse.quote(self.model, safe='')}/converse"
        body = {"system": system_blocks, "messages": messages, "inferenceConfig": config}
        if self._creds is not None:
            creds, region = self._creds, self.region
            signer = lambda u, data, h: sigv4_headers("POST", u, data, h, creds, region)  # noqa: E731
            return _post_json(url, body, {}, timeout=self.timeout, max_attempts=self.max_attempts,
                              provider="bedrock", model=self.model, hints=_bedrock_hints, signer=signer)
        return _post_json(url, body, {"Authorization": f"Bearer {self._key}"}, timeout=self.timeout,
                          max_attempts=self.max_attempts, provider="bedrock", model=self.model, hints=_bedrock_hints)

    def complete(self, system: str, user: str) -> str:
        return self.complete_with_usage(system, user)[0]

    def complete_with_usage(self, system: str, user: str):
        out = self._request(system, user)
        parts = (out.get("output") or {}).get("message", {}).get("content", [])
        u = out.get("usage") or {}
        used = None
        if u.get("inputTokens") is not None:
            used = {"input_tokens": u.get("inputTokens") or 0, "output_tokens": u.get("outputTokens") or 0}
        return "".join(p.get("text", "") for p in parts if isinstance(p, dict)), used


def llm_client(spec: str, **kwargs: Any) -> PromptClient:
    """Create a client from "provider:model". See list_providers() for every provider.

    Examples: "anthropic:claude-haiku-5-5", "claude:claude-haiku-5-5", "bedrock:us.amazon.nova-2-lite-v1:0",
    "bedrock:us.anthropic.claude-haiku-5-5", "openrouter:amazon/nova-lite-v1", "gemini:gemini-3.8-flash",
    "openai:<model>", "litellm:<model>", "ollama:llama3.2"."""
    if not isinstance(spec, str) or ":" not in spec:
        raise LLMNotConfiguredError(
            f"LLM spec must look like provider:model, got {spec!r}.",
            details={"providers": sorted([*NATIVE_PROVIDERS, *PROVIDERS])},
            suggestions=[{"description": "Pick a provider and model.", "fix": '"anthropic:claude-haiku-5-5"'}],
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    if provider == "anthropic":
        try:
            return AnthropicClient(model=model or "claude-haiku-5-5", **kwargs)
        except ImportError:  # no SDK installed: use Anthropic's OpenAI-compatible endpoint instead
            return OpenAICompatibleClient(model or "claude-haiku-5-5", provider="claude", **kwargs)
    if provider == "bedrock":
        return BedrockClient(model, **kwargs)
    if provider in ("openai-compatible", "custom"):
        # any server, configured the way the official OpenAI SDK is: OPENAI_BASE_URL (+ OPENAI_API_KEY)
        base = kwargs.pop("base_url", None) or os.environ.get("OPENAI_BASE_URL")
        if not base:
            raise LLMNotConfiguredError(
                "openai-compatible: needs a base URL.",
                suggestions=[{"description": "Set OPENAI_BASE_URL (PowerShell: $env:OPENAI_BASE_URL=\"https://host/v1\") "
                                             "and, if the server needs one, OPENAI_API_KEY; or pass base_url=."}],
            )
        key = kwargs.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        client = OpenAICompatibleClient(model, base_url=base, api_key=key, **kwargs)
        client.provider, client.model_id = "openai-compatible", f"openai-compatible:{model}"
        return client
    if provider in PROVIDERS:
        return OpenAICompatibleClient(model, provider=provider, **kwargs)
    if provider in UNAVAILABLE:
        raise LLMNotConfiguredError(f"'{provider}' isn't available: {UNAVAILABLE[provider]}",
                                    details={"provider": provider})
    raise LLMNotConfiguredError(
        f"Unknown LLM provider '{provider}'.",
        details={"providers": sorted([*NATIVE_PROVIDERS, *PROVIDERS])},
        suggestions=[{"description": "Use provider:model, or OpenAICompatibleClient(model, base_url=...) for any "
                                     "other OpenAI-compatible server.", "fix": '"gemini:gemini-3.8-flash"'}],
    )


def print_providers() -> None:
    """Print a readable table of every provider."""
    for r in list_providers():
        key = r.get("key_env") or "-"
        extra = ", ".join(f"{k} from {v}" for k, v in (r.get("url_settings") or {}).items())
        print(f"{r['provider']:<20} {key:<30} {r.get('models', '') or r.get('about', '')[:50]}"
              + (f"  [{extra}]" if extra else ""))


def detect_llm_spec() -> str | None:
    """Provider prefix of the first API key found in the environment (for tests and demos)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        return "bedrock"
    for name, p in PROVIDERS.items():
        if p.key_env and not p.key_optional and not p.params and os.environ.get(p.key_env):
            return name
    return None


def _error_text(e: urllib.error.HTTPError) -> str:
    """The provider's own error message, whatever shape it comes in."""
    try:
        raw = e.read().decode(errors="replace")
    except Exception:
        raw = ""
    e._pyursa_body = raw  # type: ignore[attr-defined]  # the body can only be read once; keep it for retry timing
    try:
        msg = _find_message(json.loads(raw)) if raw else None
    except ValueError:
        msg = raw.strip() if raw and len(raw) < 300 and "<" not in raw else None
    if not msg:
        msg = e.reason if isinstance(e.reason, str) else ""
    return " ".join(str(msg).split())[:300]


def _find_message(body: Any, depth: int = 0) -> str | None:
    """Handles {"error": {"message"}}, {"message"}, {"detail"}, {"errors": [...]}, and lists of those
    (Gemini's OpenAI layer wraps errors in a list)."""
    if depth > 4:
        return None
    if isinstance(body, list):
        for item in body:
            if (m := _find_message(item, depth + 1)):
                return m
        return None
    if isinstance(body, dict):
        for key in ("error", "errors"):
            if key in body and (m := _find_message(body[key], depth + 1)):
                return m
        for key in ("message", "Message", "detail", "error_description", "msg"):
            v = body.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, (dict, list)) and (m := _find_message(v, depth + 1)):
                return m
        return None
    if isinstance(body, str) and body:
        return body
    return None


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    """Seconds the provider asked us to wait: the Retry-After header, or Google's retryDelay / "retry in Ns"."""
    try:
        return float(e.headers.get("Retry-After"))
    except (TypeError, ValueError):
        pass
    body = getattr(e, "_pyursa_body", "") or ""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', body) or re.search(r"retry in (\d+(?:\.\d+)?)\s*s", body, re.I)
    return float(m.group(1)) if m else None


def _daily_quota_exhausted(e: urllib.error.HTTPError) -> bool:
    """A per-day quota won't recover in seconds, so retrying only wastes time."""
    body = getattr(e, "_pyursa_body", "") or ""
    return e.code == 429 and bool(re.search(r"per_?day|perday|daily", body, re.I))


def _quota_hint(detail: str) -> dict[str, str] | None:
    low = detail.lower()
    if "quota" in low or "billing" in low or "insufficient_quota" in low:
        return {"description": "Your account's quota for this model is used up (not a pyursa problem). Free tiers have "
                               "per-minute and per-day limits: wait for the reset, try a different model, or enable "
                               "billing with the provider."}
    return None


def _hints(status: int, env: str | None, detail: str = "") -> list[dict[str, str]]:
    if status in (401, 403) or _looks_like_auth_error(detail):
        return [{"description": f"Check the API key{f' in {env}' if env else ''} and that it has access to this model."}]
    if status == 404:
        return [{"description": "Check the model name and base URL; model ids differ per provider."}]
    if status == 400:
        return [{"description": "The provider rejected the request; try json_mode=False or a different model."}]
    if status == 429 and (q := _quota_hint(detail)):
        return [q]
    return [{"description": "Retry later or reduce batch_size / concurrency in pu.set_llm()."}]


def _looks_like_auth_error(text: str) -> bool:
    """Some providers report a bad key as 400 instead of 401/403 (e.g. xAI, Gemini)."""
    low = text.lower()
    return any(k in low for k in ("api key", "api_key", "apikey", "unauthorized", "authenticat", "security token",
                                  "invalid token", "token is invalid", "expired token", "access token", "credential",
                                  "permission denied", "signature"))


def _bedrock_hints(status: int, detail: str) -> list[dict[str, str]]:
    low = detail.lower()
    out = []
    if "on-demand" in low or "inference profile" in low:
        out.append({"description": "This model needs a cross-region inference profile id: prefix the model id with "
                                   "your geography, e.g. us.amazon.nova-2-lite-v1:0 or us.anthropic.claude-haiku-5-5."})
    if status in (401, 403) or "access" in low or "denied" in low or "security token" in low:
        out.append({"description": "Check the Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) or IAM permissions, and enable "
                                   "access to this model in the Bedrock console for your region."})
    if status == 404 or "not found" in low or "invalid" in low:
        out.append({"description": "Check the model id and region (AWS_REGION); ids are listed in the Bedrock console."})
    if status in (429, 503) or "throttl" in low:
        out.append({"description": "Bedrock is throttling: lower concurrency/batch_size in pu.set_llm() or raise your quota."})
    return out or [{"description": "Check the model id, region and credentials."}]
