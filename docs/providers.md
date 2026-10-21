# LLM providers

_Generated from `src/pyursa/providers.py` by `scripts/gen_api_docs.py`; do not edit by hand._

Use any of these with `pu.llm_client("<provider>:<model>")`. Every entry was checked against the
provider's official documentation on 2026-10-20. Pass the model id your provider lists.
Run `python scripts/check_providers.py` to test them from your machine.

Set keys in PowerShell with `$env:NAME="..."`, or `export NAME=...` on macOS/Linux.

## Native APIs

| Provider | What it covers |
|---|---|
| `openai-compatible` | Any other OpenAI-compatible server: base URL from OPENAI_BASE_URL (or base_url=), key from OPENAI_API_KEY if the server needs one |
| `anthropic` | Claude through the native Anthropic SDK (falls back to the OpenAI-compatible endpoint if the SDK isn't installed); key ANTHROPIC_API_KEY |
| `bedrock` | Any Bedrock model, including Amazon Nova and Claude, through the Converse API. Auth: Bedrock API key (AWS_BEARER_TOKEN_BEDROCK) or AWS access keys (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, optional AWS_SESSION_TOKEN), both without any SDK; profiles, SSO and roles with boto3. Region AWS_REGION |

## Model vendors

| Provider | Key variable | URL settings | Models | Notes |
|---|---|---|---|---|
| [`openai`](https://developers.openai.com/api/reference/resources/chat) | `OPENAI_API_KEY` | - | GPT, o-series | sends `max_completion_tokens` |
| [`claude`](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk) | `ANTHROPIC_API_KEY` | - | Claude | Claude through Anthropic's OpenAI-compatible endpoint (native SDK: anthropic:) |
| [`gemini`](https://ai.google.dev/gemini-api/docs/openai) | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | - | Gemini |  |
| [`mistral`](https://docs.mistral.ai/api/) | `MISTRAL_API_KEY` | - | Mistral, Codestral |  |
| [`cohere`](https://docs.cohere.com/docs/compatibility-api) | `COHERE_API_KEY` or `CO_API_KEY` | - | Command |  |
| [`meta`](https://dev.meta.ai/docs/protocols/chat-completions) | `MODEL_API_KEY` | - | Muse | Meta Model API (successor of the Llama API); sends `max_completion_tokens` |
| [`xai`](https://docs.x.ai/docs/guides/chat-completions) | `XAI_API_KEY` | - | Grok | Chat Completions is a legacy endpoint at xAI |
| [`deepseek`](https://api-docs.deepseek.com/) | `DEEPSEEK_API_KEY` | - | DeepSeek |  |
| [`dashscope`](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope) | `DASHSCOPE_API_KEY` | - | Qwen | Alibaba Model Studio, international (legacy shared domain); keys are region-bound |
| [`dashscope-workspace`](https://www.alibabacloud.com/help/model-studio/base-url) | `DASHSCOPE_API_KEY` | `DASHSCOPE_WORKSPACE_ID`, `DASHSCOPE_REGION` (default `ap-southeast-1`) | Qwen | Alibaba Model Studio workspace domain (recommended) |
| [`moonshot`](https://platform.kimi.ai/docs/api/chat) | `MOONSHOT_API_KEY` | - | Kimi | sends `max_completion_tokens` |
| [`zai`](https://docs.z.ai/guides/develop/openai/python) | `ZAI_API_KEY` | - | GLM |  |
| [`minimax`](https://platform.minimax.io/docs/api-reference/text-openai-api) | `MINIMAX_API_KEY` | - | MiniMax | sends `max_completion_tokens` |
| [`perplexity`](https://docs.perplexity.ai/guides/chat-completions-guide) | `PERPLEXITY_API_KEY` | - | hosted open-weight models | Perplexity's OpenAI-format router (open-weight models); Sonar moved to /v1/sonar |
| [`ai21`](https://docs.ai21.com/reference/jamba-1-6-api-ref) | `AI21_API_KEY` | - | Jamba | OpenAI-shaped API; max 4096 output tokens |
| [`reka`](https://docs.reka.ai/quick-start) | `REKA_API_KEY` | - | Reka |  |
| [`upstage`](https://console.upstage.ai/docs/capabilities/chat) | `UPSTAGE_API_KEY` | - | Solar |  |

## Inference hosts and routers

| Provider | Key variable | URL settings | Models | Notes |
|---|---|---|---|---|
| [`groq`](https://console.groq.com/docs/openai) | `GROQ_API_KEY` | - | Llama, Qwen, gpt-oss | sends `max_completion_tokens` |
| [`together`](https://docs.together.ai/docs/openai-api-compatibility) | `TOGETHER_API_KEY` | - | Llama, Qwen, DeepSeek, ... |  |
| [`fireworks`](https://docs.fireworks.ai/tools-sdks/openai-compatibility) | `FIREWORKS_API_KEY` | - | open models | Model ids are full paths: accounts/fireworks/models/... |
| [`cerebras`](https://inference-docs.cerebras.ai/resources/openai) | `CEREBRAS_API_KEY` | - | Llama, Qwen, gpt-oss |  |
| [`sambanova`](https://docs.sambanova.ai/cloud/docs/capabilities/openai-compatibility) | `SAMBANOVA_API_KEY` | - | Llama, DeepSeek, gpt-oss |  |
| [`deepinfra`](https://docs.deepinfra.com/chat/overview) | `DEEPINFRA_API_KEY` | - | open models |  |
| [`nebius`](https://docs.tokenfactory.nebius.com/quickstart.md) | `NEBIUS_API_KEY` | - | open models | Nebius Token Factory |
| [`novita`](https://docs.novita.ai/guides/llm-api) | `NOVITA_API_KEY` | - | open models |  |
| [`baseten`](https://docs.baseten.co/development/model-apis/overview) | `BASETEN_API_KEY` | - | open models |  |
| [`openrouter`](https://openrouter.ai/docs/quickstart) | `OPENROUTER_API_KEY` | - | Nova, Claude, GPT, Gemini, Llama, ... | One key for many vendors |
| [`huggingface`](https://huggingface.co/docs/inference-providers/index) | `HF_TOKEN` | - | open models (suffix :fastest, :cheapest or :<provider>) | Hugging Face Inference Providers router |
| [`nvidia`](https://docs.api.nvidia.com/nim/reference/llm-apis) | `NVIDIA_API_KEY` | - | Llama, Nemotron, ... | NVIDIA API catalog (hosted NIM) |
| [`scaleway`](https://www.scaleway.com/en/docs/generative-apis/reference-content/openai-compatibility.md) | `SCW_SECRET_KEY` | - | open models |  |
| [`ovhcloud`](https://docs.ovhcloud.com/en/guides/public-cloud/ai-machine-learning/ai-endpoints-getting-started) | `OVH_AI_ENDPOINTS_ACCESS_TOKEN` or `AI_ENDPOINT_API_KEY` | - | open models |  |

## Cloud platforms

| Provider | Key variable | URL settings | Models | Notes |
|---|---|---|---|---|
| [`azure`](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle) | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_RESOURCE` | GPT, DeepSeek, ... (your deployments) | Azure OpenAI / Microsoft Foundry v1 API; the model is your DEPLOYMENT name; resource from AZURE_OPENAI_RESOURCE or AZURE_OPENAI_ENDPOINT; sends `max_completion_tokens`; key sent in `api-key` header |
| [`vertex`](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/projects.locations.endpoints.chat/completions) | none | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `global`) | Gemini, Model Garden | Google Vertex AI; OAuth token from Application Default Credentials (pip install google-auth) or api_key=<access token>; models are google/<name> |
| [`bedrock-openai`](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html) | `AWS_BEARER_TOKEN_BEDROCK` | `AWS_REGION` (default `us-east-1`) | gpt-oss, Qwen3, DeepSeek V3.x, newer Mistral | Bedrock's OpenAI-compatible endpoint; it does NOT serve Nova or Claude (use bedrock:) |
| [`databricks`](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-chat-models) | `DATABRICKS_TOKEN` | `DATABRICKS_HOST` | Claude, Llama, gpt-oss, ... | Databricks Foundation Model APIs; model = serving endpoint name |
| [`snowflake`](https://docs.snowflake.com/en/user-guide/snowflake-cortex/open_ai_sdk) | `SNOWFLAKE_PAT` | `SNOWFLAKE_ACCOUNT` | Claude, GPT, Llama, DeepSeek | Snowflake Cortex (programmatic access token); sends `max_completion_tokens` |
| [`cloudflare`](https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/) | `CLOUDFLARE_API_TOKEN` or `CLOUDFLARE_API_KEY` | `CLOUDFLARE_ACCOUNT_ID` | Llama, Mistral, Qwen, ... | Cloudflare Workers AI; models start with @cf/ |
| [`oci`](https://docs.oracle.com/en-us/iaas/Content/generative-ai/openai-compatible-api.htm) | `OCI_GENAI_API_KEY` | `OCI_REGION` (default `us-chicago-1`) | Grok, Llama, gpt-oss | Oracle OCI Generative AI (API-key auth); Cohere models aren't served here |

## Gateways

| Provider | Key variable | URL settings | Models | Notes |
|---|---|---|---|---|
| [`litellm`](https://docs.litellm.ai/docs/proxy/quick_start) | `LITELLM_API_KEY` (optional) | - | anything the gateway is configured for | LiteLLM proxy: any model, including Bedrock Nova and Claude, in OpenAI format; model = proxy alias |
| [`portkey`](https://portkey.ai/docs/api-reference/inference-api/chat) | `PORTKEY_API_KEY` | - | anything in your Portkey catalog | Portkey gateway; model as @provider-slug/model; key sent in `x-portkey-api-key` header |
| [`vercel`](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions) | `AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` | - | anything on the gateway | Vercel AI Gateway; model as provider/model |
| [`helicone`](https://docs.helicone.ai/gateway/overview) | `HELICONE_API_KEY` | - | anything on the gateway | Helicone AI Gateway |
| [`cloudflare-gateway`](https://developers.cloudflare.com/ai-gateway/usage/chat-completion/) | `CLOUDFLARE_API_TOKEN` (optional) | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_GATEWAY_ID` (default `default`) | anything on the gateway | Cloudflare AI Gateway unified endpoint; model as provider/model; key = the upstream provider's key unless stored in the gateway |

## Local and self-hosted

| Provider | Key variable | URL settings | Models | Notes |
|---|---|---|---|---|
| [`ollama`](https://docs.ollama.com/api/openai-compatibility) | `OLLAMA_API_KEY` (optional) | - | Llama, Qwen, Gemma, gpt-oss, ... | Local Ollama (key only for Ollama Cloud) |
| [`lmstudio`](https://lmstudio.ai/docs/app/api/endpoints/openai) | `LM_API_TOKEN` (optional) | - | local models | LM Studio local server |
| [`vllm`](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) | `VLLM_API_KEY` (optional) | - | open models | vLLM server; model = served model name |
| [`llamacpp`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) | `LLAMA_API_KEY` (optional) | - | GGUF models | llama.cpp llama-server |
| [`localai`](https://localai.io/docs/features/text-generation/) | `LOCALAI_API_KEY` (optional) | - | local models | LocalAI |
| [`jan`](https://docs.jan.ai/desktop/api-server) | `JAN_API_KEY` (optional) | - | local models | Jan local API server |
| [`tgi`](https://huggingface.co/docs/text-generation-inference/messages_api) | `HF_TOKEN` (optional) | - | the served model | Hugging Face TGI Messages API; model is usually "tgi" |
| [`sglang`](https://docs.sglang.io/basic_usage/openai_api_completions.html) | `SGLANG_API_KEY` (optional) | - | open models | SGLang server |
| [`koboldcpp`](https://github.com/LostRuins/koboldcpp/wiki) | `KOBOLDCPP_PASSWORD` (optional) | - | GGUF models | KoboldCpp |
| [`gpt4all`](https://docs.gpt4all.io/gpt4all_api_server/home.html) | none | - | GPT4All models | GPT4All (enable the Local API Server in settings) |

## Retired or not OpenAI-compatible

| Name | Status |
|---|---|
| `github` | GitHub Models was retired on 2026-07-30; GitHub suggests Microsoft Foundry (use azure:). |
| `llama` | Meta's Llama API was replaced by the Meta Model API (use meta:), or run Llama via groq:, together:, cerebras:, ollama: and others. |
| `hyperbolic` | Hyperbolic retired its serverless inference API. |
| `lambda` | Lambda's Inference API was discontinued. |
| `writer` | Writer's chat API isn't OpenAI-compatible (/v1/chat); reach Palmyra through a gateway (litellm:) or a custom PromptClient. |
| `watsonx` | IBM watsonx.ai needs IAM token exchange and its OpenAI-style route is in preview; use a gateway (litellm:) for now. |
| `kong` | Kong AI Gateway routes are user-defined: use OpenAICompatibleClient(model, base_url=<your route>). |

## Anything else

Any other OpenAI-compatible server works too:

```python
pu.llm_client("openai-compatible:<model>")   # base URL from OPENAI_BASE_URL, key from OPENAI_API_KEY
pu.OpenAICompatibleClient("<model>", base_url="https://host/v1", api_key="...", headers={...})
```
