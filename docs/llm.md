# LLM operators

Use a model as a typed dataframe step. Steps before and after it still run on the engine.

```python
pu.set_llm(pu.llm_client("gemini:gemini-3.8-flash"))

tickets = (
    pu.read_parquet("tickets.parquet")
    .llm_filter("the customer complains about delivery", inputs=["body"])
    .llm_extract("the order number and a 1-5 severity",
                 fields={"order_no": "String", "severity": "Int64"}, inputs=["body"])
    .group_by("severity").agg(n=pu.len())
)
tickets.explain()["llm_steps"]     # max rows and calls, before anything is sent
tickets.collect()
pu.llm_reports()                   # rows, unique rows, cache hits, calls, tokens, cost, failures
```

| Operator | Adds | Example |
|---|---|---|
| `llm_filter(condition)` | keeps matching rows | `"mentions a competitor"` |
| `llm_map(instruction, output=, dtype=)` | one typed column | sentiment label, 1-5 score, `Decimal(10,2)` price |
| `llm_extract(instruction, fields={...})` | several typed columns | company, amount, date |

## Providers

pyursa works with **every major LLM provider**: 53 OpenAI-compatible services plus native
Anthropic and Amazon Bedrock. The **[full provider list](providers.md)** gives each one's key
variable, URL settings and documentation link. `pu.print_providers()` shows it in Python, and
`python scripts/check_providers.py` tests them from your machine.

```python
pu.llm_client("openai:<model>")                       # OPENAI_API_KEY
pu.llm_client("anthropic:claude-haiku-5-5")           # ANTHROPIC_API_KEY
pu.llm_client("gemini:gemini-3.8-flash")              # GEMINI_API_KEY or GOOGLE_API_KEY
pu.llm_client("bedrock:us.amazon.nova-2-lite-v1:0")   # Amazon Nova: Bedrock API key or AWS access keys
pu.llm_client("azure:<deployment>")                   # AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT
pu.llm_client("vertex:gemini-3.8-flash")              # GOOGLE_CLOUD_PROJECT + gcloud credentials
pu.llm_client("mistral:mistral-large-latest")         # MISTRAL_API_KEY
pu.llm_client("groq:llama-3.3-70b-versatile")         # GROQ_API_KEY
pu.llm_client("deepseek:<model>")                     # DEEPSEEK_API_KEY
pu.llm_client("dashscope:qwen-plus")                  # DASHSCOPE_API_KEY (Qwen)
pu.llm_client("openrouter:<vendor>/<model>")          # one key for many vendors
pu.llm_client("ollama:llama3.2")                      # local, no key
pu.llm_client("openai-compatible:<model>")            # anything else: OPENAI_BASE_URL (+ OPENAI_API_KEY)
```

What the client handles for you:

- **Authentication styles:** Bearer keys, Azure's `api-key` header, Portkey's header, Google
  OAuth tokens for Vertex (refreshed automatically), and for AWS: Bedrock API keys, access keys
  (signed with Signature V4, no SDK) or profiles and roles through boto3.
- **URL settings:** account IDs, regions, resource and workspace names come from environment
  variables (e.g. `CLOUDFLARE_ACCOUNT_ID`) or keyword arguments (`account_id="..."`).
- **Token-limit field:** OpenAI now deprecates `max_tokens` in favor of `max_completion_tokens`,
  and some providers accept only one of them. Each preset sends the right one, and if a server
  rejects it the client switches automatically and remembers.
- **Retired services:** asking for one (GitHub Models, Hyperbolic, Lambda, Meta's old Llama API)
  explains what replaced it instead of failing with a vague error.

### Claude in detail

| Route | Spec | Key | Notes |
|---|---|---|---|
| Anthropic, native | `anthropic:<model>` | `ANTHROPIC_API_KEY` | Recommended for production. Without the `anthropic` package installed it automatically uses the OpenAI-compatible route below. |
| Anthropic, OpenAI-compatible | `claude:<model>` | `ANTHROPIC_API_KEY` | No SDK. Anthropic describes this layer as mainly for testing and comparing models; it works for pyursa's plain-text prompts. Set `ANTHROPIC_WORKSPACE_ID` if your key spans several workspaces. |
| AWS Bedrock | `bedrock:us.anthropic.<model>` | `AWS_BEARER_TOKEN_BEDROCK` or IAM | Uses your AWS account and billing. |
| Gateways | `openrouter:anthropic/<model>`, `litellm:<name>` | gateway key | OpenAI format. |

### Amazon Nova in detail

Bedrock's own OpenAI-compatible endpoint **does not serve Nova** (or Claude, Llama and Titan),
according to AWS's API compatibility table. pyursa therefore reaches Nova through Bedrock's
native Converse API, with any of the usual AWS credentials:

| Credentials | How | SDK needed? |
|---|---|---|
| Bedrock API key | `AWS_BEARER_TOKEN_BEDROCK`, or `api_key=` | No |
| Access key and secret key | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN` for temporary keys), or `aws_access_key_id=` / `aws_secret_access_key=` / `aws_session_token=` | No: requests are signed with AWS Signature V4 using Python's standard library |
| Profiles, SSO, EC2/ECS/Lambda roles | `AWS_PROFILE` or `profile=` | Yes: `pip install boto3` |

The first one found wins, in that order; explicit arguments beat environment variables. The region
comes from `AWS_REGION` (default `us-east-1`) or `region=`.

```powershell
# PowerShell, with access keys
$env:AWS_ACCESS_KEY_ID="AKIA..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_REGION="us-east-1"
python scripts/e2e.py --only llm --llm bedrock:us.amazon.nova-2-lite-v1:0
```

```python
pu.llm_client("bedrock:us.amazon.nova-2-lite-v1:0",
              aws_access_key_id="AKIA...", aws_secret_access_key="...", region="us-east-1")
```

Gateways are an alternative if you want Nova in OpenAI format: `openrouter:amazon/nova-lite-v1`
or `litellm:<name>`.

Newer Bedrock models need a **cross-region inference profile id** (the `us.`, `eu.` or `apac.`
prefix) and model access enabled in the Bedrock console. If you use a plain id where a
profile is required, the error tells you which id to use.

### One endpoint for everything

If your organization wants every model behind one OpenAI-style URL, run a gateway such as
LiteLLM (it translates to Bedrock Converse for Nova and Claude) and point pyursa at it with
`litellm:<model name>` or `OpenAICompatibleClient(base_url=...)`.

## Cost and safety controls

| Control | Default | How |
|---|---|---|
| Deduplication | on | Identical inputs are sent once. Narrow `inputs` to cut distinct values. |
| Batching | 20 rows/request | `set_llm(batch_size=50)` |
| Caching | in memory | `set_llm(cache_path="llm.sqlite")` persists across runs. Invalid answers are never cached. |
| Row budget | 1,000 unique rows | `max_rows=` per step or in `set_llm`. Checked before any call. |
| Token budget | off | `set_llm(max_input_tokens=200_000)` refuses oversized steps before calling. |
| Cost report | tokens always | `set_llm(price_per_mtok=(0.10, 0.50))` adds `cost_usd`. |
| PII redaction | off | `set_llm(redact_pii=True)` or `redact=True` per step masks emails, phones, cards, SSNs, IPs. |
| Typed outputs | on | Answers are coerced to the declared type; anything else becomes null and counts as a failure. |
| Injection resistance | on | Rows go in a JSON data block the model is told never to follow. |

LLM steps appear in `history()`, `lineage()` and `explain()` like any other step.

## Custom models

Subclass `pu.PromptClient` and implement `complete(system, user) -> str` (optionally
`complete_with_usage` to report tokens), or wrap any Python function with
`pu.FunctionClient(fn)`.
