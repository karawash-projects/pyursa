# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow [SemVer](https://semver.org/).

## [1.0.0rc6] - 2026-10-21

### Fixed
- The DuckDB backend failed with DuckDB 2.0 previews, which removed Arrow fetching from connections
  (`pip install --pre` pulls those previews in). Results are now fetched through a relation when needed;
  the full test suite passes on DuckDB 2.0 previews, Polars 2.0 and PyArrow 26.

### Added
- CI job that runs the tests against the newest pre-release dependencies, to catch breaking upstream
  changes before they ship.

## [1.0.0rc5] - 2026-10-20

### Fixed
- Rate limits (HTTP 429): pyursa now waits as long as the provider asks (Google's `retryDelay` / "retry in Ns",
  as well as `Retry-After`); a used-up daily quota fails at once instead of retrying for over a minute, with a
  hint that the provider's quota, not pyursa, is the cause. `scripts/e2e.py` reports a provider quota as SKIP.
- Provider error messages wrapped in a list (Gemini's OpenAI layer) or in `detail`/`errors` fields were
  shown as a bare "Bad Request"; the provider's real message is now extracted.
- Bad-key errors reported as HTTP 400 (xAI, Gemini) now get the "check your API key" hint.
- `scripts/check_providers.py`: probes with real public model ids, says "needs settings" when an
  account ID or workspace is missing, recognizes bad-key replies of any status, treats "model not found"
  or "model retired" (HTTP 410) as a reachable endpoint, and checks that a local port really serves an OpenAI-compatible API.

## [1.0.0rc4] - 2026-10-20

### Added
- Bedrock accepts AWS access keys without any SDK: `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`
  (and `AWS_SESSION_TOKEN`), or `aws_access_key_id=` / `aws_secret_access_key=` / `aws_session_token=`.
  Requests are signed with AWS Signature V4 using the standard library; signatures are tested
  byte-for-byte against botocore. Works for `bedrock:` (Nova, Claude on AWS) and `bedrock-openai:`.
- `profile=` for named AWS profiles (through boto3).

## [1.0.0rc3] - 2026-10-20

### Added
- **53 OpenAI-compatible providers**, each checked against official docs: model vendors (OpenAI, Gemini,
  Mistral, Cohere, Meta, xAI, DeepSeek, Qwen/DashScope, Kimi, GLM, MiniMax, Perplexity, AI21, Reka, Upstage),
  hosts (Groq, Together, Fireworks, Cerebras, SambaNova, DeepInfra, Nebius, Novita, Baseten, OpenRouter,
  Hugging Face, NVIDIA, Scaleway, OVHcloud), clouds (Azure, Vertex AI, Bedrock OpenAI, Databricks, Snowflake,
  Cloudflare, Oracle OCI), gateways (LiteLLM, Portkey, Vercel, Helicone, Cloudflare AI Gateway) and local servers
  (Ollama, LM Studio, vLLM, llama.cpp, LocalAI, Jan, TGI, SGLang, KoboldCpp, GPT4All).
- `openai-compatible:` catch-all using `OPENAI_BASE_URL` / `OPENAI_API_KEY`.
- Auth styles (Bearer, custom headers, Google OAuth with refresh), URL settings from environment variables
  or keyword arguments, alternative key variables, `pu.print_providers()`, `scripts/check_providers.py`,
  generated `docs/providers.md`.
- Automatic `max_tokens` / `max_completion_tokens` negotiation.
- Clear messages for retired services (GitHub Models, Hyperbolic, Lambda, Meta Llama API).

### Fixed
- Together AI's base URL moved to api.together.ai; DeepSeek's documented base URL is api.deepseek.com.
- Default output limit lowered from 8192 to 4096 tokens, within every provider's cap.

## [1.0.0rc2] - 2026-10-20

### Added
- `claude:` provider: Claude through Anthropic's OpenAI-compatible endpoint (no SDK). `anthropic:` falls
  back to it automatically when the `anthropic` package isn't installed.
- `bedrock:` works with just a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`) over HTTPS, no boto3 needed;
  covers Amazon Nova and Claude on AWS. Region from `AWS_REGION` or `region=`. Clear hints for
  inference-profile ids and model access.
- `bedrock-openai:` (Bedrock's OpenAI-compatible endpoint, for gpt-oss, Qwen, DeepSeek) with a guard that
  redirects Nova/Claude/Llama to `bedrock:`; `litellm:` gateway preset; `headers=` on
  `OpenAICompatibleClient`; `pu.list_providers()`.

### Changed
- Bedrock default `max_tokens` is 4096 (fits every Nova model).

## [1.0.0rc1] - 2026-10-18

Release candidate for 1.0. The public API described in `docs/stability.md` is frozen
unless release-candidate feedback finds a problem.

### Added
- **Types:** exact `Decimal(p, s)` with defined arithmetic scales; time-zone-aware
  `Datetime(tz)` with local-time `.dt` parts; `List`/`Struct` pass-through; dictionary
  columns decode to `String`; type strings such as `"Decimal(12,2)"`.
- **Expressions:** `when/then/otherwise`, `fill_null`, `coalesce`, `is_between`, `abs`,
  `round` (half away from zero), string functions (`contains_regex`, `ends_with`, `strip`,
  `len`, `replace`, `slice`), date functions (`year`…`second`, `weekday`, `date`, `truncate`).
- **Aggregations:** `median`, `quantile`, `std`, `var`, `any`, `all`.
- **Window functions:** aggregations `.over(partition)`, `row_number`, `rank`, dense rank,
  `cum_sum`, `shift`, with deterministic tie order.
- **Operations:** full, semi and anti joins; `distinct`; `concat`.
- **I/O:** `write_parquet`, `write_csv`, `write_json`; `read_json`; multi-file, glob and
  directory Parquet scans; `read_csv(schema=..., delimiter=...)`.
- **LLM:** providers for OpenAI-compatible APIs (GPT, Gemini, Groq, Together, Mistral,
  DeepSeek, xAI, OpenRouter, Ollama, LM Studio, vLLM) and Bedrock (Amazon Nova);
  token usage and cost reports; token budgets; PII redaction; per-session LLM settings.
- **MCP:** per-client session isolation (stdio and HTTP), `--host/--port`,
  `--redact-pii`, `--llm-price`, `--llm-max-input-tokens`, `drop_frame` tool.
- **Tools:** `distinct` and `concat` steps; glob/directory/JSON loading inside the sandbox;
  `preview` returns exact numbers and JSON rows.
- Agent benchmark harness (`benchmarks/agent_bench.py`), API reference generator,
  docs site, CI matrix, security policy, Apache-2.0 license.

### Changed
- Renamed from the working name `agentframe` to **pyursa** (`import pyursa as pu`, `pyursa-mcp`,
  `PyursaError`, reserved prefix `__pu_`).
- Casts follow a strict contract: invalid text, non-whole floats to Int64, and casts without a
  single meaning now raise errors (see docs/types.md).
- `DType` is now a class (constants such as `pu.Int64` are unchanged).
- DuckDB backend binds every user value as a query parameter.
- DuckDB backend skips ordering work when a later step makes order irrelevant
  (about 2.3× faster on the end-to-end pipeline).
- CSV empty fields are read as null by default (`empty_as_null=False` restores `""`).

### Fixed (independent pre-release review)
- DuckDB resolved columns differing only by case (`A`, `a`) to the same column; such names are now rejected.
- Paths containing `[`, `*` or `?` were globbed again after the sandbox check, which could read other files.
- Engine panics (Rust `PanicException`) could escape `ToolSession` instead of returning an error.
- Polars wrapped around on Int64 overflow; overflow now raises `INTEGER_OVERFLOW` on every engine.
- Decimal literals written with an exponent (`1E+3`) lost the exponent on DuckDB.
- Casts behaved differently per engine (float→int truncation vs rounding, text parsing); one strict
  cast contract now applies everywhere, with `CAST_FAILED` errors.
- NaN handling differed between engines and between DuckDB's in-memory and Parquet paths; NaN is
  now the largest value and equal to itself everywhere.
- Hour truncation of zoned datetimes on DuckDB was wrong across a DST fall-back.
- Columns named with the internal `__pu_` prefix broke results; the prefix is now reserved.
- Malformed LLM answers could crash a step; they now become null failures. Large integer answers are exact.
- `is_in` failed on Polars for mixed numeric types; `is_in([])` of null now returns null everywhere.
- Null-typed columns failed on DuckDB.
- Several error `fix` snippets didn't work when pasted; literal values now print in parseable form.
- Bad tool arguments returned `INTERNAL_ERROR`; they now return `BAD_ARGUMENTS`. More inputs are
  validated when the plan is built (`head` bounds, `rename` values, `checkout` ids, token budgets).
- Fixed-offset time zones (`+05:00`) failed on DuckDB; whole-hour offsets map to `Etc/GMT` zones.
- Zoned literals at nonexistent local times (DST gaps) failed on Polars.
- (Second review round) Engines expanded glob characters in already-resolved file paths, so a
  sandboxed `data[1]/a.parquet` could read `data1/a.parquet`; files are now read literally.
- (Second review round) DuckDB's optimizer could drop the overflow check when an Int64 result fed a
  comparison; overflow is now checked explicitly.
- (Second review round) std/var failed on DuckDB when a NaN was present; text casts disagreed on
  `24:00`, leap seconds, year 0000 and rounding that adds a digit; non-portable regex (lookaround,
  backreferences) is now rejected when the plan is built; error codes no longer depend on column names;
  different `FunctionClient` functions no longer share cached answers; a literal file name such as
  `x[1].parquet` is no longer treated as a pattern.

### Fixed
- Polars truncated decimal products (`1.15 × 1.15 → 1.32`); results are now exact everywhere.
- Previews rounded floats to 4 significant digits.
- Instruction-like text was only flagged when it appeared in displayed values; the whole
  column is now scanned.
- String values containing NUL bytes broke DuckDB queries.
- Dates loaded from a persistent LLM cache came back as strings.
- `frame.agg()` and `select()` didn't accept keyword names.
- Some parser failures surfaced as raw Python exceptions instead of structured errors.

## [0.4.0] - 2026-10-18
- Multi-provider LLM clients; `pu.llm_client("provider:model")`.

## [0.3.0] - 2026-10-18
- `llm_filter`, `llm_map`, `llm_extract` with dedup, batching, caching and budgets.

## [0.2.0] - 2026-10-18
- MCP server and handle-based tool layer; safe expression parser.

## [0.1.0] - 2026-10-18
- Initial IR, Polars and DuckDB backends, structured errors, history, lineage,
  explain, `describe_for_llm`.
