# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow [SemVer](https://semver.org/).

## [1.1.0] - 2026-11-06

Fixes from the 1.0.1 pre-production review, plus three more silent-corruption bugs found while
verifying it. Several bugs returned wrong answers without an error; upgrading is strongly recommended.

### Security
- **The MCP server needs `--root`** and `ToolSession` needs `roots=`; there is no default folder any more
  (it used to be the current directory). To keep the old behavior: `--root .`.
- **Sessions are read-only by default.** `export` needs `--allow-write` / `ToolSession(allow_write=True)`,
  and writes only `.parquet`, `.csv` and `.json` files.
- **Hidden files and folders** (`.env`, `.git`, `.secrets.json`, ...) are never readable or writable.
- **Glob patterns are expanded only inside a root.** Absolute or `..` patterns used to be expanded before
  the sandbox check, revealing which files exist elsewhere (`/etc/pass*`) and walking the whole disk.
- **API keys never follow redirects.** A 3xx answer from a provider is an error; urllib used to re-send
  the `Authorization` / `api-key` header to whatever host the redirect named.
- **Prompt data can't break out of its block:** `<`, `>` and `&` are escaped in the JSON sent to models.
- **Column and dataset names are escaped in summaries** like cell values, and instruction-like names are
  flagged (a CSV header could inject lines into `describe_for_llm`).
- MCP sessions are freed when a client disconnects, tool calls within a session run one at a time, and a
  call without a client connection is refused instead of sharing one fallback session.

### Fixed (wrong results)
- Parquet columns kept their file types on Polars: category columns failed string functions, joins and
  concat; int32 and millisecond timestamps couldn't be combined with other data; float32 math differed
  between engines. Both engines now read file columns as their declared type.
- DuckDB dropped NaN rows from Parquet files (`x > 100` and `x == NaN` lost rows) because Parquet
  min/max statistics ignore NaN. Older Polars did the same; neither uses those statistics for float files now.
- `read_csv` turned `"NA"`, `"null"`, `"nan"` and similar text into null. Only empty fields are missing now.
- `FunctionClient` gave two closures, bound methods or callable objects the same cache key, so one
  function's answers were served for another. Keys now include the captured values.
- DuckDB could wrap a decimal `sum()` around to a wrong (negative) number when combining partial sums,
  and Polars 1.x wrapped decimal `+`, `-`, `*` and `sum()`. Results that need more than 38 digits now raise
  the new `DECIMAL_OVERFLOW` error on every engine and version.
- Decimal `+` and `-` kept the inputs' precision (`Decimal(37,2) + Decimal(37,2)` failed); they now
  return `Decimal(38, s)` as documented.
- Polars rounded decimal products to the larger operand scale; products are exact on every engine.
- Comparing a wide decimal with a literal that has more decimal places (`Decimal(38,0) > 1.00001`) failed
  with CAST_FAILED; it is now exact.
- `contains_regex` with `\d`, `\w`, `\s` or `\b` matched different characters on each engine; these
  classes are now rejected with a suggested replacement such as `[0-9]`.

### Fixed (errors and robustness)
- One failed LLM call threw away the answers of every other call in the step. Failures now cost only their
  own rows, are reported (`failed_calls`, `errors`), and all good answers are cached.
- Inside tool sessions, previews, counts and summaries of a frame reuse its LLM results instead of
  calling the model again; the in-memory LLM cache is bounded (200,000 answers by default).
- `write_json` wrote `NaN`/`Infinity`, which strict JSON parsers reject, and disagreed with MCP `export`.
  Both write strict JSON now (non-finite floats as null).
- Decimal overflow was reported as `INTEGER_OVERFLOW`; it is `DECIMAL_OVERFLOW` now.
- `read_parquet(Path(...))`, `read_parquet([])` and `filter("x == 1")` crashed with Python errors; they
  work or give structured errors now. `set_llm()` and per-step limits are validated, and `max_rows=0` /
  `batch_size=0` are no longer silently ignored.
- `describe_for_llm` skips the top-values query for columns where every value is unique (IDs).
- CSV/JSON loads in tool sessions are capped (`max_load_bytes`, `--max-load-mb`), because those formats
  are read into memory; `export` checks the file type before running the query.
- Unknown time zones and offsets outside -12:00..+14:00 are rejected; fixed offsets are normalized in every
  `Datetime` type. `describe_for_llm` has its docstring again; paste-ready fixes quote column names safely.

### Fixed (independent re-review of the fixes)
- Reusing LLM results could return stale rows after a source file changed; reuse now happens only inside
  tool sessions and only while the files are unchanged. Plain `collect()` always re-runs the step.
- `FunctionClient` ids ignored state they couldn't print exactly (NumPy arrays, pandas objects, objects
  with custom `__repr__` or `__slots__`, global helpers and objects); such functions now get a one-off id.
  Ids no longer depend on the hash seed.
- Parquet `date64` columns were read as datetimes on Polars.
- Nanosecond timestamps in Parquet files were silently cut to microseconds; values with sub-microsecond
  parts now raise `UNSUPPORTED_TYPE`, like in-memory data.
- Struct field names and line-separator characters (U+2028/U+2029 and others) could inject lines into
  `describe_for_llm`; the smallest summary printed the dataset name unescaped.
- Time-zone aliases such as `US/Eastern` were rejected on systems whose tz database lacks them.
- `llm_filter` dropped rows whose model call failed without telling anyone; this now raises an
  `LLMPartialFailureWarning`, and tool results include `llm_warnings`.
- Loading a folder failed if it contained a hidden file; hidden files and links to them are skipped.
- Glob patterns followed symlinked folders out of the root (revealing file names and walking the
  target); patterns no longer leave the roots and visit at most 200,000 entries.
- On Windows, a pattern such as `/**/*` is rooted but not absolute and was expanded against the drive.
- On Polars 2.1 the decimal overflow check was optimized away.
- `read_csv(empty_as_null=False)` turned a column of empty fields into nulls; `schema=` conversion errors
  are structured errors.
- DuckDB 1.2 and 1.3 returned rows in a different order for filters with `|` or `is_in`, and a different
  median when NaN and infinity were present. **The minimum DuckDB version is now 1.4.**
- Remaining engine differences (Polars 1.x decimal limits, `-0.0` vs `0.0`) are listed in
  `pu.SEMANTICS["known_differences"]`.

### Fixed (end-to-end journeys)
- A Parquet folder written by different tools (PyArrow dictionary columns, Polars categoricals, DuckDB,
  int32 vs int64, ms vs us timestamps) was refused as "different schemas". Files now load together as long
  as they have the same pyursa types; each file is conformed on read.
- DuckDB 2.0 printed a warning on every query (an extra copy of the result); results are fetched without it.
- Without an engine installed, queries failed with a raw `ImportError`; they now raise `ENGINE_NOT_INSTALLED`
  with `pip install "pyursa[all]"`, and the default engine falls back to whichever one is installed.
- `describe_for_llm` scanned each text column once per prompt-injection marker; one pass now (4× faster
  on 3 million rows).
- PII redaction missed a phone number at the end of a sentence ("call 416-555-0199.").
- Per-row `FunctionClient` errors were missing from `llm_reports()[...]["errors"]`.
- `llm_client("anthropic:...", base_url=..., timeout=...)` raised a raw `TypeError`.

### Added
- `scripts/journeys/`: end-to-end user journeys (analyst, data engineer, error recovery, LLM pipelines,
  provider HTTP paths, MCP over stdio and HTTP, scale, packaging) that run against the installed wheel.
  CI runs them on Linux, macOS and Windows with the newest and the oldest supported dependencies.

### Changed
- Development status is back to Beta until a follow-up review comes back clean.
- The version is defined once (`pyursa.__version__`); releases check that the tag matches and run the tests.
- New Parquet-file conformance tests read category, int8–32, uint8/32, float32, NaN, s/ms/zoned timestamps,
  dates, decimals, booleans and lists from files on every engine and compare them with in-memory data.

## [1.0.1] - 2026-10-21

### Fixed
- `pu.set_llm("provider:model")` stored the string instead of creating a client, so LLM steps failed
  later; strings are now turned into clients (`pu.set_llm("gemini:gemini-3.8-flash")`).

## [1.0.0] - 2026-10-21

First stable release. Same code as 1.0.0rc6; the public API now follows SemVer (see docs/stability.md).

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
