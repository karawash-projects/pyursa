# API reference

_Generated from the code by `scripts/gen_api_docs.py`; do not edit by hand._

Version 1.0.0rc5.

## Creating frames

### `pu.read_parquet(path: str | Sequence[str], name: str | None = None) -> Frame`

Lazily scan Parquet: a file, a list of files, a glob ("data/*.parquet") or a directory.

### `pu.read_csv(path: str, name: str | None = None, *, empty_as_null: bool = True, schema: dict[str, DType | str] | None = None, delimiter: str = ,) -> Frame`

Read a CSV with a single, engine-independent type inference (PyArrow).

Empty fields become null (like pandas); pass empty_as_null=False to keep them as "".
Pass schema={"col": "Decimal(12,2)", ...} to fix column types instead of inferring them.

### `pu.read_json(path: str, name: str | None = None) -> Frame`

Read newline-delimited JSON (one object per line) with engine-independent type inference.

### `pu.from_dict(data: dict[str, list[Any]], name: str = data) -> Frame`

### `pu.from_arrow(table: pa.Table, name: str = data) -> Frame`

### `pu.from_pandas(df, name: str = data) -> Frame`

### `pu.from_polars(df, name: str = data) -> Frame`

### `pu.concat(frames: Sequence[Frame], intent: str | None = None) -> Frame`

Stack frames vertically (same column names and types).

## Expressions

### `pu.col(name: str) -> Col`

Reference a column.

### `pu.lit(value: Any) -> Lit`

A literal value.

### `pu.len() -> Agg`

Row count aggregation (counts every row, including nulls).

### `pu.when(condition: Any) -> _When`

Conditional expression: pu.when(col("x") > 0).then("pos").otherwise("non-pos").

### `pu.coalesce(*exprs: Any) -> Coalesce`

First non-null value among the arguments.

### `pu.row_number() -> WinFn`

1, 2, 3... within each partition; use with .over(..., order_by=...). Ties keep input order.

### `pu.rank(dense: bool = False) -> WinFn`

SQL RANK (or DENSE_RANK) by the window's single order_by column; ties share a rank, nulls rank last.
Use as pu.rank().over("team", order_by="score", descending=True).

## Frame

### `Frame.schema(self) -> dict[str, DType]`

Column name -> type of the frame's output. Known without executing anything.

### `Frame.columns(self) -> list[str]`

Output column names, in order.

### `Frame.plan(self) -> P.Node`

The logical plan (IR) behind this frame.

### `Frame.filter(self, *predicates: Expr, intent: str | None = None) -> "Frame"`

Keep rows where every condition is true (null counts as false). Conditions are combined with &.

### `Frame.select(self, *exprs: Any, intent: str | None = None, **named: Any) -> "Frame"`

Choose or compute columns. Keyword arguments name new columns: select("id", total=col("a") + col("b")). Selecting only aggregations gives a one-row summary.

### `Frame.with_columns(self, *exprs: Expr, intent: str | None = None, **named: Expr) -> "Frame"`

Add or replace columns, keeping the others: with_columns(revenue=col("qty") * col("price")). Window expressions are allowed.

### `Frame.group_by(self, *keys: Any) -> "GroupBy"`

Start a grouped aggregation; finish it with .agg(...). Output is sorted by the keys.

### `Frame.agg(self, *aggs: Expr, intent: str | None = None, **named: Expr) -> "Frame"`

Aggregate the whole frame into one row: agg(total=col("x").sum(), n=pu.len()).

### `Frame.sort(self, *by: Any, descending: bool | Sequence[bool] = False, intent: str | None = None) -> "Frame"`

Stable sort by one or more columns; nulls always last.

### `Frame.head(self, n: int = 5, intent: str | None = None) -> "Frame"`

First n rows (in the current row order).

### `Frame.limit(self, n: int = 5, intent: str | None = None) -> "Frame"`

First n rows (in the current row order).

### `Frame.join(self, other: "Frame", on: str | Sequence[str], how: str = inner, suffix: str = _right, intent: str | None = None) -> "Frame"`

how: "inner", "left", "full", "semi" (left rows with a match) or "anti" (left rows without one).

### `Frame.rename(self, mapping: dict[str, str], intent: str | None = None) -> "Frame"`

Rename columns: rename({"old": "new"}).

### `Frame.drop(self, *columns: Any, intent: str | None = None) -> "Frame"`

Remove columns.

### `Frame.distinct(self, subset: str | Sequence[str] | None = None, intent: str | None = None) -> "Frame"`

Remove duplicate rows (or rows duplicated on `subset`), keeping the first occurrence in order.

### `Frame.unique(self, subset: str | Sequence[str] | None = None, intent: str | None = None) -> "Frame"`

Remove duplicate rows (or rows duplicated on `subset`), keeping the first occurrence in order.

### `Frame.concat(self, *others: "Frame", intent: str | None = None) -> "Frame"`

Append the rows of other frames (same column names and types) after this one.

### `Frame.write_parquet(self, path: str, backend: str | None = None, overwrite: bool = False) -> int`

Execute and write a Parquet file. Returns the number of rows written.

### `Frame.write_csv(self, path: str, backend: str | None = None, overwrite: bool = False) -> int`

Execute and write a CSV file (nested columns are not supported in CSV).

### `Frame.write_json(self, path: str, backend: str | None = None, overwrite: bool = False) -> int`

Execute and write newline-delimited JSON (one object per row).

### `Frame.llm_filter(self, condition: str, *, inputs: Sequence[str] | None = None, client: Any = None, max_rows: int | None = None, batch_size: int | None = None, intent: str | None = None, redact: bool | None = None, config: Any = None) -> "Frame"`

Keep rows the model judges to satisfy `condition`, e.g. "the review complains about delivery".

### `Frame.llm_map(self, instruction: str, *, output: str, dtype: DType | str = String, inputs: Sequence[str] | None = None, client: Any = None, max_rows: int | None = None, batch_size: int | None = None, intent: str | None = None, redact: bool | None = None, config: Any = None) -> "Frame"`

Add one typed column computed by the model, e.g. a sentiment label or a 1-5 score.

### `Frame.llm_extract(self, instruction: str, *, fields: dict[str, DType | str], inputs: Sequence[str] | None = None, client: Any = None, max_rows: int | None = None, batch_size: int | None = None, intent: str | None = None, redact: bool | None = None, config: Any = None) -> "Frame"`

Add several typed columns extracted from text, e.g. fields={"company": "String", "amount": "Float64"}.

### `Frame.history(self) -> list[dict[str, Any]]`

Every step from the source to this frame, oldest first.

### `Frame.undo(self, steps: int = 1) -> "Frame"`

Return the frame as it was `steps` transformations ago. Instant: no data is reloaded.

### `Frame.checkout(self, step_id: str) -> "Frame"`

Return the frame at a given step id from history().

### `Frame.lineage(self, column: str | None = None) -> dict[str, Any]`

Column-level provenance: for each column, the source columns, filters, joins and groupings behind it.

### `Frame.explain(self, backend: str | None = None) -> dict[str, Any]`

Dry run: what would execute, the output schema and a size estimate. Runs nothing.

### `Frame.collect(self, backend: str | None = None) -> pa.Table`

Execute on a backend (default: pu.default_backend()) and return Arrow.

### `Frame.to_arrow(self, backend: str | None = None) -> pa.Table`

Same as collect().

### `Frame.to_polars(self, backend: str | None = None)`

Execute and return a polars.DataFrame.

### `Frame.to_pandas(self, backend: str | None = None)`

Execute and return a pandas.DataFrame.

### `Frame.to_dicts(self, backend: str | None = None) -> list[dict[str, Any]]`

Execute and return a list of row dictionaries.

### `Frame.count(self, backend: str | None = None) -> int`

Execute and return the number of rows.

### `Frame.profile(self, backend: str | None = None, **kwargs: Any) -> dict[str, Any]`

Per-column statistics as a dict (row count, nulls, distinct, min/max/mean, top values, sample rows), computed by the engine.

### `Frame.describe_for_llm(self, tokens: int = 800, backend: str | None = None, **kwargs: Any) -> str`

## GroupBy

### `GroupBy.agg(self, *aggs: Expr, intent: str | None = None, **named: Expr) -> Frame`

Aggregations per group: agg(total=col("x").sum(), n=pu.len()). Output is sorted by the group keys, nulls last.

## Expression methods

### `Expr.alias(self, name: str) -> "Expr"`

### `Expr.is_null(self) -> "Expr"`

### `Expr.is_not_null(self) -> "Expr"`

### `Expr.is_in(self, values: Iterable[Any]) -> "Expr"`

### `Expr.is_between(self, lower: Any, upper: Any) -> "Expr"`

Inclusive on both ends.

### `Expr.cast(self, dtype: DType | str) -> "Expr"`

### `Expr.fill_null(self, value: Any) -> "Expr"`

### `Expr.abs(self) -> "Expr"`

### `Expr.round(self, decimals: int = 0) -> "Expr"`

Round half away from zero (2.5 -> 3, -2.5 -> -3) on every engine.

### `Expr.sum(self) -> "Expr"`

### `Expr.mean(self) -> "Expr"`

### `Expr.min(self) -> "Expr"`

### `Expr.max(self) -> "Expr"`

### `Expr.count(self) -> "Expr"`

### `Expr.n_unique(self) -> "Expr"`

### `Expr.median(self) -> "Expr"`

### `Expr.quantile(self, q: float) -> "Expr"`

Continuous quantile with linear interpolation, q in [0, 1].

### `Expr.std(self) -> "Expr"`

Sample standard deviation (ddof=1).

### `Expr.var(self) -> "Expr"`

Sample variance (ddof=1).

### `Expr.any(self) -> "Expr"`

### `Expr.all(self) -> "Expr"`

### `Expr.cum_sum(self) -> "Expr"`

Running total; needs .over(..., order_by=...). Nulls are skipped (SQL semantics).

### `Expr.shift(self, n: int = 1) -> "Expr"`

Value n rows earlier (n<0: later) within the window order; needs .over(order_by=...).

### `Expr.over(self, partition_by: str | Sequence[str] | None = None, *, order_by: str | Sequence[str] | None = None, descending: bool = False) -> "Expr"`

Evaluate per partition without collapsing rows, e.g. col("x").sum().over("region").

### `Expr.str(self) -> "_StrNamespace"`

### `Expr.dt(self) -> "_DtNamespace"`

## String methods (`col(...).str`)

### `str.contains(self, s: str) -> Expr`

### `str.contains_regex(self, pattern: str) -> Expr`

Regex search. Use common syntax (RE2/Rust); lookarounds and backreferences are not portable.

### `str.starts_with(self, s: str) -> Expr`

### `str.ends_with(self, s: str) -> Expr`

### `str.lower(self) -> Expr`

### `str.upper(self) -> Expr`

### `str.strip(self) -> Expr`

### `str.len(self) -> Expr`

Length in characters.

### `str.replace(self, old: str, new: str) -> Expr`

Replace every literal occurrence of `old`.

### `str.slice(self, start: int, length: int | None = None) -> Expr`

0-based substring.

## Date and time methods (`col(...).dt`)

### `dt.year(self) -> Expr`

### `dt.month(self) -> Expr`

### `dt.day(self) -> Expr`

### `dt.weekday(self) -> Expr`

ISO weekday: Monday=1 ... Sunday=7.

### `dt.hour(self) -> Expr`

### `dt.minute(self) -> Expr`

### `dt.second(self) -> Expr`

### `dt.date(self) -> Expr`

### `dt.truncate(self, unit: str) -> Expr`

Start of the year/month/week (ISO, Monday)/day/hour.

## Types

### `pu.Decimal(precision: int = 38, scale: int = 0) -> DType`

Exact decimal type, e.g. Decimal(12, 2) for money.

### `pu.List(inner: "DType | str") -> DType`

### `pu.Struct(fields: "dict[str, DType | str]") -> DType`

Constants: `pu.Int64`, `pu.Float64`, `pu.String`, `pu.Boolean`, `pu.Date`, `pu.Datetime` (call it for a zone: `pu.Datetime("UTC")`), `pu.Null`. Strings work too: `"Decimal(12,2)"`.

## Backends

### `pu.available_backends() -> list[str]`

### `pu.default_backend() -> str`

### `pu.set_default_backend(name: str) -> None`

### `pu.register_backend(name: str, factory: Callable[[], Backend]) -> None`

### `pu.get_backend(name: str | None = None) -> Backend`

## LLM operators

### `pu.set_llm(client: LLMClient | None = None, *, cache_path: str | None = None, max_rows: int | None = None, batch_size: int | None = None, concurrency: int | None = None, max_input_tokens: int | None = None, price_per_mtok: tuple[float, float] | None = None, redact_pii: bool | None = None) -> None`

Configure the default model used by llm_* operators and its limits.

### `pu.llm_client(spec: str, **kwargs: Any) -> PromptClient`

Create a client from "provider:model". See list_providers() for every provider.

Examples: "anthropic:claude-haiku-5-5", "claude:claude-haiku-5-5", "bedrock:us.amazon.nova-2-lite-v1:0",
"bedrock:us.anthropic.claude-haiku-5-5", "openrouter:amazon/nova-lite-v1", "gemini:gemini-3.8-flash",
"openai:<model>", "litellm:<model>", "ollama:llama3.2".

### `pu.llm_reports() -> list[dict[str, Any]]`

Usage reports from the most recent execution in this thread: rows, cache hits, calls, tokens, cost.

### `pu.list_providers() -> list[dict[str, Any]]`

Every provider llm_client() accepts, with its endpoint, key variable and typical models.

### `pu.print_providers() -> None`

Print a readable table of every provider.

## LLM clients

### `pu.OpenAICompatibleClient(model: str, *, provider: str | None = None, base_url: str | None = None, api_key: str | None = None, max_tokens: int = 4096, json_mode: bool = False, timeout: float = 120.0, max_attempts: int = 4, extra_body: dict[str, Any] | None = None, headers: dict[str, str] | None = None, region: str | None = None, token_param: str | None = None, query: dict[str, str] | None = None, **params: str) -> None`

Any server implementing POST {base_url}/chat/completions.

Use a preset (provider="groq") or any URL (base_url="https://host/v1"). URL placeholders such as
{account_id} or {resource} can be passed as keyword arguments or come from the environment.

### `pu.AnthropicClient(model: str = claude-haiku-5-5, max_tokens: int = 8192, api_key: str | None = None, client: Any = None) -> None`

Claude via the Anthropic API (`pip install anthropic`, ANTHROPIC_API_KEY set).

### `pu.BedrockClient(model: str, *, region: str | None = None, max_tokens: int = 4096, api_key: str | None = None, aws_access_key_id: str | None = None, aws_secret_access_key: str | None = None, aws_session_token: str | None = None, profile: str | None = None, client: Any = None, endpoint_url: str | None = None, timeout: float = 120.0, max_attempts: int = 4) -> None`

Amazon Bedrock Converse API: Amazon Nova, Claude, Llama, Mistral and every other Bedrock text model.

Authentication, first match wins (no SDK needed for 1-3):
1. api_key= or AWS_BEARER_TOKEN_BEDROCK: a Bedrock API key;
2. aws_access_key_id= / aws_secret_access_key= (and aws_session_token=);
3. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (and AWS_SESSION_TOKEN) environment variables;
4. boto3, if installed: profiles (AWS_PROFILE), SSO, instance and container roles.
Requests for 2-3 are signed with AWS Signature V4 using the standard library.
Region: region= or AWS_REGION / AWS_DEFAULT_REGION (default us-east-1).
Newer models need a cross-region inference profile id such as "us.amazon.nova-2-lite-v1:0"
or "us.anthropic.claude-haiku-5-5"; enable model access in the Bedrock console first.

### `pu.FunctionClient(fn: Callable[[dict[str, Any]], Any], model_id: str | None = None) -> None`

Wrap any Python callable (another model, a rules engine, a test stub).
`fn(row) -> dict | scalar`; a scalar is used for single-field tasks.

### `pu.PromptClient()`

Base for text-in/text-out models: builds the prompt and parses the JSON reply.
Subclasses implement complete(system, user) -> str.

## Agent tools (`pyursa.tools.ToolSession`)

### `ToolSession(roots: list[str | os.PathLike] | None = None, backend: str | None = None, llm: Any = None, redact_pii: bool = False, max_frames: int = 1000) -> None`

### `help(self) -> dict[str, Any]`

### `load_file(self, path: str, name: str | None = None) -> dict[str, Any]`

### `list_frames(self) -> dict[str, Any]`

### `describe(self, handle: str, tokens: int = 800) -> dict[str, Any]`

### `transform(self, handle: str, steps: list[dict[str, Any]], intent: str | None = None) -> dict[str, Any]`

Apply steps in order; returns a NEW handle. The input handle is unchanged.

### `preview(self, handle: str, rows: int = 10) -> dict[str, Any]`

### `llm_usage(self) -> dict[str, Any]`

Cost report for LLM steps in the most recent execution.

### `count_rows(self, handle: str) -> dict[str, Any]`

### `explain(self, handle: str) -> dict[str, Any]`

### `history(self, handle: str) -> dict[str, Any]`

### `undo(self, handle: str, steps: int = 1) -> dict[str, Any]`

### `lineage(self, handle: str, column: str | None = None) -> dict[str, Any]`

### `export(self, handle: str, path: str, overwrite: bool = False) -> dict[str, Any]`

### `set_backend(self, backend: str) -> dict[str, Any]`

### `drop_frame(self, handle: str) -> dict[str, Any]`

## Errors

Every error is an `pu.PyursaError` with `.code`, `.message`, `.details`, `.suggestions`, `.to_dict()` and `.to_json()`.

- `pu.BackendError` — code `BACKEND_ERROR`
- `pu.ColumnNotFoundError` — code `COLUMN_NOT_FOUND`
- `pu.DuplicateColumnError` — code `DUPLICATE_COLUMN`
- `pu.HistoryError` — code `HISTORY_ERROR`
- `pu.InvalidOperationError` — code `INVALID_OPERATION`
- `pu.LLMBudgetError` — code `LLM_BUDGET_EXCEEDED`
- `pu.LLMNotConfiguredError` — code `LLM_NOT_CONFIGURED`
- `pu.LLMProviderError` — code `LLM_PROVIDER_ERROR`
- `pu.PyursaError` — code `PYURSA_ERROR`
- `pu.TypeMismatchError` — code `TYPE_MISMATCH`
- `pu.UnknownBackendError` — code `UNKNOWN_BACKEND`
- `pu.UnsupportedTypeError` — code `UNSUPPORTED_TYPE`

## Semantics

The rules every backend follows (`pu.SEMANTICS`):

- **null_comparison**: Comparisons with null yield null; filter() drops rows where the condition is null.
- **division**: '/' always returns Float64 and follows IEEE 754 (x/0 = inf, 0/0 = NaN).
- **sum_of_nulls**: sum() over only nulls (or no rows) returns null, not 0.
- **n_unique**: n_unique() does not count null as a value.
- **count**: count() counts non-null values; pu.len() counts rows.
- **sort**: sort() is stable and places nulls last in both directions.
- **group_by_order**: group_by().agg() output is sorted by the keys ascending, nulls last.
- **join_order**: join() keeps left row order, then right row order within matches; full joins put unmatched right rows last; null keys never match (anti joins keep null-key rows).
- **distinct**: distinct() keeps the first occurrence of each row (or subset) in input order; nulls compare equal.
- **concat**: concat() returns all rows of the first frame, then the second, and so on.
- **rounding**: round() rounds half away from zero (2.5 -> 3, -2.5 -> -3).
- **decimals**: Decimal +,-,* are exact (scale: max for +/-, sum for *); / returns Float64; sum keeps the scale; mean/median/std return Float64.
- **quantiles**: median()/quantile(q) use linear interpolation; std()/var() are sample statistics (ddof=1).
- **windows**: Window results keep input row order; ties in order_by keep input order; rank() is SQL RANK with nulls last; cum_sum skips nulls.
- **time_zones**: Datetime(tz) keeps its zone; .dt parts are computed in that zone; comparing different zones is an error.
- **strings**: str.len counts characters; str.strip removes spaces, tabs and newlines; str.slice is 0-based.
- **row_order**: filter/select/with_columns/limit preserve input row order.
- **nan**: NaN is the largest float value and equals itself (in comparisons, sorting, grouping and min/max); max() returns NaN when one is present; std/var with an infinite value return NaN.
- **integer_overflow**: Int64 arithmetic, sums and running totals that leave the Int64 range raise INTEGER_OVERFLOW; they never wrap around.
- **casts**: Casts are strict and identical everywhere: text is trimmed of spaces/tabs/newlines and must then be a valid literal (integers like '+12', floats like '1e3', 'inf' or 'nan', decimals with up to 20 integer digits and 17 places, booleans 'true'/'false'/'1'/'0', dates 'YYYY-MM-DD', datetimes 'YYYY-MM-DD HH:MM[:SS[.ffffff]]' with ' ' or 'T', years 0001-9999, hours 00-23, no leap seconds); anything else raises CAST_FAILED. Float or Decimal to Int64 requires a whole number. Conversions to Decimal round half away from zero. Floats print as text that converts back to the exact same value ('NaN', 'inf'; notation such as '0.00001' vs '1e-5' can differ between engines); datetimes print as 'YYYY-MM-DD HH:MM:SS[.ffffff]'. Casts that have no single meaning (dates to numbers, floats to Boolean, changing a time zone) are rejected when the plan is built.
- **names**: Column names must differ by more than letter case, must not contain NUL, and must not start with the reserved prefix '__pu_'.
- **float_precision**: Float sums and means can differ in the last digits (about 1e-15 relative) between engines because they add in different orders; every other result is identical.
- **known_differences**: Rare cases engines still differ on, by design of the engines: upper()/lower() of a few special characters (e.g. 'ß', 'İ'), and DuckDB may return -0.0 as 0.0 after sorting.
