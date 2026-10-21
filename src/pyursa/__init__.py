"""pyursa: an engine-agnostic dataframe library for agentic AI workflows.

    import pyursa as pu
    from pyursa import col

    df = pu.from_dict({"region": ["east", "west"], "sales": [10, 20]})
    out = df.filter(col("sales") > 5, intent="drop tiny orders").group_by("region").agg(col("sales").sum())
    out.collect(backend="duckdb")
"""

from .backends import (
    SEMANTICS,
    available_backends,
    default_backend,
    get_backend,
    register_backend,
    set_default_backend,
)
from .errors import (
    BackendError,
    ColumnNotFoundError,
    DuplicateColumnError,
    HistoryError,
    InvalidOperationError,
    PyursaError,
    TypeMismatchError,
    UnknownBackendError,
    UnsupportedTypeError,
)
from .expr import coalesce, col, lit, rank, row_number, when
from .expr import len_ as len  # noqa: A001  (pu.len() mirrors Polars)
from .frame import (
    Frame,
    concat,
    from_arrow,
    from_dict,
    from_pandas,
    from_polars,
    read_csv,
    read_json,
    read_parquet,
)
from .providers import (
    PROVIDERS,
    BedrockClient,
    LLMProviderError,
    OpenAICompatibleClient,
    list_providers,
    llm_client,
    print_providers,
)
from .semantic import (
    AnthropicClient,
    FunctionClient,
    LLMBudgetError,
    LLMClient,
    LLMNotConfiguredError,
    PromptClient,
    llm_reports,
    set_llm,
)
from .types import (
    Boolean,
    Date,
    Datetime,
    Decimal,
    DType,
    Float64,
    Int64,
    List,
    Null,
    String,
    Struct,
)

__version__ = "1.0.0rc6"

__all__ = [
    "Frame", "col", "lit", "len", "when", "coalesce", "row_number", "rank", "concat",
    "from_arrow", "from_dict", "from_pandas", "from_polars", "read_csv", "read_json", "read_parquet",
    "DType", "Int64", "Float64", "String", "Boolean", "Date", "Datetime", "Decimal", "List", "Struct", "Null",
    "SEMANTICS", "available_backends", "default_backend", "get_backend", "register_backend", "set_default_backend",
    "set_llm", "llm_reports", "LLMClient", "PromptClient", "FunctionClient", "AnthropicClient",
    "LLMBudgetError", "LLMNotConfiguredError", "LLMProviderError",
    "llm_client", "list_providers", "print_providers", "OpenAICompatibleClient", "BedrockClient", "PROVIDERS",
    "PyursaError", "BackendError", "ColumnNotFoundError", "DuplicateColumnError", "HistoryError",
    "InvalidOperationError", "TypeMismatchError", "UnknownBackendError", "UnsupportedTypeError",
]
