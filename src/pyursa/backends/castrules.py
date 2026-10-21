"""Patterns for the strict cast contract (SEMANTICS['casts']), shared by every backend.

Text is trimmed of spaces, tabs and newlines first. A value that doesn't match its
pattern is replaced by a sentinel string that the engine then fails to convert, so
invalid input raises an error instead of becoming null or a guessed value."""

INT_RE = r"^[+-]?[0-9]+$"
FLOAT_RE = r"^[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][+-]?[0-9]+)?$"
# up to 20 integer digits and 17 decimal places, so rounding to any scale never overflows Decimal(38, 17)
DECIMAL_RE = r"^[+-]?([0-9]{1,20}\.?[0-9]{0,17}|\.[0-9]{1,17})$"
_YEAR = r"([0-9]{3}[1-9]|[0-9]{2}[1-9][0-9]|[0-9][1-9][0-9]{2}|[1-9][0-9]{3})"  # 0001-9999
DATE_RE = r"^" + _YEAR + r"-[0-9]{2}-[0-9]{2}$"
# hours 00-23, minutes and seconds 00-59 (no 24:00, no leap seconds: engines disagree on both)
DATETIME_RE = (r"^" + _YEAR + r"-[0-9]{2}-[0-9]{2}[ T]([01][0-9]|2[0-3]):[0-5][0-9]"
               r"(:[0-5][0-9](\.[0-9]{1,6})?)?$")
BOOL_TRUE = ["true", "1"]
BOOL_FALSE = ["false", "0"]
SENTINEL_PREFIX = "PYURSA_INVALID_"


def sentinel(to) -> str:
    return f"{SENTINEL_PREFIX}{str(to).replace('(', '_').replace(')', '').replace(',', '_')}"
