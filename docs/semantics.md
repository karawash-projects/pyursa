# Semantics

pyursa defines one behavior for every operation and enforces it on every backend.
Engines disagree on many details (null handling, sort stability, decimal precision, quantile
interpolation, rounding, NaN, casts, integer overflow); the conformance suites
(`tests/test_conformance*.py`, `tests/test_review_findings.py`) state the expected output for
each rule and run it on every backend.

- **null comparison**: Comparisons with null yield null; filter() drops rows where the condition is null.
- **division**: '/' always returns Float64 and follows IEEE 754 (x/0 = inf, 0/0 = NaN).
- **sum of nulls**: sum() over only nulls (or no rows) returns null, not 0.
- **n unique**: n_unique() does not count null as a value.
- **count**: count() counts non-null values; pu.len() counts rows.
- **sort**: sort() is stable and places nulls last in both directions.
- **group by order**: group_by().agg() output is sorted by the keys ascending, nulls last.
- **join order**: join() keeps left row order, then right row order within matches; full joins put unmatched right rows last; null keys never match (anti joins keep null-key rows).
- **distinct**: distinct() keeps the first occurrence of each row (or subset) in input order; nulls compare equal.
- **concat**: concat() returns all rows of the first frame, then the second, and so on.
- **rounding**: round() rounds half away from zero (2.5 -> 3, -2.5 -> -3).
- **decimals**: Decimal +,-,* are exact (scale: max for +/-, sum for *); / returns Float64; sum keeps the scale; mean/median/std return Float64.
- **quantiles**: median()/quantile(q) use linear interpolation; std()/var() are sample statistics (ddof=1).
- **windows**: Window results keep input row order; ties in order_by keep input order; rank() is SQL RANK with nulls last; cum_sum skips nulls.
- **time zones**: Datetime(tz) keeps its zone; .dt parts are computed in that zone; comparing different zones is an error.
- **strings**: str.len counts characters; str.strip removes spaces, tabs and newlines; str.slice is 0-based.
- **row order**: filter/select/with_columns/limit preserve input row order.
- **nan**: NaN is the largest float value and equals itself (in comparisons, sorting, grouping and min/max); max() returns NaN when one is present; std/var with an infinite value return NaN.
- **integer overflow**: Int64 arithmetic, sums and running totals that leave the Int64 range raise INTEGER_OVERFLOW; they never wrap around.
- **casts**: Casts are strict and identical everywhere: text is trimmed of spaces/tabs/newlines and must then be a valid literal (integers like '+12', floats like '1e3', 'inf' or 'nan', decimals with up to 18 places, booleans 'true'/'false'/'1'/'0', dates 'YYYY-MM-DD', datetimes 'YYYY-MM-DD HH:MM[:SS[.ffffff]]' with ' ' or 'T'); anything else raises CAST_FAILED. Float or Decimal to Int64 requires a whole number. Conversions to Decimal round half away from zero. Floats print like '1.5e-7' and 'NaN'; datetimes print as 'YYYY-MM-DD HH:MM:SS[.ffffff]'. Casts that have no single meaning (dates to numbers, floats to Boolean, changing a time zone) are rejected when the plan is built.
- **names**: Column names must differ by more than letter case, must not contain NUL, and must not start with the reserved prefix '__pu_'.
- **float precision**: Float sums and means can differ in the last digits (about 1e-15 relative) between engines because they add in different orders; every other result is identical.
- **known differences**: Rare cases engines still differ on, by design of the engines: upper()/lower() of a few special characters (e.g. 'ß', 'İ'), and DuckDB may return -0.0 as 0.0 after sorting.

These rules are part of the public API; see the [stability policy](stability.md).
