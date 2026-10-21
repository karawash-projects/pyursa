# Types

pyursa has a small, strict type system shared by every backend:

| Type | Arrow storage | Notes |
|---|---|---|
| `Int64` | int64 | All integer inputs. Unsigned values that don't fit raise an error. |
| `Float64` | float64 | All float inputs (float16/32 widen losslessly). |
| `Decimal(p, s)` | decimal128 | Exact. Precision up to 38. |
| `String` | string | Dictionary/categorical columns decode to String. |
| `Boolean` | bool | |
| `Date` | date32 | |
| `Datetime` / `Datetime("UTC")` | timestamp[us] | Microsecond precision; the time zone is part of the type. |
| `List(T)`, `Struct({...})` | list, struct | Carried through filters, joins, sorts by other columns, and writes. |
| `Null` | null | Columns with no type information. |

Use constants (`pu.Int64`), constructors (`pu.Decimal(12, 2)`, `pu.Datetime("UTC")`,
`pu.List(pu.String)`) or strings (`"Decimal(12,2)"`, `"Datetime(America/Toronto)"`).

## Nothing is converted silently

These raise an error (`UnsupportedTypeError`, `DuplicateColumnError`, `INTEGER_OVERFLOW` or `CAST_FAILED`) instead of corrupting data:

- Timestamps with sub-microsecond values (nanoseconds would be truncated).
- Unsigned 64-bit values above the Int64 range.
- Decimals with precision above 38.
- Binary, time-of-day, duration, map and union columns (cast them yourself first).
- Column names that differ only by letter case (`A` and `a`), contain NUL, or start with `__pu_`.
- Int64 results that overflow (sums, arithmetic, running totals) raise `INTEGER_OVERFLOW` instead of wrapping.

## NaN

NaN is a float value, not a missing value. It is the largest float and equals itself, the
same on every engine: `NaN > 0` is true, `max()` returns NaN when one is present, sorting puts
NaN after every number (and before nulls). Use `.is_null()` for missing values.

## Casts

`col("x").cast(T)` follows one strict contract on every engine:

| From → to | Rule |
|---|---|
| String → Int64 / Float64 / Decimal / Boolean / Date / Datetime | Trim spaces, tabs and newlines, then the text must be a valid literal: `"+12"`, `"1e3"`, `"inf"`, `"nan"`, `"19.99"` (up to 18 places), `"true"`/`"false"`/`"1"`/`"0"`, `"2024-01-05"`, `"2024-01-05 10:11"` or `"2024-01-05T10:11:12.5"`. Anything else raises `CAST_FAILED`. |
| Float64 / Decimal → Int64 | Must be a whole number (NaN and inf fail). Use `.round(0).cast("Int64")` to round first. |
| Float64 / Decimal → Decimal(p, s) | Rounds half away from zero to `s` places; too many digits raise an error. |
| Number / Boolean / Date / Datetime → String | Identical text on every engine (`"1.5e-7"`, `"NaN"`, `"true"`, `"2024-01-05 10:11:12.500000"`). |
| Datetime(tz) → anything else | Not allowed; use `.dt` functions. |
| Date ↔ numbers, Float → Boolean | Not allowed (no single meaning). |

## Exact decimal arithmetic

| Operation | Result |
|---|---|
| `Decimal(·,a) + Decimal(·,b)` | `Decimal(38, max(a,b))` |
| `Decimal(·,a) * Decimal(·,b)` | `Decimal(38, a+b)` (error if a+b > 38) |
| `Decimal + Int64` | `Decimal(38, a)` |
| `Decimal + Float64` | `Float64` |
| any `/` | `Float64` |
| `sum(Decimal(·,s))` | `Decimal(38, s)` |
| `mean`, `median`, `std`, `quantile` | `Float64` |

Some engines truncate decimal products by default (for example `1.15 × 1.15` → `1.32`);
pyursa computes at the result scale so you get `1.3225` everywhere.

## Time zones

A `Datetime("America/Toronto")` column stays in that zone: `.dt.hour()`, `.dt.date()` and
`.dt.truncate()` are computed in local time. Comparing datetimes in different zones (or a
zoned column with a naive literal) is an error; compare with a value in the same zone:

```python
import zoneinfo, datetime as dt
tz = zoneinfo.ZoneInfo("America/Toronto")
events.filter(col("ts") >= dt.datetime(2026, 1, 1, tzinfo=tz))
```
