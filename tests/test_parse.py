
import pytest

import pyursa as pu
from pyursa.parse import parse_expr


@pytest.mark.parametrize("src, expected", [
    ('col("sales") > 5', '(col("sales") > lit(5))'),
    ('(col("a") > 1) & col("b").is_null()', '((col("a") > lit(1)) & col("b").is_null())'),
    ('col("a") > 1 and col("b") < 2', '((col("a") > lit(1)) & (col("b") < lit(2)))'),
    ('col("r").is_in(["e", "w"])', "col(\"r\").is_in(['e', 'w'])"),
    ('col("r") in ["e"]', "col(\"r\").is_in(['e'])"),
    ('col("r") is None', 'col("r").is_null()'),
    ('not col("f")', '~col("f")'),
    ('col("s").str.contains("x")', "col(\"s\").str.contains('x')"),
    ('col("x").sum().alias("t")', 'col("x").sum().alias("t")'),
    ('len()', "len()"),
    ('2 * col("x")', '(lit(2) * col("x"))'),
    ('col("d") >= date(2026, 1, 4)', '(col("d") >= lit(date(2026, 1, 4)))'),
    ('col("p") > decimal("1.10")', '(col("p") > lit(decimal(\'1.10\')))'),
    ('"qty"', 'col("qty")'),
])
def test_valid(src, expected):
    assert repr(parse_expr(src)) == expected


@pytest.mark.parametrize("src", [
    '__import__("os").system("rm -rf /")',
    'open("/etc/passwd")',
    'col("a").__class__',
    'col("a").to_pandas()',
    '[x for x in range(10**9)]',
    'lambda: 1',
    'col("a") ** 2',
    "(" * 300 + "1" + ")" * 300,
    'qty > 5',
])
def test_rejected_safely(src):
    with pytest.raises(pu.PyursaError) as ei:
        parse_expr(src)
    assert ei.value.code == "EXPRESSION_SYNTAX"


def test_bare_name_suggests_col():
    with pytest.raises(pu.PyursaError) as ei:
        parse_expr("qty > 5")
    assert ei.value.suggestions[0]["fix"] == 'col("qty")'


def test_parsed_expression_runs(orders, backend):
    out = orders.filter(parse_expr('col("sales") > 5 and col("region") in ["east"]')).select("id")
    assert out.to_dicts(backend) == [{"id": 1}, {"id": 6}]
