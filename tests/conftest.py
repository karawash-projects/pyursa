import datetime as dt

import pytest

import pyursa as pu

BACKENDS = ["polars", "duckdb"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    return request.param


@pytest.fixture
def orders():
    return pu.from_dict(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "region": ["east", "west", "east", None, "west", "east"],
            "sales": [10, 20, None, 5, 20, 10],
            "price": [1.5, 2.0, 3.0, None, 1.0, 0.0],
            "day": [dt.date(2026, 1, d) for d in (3, 1, 2, 5, 4, 6)],
        },
        name="orders",
    )


@pytest.fixture
def customers():
    return pu.from_dict(
        {"region": ["east", "west", "west", "north"], "manager": ["Ana", "Bo", "Cy", "Di"]},
        name="customers",
    )
