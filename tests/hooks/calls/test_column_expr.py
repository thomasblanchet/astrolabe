"""Check `ColumnExpr`, which reads a column name that the caller wrote as a string.

polars accepts a string in place of `pl.col(...)`, as in
`df.select("price")`. Such an argument has the type `str` and no
expression type, so a hook reads the name from the AST node.
`ColumnExpr.from_node` reads only this string form. Every argument
that is a real expression, such as `pl.col("price")`, already has an
expression type that the hook reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import NameExpr, StrExpr

from astrolabe._hooks.calls.column_expr import ColumnExpr

if TYPE_CHECKING:
    from mypy.test.typefixture import TypeFixture
    from mypy.types import Instance

    from tests.hooks.conftest import FakeApi


def test_from_node_recognizes_the_bare_string_shorthand(fixture: TypeFixture, api: FakeApi) -> None:
    parsed = ColumnExpr.from_node(StrExpr("foo"), api)
    assert parsed == ColumnExpr(source_name="foo")


def test_from_node_rejects_an_unrecognized_shape(
    fixture: TypeFixture,
    api: FakeApi,
    str_dtype: Instance,
) -> None:
    # str_dtype stands in for the type of an ordinary variable. Like that
    # type, it holds no known value, so the reader cannot find the string.
    node = NameExpr("some_variable")
    api.expression_types[id(node)] = str_dtype
    assert ColumnExpr.from_node(node, api) is None
