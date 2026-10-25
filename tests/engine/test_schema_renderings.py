"""Check how the declared columns of a frame become the plain schema that the engine reads.

The engine knows nothing about mypy. It reads the schema of a frame
as a plain map from each column name to the text that polars shows
for its data type, for example `"List(Boolean)"`. Each hook that
applies an expression to a frame first turns the declared columns
into that map with `SchemaRenderings`.

These tests cover each form of data type: a plain one such as
`Int64`, `Unchecked`, and each data type with parameters. A data type
that has no text becomes `Unchecked`, and a column with no known name
is left out.
"""

from __future__ import annotations

from mypy.types import Instance

from astrolabe._hooks.domain.column import ColumnRef

from astrolabe._hooks.encoding.schema_renderings import SchemaRenderings
from tests.support import (
    array_dtype,
    boolean_dtype,
    column_ref,
    duration_dtype,
    int64_dtype,
    list_dtype,
    new_type_fixture,
    parametrized_datetime_dtype,
    unchecked_instance,
)


class TestSchemaRenderings:
    """The declared columns of a frame become the map that the engine reads."""

    def test_the_vocabulary_renders(self) -> None:
        """Plain data types, `Unchecked`, and data types with parameters each get their text."""
        fixture = new_type_fixture()
        rendered = SchemaRenderings().of([
            column_ref(fixture, "price", int64_dtype(fixture)),
            column_ref(fixture, "open", unchecked_instance(fixture)),
            column_ref(fixture, "flags", list_dtype(fixture, boolean_dtype(fixture))),
            column_ref(fixture, "span", duration_dtype(fixture)),
            column_ref(
                fixture,
                "stamp",
                parametrized_datetime_dtype(fixture, "ns", None),
            ),
            column_ref(fixture, "grid", array_dtype(fixture, int64_dtype(fixture))),
        ])
        assert rendered == {
            "price": "Int64",
            "open": "Unchecked",
            "flags": "List(Boolean)",
            "span": "Duration",
            "stamp": "Datetime(time_unit='ns', time_zone=None)",
            "grid": "Array(Int64, shape=(3,))",
        }

    def test_an_unrenderable_dtype_renders_as_unchecked(self) -> None:
        """A type that is not a polars data type becomes `Unchecked`."""
        fixture = new_type_fixture()
        columns = [
            column_ref(fixture, "price", int64_dtype(fixture)),
            column_ref(fixture, "blob", Instance(fixture.oi, [])),
        ]
        assert SchemaRenderings().of(columns) == {"price": "Int64", "blob": "Unchecked"}

    def test_an_unnamed_column_is_left_out(self) -> None:
        """A column whose name is not a literal is left out, because no read can find it."""
        fixture = new_type_fixture()
        info = fixture.make_type_info(ColumnRef.TARGET.fullname)
        unnamed = ColumnRef(Instance(info, [int64_dtype(fixture), int64_dtype(fixture)]))
        named = column_ref(fixture, "price", int64_dtype(fixture))
        assert SchemaRenderings().of([unnamed, named]) == {"price": "Int64"}
