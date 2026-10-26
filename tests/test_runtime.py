# mypy: ignore-errors
"""Check what astrolabe does when the program runs, apart from the checker.

The public classes of astrolabe also exist at run time. `TypedDataFrame[X]`
builds a real class whose `schema` holds the declared columns as polars
data types. `check_schema`, `assume_schema` and `force_schema` compare a
real frame with such a class, or cast it to that class. Each declaration,
such as `typed_expr(...)`, keeps its parts as a plain record. These
tests check that behavior with plain pytest.

Some tests call `isinstance()` with a class such as `TypedDataFrame[X]`.
mypy refuses `isinstance()` with a subscripted generic class, even though
`__class_getitem__` gives a real class here. The directive
`# mypy: ignore-errors` on the first line therefore turns off type
checking for this whole module. The `.yml` cases under `tests/typing/`
check the static types.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import polars as pl
import polars.selectors as cs
import pytest

from astrolabe import (
    Unchecked,
    Array,
    Column,
    Columns,
    Datetime,
    Decimal,
    Duration,
    Enum,
    List,
    Struct,
    TypedDataFrame,
    TypedLazyFrame,
    TypedSeries,
    assume_schema,
    check_schema,
    column,
    column_of,
    force_schema,
    row,
    same,
    some_column,
    subset_of,
    typed_agg,
    typed_agg_selection,
    typed_expr,
    typed_lit,
    typed_selection,
    wrap,
)
from astrolabe.frame import _TypedFrameBase
from tests.releases import polars_before


PriceColumn = column("price", pl.Float64)


class Priced(Columns):
    """A schema with one `Struct` column, whose one field is a declared column."""

    s: Struct[PriceColumn]


class FooBar(Columns):
    """The schema that most tests in this module give a frame."""

    foo: pl.String
    bar: pl.Int64


class FooUnchecked(Columns):
    """The schema of `FooBar`, with `bar` as an `Unchecked` column."""

    foo: pl.String
    bar: Unchecked


Frame = TypedDataFrame[FooBar]
LazyFrame = TypedLazyFrame[FooBar]
AnyFrame = TypedDataFrame[FooUnchecked]


def test_class_level_schema_is_the_declared_schema() -> None:
    assert Frame.schema == {"foo": pl.String, "bar": pl.Int64}


class RuntimeLegislators(Columns):
    """A `Columns` subclass written by hand, which `TypedDataFrame[...]` reads at run time."""

    first_name: pl.String
    last_name: pl.Int64


def test_columns_class_schema_resolves_at_runtime() -> None:
    ColumnsFrame = TypedDataFrame[RuntimeLegislators]
    assert ColumnsFrame.schema == {"first_name": pl.String, "last_name": pl.Int64}


class BadRuntimeColumns(Columns):
    """A `Columns` subclass with an annotation that is not a polars data type."""

    # `int` is not a polars data type. The checker reports this annotation, but
    # this module has no type checking, so the class still builds. Only the
    # schema of a frame over this class shows the fault: it is None.
    a: int


def test_columns_class_with_unresolvable_field_type_has_no_concrete_schema() -> None:
    BadFrame = TypedDataFrame[BadRuntimeColumns]
    assert BadFrame.schema is None


MARKERS_OF_ANOTHER_KIND: dict[str, object] = {
    "a struct field of a name that is a number": Struct[Column[Literal[3], pl.Int64]],
    "a struct field of a name that is a type": Struct[Column[int, pl.Int64]],
    "a struct field of a type that is not polars": Struct[Column[Literal["b"], int]],
    "a list of a type that is not polars": List[int],
    "an array of a type that is not polars": Array[int, Literal[3]],
    "an array of a width of text": Array[pl.Int64, Literal["x"]],
    "an array of a width that is not a literal": Array[pl.Int64, int],
    "a datetime of a unit that is a number": Datetime[Literal[3], None],
    "a datetime of a unit that is not a literal": Datetime[int, None],
    "a datetime of a zone that is a number": Datetime[Literal["us"], Literal[3]],
    "a duration of a unit that is a number": Duration[Literal[3]],
    "a decimal of a precision of text": Decimal[Literal["x"], Literal[2]],
    "a decimal of a scale of text": Decimal[Literal[10], Literal["x"]],
    "an enum of a category that is a number": Enum[Literal["a"], Literal[3]],
}


@pytest.mark.parametrize("name", sorted(MARKERS_OF_ANOTHER_KIND))
def test_a_marker_with_a_parameter_of_another_kind_gives_no_schema(name: str) -> None:
    # The checker reports such a marker. Without the checker, the class still
    # builds, and the frame over it has no schema, as for `BadRuntimeColumns`.
    annotations = {"a": MARKERS_OF_ANOTHER_KIND[name]}
    bad = type("Bad", (Columns,), {"__annotations__": annotations, "__module__": __name__})
    assert TypedDataFrame[bad].schema is None


def test_a_generic_columns_class_gives_the_schema_of_its_argument() -> None:
    class Listed[T](Columns):
        baz: List[T]

    assert TypedDataFrame[Listed[pl.Int64]].schema == {"baz": pl.List(pl.Int64)}


def test_a_generic_argument_names_a_struct_field_by_a_bare_string() -> None:
    # `typing` wraps the bare string in a forward reference, and the argument
    # of a generic class reaches the schema with no evaluation of it.
    class Keyed[T](Columns):
        v: T

    keyed = TypedDataFrame[Keyed[Struct[Column["b", pl.Int64]]]]
    assert keyed.schema == {"v": pl.Struct({"b": pl.Int64})}


def test_list_dtype_resolves_to_a_real_pl_list_at_runtime() -> None:
    class Listed(Columns):
        baz: List[pl.Int64]

    ListFrame = TypedDataFrame[Listed]
    assert ListFrame.schema == {"baz": pl.List(pl.Int64)}


def test_nested_list_dtype_resolves_to_a_real_pl_list_at_runtime() -> None:
    class NestedListed(Columns):
        baz: List[List[pl.Int64]]

    NestedListFrame = TypedDataFrame[NestedListed]
    assert NestedListFrame.schema == {"baz": pl.List(pl.List(pl.Int64))}


def test_array_dtype_resolves_to_a_real_pl_array_at_runtime() -> None:
    class Arrayed(Columns):
        baz: Array[pl.Int64, Literal[3]]

    ArrayFrame = TypedDataFrame[Arrayed]
    assert ArrayFrame.schema == {"baz": pl.Array(pl.Int64, 3)}


def test_struct_dtype_resolves_to_a_real_pl_struct_at_runtime() -> None:
    class Structured(Columns):
        bar: Struct[Column[Literal["baz"], pl.Int64], Column[Literal["qux"], pl.Int64]]

    StructFrame = TypedDataFrame[Structured]
    assert StructFrame.schema == {"bar": pl.Struct({"baz": pl.Int64, "qux": pl.Int64})}


def test_instance_level_schema_reflects_the_real_dataframe() -> None:
    # The `schema` of an instance must be the real schema that polars computes,
    # not the declared one. The class attribute `schema` must not hide the
    # polars property that the instance inherits.
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    df.__class__ = Frame
    assert df.schema == pl.DataFrame({"foo": ["x"], "bar": [30]}).schema


def test_assume_schema_does_not_reorder_columns() -> None:
    df = pl.DataFrame({"bar": [30], "foo": ["x"]})
    result = assume_schema(df, Frame)
    assert list(result.columns) == ["bar", "foo"]
    assert isinstance(result, Frame)


def test_assume_schema_does_not_raise_on_a_missing_column() -> None:
    df = pl.DataFrame({"bar": [30]})
    result = assume_schema(df, Frame)
    assert isinstance(result, Frame)


def test_assume_schema_does_not_raise_on_a_dtype_mismatch() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30.0]})
    result = assume_schema(df, Frame)
    assert isinstance(result, Frame)


def test_assume_schema_does_not_mutate_the_original_object() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    result = assume_schema(df, Frame)
    assert result is not df
    assert df.__class__ is pl.DataFrame


def test_check_schema_does_not_reorder_columns() -> None:
    df = pl.DataFrame({"bar": [30], "foo": ["x"]})
    result = check_schema(df, Frame)
    assert list(result.columns) == ["bar", "foo"]
    assert isinstance(result, Frame)


def test_check_schema_does_not_mutate_the_original_object() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    result = check_schema(df, Frame)
    assert result is not df
    assert df.__class__ is pl.DataFrame


def test_check_schema_raises_on_missing_column() -> None:
    df = pl.DataFrame({"bar": [30]})
    with pytest.raises(pl.exceptions.SchemaError):
        check_schema(df, Frame)


def test_check_schema_raises_on_an_extra_undeclared_column() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30], "extra": [True]})
    with pytest.raises(pl.exceptions.SchemaError):
        check_schema(df, Frame)


def test_check_schema_raises_on_dtype_mismatch() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30.0]})
    with pytest.raises(pl.exceptions.SchemaError):
        check_schema(df, Frame)


def test_check_schema_does_not_coerce_dtypes() -> None:
    # The same input as the test above, for the rule that check_schema only
    # checks. It never casts a column to a wider data type.
    df = pl.DataFrame({"foo": ["x"], "bar": [30.0]})
    with pytest.raises(pl.exceptions.SchemaError):
        check_schema(df, Frame)


def test_check_schema_raises_when_frame_type_has_no_concrete_schema() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    with pytest.raises(TypeError, match="no concrete schema"):
        check_schema(df, TypedDataFrame)


def test_unchecked_schema_reports_the_marker_class() -> None:
    assert AnyFrame.schema == {"foo": pl.String, "bar": Unchecked}


def test_check_schema_accepts_any_actual_dtype_for_an_unchecked_column() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    result = check_schema(df, AnyFrame)
    assert isinstance(result, AnyFrame)
    other = pl.DataFrame({"foo": ["x"], "bar": ["not an int either"]})
    other_result = check_schema(other, AnyFrame)
    assert isinstance(other_result, AnyFrame)


def test_check_schema_still_requires_an_unchecked_column_present_by_name() -> None:
    df = pl.DataFrame({"foo": ["x"]})
    with pytest.raises(pl.exceptions.SchemaError):
        check_schema(df, AnyFrame)


def test_force_schema_reorders_and_narrows_columns_to_the_declared_set() -> None:
    df = pl.DataFrame({"bar": [30], "extra": [True], "foo": ["x"]})
    result = force_schema(df, Frame)
    assert list(result.columns) == ["foo", "bar"]
    assert isinstance(result, Frame)


def test_force_schema_casts_a_convertible_dtype_mismatch() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30.0]})
    result = force_schema(df, Frame)
    assert isinstance(result, Frame)
    assert result.schema["bar"] == pl.Int64


def test_force_schema_raises_on_a_missing_column() -> None:
    df = pl.DataFrame({"bar": [30]})
    with pytest.raises(pl.exceptions.SchemaError):
        force_schema(df, Frame)


def test_force_schema_raises_on_an_unconvertible_value_by_default() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": ["not a number"]})
    with pytest.raises(pl.exceptions.InvalidOperationError):
        force_schema(df, Frame)


def test_force_schema_nulls_an_unconvertible_value_under_strict_false() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": ["not a number"]})
    result = force_schema(df, Frame, strict=False)
    assert isinstance(result, Frame)
    assert result["bar"].to_list() == [None]


def test_force_schema_does_not_cast_an_unchecked_column() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30.5]})
    result = force_schema(df, AnyFrame)
    assert isinstance(result, AnyFrame)
    assert result.schema["bar"] == pl.Float64


def test_force_schema_raises_when_frame_type_has_no_concrete_schema() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [30]})
    with pytest.raises(TypeError, match="no concrete schema"):
        force_schema(df, TypedDataFrame)


def test_typed_dataframe_methods_delegate_to_the_real_dataframe() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [1]})
    df.__class__ = Frame
    assert df.select("foo").columns == ["foo"]
    assert df.select_seq("foo").columns == ["foo"]
    assert df.drop("bar").columns == ["foo"]
    assert df.rename({"bar": "baz"}).columns == ["foo", "baz"]
    assert df.with_columns(pl.col("bar").alias("qux")).columns == ["foo", "bar", "qux"]
    assert df.with_columns_seq(pl.col("bar").alias("qux")).columns == ["foo", "bar", "qux"]


def test_typed_dataframe_passthrough_methods_delegate_to_the_real_dataframe() -> None:
    df = pl.DataFrame({"foo": ["x", "y"], "bar": [1, 2]})
    df.__class__ = Frame
    assert df.filter(pl.col("bar") > 0).columns == ["foo", "bar"]
    assert df.sort("bar").columns == ["foo", "bar"]
    assert df.unique().columns == ["foo", "bar"]
    assert df.head(1).columns == ["foo", "bar"]
    assert df.tail(1).columns == ["foo", "bar"]
    assert df.limit(1).columns == ["foo", "bar"]
    assert df.reverse().columns == ["foo", "bar"]
    assert df.drop_nulls().columns == ["foo", "bar"]
    assert df.clone().columns == ["foo", "bar"]
    assert df.gather_every(1).columns == ["foo", "bar"]
    assert df.slice(0, 1).columns == ["foo", "bar"]
    assert df.shift(1).columns == ["foo", "bar"]
    assert df.drop_nans().columns == ["foo", "bar"]
    assert df.remove(pl.col("bar") == 0).columns == ["foo", "bar"]
    assert df.top_k(1, by="bar").columns == ["foo", "bar"]
    assert df.bottom_k(1, by="bar").columns == ["foo", "bar"]
    assert df.clear().columns == ["foo", "bar"]
    assert df.rechunk().columns == ["foo", "bar"]
    assert df.sample(1).columns == ["foo", "bar"]


def test_typed_dataframe_merge_sorted_and_update_delegate_to_the_real_dataframe() -> None:
    df = pl.DataFrame({"foo": [1], "bar": [1]})
    df.__class__ = Frame
    other = pl.DataFrame({"foo": [2], "bar": [2]})
    assert df.merge_sorted(other, key="foo").columns == ["foo", "bar"]
    assert df.update(other, on="foo").columns == ["foo", "bar"]


def test_typed_series_passthrough_methods_delegate_to_the_real_series() -> None:
    s = pl.Series("foo", [3, 1, 2])
    s.__class__ = TypedSeries
    assert s.drop_nulls().to_list() == [3, 1, 2]
    assert s.alias("renamed").name == "renamed"
    assert s.rename("renamed").name == "renamed"
    assert s.slice(0, 2).to_list() == [3, 1]
    assert s.head(2).to_list() == [3, 1]
    assert s.tail(2).to_list() == [1, 2]
    assert s.limit(2).to_list() == [3, 1]
    assert s.gather_every(2).to_list() == [3, 2]
    assert sorted(s.top_k(2).to_list()) == [2, 3]
    assert sorted(s.bottom_k(2).to_list()) == [1, 2]
    assert s.unique().to_list() == [1, 2, 3]
    assert s.reverse().to_list() == [2, 1, 3]
    assert s.clear().to_list() == []
    assert s.round().to_list() == [3, 1, 2]
    assert s.shift(1).to_list() == [None, 3, 1]
    assert sorted(s.shuffle(seed=0).to_list()) == [1, 2, 3]
    assert len(s.sample(2, seed=0).to_list()) == 2


def test_typed_dataframe_vstack_and_extend_delegate_to_the_real_dataframe() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [1]})
    df.__class__ = Frame
    other = pl.DataFrame({"foo": ["y"], "bar": [2]})
    assert df.vstack(other).columns == ["foo", "bar"]
    assert df.extend(other).columns == ["foo", "bar"]


def test_typed_dataframe_pipe_calls_function_with_the_receiver_first() -> None:
    df = pl.DataFrame({"foo": ["x"], "bar": [1]})
    df.__class__ = Frame
    result = df.pipe(lambda frame, suffix: [*frame.columns, suffix], "extra")
    assert result == ["foo", "bar", "extra"]


def test_typed_lazyframe_methods_delegate_to_the_real_lazyframe() -> None:
    lf = pl.DataFrame({"foo": ["x"], "bar": [1]}).lazy()
    lf.__class__ = LazyFrame
    assert lf.select("foo").collect().columns == ["foo"]
    assert lf.select_seq("foo").collect().columns == ["foo"]
    assert lf.drop("bar").collect().columns == ["foo"]
    assert lf.rename({"bar": "baz"}).collect().columns == ["foo", "baz"]
    assert lf.with_columns(pl.col("bar").alias("qux")).collect().columns == ["foo", "bar", "qux"]
    assert lf.with_columns_seq(pl.col("bar").alias("qux")).collect().columns == [
        "foo",
        "bar",
        "qux",
    ]


@pytest.mark.skipif(polars_before("1.41.0"), reason="polars 1.41.0 adds gather to the frames")
def test_typed_frames_pass_gather_on_to_polars() -> None:
    df = pl.DataFrame({"foo": ["x", "y"], "bar": [1, 2]})
    df.__class__ = Frame
    assert df.gather([0]).columns == ["foo", "bar"]
    lf = df.lazy()
    lf.__class__ = LazyFrame
    assert lf.gather([0]).collect().columns == ["foo", "bar"]


@pytest.mark.skipif(not polars_before("1.41.0"), reason="polars 1.41.0 adds gather to the frames")
def test_typed_frames_have_no_gather_before_polars_1_41() -> None:
    df = pl.DataFrame({"foo": ["x", "y"], "bar": [1, 2]})
    df.__class__ = Frame
    with pytest.raises(AttributeError):
        df.gather([0])
    lf = df.lazy()
    lf.__class__ = LazyFrame
    with pytest.raises(AttributeError):
        lf.gather([0])


def test_typed_lazyframe_passthrough_methods_delegate_to_the_real_lazyframe() -> None:
    lf = pl.DataFrame({"foo": ["x", "y"], "bar": [1, 2]}).lazy()
    lf.__class__ = LazyFrame
    assert lf.filter(pl.col("bar") > 0).collect().columns == ["foo", "bar"]
    assert lf.sort("bar").collect().columns == ["foo", "bar"]
    assert lf.unique().collect().columns == ["foo", "bar"]
    assert lf.head(1).collect().columns == ["foo", "bar"]
    assert lf.tail(1).collect().columns == ["foo", "bar"]
    assert lf.limit(1).collect().columns == ["foo", "bar"]
    assert lf.reverse().collect().columns == ["foo", "bar"]
    assert lf.drop_nulls().collect().columns == ["foo", "bar"]
    assert lf.clone().collect().columns == ["foo", "bar"]
    assert lf.gather_every(1).collect().columns == ["foo", "bar"]
    assert lf.slice(0, 1).collect().columns == ["foo", "bar"]
    assert lf.shift(1).collect().columns == ["foo", "bar"]
    assert lf.drop_nans().collect().columns == ["foo", "bar"]
    assert lf.remove(pl.col("bar") == 0).collect().columns == ["foo", "bar"]
    assert lf.top_k(1, by="bar").collect().columns == ["foo", "bar"]
    assert lf.bottom_k(1, by="bar").collect().columns == ["foo", "bar"]
    assert lf.clear().collect().columns == ["foo", "bar"]


def test_typed_lazyframe_merge_sorted_and_update_delegate_to_the_real_lazyframe() -> None:
    lf = pl.DataFrame({"foo": [1], "bar": [1]}).lazy()
    lf.__class__ = LazyFrame
    other = pl.DataFrame({"foo": [2], "bar": [2]}).lazy()
    assert lf.merge_sorted(other, key="foo").collect().columns == ["foo", "bar"]
    assert lf.update(other, on="foo").collect().columns == ["foo", "bar"]


def test_typed_lazyframe_pipe_calls_function_with_the_receiver_first() -> None:
    lf = pl.DataFrame({"foo": ["x"], "bar": [1]}).lazy()
    lf.__class__ = LazyFrame
    assert lf.pipe(lambda frame, suffix: [*frame.collect().columns, suffix], "extra") == [
        "foo",
        "bar",
        "extra",
    ]


def test_lazyframe_instance_level_schema_reflects_the_real_lazyframe() -> None:
    lf = pl.DataFrame({"foo": ["x"], "bar": [30]}).lazy()
    lf.__class__ = LazyFrame
    assert lf.schema == pl.DataFrame({"foo": ["x"], "bar": [30]}).lazy().schema


def test_instance_schema_raises_when_no_base_defines_a_real_schema() -> None:
    class NoSchemaBase(_TypedFrameBase):
        pass

    with pytest.raises(AttributeError, match="schema"):
        _ = NoSchemaBase().schema


def test_instance_schema_returns_a_non_descriptor_real_schema_unchanged() -> None:
    class PlainSchemaBase:
        schema: ClassVar = {"plain": "value"}

    class PlainSchemaFrame(_TypedFrameBase, PlainSchemaBase):
        pass

    assert PlainSchemaFrame().schema == {"plain": "value"}


class TestDeclarationRecords:
    """Each declaration function gives a plain record of its arguments at run time."""

    def test_typed_expr_records_its_parts_and_defaults(self) -> None:
        """Each declaration keeps its parts and its defaults."""
        quantity = column_of(RuntimeLegislators, "first_name")
        flag = typed_expr(quantity, gives=pl.Boolean, named="is_first")
        assert flag.character == "elementwise"
        assert flag.parts == (quantity,)
        assert flag.gives is pl.Boolean
        assert flag.named == "is_first"
        assert typed_agg(gives=pl.UInt32, named="len").character == "aggregating"
        assert typed_lit(gives=pl.Boolean).named is None
        rows = typed_agg(row(quantity, gives=same), row(quantity, gives=pl.Int64))
        assert rows.parts[0].gives is same
        assert some_column(pl.Float64).region is pl.Float64
        names = subset_of(RuntimeLegislators, "first_name", "last_name").names
        assert names == ("first_name", "last_name")
        assert wrap(pl.List, same("x")).inner.read == "x"

    def test_column_keeps_its_struct_field_runtime_value(self) -> None:
        """A `column()` declaration inside a `Struct` names a field at run time."""
        assert TypedDataFrame[Priced].schema == {"s": pl.Struct({"price": pl.Float64})}

    def test_selection_declarations_record_their_parts(self) -> None:
        """A selection declaration keeps its selector, its rows and its rename."""
        texts = typed_selection(cs.numeric(), gives=pl.String, suffix="_s")
        assert texts.character == "elementwise"
        assert texts.suffix == "_s"
        assert texts.prefix is None
        assert texts.gives is pl.String
        means = typed_agg_selection(cs.numeric(), row(pl.Float64, gives=same))
        assert means.character == "aggregating"
        assert means.parts[0].gives is same
