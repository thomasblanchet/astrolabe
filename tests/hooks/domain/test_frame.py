"""Check `Frame`, the plugin class that wraps the mypy type of a typed frame.

A typed frame is written `TypedDataFrame[Columns, Rest]`. The first
argument holds the declared columns, that is, the columns that the
type names. The second holds the remainder, the columns that the type
does not name. `Frame` reads both parts, builds a new frame type with
new columns, and gives the plain polars frame type when a hook falls
back. The fixtures of `tests/hooks/conftest.py` build the mypy types
by hand, so these tests need no mypy build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import AnyType, Instance, TypeOfAny, get_proper_type
from mypy.types import Type as MypyType

from astrolabe._hooks.domain.frame import Frame
from tests.hooks.conftest import FakeApi, make_column_ref, make_frame
from tests.support import declared_rest

if TYPE_CHECKING:
    from mypy.test.typefixture import TypeFixture

    from tests.hooks.conftest import FakePlugin


def test_from_proper_type_wraps_a_known_frame_fullname(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    column = make_column_ref(fixture, "foo", int_dtype)
    frame = make_frame(fixture, [column.as_instance()])
    wrapped = Frame.from_proper_type(frame.as_instance())
    assert wrapped is not None


def test_from_proper_type_rejects_a_non_instance(fixture: TypeFixture) -> None:
    not_an_instance: MypyType = AnyType(TypeOfAny.explicit)
    assert Frame.from_proper_type(not_an_instance) is None


def test_from_proper_type_rejects_an_unrelated_instance(fixture: TypeFixture) -> None:
    unrelated = Instance(fixture.str_type_info, [])
    assert Frame.from_proper_type(unrelated) is None


def test_columns_returns_every_resolved_column(fixture: TypeFixture, int_dtype: Instance) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    bar = make_column_ref(fixture, "bar", int_dtype)
    frame = make_frame(fixture, [foo.as_instance(), bar.as_instance()])
    columns = frame.columns
    assert columns is not None
    assert [column.name for column in columns] == ["foo", "bar"]


def test_columns_is_none_when_a_remainder_is_present(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    rest = declared_rest(fixture)
    frame = make_frame(fixture, [foo.as_instance(), rest])
    assert frame.columns is None


def test_known_columns_carries_a_trailing_remainder(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    rest = declared_rest(fixture)
    frame = make_frame(fixture, [foo.as_instance(), rest])
    result = frame.known_columns
    assert result is not None
    assert [column.name for column in result.columns] == ["foo"]
    assert result.rest is rest


def test_known_columns_has_no_rest_when_schema_is_fully_concrete(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    frame = make_frame(fixture, [foo.as_instance()])
    result = frame.known_columns
    assert result is not None
    assert result.rest is None


def test_known_columns_is_none_without_a_columns_tuple(fixture: TypeFixture) -> None:
    frame_info = fixture.make_type_info(next(iter(Frame.TARGETS)).fullname)
    frame = Frame(Instance(frame_info, []))
    assert frame.known_columns is None


def test_schema_items_returns_only_the_rest_slots_own_items(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    garbage: MypyType = AnyType(TypeOfAny.explicit)
    frame = make_frame(fixture, [foo.as_instance(), garbage])
    items = frame.schema_items()
    assert items is not None
    assert items == [garbage]


def test_schema_items_is_none_without_a_columns_tuple(fixture: TypeFixture) -> None:
    frame_info = fixture.make_type_info(next(iter(Frame.TARGETS)).fullname)
    frame = Frame(Instance(frame_info, []))
    assert frame.schema_items() is None


def test_rebuilt_replaces_the_schema_with_new_columns(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    bar = make_column_ref(fixture, "bar", int_dtype)
    frame = make_frame(fixture, [foo.as_instance()])
    rebuilt = frame.rebuilt([bar], api)
    columns = rebuilt.columns
    assert columns is not None
    assert [column.name for column in columns] == ["bar"]


def test_rebuilt_keeps_the_same_frame_type(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    frame = make_frame(fixture, [foo.as_instance()])
    rebuilt = frame.rebuilt([foo], api)
    assert rebuilt.as_instance().type is frame.as_instance().type


def test_rebuilt_accepts_an_explicit_rest_argument(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    rest = declared_rest(fixture)
    frame = make_frame(fixture, [foo.as_instance()])
    rebuilt = frame.rebuilt([foo], api, rest=rest)
    result = rebuilt.known_columns
    assert result is not None
    assert result.rest is rest


def test_as_plain_builds_the_plain_dataframe_type(
    fixture: TypeFixture,
    int_dtype: Instance,
    plugin: FakePlugin,
) -> None:
    foo = make_column_ref(fixture, "foo", int_dtype)
    frame = make_frame(fixture, [foo.as_instance()])
    plain = frame.as_plain(plugin)
    assert plain is not None
    proper = get_proper_type(plain)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "polars.dataframe.frame.DataFrame"
