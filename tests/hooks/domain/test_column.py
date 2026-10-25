"""Check `ColumnRef`, the plugin class that wraps the mypy type of one column.

A column is written `Column[Literal[name], data type]`. `ColumnRef`
reads its name and its data type, and builds a new column type from
an old one, for example with a new name. The fixtures of
`tests/hooks/conftest.py` build the mypy types by hand, so these tests
need no mypy build.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import AnyType, Instance, TypeOfAny
from mypy.types import Type as MypyType

from astrolabe._hooks.domain.column import ColumnRef
from tests.hooks.conftest import FakeApi, make_column_instance, make_column_ref

if TYPE_CHECKING:
    from mypy.test.typefixture import TypeFixture


def test_name_reads_the_literal_value(fixture: TypeFixture, int_dtype: Instance) -> None:
    column = make_column_ref(fixture, "foo", int_dtype)
    assert column.name == "foo"


def test_dtype_returns_the_raw_dtype_arg(fixture: TypeFixture, int_dtype: Instance) -> None:
    column = make_column_ref(fixture, "foo", int_dtype)
    assert column.dtype is int_dtype


def test_from_proper_type_wraps_a_column_instance(
    fixture: TypeFixture,
    int_dtype: Instance,
) -> None:
    instance = make_column_instance(fixture, "foo", int_dtype)
    column = ColumnRef.from_proper_type(instance)
    assert column is not None
    assert column.name == "foo"


def test_from_proper_type_rejects_a_non_instance(fixture: TypeFixture) -> None:
    not_an_instance: MypyType = AnyType(TypeOfAny.explicit)
    assert ColumnRef.from_proper_type(not_an_instance) is None


def test_with_name_changes_the_name_but_keeps_the_dtype(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    column = make_column_ref(fixture, "foo", int_dtype)
    renamed = column.with_name("bar", api)
    assert renamed.name == "bar"
    assert renamed.dtype is int_dtype


def test_with_name_reuses_the_same_type_info(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    column = make_column_ref(fixture, "foo", int_dtype)
    renamed = column.with_name("bar", api)
    assert renamed.as_instance().type is column.as_instance().type


def test_build_reuses_the_models_type_info(
    fixture: TypeFixture,
    int_dtype: Instance,
    api: FakeApi,
) -> None:
    model = make_column_ref(fixture, "foo", int_dtype)
    built = ColumnRef.build("bar", int_dtype, model, api)
    assert built.name == "bar"
    assert built.dtype is int_dtype
    assert built.as_instance().type is model.as_instance().type


def test_as_instance_round_trips(fixture: TypeFixture, int_dtype: Instance) -> None:
    instance = make_column_instance(fixture, "foo", int_dtype)
    assert ColumnRef(instance).as_instance() is instance
