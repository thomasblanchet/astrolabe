"""Give the fixtures that a unit test of a plugin class takes, with no mypy build.

A unit test of a class in `_hooks/domain/` or `_hooks/calls/` needs mypy
objects: a `TypeFixture`, a stand-in for the checker API, a stand-in
for the plugin, and a few data types. Each fixture here builds one of
them with a builder from `tests/support.py`. A test module imports the
fixture types from this module under `TYPE_CHECKING`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import support
from tests.support import FakeApi, FakePlugin, column_instance
from tests.support import column_ref as make_column_ref
from tests.support import frame as make_frame

make_column_instance = column_instance

if TYPE_CHECKING:
    from mypy.test.typefixture import TypeFixture
    from mypy.types import Instance

__all__ = [
    "FakeApi",
    "FakePlugin",
    "make_column_instance",
    "make_column_ref",
    "make_frame",
]


@pytest.fixture
def fixture() -> TypeFixture:
    return support.new_type_fixture()


@pytest.fixture
def api(fixture: TypeFixture) -> FakeApi:
    return support.new_api(fixture)


@pytest.fixture
def plugin(fixture: TypeFixture) -> FakePlugin:
    return support.new_plugin(fixture)


@pytest.fixture
def int_dtype(fixture: TypeFixture) -> Instance:
    return support.int_dtype(fixture)


@pytest.fixture
def str_dtype(fixture: TypeFixture) -> Instance:
    return support.str_dtype(fixture)
