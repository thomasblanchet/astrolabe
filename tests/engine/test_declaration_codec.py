"""Check that each requirement on a declared remainder reads back as the rule it states.

A user declares a remainder with `other_columns(...)`: the columns of
a frame that its type does not name. Keywords add requirements, for
example `without=["ticker"]` or `all_of=pl.NUMERIC_DTYPES`. The plugin
stores each requirement as a marker base class of the declared class,
and the declaration codec reads those bases back.

Each requirement is read in up to three ways, and each way has tests
here. The record is what the checker then knows about the remainder:
the names it cannot hold and the data types it can hold. The predicate
decides, one column at a time, whether a column belongs to one part of
a `SplitColumns` declaration. The relations are the requirements that
compare two declarations, which the checker can only check at a call.
A base with a wrong shape reads as None, never as a weaker rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.test.typefixture import TypeFixture
from mypy.types import Instance, UnionType

import astrolabe.remainder as markers
from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
)
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.encoding.declaration_codec import (
    DeclarationCodec,
    MarkedDeclarations,
    Relations,
)
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, NameSetCodec
from astrolabe._hooks.remainder.record import ColumnSet
from tests.support import FakeApi, FakePlugin

if TYPE_CHECKING:
    from mypy.nodes import TypeInfo
    from mypy.types import Type as MypyType


class Declared:
    """Build declared remainder classes and their requirement bases for one test.

    All classes of one test come from one `TypeFixture`, so the types
    that they share compare equal.
    """

    def __init__(self) -> None:
        """Make the fixture, the minter of marker types, and the codec under test."""
        self.fixture = TypeFixture()
        self.minter = MarkerMinter(
            FakePlugin(self.fixture),
            CheckerLiteralTypeBuilder(),
            FakeApi(self.fixture),
        )
        self.names = NameSetCodec(self.minter)
        self.codec = DeclarationCodec()

    def info(self, *requirements: Instance) -> TypeInfo:
        """Build the class that `other_columns(...)` registers, with the given requirements.

        Args:
            *requirements: The requirement bases, from the other methods.

        Returns:
            The declared class. Its full name is `decl.Rest`.
        """
        marked = checked_not_none(self.minter.marker(markers.DeclaredRemainder, []))
        bases = [Instance(self.fixture.oi, []), marked, *requirements]
        return self.fixture.make_type_info(
            "decl.Rest",
            bases=bases,
            mro=[marked.type, self.fixture.oi],
        )

    def base(self, marker: type, argument: MypyType) -> Instance:
        """Build one requirement base from its marker class and its one argument.

        Args:
            marker: The marker class of the requirement, for example `Without`.
            argument: The type argument of the marker.

        Returns:
            The base.
        """
        return checked_not_none(self.minter.marker(marker, [argument]))

    def name_base(self, marker: type, *names: str) -> Instance:
        """Build a requirement that lists column names, for `without=` or `only=`.

        Args:
            marker: `Without` or `OnlyNames`.
            *names: The names.

        Returns:
            The base.
        """
        return self.base(marker, self.names.built(frozenset(names)))

    def text_base(self, marker: type, text: str) -> Instance:
        """Build a requirement on a prefix or a suffix of the column names.

        Args:
            marker: One of `WithoutPrefix`, `OnlyPrefix`, `WithoutSuffix`
                and `OnlySuffix`.
            text: The prefix or the suffix.

        Returns:
            The base.
        """
        return self.base(marker, self.minter.string(text))

    def family_base(self, marker: type, fullname: str) -> Instance:
        """Build a requirement on the data types of the columns, for `all_of=` or `none_of=`.

        Args:
            marker: `AllOf` or `NoneOf`.
            fullname: The full name of a polars data type class. A class
                such as `NumericType` stands for its whole family.

        Returns:
            The base.
        """
        return self.base(marker, checked_not_none(self.minter.resolved(fullname)))


class TestTheGate:
    """Only a class with the base `DeclaredRemainder` reads as a declared remainder."""

    def test_the_marked_base_decides(self) -> None:
        """The class with the base is a declaration, and a plain class is not."""
        declared = Declared()
        info = declared.info()
        assert MarkedDeclarations().is_declaration(Instance(info, []))
        plain = declared.fixture.make_type_info("kept.Other")
        assert not MarkedDeclarations().is_declaration(Instance(plain, []))


class TestRecords:
    """Each requirement gives a record with exactly the names and data types it states."""

    def test_without_proves_the_names_absent(self) -> None:
        """`without=` makes the listed names absent, and no other name."""
        declared = Declared()
        info = declared.info(declared.name_base(markers.Without, "a", "b"))
        record = declared.codec.record_of(info)
        assert record is not None
        assert record.absence.holds_of("a")
        assert not record.absence.holds_of("c")

    def test_only_names_proves_everything_else_absent(self) -> None:
        """`only=` makes every name that it does not list absent."""
        declared = Declared()
        info = declared.info(declared.name_base(markers.OnlyNames, "a"))
        record = declared.codec.record_of(info)
        assert record is not None
        assert record.absence.holds_of("z")
        assert not record.absence.holds_of("a")

    def test_the_four_affix_requirements_prove_their_regions(self) -> None:
        """Each prefix or suffix requirement makes the right set of names absent."""
        declared = Declared()
        cases = (
            (markers.WithoutPrefix, "sec_", "sec_x", "x"),
            (markers.OnlyPrefix, "sec_", "x", "sec_x"),
            (markers.WithoutSuffix, "_p", "x_p", "x"),
            (markers.OnlySuffix, "_p", "x", "x_p"),
        )
        for marker, affix, absent, present in cases:
            info = declared.info(declared.text_base(marker, affix))
            record = declared.codec.record_of(info)
            assert record is not None, marker
            assert record.absence.holds_of(absent), marker
            assert not record.absence.holds_of(present), marker

    def test_all_of_narrows_the_type_set(self) -> None:
        """`all_of=` keeps only the data types of the family."""
        declared = Declared()
        numeric = declared.family_base(markers.AllOf, "polars.datatypes.classes.NumericType")
        record = declared.codec.record_of(declared.info(numeric))
        assert record is not None
        assert "Int64" in record.types.members
        assert "String" not in record.types.members

    def test_none_of_removes_the_family(self) -> None:
        """`none_of=` takes the data types of the family out of the record."""
        declared = Declared()
        no_text = declared.family_base(markers.NoneOf, "polars.datatypes.classes.String")
        record = declared.codec.record_of(declared.info(no_text))
        assert record is not None
        assert "String" not in record.types.members
        assert "Int64" in record.types.members

    def test_requirements_combine(self) -> None:
        """Two keywords on one declaration both hold."""
        declared = Declared()
        info = declared.info(
            declared.name_base(markers.Without, "audit_id"),
            declared.text_base(markers.WithoutPrefix, "sec_"),
        )
        record = declared.codec.record_of(info)
        assert record is not None
        assert record.absence.holds_of("audit_id")
        assert record.absence.holds_of("sec_x")
        assert not record.absence.holds_of("venue")

    def test_a_united_family_is_the_union_of_its_parts(self) -> None:
        """A union of data type classes stands for every data type of each class."""
        declared = Declared()
        pair = UnionType([
            checked_not_none(declared.minter.resolved("polars.datatypes.classes.String")),
            checked_not_none(declared.minter.resolved("polars.datatypes.classes.Boolean")),
        ])
        info = declared.info(declared.base(markers.NoneOf, pair))
        record = declared.codec.record_of(info)
        assert record is not None
        assert {"String", "Boolean"} & record.types.members == set()

    def test_the_record_describes_a_lawful_column_set(self) -> None:
        """The record accepts a set of columns that obeys every requirement, and no other."""
        declared = Declared()
        info = declared.info(
            declared.name_base(markers.Without, "ticker"),
            declared.family_base(markers.AllOf, "polars.datatypes.classes.NumericType"),
        )
        record = declared.codec.record_of(info)
        assert record is not None
        assert record.describes(ColumnSet.of(price="Int64"))
        assert not record.describes(ColumnSet.of(ticker="Int64"))
        assert not record.describes(ColumnSet.of(venue="String"))

    def test_a_malformed_base_reads_as_none(self) -> None:
        """A base whose argument has the wrong shape gives no record."""
        declared = Declared()
        broken = declared.base(
            markers.Without,
            checked_not_none(declared.minter.resolved("kept.NotNames")),
        )
        assert declared.codec.record_of(declared.info(broken)) is None
        unknown = declared.base(
            markers.AllOf,
            checked_not_none(declared.minter.resolved("kept.NoFamily")),
        )
        assert declared.codec.record_of(declared.info(unknown)) is None


class TestRelations:
    """`disjoint_from=` and `within=` read as sets of declared class names, one set per base."""

    def test_one_declaration_reads_as_a_singleton_set(self) -> None:
        """One declared class as the argument reads as a set of one name."""
        declared = Declared()
        other = checked_not_none(declared.minter.resolved("decl.L"))
        info = declared.info(declared.base(markers.DisjointFrom, other))
        assert declared.codec.relations_of(info) == Relations((frozenset({"decl.L"}),), ())

    def test_a_set_of_parts_stays_one_set(self) -> None:
        """A union of classes stays one set, because the rule is about their union."""
        declared = Declared()
        both = UnionType([
            checked_not_none(declared.minter.resolved("decl.A")),
            checked_not_none(declared.minter.resolved("decl.B")),
        ])
        info = declared.info(declared.base(markers.Within, both))
        assert declared.codec.relations_of(info) == Relations(
            (),
            (frozenset({"decl.A", "decl.B"}),),
        )

    def test_a_malformed_relational_base_reads_as_none(self) -> None:
        """A literal in place of a class does not read."""
        declared = Declared()
        broken = declared.base(markers.Within, declared.minter.string("L"))
        assert declared.codec.relations_of(declared.info(broken)) is None

    def test_a_malformed_union_item_reads_as_none(self) -> None:
        """One bad member of a union makes the whole base unreadable."""
        declared = Declared()
        mixed = UnionType([
            checked_not_none(declared.minter.resolved("decl.A")),
            declared.minter.string("B"),
        ])
        info = declared.info(declared.base(markers.DisjointFrom, mixed))
        assert declared.codec.relations_of(info) is None
