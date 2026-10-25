"""Read the requirements of a declared remainder from the bases of its class.

A remainder is the set of columns of an open frame that its type does not name.
The user declares one with `other_columns(...)`, and can state requirements on
it there: for example `without=["id"]` says that it never holds a column named
`id`, and `disjoint_from=Other` says that it shares no name with another
declared remainder. `other_columns(...)` makes a class, and each requirement
becomes one extra base of that class: a marker such as `Without[...]` whose type
argument holds the value.

The requirements must live in the class itself. mypy keeps a cache of each
module that did not change and loads it on a later run without running the
plugin again. A class and its bases survive that cache, so this module can read
every requirement from the `TypeInfo` alone, on the first run and on every later
one.

The module reads the bases in three ways. The record is what the checker knows
about the remainder: the names that it cannot hold and the data types that its
columns can have. The predicate is the test on one column that a split
declaration uses to place each column in a part. The relations are the
requirements between two declared remainders, which only a call can check,
because only there are both remainders bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mypy.types import Instance, UnionType, get_proper_type

import astrolabe.remainder as markers
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.marker_minter import NameSetReads, ProperReads
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.remainder import absence
from astrolabe._hooks.remainder.record import Record
from astrolabe._hooks.encoding.family_codec import FamilyReads

if TYPE_CHECKING:
    from astrolabe._hooks.remainder.type_set import Family
    from collections.abc import Callable, Iterator

    from mypy.nodes import TypeInfo
    from mypy.types import Type as MypyType


class MarkedDeclarations:
    """Accept a class as a declared remainder if it has the `DeclaredRemainder` base.

    This is the declaration gate of a real run. `other_columns(...)` gives every
    class that it makes that base. The remainder reader accepts a plain instance
    as a remainder only if this gate accepts its class, so no other class reads
    as a remainder.

    Example:
        >>> fixture = new_type_fixture()
        >>> minter = new_marker_minter(fixture)
        >>> MarkedDeclarations().is_declaration(minter.resolved("kept.R"))
        False
    """

    def is_declaration(self, instance: Instance) -> bool:
        """Tell whether the class of one instance is a declared remainder.

        Args:
            instance: The plain instance to test.

        Returns:
            True if the class has the `DeclaredRemainder` base.
        """
        return instance.type.has_base(Target.of(markers.DeclaredRemainder).fullname)


@dataclass(frozen=True)
class Relations:
    """The requirements that tie one declared remainder to other declared remainders.

    Each entry is the set of full names that one requirement names.
    `disjoint_from=A | B` holds if the remainder shares no name with `A` and
    none with `B`, so a call can check each part alone. `within=A | B` holds
    only against the union of `A` and `B`, so the call must check the set as a
    whole. Each set therefore stays one entry.

    Attributes:
        disjoint_from: For each `disjoint_from=` requirement, the declared
            remainders that this one shares no name with.
        within: For each `within=` requirement, the declared remainders that
            hold every name of this one between them.

    Example:
        >>> Relations((), ()).disjoint_from
        ()
    """

    disjoint_from: tuple[frozenset[str], ...]
    within: tuple[frozenset[str], ...]


class DeclarationCodec:
    """Read the requirements of a declared remainder in the two forms that the checker needs.

    A frame hook needs the record of a remainder to know which names and data
    types it can hold. The check of a call needs the relations between
    declared remainders. Both come from the same requirement bases, so this
    class walks those bases and gives each form. It reads the arguments through
    `NameSetReads`, `ProperReads` and `FamilyReads`, and it builds no type.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_declaration_codec(fixture)
        >>> info = declaration_info(fixture, without=("ticker",))
        >>> codec.record_of(info).absence.holds_of("ticker")
        True
    """

    def __init__(self) -> None:
        """Set up the readers."""
        self._names = NameSetReads()
        self._reads = ProperReads()

    @propagates(None)
    def record_of(self, info: TypeInfo) -> Record | None:
        """Give the record that the requirements of a declaration state.

        A body of a function can rely on what its callers promise. So the record
        of a declared remainder comes from its own requirements. A requirement
        on names adds to the names that the remainder cannot hold. `AllOf`
        keeps only the data types of one family, and `NoneOf` takes the data
        types of one family out. A relational base states nothing about the
        columns of this remainder, so it does not change the record.

        Args:
            info: The declared remainder class.

        Returns:
            The record, or None if a requirement base is in a wrong form.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> bare = declaration_info(fixture)
            >>> codec.record_of(bare).absence
            Never()
            >>> minter = new_marker_minter(fixture)
            >>> apart = minter.marker(remainder_markers.DisjointFrom, [minter.resolved("kept.L")])
            >>> codec.record_of(declaration_info(fixture, apart)).absence
            Never()
        """
        rules = self._absence_rules()
        record = Record.unconstrained()
        for marker, argument in self._requirement_bases(info):
            rule = rules.get(marker)
            if rule is not None:
                tightened = absence.Or(record.absence, not_none(rule(argument)))
                record = Record(tightened, record.types)
            elif marker == "AllOf":
                family = not_none(self._family(argument))
                record = Record(record.absence, record.types.narrowed_to(family))
            elif marker == "NoneOf":
                family = not_none(self._family(argument))
                record = Record(record.absence, record.types.minus(family.type_set()))
        return record

    @propagates(None)
    def relations_of(self, info: TypeInfo) -> Relations | None:
        """Give the requirements that tie this declared remainder to others.

        A base that is not relational, for example `Without`, adds nothing.

        Args:
            info: The declared remainder class.

        Returns:
            The relations, or None if a relational base is in a wrong form.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> codec.relations_of(declaration_info(fixture, without=("a",)))
            Relations(disjoint_from=(), within=())
        """
        disjoint: list[frozenset[str]] = []
        within: list[frozenset[str]] = []
        for marker, argument in self._requirement_bases(info):
            if marker == "DisjointFrom":
                disjoint.append(not_none(self._declarations(argument)))
            if marker == "Within":
                within.append(not_none(self._declarations(argument)))
        return Relations(tuple(disjoint), tuple(within))

    def _requirement_bases(self, info: TypeInfo) -> Iterator[tuple[str, MypyType]]:
        """Walk the requirement bases of a declared remainder class.

        A requirement base is a marker class from the module of the markers,
        with a type argument. Any other base, for example `object`, is not one.

        Args:
            info: The declared remainder class.

        Yields:
            The name of the marker class of each requirement base, with its one
            type argument.
        """
        prefix = markers.Without.__module__ + "."
        for base in info.bases:
            if base.type.fullname.startswith(prefix) and base.args:
                yield base.type.fullname.removeprefix(prefix), base.args[0]

    def _absence_rules(self) -> dict[str, Callable[[MypyType], absence.Absence | None]]:
        """Map each requirement on names to a reader of the names that it rules out.

        Returns:
            One reader for each marker. Each reader gives the rule for the names
            that the remainder cannot hold.

        Example:
            >>> fixture = new_type_fixture()
            >>> sorted(new_declaration_codec(fixture)._absence_rules())[:2]
            ['OnlyNames', 'OnlyPrefix']
        """
        return {
            "Without": self._one_of,
            "OnlyNames": lambda a: self._negated(self._one_of(a)),
            "WithoutPrefix": self._starts_with,
            "OnlyPrefix": lambda a: self._negated(self._starts_with(a)),
            "WithoutSuffix": self._ends_with,
            "OnlySuffix": lambda a: self._negated(self._ends_with(a)),
        }

    def _one_of(self, argument: MypyType) -> absence.Absence | None:
        """Read a list of names as a rule that holds for exactly those names.

        Args:
            argument: The type argument that holds the set of names.

        Returns:
            The rule, or None if the argument is not a set of names.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> codec._one_of(minter.string("a"))
            OneOf(names=frozenset({'a'}))
        """
        names = self._names.read(argument)
        if names is None:
            return None
        return absence.OneOf(names)

    def _starts_with(self, argument: MypyType) -> absence.Absence | None:
        """Read a prefix as a rule that holds for the names that start with it.

        Args:
            argument: The type argument that holds the prefix as a string
                literal.

        Returns:
            The rule, or None if the argument is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> codec._starts_with(minter.string("sec_"))
            StartsWith(prefix='sec_')
            >>> codec._starts_with(minter.resolved("kept.R")) is None
            True
        """
        prefix = self._reads.string_literal(argument)
        if prefix is None:  # pragma: no cover -- other_columns writes a literal
            return None
        return absence.StartsWith(prefix)

    def _ends_with(self, argument: MypyType) -> absence.Absence | None:
        """Read a suffix as a rule that holds for the names that end with it.

        Args:
            argument: The type argument that holds the suffix as a string
                literal.

        Returns:
            The rule, or None if the argument is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> codec._ends_with(minter.string("_prev"))
            EndsWith(suffix='_prev')
            >>> codec._ends_with(minter.resolved("kept.R")) is None
            True
        """
        suffix = self._reads.string_literal(argument)
        if suffix is None:  # pragma: no cover -- other_columns writes a literal
            return None
        return absence.EndsWith(suffix)

    def _negated(self, rule: absence.Absence | None) -> absence.Absence | None:
        """Give the negation of a rule on names, and pass None through.

        Args:
            rule: The rule to negate, or None.

        Returns:
            The negation, or None.

        Example:
            >>> fixture = new_type_fixture()
            >>> new_declaration_codec(fixture)._negated(None) is None
            True
        """
        if rule is None:  # pragma: no cover -- other_columns writes a literal
            return None
        return absence.Not(rule)

    def _family(self, argument: MypyType) -> Family | None:
        """Read one type argument as a family of data types.

        Args:
            argument: The type argument that holds the family.

        Returns:
            The family, or None for a class that is not a family.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> codec._family(minter.resolved("polars.datatypes.classes.String"))
            String
        """
        return FamilyReads(required=True).read(argument)

    @propagates(None)
    def _declarations(self, argument: MypyType) -> frozenset[str] | None:
        """Read a relational requirement as the full names of the declared remainders that it names.

        Args:
            argument: One declared remainder class, or a union of them.

        Returns:
            The full names, or None for a type of another form, or for a union
            with an item that is not a class.

        Example:
            >>> from mypy.types import UnionType
            >>> fixture = new_type_fixture()
            >>> codec = new_declaration_codec(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> codec._declarations(minter.resolved("kept.L"))
            frozenset({'kept.L'})
            >>> pair = UnionType([minter.resolved("decl.A")])
            >>> codec._declarations(pair)
            frozenset({'decl.A'})
        """
        proper = get_proper_type(argument)
        items = proper.items if isinstance(proper, UnionType) else [proper]
        found = [not_none(self._reads.instance(item)) for item in items]
        return frozenset(one.type.fullname for one in found)
