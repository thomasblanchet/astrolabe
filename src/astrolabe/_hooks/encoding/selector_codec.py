"""Write a selector as a mypy type, and read it back.

A selector picks columns by their name, by their data type, or by a mix of both,
for example `cs.starts_with("s_")` or `cs.numeric()`. The engine uses selectors
in two places. A selection type holds the selector that picks its columns. A
remainder can hold a filter that is a selector, where the remainder is the part
of an open frame that its type does not name. Both must travel inside a mypy
type, so a selector needs a type too.

The codec writes a test on names by what it accepts, not by how the user wrote
it. A finite set of names becomes `OneOf[...]`. Any other test becomes
`Matching[...]`, with a text that only the names fix: a regular expression of
the names where it reads well, as `ab.*` for the names that begin with `ab`, or
else the text form of the smallest automaton that accepts them. Two selectors
that pick the same columns therefore get the same type. The forms `StartsWith`,
`EndsWith` and `Contains` are for the reader only: a user can write them in a
declaration and the checker reads them, but the codec never writes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import Instance, LiteralType, get_proper_type
from mypy.types import Type as MypyType

import astrolabe.remainder as markers
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.family_codec import FamilyCodec, FamilyReads
from astrolabe._hooks.encoding.marker_minter import (
    MarkerMinter,
    NameSetCodec,
    NameSetReads,
)
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._hooks.remainder import absence
from astrolabe._hooks.remainder.name_language import NameSet
from astrolabe._hooks.remainder.selector import (
    NameTest,
    OfType,
    SelAnd,
    Selector,
    SelNot,
    SelOr,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class SelectorCodec:
    """Build the mypy type for one selector, and read it back.

    A hook that types a selector expression, or that changes the remainder of a
    frame with a selector, writes the selector through this class. The class
    writes each test on names in its normal form, so the type of a selector
    depends only on the columns that it picks. It uses `NameSetCodec` for sets
    of names and `FamilyCodec` for tests on data types. The reading half is
    `SelectorReader`.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_selector_codec(fixture)
        >>> from astrolabe._hooks.remainder.absence import OneOf
        >>> codec.encoded(NameTest(OneOf(frozenset({"a", "b"}))))
        astrolabe.remainder.OneOf[Literal['a'] | Literal['b']]
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter and the codec for sets of names.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._names = NameSetCodec(minter)

    @propagates(None)
    def encoded(self, selector: Selector) -> MypyType | None:
        """Build the type of one selector.

        Args:
            selector: The selector from the engine.

        Returns:
            The marker type, or None if a family of data types has no polars
            class or member type.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> spelled = str(codec.encoded(SelNot(OfType(Family.numeric()))))
            >>> spelled.startswith("astrolabe.remainder.SelNot[")
            True
            >>> "OfType[polars.datatypes.classes.NumericType]" in spelled
            True
        """
        if isinstance(selector, NameTest):
            return self._encoded_name_test(selector)
        if isinstance(selector, OfType):
            return self._encoded_family(selector)
        return self._encoded_combinator(selector)

    @propagates(None)
    def _encoded_combinator(self, selector: Selector) -> MypyType | None:
        """Build the type of a negation, a conjunction or a disjunction of selectors.

        Args:
            selector: The selector to write.

        Returns:
            The `SelNot`, `SelAnd` or `SelOr` type over the types of its sides,
            or None if a side has no type.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> both = SelAnd(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> spelled = str(codec._encoded_combinator(both))
            >>> spelled.count("astrolabe.remainder.OfType[")
            2
        """
        if isinstance(selector, SelNot):
            inner = not_none(self.encoded(selector.inner))
            return self._minter.marker(markers.SelNot, [inner])
        pair = checked_pair(selector)
        left = not_none(self.encoded(pair[0]))
        right = not_none(self.encoded(pair[1]))
        cls = markers.SelAnd if isinstance(selector, SelAnd) else markers.SelOr
        return self._minter.marker(cls, [left, right])

    def _encoded_name_test(self, selector: NameTest) -> MypyType | None:
        """Build the type of a test on names by the names that it accepts.

        Args:
            selector: The test on names.

        Returns:
            `OneOf[...]` for a finite set of names. Otherwise `Matching[...]`
            with the spelling of the names (`NameSet.spelling`).

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> str(codec._encoded_name_test(NameTest(StartsWith("a"))))
            "astrolabe.remainder.Matching[Literal['a.*']]"
        """
        language = selector.rule.language()
        names = language.finite_names()
        if names is not None:
            return self._minter.marker(markers.OneOf, [self._names.built(names)])
        spelling = self._minter.string(language.spelling())
        return self._minter.marker(markers.Matching, [spelling])

    def _encoded_family(self, selector: OfType) -> MypyType | None:
        """Build the type of a test on data types.

        Args:
            selector: The test on data types.

        Returns:
            The `OfType[...]` type over the type of the family, or None if a
            member of the family has no type.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> codec._encoded_family(OfType(Family.temporal()))
            astrolabe.remainder.OfType[polars.datatypes.classes.TemporalType]
            >>> codec._encoded_family(OfType(Family("Int", frozenset({"Int64"}))))
            astrolabe.remainder.OfType[polars.datatypes.classes.Int64]
            >>> weird = OfType(Family("Weird", frozenset({"Nothing(x)"})))
            >>> codec._encoded_family(weird) is None
            True
        """
        family = FamilyCodec(self._minter).built(selector.family)
        if family is None:
            return None
        return self._minter.marker(markers.OfType, [family])

    def decoded(self, typ: MypyType) -> Selector | None:
        """Read the type of a selector back into the selector.

        Args:
            typ: The type to read.

        Returns:
            The selector, or None for a type that is not a selector.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.absence import OneOf
            >>> spelled = codec.encoded(NameTest(OneOf(frozenset({"a"}))))
            >>> codec.decoded(spelled)
            NameTest(rule=OneOf(names=frozenset({'a'})))
        """
        return SelectorReader().decoded(typ)


class SelectorReader:
    """Read the type of a selector back into the selector, with no access to the plugin.

    This is the reading half of `SelectorCodec`. It reads every selector marker,
    the forms that only a user writes among them. Code that only reads
    selectors, and builds nothing, uses it alone.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> SelectorReader().decoded(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the reader of sets of names."""
        self._names = NameSetReads()

    def decoded(self, typ: MypyType) -> Selector | None:
        """Read one type as a selector.

        The full name of the marker class selects the reader.

        Args:
            typ: The type to read.

        Returns:
            The selector, or None for a type that is not a selector.

        Example:
            >>> from mypy.types import Instance
            >>> from astrolabe.remainder import Contains
            >>> reader = SelectorReader()
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> reader.decoded(minter.string("a")) is None
            True
            >>> reader.decoded(Instance(fixture.oi, [])) is None
            True
            >>> reader.decoded(minter.marker(Contains, [minter.string("tmp")]))
            NameTest(rule=Contains(part='tmp'))
        """
        proper = get_proper_type(typ)
        if not isinstance(proper, Instance):
            return None
        by_marker: dict[type, Callable[[Instance], Selector | None]] = {
            markers.StartsWith: self._read_starts_with,
            markers.EndsWith: self._read_ends_with,
            markers.Contains: self._read_contains,
            markers.OneOf: self._read_one_of,
            markers.Matching: self._read_matching,
            markers.OfType: self._read_of_type,
            markers.SelNot: self._read_not,
            markers.SelAnd: self._read_and,
            markers.SelOr: self._read_or,
        }
        readers = {Target.of(marker).fullname: reader for marker, reader in by_marker.items()}
        reader = readers.get(proper.type.fullname)
        if reader is None:
            return None
        return reader(proper)

    @propagates(None)
    def _read_starts_with(self, instance: Instance) -> Selector | None:
        """Read a `StartsWith[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The test on names, or None if the argument is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = SelectorReader()
            >>> minter = new_marker_minter(fixture)
            >>> from astrolabe.remainder import StartsWith
            >>> spelled = minter.marker(StartsWith, [minter.string("s_")])
            >>> reader._read_starts_with(spelled)
            NameTest(rule=StartsWith(prefix='s_'))
        """
        return NameTest(absence.StartsWith(not_none(self._literal(instance, 0))))

    @propagates(None)
    def _read_ends_with(self, instance: Instance) -> Selector | None:
        """Read an `EndsWith[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The test on names, or None if the argument is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = SelectorReader()
            >>> minter = new_marker_minter(fixture)
            >>> from astrolabe.remainder import EndsWith
            >>> spelled = minter.marker(EndsWith, [minter.string("_p")])
            >>> reader._read_ends_with(spelled)
            NameTest(rule=EndsWith(suffix='_p'))
        """
        return NameTest(absence.EndsWith(not_none(self._literal(instance, 0))))

    @propagates(None)
    def _read_contains(self, instance: Instance) -> Selector | None:
        """Read a `Contains[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The test on names, or None if the argument is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> reader = SelectorReader()
            >>> minter = new_marker_minter(fixture)
            >>> from astrolabe.remainder import Contains
            >>> spelled = minter.marker(Contains, [minter.string("tmp")])
            >>> reader._read_contains(spelled)
            NameTest(rule=Contains(part='tmp'))
        """
        return NameTest(absence.Contains(not_none(self._literal(instance, 0))))

    @propagates(None)
    def _read_one_of(self, instance: Instance) -> Selector | None:
        """Read a `OneOf[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The test on the listed names, or None if the argument is not a set
            of names.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.absence import OneOf
            >>> spelled = codec.encoded(NameTest(OneOf(frozenset({"a"}))))
            >>> SelectorReader()._read_one_of(spelled)
            NameTest(rule=OneOf(names=frozenset({'a'})))
        """
        names = not_none(self._names.read(not_none(self._argument(instance, 0))))
        return NameTest(absence.OneOf(names))

    @propagates(None)
    def _read_matching(self, instance: Instance) -> Selector | None:
        """Read a `Matching[...]` type, and build the automaton again from its text.

        The checker writes that text itself, but a user can write a
        `Matching[...]` by hand with any string. A text that is not the
        spelling of its names is not one (`NameSet.from_spelling`), and
        neither is a finite set of names, which the codec writes as
        `OneOf[...]`.

        Args:
            instance: The marker instance.

        Returns:
            The test on names, or None if the argument is not a string literal
            or not a spelling.

        Example:
            >>> from astrolabe.remainder import Matching
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> spelled = codec.encoded(NameTest(StartsWith("ab")))
            >>> SelectorReader()._read_matching(spelled).picks("abc", "Int64")
            True
            >>> minter = new_marker_minter(fixture)
            >>> by_hand = minter.marker(Matching, [minter.string("^p$")])
            >>> SelectorReader()._read_matching(by_hand) is None
            True
            >>> another = minter.marker(Matching, [minter.string("a(b)?.*")])
            >>> SelectorReader()._read_matching(another) is None
            True
        """
        text = not_none(self._literal(instance, 0))
        try:
            names = NameSet.from_spelling(text)
        except ValueError:
            return None
        not_falsy(names.finite_names() is None)
        return NameTest(absence.Matching.of(names))

    @propagates(None)
    def _read_of_type(self, instance: Instance) -> Selector | None:
        """Read an `OfType[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The test on data types, or None for a class that is not a family.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> spelled = codec.encoded(OfType(Family.numeric()))
            >>> SelectorReader()._read_of_type(spelled).family
            NumericType
        """
        return OfType(not_none(FamilyReads().read(not_none(self._argument(instance, 0)))))

    @propagates(None)
    def _read_not(self, instance: Instance) -> Selector | None:
        """Read a `SelNot[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The negation of the selector inside, or None if that selector is in
            a wrong form.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> spelled = codec.encoded(SelNot(OfType(Family.numeric())))
            >>> SelectorReader()._read_not(spelled).inner.family
            NumericType
        """
        return SelNot(not_none(self.decoded(not_none(self._argument(instance, 0)))))

    @propagates(None)
    def _read_and(self, instance: Instance) -> Selector | None:
        """Read a `SelAnd[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The conjunction of the two selectors inside, or None if one of them
            is in a wrong form.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> pair = SelAnd(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> SelectorReader()._read_and(codec.encoded(pair)) == pair
            True
        """
        left = not_none(self.decoded(not_none(self._argument(instance, 0))))
        right = not_none(self.decoded(not_none(self._argument(instance, 1))))
        return SelAnd(left, right)

    @propagates(None)
    def _read_or(self, instance: Instance) -> Selector | None:
        """Read a `SelOr[...]` type.

        Args:
            instance: The marker instance.

        Returns:
            The disjunction of the two selectors inside, or None if one of them
            is in a wrong form.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_selector_codec(fixture)
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> pair = SelOr(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> SelectorReader()._read_or(codec.encoded(pair)) == pair
            True
        """
        left = not_none(self.decoded(not_none(self._argument(instance, 0))))
        right = not_none(self.decoded(not_none(self._argument(instance, 1))))
        return SelOr(left, right)

    def _argument(self, instance: Instance, position: int) -> MypyType | None:
        """Read one type argument of a marker by its position.

        Args:
            instance: The marker instance.
            position: The position of the argument.

        Returns:
            The argument, or None if the marker has fewer arguments.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> from astrolabe.remainder import Matching
            >>> one = minter.marker(Matching, [minter.string("x")])
            >>> SelectorReader()._argument(one, 1) is None
            True
        """
        if position >= len(instance.args):
            return None
        return instance.args[position]

    def _literal(self, instance: Instance, position: int) -> str | None:
        """Read one type argument of a marker as a string literal.

        Args:
            instance: The marker instance.
            position: The position of the argument.

        Returns:
            The string that the literal holds, or None if the argument is
            missing or is not a string literal.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> from astrolabe.remainder import Contains
            >>> one = minter.marker(Contains, [minter.string("tmp")])
            >>> SelectorReader()._literal(one, 0)
            'tmp'
            >>> SelectorReader()._literal(one, 5) is None
            True
            >>> bare = minter.marker(Contains, [minter.resolved("kept.R")])
            >>> SelectorReader()._literal(bare, 0) is None
            True
        """
        argument = self._argument(instance, position)
        if argument is None:
            return None
        proper = get_proper_type(argument)
        if isinstance(proper, LiteralType) and isinstance(proper.value, str):
            return proper.value
        return None


def checked_pair(selector: Selector) -> tuple[Selector, Selector]:
    """Give the two sides of a conjunction or a disjunction of selectors.

    The encoder handles every other form of selector first, so only a selector
    with two sides gets here. If that invariant breaks, the function raises an
    error in place of a wrong answer.

    Args:
        selector: The selector, which is not a test on names, a test on data
            types or a negation.

    Returns:
        The two sides.

    Raises:
        TypeError: The selector has no two sides. That is a bug in this code.

    Example:
        >>> from astrolabe._hooks.remainder.absence import Always
        >>> left = NameTest(Always())
        >>> checked_pair(SelOr(left, left))[0] is left
        True
    """
    if not isinstance(selector, SelAnd | SelOr):  # pragma: no cover
        # The encoder handles the tests and the negation first, so only
        # a conjunction or a disjunction gets here.
        problem = f"not a two-sided selector: {selector!r}"
        raise TypeError(problem)
    return (selector.left, selector.right)
