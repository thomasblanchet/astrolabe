"""Write the remainder of an open frame as a mypy type, and read it back.

An open frame has columns that its type does not name, and those columns are its
remainder. The user declares a remainder with `other_columns(...)`, which makes
a class that stands for the unknown columns. Frame operations then change the
remainder: `drop` takes names out of it, a selection in `with_columns` can
retype it, and `join` can add a suffix to its names. The engine holds the result
in a normal form with five parts, and the frame type must hold that form,
because between two calls the remainder lives only in the type.

The five parts nest in one fixed order, the outermost first:

    Retyped?  over  Prefixed?  over  Suffixed?  over
    filter?  over  (declaration | Both)

The base is the declared class, or `Both[...]` for the union of several
remainders. The filter is `Minus[...]` if it takes a finite set of names out,
`Except[...]` for any other filter that takes columns out, and `Only[...]` for a
filter that keeps the columns that a selector picks. A remainder that the
checker proves empty has no type at all: the frame type then has no remainder,
and the frame is closed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import astrolabe.remainder as markers
from astrolabe.remainder import Prefixed as _PrefixedMarker
from astrolabe.remainder import Suffixed as _SuffixedMarker
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.encoding.marker_minter import (
    MarkerMinter,
    NameSetCodec,
    NameSetReads,
    ProperReads,
)
from astrolabe._hooks.encoding.rendering_codec import (
    ConstDtypeCodec,
    ConstDtypeReads,
)
from astrolabe._hooks.encoding.selector_codec import SelectorCodec, SelectorReader
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.remainder.absence import OneOf
from astrolabe._hooks.remainder.canonical import (
    Base,
    Canonical,
    Nothing,
    Union,
    Variable,
)
from astrolabe._hooks.remainder.selector import NameTest, Selector, SelNot

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from mypy.types import Instance
    from mypy.types import Type as MypyType


class DeclarationGate(Protocol):
    """Decide whether a class is a declared remainder, one that `other_columns(...)` made.

    The base of a remainder type is a plain instance of a class. Any other plain
    instance, for example a data type class, has the same shape. The reader
    accepts a plain instance as a remainder only if the gate accepts its class,
    so a type that is not a remainder never reads as one.
    """

    def is_declaration(self, instance: Instance) -> bool:
        """Tell whether the class of one instance is a declared remainder.

        Args:
            instance: The plain instance to test.

        Returns:
            True for a declared remainder.
        """
        ...


class RemainderCodec:
    """Build the mypy type for the remainder of an open frame, and read it back.

    Every hook that gives a frame type with a remainder writes the remainder
    through this class. The class writes only the normal form that the engine
    makes, so one remainder always gives one type, and mypy can compare two
    frame types without help. It uses `SelectorCodec` for the filters,
    `NameSetCodec` for sets of names and `ConstDtypeCodec` for the data type of
    a retype. The reading half is `RemainderReader`.

    Example:
        >>> fixture = new_type_fixture()
        >>> codec = new_remainder_codec(fixture, {"kept.R"})
        >>> codec.encoded(Canonical.of(Variable("kept.R")))
        kept.R
        >>> codec.encoded(Canonical.of(Nothing())) is None
        True
    """

    def __init__(self, minter: MarkerMinter, gate: DeclarationGate) -> None:
        """Keep the minter, the gate and the codecs for the parts.

        Args:
            minter: The marker minter for the current phase.
            gate: The test that tells a declared remainder from any other plain
                instance.
        """
        self._minter = minter
        self._gate = gate
        self._selectors = SelectorCodec(minter)
        self._names = NameSetCodec(minter)
        self._reads = ProperReads()

    def encoded(self, form: Canonical) -> MypyType | None:
        """Build the type of one remainder in normal form.

        Args:
            form: The remainder, in the normal form that `normalise` gives.

        Returns:
            The type, or None for a remainder that the checker proves empty. The
            frame type then has no remainder at all.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> named = Canonical(Variable("kept.R"), suffix="_p")
            >>> print(codec.encoded(named))
            astrolabe.remainder.Suffixed[kept.R, Literal['_p']]
        """
        if isinstance(form.base, Nothing):
            return None
        spelled = self._based(form.base)
        spelled = self._filtered(spelled, form.where)
        spelled = self._affixed(spelled, form)
        return self._retyped(spelled, form.retype)

    def _based(self, base: Base) -> MypyType:
        """Build the base, the innermost part.

        Args:
            base: A declared remainder or a union. The empty base never gets
                here.

        Returns:
            The instance of the declared class, or a chain of `Both[...]` over
            the sorted parts of the union.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> codec._based(Variable("kept.R"))
            kept.R
        """
        if isinstance(base, Variable):
            return checked_not_none(self._minter.resolved(base.name))
        parts = [self.encoded(part) for part in checked_union(base).parts]
        present = [checked_not_none(part) for part in parts]
        return self._both_chain(present)

    def _both_chain(self, parts: list[MypyType]) -> MypyType:
        """Nest the parts of a union into a chain of `Both[...]` that leans to the right.

        Args:
            parts: The types of the parts, in the order of the normal form.
                There is at least one.

        Returns:
            The one part, or `Both[first, rest]`.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R", "kept.S"})
            >>> a = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> b = codec.encoded(Canonical.of(Variable("kept.S")))
            >>> print(codec._both_chain([a, b]))
            astrolabe.remainder.Both[kept.R, kept.S]
        """
        head = parts[0]
        if len(parts) == 1:
            return head
        rest = self._both_chain(parts[1:])
        return checked_not_none(self._minter.marker(markers.Both, [head, rest]))

    def _filtered(self, spelled: MypyType, where: Selector | None) -> MypyType:
        """Put the filter around the base, if the remainder has one.

        Args:
            spelled: The type of the base.
            where: The filter in normal form, or None for no filter.

        Returns:
            `Minus[...]`, `Except[...]` or `Only[...]` around the base, or the
            base alone.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> base = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> codec._filtered(base, None)
            kept.R
        """
        if where is None:
            return spelled
        removed = self._removed_names(where)
        if removed is not None:
            names = self._names.built(removed)
            return checked_not_none(self._minter.marker(markers.Minus, [spelled, names]))
        return self._selector_filtered(spelled, where)

    def filtered(self, spelled: MypyType, where: Selector) -> MypyType:
        """Put a filter that keeps the columns that a selector picks around one type.

        Code that splits a remainder into parts, one part for each filter, uses
        this to write each part.

        Args:
            spelled: The type to filter.
            where: The selector that the kept columns match.

        Returns:
            The `Only[...]` type, or `Except[...]` for a selector that is a
            negation.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> base = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> str(codec.filtered(base, OfType(Family.numeric()))).startswith(
            ...     "astrolabe.remainder.Only["
            ... )
            True
        """
        return self._selector_filtered(spelled, where)

    def _selector_filtered(self, spelled: MypyType, where: Selector) -> MypyType:
        """Put a filter around the base that does not take out a finite set of names.

        Args:
            spelled: The type of the base.
            where: The filter in normal form.

        Returns:
            `Except[...]` over the inside of a negation, and `Only[...]`
            otherwise.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> base = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> spelled = str(codec._selector_filtered(base,
            ...     SelNot(OfType(Family.numeric()))))
            >>> spelled.startswith("astrolabe.remainder.Except[")
            True
        """
        if isinstance(where, SelNot):
            inner = checked_not_none(self._selectors.encoded(where.inner))
            return checked_not_none(self._minter.marker(markers.Except, [spelled, inner]))
        selector = checked_not_none(self._selectors.encoded(where))
        return checked_not_none(self._minter.marker(markers.Only, [spelled, selector]))

    def _removed_names(self, where: Selector) -> frozenset[str] | None:
        """Give the names that a filter takes out, if that is all the filter does.

        Args:
            where: The filter in normal form.

        Returns:
            The finite set of names that the filter takes out, or None for a
            filter of another form.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> gone = SelNot(NameTest(OneOf(frozenset({"z"}))))
            >>> codec._removed_names(gone)
            frozenset({'z'})
        """
        if not isinstance(where, SelNot):
            return None
        inner = where.inner
        if not isinstance(inner, NameTest):
            return None
        return inner.rule.language().finite_names()

    def _affixed(self, spelled: MypyType, form: Canonical) -> MypyType:
        """Put the suffix, and then the prefix, around the filtered base.

        Args:
            spelled: The type so far.
            form: The normal form that gives the suffix and the prefix.

        Returns:
            The type with `Suffixed[...]` and `Prefixed[...]` around it, for
            each one that the form has.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> base = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> codec._affixed(base, Canonical.of(Variable("kept.R")))
            kept.R
        """
        if form.suffix:
            suffix = self._minter.string(form.suffix)
            spelled = checked_not_none(self._minter.marker(_SuffixedMarker, [spelled, suffix]))
        if form.prefix:
            prefix = self._minter.string(form.prefix)
            spelled = checked_not_none(self._minter.marker(_PrefixedMarker, [prefix, spelled]))
        return spelled

    def _retyped(self, spelled: MypyType, retype: str | None) -> MypyType:
        """Put the retype, the outermost part, around the type so far.

        Args:
            spelled: The type so far.
            retype: The rendering of the new data type of every column, or None.
                A rendering is the text that polars shows for a data type.

        Returns:
            The type with `Retyped[...]` around it, or the type alone if there
            is no retype.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> base = codec.encoded(Canonical.of(Variable("kept.R")))
            >>> print(codec._retyped(base, "UInt32"))
            astrolabe.remainder.Retyped[kept.R, polars.datatypes.classes.UInt32]
            >>> print(codec._retyped(base, "List(UInt32)").args[1])
            astrolabe.frame.List[polars.datatypes.classes.UInt32]
        """
        if retype is None:
            return spelled
        dtype = checked_not_none(ConstDtypeCodec(self._minter).built(retype))
        return checked_not_none(self._minter.marker(markers.Retyped, [spelled, dtype]))

    def decoded(self, typ: MypyType) -> Canonical | None:
        """Read the type of a remainder back into its normal form.

        Args:
            typ: The type to read.

        Returns:
            The normal form, or None for a type that is not a remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> named = Canonical(Variable("kept.R"), suffix="_p", retype="UInt32")
            >>> codec.decoded(codec.encoded(named)) == named
            True
        """
        return RemainderReader(self._gate).decoded(typ)


class RemainderReader:
    """Read the type of a remainder back into its normal form, with no access to the plugin.

    This is the reading half of `RemainderCodec`. Code that reads the remainder
    of a frame, and builds nothing, uses it alone. It needs only the declaration
    gate.

    Example:
        >>> reader = RemainderReader(NamedDeclarations(frozenset()))
        >>> fixture = new_type_fixture()
        >>> reader.decoded(new_marker_minter(fixture).string("x")) is None
        True
    """

    def __init__(self, gate: DeclarationGate) -> None:
        """Keep the gate.

        Args:
            gate: The test that tells a declared remainder from any other plain
                instance.
        """
        self._gate = gate
        self._reads = ProperReads()

    @propagates(None)
    def decoded(self, typ: MypyType) -> Canonical | None:
        """Read one type as a remainder.

        The parts must nest in the order of the normal form. A type with any
        other order reads as no remainder. Where a user wrote such a type, the
        check of the declaration reports it.

        Args:
            typ: The type to read.

        Returns:
            The normal form, or None for a type that is not a remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = new_remainder_codec(fixture, {"kept.R"})
            >>> reader = RemainderReader(NamedDeclarations(frozenset({"kept.R"})))
            >>> named = Canonical(Variable("kept.R"), prefix="s_")
            >>> reader.decoded(codec.encoded(named)) == named
            True
        """
        pieces = _DecodePieces(
            selectors=SelectorReader(),
            names=NameSetReads(),
            gate=self._gate,
            decode=self.decoded,
        )
        peeler = _FormPeeler(pieces, not_none(self._reads.instance(typ)))
        retype = peeler.retype()
        prefix = peeler.prefix()
        suffix = peeler.suffix()
        where = peeler.where()
        base = not_none(peeler.base())
        return Canonical(base, where, suffix, prefix, retype)


class _DecodePieces:
    """The readers that one pass over a remainder type needs.

    This parameter object keeps the constructor of `_FormPeeler` small. Every
    part of a union is read with the same bundle.
    """

    def __init__(
        self,
        selectors: SelectorReader,
        names: NameSetReads,
        gate: DeclarationGate,
        decode: Callable[[MypyType], Canonical | None],
    ) -> None:
        """Keep the readers.

        Args:
            selectors: The reader of selectors.
            names: The reader of sets of names.
            gate: The test for a declared remainder.
            decode: The `decoded` method of the reader, to read each part of a
                union.
        """
        self.selectors = selectors
        self.names = names
        self.gate = gate
        self.decode = decode
        self.reads = ProperReads()


class _FormPeeler:
    """Read the parts of a remainder type from the outside in, in one pass.

    Each method reads its part if the current layer is that part, and then moves
    to the next layer. If the current layer is another part, the method reads
    nothing. A part out of order is therefore left over, and the base then
    refuses it.
    """

    def __init__(self, pieces: _DecodePieces, instance: Instance) -> None:
        """Start the pass at the outermost instance.

        Args:
            pieces: The readers for the pass.
            instance: The outermost instance of the type.
        """
        self._pieces = pieces
        self._at: Instance | None = instance

    def retype(self) -> str | None:
        """Read the `Retyped[...]` layer, if the current layer is one.

        Returns:
            The rendering of the new data type, or None if there is no retype.
        """
        instance = self._stepped(markers.Retyped)
        if instance is None:
            return None
        argument = self._pieces.reads.instance(instance.args[1])
        return not_none(ConstDtypeReads().read(not_none(argument)))

    def prefix(self) -> str:
        """Read the `Prefixed[...]` layer, if the current layer is one.

        Returns:
            The prefix, or an empty string if there is none.
        """
        instance = self._stepped(_PrefixedMarker, inner_at=1)
        if instance is None:
            return ""
        return not_none(self._pieces.reads.string_literal(instance.args[0]))

    def suffix(self) -> str:
        """Read the `Suffixed[...]` layer, if the current layer is one.

        Returns:
            The suffix, or an empty string if there is none.
        """
        instance = self._stepped(_SuffixedMarker)
        if instance is None:
            return ""
        return not_none(self._pieces.reads.string_literal(instance.args[1]))

    def where(self) -> Selector | None:
        """Read the filter layer, if the current layer is one.

        Returns:
            The selector of the filter, or None if there is no filter.
        """
        for reader in (self._minus, self._excepted, self._only):
            selector = reader()
            if selector is not None:
                return selector
        return None

    def _minus(self) -> Selector | None:
        """Read a `Minus[...]` layer, if the current layer is one.

        Returns:
            The negation of a test for the listed names, or None.
        """
        instance = self._stepped(markers.Minus)
        if instance is None:
            return None
        names = not_none(self._pieces.names.read(instance.args[1]))
        return SelNot(NameTest(OneOf(names)))

    def _excepted(self) -> Selector | None:
        """Read an `Except[...]` layer, if the current layer is one.

        Returns:
            The negation of the selector inside, or None.
        """
        instance = self._stepped(markers.Except)
        if instance is None:
            return None
        return SelNot(not_none(self._pieces.selectors.decoded(instance.args[1])))

    def _only(self) -> Selector | None:
        """Read an `Only[...]` layer, if the current layer is one.

        Returns:
            The selector inside, or None.
        """
        instance = self._stepped(markers.Only)
        if instance is None:
            return None
        return not_none(self._pieces.selectors.decoded(instance.args[1]))

    def base(self) -> Base | None:
        """Read the base, the innermost layer.

        A plain instance reads as a declared remainder only if the gate accepts
        its class.

        Returns:
            The declared remainder or the union, or None for any other type.
        """
        instance = self._at
        if instance is None:
            return None
        if self._pieces.reads.matches(instance, markers.Both):
            return Union(tuple(self._parts(instance)))
        if self._pieces.gate.is_declaration(instance):
            return Variable(instance.type.fullname)
        return None

    def _parts(self, instance: Instance) -> Iterator[Canonical]:
        """Read the parts of a `Both[...]` chain, and flatten any chain inside it.

        Args:
            instance: The `Both[...]` instance.

        Yields:
            One normal form for each part.
        """
        for argument in instance.args:
            inner = self._pieces.reads.instance(argument)
            if inner is not None and self._pieces.reads.matches(inner, markers.Both):
                yield from self._parts(inner)
            else:
                yield not_none(self._pieces.decode(argument))

    def _stepped(self, marker: type, inner_at: int = 0) -> Instance | None:
        """Take the current layer if it is of the given marker, and move to the next one.

        Args:
            marker: The marker class of the layer.
            inner_at: The position of the argument that holds the next layer.

        Returns:
            The instance of the layer, or None if the current layer is of
            another marker.
        """
        instance = self._at
        if instance is None or not self._pieces.reads.matches(instance, marker):
            return None
        self._at = self._pieces.reads.instance(instance.args[inner_at])
        return instance


def checked_union(base: Base) -> Union:
    """Give the base as a union, and fail loudly if it is not one.

    The encoder handles a declared remainder and the empty base first, so only a
    union gets here. If that invariant breaks, the function raises an error in
    place of a wrong answer.

    Args:
        base: The base, which is not a declared remainder and not the empty
            base.

    Returns:
        The union.

    Raises:
        TypeError: The base is not a union. That is a bug in this code.

    Example:
        >>> parts = (Canonical.of(Variable("kept.R")),)
        >>> checked_union(Union(parts)).parts is parts
        True
    """
    if not isinstance(base, Union):  # pragma: no cover
        # The encoder handles a declared remainder and the empty base
        # first, so only a union gets here.
        problem = f"not a union base: {base!r}"
        raise TypeError(problem)
    return base
