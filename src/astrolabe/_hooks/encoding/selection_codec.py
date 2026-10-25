"""Write the type of a declared selection as a mypy type, and read it back.

A user declares a selection that a function takes or returns, for example with
`typed_selection(cs.numeric(), gives=same)`. It gives one output column for each
column that its selector picks. Its type holds the selector, a map from the
data type of each picked column to the data type of its output, the character,
and the rename. The character tells whether the selection works row by row or
gives one value for each group. The declared type lives only in its mypy type,
so all of these parts must travel inside it.

A selector with no operation on it is `TypedSelector[selector]`. Any other
selection is `TypedSelection[selector, Rows[...], rename]`, or
`TypedAggSelection[...]` if it gives one value for each group. The map has one
read with the blank name, which stands for the picked column. The rename is
`None`, or `Affix[prefix, suffix, fixed]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import Instance, NoneType, get_proper_type

import astrolabe.expr as markers
from astrolabe._hooks.encoding.expr_codec import (
    OutcomeCodec,
    OutcomeReads,
)
from astrolabe._hooks.deferred.code import Code
from astrolabe._hooks.deferred.remainder import DeferredRemainder
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.selector_codec import SelectorCodec, SelectorReader
from astrolabe._hooks.expression.character import AGGREGATING, ELEMENTWISE
from astrolabe._hooks.expression.selection import Affix, SelectionType
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe._hooks.recording.deferred_types import DeferredReader
from astrolabe.expr import TypedAggSelection, TypedSelection, TypedSelector

if TYPE_CHECKING:
    from mypy.types import Type as MypyType

    from astrolabe._hooks.encoding.marker_minter import MarkerMinter


class RenameCodec:
    """Write the rename of a selection as a mypy type, and read it back.

    A selection can rename each column that it gives: it can add a prefix and a
    suffix, or give every output one fixed name. The rename has no effect if it
    adds nothing, and that case is written as `None`. An open `when`/`then`
    chain over a selection holds the same rename, so both codecs write it
    through this one class.

    Example:
        >>> fixture = new_type_fixture()
        >>> RenameCodec(new_marker_minter(fixture)).encoded(Affix())
        None
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter that builds the types.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter

    def encoded(self, affix: Affix) -> MypyType:
        """Build the type of one rename.

        The last part is the fixed name: the type of `None` if each output
        keeps the name of its picked column, and otherwise the literal of the
        one name that every output takes.

        Args:
            affix: The rename.

        Returns:
            The type of `None` for a rename that changes nothing, and the
            `Affix[...]` type otherwise.

        Example:
            >>> fixture = new_type_fixture()
            >>> codec = RenameCodec(new_marker_minter(fixture))
            >>> codec.encoded(Affix(suffix="_r"))
            astrolabe.expr.Affix[Literal[''], Literal['_r'], None]
            >>> codec.encoded(Affix(fixed="x"))
            astrolabe.expr.Affix[Literal[''], Literal[''], Literal['x']]
        """
        if affix.is_identity():
            return NoneType()
        fixed: MypyType = NoneType()
        if affix.fixed is not None:
            fixed = self._minter.string(affix.fixed)
        parts = [
            self._minter.string(affix.prefix),
            self._minter.string(affix.suffix),
            fixed,
        ]
        return not_none(self._minter.marker(markers.Affix, parts))


class RenameReader:
    """Read the type of a rename back into the engine value, with no access to the plugin.

    This is the reading half of `RenameCodec`. The selection reader and the
    chain reader both use it.

    Example:
        >>> from mypy.types import NoneType
        >>> RenameReader().decoded(NoneType())
        Affix(prefix='', suffix='', fixed=None)
    """

    def __init__(self) -> None:
        """Set up the reader of instances."""
        self._reads = ProperReads()

    @propagates(None)
    def decoded(self, typ: MypyType) -> Affix | None:
        """Read one type as a rename.

        Args:
            typ: The type of the rename part.

        Returns:
            The rename that changes nothing for `None`, the rename in an
            `Affix[...]` type, or None for a type in another form.

        Example:
            >>> from mypy.types import UninhabitedType
            >>> RenameReader().decoded(UninhabitedType()) is None
            True
        """
        if isinstance(get_proper_type(typ), NoneType):
            return Affix()
        instance = not_none(self._reads.instance(typ))
        not_falsy(self._reads.matches(instance, markers.Affix) and len(instance.args) == 3)
        prefix = not_none(self._reads.string_literal(instance.args[0]))
        suffix = not_none(self._reads.string_literal(instance.args[1]))
        return Affix(prefix, suffix, self._reads.string_literal(instance.args[2]))


class SelectionCodec:
    """Build the mypy type for one selection, and read it back.

    Every hook that types a selection, for example a `cs.*` function or an
    operation on a selection, returns its answer through this class. It writes
    the selector through `SelectorCodec` and the table through `OutcomeCodec`,
    so one selection always gives the same type. The reading half is
    `SelectionReader`.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> fixture = new_type_fixture()
        >>> codec = new_selection_codec(fixture)
        >>> numeric = SelectionType.identity(OfType(Family.numeric()))
        >>> codec.encoded(numeric)
        astrolabe.expr.TypedSelector[astrolabe.remainder.OfType[polars.datatypes.classes.NumericType]]
        >>> SelectionReader().decoded(codec.encoded(numeric)) == numeric
        True
    """

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the minter and the codecs for the parts.

        Args:
            minter: The marker minter for the current phase.
        """
        self._minter = minter
        self._selectors = SelectorCodec(minter)
        self._outcomes = OutcomeCodec(minter)

    @propagates(None)
    def encoded(self, selection: SelectionType) -> MypyType | None:
        """Build the type of one selection.

        A selector with no operation on it takes the short form
        `TypedSelector[...]`. Any other selection takes the head class
        `TypedAggSelection` if it gives one value for each group, and
        `TypedSelection` otherwise. The parts are the selector, the table and
        the rename.

        Args:
            selection: The selection from the engine.

        Returns:
            The type, or None if a part has no type.
        """
        selector = not_none(self._selectors.encoded(selection.selector))
        if selection.is_identity():
            return self._minter.marker(TypedSelector, [selector])
        gives = not_none(self._outcomes.rows(selection.table))
        head = TypedAggSelection if selection.character is AGGREGATING else TypedSelection
        rename = RenameCodec(self._minter).encoded(selection.affix)
        return self._minter.marker(head, [selector, gives, rename])


class SelectionReader:
    """Read the type of a selection back into the engine value, with no access to the plugin.

    This is the reading half of `SelectionCodec`. Code that only needs to know
    whether an argument is a selection, and what it is, uses it alone.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> SelectionReader().decoded(UninhabitedType()) is None
        True
    """

    def __init__(self) -> None:
        """Set up the readers for the parts."""
        self._reads = ProperReads()
        self._selectors = SelectorReader()
        self._outcomes = OutcomeReads()

    def decoded(self, typ: MypyType) -> SelectionType | None:
        """Read one type as a selection.

        Args:
            typ: The type to read.

        Returns:
            The selection, or None for a type that is not a selection.
        """
        instance = self._reads.instance(typ)
        if instance is None:
            return None
        if self._reads.matches(instance, TypedSelector):
            return self._deferred(instance) or self._pure(instance)
        return self._applied(instance)

    def _deferred(self, instance: Instance) -> SelectionType | None:
        """Read a selector of the deferred model, whose type holds its code, as a selection.

        Args:
            instance: The type of the selector.

        Returns:
            The selection that gives each picked column back, or `None` where
            the type holds no code of a selector alone.
        """
        code = DeferredReader().code(instance)
        if code is None:
            return None
        rule = DeferredRemainder("row").rule(Code(code.text, tuple(None for _ in code.leaves)))
        return None if rule is None else SelectionType.identity(rule)

    @propagates(None)
    def _pure(self, instance: Instance) -> SelectionType | None:
        """Expand the short form `TypedSelector[...]` into the selection with no operation.

        Args:
            instance: The `TypedSelector[...]` instance.

        Returns:
            The selection that gives each picked column back unchanged, or None
            if the selector is in a wrong form.
        """
        not_falsy(len(instance.args) == 1)
        selector = not_none(self._selectors.decoded(instance.args[0]))
        return SelectionType.identity(selector)

    @propagates(None)
    def _applied(self, instance: Instance) -> SelectionType | None:
        """Read the full form, with a head class, as a selection.

        The head class `TypedSelection` gives a selection that works row by
        row, and `TypedAggSelection` gives one value for each group. Any other
        class is not a selection.

        Args:
            instance: The instance of the head class.

        Returns:
            The selection, or None if a part is in a wrong form.
        """
        if self._reads.matches(instance, TypedSelection):
            character = ELEMENTWISE
        else:
            not_falsy(self._reads.matches(instance, TypedAggSelection))
            character = AGGREGATING
        not_falsy(len(instance.args) == 3)
        selector = not_none(self._selectors.decoded(instance.args[0]))
        rows = not_none(self._reads.instance(instance.args[1]))
        not_falsy(self._reads.matches(rows, markers.Rows))
        table = not_none(self._outcomes.rows_table((SelectionType.AXIS,), rows))
        affix = not_none(RenameReader().decoded(instance.args[2]))
        return SelectionType(selector, table, affix, character)
