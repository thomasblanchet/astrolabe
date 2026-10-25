"""Apply one change to the remainder of a frame type, and give the new remainder type.

The remainder of an open frame is the set of columns that its type does not
name. A frame type holds its remainder as a mypy type, and a frame hook that
changes the remainder, for example `drop` or `rename`, needs the changed type.
That takes four steps: read the type back into the normal form of the engine,
apply the change, bring the result back to normal form, and write it as a type
again. This module does those four steps, so that no hook does them by hand.

A closed frame has no remainder, and every change here keeps it closed. A hook
can therefore call these methods on any frame, with no test first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
)
from astrolabe._hooks.encoding.declaration_codec import MarkedDeclarations
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.remainder_codec import RemainderCodec
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.remainder.absence import OneOf
from astrolabe._hooks.remainder.canonical import (
    Canonical,
    Carried,
    Combined,
    Filter,
    Grafted,
    Nothing,
    Prefix,
    Retype,
    Suffix,
)
from astrolabe._hooks.remainder.record import ColumnSet
from astrolabe._hooks.remainder.selector import NameTest, OfType, SelAnd, Selector, SelNot
from astrolabe._hooks.remainder.type_set import Family

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.expression.selection import RemainderEffect
    from astrolabe._hooks.remainder.canonical import Expr


class RemainderEdit:
    """Apply one change to the remainder type of a frame, and give the changed type.

    A frame hook gives the remainder type of its receiver, and gets back the
    remainder type of its result. In between, the class reads the type through
    `RemainderCodec`, lets the engine apply the change and bring the result to
    normal form, and writes it back. The changes are named after what they do:
    take names out, keep or drop the columns that a selector picks, add a prefix
    or a suffix, retype, or unite two remainders. A remainder type of None
    stands for a closed frame, in the argument and in the result.

    The class also serves the check of a call to a function with a declared
    frame parameter. There it replaces each declared remainder in the return
    type with the remainder of the caller, or with the concrete columns of the
    caller.

    Example:
        >>> fixture = new_type_fixture()
        >>> edit = new_remainder_edit(fixture, {"kept.R"})
        >>> edit.minus(None, ["z"]) is None
        True
    """

    def __init__(self, codec: RemainderCodec) -> None:
        """Keep the codec for this call.

        Args:
            codec: The remainder codec for the current phase.
        """
        self._codec = codec

    @classmethod
    def for_checker(cls, plugin: PluginLookup, api: object) -> RemainderEdit:
        """Build the editor for a hook that runs in the type checking phase.

        Args:
            plugin: The mypy plugin.
            api: The plugin API of the type checking phase, which builds literal
                types.

        Returns:
            The editor, with a gate that reads the declared remainder classes
            from their marker bases.
        """
        minter = MarkerMinter(plugin, CheckerLiteralTypeBuilder(), api)
        return cls(RemainderCodec(minter, MarkedDeclarations()))

    def narrowed(self, rest: MypyType, where: Selector) -> MypyType:
        """Narrow one remainder type to the columns that a selector picks.

        A split declaration divides the columns of a frame into parts. For an
        open argument, each part takes the remainder of the caller narrowed in
        this way, so the part knows what its own filter keeps.

        Args:
            rest: The remainder type to narrow.
            where: The selector of the part.

        Returns:
            The narrowed remainder type.
        """
        return self._codec.filtered(rest, where)

    def minus(self, rest: MypyType | None, names: Sequence[str]) -> MypyType | None:
        """Take named columns out of a remainder.

        A name that the remainder already cannot hold is left out of the change.
        The remainder stays the same for every caller either way, and the old
        type is shorter.

        Args:
            rest: The remainder type of the frame, or None for a closed frame.
            names: The names that the operation takes out or claims.

        Returns:
            The changed remainder type. It is the same type if the remainder
            already cannot hold any of the names. None if the frame was closed
            or the checker proves the result empty.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.minus(minter.resolved("kept.R"), ["z"])
            >>> "Minus[" in str(out) and "'z'" in str(out)
            True
            >>> edit.minus(None, ["z"]) is None
            True
        """
        knowledge = RemainderKnowledge()
        removed = frozenset(name for name in names if not knowledge.proves_absent(rest, name))
        if not removed:
            return rest
        gone = SelNot(NameTest(OneOf(removed)))
        return self._applied(rest, lambda inner: Filter(inner, gone))

    @propagates(None)
    def both(self, left: MypyType | None, right: MypyType | None) -> MypyType | None:
        """Unite the remainders of two frames, for example in `concat` or `join`.

        Args:
            left: The remainder type of one frame, or None for a closed frame.
            right: The remainder type of the other frame, or None.

        Returns:
            The remainder type of the union. If only one frame is open, its own
            remainder type. None if both frames are closed or a type does not
            read as a remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R", "kept.S"})
            >>> minter = new_marker_minter(fixture)
            >>> left = minter.resolved("kept.R")
            >>> edit.both(left, None) is left
            True
            >>> out = edit.both(left, minter.resolved("kept.S"))
            >>> "Both[" in str(out)
            True
        """
        if left is None:
            return right
        if right is None:
            return left
        left_form = not_none(self._codec.decoded(left))
        right_form = not_none(self._codec.decoded(right))
        joined = Combined(Carried(left_form), Carried(right_form))
        return self._codec.encoded(joined.normalise())

    def only(self, rest: MypyType | None, selector: Selector) -> MypyType | None:
        """Keep the columns of the remainder that a selector picks.

        Args:
            rest: The remainder type of the frame, or None.
            selector: The selector.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.only(minter.resolved("kept.R"), OfType(Family.numeric()))
            >>> "Only[" in str(out)
            True
        """
        return self._applied(rest, lambda inner: Filter(inner, selector))

    def excepted(self, rest: MypyType | None, selector: Selector) -> MypyType | None:
        """Drop the columns of the remainder that a selector picks.

        Args:
            rest: The remainder type of the frame, or None.
            selector: The selector.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.excepted(minter.resolved("kept.R"), OfType(Family.numeric()))
            >>> "Except[" in str(out)
            True
        """
        return self._applied(rest, lambda inner: Filter(inner, SelNot(selector)))

    def selected(self, rest: MypyType | None, effect: RemainderEffect) -> MypyType | None:
        """Apply the whole effect of a selection: keep, then retype, then rename the picked columns.

        This is the remainder of a result that holds only what the selection
        gives, for example the result of `select`.

        Args:
            rest: The remainder type of the frame, or None.
            effect: The effect of the selection on the remainder.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> effect = RemainderEffect(OfType(Family.numeric()), "String", Affix(suffix="_s"))
            >>> out = str(edit.selected(minter.resolved("kept.R"), effect))
            >>> "Suffixed[" in out and "Retyped[" in out and "Only[" in out
            True
            >>> plain = RemainderEffect(OfType(Family.numeric()), None, Affix())
            >>> str(edit.selected(minter.resolved("kept.R"), plain)).startswith(
            ...     "astrolabe.remainder.Only["
            ... )
            True
        """
        kept = self._retyped(self.only(rest, effect.selector), effect)
        if effect.affix.prefix:
            kept = self.prefixed(kept, effect.affix.prefix)
        if effect.affix.suffix:
            kept = self.suffixed(kept, effect.affix.suffix)
        return kept

    def with_effect(
        self,
        original: MypyType | None,
        rest: MypyType | None,
        effect: RemainderEffect,
    ) -> MypyType | None:
        """Apply the effect of a selection beside the columns that the operation keeps.

        An operation such as `with_columns` keeps every column of its receiver
        and adds or replaces the columns that the selection gives. If the
        selection keeps the names of the picked columns, it replaces them, so
        the picked part of the remainder is retyped in place. If it renames
        them, the renamed columns are new, and they join the remainder beside
        the old ones.

        Args:
            original: The remainder type of the receiver.
            rest: The remainder type so far, after the other changes of this
                call.
            effect: The effect of the selection on the remainder.

        Returns:
            The remainder type so far if the selection gives each picked column
            back with its own name and data type. Otherwise the type with the
            picked part retyped in place, or with the renamed part added.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> rest = new_marker_minter(fixture).resolved("kept.R")
            >>> retype = RemainderEffect(OfType(Family.numeric()), "String", Affix())
            >>> out = str(edit.with_effect(rest, rest, retype))
            >>> "Both[" in out and "Retyped[" in out and "Except[" in out
            True
            >>> same = RemainderEffect(OfType(Family.numeric()), None, Affix())
            >>> edit.with_effect(rest, rest, same) is rest
            True
        """
        if not effect.changes() and effect.affix.is_identity():
            return rest
        if not effect.affix.is_identity():
            return self.both(rest, self.selected(original, effect))
        kept = self.only(original, effect.selector)
        if effect.retype is not None:
            return self.both(self.excepted(rest, effect.selector), self._retyped(kept, effect))
        changed = SelAnd(effect.selector, self._changed(effect))
        return self.both(self.excepted(rest, changed), self._kinds(kept, effect))

    def _retyped(self, kept: MypyType | None, effect: RemainderEffect) -> MypyType | None:
        """Give the picked columns of the remainder with the data types that an effect gives them.

        Where the effect keeps the data type of the kinds that it does not list,
        those columns stay as they are.

        Args:
            kept: The remainder type of the picked columns.
            effect: The effect.

        Returns:
            The retyped columns, or None for a closed frame.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> rest = new_marker_minter(fixture).resolved("kept.R")
            >>> effect = RemainderEffect(OfType(Family.numeric()), "String", Affix())
            >>> str(edit._retyped(rest, effect)).count("Retyped[")
            1
        """
        if not effect.retypes:
            return kept if effect.retype is None else self.retyped(kept, effect.retype)
        others = self.excepted(kept, self._changed(effect))
        if effect.retype is not None:
            others = self.retyped(others, effect.retype)
        return self.both(others, self._kinds(kept, effect))

    def _changed(self, effect: RemainderEffect) -> Selector:
        """Give the rule that picks the columns of the kinds that an effect retypes, kind by kind.

        Args:
            effect: The effect, with a data type for some kinds.

        Returns:
            The rule.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> retypes = (("Int16", frozenset({"Int8"})), ("Int32", frozenset({"Null"})))
            >>> effect = RemainderEffect(OfType(Family.numeric()), None, Affix(), retypes)
            >>> new_remainder_edit(new_type_fixture(), set())._changed(effect)
            OfType(family=Int8|Null)
        """
        kinds = frozenset().union(*(kinds for _, kinds in effect.retypes))
        return OfType(Family.of_members(kinds))

    def _kinds(self, kept: MypyType | None, effect: RemainderEffect) -> MypyType | None:
        """Give the columns of the kinds that an effect retypes, each kind with its data type.

        Args:
            kept: The remainder type of the picked columns.
            effect: The effect, with a data type for some kinds.

        Returns:
            The retyped columns, or None for a closed frame.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> rest = new_marker_minter(fixture).resolved("kept.R")
            >>> retypes = (("Int16", frozenset({"Int8"})), ("Int32", frozenset({"Null"})))
            >>> effect = RemainderEffect(OfType(Family.numeric()), None, Affix(), retypes)
            >>> str(edit._kinds(rest, effect)).count("Retyped[")
            2
        """
        retyped: MypyType | None = None
        for answer, kinds in effect.retypes:
            part = self.only(kept, OfType(Family.of_members(kinds)))
            retyped = self.both(retyped, self.retyped(part, answer))
        return retyped

    def suffixed(self, rest: MypyType | None, suffix: str) -> MypyType | None:
        """Add a suffix to the name of every column in the remainder.

        Args:
            rest: The remainder type of the frame, or None.
            suffix: The suffix to add.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.suffixed(minter.resolved("kept.R"), "_right")
            >>> "Suffixed[" in str(out)
            True
            >>> edit.suffixed(None, "_right") is None
            True
        """
        return self._applied(rest, lambda inner: Suffix(inner, suffix))

    def prefixed(self, rest: MypyType | None, prefix: str) -> MypyType | None:
        """Add a prefix to the name of every column in the remainder.

        Args:
            rest: The remainder type of the frame, or None.
            prefix: The prefix to add.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.prefixed(minter.resolved("kept.R"), "s_")
            >>> "Prefixed[" in str(out)
            True
        """
        return self._applied(rest, lambda inner: Prefix(inner, prefix))

    def retyped(self, rest: MypyType | None, dtype: str) -> MypyType | None:
        """Give every column in the remainder a new data type, and keep every name.

        Args:
            rest: The remainder type of the frame, or None.
            dtype: The rendering of the new data type, or `Unchecked` if no one
                data type fits. A rendering is the text that polars shows for a
                data type.

        Returns:
            The changed remainder type, or None for a closed frame.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> out = edit.retyped(minter.resolved("kept.R"), "UInt32")
            >>> "Retyped[" in str(out)
            True
        """
        return self._applied(rest, lambda inner: Retype(inner, dtype))

    @propagates(None)
    def _applied(
        self,
        rest: MypyType | None,
        effect: Callable[[Carried], Expr],
    ) -> MypyType | None:
        """Read the type, apply one change, normalise the result, and write it as a type.

        Args:
            rest: The remainder type of the frame, or None.
            effect: A function that puts the change around the remainder that
                was read.

        Returns:
            The changed remainder type. None for a closed frame, for a result
            that the checker proves empty, or for a type that does not read as a
            remainder.
        """
        if rest is None:
            return None
        form = not_none(self._codec.decoded(rest))
        wrapped = effect(Carried(form))
        return self._codec.encoded(wrapped.normalise())

    @propagates(None)
    def substituted(
        self,
        rest: MypyType | None,
        bindings: Mapping[str, MypyType | None],
    ) -> MypyType | None:
        """Put the remainder of the caller in place of each declared remainder in a type.

        A function can declare a frame parameter with a remainder, and use that
        remainder in its return type. At each call, the checker binds the
        declared remainder to the remainder of the argument, and this method
        puts the bound remainder into the return type.

        Args:
            rest: The remainder type of the declared result, or None for a
                closed result, which names no declared remainder.
            bindings: The remainder type that each declared remainder is bound
                to, by full name. A binding to None means the empty remainder.

        Returns:
            The remainder type with each declared remainder replaced. None for a
            closed declared result, a declared remainder with no binding, a
            result that the checker proves empty, or a type that does not read
            as a remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"decl.Rest", "kept.R"})
            >>> minter = new_marker_minter(fixture)
            >>> declared = minter.resolved("decl.Rest")
            >>> out = edit.substituted(declared, {"decl.Rest": minter.resolved("kept.R")})
            >>> str(out)
            'kept.R'
            >>> edit.substituted(declared, {"decl.Rest": None}) is None
            True
            >>> edit.substituted(None, {}) is None
            True
        """
        if rest is None:
            return None
        form = not_none(self._codec.decoded(rest))
        parts = {name: not_none(self.decoded(slot)) for name, slot in bindings.items()}
        grafted = not_none(Grafted()(form, parts))
        return self._codec.encoded(grafted)

    @propagates(None)
    def denoted(
        self,
        rest: MypyType | None,
        bindings: Mapping[str, ColumnSet],
    ) -> ColumnSet | None:
        """Give the concrete columns that a remainder type stands for under concrete bindings.

        If every argument of a call is a closed frame, each declared remainder
        is bound to a known set of columns. The return type then has known
        columns too.

        Args:
            rest: The remainder type of the declared result, or None for a
                closed result.
            bindings: The concrete columns that each declared remainder is bound
                to, by full name.

        Returns:
            The columns, or an empty set for a closed result. None for a type
            that does not read as a remainder, or for a declared remainder with
            no binding.

        Example:
            >>> from astrolabe._hooks.remainder.record import ColumnSet
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"decl.Rest"})
            >>> minter = new_marker_minter(fixture)
            >>> declared = minter.resolved("decl.Rest")
            >>> bound = {"decl.Rest": ColumnSet.of(venue="String")}
            >>> edit.denoted(declared, bound).entries
            (('venue', 'String'),)
            >>> edit.denoted(None, {}).entries
            ()
            >>> edit.denoted(declared, {}) is None
            True
        """
        if rest is None:
            return ColumnSet(())
        form = not_none(self._codec.decoded(rest))
        try:
            return form.denote(bindings)
        except KeyError:
            return None

    def decoded(self, rest: MypyType | None) -> Canonical | None:
        """Read a remainder type back into its normal form.

        Args:
            rest: The remainder type of the frame, or None.

        Returns:
            The normal form, the empty form for a closed frame, or None for a
            type that does not read as a remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> edit.decoded(None).base
            Nothing()
            >>> minter = new_marker_minter(fixture)
            >>> edit.decoded(minter.resolved("kept.R")).base
            Variable(name='kept.R')
        """
        if rest is None:
            return Canonical.of(Nothing())
        return self._codec.decoded(rest)

    def encoded(self, form: Canonical) -> MypyType | None:
        """Write a normal form as a remainder type.

        Args:
            form: The normal form.

        Returns:
            The remainder type, or None for a remainder that the checker proves
            empty.

        Example:
            >>> fixture = new_type_fixture()
            >>> edit = new_remainder_edit(fixture, {"kept.R"})
            >>> edit.encoded(Canonical.of(Nothing())) is None
            True
        """
        return self._codec.encoded(form)
