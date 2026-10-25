"""Answer what the checker knows about the remainder of a frame, from its type alone.

The remainder of an open frame is the set of columns that its type does not
name. Many frame hooks must know facts about it before they answer. For example,
`with_columns` must know whether the remainder can hold a name that it adds, and
`join` must know whether the remainders of its two frames can share a name.

Every question here only reads the type, and none of them builds a new one.
Each declared remainder class sits inside the type, and the requirements of the
class come from its bases. So a hook can ask any of these questions with the
remainder type alone, and it needs no access to the plugin or to mypy's lookup
of names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.declaration_codec import (
    DeclarationCodec,
    MarkedDeclarations,
)
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.remainder_codec import RemainderReader
from astrolabe._hooks.propagate import not_none, propagates
from astrolabe._hooks.remainder.canonical import (
    Canonical,
    Carried,
    Nothing,
    Unioned,
    Variable,
)
from astrolabe.remainder import Prefixed as _PrefixedMarker
from astrolabe.remainder import SplitColumns as _SplitMarker
from astrolabe.remainder import Suffixed as _SuffixedMarker

if TYPE_CHECKING:
    from mypy.nodes import TypeInfo
    from mypy.types import Type as MypyType

    from astrolabe._hooks.remainder.record import Record


class RemainderKnowledge:
    """Answer what one remainder type proves about the columns that it can hold.

    The central answer is the record: the names that the remainder cannot hold,
    and the data types that its columns can have. The class reads the remainder
    type into its normal form, takes the record of each declared remainder from
    the requirements on its class, and lets the engine derive the record of the
    whole form. A closed frame has no remainder type, and its record allows no
    column.

    The class also reads the form of a remainder type: the declared remainder
    classes that it names, and whether it renames them. The check of a call to a
    function with declared frame parameters needs these answers, and so does the
    check of whether two remainders can share a name. The reads also handle the
    `SplitColumns[...]` form, which divides a remainder into parts.

    Example:
        >>> RemainderKnowledge().proves_absent(None, "anything")
        True
    """

    RENAMING: ClassVar[tuple[Target, ...]] = (
        Target.of(_SuffixedMarker),
        Target.of(_PrefixedMarker),
    )
    SPLIT: ClassVar[Target] = Target.of(_SplitMarker)

    def __init__(self) -> None:
        """Set up the readers, with the declaration gate of a real run."""
        self._gate = MarkedDeclarations()
        self._reader = RemainderReader(self._gate)
        self._reads = ProperReads()
        self._declarations = DeclarationCodec()

    def proves_absent(self, rest: MypyType | None, name: str) -> bool:
        """Tell whether the remainder cannot hold a column with the given name.

        Args:
            rest: The remainder type of the frame, or None for a closed frame,
                which holds no column at all.
            name: The column name to test.

        Returns:
            True if the record of the remainder proves that the name is not
            there. A type that does not read as a remainder proves nothing.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> bare = minter.resolved("kept.R")
            >>> RemainderKnowledge().proves_absent(bare, "ticker")
            False
        """
        record = self.record(rest)
        return record is not None and record.absence.holds_of(name)

    @propagates(None)
    def record(self, rest: MypyType | None) -> Record | None:
        """Give the record of one remainder type.

        The parts of a `SplitColumns[...]` type hold the whole remainder between
        them, so its record is the record of the union of its parts.

        Args:
            rest: The remainder type of the frame, or None for a closed frame.

        Returns:
            The record, or the record of an empty remainder for a closed frame.
            None for a type that does not read as a remainder, or for a
            declaration whose requirements are in a wrong form.

        Example:
            >>> RemainderKnowledge().record(None).types.members
            frozenset()
        """
        if rest is None:
            return Carried(Canonical.of(Nothing())).know({})
        form = not_none(self._form(rest))
        records = {
            fullname: not_none(self._declarations.record_of(info))
            for fullname, info in self._declared_infos(rest).items()
        }
        return Carried(form).know(records)

    @propagates(None)
    def _form(self, rest: MypyType) -> Canonical | None:
        """Read one remainder type into its normal form, and read a split as the union of its parts.

        Args:
            rest: The remainder type of the frame.

        Returns:
            The normal form, or None for a type that does not read as a
            remainder, or for a split with a part that does not.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> split = minter.marker(
            ...     remainder_markers.SplitColumns,
            ...     [declared_rest(fixture), declared_rest(fixture, name="decl.S")],
            ... )
            >>> [part.base.name for part in RemainderKnowledge()._form(split).base.parts]
            ['decl.Rest', 'decl.S']
            >>> single = minter.marker(remainder_markers.SplitColumns, [declared_rest(fixture)])
            >>> RemainderKnowledge()._form(single).base.name
            'decl.Rest'
        """
        instance = self._reads.instance(rest)
        if instance is None or not self.SPLIT.matches(instance):
            return self._reader.decoded(rest)
        parts = [not_none(self._reader.decoded(part)) for part in instance.args]
        return Unioned()(parts)

    @propagates(None)
    def part_declarations(self, rest: MypyType) -> tuple[str, ...] | None:
        """Read a remainder type as the declared remainders that its parts name, in order.

        A type that is not a split names one declared remainder, and a
        `SplitColumns[...]` type names one for each part.

        Args:
            rest: The remainder type to read.

        Returns:
            The full names, or None if a part carries a change, for example a
            filter, or does not name exactly one declared remainder.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> knowledge = RemainderKnowledge()
            >>> knowledge.part_declarations(declared_rest(fixture))
            ('decl.Rest',)
            >>> split = new_marker_minter(fixture).marker(
            ...     remainder_markers.SplitColumns,
            ...     [declared_rest(fixture), declared_rest(fixture, name="decl.S")],
            ... )
            >>> knowledge.part_declarations(split)
            ('decl.Rest', 'decl.S')
        """
        instance = self._reads.instance(rest)
        if instance is None or not self.SPLIT.matches(instance):
            name = self.whole_declaration(rest)
            return None if name is None else (name,)
        return tuple(not_none(self.whole_declaration(part)) for part in instance.args)

    def whole_declaration(self, rest: MypyType) -> str | None:
        """Read a remainder type as one declared remainder with no change on it.

        Args:
            rest: The remainder type to read.

        Returns:
            The full name of the declared remainder, or None if the type carries
            a change, for example a filter or a suffix, or does not name exactly
            one declared remainder.

        Example:
            >>> fixture = new_type_fixture()
            >>> knowledge = RemainderKnowledge()
            >>> knowledge.whole_declaration(declared_rest(fixture))
            'decl.Rest'
            >>> knowledge.whole_declaration(new_marker_minter(fixture).string("x"))
            >>> import astrolabe.remainder as remainder_markers
            >>> pair = new_marker_minter(fixture).marker(
            ...     remainder_markers.Both,
            ...     [declared_rest(fixture), declared_rest(fixture, name="decl.S")],
            ... )
            >>> knowledge.whole_declaration(pair)
        """
        form = self._reader.decoded(rest)
        if form is None or form != Canonical.of(form.base):
            return None
        if not isinstance(form.base, Variable):
            return None
        return form.base.name

    def renames(self, rest: MypyType) -> bool:
        """Tell whether a remainder type renames the columns of its declared remainders.

        A `Suffixed[...]` or `Prefixed[...]` marker anywhere in the type gives
        its columns names that the declared remainders do not hold. A promise
        about the names of a declared remainder then no longer applies.

        Args:
            rest: The remainder type of the frame.

        Returns:
            True if a marker in the type is `Suffixed[...]` or `Prefixed[...]`.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> knowledge = RemainderKnowledge()
            >>> knowledge.renames(declared_rest(fixture))
            False
            >>> suffixed = minter.marker(
            ...     remainder_markers.Suffixed,
            ...     [declared_rest(fixture), minter.string("_x")],
            ... )
            >>> knowledge.renames(suffixed)
            True
        """
        instance = self._reads.instance(rest)
        if instance is None:
            return False
        if any(target.matches(instance) for target in self.RENAMING):
            return True
        return any(self.renames(argument) for argument in instance.args)

    def declarations(self, rest: MypyType | None) -> dict[str, TypeInfo]:
        """Give the declared remainder classes that one remainder type names.

        Args:
            rest: The remainder type of the frame, or None for a closed frame,
                which names none.

        Returns:
            The class of each declared remainder, by full name.

        Example:
            >>> RemainderKnowledge().declarations(None)
            {}
        """
        if rest is None:
            return {}
        return self._declared_infos(rest)

    def _declared_infos(self, rest: MypyType) -> dict[str, TypeInfo]:
        """Collect the declared remainder classes that one remainder type names.

        The normal form names each declared remainder only by its full name. The
        remainder type still holds the instances of the classes, so this method
        walks the type and takes each class from there. It does not look the
        class up again by name.

        Args:
            rest: The remainder type of the frame.

        Returns:
            The class of each declared remainder, by full name.

        Example:
            >>> fixture = new_type_fixture()
            >>> minter = new_marker_minter(fixture)
            >>> RemainderKnowledge()._declared_infos(minter.string("x"))
            {}
        """
        found: dict[str, TypeInfo] = {}
        self._collect(rest, found)
        return found

    def _collect(self, typ: MypyType, found: dict[str, TypeInfo]) -> None:
        """Add the declared remainder classes in one type and in its type arguments.

        The gate tells a declared remainder from any other class.

        Args:
            typ: The type to walk.
            found: The classes found so far. This method adds to it in place.
        """
        instance = self._reads.instance(typ)
        if instance is None:
            return
        if self._gate.is_declaration(instance):
            found[instance.type.fullname] = instance.type
        for argument in instance.args:
            self._collect(argument, found)


class SlotRelations:
    """Answer questions about the remainders of two frames together.

    An operation that combines two frames, for example `join` or `concat`, must
    know how their remainders relate. Can a column of one share its name with a
    column of the other? Which of the names that the operation needs gone can a
    remainder still hold? The answers read the records and the declared
    remainders of both types, in the same way as `RemainderKnowledge`. They need
    no state from the plugin.

    Example:
        >>> SlotRelations().unprovable(None, frozenset({"a"}))
        frozenset()
    """

    def __init__(self) -> None:
        """Set up the readers, with the declaration gate of a real run."""
        self._knowledge = RemainderKnowledge()
        self._declarations = DeclarationCodec()

    def unprovable(self, rest: MypyType | None, names: frozenset[str]) -> frozenset[str]:
        """Keep the names that the record of one remainder cannot prove absent.

        Args:
            rest: The remainder type to ask, or None for a closed frame, which
                proves every name absent.
            names: The names that the operation needs to be absent from that
                remainder.

        Returns:
            The names that are not proven absent.

        Example:
            >>> fixture = new_type_fixture()
            >>> relations = SlotRelations()
            >>> slot = declared_rest(fixture, without=("a",))
            >>> sorted(relations.unprovable(slot, frozenset({"a", "b"})))
            ['b']
        """
        knowledge = self._knowledge
        return frozenset(name for name in names if not knowledge.proves_absent(rest, name))

    def can_share(self, left: MypyType | None, right: MypyType | None) -> bool:
        """Tell whether a column in one remainder can have the same name as a column in the other.

        Args:
            left: The remainder type of the receiver, or None.
            right: The remainder type of the other frame, or None.

        Returns:
            False if one side is closed, if no name is possible in both records,
            or if the declarations promise that the two share no name. True
            otherwise.
        """
        if left is None or right is None:
            return False
        if self._languages_disjoint(left, right):
            return False
        return not self._declared_disjoint(left, right)

    def _languages_disjoint(self, left: MypyType, right: MypyType) -> bool:
        """Tell whether the records of the two remainders have no possible name in common.

        A record that allows no column shares no name with any other record.

        Args:
            left: The remainder type of the receiver.
            right: The remainder type of the other frame.

        Returns:
            True if the checker proves that no name is possible in both. A type
            that does not read as a remainder proves nothing.

        Example:
            >>> fixture = new_type_fixture()
            >>> relations = SlotRelations()
            >>> bare = declared_rest(fixture)
            >>> relations._languages_disjoint(bare, declared_rest(fixture))
            False
            >>> opaque = new_marker_minter(fixture).string("x")
            >>> relations._languages_disjoint(bare, opaque)
            False
        """
        left_record = self._knowledge.record(left)
        right_record = self._knowledge.record(right)
        if left_record is None or right_record is None:
            return False
        if left_record.permits_nothing() or right_record.permits_nothing():
            return True
        left_present = left_record.absence.language().complement()
        right_present = right_record.absence.language().complement()
        return left_present.intersection(right_present).empty()

    def _declared_disjoint(self, left: MypyType, right: MypyType) -> bool:
        """Tell whether the declarations promise that the two remainders share no name.

        Each pair of a declared remainder on the left and one on the right must
        have a `disjoint_from=` requirement on one of the two that names the
        other. The promise is
        part of what a declaration states, so the checker can rely on it inside
        a function body that has the declaration.

        Args:
            left: The remainder type of the receiver.
            right: The remainder type of the other frame.

        Returns:
            True if every pair has the promise.

        Example:
            >>> fixture = new_type_fixture()
            >>> relations = SlotRelations()
            >>> opaque = new_marker_minter(fixture).string("x")
            >>> relations._declared_disjoint(declared_rest(fixture), opaque)
            False
        """
        lefts = self._promised_declarations(left)
        rights = self._promised_declarations(right)
        if not lefts or not rights:
            return False
        pairs = (
            self._pair_promised(left_name, left_info, right_name, right_info)
            for left_name, left_info in lefts.items()
            for right_name, right_info in rights.items()
        )
        return all(pairs)

    def _pair_promised(
        self,
        left_name: str,
        left_info: TypeInfo,
        right_name: str,
        right_info: TypeInfo,
    ) -> bool:
        """Tell whether one pair of declared remainders has the promise that they share no name.

        Args:
            left_name: The full name of the left declared remainder.
            left_info: The class of the left declared remainder.
            right_name: The full name of the right declared remainder.
            right_info: The class of the right declared remainder.

        Returns:
            True if the `disjoint_from=` of either one names the other.
        """
        if right_name in self._disjoint_names(left_info):
            return True
        return left_name in self._disjoint_names(right_info)

    def _promised_declarations(self, rest: MypyType) -> dict[str, TypeInfo]:
        """Give the declared remainders whose promises apply to one remainder type.

        A `disjoint_from=` promise is about the names of the declared
        remainders. A type that adds a suffix or a prefix to them holds other
        names, so no promise applies to it.

        Args:
            rest: The remainder type of a frame.

        Returns:
            The declared remainders of the type, by full name, or none for a
            type that renames.

        Example:
            >>> fixture = new_type_fixture()
            >>> sorted(SlotRelations()._promised_declarations(declared_rest(fixture)))
            ['decl.Rest']
        """
        if self._knowledge.renames(rest):
            return {}
        return self._knowledge.declarations(rest)

    def _disjoint_names(self, info: TypeInfo) -> frozenset[str]:
        """Join the `disjoint_from=` sets of one declared remainder into one set.

        A remainder that shares no name with a union shares no name with each
        member of it, so the sets join into one.

        Args:
            info: The class of the declared remainder.

        Returns:
            Every full name that the declaration promises to share no name with.
            An empty set for a declaration in a wrong form, which promises
            nothing.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> relations = SlotRelations()
            >>> minter = new_marker_minter(fixture)
            >>> bad = minter.marker(
            ...     remainder_markers.DisjointFrom, [minter.string("x")]
            ... )
            >>> relations._disjoint_names(declaration_info(fixture, bad))
            frozenset()
        """
        relations = self._declarations.relations_of(info)
        if relations is None:
            return frozenset()
        flattened: set[str] = set()
        for group in relations.disjoint_from:
            flattened |= group
        return frozenset(flattened)
