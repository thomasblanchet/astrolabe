"""Decide the remainder of a join result, or name the columns that depend on the caller.

The remainder of an open frame is the set of columns that its type does not
name. A polars join keeps the columns of both frames. If a right column has the
name of a left column, polars adds a suffix to the right one, `_right` by
default. The checker can decide those renames for the declared columns. It
cannot see the columns in a remainder, so a join over a remainder is allowed
only if it renames the same columns for every caller.

This module decides the part that the remainders affect. A column hidden in the
left remainder can force a rename on a declared right column. A declared right
column, once renamed, can collide with a hidden left column. A hidden right
column can collide with a declared left column. And the two remainders can share
a name. The answer to each question comes from the two remainder types alone:
what the checker knows about each remainder, and the classes that declare them.
So a join hook needs nothing more to decide them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TYPE_CHECKING

from astrolabe._hooks.encoding.declaration_codec import DeclarationCodec
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import (
    RemainderKnowledge,
    SlotRelations,
)
from astrolabe._hooks.errors import (
    JoinRemainderOverlapError,
    JoinRenameAmbiguityError,
    JoinSuffixTakenError,
)
from astrolabe._hooks.remainder.absence import EndsWith, OneOf
from astrolabe._hooks.remainder.entailment import AbsenceEntailment

if TYPE_CHECKING:
    from mypy.nodes import TypeInfo
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unproven_gate import UnprovenGate


class ProbeNames:
    """Sort the output names of a probe merge by what happens to each column.

    A join hook merges the columns of its two frames with its ordinary rules.
    For the probe, it runs that merge with a special suffix that no real column
    name can hold, because it starts with a NUL character. An output name that
    ends with that suffix is then, for sure, a right column that the join
    renames.
    """

    SUFFIX: Final[str] = "\0astrolabe_join_probe"

    def __init__(self, names: list[str]) -> None:
        """Keep the output names of one probe merge.

        Args:
            names: The output names of a merge that ran with
                `ProbeNames.SUFFIX`, in order.
        """
        self._names = names

    def renamed_right(self) -> frozenset[str]:
        """Give the right names that the join renames for sure.

        Returns:
            The name of each output that has the probe suffix, with the suffix
            taken off.

        Example:
            >>> ProbeNames(["k", "v" + ProbeNames.SUFFIX]).renamed_right()
            frozenset({'v'})
        """
        suffixed = [name for name in self._names if name.endswith(self.SUFFIX)]
        return frozenset(name.removesuffix(self.SUFFIX) for name in suffixed)

    def plain(self) -> frozenset[str]:
        """Give the output names that the join does not rename.

        Returns:
            Each output name that has no probe suffix.

        Example:
            >>> ProbeNames(["k", "v" + ProbeNames.SUFFIX]).plain()
            frozenset({'k'})
        """
        return frozenset(name for name in self._names if not name.endswith(self.SUFFIX))


@dataclass(frozen=True)
class DeclaredNames:
    """The declared column names of the two frames of a join.

    Attributes:
        left: The names that the type of the left frame names.
        right: The names that the type of the right frame names.

    Example:
        >>> DeclaredNames(frozenset({"k"}), frozenset()).left
        frozenset({'k'})
    """

    left: frozenset[str]
    right: frozenset[str]


@dataclass(frozen=True)
class JoinSides:
    """The names of one join, sorted by what the join does to each of them.

    Attributes:
        left_rest: The remainder type of the receiver, or None for a closed
            receiver.
        right_rest: The remainder type of the other frame, or None.
        surviving_left: The declared left names that the result keeps with their
            own name, less the names that the right frame declares. A remainder
            never holds a name that its own frame declares, so those names
            cannot hide in the right remainder.
        unrenamed_right: The declared right names that the result keeps with
            their own name.
        renamed_right: The declared right names that the result holds with the
            suffix added.
        suffix: The suffix that a right name takes if it collides.
        left_declared: All the declared left names, whatever happens to them.
            The check for a right remainder that sits inside the left names
            tests these against the suffix.
    """

    left_rest: MypyType | None
    right_rest: MypyType | None
    surviving_left: frozenset[str]
    unrenamed_right: frozenset[str]
    renamed_right: frozenset[str]
    suffix: str
    left_declared: frozenset[str] = frozenset()

    @classmethod
    def from_probe(
        cls,
        probe: ProbeNames,
        left_rest: MypyType | None,
        right_rest: MypyType | None,
        declared: DeclaredNames,
        suffix: str,
    ) -> JoinSides:
        """Sort the output names of one probe merge into the names of the join.

        An output without the probe suffix that names a left column is a left
        column that the result keeps. Every other output without the suffix is a
        right column that keeps its own name.

        Args:
            probe: The output names of the probe merge.
            left_rest: The remainder type of the receiver, or None.
            right_rest: The remainder type of the other frame, or None.
            declared: The declared names of the two frames.
            suffix: The real suffix that a right name takes if it collides.

        Returns:
            The names of the join, with the real suffix.

        Example:
            >>> sides = JoinSides.from_probe(
            ...     ProbeNames(["k", "v"]), None, None,
            ...     DeclaredNames(frozenset({"k"}), frozenset({"k", "v"})),
            ...     "_right",
            ... )
            >>> sorted(sides.unrenamed_right)
            ['v']
        """
        plain = probe.plain()
        return cls(
            left_rest,
            right_rest,
            (plain & declared.left) - declared.right,
            plain - declared.left,
            probe.renamed_right(),
            suffix,
            declared.left,
        )


@dataclass(frozen=True)
class JoinRemainderOutcome:
    """What the checker decided about the remainder of one join.

    Attributes:
        slot: The remainder type of the result, or None for a closed result. It
            has a meaning only if the join is allowed.
        at_risk: The names in the result that depend on the caller, sorted.
            Empty if every name is certain.
        remainders_can_share: True if the checker cannot prove that the two
            remainders share no name, so a hidden pair can collide.
        suffix_collision: True if the right remainder sits inside the left names
            but the left frame can already hold a name with the suffix, so a
            renamed right column can collide.

    Example:
        >>> JoinRemainderOutcome(None, (), False).allowed
        True
    """

    slot: MypyType | None
    at_risk: tuple[str, ...]
    remainders_can_share: bool
    suffix_collision: bool = False

    @property
    def allowed(self) -> bool:
        """Tell whether the join renames the same columns for every caller.

        Returns:
            True if no name is at risk, the remainders cannot share a name, and
            no name with the suffix can collide.

        Example:
            >>> JoinRemainderOutcome(None, ("sector",), False).allowed
            False
        """
        if self.at_risk or self.remainders_can_share:
            return False
        return not self.suffix_collision


class JoinRemainderResolver:
    """Decide the remainder of one join from the two remainder types and the names of the join.

    The decision tries two rules. The first rule proves that nothing hidden
    collides, and the remainder of the result is then the union of the two
    remainders. If that fails, the second rule tries to prove that every hidden
    right column collides: if every name that the right remainder can hold is a
    left name, each of those columns gets the suffix for every caller. The
    result then holds the left remainder and the right remainder with the
    suffix.

    This class holds the decision and no mypy context, so a doctest can check
    it. It reads the remainders through `RemainderKnowledge`, `SlotRelations`
    and `DeclarationCodec`, and it writes the result through `RemainderEdit`.

    Example:
        >>> fixture = new_type_fixture()
        >>> resolver = new_join_remainder_resolver(fixture)
        >>> closed = JoinSides(
        ...     None, None, frozenset({"k"}), frozenset({"v"}), frozenset(), "_right"
        ... )
        >>> resolver.outcome(closed).allowed
        True
    """

    def __init__(self, edit: RemainderEdit) -> None:
        """Keep the editor and the readers of remainder types.

        Args:
            edit: The remainder editor for the current phase. It writes the
                remainder type of an allowed result.
        """
        self._edit = edit
        self._knowledge = RemainderKnowledge()
        self._declarations = DeclarationCodec()
        self._relations = SlotRelations()

    def outcome(self, sides: JoinSides) -> JoinRemainderOutcome:
        """Decide the remainder of the join.

        Args:
            sides: The names of the join.

        Returns:
            The outcome: the remainder of the result if every name is certain,
            and what stands in the way otherwise.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> open_left = JoinSides(
            ...     declared_rest(fixture, without=("sector",)), None,
            ...     frozenset({"k"}), frozenset({"sector"}), frozenset(), "_right",
            ... )
            >>> resolver.outcome(open_left).slot
            decl.Rest
        """
        plain = self._plain_outcome(sides)
        if plain.allowed:
            return plain
        if not self._right_within_left(sides):
            return plain
        return self._containment_outcome(sides)

    def _plain_outcome(self, sides: JoinSides) -> JoinRemainderOutcome:
        """Decide the join under the rule that nothing hidden collides.

        Three groups of names in the result depend on the caller unless a
        record proves them absent from the other side. A record is what the
        checker knows about one remainder, for example the names that it
        cannot hold. The groups are the right names that keep their own name,
        the right names with the suffix, and the left names that the result
        keeps. A closed side proves every name absent.

        Args:
            sides: The names of the join.

        Returns:
            The union of the two remainders if no hidden collision is possible,
            and what stands in the way otherwise. The names at risk are sorted.
        """
        risky = self._relations.unprovable(sides.left_rest, sides.unrenamed_right)
        suffixed = frozenset(name + sides.suffix for name in sides.renamed_right)
        risky |= self._relations.unprovable(sides.left_rest, suffixed)
        risky |= self._relations.unprovable(sides.right_rest, sides.surviving_left)
        at_risk = tuple(sorted(risky))
        shared = self._relations.can_share(sides.left_rest, sides.right_rest)
        if at_risk or shared:
            return JoinRemainderOutcome(None, at_risk, shared)
        slot = self._edit.both(sides.left_rest, sides.right_rest)
        return JoinRemainderOutcome(slot, (), False)

    def _containment_outcome(self, sides: JoinSides) -> JoinRemainderOutcome:
        """Decide the join under the rule that every hidden right column collides.

        Every name that the right remainder can hold is a left name, so every
        column in it gets the suffix, whatever the caller passes. Two facts are
        still to prove: each new name with the suffix is free on the left, and
        the declared right columns still have a certain fate.

        Args:
            sides: The names of the join.

        Returns:
            The union of the left remainder and the right remainder with the
            suffix, or what stands in the way.
        """
        at_risk = self._relations.unprovable(sides.left_rest, sides.unrenamed_right)
        suffixed = frozenset(name + sides.suffix for name in sides.renamed_right)
        at_risk |= self._relations.unprovable(sides.left_rest, suffixed)
        suffix_taken = not self._suffix_free_on_left(sides)
        if at_risk or suffix_taken:
            return JoinRemainderOutcome(None, tuple(sorted(at_risk)), False, suffix_taken)
        renamed = self._edit.suffixed(sides.right_rest, sides.suffix)
        slot = self._edit.both(sides.left_rest, renamed)
        return JoinRemainderOutcome(slot, (), False)

    def _suffix_free_on_left(self, sides: JoinSides) -> bool:
        """Tell whether no left name can already end with the suffix.

        A renamed right column takes its own name plus the suffix. If a left
        name already ends with the suffix, the new name can be that left name,
        and polars then raises an error in place of adding the suffix twice.

        Args:
            sides: The names of the join.

        Returns:
            True if no declared left name ends with the suffix, and the left
            frame is closed or the record of its remainder proves every such
            name absent. A type that does not read as a remainder proves
            nothing.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> clean = JoinSides(
            ...     None, None, frozenset(), frozenset(), frozenset(),
            ...     "_prev", frozenset({"price"}),
            ... )
            >>> resolver._suffix_free_on_left(clean)
            True
            >>> taken = JoinSides(
            ...     None, None, frozenset(), frozenset(), frozenset(),
            ...     "_prev", frozenset({"price_prev"}),
            ... )
            >>> resolver._suffix_free_on_left(taken)
            False
            >>> opaque = JoinSides(
            ...     new_marker_minter(fixture).string("x"), None,
            ...     frozenset(), frozenset(), frozenset(), "_prev",
            ... )
            >>> resolver._suffix_free_on_left(opaque)
            False
        """
        if any(name.endswith(sides.suffix) for name in sides.left_declared):
            return False
        if sides.left_rest is None:
            return True
        record = self._knowledge.record(sides.left_rest)
        if record is None:
            return False
        return AbsenceEntailment()(EndsWith(sides.suffix), record.absence)

    def _right_within_left(self, sides: JoinSides) -> bool:
        """Tell whether every name that the right remainder can hold is a left name.

        A record gives an upper limit on the names that a remainder can hold. So
        it can prove the containment only against a set that is known to be
        present: the declared left names. The columns of the left remainder are
        reached through a `within=` promise. No promise can point into a closed
        frame.

        Args:
            sides: The names of the join.

        Returns:
            True if every name that the right record allows is a declared left
            name, or if the left frame is open and the right declarations
            promise that their names sit inside the left declaration.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> closed_left = JoinSides(
            ...     None, declared_rest(fixture),
            ...     frozenset(), frozenset(), frozenset(), "_right",
            ... )
            >>> resolver._right_within_left(closed_left)
            False
        """
        if sides.right_rest is None:
            return False
        if self._contained_in_declared(sides.left_declared, sides.right_rest):
            return True
        if sides.left_rest is None:
            return False
        return self._declared_within(sides.left_rest, sides.right_rest)

    def _contained_in_declared(self, left_declared: frozenset[str], right: MypyType) -> bool:
        """Tell whether the right record allows only declared left names.

        Args:
            left_declared: The declared names of the left frame.
            right: The remainder type of the other frame.

        Returns:
            True if every name that the right record allows is a declared left
            name. A type that does not read as a remainder proves nothing.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> bare = declared_rest(fixture)
            >>> resolver._contained_in_declared(frozenset({"a"}), bare)
            False
            >>> opaque = new_marker_minter(fixture).string("x")
            >>> resolver._contained_in_declared(frozenset({"a"}), opaque)
            False
        """
        right_record = self._knowledge.record(right)
        if right_record is None:
            return False
        right_present = right_record.absence.language().complement()
        return right_present.contained_in(OneOf(left_declared).language())

    def _declared_within(self, left: MypyType, right: MypyType) -> bool:
        """Tell whether the declarations promise that the right remainder sits inside the left one.

        The promise applies to the remainder types only if no change broke it.
        The left type must be one declared remainder with no change on it, so
        nothing that the promise points to has been taken out. The right type
        must not rename its columns.

        Args:
            left: The remainder type of the receiver.
            right: The remainder type of the other frame.

        Returns:
            True if every right declared remainder has a `within=` that the left
            declared remainder covers.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> opaque = new_marker_minter(fixture).string("x")
            >>> resolver._declared_within(opaque, declared_rest(fixture))
            False
            >>> unmarked = new_marker_minter(fixture).resolved("kept.R")
            >>> resolver._declared_within(declared_rest(fixture), unmarked)
            False
        """
        left_name = self._knowledge.whole_declaration(left)
        if left_name is None or not self._names_unchanged(right):
            return False
        rights = self._knowledge.declarations(right)
        if not rights:
            return False
        return all(self._within_covered(info, left_name) for info in rights.values())

    def _names_unchanged(self, rest: MypyType) -> bool:
        """Tell whether the changes on one remainder type keep the name of every column.

        A filter only takes columns out, and a retype keeps every name, so a
        promise about names still holds after either one. A prefix or a suffix
        renames, which breaks the promise.

        Args:
            rest: The remainder type to read.

        Returns:
            True if the type reads as a remainder with no prefix and no suffix.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> resolver._names_unchanged(declared_rest(fixture))
            True
            >>> opaque = new_marker_minter(fixture).string("x")
            >>> resolver._names_unchanged(opaque)
            False
        """
        form = self._edit.decoded(rest)
        if form is None:
            return False
        return not form.suffix and not form.prefix

    def _within_covered(self, info: TypeInfo, left_name: str) -> bool:
        """Tell whether the `within=` of one declared remainder names the left declared remainder.

        Args:
            info: The class of the right declared remainder.
            left_name: The full name of the left declared remainder.

        Returns:
            True if one `within=` entry names only the left declared remainder.

        Example:
            >>> import astrolabe.remainder as remainder_markers
            >>> fixture = new_type_fixture()
            >>> resolver = new_join_remainder_resolver(fixture)
            >>> minter = new_marker_minter(fixture)
            >>> bad = minter.marker(
            ...     remainder_markers.Within, [minter.string("x")]
            ... )
            >>> info = declaration_info(fixture, bad, name="decl.S")
            >>> resolver._within_covered(info, "decl.Rest")
            False
        """
        relations = self._declarations.relations_of(info)
        if relations is None:
            return False
        return any(group <= {left_name} for group in relations.within)


class JoinRemainderStep:
    """Decide the remainder of one join call, and report the call if it depends on the caller.

    Every hook that merges the columns of two frames and adds a suffix on a
    collision uses this step, so `.join()`, `.join_where()` and `.join_asof()`
    always decide in the same way. The step builds the editor for the type
    checking phase, asks `JoinRemainderResolver`, and reports a refused join
    through `UnprovenGate`, with the error that names the cause.
    """

    def __init__(self, plugin: Plugin, gate: UnprovenGate, operation: str) -> None:
        """Keep the plugin, the gate and the name of the operation.

        The editor finds the marker classes through the plugin.

        Args:
            plugin: The mypy plugin.
            gate: The gate that decides whether a join that depends on the
                caller is reported.
            operation: The name of the join operation, for the error.
        """
        self._plugin = plugin
        self._gate = gate
        self._operation = operation

    def decided(self, ctx: MethodContext, sides: JoinSides) -> JoinRemainderOutcome | None:
        """Decide the remainder of the result, and report a join that depends on the caller.

        Args:
            ctx: The mypy context of the join call.
            sides: The names of the join.

        Returns:
            The allowed outcome, or None for a join whose renames depend on the
            caller. That join is reported, unless the `allow_unproven` setting
            lets it pass.
        """
        resolver = JoinRemainderResolver(RemainderEdit.for_checker(self._plugin, ctx.api))
        outcome = resolver.outcome(sides)
        if outcome.allowed:
            return outcome
        self._report(ctx, outcome, sides.suffix)
        return None

    def _report(self, ctx: MethodContext, outcome: JoinRemainderOutcome, suffix: str) -> None:
        """Report why the renames of one join depend on the caller.

        A name at risk gets the error that names it. Otherwise a suffix that the
        left frame can already hold gets its own error. Otherwise the error says
        that the two remainders can share a name.

        Args:
            ctx: The mypy context of the join call.
            outcome: The outcome that is not allowed.
            suffix: The suffix that a right name takes if it collides.
        """
        if outcome.at_risk:
            self._gate.report(ctx, JoinRenameAmbiguityError(outcome.at_risk, self._operation))
            return
        if outcome.suffix_collision:
            self._gate.report(ctx, JoinSuffixTakenError(suffix, self._operation))
            return
        self._gate.report(ctx, JoinRemainderOverlapError(self._operation))
