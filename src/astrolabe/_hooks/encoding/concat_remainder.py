"""Decide the remainder of the result of one `pl.concat` call, or say what stops the decision.

The remainder of an open frame is the set of columns that its type does not
name. A concat that lines up columns by name, for example a diagonal concat,
treats a name that two members share differently from a name that only one
member has. If one member declares a name and another member can hold the same
name in its remainder, the set of shared names depends on the frames that each
caller passes. The checker then cannot name the columns of the result. The same
holds if the remainders of two members can share a name.

So the checker needs proof. Every name that one member declares must be absent
from the remainder of each other member, and no two remainders can share a name.
If the proof holds, the remainder of the result is the union of the remainders
of the members. If not, the call is reported through the gate of the
`allow_unproven` setting, which decides whether a result that the checker cannot
prove passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import SlotRelations
from astrolabe._hooks.errors import (
    ConcatHiddenCollisionError,
    ConcatRemainderOverlapError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import FunctionContext, MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unproven_gate import UnprovenGate


@dataclass(frozen=True)
class ConcatMember:
    """One frame passed to `pl.concat`: its declared names and its remainder type.

    Attributes:
        rest: The remainder type of the frame, or None for a closed frame.
        declared: The names of the columns that the type of the frame names.

    Example:
        >>> ConcatMember(None, frozenset({"k"})).declared
        frozenset({'k'})
    """

    rest: MypyType | None
    declared: frozenset[str]


@dataclass(frozen=True)
class ConcatRemainderOutcome:
    """What the checker decided about the remainder of one concat.

    Attributes:
        slot: The remainder type of the result, or None for a closed result. It
            has a meaning only if the concat is allowed.
        at_risk: The declared names that the remainder of another member can
            also hold, sorted.
        remainders_can_share: True if the checker cannot prove that the
            remainders of every two members share no name.

    Example:
        >>> ConcatRemainderOutcome(None, (), False).allowed
        True
    """

    slot: MypyType | None
    at_risk: tuple[str, ...]
    remainders_can_share: bool

    @property
    def allowed(self) -> bool:
        """Tell whether the checker proved every pair of members safe.

        Returns:
            True if no name is at risk and no two remainders can share a name.

        Example:
            >>> ConcatRemainderOutcome(None, ("v",), False).allowed
            False
        """
        return not self.at_risk and not self.remainders_can_share


class ConcatRemainderResolver:
    """Decide the remainder of one concat from the remainders and declared names of its members.

    This class holds the decision and no mypy context, so a doctest can check
    it. It reads the relations between remainders through `SlotRelations`, and
    it writes the remainder of an allowed result through `RemainderEdit`.

    Example:
        >>> fixture = new_type_fixture()
        >>> resolver = new_concat_remainder_resolver(fixture)
        >>> members = [ConcatMember(None, frozenset({"k"}))] * 2
        >>> resolver.outcome(members)
        ConcatRemainderOutcome(slot=None, at_risk=(), remainders_can_share=False)
    """

    def __init__(self, edit: RemainderEdit) -> None:
        """Keep the editor and the reader of relations between remainders.

        Args:
            edit: The remainder editor for the current phase. It writes the
                remainder type of an allowed result.
        """
        self._edit = edit
        self._relations = SlotRelations()

    def outcome(self, members: Sequence[ConcatMember]) -> ConcatRemainderOutcome:
        """Decide the remainder of the concat.

        Args:
            members: The frames of the call, in the order of the call.

        Returns:
            The union of the remainders if every pair is proven safe, and what
            stands in the way otherwise. The union is None if every member is
            closed.

        Example:
            >>> fixture = new_type_fixture()
            >>> resolver = new_concat_remainder_resolver(fixture)
            >>> open_member = ConcatMember(declared_rest(fixture), frozenset())
            >>> closed = ConcatMember(None, frozenset({"v"}))
            >>> resolver.outcome([open_member, closed]).at_risk
            ('v',)
        """
        at_risk = self._risky_names(members)
        shared = self._any_pair_can_share(members)
        if at_risk or shared:
            return ConcatRemainderOutcome(None, at_risk, shared)
        slot: MypyType | None = None
        for member in members:
            slot = self._edit.both(slot, member.rest)
        return ConcatRemainderOutcome(slot, (), False)

    def _risky_names(self, members: Sequence[ConcatMember]) -> tuple[str, ...]:
        """Collect every declared name that the remainder of another member can hold.

        The remainder of a member never holds a name that the same member
        declares. So the declared names of each member are tested against the
        remainder of every other member, less the names that the other member
        declares itself.

        Args:
            members: The frames of the call, in the order of the call.

        Returns:
            The names at risk, sorted.
        """
        risky: frozenset[str] = frozenset()
        for owner in members:
            for other in members:
                if owner is other:
                    continue
                tested = owner.declared - other.declared
                risky |= self._relations.unprovable(other.rest, tested)
        return tuple(sorted(risky))

    def _any_pair_can_share(self, members: Sequence[ConcatMember]) -> bool:
        """Tell whether the remainders of any two members can share a name.

        Args:
            members: The frames of the call, in the order of the call.

        Returns:
            True if the checker cannot prove, for some pair, that the two
            remainders share no name.
        """
        pairs = (
            self._relations.can_share(left.rest, right.rest)
            for index, left in enumerate(members)
            for right in members[index + 1 :]
        )
        return any(pairs)


class ConcatRemainderStep:
    """Decide the remainder of one `pl.concat` call, and report the call if no proof holds.

    The hook for `pl.concat` uses this step. It builds the editor for the type
    checking phase, asks `ConcatRemainderResolver`, and reports a refused call
    through `UnprovenGate`, with the error that names the cause.
    """

    def __init__(self, plugin: Plugin, gate: UnprovenGate, operation: str = "concat") -> None:
        """Keep the plugin, the gate and the name of the operation.

        The editor finds the marker classes through the plugin.

        Args:
            plugin: The mypy plugin.
            gate: The gate that decides whether a concat that depends on the
                caller is reported.
            operation: The operation, for the error and for the setting that
                lets it pass: `concat` or `hstack`.
        """
        self._plugin = plugin
        self._gate = gate
        self._operation = operation

    def decided(
        self,
        ctx: FunctionContext | MethodContext,
        members: Sequence[ConcatMember],
    ) -> ConcatRemainderOutcome | None:
        """Decide the remainder of the result, and report a concat that depends on the caller.

        Args:
            ctx: The mypy context of the call to `pl.concat(...)`.
            members: The frames of the call, in the order of the call.

        Returns:
            The allowed outcome, or None for a concat whose result depends on
            the caller. That concat is reported, unless the `allow_unproven`
            setting lets it pass.
        """
        resolver = ConcatRemainderResolver(RemainderEdit.for_checker(self._plugin, ctx.api))
        outcome = resolver.outcome(members)
        if outcome.allowed:
            return outcome
        self._report(ctx, outcome)
        return None

    def _report(
        self,
        ctx: FunctionContext | MethodContext,
        outcome: ConcatRemainderOutcome,
    ) -> None:
        """Report why the result of one concat depends on the caller.

        A declared name at risk gets the error that names it. Otherwise the
        error says that two remainders can share a name.

        Args:
            ctx: The mypy context of the call to `pl.concat(...)`.
            outcome: The outcome that is not allowed.
        """
        if outcome.at_risk:
            self._gate.report(ctx, ConcatHiddenCollisionError(outcome.at_risk, self._operation))
            return
        self._gate.report(ctx, ConcatRemainderOverlapError(self._operation))
