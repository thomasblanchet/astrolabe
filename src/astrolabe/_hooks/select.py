"""Type `.select()` on a typed frame: the result holds exactly the columns that the call asks for.

The typed frame declares `.select()` to return its own type, which is wrong
for almost every call. This hook computes the real columns of the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import DuplicateColumnError
from astrolabe._hooks.expression_requests import (
    ExpressionRequest,
    MemberSplit,
    RequestPolicies,
    RequestReader,
    RequestResolver,
    Resolution,
    SelectionArgument,
    Unpicked,
)
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, not_refused

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
    from astrolabe._hooks.expression.selection import RemainderEffect


@dataclass(frozen=True)
class _SelectPlan:
    """Hold what the hook reads from one `.select()` call before it types the arguments.

    Attributes:
        frame: The receiver frame.
        requests: Each argument of the call, in call order.
        known: The declared columns of the receiver, and its remainder.
    """

    frame: Frame
    requests: list[ExpressionRequest]
    known: KnownColumns

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(
        cls,
        ctx: MethodContext,
        requests: list[ExpressionRequest] | None,
    ) -> _SelectPlan:
        """Read the receiver of a `.select()` call, with its arguments.

        Args:
            ctx: The context that mypy gives for the call.
            requests: The arguments, or None where an argument gives no
                keyword to read.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if an
            argument comes from `**kwargs` unpacking, which gives no
            keyword to read. It is `CHECKER_GAP` if the receiver is not a
            typed frame.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        arguments = not_none(requests, reason=Reason.DYNAMIC_NAME)
        known = not_none(frame.known_columns)
        return cls(frame, arguments, known)


@dataclass(frozen=True)
class _Selected:
    """Hold the typed outputs of one `.select()` call.

    Attributes:
        columns: Each output column, in call order.
        effects: For each selection argument, what it keeps of the
            remainder of the receiver and how it changes those columns.
    """

    columns: list[ColumnRef]
    effects: list[RemainderEffect]


class SelectHook:
    """Give `.select()` on a typed frame the type of its result, and report the calls that fail.

    Each argument gives one or more output columns. An expression, such
    as `pl.col("price") * 2`, is evaluated against the declared columns of
    the receiver, where each expression gives one value per row. A bare
    string names a declared column, which is kept as it is. A bare scalar
    literal gives a new column. A keyword argument, as in
    `select(total=...)`, names its output, and wins over an outermost
    `.alias(...)`.

    A selection is a typed selector, such as `cs.numeric()`. It picks the
    declared columns that it matches, and it can also pick columns from
    the remainder of an open frame. The remainder is the set of columns
    of an open frame that its type does not name. The result then keeps
    the part of the remainder that the selection matches. With no
    selection, the result has no remainder, so it is a closed frame.

    polars fails if two outputs share a name. The hook reports that, and
    also reports a new output name that the kept remainder can hold.

    It is a mypy method hook in the plugin layer. `RequestResolver` types
    each argument, and `RemainderEdit` computes the kept remainder. If an
    argument cannot be typed, the hook falls back through
    `UntypedFallback`, which gives the plain frame that polars declares
    and reports it unless the `allow_unproven` setting lets it pass.
    """

    def __init__(
        self,
        untyped_fallback: UntypedFallback,
        plugin: Plugin,
        unchecked_fallback: UncheckedFallback,
        built_for: str | None = None,
    ) -> None:
        """Keep the tools that the hook needs for each call.

        Args:
            untyped_fallback: Gives the plain frame for a call that the
                hook cannot type, and reports it unless the setting lets
                it pass.
            plugin: The mypy plugin, which finds the classes that the
                output column types need.
            unchecked_fallback: Gives the data type `Unchecked` to an
                output whose name is known and whose data type is not,
                and reports it unless the setting lets it pass.
            built_for: The name of a method that polars runs as
                `select` over an expression that it builds, such as
                `fill_null`, or None for `select` itself. Errors and the
                `allow_unproven` setting name that method. The user does not
                write the selection, so where polars refuses it at a kind
                that the remainder can hold, the result depends on the
                columns that the caller passes.
        """
        self._untyped = untyped_fallback
        self._plugin = plugin
        self._operation = built_for or "select"
        refused = None if built_for is None else Reason.CALLER_DEPENDENT
        self._policies = RequestPolicies(
            self._operation,
            "row",
            plugin,
            unchecked_fallback,
            refused,
            unpicked=Unpicked.ABSENT,
        )
        self._duplicate_check = DuplicateNameCheck(self._operation)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.select()` call.

        Args:
            ctx: The context that mypy gives for `.select()`.

        Returns:
            The frame that `typed` gives for the arguments of the call.
        """
        return self.typed(ctx, RequestReader("exprs", "named_exprs").of(ctx))

    def typed(self, ctx: MethodContext, requests: list[ExpressionRequest] | None) -> MypyType:
        """Give the frame that `select` gives for some arguments.

        A method that polars runs as `select` over an expression that the
        user does not write, such as `fill_null` with a strategy, gives the
        arguments itself.

        Args:
            ctx: The context that mypy gives for the call.
            requests: The arguments, or None where an argument gives no
                keyword to read.

        Returns:
            A typed frame of the same kind as the receiver, with the
            output columns and the kept remainder. If the hook cannot
            type the call, the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        split = MemberSplit()
        choices = split.choices(requests)
        if choices is not None:
            return split.joined([self.typed(ctx, choice) for choice in choices])
        plan = _SelectPlan.from_context(ctx, requests)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                self._operation,
                Frame.from_proper_type(ctx.type),
            )
        selected = self._selected(ctx, plan)
        if isinstance(selected, Refusal):
            return self._untyped.resolve(ctx, selected.reason, self._operation, plan.frame)
        rest = self._remainder(ctx, plan, selected)
        return plan.frame.rebuilt(selected.columns, ctx.api, rest=rest).as_instance()

    @explains(Reason.CHECKER_GAP)
    def _selected(self, ctx: MethodContext, plan: _SelectPlan) -> _Selected:
        """Turn each argument into its output columns.

        Args:
            ctx: The context that mypy gives for `.select()`.
            plan: What the hook read from the call.

        Returns:
            The output columns in call order, and the effect of each
            selection on the remainder. A `Refusal` with its reason if an
            argument cannot be typed. A `Refusal` with `REPORTED` if two
            outputs share a name, because that error is already reported.
        """
        selection = any(SelectionArgument().matches(request.arg_type) for request in plan.requests)
        alone = len(plan.requests) == 1 and (plan.known.rest is None or not selection)
        resolver = RequestResolver(ctx, plan.known, replace(self._policies, alone=alone))
        resolutions = [not_refused(resolver.resolved_all(request)) for request in plan.requests]
        columns = Resolution.columns_of(resolutions)
        names = [column.name for column in columns]
        not_falsy(not self._duplicate_check.fails(ctx, names), reason=Reason.REPORTED)
        return _Selected(columns, Resolution.effects_of(resolutions))

    def _remainder(
        self,
        ctx: MethodContext,
        plan: _SelectPlan,
        selected: _Selected,
    ) -> MypyType | None:
        """Compute the remainder of the result, from what each selection keeps.

        Each selection keeps its own part of the remainder of the
        receiver, and the result holds all of those parts together. If a
        new output name can also be a column in a kept part, polars can
        see two columns with that name. The hook reports that as a
        duplicate, the same error as two outputs with one name.

        Args:
            ctx: The context that mypy gives for `.select()`.
            plan: What the hook read from the call.
            selected: The output columns and the selection effects.

        Returns:
            The encoded remainder of the result, or None if the call has
            no selection.
        """
        edit = RemainderEdit.for_checker(self._plugin, ctx.api)
        rest: MypyType | None = None
        for effect in selected.effects:
            kept = edit.selected(plan.known.rest, effect)
            for name in self._fresh_names(plan, selected.columns, effect):
                if not RemainderKnowledge().proves_absent(kept, name):
                    DuplicateColumnError(self._operation, name).fail(ctx)
            rest = edit.both(rest, kept)
        return rest

    def _fresh_names(
        self,
        plan: _SelectPlan,
        columns: list[ColumnRef],
        effect: RemainderEffect,
    ) -> list[str]:
        """List the output names that can clash with the part of the remainder a selection keeps.

        A declared name is never in the remainder, because the remainder
        holds only the columns that the type does not name. A declared
        name with the prefix or suffix of the selection added is also
        outside what the selection keeps. So only a new name, from a
        literal or an alias, can clash.

        Args:
            plan: What the hook read from the call.
            columns: Each output column.
            effect: The effect of one selection.

        Returns:
            The new names, in call order. A column with no literal name is
            not in the list.

        Example:
            >>> from astrolabe._hooks.expression.selection import Affix, RemainderEffect
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import NameTest
            >>> fixture = new_type_fixture()
            >>> hook = SelectHook(untyped_fallback(), None, unchecked_fallback())
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> shown = frame(fixture, [price.as_instance()])
            >>> plan = _SelectPlan(shown, [], known_columns([price]))
            >>> suffixed = column_ref(fixture, "price_r", int_dtype(fixture))
            >>> fresh = column_ref(fixture, "one", int_dtype(fixture))
            >>> effect = RemainderEffect(NameTest(Always()), None, Affix(suffix="_r"))
            >>> hook._fresh_names(plan, [suffixed, fresh], effect)
            ['one']
        """
        declared = set(plan.known.by_name())
        excluded = declared | {effect.affix.applied(name) for name in declared}
        fresh: list[str] = []
        for column in columns:
            if column.name is not None and column.name not in excluded:
                fresh.append(column.name)
        return fresh
