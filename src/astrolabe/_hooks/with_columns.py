"""Type `.with_columns()` on a typed frame: add new columns, or replace columns by name.

The typed frame declares `.with_columns()` to return its own type, which is
wrong as soon as a call adds a column or changes a data type. This hook
computes the real columns of the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.encoding.remainder_knowledge import SlotRelations
from astrolabe._hooks.expression_requests import (
    ExpressionRequest,
    MemberSplit,
    RequestPolicies,
    RequestReader,
    RequestResolver,
    Resolution,
    Unpicked,
)
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, not_refused

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
    from astrolabe._hooks.expression.selection import RemainderEffect


@dataclass(frozen=True)
class _WithColumnsPlan:
    """Hold what the hook reads from one `.with_columns()` call before it types the arguments.

    Attributes:
        frame: The receiver frame.
        known: The declared columns of the receiver, and its remainder.
        requests: Each argument of the call, in call order.
    """

    frame: Frame
    known: KnownColumns
    requests: list[ExpressionRequest]

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(
        cls,
        ctx: MethodContext,
        requests: list[ExpressionRequest] | None,
    ) -> _WithColumnsPlan:
        """Read the receiver of a `.with_columns()` call, with its arguments.

        Args:
            ctx: The context that mypy gives for the call.
            requests: The arguments, or None where an argument gives no
                keyword to read.

        Returns:
            The plan, or a `Refusal`. The reason is `DYNAMIC_NAME` if an
            argument comes from `**kwargs` unpacking, which gives no
            keyword to read. It is `CHECKER_GAP` if the receiver is not a
            typed frame, or if the call has no argument.
        """
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        arguments = not_none(requests, reason=Reason.DYNAMIC_NAME)
        return cls(frame, known, not_falsy(arguments))


class WithColumnsHook:
    """Give `.with_columns()` on a typed frame its result type, and report the calls that fail.

    Each argument gives one or more output columns, in the same way as for
    `.select()`. An output whose name is already a column replaces that
    column in place. An output with a new name is added after the old
    columns, in argument order: the positional arguments first, then the
    keyword arguments, as polars does. Each argument is evaluated against
    the receiver as it was before the call, so no argument can read the
    output of another. A keyword names its output, and wins over an
    outermost `.alias(...)`.

    The receiver can be an open frame, whose type does not name all of its
    columns. The columns that it does not name are its remainder. polars
    replaces a remainder column that has the name of an output, so each
    output name leaves the remainder. A selection, that is a typed
    selector such as `cs.numeric()`, can change the remainder columns
    that it matches. It either changes their data type in place, or adds
    renamed copies of them with a prefix or a suffix. If such a copy can
    have the name of a column that is already there, the result depends on
    the columns of the caller, and the hook gives up.

    It is a mypy method hook in the plugin layer. `RequestResolver` types
    each argument, and `RemainderEdit` computes the new remainder. If the
    hook gives up, it falls back through `UntypedFallback`, which gives the
    plain frame that polars declares and reports it unless the
    `allow_unproven` setting lets it pass.
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
                output columns and the new remainder need.
            unchecked_fallback: Gives the data type `Unchecked` to an
                output whose name is known and whose data type is not,
                and reports it unless the setting lets it pass.
            built_for: The name of a method that polars runs as
                `with_columns` over an expression that it builds, such as
                `fill_nan`, or None for `with_columns` itself. Errors and the
                `allow_unproven` setting name that method. The user does not
                write the selection, so where polars refuses it at a kind
                that the remainder can hold, the result depends on the
                columns that the caller passes.
        """
        self._untyped = untyped_fallback
        self._plugin = plugin
        self._operation = built_for or "with_columns"
        refused = None if built_for is None else Reason.CALLER_DEPENDENT
        self._policies = RequestPolicies(
            self._operation,
            "row",
            plugin,
            unchecked_fallback,
            refused,
            unpicked=Unpicked.KEPT,
        )
        self._duplicate_check = DuplicateNameCheck(self._operation)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.with_columns()` call.

        Args:
            ctx: The context that mypy gives for `.with_columns()`.

        Returns:
            The frame that `typed` gives for the arguments of the call.
        """
        return self.typed(ctx, RequestReader("exprs", "named_exprs").of(ctx))

    def typed(self, ctx: MethodContext, requests: list[ExpressionRequest] | None) -> MypyType:
        """Give the frame that `with_columns` gives for some arguments.

        A method that polars runs as `with_columns` over an expression that
        the user does not write, such as `fill_nan`, gives the arguments
        itself.

        Args:
            ctx: The context that mypy gives for the call.
            requests: The arguments, or None where an argument gives no
                keyword to read.

        Returns:
            A typed frame of the same kind as the receiver, with each
            output added or replaced. If the hook cannot type the call,
            the plain `pl.DataFrame` or `pl.LazyFrame`: an argument cannot
            be typed, two outputs share a name, or a renamed copy can clash
            with a column.
        """
        split = MemberSplit()
        choices = split.choices(requests)
        if choices is not None:
            return split.joined([self.typed(ctx, choice) for choice in choices])
        plan = _WithColumnsPlan.from_context(ctx, requests)
        if isinstance(plan, Refusal):
            return self._untyped.resolve(
                ctx,
                plan.reason,
                self._operation,
                Frame.from_proper_type(ctx.type),
            )
        resolved = self._resolve(ctx, plan)
        if isinstance(resolved, Refusal):
            return self._untyped.resolve(ctx, resolved.reason, self._operation, plan.frame)
        return self._rebuilt(ctx, plan, resolved)

    def _every_affix_proven(
        self,
        ctx: MethodContext,
        plan: _WithColumnsPlan,
        resolved: list[Resolution],
    ) -> bool:
        """Answer whether each selection with a prefix or suffix adds only new names.

        Such a selection adds renamed copies of the remainder columns it
        matches. polars replaces a column that already has one of those
        names. So the result depends on the columns of the caller, unless
        the new names cannot be in the remainder and are not declared. The
        declared names are those of the receiver and each output name.

        A remainder never holds a declared name, because a frame holds
        each name once. The method takes the declared names out of the
        remainder first, so that the remainder type states this fact and
        the proof can use it.

        Args:
            ctx: The context that mypy gives for `.with_columns()`.
            plan: What the hook read from the call.
            resolved: The typed arguments.

        Returns:
            True if no added name can be a remainder column or a declared
            column.
        """
        edit = RemainderEdit.for_checker(self._plugin, ctx.api)
        receiver = plan.known.columns
        receiver_names = [column.name for column in receiver if column.name is not None]
        result_columns = [*receiver, *Resolution.columns_of(resolved)]
        declared = frozenset(column.name for column in result_columns if column.name is not None)
        rest = edit.minus(plan.known.rest, receiver_names)
        effects = [resolution.effect for resolution in resolved]
        return all(self._affix_proven(edit, rest, effect, declared) for effect in effects)

    def _affix_proven(
        self,
        edit: RemainderEdit,
        rest: MypyType | None,
        effect: RemainderEffect | None,
        declared: frozenset[str],
    ) -> bool:
        """Answer whether the renamed copies of one selection all have new names.

        The copies can clash in two ways. A copy can have the name of a
        column in the remainder, which `SlotRelations.can_share` decides.
        A copy can have a declared name, which `SlotRelations.unprovable`
        decides.

        Args:
            edit: The remainder editor for this call.
            rest: The encoded remainder of the receiver.
            effect: The effect of one selection, or None for an argument
                that is not a selection.
            declared: Each name that the result declares.

        Returns:
            True for an argument that is not a selection, for a selection
            with no prefix or suffix, and for copies whose names are
            proven new. Else False.
        """
        if rest is None or effect is None or effect.affix.is_identity():
            return True
        image = edit.selected(rest, effect)
        relations = SlotRelations()
        return not relations.can_share(rest, image) and not relations.unprovable(image, declared)

    @explains(Reason.CHECKER_GAP)
    def _resolve(self, ctx: MethodContext, plan: _WithColumnsPlan) -> list[Resolution]:
        """Turn each argument into its output columns, and check the output names.

        polars fails if two arguments of one call give the same output
        name, even though an output can replace an old column. The method
        reports that duplicate. Each selection with a prefix or suffix must
        also add only new names.

        Args:
            ctx: The context that mypy gives for `.with_columns()`.
            plan: What the hook read from the call.

        Returns:
            The typed arguments in argument order, or a `Refusal`. It is the
            `Refusal` of the first argument that cannot be typed.
            `REPORTED` if two outputs share a name, after the hook reports
            the duplicate. `CALLER_DEPENDENT` if a renamed copy can clash
            with a column.
        """
        resolver = RequestResolver(ctx, plan.known, self._policies)
        resolved = [not_refused(resolver.resolved_all(request)) for request in plan.requests]
        names = [column.name for column in Resolution.columns_of(resolved)]
        not_falsy(not self._duplicate_check.fails(ctx, names), reason=Reason.REPORTED)
        proven = self._every_affix_proven(ctx, plan, resolved)
        not_falsy(proven, reason=Reason.CALLER_DEPENDENT)
        return resolved

    def _rebuilt(
        self,
        ctx: MethodContext,
        plan: _WithColumnsPlan,
        resolved: list[Resolution],
    ) -> MypyType:
        """Build the result frame, with each output name taken out of the remainder.

        polars replaces a column that has the name of an output. It does
        not add a second column with that name. So each output name leaves
        the remainder and becomes a declared column of the result. Then
        each selection changes the remainder: it retypes the columns it
        matches in place, or it adds their renamed copies. The selection
        matches against the remainder of the receiver as it was before the
        call, because every argument reads the original frame.

        Args:
            ctx: The context that mypy gives for `.with_columns()`.
            plan: What the hook read from the call.
            resolved: The typed arguments, in argument order.

        Returns:
            The result frame.
        """
        outputs = Resolution.columns_of(resolved)
        names = [column.name for column in outputs if column.name is not None]
        edit = RemainderEdit.for_checker(self._plugin, ctx.api)
        rest = edit.minus(plan.known.rest, names)
        for resolution in resolved:
            if resolution.effect is not None:
                rest = edit.with_effect(plan.known.rest, rest, resolution.effect)
        columns = self._overlaid(plan.known.columns, outputs)
        return plan.frame.rebuilt(columns, ctx.api, rest=rest).as_instance()

    def _overlaid(
        self,
        existing: Sequence[ColumnRef],
        resolved: Sequence[ColumnRef],
    ) -> list[ColumnRef]:
        """Put the output columns over the declared columns of the receiver.

        Args:
            existing: The declared columns of the receiver, in order.
            resolved: Each output column, in argument order.

        Returns:
            The declared columns, with each one that has the name of an
            output replaced in place, then each new output in argument
            order.

        Example:
            >>> fixture = new_type_fixture()
            >>> hook = WithColumnsHook(untyped_fallback(), None, unchecked_fallback())
            >>> price = column_ref(fixture, "price", int_dtype(fixture))
            >>> tax = column_ref(fixture, "tax", str_dtype(fixture))
            >>> total = column_ref(fixture, "price", str_dtype(fixture))
            >>> [column.name for column in hook._overlaid([price], [total, tax])]
            ['price', 'tax']
        """
        by_name = {column.name: column for column in existing}
        for column in resolved:
            by_name[column.name] = column
        return list(by_name.values())
