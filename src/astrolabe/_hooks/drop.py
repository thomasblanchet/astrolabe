"""The hook for `.drop()`, which takes the named columns out of the type of the result.

Without this hook, mypy gives `.drop()` the type of the receiver, and
the dropped columns stay in the type. A later read of such a column then
passes the checker and fails at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mypy.nodes import StrExpr
from mypy.types import Instance, get_proper_type

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.deferred.code_reader import HoleValues
from astrolabe._hooks.deferred.polars_selector import SelectionRoot
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.marker_minter import ProperReads
from astrolabe._hooks.encoding.remainder_knowledge import RemainderKnowledge
from astrolabe._hooks.errors import InvalidCallShapeError
from astrolabe._hooks.expression.selection import RemainderEffect
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.frame import Frame
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.domain.requested_columns import RequestedColumns
from astrolabe._hooks.propagate import Refusal, explains, not_falsy, not_none, not_refused
from astrolabe.expr import TypedColumn

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.deferred.code import Code
    from astrolabe._hooks.domain.frame import KnownColumns
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback
    from astrolabe._hooks.remainder.selector import Selector


@dataclass(frozen=True)
class _DropPlan:
    """The inputs of one `.drop()` call, read from the call once.

    The hook needs the receiver, the names to drop and the `strict`
    argument together. Each of them can be missing or unreadable, so the
    plan reads them in one chain that stops at the first gap and names the
    reason.

    Attributes:
        frame: The receiver.
        requested: The literal names of the columns to drop.
        known: The columns of the receiver: the declared columns and the
            remainder. The remainder holds the columns of an open frame that
            its type does not name.
        strict: The literal value of the `strict` argument. With False,
            polars ignores a name that it does not find. With True, the
            default, polars raises `ColumnNotFoundError`. None means that the
            argument is not a literal, so the checker cannot know which case
            applies.
        codes: The code of each argument that is no literal name: a column
            or a selector, which polars expands to names.
        columns: The name of each declared column that an argument passes,
            as a parameter typed `TypedColumn[Literal["a"], ...]`.
    """

    frame: Frame
    requested: list[str]
    known: KnownColumns
    strict: bool | None
    codes: list[Code[MypyType]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def from_context(cls, ctx: MethodContext) -> _DropPlan:
        """Read the plan from a `.drop()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The plan, or a refusal with its reason. The reason is a dynamic name
            if an argument is no literal string and holds no code that the
            checker reads. It is a checker gap if the receiver is not a typed
            frame.
        """
        args = CallArgs(ctx)
        frame = not_none(Frame.from_proper_type(ctx.type))
        known = not_none(frame.known_columns)
        strict = args.literal_bool("strict", default=True)
        names = args.vararg_str_list("args")
        if names is not None:
            return cls(frame, names, known, strict)
        plan = cls(frame, [], known, strict)
        for node in args.vararg_node_list("args"):
            if isinstance(node, StrExpr):
                plan.requested.append(node.value)
                continue
            not_refused(plan._read(ctx.api.get_expression_type(node)))
        return plan

    @explains(Reason.DYNAMIC_NAME)
    def _read(self, typ: MypyType) -> _DropPlan:
        """Add one argument that is no literal name: a declared column, or a code.

        Args:
            typ: The type of the argument.

        Returns:
            This plan, or a refusal where the argument names no column that
            the checker can read.
        """
        proper = get_proper_type(typ)
        if isinstance(proper, Instance) and Target.of(TypedColumn).matches(proper):
            self.columns.append(not_none(ProperReads().string_literal(proper.args[0])))
            return self
        found = not_none(DeferredArgument("row").codes(typ))
        not_falsy(len(found) == 1)
        self.codes.append(found[0])
        return self


@dataclass(frozen=True)
class _Expanded:
    """The columns that the codes of one `.drop()` call drop, as polars expands them.

    Attributes:
        names: The declared columns that a code picks, and each column that
            a code names.
        required: The names that polars requires the frame to hold, whatever
            `strict` says: each column that a code names, and each name of
            a list that a selector requires.
        selectors: The selector of each code, which also picks columns of
            the remainder of an open frame.
    """

    names: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    selectors: list[Selector] = field(default_factory=list)

    @classmethod
    @explains(Reason.CHECKER_GAP)
    def of(cls, ctx: MethodContext, plan: _DropPlan) -> _Expanded:
        """Expand each code of a call to the columns that it drops.

        Polars refuses an expression that is no column and no selector,
        so the method reports it. Where the data type of a column decides
        whether a selector picks it, the columns of the result depend on
        the data.

        Args:
            ctx: The mypy context of the method call.
            plan: The plan of the call.

        Returns:
            The expansion, or a refusal with its reason: `REPORTED` after an
            error, `DYNAMIC_NAME` for a column or a pattern from a variable,
            `PARTIAL_PICK` where the data decide a pick, and a checker gap
            where the checker cannot read a selector.
        """
        expanded = cls(list(plan.columns), list(plan.columns))
        for code in plan.codes:
            declared = DeferredArgument("row").declared(code)
            refused = SelectionRoot().refused(declared)
            if refused:
                InvalidCallShapeError.not_a_selector("drop").fail(ctx)
            not_falsy(not refused, reason=Reason.REPORTED)
            not_refused(expanded._read(code, plan.known, column=SelectionRoot().column(declared)))
        return expanded

    @explains(Reason.CHECKER_GAP)
    def _read(self, code: Code[MypyType], known: KnownColumns, *, column: bool) -> _Expanded:
        """Add the columns that one code drops.

        Args:
            code: The code: a column, or a selector.
            known: The columns of the receiver.
            column: Whether the code is one column.

        Returns:
            This expansion, or a refusal with its reason.
        """
        argument = DeferredArgument("row")
        if column:
            name = argument.answer(code, known.columns).name
            self.names.append(not_none(name, reason=Reason.DYNAMIC_NAME))
            self.required.append(self.names[-1])
            return self
        selector = not_none(argument.selector(code))
        not_falsy(not HoleValues().held(selector), reason=Reason.DYNAMIC_NAME)
        picked = not_none(argument.picked(code, known.columns))
        not_falsy(not picked.partial, reason=Reason.PARTIAL_PICK)
        self.names.extend(picked.names)
        self.required.extend(argument.missing(selector, known.columns))
        if known.rest is not None:
            record = not_none(RemainderKnowledge().record(known.rest))
            effect = argument.effect(code, known.columns, record)
            read = effect if isinstance(effect, RemainderEffect) else None
            self.selectors.append(not_none(read).selector)
        return self


class DropHook:
    """Give the type of a frame after `.drop()`, without the dropped columns.

    The hook works on an open frame too. The declared columns lose each
    dropped name. The remainder loses each dropped name as well, and the
    hook spells that as `Minus[...]` around it. This matters for a declared
    name as much as for an undeclared one. After the drop the name is free,
    so a later `.with_row_index(name)` or `.rename({other: name})` is valid,
    and the remainder still proves that no column of the caller holds the
    name.

    With the default `strict=True`, polars raises `ColumnNotFoundError` for a
    name that the frame does not hold. If the frame does not declare a name
    and its remainder cannot hold it, the hook reports an error. If the
    remainder can hold it, the result is unproven, and the `allow_unproven`
    setting decides whether the hook reports it. When the setting lets it
    pass, the hook types the result as if the name were dropped. With
    `strict=False`, polars ignores a missing name, so the call always
    succeeds.

    The hook sits in the plugin layer. It reads the receiver through
    `Frame`, checks the names through `RequestedColumns`, and edits the
    remainder through `RemainderEdit`. If it cannot read the call, it falls
    back: it gives the plain frame type in place of a typed result.
    """

    def __init__(self, untyped_fallback: UntypedFallback, plugin: Plugin) -> None:
        """Keep the fallback and the plugin that the hook needs.

        Args:
            untyped_fallback: The fallback that gives the plain frame type when
                the hook cannot read the call, subject to the `allow_unproven`
                setting.
            plugin: The mypy plugin, which mints the new remainder.
        """
        self._requested_columns = RequestedColumns(untyped_fallback, "drop")
        self._untyped = untyped_fallback
        self._plugin = plugin

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the result type of one `.drop()` call.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The frame without the dropped columns. If the hook cannot read the
            call, the plain `pl.DataFrame` or `pl.LazyFrame`. See
            `UntypedFallback`.
        """
        narrowed = self._narrowed(ctx)
        if isinstance(narrowed, Refusal):
            return self._untyped.resolve(
                ctx,
                narrowed.reason,
                "drop",
                Frame.from_proper_type(ctx.type),
            )
        return narrowed

    @explains(Reason.CHECKER_GAP)
    def _narrowed(self, ctx: MethodContext) -> MypyType:
        """Build the result frame, with each dropped name out of the remainder as well.

        If `strict` is not a literal, the hook types the result only when the frame
        declares every requested name. Both modes then give the same result, and
        neither mode can raise, so the hook has nothing to report.

        Under `strict=True`, the hook checks each requested name against the
        receiver. A name passes if the frame declares it, or if the remainder
        can hold it and the setting lets it pass. The check reports an error
        for the first name that fails. Under `strict=False`, polars ignores a
        missing name, so the hook checks nothing.

        Args:
            ctx: The mypy context of the method call.

        Returns:
            The result frame, or a refusal with its reason. The reason is the
            plan's own reason if the call cannot be read. It is a checker gap if
            `strict` is not a literal and a name is not declared. It is
            `Reason.REPORTED` if a name failed under `strict=True` and the hook
            already reported the error.
        """
        plan = not_refused(_DropPlan.from_context(ctx))
        expanded = not_refused(_Expanded.of(ctx, plan))
        undeclared = set(plan.requested) - set(plan.known.by_name())
        not_falsy(plan.strict is not None or not undeclared)
        checked = [*(plan.requested if plan.strict else []), *expanded.required]
        if checked:
            resolved = self._requested_columns.resolved(ctx, checked, plan.known)
            if isinstance(resolved, Refusal):
                may_hold = resolved.reason is Reason.REMAINDER_MAY_HOLD
                not_falsy(may_hold, reason=Reason.REPORTED)
        dropped = [*plan.requested, *expanded.names]
        editor = RemainderEdit.for_checker(self._plugin, ctx.api)
        rest = editor.minus(plan.known.rest, dropped)
        for selector in expanded.selectors:
            rest = editor.excepted(rest, selector)
        remaining = [column for column in plan.known.columns if column.name not in dropped]
        return plan.frame.rebuilt(remaining, ctx.api, rest=rest).as_instance()
