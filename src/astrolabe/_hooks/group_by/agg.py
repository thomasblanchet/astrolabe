"""Type `group_by(...).agg(...)`: the keys, then one column for each aggregation.

Polars reads each argument of `.agg` once for each group. An expression that works row by row
gives one list for each group, so its column becomes a `List` of its data type. An expression
that combines rows into one value keeps its data type, and a literal stays one value. A column
name written as a string is `pl.col(name)`, so it becomes a `List` too. A keyword argument
always names its output column, even over an `.alias(...)` inside it. The result has the keys
first, then each aggregation in call order.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING


from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.domain.frame import KnownColumns
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.remainder_edit import RemainderEdit
from astrolabe._hooks.group_by.receiver import GroupByReceiver
from astrolabe._hooks.group_by.result import GroupByResult
from astrolabe._hooks.group_by.window_columns import WindowColumns
from astrolabe._hooks.expression_requests import (
    MemberSplit,
    RequestPolicies,
    RequestReader,
    RequestResolver,
    Resolution,
    Unpicked,
)
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.expression_requests import ExpressionRequest
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


@dataclass(frozen=True)
class _PreparedAgg:
    """Everything that one `.agg` call needs to build its result frame.

    Attributes:
        key_columns: The key columns.
        resolved: The output column of each aggregation, in call order.
        rest: The remainder of the result: the columns of the remainder that a selection such
            as `cs.numeric().sum()` keeps. None if no selection keeps any.
        receiver_fullname: The mypy full name of the group-by class.
    """

    key_columns: list[ColumnRef]
    resolved: list[ColumnRef]
    rest: MypyType | None
    receiver_fullname: str


class AggHook:
    """Type `.agg(...)` on a typed group-by as the typed frame that it gives.

    The hook reads each argument against the columns of the frame at the time of the
    `group_by`. It uses the same request resolver as `select`, in the aggregation context. A
    selection, such as `cs.numeric().sum()`, picks among the columns that are not keys, and it
    can also keep columns of the remainder. If the hook cannot type an argument, it falls back
    to the plain frame. If the result has two columns with the same name, it reports an error
    and falls back.
    """

    def __init__(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the plugin and the two fallbacks that the hook can need.

        Args:
            plugin: The mypy plugin, which finds the classes of the result.
            untyped_fallback: The gate that reports a call that the hook cannot type.
            unchecked_fallback: The gate that reports an aggregation whose output name is known
                and whose data type is not. The column then gets the data type `Unchecked`.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback
        self._policies = RequestPolicies(
            "agg",
            "agg",
            plugin,
            unchecked_fallback,
            unpicked=Unpicked.ABSENT,
        )

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.agg` call.

        Args:
            ctx: The mypy context of the `.agg` call.

        Returns:
            The result `TypedDataFrame` or `TypedLazyFrame`. If the hook cannot type the call,
            the mypy default, which is the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        requests = RequestReader("aggs", "named_aggs").of(ctx)
        return self.aggregated(ctx, requests, "agg")

    def aggregated(
        self,
        ctx: MethodContext,
        requests: Sequence[ExpressionRequest] | None,
        operation: str,
    ) -> MypyType:
        """Give the frame that a group-by gives for `agg` over some arguments.

        A short form of a group-by, such as `.sum()`, runs `agg` over an expression that the
        user does not write, so it gives the arguments itself. The result has the keys, then
        each aggregation. A `group_by()` with no key is valid, and the result then has the
        aggregations only. If two columns of the result, keys included, have the same name,
        the hook reports the duplicate and falls back.

        Args:
            ctx: The mypy context of the call.
            requests: The arguments of the `agg`, or None where an argument gives no keyword
                that the hook can read.
            operation: The name of the method that the error of a duplicate name gives, as
                `agg` or `len`.

        Returns:
            The result `TypedDataFrame` or `TypedLazyFrame`. If the hook cannot type the call,
            the mypy default, which is the plain `pl.DataFrame` or `pl.LazyFrame`.
        """
        split = MemberSplit()
        choices = split.choices(requests)
        if choices is not None:
            return split.joined([self.aggregated(ctx, choice, operation) for choice in choices])
        prepared = self._prepared(ctx, requests)
        if isinstance(prepared, Refusal):
            return self._untyped_fallback(ctx, prepared.reason)
        columns = prepared.key_columns + prepared.resolved
        if DuplicateNameCheck(operation).fails(ctx, [column.name for column in columns]):
            return self._untyped_fallback(ctx, Reason.REPORTED)
        frame = GroupByResult(self._plugin).frame(
            prepared.receiver_fullname,
            columns,
            prepared.rest,
            ctx.api,
        )
        if frame is None:  # pragma: no cover
            # The receiver is always one of the two group-by classes. See GroupByResult.
            return self._untyped_fallback(ctx, Reason.CHECKER_GAP)
        return frame

    def _untyped_fallback(self, ctx: MethodContext, reason: Reason) -> MypyType:
        """Fall back to the plain frame for a call that the hook cannot type.

        Args:
            ctx: The mypy context of the `.agg` call.
            reason: Why the hook cannot type the call.

        Returns:
            The mypy default result. The method first reports `ImplicitlyUntypedError` unless
            the `allow_unproven` setting lets the reason pass.
        """
        self._untyped.report_if_disallowed(ctx, reason, "agg")
        return ctx.default_return_type

    @explains(Reason.CHECKER_GAP)
    def _prepared(
        self,
        ctx: MethodContext,
        requests: Sequence[ExpressionRequest] | None,
    ) -> _PreparedAgg:
        """Read the receiver and type each argument of the call.

        A selection can pick every declared column but a key. Polars leaves the keys out of a
        selection in `.agg`, because each key is already a column of the result. A group-by
        over windows also leaves out its index column, and evaluates each argument over its
        windows (`WindowColumns`).

        Args:
            ctx: The mypy context of the `.agg` call.
            requests: The arguments of the `agg`, or None where an argument gives no keyword.

        Returns:
            The inputs of the result, or a refusal. The reason is `CHECKER_GAP` if the receiver
            is not a typed group-by, and `DYNAMIC_NAME` if an argument has no keyword that the
            hook can read. Otherwise it is the reason of the first argument that the hook
            cannot type.
        """
        receiver_columns = not_none(GroupByReceiver.from_receiver_type(ctx.type))
        arguments = not_none(requests, reason=Reason.DYNAMIC_NAME)
        leading = receiver_columns.key_columns + receiver_columns.index_columns
        keys = {column.name for column in leading}
        selectable: list[ColumnRef] = []
        for column in receiver_columns.original_columns:
            if column.name not in keys:
                selectable.append(column)
        windows = WindowColumns(self._plugin)
        resolver = RequestResolver(
            ctx,
            KnownColumns(receiver_columns.original_columns, receiver_columns.rest),
            replace(self._policies, windows=windows.kind(receiver_columns)),
            selectable,
        )
        resolutions = [not_refused(resolver.resolved_all(request)) for request in arguments]
        return _PreparedAgg(
            windows.leading(receiver_columns, ctx.api),
            Resolution.columns_of(resolutions),
            self._kept(ctx, receiver_columns.rest, resolutions),
            receiver_columns.receiver_fullname,
        )

    def _kept(
        self,
        ctx: MethodContext,
        rest: MypyType | None,
        resolutions: list[Resolution],
    ) -> MypyType | None:
        """Give the part of the remainder that the selections of the call keep.

        A selection also picks columns of the remainder, and the result keeps them. Any other
        aggregation keeps no column of the remainder. The kept parts of all the selections join
        into one remainder.

        Args:
            ctx: The mypy context of the `.agg` call.
            rest: The remainder of the receiver, or None.
            resolutions: The result of each argument.

        Returns:
            The kept remainder, or None if no selection keeps any.
        """
        edit = RemainderEdit.for_checker(self._plugin, ctx.api)
        kept: MypyType | None = None
        for resolution in resolutions:
            effect = resolution.effect
            selected = None if effect is None else edit.selected(rest, effect)
            kept = edit.both(kept, selected)
        return kept
