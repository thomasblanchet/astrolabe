"""Type the short forms on a group-by, such as `.sum()`, `.head(n)` and `.len()`.

Polars runs most short forms as `agg` over one expression. `group_by(...).sum()` runs
`group_by(...).agg(pl.all().sum())`, which applies one member to every column that is not a key.
`.all()` runs `.agg(pl.all())`, which keeps each column as a list for each group. `.len()` and
`.count()` run `.agg(pl.len())` under the name of the form. `.head(n)` and `.tail(n)` keep every
column, with the keys first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import CallExpr

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.errors import InvalidCallShapeError
from astrolabe._hooks.expression_requests import ExpressionRequest
from astrolabe._hooks.group_by.agg import AggHook
from astrolabe._hooks.group_by.receiver import GroupByReceiver
from astrolabe._hooks.group_by.result import GroupByResult
from astrolabe._hooks.recording.recorder import RecordedShortForm, ShortForm
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.row_counts import RowCountCheck

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class GroupByShorthandHook:
    """Type one short form on a typed group-by, such as `.sum()`, as the frame that it gives.

    There is one instance of the hook for each short form. A form that polars runs as `agg`
    records its expression with the arguments of the call, and the hook types the call as that
    `agg`. So the rules of each member apply to each column, as in `.agg(pl.all().sum())`, and
    the errors are the errors of `agg`. A keeping form keeps the columns. If the hook cannot
    type a call, it falls back to the plain frame.
    """

    # The expression that each form runs in `agg`. `len` with a name runs it under that name.
    RUNS: ClassVar[dict[str, ShortForm]] = {
        "all": ShortForm("pl.all()"),
        "first": ShortForm("pl.all().first()", ("ignore_nulls",)),
        "last": ShortForm("pl.all().last()", ("ignore_nulls",)),
        "max": ShortForm("pl.all().max()"),
        "mean": ShortForm("pl.all().mean()"),
        "median": ShortForm("pl.all().median()"),
        "min": ShortForm("pl.all().min()"),
        "n_unique": ShortForm("pl.all().n_unique()"),
        "quantile": ShortForm("pl.all().quantile()", ("quantile", "interpolation")),
        "sum": ShortForm("pl.all().sum()"),
        "count": ShortForm("pl.len().alias('count')"),
        "len": ShortForm("pl.len().alias()", ("name",)),
    }
    UNNAMED: ClassVar[ShortForm] = ShortForm("pl.len()")

    # The forms that keep every column of the frame, with the keys first.
    KEEPING: ClassVar[frozenset[str]] = frozenset({"head", "tail"})

    def __init__(
        self,
        plugin: Plugin,
        member: str,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the plugin, the name of the short form, and the hook of `agg`.

        Args:
            plugin: The mypy plugin, which finds the classes of the result.
            member: The name of the short form, for example `"sum"`.
            untyped_fallback: The gate that reports a call that the hook cannot type.
            unchecked_fallback: The gate that reports a column whose name is known and whose
                data type is not.
        """
        self._plugin = plugin
        self._member = member
        self._untyped = untyped_fallback
        self._agg = AggHook(plugin, untyped_fallback, unchecked_fallback)

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one short form call.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The result frame. If the hook cannot type the call, the plain frame, after it
            reports `ImplicitlyUntypedError` unless the setting lets the reason pass.
        """
        if self._member in self.KEEPING:
            if RowCountCheck.of_receiver(ctx.type, self._member).reported(ctx):
                return ctx.default_return_type
            return self._kept(ctx)
        typ = RecordedShortForm(self._plugin, self._form(ctx)).recorded(ctx)
        if typ is None:
            # Polars refuses to build the expression, and the recorder reported it.
            return ctx.default_return_type
        request = ExpressionRequest(checked_cast(ctx.context, CallExpr), typ, None)
        return self._agg.aggregated(ctx, [request], self._member)

    def _form(self, ctx: MethodContext) -> ShortForm:
        """Give the expression that one call runs in `agg`.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The short form. `len` with no name runs `pl.len()`, which polars names `len`.
        """
        if self._member == "len" and CallArgs(ctx).is_omitted_or_none("name"):
            return self.UNNAMED
        return self.RUNS[self._member]

    def _kept(self, ctx: MethodContext) -> MypyType:
        """Give every column of the frame, with the keys first, for `.head` and `.tail`.

        Over windows of the rows, polars refuses both: it explodes the index
        column, which holds one value for each window.

        Args:
            ctx: The mypy context of the call.

        Returns:
            The result frame, or the plain frame where the receiver does not read.
        """
        receiver = GroupByReceiver.from_receiver_type(ctx.type)
        if receiver is None:  # pragma: no cover
            # The plugin registers the hook only on the typed group-by classes.
            self._untyped.report_if_disallowed(ctx, Reason.CHECKER_GAP, "agg")
            return ctx.default_return_type
        if receiver.windows:
            reason = "polars explodes the index, which holds one value for each window"
            InvalidCallShapeError(f"{self._member}() raises InvalidOperationError: {reason}").fail(
                ctx
            )
            return ctx.default_return_type
        keys = {column.name for column in receiver.key_columns}
        others = [column for column in receiver.original_columns if column.name not in keys]
        frame = GroupByResult(self._plugin).frame(
            receiver.receiver_fullname,
            [*receiver.key_columns, *others],
            receiver.rest,
            ctx.api,
        )
        if frame is None:  # pragma: no cover
            # The receiver is always one of the two group-by classes. See GroupByResult.
            return ctx.default_return_type
        return frame
