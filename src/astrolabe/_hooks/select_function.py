"""Type `pl.select(...)`, which builds a new frame from expressions with no input frame.

`pl.select(pl.lit(1).alias("one"))` runs its expressions against an empty
frame. It takes the same arguments as the frame method `.select()`. Each
argument must be a literal or an expression that reads no column, because
there is no column to read. With `eager=False`, the call gives a lazy frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.expression_requests import (
    RequestPolicies,
    RequestReader,
    RequestResolver,
    Resolution,
    Unpicked,
)
from astrolabe._hooks.propagate import Refusal, explains, not_none, not_refused
from astrolabe._hooks.unproven import Reason
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.untyped_fallback import UntypedFallback


class SelectFunctionHook:
    """Give `pl.select(...)` a typed frame with one column for each argument.

    Without this hook, polars types the call as a plain frame, and the
    checker knows none of its columns. The hook types each argument the
    way `.select()` does on a frame, but against a receiver with no
    columns. So an expression that reads a column gets an error, the same
    error as a read of an unknown column on a frame.

    It is a mypy function hook in the plugin layer. It gives the arguments
    to `RequestResolver`, which turns each one into its output columns. It
    checks the output names with `DuplicateNameCheck`, and mints the frame
    through `Frame`. If an argument cannot be typed, the hook falls back to
    the plain frame that polars declares, through `UntypedFallback`. To
    fall back means to give that plain frame in place of a typed result.
    """

    def __init__(
        self,
        plugin: Plugin,
        untyped_fallback: UntypedFallback,
        unchecked_fallback: UncheckedFallback,
    ) -> None:
        """Keep the tools that the hook needs for each call.

        Args:
            plugin: The mypy plugin, which finds the frame classes.
            untyped_fallback: Reports a call that the hook cannot type,
                unless the `allow_unproven` setting lets it pass.
            unchecked_fallback: Gives the data type `Unchecked` to a column
                whose name is known and whose data type is not, and reports
                it unless the setting lets it pass.
        """
        self._plugin = plugin
        self._untyped = untyped_fallback
        self._policies = RequestPolicies(
            "select",
            "row",
            plugin,
            unchecked_fallback,
            unpicked=Unpicked.ABSENT,
        )
        self._duplicate_check = DuplicateNameCheck("select")

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one `pl.select(...)` call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A typed frame with the output columns, or the plain frame that
            polars declares if an argument cannot be typed. The frame is a
            `TypedLazyFrame` if the call passes `eager=False`, else a
            `TypedDataFrame`.
        """
        columns = self._columns(ctx)
        if isinstance(columns, Refusal):
            self._untyped.report_if_disallowed(ctx, columns.reason, "select")
            return ctx.default_return_type
        eager = CallArgs(ctx).literal_bool("eager", default=True)
        target = Target.of(TypedLazyFrame) if eager is False else Target.of(TypedDataFrame)
        built = Frame.build_fresh(target, columns, self._plugin, ctx.api)
        if built is None:  # pragma: no cover
            # The frame classes are in `astrolabe.frame`, and mypy loads that
            # module in every run with the plugin.
            return ctx.default_return_type
        return built.as_instance()

    @explains(Reason.CHECKER_GAP)
    def _columns(self, ctx: FunctionContext) -> list[ColumnRef]:
        """Turn each argument into its output columns, against an empty receiver.

        A duplicate output name is an error that this step reports itself.
        It then refuses with `Reason.REPORTED`, so that the fallback adds
        no second error.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The output columns in the order of the arguments, or a
            `Refusal` that says why an argument cannot be typed.
        """
        requests = not_none(
            RequestReader("exprs", "named_exprs").of(ctx),
            reason=Reason.DYNAMIC_NAME,
        )
        resolver = RequestResolver(ctx, KnownColumns([], None), self._policies)
        resolutions = [not_refused(resolver.resolved_all(request)) for request in requests]
        columns = Resolution.columns_of(resolutions)
        names = [column.name for column in columns]
        not_none(
            None if self._duplicate_check.fails(ctx, names) else columns,
            reason=Reason.REPORTED,
        )
        return columns
