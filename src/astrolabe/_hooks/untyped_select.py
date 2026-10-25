"""Type `.select()` on a plain frame, whose columns the checker does not know.

A plain frame is a `pl.DataFrame` or `pl.LazyFrame` that the checker does
not track, for example the result of `pl.read_csv(path)` with no schema.
Its data types are unknown, but a `.select()` call on it names each output
column. This hook turns that call into a typed frame, so that the checker
tracks the columns from this point on.

The output names come from the arguments. A bare column name gives a
column of data type `Unchecked`, which means that its name is known and
its data type is not. An expression is evaluated against a stand-in frame
that declares each column the expression reads as `Unchecked`. The rules
for expressions read `Unchecked` as "any data type". So an expression
whose answer is the same for every input, such as
`pl.col("a").is_null()`, still gives `Boolean`. An answer that depends on
the unknown input stays `Unchecked`. A selector, from `polars.selectors`,
needs real columns to match, so a call with one stays plain.

The `elevate_untyped_select` setting turns this hook on or off. It is on
by default.
"""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

import polars as pl
from mypy.types import get_proper_type, Instance

from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.deferred.argument import DeferredArgument
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.domain.frame import Frame, KnownColumns
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.expr_evaluation import ExpressionArgument
from astrolabe._hooks.expression_requests import (
    ExpressionRequest,
    RequestPolicies,
    RequestReader,
    RequestResolver,
    SelectionArgument,
    Unpicked,
)
from astrolabe._hooks.propagate import not_falsy, not_none, not_refused, propagates
from astrolabe.frame import TypedDataFrame, TypedLazyFrame

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback


class UntypedSelectHook:
    """Mint a typed frame from `.select()` on a plain frame, with the output names of the call.

    The receiver has no declared columns, so the hook cannot check a name
    against them. It builds each output column from the argument alone,
    and gives the result the typed kind of the receiver: `TypedDataFrame`
    for `pl.DataFrame`, and `TypedLazyFrame` for `pl.LazyFrame`. Each
    column read from the receiver has the data type `Unchecked`, and
    `UncheckedFallback` reports it unless the `allow_unproven` setting
    lets it pass.

    It is a mypy method hook in the plugin layer. It gives each expression
    to `RequestResolver`, the same code that types `.select()` on a typed
    frame. If an argument has a shape that the hook cannot read, or two
    outputs share a name, the call keeps the plain frame.
    """

    _TYPED_KIND_TARGET: ClassVar[dict[Target, Target]] = {
        Target.of(pl.DataFrame): Target.of(TypedDataFrame),
        Target.of(pl.LazyFrame): Target.of(TypedLazyFrame),
    }

    def __init__(self, plugin: Plugin, unchecked_fallback: UncheckedFallback) -> None:
        """Keep the tools that the hook needs for each call.

        Args:
            plugin: The mypy plugin, which finds the typed frame classes.
            unchecked_fallback: Gives the data type `Unchecked` to each
                column read from the receiver, and reports it unless the
                setting lets it pass. A plain receiver gives no data type
                for any column.
        """
        self._plugin = plugin
        self._unchecked_fallback = unchecked_fallback
        self._policies = RequestPolicies(
            "select",
            "row",
            plugin,
            unchecked_fallback,
            unpicked=Unpicked.ABSENT,
        )
        self._literals = BareLiteralReader()
        self._duplicate_check = DuplicateNameCheck("select")

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.select()` call on a plain frame.

        Args:
            ctx: The context that mypy gives for `.select()`.

        Returns:
            A new typed frame of the same kind as the receiver, if the
            hook can read every argument and no two outputs share a name.
            Else the plain frame that polars declares.
        """
        minted = self._minted(ctx)
        if minted is None:
            return ctx.default_return_type
        return minted

    @propagates(None)
    def _minted(self, ctx: MethodContext) -> MypyType | None:
        """Mint the result frame, step by step, or give None if a step fails.

        The typed frame class matches the kind of the receiver:
        `TypedDataFrame` for a `pl.DataFrame` receiver, and
        `TypedLazyFrame` for a `pl.LazyFrame` receiver. If two outputs share
        a name, the method reports the duplicate.

        Args:
            ctx: The context that mypy gives for `.select()`.

        Returns:
            The typed frame, or None. None comes if the receiver is not a
            plain frame, if an argument has a shape that the hook cannot
            read, if two outputs share a name, or if mypy cannot find the
            frame class.
        """
        proper = get_proper_type(ctx.type)
        if not isinstance(proper, Instance):  # pragma: no cover
            # The plugin registers this hook only on `pl.DataFrame.select`
            # and `pl.LazyFrame.select`, so the receiver is always an
            # instance of one of them.
            return None
        target = not_none(self._TYPED_KIND_TARGET.get(Target(proper.type.fullname)))
        requests = not_none(RequestReader("exprs", "named_exprs").of(ctx))
        columns = [not_none(self._resolved_column(ctx, request)) for request in requests]
        names = [column.name for column in columns]
        not_falsy(not self._duplicate_check.fails(ctx, names))
        frame = not_none(Frame.build_fresh(target, columns, self._plugin, ctx.api))
        return frame.as_instance()

    def _resolved_column(self, ctx: MethodContext, request: ExpressionRequest) -> ColumnRef | None:
        """Turn one argument into its output column.

        A bare string names a column, which gets the data type
        `Unchecked`. In `select(total="price")`, the keyword names the
        output, so the keyword wins over the column name.

        An expression is evaluated against a stand-in frame with no
        remainder. The typed expression lists the names of the columns it
        reads, and the stand-in frame declares each of them as `Unchecked`,
        sorted by name so that the result is the same on every run. The
        expression rules can then evaluate the expression without a real
        schema. An argument that is not a typed expression gets a stand-in
        frame with no columns.

        A selection, that is a typed selector such as `cs.numeric()`,
        needs real columns to match, so it stops the whole call.

        Args:
            ctx: The context that mypy gives for `.select()`.
            request: One argument of the call.

        Returns:
            The output column, or None if the argument is a selection or
            has a shape that the hook cannot read.
        """
        if SelectionArgument().matches(request.arg_type):
            return None
        name = self._literals.string(request.node, request.arg_type)
        if name is not None:
            forced = request.forced_name
            return self._unchecked_column(ctx, forced if forced is not None else name)
        stand_in = [self._unchecked_column(ctx, read) for read in self._reads(request)]
        known = KnownColumns(stand_in, None)
        return not_refused(RequestResolver(ctx, known, self._policies).resolved(request))

    def _reads(self, request: ExpressionRequest) -> list[str]:
        """Give the names of the columns that an argument reads, sorted.

        Args:
            request: One argument of the call.

        Returns:
            The names. No name for an argument that is not an expression.
        """
        deferred = DeferredArgument("row").reads(request.arg_type)
        if deferred is not None:
            return deferred
        expression = ExpressionArgument().decoded(request.arg_type)
        return [] if expression is None else sorted(expression.needs)

    def _unchecked_column(self, ctx: MethodContext, name: str) -> ColumnRef:
        """Build one `Unchecked` column, and report it unless the setting lets it pass.

        The reason is `NO_SCHEMA`: the receiver gives no data type.

        Args:
            ctx: The context that mypy gives for `.select()`.
            name: The name of the column.

        Returns:
            The column.
        """
        # `Unchecked` is declared in `astrolabe.frame`, and the plugin
        # makes mypy load that module in every run. So the lookup never
        # fails.
        dtype = checked_not_none(
            self._unchecked_fallback.substituted(ctx, Reason.NO_SCHEMA, "select", name),
        )
        return ColumnRef.from_name_and_dtype(name, dtype)
