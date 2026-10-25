"""Give a call such as `pl.Series("price", ...)` a type that keeps its name and data type."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.unproven import Reason
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.domain.dtype_class_guard import DtypeClassGuard
from astrolabe._hooks.domain.series import SeriesRef
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext
    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from astrolabe._hooks.domain.frame import PluginLookup


class SeriesConstructorHook:
    """Give `pl.Series(name, values, dtype=...)` the type `TypedSeries[Literal[name], dtype]`.

    polars types every `pl.Series(...)` call as a plain `Series`. The
    checker then knows neither the name nor the data type, so it cannot
    check the series where a frame receives it as a column, for example
    in `.hstack(...)`. This hook replaces the return type with a
    `TypedSeries` that holds both. The type travels with the value through
    a variable, like any other type.

    The name must be a literal string, or a string variable declared
    `Final`. Else the hook keeps the plain `Series`. The data type comes
    only from `dtype=`, read by `DtypeExpr` in the same way as the
    argument of `.cast(...)`. The hook does not infer a data type from the
    values. If `dtype=` is missing or is not a literal data type, the
    series gets the data type `Unchecked`, which means that its name is
    known and its data type is not. `UncheckedFallback` reports that
    unless the `allow_unproven` setting lets it pass.

    It is a mypy function hook in the plugin layer, and it mints the
    series type through `SeriesRef`.
    """

    def __init__(self, plugin: PluginLookup, unchecked_fallback: UncheckedFallback) -> None:
        """Keep the plugin and the fallback for a missing data type.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedSeries`.
            unchecked_fallback: Gives the data type `Unchecked` to a series
                whose data type the hook cannot read, and reports it
                unless the setting lets it pass.
        """
        self._plugin = plugin
        self._unchecked_fallback = unchecked_fallback
        self._literal_type = CheckerLiteralTypeBuilder()
        self._dtype_expr = DtypeExpr()

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one `pl.Series(...)` call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            A new `TypedSeries[Literal[name], dtype]`. If the name is not
            a literal string or a `Final` string variable, or mypy cannot
            find a class that the type needs, the plain `Series`.
        """
        result = self._result(ctx)
        if result is None:
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: FunctionContext) -> MypyType | None:
        """Build the `TypedSeries` from the name and the data type of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The series type, or None if a step fails.
        """
        name = not_none(CallArgs(ctx).literal_str("name"))
        name_type = self._literal_type.string(name, ctx.api)
        dtype = not_none(self._dtype(ctx, name))
        return not_none(SeriesRef.build(name_type, dtype, self._plugin))

    def _dtype(self, ctx: FunctionContext, name: str) -> MypyType | None:
        """Read the data type from `dtype=`, or give `Unchecked` if the hook cannot.

        A class with parameters, such as `pl.Datetime`, gives the instance
        that polars builds for it, and `pl.Decimal` gives `Decimal(38, 0)`.
        A marker class of astrolabe that polars cannot read here, such as
        `List[pl.Int64]` or the bare class `List`, is reported, and the
        series gets `Unchecked`. So is a subclass of a polars data type that
        user code defines. If that class subclasses `pl.List`, `pl.Array` or
        `pl.Struct`, polars ignores it and the values decide the data type,
        so the series gets `Unchecked` as if `dtype=` were missing. An
        argument that is no data type class at all, which mypy reports,
        counts as a `dtype=` that is not a literal data type.
        The two cases of a missing data type have different reasons. A missing `dtype=` is
        `Reason.SERIES_NO_DTYPE`. A `dtype=` that is not a literal data
        type is `Reason.NON_LITERAL_DTYPE`. The fallback names the reason
        in its error, and the setting can let one pass and not the other.

        Args:
            ctx: The context that mypy gives for the call.
            name: The literal name of the series, for the error message.

        Returns:
            The data type from `dtype=`, or a new `Unchecked`. None if
            mypy cannot find the class `Unchecked`.
        """
        nodes = CallArgs(ctx).node_list("dtype")
        if not nodes:
            return self._unchecked_fallback.substituted(ctx, Reason.SERIES_NO_DTYPE, "Series", name)
        if DtypeClassGuard().refused_in_series(ctx, nodes[0]):
            return self._unchecked_fallback.substituted(ctx, Reason.REPORTED, "Series", name)
        if self._dtype_expr.nested_user_class(ctx.api, nodes[0]):
            return self._unchecked_fallback.substituted(ctx, Reason.SERIES_NO_DTYPE, "Series", name)
        minter = MarkerMinter(self._plugin, self._literal_type, ctx.api)
        resolved = self._dtype_expr.schema_typed(ctx.api, nodes[0], ConstDtypeCodec(minter))
        if resolved is not None and self._dtype_expr.names_data_type(ctx.api, nodes[0]):
            return resolved
        return self._unchecked_fallback.substituted(ctx, Reason.NON_LITERAL_DTYPE, "Series", name)
