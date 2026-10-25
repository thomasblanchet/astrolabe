"""Type `.cast()` and `.round()` on a `TypedSeries`, and report a call that polars rejects.

A `TypedSeries` is a series whose type holds its name and its data type.
Most of its methods keep both, and they are declared to return the type of
the receiver. `.cast()` changes the data type, and `.round()` fails on a data
type that is not a number. These two hooks give the right type, and they
report the calls that fail when the program runs.
"""

from __future__ import annotations

from typing import Final, TYPE_CHECKING

from mypy.types import Instance

from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.cast_compatibility import CastCompatibility
from astrolabe._hooks.domain.dtype_class_guard import DtypeClassGuard
from astrolabe._hooks.domain.dtype_families import DtypeFamilies
from astrolabe._hooks.domain.series import SeriesRef
from astrolabe._hooks.domain.unchecked import UncheckedCompatibility
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec, ConstDtypeReads
from astrolabe._hooks.errors import UncastableDtypeError, WrongColumnShapeError
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.propagate import not_falsy, propagates
from astrolabe._hooks.unproven import Reason
from astrolabe.frame import Unchecked as _UncheckedMarker

if TYPE_CHECKING:
    from mypy.plugin import MethodContext, Plugin

    from astrolabe._hooks.domain.unchecked import UncheckedFallback
    from mypy.types import Type as MypyType


class _UncheckedSeries:
    """Mint `TypedSeries[name, Unchecked]` for a call that the hook reported as an error.

    `Unchecked` is the data type of a column whose name is known and whose
    data type is not. After a failed `.cast()` or `.round()`, the checker
    has reported the error, and it cannot state a data type for the
    result. Neither method renames the series, so the name stays known.
    This result keeps the name and marks the data type as unknown, so that
    later code does not get a second error for the same cause.

    Example:
        >>> fixture = new_type_fixture()
        >>> name = literal_str_type(fixture, "a")
        >>> _UncheckedSeries(new_plugin(fixture)).built(name)
        astrolabe.frame.TypedSeries[Literal['a'], astrolabe.frame.Unchecked]
    """

    _UNCHECKED_TARGET: Final[Target] = Target.of(_UncheckedMarker)

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, to look up the classes of the result.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `Unchecked` and of `TypedSeries`.
        """
        self._plugin = plugin
        self._type_info = FreshTypeInfo(plugin)

    def built(self, name: MypyType) -> MypyType | None:
        """Build `TypedSeries[name, Unchecked]`.

        Args:
            name: The literal name type of the receiver, kept unchanged.

        Returns:
            The series type, or None if mypy cannot find `Unchecked` or
            `TypedSeries`.

        Example: see class docstring.
        """
        info = self._type_info.resolved(self._UNCHECKED_TARGET.fullname)
        if info is None:  # pragma: no cover
            # `Unchecked` is declared in the same module as `TypedSeries`.
            # The hook runs only on a `TypedSeries`, so mypy has already
            # loaded that module.
            return None
        return SeriesRef.build(name, Instance(info, []), self._plugin)


class CastHook:
    """Give `.cast(dtype)` on a `TypedSeries` the new data type, or report a cast that always fails.

    polars declares that `.cast()` returns the type of its receiver, which
    for a `TypedSeries` keeps the old data type. That is wrong for a cast,
    so this hook replaces it with `TypedSeries[name, dtype]`, with the same
    name and the new data type.

    `CastCompatibility` decides the result from the source and target data
    types. The result is the target, or the data type that polars gives in
    its place: a `Struct` cast to `Float64` casts each field and stays a
    `Struct`. It is the same rule that the checker applies to `.cast()` on an
    expression, so a series and an expression always agree. If the cast
    fails for all data, the hook reports an `UncastableDtypeError`. If the
    cast depends on the data, for example `String` to `Int64`, the hook
    assumes that it succeeds. The arguments `strict=` and
    `wrap_numerical=` do not change the target data type, so the hook does
    not read them. It is a mypy method hook in the plugin layer.
    """

    def __init__(self, plugin: Plugin, unchecked_fallback: UncheckedFallback) -> None:
        """Build the readers and the cast rule.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of
                `TypedSeries` and the classes that the cast rule needs.
            unchecked_fallback: The fallback that reports a target that the
                hook cannot read, unless the `allow_unproven` setting lets it
                pass, and gives the data type `Unchecked`.
        """
        self._plugin = plugin
        self._dtype_expr = DtypeExpr()
        self._compatibility = CastCompatibility()
        self._reads = ConstDtypeReads()
        self._unchecked = _UncheckedSeries(plugin)
        self._fallback = unchecked_fallback

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.cast(dtype)` call.

        Args:
            ctx: The context that mypy gives for `.cast()`.

        Returns:
            `TypedSeries[name, dtype]` if the cast can succeed, with the
            name of the receiver. `TypedSeries[name, Unchecked]` if the
            cast always fails, after the hook reports
            `UncastableDtypeError`, and also if the hook cannot read
            `dtype`, as a variable of the type `pl.DataType`. The declared
            type where the call passes no `dtype`, which mypy reports, and
            after a report of the class `pl.Decimal` or of a marker class
            such as `List[pl.Int64]`, which polars refuses.
        """
        nodes = CallArgs(ctx).node_list("dtype")
        if not nodes or DtypeClassGuard().refused(ctx, nodes[0]):
            return ctx.default_return_type
        target = self._resolved_dtype(ctx)
        if target is None:
            return self._unread_result(ctx)
        return self._checked_result(ctx, target)

    def _unread_result(self, ctx: MethodContext) -> MypyType:
        """Give the result of a cast to a data type that the hook cannot read.

        The cast gives the series the type that the target holds at run time,
        so the result keeps the name of the receiver under `Unchecked`. The
        fallback first reports it unless the `allow_unproven` setting lets
        it pass.

        Args:
            ctx: The context that mypy gives for `.cast()`.

        Returns:
            `TypedSeries[name, Unchecked]`, or the declared type where the
            plugin cannot build it.
        """
        receiver = SeriesRef.from_proper_type(ctx.type)
        if receiver is None:  # pragma: no cover -- the plugin runs this hook on a TypedSeries
            return ctx.default_return_type
        name = SeriesRef.name(receiver)
        text = ProperReads().string_literal(name)
        self._fallback.reported(ctx, Reason.NON_LITERAL_DTYPE, "Series", text or "")
        built = self._unchecked.built(name)
        return built if built is not None else ctx.default_return_type  # pragma: no cover

    def _checked_result(self, ctx: MethodContext, target: MypyType) -> MypyType:
        """Check the cast from the data type of the receiver to `target`, and build the result.

        If the cast always fails, the method reports `UncastableDtypeError`.
        The result then has the data type `Unchecked`.

        Args:
            ctx: The context that mypy gives for `.cast()`.
            target: The target data type, read from the call.

        Returns:
            The same answer as `__call__`.
        """
        receiver = SeriesRef.from_proper_type(ctx.type)
        if receiver is None:  # pragma: no cover
            # The plugin runs this hook only on a `TypedSeries` receiver.
            return ctx.default_return_type
        name = SeriesRef.name(receiver)
        source = SeriesRef.dtype(receiver)
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), ctx.api)
        resolved = self._compatibility.resolve(source, target, minter)
        if resolved is None:
            # The rule refuses a cast only where it read both data types.
            error = UncastableDtypeError.on_series(
                source=checked_not_none(self._reads.read_type(source)),
                target=checked_not_none(self._reads.read_type(target)),
            )
            error.fail(ctx)
            substituted = self._unchecked.built(name)
            return substituted if substituted is not None else ctx.default_return_type
        built = SeriesRef.build(name, resolved, self._plugin)
        return built if built is not None else ctx.default_return_type  # pragma: no cover

    @propagates(None)
    def _resolved_dtype(self, ctx: MethodContext) -> MypyType | None:
        """Read the target data type from the `dtype` argument.

        A data type can arrive as a class, for example `pl.Int64`, or as an
        instance, for example `pl.Int64()`. `DtypeExpr` reads both, and it
        reads a class with parameters, such as `pl.Datetime`, as the
        instance that polars builds for it.

        Args:
            ctx: The context that mypy gives for `.cast()`.

        Returns:
            The data type, or None if the call passes no `dtype`, or its
            type is neither a class nor an instance.
        """
        nodes = not_falsy(CallArgs(ctx).node_list("dtype"))
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), ctx.api)
        return self._dtype_expr.typed(ctx.api, nodes[0], ConstDtypeCodec(minter))


class RoundHook:
    """Report `.round()` on a `TypedSeries` whose data type is not a number.

    `.round()` keeps the data type of a numeric or `Decimal` series. On any
    other series, polars raises `InvalidOperationError` when the program
    runs. The hook reads the data type family of the receiver through
    `DtypeFamilies`. A numeric, `Decimal` or `Unchecked` series keeps its
    type. Any other series gets a `WrongColumnShapeError` and the result
    `TypedSeries[name, Unchecked]`. It is a mypy method hook in the plugin
    layer.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Build the data type families and the fallback result.

        Args:
            plugin: The mypy plugin, which finds the classes that the
                family test and the fallback result need.
        """
        self._families = DtypeFamilies(plugin)
        self._unchecked = _UncheckedSeries(plugin)
        self._marker = UncheckedCompatibility()

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.round(decimals)` call.

        An `Unchecked` receiver passes, because the unknown data type can be
        a number. The result then keeps `Unchecked`.

        Args:
            ctx: The context that mypy gives for `.round()`.

        Returns:
            The declared type, which is the type of the receiver, for a
            numeric, `Decimal` or `Unchecked` receiver. Else
            `TypedSeries[name, Unchecked]`, after the hook reports
            `WrongColumnShapeError`.
        """
        receiver = SeriesRef.from_proper_type(ctx.type)
        if receiver is None:  # pragma: no cover
            # The plugin runs this hook only on a `TypedSeries` receiver.
            return ctx.default_return_type
        dtype = SeriesRef.dtype(receiver)
        if self._families.is_numeric_or_decimal(dtype) or self._marker.matches(dtype):
            return ctx.default_return_type
        error = WrongColumnShapeError.on_series(expected="numeric or Decimal", operation=".round()")
        error.fail(ctx)
        substituted = self._unchecked.built(SeriesRef.name(receiver))
        return substituted if substituted is not None else ctx.default_return_type


class SeriesRenameHook:
    """Give `.alias(name)` and `.rename(name)` on a `TypedSeries` the new name.

    polars gives the series the new name and keeps its data type. The type
    of the receiver holds the old name, so without this hook a later
    operation, for example `hstack`, sees the old name and reports a
    duplicate that is not there. The hook gives `TypedSeries[name, dtype]`,
    with the new name as a literal string. A name that is not a literal
    becomes `str`, so the data type is kept and the name is unknown. It is
    a mypy method hook in the plugin layer.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, which finds the `TypeInfo` of `TypedSeries`.

        Args:
            plugin: The mypy plugin.
        """
        self._plugin = plugin

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one `.alias(name)` or `.rename(name)` call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            `TypedSeries[name, dtype]`, or the declared type if the hook
            cannot read the receiver. The name is a string literal type for
            a literal name, and the type that mypy gives the argument
            otherwise.
        """
        receiver = SeriesRef.from_proper_type(ctx.type)
        if receiver is None:  # pragma: no cover
            # The plugin runs this hook only on a `TypedSeries` receiver.
            return ctx.default_return_type
        args = CallArgs(ctx)
        literal = args.literal_str("name")
        if literal is None:
            name = checked_not_none(args.node_type("name"))
        else:
            name = CheckerLiteralTypeBuilder().string(literal, ctx.api)
        built = SeriesRef.build(name, SeriesRef.dtype(receiver), self._plugin)
        return built or ctx.default_return_type
