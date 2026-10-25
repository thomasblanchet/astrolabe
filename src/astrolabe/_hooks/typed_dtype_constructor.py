"""Give a call such as `pl.Datetime("ms")` or `pl.Struct({...})` a type that keeps its parts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.nodes import CallExpr, ListExpr, TupleExpr
from mypy.types import Instance

from astrolabe._hooks.calls.bare_literal import BareLiteralReader
from astrolabe._hooks.calls.call_args import CallArgs
from astrolabe._hooks.calls.dtype_constructor_args import DtypeConstructorArgs
from astrolabe._hooks.calls.dtype_constructor_readers import CheckerLiteralTypeBuilder
from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.calls.dtype_mapping import DtypeMapping
from astrolabe._hooks.checked import checked_cast
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.domain.dtype_class_guard import DtypeClassGuard
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.errors import InvalidCallShapeError
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeCodec
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
from astrolabe.frame import Array as ArrayMarker
from astrolabe.frame import List as ListMarker

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import FunctionContext, Plugin
    from mypy.types import Type as MypyType

    from astrolabe._hooks.calls.dtype_constructor_readers import _ParametrizedDtypeReader


class DtypeConstructorHook:
    """Give a call to `pl.Datetime`, `pl.Duration`, `pl.Decimal` or `pl.Enum` a parametrized type.

    These four data types have parameters: a time unit and a time zone,
    a precision and a scale, or a list of categories. The parameters are
    part of the data type, because a column of `Datetime("ms")` and a
    column of `Datetime("us")` hold different data. polars types the call
    `pl.Datetime("ms")` only as `Datetime`, so the checker loses the
    parameters as soon as the call ends.

    This hook replaces that return type with a marker class of this
    project, for example `Datetime[Literal["ms"], None]`, whose type
    arguments hold the literal parameters. The parameters then travel
    with the value through a variable, like any other type. Code that
    later reads a data type, for example in `.cast(...)` or in a
    `schema=` argument, finds them in the type.

    It is a mypy function hook in the plugin layer, with one instance for
    each of the four classes. The injected reader knows the parameters of
    its class and turns the literal arguments into type arguments. If an
    argument is not a literal, the hook keeps the plain type of polars.
    """

    def __init__(self, reader: _ParametrizedDtypeReader, plugin: Plugin) -> None:
        """Keep the reader for one data type class, and a way to find its marker class.

        Args:
            reader: Reads the literal arguments of a call to one data type
                class, and gives the type arguments of its marker.
            plugin: The mypy plugin, which finds the `TypeInfo` of the
                marker class.
        """
        self._reader = reader
        self._type_info = FreshTypeInfo(plugin)

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one call to a data type class.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker with the parameters as type arguments. If an
            argument is not a literal, or mypy cannot find the marker
            class, the plain type that polars declares.
        """
        result = self._result(ctx)
        if result is None:
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: FunctionContext) -> MypyType | None:
        """Build the marker type from the literal arguments of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker type, or None if a step fails.
        """
        call = checked_cast(ctx.context, CallExpr)
        constructor_args = DtypeConstructorArgs(call, self._reader.PARAM_NAMES)
        type_args = not_none(self._reader(constructor_args, ctx.api))
        info = not_none(self._type_info.resolved(self._reader.MARKER_TARGET.fullname))
        return Instance(info, type_args)


class NestedDtypeConstructorHook:
    """Give a call such as `pl.List(pl.Int64)` or `pl.Array(pl.Int8, 2)` a type with its parts.

    A `List` has an element type, and an `Array` has an element type and a
    width. polars types the call `pl.List(pl.Int64)` only as `List`, so the
    checker would lose the element type as soon as the call ends. A cast to
    that list, or a `dtype=` of that list, would then give a bare `List`.

    This hook replaces that return type with the marker class of this
    project, `List[pl.Int64]` or `Array[pl.Int8, Literal[2]]`, which holds
    the parts as type arguments, the same form that a `Columns` field
    writes. Code that later reads a data type, for example in `.cast(...)`,
    then finds the whole data type.

    It is a mypy function hook in the plugin layer, with one instance for
    `pl.List` and one for `pl.Array`. If the element is not a data type that
    the hook can read, or a width of the shape is not a literal number, the
    hook keeps the plain type of polars. A shape of several widths nests an
    array for each width after the first. An element that is a class with parameters,
    such as `pl.Datetime`, is the instance that polars builds for it. polars
    raises for an element that is a marker class with its type arguments,
    such as `List[pl.Int64]`, so the hook reports it. polars also raises
    for `pl.Array` with no shape, so the hook reports that call too.
    """

    def __init__(self, plugin: Plugin, array: bool) -> None:
        """Keep the plugin, and whether this instance serves `pl.Array`.

        Args:
            plugin: The mypy plugin, which finds the `TypeInfo` of the
                marker class.
            array: True for `pl.Array`, and False for `pl.List`.
        """
        self._plugin = plugin
        self._type_info = FreshTypeInfo(plugin)
        self._array = array

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one call to `pl.List` or `pl.Array`.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker with the element, and the width of an array, as type
            arguments. If a part cannot be read, the plain type that polars
            declares.
        """
        result = self._result(ctx)
        if result is None:
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: FunctionContext) -> MypyType | None:
        """Build the marker type from the arguments of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker type, or None if a step fails.
        """
        args = CallArgs(ctx)
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), ctx.api)
        nodes = not_falsy(args.node_list("inner"))
        not_falsy(not DtypeClassGuard().refused_in_nested(ctx, nodes[0]))
        inner = not_none(DtypeExpr().typed(ctx.api, nodes[0], ConstDtypeCodec(minter)))
        if not self._array:
            info = not_none(self._type_info.resolved(Target.of(ListMarker).fullname))
            return Instance(info, [inner])
        if args.is_omitted_or_none("shape") and args.is_omitted_or_none("width"):
            InvalidCallShapeError.array_without_shape().fail(ctx)
            return None
        node = (args.node_list("shape") or args.node_list("width"))[0]
        widths = not_falsy(self._shape(node, ctx.api))
        info = not_none(self._type_info.resolved(Target.of(ArrayMarker).fullname))
        nested = inner
        for width in reversed(widths):
            literal = CheckerLiteralTypeBuilder().integer(width, ctx.api)
            nested = Instance(info, [nested, literal])
        return nested

    def _shape(self, node: Expression, api: object) -> list[int] | None:
        """Read the shape of an array: one width, or a tuple or a list of widths.

        polars nests an array for each width after the first, so the shape
        `(2, 3)` gives an array of two arrays of three values.

        Args:
            node: The shape that the call passes.
            api: The checker, which reads the value of a variable declared `Final`.

        Returns:
            The widths, from the outer array in, or `None` where a width is not
            a literal integer.
        """
        items = node.items if isinstance(node, TupleExpr | ListExpr) else [node]
        widths: list[int] = []
        for item in items:
            width = BareLiteralReader().scalar_via_api(item, api)
            if not isinstance(width, int) or isinstance(width, bool):
                return None
            widths.append(width)
        return widths


class StructDtypeConstructorHook:
    """Give a call such as `pl.Struct({"p": pl.Int8})` a type with its fields.

    polars types the call `pl.Struct({...})` only as `Struct`, so the checker
    would lose the fields as soon as the call ends. A cast to that struct would
    then give a bare `Struct`, and a later `.struct.field(...)` would know no
    field.

    This hook reads the dictionary of the call, one field name and one data
    type for each entry, and gives the marker `Struct[...]` that a `Columns`
    field writes. It is a mypy function hook in the plugin layer. If the
    argument is not a dictionary literal with a string key and a data type
    that the hook can read for each entry, the hook keeps the plain type of
    polars. A list of `pl.Field(...)` keeps the plain type too. polars
    raises for a field that is a marker class with its type arguments, such
    as `List[pl.Int64]`, so the hook reports each one.
    """

    def __init__(self, plugin: Plugin) -> None:
        """Keep the plugin, which finds the classes of the marker types.

        Args:
            plugin: The mypy plugin.
        """
        self._plugin = plugin

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Give the type of one call to `pl.Struct`.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker with one `Column[...]` item for each field. If a field
            cannot be read, the plain type that polars declares.
        """
        result = self._result(ctx)
        if result is None:
            return ctx.default_return_type
        return result

    @propagates(None)
    def _result(self, ctx: FunctionContext) -> MypyType | None:
        """Build the marker type from the dictionary of the call.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The marker type, or None if a step fails.
        """
        nodes = CallArgs(ctx).node_list("fields")
        not_falsy(len(nodes) == 1)
        guard = DtypeClassGuard()
        reported = [
            guard.refused_in_nested(ctx, value) for value in DtypeMapping().values(nodes[0])
        ]
        not_falsy(not any(reported))
        entries = not_none(DtypeMapping().entries(ctx.api, nodes[0]))
        spelled = ", ".join(f"{name!r}: {rendering}" for name, rendering in entries)
        minter = MarkerMinter(self._plugin, CheckerLiteralTypeBuilder(), ctx.api)
        return ConstDtypeCodec(minter).built("Struct({" + spelled + "})")


class FieldConstructorHook:
    """Report a call such as `pl.Field("a", List[pl.Int64])`, which polars refuses.

    `pl.Field` pairs a field name with a data type, and a list of such
    pairs builds a `Struct`. polars parses the data type when the call
    runs, and it raises `TypeError` for a marker class of this project
    with its type arguments, such as `List[pl.Int64]`. That marker
    writes a data type in a type annotation, and it is not a data type
    of polars. polars accepts `Unchecked` and the bare class `Decimal`
    here, and only a later use of the data type raises.

    It is a mypy function hook in the plugin layer. It reports the
    argument and keeps the plain type of the call, because no code reads
    the data type of a field.
    """

    def __call__(self, ctx: FunctionContext) -> MypyType:
        """Report a refused data type of one call to `pl.Field`.

        Args:
            ctx: The context that mypy gives for the call.

        Returns:
            The plain type that polars declares.
        """
        guard = DtypeClassGuard()
        for node in CallArgs(ctx).node_list("dtype"):
            guard.refused_in_nested(ctx, node)
        return ctx.default_return_type
