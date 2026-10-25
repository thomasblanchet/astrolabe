"""Check each annotation that uses a marker class of this project, and build its type.

A user describes columns with marker classes in annotations, for example
`TypedDataFrame[Trades]`, `Column[Literal["price"], pl.Float64]` or
`Datetime[Literal["ms"], Literal["UTC"]]`. mypy calls a type analysis hook
when it reads such an annotation, before it checks any code. The hook
checks the type arguments and gives back the type that mypy stores.

Two needs meet here. First, many wrong arguments fail only when the program
runs, for example a time zone that does not exist or a `Decimal` scale
larger than its precision. The hook reports them at the annotation.
Second, every other hook reads these types back later, and it can read only
the shapes that it expects. So the hook gives each marker one fixed shape:
a bare `pl.Decimal` becomes `Decimal[38, 0]`, a schema class becomes a
closed `TypedDict`, and a remainder is written in its canonical form.

This module has one hook class for each marker, and a few helpers that
several hooks share. All of them sit in the plugin layer, and run in the
semantic analysis phase of mypy.
"""

from __future__ import annotations

import zoneinfo
from collections import Counter
from typing import ClassVar, Final, TYPE_CHECKING

import polars as pl
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    NoneType,
    TupleType,
    Type,
    TypedDictType,
    TypeOfAny,
    TypeVarType,
    UnboundType,
    get_proper_type,
)
from mypy.types import Type as MypyType

from astrolabe._hooks.checked import checked_cast, checked_not_none
from astrolabe import frame
from astrolabe.declare import InferredRemainder
from astrolabe._hooks.dispatch.generic import Target
from astrolabe._hooks.encoding.declaration_codec import DeclarationCodec
from astrolabe._hooks.remainder.split import Split, SplitPredicates
from astrolabe._hooks.domain.column import ColumnRef
from astrolabe._hooks.duplicate_name import DuplicateNameCheck
from astrolabe._hooks.encoding.declaration_codec import MarkedDeclarations
from astrolabe._hooks.encoding.rendering_codec import ConstDtypeReads
from astrolabe._hooks.encoding.marker_minter import MarkerMinter, ProperReads
from astrolabe._hooks.encoding.remainder_codec import RemainderCodec
from astrolabe._hooks.remainder.canonical import Carried
from astrolabe._hooks.errors import (
    ColumnsFieldTypeError,
    ColumnsNotClosedError,
    ColumnsNotTotalError,
    MarkerArgumentError,
    MarkerArityError,
)
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.calls.dtype_constructor_readers import SemanticAnalyzerLiteralTypeBuilder
from astrolabe._hooks.propagate import not_falsy, not_none, propagates
import astrolabe.remainder as remainder_markers
from astrolabe.remainder import Prefixed, Suffixed
from astrolabe.expr import TypedColumn
from astrolabe.frame import Array as _ArrayMarker
from astrolabe.frame import Datetime as _DatetimeMarker
from astrolabe.frame import Decimal as _DecimalMarker
from astrolabe.frame import Duration as _DurationMarker
from astrolabe.frame import Enum as _EnumMarker
from astrolabe.frame import List as _ListMarker
from astrolabe.frame import Struct as _StructMarker


if TYPE_CHECKING:
    from astrolabe._hooks.remainder.record import Record
    from collections.abc import Callable

    from mypy.nodes import TypeInfo
    from mypy.nodes import Context
    from mypy.plugin import AnalyzeTypeContext
    from mypy.types import ProperType

    from astrolabe._hooks.calls.dtype_constructor_readers import LiteralTypeBuilder
    from astrolabe._hooks.domain.frame import PluginLookup
    from astrolabe._hooks.errors import _FailApi, _FailContext


class TypeAnalysisHook:
    """Give the shared steps of the type analysis hooks for the marker classes.

    Each marker hook subclasses this class, and implements `__call__`
    with its own checks of the count and the shape of its arguments. Most
    markers take a literal string somewhere and report errors in the same
    way, so those two steps live here.
    """

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one annotation of the marker, and give the type that mypy stores.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type of the marker.
        """
        raise NotImplementedError

    def _literal_string_arg(
        self,
        arg: Type,
        ctx: AnalyzeTypeContext,
        error_message: str,
    ) -> LiteralType | None:
        """Read a type argument as a literal string type, or report it.

        A bare string, as in `Column["foo", ...]`, is rejected. mypy reads
        a bare string in a type argument as a forward reference to a
        class, not as a literal.

        Args:
            arg: The type argument.
            ctx: The context that mypy gives for the annotation.
            error_message: The message to report if `arg` is not a literal
                string.

        Returns:
            The literal type, or None after a `MarkerArgumentError` if
            `arg` is a bare string or not a literal string.
        """
        if not self._is_bare_string(arg):
            analyzed = get_proper_type(ctx.api.analyze_type(arg))
            if isinstance(analyzed, LiteralType) and isinstance(analyzed.value, str):
                return analyzed
        MarkerArgumentError(error_message).fail(ctx)
        return None

    def _is_bare_string(self, arg: Type) -> bool:
        """Answer whether a type argument is a bare string, not wrapped in `Literal[...]`.

        mypy reads a bare string in a type argument, as in
        `Column["foo", ...]`, as a forward reference. It keeps the text in
        `original_str_expr`, which marks the case.

        Args:
            arg: The type argument, before analysis.

        Returns:
            True if `arg` is a bare string.

        Example:
            >>> hook = ListTypeHook()
            >>> hook._is_bare_string(UnboundType("foo", original_str_expr="foo"))
            True
            >>> hook._is_bare_string(UnboundType("foo"))
            False
        """
        return isinstance(arg, UnboundType) and arg.original_str_expr is not None

    def _error(
        self,
        ctx: AnalyzeTypeContext,
        error: MarkerArityError | MarkerArgumentError,
    ) -> MypyType:
        """Report an error, and give the type that mypy uses after an error.

        mypy reports no second error for a type that comes from an error.

        Args:
            ctx: The context that mypy gives for the annotation.
            error: The error to report.

        Returns:
            `Any`, marked as the result of an error.
        """
        error.fail(ctx)
        return AnyType(TypeOfAny.from_error)


class GroupByTypeHook(TypeAnalysisHook):
    """Check `TypedGroupBy[...]` and `TypedLazyGroupBy[...]` in an annotation.

    The checker gives these types to `group_by(...)`, with a tuple of
    `Column[...]` items for the keys, a tuple for the declared columns
    and the remainder. A user can also write the type by hand. The hook
    for `.agg(...)` reads the two tuples, so an annotation that does not
    give them is an error here.
    """

    def __init__(self, target: Target) -> None:
        """Keep the marker class that this instance checks.

        Args:
            target: `TypedGroupBy` or `TypedLazyGroupBy`.
        """
        self._target = target

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one annotation of the marker.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The marker type, or `Any` after an error if the annotation does
            not give three arguments, the first two of them tuples of
            `Column[...]` items.
        """
        args = [ctx.api.analyze_type(arg) for arg in ctx.type.args]
        if len(args) != 3 or not all(self._columns(arg) for arg in args[:2]):
            message = (
                f"{self._target.fullname.rsplit('.', 1)[-1]}[...] takes a tuple of Column[...] for "
                "the keys, a tuple of Column[...] for the declared columns, and the remainder"
            )
            return self._error(ctx, MarkerArgumentError(message))
        return ctx.api.named_type(self._target.fullname, args)

    def _columns(self, arg: MypyType) -> bool:
        """Answer whether an argument is a tuple of `Column[...]` items.

        Each item must be an instance of `Column`.

        Args:
            arg: The analyzed argument.

        Returns:
            True for such a tuple.
        """
        proper = get_proper_type(arg)
        if not isinstance(proper, TupleType):
            return False
        for item in proper.items:
            item_proper = get_proper_type(item)
            if not isinstance(item_proper, Instance) or item_proper.type.fullname != self._COLUMN:
                return False
        return True

    _COLUMN: ClassVar[str] = Target.of(frame.Column).fullname


class ColumnTypeHook(TypeAnalysisHook):
    """Check `Column[name, dtype]`, one named and typed column.

    `name` must be a literal string, `Literal["price"]`. Anything else is
    an error, a bare string included. A bare `pl.Decimal`, `pl.Datetime`
    or `pl.Duration` in `dtype` takes the default parameters of polars,
    through `BareDtypeDefaults`, so that it compares equal to the same
    data type written in full.
    """

    TARGET: Final[Target] = ColumnRef.TARGET

    def __init__(self, bare_defaults: BareDtypeDefaults) -> None:
        """Keep the rule that gives a bare data type class its default parameters.

        Args:
            bare_defaults: Gives a bare `pl.Decimal`, `pl.Datetime` or
                `pl.Duration` its default parameters, and keeps every other
                data type unchanged.
        """
        self._bare_defaults = bare_defaults

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Column[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Column[name, dtype]`, or `Any` after an error if the
            arguments do not have this shape.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Column[...] requires exactly 2 type arguments: Column[name, dtype]"
            return self._error(ctx, MarkerArityError(msg))

        name_arg, dtype_arg = args
        name_type = self._literal_string_arg(
            name_arg,
            ctx,
            'Column[...] name argument must be a string literal, e.g. Column[Literal["foo"], ...]',
        )
        if name_type is None:
            return AnyType(TypeOfAny.from_error)

        dtype_type = self._bare_defaults.applied(ctx.api, ctx.api.analyze_type(dtype_arg))
        return ctx.api.named_type(self.TARGET.fullname, [name_type, dtype_type])


class SuffixedTypeHook(TypeAnalysisHook):
    """Check `Suffixed[name_expr, suffix]`, a remainder whose names take a suffix.

    `suffix` must be a literal string. `name_expr` can be any remainder
    type, and the hook passes it on unchecked.
    """

    TARGET: Final[Target] = Target.of(Suffixed)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Suffixed[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Suffixed[name_expr, suffix]`, or `Any` after an
            error if the arguments do not have this shape.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Suffixed[...] requires exactly 2 type arguments: Suffixed[name_expr, suffix]"
            return self._error(ctx, MarkerArityError(msg))
        name_expr_type = ctx.api.analyze_type(args[0])
        suffix_type = self._literal_string_arg(
            args[1],
            ctx,
            "Suffixed[...] suffix argument must be a string literal, e.g. "
            'Suffixed[..., Literal["_right"]]',
        )
        if suffix_type is None:
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [name_expr_type, suffix_type])


class PrefixedTypeHook(TypeAnalysisHook):
    """Check `Prefixed[prefix, name_expr]`, a remainder whose names take a prefix.

    `prefix` must be a literal string. `name_expr` can be any remainder
    type, and the hook passes it on unchecked.
    """

    TARGET: Final[Target] = Target.of(Prefixed)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Prefixed[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Prefixed[prefix, name_expr]`, or `Any` after an
            error if the arguments do not have this shape.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Prefixed[...] requires exactly 2 type arguments: Prefixed[prefix, name_expr]"
            return self._error(ctx, MarkerArityError(msg))
        prefix_type = self._literal_string_arg(
            args[0],
            ctx,
            "Prefixed[...] prefix argument must be a string literal, e.g. "
            'Prefixed[Literal["pre_"], ...]',
        )
        name_expr_type = ctx.api.analyze_type(args[1])
        if prefix_type is None:
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [prefix_type, name_expr_type])


class ListTypeHook(TypeAnalysisHook):
    """Check `List[dtype]`, the data type of a column that holds a list in each row.

    mypy analyzes `dtype` with the hooks too, so a nested `List[List[...]]`
    needs no special case.
    """

    TARGET: Final[Target] = Target.of(_ListMarker)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `List[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `List[dtype]`, or `Any` after an error if the
            annotation does not have exactly one type argument.
        """
        args = ctx.type.args
        if len(args) != 1:
            msg = "List[...] requires exactly 1 type argument: List[dtype]"
            return self._error(ctx, MarkerArityError(msg))

        dtype_type = ctx.api.analyze_type(args[0])
        return ctx.api.named_type(self.TARGET.fullname, [dtype_type])


class ArrayTypeHook(TypeAnalysisHook):
    """Check `Array[dtype, size]`, the data type of a column with a fixed-length array in each row.

    `size` must be a literal integer that is not negative, for example
    `Literal[3]`. mypy itself rejects a bare integer there. A `bool` literal
    is not an integer here: mypy keeps `Literal[True]` apart from
    `Literal[1]`, so no data type that the checker mints could match it.
    """

    TARGET: Final[Target] = Target.of(_ArrayMarker)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Array[...]` annotation.

        polars raises `OverflowError` for a negative size, even for a frame
        with no rows. If mypy already rejected the `size` argument, for
        example a bare integer with its own `[valid-type]` error, the hook
        keeps the `Any` that mypy gives, with no second error.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Array[dtype, size]`, or `Any` after an error if the
            annotation does not have two type arguments or `size` is not a
            literal integer of zero or more.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Array[...] requires exactly 2 type arguments: Array[dtype, size]"
            return self._error(ctx, MarkerArityError(msg))

        dtype_type = ctx.api.analyze_type(args[0])
        size_type = get_proper_type(ctx.api.analyze_type(args[1]))
        is_non_negative_int = (
            isinstance(size_type, LiteralType)
            and type(size_type.value) is int
            and size_type.value >= 0
        )
        if not (isinstance(size_type, AnyType) or is_non_negative_int):
            error = MarkerArgumentError(
                "Array[...] size argument must be a non-negative integer literal",
            )
            error.fail(ctx)
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [dtype_type, size_type])


class DeclaredColumnItem:
    """Read a `column(...)` declaration as a `Column[...]` item, for a field of `Struct[...]`.

    A user can declare a column once, `Price = column("price",
    pl.Float64)`, and use the name `Price` in many places. As a type,
    `Price` is `TypedColumn[Literal["price"], pl.Float64]`, the typed
    expression that reads that column. Inside `Struct[...]`, the same name
    means the field `Column[Literal["price"], pl.Float64]`. This class
    makes that translation.

    A declaration can also give a set of data types, for example a
    family such as all integers, or `EveryDType`. Such a declaration names
    no single field type, so it stays as it is and the struct check
    rejects it.

    Example:
        >>> fixture = new_type_fixture()
        >>> item = DeclaredColumnItem(new_plugin(fixture))
        >>> item.as_column(int_dtype(fixture)) is None
        True
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep a way to look up the `Column` class, and the reader for marker types.

        Args:
            plugin: The mypy plugin, which finds `Column` by its full name.
        """
        self._types = FreshTypeInfo(plugin)
        self._reads = ProperReads()

    def analyzed_item(self, ctx: AnalyzeTypeContext, arg: Type) -> ProperType:
        """Analyze one type argument, and turn a declared column into its `Column[...]`.

        Args:
            ctx: The context that mypy gives for the annotation.
            arg: The type argument.

        Returns:
            A `Column[...]` instance for a declared column with one data
            type. Else the analyzed type unchanged.
        """
        analyzed = get_proper_type(ctx.api.analyze_type(arg))
        column = self.as_column(analyzed)
        if column is None:
            return analyzed
        return column

    @propagates(None)
    def as_column(self, analyzed: MypyType) -> Instance | None:
        """Give a declared column as a `Column[...]` instance.

        A declaration with a blank name, one that stands for any column,
        has no literal name, so it gives None.

        Args:
            analyzed: The analyzed type argument.

        Returns:
            The column instance, or None for any other type.
        """
        instance = not_none(self._reads.instance(analyzed))
        not_falsy(self._reads.matches(instance, TypedColumn))
        not_falsy(len(instance.args) == 2)
        not_none(self._reads.string_literal(instance.args[0]))
        dtype = not_none(self._concrete(instance.args[1]))
        info = not_none(self._types.resolved(ColumnRef.TARGET.fullname))
        return Instance(info, [instance.args[0], dtype])

    def _concrete(self, region: MypyType) -> Instance | None:
        """Give the declared data type if it is one data type, and not a set of them.

        Args:
            region: The data type part of the declaration.

        Returns:
            The data type, or None for a family, a union or `EveryDType`.

        Example:
            >>> fixture = new_type_fixture()
            >>> item = DeclaredColumnItem(new_plugin(fixture))
            >>> item._concrete(unchecked_instance(fixture)) is not None
            True
            >>> item._concrete(union_type(int_dtype(fixture), str_dtype(fixture))) is None
            True
        """
        instance = self._reads.instance(region)
        if instance is None or ConstDtypeReads().read(instance) is None:
            return None
        return instance


class StructTypeHook(TypeAnalysisHook):
    """Check `Struct[Column[...], ...]`, the data type of a column that holds a record in each row.

    Each argument must be a `Column[...]`, or a `column(...)` declaration
    with one data type. A struct names all of its fields, so it takes no
    remainder, unlike a frame. polars rejects two fields with one name, and
    the hook reports that too.
    """

    TARGET: Final[Target] = Target.of(_StructMarker)

    def __init__(self, plugin: PluginLookup) -> None:
        """Build the duplicate check and the reader for declared columns.

        Args:
            plugin: The mypy plugin, which finds `Column` for a declared
                column.
        """
        self._duplicate_check = DuplicateNameCheck("Struct[...]")
        self._declared = DeclaredColumnItem(plugin)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Struct[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Struct[...]`, with its fields in one tuple type
            argument. An argument that is not a `Column[...]` gets a
            `MarkerArgumentError` and is left out. If two fields share a
            name, `Any` after the error. A field whose name is a type
            variable has no literal name, so the duplicate check skips it.
        """
        items: list[MypyType] = []
        for arg in ctx.type.args:
            analyzed = self._declared.analyzed_item(ctx, arg)
            if ColumnRef.TARGET.matches(analyzed):
                items.append(analyzed)
                continue
            MarkerArgumentError("Struct[...] arguments must be Column[...]").fail(ctx)
        names = [checked_not_none(ColumnRef.from_proper_type(item)).name for item in items]
        if self._duplicate_check.fails(ctx, names):
            return AnyType(TypeOfAny.from_error)
        tuple_fallback = ctx.api.named_type(
            "builtins.tuple",
            [ctx.api.named_type("builtins.object", [])],
        )
        fields_tuple = TupleType(items, fallback=tuple_fallback)
        return ctx.api.named_type(self.TARGET.fullname, [fields_tuple])


class TimeUnitValidator:
    """Check the time unit of `Datetime[...]` or `Duration[...]`.

    polars knows only three time units: `"ms"`, `"us"` and `"ns"`. The
    two markers share this rule, so both hooks use this one class.
    """

    _RECOGNIZED: ClassVar[frozenset[str]] = frozenset({"ms", "us", "ns"})

    def checked(self, unit: LiteralType, ctx: AnalyzeTypeContext) -> LiteralType | None:
        """Check the value of a time unit that is already a literal string.

        Args:
            unit: The `Unit` argument, as a literal string type.
            ctx: The context that mypy gives for the annotation.

        Returns:
            `unit` unchanged, or None after a `MarkerArgumentError` if its
            value is not `"ms"`, `"us"` or `"ns"`.
        """
        if unit.value in self._RECOGNIZED:
            return unit
        error = MarkerArgumentError(
            'Unit must be Literal["ms"], Literal["us"], or Literal["ns"]',
        )
        error.fail(ctx)
        return None


class DatetimeTypeHook(TypeAnalysisHook):
    """Check `Datetime[Unit, Zone]`, the data type of a date and time column.

    `Unit` must be `Literal["ms"]`, `Literal["us"]` or `Literal["ns"]`.
    `Zone` must be None, a literal string that names a real time zone, such
    as `Literal["UTC"]`, or `AnyZone`, which stands for every zone. polars
    fails on an unknown zone name only when it builds the column, so the
    hook checks the name against the time zone database of Python.
    """

    TARGET: Final[Target] = Target.of(_DatetimeMarker)
    _ANY_ZONE: Final[str] = Target.of(frame.AnyZone).fullname

    def __init__(self, time_unit: TimeUnitValidator) -> None:
        """Keep the time unit check, and load the names of the IANA time zones.

        Args:
            time_unit: Checks the time unit.
        """
        self._time_unit = time_unit
        self._available_zones = zoneinfo.available_timezones()

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Datetime[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Datetime[unit, zone]`, or `Any` after an error if
            the arguments do not have this shape.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Datetime[...] requires exactly 2 type arguments: Datetime[unit, zone]"
            return self._error(ctx, MarkerArityError(msg))
        literal = self._literal_string_arg(
            args[0],
            ctx,
            'Datetime[...] unit argument must be Literal["ms"], Literal["us"], or Literal["ns"]',
        )
        if literal is None:
            return AnyType(TypeOfAny.from_error)
        unit_type = self._time_unit.checked(literal, ctx)
        if unit_type is None:
            return AnyType(TypeOfAny.from_error)
        zone_type = self._zone_type(ctx, args[1])
        if zone_type is None:
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [unit_type, zone_type])

    def _zone_type(self, ctx: AnalyzeTypeContext, zone_arg: Type) -> MypyType | None:
        """Read and check the `Zone` argument of `Datetime[...]`.

        None means no zone, and `AnyZone` stands for every zone of the unit.
        polars raises `ComputeError: unable to parse time zone` for an
        unknown zone as soon as it builds a column of this data type. So the
        method checks a zone name against the names of the IANA time zones.

        Args:
            ctx: The context that mypy gives for the annotation.
            zone_arg: The `Zone` type argument.

        Returns:
            None as a type, the `AnyZone` marker, or a literal string that
            names an IANA time zone. None after an error for any other
            argument, or for an unknown zone name.
        """
        analyzed = get_proper_type(ctx.api.analyze_type(zone_arg))
        if isinstance(analyzed, NoneType):
            return analyzed
        if isinstance(analyzed, Instance) and analyzed.type.fullname == self._ANY_ZONE:
            return analyzed
        if isinstance(analyzed, LiteralType) and isinstance(analyzed.value, str):
            if analyzed.value in self._available_zones:
                return analyzed
            message = f"Zone {analyzed.value!r} is not a recognized IANA time zone name"
            MarkerArgumentError(message).fail(ctx)
            return None
        error = MarkerArgumentError(
            "Datetime[...] zone argument must be None, a string literal such as "
            'Literal["UTC"], or AnyZone for every zone of the unit',
        )
        error.fail(ctx)
        return None


class DurationTypeHook(TypeAnalysisHook):
    """Check `Duration[Unit]`, the data type of a column of time spans.

    `Unit` must be `Literal["ms"]`, `Literal["us"]` or `Literal["ns"]`.
    """

    TARGET: Final[Target] = Target.of(_DurationMarker)

    def __init__(self, time_unit: TimeUnitValidator) -> None:
        """Keep the time unit check.

        Args:
            time_unit: Checks the time unit.
        """
        self._time_unit = time_unit

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Duration[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Duration[unit]`, or `Any` after an error if the
            annotation does not have exactly one valid argument.
        """
        args = ctx.type.args
        if len(args) != 1:
            msg = "Duration[...] requires exactly 1 type argument: Duration[unit]"
            return self._error(ctx, MarkerArityError(msg))
        literal = self._literal_string_arg(
            args[0],
            ctx,
            'Duration[...] unit argument must be Literal["ms"], Literal["us"], or Literal["ns"]',
        )
        if literal is None:
            return AnyType(TypeOfAny.from_error)
        unit_type = self._time_unit.checked(literal, ctx)
        if unit_type is None:
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [unit_type])


class DecimalShapeValidator:
    """Check the precision and the scale of `Decimal[Precision, Scale]` against the rules of polars.

    polars raises `OverflowError` for a negative precision or scale, and
    `InvalidOperationError` for a scale larger than the precision. It
    raises as soon as it builds a schema with the data type, even with no
    rows. This class holds the two rules for the Decimal hook.
    """

    def checked_non_negative(
        self,
        value: LiteralType,
        label: str,
        ctx: AnalyzeTypeContext,
    ) -> LiteralType | None:
        """Check that a literal integer is zero or more.

        Args:
            value: The precision or the scale, as a literal integer type.
            label: `"precision"` or `"scale"`, for the error message.
            ctx: The context that mypy gives for the annotation.

        Returns:
            `value` unchanged, or None after a `MarkerArgumentError` if it
            is negative.
        """
        if checked_cast(value.value, int) >= 0:
            return value
        MarkerArgumentError(f"Decimal[...] {label} argument must be non-negative").fail(ctx)
        return None

    def checked_scale_within_precision(
        self,
        precision: LiteralType,
        scale: LiteralType,
        ctx: AnalyzeTypeContext,
    ) -> bool:
        """Check that the scale is not larger than the precision.

        Args:
            precision: The precision, as a literal integer type of zero or
                more.
            scale: The scale, as a literal integer type of zero or more.
            ctx: The context that mypy gives for the annotation.

        Returns:
            True if `scale` is not larger than `precision`. Else False,
            after a `MarkerArgumentError`.
        """
        if checked_cast(scale.value, int) <= checked_cast(precision.value, int):
            return True
        MarkerArgumentError("Decimal[...] scale must not exceed precision").fail(ctx)
        return False


class ImplicitDecimalPrecision:
    """Mint `Literal[38]`, the precision that polars gives a `Decimal` with none.

    polars always sets the precision to 38 if the code gives none. So
    `Decimal[None, 2]` and `Decimal[38, 2]` are the same data type, and
    their types must be equal too. Two places need the value: a `Decimal`
    marker with no precision, and a bare `pl.Decimal` in a column. Both
    take it from this class, so they always agree.
    """

    VALUE: Final[int] = 38

    def __init__(self, literal_type: LiteralTypeBuilder) -> None:
        """Keep the builder of literal types for one phase of mypy.

        Args:
            literal_type: Builds a `LiteralType` from a Python value with
                the interface that mypy gives in that phase.
        """
        self._literal_type = literal_type

    def build(self, api: object) -> LiteralType:
        """Mint a new `Literal[38]`.

        This method has no doctest, because it needs the real interface
        that mypy gives in a phase.

        Args:
            api: The mypy plugin interface, which builds the `int` type
                under the literal.

        Returns:
            `Literal[38]`.
        """
        return self._literal_type.integer(self.VALUE, api)


class BareDtypeDefaults:
    """Give a bare data type class in a column the default parameters of polars.

    `pl.Decimal()` is `Decimal(38, 0)`, `pl.Datetime()` is
    `Datetime("us", None)` and `pl.Duration()` is `Duration("us")`. So a
    column declared with the bare class has that data type. Without this
    rule, the bare class carries no parameters, and each operation on the
    column answers as if the parameters were unknown.

    The default applies only to the data type of a column. In a selector
    or a requirement, a bare class keeps its meaning of every data type of
    that kind, as `cs.by_dtype(pl.Datetime)` has in polars.

    The rule runs in both phases of mypy: in an annotation, and in a call
    that takes a frame type as a value. The builder of literal types
    belongs to one phase, and the lookup of the marker classes works in
    both.
    """

    _BARE_DECIMAL: Final[Target] = Target.of(pl.Decimal)
    _BARE_DATETIME: Final[Target] = Target.of(pl.Datetime)
    _BARE_DURATION: Final[Target] = Target.of(pl.Duration)

    def __init__(
        self,
        implicit_precision: ImplicitDecimalPrecision,
        literal_type: LiteralTypeBuilder,
        markers: FreshTypeInfo,
    ) -> None:
        """Keep the builders of the literal arguments, and the lookup of the marker classes.

        Args:
            implicit_precision: Mints `Literal[38]` for the precision.
            literal_type: Builds the other literal arguments.
            markers: Finds the class of each marker that the rule mints.
        """
        self._implicit_precision = implicit_precision
        self._literal_type = literal_type
        self._markers = markers

    def applied(self, api: object, dtype_type: MypyType) -> MypyType:
        """Give a bare class its default parameters, and keep any other data type unchanged.

        Args:
            api: The mypy plugin interface of the phase, which builds the
                types under the literal arguments.
            dtype_type: The analyzed data type of one column.

        Returns:
            The marker with parameters for a bare `pl.Decimal`,
            `pl.Datetime` or `pl.Duration`. Else `dtype_type`.
        """
        proper = get_proper_type(dtype_type)
        if not isinstance(proper, Instance) or proper.args:
            return dtype_type
        minted: dict[Target, Callable[[object], MypyType]] = {
            self._BARE_DECIMAL: self._decimal,
            self._BARE_DATETIME: self._datetime,
            self._BARE_DURATION: self._duration,
        }
        for target, mint in minted.items():
            if target.matches(proper):
                return mint(api)
        return dtype_type

    def _decimal(self, api: object) -> MypyType:
        """Mint `Decimal[Literal[38], Literal[0]]`.

        Args:
            api: The mypy plugin interface of the phase.

        Returns:
            The marker.
        """
        precision = self._implicit_precision.build(api)
        scale = self._literal_type.integer(0, api)
        return self._marker(_DecimalMarker, [precision, scale])

    def _datetime(self, api: object) -> MypyType:
        """Mint `Datetime[Literal["us"], None]`.

        Args:
            api: The mypy plugin interface of the phase.

        Returns:
            The marker.
        """
        unit = self._literal_type.string("us", api)
        return self._marker(_DatetimeMarker, [unit, NoneType()])

    def _duration(self, api: object) -> MypyType:
        """Mint `Duration[Literal["us"]]`.

        Args:
            api: The mypy plugin interface of the phase.

        Returns:
            The marker.
        """
        unit = self._literal_type.string("us", api)
        return self._marker(_DurationMarker, [unit])

    def _marker(self, marker: type, args: list[MypyType]) -> Instance:
        """Mint one marker of a data type with its parameters.

        Args:
            marker: The marker class, for example `Decimal`.
            args: The parameters, as type arguments.

        Returns:
            The marker.
        """
        info = checked_not_none(self._markers.resolved(Target.of(marker).fullname))
        return Instance(info, args)


class DecimalTypeHook(TypeAnalysisHook):
    """Check `Decimal[Precision, Scale]`, the data type of a column of exact decimal numbers.

    `Precision` must be None or a literal integer of zero or more, such as
    `Literal[38]`. None becomes `Literal[38]`, the default of polars, so
    that `Decimal[None, 2]` and `Decimal[38, 2]` give the same type, as
    they give the same data type. `Scale` must be a literal integer of zero
    or more, and not larger than the precision. The bound of the type
    parameter already rejects a scale that is not an integer, but it
    accepts a negative one, so the hook checks that. A `bool` literal is not
    an integer here: mypy keeps `Literal[True]` apart from `Literal[1]`, so
    no data type that the checker mints could match it.
    """

    TARGET: Final[Target] = Target.of(_DecimalMarker)

    def __init__(
        self,
        shape: DecimalShapeValidator,
        implicit_precision: ImplicitDecimalPrecision,
    ) -> None:
        """Keep the checks of the arguments and the builder of the default precision.

        Args:
            shape: Checks the precision and the scale against the rules
                of polars.
            implicit_precision: Mints `Literal[38]` for a precision of
                None.
        """
        self._shape = shape
        self._implicit_precision = implicit_precision

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Decimal[...]` annotation.

        If mypy already rejected the precision or the scale, that side is
        `Any`, and the hook has nothing to compare. Else the scale must not
        be larger than the precision.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Decimal[precision, scale]`. `Any` after an error if
            the annotation does not have two arguments, an argument is
            wrong, or the scale is larger than the precision.
        """
        args = ctx.type.args
        if len(args) != 2:
            msg = "Decimal[...] requires exactly 2 type arguments: Decimal[precision, scale]"
            return self._error(ctx, MarkerArityError(msg))
        precision_type = self._precision_type(ctx, args[0])
        if precision_type is None:
            return AnyType(TypeOfAny.from_error)
        scale_type = self._scale_type(ctx, args[1])
        if scale_type is None:
            return AnyType(TypeOfAny.from_error)
        if isinstance(precision_type, LiteralType) and isinstance(scale_type, LiteralType):
            shape = self._shape
            if not shape.checked_scale_within_precision(precision_type, scale_type, ctx):
                return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self.TARGET.fullname, [precision_type, scale_type])

    def _precision_type(self, ctx: AnalyzeTypeContext, precision_arg: Type) -> MypyType | None:
        """Read and check the `Precision` argument of `Decimal[...]`.

        Args:
            ctx: The context that mypy gives for the annotation.
            precision_arg: The `Precision` type argument.

        Returns:
            `Literal[38]`, the default of polars, for None. The literal
            integer if it is zero or more. The `Any` that mypy gives if
            mypy already rejected the argument, for example a bare integer
            with its own `[valid-type]` error, with no second error. None
            after an error for any other argument.
        """
        analyzed = get_proper_type(ctx.api.analyze_type(precision_arg))
        if isinstance(analyzed, AnyType):
            return analyzed
        if isinstance(analyzed, NoneType):
            return self._implicit_precision.build(ctx.api)
        if isinstance(analyzed, LiteralType) and type(analyzed.value) is int:
            return self._shape.checked_non_negative(analyzed, "precision", ctx)
        error = MarkerArgumentError(
            "Decimal[...] precision argument must be None or a literal int, e.g. Literal[38]",
        )
        error.fail(ctx)
        return None

    def _scale_type(self, ctx: AnalyzeTypeContext, scale_arg: Type) -> MypyType | None:
        """Read and check the `Scale` argument of `Decimal[...]`.

        Args:
            ctx: The context that mypy gives for the annotation.
            scale_arg: The `Scale` type argument.

        Returns:
            The literal integer if it is zero or more. The `Any` that mypy
            gives if mypy already rejected the argument, with no second
            error. None after an error for any other argument.
        """
        analyzed = get_proper_type(ctx.api.analyze_type(scale_arg))
        if isinstance(analyzed, AnyType):
            return analyzed
        if isinstance(analyzed, LiteralType) and type(analyzed.value) is int:
            return self._shape.checked_non_negative(analyzed, "scale", ctx)
        error = MarkerArgumentError(
            "Decimal[...] scale argument must be a literal int, e.g. Literal[2]",
        )
        error.fail(ctx)
        return None


class EnumTypeHook(TypeAnalysisHook):
    """Check `Enum[*Categories]`, the data type of a column with a fixed set of text values.

    Each argument must be a literal string. `Enum[()]` is the enum with no
    category, which polars accepts as `pl.Enum([])`. A bare `Enum` with no
    brackets names no categories at all, so the hook reports it. polars
    rejects an `Enum` with a repeated category, so the hook reports that
    too.
    """

    TARGET: Final[Target] = Target.of(_EnumMarker)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one `Enum[...]` annotation.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The type `Enum[...]`, with its categories in one tuple type
            argument. `Any` after an error if it is a bare `Enum`, an
            argument is not a literal string, or two categories are the
            same.
        """
        args = ctx.type.args
        if not args and not ctx.type.empty_tuple_index:
            msg = "Enum needs its categories: Enum[category, ...], or Enum[()] for none"
            return self._error(ctx, MarkerArityError(msg))
        items = self._items(ctx, args)
        if items is None:
            return AnyType(TypeOfAny.from_error)
        if self._has_duplicate(ctx, items):
            return AnyType(TypeOfAny.from_error)
        tuple_fallback = ctx.api.named_type(
            "builtins.tuple",
            [ctx.api.named_type("builtins.object", [])],
        )
        categories = TupleType(items, fallback=tuple_fallback)
        return ctx.api.named_type(self.TARGET.fullname, [categories])

    def _has_duplicate(self, ctx: AnalyzeTypeContext, items: list[MypyType]) -> bool:
        """Report a repeated category, if there is one.

        polars raises `ValueError` for an `Enum` with a repeated category
        as soon as it builds the data type, with no data involved. The error
        names the repeated category that comes first in sorted order, so the
        message is the same on every run.

        Args:
            ctx: The context that mypy gives for the annotation.
            items: The category literals, each a literal string.

        Returns:
            True if a category repeats, after a `MarkerArgumentError`.
        """
        values = [checked_cast(checked_cast(item, LiteralType).value, str) for item in items]
        counts = Counter(values)
        repeated = sorted(value for value, count in counts.items() if count > 1)
        if not repeated:
            return False
        error = MarkerArgumentError(
            f"Enum[...] arguments must all be different, found duplicate {repeated[0]!r}",
        )
        error.fail(ctx)
        return True

    def _items(self, ctx: AnalyzeTypeContext, args: tuple[Type, ...]) -> list[MypyType] | None:
        """Read each argument of `Enum[...]` as a literal string type.

        Args:
            ctx: The context that mypy gives for the annotation.
            args: The type arguments of the marker.

        Returns:
            The literal types in order, or None after an error if an
            argument is not a literal string.
        """
        items: list[MypyType] = []
        for arg in args:
            literal = self._literal_string_arg(
                arg,
                ctx,
                'Enum[...] arguments must be string literals, e.g. Literal["a"]',
            )
            if literal is None:
                return None
            items.append(literal)
        return items


class _At:
    """Point an error at a chosen node, in place of the annotation that mypy is reading.

    A schema class is checked when a frame annotation uses it, not where
    the class is written. A wrong field is best reported at the field
    itself, and not at each annotation that uses the class. An error class
    of this project reports through any object with a `.context` and an
    `.api`, so this small class holds the chosen node and the interface of
    the current hook.
    """

    def __init__(self, context: Context, api: _FailApi) -> None:
        """Keep the node to report at, and the interface that reports.

        Args:
            context: The node where the error appears.
            api: The mypy plugin interface that reports the error.
        """
        self.context = context
        self.api = api


class ColumnsRestrictionCheck:
    """Check that a schema class can describe the columns of a frame.

    A schema class is a `TypedDict` class, usually one that inherits
    `Columns`, whose fields are the columns. Three rules apply. Each field
    must be required, because a frame always has each declared column. The
    class must not be `closed=False`, because a schema lists all of the
    declared columns. And each field must be a data type.

    mypy runs no plugin hook for the class definition alone. So the check
    runs each time a frame type uses the class as its first type argument,
    in an annotation or as a value, and it points each error at the class
    or the field.
    """

    _DTYPE_TARGET: Final[Target] = Target.of(pl.DataType)

    @propagates(None)
    def checked(self, ctx: _FailContext, analyzed: TypedDictType) -> TypedDictType | None:
        """Run the three checks on the schema class of one frame type.

        Args:
            ctx: The context that mypy gives for the frame type.
            analyzed: The `TypedDict` type of the schema class.

        Returns:
            `analyzed` unchanged, or None after the error of the first
            check that fails.
        """
        info = analyzed.fallback.type
        not_falsy(self._checked_total(ctx, info, analyzed))
        not_falsy(self._checked_closed(ctx, info))
        not_falsy(self._checked_fields(ctx, info, analyzed))
        return analyzed

    def _checked_total(
        self,
        ctx: _FailContext,
        info: TypeInfo,
        analyzed: TypedDictType,
    ) -> bool:
        """Check that each field of the schema class is required.

        Args:
            ctx: The context that mypy gives for the frame type.
            info: The schema class.
            analyzed: The `TypedDict` type of the class.

        Returns:
            True if each field is required. Else False, after a
            `ColumnsNotTotalError` at the `total=` keyword. If the class has
            no such keyword, the cause is a `NotRequired[...]` field, and the
            error is at the class.
        """
        if analyzed.required_keys == set(analyzed.items):
            return True
        context: Context = info.defn
        keyword = info.defn.keywords.get("total")
        if keyword is not None:
            context = keyword
        ColumnsNotTotalError(info.name).fail(_At(context, ctx.api))
        return False

    def _checked_closed(self, ctx: _FailContext, info: TypeInfo) -> bool:
        """Check that the schema class does not declare `closed=False`.

        Args:
            ctx: The context that mypy gives for the frame type.
            info: The schema class.

        Returns:
            True if the class does not declare `closed=False`. Else False,
            after a `ColumnsNotClosedError` at that keyword.
        """
        keyword = info.defn.keywords.get("closed")
        if keyword is None or getattr(keyword, "fullname", None) != "builtins.False":
            return True
        ColumnsNotClosedError(info.name).fail(_At(keyword, ctx.api))
        return False

    def _checked_fields(
        self,
        ctx: _FailContext,
        info: TypeInfo,
        analyzed: TypedDictType,
    ) -> bool:
        """Check that each field of the schema class is a data type.

        Args:
            ctx: The context that mypy gives for the frame type.
            info: The schema class.
            analyzed: The `TypedDict` type of the class.

        Returns:
            True if each field is a polars data type or a data type marker
            of this project. Else False, after a `ColumnsFieldTypeError` at
            the first wrong field, in field order.
        """
        for name, field_type in analyzed.items.items():
            if self._is_dtype(get_proper_type(field_type)):
                continue
            context = self._field_context(info, name)
            ColumnsFieldTypeError(name).fail(_At(context, ctx.api))
            return False
        return True

    def _is_dtype(self, field_type: ProperType) -> bool:
        """Answer whether the type of a field is a data type, or a type variable bound to one.

        A generic schema class, for example `class Keyed(Columns,
        Generic[T])`, can give a field the type variable `T`. The field is
        a data type if the bound of `T` is one.

        Args:
            field_type: The type of the field.

        Returns:
            True for a polars data type, a data type marker of this
            project, a type variable bound to one of those, or a field
            whose annotation already has an error.
        """
        if isinstance(field_type, TypeVarType):
            return self._is_dtype(get_proper_type(field_type.upper_bound))
        if isinstance(field_type, AnyType) and field_type.type_of_any == TypeOfAny.from_error:
            return True
        return self._DTYPE_TARGET.is_base_of(field_type)

    def _field_context(self, info: TypeInfo, name: str) -> Context:
        """Choose the node for an error about one field.

        A field written in the class itself gets the error at its own
        line. A field inherited from a base class gets it at the class,
        because mypy records the class that composes the fields as the
        source of an inherited field, not the base class.

        Args:
            info: The schema class.
            name: The name of the field.

        Returns:
            The node that mypy records as the source of the field.
        """
        data = info.typeddict_data
        if data is not None:
            source = data.field_sources.get(name)
            if source is not None:
                return source.ctx
        # mypy gives a `TypedDictType` only for a class that its semantic
        # analysis has read as a `TypedDict`. That analysis sets
        # `typeddict_data`, and its `field_sources` hold every field, own
        # or inherited. So the lookup above always finds the field.
        return info.defn  # pragma: no cover


class ClosedSchema:
    """Check a schema class, and give the closed schema that a frame type holds.

    A frame type holds its declared columns as a closed `TypedDict`, in
    which each bare data type class has its default parameters. A schema
    class such as `class Trades(Columns)` gives a `TypedDict` that is not
    closed. If the schema stays open, a schema with one more field becomes
    a subtype of the other, and mypy then drops one of them from a union of
    frame types. Every place that reads a schema class into a frame type,
    in an annotation or as a value, takes this one step, so that the two
    forms give the same type.
    """

    def __init__(self, bare_defaults: BareDtypeDefaults) -> None:
        """Keep the rule of the bare data type classes, and build the checks of a schema class.

        Args:
            bare_defaults: Gives a field with a bare data type class the
                default parameters of polars, with the builders of one phase
                of mypy.
        """
        self._bare_defaults = bare_defaults
        self._restrictions = ColumnsRestrictionCheck()

    def closed(self, ctx: _FailContext, analyzed: TypedDictType) -> TypedDictType | None:
        """Check the schema class of one frame type, and close its schema.

        Args:
            ctx: The context that mypy gives for the frame type.
            analyzed: The `TypedDict` type of the schema class.

        Returns:
            The closed schema, with the default parameters of each bare data
            type class. None after the error of a check that fails
            (`ColumnsRestrictionCheck`).
        """
        validated = self._restrictions.checked(ctx, analyzed)
        if validated is None:
            return None
        fields = validated.items.values()
        field_types = [self._bare_defaults.applied(ctx.api, field) for field in fields]
        return validated.copy_modified(is_closed=True, item_types=field_types)


class RemainderSlot:
    """Check the remainder argument of a frame annotation, and write it in canonical form.

    The remainder of an open frame is the set of columns that its type does
    not name. In `TypedDataFrame[Trades, Rest]`, `Rest` is the remainder
    argument. It can be a class declared with `other_columns(...)`, a tree
    of remainder markers over such a class, for example `Suffixed[...]`,
    or `SplitColumns[...]` whose parts are each a declared class. In a
    return type it can also be `InferredRemainder`, which the checker
    replaces at each call with the remainder of each returned value.

    The slot accepts a shape only if `RemainderCodec` can decode it, so no
    annotation carries a form that later hooks cannot read. It then writes
    the remainder in its canonical form. Two remainders that stand for the
    same columns then have equal types, which is how mypy compares them.
    """

    _SPLIT_TARGET: Final[Target] = Target.of(remainder_markers.SplitColumns)
    _INFERRED: Final[Target] = Target.of(InferredRemainder)

    def __init__(self, plugin: PluginLookup) -> None:
        """Keep the plugin, which the codec of each call uses to find marker classes.

        Args:
            plugin: The mypy plugin.
        """
        self._plugin = plugin
        self._reads = ProperReads()

    def resolved(self, ctx: AnalyzeTypeContext, arg: Type) -> MypyType | None:
        """Analyze one type argument, and accept it if it is a remainder.

        Args:
            ctx: The context that mypy gives for the annotation.
            arg: The type argument, before analysis.

        Returns:
            The remainder in canonical form, or None if the argument is
            not a remainder.
        """
        analyzed = get_proper_type(ctx.api.analyze_type(arg))
        if self._INFERRED.matches(analyzed):
            return analyzed
        if self.spelled(ctx, analyzed):
            return self.canonical(ctx, analyzed)
        return None

    def canonical(self, ctx: AnalyzeTypeContext, analyzed: ProperType) -> MypyType:
        """Write an accepted remainder in its canonical form.

        A union that a user writes keeps the order and the grouping that
        the user chose, and an effect written over a union stays outside
        it. A remainder that a hook mints has neither. Without this step,
        the two types differ although they stand for the same columns.

        Args:
            ctx: The context that mypy gives for the annotation.
            analyzed: The accepted remainder.

        Returns:
            The canonical form. `analyzed` if it is already canonical, if
            it is a split, or if the codec cannot encode the canonical form.
        """
        codec = self._codec(ctx)
        form = codec.decoded(analyzed)
        if form is None:
            return analyzed
        canonical = Carried(form).normalise()
        if canonical == form:
            return analyzed
        spelled = codec.encoded(canonical)
        return analyzed if spelled is None else spelled

    def spelled(self, ctx: AnalyzeTypeContext, analyzed: ProperType) -> bool:
        """Answer whether one analyzed type is a remainder.

        A valid split is also checked for parts that can take no column,
        and each such part is reported.

        Args:
            ctx: The context that mypy gives for the annotation.
            analyzed: The analyzed type argument.

        Returns:
            True for a remainder that the codec decodes, or for a valid
            `SplitColumns[...]`.
        """
        if self._codec(ctx).decoded(analyzed) is not None:
            return True
        if not self._is_split(analyzed):
            return False
        split = self._split_of(checked_cast(analyzed, Instance))
        if split is not None:
            self._report_dead(ctx, split)
        return True

    def _report_dead(self, ctx: AnalyzeTypeContext, split: Split) -> None:
        """Report each part of a split that can take no column.

        A column goes to the first part that accepts it. A part whose
        columns an earlier part all accepts is dead: it never gets a
        column, which is almost surely a mistake in the declaration.

        Args:
            ctx: The context that mypy gives for the annotation.
            split: The declared split.
        """
        dead = split.dead_parts()
        if not dead:
            return
        positions = ", ".join(str(index + 1) for index in dead)
        error = MarkerArgumentError(
            f"SplitColumns[...] part {positions} takes no column, because an "
            "earlier part accepts every column it accepts",
        )
        error.fail(ctx)

    @propagates(None)
    def _split_of(self, analyzed: Instance) -> Split | None:
        """Build the split that the declared parts stand for.

        Args:
            analyzed: A valid `SplitColumns[...]` instance type.

        Each part is a declared class, and its declaration states what the
        part holds.

        Returns:
            The split, or None if the record of a part gives no selector.
        """
        records: list[Record] = []
        for part in analyzed.args:
            instance = not_none(self._reads.instance(part))
            record = DeclarationCodec().record_of(instance.type)
            records.append(not_none(record))
        return SplitPredicates().split(tuple(records))

    def _codec(self, ctx: AnalyzeTypeContext) -> RemainderCodec:
        """Build the remainder codec for one annotation.

        The codec needs the interface of the current call to build
        literal types, so each call builds its own.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The codec, which reads declarations from their marker bases.
        """
        minter = MarkerMinter(self._plugin, SemanticAnalyzerLiteralTypeBuilder(), ctx.api)
        return RemainderCodec(minter, MarkedDeclarations())

    def _is_split(self, analyzed: ProperType) -> bool:
        """Answer whether one analyzed type is a valid `SplitColumns[...]`.

        Args:
            analyzed: The analyzed type argument.

        Returns:
            True for `SplitColumns[...]` with at least one part, each part
            an instance of a class declared with `other_columns(...)`.
        """
        if not isinstance(analyzed, Instance):
            return False
        if not self._SPLIT_TARGET.matches(analyzed) or not analyzed.args:
            return False
        for part in analyzed.args:
            instance = self._reads.instance(part)
            if instance is None or not MarkedDeclarations().is_declaration(instance):
                return False
        return True


class FrameTypeHook(TypeAnalysisHook):
    """Check `TypedDataFrame[...]` and `TypedLazyFrame[...]`, and give the frame one fixed shape.

    A user writes a typed frame in one of three shapes:

    - `TypedDataFrame[Schema]`: a closed frame, whose type names all of
      its columns. `Schema` is a schema class, usually one that inherits
      `Columns`, and its fields are the declared columns.
    - `TypedDataFrame[Schema, Remainder]`: an open frame. `Remainder`
      stands for the columns that the schema does not name.
    - `TypedDataFrame[Remainder]`: an open frame with no declared columns.

    Each hook that reads a frame later expects one shape. So this hook
    always gives two type arguments: the declared columns as a closed
    `TypedDict`, and the remainder in canonical form, or None for a closed
    frame. `TypedDataFrame[()]` is a frame of no column. A frame class with
    no argument at all names no columns, so it is the plain frame of polars.

    The plugin builds one instance for each of the two frame classes.
    """

    _TYPED_DICT_FALLBACK: Final[str] = "typing._TypedDict"

    def __init__(self, owner: type, plugin: PluginLookup, closed_schema: ClosedSchema) -> None:
        """Keep the frame class of this hook, and build its checks.

        Args:
            owner: `TypedDataFrame` or `TypedLazyFrame`.
            plugin: The mypy plugin, which the remainder check needs to
                find marker classes.
            closed_schema: Checks a schema class and closes its schema, with
                the default parameters of each bare data type class.
        """
        self._closed_schema = closed_schema
        self._fullname = Target.of(owner).fullname
        polars_bases = [base for base in owner.__mro__ if base.__module__.startswith("polars.")]
        self._plain = Target.of(polars_bases[0]).fullname
        self._remainder = RemainderSlot(plugin)

    def __call__(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Check one frame annotation.

        `TypedDataFrame[()]` is a frame of no column. A frame class with no
        argument at all, as in `type[TypedDataFrame]`, names no columns, so
        it is the plain frame of polars that it extends, as `pl.DataFrame`.
        If the first argument is a `TypedDict`, it is the schema. Else it must
        be a lone remainder, which gives an open frame with no declared
        columns.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            The frame type with exactly two type arguments, the plain frame
            for a frame class with no argument, or `Any` after an error.
        """
        args = ctx.type.args
        if not args and ctx.type.empty_tuple_index:
            return ctx.api.named_type(self._fullname, [self._no_columns(ctx), NoneType()])
        if not args:
            return ctx.api.named_type(self._plain, [])
        if len(args) > 2:
            return self._malformed(ctx)
        first = get_proper_type(ctx.api.analyze_type(args[0]))
        if isinstance(first, TypedDictType):
            return self._built_from_columns_class(ctx, args, first)
        if len(args) != 1 or not self._remainder.spelled(ctx, first):
            return self._malformed(ctx)
        remainder = self._remainder.canonical(ctx, first)
        return ctx.api.named_type(self._fullname, [self._no_columns(ctx), remainder])

    def _no_columns(self, ctx: AnalyzeTypeContext) -> TypedDictType:
        """Build the empty schema of a frame with no declared columns.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            A closed `TypedDict` type with no fields.
        """
        fallback = ctx.api.named_type(self._TYPED_DICT_FALLBACK, [])
        return TypedDictType({}, set(), set(), fallback, is_closed=True)

    def _malformed(self, ctx: AnalyzeTypeContext) -> MypyType:
        """Report a frame annotation whose shape is none of the three.

        Args:
            ctx: The context that mypy gives for the annotation.

        Returns:
            `Any`, marked as the result of an error.
        """
        error = MarkerArgumentError(
            "TypedDataFrame[...]/TypedLazyFrame[...] takes a schema class, then an "
            "optional remainder: declare the columns in a class that inherits Columns, "
            'or with columns("Name", {...}) for names that are not identifiers',
        )
        error.fail(ctx)
        return AnyType(TypeOfAny.from_error)

    def _built_from_columns_class(
        self,
        ctx: AnalyzeTypeContext,
        args: tuple[Type, ...],
        analyzed: TypedDictType,
    ) -> MypyType:
        """Check and build a frame whose first argument is a schema class.

        Args:
            ctx: The context that mypy gives for the annotation.
            args: The type arguments before analysis.
            analyzed: The `TypedDict` type of the first argument.

        Each field with a bare data type class gets the default parameters
        of polars. For example, a bare `pl.Decimal` becomes `Decimal[38, 0]`.

        Returns:
            The frame type. `Any` after an error if the schema class fails
            `ColumnsRestrictionCheck`, or the second argument is not a
            remainder.
        """
        closed = self._closed_schema.closed(ctx, analyzed)
        if closed is None:
            return AnyType(TypeOfAny.from_error)
        rest_type = self._resolved_rest_argument(ctx, args)
        if rest_type is None:
            return AnyType(TypeOfAny.from_error)
        return ctx.api.named_type(self._fullname, [closed, rest_type])

    def _resolved_rest_argument(
        self,
        ctx: AnalyzeTypeContext,
        args: tuple[Type, ...],
    ) -> MypyType | None:
        """Read the second argument of a frame annotation as its remainder.

        Args:
            ctx: The context that mypy gives for the annotation.
            args: The type arguments before analysis.

        Returns:
            None as a type, which marks a closed frame, if there is only
            one argument. Else the remainder in canonical form. None after
            a `MarkerArgumentError` if the second argument is not a
            remainder.
        """
        if len(args) == 1:
            return NoneType()
        analyzed = self._remainder.resolved(ctx, args[1])
        if analyzed is not None:
            return analyzed
        error = MarkerArgumentError(
            "TypedDataFrame[...]/TypedLazyFrame[...]'s second argument must be a "
            "remainder: a class declared with other_columns(...), a remainder "
            "marker over one, or SplitColumns[...]",
        )
        error.fail(ctx)
        return None
