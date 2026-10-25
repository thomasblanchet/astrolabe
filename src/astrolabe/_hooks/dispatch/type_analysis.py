"""Register the hooks that mypy calls when it reads a type that the user writes.

mypy calls a type-analyze hook when it reads a type in an annotation,
for example `Column[Literal["a"], pl.Int64]`. The hook checks the type
arguments and builds the type that the rest of the plugin reads. For
example, a bare `pl.Datetime` gets the default parameters of polars.

This module registers one hook for each marker type of the project
(`Column`, `List`, `Array`, `Struct`, `Datetime`, `Duration`, `Decimal`,
`Enum`, `Suffixed` and `Prefixed`), one hook each for
`TypedDataFrame` and `TypedLazyFrame`, and one for each placeholder of an
inferred return type.
"""

from __future__ import annotations

from mypy.plugin import AnalyzeTypeContext
from mypy.types import Type as MypyType

from typing import TYPE_CHECKING

from astrolabe._hooks.dispatch.generic import HookDispatcher, Target
from astrolabe._hooks.calls.dtype_constructor_readers import SemanticAnalyzerLiteralTypeBuilder
from astrolabe._hooks.fresh_type_info import FreshTypeInfo
from astrolabe._hooks.inferred.interface import PlaceholderTypeHook
from astrolabe._hooks.type_analysis import (
    ArrayTypeHook,
    BareDtypeDefaults,
    ClosedSchema,
    ColumnTypeHook,
    DatetimeTypeHook,
    DecimalShapeValidator,
    DecimalTypeHook,
    DurationTypeHook,
    EnumTypeHook,
    FrameTypeHook,
    GroupByTypeHook,
    ImplicitDecimalPrecision,
    ListTypeHook,
    PrefixedTypeHook,
    StructTypeHook,
    SuffixedTypeHook,
    TimeUnitValidator,
)
from astrolabe.declare import InferredDataFrame, InferredExpr, InferredLazyFrame, InferredRemainder
from astrolabe.remainder import Prefixed, Suffixed
from astrolabe.frame import (
    Array,
    Column,
    Datetime,
    Decimal,
    Duration,
    Enum,
    List,
    Struct,
    TypedDataFrame,
    TypedGroupBy,
    TypedLazyFrame,
    TypedLazyGroupBy,
)


if TYPE_CHECKING:
    from astrolabe._hooks.domain.frame import PluginLookup


class TypeAnalyzeHooks:
    """Build the type-analyze hooks, and register each one by the full name of its class.

    The plugin asks this class for the hook of a type when mypy reads
    that type in an annotation. The class also builds the helpers that
    several hooks share, so that a bare data type, such as `pl.Decimal`,
    gets the same default parameters in every place.
    """

    def __init__(self, plugin: PluginLookup) -> None:
        """Build every type-analyze hook, and register it in a new dispatcher.

        Args:
            plugin: The mypy plugin, which the hooks for `Struct` and
                for the frames use to look up a class by its full name.
        """
        self._dispatcher = HookDispatcher[AnalyzeTypeContext, MypyType]()
        literal_type = SemanticAnalyzerLiteralTypeBuilder()
        implicit_precision = ImplicitDecimalPrecision(literal_type)
        bare_defaults = BareDtypeDefaults(implicit_precision, literal_type, FreshTypeInfo(plugin))
        closed_schema = ClosedSchema(bare_defaults)
        self._dispatcher.register(Target.of(Column), ColumnTypeHook(bare_defaults))
        self._dispatcher.register(Target.of(List), ListTypeHook())
        self._dispatcher.register(Target.of(Array), ArrayTypeHook())
        self._dispatcher.register(Target.of(Struct), StructTypeHook(plugin))
        time_unit = TimeUnitValidator()
        self._dispatcher.register(Target.of(Datetime), DatetimeTypeHook(time_unit))
        self._dispatcher.register(Target.of(Duration), DurationTypeHook(time_unit))
        self._dispatcher.register(
            Target.of(Decimal),
            DecimalTypeHook(DecimalShapeValidator(), implicit_precision),
        )
        self._dispatcher.register(Target.of(Enum), EnumTypeHook())
        self._dispatcher.register(Target.of(Suffixed), SuffixedTypeHook())
        self._dispatcher.register(Target.of(Prefixed), PrefixedTypeHook())
        self._dispatcher.register(
            Target.of(TypedDataFrame),
            FrameTypeHook(TypedDataFrame, plugin, closed_schema),
        )
        self._dispatcher.register(
            Target.of(TypedLazyFrame),
            FrameTypeHook(TypedLazyFrame, plugin, closed_schema),
        )
        for group_by in (TypedGroupBy, TypedLazyGroupBy):
            self._dispatcher.register(Target.of(group_by), GroupByTypeHook(Target.of(group_by)))
        placeholders = (InferredExpr, InferredDataFrame, InferredLazyFrame, InferredRemainder)
        for placeholder in placeholders:
            target = Target.of(placeholder)
            self._dispatcher.register(target, PlaceholderTypeHook(target))

    @property
    def dispatcher(self) -> HookDispatcher[AnalyzeTypeContext, MypyType]:
        """The dispatcher that holds every registered type-analyze hook."""
        return self._dispatcher
