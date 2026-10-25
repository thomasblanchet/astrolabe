"""The definitions of the functions of the `dt` namespace, and of the functions of business days.

The `dt` namespace holds the functions of dates, times and durations. The
identity of such a node names the namespace, for example
`TemporalExpr.CastTimeUnit`. Polars declares every one of them elementwise,
with no casting rule. Most of them read one input and give a part of a moment,
such as its year. The others read a unit, a zone or a format from the options
of their node, or read more inputs: the duration of `truncate`, the parts of
`replace` and the time of `combine`.

The functions of business days, `dt.add_business_days`, `dt.is_business_day`
and `pl.business_day_count`, are in a namespace of their own, `Business`. They
read their dates and a list of holidays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.cast_pass import DurationCasts, HolidaysCasts
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import (
    AllInputs,
    LiteralInputs,
    OneInput,
)
from astrolabe._rules.definitions.node_options import (
    FromOptions,
    NodeIdentity,
    NodeOption,
    TemporalOptions,
)
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.dates import (
    BusinessKernels,
    DateKernels,
    EveryKernel,
    OffsetBy,
    Replace,
)
from astrolabe._rules.kernels.temporal import TemporalParts
from astrolabe._rules.options import FunctionFlag, FunctionOptions, KernelLength, Shape
from astrolabe._rules.type_functions import (
    FieldsMapper,
    TemporalClosures,
)

if TYPE_CHECKING:
    from astrolabe._rules.kernels.dispatch import Dispatch


class PartNode:
    """Build the definition of a function of the `dt` namespace that gives a fixed type.

    The functions that give a part of a moment, such as `dt.year` or
    `dt.hour`, have the same parts: the type function `with_dtype` of a fixed
    type, the options of an elementwise function, no casting rule, and a
    kernel of one input that reads the kinds that hold the part. So one call
    builds the definition of each of them.

    Example:
        >>> PartNode().of("Year", DType("Int32"), DateKernels.YEAR).identity
        NodeIdentity(shape='Function', function='TemporalExpr.Year')
    """

    def of(self, function: str, dtype: DType, kernel: Dispatch) -> Definition:
        """Give the definition of one function.

        Args:
            function: The variant of the function in the namespace, as polars names it.
            dtype: The type that the type function gives.
            kernel: The dispatch of the kernel.

        Returns:
            The definition.

        Example:
            >>> PartNode().of("Hour", DType("Int8"), DateKernels.HOUR).type_function
            Mapper(with_dtype, dtype=Int8)
        """
        return Definition(
            identity=NodeIdentity("Function", f"TemporalExpr.{function}"),
            type_function=Mapper(FieldsMapper.with_dtype, dtype=dtype),
            options=Shape.function(FunctionOptions.elementwise()),
            casts=FunctionCasting(),
            kernel=OneInput(kernel),
            nulls=NullRule.FIRST,
        )


class TotalNode:
    """Build the definition of a total of a `Duration`, such as `dt.total_days`.

    The totals differ only by their unit, which the variant of the function
    names. Each one reads the option `fractional`: its type is `Float64` where
    the count keeps its fraction, and `Int64` otherwise. So one call builds
    the definition of each of them, and the definition reads the option.

    Example:
        >>> TotalNode().of("TotalHours").identity
        NodeIdentity(shape='Function', function='TemporalExpr.TotalHours')
    """

    def of(self, function: str) -> Definition:
        """Give the definition of one total.

        Args:
            function: The variant of the function in the namespace, as polars names it.

        Returns:
            The definition.

        Example:
            >>> from astrolabe._rules.definitions.node_options import NodeOptions
            >>> function = {"TemporalExpr": {"TotalDays": {"fractional": True}}}
            >>> node = NodeOptions({"Function": {"function": function}})
            >>> TotalNode().of("TotalDays").at(node).type_function
            Mapper(with_dtype, dtype=Float64)
        """
        fractional = NodeOption.flag("function", "TemporalExpr", function, "fractional")
        return Definition(
            identity=NodeIdentity("Function", f"TemporalExpr.{function}"),
            type_function=Mapper(
                FieldsMapper.with_dtype,
                dtype=FromOptions(TemporalClosures().total, fractional),
            ),
            options=Shape.function(FunctionOptions.elementwise()),
            casts=FunctionCasting(),
            kernel=OneInput(FromOptions(DateKernels().total, fractional)),
            nulls=NullRule.FIRST,
        )


class Temporal:
    """Hold the definitions of the `dt` namespace, of `pl.datetime` and of `pl.duration`.

    Example:
        >>> Temporal.CAST_TIME_UNIT.identity
        NodeIdentity(shape='Function', function='TemporalExpr.CastTimeUnit')
    """

    CAST_TIME_UNIT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.CastTimeUnit"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                TemporalClosures().with_time_unit,
                NodeOption.unit("function", "TemporalExpr", "CastTimeUnit"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                DateKernels().cast_time_unit,
                NodeOption.unit("function", "TemporalExpr", "CastTimeUnit"),
            ),
        ),
    )
    WITH_TIME_UNIT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.WithTimeUnit"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                TemporalClosures().with_time_unit,
                NodeOption.unit("function", "TemporalExpr", "WithTimeUnit"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                DateKernels().with_time_unit,
                NodeOption.unit("function", "TemporalExpr", "WithTimeUnit"),
            ),
        ),
    )
    CONVERT_TIME_ZONE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.ConvertTimeZone"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                TemporalClosures().convert_time_zone,
                NodeOption.text("function", "TemporalExpr", "ConvertTimeZone", "inner"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                DateKernels().convert_time_zone,
                NodeOption.text("function", "TemporalExpr", "ConvertTimeZone", "inner"),
            ),
        ),
    )
    MILLENNIUM: ClassVar[Definition] = PartNode().of(
        "Millennium",
        DType("Int32"),
        DateKernels.MILLENNIUM,
    )
    CENTURY: ClassVar[Definition] = PartNode().of("Century", DType("Int32"), DateKernels.CENTURY)
    YEAR: ClassVar[Definition] = PartNode().of("Year", DType("Int32"), DateKernels.YEAR)
    ISO_YEAR: ClassVar[Definition] = PartNode().of("IsoYear", DType("Int32"), DateKernels.ISO_YEAR)
    IS_LEAP_YEAR: ClassVar[Definition] = PartNode().of(
        "IsLeapYear",
        DType("Boolean"),
        DateKernels.IS_LEAP_YEAR,
    )
    QUARTER: ClassVar[Definition] = PartNode().of("Quarter", DType("Int8"), DateKernels.QUARTER)
    MONTH: ClassVar[Definition] = PartNode().of("Month", DType("Int8"), DateKernels.MONTH)
    DAYS_IN_MONTH: ClassVar[Definition] = PartNode().of(
        "DaysInMonth",
        DType("Int8"),
        DateKernels.DAYS_IN_MONTH,
    )
    WEEK: ClassVar[Definition] = PartNode().of("Week", DType("Int8"), DateKernels.WEEK)
    WEEKDAY: ClassVar[Definition] = PartNode().of("WeekDay", DType("Int8"), DateKernels.WEEKDAY)
    DAY: ClassVar[Definition] = PartNode().of("Day", DType("Int8"), DateKernels.DAY)
    ORDINAL_DAY: ClassVar[Definition] = PartNode().of(
        "OrdinalDay",
        DType("Int16"),
        DateKernels.ORDINAL_DAY,
    )
    HOUR: ClassVar[Definition] = PartNode().of("Hour", DType("Int8"), DateKernels.HOUR)
    MINUTE: ClassVar[Definition] = PartNode().of("Minute", DType("Int8"), DateKernels.MINUTE)
    SECOND: ClassVar[Definition] = PartNode().of("Second", DType("Int8"), DateKernels.SECOND)
    MILLISECOND: ClassVar[Definition] = PartNode().of(
        "Millisecond",
        DType("Int32"),
        DateKernels.MILLISECOND,
    )
    MICROSECOND: ClassVar[Definition] = PartNode().of(
        "Microsecond",
        DType("Int32"),
        DateKernels.MICROSECOND,
    )
    NANOSECOND: ClassVar[Definition] = PartNode().of(
        "Nanosecond",
        DType("Int32"),
        DateKernels.NANOSECOND,
    )
    TIME: ClassVar[Definition] = PartNode().of("Time", DType("Time"), DateKernels.TIME)
    DATE: ClassVar[Definition] = PartNode().of("Date", DType("Date"), DateKernels.DATE)
    DATETIME: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Datetime"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.try_map_dtype, func=TemporalClosures().naive),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(DateKernels.DATETIME),
    )
    DATETIME_FUNCTION: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.DatetimeFunction"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption(
                ("function", "TemporalExpr", "DatetimeFunction"),
                TemporalOptions().datetime,
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(
                TemporalParts.datetime,
                NodeOption(
                    ("function", "TemporalExpr", "DatetimeFunction"), TemporalOptions().datetime
                ),
            ),
        ),
    )
    DURATION: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Duration"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption(("function", "TemporalExpr", "Duration"), TemporalOptions().duration),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(DurationCasts()),
        kernel=AllInputs(
            FromOptions(
                TemporalParts.duration,
                NodeOption(("function", "TemporalExpr", "Duration"), TemporalOptions().duration),
            ),
        ),
    )
    MONTH_START: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.MonthStart"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(DateKernels.MONTH_START),
    )
    MONTH_END: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.MonthEnd"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(DateKernels.MONTH_END),
    )
    BASE_UTC_OFFSET: ClassVar[Definition] = PartNode().of(
        "BaseUtcOffset",
        DType("Duration", ("ms",)),
        DateKernels().offset("base_utc_offset"),
    )
    DST_OFFSET: ClassVar[Definition] = PartNode().of(
        "DSTOffset",
        DType("Duration", ("ms",)),
        DateKernels().offset("dst_offset"),
    )
    TIMESTAMP: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.TimeStamp"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Int64")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                DateKernels().timestamp,
                NodeOption.unit("function", "TemporalExpr", "TimeStamp"),
            ),
        ),
    )
    TO_STRING: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.ToString"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                DateKernels().to_string,
                NodeOption.text("function", "TemporalExpr", "ToString"),
            ),
        ),
    )
    TOTAL_DAYS: ClassVar[Definition] = TotalNode().of("TotalDays")
    TOTAL_HOURS: ClassVar[Definition] = TotalNode().of("TotalHours")
    TOTAL_MINUTES: ClassVar[Definition] = TotalNode().of("TotalMinutes")
    TOTAL_SECONDS: ClassVar[Definition] = TotalNode().of("TotalSeconds")
    TOTAL_MILLISECONDS: ClassVar[Definition] = TotalNode().of("TotalMilliseconds")
    TOTAL_MICROSECONDS: ClassVar[Definition] = TotalNode().of("TotalMicroseconds")
    TOTAL_NANOSECONDS: ClassVar[Definition] = TotalNode().of("TotalNanoseconds")
    TRUNCATE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Truncate"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(EveryKernel("truncate")),
    )
    ROUND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Round"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(EveryKernel("round")),
    )
    OFFSET_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.OffsetBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(OffsetBy()),
    )
    REPLACE_TIME_ZONE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.ReplaceTimeZone"),
        type_function=Mapper(
            FieldsMapper.map_datetime_dtype_timezone,
            time_zone=NodeOption.zone("function", "TemporalExpr", "ReplaceTimeZone", 0),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                DateKernels().replace_time_zone,
                NodeOption.zone("function", "TemporalExpr", "ReplaceTimeZone", 0),
            ),
        ),
    )
    COMBINE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Combine"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                TemporalClosures().combine,
                NodeOption.unit("function", "TemporalExpr", "Combine"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(
                DateKernels().combine,
                NodeOption.unit("function", "TemporalExpr", "Combine"),
            ),
        ),
    )
    REPLACE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TemporalExpr.Replace"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise(), KernelLength.first()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Replace()),
    )
    IS_BUSINESS_DAY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Business.IsBusinessDay"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(HolidaysCasts(1)),
        kernel=AllInputs(
            FromOptions(
                BusinessKernels().is_business_day,
                NodeOption.flags("function", "Business", "IsBusinessDay", "week_mask"),
            ),
        ),
    )
    ADD_BUSINESS_DAYS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Business.AddBusinessDay"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(HolidaysCasts(2)),
        kernel=AllInputs(
            FromOptions(
                BusinessKernels().add_business_days,
                NodeOption.flags("function", "Business", "AddBusinessDay", "week_mask"),
            ),
        ),
    )
    BUSINESS_DAY_COUNT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Business.BusinessDayCount"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Int32")),
        options=Shape.function(FunctionOptions.elementwise().flag(FunctionFlag.ALLOW_RENAME)),
        casts=OwnRuleCasts(HolidaysCasts(2)),
        kernel=AllInputs(
            FromOptions(
                BusinessKernels().business_day_count,
                NodeOption.flags("function", "Business", "BusinessDayCount", "week_mask"),
            ),
        ),
    )
