"""The definitions of the range functions, such as `int_range` and `date_ranges`.

A function of one range, such as `int_range`, `linear_space`, `date_range`,
`datetime_range` or `time_range`, gives a column of the values between its
bounds. Polars requires each bound to give one value when it converts the
expression, and the number of rows depends on the values. A function of many
ranges, such as `int_ranges`, reads its bounds row by row and gives a list
for each row. The cast pass has a rule of its own for the integer ranges and
for the ranges of dates and of datetimes (`IntRangeCasts`,
`TemporalRangeCasts`). The ranges of times and the linear spaces have none,
and their kernels cast or read their inputs (`TemporalRange`, `LinearSpace`).
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import IntRangeCasts, TemporalRangeCasts
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import AllInputs, LiteralInputs
from astrolabe._rules.definitions.node_options import (
    FromOptions,
    NodeIdentity,
    NodeOption,
    OptionReader,
)
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.windows import WindowDuration
from astrolabe._rules.kernels.ranges import (
    IntRange,
    IntRanges,
    LinearSpace,
    LinearSpaces,
    RangeKernels,
)
from astrolabe._rules.options import (
    FunctionFlag,
    FunctionOptions,
    KernelLength,
    ScalarInput,
    Shape,
)
from astrolabe._rules.type_functions import AggFields, FieldsMapper


class Ranges:
    """Hold the definitions of the range functions.

    Example:
        >>> Ranges.INT_RANGE.identity
        NodeIdentity(shape='Function', function='Range.IntRange')
    """

    RENAMED: ClassVar[FunctionFlag] = FunctionFlag.ALLOW_RENAME
    RANGED: ClassVar[FunctionFlag] = FunctionFlag.ALLOW_RENAME | FunctionFlag.RANGE
    ONE_RANGE: ClassVar[Shape] = Shape.function(
        FunctionOptions.groupwise().flag(RANGED),
        KernelLength.values(),
    )
    ONE_SPAN: ClassVar[Shape] = Shape.function(
        FunctionOptions.groupwise().flag(RENAMED),
        KernelLength.values(),
    )
    MANY_RANGES: ClassVar[Shape] = Shape.function(FunctionOptions.elementwise().flag(RANGED))
    MANY_SPANS: ClassVar[Shape] = Shape.function(FunctionOptions.elementwise().flag(RENAMED))
    DATE: ClassVar[DType] = DType("Date")
    TIME: ClassVar[DType] = DType("Time")

    INT_RANGE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.IntRange"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption.dtype("function", "Range", "IntRange", "dtype"),
        ),
        options=ONE_RANGE.with_scalar_inputs(
            ScalarInput.passed(0, "start", "int_range"),
            ScalarInput.passed(1, "stop", "int_range"),
        ),
        casts=OwnRuleCasts(
            FromOptions(IntRangeCasts, NodeOption.dtype("function", "Range", "IntRange", "dtype")),
        ),
        kernel=AllInputs(
            FromOptions(
                IntRange,
                NodeOption.dtype("function", "Range", "IntRange", "dtype"),
                NodeOption.integer("function", "Range", "IntRange", "step"),
            ),
        ),
    )
    INT_RANGES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.IntRanges"),
        nulls=NullRule.ANY,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                AggFields().implode,
                NodeOption.dtype("function", "Range", "IntRanges", "dtype"),
            ),
        ),
        options=MANY_SPANS,
        casts=OwnRuleCasts(IntRangeCasts(DType("Int64"), ("start", "end", "step"))),
        kernel=LiteralInputs(
            FromOptions(IntRanges, NodeOption.dtype("function", "Range", "IntRanges", "dtype")),
        ),
    )
    LINEAR_SPACE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.LinearSpace"),
        type_function=Mapper(FieldsMapper.linear_space_dtype),
        options=Shape.function(
            FunctionOptions.groupwise().flag(RANGED),
            KernelLength.values(),
        ).with_scalar_inputs(
            ScalarInput.passed(0, "start", "linear_space"),
            ScalarInput.passed(1, "end", "linear_space"),
            ScalarInput.passed(2, "num_samples", "linear_space"),
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(LinearSpace()),
    )
    LINEAR_SPACES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.LinearSpaces"),
        nulls=NullRule.ANY,
        type_function=Mapper(
            FieldsMapper.linear_spaces_dtype,
            array_width=NodeOption(
                ("function", "Range", "LinearSpaces", "array_width"),
                OptionReader().integer_or_none,
            ),
        ),
        options=MANY_SPANS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                LinearSpaces,
                NodeOption(
                    ("function", "Range", "LinearSpaces", "array_width"),
                    OptionReader().integer_or_none,
                ),
            ),
        ),
    )
    DATE_RANGE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.DateRange"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DATE),
        options=ONE_RANGE.with_scalar_inputs(
            ScalarInput.passed(0, "start", "date_range"),
            ScalarInput.passed(1, "end", "date_range"),
        ),
        casts=OwnRuleCasts(TemporalRangeCasts.dates()),
        kernel=LiteralInputs(RangeKernels.DATE),
    )
    DATE_RANGES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.DateRanges"),
        nulls=NullRule.ANY,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (DATE,))),
        options=MANY_RANGES,
        casts=OwnRuleCasts(TemporalRangeCasts.dates()),
        kernel=LiteralInputs(RangeKernels.DATES),
    )
    DATETIME_RANGE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.DatetimeRange"),
        type_function=Mapper(
            FieldsMapper.map_to_datetime_range_dtype,
            time_unit=NodeOption(
                ("function", "Range", "DatetimeRange", "time_unit"),
                OptionReader().unit_or_none,
            ),
            time_zone=NodeOption.zone("function", "Range", "DatetimeRange", "time_zone"),
        ),
        options=ONE_SPAN.with_scalar_inputs(
            ScalarInput.passed(0, "start", "datetime_range"),
            ScalarInput.passed(1, "end", "datetime_range"),
        ),
        casts=OwnRuleCasts(
            FromOptions(
                TemporalRangeCasts.datetimes,
                NodeOption(
                    ("function", "Range", "DatetimeRange", "time_unit"),
                    OptionReader().unit_or_none,
                ),
                NodeOption.zone("function", "Range", "DatetimeRange", "time_zone"),
                NodeOption(("function", "Range", "DatetimeRange", "interval"), WindowDuration.of),
            ),
        ),
        kernel=LiteralInputs(RangeKernels.DATETIME),
    )
    DATETIME_RANGES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.DatetimeRanges"),
        nulls=NullRule.ANY,
        type_function=Mapper(
            FieldsMapper.map_to_datetime_ranges_dtype,
            time_unit=NodeOption(
                ("function", "Range", "DatetimeRanges", "time_unit"),
                OptionReader().unit_or_none,
            ),
            time_zone=NodeOption.zone("function", "Range", "DatetimeRanges", "time_zone"),
        ),
        options=MANY_RANGES,
        casts=OwnRuleCasts(
            FromOptions(
                TemporalRangeCasts.datetimes,
                NodeOption(
                    ("function", "Range", "DatetimeRanges", "time_unit"),
                    OptionReader().unit_or_none,
                ),
                NodeOption.zone("function", "Range", "DatetimeRanges", "time_zone"),
                NodeOption(("function", "Range", "DatetimeRanges", "interval"), WindowDuration.of),
            ),
        ),
        kernel=LiteralInputs(RangeKernels.DATETIMES),
    )
    TIME_RANGE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.TimeRange"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=TIME),
        options=ONE_SPAN.with_scalar_inputs(
            ScalarInput.passed(0, "start", "time_range"),
            ScalarInput.passed(1, "end", "time_range"),
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(RangeKernels.TIME),
    )
    TIME_RANGES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Range.TimeRanges"),
        nulls=NullRule.ANY,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("List", (TIME,))),
        options=MANY_RANGES,
        casts=FunctionCasting(),
        kernel=LiteralInputs(RangeKernels.TIMES),
    )
