"""The definitions of the aggregations, such as `sum`, `mean`, `any` or `quantile`.

An aggregation gives one value for each group. In a `select`, the whole frame
is one group, so the aggregation gives one row. In `group_by().agg()`, its
kernel runs on each group of the frame, and gives one row for each group.

Polars stores most aggregations as nodes of their own kind (`Agg`), and the
others as functions whose options give one value (`aggregation()`). The run
of a function in `agg` depends on the function: some run as an aggregation,
some have a kernel of their own for the groups, and `min_by` and `max_by`
have a physical expression of their own (`Definition.physical`).
`lower_bound` and `upper_bound` become a literal when polars converts the
expression.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import CastOption, FirstInputCast
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, NoCasts, OwnRuleCasts
from astrolabe._rules.definitions.value_adapters import KeptValue
from astrolabe._rules.definitions.definition import DdofNulls, Definition, NullRule, ShapeSince
from astrolabe._rules.definitions.kernel_adapters import (
    AllInputs,
    ItemReduction,
    LiteralReduction,
    OneInput,
    PlannedInput,
    Reduction,
    TwoInputs,
)
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.overrides import CountOfNullLiteral
from astrolabe._rules.definitions.type_function_adapters import AggField, ExprField, Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import PhysicalExpr
from astrolabe._rules.kernels.numeric import NumericKernels
from astrolabe._rules.kernels.reductions import (
    Correlation,
    GroupQuantile,
    IndexOf,
    Quantile,
    ReductionKernels,
    ValueAtKey,
)
from astrolabe._rules.options import (
    CastingRules,
    FunctionFlag,
    FunctionOptions,
    NodeKind,
    ScalarInput,
    Shape,
)
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import AggFields, ExprFields, FieldsMapper, FunctionClosures


class Aggregations:
    """Hold the definitions of the aggregations.

    Example:
        >>> Aggregations.SUM.identity
        NodeIdentity(shape='Agg', function='Sum')
        >>> Aggregations.MIN_BY.physical
        <PhysicalExpr.MIN_MAX_BY: 'min_max_by'>
    """

    AGG: ClassVar[Shape] = Shape.of(NodeKind.AGG)
    SCALAR: ClassVar[Shape] = Shape.function(FunctionOptions.aggregation())
    UNORDERED: ClassVar[Shape] = Shape.function(
        FunctionOptions.aggregation().flag(FunctionFlag.NON_ORDER_OBSERVING),
    )
    INDEXED: ClassVar[Shape] = Shape.function(
        FunctionOptions.aggregation().with_casting_rules(CastingRules.first_arg_lossless()),
    )
    SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Sum"),
        type_function=AggField(AggFields.sum),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(NumericKernels.SUM, NumericKernels.SUM_GROUPS),
    )
    MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Min"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.EXTREMUM, ReductionKernels.SAME),
    )
    MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Max"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.EXTREMUM, ReductionKernels.SAME),
    )
    FIRST: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "First"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.SAME, ReductionKernels.SAME),
    )
    LAST: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Last"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.SAME, ReductionKernels.SAME),
    )
    FIRST_NON_NULL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "FirstNonNull"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.SAME, ReductionKernels.SAME),
    )
    LAST_NON_NULL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "LastNonNull"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.SAME, ReductionKernels.SAME),
    )
    ITEM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Item"),
        computes=KeptValue(),
        type_function=AggField(AggFields.same),
        options=AGG,
        casts=NoCasts(),
        kernel=ItemReduction(
            Reduction(ReductionKernels.ITEM, ReductionKernels.ITEM),
            allow_empty=NodeOption.flag("Item", "allow_empty"),
        ),
    )
    MEAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Mean"),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.mean),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(NumericKernels.MEAN, ReductionKernels.MOMENT_GROUPS),
    )
    MEDIAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Median"),
        nulls=NullRule.FIRST,
        type_function=AggField(AggFields.mean),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.MEDIAN, ReductionKernels.MOMENT_GROUPS),
    )
    STD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Std"),
        nulls=NullRule.FIRST,
        short=DdofNulls(NodeOption.integer("Std", 1)),
        type_function=AggField(AggFields.mean),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.STD, ReductionKernels.STD_GROUPS),
    )
    VAR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Var"),
        nulls=NullRule.FIRST,
        short=DdofNulls(NodeOption.integer("Var", 1)),
        type_function=AggField(AggFields.var),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.VAR, ReductionKernels.STD_GROUPS),
    )
    N_UNIQUE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "NUnique"),
        type_function=AggField(AggFields.count),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.COUNT, ReductionKernels.COUNT),
    )
    COUNT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Count"),
        type_function=AggField(AggFields.count),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.COUNT, ReductionKernels.COUNT),
    )
    IMPLODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "Implode"),
        type_function=AggField(AggFields.implode),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.IMPLODE, ReductionKernels.IMPLODE),
        physical=PhysicalExpr.IMPLODE,
    )
    AGG_GROUPS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Agg", "AggGroups"),
        type_function=AggField(AggFields.agg_groups),
        options=AGG,
        casts=NoCasts(),
        kernel=Reduction(ReductionKernels.AGG_GROUPS_WHOLE, ReductionKernels.AGG_GROUPS),
    )
    APPROX_N_UNIQUE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ApproxNUnique"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=OneInput(ReductionKernels.APPROX_N_UNIQUE),
    )
    ARG_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArgMax"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.COUNT, ReductionKernels.COUNT),
        physical=PhysicalExpr.AGGREGATION,
    )
    ARG_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArgMin"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.COUNT, ReductionKernels.COUNT),
        physical=PhysicalExpr.AGGREGATION,
    )
    BITWISE_AND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.And"),
        type_function=Mapper(FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("and")),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.BITWISE, ReductionKernels.SAME),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    BITWISE_OR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.Or"),
        type_function=Mapper(FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("or")),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.BITWISE, ReductionKernels.SAME),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    BITWISE_XOR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.Xor"),
        type_function=Mapper(FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("xor")),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.BITWISE, ReductionKernels.SAME),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    ALL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.All"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=UNORDERED,
        casts=OwnRuleCasts(FirstInputCast(DType("Boolean"), CastOption.NON_STRICT)),
        kernel=Reduction(ReductionKernels.ANY_ALL, ReductionKernels.ANY_ALL),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    ANY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.Any"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=UNORDERED,
        casts=OwnRuleCasts(FirstInputCast(DType("Boolean"), CastOption.NON_STRICT)),
        kernel=Reduction(ReductionKernels.ANY_ALL, ReductionKernels.ANY_ALL),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    HAS_NULLS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.HasNulls"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.BOOLEAN, ReductionKernels.BOOLEAN),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    IS_EMPTY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsEmpty"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=CountOfNullLiteral(
            Reduction(ReductionKernels.BOOLEAN, ReductionKernels.BOOLEAN),
            ignore_nulls=NodeOption.flag("function", "Boolean", "IsEmpty", "ignore_nulls"),
        ),
        physical=PhysicalExpr.COUNT_GROUPS,
    )
    IS_SORTED: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsSorted"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                ReductionKernels().is_sorted,
                NodeOption.flag_or_none("function", "Boolean", "IsSorted", "descending"),
            ),
        ),
    )
    NULL_COUNT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "NullCount"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=Reduction(ReductionKernels.COUNT, ReductionKernels.COUNT),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    ENTROPY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Entropy"),
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=OneInput(ReductionKernels.ENTROPY),
    )
    PRODUCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Product"),
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().product),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=OneInput(ReductionKernels.PRODUCT),
    )
    SKEW: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Skew"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Float64")),
        options=UNORDERED,
        casts=OwnRuleCasts(FirstInputCast(DType("Float64"), CastOption.STRICT)),
        kernel=Reduction(ReductionKernels.SKEW, ReductionKernels.SKEW_GROUPS),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    KURTOSIS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Kurtosis"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Float64")),
        options=UNORDERED,
        casts=OwnRuleCasts(FirstInputCast(DType("Float64"), CastOption.STRICT)),
        kernel=Reduction(ReductionKernels.SKEW, ReductionKernels.SKEW_GROUPS),
        physical=PhysicalExpr.GROUPS_FUNCTION,
    )
    LOWER_BOUND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "LowerBound"),
        type_function=ExprField(ExprFields.bound),
        options=Shape.literal(),
        casts=NoCasts(),
        kernel=PlannedInput(ReductionKernels().bound),
    )
    UPPER_BOUND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "UpperBound"),
        type_function=ExprField(ExprFields.bound),
        options=Shape.literal(),
        casts=NoCasts(),
        kernel=PlannedInput(ReductionKernels().bound),
    )
    MAX_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "MaxBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=TwoInputs(ValueAtKey()),
        physical=PhysicalExpr.MIN_MAX_BY,
    )
    MIN_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "MinBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=SCALAR,
        casts=FunctionCasting(),
        kernel=TwoInputs(ValueAtKey()),
        physical=PhysicalExpr.MIN_MAX_BY,
    )
    INDEX_OF: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "IndexOf"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=ShapeSince(
            INDEXED.with_scalar_inputs(ScalarInput.passed(1, "value", "index_of")),
            "1.40.0",
            INDEXED,
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(IndexOf()),
    )
    CORRELATION: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Correlation"),
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(
            FunctionOptions.aggregation().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(Correlation, NodeOption.variant("function", "Correlation", "method")),
        ),
    )
    QUANTILE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Quantile"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype),
        options=UNORDERED,
        casts=FunctionCasting(),
        kernel=LiteralReduction(Quantile(), GroupQuantile()),
        physical=PhysicalExpr.AGGREGATION,
    )
