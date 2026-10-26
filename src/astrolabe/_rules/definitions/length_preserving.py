"""The definitions of the nodes that keep the length of their input, such as `cum_sum`.

Such a node gives one value for each row of its input, but the value of a row
can depend on other rows. So polars does not run it as an elementwise node.
Inside `group_by().agg()`, it runs on each group, and gives a list for each
group. The family holds the cumulative functions, the exponentially weighted
functions, the rolling functions of a fixed window and of a window that a key
column orders, `shift`, `diff` and `pct_change`, the interpolations, the
ranks, the fills of nulls by a strategy, and the tests of duplicates.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import EwmCasts, ShiftFillCasts
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.value_adapters import KeptValue
from astrolabe._rules.definitions.definition import (
    Definition,
    NullRule,
    PeriodNulls,
    WeightedNulls,
    WindowNulls,
)
from astrolabe._rules.definitions.kernel_adapters import AllInputs, LiteralInputs, OneInput
from astrolabe._rules.definitions.node_options import (
    FromOptions,
    NodeIdentity,
    NodeOption,
    WhenIn,
)
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import PhysicalExpr
from astrolabe._rules.kernels.numeric import NumericKernels
from astrolabe._rules.kernels.windows import (
    Difference,
    EwmBy,
    InterpolateBy,
    PercentChange,
    RollingByKernels,
    RollingKernels,
    Shift,
    ShiftFilled,
    WindowDuration,
    WindowKernels,
)
from astrolabe._rules.options import FunctionFlag, FunctionOptions, ScalarInput, Shape
from astrolabe._rules.type_functions import FieldsMapper, FunctionClosures, WindowClosures


class WindowOptions:
    """Name the options of the rolling and exponentially weighted nodes that their rules read.

    A rolling node of a fixed window holds its options under
    `RollingExpr`, and a rolling node whose windows a key column orders under
    `RollingExprBy`. Each definition of such a node reads the same options.

    Example:
        >>> WindowOptions.SIZE
        NodeOption.integer('function', 'RollingExpr', 'options', 'window_size')
    """

    SIZE: ClassVar[NodeOption[int]] = NodeOption.integer(
        "function", "RollingExpr", "options", "window_size"
    )
    PERIODS: ClassVar[NodeOption[int]] = NodeOption.integer(
        "function", "RollingExpr", "options", "min_periods"
    )
    WEIGHTS: ClassVar[NodeOption[tuple[float, ...] | None]] = NodeOption.floats_or_none(
        "function", "RollingExpr", "options", "weights"
    )
    RANK: ClassVar[NodeOption[str | None]] = NodeOption.text(
        "function", "RollingExpr", "options", "fn_params", "Rank", "method"
    )
    DURATION: ClassVar[NodeOption[WindowDuration]] = NodeOption(
        ("function", "RollingExprBy", "options", "window_size"),
        WindowDuration.of,
    )
    CLOSED: ClassVar[NodeOption[str | None]] = NodeOption.text(
        "function", "RollingExprBy", "options", "closed_window"
    )
    RANK_BY: ClassVar[NodeOption[str | None]] = NodeOption.text(
        "function", "RollingExprBy", "options", "fn_params", "Rank", "method"
    )
    MEAN_HALF_LIFE: ClassVar[NodeOption[WindowDuration]] = NodeOption(
        ("function", "EwmMeanBy", "half_life"),
        WindowDuration.of,
    )
    SUM_HALF_LIFE: ClassVar[NodeOption[WindowDuration]] = NodeOption(
        ("function", "EwmSumBy", "half_life"),
        WindowDuration.of,
    )


class LengthPreserving:
    """Hold the definitions of the nodes that keep the length of their input.

    Example:
        >>> LengthPreserving.CUM_SUM.identity
        NodeIdentity(shape='Function', function='CumSum')
    """

    KEEPS: ClassVar[Shape] = Shape.function(FunctionOptions.length_preserving())
    COUNTED: ClassVar[Shape] = KEEPS.with_scalar_inputs(ScalarInput.named(1, "n"))

    CUM_SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "CumSum"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().cum_sum),
        options=Shape.function(FunctionOptions.length_preserving()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.CUM_SUM),
    )
    CUM_COUNT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "CumCount"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.INDICES),
    )
    CUM_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "CumMax"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.CUM_EXTREME),
    )
    CUM_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "CumMin"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.CUM_EXTREME),
    )
    CUM_PROD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "CumProd"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().cum_prod),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.CUM_PROD),
    )
    ARG_SORT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArgSort"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.INDICES),
    )
    RLE_ID: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RLEID"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.INDICES),
    )
    IS_DUPLICATED: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsDuplicated"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.UNIQUE),
    )
    IS_UNIQUE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsUnique"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.UNIQUE),
    )
    IS_FIRST_DISTINCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsFirstDistinct"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.FIRST_DISTINCT),
    )
    IS_LAST_DISTINCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsLastDistinct"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.FIRST_DISTINCT),
    )
    PEAK_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "PeakMax"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.PEAKS),
    )
    PEAK_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "PeakMin"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.PEAKS),
    )
    RANK: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Rank"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                WindowClosures().rank,
                NodeOption.text("function", "Rank", "options", "method"),
            ),
        ),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                WindowKernels().rank,
                NodeOption.text("function", "Rank", "options", "method"),
            ),
        ),
    )
    RECHUNK: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Rechunk"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.SAME),
    )
    REVERSE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Reverse"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.length_preserving().flag(FunctionFlag.NON_ORDER_OBSERVING),
        ),
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.REVERSE),
        physical=PhysicalExpr.REGROUP,
    )
    FILL_NULL_WITH_STRATEGY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "FillNullWithStrategy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=WhenIn(
            NodeOption.variant("function", "FillNullWithStrategy"),
            ("Zero", "One"),
            Shape.function(FunctionOptions.elementwise()),
            Shape.function(FunctionOptions.length_preserving()),
        ),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                WindowKernels().fill,
                NodeOption.variant("function", "FillNullWithStrategy"),
            ),
        ),
        physical=WhenIn(
            NodeOption.variant("function", "FillNullWithStrategy"),
            ("Forward", "Backward"),
            PhysicalExpr.REGROUP,
            None,
        ),
    )
    INTERPOLATE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Interpolate"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.map_dtype,
            func=FromOptions(
                WindowClosures().interpolate,
                NodeOption.text("function", "Interpolate"),
            ),
        ),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(WindowKernels().interpolate, NodeOption.text("function", "Interpolate")),
        ),
    )
    QCUT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "QCut"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                WindowClosures().qcut,
                include_breaks=NodeOption.flag("function", "QCut", "include_breaks"),
            ),
        ),
        options=Shape.function(
            FunctionOptions.length_preserving().flag(FunctionFlag.PASS_NAME_TO_APPLY),
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                WindowKernels().qcut,
                NodeOption.floats("function", "QCut", "probs"),
                NodeOption.texts("function", "QCut", "labels"),
                include_breaks=NodeOption.flag("function", "QCut", "include_breaks"),
                allow_duplicates=NodeOption.flag("function", "QCut", "allow_duplicates"),
            ),
        ),
    )
    SHIFT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Shift"),
        short=PeriodNulls(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=COUNTED,
        casts=FunctionCasting(),
        kernel=LiteralInputs(Shift(), group_lengths=True),
    )
    SHIFT_AND_FILL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ShiftAndFill"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS.with_scalar_inputs(
            ScalarInput.named(1, "n"),
            ScalarInput.named(2, "fill_value"),
        ),
        casts=OwnRuleCasts(ShiftFillCasts()),
        kernel=LiteralInputs(ShiftFilled(), group_lengths=True),
    )
    DIFF: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Diff"),
        short=PeriodNulls(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().diff),
        options=WhenIn(
            NodeOption.text("function", "Diff"),
            ("Drop",),
            Shape.function(FunctionOptions.groupwise()).with_scalar_inputs(
                ScalarInput.named(1, "n"),
            ),
            COUNTED,
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(Difference.of, NodeOption.text("function", "Diff")),
            group_lengths=True,
        ),
    )
    PCT_CHANGE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "PctChange"),
        short=PeriodNulls(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().pct_change),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(PercentChange(), group_lengths=True),
    )
    INTERPOLATE_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "InterpolateBy"),
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(InterpolateBy(), group_lengths=True),
    )
    EWM_MEAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmMean"),
        nulls=NullRule.FIRST,
        short=WeightedNulls(NodeOption.integer("function", "EwmMean", "options", "min_periods")),
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=KEEPS,
        casts=OwnRuleCasts(EwmCasts()),
        kernel=OneInput(WindowKernels.EWM_LOOP),
    )
    EWM_SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmSum"),
        nulls=NullRule.FIRST,
        short=WeightedNulls(NodeOption.integer("function", "EwmSum", "options", "min_periods")),
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=KEEPS,
        casts=OwnRuleCasts(EwmCasts()),
        kernel=OneInput(WindowKernels.EWM_LOOP),
    )
    EWM_STD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmStd"),
        nulls=NullRule.FIRST,
        short=WeightedNulls(
            NodeOption.integer("function", "EwmStd", "options", "min_periods"),
            bias=NodeOption.flag("function", "EwmStd", "options", "bias"),
        ),
        type_function=Mapper(FieldsMapper.ewm_var_dtype, op="ewm_std"),
        options=KEEPS,
        casts=OwnRuleCasts(EwmCasts()),
        kernel=OneInput(WindowKernels.EWM_RECAST),
    )
    EWM_VAR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmVar"),
        nulls=NullRule.FIRST,
        short=WeightedNulls(
            NodeOption.integer("function", "EwmVar", "options", "min_periods"),
            bias=NodeOption.flag("function", "EwmVar", "options", "bias"),
        ),
        type_function=Mapper(FieldsMapper.ewm_var_dtype, op="ewm_var"),
        options=KEEPS,
        casts=OwnRuleCasts(EwmCasts()),
        kernel=OneInput(WindowKernels.EWM_RECAST),
    )
    EWM_MEAN_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmMeanBy"),
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(FromOptions(EwmBy, WindowOptions.MEAN_HALF_LIFE), group_lengths=True),
    )
    EWM_SUM_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "EwmSumBy"),
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(FromOptions(EwmBy, WindowOptions.SUM_HALF_LIFE), group_lengths=True),
    )
    ROLLING_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Min"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().extreme,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Max"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().extreme,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Sum"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.sum_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().sum,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_MEAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Mean"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().mean,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_QUANTILE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Quantile"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().quantile,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_STD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Std"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.var_dtype, op="std"),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().spread,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_VAR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Var"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.var_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                RollingKernels().spread,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.WEIGHTS,
            ),
        ),
    )
    ROLLING_RANK: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Rank"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(WindowClosures().rank, WindowOptions.RANK),
        ),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                RollingKernels().rank,
                WindowOptions.SIZE,
                WindowOptions.PERIODS,
                WindowOptions.RANK,
            ),
        ),
    )
    ROLLING_SKEW: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Skew"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.MOMENT),
    )
    ROLLING_KURTOSIS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.Kurtosis"),
        short=WindowNulls(WindowOptions.PERIODS),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=OneInput(WindowKernels.MOMENT),
    )
    ROLLING_CORR_COV: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExpr.CorrCov"),
        short=WindowNulls(WindowOptions.PERIODS),
        type_function=Mapper(FieldsMapper.try_map_dtypes, func=FunctionClosures().corr_cov),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(RollingKernels().corr_cov, WindowOptions.SIZE, WindowOptions.PERIODS),
            group_lengths=True,
        ),
    )
    ROLLING_MIN_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.MinBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().extreme, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_MAX_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.MaxBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().extreme, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_SUM_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.SumBy"),
        type_function=Mapper(FieldsMapper.sum_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().sum, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_MEAN_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.MeanBy"),
        type_function=Mapper(FieldsMapper.moment_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().mean, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_QUANTILE_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.QuantileBy"),
        type_function=Mapper(FieldsMapper.moment_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().mean, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_STD_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.StdBy"),
        type_function=Mapper(FieldsMapper.var_dtype, op="std"),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().spread, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_VAR_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.VarBy"),
        type_function=Mapper(FieldsMapper.var_dtype),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(RollingByKernels().spread, WindowOptions.DURATION),
            group_lengths=True,
        ),
    )
    ROLLING_RANK_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RollingExprBy.RankBy"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(WindowClosures().rank, WindowOptions.RANK_BY),
        ),
        options=KEEPS,
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(
                RollingByKernels().rank,
                WindowOptions.DURATION,
                WindowOptions.CLOSED,
                WindowOptions.RANK_BY,
            ),
            group_lengths=True,
        ),
    )
