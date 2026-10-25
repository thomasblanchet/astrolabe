"""The definitions of the functions of the `list` namespace, such as `list.sum` or `list.get`.

The `list` namespace holds the functions of list data. The identity of such a
node names the namespace, for example `ListExpr.Sum`. Polars checks most of
these functions in its planner: the first input must be a list, or the planner
refuses the node. The kernel then works on the values inside each list, so its
type depends on the inner type of the data. `concat_list` builds the node of
`list.concat`, and `arr.head`, `arr.slice` and `arr.tail` build the node of
`list.slice` over the array as a list.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import GatherIndexCasts, IsInCasts, Namespace
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import AllInputs, LiteralInputs, OneInput
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import Mapper, MapperCheck
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.lists import (
    Concatenated,
    Contains,
    CountMatches,
    Gathered,
    GatherEvery,
    Joined,
    ListKernels,
    SetOperated,
    Shifted,
    Sliced,
)
from astrolabe._rules.options import CastingRules, FunctionFlag, FunctionOptions, Shape
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import FieldsMapper, ListClosures


class Lists:
    """Hold the definitions of the functions of the `list` namespace.

    Example:
        >>> Lists.SUM.identity
        NodeIdentity(shape='Function', function='ListExpr.Sum')
    """

    IS_LIST: ClassVar[MapperCheck] = MapperCheck(FieldsMapper.ensure_is_list)
    ELEMENTWISE: ClassVar[Shape] = Shape.function(FunctionOptions.elementwise())

    LENGTH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Length"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_LIST], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.LENGTH),
    )
    SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Sum"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.nested_sum_type),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.SUM),
    )
    MEAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Mean"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.nested_mean_median_type),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.MEAN),
    )
    MEDIAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Median"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.nested_mean_median_type),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.MEDIAN),
    )
    STD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Std"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.STD),
    )
    VAR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Var"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.var_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.VAR),
    )
    MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Min"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.EXTREMUM),
    )
    MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Max"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.EXTREMUM),
    )
    ARG_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.ArgMin"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_LIST], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.ARG_EXTREMUM),
    )
    ARG_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.ArgMax"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_LIST], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.ARG_EXTREMUM),
    )
    DIFF: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Diff"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.try_map_dtype, func=ListClosures().diff),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.DIFF),
    )
    SORT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Sort"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.SAME),
    )
    DROP_NULLS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.DropNulls"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ListKernels.SAME),
    )
    GET: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Get"),
        type_function=Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(ListKernels().get, NodeOption.flag("function", "ListExpr", "Get")),
            can_fail=FromOptions(
                ListKernels().fails_out_of_bounds,
                NodeOption.flag("function", "ListExpr", "Get"),
            ),
            group_lengths=True,
        ),
    )
    GATHER: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Gather"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=OwnRuleCasts(GatherIndexCasts()),
        kernel=AllInputs(
            Gathered(),
            can_fail=FromOptions(
                ListKernels().fails_out_of_bounds,
                NodeOption.flag("function", "ListExpr", "Gather"),
            ),
            group_lengths=True,
        ),
    )
    GATHER_EVERY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.GatherEvery"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=LiteralInputs(GatherEvery(), group_lengths=True),
    )
    SHIFT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Shift"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(Shifted("List"), group_lengths=True),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Slice"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=LiteralInputs(Sliced(), group_lengths=True),
    )
    SAMPLE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Sample"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(
                ListKernels().sample,
                fraction=NodeOption.flag("function", "ListExpr", "Sample", "is_fraction"),
                replaces=NodeOption.flag("function", "ListExpr", "Sample", "with_replacement"),
            ),
            group_lengths=True,
        ),
    )
    JOIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Join"),
        type_function=Mapper(FieldsMapper.try_map_dtype, func=ListClosures().join),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(Joined("List", checks_lengths=False), group_lengths=True),
    )
    CONTAINS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Contains"),
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_LIST], dtype=DType("Boolean")),
        options=ELEMENTWISE,
        casts=OwnRuleCasts(IsInCasts.list_contains()),
        kernel=AllInputs(Contains("List"), group_lengths=True),
    )
    COUNT_MATCHES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.CountMatches"),
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_LIST], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(CountMatches("List"), group_lengths=True),
    )
    SET_OPERATION: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.SetOperation"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_LIST]),
        options=Shape.function(
            FunctionOptions.elementwise().with_casting_rules(
                CastingRules.supertype(
                    SupertypeFlag.default() | SupertypeFlag.ALLOW_IMPLODE_LIST,
                ),
            ),
        ),
        casts=FunctionCasting(Namespace.LIST),
        kernel=AllInputs(SetOperated(), group_lengths=True),
    )
    CONCAT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.Concat"),
        type_function=Mapper(FieldsMapper.map_to_list_supertype),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.INPUT_WILDCARD_EXPANSION),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(Concatenated(), group_lengths=True),
    )
    TO_ARRAY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.ToArray"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            checks=[IS_LIST],
            func=FromOptions(
                ListClosures().to_array,
                NodeOption.integer("function", "ListExpr", "ToArray"),
            ),
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                ListKernels().to_array,
                NodeOption.integer("function", "ListExpr", "ToArray"),
            ),
        ),
    )
    TO_STRUCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ListExpr.ToStruct"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                ListClosures().to_struct,
                NodeOption.texts("function", "ListExpr", "ToStruct"),
            ),
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                ListKernels().to_struct,
                NodeOption.texts("function", "ListExpr", "ToStruct"),
            ),
        ),
    )
