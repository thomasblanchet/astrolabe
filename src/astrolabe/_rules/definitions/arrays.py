"""The definitions of the functions of the `arr` namespace, such as `arr.sum` or `arr.get`.

The `arr` namespace holds the functions of array data, that is, lists of one
fixed width. The identity of such a node names the namespace, for example
`ArrayExpr.Sum`. The planner checks that the first input is an array, and
the kernel works on the values inside each array, so its type depends on the
inner type of the data. `concat_arr` builds the node `ArrayExpr.Concat`.
`arr.head`, `arr.slice` and `arr.tail` build the node of `list.slice`, and the
node `ArrayExpr.Slice` where they keep the array (`as_array=True`).
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import IsInCasts
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import AllInputs, LiteralInputs, OneInput
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import (
    CheckedInputs,
    Mapper,
    MapperCheck,
)
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.arrays import ArrayConcat, ArrayKernels
from astrolabe._rules.kernels.lists import (
    Contains,
    CountMatches,
    Joined,
    ListKernels,
    Shifted,
)
from astrolabe._rules.options import FunctionFlag, FunctionOptions, Shape
from astrolabe._rules.type_functions import ArrayClosures, ArrayIndexBounds, FieldsMapper


class Arrays:
    """Hold the definitions of the functions of the `arr` namespace.

    Example:
        >>> Arrays.SUM.identity
        NodeIdentity(shape='Function', function='ArrayExpr.Sum')
    """

    IS_ARRAY: ClassVar[MapperCheck] = MapperCheck(FieldsMapper.ensure_is_array)
    ELEMENTWISE: ClassVar[Shape] = Shape.function(FunctionOptions.elementwise())

    LENGTH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Length"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.LENGTH),
    )
    MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Min"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.EXTREMUM),
    )
    MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Max"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.EXTREMUM),
    )
    SUM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Sum"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.nested_sum_type, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.SUM),
    )
    TO_LIST: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.ToList"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            checks=[IS_ARRAY],
            func=ArrayClosures().to_list,
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.TO_LIST),
    )
    STD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Std"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.MOMENT),
    )
    VAR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Var"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.var_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.VAR),
    )
    MEAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Mean"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.MOMENT),
    )
    MEDIAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Median"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.moment_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.MOMENT),
    )
    SORT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Sort"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.SAME),
    )
    ARG_MIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.ArgMin"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.ARG_EXTREMUM),
    )
    ARG_MAX: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.ArgMax"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.ARG_EXTREMUM),
    )
    EXPLODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Explode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.try_map_to_array_inner_dtype, checks=[IS_ARRAY]),
        options=Shape.function(FunctionOptions.row_separable()),
        casts=FunctionCasting(),
        kernel=OneInput(ArrayKernels.EXPLODE),
    )
    TO_STRUCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.ToStruct"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            checks=[IS_ARRAY],
            func=FromOptions(
                ArrayClosures().to_struct,
                NodeOption.texts("function", "ArrayExpr", "ToStruct"),
            ),
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                ArrayKernels().to_struct,
                NodeOption.texts("function", "ArrayExpr", "ToStruct"),
            ),
        ),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Slice"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            checks=[IS_ARRAY],
            func=FromOptions(
                ArrayClosures().slice,
                NodeOption.integer("function", "ArrayExpr", "Slice", 0),
                NodeOption.integer("function", "ArrayExpr", "Slice", 1),
            ),
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                ArrayKernels().slice,
                NodeOption.integer("function", "ArrayExpr", "Slice", 0),
                NodeOption.integer("function", "ArrayExpr", "Slice", 1),
            ),
        ),
    )
    GET: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Get"),
        type_function=CheckedInputs(
            FromOptions(
                ArrayIndexBounds,
                null_on_oob=NodeOption.flag("function", "ArrayExpr", "Get"),
            ),
            Mapper(FieldsMapper.map_to_list_and_array_inner_dtype, checks=[IS_ARRAY]),
        ),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=LiteralInputs(
            FromOptions(ArrayKernels().get, NodeOption.flag("function", "ArrayExpr", "Get")),
            can_fail=FromOptions(
                ListKernels().fails_out_of_bounds,
                NodeOption.flag("function", "ArrayExpr", "Get"),
            ),
            group_lengths=True,
        ),
    )
    SHIFT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Shift"),
        type_function=Mapper(FieldsMapper.with_same_dtype, checks=[IS_ARRAY]),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(Shifted("Array"), group_lengths=True),
    )
    JOIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Join"),
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType("String")),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(Joined("Array", checks_lengths=True), group_lengths=True),
    )
    CONTAINS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Contains"),
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType("Boolean")),
        options=ELEMENTWISE,
        casts=OwnRuleCasts(IsInCasts("arr.contains", contains=True)),
        kernel=AllInputs(Contains("Array"), group_lengths=True),
    )
    COUNT_MATCHES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.CountMatches"),
        type_function=Mapper(FieldsMapper.with_dtype, checks=[IS_ARRAY], dtype=DType.idx()),
        options=ELEMENTWISE,
        casts=FunctionCasting(),
        kernel=AllInputs(CountMatches("Array"), group_lengths=True),
    )
    CONCAT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArrayExpr.Concat"),
        type_function=Mapper(FieldsMapper.try_map_dtypes, func=ArrayClosures().concat),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.INPUT_WILDCARD_EXPANSION),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(ArrayConcat(), group_lengths=True),
    )
