"""The definitions of the functions whose output has a length of its own, such as `unique`.

Such a function gives a number of rows that its values or its options
decide, so polars runs it on the whole input, and on each group in `agg`.
The family holds `unique`, `mode`, `arg_unique`, `unique_counts`,
`drop_nulls`, `drop_nans`, `arg_where`, `gather_every`, `rle`, `reshape`,
`hist`, `shuffle` and `sample`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.definitions.cast_adapters import FunctionCasting
from astrolabe._rules.definitions.value_adapters import KeptValue
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import (
    AllInputs,
    LiteralInputs,
    OneInput,
    PlannedInputs,
)
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption, WhenIn
from astrolabe._rules.definitions.type_function_adapters import ExprField, Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.lengths import (
    Appended,
    Counted,
    Dimensions,
    Extended,
    LengthKernels,
    Mapped,
    Repeated,
    RepeatedBy,
    Sampled,
    Searched,
    TopKBy,
)
from astrolabe._rules.options import (
    FunctionFlag,
    FunctionOptions,
    KernelLength,
    ScalarInput,
    Shape,
)
from astrolabe._rules.supertype import SupertypeFlag

if TYPE_CHECKING:
    from collections.abc import Callable

from astrolabe._rules.type_functions import (
    ExprFields,
    FieldsMapper,
    FunctionClosures,
    LengthClosures,
)


class OtherLengths:
    """Hold the definitions of the functions whose output has a length of its own.

    Example:
        >>> OtherLengths.UNIQUE.identity
        NodeIdentity(shape='Function', function='Unique')
    """

    GROUPWISE: ClassVar[Shape] = Shape.function(FunctionOptions.groupwise(), KernelLength.shrinks())
    COUNTS: ClassVar[FromOptions[Callable[[DType], DType]]] = FromOptions(
        FunctionClosures().value_counts,
        NodeOption.first_name("input"),
        NodeOption.text("function", "ValueCounts", "name"),
        normalize=NodeOption.flag("function", "ValueCounts", "normalize"),
    )
    DROPS: ClassVar[Shape] = Shape.function(
        FunctionOptions.row_separable().flag(FunctionFlag.NON_ORDER_PRODUCING),
        KernelLength.values(),
    )

    UNIQUE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Unique"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.SAME),
    )
    MODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Mode"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.SAME),
    )
    ARG_UNIQUE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArgUnique"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.INDICES),
    )
    UNIQUE_COUNTS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "UniqueCounts"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.INDICES),
    )
    DROP_NULLS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "DropNulls"),
        computes=KeptValue(),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.row_separable().flag(
                FunctionFlag.ALLOW_EMPTY_INPUTS | FunctionFlag.NON_ORDER_PRODUCING,
            ),
            KernelLength.values(),
        ),
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.SAME),
    )
    DROP_NANS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "DropNans"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=DROPS,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.SAME),
    )
    ARG_WHERE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ArgWhere"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.WHERE),
    )
    GATHER_EVERY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "GatherEvery"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.SAME),
    )
    RLE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RLE"),
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().rle),
        options=GROUPWISE,
        casts=FunctionCasting(),
        kernel=OneInput(LengthKernels.RLE),
    )
    RANDOM: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Random"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=WhenIn(
            NodeOption.variant("function", "Random", "method"),
            ("Shuffle",),
            Shape.function(FunctionOptions.length_preserving()),
            Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        ),
        casts=FunctionCasting(),
        kernel=LiteralInputs(NodeOption(("function", "Random", "method"), Sampled.of)),
    )
    HIST: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Hist"),
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=FromOptions(
                LengthClosures().hist,
                include_breakpoint=NodeOption.flag("function", "Hist", "include_breakpoint"),
                include_category=NodeOption.flag("function", "Hist", "include_category"),
            ),
        ),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                LengthKernels().hist,
                include_breakpoint=NodeOption.flag("function", "Hist", "include_breakpoint"),
                include_category=NodeOption.flag("function", "Hist", "include_category"),
            ),
        ),
    )
    RESHAPE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Reshape"),
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                FunctionClosures().reshape,
                NodeOption(("function", "Reshape"), Dimensions.of),
            ),
        ),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(
                LengthKernels().reshape,
                NodeOption(("function", "Reshape"), Dimensions.of),
            ),
        ),
    )
    TOP_K: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TopK"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Counted((1,), Mapped(LengthClosures().same), index=True)),
    )
    TOP_K_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "TopKBy"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.values()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(TopKBy()),
    )
    REPEAT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Repeat"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.groupwise().flag(FunctionFlag.ALLOW_RENAME),
            KernelLength.values(),
        ).with_scalar_inputs(ScalarInput.named(0, "value"), ScalarInput.named(1, "n")),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Repeated()),
    )
    EXTEND_CONSTANT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ExtendConstant"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(
            FunctionOptions.groupwise(),
            KernelLength.values(),
        ).with_scalar_inputs(ScalarInput.named(1, "value"), ScalarInput.named(2, "n")),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Extended()),
    )
    REPEAT_BY: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RepeatBy"),
        type_function=Mapper(FieldsMapper.map_dtype, func=FunctionClosures().repeat_by),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(RepeatedBy()),
    )
    SEARCH_SORTED: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "SearchSorted"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType.idx()),
        options=Shape.function(
            FunctionOptions.groupwise().with_supertyping(
                SupertypeFlag.default() & ~SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING,
            ),
            KernelLength.of_input(1),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(Searched()),
    )
    APPEND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Append"),
        type_function=ExprField(ExprFields.append),
        options=Shape.function(FunctionOptions.groupwise(), KernelLength.joined()),
        casts=FunctionCasting(),
        kernel=PlannedInputs(Appended()),
    )
    VALUE_COUNTS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ValueCounts"),
        type_function=Mapper(FieldsMapper.map_dtype, func=COUNTS),
        options=Shape.function(
            FunctionOptions.groupwise().flag(
                FunctionFlag.PASS_NAME_TO_APPLY | FunctionFlag.NON_ORDER_OBSERVING,
            ),
            KernelLength.shrinks(),
        ),
        casts=FunctionCasting(),
        kernel=OneInput(FromOptions(LengthKernels().mapped, COUNTS)),
        names=True,
    )
