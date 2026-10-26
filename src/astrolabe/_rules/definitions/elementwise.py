"""The definitions of the elementwise nodes, such as `abs`, `exp` or a cast.

An elementwise node gives one value for each row of its inputs, and the value
of a row depends only on that row. This module holds the functions of
arithmetic, the trigonometric functions, the tests for a null or a NaN, the
bit counts, `fill_null` with a value, and the cast node. The functions of a
namespace, such as `str.strptime`, are in their own module, even where they
are elementwise.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.definitions.cast_adapters import CastNode, FunctionCasting
from astrolabe._rules.definitions.value_adapters import CastValue
from astrolabe._rules.definitions.definition import Definition, NullRule, ShapeSince
from astrolabe._rules.definitions.kernel_adapters import (
    AllInputs,
    OneInput,
    PlannedInput,
    StepData,
    TwoInputs,
)
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.simplify_adapters import NegateSimplify
from astrolabe._rules.definitions.type_function_adapters import ExprField, Mapper, MapperCheck
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.fill import FillNull
from astrolabe._rules.kernels.floats import FloatKernels, Log
from astrolabe._rules.kernels.layout import LayoutKernels
from astrolabe._rules.kernels.logic import BitwiseKernels
from astrolabe._rules.kernels.numeric import NumericKernels
from astrolabe._rules.options import FunctionFlag, FunctionOptions, NodeKind, Shape
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import ExprFields, FieldsMapper, FunctionClosures


class Elementwise:
    """Hold the definitions of the elementwise nodes.

    Example:
        >>> Elementwise.ABS.identity
        NodeIdentity(shape='Function', function='Abs')
    """

    ABS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Abs"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.ABS),
    )
    SQRT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Pow.Sqrt"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.SQRT),
    )
    FILL_NULL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "FillNull"),
        type_function=Mapper(FieldsMapper.map_to_supertype),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(),
        kernel=TwoInputs(FillNull()),
    )
    CAST: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Cast", ""),
        computes=CastValue(),
        nulls=NullRule.FIRST,
        type_function=ExprField(ExprFields.cast, target=NodeOption.dtype("dtype")),
        options=Shape.of(NodeKind.CAST),
        casts=CastNode(NodeOption.dtype("dtype"), NodeOption.cast_option("options")),
        kernel=StepData(),
    )
    CBRT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Pow.Cbrt"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_numeric_to_float_dtype, coerce_decimal=True),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.CBRT),
    )
    EXP: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Exp"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.EXP),
    )
    LOG1P: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Log1p"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.LOG1P),
    )
    LOG: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Log"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.log_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=AllInputs(Log()),
    )
    SIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Sin"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    COS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Cos"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    TAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Tan"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    COT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Cot"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCSIN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcSin"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCCOS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcCos"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCTAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcTan"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    SINH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Sinh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    COSH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Cosh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    TANH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Tanh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCSINH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcSinh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCCOSH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcCosh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    ARCTANH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.ArcTanh"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    DEGREES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Degrees"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    RADIANS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Trigonometry.Radians"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.TRIGONOMETRY),
    )
    NEGATE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Negate"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.NEGATE),
        simplify=NegateSimplify(),
    )
    SIGN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Sign"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_same_dtype,
            checks=[
                MapperCheck(
                    FieldsMapper.ensure_satisfies,
                    check=FunctionClosures().numeric,
                    op="sign",
                ),
            ],
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.KEEP_NUMBER),
    )
    FLOOR: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Floor"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.KEEP_NUMBER),
    )
    CEIL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Ceil"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.KEEP_NUMBER),
    )
    ROUND: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Round"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.KEEP_NUMBER),
    )
    TRUNCATE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Truncate"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(NumericKernels.KEEP_NUMBER),
    )
    ROUND_SIG_FIGS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "RoundSF"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                NumericKernels().round_sig_figs,
                NodeOption.integer("function", "RoundSF", "digits"),
            ),
        ),
    )
    IS_NAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsNan"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.PRESERVES_NULL_FIRST_INPUT)
        ),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.IS_NAN),
    )
    IS_NOT_NAN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsNotNan"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.PRESERVES_NULL_FIRST_INPUT)
        ),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.IS_NOT_NAN),
    )
    IS_FINITE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsFinite"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.PRESERVES_NULL_FIRST_INPUT)
        ),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.IS_FINITE),
    )
    IS_INFINITE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsInfinite"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.PRESERVES_NULL_FIRST_INPUT)
        ),
        casts=FunctionCasting(),
        kernel=OneInput(FloatKernels.IS_INFINITE),
    )
    IS_NULL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsNull"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(LayoutKernels.IS_NULL),
    )
    IS_NOT_NULL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsNotNull"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(LayoutKernels.IS_NULL),
    )
    NOT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.Not"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.try_map_dtype, func=FunctionClosures().not_),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.PRESERVES_NULL_FIRST_INPUT)
        ),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.NOT),
    )
    BITWISE_COUNT_ONES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.CountOnes"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("count_ones")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    BITWISE_COUNT_ZEROS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.CountZeros"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("count_zeros")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    BITWISE_LEADING_ONES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.LeadingOnes"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("leading_ones")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    BITWISE_LEADING_ZEROS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.LeadingZeros"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("leading_zeros")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    BITWISE_TRAILING_ONES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.TrailingOnes"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("trailing_ones")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    BITWISE_TRAILING_ZEROS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Bitwise.TrailingZeros"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype, func=FunctionClosures().bitwise("trailing_zeros")
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(BitwiseKernels.COUNT),
    )
    HASH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Hash"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt64")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(LayoutKernels.HASH),
    )
    TO_PHYSICAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ToPhysical"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.to_physical_type),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(LayoutKernels.TO_PHYSICAL),
    )
    SET_SORTED: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "SetSortedFlag"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(LayoutKernels.SET_SORTED),
    )
    REINTERPRET: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Reinterpret"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                FunctionClosures().reinterpret,
                NodeOption.flag_or_none("function", "Reinterpret", 0),
                NodeOption.data_type("function", "Reinterpret", 1),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=PlannedInput(LayoutKernels().reinterpret),
    )
    CUT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Cut"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                FunctionClosures().cut,
                NodeOption.floats("function", "Cut", "breaks"),
                NodeOption.texts("function", "Cut", "labels"),
                left_closed=NodeOption.flag("function", "Cut", "left_closed"),
                include_breaks=NodeOption.flag("function", "Cut", "include_breaks"),
            ),
        ),
        options=ShapeSince(
            Shape.function(FunctionOptions.elementwise().flag(FunctionFlag.PASS_NAME_TO_APPLY)),
            "1.40.0",
            Shape.function(
                FunctionOptions.length_preserving().flag(FunctionFlag.PASS_NAME_TO_APPLY),
            ),
        ),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                NumericKernels().cut,
                NodeOption.floats("function", "Cut", "breaks"),
                NodeOption.texts("function", "Cut", "labels"),
                left_closed=NodeOption.flag("function", "Cut", "left_closed"),
                include_breaks=NodeOption.flag("function", "Cut", "include_breaks"),
            ),
        ),
    )
