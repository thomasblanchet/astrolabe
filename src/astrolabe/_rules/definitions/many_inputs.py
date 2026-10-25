"""The definitions of the functions of several inputs that work value by value, such as `is_in`.

Each function reads one row of each input to give one row: `coalesce`, the
horizontal functions such as `max_horizontal`, `clip`, `is_between`,
`is_close`, `is_in`, `arctan2`, `pow`, `replace` and `replace_strict`. Most
join their inputs to a supertype in the cast pass, from the casting rule of
their options, and `is_in` has a rule of its own. Polars turns
`all_horizontal` and `any_horizontal` of two inputs into `&` and `|` when it
converts the expression: the logical `and` and `or`, which read their
inputs as `Boolean`. So they take the rules of those operators.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from astrolabe._rules.cast_pass import IsBetweenCasts, IsInCasts, ReplaceCasts
from astrolabe._rules.definitions.binary_operators import OperatorNode
from astrolabe._rules.definitions.cast_adapters import FunctionCasting, OwnRuleCasts
from astrolabe._rules.definitions.definition import Definition
from astrolabe._rules.definitions.kernel_adapters import AllInputs, LiteralInputs
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.overrides import NullsGiveFirstInput
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.comparison import CompareOp
from astrolabe._rules.kernels.horizontal import (
    Between,
    Clipped,
    CloseTest,
    Coalesced,
    HorizontalExtreme,
    HorizontalMean,
    HorizontalSum,
)
from astrolabe._rules.kernels.lists import Membership
from astrolabe._rules.kernels.nodes import Replaced, ReplacedStrict, Structured
from astrolabe._rules.kernels.powers import ArcTangent, Power
from astrolabe._rules.operators import Operator
from astrolabe._rules.options import FunctionFlag, FunctionOptions, KernelLength, Shape
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import FieldsMapper, FunctionClosures


class ManyInputs:
    """Hold the definitions of the functions of several inputs that work value by value.

    Example:
        >>> ManyInputs.COALESCE.identity
        NodeIdentity(shape='Function', function='Coalesce')
    """

    NO_TEXT: ClassVar[SupertypeFlag] = (
        SupertypeFlag.default() & ~SupertypeFlag.ALLOW_PRIMITIVE_TO_STRING
    )
    JOINED: ClassVar[Shape] = Shape.function(
        FunctionOptions
        .elementwise()
        .flag(FunctionFlag.INPUT_WILDCARD_EXPANSION)
        .with_supertyping(SupertypeFlag.default()),
    )
    BOOLEAN: ClassVar[DType] = DType("Boolean")

    COALESCE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Coalesce"),
        type_function=Mapper(FieldsMapper.map_to_supertype),
        options=JOINED,
        casts=FunctionCasting(),
        kernel=AllInputs(Coalesced(), own_lengths=False),
    )
    MAX_HORIZONTAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "MaxHorizontal"),
        type_function=Mapper(FieldsMapper.map_to_supertype),
        options=Shape.function(
            FunctionOptions
            .elementwise()
            .flag(FunctionFlag.INPUT_WILDCARD_EXPANSION | FunctionFlag.ALLOW_RENAME)
            .with_supertyping(NO_TEXT),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(HorizontalExtreme(CompareOp.GT), own_lengths=False),
    )
    MIN_HORIZONTAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "MinHorizontal"),
        type_function=Mapper(FieldsMapper.map_to_supertype),
        options=Shape.function(
            FunctionOptions
            .elementwise()
            .flag(FunctionFlag.INPUT_WILDCARD_EXPANSION | FunctionFlag.ALLOW_RENAME)
            .with_supertyping(NO_TEXT),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(HorizontalExtreme(CompareOp.LT), own_lengths=False),
    )
    SUM_HORIZONTAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "SumHorizontal"),
        type_function=Mapper(
            FieldsMapper.map_to_supertype_with,
            func=FunctionClosures().sum_horizontal,
        ),
        options=JOINED,
        casts=FunctionCasting(),
        kernel=NullsGiveFirstInput(
            AllInputs(
                FromOptions(
                    HorizontalSum,
                    NodeOption.flag("function", "SumHorizontal", "ignore_nulls"),
                ),
                own_lengths=False,
            ),
        ),
    )
    MEAN_HORIZONTAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "MeanHorizontal"),
        type_function=Mapper(
            FieldsMapper.map_to_supertype_with,
            func=FunctionClosures().mean_horizontal,
        ),
        options=Shape.function(
            FunctionOptions.elementwise().flag(FunctionFlag.INPUT_WILDCARD_EXPANSION),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(HorizontalMean(), own_lengths=False),
    )
    ALL_HORIZONTAL: ClassVar[Definition] = replace(
        OperatorNode().of(Operator.LOGICAL_AND),
        identity=NodeIdentity("Function", "Boolean.AllHorizontal"),
    )
    ANY_HORIZONTAL: ClassVar[Definition] = replace(
        OperatorNode().of(Operator.LOGICAL_OR),
        identity=NodeIdentity("Function", "Boolean.AnyHorizontal"),
    )
    CLIP: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Clip"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise(), KernelLength.first()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Clipped()),
    )
    IS_BETWEEN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsBetween"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=BOOLEAN),
        options=Shape.function(FunctionOptions.elementwise().with_supertyping(NO_TEXT)),
        casts=OwnRuleCasts(
            FromOptions(
                IsBetweenCasts, NodeOption.text("function", "Boolean", "IsBetween", "closed")
            ),
        ),
        kernel=AllInputs(
            FromOptions(Between, NodeOption.text("function", "Boolean", "IsBetween", "closed")),
        ),
    )
    IS_CLOSE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsClose"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=BOOLEAN),
        options=Shape.function(FunctionOptions.elementwise().with_supertyping(NO_TEXT)),
        casts=FunctionCasting(),
        kernel=AllInputs(
            FromOptions(
                CloseTest,
                NodeOption.number("function", "Boolean", "IsClose", "abs_tol"),
                NodeOption.number("function", "Boolean", "IsClose", "rel_tol"),
            ),
        ),
    )
    IS_IN: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Boolean.IsIn"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=BOOLEAN),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(IsInCasts("is_in", contains=False)),
        kernel=AllInputs(Membership(), own_lengths=False),
    )
    ATAN2: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Atan2"),
        type_function=Mapper(FieldsMapper.map_to_float_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=AllInputs(ArcTangent()),
    )
    POW: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Pow.Generic"),
        type_function=Mapper(FieldsMapper.pow_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=LiteralInputs(Power()),
    )
    REPLACE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Replace"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(ReplaceCasts()),
        kernel=LiteralInputs(Replaced()),
    )
    REPLACE_STRICT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "ReplaceStrict"),
        type_function=Mapper(
            FieldsMapper.replace_dtype,
            return_dtype=NodeOption.dtype_or_none("function", "ReplaceStrict", "return_dtype"),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=OwnRuleCasts(ReplaceCasts()),
        kernel=LiteralInputs(
            FromOptions(
                ReplacedStrict,
                NodeOption.dtype_or_none("function", "ReplaceStrict", "return_dtype"),
            ),
        ),
    )
    AS_STRUCT: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "AsStruct"),
        type_function=Mapper(FieldsMapper.as_struct, names=NodeOption.names("input")),
        options=Shape.function(
            FunctionOptions.elementwise().flag(
                FunctionFlag.PASS_NAME_TO_APPLY | FunctionFlag.INPUT_WILDCARD_EXPANSION,
            ),
        ),
        casts=FunctionCasting(),
        kernel=AllInputs(FromOptions(Structured, NodeOption.names("input")), own_lengths=False),
        names=True,
    )
