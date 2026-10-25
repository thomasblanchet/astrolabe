"""The definitions of the functions of the `cat` namespace, such as `cat.len_chars`.

The `cat` namespace holds the functions of a `Categorical` or an `Enum`. The
identity of such a node names the namespace, for example
`Categorical.LenChars`. The type function of such a function gives a fixed
type and does not check its input, except `cat.physical`. No function of the
namespace casts its inputs, so the cast pass never checks the namespace
either. The kernel reads the categories of the input.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import Namespace
from astrolabe._rules.definitions.cast_adapters import FunctionCasting
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import OneInput
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.strings import TextKernels
from astrolabe._rules.options import FunctionOptions, Shape
from astrolabe._rules.type_functions import CategoricalClosures, FieldsMapper


class Categories:
    """Hold the definitions of the functions of the `cat` namespace.

    Example:
        >>> Categories.PHYSICAL.identity
        NodeIdentity(shape='Function', function='Categorical.Physical')
    """

    GET_CATEGORIES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.GetCategories"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.groupwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.GET_CATEGORIES),
    )
    LEN_BYTES: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.LenBytes"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_LENGTHS),
    )
    LEN_CHARS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.LenChars"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_LENGTHS),
    )
    STARTS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.StartsWith"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_AFFIX),
    )
    ENDS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.EndsWith"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_AFFIX),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.Slice"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_SLICE),
    )
    TO: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.To"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.with_dtype,
            dtype=NodeOption.dtype("function", "Categorical", "To", 0),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(
            FromOptions(
                TextKernels().to_category,
                NodeOption.dtype("function", "Categorical", "To", 0),
            ),
        ),
    )
    PHYSICAL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "Categorical.Physical"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.try_map_dtype, func=CategoricalClosures().physical),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.CATEGORICAL),
        kernel=OneInput(TextKernels.CATEGORY_PHYSICAL),
    )
