"""The definitions of the functions of the `bin` namespace, such as `bin.contains`.

The `bin` namespace holds the functions of bytes, the `Binary` data type. The
identity of such a node names the namespace, for example
`BinaryExpr.Contains`. The type function of such a function does not check
its first input. `bin.contains`, `bin.starts_with` and `bin.ends_with` cast
their inputs to their supertype, and their cast pass checks that the first
input is `Binary`. Every kernel reads its first input as `Binary` data.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.cast_pass import Namespace
from astrolabe._rules.definitions.cast_adapters import FunctionCasting
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import LiteralInputs, OneInput
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.strings import TextKernels
from astrolabe._rules.options import FunctionOptions, Shape
from astrolabe._rules.supertype import SupertypeFlag
from astrolabe._rules.type_functions import BinaryClosures, FieldsMapper


class Binaries:
    """Hold the definitions of the functions of the `bin` namespace.

    Example:
        >>> Binaries.SIZE.identity
        NodeIdentity(shape='Function', function='BinaryExpr.Size')
    """

    CONTAINS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Contains"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_AFFIX),
    )
    STARTS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.StartsWith"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_AFFIX),
    )
    ENDS_WITH: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.EndsWith"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("Boolean")),
        options=Shape.function(
            FunctionOptions.elementwise().with_supertyping(SupertypeFlag.default()),
        ),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_AFFIX),
    )
    HEX_DECODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.HexDecode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=OneInput(TextKernels.BINARY_DECODE),
    )
    HEX_ENCODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.HexEncode"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=OneInput(TextKernels.BINARY_ENCODE),
    )
    SIZE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Size"),
        nulls=NullRule.FIRST,
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt32")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=OneInput(TextKernels.BINARY_SIZE),
    )
    REINTERPRET: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Reinterpret"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_dtype,
            func=FromOptions(
                BinaryClosures().reinterpret,
                NodeOption.dtype("function", "BinaryExpr", "Reinterpret", 0),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=OneInput(
            FromOptions(
                TextKernels().reinterpret,
                NodeOption.dtype("function", "BinaryExpr", "Reinterpret", 0),
            ),
        ),
    )
    SLICE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Slice"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_SLICE),
    )
    HEAD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Head"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_HEAD),
    )
    TAIL: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Tail"),
        type_function=Mapper(FieldsMapper.with_same_dtype),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_HEAD),
    )
    GET: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "BinaryExpr.Get"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("UInt8")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(Namespace.BINARY),
        kernel=LiteralInputs(TextKernels.BINARY_GET),
    )
