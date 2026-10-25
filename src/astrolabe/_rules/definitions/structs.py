"""The definitions of the functions of the `struct` namespace, such as `struct.field`.

The `struct` namespace holds the functions of struct data. The identity of
such a node names the namespace, for example `StructExpr.FieldByName`.
`struct.unnest()` and `struct.field` with several names build the node
`StructExpr.SelectFields`, which polars expands into one `struct.field` for
each field that it picks.
"""

from __future__ import annotations

from typing import ClassVar

from astrolabe._rules.definitions.cast_adapters import FunctionCasting
from astrolabe._rules.definitions.definition import Definition, NullRule
from astrolabe._rules.definitions.kernel_adapters import OneInput
from astrolabe._rules.definitions.node_options import FromOptions, NodeIdentity, NodeOption
from astrolabe._rules.definitions.type_function_adapters import Mapper
from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.structs import StructKernels
from astrolabe._rules.options import FunctionFlag, FunctionOptions, Shape
from astrolabe._rules.type_functions import (
    FieldSelection,
    FieldsMapper,
    StructClosures,
)


class Structs:
    """Hold the definitions of the functions of the `struct` namespace.

    Example:
        >>> Structs.STRUCT_FIELD.identity
        NodeIdentity(shape='Function', function='StructExpr.FieldByName')
    """

    SELECTOR: ClassVar[NodeOption[FieldSelection]] = NodeOption(
        ("function", "StructExpr", "SelectFields"),
        FieldSelection,
    )

    STRUCT_FIELD: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StructExpr.FieldByName"),
        type_function=Mapper(
            FieldsMapper.try_map_field,
            func=FromOptions(
                StructClosures().field_by_name,
                NodeOption.text("function", "StructExpr", "FieldByName"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise().flag(FunctionFlag.ALLOW_RENAME)),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                StructKernels().field_by_name,
                NodeOption.text("function", "StructExpr", "FieldByName"),
            ),
        ),
    )
    SELECT_FIELDS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StructExpr.SelectFields"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.try_map_field,
            func=FromOptions(StructClosures().select_fields, SELECTOR),
        ),
        options=Shape.function(FunctionOptions.elementwise().flag(FunctionFlag.ALLOW_RENAME)),
        casts=FunctionCasting(),
        kernel=OneInput(FromOptions(StructKernels().select_fields, SELECTOR)),
    )
    RENAME_FIELDS: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StructExpr.RenameFields"),
        nulls=NullRule.FIRST,
        type_function=Mapper(
            FieldsMapper.map_dtype,
            func=FromOptions(
                StructClosures().rename_fields,
                NodeOption.texts("function", "StructExpr", "RenameFields"),
            ),
        ),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(
            FromOptions(
                StructKernels().rename_fields,
                NodeOption.texts("function", "StructExpr", "RenameFields"),
            ),
        ),
    )
    JSON_ENCODE: ClassVar[Definition] = Definition(
        identity=NodeIdentity("Function", "StructExpr.JsonEncode"),
        type_function=Mapper(FieldsMapper.with_dtype, dtype=DType("String")),
        options=Shape.function(FunctionOptions.elementwise()),
        casts=FunctionCasting(),
        kernel=OneInput(StructKernels.JSON_ENCODE),
    )
