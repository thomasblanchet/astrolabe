"""astrolabe: a mypy plugin that checks the columns of polars frames before the code runs.

A `TypedDataFrame` or a `TypedLazyFrame` carries its columns in its type:
the name and the data type of each one. A polars expression gets a type
too, which says what it reads and what it gives. With the plugin loaded,
mypy reports a missing column or a wrong data type when it checks the code,
and not when the code runs. At run time, each typed frame is an ordinary
polars frame.

This package holds every name that a user imports: the frame types, the
data type markers, the declarations of expression types, and the markers
for the columns that a frame type does not name.

`Decimal` and `Enum` here have the same names as `decimal.Decimal` and
`enum.Enum` of the standard library. A file that needs both can import
the modules, `import decimal` and `import enum`, and write
`decimal.Decimal` and `enum.Enum`.
"""

from astrolabe.declare import (
    InferredDataFrame,
    InferredExpr,
    InferredLazyFrame,
    InferredRemainder,
    infer,
)
from astrolabe.remainder import (
    Prefixed,
    Suffixed,
    other_columns,
)
from astrolabe.remainder import (
    Both,
    Contains,
    EndsWith,
    Except,
    Matching,
    Minus,
    OfType,
    OneOf,
    Only,
    Retyped,
    SelAnd,
    SelNot,
    SelOr,
    SplitColumns,
    StartsWith,
)
from polars.datatypes.classes import (
    FloatType,
    IntegerType,
    NestedType,
    NumericType,
    SignedIntegerType,
    TemporalType,
    UnsignedIntegerType,
)
from astrolabe.expr import EveryDType
from astrolabe.declare import (
    column_of,
    row,
    same,
    some_column,
    subset_of,
    typed_agg,
    typed_agg_selection,
    typed_expr,
    typed_lit,
    typed_selection,
    wrap,
)
from astrolabe.expr import TypedAgg, TypedColumn, TypedExpr, TypedLit
from astrolabe.frame import (
    AnyZone,
    Unchecked,
    Array,
    Column,
    Columns,
    Datetime,
    Decimal,
    Duration,
    Enum,
    List,
    Planned,
    Struct,
    TypedDataFrame,
    TypedLazyFrame,
    TypedSeries,
    assume_schema,
    check_schema,
    column,
    columns,
    force_schema,
)

__all__ = [
    "AnyZone",
    "Array",
    "Both",
    "Column",
    "Columns",
    "Contains",
    "Datetime",
    "Decimal",
    "Duration",
    "EndsWith",
    "Enum",
    "EveryDType",
    "Except",
    "FloatType",
    "InferredDataFrame",
    "InferredExpr",
    "InferredLazyFrame",
    "InferredRemainder",
    "IntegerType",
    "List",
    "Matching",
    "Minus",
    "NestedType",
    "NumericType",
    "OfType",
    "OneOf",
    "Only",
    "Planned",
    "Prefixed",
    "Retyped",
    "SelAnd",
    "SelNot",
    "SelOr",
    "SignedIntegerType",
    "SplitColumns",
    "StartsWith",
    "Struct",
    "Suffixed",
    "TemporalType",
    "TypedAgg",
    "TypedColumn",
    "TypedDataFrame",
    "TypedExpr",
    "TypedLazyFrame",
    "TypedLit",
    "TypedSeries",
    "Unchecked",
    "UnsignedIntegerType",
    "assume_schema",
    "check_schema",
    "column",
    "column_of",
    "columns",
    "force_schema",
    "infer",
    "other_columns",
    "row",
    "same",
    "some_column",
    "subset_of",
    "typed_agg",
    "typed_agg_selection",
    "typed_expr",
    "typed_lit",
    "typed_selection",
    "wrap",
]
