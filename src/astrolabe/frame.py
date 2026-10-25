"""The frame types, the data type markers, and the runtime half of the checker.

The checker reads the types in this module. `TypedDataFrame[Trades]` and
`TypedLazyFrame[Trades]` are frames whose type states their columns, and
`Columns` is the base of a class that lists those columns. The markers
`List`, `Datetime`, `Decimal` and the others write a data type with
parameters in a type annotation, where a polars call such as
`pl.Datetime("us")` cannot stand.

The module also works at run time. A typed frame is a real polars frame.
`TypedDataFrame[Trades].schema` gives the columns that `Trades` declares, and
`assume_schema`, `check_schema` and `force_schema` turn a plain frame into a
typed one. The checker does not use this runtime code: it reads only the
types.
"""

from __future__ import annotations

import types
from functools import cached_property
from typing import (
    Any,
    Concatenate,
    Final,
    ForwardRef,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
    Literal,
    Self,
    TYPE_CHECKING,
    TypedDict,
)

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from collections.abc import Callable
    from typing import TypeGuard

DTypeLike = pl.DataType | type[pl.DataType]
"""A polars data type, written as a class (`pl.Int64`) or as an instance (`pl.Int64()`).

polars accepts both forms, and users write both.
"""


class Columns(TypedDict):
    """The base class of a class that lists the columns of a frame.

    Each field of the class is one column: the name of the field is the name
    of the column, and its annotation is the data type:

    ```python
    class Legislators(Columns):
        first_name: pl.String
        last_name: pl.String
    ```

    `TypedDataFrame[Legislators]` is then a frame with exactly these columns.
    `Columns` declares no column of its own. It is a `TypedDict`, so the
    checker compares two such classes by their fields, and the order of the
    fields does not matter. A class can inherit from several `Columns`
    classes to join their columns. A generic class, for example
    `class Keyed[T: pl.DataType](Columns)`, takes a data type at each use.
    For a column name that is not a Python identifier, use `columns(...)`.

    A subclass must not set `total=False` or `closed=False`. Each annotation
    must be a polars data type or a marker of this module, for example
    `List[pl.Int64]`. The checker reports a class that breaks these rules
    where a frame type first uses it. A class that no frame type uses gets
    no check.

    Example:
        >>> class Prices(Columns):
        ...     price: pl.Int64
        >>> Prices.__required_keys__ == frozenset({"price"})
        True
    """


def columns(name: str, fields: Mapping[str, Any]) -> Any:
    """Declare a `Columns` class from a dict, for column names that are not Python identifiers.

    A class body declares each column as an attribute, so it cannot hold a
    name such as `"1"`, `"v1|a"` or `"total revenue"`. Write
    `Pivoted = columns("Pivoted", {"1": pl.Int64})` in its place. The checker
    reads the call and makes `Pivoted` a `Columns` class with those columns.
    The first argument must be the name that the call is assigned to.

    Args:
        name: The name of the class.
        fields: Each column name, mapped to its data type as polars writes
            it, for example `pl.Datetime("us")`, or `Unchecked`.

    Returns:
        A `Columns` subclass with one field for each column.

    Example:
        >>> Pivoted = columns("Pivoted", {"1": pl.Int64, "v1|a": pl.String})
        >>> Pivoted.__annotations__
        {'1': Int64, 'v1|a': String}
    """
    body = {"__annotations__": dict(fields)}
    return types.new_class(name, (Columns,), exec_body=lambda namespace: namespace.update(body))


class Column[Name, DType: pl.DataType]:
    """One column, with its name and its data type, in a type.

    `Column[Literal["price"], pl.Float64]` is the column `price` of type
    `Float64`. The checker writes this marker in the types that it shows, for
    example in the reads and the result of an expression type, and inside a
    `Struct[...]` for each field. No code creates an instance of the class.

    The runtime parser of this module reads a `Column[...]` inside a
    `Struct[...]`, to build the `pl.Struct` that `.schema` gives.

    Example:
        >>> Column[Literal["foo"], pl.Int64]
        astrolabe.frame.Column[typing.Literal['foo'], polars.datatypes.classes.Int64]
    """


def column(name: str, dtype: object) -> Any:
    """Declare an expression that reads one column and gives it back unchanged.

    The checker reads `Price = column("price", pl.Float64)` and makes `Price`
    the type of `pl.col("price")` on a `Float64` column:
    `TypedColumn[Literal["price"], Float64]`. `Price` can then stand in a
    signature, and each call checks that its argument fits. If the data type
    is one data type, the same name can declare a field inside `Struct[...]`.

    The name is a call argument, not a subscript, because mypy reads a bare
    string in a subscript as a forward reference to a class.

    Args:
        name: The name of the column.
        dtype: The data types that the column can have: a data type as a
            class or an instance, a polars family such as `NumericType`, a
            set of data types, `EveryDType`, or `Unchecked`.

    Returns:
        The marker `Column[Literal[name], dtype]`, at run time.

    Example:
        >>> column("foo", pl.Int64)
        astrolabe.frame.Column[typing.Literal['foo'], polars.datatypes.classes.Int64]
    """
    return Column[Literal[name], dtype]  # type: ignore[valid-type]


class Unchecked(pl.DataType):
    """The data type of a column whose name is known and whose data type is not.

    The checker gives a column this data type when it knows the name but
    cannot prove the data type, for example after `map_elements` with no
    `return_dtype=`. It keeps every other column of the result exact, and does
    not give up on the whole frame. A user can also write it, for example
    `Column[Literal["x"], Unchecked]`, for a column whose data type is not
    worth a check.

    Each later operation accepts an `Unchecked` column, as mypy accepts a
    value of type `Any`. An operation on it gives `Unchecked` again. A check
    that compares data types, for example of a join key, a `pl.concat` or a
    declared parameter, treats `Unchecked` as equal to any data type. At run
    time, `check_schema` does not check the data type of such a column, and
    `force_schema` does not cast it.

    Example:
        >>> Column[Literal["x"], Unchecked]
        astrolabe.frame.Column[typing.Literal['x'], astrolabe.frame.Unchecked]
    """


class List[DType: pl.DataType](pl.DataType):
    """The data type `pl.List(...)`, in a form that a type annotation accepts.

    `pl.List(pl.Int64)` is a call, and a type annotation cannot hold a call.
    So a `Columns` field writes `List[pl.Int64]` in its place. At run time,
    `.schema` turns it into `pl.List(pl.Int64)`.

    Example:
        >>> List[pl.Int64]
        astrolabe.frame.List[polars.datatypes.classes.Int64]
    """


class Array[DType: pl.DataType, Shape: int](pl.DataType):
    """The data type `pl.Array(...)`, in a form that a type annotation accepts.

    `Array[pl.Int64, Literal[3]]` is an array of three `Int64` values. The
    width must be `Literal[N]`, not a bare `int`. At run time, `.schema` turns
    it into `pl.Array(pl.Int64, 3)`.

    Example:
        >>> Array[pl.Int64, Literal[3]]
        astrolabe.frame.Array[polars.datatypes.classes.Int64, typing.Literal[3]]
    """


class Struct[*Fields](pl.DataType):
    """The data type `pl.Struct(...)`, in a form that a type annotation accepts.

    Each argument is one field, written `Column[Literal[name], data_type]`, or
    a name declared with `column(name, data_type)`. At run time, `.schema`
    turns it into a `pl.Struct` with those fields.

    Example:
        >>> field = Column[Literal["bar"], pl.Int64]
        >>> DTypeMarkerResolver(ColumnSchemaParser()).resolve(Struct[field])
        Struct({'bar': Int64})
    """


class Datetime[Unit: str, Zone](pl.Datetime):
    """A `Datetime` data type with its time unit and its time zone, in a type annotation.

    `Unit` must be `Literal["ms"]`, `Literal["us"]` or `Literal["ns"]`. `Zone`
    must be `None` for no time zone, or a string literal such as
    `Literal["UTC"]`. `AnyZone` in that place stands for every zone at once.
    The class subclasses `pl.Datetime`, so a
    selector such as `cs.temporal()` still picks the column. At run time,
    `.schema` turns it into `pl.Datetime(unit, zone)`.

    Example:
        >>> Datetime[Literal["us"], None]
        astrolabe.frame.Datetime[typing.Literal['us'], NoneType]
    """


class AnyZone:
    """Stand for every time zone in the zone position of `Datetime`.

    `cs.datetime("us")` picks a column of the unit `us`, whatever its time
    zone. No list can name every zone, because the set of zone names is open.
    So `Datetime[Literal["us"], AnyZone]` stands for the data types of one
    unit with every zone, the naive one included.

    Example:
        >>> Datetime[Literal["us"], AnyZone]
        astrolabe.frame.Datetime[typing.Literal['us'], astrolabe.frame.AnyZone]
    """


class Duration[Unit: str](pl.Duration):
    """A `Duration` data type with its time unit, in a type annotation.

    `Unit` must be `Literal["ms"]`, `Literal["us"]` or `Literal["ns"]`. The
    class subclasses `pl.Duration`, as `Datetime` subclasses `pl.Datetime`.
    At run time, `.schema` turns it into `pl.Duration(unit)`.

    Example:
        >>> Duration[Literal["ms"]]
        astrolabe.frame.Duration[typing.Literal['ms']]
    """


class Decimal[Precision, Scale: int](pl.Decimal):
    """A `Decimal` data type with its precision and its scale, in a type annotation.

    `Precision` must be `None`, for the largest precision that polars
    allows, or an integer literal such as `Literal[38]`. `Scale` must be an
    integer literal such as `Literal[2]`. The class subclasses `pl.Decimal`.
    At run time, `.schema` turns it into `pl.Decimal(precision, scale)`.

    The class has the same name as `decimal.Decimal` of the standard
    library. A file that needs both can import the module, `import decimal`,
    and write `decimal.Decimal`.

    Example:
        >>> Decimal[None, Literal[2]]
        astrolabe.frame.Decimal[NoneType, typing.Literal[2]]
    """


class Enum[*Categories](pl.Enum):
    """An `Enum` data type with its categories, in order, in a type annotation.

    Each argument must be a string literal, for example
    `Enum[Literal["buy"], Literal["sell"]]`. The categories belong to the
    type: two `Enum` columns with different categories have different data
    types. The class subclasses `pl.Enum`. At run time, `.schema` turns it
    into `pl.Enum([...])`.

    The class has the same name as `enum.Enum` of the standard library. A
    file that needs both can import the module, `import enum`, and write
    `enum.Enum`.

    Example:
        >>> Enum[Literal["a"], Literal["b"]]
        astrolabe.frame.Enum[typing.Literal['a'], typing.Literal['b']]
    """


class Planned[Run: pl.DataType, Planner: pl.DataType](pl.DataType):
    """The two data types of a column of a lazy frame, where polars plans it at another type.

    A lazy frame runs no step until `collect()`. Polars plans each step at the
    data types that it stores for the columns, and these are the planner types
    of the steps before. The data of a column can have another type: in
    `lf.with_columns(d=pl.col("i8") * pl.col("dec"))` over a `Decimal(10, 2)`
    column `dec`, polars plans `d` as `Decimal(10, 2)`, and the run gives a
    `Decimal(38, 2)`. A later step casts `d` at the planner type, and reads
    the data at the run type, so the column needs both types.

    `Run` is the data type of the data, which `collect()` gives. `Planner` is
    the data type that polars stores, which `collect_schema()` gives. The
    checker writes this marker only where the two types differ. At run time,
    `.schema` turns it into the planner type.

    Example:
        >>> wide, narrow = Decimal[Literal[38], Literal[2]], Decimal[Literal[10], Literal[2]]
        >>> DTypeMarkerResolver(ColumnSchemaParser()).resolve(Planned[wide, narrow])
        Decimal(precision=10, scale=2)
    """


class ColumnSchemaParser:
    """Read the columns of a frame type at run time, to give `.schema` its value.

    `TypedDataFrame[Trades].schema` must give a real polars schema: each
    column name mapped to a polars data type. This class reads the `Columns`
    class of a frame type and each `Column[...]` marker, and builds that
    dict. It works on runtime objects from `typing`, and not on mypy types.
    `DTypeMarkerResolver` turns each annotation into a data type.

    Example:
        >>> ColumnSchemaParser().parse_column(Column["foo", pl.Int64])
        ('foo', Int64)
    """

    def __init__(self) -> None:
        """Build the resolver of data types.

        The resolver reads each field of a `Struct[...]` back through this parser.
        """
        self._dtype_resolver = DTypeMarkerResolver(self)

    def parse_column(self, arg: Any) -> tuple[str, DTypeLike] | None:
        """Read one `Column[name, data_type]` marker.

        Args:
            arg: The marker, or any other object.

        Returns:
            The name and the data type, or None if `arg` is not a `Column[...]`
            or one of its parts cannot be read.

        Example:
            >>> parser = ColumnSchemaParser()
            >>> parser.parse_column(Column["foo", pl.Int64])
            ('foo', Int64)
            >>> parser.parse_column(Column[Literal["foo"], pl.Int64])
            ('foo', Int64)
            >>> parser.parse_column(int)
            >>> parser.parse_column(Column[123, pl.Int64])
            >>> parser.parse_column(Column[Literal["foo"], int])
        """
        if get_origin(arg) is not Column:
            return None
        name_arg, dtype_arg = get_args(arg)
        name = self._column_name(name_arg)
        if name is None:
            return None
        resolved_dtype = self._dtype_resolver.resolve(dtype_arg)
        if resolved_dtype is None:
            return None
        return name, resolved_dtype

    def _column_name(self, name_arg: Any) -> str | None:
        """Read the name argument of a `Column[...]`.

        A user can write the name as `Literal["foo"]` or as a bare string.
        `typing` takes a bare string in a subscript for a forward reference to
        a class, and wraps it in a `ForwardRef`. This method takes the name
        back out of it.

        Args:
            name_arg: The name argument.

        Returns:
            The name, or None if the argument is not a string literal or a
            `ForwardRef`.

        Example:
            >>> parser = ColumnSchemaParser()
            >>> parser._column_name(Literal["foo"])
            'foo'
            >>> parser._column_name(ForwardRef("foo"))
            'foo'
            >>> parser._column_name(123)
            >>> parser._column_name(Literal[1])
        """
        if get_origin(name_arg) is Literal:
            (name,) = get_args(name_arg)
            return name if isinstance(name, str) else None
        if isinstance(name_arg, ForwardRef):
            return name_arg.__forward_arg__
        return None

    def extract_schema(self, args: tuple[Any, ...]) -> dict[str, DTypeLike] | None:
        """Build the schema dict from the type arguments of a frame type.

        Only a closed frame type with one `Columns` class has a full schema. An
        open frame type, with a remainder of undeclared columns, has none, and
        neither has a frame type with a type variable. A `Columns` class is a
        `TypedDict` class. If the class is generic and has its arguments, for
        example `Keyed[pl.Int64]`, each type variable in a field reads as its
        argument.

        Args:
            args: The type arguments of the frame type, in order.

        Returns:
            The schema, in the order of the fields of the class. An empty dict
            for no arguments. None for any other arguments, for example a
            `Columns` class and a remainder, a type that is not a `Columns`
            class, or a field that is not a data type.

        Example:
            >>> parser = ColumnSchemaParser()
            >>> class Prices(Columns):
            ...     price: pl.Int64
            >>> parser.extract_schema((Prices,))
            {'price': Int64}
            >>> parser.extract_schema(())
            {}
            >>> parser.extract_schema((int,))
            >>> class Keyed[T: pl.DataType](Columns):
            ...     v: T
            >>> parser.extract_schema((Keyed[pl.Int64],))
            {'v': Int64}
            >>> class Odd(Columns):
            ...     v: int
            >>> parser.extract_schema((Odd,))
        """
        if not args:
            return {}
        if len(args) != 1:
            return None
        candidate = args[0]
        origin = get_origin(candidate) or candidate
        if not is_typeddict(origin):
            return None
        hints = get_type_hints(origin, include_extras=True, localns=self._type_parameters(origin))
        parameters = getattr(origin, "__parameters__", ())
        bound = dict(zip(parameters, get_args(candidate), strict=False))
        substituted = {name: self._substituted(hint, bound) for name, hint in hints.items()}
        schema: dict[str, DTypeLike] = {}
        for name, dtype_arg in substituted.items():
            resolved_dtype = self._dtype_resolver.resolve(dtype_arg)
            if resolved_dtype is None:
                return None
            schema[name] = resolved_dtype
        return schema

    def _type_parameters(self, origin: type) -> dict[str, Any]:
        """Give the type parameters of a generic class, by name, to read its field annotations.

        A field annotation of a generic class, for example `v: T`, names a type
        parameter of the class. Python 3.13 reads those names by itself. Python
        3.12 does not, so the names must reach `get_type_hints` as its local
        names.

        Args:
            origin: The class.

        Returns:
            Each type parameter of the class, by its name. An empty dict for a
            class that is not generic.

        Example:
            >>> class Keyed[T: pl.DataType](Columns):
            ...     v: T
            >>> ColumnSchemaParser()._type_parameters(Keyed)
            {'T': T}
        """
        parameters = getattr(origin, "__type_params__", ())
        return {parameter.__name__: parameter for parameter in parameters}

    def _substituted(self, hint: Any, bound: dict[Any, Any]) -> Any:
        """Put the arguments in the place of the type variables in one field annotation.

        Args:
            hint: The annotation of the field.
            bound: Each type variable, mapped to its argument.

        Returns:
            The annotation with each known type variable replaced. A marker such
            as `List[T]` gets its arguments again, as `List[pl.Int64]`.

        Example:
            >>> from typing import TypeVar
            >>> T = TypeVar("T")
            >>> ColumnSchemaParser()._substituted(List[T], {T: pl.Int64})
            astrolabe.frame.List[polars.datatypes.classes.Int64]
        """
        if hint in bound:
            return bound[hint]
        parameters = getattr(hint, "__parameters__", ())
        if not parameters:
            return hint
        return hint[tuple(bound.get(parameter, parameter) for parameter in parameters)]


class DTypeMarkerResolver:
    """Turn one data type annotation into a real polars data type, at run time.

    An annotation is a polars data type, as a class or an instance, or a
    marker of this module, for example `List[pl.Int64]` or
    `Datetime[Literal["us"], None]`. This class builds the polars object that
    each one stands for, and it works through nested markers. A `Struct[...]`
    holds `Column[...]` fields, so this class uses a `ColumnSchemaParser` to
    read them. The parser also needs this class, so it builds one and passes
    itself in. Each class then reaches the other without a module-level
    function.

    Example:
        >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
        >>> resolver.resolve(pl.Int64)
        Int64
        >>> resolver.resolve(List[pl.Int64])
        List(Int64)
        >>> resolver.resolve(int)
    """

    def __init__(self, column_parser: ColumnSchemaParser) -> None:
        """Keep the parser for `Struct[...]` fields, and map each marker to its method.

        Args:
            column_parser: The parser that reads each `Column[name, data_type]`
                field of a `Struct[...]`.
        """
        self._column_parser = column_parser
        self._resolvers: dict[Any, Callable[[Any], DTypeLike | None]] = {
            List: self._resolve_list_dtype,
            Array: self._resolve_array_dtype,
            Struct: self._resolve_struct_dtype,
            Datetime: self._resolve_datetime_dtype,
            Duration: self._resolve_duration_dtype,
            Decimal: self._resolve_decimal_dtype,
            Enum: self._resolve_enum_dtype,
            Planned: self._resolve_planned_dtype,
        }

    def resolve(self, dtype_arg: Any) -> DTypeLike | None:
        """Turn one annotation into a polars data type.

        A plain polars data type comes back as it is. A marker becomes the data
        type that it stands for, and a marker inside a marker works too, for
        example `List[List[pl.Int64]]`.

        Args:
            dtype_arg: The annotation.

        Returns:
            The data type, or None if the annotation is not a data type or a
            marker with valid arguments.
        """
        if self._is_plain_dtype(dtype_arg):
            return dtype_arg
        handler = self._resolvers.get(get_origin(dtype_arg))
        if handler is None:
            return None
        return handler(dtype_arg)

    def _is_plain_dtype(self, dtype_arg: Any) -> TypeGuard[DTypeLike]:
        """Tell whether an annotation is a polars data type and not a marker.

        Args:
            dtype_arg: The annotation.

        Returns:
            True for a `pl.DataType` class or instance.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._is_plain_dtype(pl.Int64)
            True
            >>> resolver._is_plain_dtype(pl.Int64())
            True
            >>> resolver._is_plain_dtype(int)
            False
        """
        if isinstance(dtype_arg, pl.DataType):
            return True
        return isinstance(dtype_arg, type) and issubclass(dtype_arg, pl.DataType)

    def _resolve_planned_dtype(self, dtype_arg: Any) -> DTypeLike | None:
        """Turn `Planned[...]` into its planner type, which a lazy frame stores.

        Args:
            dtype_arg: The `Planned[...]` annotation.

        Returns:
            The planner type, or None if it is not a data type.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_planned_dtype(Planned[pl.Float64, pl.Float32])
            Float32
        """
        return self.resolve(get_args(dtype_arg)[1])

    def _resolve_list_dtype(self, dtype_arg: Any) -> pl.List | None:
        """Turn `List[...]` into `pl.List(...)`.

        Args:
            dtype_arg: The `List[...]` annotation.

        Returns:
            The `pl.List`, or None if the inner annotation is not a data type.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_list_dtype(List[pl.Int64])
            List(Int64)
            >>> resolver._resolve_list_dtype(List[int])
        """
        (inner_arg,) = get_args(dtype_arg)
        inner = self.resolve(inner_arg)
        if inner is None:
            return None
        return pl.List(inner)

    def _resolve_array_dtype(self, dtype_arg: Any) -> pl.Array | None:
        """Turn `Array[...]` into `pl.Array(...)`.

        Args:
            dtype_arg: The `Array[...]` annotation.

        Returns:
            The `pl.Array`, or None if the inner annotation is not a data type or
            the width is not `Literal[N]`.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_array_dtype(Array[pl.Int64, Literal[3]])
            Array(Int64, shape=(3,))
            >>> resolver._resolve_array_dtype(Array[int, Literal[3]])
            >>> resolver._resolve_array_dtype(Array[pl.Int64, str])
        """
        inner_arg, shape_arg = get_args(dtype_arg)
        inner = self.resolve(inner_arg)
        if inner is None:
            return None
        shape = self._literal_int(shape_arg)
        if shape is None:
            return None
        return pl.Array(inner, shape)

    def _literal_int(self, arg: Any) -> int | None:
        """Read an integer literal, for the width of an `Array` or the digits of a `Decimal`.

        Args:
            arg: The argument.

        Returns:
            The integer, or None if `arg` is not `Literal[N]` with an integer.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._literal_int(Literal[3])
            3
            >>> resolver._literal_int(int)
            >>> resolver._literal_int(Literal["x"])
        """
        if get_origin(arg) is not Literal:
            return None
        (value,) = get_args(arg)
        if not isinstance(value, int):
            return None
        return value

    def _literal_str(self, arg: Any) -> str | None:
        """Read a string literal, for a time unit, a time zone or a category.

        A bare string does not count here, unlike the name of a `Column`. A user
        writes a unit or a category in `Literal[...]`, and a bare string there
        is a mistake.

        Args:
            arg: The argument.

        Returns:
            The string, or None if `arg` is not `Literal["..."]` with a string.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._literal_str(Literal["us"])
            'us'
            >>> resolver._literal_str(str)
            >>> resolver._literal_str(Literal[3])
        """
        if get_origin(arg) is not Literal:
            return None
        (value,) = get_args(arg)
        if not isinstance(value, str):
            return None
        return value

    def _resolve_struct_dtype(self, dtype_arg: Any) -> pl.Struct | None:
        """Turn `Struct[...]` into `pl.Struct(...)`.

        Args:
            dtype_arg: The `Struct[...]` annotation.

        Returns:
            The `pl.Struct`, or None if a field is not a valid `Column[...]`.

        Example:
            >>> field = Column[Literal["bar"], pl.Int64]
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_struct_dtype(Struct[field])
            Struct({'bar': Int64})
            >>> resolver._resolve_struct_dtype(Struct[int])
        """
        fields: dict[str, DTypeLike] = {}
        for field_arg in get_args(dtype_arg):
            parsed = self._column_parser.parse_column(field_arg)
            if parsed is None:
                return None
            name, resolved = parsed
            fields[name] = resolved
        return pl.Struct(fields)

    def _resolve_datetime_dtype(self, dtype_arg: Any) -> pl.Datetime | None:
        """Turn `Datetime[...]` into `pl.Datetime(...)`.

        Args:
            dtype_arg: The `Datetime[...]` annotation.

        Returns:
            The `pl.Datetime`, or None if the unit is not a string literal or the
            zone is neither `None` nor a string literal.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_datetime_dtype(Datetime[Literal["us"], None])
            Datetime(time_unit='us', time_zone=None)
            >>> resolver._resolve_datetime_dtype(Datetime[Literal["us"], Literal["UTC"]])
            Datetime(time_unit='us', time_zone='UTC')
            >>> resolver._resolve_datetime_dtype(Datetime[str, None])
            >>> resolver._resolve_datetime_dtype(Datetime[Literal["us"], int])
        """
        unit_arg, zone_arg = get_args(dtype_arg)
        unit = self._literal_str(unit_arg)
        if unit is None:
            return None
        if zone_arg is type(None):
            return pl.Datetime(unit, None)  # type: ignore[arg-type]
        zone = self._literal_str(zone_arg)
        if zone is None:
            return None
        return pl.Datetime(unit, zone)  # type: ignore[arg-type]

    def _resolve_duration_dtype(self, dtype_arg: Any) -> pl.Duration | None:
        """Turn `Duration[...]` into `pl.Duration(...)`.

        Args:
            dtype_arg: The `Duration[...]` annotation.

        Returns:
            The `pl.Duration`, or None if the unit is not a string literal.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_duration_dtype(Duration[Literal["ms"]])
            Duration(time_unit='ms')
            >>> resolver._resolve_duration_dtype(Duration[str])
        """
        (unit_arg,) = get_args(dtype_arg)
        unit = self._literal_str(unit_arg)
        if unit is None:
            return None
        return pl.Duration(unit)  # type: ignore[arg-type]

    def _resolve_decimal_dtype(self, dtype_arg: Any) -> pl.Decimal | None:
        """Turn `Decimal[...]` into `pl.Decimal(...)`.

        Args:
            dtype_arg: The `Decimal[...]` annotation.

        Returns:
            The `pl.Decimal`, or None if the scale is not an integer literal or
            the precision is neither `None` nor an integer literal.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_decimal_dtype(Decimal[None, Literal[2]])
            Decimal(precision=38, scale=2)
            >>> resolver._resolve_decimal_dtype(Decimal[Literal[38], Literal[2]])
            Decimal(precision=38, scale=2)
            >>> resolver._resolve_decimal_dtype(Decimal[None, str])
            >>> resolver._resolve_decimal_dtype(Decimal[str, Literal[2]])
        """
        precision_arg, scale_arg = get_args(dtype_arg)
        scale = self._literal_int(scale_arg)
        if scale is None:
            return None
        if precision_arg is type(None):
            return pl.Decimal(None, scale)
        precision = self._literal_int(precision_arg)
        if precision is None:
            return None
        return pl.Decimal(precision, scale)

    def _resolve_enum_dtype(self, dtype_arg: Any) -> pl.Enum | None:
        """Turn `Enum[...]` into `pl.Enum(...)`, with the categories in order.

        Args:
            dtype_arg: The `Enum[...]` annotation.

        Returns:
            The `pl.Enum`, or None if a category is not a string literal.

        Example:
            >>> resolver = DTypeMarkerResolver(ColumnSchemaParser())
            >>> resolver._resolve_enum_dtype(Enum[Literal["a"], Literal["b"]])
            Enum(categories=['a', 'b'])
            >>> resolver._resolve_enum_dtype(Enum[int])
        """
        categories: list[str] = []
        for category_arg in get_args(dtype_arg):
            category = self._literal_str(category_arg)
            if category is None:
                return None
            categories.append(category)
        return pl.Enum(categories)


class _SchemaDescriptor:
    """Give `.schema` on a typed frame class the declared columns, and on a frame its real ones.

    On the class, for example `TypedDataFrame[Trades].schema`, the value is
    the dict of columns that `Trades` declares. On a frame, the value is the
    schema that polars reads from the data. So the declared columns never
    hide the real ones: code that reads `.schema` on a frame sees what the
    frame holds.
    """

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Give the declared schema or the real one.

        On a frame, the method finds the `schema` attribute that polars
        defines, in the bases after the typed class. If that attribute is a
        descriptor, the method binds it to the frame.

        Args:
            obj: The frame that `.schema` is read on, or None when it is read on
                the class.
            objtype: The class that `.schema` is read through.

        Returns:
            The declared schema when it is read on the class, else the real
            schema of the frame.

        Raises:
            AttributeError: If no polars base class defines `schema`.
        """
        if obj is None:
            return getattr(objtype, "_declared_schema", None)
        mro = type(obj).__mro__
        for base in mro[mro.index(_TypedFrameBase) + 1 :]:
            real = base.__dict__.get("schema")
            if real is None:
                continue
            if hasattr(real, "__get__"):
                return real.__get__(obj, type(obj))
            return real
        msg = "schema"
        raise AttributeError(msg)


class _TypedFrameBase:
    """Give `TypedDataFrame` and `TypedLazyFrame` a runtime class for each subscript.

    A plain generic class forgets its type arguments at run time. Then
    `TypedDataFrame[Trades].schema` has nothing to read. So a subscript here
    builds a real subclass that keeps the schema of its arguments. A cache
    keeps that subclass, and the same subscript always gives the same class.
    """

    _synthesized_cache: Final[dict[tuple[Any, ...], type]] = {}
    _declared_schema: Final[dict[str, DTypeLike] | None] = None
    schema = _SchemaDescriptor()

    def __class_getitem__(cls, item: Any) -> Any:
        """Build, or take from the cache, the subclass for one subscript.

        Args:
            item: The type argument or arguments of the subscript.

        Returns:
            A subclass of `cls` whose `_declared_schema` holds the schema of
            `item`. That schema is None if `item` has no full schema.
        """
        args = item if isinstance(item, tuple) else (item,)
        cache_key = (cls, args)
        cached = cls._synthesized_cache.get(cache_key)
        if cached is not None:
            return cached

        schema = ColumnSchemaParser().extract_schema(args)
        synthesized = type(cls.__name__, (cls,), {"_declared_schema": schema})
        cls._synthesized_cache[cache_key] = synthesized
        return synthesized


class TypedDataFrame[Schema, Rest](_TypedFrameBase, pl.DataFrame):
    """A `pl.DataFrame` whose type states its columns.

    `TypedDataFrame[Trades]` is a frame with the columns of the `Columns`
    class `Trades`, each with its data type. The checker reads this type at
    each operation. It reports a column that the frame does not have, and it
    computes the columns of the result. `TypedDataFrame[Trades, Rest]`, with
    `Rest = other_columns()`, is an open frame: it has the columns of
    `Trades` and can hold any other columns, which `Rest` stands for. A
    closed frame shows `None` in that place.

    At run time the class is a `pl.DataFrame`, and the data does not use the
    type arguments. `TypedDataFrame[Trades].schema` gives the columns that
    `Trades` declares. `assume_schema`, `check_schema` and `force_schema` turn
    a plain frame into a typed one.

    polars declares most methods of `pl.DataFrame` to return a plain
    `DataFrame`, and that type holds no columns. So this class declares each
    method that keeps the columns, for example `sort` and `head`, to return
    `Self`, and the result keeps the type of the receiver. The checker types
    the methods that change the columns, for example `select`.

    Example:
        >>> class Prices(Columns):
        ...     foo: pl.String
        ...     bar: pl.Int64
        >>> Frame = TypedDataFrame[Prices]
        >>> Frame.schema
        {'foo': String, 'bar': Int64}
    """

    def select(self, *args: Any, **kwargs: Any) -> Self:
        """Select or compute columns. The checker gives the result its type.

        The checker's hook reads the arguments and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.DataFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.select`.
            **kwargs: The keyword arguments of `pl.DataFrame.select`.

        Returns:
            The result of `pl.DataFrame.select`.
        """
        return super().select(*args, **kwargs)  # type: ignore[return-value]

    def drop(self, *args: Any, strict: bool = True) -> Self:
        """Delete columns. The checker gives the result its type.

        The checker's hook reads the names and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.DataFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `strict`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.drop`.
            strict: If True, polars raises an error for a name that the
                frame does not have. If False, polars ignores that name.
                The parameter has its own name in this signature so that
                the checker's hook can read it: the value decides whether
                a missing name is an error.

        Returns:
            The result of `pl.DataFrame.drop`.
        """
        return super().drop(*args, strict=strict)  # type: ignore[return-value]

    def rename(self, *args: Any, **kwargs: Any) -> Self:
        """Rename columns. The checker gives the result its type.

        The checker's hook reads the mapping and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.DataFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.rename`.
            **kwargs: The keyword arguments of `pl.DataFrame.rename`.

        Returns:
            The result of `pl.DataFrame.rename`.
        """
        return super().rename(*args, **kwargs)  # type: ignore[return-value]

    def with_columns(self, *args: Any, **kwargs: Any) -> Self:
        """Add or replace columns. The checker gives the result its type.

        The checker's hook reads the arguments and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.DataFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.with_columns`.
            **kwargs: The keyword arguments of `pl.DataFrame.with_columns`.

        Returns:
            The result of `pl.DataFrame.with_columns`.
        """
        return super().with_columns(*args, **kwargs)  # type: ignore[return-value]

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the rows where the predicates hold, and keep the type of the frame.

        polars declares `pl.DataFrame.filter` to return a plain `DataFrame`, and
        that type holds no columns. A filter deletes rows and never changes a
        column. So this method returns `Self`, and the result keeps the columns of
        the receiver. The checker's hook still checks each predicate: it must read
        columns that the frame has, and give a `Boolean`.

        Args:
            *args: The positional arguments of `pl.DataFrame.filter`.
            **kwargs: The keyword arguments of `pl.DataFrame.filter`.

        Returns:
            The result of `pl.DataFrame.filter`.
        """
        return super().filter(*args, **kwargs)  # type: ignore[return-value]

    def sort(self, *args: Any, **kwargs: Any) -> Self:
        """Sort the rows, and keep the type of the frame.

        polars declares `pl.DataFrame.sort` to return a plain `DataFrame`, and that
        type holds no columns. A sort changes only the order of the rows. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.sort`.
            **kwargs: The keyword arguments of `pl.DataFrame.sort`.

        Returns:
            The result of `pl.DataFrame.sort`.
        """
        return super().sort(*args, **kwargs)  # type: ignore[return-value]

    def unique(self, *args: Any, **kwargs: Any) -> Self:
        """Keep one row of each set of equal rows, and keep the type of the frame.

        polars declares `pl.DataFrame.unique` to return a plain `DataFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.unique`.
            **kwargs: The keyword arguments of `pl.DataFrame.unique`.

        Returns:
            The result of `pl.DataFrame.unique`.
        """
        return super().unique(*args, **kwargs)  # type: ignore[return-value]

    def head(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` rows, and keep the type of the frame.

        polars declares `pl.DataFrame.head` to return a plain `DataFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.head`.
            **kwargs: The keyword arguments of `pl.DataFrame.head`.

        Returns:
            The result of `pl.DataFrame.head`.
        """
        return super().head(*args, **kwargs)  # type: ignore[return-value]

    def tail(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the last `n` rows, and keep the type of the frame.

        polars declares `pl.DataFrame.tail` to return a plain `DataFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.tail`.
            **kwargs: The keyword arguments of `pl.DataFrame.tail`.

        Returns:
            The result of `pl.DataFrame.tail`.
        """
        return super().tail(*args, **kwargs)  # type: ignore[return-value]

    def limit(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` rows, as `head` does, and keep the type of the frame.

        polars declares `pl.DataFrame.limit` to return a plain `DataFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.limit`.
            **kwargs: The keyword arguments of `pl.DataFrame.limit`.

        Returns:
            The result of `pl.DataFrame.limit`.
        """
        return super().limit(*args, **kwargs)  # type: ignore[return-value]

    def reverse(self, *args: Any, **kwargs: Any) -> Self:
        """Reverse the order of the rows, and keep the type of the frame.

        polars declares `pl.DataFrame.reverse` to return a plain `DataFrame`, and
        that type holds no columns. It changes only the order of the rows. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.reverse`.
            **kwargs: The keyword arguments of `pl.DataFrame.reverse`.

        Returns:
            The result of `pl.DataFrame.reverse`.
        """
        return super().reverse(*args, **kwargs)  # type: ignore[return-value]

    def drop_nulls(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows that hold a null, and keep the type of the frame.

        polars declares `pl.DataFrame.drop_nulls` to return a plain `DataFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.drop_nulls`.
            **kwargs: The keyword arguments of `pl.DataFrame.drop_nulls`.

        Returns:
            The result of `pl.DataFrame.drop_nulls`.
        """
        return super().drop_nulls(*args, **kwargs)  # type: ignore[return-value]

    def clone(self, *args: Any, **kwargs: Any) -> Self:
        """Copy the frame, and keep its type.

        polars declares `pl.DataFrame.clone` to return a plain `DataFrame`, and that
        type holds no columns. The copy has the same columns. So this method returns
        `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.clone`.
            **kwargs: The keyword arguments of `pl.DataFrame.clone`.

        Returns:
            The result of `pl.DataFrame.clone`.
        """
        return super().clone(*args, **kwargs)  # type: ignore[return-value]

    def gather_every(self, *args: Any, **kwargs: Any) -> Self:
        """Keep every `n`-th row, and keep the type of the frame.

        polars declares `pl.DataFrame.gather_every` to return a plain `DataFrame`,
        and that type holds no columns. It deletes rows and never changes a column.
        So this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.gather_every`.
            **kwargs: The keyword arguments of `pl.DataFrame.gather_every`.

        Returns:
            The result of `pl.DataFrame.gather_every`.
        """
        return super().gather_every(*args, **kwargs)  # type: ignore[return-value]

    def slice(self, *args: Any, **kwargs: Any) -> Self:
        """Keep a range of rows, and keep the type of the frame.

        polars declares `pl.DataFrame.slice` to return a plain `DataFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.slice`.
            **kwargs: The keyword arguments of `pl.DataFrame.slice`.

        Returns:
            The result of `pl.DataFrame.slice`.
        """
        return super().slice(*args, **kwargs)  # type: ignore[return-value]

    def shift(self, *args: Any, **kwargs: Any) -> Self:
        """Move the values of each column down or up, and keep the type of the frame.

        polars declares `pl.DataFrame.shift` to return a plain `DataFrame`, and that
        type holds no columns. Nulls fill the rows that the move leaves empty. Each
        column keeps its name and its data type. So this method returns `Self`, and
        the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.shift`.
            **kwargs: The keyword arguments of `pl.DataFrame.shift`.

        Returns:
            The result of `pl.DataFrame.shift`.
        """
        return super().shift(*args, **kwargs)  # type: ignore[return-value]

    def drop_nans(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows that hold a NaN value, and keep the type of the frame.

        polars declares `pl.DataFrame.drop_nans` to return a plain `DataFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.drop_nans`.
            **kwargs: The keyword arguments of `pl.DataFrame.drop_nans`.

        Returns:
            The result of `pl.DataFrame.drop_nans`.
        """
        return super().drop_nans(*args, **kwargs)  # type: ignore[return-value]

    def gather(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the rows at the given positions, and keep the type of the frame.

        polars declares `pl.DataFrame.gather` to return a plain `DataFrame`, and
        that type holds no columns. It picks rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.gather`.
            **kwargs: The keyword arguments of `pl.DataFrame.gather`.

        Returns:
            The result of `pl.DataFrame.gather`.
        """
        return super().gather(*args, **kwargs)  # type: ignore[return-value]

    def remove(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows where the predicates hold, and keep the type of the frame.

        polars declares `pl.DataFrame.remove` to return a plain `DataFrame`, and
        that type holds no columns. It is the opposite of `filter`, and it never
        changes a column. So this method returns `Self`, and the result keeps the
        columns of the receiver. The checker's hook still checks each predicate: it
        must read columns that the frame has, and give a `Boolean`.

        Args:
            *args: The positional arguments of `pl.DataFrame.remove`.
            **kwargs: The keyword arguments of `pl.DataFrame.remove`.

        Returns:
            The result of `pl.DataFrame.remove`.
        """
        return super().remove(*args, **kwargs)  # type: ignore[return-value]

    def top_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` rows with the largest values, and keep the type of the frame.

        polars declares `pl.DataFrame.top_k` to return a plain `DataFrame`, and that
        type holds no columns. It picks rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.top_k`.
            **kwargs: The keyword arguments of `pl.DataFrame.top_k`.

        Returns:
            The result of `pl.DataFrame.top_k`.
        """
        return super().top_k(*args, **kwargs)  # type: ignore[return-value]

    def bottom_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` rows with the smallest values, and keep the type of the frame.

        polars declares `pl.DataFrame.bottom_k` to return a plain `DataFrame`, and
        that type holds no columns. It picks rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.DataFrame.bottom_k`.
            **kwargs: The keyword arguments of `pl.DataFrame.bottom_k`.

        Returns:
            The result of `pl.DataFrame.bottom_k`.
        """
        return super().bottom_k(*args, **kwargs)  # type: ignore[return-value]

    def select_seq(self, *args: Any, **kwargs: Any) -> Self:
        """Select or compute columns as `select` does, and run the expressions in turn.

        The result has the same columns as with `select`, so the same hook of the
        checker computes them. If the hook cannot compute the columns, the result is
        a plain `pl.DataFrame`. The hook reads the arguments by the parameter
        names of this signature, `args` and `kwargs`, and polars gives its parameters
        other names, so the override must stay. The declared return type `Self`
        applies only where the checker is not loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.select_seq`.
            **kwargs: The keyword arguments of `pl.DataFrame.select_seq`.

        Returns:
            The result of `pl.DataFrame.select_seq`.
        """
        return super().select_seq(*args, **kwargs)  # type: ignore[return-value]

    def with_columns_seq(self, *args: Any, **kwargs: Any) -> Self:
        """Add or replace columns as `with_columns` does, and run the expressions in turn.

        The result has the same columns as with `with_columns`, so the same hook of
        the checker computes them. If the hook cannot compute the columns, the
        result is a plain `pl.DataFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars gives its
        parameters other names, so the override must stay. The declared return type
        `Self` applies only where the checker is not loaded.

        Args:
            *args: The positional arguments of `pl.DataFrame.with_columns_seq`.
            **kwargs: The keyword arguments of `pl.DataFrame.with_columns_seq`.

        Returns:
            The result of `pl.DataFrame.with_columns_seq`.
        """
        return super().with_columns_seq(*args, **kwargs)  # type: ignore[return-value]

    def merge_sorted(
        self,
        other: pl.DataFrame,
        key: str | Sequence[str],
        *,
        maintain_order: bool = False,
    ) -> Self:
        """Merge the rows of another sorted frame into this one, and keep the type of the frame.

        polars declares `pl.DataFrame.merge_sorted` to return a plain `DataFrame`,
        and that type holds no columns. polars raises an error if the other frame
        has other columns. So this method returns `Self`, and the result keeps the
        columns of the receiver. The checker's hook reports the other frame if its
        type proves that its columns differ. The parameters are those of polars, so
        the hook finds `other` whether the caller passes it by position or by name.

        Args:
            other: The other sorted frame.
            key: The column or columns that both frames are sorted by.
            maintain_order: Whether rows with equal keys keep their order.

        Returns:
            The result of `pl.DataFrame.merge_sorted`.
        """
        return super().merge_sorted(other, key, maintain_order=maintain_order)  # type: ignore[return-value]

    def clear(self, *args: Any, **kwargs: Any) -> Self:
        """Give a frame with the same columns and no rows, or `n` rows of nulls.

        polars declares `pl.DataFrame.clear` to return a plain `DataFrame`, and that
        type holds no columns. The result has the same columns as the receiver, so
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.DataFrame.clear`.
            **kwargs: The keyword arguments of `pl.DataFrame.clear`.

        Returns:
            The result of `pl.DataFrame.clear`.
        """
        return super().clear(*args, **kwargs)  # type: ignore[return-value]

    def rechunk(self, *args: Any, **kwargs: Any) -> Self:
        """Put the data of each column in one block of memory, and keep the type of the frame.

        polars declares `pl.DataFrame.rechunk` to return a plain `DataFrame`, and
        that type holds no columns. It changes how polars stores the data, and not
        the rows or the columns. So this method returns `Self`, and the result keeps
        the columns of the receiver. `pl.LazyFrame` has no `rechunk`, so
        `TypedLazyFrame` has no such method.

        Args:
            *args: The positional arguments of `pl.DataFrame.rechunk`.
            **kwargs: The keyword arguments of `pl.DataFrame.rechunk`.

        Returns:
            The result of `pl.DataFrame.rechunk`.
        """
        return super().rechunk(*args, **kwargs)  # type: ignore[return-value]

    def sample(self, *args: Any, **kwargs: Any) -> Self:
        """Keep a random sample of the rows, and keep the type of the frame.

        polars declares `pl.DataFrame.sample` to return a plain `DataFrame`, and
        that type holds no columns. It picks rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver. `pl.LazyFrame` has no `sample`, so `TypedLazyFrame` has no such
        method.

        Args:
            *args: The positional arguments of `pl.DataFrame.sample`.
            **kwargs: The keyword arguments of `pl.DataFrame.sample`.

        Returns:
            The result of `pl.DataFrame.sample`.
        """
        return super().sample(*args, **kwargs)  # type: ignore[return-value]

    def vstack(self, other: pl.DataFrame, *, in_place: bool = False) -> Self:
        """Add the rows of another frame below the rows of this one, and keep the type of the frame.

        polars declares `pl.DataFrame.vstack` to return a plain `DataFrame`, and
        that type holds no columns. polars raises an error if the other frame has
        other columns. So this method returns `Self`, and the result keeps the
        columns of the receiver. The checker's hook reports the other frame if its
        type proves that its columns differ. `pl.LazyFrame` has no `vstack`, so
        `TypedLazyFrame` has no such method.

        Args:
            other: The frame whose rows to add.
            in_place: Whether to change this frame instead of building a new one.

        Returns:
            The result of `pl.DataFrame.vstack`.
        """
        return super().vstack(other, in_place=in_place)  # type: ignore[return-value]

    def extend(self, other: pl.DataFrame) -> Self:
        """Add the rows of another frame to this one in place, and keep the type of the frame.

        polars declares `pl.DataFrame.extend` to return a plain `DataFrame`, and
        that type holds no columns. As in `vstack`, polars raises an error if the
        other frame has other columns. So this method returns `Self`, and the result
        keeps the columns of the receiver. The checker's hook reports the other
        frame if its type proves that its columns differ. `pl.LazyFrame` has no
        `extend`, so `TypedLazyFrame` has no such method.

        Args:
            other: The frame whose rows to add.

        Returns:
            The result of `pl.DataFrame.extend`.
        """
        return super().extend(other)  # type: ignore[return-value]

    def pipe[**P, T](
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Call a function with the frame as its first argument.

        polars types the first parameter of `function` as a plain `DataFrame`. A
        parameter type of a `Callable` is contravariant, so mypy then rejects a
        function whose first parameter is a typed frame, for example
        `TypedDataFrame[Trades]`. This signature takes the type of the receiver in
        its place.

        Args:
            function: The function to call. Its first parameter accepts the
                frame.
            *args: More positional arguments for `function`, after the frame.
            **kwargs: The keyword arguments for `function`.

        Returns:
            What `function` returns.
        """
        return function(self, *args, **kwargs)


class TypedLazyFrame[Schema, Rest](_TypedFrameBase, pl.LazyFrame):
    """A `pl.LazyFrame` whose type states its columns.

    `TypedLazyFrame[Trades]` is a lazy frame with the columns of the
    `Columns` class `Trades`, each with its data type. The checker reads this
    type at each operation, as it does for `TypedDataFrame`, and
    `TypedLazyFrame[Trades, Rest]` is an open frame in the same way.
    `.collect()` gives a `TypedDataFrame` with the same columns.

    At run time the class is a `pl.LazyFrame`, and `TypedLazyFrame[Trades].schema`
    gives the columns that `Trades` declares. As in `TypedDataFrame`, each
    method that keeps the columns returns `Self`, because polars declares it
    to return a plain `LazyFrame`.

    Example:
        >>> class Prices(Columns):
        ...     foo: pl.String
        ...     bar: pl.Int64
        >>> Frame = TypedLazyFrame[Prices]
        >>> Frame.schema
        {'foo': String, 'bar': Int64}
    """

    def select(self, *args: Any, **kwargs: Any) -> Self:
        """Select or compute columns. The checker gives the result its type.

        The checker's hook reads the arguments and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.LazyFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.select`.
            **kwargs: The keyword arguments of `pl.LazyFrame.select`.

        Returns:
            The result of `pl.LazyFrame.select`.
        """
        return super().select(*args, **kwargs)  # type: ignore[return-value]

    def drop(self, *args: Any, strict: bool = True) -> Self:
        """Delete columns. The checker gives the result its type.

        The checker's hook reads the names and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.LazyFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `strict`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.drop`.
            strict: If True, polars raises an error for a name that the
                frame does not have. If False, polars ignores that name.
                The parameter has its own name in this signature so that
                the checker's hook can read it: the value decides whether
                a missing name is an error.

        Returns:
            The result of `pl.LazyFrame.drop`.
        """
        return super().drop(*args, strict=strict)  # type: ignore[return-value]

    def rename(self, *args: Any, **kwargs: Any) -> Self:
        """Rename columns. The checker gives the result its type.

        The checker's hook reads the mapping and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.LazyFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.rename`.
            **kwargs: The keyword arguments of `pl.LazyFrame.rename`.

        Returns:
            The result of `pl.LazyFrame.rename`.
        """
        return super().rename(*args, **kwargs)  # type: ignore[return-value]

    def with_columns(self, *args: Any, **kwargs: Any) -> Self:
        """Add or replace columns. The checker gives the result its type.

        The checker's hook reads the arguments and the columns of the receiver, and
        computes the columns of the result. If the hook cannot compute the columns,
        the result is a plain `pl.LazyFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars
        gives its parameters other names, so the override must stay. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.with_columns`.
            **kwargs: The keyword arguments of `pl.LazyFrame.with_columns`.

        Returns:
            The result of `pl.LazyFrame.with_columns`.
        """
        return super().with_columns(*args, **kwargs)  # type: ignore[return-value]

    def filter(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the rows where the predicates hold, and keep the type of the frame.

        polars declares `pl.LazyFrame.filter` to return a plain `LazyFrame`, and
        that type holds no columns. A filter deletes rows and never changes a
        column. So this method returns `Self`, and the result keeps the columns of
        the receiver. The checker's hook still checks each predicate: it must read
        columns that the frame has, and give a `Boolean`.

        Args:
            *args: The positional arguments of `pl.LazyFrame.filter`.
            **kwargs: The keyword arguments of `pl.LazyFrame.filter`.

        Returns:
            The result of `pl.LazyFrame.filter`.
        """
        return super().filter(*args, **kwargs)  # type: ignore[return-value]

    def sort(self, *args: Any, **kwargs: Any) -> Self:
        """Sort the rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.sort` to return a plain `LazyFrame`, and that
        type holds no columns. A sort changes only the order of the rows. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.sort`.
            **kwargs: The keyword arguments of `pl.LazyFrame.sort`.

        Returns:
            The result of `pl.LazyFrame.sort`.
        """
        return super().sort(*args, **kwargs)  # type: ignore[return-value]

    def unique(self, *args: Any, **kwargs: Any) -> Self:
        """Keep one row of each set of equal rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.unique` to return a plain `LazyFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.unique`.
            **kwargs: The keyword arguments of `pl.LazyFrame.unique`.

        Returns:
            The result of `pl.LazyFrame.unique`.
        """
        return super().unique(*args, **kwargs)  # type: ignore[return-value]

    def head(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.head` to return a plain `LazyFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.head`.
            **kwargs: The keyword arguments of `pl.LazyFrame.head`.

        Returns:
            The result of `pl.LazyFrame.head`.
        """
        return super().head(*args, **kwargs)  # type: ignore[return-value]

    def tail(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the last `n` rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.tail` to return a plain `LazyFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.tail`.
            **kwargs: The keyword arguments of `pl.LazyFrame.tail`.

        Returns:
            The result of `pl.LazyFrame.tail`.
        """
        return super().tail(*args, **kwargs)  # type: ignore[return-value]

    def limit(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` rows, as `head` does, and keep the type of the frame.

        polars declares `pl.LazyFrame.limit` to return a plain `LazyFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.limit`.
            **kwargs: The keyword arguments of `pl.LazyFrame.limit`.

        Returns:
            The result of `pl.LazyFrame.limit`.
        """
        return super().limit(*args, **kwargs)  # type: ignore[return-value]

    def reverse(self, *args: Any, **kwargs: Any) -> Self:
        """Reverse the order of the rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.reverse` to return a plain `LazyFrame`, and
        that type holds no columns. It changes only the order of the rows. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.reverse`.
            **kwargs: The keyword arguments of `pl.LazyFrame.reverse`.

        Returns:
            The result of `pl.LazyFrame.reverse`.
        """
        return super().reverse(*args, **kwargs)  # type: ignore[return-value]

    def drop_nulls(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows that hold a null, and keep the type of the frame.

        polars declares `pl.LazyFrame.drop_nulls` to return a plain `LazyFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.drop_nulls`.
            **kwargs: The keyword arguments of `pl.LazyFrame.drop_nulls`.

        Returns:
            The result of `pl.LazyFrame.drop_nulls`.
        """
        return super().drop_nulls(*args, **kwargs)  # type: ignore[return-value]

    def clone(self, *args: Any, **kwargs: Any) -> Self:
        """Copy the frame, and keep its type.

        polars declares `pl.LazyFrame.clone` to return a plain `LazyFrame`, and that
        type holds no columns. The copy has the same columns. So this method returns
        `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.clone`.
            **kwargs: The keyword arguments of `pl.LazyFrame.clone`.

        Returns:
            The result of `pl.LazyFrame.clone`.
        """
        return super().clone(*args, **kwargs)  # type: ignore[return-value]

    def gather_every(self, *args: Any, **kwargs: Any) -> Self:
        """Keep every `n`-th row, and keep the type of the frame.

        polars declares `pl.LazyFrame.gather_every` to return a plain `LazyFrame`,
        and that type holds no columns. It deletes rows and never changes a column.
        So this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.gather_every`.
            **kwargs: The keyword arguments of `pl.LazyFrame.gather_every`.

        Returns:
            The result of `pl.LazyFrame.gather_every`.
        """
        return super().gather_every(*args, **kwargs)  # type: ignore[return-value]

    def slice(self, *args: Any, **kwargs: Any) -> Self:
        """Keep a range of rows, and keep the type of the frame.

        polars declares `pl.LazyFrame.slice` to return a plain `LazyFrame`, and that
        type holds no columns. It deletes rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.slice`.
            **kwargs: The keyword arguments of `pl.LazyFrame.slice`.

        Returns:
            The result of `pl.LazyFrame.slice`.
        """
        return super().slice(*args, **kwargs)  # type: ignore[return-value]

    def shift(self, *args: Any, **kwargs: Any) -> Self:
        """Move the values of each column down or up, and keep the type of the frame.

        polars declares `pl.LazyFrame.shift` to return a plain `LazyFrame`, and that
        type holds no columns. Nulls fill the rows that the move leaves empty. Each
        column keeps its name and its data type. So this method returns `Self`, and
        the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.shift`.
            **kwargs: The keyword arguments of `pl.LazyFrame.shift`.

        Returns:
            The result of `pl.LazyFrame.shift`.
        """
        return super().shift(*args, **kwargs)  # type: ignore[return-value]

    def drop_nans(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows that hold a NaN value, and keep the type of the frame.

        polars declares `pl.LazyFrame.drop_nans` to return a plain `LazyFrame`, and
        that type holds no columns. It deletes rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.drop_nans`.
            **kwargs: The keyword arguments of `pl.LazyFrame.drop_nans`.

        Returns:
            The result of `pl.LazyFrame.drop_nans`.
        """
        return super().drop_nans(*args, **kwargs)  # type: ignore[return-value]

    def gather(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the rows at the given positions, and keep the type of the frame.

        polars declares `pl.LazyFrame.gather` to return a plain `LazyFrame`, and
        that type holds no columns. It picks rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.gather`.
            **kwargs: The keyword arguments of `pl.LazyFrame.gather`.

        Returns:
            The result of `pl.LazyFrame.gather`.
        """
        return super().gather(*args, **kwargs)  # type: ignore[return-value]

    def remove(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the rows where the predicates hold, and keep the type of the frame.

        polars declares `pl.LazyFrame.remove` to return a plain `LazyFrame`, and
        that type holds no columns. It is the opposite of `filter`, and it never
        changes a column. So this method returns `Self`, and the result keeps the
        columns of the receiver. The checker's hook still checks each predicate: it
        must read columns that the frame has, and give a `Boolean`.

        Args:
            *args: The positional arguments of `pl.LazyFrame.remove`.
            **kwargs: The keyword arguments of `pl.LazyFrame.remove`.

        Returns:
            The result of `pl.LazyFrame.remove`.
        """
        return super().remove(*args, **kwargs)  # type: ignore[return-value]

    def top_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` rows with the largest values, and keep the type of the frame.

        polars declares `pl.LazyFrame.top_k` to return a plain `LazyFrame`, and that
        type holds no columns. It picks rows and never changes a column. So this
        method returns `Self`, and the result keeps the columns of the receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.top_k`.
            **kwargs: The keyword arguments of `pl.LazyFrame.top_k`.

        Returns:
            The result of `pl.LazyFrame.top_k`.
        """
        return super().top_k(*args, **kwargs)  # type: ignore[return-value]

    def bottom_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` rows with the smallest values, and keep the type of the frame.

        polars declares `pl.LazyFrame.bottom_k` to return a plain `LazyFrame`, and
        that type holds no columns. It picks rows and never changes a column. So
        this method returns `Self`, and the result keeps the columns of the
        receiver.

        Args:
            *args: The positional arguments of `pl.LazyFrame.bottom_k`.
            **kwargs: The keyword arguments of `pl.LazyFrame.bottom_k`.

        Returns:
            The result of `pl.LazyFrame.bottom_k`.
        """
        return super().bottom_k(*args, **kwargs)  # type: ignore[return-value]

    def select_seq(self, *args: Any, **kwargs: Any) -> Self:
        """Select or compute columns as `select` does, and run the expressions in turn.

        The result has the same columns as with `select`, so the same hook of the
        checker computes them. If the hook cannot compute the columns, the result is
        a plain `pl.LazyFrame`. The hook reads the arguments by the parameter
        names of this signature, `args` and `kwargs`, and polars gives its parameters
        other names, so the override must stay. The declared return type `Self`
        applies only where the checker is not loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.select_seq`.
            **kwargs: The keyword arguments of `pl.LazyFrame.select_seq`.

        Returns:
            The result of `pl.LazyFrame.select_seq`.
        """
        return super().select_seq(*args, **kwargs)  # type: ignore[return-value]

    def with_columns_seq(self, *args: Any, **kwargs: Any) -> Self:
        """Add or replace columns as `with_columns` does, and run the expressions in turn.

        The result has the same columns as with `with_columns`, so the same hook of
        the checker computes them. If the hook cannot compute the columns, the
        result is a plain `pl.LazyFrame`. The hook reads the arguments by the
        parameter names of this signature, `args` and `kwargs`, and polars gives its
        parameters other names, so the override must stay. The declared return type
        `Self` applies only where the checker is not loaded.

        Args:
            *args: The positional arguments of `pl.LazyFrame.with_columns_seq`.
            **kwargs: The keyword arguments of `pl.LazyFrame.with_columns_seq`.

        Returns:
            The result of `pl.LazyFrame.with_columns_seq`.
        """
        return super().with_columns_seq(*args, **kwargs)  # type: ignore[return-value]

    def merge_sorted(
        self,
        other: pl.LazyFrame,
        key: str | Sequence[str],
        *,
        maintain_order: bool = False,
    ) -> Self:
        """Merge the rows of another sorted frame into this one, and keep the type of the frame.

        polars declares `pl.LazyFrame.merge_sorted` to return a plain `LazyFrame`,
        and that type holds no columns. polars raises an error if the other frame
        has other columns. So this method returns `Self`, and the result keeps the
        columns of the receiver. The checker's hook reports the other frame if its
        type proves that its columns differ. The parameters are those of polars, so
        the hook finds `other` whether the caller passes it by position or by name.

        Args:
            other: The other sorted frame.
            key: The column or columns that both frames are sorted by.
            maintain_order: Whether rows with equal keys keep their order.

        Returns:
            The result of `pl.LazyFrame.merge_sorted`.
        """
        return super().merge_sorted(other, key, maintain_order=maintain_order)  # type: ignore[return-value]

    def clear(self, *args: Any, **kwargs: Any) -> Self:
        """Give a frame with the same columns and no rows, or `n` rows of nulls.

        polars declares `pl.LazyFrame.clear` to return a plain `LazyFrame`, and that
        type holds no columns. The result has the same columns as the receiver, so
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.LazyFrame.clear`.
            **kwargs: The keyword arguments of `pl.LazyFrame.clear`.

        Returns:
            The result of `pl.LazyFrame.clear`.
        """
        return super().clear(*args, **kwargs)  # type: ignore[return-value]

    def pipe[**P, T](
        self,
        function: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        """Call a function with the frame as its first argument.

        polars types the first parameter of `function` as a plain `LazyFrame`. A
        parameter type of a `Callable` is contravariant, so mypy then rejects a
        function whose first parameter is a typed frame, for example
        `TypedLazyFrame[Trades]`. This signature takes the type of the receiver in
        its place.

        Args:
            function: The function to call. Its first parameter accepts the
                frame.
            *args: More positional arguments for `function`, after the frame.
            **kwargs: The keyword arguments for `function`.

        Returns:
            What `function` returns.
        """
        return function(self, *args, **kwargs)


class TypedSeries[Name, DType: pl.DataType](pl.Series):
    """A `pl.Series` whose type states its name and its data type.

    `df["price"]` and `df.get_column("price")` on a typed frame give
    `TypedSeries[Literal["price"], Float64]`. The checker also types
    `pl.Series(...)` with a name and a `dtype=`, and the methods `.cast()` and
    `.round()`. The type arguments are in the same order as in `Column`: the
    name, then the data type. At run time the class is a `pl.Series`.

    polars declares most methods of `pl.Series` to return a plain `Series`.
    So this class declares each method that keeps the name and the data type
    to return `Self`, and the result keeps the type of the receiver.
    """

    def drop_nulls(self, *args: Any, **kwargs: Any) -> Self:
        """Delete the null values, and keep the type of the series.

        polars declares `pl.Series.drop_nulls` to return a plain `Series`, which has
        no name or data type in its type. The name and the data type stay the same.
        So this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.drop_nulls`.
            **kwargs: The keyword arguments of `pl.Series.drop_nulls`.

        Returns:
            The result of `pl.Series.drop_nulls`.
        """
        return super().drop_nulls(*args, **kwargs)  # type: ignore[return-value]

    def alias(self, name: str) -> Self:
        """Give the series a new name. The checker gives the result its type.

        The checker's hook reads the new name and gives the result the type
        `TypedSeries[name, dtype]`, with the data type of the receiver. The
        declared return type `Self` applies only where the checker is not
        loaded.

        Args:
            name: The new name.

        Returns:
            The result of `pl.Series.alias`.
        """
        return super().alias(name)  # type: ignore[return-value]

    def rename(self, name: str) -> Self:
        """Give the series a new name, as `alias` does. The checker gives the result its type.

        Args:
            name: The new name.

        Returns:
            The result of `pl.Series.rename`.
        """
        return super().rename(name)  # type: ignore[return-value]

    def slice(self, *args: Any, **kwargs: Any) -> Self:
        """Keep a range of values, and keep the type of the series.

        polars declares `pl.Series.slice` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.slice`.
            **kwargs: The keyword arguments of `pl.Series.slice`.

        Returns:
            The result of `pl.Series.slice`.
        """
        return super().slice(*args, **kwargs)  # type: ignore[return-value]

    def head(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` values, and keep the type of the series.

        polars declares `pl.Series.head` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.head`.
            **kwargs: The keyword arguments of `pl.Series.head`.

        Returns:
            The result of `pl.Series.head`.
        """
        return super().head(*args, **kwargs)  # type: ignore[return-value]

    def tail(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the last `n` values, and keep the type of the series.

        polars declares `pl.Series.tail` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.tail`.
            **kwargs: The keyword arguments of `pl.Series.tail`.

        Returns:
            The result of `pl.Series.tail`.
        """
        return super().tail(*args, **kwargs)  # type: ignore[return-value]

    def limit(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the first `n` values, as `head` does, and keep the type of the series.

        polars declares `pl.Series.limit` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.limit`.
            **kwargs: The keyword arguments of `pl.Series.limit`.

        Returns:
            The result of `pl.Series.limit`.
        """
        return super().limit(*args, **kwargs)  # type: ignore[return-value]

    def gather_every(self, *args: Any, **kwargs: Any) -> Self:
        """Keep every `n`-th value, and keep the type of the series.

        polars declares `pl.Series.gather_every` to return a plain `Series`, which
        has no name or data type in its type. The name and the data type stay the
        same. So this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.gather_every`.
            **kwargs: The keyword arguments of `pl.Series.gather_every`.

        Returns:
            The result of `pl.Series.gather_every`.
        """
        return super().gather_every(*args, **kwargs)  # type: ignore[return-value]

    def top_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` largest values, and keep the type of the series.

        polars declares `pl.Series.top_k` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.top_k`.
            **kwargs: The keyword arguments of `pl.Series.top_k`.

        Returns:
            The result of `pl.Series.top_k`.
        """
        return super().top_k(*args, **kwargs)  # type: ignore[return-value]

    def bottom_k(self, *args: Any, **kwargs: Any) -> Self:
        """Keep the `k` smallest values, and keep the type of the series.

        polars declares `pl.Series.bottom_k` to return a plain `Series`, which has
        no name or data type in its type. The name and the data type stay the same.
        So this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.bottom_k`.
            **kwargs: The keyword arguments of `pl.Series.bottom_k`.

        Returns:
            The result of `pl.Series.bottom_k`.
        """
        return super().bottom_k(*args, **kwargs)  # type: ignore[return-value]

    def unique(self, *args: Any, **kwargs: Any) -> Self:
        """Keep one of each set of equal values, and keep the type of the series.

        polars declares `pl.Series.unique` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.unique`.
            **kwargs: The keyword arguments of `pl.Series.unique`.

        Returns:
            The result of `pl.Series.unique`.
        """
        return super().unique(*args, **kwargs)  # type: ignore[return-value]

    def reverse(self, *args: Any, **kwargs: Any) -> Self:
        """Reverse the order of the values, and keep the type of the series.

        polars declares `pl.Series.reverse` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.reverse`.
            **kwargs: The keyword arguments of `pl.Series.reverse`.

        Returns:
            The result of `pl.Series.reverse`.
        """
        return super().reverse(*args, **kwargs)  # type: ignore[return-value]

    def clear(self, *args: Any, **kwargs: Any) -> Self:
        """Give a series with the same name and data type and no values, or `n` nulls.

        polars declares `pl.Series.clear` to return a plain `Series`, which has no
        name or data type in its type. The result keeps the type of the series. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.clear`.
            **kwargs: The keyword arguments of `pl.Series.clear`.

        Returns:
            The result of `pl.Series.clear`.
        """
        return super().clear(*args, **kwargs)  # type: ignore[return-value]

    def round(self, *args: Any, **kwargs: Any) -> Self:
        """Round each value, and keep the type of the series.

        polars declares `pl.Series.round` to return a plain `Series`, which has no
        name or data type in its type. polars keeps the data type of a numeric or
        `Decimal` series. On any other series, polars raises
        `InvalidOperationError`. The checker's hook reports that case, and then
        gives the data type `Unchecked`. So this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.round`.
            **kwargs: The keyword arguments of `pl.Series.round`.

        Returns:
            The result of `pl.Series.round`.
        """
        return super().round(*args, **kwargs)  # type: ignore[return-value]

    def shift(self, *args: Any, **kwargs: Any) -> Self:
        """Move the values down or up, and keep the type of the series.

        polars declares `pl.Series.shift` to return a plain `Series`, which has no
        name or data type in its type. Nulls fill the positions that the move leaves
        empty. The name and the data type stay the same. So this method returns
        `Self`.

        Args:
            *args: The positional arguments of `pl.Series.shift`.
            **kwargs: The keyword arguments of `pl.Series.shift`.

        Returns:
            The result of `pl.Series.shift`.
        """
        return super().shift(*args, **kwargs)  # type: ignore[return-value]

    def shuffle(self, *args: Any, **kwargs: Any) -> Self:
        """Put the values in a random order, and keep the type of the series.

        polars declares `pl.Series.shuffle` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.shuffle`.
            **kwargs: The keyword arguments of `pl.Series.shuffle`.

        Returns:
            The result of `pl.Series.shuffle`.
        """
        return super().shuffle(*args, **kwargs)  # type: ignore[return-value]

    def sample(self, *args: Any, **kwargs: Any) -> Self:
        """Keep a random sample of the values, and keep the type of the series.

        polars declares `pl.Series.sample` to return a plain `Series`, which has no
        name or data type in its type. The name and the data type stay the same. So
        this method returns `Self`.

        Args:
            *args: The positional arguments of `pl.Series.sample`.
            **kwargs: The keyword arguments of `pl.Series.sample`.

        Returns:
            The result of `pl.Series.sample`.
        """
        return super().sample(*args, **kwargs)  # type: ignore[return-value]


class TypedGroupBy[Keys, OriginalColumns, Rest](pl.dataframe.group_by.GroupBy):
    """The type of `group_by(...)` on a `TypedDataFrame`, before its `.agg(...)`.

    The checker types `.agg(...)` from three facts that it keeps here: `Keys`
    is the columns of the groups, in order, `OriginalColumns` is the declared
    columns of the frame, and `Rest` is its remainder, that is, its
    undeclared columns. A user does not write this type, and no code creates
    an instance of it.
    """


class TypedLazyGroupBy[Keys, OriginalColumns, Rest](pl.lazyframe.group_by.LazyGroupBy):
    """The type of `group_by(...)` on a `TypedLazyFrame`, before its `.agg(...)`.

    It holds the same three facts as `TypedGroupBy`, over the `LazyGroupBy`
    of polars.
    """


class TypedRollingGroupBy[Keys, OriginalColumns, Rest, Index, Windows](
    pl.dataframe.group_by.RollingGroupBy,
):
    """The type of `rolling(...)` on a `TypedDataFrame`, before its `.agg(...)`.

    It holds the facts of `TypedGroupBy`, and two more. `Index` is the index
    column, as a tuple of one column, and `Windows` names the windows of the
    rows, as a `Literal` text that the checker reads. A user does not write
    this type, and no code creates an instance of it.
    """


class TypedDynamicGroupBy[Keys, OriginalColumns, Rest, Index, Windows](
    pl.dataframe.group_by.DynamicGroupBy,
):
    """The type of `group_by_dynamic(...)` on a `TypedDataFrame`, before its `.agg(...)`.

    It holds the same five facts as `TypedRollingGroupBy`, over the
    `DynamicGroupBy` of polars.
    """


class TypedLazyRollingGroupBy[Keys, OriginalColumns, Rest, Index, Windows](
    pl.lazyframe.group_by.LazyGroupBy,
):
    """The type of `rolling(...)` on a `TypedLazyFrame`, before its `.agg(...)`.

    It holds the same five facts as `TypedRollingGroupBy`, over the
    `LazyGroupBy` of polars.
    """


class TypedLazyDynamicGroupBy[Keys, OriginalColumns, Rest, Index, Windows](
    pl.lazyframe.group_by.LazyGroupBy,
):
    """The type of `group_by_dynamic(...)` on a `TypedLazyFrame`, before its `.agg(...)`.

    It holds the same five facts as `TypedRollingGroupBy`, over the
    `LazyGroupBy` of polars.
    """


def assume_schema[F: _TypedFrameBase](df: pl.DataFrame | pl.LazyFrame, frame_type: type[F]) -> F:
    """Give a frame the type of a typed frame, with no check at all.

    mypy treats the call as `typing.cast`: the result has the type
    `frame_type`, and nothing proves it. At run time, the function does not
    read the schema of `frame_type`, and it does not check or reorder the
    columns. It makes a cheap copy of `df`, which shares the data, and gives
    the copy the class `frame_type`. `df` keeps its own class. Use
    `check_schema` to check the columns at run time, or `force_schema` to
    cast them.

    Args:
        df: The frame.
        frame_type: The typed frame type, for example
            `TypedDataFrame[Trades]`.

    Returns:
        A copy of `df` of the class `frame_type`, with its columns in the
        order that `df` has.

    Example:
        >>> class Foo(Columns):
        ...     foo: pl.Int64
        >>> Frame = TypedDataFrame[Foo]
        >>> df = pl.DataFrame({"foo": [1, 2, 3]})
        >>> result = assume_schema(df, Frame)
        >>> isinstance(result, Frame)
        True
        >>> isinstance(df, Frame)
        False

    Example:
        >>> class Missing(Columns):
        ...     missing: pl.Int64
        >>> WrongFrame = TypedDataFrame[Missing]
        >>> assume_schema(df, WrongFrame)  # no check: never raises
        shape: (3, 1)
        ┌─────┐
        │ foo │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 1   │
        │ 2   │
        │ 3   │
        └─────┘
    """
    return SchemaCast(df, frame_type).assume()


def check_schema[F: _TypedFrameBase](df: pl.DataFrame | pl.LazyFrame, frame_type: type[F]) -> F:
    """Check at run time that a frame has exactly the declared columns, and type it.

    mypy treats the call as `typing.cast`. At run time, each declared column
    must be present with its declared data type, and the frame must have no
    other column. The function casts nothing: a column of another data type
    is an error. A column declared `Unchecked` must be present, and its data
    type is not checked. The order of the columns does not matter, and the
    function keeps it.

    Args:
        df: The frame.
        frame_type: The typed frame type, for example
            `TypedDataFrame[Trades]`.

    Returns:
        A copy of `df` of the class `frame_type`, with its columns in the
        order that `df` has. The function raises `_SchemaNotConcreteError` if
        `frame_type` has no full schema, `_MissingColumnError` for a missing
        column, `_DtypeMismatchError` for a column of another data type, and
        `_ExtraColumnError` for a column that `frame_type` does not declare.

    Example:
        >>> class Foo(Columns):
        ...     foo: pl.Int64
        >>> Frame = TypedDataFrame[Foo]
        >>> df = pl.DataFrame({"foo": [1, 2, 3]})
        >>> result = check_schema(df, Frame)
        >>> isinstance(result, Frame)
        True
    """
    return SchemaCast(df, frame_type).check()


def force_schema[F: _TypedFrameBase](
    df: pl.DataFrame | pl.LazyFrame,
    frame_type: type[F],
    *,
    strict: bool = True,
) -> F:
    """Change a frame to the declared columns, and give it the typed frame type.

    mypy treats the call as `typing.cast`. At run time, the function keeps
    only the declared columns, in their declared order, and casts each one
    to its declared data type. It does not cast a column declared
    `Unchecked`, because that column has no data type to cast to.

    Args:
        df: The frame.
        frame_type: The typed frame type, for example
            `TypedDataFrame[Trades]`.
        strict: The `strict` argument of `pl.DataFrame.cast`. If True, a value
            that polars cannot cast raises
            `pl.exceptions.InvalidOperationError`. If False, that value
            becomes null.

    Returns:
        The frame with the declared columns, in the declared order and with
        the declared data types, of the class `frame_type`. The function
        raises `_SchemaNotConcreteError` if `frame_type` has no full schema,
        and `_MissingColumnError` for a missing column. It never raises
        `_DtypeMismatchError`, because it casts in place of a check.

    Example:
        >>> class Foo(Columns):
        ...     foo: pl.Int64
        >>> Frame = TypedDataFrame[Foo]
        >>> df = pl.DataFrame({"foo": [1.0, 2.0, 3.0]})
        >>> result = force_schema(df, Frame)
        >>> isinstance(result, Frame)
        True
        >>> result.schema
        Schema({'foo': Int64})
    """
    return SchemaCast(df, frame_type).force(strict=strict)


class _SchemaNotConcreteError(TypeError):
    """The frame type has no full schema, so there is nothing to compare the frame with.

    `check_schema` and `force_schema` raise it, for example for an open frame
    type.
    """

    def __init__(self, frame_type: type[_TypedFrameBase]) -> None:
        """Build the message that names the frame type.

        Args:
            frame_type: The frame type, for example an open frame type.
        """
        super().__init__(f"{frame_type!r} has no concrete schema to assume")


class _MissingColumnError(pl.exceptions.SchemaError):
    """A declared column is missing from the frame, at `check_schema` or `force_schema`."""

    def __init__(self, name: str) -> None:
        """Build the message that names the column.

        Args:
            name: The name of the missing column.
        """
        super().__init__(f"missing column {name!r}")


class _DtypeMismatchError(pl.exceptions.SchemaError):
    """A declared column has another data type in the frame, at `check_schema`."""

    def __init__(self, name: str, actual_dtype: Any, expected_dtype: DTypeLike) -> None:
        """Build the message that names the column and the two data types.

        Args:
            name: The name of the column.
            actual_dtype: The data type of the column in the frame.
            expected_dtype: The declared data type.
        """
        super().__init__(f"column {name!r} has dtype {actual_dtype!r}, expected {expected_dtype!r}")


class _ExtraColumnError(pl.exceptions.SchemaError):
    """The frame has a column that the frame type does not declare, at `check_schema`."""

    def __init__(self, name: str) -> None:
        """Build the message that names the column.

        Args:
            name: The name of the column that is not declared.
        """
        super().__init__(f"unexpected column {name!r}")


class SchemaCast[F: _TypedFrameBase]:
    """Give a frame a typed frame type at run time: with no check, with a check, or with a cast.

    `assume_schema`, `check_schema` and `force_schema` share the work here.
    The class reads the declared schema and the real schema of the frame only
    when a step needs them, and only once. So `assume` never reads the
    declared schema, and costs no more than a copy of the frame.

    Example:
        >>> class Foo(Columns):
        ...     foo: pl.Int64
        >>> Frame = TypedDataFrame[Foo]
        >>> df = pl.DataFrame({"foo": [1, 2, 3]})
        >>> isinstance(SchemaCast(df, Frame).assume(), Frame)
        True
    """

    def __init__(self, df: pl.DataFrame | pl.LazyFrame, frame_type: type[F]) -> None:
        """Keep the frame and the typed frame type.

        Args:
            df: The frame.
            frame_type: The typed frame type.
        """
        self._df = df
        self._frame_type = frame_type

    @cached_property
    def _target_schema(self) -> dict[str, DTypeLike]:
        """The declared schema of the typed frame type.

        Raises:
            _SchemaNotConcreteError: If the frame type has no full schema.
        """
        target_schema = self._frame_type.schema
        if not isinstance(target_schema, dict):
            raise _SchemaNotConcreteError(self._frame_type)
        return target_schema

    @cached_property
    def _actual_schema(self) -> Any:
        """The real schema of the frame.

        For a lazy frame, polars computes it from the query plan.
        """
        return self._df.collect_schema() if isinstance(self._df, pl.LazyFrame) else self._df.schema

    def assume(self) -> F:
        """Give the frame the typed frame type, with no check at all.

        See `assume_schema`: the function calls this method.

        Returns:
            A copy of the frame of the class `frame_type`, with its columns in
            their own order.
        """
        return self._relabeled(self._df.clone())

    def check(self) -> F:
        """Check that the frame has exactly the declared columns, then give it the typed frame type.

        See `check_schema`: the function calls this method. Each declared
        column must be present, with its declared data type. A column declared
        `Unchecked` passes with any data type: only its presence counts. The
        frame must have no column outside the declared ones.

        Returns:
            A copy of the frame of the class `frame_type`, with its columns in
            their own order. The method raises `_SchemaNotConcreteError`,
            `_MissingColumnError`, `_DtypeMismatchError` or `_ExtraColumnError`,
            as `check_schema` says.

        Raises:
            _DtypeMismatchError: If a column has another data type than the
                declared one, and the declared one is not `Unchecked`.
            _ExtraColumnError: If the frame has a column that is not declared.

        Example:
            >>> X = columns("X", {"x": pl.Int64})
            >>> SchemaCast(pl.DataFrame({"x": [1]}), TypedDataFrame[X]).check().columns
            ['x']
            >>> Loose = columns("Loose", {"x": Unchecked})
            >>> SchemaCast(pl.DataFrame({"x": ["a"]}), TypedDataFrame[Loose]).check().columns
            ['x']
            >>> SchemaCast(pl.DataFrame({"x": ["a"]}), TypedDataFrame[X]).check()
            Traceback (most recent call last):
                ...
            astrolabe.frame._DtypeMismatchError: ...
            >>> extra = SchemaCast(pl.DataFrame({"x": [1]}), TypedDataFrame[()])
            >>> extra.check()
            Traceback (most recent call last):
                ...
            astrolabe.frame._ExtraColumnError: unexpected column 'x'
        """
        actual = self._actual_schema
        for name, dtype in self._target_schema.items():
            self._require_present(name)
            if dtype != Unchecked and actual[name] != dtype:
                raise _DtypeMismatchError(name, actual[name], dtype)
        for name in actual:
            if name not in self._target_schema:
                raise _ExtraColumnError(name)
        return self._relabeled(self._df.clone())

    def force(self, *, strict: bool = True) -> F:
        """Keep and cast the declared columns, then give the frame the typed frame type.

        See `force_schema`: the function calls this method. It checks only that
        each declared column is present, and not its data type, because it
        casts each column to its declared data type.

        Args:
            strict: The `strict` argument of `pl.DataFrame.cast`. If True, a
                value that polars cannot cast raises
                `pl.exceptions.InvalidOperationError`. If False, that value
                becomes null.

        Returns:
            The frame with the declared columns, in the declared order and with
            the declared data types, of the class `frame_type`. The method raises
            `_SchemaNotConcreteError` or `_MissingColumnError`, as
            `force_schema` says.
        """
        for name in self._target_schema:
            self._require_present(name)
        entries = self._target_schema.items()
        checked = {name: dtype for name, dtype in entries if dtype != Unchecked}
        cast_map = pl.Schema(checked)
        result = self._df.select(list(self._target_schema)).cast(cast_map, strict=strict)
        return self._relabeled(result)

    def _relabeled(self, result: pl.DataFrame | pl.LazyFrame) -> F:
        """Change the class of a frame to the typed frame type.

        The frame keeps its data. Only `__class__` changes, so the step costs
        nothing.

        Args:
            result: The frame: a copy for `assume` and `check`, or the cast
                result for `force`.

        Returns:
            `result`, now of the class `frame_type`.
        """
        result.__class__ = self._frame_type  # type: ignore[assignment]
        return result  # type: ignore[return-value]

    def _require_present(self, name: str) -> None:
        """Check that one declared column is present.

        Args:
            name: The name of the column.

        Raises:
            _MissingColumnError: If the frame has no column with this name.
        """
        if name not in self._actual_schema:
            raise _MissingColumnError(name)
