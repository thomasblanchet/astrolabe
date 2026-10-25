"""The data types of polars as plain values, for rules that compute types without data.

A rule of polars reads data types and gives data types. Some of those types
exist only in the planner of polars: a literal such as `pl.lit(300)` has the
type `Unknown(Int(300))` until polars joins it with a column. This module holds
one value for every such type, and the tests on types that polars uses. It also
holds the facts of the parameters that many rules read: the range of each
integer type, the fit of a number in a `Decimal`, and the time units.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars.datatypes import DataTypeClass


class NoPartError(TypeError):
    """Report a request for a part that a data type does not have.

    A bug in a rule causes it: for example, a rule that asks a `String`
    for its inner type.

    Example:
        >>> str(NoPartError("String", "inner type"))
        'String has no inner type'
    """

    def __init__(self, dtype: str, part: str) -> None:  # pragma: no cover -- a guard raises it
        """Build the message.

        Args:
            dtype: The rendering of the data type.
            part: The part that was asked for.

        Example:
            >>> NoPartError("Int8", "fields").args
            ('Int8 has no fields',)
        """
        super().__init__(f"{dtype} has no {part}")


@dataclass(frozen=True)
class DType:
    """Hold one data type of polars, or a type that only its planner knows.

    `name` is the name of the polars class, for example `Int64`, `Datetime` or
    `List`. `args` holds the parameters, in the order of polars: the precision
    and the scale of a `Decimal`, the unit and the zone of a `Datetime`, the
    unit of a `Duration`, the categories of an `Enum`, the inner type of a
    `List`, the inner type and the width of an `Array`, and the fields of a
    `Struct` as pairs of a name and a type. The planner type `Unknown` holds
    what it is, `int`, `float`, `str` or `any`, and the value of an int.

    Example:
        >>> DType.of(pl.List(pl.Int8()))
        List(Int8)
        >>> Is.integer(DType.dynamic_int(300))
        True
    """

    name: str
    args: tuple[object, ...] = ()

    PHYSICAL: ClassVar[dict[str, str]] = {
        "Date": "Int32",
        "Datetime": "Int64",
        "Duration": "Int64",
        "Time": "Int64",
        "Decimal": "Int128",
        "Categorical": "UInt32",
    }

    def __repr__(self) -> str:
        """Give the rendering, as `str()` of the polars data type gives it.

        Returns:
            The rendering.

        Example:
            >>> DType("Decimal", (10, 2))
            Decimal(precision=10, scale=2)
        """
        return self.render()

    @classmethod
    def of(cls, dtype: pl.DataType | DataTypeClass) -> DType:
        """Read a polars data type, as an instance or as a class.

        Polars gives a data type with no parameter as a class in some places,
        for example `pl.Int8` as the inner type of a list, so this reads both.
        It shows the planner type `Unknown` with no detail of what it is, so
        this reads it as `Unknown` of any kind.

        Args:
            dtype: The polars data type.

        Returns:
            The value.

        Example:
            >>> DType.of(pl.Struct({"a": pl.Datetime("ms")}))
            Struct({'a': Datetime(time_unit='ms', time_zone=None)})
            >>> DType.of(pl.List(pl.Unknown())) == DType("List", (DType.unknown("any"),))
            True
            >>> DType.of(pl.Int8)
            Int8
        """
        if not isinstance(dtype, pl.DataType):
            return cls.of(dtype())
        readers: dict[type, Callable[[pl.DataType], tuple[object, ...]]] = {
            pl.Decimal: lambda d: (d.precision, d.scale),  # type: ignore[attr-defined]
            pl.Datetime: lambda d: (d.time_unit, d.time_zone),  # type: ignore[attr-defined]
            pl.Duration: lambda d: (d.time_unit,),  # type: ignore[attr-defined]
            pl.Enum: lambda d: (tuple(d.categories.to_list()),),  # type: ignore[attr-defined]
            pl.List: lambda d: (cls._inner_of(d),),
            pl.Array: lambda d: (cls._inner_of(d), d.size),  # type: ignore[attr-defined]
            pl.Struct: lambda d: (cls._fields_of(d),),
            pl.Unknown: lambda _: ("any", None),
        }
        reader = readers.get(type(dtype))
        return cls(type(dtype).__name__, reader(dtype) if reader is not None else ())

    @classmethod
    def _inner_of(cls, dtype: pl.DataType) -> DType:
        """Read the inner type of a polars `List` or `Array`.

        Args:
            dtype: The polars data type.

        Returns:
            The inner type.

        Example:
            >>> DType._inner_of(pl.Array(pl.Int8(), 2))
            Int8
        """
        return cls.of(dtype.inner)  # type: ignore[attr-defined]

    @classmethod
    def _fields_of(cls, dtype: pl.DataType) -> tuple[tuple[str, DType], ...]:
        """Read the fields of a polars `Struct`.

        Args:
            dtype: The polars data type.

        Returns:
            The fields, each a pair of a name and a type.

        Example:
            >>> DType._fields_of(pl.Struct({"a": pl.Int8()}))
            (('a', Int8),)
        """
        fields = dtype.fields  # type: ignore[attr-defined]
        return tuple((field.name, cls.of(field.dtype)) for field in fields)

    @classmethod
    def dynamic_int(cls, value: int) -> DType:
        """Give the planner type of an integer literal with no data type, such as `pl.lit(300)`.

        Args:
            value: The value of the literal.

        Returns:
            The type `Unknown(Int(value))`.

        Example:
            >>> DType.dynamic_int(-1)
            Unknown(int, -1)
        """
        return cls("Unknown", ("int", value))

    @classmethod
    def unknown(cls, what: str) -> DType:
        """Give a planner type `Unknown` that holds no value.

        Args:
            what: `float` or `str` for a literal of that Python type, or `any`.

        Returns:
            The type.

        Example:
            >>> Is.float(DType.unknown("float"))
            True
        """
        return cls("Unknown", (what, None))

    @classmethod
    def idx(cls) -> DType:
        """Give the type of the row indices and counts of polars.

        The standard build of polars has 32-bit indices, so the type is
        `UInt32`.

        Returns:
            The type.

        Example:
            >>> DType.idx()
            UInt32
        """
        return cls("UInt32")

    def polars(self) -> pl.DataType:
        """Give the polars data type, the reverse of `of`.

        Code that asks polars a question about a schema, such as which
        columns a selector picks, needs the data types of polars.

        Returns:
            The polars data type.

        Raises:
            NoPartError: The type is a literal type with no polars data type,
                such as an integer literal.

        Example:
            >>> DType.of(pl.Struct({"a": pl.List(pl.Datetime("ms"))})).polars()
            Struct({'a': List(Datetime(time_unit='ms', time_zone=None))})
            >>> DType("Enum", (("a", "b"),)).polars()
            Enum(categories=['a', 'b'])
            >>> DType("Literal").polars()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        builders: dict[str, Callable[[], pl.DataType]] = {
            "Decimal": lambda: pl.Decimal(*self.args),  # type: ignore[arg-type]
            "Datetime": lambda: pl.Datetime(*self.args),  # type: ignore[arg-type]
            "Duration": lambda: pl.Duration(*self.args),  # type: ignore[arg-type]
            "Enum": lambda: pl.Enum(list(Parameter().categories(self))),
            "List": lambda: pl.List(self.inner().polars()),
            "Array": lambda: pl.Array(self.inner().polars(), self.args[1]),  # type: ignore[arg-type]
            "Struct": lambda: pl.Struct({name: dtype.polars() for name, dtype in self.fields()}),
            "Unknown": pl.Unknown,
        }
        if self.name in builders:
            return builders[self.name]()
        kind = getattr(pl, self.name, None)
        if (  # pragma: no cover -- each caller names a type of polars
            not isinstance(kind, type) or not issubclass(kind, pl.DataType)
        ):
            raise NoPartError(self.render(), "polars data type")
        return kind()

    def _ids(self) -> DType:
        """Give the type of the ids of the categories of an `Enum`.

        Returns:
            The narrowest unsigned type whose largest value is above the
            count of the categories.

        Example:
            >>> DType("Enum", (("a", "b"),))._ids()
            UInt8
            >>> DType("Enum", (tuple(str(n) for n in range(2**16)),))._ids()
            UInt32
        """
        count = len(Parameter().categories(self))
        for name, bits in (("UInt8", 8), ("UInt16", 16)):
            if count < 2**bits:
                return DType(name)
        return DType("UInt32")

    def render(self) -> str:
        """Give the rendering, as `str()` of the polars data type gives it.

        `Unknown` renders with what it is, because polars has no rendering for it.
        A kind with parameters and no argument renders as its name, as the
        class of polars does: `Datetime` with its parameters open.

        Returns:
            The rendering.

        Example:
            >>> DType("Array", (DType("Int64"), 2)).render()
            'Array(Int64, shape=(2,))'
            >>> DType("Array", (DType("Array", (DType("Int64"), 3)), 2)).render()
            'Array(Int64, shape=(2, 3))'
            >>> DType("Datetime").render()
            'Datetime'
        """
        renders: dict[str, Callable[[tuple[object, ...]], str]] = {
            "Decimal": lambda a: f"Decimal(precision={a[0]}, scale={a[1]})",
            "Datetime": lambda a: f"Datetime(time_unit='{a[0]}', time_zone={a[1]!r})",
            "Duration": lambda a: f"Duration(time_unit='{a[0]}')",
            "Enum": lambda a: f"Enum(categories={list(a[0])!r})",  # type: ignore[call-overload]
            "List": lambda a: f"List({a[0]!r})",
            "Array": lambda _: self._array_text(),
            "Struct": lambda _: "Struct({" + self._fields_text() + "})",
            "Unknown": lambda a: "Unknown" if a[1] is None else f"Unknown({a[0]}, {a[1]})",
        }
        render = renders.get(self.name)
        return render(self.args) if render is not None and self.args else self.name

    def _array_text(self) -> str:
        """Give the rendering of an array, with one shape for the arrays nested in it.

        Returns:
            The rendering.

        Example:
            >>> DType("Array", (DType("Int8"), 2))._array_text()
            'Array(Int8, shape=(2,))'
        """
        widths: list[str] = []
        inner = self
        while inner.name == "Array" and inner.args:
            widths.append(str(inner.args[1]))
            inner = inner.inner()
        shape = ", ".join(widths) + ("," if len(widths) == 1 else "")
        return f"Array({inner!r}, shape=({shape}))"

    def _fields_text(self) -> str:
        """Give the fields of a `Struct` as polars renders them.

        Returns:
            The fields, separated by commas.

        Example:
            >>> DType("Struct", ((("a", DType("Int8")),),))._fields_text()
            "'a': Int8"
        """
        return ", ".join(f"'{name}': {dtype!r}" for name, dtype in self.fields())

    def inner(self) -> DType:
        """Give the inner type of a `List` or an `Array`.

        Returns:
            The inner type.

        Raises:
            NoPartError: The type is not a `List` or an `Array`.

        Example:
            >>> DType.of(pl.List(pl.String())).inner()
            String
            >>> DType("Int8").inner()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        inner = self.args[0] if self.args else None
        if not isinstance(inner, DType):  # pragma: no cover -- each caller checks the kind
            raise NoPartError(self.render(), "inner type")
        return inner

    def fields(self) -> tuple[tuple[str, DType], ...]:
        """Give the fields of a `Struct`, each a pair of a name and a type.

        Returns:
            The fields.

        Raises:
            NoPartError: The type is not a `Struct`.

        Example:
            >>> DType.of(pl.Struct({"a": pl.Int8()})).fields()
            (('a', Int8),)
            >>> DType("Int8").fields()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        fields = self.args[0] if self.args else None
        if (  # pragma: no cover -- each caller checks the kind
            self.name != "Struct" or not isinstance(fields, tuple)
        ):
            raise NoPartError(self.render(), "fields")
        return fields

    def physical(self) -> DType:
        """Give the physical type that holds the data.

        A `Categorical` holds 32-bit ids. An `Enum` holds ids of the narrowest
        unsigned type that numbers its categories: 8 bits up to 255
        categories, 16 bits up to 65,535, and 32 bits above.

        Returns:
            The physical type.

        Example:
            >>> DType("Date").physical(), DType.of(pl.List(pl.Time())).physical()
            (Int32, List(Int64))
            >>> DType("Enum", (tuple(str(n) for n in range(256)),)).physical()
            UInt16
        """
        if self.name == "Enum":
            return self._ids()
        if self.name in self.PHYSICAL:
            return DType(self.PHYSICAL[self.name])
        return self._mapped(lambda dtype: dtype.physical())

    def leaf(self) -> DType:
        """Give the type under every `List` and `Array`.

        Returns:
            The leaf type.

        Example:
            >>> DType.of(pl.List(pl.Array(pl.Int8(), 2))).leaf()
            Int8
        """
        if self.name in Is.LISTS:
            return self.inner().leaf()
        return self

    def with_leaf(self, leaf: DType) -> DType:
        """Give the type with its leaf replaced.

        Args:
            leaf: The new leaf.

        Returns:
            The type.

        Example:
            >>> DType.of(pl.List(pl.Int8())).with_leaf(DType("Float64"))
            List(Float64)
        """
        if self.name not in Is.LISTS:
            return leaf
        return DType(self.name, (self.inner().with_leaf(leaf), *self.args[1:]))

    def same(self, other: DType) -> bool:
        """Say whether polars takes two types as equal.

        Two integer literals are equal whatever their values. So the planner
        does not join `pl.lit(1)` and `pl.lit(2**40)`, and keeps the first.

        Args:
            other: The other type.

        Returns:
            Whether they are equal.

        Example:
            >>> DType.dynamic_int(1).same(DType.dynamic_int(2**40))
            True
        """
        if Is.dynamic_int(self) and Is.dynamic_int(other):
            return True
        if self.name != other.name:
            return False
        if self.name in Is.LISTS:
            return self.inner().same(other.inner()) and self.args[1:] == other.args[1:]
        if self.name == "Struct":
            mine, theirs = self.fields(), other.fields()
            pairs = zip(mine, theirs, strict=False)
            matched = all(a[0] == b[0] and a[1].same(b[1]) for a, b in pairs)
            return len(mine) == len(theirs) and matched
        return self.args == other.args

    def materialized(self) -> DType:
        """Give the type that a literal takes alone, beside no column.

        Returns:
            The type: `Int32`, `Int64`, `UInt64` or `Int128` for an integer
            by its value, `Float64` for a float and `String` for a string.

        Raises:
            NoPartError: The type is an integer literal that holds no value.

        Example:
            >>> DType.dynamic_int(2**40).materialized()
            Int64
            >>> DType("Unknown", ("int", "a")).materialized()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        if Is.dynamic_int(self):
            value = self.args[1]
            if not isinstance(value, int):  # pragma: no cover -- a dynamic integer holds its value
                raise NoPartError(self.render(), "integer value")
            return DynamicInt.standalone(value)
        fixed = {"float": DType("Float64"), "str": DType("String")}
        what = self.args[0] if Is.unknown(self) else None
        if isinstance(what, str) and what in fixed:
            return fixed[what]
        return self._mapped(lambda dtype: dtype.materialized())

    def _mapped(self, change: Callable[[DType], DType]) -> DType:
        """Give a nested type with a change applied to each inner type, or the type itself.

        Args:
            change: The change of one inner type.

        Returns:
            The type.

        Example:
            >>> DType.of(pl.Array(pl.Date(), 2))._mapped(lambda d: DType("Int32"))
            Array(Int32, shape=(2,))
        """
        if self.name in Is.LISTS:
            return DType(self.name, (change(self.inner()), *self.args[1:]))
        if self.name == "Struct":
            return DType("Struct", (tuple((n, change(d)) for n, d in self.fields()),))
        return self


class Is:
    """Give the tests that polars puts on a data type.

    Each test reads one type. The tests of a number include the planner
    types of literals, as polars does: an integer literal with no data type
    is an integer, and a float literal is a float. The sets name the kinds
    that the arms of polars match together: `LISTS` holds `List` and `Array`,
    `MOMENTS` the temporal kinds that are not a `Duration`, and
    `SMALL_INTEGERS` the integers of 8 and 16 bits, which a sum widens to
    `Int64`.

    Example:
        >>> Is.float(DType("Float16")), Is.numeric(DType("Decimal", (10, 2)))
        (True, True)
    """

    SIGNED: ClassVar[frozenset[str]] = frozenset({"Int8", "Int16", "Int32", "Int64", "Int128"})
    UNSIGNED: ClassVar[frozenset[str]] = frozenset({
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt128",
    })
    FLOATS: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32", "Float64"})
    TEMPORAL: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime", "Duration", "Time"})
    NESTED: ClassVar[frozenset[str]] = frozenset({"List", "Array", "Struct"})
    LISTS: ClassVar[frozenset[str]] = frozenset({"List", "Array"})
    CATEGORIES: ClassVar[frozenset[str]] = frozenset({"Categorical", "Enum"})
    MOMENTS: ClassVar[frozenset[str]] = frozenset({"Date", "Datetime", "Time"})
    SMALL_INTEGERS: ClassVar[frozenset[str]] = frozenset({"Int8", "UInt8", "Int16", "UInt16"})

    @classmethod
    def float(cls, dtype: DType) -> bool:
        """Say whether the type is a float, a float literal included.

        Args:
            dtype: The type.

        Returns:
            Whether it is a float.

        Example:
            >>> Is.float(DType("Decimal", (10, 2)))
            False
        """
        return dtype.name in cls.FLOATS or (dtype.name == "Unknown" and dtype.args[0] == "float")

    @classmethod
    def dynamic_int(cls, dtype: DType) -> bool:
        """Say whether the type is the planner type of an integer literal with no data type.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.dynamic_int(DType.dynamic_int(1)), Is.dynamic_int(DType("Int32"))
            (True, False)
        """
        return dtype.name == "Unknown" and dtype.args[0] == "int"

    @classmethod
    def integer(cls, dtype: DType) -> bool:
        """Say whether the type is an integer, an integer literal included.

        Args:
            dtype: The type.

        Returns:
            Whether it is an integer.

        Example:
            >>> Is.integer(DType("UInt128"))
            True
        """
        return dtype.name in cls.SIGNED | cls.UNSIGNED or cls.dynamic_int(dtype)

    @classmethod
    def signed(cls, dtype: DType) -> bool:
        """Say whether the type is a signed integer type.

        Args:
            dtype: The type.

        Returns:
            Whether it is one. A literal is not.

        Example:
            >>> Is.signed(DType("Int8")), Is.signed(DType.dynamic_int(-1))
            (True, False)
        """
        return dtype.name in cls.SIGNED

    @classmethod
    def unsigned(cls, dtype: DType) -> bool:
        """Say whether the type is an unsigned integer type.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.unsigned(DType("UInt8"))
            True
        """
        return dtype.name in cls.UNSIGNED

    @classmethod
    def primitive_numeric(cls, dtype: DType) -> bool:
        """Say whether the type is an integer or a float.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.primitive_numeric(DType("Decimal", (10, 2)))
            False
        """
        return cls.float(dtype) or cls.integer(dtype)

    @classmethod
    def numeric(cls, dtype: DType) -> bool:
        """Say whether the type is an integer, a float or a `Decimal`.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.numeric(DType("Decimal", (10, 2)))
            True
        """
        return cls.primitive_numeric(dtype) or dtype.name == "Decimal"

    @classmethod
    def string(cls, dtype: DType) -> bool:
        """Say whether the type is `String`, a string literal included.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.string(DType.unknown("str"))
            True
        """
        return dtype.name == "String" or (dtype.name == "Unknown" and dtype.args[0] == "str")

    @classmethod
    def temporal(cls, dtype: DType) -> bool:
        """Say whether the type is `Date`, `Datetime`, `Duration` or `Time`.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.temporal(DType("Time"))
            True
        """
        return dtype.name in cls.TEMPORAL

    @classmethod
    def nested(cls, dtype: DType) -> bool:
        """Say whether the type is a `List`, an `Array` or a `Struct`.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.nested(DType("Array", (DType("Int8"), 2)))
            True
        """
        return dtype.name in cls.NESTED

    @classmethod
    def unknown(cls, dtype: DType) -> bool:
        """Say whether the type is a planner type `Unknown`.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.unknown(DType.unknown("any"))
            True
        """
        return dtype.name == "Unknown"

    @classmethod
    def unknown_any(cls, dtype: DType) -> bool:
        """Say whether the type is the planner type `Unknown(Any)`, of no literal kind.

        The planner gives this type where it cannot join two types, for
        example a `Date` beside an integer literal.

        Args:
            dtype: The type.

        Returns:
            Whether it is `Unknown(Any)`.

        Example:
            >>> Is.unknown_any(DType.unknown("any")), Is.unknown_any(DType.dynamic_int(1))
            (True, False)
        """
        return cls.unknown(dtype) and dtype.args[0] == "any"

    @classmethod
    def known(cls, dtype: DType) -> bool:
        """Say whether no part of the type is `Unknown`.

        Args:
            dtype: The type.

        Returns:
            Whether every part is known.

        Example:
            >>> Is.known(DType("List", (DType.dynamic_int(1),)))
            False
        """
        if dtype.name in Is.LISTS:
            return cls.known(dtype.inner())
        if dtype.name == "Struct":
            return all(cls.known(inner) for _, inner in dtype.fields())
        return not cls.unknown(dtype)

    @classmethod
    def primitive(cls, dtype: DType) -> bool:
        """Say whether the type is a number, `Boolean`, `String` or `Binary`.

        Args:
            dtype: The type.

        Returns:
            Whether it is one.

        Example:
            >>> Is.primitive(DType("Binary")), Is.primitive(DType("Date"))
            (True, False)
        """
        return cls.primitive_numeric(dtype) or dtype.name in {"Boolean", "String", "Binary"}

    @classmethod
    def list_arithmetic_input(cls, dtype: DType) -> bool:
        """Say whether list arithmetic takes the type as a leaf.

        The planner and the kernels of lists and arrays both read this test.

        Args:
            dtype: The type.

        Returns:
            Whether it is an integer, a float, `Boolean` or `Null`.

        Example:
            >>> Is.list_arithmetic_input(DType("Null")), Is.list_arithmetic_input(DType("String"))
            (True, False)
        """
        return cls.primitive_numeric(dtype) or dtype.name in {"Boolean", "Null"}


class DynamicInt:
    """Give the type that an integer literal takes, by its value.

    Polars gives an integer literal with no data type one of three types, by
    where it meets a type: the type that it takes alone, the smallest unsigned
    type that holds it beside an unsigned column, and the smallest type that
    holds it at all. Each is the first type of a list that holds the value
    (`IntegerRange`).

    Example:
        >>> DynamicInt.standalone(300), DynamicInt.positive(300), DynamicInt.smallest(300)
        (Int32, UInt16, Int16)
    """

    STANDALONE: ClassVar[tuple[str, ...]] = ("Int32", "Int64", "UInt64")
    POSITIVE: ClassVar[tuple[str, ...]] = ("UInt8", "UInt16", "UInt32", "UInt64")
    SMALLEST: ClassVar[tuple[str, ...]] = ("Int8", "Int16", "Int32", "Int64", "UInt64")

    @classmethod
    def standalone(cls, value: int) -> DType:
        """Give the type of the literal alone.

        Args:
            value: The value.

        Returns:
            `Int32`, `Int64`, `UInt64`, or `Int128` past them.

        Example:
            >>> DynamicInt.standalone(2**63)
            UInt64
        """
        return cls._first(cls.STANDALONE, value, "Int128")

    @classmethod
    def positive(cls, value: int) -> DType:
        """Give the smallest unsigned type that holds the literal.

        Args:
            value: The value.

        Returns:
            The type, or `Null` where no unsigned type holds the value.

        Example:
            >>> DynamicInt.positive(-1)
            Null
        """
        return cls._first(cls.POSITIVE, value, "Null")

    @classmethod
    def smallest(cls, value: int) -> DType:
        """Give the smallest integer type that holds the literal.

        Args:
            value: The value.

        Returns:
            The type, or `Null` where no 64-bit type holds the value.

        Example:
            >>> DynamicInt.smallest(-129)
            Int16
        """
        return cls._first(cls.SMALLEST, value, "Null")

    @classmethod
    def _first(cls, names: tuple[str, ...], value: int, last: str) -> DType:
        """Give the first integer type of a list that holds the value.

        Args:
            names: The names of the integer types, in the order of polars.
            value: The value.
            last: The type where no type of the list holds the value.

        Returns:
            The type.

        Example:
            >>> DynamicInt._first(("UInt8",), 256, "Null")
            Null
        """
        for name in names:
            if IntegerRange().holds(name, value):
                return DType(name)
        return DType(last)


class IntegerRange:
    """Give the range of each integer type, for the rules that read the values that a type holds.

    A cast keeps only the values that its target holds, a literal takes the
    first type that holds its value, and an integer wraps around at the ends
    of its range in the arithmetic of a literal.

    Example:
        >>> IntegerRange().holds("UInt8", 256)
        False
    """

    BOUNDS: ClassVar[dict[str, tuple[int, int]]] = {
        "Int8": (-(2**7), 2**7 - 1),
        "Int16": (-(2**15), 2**15 - 1),
        "Int32": (-(2**31), 2**31 - 1),
        "Int64": (-(2**63), 2**63 - 1),
        "Int128": (-(2**127), 2**127 - 1),
        "UInt8": (0, 2**8 - 1),
        "UInt16": (0, 2**16 - 1),
        "UInt32": (0, 2**32 - 1),
        "UInt64": (0, 2**64 - 1),
        "UInt128": (0, 2**128 - 1),
    }

    def holds(self, name: str, value: int) -> bool:
        """Say whether an integer type holds a value.

        Args:
            name: The name of the integer type.
            value: The value.

        Returns:
            Whether the value is in the range of the type.

        Example:
            >>> IntegerRange().holds("Int8", -128)
            True
        """
        low, high = self.BOUNDS[name]
        return low <= value <= high

    def wrapped(self, name: str, value: int) -> int:
        """Give the value that an integer type holds after the value wraps around its range.

        Args:
            name: The name of the integer type.
            value: The value, which can be out of the range.

        Returns:
            The value in the range, the same modulo the size of the range.

        Example:
            >>> IntegerRange().wrapped("Int8", 200), IntegerRange().wrapped("UInt8", 256)
            (-56, 0)
        """
        low, high = self.BOUNDS[name]
        return (value - low) % (high - low + 1) + low


class BitWidth:
    """Give the number of bits of each primitive number, for a rule that reads its bits.

    `reinterpret` reads the bits of each value as another number of the
    same width: the bits of an `Int8` as a `UInt8`, or of a `Float32` as an
    `Int32`. A `Decimal`, a temporal type and a literal with no data type
    have no width here, so no other type reads their bits.

    Example:
        >>> BitWidth().of(DType("Float16")), BitWidth().of(DType("Date")) is None
        (16, True)
    """

    WIDTHS: ClassVar[dict[str, int]] = {
        "Int8": 8,
        "UInt8": 8,
        "Int16": 16,
        "UInt16": 16,
        "Float16": 16,
        "Int32": 32,
        "UInt32": 32,
        "Float32": 32,
        "Int64": 64,
        "UInt64": 64,
        "Float64": 64,
        "Int128": 128,
        "UInt128": 128,
    }

    def of(self, dtype: DType) -> int | None:
        """Give the number of bits of a primitive number.

        Args:
            dtype: The data type.

        Returns:
            The number of bits, or `None` where the type is not a primitive number.

        Example:
            >>> BitWidth().of(DType("UInt128")), BitWidth().of(DType("Decimal", (10, 2))) is None
            (128, True)
        """
        return self.WIDTHS.get(dtype.name)


class Decimals:
    """Give the facts of the `Decimal` type: its largest precision and the fit of a value.

    A `Decimal` holds its digits as one 128-bit integer with no point. A value
    fits a precision where its digits are fewer than the precision. An integer
    fits a precision and a scale where it times ten to the scale fits the
    precision. A sum or a product of decimals takes the largest precision, 38.

    Example:
        >>> Decimals().holds(1000, 5, 2), Decimals().widest(2)
        (False, Decimal(precision=38, scale=2))
    """

    MAX_PRECISION: ClassVar[int] = 38

    def widest(self, scale: int) -> DType:
        """Give the `Decimal` of the largest precision with a scale.

        Args:
            scale: The scale.

        Returns:
            The type.

        Example:
            >>> Decimals().widest(0)
            Decimal(precision=38, scale=0)
        """
        return DType("Decimal", (self.MAX_PRECISION, scale))

    def fits(self, digits: int, precision: int) -> bool:
        """Say whether the digits of a decimal value fit a precision.

        Args:
            digits: The digits of the value, with no point.
            precision: The precision.

        Returns:
            Whether the digits are fewer than the precision.

        Example:
            >>> Decimals().fits(999, 3), Decimals().fits(-1000, 3)
            (True, False)
        """
        limit: int = 10**precision
        return abs(digits) < limit

    def holds(self, value: int, precision: int, scale: int) -> bool:
        """Say whether a `Decimal` holds an integer value.

        Args:
            value: The integer value.
            precision: The precision of the decimal.
            scale: The scale of the decimal.

        Returns:
            Whether the value times ten to the scale fits the precision.

        Example:
            >>> Decimals().holds(127, 5, 2), Decimals().holds(127, 4, 2)
            (True, False)
        """
        return self.fits(value * 10**scale, precision)


class TimeUnit:
    """Give the facts of the time units of `Datetime` and `Duration`: their size and their join.

    A unit is `ns`, `us` or `ms`. A value of a `Datetime` or a `Duration`
    counts its unit, so a cast to a finer unit multiplies the count. A
    difference of two values of a kind with no unit, `Date` or `Time`, has a
    fixed unit. Two units join into one of the two (`joined`).

    Example:
        >>> TimeUnit.joined("ns", "us"), TimeUnit.joined("us", "ms"), TimeUnit.joined("ms", "ns")
        ('us', 'ms', 'ms')
    """

    PER_SECOND: ClassVar[dict[str, int]] = {"ms": 10**3, "us": 10**6, "ns": 10**9}
    SECONDS_PER_DAY: ClassVar[int] = 86_400
    DIFFERENCE: ClassVar[dict[str, str]] = {"Date": "us", "Time": "ns"}

    @classmethod
    def joined(cls, left: str, right: str) -> str:
        """Give the unit that polars keeps for two units.

        Args:
            left: The unit of the left type.
            right: The unit of the right type.

        Returns:
            The unit.

        Example:
            >>> TimeUnit.joined("ns", "ns")
            'ns'
        """
        if (left, right) == ("ns", "us"):
            return "us"
        if right == "ms":
            return "ms"
        return left

    @classmethod
    def per_day(cls, unit: str) -> int:
        """Give the number of values of a unit in one day.

        Args:
            unit: The unit.

        Returns:
            The number.

        Example:
            >>> TimeUnit.per_day("ms")
            86400000
        """
        return cls.SECONDS_PER_DAY * cls.PER_SECOND[unit]

    @classmethod
    def as_fine(cls, unit: str, other: str) -> bool:
        """Say whether a unit is as fine as another unit, or finer.

        Args:
            unit: The unit.
            other: The other unit.

        Returns:
            Whether one second holds at least as many values of the unit.

        Example:
            >>> TimeUnit.as_fine("ns", "us"), TimeUnit.as_fine("ms", "us")
            (True, False)
        """
        return cls.PER_SECOND[unit] >= cls.PER_SECOND[other]

    @classmethod
    def finer(cls, unit: str, other: str) -> bool:
        """Say whether a unit is finer than another unit.

        Args:
            unit: The unit.
            other: The other unit.

        Returns:
            Whether one second holds more values of the unit.

        Example:
            >>> TimeUnit.finer("us", "us")
            False
        """
        return cls.PER_SECOND[unit] > cls.PER_SECOND[other]


class KnownNan(float):
    """Hold a NaN that the code writes, as a float that a rule can tell from a null of the tree.

    The JSON form of the tree of polars writes NaN, `inf` and `-inf` alike as
    `null`, so a rule reads a null as a value that it does not know. A NaN that
    the code holds is known, and this class marks it. It is a float, so every
    rule reads it as NaN.

    Example:
        >>> import math
        >>> math.isnan(KnownNan("nan")), isinstance(KnownNan("nan"), float)
        (True, True)
    """


class Parameter:
    """Read one parameter of a data type with its Python type.

    `DType` keeps its parameters as plain objects, so a rule that computes
    with a precision or a time unit reads it here. A parameter of the wrong
    type is a bug in a rule.

    Example:
        >>> Parameter().integer(DType("Decimal", (10, 2)), 1)
        2
    """

    def integer(self, dtype: DType, index: int) -> int:
        """Give a parameter that is an integer, such as a precision, a scale or a width.

        Args:
            dtype: The data type.
            index: The place of the parameter.

        Returns:
            The parameter.

        Raises:
            NoPartError: The parameter is not an integer.

        Example:
            >>> Parameter().integer(DType("Array", (DType("Int8"), 3)), 1)
            3
            >>> Parameter().integer(DType("Int8"), 1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        value = dtype.args[index] if index < len(dtype.args) else None
        if not isinstance(value, int):  # pragma: no cover -- each caller checks the kind
            raise NoPartError(dtype.render(), f"integer parameter {index}")
        return value

    def text(self, dtype: DType, index: int) -> str | None:
        """Give a parameter that is text or nothing, such as a time unit or a time zone.

        Args:
            dtype: The data type.
            index: The place of the parameter.

        Returns:
            The parameter, or `None`.

        Raises:
            NoPartError: The parameter is not text.

        Example:
            >>> Parameter().text(DType("Datetime", ("us", None)), 0)
            'us'
            >>> Parameter().text(DType("Array", (DType("Int8"), 3)), 1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        value = dtype.args[index] if index < len(dtype.args) else None
        if (  # pragma: no cover -- each caller checks the kind
            value is not None and not isinstance(value, str)
        ):
            raise NoPartError(dtype.render(), f"text parameter {index}")
        return value

    def unit(self, dtype: DType) -> str:
        """Give the time unit of a `Datetime` or a `Duration`.

        Args:
            dtype: The data type.

        Returns:
            The unit: `ns`, `us` or `ms`.

        Raises:
            NoPartError: The type has no time unit.

        Example:
            >>> Parameter().unit(DType("Duration", ("ms",)))
            'ms'
            >>> Parameter().unit(DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        unit = self.text(dtype, 0)
        if unit is None:  # pragma: no cover -- each caller checks the kind
            raise NoPartError(dtype.render(), "time unit")
        return unit

    def categories(self, dtype: DType) -> tuple[str, ...]:
        """Give the categories of an `Enum`, in order.

        Args:
            dtype: The data type.

        Returns:
            The categories.

        Raises:
            NoPartError: The type has no categories.

        Example:
            >>> Parameter().categories(DType("Enum", (("a", "b"),)))
            ('a', 'b')
            >>> Parameter().categories(DType("Categorical"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.NoPartError: ...
        """
        categories = dtype.args[0] if dtype.name == "Enum" and dtype.args else None
        if not isinstance(categories, tuple):  # pragma: no cover -- each caller checks the kind
            raise NoPartError(dtype.render(), "categories")
        return tuple(str(category) for category in categories)


class RenderingError(ValueError):
    """Say that a text is not the rendering of a data type that the rules know.

    Example:
        >>> str(RenderingError("Foo"))
        'no data type has the rendering Foo'
    """

    def __init__(self, text: str) -> None:
        """Keep the text in the message.

        Args:
            text: The text.
        """
        super().__init__(f"no data type has the rendering {text}")


class Rendering:
    """Read the rendering of a data type, the text that polars prints for it, as a `DType`.

    The checker holds the data type of each column of a frame as its
    rendering, for example `Datetime(time_unit='ms', time_zone='UTC')`. The
    rules read a `DType`. A rendering has the syntax of a Python expression,
    so the parser reads it with `ast.parse` and walks the syntax tree. It runs
    nothing. It reads the form of polars for an array of several dimensions,
    `Array(Int8, shape=(2, 3))`, and the nested form that `DType.render` gives.

    Example:
        >>> Rendering().parse("List(Datetime(time_unit='ms', time_zone=None))")
        List(Datetime(time_unit='ms', time_zone=None))
        >>> Rendering().parse("Array(Int8, shape=(2, 3))").args[1]
        2
    """

    PLAIN: ClassVar[frozenset[str]] = frozenset({
        *Is.SIGNED,
        *Is.UNSIGNED,
        *Is.FLOATS,
        "Boolean",
        "String",
        "Binary",
        "Date",
        "Time",
        "Null",
        "Categorical",
        "Object",
    })
    KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "Decimal": ("precision", "scale"),
        "Datetime": ("time_unit", "time_zone"),
        "Duration": ("time_unit",),
        "Enum": ("categories",),
    }

    def parse(self, text: str) -> DType:
        """Give the data type of a rendering.

        Args:
            text: The rendering.

        Returns:
            The data type.

        Raises:
            RenderingError: The text is not the rendering of a data type that
                the rules know, for example `Unchecked`.

        Example:
            >>> Rendering().parse("Struct({'a': Int8})")
            Struct({'a': Int8})
            >>> Rendering().parse("Unchecked")
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.RenderingError: no data type has the rendering Unchecked
            >>> Rendering().parse("List(")
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.RenderingError: no data type has the rendering List(
        """
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as error:  # pragma: no cover -- each rendering is Python
            raise RenderingError(text) from error
        return self._dtype(tree.body, text)

    def _dtype(self, node: ast.expr, text: str) -> DType:
        """Give the data type of one part of the syntax tree.

        Args:
            node: The part.
            text: The whole rendering, for the error.

        Returns:
            The data type.

        Raises:
            RenderingError: The part is neither the name of a kind nor a call
                of a kind with parameters.

        Example:
            >>> Rendering()._dtype(ast.parse("Unknown", mode="eval").body, "Unknown")
            Unknown
        """
        if isinstance(node, ast.Name) and node.id in self.PLAIN:
            return DType(node.id)
        if isinstance(node, ast.Name) and node.id == "Unknown":
            return DType.unknown("any")
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise RenderingError(text)
        values = [self._value(argument, text) for argument in node.args]
        keywords = {item.arg: self._value(item.value, text) for item in node.keywords}
        return self._built(node.func.id, values, keywords, text)

    def _value(self, node: ast.expr, text: str) -> object:
        """Give the value of one argument of a call: a constant, a sequence, a dict or a data type.

        Args:
            node: The argument.
            text: The whole rendering, for the error.

        Returns:
            The value. A dict becomes a tuple of pairs, as the fields of a struct.

        Example:
            >>> Rendering()._value(ast.parse("{'a': Int8}", mode="eval").body, "")
            (('a', Int8),)
        """
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Tuple | ast.List):
            return tuple(self._value(item, text) for item in node.elts)
        if isinstance(node, ast.Dict):
            return self._pairs(node, text)
        return self._dtype(node, text)

    def _pairs(self, node: ast.Dict, text: str) -> tuple[tuple[object, object], ...]:
        """Give the pairs of a dict, as the fields of a struct.

        Args:
            node: The dict.
            text: The whole rendering, for the error.

        Returns:
            The pairs of a key and a value, in order.

        Raises:
            RenderingError: The dict unpacks another dict, which no rendering does.

        Example:
            >>> Rendering()._pairs(ast.parse("{'a': Int8}", mode="eval").body, "")
            (('a', Int8),)
            >>> Rendering()._pairs(ast.parse("{**x}", mode="eval").body, "{**x}")
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.RenderingError: no data type has the rendering {**x}
        """
        pairs: list[tuple[object, object]] = []
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:  # pragma: no cover -- a rendering names each field
                raise RenderingError(text)
            pairs.append((self._value(key, text), self._value(value, text)))
        return tuple(pairs)

    def _built(
        self,
        name: str,
        values: list[object],
        keywords: dict[str | None, object],
        text: str,
    ) -> DType:
        """Give the data type of a kind with parameters.

        Args:
            name: The name of the kind.
            values: The arguments by position.
            keywords: The arguments by name.
            text: The whole rendering, for the error.

        Returns:
            The data type.

        Raises:
            RenderingError: The kind has no parameters, or an argument is missing.

        Example:
            >>> Rendering()._built("Duration", [], {"time_unit": "ms"}, "")
            Duration(time_unit='ms')
            >>> Rendering()._built("Array", [DType("Int8")], {"shape": ()}, "Array")
            Traceback (most recent call last):
            ...
            astrolabe._rules.dtypes.RenderingError: no data type has the rendering Array
        """
        try:
            if name == "Array":
                return self._array(values[0], keywords["shape"])
            if name in {"List", "Struct"}:
                return DType(name, (values[0],))
            return DType(name, tuple(keywords[key] for key in self.KEYWORDS[name]))
        except (KeyError, IndexError) as error:  # pragma: no cover -- each parameter is written
            raise RenderingError(text) from error

    def _array(self, inner: object, shape: object) -> DType:
        """Give the data type of an array, one array for each width of its shape.

        Args:
            inner: The data type of the values.
            shape: The widths, the outer width first.

        Returns:
            The array.

        Raises:
            KeyError: The shape is not a tuple of widths.

        Example:
            >>> Rendering()._array(DType("Int8"), (2, 3))
            Array(Int8, shape=(2, 3))
        """
        if (  # pragma: no cover -- a rendering writes a shape as a tuple
            not isinstance(shape, tuple) or not shape
        ):
            raise KeyError(shape)
        built = DType("Array", (inner, shape[-1]))
        for width in reversed(shape[:-1]):
            built = DType("Array", (built, width))
        return built
