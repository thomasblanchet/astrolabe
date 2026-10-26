"""The type functions of polars: the planner type of a node from the types of its inputs.

Polars types each node of an expression before any data runs. A function node
gets its type from one call of its `FieldsMapper`. Every other node gets its
type from the rule of its kind (`ExprFields`). Examples are a binary operator,
a `when`, a cast and an aggregation. That type is the planner type, the type
that `collect_schema()` shows. This module translates both parts, so that the
type function of a node is one call that polars itself names, for example
`FieldsMapper(dtype).with_same_dtype()`.

A type function can refuse a query, as a type function of polars does with an
error. It then raises `PolarsRefusal`, with the Python class of the error that
polars raises and the message of the refusal.
"""

from __future__ import annotations

import decimal
import itertools
import math
import re
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import BitWidth, Decimals, DType, Is, KnownNan, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Undecided, UndecidedError
from astrolabe._rules.operators import NumericOp, Operator
from astrolabe._rules.releases import polars_releases
from astrolabe._rules.supertype import Supertype

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class Operand:
    """Hold one input of a node as the planner reads it: its type, and its value as a literal.

    Most rules read only the type. A few rules also read whether the input is a
    literal node at that moment, and its value: the arithmetic of a literal
    with no data type, and the casts that the cast pass folds into a literal.
    `pl.lit(value)` of a Python `int` or `float` is a literal with no data
    type, of the planner type `Unknown`. A `str`, a `bool` and `None` give a
    literal of type `String`, `Boolean` and `Null`. `scalar` says that the
    input gives one value, as an aggregation or a literal does, which the
    type of an `over` and a `rolling` reads. It takes no part in the equality
    of two inputs.

    Example:
        >>> Operand.of_value(300)
        Operand(dtype=Unknown(int, 300), literal=True, value=300)
        >>> Operand.column(DType("Int8")).literal
        False
    """

    dtype: DType
    literal: bool = False
    value: object = None
    scalar: bool = dataclasses.field(default=False, repr=False, compare=False)

    @classmethod
    def column(cls, dtype: DType) -> Operand:
        """Give an input that is not a literal, for example a column.

        Args:
            dtype: Its planner type.

        Returns:
            The input.

        Example:
            >>> Operand.column(DType("Date"))
            Operand(dtype=Date, literal=False, value=None)
        """
        return cls(dtype)

    @classmethod
    def of_value(cls, value: bool | int | float | str | None) -> Operand:
        """Give the input `pl.lit(value)` of a Python value with no data type.

        Args:
            value: The Python value.

        Returns:
            The literal input.

        Example:
            >>> Operand.of_value(1.5).dtype.args, Operand.of_value(None).dtype
            (('float', None), Null)
        """
        if isinstance(value, bool):
            return cls(DType("Boolean"), literal=True, value=value, scalar=True)
        if isinstance(value, int):
            return cls(DType.dynamic_int(value), literal=True, value=value, scalar=True)
        if isinstance(value, float):
            return cls(DType.unknown("float"), literal=True, value=value, scalar=True)
        if isinstance(value, str):
            return cls(DType("String"), literal=True, value=value, scalar=True)
        return cls(DType("Null"), literal=True, scalar=True)

    @classmethod
    def typed_literal(cls, value: object, dtype: DType) -> Operand:
        """Give a literal node that holds a value of a known type.

        The cast pass makes such a node where it folds a cast into a literal.

        Args:
            value: The Python value.
            dtype: The type of the literal.

        Returns:
            The literal input.

        Example:
            >>> Operand.typed_literal(1, DType("UInt8")).dtype
            UInt8
        """
        return cls(dtype, literal=True, value=value, scalar=True)


class FieldsMapper:  # ruff: ignore[too-many-public-methods] -- one method per method of polars
    """Type a function from the types of its inputs, with the `FieldsMapper` of polars.

    Polars declares the type of each function as one call of a method of its
    mapper. Each method here has the name of the method of polars and gives the
    planner type of the output, or raises `PolarsRefusal` where polars returns
    an error. A method that polars passes a closure takes a function of a type.
    `ensure_is_list`, `ensure_is_array` and `ensure_satisfies` check the inputs
    and give the mapper back, so a call chains as in polars:
    `FieldsMapper(dtype).ensure_is_list().with_same_dtype()`.

    The class has one method for each method of the mapper of polars, so it
    has as many. It reads only the types. The output takes the name of the
    first input, as in polars, and code that needs the name reads it there.

    Example:
        >>> FieldsMapper(DType("List", (DType("Int8"),))).ensure_is_list().with_same_dtype()
        List(Int8)
        >>> FieldsMapper(DType("Int8"), DType("UInt8")).map_to_supertype()
        Int16
    """

    NARROW_FLOATS: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32"})

    def __init__(self, *dtypes: DType) -> None:
        """Keep the planner types of the inputs, in the order of the node.

        Args:
            dtypes: The types. A function has at least one input.

        Example:
            >>> FieldsMapper(DType("Int8"), DType("String")).with_same_dtype()
            Int8
        """
        self._dtypes = dtypes

    def with_same_dtype(self) -> DType:
        """Give the type of the first input.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Date")).with_same_dtype()
            Date
        """
        return self.map_dtype(lambda dtype: dtype)

    def with_dtype(self, dtype: DType) -> DType:
        """Give a fixed type, whatever the inputs.

        Args:
            dtype: The type.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("String")).with_dtype(DType.idx())
            UInt32
        """
        return dtype

    def map_dtype(self, func: Callable[[DType], DType]) -> DType:
        """Give a function of the type of the first input.

        Args:
            func: The function, which cannot refuse.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("UInt8")).map_dtype(FunctionClosures().diff)
            Int16
        """
        return func(self._dtypes[0])

    def try_map_field(self, func: Callable[[DType], DType]) -> DType:
        """Give a function of the first input that can refuse.

        Polars passes the whole input, its name and its type. The functions
        that use this method read only the type, or the fields of a struct.
        The function raises `PolarsRefusal` to refuse.

        Args:
            func: The function.

        Returns:
            The type.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> FieldsMapper(struct).try_map_field(StructClosures().field_by_name("a"))
            Int8
        """
        return func(self._dtypes[0])

    def try_map_dtype(self, func: Callable[[DType], DType]) -> DType:
        """Give a function of the type of the first input that can refuse.

        The function raises `PolarsRefusal` to refuse.

        Args:
            func: The function.

        Returns:
            The type.

        Example:
            >>> moment = DType("Datetime", ("us", None))
            >>> FieldsMapper(moment).try_map_dtype(TemporalClosures().with_time_unit("ms"))
            Datetime(time_unit='ms', time_zone=None)
        """
        return func(self._dtypes[0])

    def try_map_dtypes(self, func: Callable[[tuple[DType, ...]], DType]) -> DType:
        """Give a function of the types of all the inputs that can refuse.

        The function raises `PolarsRefusal` to refuse.

        Args:
            func: The function.

        Returns:
            The type.

        Example:
            >>> mapper = FieldsMapper(DType("Int8"), DType("Float32"))
            >>> mapper.try_map_dtypes(FunctionClosures().corr_cov)
            Float32
        """
        return func(self._dtypes)

    def to_physical_type(self) -> DType:
        """Give the physical type that holds the data.

        Returns:
            The type: for example `Int32` for a `Date`.

        Example:
            >>> FieldsMapper(DType("List", (DType("Date"),))).to_physical_type()
            List(Int32)
        """
        return self.map_dtype(lambda dtype: dtype.physical())

    def map_to_supertype(self) -> DType:
        """Give the supertype of all the inputs.

        The fold starts at the first type. A `Categorical` first input keeps
        its type where the join is `String`. A literal type takes the type that
        it takes alone: `Int32` for `pl.lit(1)`. A step with no supertype
        refuses with `PolarsRefusal`, as `Supertype.require` does.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Categorical"), DType("String")).map_to_supertype()
            Categorical
            >>> FieldsMapper(DType.dynamic_int(1), DType.dynamic_int(2)).map_to_supertype()
            Int32
        """
        first = self._dtypes[0]
        joined = self._fold(self._dtypes)
        if first.name == "Categorical" and joined.name == "String":
            return first
        return joined.materialized() if Is.unknown(joined) else joined

    def map_to_datetime_ranges_dtype(self, time_unit: str | None, time_zone: str | None) -> DType:
        """Give the type of `datetime_ranges`: a list of the type of `datetime_range`.

        Args:
            time_unit: The time unit that the call names, or `None`.
            time_zone: The time zone that the call names, or `None`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Date"), DType("Date")).map_to_datetime_ranges_dtype("ms", None)
            List(Datetime(time_unit='ms', time_zone=None))
        """
        return DType("List", (self.map_to_datetime_range_dtype(time_unit, time_zone),))

    def linear_space_dtype(self) -> DType:
        """Give the type of the values of `linear_space`, from the types of its two bounds.

        Two bounds of `Float16` or of `Float32` keep that type. Two bounds of
        one temporal type keep it, and a `Date` becomes a `Datetime` in
        microseconds. Other numbers give `Float64`, and any other pair raises.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A bound is not a number, and the bounds are not of one
                temporal type.

        Example:
            >>> FieldsMapper(DType("Date"), DType("Date")).linear_space_dtype()
            Datetime(time_unit='us', time_zone=None)
            >>> FieldsMapper(DType("Int8"), DType("Float32")).linear_space_dtype()
            Float64
        """
        start, end = self._dtypes[0], self._dtypes[1]
        if start.name in {"Float16", "Float32"} and start.same(end):
            return start
        if Is.temporal(start) and start.same(end):
            return DType("Datetime", ("us", None)) if start.name == "Date" else start
        if not Is.primitive_numeric(start) or not Is.primitive_numeric(end):
            message = f"'start' and 'end' have incompatible dtypes, got {start!r} and {end!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return DType("Float64")

    def linear_spaces_dtype(self, array_width: int | None) -> DType:
        """Give the type of `linear_spaces`: a list of the values of `linear_space`, or an array.

        Args:
            array_width: The width of each array, or `None` for a list.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Int8"), DType("Int8")).linear_spaces_dtype(3)
            Array(Float64, shape=(3,))
        """
        inner = self.linear_space_dtype()
        if array_width is None:
            return DType("List", (inner,))
        return DType("Array", (inner, array_width))

    def map_to_supertype_with(self, func: Callable[[DType], DType]) -> DType:
        """Give a closure of the supertype of all the inputs, as polars maps that field.

        Args:
            func: The closure.

        Returns:
            The type.

        Example:
            >>> booleans = FieldsMapper(DType("Boolean"), DType("Boolean"))
            >>> booleans.map_to_supertype_with(FunctionClosures().sum_horizontal)
            UInt32
        """
        return func(self.map_to_supertype())

    def as_struct(self, names: Sequence[str]) -> DType:
        """Give the type of `pl.struct`: one field for each input, by its name and its type.

        Args:
            names: The output name of each input.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: Two inputs have one name.

        Example:
            >>> FieldsMapper(DType("Int8"), DType("Date")).as_struct(("a", "b"))
            Struct({'a': Int8, 'b': Date})
        """
        if len(set(names)) < len(names):
            raise PolarsRefusal(ErrorClass.DUPLICATE, f"multiple fields with names {names}")
        fields = tuple(zip(names, (dtype.materialized() for dtype in self._dtypes), strict=True))
        return DType("Struct", (fields,))

    def _fold(self, dtypes: Sequence[DType]) -> DType:
        """Join types from the first one, each join as `Supertype.require` gives it.

        Args:
            dtypes: The types.

        Returns:
            The supertype of all of them.

        Example:
            >>> FieldsMapper(DType("Int8"))._fold([DType("Int8"), DType("Float32")])
            Float32
        """
        joined = dtypes[0]
        for dtype in dtypes[1:]:
            joined = Supertype().require(joined, dtype)
        return joined

    def replace_dtype(self, return_dtype: DType | None) -> DType:
        """Give the type of `replace_strict`, of the column, `old`, `new` and the default.

        Without a return type, the type is the inner type of `new`, or the type
        of `new` where it is not a list, joined with the type of the default
        where there is one. A default with no join refuses with
        `PolarsRefusal`.

        Args:
            return_dtype: The type that the call names, or `None`.

        Returns:
            The type.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> FieldsMapper(DType("String"), listed, listed, DType("Int16")).replace_dtype(None)
            Int16
            >>> FieldsMapper(DType("String"), listed, listed).replace_dtype(None)
            Int8
            >>> FieldsMapper(DType("String"), listed, listed).replace_dtype(DType("Date"))
            Date
        """
        if return_dtype is not None:
            return return_dtype
        new = self._dtypes[2]
        inner = new.inner() if new.name in Is.LISTS else new
        if len(self._dtypes) < 4:
            return inner
        return Supertype().require(self._dtypes[3], inner)

    def map_to_list_and_array_inner_dtype(self) -> DType:
        """Give the inner type of a list or an array, or `Unknown`.

        Returns:
            The inner type, or `Unknown` for a type that is not a list or an array.

        Example:
            >>> FieldsMapper(DType("Array", (DType("Int8"), 2))).map_to_list_and_array_inner_dtype()
            Int8
            >>> FieldsMapper(DType("Int8")).map_to_list_and_array_inner_dtype()
            Unknown
        """
        first = self._dtypes[0]
        return first.inner() if first.name in Is.LISTS else DType.unknown("any")

    def try_map_to_array_inner_dtype(self) -> DType:
        """Give the inner type of an array, and refuse any other type.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is not an array.

        Example:
            >>> FieldsMapper(DType("Int8")).try_map_to_array_inner_dtype()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        first = self._dtypes[0]
        if first.name != "Array":  # pragma: no cover -- the check refuses any other type
            message = f"expected Array type, got: {first!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return first.inner()

    def map_to_list_supertype(self) -> DType:
        """Give a list of the supertype of the inputs, a list by its inner type.

        An array input counts as one element, as it is. A step with no
        supertype refuses with `PolarsRefusal`.

        Returns:
            The type.

        Example:
            >>> listed = DType("List", (DType("UInt8"),))
            >>> FieldsMapper(DType("Int8"), listed).map_to_list_supertype()
            List(Int16)
        """
        inners = [dtype.inner() if dtype.name == "List" else dtype for dtype in self._dtypes]
        return DType("List", (self._fold(inners),))

    def map_datetime_dtype_timezone(self, time_zone: str | None) -> DType:
        """Give a `Datetime` with a new time zone, and refuse any other type.

        Args:
            time_zone: The new time zone, or `None`.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is not a `Datetime`.

        Example:
            >>> FieldsMapper(DType("Datetime", ("ms", None))).map_datetime_dtype_timezone("UTC")
            Datetime(time_unit='ms', time_zone='UTC')
        """
        first = self._dtypes[0]
        if first.name != "Datetime":
            message = f"replace-time-zone operation not supported for dtype `{first!r}`"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return DType("Datetime", (Parameter().unit(first), time_zone))

    def map_to_datetime_range_dtype(self, time_unit: str | None, time_zone: str | None) -> DType:
        """Give the `Datetime` of a range of moments.

        The range takes the unit and the zone of the supertype of its inputs
        where that is a `Datetime`, and microseconds with no zone otherwise.
        A unit or a zone that the call names wins. Inputs with no supertype
        refuse with `PolarsRefusal`.

        Args:
            time_unit: The unit that the call names, or `None`.
            time_zone: The zone that the call names, or `None`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Date"), DType("Date")).map_to_datetime_range_dtype(None, "UTC")
            Datetime(time_unit='us', time_zone='UTC')
        """
        joined = self.map_to_supertype()
        unit, zone = "us", None
        if joined.name == "Datetime":
            unit, zone = Parameter().unit(joined), Parameter().text(joined, 1)
        return DType("Datetime", (time_unit or unit, time_zone or zone))

    def ensure_satisfies(self, check: Callable[[int, DType], bool], op: str) -> FieldsMapper:
        """Check every input with a test of its place and its type.

        Args:
            check: The test of the place and the type of one input.
            op: The name of the function, for the message.

        Returns:
            The mapper, so that the call goes on.

        Raises:
            PolarsRefusal: An input does not pass the test.

        Example:
            >>> mapper = FieldsMapper(DType("Int8"))
            >>> checked = mapper.ensure_satisfies(lambda _, dtype: Is.numeric(dtype), "sign")
            >>> checked.with_same_dtype()
            Int8
        """
        for index, dtype in enumerate(self._dtypes):
            if not check(index, dtype):
                refused = f"`{op}` operation not supported for dtype `{dtype!r}`"
                message = f"{refused} as argument {index}"
                raise PolarsRefusal(
                    ErrorClass.INVALID_OPERATION,
                    message,
                )
        return self

    def ensure_is_list(self) -> FieldsMapper:
        """Check that the first input is a list.

        Returns:
            The mapper, so that the call goes on.

        Raises:
            PolarsRefusal: The first input is not a list.

        Example:
            >>> FieldsMapper(DType("Int8")).ensure_is_list()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        first = self._dtypes[0]
        if first.name != "List":
            message = f"expected List data type for list operation, got: {first!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return self

    def ensure_is_array(self) -> FieldsMapper:
        """Check that the first input is an array.

        Returns:
            The mapper, so that the call goes on.

        Raises:
            PolarsRefusal: The first input is not an array.

        Example:
            >>> FieldsMapper(DType("Array", (DType("Int8"), 2))).ensure_is_array().with_same_dtype()
            Array(Int8, shape=(2,))
        """
        first = self._dtypes[0]
        if first.name != "Array":
            message = f"expected Array datatype for array operation, got: {first!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return self

    @polars_releases
    def var_dtype(self, op: str = "var") -> DType:
        """Give the type of a variance or a standard deviation.

        A `Duration`, at the top or under any number of lists or arrays,
        refuses with `PolarsRefusal`. Every other input takes the type of
        `moment_dtype`.

        Args:
            op: The name of the function, `var` or `std`, for the message.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The leaf of the input is a `Duration`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> nested = FieldsMapper(DType("List", (DType("Duration", ("us",)),)))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Int8")).var_dtype("std")
            Float64
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     nested.var_dtype("std")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: operation `std` ...
        """
        leaf = self._dtypes[0].leaf()
        if leaf.name == "Duration":
            message = f"operation `{op}` is not supported for `{leaf!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self.moment_dtype()

    @var_dtype.before("2.0.0")
    def _var_one_level(self, op: str = "var") -> DType:
        """Give the type of a variance or a standard deviation in the older releases.

        A standard deviation takes the type of `moment_dtype`. A variance of a
        `Duration`, or of a list or an array of them, refuses with
        `PolarsRefusal`. The test reads the leaf, the type under every list,
        but the refusal reads one level.

        Args:
            op: The name of the function, `var` or `std`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Int8"))._var_one_level()
            Float64
            >>> FieldsMapper(DType("Duration", ("us",)))._var_one_level("std")
            Duration(time_unit='us')
        """
        if op != "var" or self._dtypes[0].leaf().name != "Duration":
            return self.moment_dtype()
        return self.try_map_dtype(lambda dtype: self._var_inner(self._first_level(dtype)))

    @polars_releases
    def ewm_var_dtype(self, op: str) -> DType:
        """Give the type of `ewm_std` or `ewm_var`: a float, after a refusal of a `Duration`.

        Args:
            op: The name of the function, `ewm_std` or `ewm_var`, for the message.

        Returns:
            The type of `map_numeric_to_float_dtype`, with a `Decimal` cast.

        Raises:
            PolarsRefusal: The leaf of the input is a `Duration`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Date")).ewm_var_dtype("ewm_var")
            Date
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Duration", ("us",))).ewm_var_dtype("ewm_var")
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: operation `ewm_var` ...
        """
        leaf = self._dtypes[0].leaf()
        if leaf.name == "Duration":
            message = f"operation `{op}` is not supported for `{leaf!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self.map_numeric_to_float_dtype(coerce_decimal=True)

    @ewm_var_dtype.before("2.0.0")
    def _ewm_older(self, op: str) -> DType:
        """Give the type of `ewm_std` or `ewm_var` in the older releases.

        `ewm_std` gives a float, and `ewm_var` the type of a variance.

        Args:
            op: The name of the function, `ewm_std` or `ewm_var`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Date"))._ewm_older("ewm_var")
            Datetime(time_unit='us', time_zone=None)
        """
        if op == "ewm_std":
            return self.map_numeric_to_float_dtype(coerce_decimal=True)
        return self.var_dtype()

    @polars_releases
    def ensure_number_or_boolean(self, op: str) -> FieldsMapper:
        """Check that every input is a number or a `Boolean`, as `exp`, `log` and `entropy` need.

        Args:
            op: The name of the function, for the message.

        Returns:
            The mapper, so that the call goes on.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("String")).ensure_number_or_boolean("exp")
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `exp` operation ...
        """
        return self.ensure_satisfies(
            lambda _, dtype: Is.numeric(dtype) or dtype.name == "Boolean",
            op,
        )

    @ensure_number_or_boolean.before("2.0.0")
    def _no_number_check(self, op: str) -> FieldsMapper:
        """Check nothing, as the older releases do.

        Args:
            op: The name of the function.

        Returns:
            The mapper.

        Example:
            >>> FieldsMapper(DType("String"))._no_number_check("exp").map_to_float_dtype()
            Float64
        """
        del op
        return self

    @polars_releases
    def sign_dtype(self) -> DType:
        """Give the type of `sign`: the type of the input, with an integer digit for a `Decimal`.

        The result 1 needs one digit left of the point, so a `Decimal` takes
        at least its scale plus one as its precision, up to 38.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Decimal", (3, 3))).sign_dtype()
            Decimal(precision=4, scale=3)
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Int8")).sign_dtype()
            Int8
        """
        first = self._dtypes[0]
        if first.name != "Decimal":
            return first
        precision, scale = Parameter().integer(first, 0), Parameter().integer(first, 1)
        return DType("Decimal", (min(max(precision, scale + 1), Decimals.MAX_PRECISION), scale))

    @polars_releases
    def carry_dtype(self, decimals: int | None = 0, mode: str = "HalfToEven") -> DType:
        """Give the type of a rounding: one more digit for a `Decimal` where the rounding can carry.

        A rounding can carry into one more integer digit, as 9.9 rounds to
        10. So a `Decimal` gets one more digit of precision, up to 38, where
        its scale is larger than `decimals`. The mode `ToZero` never carries.
        `floor` and `ceil` keep no decimal, and `round_sig_figs` can carry at
        every scale (`decimals=None`). Every other type keeps its type.

        Args:
            decimals: The number of decimals that the rounding keeps, or
                `None` where it can carry at every scale.
            mode: The mode of the rounding, as polars writes it.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> decimal = FieldsMapper(DType("Decimal", (5, 2)))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     decimal.carry_dtype(1), decimal.carry_dtype(2)
            (Decimal(precision=6, scale=2), Decimal(precision=5, scale=2))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     decimal.carry_dtype(1, "ToZero")
            Decimal(precision=5, scale=2)
        """
        first = self._dtypes[0]
        if first.name != "Decimal" or mode == "ToZero":
            return first
        precision, scale = Parameter().integer(first, 0), Parameter().integer(first, 1)
        if decimals is not None and scale <= decimals:
            return first
        return DType("Decimal", (min(precision + 1, Decimals.MAX_PRECISION), scale))

    @carry_dtype.before("2.0.0")
    def _no_carry(self, decimals: int | None = 0, mode: str = "HalfToEven") -> DType:
        """Give the type of a rounding in the older releases: the type of the input.

        Args:
            decimals: The number of decimals that the rounding keeps.
            mode: The mode of the rounding.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Decimal", (5, 2)))._no_carry(None)
            Decimal(precision=5, scale=2)
        """
        del decimals, mode
        return self.with_same_dtype()

    @sign_dtype.before("2.0.0")
    def _same_sign(self) -> DType:
        """Give the type of `sign` in the older releases: the type of the input.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Decimal", (3, 3)))._same_sign()
            Decimal(precision=3, scale=3)
        """
        return self.with_same_dtype()

    def _var_inner(self, dtype: DType) -> DType:
        """Refuse a temporal type, and keep any other type.

        Args:
            dtype: The type under one list or array.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The type is temporal.

        Example:
            >>> mapper = FieldsMapper(DType("Int8"))
            >>> mapper._var_inner(DType("Duration", ("us",)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> mapper._var_inner(DType("Int8"))
            Int8
        """
        if Is.temporal(dtype):
            message = f"operation `var` is not supported for `{dtype!r}`"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return dtype

    def moment_dtype(self) -> DType:
        """Give the type of a mean, a median, a standard deviation or a quantile.

        It reads the type under one list or array. A number becomes `Float64`,
        a float stays, and a `Date` becomes a `Datetime` in microseconds. Every
        other type stays.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("List", (DType("Float32"),))).moment_dtype()
            Float32
            >>> FieldsMapper(DType("Date")).moment_dtype()
            Datetime(time_unit='us', time_zone=None)
        """
        return self.map_dtype(lambda dtype: self._moment_inner(self._first_level(dtype)))

    @polars_releases
    def quantile_dtype(self) -> DType:
        """Give the type of `quantile`: that of `moment_dtype`, in a list for a list of quantiles.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> quantiles = DType("List", (DType("Float64"),))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Int8"), quantiles).quantile_dtype()
            List(Float64)
        """
        moment = self.moment_dtype()
        listed = len(self._dtypes) > 1 and self._dtypes[1].name == "List"
        return DType("List", (moment,)) if listed else moment

    @quantile_dtype.before("2.0.0")
    def _moment_only(self) -> DType:
        """Give the type of `quantile` in the older releases: that of `moment_dtype`.

        Returns:
            The type.

        Example:
            >>> quantiles = DType("List", (DType("Float64"),))
            >>> FieldsMapper(DType("Int8"), quantiles)._moment_only()
            Float64
        """
        return self.moment_dtype()

    def _first_level(self, dtype: DType) -> DType:
        """Give the type inside a list or an array, or the type itself.

        Args:
            dtype: The type.

        Returns:
            The inner type of a list or an array, and any other type as it is.

        Example:
            >>> FieldsMapper(DType("Int8"))._first_level(DType("Array", (DType("Int8"), 2)))
            Int8
        """
        return dtype.inner() if dtype.name in Is.LISTS else dtype

    def _moment_inner(self, dtype: DType) -> DType:
        """Give the moment type of one type.

        Args:
            dtype: The type.

        Returns:
            The type of the moment.

        Example:
            >>> FieldsMapper(DType("Int8"))._moment_inner(DType("Decimal", (10, 2)))
            Float64
        """
        if dtype.name in Is.FLOATS:
            return dtype
        if dtype.name in {"Boolean", "Decimal"} or Is.primitive_numeric(dtype):
            return DType("Float64")
        if dtype.name == "Date":
            return DType("Datetime", ("us", None))
        return dtype

    def map_to_float_dtype(self) -> DType:
        """Give the float type of an input, as polars types `sin`.

        Returns:
            `Float16` and `Float32` as they are, and `Float64` for every other type.

        Example:
            >>> FieldsMapper(DType("String")).map_to_float_dtype()
            Float64
        """
        return self.map_dtype(FunctionClosures().pct_change)

    def map_numeric_to_float_dtype(self, *, coerce_decimal: bool) -> DType:
        """Give the float type of a number, and keep every other type.

        A number and a `Boolean` become `Float64`. A narrow float stays. A
        `Decimal` becomes `Float64` only with `coerce_decimal`.

        Args:
            coerce_decimal: Whether a `Decimal` becomes `Float64`.

        Returns:
            The type.

        Example:
            >>> decimal = FieldsMapper(DType("Decimal", (10, 2)))
            >>> decimal.map_numeric_to_float_dtype(coerce_decimal=False)
            Decimal(precision=10, scale=2)
        """
        return self.map_dtype(
            lambda dtype: self._numeric_float(dtype, coerce_decimal=coerce_decimal),
        )

    def _numeric_float(self, dtype: DType, *, coerce_decimal: bool) -> DType:
        """Give the float type of one type, the closure of `map_numeric_to_float_dtype`.

        Args:
            dtype: The type.
            coerce_decimal: Whether a `Decimal` becomes `Float64`.

        Returns:
            The type.

        Example:
            >>> mapper = FieldsMapper(DType("Int8"))
            >>> mapper._numeric_float(DType("Boolean"), coerce_decimal=False)
            Float64
        """
        if dtype.name in self.NARROW_FLOATS:
            return dtype
        if dtype.name == "Decimal":
            return DType("Float64") if coerce_decimal else dtype
        coerce = dtype.name == "Boolean" or Is.primitive_numeric(dtype)
        return DType("Float64") if coerce else dtype

    def sum_dtype(self) -> DType:
        """Give the type of the sum of a window, as `rolling_sum` has it.

        Returns:
            `Int64` for a small integer, `UInt32` for a `Boolean`, the type of
            `_summed_decimal` for a `Decimal`, and the input type for the rest.

        Example:
            >>> FieldsMapper(DType("UInt16")).sum_dtype()
            Int64
            >>> FieldsMapper(DType("Boolean")).sum_dtype()
            UInt32
        """
        first = self._dtypes[0]
        if first.name in Is.SMALL_INTEGERS:
            return DType("Int64")
        if first.name == "Decimal":
            return self._summed_decimal(first)
        return DType.idx() if first.name == "Boolean" else first

    @polars_releases
    def _summed_decimal(self, dtype: DType) -> DType:
        """Give the type of a sum of a `Decimal`: the widest precision at its scale.

        Args:
            dtype: The decimal.

        Returns:
            The decimal of precision 38.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FieldsMapper(DType("Decimal", (10, 2))).sum_dtype()
            Decimal(precision=38, scale=2)
        """
        return Decimals().widest(Parameter().integer(dtype, 1))

    @_summed_decimal.before("2.0.0")
    def _kept_decimal(self, dtype: DType) -> DType:
        """Give the type of a sum of a `Decimal` in the older releases: the decimal itself.

        Args:
            dtype: The decimal.

        Returns:
            The same decimal.

        Example:
            >>> FieldsMapper(DType("Decimal", (10, 2)))._kept_decimal(DType("Decimal", (10, 2)))
            Decimal(precision=10, scale=2)
        """
        return dtype

    def nested_sum_type(self) -> DType:
        """Give the type of the sum of each list or array.

        The inner type maps as `sum_dtype` maps a type. An input that is not
        a list or an array refuses with `PolarsRefusal`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Array", (DType("Int8"), 2))).nested_sum_type()
            Int64
        """
        return FieldsMapper(self._nested_inner()).sum_dtype()

    def nested_mean_median_type(self) -> DType:
        """Give the type of the mean or the median of each list or array.

        A `Date` gives a `Datetime` in microseconds. A temporal type and a
        narrow float stay. Every other type gives `Float64`. An input that is
        not a list or an array refuses with `PolarsRefusal`.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("List", (DType("String"),))).nested_mean_median_type()
            Float64
            >>> FieldsMapper(DType("List", (DType("Duration", ("ms",)),))).nested_mean_median_type()
            Duration(time_unit='ms')
        """
        inner = self._nested_inner()
        if inner.name == "Date":
            return DType("Datetime", ("us", None))
        if Is.temporal(inner) or inner.name in self.NARROW_FLOATS:
            return inner
        return DType("Float64")

    def _nested_inner(self) -> DType:
        """Give the inner type of a list or an array, and refuse another type.

        Returns:
            The inner type.

        Raises:
            PolarsRefusal: The input is not a list or an array.

        Example:
            >>> FieldsMapper(DType("Int8"))._nested_inner()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        first = self._dtypes[0]
        if first.name not in Is.LISTS:
            message = f"expected List or Array type, got dtype: {first!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return first.inner()

    def pow_dtype(self) -> DType:
        """Give the type of `base ** exponent`.

        An integer base takes the type of a float exponent. Every other base
        keeps its type.

        Returns:
            The type.

        Example:
            >>> FieldsMapper(DType("Float32"), DType("Int64")).pow_dtype()
            Float32
        """
        base, exponent = self._dtypes[0], self._dtypes[1]
        if Is.integer(base) and Is.float(exponent):
            return exponent
        return base

    def log_dtype(self) -> DType:
        """Give the type of a logarithm with a base.

        Returns:
            The float type of the input, else the float type of the base, else `Float64`.

        Example:
            >>> FieldsMapper(DType("Int8"), DType("Float32")).log_dtype()
            Float32
        """
        value, base = self._dtypes[0], self._dtypes[1]
        if Is.float(value):
            return value
        return base if Is.float(base) else DType("Float64")


class FunctionClosures:
    """Give the closures that polars passes to the mapper, for the functions that need one.

    Polars types some functions with a closure of its own, for example the
    closure that `map_dtype` takes for `diff`. Each method here is one such
    closure. A method that takes the options of a function gives the closure
    for them. A closure that refuses raises `PolarsRefusal`.

    Example:
        >>> FieldsMapper(DType("Boolean")).map_dtype(FunctionClosures().cum_sum)
        UInt32
    """

    CUM_SUM_KEEPS: ClassVar[frozenset[str]] = frozenset({
        "Int32",
        "Int128",
        "UInt32",
        "UInt64",
        "Float16",
        "Float32",
        "Float64",
    })
    PRODUCT_KEEPS: ClassVar[frozenset[str]] = frozenset({
        "Float16",
        "Float32",
        "Float64",
        "UInt64",
        "Int128",
    })
    BIT_COUNTS: ClassVar[frozenset[str]] = frozenset({
        "count_ones",
        "count_zeros",
        "leading_ones",
        "leading_zeros",
        "trailing_ones",
        "trailing_zeros",
    })
    SIGNED: ClassVar[dict[str, str]] = {
        "UInt8": "Int8",
        "UInt16": "Int16",
        "UInt32": "Int32",
        "UInt64": "Int64",
        "UInt128": "Int128",
    }
    UNSIGNED: ClassVar[dict[str, str]] = {signed: unsigned for unsigned, signed in SIGNED.items()}

    def product(self, dtype: DType) -> DType:
        """Give the type of `product`.

        Args:
            dtype: The input type.

        Returns:
            A float, `UInt64` and `Int128` as they are, a `Decimal` of the
            largest precision, and `Int64` for every other type.

        Example:
            >>> FunctionClosures().product(DType("UInt8"))
            Int64
        """
        if dtype.name in self.PRODUCT_KEEPS:
            return dtype
        if dtype.name == "Decimal":
            return Decimals().widest(Parameter().integer(dtype, 1))
        return DType("Int64")

    def cum_sum(self, dtype: DType) -> DType:
        """Give the type of `cum_sum`.

        A `Decimal` takes the type of `_cum_decimal`. Any other logical type,
        that is, one whose physical type differs, such as a `Date`, stays. A
        literal type sums as the type that it takes alone.

        Args:
            dtype: The input type.

        Returns:
            The type.

        Example:
            >>> FunctionClosures().cum_sum(DType("Int8")), FunctionClosures().cum_sum(DType("Date"))
            (Int64, Date)
        """
        if not dtype.same(dtype.physical()):
            return self._cum_decimal(dtype) if dtype.name == "Decimal" else dtype
        if dtype.name == "Boolean":
            return DType("UInt32")
        if dtype.name in self.CUM_SUM_KEEPS:
            return dtype
        if Is.dynamic_int(dtype):
            return self.cum_sum(dtype.materialized())
        if Is.unknown(dtype):
            return DType("Float64") if Is.float(dtype) else dtype
        return DType("Int64")

    @polars_releases
    def _cum_decimal(self, dtype: DType) -> DType:
        """Give the type of `cum_sum` of a `Decimal`: precision 38, with the scale of the input.

        Args:
            dtype: The decimal type.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     FunctionClosures()._cum_decimal(DType("Decimal", (10, 2)))
            Decimal(precision=38, scale=2)
        """
        return Decimals().widest(Parameter().integer(dtype, 1))

    @_cum_decimal.before("2.0.0")
    def _kept_cum_decimal(self, dtype: DType) -> DType:
        """Give the type of `cum_sum` of a `Decimal` in the older releases: the input type.

        Args:
            dtype: The decimal type.

        Returns:
            The type.

        Example:
            >>> FunctionClosures()._kept_cum_decimal(DType("Decimal", (10, 2)))
            Decimal(precision=10, scale=2)
        """
        return dtype

    def cum_prod(self, dtype: DType) -> DType:
        """Give the type of `cum_prod`.

        Args:
            dtype: The input type.

        Returns:
            A float, `UInt64` and `Int128` as they are, and `Int64` for every other type.

        Example:
            >>> FunctionClosures().cum_prod(DType("Boolean"))
            Int64
        """
        return dtype if dtype.name in self.PRODUCT_KEEPS else DType("Int64")

    def diff(self, dtype: DType) -> DType:
        """Give the type of `diff`.

        Args:
            dtype: The input type.

        Returns:
            A `Duration` for a temporal type, a signed type for an unsigned
            one, a `Decimal` of the largest precision, and the input type otherwise.

        Example:
            >>> FunctionClosures().diff(DType("Date")), FunctionClosures().diff(DType("UInt32"))
            (Duration(time_unit='us'), Int64)
        """
        if dtype.name == "Decimal":
            return Decimals().widest(Parameter().integer(dtype, 1))
        return ListClosures().difference(dtype)

    def pct_change(self, dtype: DType) -> DType:
        """Give the type of `pct_change`, the rule of `map_to_float_dtype`.

        Args:
            dtype: The input type.

        Returns:
            `Float16` and `Float32` as they are, and `Float64` for every other type.

        Example:
            >>> FunctionClosures().pct_change(DType("Float16"))
            Float16
        """
        narrow = dtype.name in FieldsMapper.NARROW_FLOATS
        return dtype if narrow else DType("Float64")

    def repeat_by(self, dtype: DType) -> DType:
        """Give the type of `repeat_by`: a list of the input type.

        Args:
            dtype: The input type.

        Returns:
            The list type.

        Example:
            >>> FunctionClosures().repeat_by(DType("Int8"))
            List(Int8)
        """
        return DType("List", (dtype,))

    def rle(self, dtype: DType) -> DType:
        """Give the type of `rle`: a struct of the run length and the value.

        Args:
            dtype: The input type.

        Returns:
            The struct type.

        Example:
            >>> FunctionClosures().rle(DType("Int8"))
            Struct({'len': UInt32, 'value': Int8})
        """
        return DType("Struct", ((("len", DType.idx()), ("value", dtype)),))

    def value_counts(
        self,
        name: str,
        count_name: str,
        *,
        normalize: bool,
    ) -> Callable[[DType], DType]:
        """Give the closure of `value_counts`.

        Args:
            name: The name of the input, which names the field of the values.
            count_name: The name of the field of the counts.
            normalize: Whether the counts are fractions.

        Returns:
            The closure: a struct of the value and its count.

        Example:
            >>> FunctionClosures().value_counts("x", "count", normalize=True)(DType("Int8"))
            Struct({'x': Int8, 'count': Float64})
        """
        count = DType("Float64") if normalize else DType.idx()
        return lambda dtype: DType("Struct", (((name, dtype), (count_name, count)),))

    def reshape(self, dimensions: Sequence[int]) -> Callable[[DType], DType]:
        """Give the closure of `reshape`.

        The leaf of the input goes into one array for each dimension after
        the first, from the last one. A dimension of `-1` asks polars to
        infer it, which only the first dimension allows: the closure refuses
        it later with `PolarsRefusal`.

        Args:
            dimensions: The dimensions of the call.

        Returns:
            The closure.

        Example:
            >>> FunctionClosures().reshape((-1, 2, 3))(DType("List", (DType("Int8"),)))
            Array(Int8, shape=(2, 3))
        """
        return lambda dtype: self._reshaped(dtype.leaf(), dimensions)

    def _reshaped(self, leaf: DType, dimensions: Sequence[int]) -> DType:
        """Wrap a leaf in one array for each dimension after the first.

        Args:
            leaf: The leaf type.
            dimensions: The dimensions.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A dimension after the first is `-1`.

        Example:
            >>> FunctionClosures()._reshaped(DType("Int8"), (2, -1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        wrapped = leaf
        for dimension in reversed(dimensions[1:]):
            if dimension < 0:
                message = "can only infer the first dimension"
                raise PolarsRefusal(
                    ErrorClass.INVALID_OPERATION,
                    message,
                )
            wrapped = DType("Array", (wrapped, dimension))
        return wrapped

    def corr_cov(self, dtypes: tuple[DType, ...]) -> DType:
        """Give the type of a rolling correlation or covariance.

        Two inputs with no supertype refuse with `PolarsRefusal`.

        Args:
            dtypes: The types of the two inputs.

        Returns:
            The supertype where it is a float, and `Float64` otherwise.

        Example:
            >>> FunctionClosures().corr_cov((DType("Int8"), DType("Int16")))
            Float64
        """
        joined = Supertype().require(dtypes[0], dtypes[1])
        return joined if Is.float(joined) else DType("Float64")

    def sum_horizontal(self, dtype: DType) -> DType:
        """Give the type of `sum_horizontal` from the supertype of its inputs.

        Args:
            dtype: The supertype of the inputs.

        Returns:
            `UInt32` for a `Boolean`, and the supertype otherwise.

        Example:
            >>> FunctionClosures().sum_horizontal(DType("Boolean"))
            UInt32
        """
        return DType.idx() if dtype.name == "Boolean" else dtype

    def mean_horizontal(self, dtype: DType) -> DType:
        """Give the type of `mean_horizontal` from the supertype of its inputs.

        Args:
            dtype: The supertype of the inputs.

        Returns:
            `Float16` and `Float32` as they are, and `Float64` for every other type.

        Example:
            >>> FunctionClosures().mean_horizontal(DType("Int8"))
            Float64
        """
        return self.pct_change(dtype)

    def linear_space(self, dtypes: tuple[DType, ...]) -> DType:
        """Give the type of the values of `linear_space`.

        Args:
            dtypes: The types of the start and the end.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The two types are not numbers, and not one temporal type.

        Example:
            >>> FunctionClosures().linear_space((DType("Date"), DType("Date")))
            Datetime(time_unit='us', time_zone=None)
        """
        start, end = dtypes[0], dtypes[1]
        if start.name in FieldsMapper.NARROW_FLOATS and start.same(end):
            return start
        if Is.temporal(start) and start.same(end):
            return DType("Datetime", ("us", None)) if start.name == "Date" else start
        if not (Is.primitive_numeric(start) and Is.primitive_numeric(end)):
            message = f"'start' and 'end' have incompatible dtypes, got {start!r} and {end!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return DType("Float64")

    def bitwise(self, function: str) -> Callable[[DType], DType]:
        """Give the closure of a bitwise function.

        Args:
            function: The name of the function, for example `count_ones` or `and`.

        Returns:
            The closure: `UInt32` for a count, and the input type for `and`, `or` and `xor`.

        Example:
            >>> FunctionClosures().bitwise("count_ones")(DType("Int8"))
            UInt32
        """
        return lambda dtype: self._bitwise(function, dtype)

    def _bitwise(self, function: str, dtype: DType) -> DType:
        """Type one bitwise function on one type.

        Args:
            function: The name of the function.
            dtype: The input type.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is not a `Boolean` or an integer.

        Example:
            >>> FunctionClosures()._bitwise("xor", DType("Float32"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if not (dtype.name == "Boolean" or Is.integer(dtype)):
            message = f"dtype {dtype!r} not supported in '{function}' operation"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return DType("UInt32") if function in self.BIT_COUNTS else dtype

    def not_(self, dtype: DType) -> DType:
        """Give the type of `not_`.

        Args:
            dtype: The input type.

        Returns:
            `Boolean` for a `Boolean`, and the input type for an integer.

        Raises:
            PolarsRefusal: The input is neither.

        Example:
            >>> FunctionClosures().not_(DType("UInt8"))
            UInt8
        """
        if dtype.name == "Boolean" or Is.integer(dtype):
            return dtype
        message = f"dtype {dtype!r} not supported in 'not' operation"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def numeric(self, index: int, dtype: DType) -> bool:
        """Give the check that `sign` puts on each input: a number.

        Args:
            index: The place of the input, which the check does not read.
            dtype: The type of the input.

        Returns:
            Whether the type is an integer, a float or a `Decimal`, a literal included.

        Example:
            >>> FunctionClosures().numeric(0, DType.dynamic_int(1))
            True
            >>> FieldsMapper(DType("Date")).ensure_satisfies(FunctionClosures().numeric, "sign")
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `sign` operation ...
        """
        del index
        return Is.numeric(dtype)

    def reinterpret(self, signed: bool | None, target: DType | None) -> Callable[[DType], DType]:
        """Give the closure of `reinterpret`, which reads the bits of each value as another number.

        Polars computes the type when it converts the expression, from the
        planner type of the input, and keeps it in the function of the node.
        The input must be a number. With `signed`, an integer takes the
        integer of the same width with a sign, or without a sign where
        `signed` is false, and every other number keeps its type. With
        `target`, the input and the target must be primitive numbers of the
        same width (`BitWidth`). A call gives exactly one of the two.

        Args:
            signed: Whether the integer takes a sign, or `None` where the call gives a target.
            target: The type to read the bits as, or `None` where the call gives `signed`.

        Returns:
            The closure.

        Example:
            >>> FunctionClosures().reinterpret(True, None)(DType("UInt16"))
            Int16
            >>> FunctionClosures().reinterpret(None, DType("Float32"))(DType("UInt32"))
            Float32
        """
        return lambda dtype: self._reinterpret(dtype, signed, target)

    def _reinterpret(self, dtype: DType, signed: bool | None, target: DType | None) -> DType:
        """Type `reinterpret` on one input type.

        Args:
            dtype: The input type.
            signed: Whether the integer takes a sign, or `None`.
            target: The type to read the bits as, or `None`.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The call gives both `signed` and `target`, or
                neither, or the input is not a number, or the target does
                not have the width of the input.

        Example:
            >>> FunctionClosures()._reinterpret(DType("Int8"), False, None)
            UInt8
            >>> FunctionClosures()._reinterpret(DType("Decimal", (10, 2)), True, None)
            Decimal(precision=10, scale=2)
            >>> FunctionClosures()._reinterpret(DType("Int8"), True, DType("UInt8"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: reinterpret requires ...
            >>> FunctionClosures()._reinterpret(DType("String"), True, None)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot reinterpret ...
            >>> FunctionClosures()._reinterpret(DType("Int8"), None, DType("Int16"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot reinterpret ...
        """
        if (signed is None) == (target is None):  # pragma: no cover -- polars refuses it
            message = "reinterpret requires exactly one of `signed` or `dtype` to be specified"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if not Is.numeric(dtype):
            message = f"cannot reinterpret non-numeric input dtype '{dtype!r}'"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if target is None:
            names = self.SIGNED if signed else self.UNSIGNED
            return DType(names[dtype.name]) if dtype.name in names else dtype
        width = BitWidth().of(dtype)
        if width is None or width != BitWidth().of(target):
            message = f"cannot reinterpret from {dtype!r} to {target!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return target

    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None,
        *,
        left_closed: bool,
        include_breaks: bool,
    ) -> Callable[[DType], DType]:
        """Give the closure of `cut`, which reads its options and not its input.

        `cut` puts each value in a bin between two breaks, and gives the label
        of the bin. The type is an `Enum` of the labels, or with
        `include_breaks` a struct of the upper break and the label of each
        value. The labels are the labels of the call, or the intervals of the
        breaks in the order of the call (`CutBins`). Before polars 1.40.0,
        the labels are `Categorical` (`CutBins.dtype`).

        Args:
            breaks: The breaks, in the order of the call.
            labels: The labels of the call, or `None`.
            left_closed: Whether each bin holds its lower break and not its upper break.
            include_breaks: Whether the output holds the upper break of each value.

        Returns:
            The closure. It refuses where the labels do not number one more
            than the breaks, or where two labels are equal.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> closures = FunctionClosures()
            >>> cut = closures.cut([2.0, 0.0], None, left_closed=False, include_breaks=False)
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     cut(DType("Int8"))
            Enum(categories=['(-inf, 2]', '(2, 0]', '(0, inf]'])
        """
        names = None if labels is None else tuple(labels)
        bins = CutBins(tuple(breaks), names, left_closed=left_closed, include_breaks=include_breaks)
        return lambda dtype: bins.dtype()


@dataclass(frozen=True)
class CutBins:
    """Hold the options of one call of `cut`, and give the labels of its bins and their type.

    `cut` puts each value in a bin between two breaks, and gives the label of
    the bin. Polars takes the labels of the call. Otherwise it writes each bin
    as the interval of its two breaks: `(0, 2]`, or `[0, 2)` where each bin
    holds its lower break. The first bin starts at `-inf`, and the last bin
    ends at `inf`. Polars writes a break as Rust writes a float: the shortest
    digits that give the float back, with no exponent and no `.0`.

    The type function writes the intervals of the breaks in the order of the
    call, and the kernel in their sorted order (`sorted`). So with breaks that
    are not sorted, the planner type and the run type are two `Enum` types
    with other categories. Before polars 1.40.0, the type is `Categorical`
    whatever the labels (`dtype`).

    Example:
        >>> CutBins((0.0, 2.0), None, left_closed=False, include_breaks=False).names()
        ('(-inf, 0]', '(0, 2]', '(2, inf]')
    """

    breaks: tuple[float, ...]
    labels: tuple[str, ...] | None
    left_closed: bool
    include_breaks: bool

    def sorted(self) -> CutBins:
        """Give the same call with its breaks in ascending order, as the kernel reads them.

        Returns:
            The call.

        Example:
            >>> CutBins((2.0, 0.0), None, left_closed=True, include_breaks=False).sorted().names()
            ('[-inf, 0)', '[0, 2)', '[2, inf)')
        """
        return CutBins(
            tuple(sorted(self.breaks)),
            self.labels,
            left_closed=self.left_closed,
            include_breaks=self.include_breaks,
        )

    def names(self) -> tuple[str, ...]:
        """Give the label of each bin.

        A break of NaN stands for a break that the tree of the call does not
        hold: NaN, `inf` or `-inf`, which the JSON form of the tree writes
        alike. The interval of such a break is not known, so the rules
        cannot answer. A NaN that the code holds (`KnownNan`) is known, and
        its interval names it `NaN`.

        Returns:
            The labels of the call, or the intervals of the breaks.

        Raises:
            PolarsRefusal: The labels of the call do not number one more than the breaks.
            UndecidedError: The rules do not know the interval of a break.

        Example:
            >>> CutBins((0.0,), ("a",), left_closed=False, include_breaks=False).names()
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: provide len(breaks) + 1 labels
            >>> CutBins((math.nan,), None, left_closed=False, include_breaks=False).names()
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not serialized: a break that is NaN or ...
        """
        unknown = [value for value in self.breaks if math.isnan(value)]
        if self.labels is None and not all(isinstance(value, KnownNan) for value in unknown):
            question = "a break that is NaN or infinite, which the tree writes as null"
            raise UndecidedError(Undecided.NOT_SERIALIZED, question)
        if self.labels is None:
            lows = (-math.inf, *self.breaks)
            highs = (*self.breaks, math.inf)
            return tuple(itertools.starmap(self._interval, zip(lows, highs, strict=True)))
        if len(self.labels) != len(self.breaks) + 1:
            raise PolarsRefusal(ErrorClass.SHAPE, "provide len(breaks) + 1 labels")
        return self.labels

    @polars_releases
    def dtype(self) -> DType:
        """Give the type of the output: an `Enum` of the labels, or a struct of a break and a label.

        Returns:
            The type.

        Raises:
            PolarsRefusal: Two labels are equal, so they do not make an `Enum`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> pair = CutBins((0.0,), ("a", "b"), left_closed=False, include_breaks=True)
            >>> equal = CutBins((0.0,), ("a", "a"), left_closed=False, include_breaks=False)
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     pair.dtype()
            Struct({'breakpoint': Float64, 'category': Enum(categories=['a', 'b'])})
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     equal.dtype()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: the categories of an Enum ...
        """
        names = self.names()
        if len(set(names)) != len(names):
            message = "the categories of an Enum must be unique"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return self._wrapped(DType("Enum", (names,)))

    @dtype.before("1.40.0")
    def _categorical(self) -> DType:
        """Give the type of the output in the older releases: `Categorical`, whatever the labels.

        Returns:
            `Categorical`, or a struct of a break and a `Categorical`.

        Example:
            >>> CutBins((0.0,), ("a", "a"), left_closed=False, include_breaks=True)._categorical()
            Struct({'breakpoint': Float64, 'category': Categorical})
        """
        return self._wrapped(DType("Categorical"))

    def _wrapped(self, categories: DType) -> DType:
        """Give the type of the labels, in a struct with the break where the call includes it.

        Args:
            categories: The type of the labels.

        Returns:
            The type of the output.

        Example:
            >>> CutBins((), None, left_closed=False, include_breaks=False)._wrapped(DType("Int8"))
            Int8
        """
        if not self.include_breaks:
            return categories
        return DType("Struct", ((("breakpoint", DType("Float64")), ("category", categories)),))

    def _interval(self, low: float, high: float) -> str:
        """Give the label of the bin between two breaks.

        Args:
            low: The lower break.
            high: The upper break.

        Returns:
            The interval, closed on the side that holds its break.

        Example:
            >>> CutBins((), None, left_closed=False, include_breaks=False)._interval(-math.inf, 0.5)
            '(-inf, 0.5]'
        """
        if self.left_closed:
            return f"[{self._number(low)}, {self._number(high)})"
        return f"({self._number(low)}, {self._number(high)}]"

    def _number(self, value: float) -> str:
        """Give a float as Rust writes it: the shortest digits that give it back, with no exponent.

        Args:
            value: The float.

        Returns:
            The text.

        Example:
            >>> bins = CutBins((), None, left_closed=False, include_breaks=False)
            >>> bins._number(2.0), bins._number(1e20), bins._number(1e-7), bins._number(-0.0)
            ('2', '100000000000000000000', '0.0000001', '-0')
            >>> bins._number(math.inf), bins._number(math.nan)
            ('inf', 'NaN')
        """
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        text = format(decimal.Decimal(repr(value)), "f")
        return text.rstrip("0").rstrip(".") if "." in text else text


class LengthClosures:
    """Give the types of the functions whose output has a length of its own, from their options.

    Example:
        >>> LengthClosures().hist(include_breakpoint=False, include_category=False)
        UInt32
    """

    def hist(self, *, include_breakpoint: bool, include_category: bool) -> DType:
        """Give the type of `hist`: the count of each bin, in a struct where the call asks for more.

        Args:
            include_breakpoint: Whether the output holds the upper break of each bin.
            include_category: Whether the output holds the category of each bin.

        Returns:
            The index type, or a struct of the parts that the call asks for and the count.

        Example:
            >>> LengthClosures().hist(include_breakpoint=True, include_category=False)
            Struct({'breakpoint': Float64, 'count': UInt32})
            >>> LengthClosures().hist(include_breakpoint=False, include_category=True)
            Struct({'category': Categorical, 'count': UInt32})
        """
        if not (include_breakpoint or include_category):
            return DType.idx()
        fields: list[tuple[str, DType]] = []
        if include_breakpoint:
            fields.append(("breakpoint", DType("Float64")))
        if include_category:
            fields.append(("category", DType("Categorical")))
        fields.append(("count", DType.idx()))
        return DType("Struct", (tuple(fields),))

    def same(self, dtype: DType) -> DType:
        """Give the type of the values, as a kernel that moves or picks them gives it.

        Args:
            dtype: The type of the values.

        Returns:
            The same type.

        Example:
            >>> LengthClosures().same(DType("Date"))
            Date
        """
        return dtype


class WindowClosures:
    """Give the closures and the fixed types of the functions whose type reads one option.

    `rank` and `rolling_rank` read their method, `qcut` reads whether it
    gives its breaks, and `interpolate` reads its method. Polars matches the
    option in the type function of each, and each method here gives the type
    or the closure for one value of the option.

    Example:
        >>> FieldsMapper(DType("Int8")).map_dtype(WindowClosures().interpolate("Linear"))
        Float64
    """

    def rank(self, method: str | None) -> DType:
        """Give the type of `rank` and `rolling_rank` with a method of ranking.

        Args:
            method: The method, as polars serializes it, for example `Average`.

        Returns:
            `Float64` for the average rank, and the index type for the other methods.

        Example:
            >>> WindowClosures().rank("Average"), WindowClosures().rank("Dense")
            (Float64, UInt32)
        """
        return DType("Float64") if method == "Average" else DType.idx()

    def qcut(self, *, include_breaks: bool) -> DType:
        """Give the type of `qcut`: the categories of the global categories, with the breaks or not.

        Args:
            include_breaks: Whether the output holds the upper break of each value.

        Returns:
            `Categorical`, or a struct of the break and the category.

        Example:
            >>> WindowClosures().qcut(include_breaks=True)
            Struct({'breakpoint': Float64, 'category': Categorical})
        """
        category = DType("Categorical")
        if not include_breaks:
            return category
        return DType("Struct", ((("breakpoint", DType("Float64")), ("category", category)),))

    def interpolate(self, method: str | None) -> Callable[[DType], DType]:
        """Give the closure of `interpolate` with a method.

        The linear method gives a float for a number, as
        `map_numeric_to_float_dtype` does without the `Decimal`. The nearest
        method keeps the type.

        Args:
            method: The method, as polars serializes it: `Linear` or `Nearest`.

        Returns:
            The closure.

        Example:
            >>> WindowClosures().interpolate("Linear")(DType("Int8"))
            Float64
            >>> WindowClosures().interpolate("Nearest")(DType("Int8"))
            Int8
        """
        if method != "Linear":
            return lambda dtype: dtype
        return lambda dtype: FieldsMapper(dtype).map_numeric_to_float_dtype(coerce_decimal=False)


class ListClosures:
    """Give the closures of the functions of the `list` namespace.

    The other functions of the namespace use only the methods of the mapper.

    Example:
        >>> ListClosures().diff(DType("List", (DType("Date"),)))
        List(Duration(time_unit='us'))
    """

    DIFFERENCES: ClassVar[dict[str, str]] = {
        "UInt64": "Int64",
        "UInt32": "Int64",
        "UInt16": "Int32",
        "UInt8": "Int16",
    }

    def difference(self, dtype: DType) -> DType:
        """Give the type of a difference of two values of one type.

        Args:
            dtype: The type of the values.

        Returns:
            A `Duration` for a temporal type, a signed type for an unsigned one,
            and the type itself otherwise.

        Example:
            >>> ListClosures().difference(DType("Time"))
            Duration(time_unit='ns')
        """
        if dtype.name == "Datetime":
            return DType("Duration", (Parameter().unit(dtype),))
        if dtype.name in TimeUnit.DIFFERENCE:
            return DType("Duration", (TimeUnit.DIFFERENCE[dtype.name],))
        signed = self.DIFFERENCES.get(dtype.name)
        return dtype if signed is None else DType(signed)

    def diff(self, dtype: DType) -> DType:
        """Give the type of `list.diff`.

        A `Decimal` inner type stays, unlike the `diff` of a column.

        Args:
            dtype: The input type.

        Returns:
            The list type.

        Raises:
            PolarsRefusal: The input is not a list.

        Example:
            >>> ListClosures().diff(DType("List", (DType("UInt8"),)))
            List(Int16)
        """
        if dtype.name != "List":
            message = f"list.diff operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return DType("List", (self.difference(dtype.inner()),))

    def join(self, dtype: DType) -> DType:
        """Give the type of `list.join`.

        Args:
            dtype: The input type.

        Returns:
            `String`.

        Raises:
            PolarsRefusal: The input is not a list of strings.

        Example:
            >>> ListClosures().join(DType("List", (DType("String"),)))
            String
        """
        if dtype.name != "List" or not Is.string(dtype.inner()):
            message = f"attempted list join with non-string dtype: {dtype!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return DType("String")

    def to_array(self, width: int) -> Callable[[DType], DType]:
        """Give the closure of `list.to_array`.

        The closure refuses an input that is not a list with `PolarsRefusal`.

        Args:
            width: The width of the array.

        Returns:
            The closure.

        Example:
            >>> ListClosures().to_array(2)(DType("List", (DType("Int8"),)))
            Array(Int8, shape=(2,))
        """
        return lambda dtype: DType("Array", (self._to_array_inner(dtype), width))

    def to_struct(self, names: Sequence[str]) -> Callable[[DType], DType]:
        """Give the closure of `list.to_struct`.

        Args:
            names: The names of the fields.

        Returns:
            The closure: a struct with one field of the inner type for each name.

        Example:
            >>> ListClosures().to_struct(("a", "b"))(DType("List", (DType("Int8"),)))
            Struct({'a': Int8, 'b': Int8})
        """
        return lambda dtype: StructOf().of(self._to_struct_inner(dtype), names)

    def _to_array_inner(self, dtype: DType) -> DType:
        """Give the inner type of a list for `list.to_array`, and refuse any other type.

        Args:
            dtype: The input type.

        Returns:
            The inner type.

        Raises:
            PolarsRefusal: The input is not a list. Polars raises a `ComputeError`.

        Example:
            >>> ListClosures()._to_array_inner(DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if dtype.name != "List":  # pragma: no cover -- the check refuses any other type
            message = f"expected List dtype, got {dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return dtype.inner()

    def _to_struct_inner(self, dtype: DType) -> DType:
        """Give the inner type of a list for `list.to_struct`, and refuse any other type.

        Args:
            dtype: The input type.

        Returns:
            The inner type.

        Raises:
            PolarsRefusal: The input is not a list. Polars raises an `InvalidOperationError`.

        Example:
            >>> ListClosures()._to_struct_inner(DType("List", (DType("Int8"),)))
            Int8
        """
        if dtype.name != "List":
            message = f"expected List dtype, got {dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return dtype.inner()


class StructOf:
    """Build a struct with one field of one type for each name.

    The functions `list.to_struct`, `arr.to_struct` and `struct.rename_fields`
    over a type that is not a struct all build it.

    Example:
        >>> StructOf().of(DType("Int8"), ("a", "b"))
        Struct({'a': Int8, 'b': Int8})
    """

    def of(self, dtype: DType, names: Sequence[str]) -> DType:
        """Give the struct.

        Args:
            dtype: The type of every field.
            names: The names.

        Returns:
            The struct type.

        Example:
            >>> StructOf().of(DType("Int8"), ())
            Struct({})
        """
        return DType("Struct", (tuple((name, dtype) for name in names),))


class ArrayClosures:
    """Give the closures of the functions of the `arr` namespace.

    The other functions of the namespace use only the methods of the mapper.

    Example:
        >>> ArrayClosures().to_list(DType("Array", (DType("Int8"), 2)))
        List(Int8)
    """

    def to_list(self, dtype: DType) -> DType:
        """Give the type of `arr.to_list`.

        Args:
            dtype: The input type.

        Returns:
            A list of the inner type.

        Raises:
            PolarsRefusal: The input is not an array.

        Example:
            >>> ArrayClosures().to_list(DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if dtype.name != "Array":  # pragma: no cover -- the check refuses any other type
            message = "expected array dtype"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return DType("List", (dtype.inner(),))

    def slice(self, offset: int, length: int) -> Callable[[DType], DType]:
        """Give the closure of `arr.slice`.

        A negative offset or length counts from the end of the array. The width
        of the result is the part of the array that the slice covers.

        Args:
            offset: The offset.
            length: The length.

        Returns:
            The closure: an array of the inner type and the new width.

        Example:
            >>> ArrayClosures().slice(1, 5)(DType("Array", (DType("Int8"), 3)))
            Array(Int8, shape=(2,))
        """
        return lambda dtype: self._sliced(dtype, offset, length)

    def _sliced(self, dtype: DType, offset: int, length: int) -> DType:
        """Give the array type that a slice of an array leaves.

        Args:
            dtype: The input type.
            offset: The offset.
            length: The length.

        Returns:
            The array type.

        Raises:
            PolarsRefusal: The input is not an array.

        Example:
            >>> ArrayClosures()._sliced(DType("Array", (DType("Int8"), 3)), -2, -1)
            Array(Int8, shape=(2,))
            >>> ArrayClosures()._sliced(DType("Int8"), 0, 1)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: expected array dtype, ...
        """
        if dtype.name != "Array":  # pragma: no cover -- the caller checks the array first
            message = f"expected array dtype, got {dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        width = Parameter().integer(dtype, 1)
        covered = max(width + length, 0) if length < 0 else length
        start = offset + width if offset < 0 else offset
        clamped_start = min(max(start, 0), width)
        clamped_stop = min(max(start + covered, 0), width)
        return DType("Array", (dtype.inner(), clamped_stop - clamped_start))

    def to_struct(self, names: Sequence[str] | None) -> Callable[[DType], DType]:
        """Give the closure of `arr.to_struct`.

        Without names, the fields are `field_0`, `field_1` and on.

        Args:
            names: The names of the fields, one for each place of the array, or `None`.

        Returns:
            The closure: a struct with one field of the inner type for each place.

        Example:
            >>> ArrayClosures().to_struct(None)(DType("Array", (DType("Int8"), 2)))
            Struct({'field_0': Int8, 'field_1': Int8})
        """
        return lambda dtype: self._places(dtype, names)

    def _places(self, dtype: DType, names: Sequence[str] | None) -> DType:
        """Give the struct of the places of an array.

        Args:
            dtype: The array type.
            names: The names, or `None` for the default names.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: The input is not an array, or the number of names is
                not the width of the array.

        Example:
            >>> ArrayClosures()._places(DType("Array", (DType("Int8"), 1)), ("a",))
            Struct({'a': Int8})
            >>> ArrayClosures()._places(DType("Int8"), ("a",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected Array ...
            >>> ArrayClosures()._places(DType("Array", (DType("Int8"), 2)), ("a",))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: arr.to_struct() number of ...
        """
        if dtype.name != "Array":  # pragma: no cover -- the check refuses any other type
            message = f"expected Array type, got: {dtype!r}"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        width = Parameter().integer(dtype, 1)
        if names is not None and len(names) != width:
            message = (
                f"arr.to_struct() number of field names provided ({len(names)}) "
                f"does not match width of input ({width})"
            )
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        chosen = names if names is not None else [f"field_{index}" for index in range(width)]
        return StructOf().of(dtype.inner(), chosen)

    def concat(self, dtypes: tuple[DType, ...]) -> DType:
        """Give the type of `concat_arr`, of all its inputs.

        Each input is an array of a width, or a value of any other type, which
        counts as an array of width 1. The inner type of each input must be
        the inner type of the first input. The output is an array of that
        inner type, as wide as all the inputs together.

        Args:
            dtypes: The types of the inputs.

        Returns:
            The array type.

        Raises:
            PolarsRefusal: An input has another inner type. Polars raises a `SchemaError`.

        Example:
            >>> pair = DType("Array", (DType("Int8"), 2))
            >>> ArrayClosures().concat((pair, DType("Int8")))
            Array(Int8, shape=(3,))
            >>> ArrayClosures().concat((pair, DType("Int16")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        inner, width = self._piece(dtypes[0])
        for dtype in dtypes[1:]:
            other, extra = self._piece(dtype)
            if not other.same(inner):
                message = f"concat_arr dtype mismatch: expected {inner!r}, got {dtype!r}"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
            width += extra
        return DType("Array", (inner, width))

    def _piece(self, dtype: DType) -> tuple[DType, int]:
        """Give the inner type and the width of one input of `concat_arr`.

        Args:
            dtype: The type of the input.

        Returns:
            The inner type and the width of an array, or the type and 1 for any other type.

        Example:
            >>> ArrayClosures()._piece(DType("Date"))
            (Date, 1)
        """
        if dtype.name == "Array":
            return dtype.inner(), Parameter().integer(dtype, 1)
        return dtype, 1


@dataclass(frozen=True)
class ArrayIndexBounds:
    """Refuse an index of `arr.get` that is a literal outside the width of the arrays.

    The planner of polars checks one input of `arr.get` before its type
    function: where the arrays have a fixed width and the index is an
    integer literal node with no data type, the index must lie inside that
    width. A negative index counts from the end. The check applies only where
    an index outside an array raises, that is, where `null_on_oob` is off. An
    index that the planner computes from other literals, such as
    `pl.lit(2) + pl.lit(0)`, is not a literal node, so the check skips it.

    Example:
        >>> array = Operand.column(DType("Array", (DType("Int8"), 2)))
        >>> ArrayIndexBounds(null_on_oob=False).check([array, Operand.of_value(-2)])
        >>> ArrayIndexBounds(null_on_oob=False).check([array, Operand.of_value(2)])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    null_on_oob: bool

    def check(self, inputs: Sequence[Operand]) -> None:
        """Refuse an index outside the width.

        Args:
            inputs: The arrays and the index.

        Raises:
            PolarsRefusal: The index is a literal outside the width.

        Example:
            >>> array = Operand.column(DType("Array", (DType("Int8"), 2)))
            >>> ArrayIndexBounds(null_on_oob=True).check([array, Operand.of_value(5)])
        """
        array, index = inputs[0], inputs[1]
        if self.null_on_oob or array.dtype.name != "Array" or not index.literal:
            return
        if not Is.dynamic_int(index.dtype) or not isinstance(index.value, int):
            return
        width = Parameter().integer(array.dtype, 1)
        if not -width <= index.value < width:
            message = f"get index {index.value} is out of bounds for array of width {width}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class TemporalClosures:
    """Give the closures of the functions of the `dt` namespace.

    The other functions of the namespace use only the methods of the mapper.

    Example:
        >>> TemporalClosures().with_time_unit("ns")(DType("Duration", ("us",)))
        Duration(time_unit='ns')
    """

    def with_time_unit(self, time_unit: str) -> Callable[[DType], DType]:
        """Give the closure of `with_time_unit` and `cast_time_unit`.

        Args:
            time_unit: The new unit.

        Returns:
            The closure: a `Duration` or a `Datetime` with the new unit.

        Example:
            >>> TemporalClosures().with_time_unit("ms")(DType("Datetime", ("us", "UTC")))
            Datetime(time_unit='ms', time_zone='UTC')
        """
        return lambda dtype: self._with_unit(dtype, time_unit)

    def _with_unit(self, dtype: DType, time_unit: str) -> DType:
        """Give a `Duration` or a `Datetime` with a new unit.

        Args:
            dtype: The input type.
            time_unit: The new unit.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is neither.

        Example:
            >>> TemporalClosures()._with_unit(DType("Date"), "ms")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if dtype.name == "Duration":
            return DType("Duration", (time_unit,))
        if dtype.name == "Datetime":
            return DType("Datetime", (time_unit, Parameter().text(dtype, 1)))
        message = f"expected duration or datetime, got {dtype!r}"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def convert_time_zone(self, time_zone: str) -> Callable[[DType], DType]:
        """Give the closure of `convert_time_zone`.

        The closure refuses a type that is not a `Datetime` with `PolarsRefusal`.

        Args:
            time_zone: The new zone.

        Returns:
            The closure: the `Datetime` with the new zone.

        Example:
            >>> TemporalClosures().convert_time_zone("UTC")(DType("Datetime", ("ms", None)))
            Datetime(time_unit='ms', time_zone='UTC')
        """
        return lambda dtype: DType("Datetime", (self._datetime_unit(dtype), time_zone))

    def naive(self, dtype: DType) -> DType:
        """Give the type of `dt.datetime`: the `Datetime` with no zone.

        A type that is not a `Datetime` refuses with `PolarsRefusal`.

        Args:
            dtype: The input type.

        Returns:
            The type.

        Example:
            >>> TemporalClosures().naive(DType("Datetime", ("ns", "UTC")))
            Datetime(time_unit='ns', time_zone=None)
        """
        return DType("Datetime", (self._datetime_unit(dtype), None))

    def _datetime_unit(self, dtype: DType) -> str:
        """Give the unit of a `Datetime`, and refuse another type.

        Args:
            dtype: The input type.

        Returns:
            The unit.

        Raises:
            PolarsRefusal: The input is not a `Datetime`.

        Example:
            >>> TemporalClosures()._datetime_unit(DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if dtype.name != "Datetime":
            message = f"expected Datetime, got {dtype!r}"
            raise PolarsRefusal(
                ErrorClass.COMPUTE,
                message,
            )
        return Parameter().unit(dtype)

    def combine(self, time_unit: str) -> Callable[[DType], DType]:
        """Give the closure of `dt.combine`.

        Args:
            time_unit: The unit of the result.

        Returns:
            The closure: a `Datetime` in the unit, with the zone of a `Datetime` input.

        Example:
            >>> TemporalClosures().combine("ms")(DType("Date"))
            Datetime(time_unit='ms', time_zone=None)
        """
        return lambda dtype: self._combined(dtype, time_unit)

    def _combined(self, dtype: DType, time_unit: str) -> DType:
        """Give the `Datetime` that `dt.combine` gives for one input type.

        Args:
            dtype: The input type.
            time_unit: The unit of the result.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is neither a `Date` nor a `Datetime`.

        Example:
            >>> TemporalClosures()._combined(DType("Time"), "ms")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if dtype.name == "Datetime":
            return DType("Datetime", (time_unit, Parameter().text(dtype, 1)))
        if dtype.name == "Date":
            return DType("Datetime", (time_unit, None))
        message = f"expected Date or Datetime, got {dtype!r}"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def total(self, fractional: bool) -> DType:
        """Give the type of `total_days` and the other totals of a `Duration`.

        Args:
            fractional: Whether the count keeps its fraction.

        Returns:
            `Float64` for a count with its fraction, and `Int64` otherwise.

        Example:
            >>> TemporalClosures().total(True), TemporalClosures().total(False)
            (Float64, Int64)
        """
        return DType("Float64" if fractional else "Int64")


class StringClosures:
    """Give the type rules of the functions of the `str` namespace that read an option of a node.

    Example:
        >>> StringClosures().strptime(DType("Date"), "%Y-%m-%d")
        Date
    """

    ZONE_AWARE: ClassVar[re.Pattern[str]] = re.compile(r"(%z)|(%:z)|(%::z)|(%:::z)|(%#z)|(^%\+$)")

    def strptime(self, target: DType, pattern: str | None) -> DType:
        """Give the type of `str.strptime`: its target, in the zone `UTC` for a format with a zone.

        A format that reads the offset of a zone, for example `%z`, gives a
        moment in `UTC`. So a `Datetime` target with no zone takes the zone
        `UTC`. Every other target stays as it is.

        Args:
            target: The target type: a `Date`, a `Datetime` or a `Time`.
            pattern: The format, or `None` where polars infers it from the text.

        Returns:
            The type.

        Example:
            >>> StringClosures().strptime(DType("Datetime", ("us", None)), "%Y-%m-%d %z")
            Datetime(time_unit='us', time_zone='UTC')
            >>> StringClosures().strptime(DType("Datetime", ("ms", None)), None)
            Datetime(time_unit='ms', time_zone=None)
        """
        aware = pattern is not None and self.ZONE_AWARE.search(pattern) is not None
        if target.name != "Datetime" or not aware or Parameter().text(target, 1) is not None:
            return target
        return DType("Datetime", (Parameter().unit(target), "UTC"))

    def to_integer(self, target: DType | None) -> DType:
        """Give the type of `str.to_integer`: its target, or `Int64` where the node names none.

        Args:
            target: The target type of the node, or `None`.

        Returns:
            The type.

        Example:
            >>> StringClosures().to_integer(None), StringClosures().to_integer(DType("UInt8"))
            (Int64, UInt8)
        """
        return DType("Int64") if target is None else target

    def to_decimal(self, scale: int) -> DType:
        """Give the type of `str.to_decimal`: a `Decimal` of the largest precision, at the scale.

        Args:
            scale: The scale of the node.

        Returns:
            The type.

        Example:
            >>> StringClosures().to_decimal(2)
            Decimal(precision=38, scale=2)
        """
        return Decimals().widest(scale)

    def split_exact(self, count: int) -> DType:
        """Give the type of `str.split_exact`: a struct of `count + 1` text fields.

        Args:
            count: The number of splits of the node.

        Returns:
            The struct, with the fields `field_0`, `field_1` and so on.

        Example:
            >>> StringClosures().split_exact(1)
            Struct({'field_0': String, 'field_1': String})
        """
        return self.splitn(count + 1)

    def splitn(self, count: int) -> DType:
        """Give the type of `str.splitn`: a struct of `count` text fields.

        Args:
            count: The number of parts of the node.

        Returns:
            The struct, with the fields `field_0`, `field_1` and so on.

        Example:
            >>> StringClosures().splitn(1)
            Struct({'field_0': String})
        """
        fields = tuple((f"field_{index}", DType("String")) for index in range(count))
        return DType("Struct", (fields,))


class BinaryClosures:
    """Give the type rules of the functions of the `bin` namespace that read an option of a node.

    Example:
        >>> BinaryClosures().reinterpret(DType("UInt8"))(DType("Binary"))
        UInt8
    """

    def reinterpret(self, target: DType) -> Callable[[DType], DType]:
        """Give the closure of `bin.reinterpret`, which gives its target.

        Polars accepts a target that is a number, a temporal type, or an
        `Array` of one of them. It refuses any other target when it converts
        the expression, before it types the node.

        Args:
            target: The target type of the node.

        Returns:
            The closure, which reads no input.

        Example:
            >>> BinaryClosures().reinterpret(DType("String"))(DType("Binary"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot reinterpret ...
        """

        def reinterpret(dtype: DType) -> DType:
            del dtype
            inner = target.inner() if target.name == "Array" else target
            if not Is.primitive_numeric(inner) and not Is.temporal(inner):
                message = f"cannot reinterpret binary to dtype {target!r}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            return target

        return reinterpret


class CategoricalClosures:
    """Give the type rules of the functions of the `cat` namespace.

    Example:
        >>> CategoricalClosures().physical(DType("Categorical"))
        UInt32
    """

    def physical(self, dtype: DType) -> DType:
        """Give the type of `cat.physical`: the physical type of a `Categorical` or an `Enum`.

        Args:
            dtype: The type of the input.

        Returns:
            The type of the ids of the categories.

        Raises:
            PolarsRefusal: The input has no categories.

        Example:
            >>> CategoricalClosures().physical(DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid dtype: ...
        """
        if dtype.name not in Is.CATEGORIES:
            message = f"invalid dtype: expected an Enum or Categorical type, received '{dtype!r}'"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return dtype.physical()


class FieldSelection:
    """Give the names of the fields of a struct that a selector of polars picks.

    `struct.unnest()` and `struct.field` with several names build one node
    that holds a selector over the fields of its struct input. When polars
    converts the expression, it expands that node into one `struct.field` for
    each field that the selector picks. A selector is a tree: a leaf picks
    fields by name, by place, by a regular expression, or picks all or none,
    and a branch joins two selectors as sets. The result keeps the order of
    the fields of the struct, except for a leaf of names or of places, which
    keeps its own order. A strict leaf refuses a name or a place that the
    struct does not hold. A leaf of data types needs the selectors of data
    types, which this class does not translate.

    Example:
        >>> unnest = {"Union": [{"Union": ["Empty", "Wildcard"]}, {"ByName": {"names": []}}]}
        >>> FieldSelection(unnest).names(("a", "b"))
        ('a', 'b')
        >>> FieldSelection({"ByName": {"names": ["b", "a"], "strict": True}}).names(("a", "b"))
        ('b', 'a')
    """

    SETS: ClassVar[frozenset[str]] = frozenset({
        "Union",
        "Difference",
        "ExclusiveOr",
        "Intersect",
    })

    def __init__(self, selector: object) -> None:
        """Keep the selector, as `json.loads` gives the serialized tree of polars.

        Args:
            selector: The selector.

        Example:
            >>> FieldSelection("Wildcard")._selector
            'Wildcard'
        """
        self._selector = selector

    def __repr__(self) -> str:
        """Give the selector.

        Returns:
            The text.

        Example:
            >>> FieldSelection("Empty")
            FieldSelection('Empty')
        """
        return f"FieldSelection({self._selector!r})"

    def names(self, fields: Sequence[str]) -> tuple[str, ...]:
        """Give the names that the selector picks among the fields of a struct.

        It raises `PolarsRefusal` where a strict leaf names a field that the
        struct does not hold.

        Args:
            fields: The names of the fields, in order.

        Returns:
            The picked names.

        Raises:
            UndecidedError: The selector picks fields by their data types.

        Example:
            >>> picked = {"Difference": ["Wildcard", {"ByName": {"names": ["a"]}}]}
            >>> FieldSelection(picked).names(("a", "b", "c"))
            ('b', 'c')
            >>> FieldSelection({"ByDType": {}}).names(("a",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not translated: a selector of fields ByDType
        """
        kind, part = self._variant(self._selector)
        if kind in self.SETS:
            return self._joined(kind, part, fields)
        leaves: dict[str, Callable[[object, Sequence[str]], tuple[str, ...]]] = {
            "Empty": self._empty,
            "Wildcard": self._wildcard,
            "ByName": self._by_name,
            "ByIndex": self._by_index,
            "Matches": self._matches,
        }
        if kind not in leaves:  # pragma: no cover -- polars selects fields by names
            raise UndecidedError(Undecided.NOT_TRANSLATED, f"a selector of fields {kind}")
        return leaves[kind](part, fields)

    def _empty(self, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Pick no field.

        Args:
            part: What the leaf holds, which is nothing.
            fields: The names of the fields.

        Returns:
            No name.

        Example:
            >>> FieldSelection("Empty")._empty(None, ("a",))
            ()
        """
        del part, fields
        return ()

    def _wildcard(self, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Pick every field.

        Args:
            part: What the leaf holds, which is nothing.
            fields: The names of the fields.

        Returns:
            Every name.

        Example:
            >>> FieldSelection("Wildcard")._wildcard(None, ("a", "b"))
            ('a', 'b')
        """
        del part
        return tuple(fields)

    def _matches(self, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Pick the fields whose name a regular expression matches at any place.

        Args:
            part: What the leaf holds: the regular expression.
            fields: The names of the fields.

        Returns:
            The picked names.

        Example:
            >>> FieldSelection("Empty")._matches("^b", ("a", "b", "bc"))
            ('b', 'bc')
        """
        return tuple(name for name in fields if re.search(str(part), name))

    def _variant(self, selector: object) -> tuple[str, object]:
        """Give the name of a variant of the selector and what it holds.

        Args:
            selector: The variant, as a name or as a map of its name to what it holds.

        Returns:
            The name and what it holds, or `None` for a variant that holds nothing.

        Example:
            >>> FieldSelection("Empty")._variant({"Matches": "^a$"})
            ('Matches', '^a$')
        """
        if isinstance(selector, dict) and len(selector) == 1:
            name, part = next(iter(selector.items()))
            return str(name), part
        return str(selector), None

    def _joined(self, kind: str, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Join the picks of two selectors as sets, in the order of the fields.

        Args:
            kind: The join: `Union`, `Difference`, `ExclusiveOr` or `Intersect`.
            part: The two selectors.
            fields: The names of the fields, in order.

        Returns:
            The picked names.

        Example:
            >>> join = ["Wildcard", {"ByName": {"names": ["b"]}}]
            >>> FieldSelection("Empty")._joined("Intersect", join, ("a", "b"))
            ('b',)
        """
        left, right = (FieldSelection(side).names(fields) for side in self._sides(part))
        kept = {
            "Union": set(left) | set(right),
            "Difference": set(left) - set(right),
            "ExclusiveOr": set(left) ^ set(right),
            "Intersect": set(left) & set(right),
        }
        return tuple(name for name in fields if name in kept[kind])

    def _sides(self, part: object) -> tuple[object, object]:
        """Give the two selectors of a join.

        Args:
            part: What the join holds: a list of two selectors.

        Returns:
            The two selectors.

        Raises:
            TypeError: The join does not hold two selectors.

        Example:
            >>> FieldSelection("Empty")._sides(["Empty", "Wildcard"])
            ('Empty', 'Wildcard')
            >>> FieldSelection("Empty")._sides(["Empty"])
            Traceback (most recent call last):
            ...
            TypeError: no two selectors in ['Empty']
        """
        if (  # pragma: no cover -- polars writes two selectors
            not isinstance(part, list) or len(part) != 2
        ):
            message = f"no two selectors in {part!r}"
            raise TypeError(message)
        return part[0], part[1]

    def _by_name(self, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Pick fields by name, in the order of the names.

        Args:
            part: What the leaf holds: the names, and whether it is strict.
            fields: The names of the fields.

        Returns:
            The picked names.

        Raises:
            PolarsRefusal: The leaf is strict, and a name is not a field.

        Example:
            >>> leaf = {"names": ["c"], "strict": True}
            >>> FieldSelection("Empty")._by_name(leaf, ("a",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ColumnNotFoundError: ...
            >>> FieldSelection("Empty")._by_name({"names": ["c", "a"], "strict": False}, ("a",))
            ('a',)
        """
        leaf = part if isinstance(part, dict) else {}
        picked: list[str] = []
        for name in leaf.get("names", []):
            if name in fields:
                picked.append(str(name))
            elif leaf.get("strict", True):
                raise PolarsRefusal(ErrorClass.COLUMN_NOT_FOUND, str(name))
        return tuple(dict.fromkeys(picked))

    def _by_index(self, part: object, fields: Sequence[str]) -> tuple[str, ...]:
        """Pick fields by place, in the order of the places, where a negative place counts back.

        Args:
            part: What the leaf holds: the places, and whether it is strict.
            fields: The names of the fields.

        Returns:
            The picked names.

        Raises:
            PolarsRefusal: The leaf is strict and a place is outside the fields,
                or the leaf names one place twice.

        Example:
            >>> FieldSelection("Empty")._by_index({"indices": [-1], "strict": True}, ("a", "b"))
            ('b',)
            >>> FieldSelection("Empty")._by_index({"indices": [5, 0], "strict": False}, ("a", "b"))
            ('a',)
            >>> FieldSelection("Empty")._by_index({"indices": [0, -2]}, ("a", "b"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: a
        """
        leaf = part if isinstance(part, dict) else {}
        strict = bool(leaf.get("strict", True))
        picked: list[str] = []
        for index in leaf.get("indices", []):
            place = self._place(index, len(fields), strict=strict)
            if place is None:
                continue
            if fields[place] in picked:
                raise PolarsRefusal(ErrorClass.DUPLICATE, fields[place])
            picked.append(fields[place])
        return tuple(picked)

    def _place(self, index: int, count: int, *, strict: bool) -> int | None:
        """Give the place of a field from its index, where a negative index counts back.

        Args:
            index: The index of the leaf.
            count: The number of fields.
            strict: Whether an index outside the fields raises.

        Returns:
            The place, or `None` for an index outside the fields that is not strict.

        Raises:
            PolarsRefusal: The index is outside the fields, and the leaf is strict.

        Example:
            >>> FieldSelection("Empty")._place(-1, 2, strict=True)
            1
            >>> FieldSelection("Empty")._place(5, 2, strict=False) is None
            True
            >>> FieldSelection("Empty")._place(5, 2, strict=True)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ColumnNotFoundError: no field at 5
        """
        place = index + count if index < 0 else index
        if 0 <= place < count:
            return place
        if strict:
            raise PolarsRefusal(ErrorClass.COLUMN_NOT_FOUND, f"no field at {index}")
        return None


class StructClosures:
    """Give the closures of the functions of the `struct` namespace.

    Example:
        >>> StructClosures().prefix_fields("p_")(DType("Struct", ((("a", DType("Int8")),),)))
        Struct({'p_a': Int8})
    """

    def field_by_name(self, name: str) -> Callable[[DType], DType]:
        """Give the closure of `struct.field`.

        The closure refuses a missing field, or a type that is not a struct.

        Args:
            name: The name of the field.

        Returns:
            The closure: the type of the field.

        Example:
            >>> StructClosures().field_by_name("a")(DType("Struct", ((("a", DType("Int8")),),)))
            Int8
        """
        return lambda dtype: self._field(dtype, name)

    def _field(self, dtype: DType, name: str) -> DType:
        """Give the type of one field of a struct, and refuse a missing field or another type.

        Args:
            dtype: The input type.
            name: The name of the field.

        Returns:
            The type of the field.

        Raises:
            PolarsRefusal: The input is not a struct, or it has no such field.

        Example:
            >>> StructClosures()._field(DType("Int8"), "a")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: StructFieldNotFoundError: ...
        """
        fields = dict(dtype.fields()) if dtype.name == "Struct" else {}
        if name not in fields:
            message = name
            raise PolarsRefusal(
                ErrorClass.STRUCT_FIELD_NOT_FOUND,
                message,
            )
        return fields[name]

    def rename_fields(self, names: Sequence[str]) -> Callable[[DType], DType]:
        """Give the closure of `struct.rename_fields` (`_renamed`).

        Args:
            names: The new names.

        Returns:
            The closure.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> StructClosures().rename_fields(("x", "y"))(pair)
            Struct({'x': Int8, 'y': Date})
        """
        return lambda dtype: self._renamed(dtype, names)

    @polars_releases
    def _renamed(self, dtype: DType, names: Sequence[str]) -> DType:
        """Give a struct with new names for its fields, in order.

        The input must be a struct, with as many fields as there are names.

        Args:
            dtype: The input type.
            names: The new names.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: The input is not a struct, or the number of names
                is not the number of fields.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     StructClosures()._renamed(pair, ("x",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: struct.rename_fields() ...
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     StructClosures()._renamed(pair, ("x", "y"))
            Struct({'x': Int8, 'y': Date})
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     StructClosures()._renamed(DType("Int8"), ("x",))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `rename_fields` ...
        """
        if dtype.name != "Struct":
            message = f"`rename_fields` operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        fields = dtype.fields()
        if len(names) != len(fields):
            message = (
                "struct.rename_fields() argument has a different number of fields than the "
                f"struct it operates on ({len(names)} vs {len(fields)})."
            )
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        pairs = zip(fields, names, strict=True)
        return DType("Struct", (tuple((name, field[1]) for field, name in pairs),))

    @_renamed.before("2.0.0")
    def _renamed_loosely(self, dtype: DType, names: Sequence[str]) -> DType:
        """Give a struct with new names for its fields, as the older releases do.

        A struct keeps as many fields as there are names. Another type gives
        a struct with one field of that type for each name.

        Args:
            dtype: The input type.
            names: The new names.

        Returns:
            The struct type.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> StructClosures()._renamed_loosely(pair, ("x",))
            Struct({'x': Int8})
            >>> StructClosures()._renamed_loosely(DType("Int8"), ("x", "y"))
            Struct({'x': Int8, 'y': Int8})
        """
        if dtype.name != "Struct":
            return StructOf().of(dtype, names)
        pairs = zip(dtype.fields(), names, strict=False)
        return DType("Struct", (tuple((name, field[1]) for field, name in pairs),))

    def prefix_fields(self, prefix: str) -> Callable[[DType], DType]:
        """Give the closure of `struct.prefix_fields`.

        The closure refuses a type that is not a struct with `PolarsRefusal`.

        Args:
            prefix: The text before each name.

        Returns:
            The closure.

        Example:
            >>> StructClosures().prefix_fields("p_")(DType("Struct", ((),)))
            Struct({})
        """
        return lambda dtype: self._named(dtype, "prefix_fields", lambda name: f"{prefix}{name}")

    def suffix_fields(self, suffix: str) -> Callable[[DType], DType]:
        """Give the closure of `struct.suffix_fields`.

        The closure refuses a type that is not a struct with `PolarsRefusal`.

        Args:
            suffix: The text after each name.

        Returns:
            The closure.

        Example:
            >>> StructClosures().suffix_fields("_s")(DType("Struct", ((("a", DType("Int8")),),)))
            Struct({'a_s': Int8})
        """
        return lambda dtype: self._named(dtype, "suffix_fields", lambda name: f"{name}{suffix}")

    def _named(self, dtype: DType, op: str, rename: Callable[[str], str]) -> DType:
        """Give a struct with each name of a field changed.

        Args:
            dtype: The input type.
            op: The name of the function, for the message.
            rename: The change of one name.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: The input is not a struct.

        Example:
            >>> StructClosures()._named(DType("Int8"), "prefix_fields", str.upper)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if dtype.name != "Struct":
            message = f"{op} operation not supported for dtype `{dtype!r}` (expected: Struct)"
            raise PolarsRefusal(
                ErrorClass.INVALID_OPERATION,
                message,
            )
        return DType("Struct", (tuple((rename(name), field) for name, field in dtype.fields()),))

    def select_fields(self, selection: FieldSelection) -> Callable[[DType], DType]:
        """Give the closure of `struct.unnest`, and of `struct.field` with several names.

        Polars expands the node into one `struct.field` for each field that
        the selector picks (`FieldSelection`), before it types the node. An
        input that is not a struct refuses the plan. One picked field gives a
        node of the type of that field. Two fields or more give two columns
        or more. They all take the one output name of the expression, so the
        frame refuses them as duplicates. A selector that picks no field gives
        no column, which the rules do not follow.

        Args:
            selection: The selector of the node.

        Returns:
            The closure.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Date"))),))
            >>> StructClosures().select_fields(FieldSelection({"ByName": {"names": ["b"]}}))(pair)
            Date
            >>> StructClosures().select_fields(FieldSelection("Wildcard"))(pair)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: ...
        """
        return lambda dtype: self._selected(dtype, selection)

    def _selected(self, dtype: DType, selection: FieldSelection) -> DType:
        """Give the type of the one field that a selector picks in a struct.

        Args:
            dtype: The input type.
            selection: The selector.

        Returns:
            The type of the field.

        Raises:
            PolarsRefusal: The input is not a struct, or the selector picks two
                fields or more.
            UndecidedError: The selector picks no field.

        Example:
            >>> StructClosures()._selected(DType("Int8"), FieldSelection("Wildcard"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if dtype.name != "Struct":
            message = f"struct.field operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        fields = dict(dtype.fields())
        names = selection.names(tuple(fields))
        if not names:
            raise UndecidedError(Undecided.NOT_TRANSLATED, "a struct.unnest that gives no column")
        if len(names) > 1:
            message = f"the fields {', '.join(names)} give columns of one name"
            raise PolarsRefusal(ErrorClass.DUPLICATE, message)
        return fields[names[0]]


class ListArithmeticCheck:
    """Refuse list arithmetic on two lists whose inner types it does not take.

    Arithmetic on two lists needs inner types that list arithmetic takes
    (`Is.list_arithmetic_input`): numbers, `Boolean` or `Null`. The planner
    refuses the other pairs of lists, for every arithmetic operator.

    Example:
        >>> listed = DType("List", (DType("Int8"),))
        >>> ListArithmeticCheck().refuse(Operator.PLUS, listed, listed) is None
        True
    """

    def refuse(self, op: Operator, left: DType, right: DType) -> None:
        """Refuse two lists of an inner type that list arithmetic does not take.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Raises:
            PolarsRefusal: Both inputs are lists, and one inner type is not taken.

        Example:
            >>> listed = DType("List", (DType("String"),))
            >>> ListArithmeticCheck().refuse(Operator.PLUS, listed, listed)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if left.name != "List" or right.name != "List":
            return
        inners = (left.inner(), right.inner())
        if all(Is.list_arithmetic_input(inner) for inner in inners):
            return
        message = f"cannot {op.value} two list columns with non-numeric inner types"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class ArithmeticField:
    """Give the planner type of an arithmetic or bitwise operator.

    It has three groups of arms: `-`, `+`, and every other operator. A struct
    beside a number keeps the struct. A temporal type meets only the types that
    its arithmetic allows. A list or an array joins its leaf with the other
    input. Two decimals give the largest precision. The rest is the supertype.
    One arm reads the nodes: in `*`, `//` and the other operators of the last
    group, a literal with no data type on the left takes the type of an integer
    or a float on the right. A pair that the arms refuse raises
    `PolarsRefusal`.

    Example:
        >>> field = ArithmeticField()
        >>> field.of(Operator.MINUS, Operand.column(DType("Date")), Operand.column(DType("Date")))
        Duration(time_unit='us')
        >>> field.of(Operator.MULTIPLY, Operand.of_value(300), Operand.column(DType("Int8")))
        Int8
    """

    MOMENT_PAIRS: ClassVar[frozenset[tuple[str, str]]] = frozenset({
        ("Duration", "Datetime"),
        ("Datetime", "Duration"),
        ("Duration", "Date"),
        ("Date", "Duration"),
        ("Duration", "Time"),
        ("Time", "Duration"),
    })
    DIFFERENCE_ORDER: ClassVar[tuple[str, ...]] = ("Datetime", "Date", "Duration", "Time")

    def of(self, op: Operator, left: Operand, right: Operand) -> DType:
        """Give the type of the node.

        Args:
            op: The operator, which is neither a comparison nor `/`.
            left: The left input.
            right: The right input.

        Returns:
            The type.

        Example:
            >>> boolean = Operand.column(DType("Boolean"))
            >>> ArithmeticField().of(Operator.PLUS, boolean, boolean)
            UInt32
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = Operand.column(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     ArithmeticField().of(Operator.PLUS, struct, Operand.column(DType("Int16")))
            Struct({'a': Int16})
        """
        ltype, rtype = left.dtype, right.dtype
        struct = StructBesideNumber().joined(op, ltype, rtype)
        if struct is not None:
            return struct
        if op is Operator.MINUS:
            return self._minus(ltype, rtype)
        if op is Operator.PLUS:
            return self._plus(ltype, rtype)
        if op.is_bitwise() and self._boolean_beside_number(ltype, rtype):
            return self._mixed_bitwise(op, ltype, rtype)
        return self._other(op, left, right)

    @polars_releases
    def _mixed_bitwise(self, op: Operator, left: DType, right: DType) -> DType:
        """Refuse `&`, `|` or `^` between a `Boolean` and a number.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Raises:
            PolarsRefusal: The operator does not take the pair.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     ArithmeticField()._mixed_bitwise(Operator.AND, DType("Boolean"), DType("Int8"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: And on Boolean and Int8 ...
        """
        Supertype().require(left, right)
        message = f"{op.value} on {left!r} and {right!r} is not supported"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)

    @_mixed_bitwise.before("2.0.0")
    def _joined_bitwise(self, op: Operator, left: DType, right: DType) -> DType:
        """Give the supertype of a `Boolean` and a number, as the older releases do.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The supertype.

        Example:
            >>> ArithmeticField()._joined_bitwise(Operator.AND, DType("Boolean"), DType("Int8"))
            Int8
        """
        del op
        return Supertype().require(left, right)

    @polars_releases
    def _decimal_beside_integer(self, left: DType, right: DType) -> DType | None:
        """Give the type of a `Decimal` beside an integer: the largest precision at its scale.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The decimal, or `None` where the pair is not a decimal beside an integer.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     field = ArithmeticField()
            ...     field._decimal_beside_integer(DType("Int8"), DType("Decimal", (5, 2)))
            Decimal(precision=38, scale=2)
        """
        for side, other in ((left, right), (right, left)):
            if side.name == "Decimal" and Is.integer(other) and Is.known(other):
                return Decimals().widest(Parameter().integer(side, 1))
        return None

    @_decimal_beside_integer.before("2.0.0")
    def _no_decimal_arm(self, left: DType, right: DType) -> DType | None:
        """Give no type: the older releases join a decimal and an integer as the supertype does.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            `None`.

        Example:
            >>> ArithmeticField()._no_decimal_arm(DType("Int8"), DType("Decimal", (5, 2))) is None
            True
        """
        del left, right
        return None

    def _boolean_beside_number(self, left: DType, right: DType) -> bool:
        """Say whether only one input is a `Boolean` and only one is a number.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether the deprecated bitwise form applies.

        Example:
            >>> ArithmeticField()._boolean_beside_number(DType("Boolean"), DType("Int8"))
            True
        """
        booleans = (left.name == "Boolean") != (right.name == "Boolean")
        return booleans and Is.numeric(left) != Is.numeric(right)

    def _struct_first(self, left: DType, right: DType) -> bool:
        """Say whether a left struct is the type, beside a struct or a number.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            Whether the arm applies.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> ArithmeticField()._struct_first(struct, DType("Decimal", (10, 2)))
            True
        """
        return left.name == "Struct" and (right.name == "Struct" or Is.numeric(right))

    def _not_allowed(self, op: Operator, left: DType, right: DType) -> str:
        """Give the message of the refusal of an operator on two types.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The message.

        Example:
            >>> ArithmeticField()._not_allowed(Operator.PLUS, DType("Date"), DType("Int8"))
            'Plus not allowed on Date and Int8'
        """
        return f"{op.value} not allowed on {left!r} and {right!r}"

    def _minus(self, left: DType, right: DType) -> DType:
        """Give the type of `-`.

        A temporal type that meets a type it cannot subtract refuses with
        `PolarsRefusal`, as does a pair with no supertype.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The type.

        Example:
            >>> ArithmeticField()._minus(DType("Datetime", ("ms", None)), DType("Date"))
            Duration(time_unit='ms')
        """
        if self._struct_first(left, right):
            return left
        if (left.name, right.name) in self.MOMENT_PAIRS:
            return Supertype().require(left, right)
        difference = self._difference(left, right)
        if difference is not None:
            return difference
        return self._nested(NumericOp.SUB, Operator.MINUS, left, right)

    def _difference(self, left: DType, right: DType) -> DType | None:
        """Give the `Duration` of a difference of temporal types.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The `Duration`, or `None` where neither type is temporal.

        Raises:
            PolarsRefusal: Only one of the two types is of the temporal kind of an arm.

        Example:
            >>> ArithmeticField()._difference(DType("Time"), DType("Time"))
            Duration(time_unit='ns')
        """
        names = {left.name, right.name}
        if names == {"Datetime", "Date"}:
            moment = left if left.name == "Datetime" else right
            return DType("Duration", (Parameter().unit(moment),))
        for kind in self.DIFFERENCE_ORDER:
            if names == {kind}:
                return self._same_kind_difference(left, right)
            if kind in names:
                message = self._not_allowed(Operator.MINUS, left, right)
                raise PolarsRefusal(
                    ErrorClass.INVALID_OPERATION,
                    message,
                )
        return None

    def _same_kind_difference(self, left: DType, right: DType) -> DType:
        """Give the `Duration` of the difference of two values of one temporal kind.

        The sum of two durations takes the same type.

        Args:
            left: The left type.
            right: The right type, of the same kind.

        Returns:
            The `Duration`.

        Example:
            >>> nano, micro = DType("Duration", ("ns",)), DType("Duration", ("us",))
            >>> ArithmeticField()._same_kind_difference(nano, micro)
            Duration(time_unit='us')
        """
        if left.name in TimeUnit.DIFFERENCE:
            return DType("Duration", (TimeUnit.DIFFERENCE[left.name],))
        unit = TimeUnit.joined(Parameter().unit(left), Parameter().unit(right))
        return DType("Duration", (unit,))

    def _plus(self, left: DType, right: DType) -> DType:
        """Give the type of `+`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A temporal type meets a type that it cannot add.

        Example:
            >>> ArithmeticField()._plus(DType("Duration", ("ns",)), DType("Duration", ("ms",)))
            Duration(time_unit='ms')
        """
        if self._struct_first(left, right):
            return left
        names = {left.name, right.name}
        if (left.name, right.name) in self.MOMENT_PAIRS:
            return Supertype().require(left, right)
        moment = bool(names & Is.MOMENTS)
        if moment or (names != {"Duration"} and "Duration" in names):
            message = self._not_allowed(Operator.PLUS, left, right)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if names == {"Duration"}:
            return self._same_kind_difference(left, right)
        if names == {"Boolean"}:
            return DType.idx()
        return self._nested(NumericOp.ADD, Operator.PLUS, left, right)

    def _nested(self, list_op: NumericOp, op: Operator, left: DType, right: DType) -> DType:
        """Give the type of `+` or `-` for lists, arrays and decimals, and the supertype otherwise.

        A list joins the leaves as list arithmetic does, and an array joins
        them as the supertype does. The leaf of the list or the array comes
        first. A pair with no join refuses with `PolarsRefusal`, as
        `ListArithmeticCheck` does.

        Args:
            list_op: The list operation of the operator.
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The type.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> ArithmeticField()._nested(NumericOp.ADD, Operator.PLUS, DType("Float32"), listed)
            List(Float32)
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> decimal = DType("Decimal", (10, 2))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     ArithmeticField()._nested(NumericOp.ADD, Operator.PLUS, decimal, DType("Int8"))
            Decimal(precision=38, scale=2)
        """
        ListArithmeticCheck().refuse(op, left, right)
        for kind in ("List", "Array"):
            nested, other = (left, right) if left.name == kind else (right, left)
            if nested.name != kind:
                continue
            if kind == "List":
                return nested.with_leaf(list_op.leaf_supertype(nested.leaf(), other.leaf()))
            return nested.with_leaf(Supertype().require(nested.leaf(), other.leaf()))
        decimal = self._decimal_beside_integer(left, right)
        if decimal is not None:
            return decimal
        if left.name == right.name == "Decimal":
            return self.decimal(left, right)
        return Supertype().require(left, right)

    def decimal(self, left: DType, right: DType) -> DType:
        """Give the type of arithmetic on two decimals: the largest precision and the larger scale.

        Args:
            left: The left decimal.
            right: The right decimal.

        Returns:
            The decimal type.

        Example:
            >>> ArithmeticField().decimal(DType("Decimal", (10, 2)), DType("Decimal", (5, 4)))
            Decimal(precision=38, scale=4)
        """
        scale = max(Parameter().integer(left, 1), Parameter().integer(right, 1))
        return Decimals().widest(scale)

    def _other(self, op: Operator, left: Operand, right: Operand) -> DType:
        """Give the type of an operator that is not `+` or `-`.

        A pair that the arms refuse raises `PolarsRefusal`.

        Args:
            op: The operator.
            left: The left input.
            right: The right input.

        Returns:
            The type.

        Example:
            >>> duration = Operand.column(DType("Duration", ("us",)))
            >>> ArithmeticField()._other(Operator.MULTIPLY, Operand.column(DType("Int8")), duration)
            Duration(time_unit='us')
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> decimal = Operand.column(DType("Decimal", (10, 2)))
            >>> number = Operand.column(DType("Int8"))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     ArithmeticField()._other(Operator.MULTIPLY, decimal, number)
            Decimal(precision=38, scale=2)
        """
        ltype, rtype = left.dtype, right.dtype
        if self._struct_first(ltype, rtype):
            return ltype
        temporal = self._other_temporal(op, ltype, rtype)
        if temporal is not None:
            return temporal
        decimal = self._decimal_beside_integer(ltype, rtype)
        if decimal is not None:
            return decimal
        if ltype.name == rtype.name == "Decimal":
            return self.decimal(ltype, rtype)
        ListArithmeticCheck().refuse(op, ltype, rtype)
        for kind in ("List", "Array"):
            nested, other = (ltype, rtype) if ltype.name == kind else (rtype, ltype)
            if nested.name == kind:
                return nested.with_leaf(Supertype().require(nested.leaf(), other.leaf()))
        literal = self._literal_arm(left, right)
        return Supertype().require(ltype, rtype) if literal is None else literal

    def _literal_arm(self, left: Operand, right: Operand) -> DType | None:
        """Give the type that an arm of a lone literal gives.

        A lone literal on the left takes the type on the right
        (`_literal_takes_right`). In some releases, a lone literal on the right
        keeps the type on the left (`_literal_keeps_left`).

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type, or `None` where no such arm applies.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> narrow = Operand.column(DType("Float32"))
            >>> ArithmeticField()._literal_arm(Operand.of_value(1.5), narrow)
            Float32
            >>> with PolarsRelease.pretend("1.42.1"):
            ...     ArithmeticField()._literal_arm(narrow, Operand.of_value(1.5))
            Float32
        """
        if self._literal_takes_right(left, right):
            return right.dtype
        if self._literal_keeps_left(left, right):
            return left.dtype
        return None

    def _other_temporal(self, op: Operator, left: DType, right: DType) -> DType | None:
        """Give the type of an operator other than `+` and `-` on a temporal type.

        Only `*` of a `Duration` and a number has a type.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The `Duration`, or `None` where no type is temporal in the sense of the arms.

        Raises:
            PolarsRefusal: The arms refuse the pair.

        Example:
            >>> duration = DType("Duration", ("us",))
            >>> ArithmeticField()._other_temporal(Operator.MULTIPLY, DType("Int8"), duration)
            Duration(time_unit='us')
        """
        names = {left.name, right.name}
        refused = bool(names & Is.MOMENTS) or names == {"Duration"}
        for duration, number in ((right, left), (left, right)):
            if refused or duration.name != "Duration" or not Is.primitive_numeric(number):
                continue
            if op is Operator.MULTIPLY:
                return duration
            refused = True
        if refused:
            message = self._not_allowed(op, left, right)
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return None

    def _literal_takes_right(self, left: Operand, right: Operand) -> bool:
        """Say whether a literal on the left takes the type on the right.

        The arm applies to two integers or two floats where only the left
        input is a literal node and its type is `Unknown`.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Whether the left literal takes the right type.

        Example:
            >>> narrow = Operand.column(DType("Float32"))
            >>> ArithmeticField()._literal_takes_right(Operand.of_value(1.5), narrow)
            True
        """
        ltype, rtype = left.dtype, right.dtype
        integers = Is.integer(ltype) and Is.integer(rtype)
        floats = Is.float(ltype) and Is.float(rtype)
        alone = left.literal and not right.literal
        return (integers or floats) and alone and Is.unknown(ltype)

    @polars_releases
    def _literal_keeps_left(self, left: Operand, right: Operand) -> bool:
        """Say whether a literal on the right keeps the type on the left: never.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Always `False`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> narrow = Operand.column(DType("Int8"))
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     ArithmeticField()._literal_keeps_left(narrow, Operand.of_value(300))
            False
        """
        del left, right
        return False

    @_literal_keeps_left.before("1.43.0")
    def _right_literal_keeps_left(self, left: Operand, right: Operand) -> bool:
        """Say whether a literal on the right keeps the type on the left, as the older releases do.

        The arm mirrors `_literal_takes_right`: it applies to two integers or
        two floats where only the right input is a literal node and its type
        is `Unknown`.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Whether the left type stays.

        Example:
            >>> narrow = Operand.column(DType("Int8"))
            >>> ArithmeticField()._right_literal_keeps_left(narrow, Operand.of_value(300))
            True
        """
        ltype, rtype = left.dtype, right.dtype
        integers = Is.integer(ltype) and Is.integer(rtype)
        floats = Is.float(ltype) and Is.float(rtype)
        alone = right.literal and not left.literal
        return (integers or floats) and alone and Is.unknown(rtype)


class StructBesideNumber:
    """Give the type of arithmetic of a struct beside a number, field by field.

    From polars 2.0.0, each field joins the number: a struct field recurses, a
    `Duration` stays under `*`, a list or an array joins its leaf as list
    arithmetic does, and a number, `Boolean` or `Null` joins the number as the
    supertype does. A division then gives the float of each field. Any other
    field refuses. Before that release, the arms of each operator type the
    pair (`joined`).

    Example:
        >>> struct = DType("Struct", ((("a", DType("Int8")),),))
        >>> StructBesideNumber()._fields(Operator.PLUS, struct, DType("Int16"))
        Struct({'a': Int16})
    """

    LIST_OPS: ClassVar[dict[Operator, NumericOp]] = {
        Operator.PLUS: NumericOp.ADD,
        Operator.MINUS: NumericOp.SUB,
        Operator.MULTIPLY: NumericOp.MUL,
        Operator.TRUE_DIVIDE: NumericOp.DIV,
        Operator.RUST_DIVIDE: NumericOp.DIV,
        Operator.FLOOR_DIVIDE: NumericOp.FLOOR_DIV,
        Operator.MODULUS: NumericOp.REM,
    }

    @polars_releases
    def joined(self, op: Operator, left: DType, right: DType) -> DType | None:
        """Give the struct of an arithmetic operator on a struct and a number.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            The struct, or `None` where the pair is not a struct beside a number.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     StructBesideNumber().joined(Operator.TRUE_DIVIDE, DType("Int8"), struct)
            Struct({'a': Float64})
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     StructBesideNumber().joined(Operator.EQ, struct, DType("Int8")) is None
            True
        """
        if op.is_comparison_or_bitwise():
            return None
        for struct, number in ((left, right), (right, left)):
            if struct.name == "Struct" and Is.primitive_numeric(number):
                return self._fields(op, struct, number)
        return None

    @joined.before("2.0.0")
    def _per_operator(self, op: Operator, left: DType, right: DType) -> DType | None:
        """Give no struct: the older releases type the pair in the arms of each operator.

        Args:
            op: The operator.
            left: The left type.
            right: The right type.

        Returns:
            `None`.

        Example:
            >>> StructBesideNumber()._per_operator(Operator.PLUS, DType("Int8"), DType("Int8"))
        """
        del op, left, right
        return None

    def _fields(self, op: Operator, struct: DType, number: DType) -> DType:
        """Join each field of a struct with a number.

        Args:
            op: The operator.
            struct: The struct.
            number: The number.

        Returns:
            The struct of the joined fields.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Null")), ("b", DType("Boolean"))),))
            >>> StructBesideNumber()._fields(Operator.MINUS, pair, DType("UInt8"))
            Struct({'a': UInt8, 'b': UInt8})
        """
        fields = tuple(
            (name, self._field(op, name, field, number)) for name, field in struct.fields()
        )
        return DType("Struct", (fields,))

    def _field(self, op: Operator, name: str, field: DType, number: DType) -> DType:
        """Join one field of a struct with a number.

        Args:
            op: The operator.
            name: The name of the field, for the message.
            field: The type of the field.
            number: The number.

        Returns:
            The type of the field.

        Raises:
            PolarsRefusal: The field is not a number, and not a type that the arms take.

        Example:
            >>> StructBesideNumber()._field(Operator.PLUS, "a", DType("String"), DType("Int8"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot Plus a struct ...
            >>> inner = DType("Struct", ((("b", DType("Int8")),),))
            >>> StructBesideNumber()._field(Operator.PLUS, "a", inner, DType("Int16"))
            Struct({'b': Int16})
            >>> span, listed = DType("Duration", ("us",)), DType("List", (DType("Int8"),))
            >>> StructBesideNumber()._field(Operator.MULTIPLY, "a", span, DType("Int8"))
            Duration(time_unit='us')
            >>> StructBesideNumber()._field(Operator.PLUS, "a", listed, DType("Int16"))
            List(Int16)
        """
        if field.name == "Struct":
            return self._fields(op, field, number)
        if field.name == "Duration" and op is Operator.MULTIPLY:
            return field
        if field.name in Is.LISTS:
            return self._nested(op, name, field, number)
        if Is.numeric(field) or field.name in {"Null", "Boolean"}:
            joined = Supertype().require(field, number)
            return TrueDivField().of(joined, joined) if op is Operator.TRUE_DIVIDE else joined
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, self._refused(op, name, field))

    def _refused(self, op: Operator, name: str, field: DType) -> str:
        """Give the message of a field that the operator does not take.

        Args:
            op: The operator.
            name: The name of the field.
            field: The type of the field.

        Returns:
            The message.

        Example:
            >>> StructBesideNumber()._refused(Operator.PLUS, "a", DType("String"))
            'cannot Plus a struct with non-numeric field: (field: a, dtype: String)'
        """
        return (
            f"cannot {op.value} a struct with non-numeric field: (field: {name}, dtype: {field!r})"
        )

    def _nested(self, op: Operator, name: str, field: DType, number: DType) -> DType:
        """Join the leaf of a list or an array field with a number, as list arithmetic does.

        The leaf here is the type under the lists or the arrays of the kind of
        the field, so a list of arrays has an array as its leaf.

        Args:
            op: The operator.
            name: The name of the field, for the message.
            field: The type of the field.
            number: The number.

        Returns:
            The type of the field.

        Raises:
            PolarsRefusal: List arithmetic does not take the leaf.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> StructBesideNumber()._nested(Operator.PLUS, "a", listed, DType("Int16"))
            List(Int16)
            >>> texts = DType("List", (DType("String"),))
            >>> StructBesideNumber()._nested(Operator.PLUS, "a", texts, DType("Int8"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot Plus a struct ...
        """
        leaf = field
        while leaf.name == field.name:
            leaf = leaf.inner()
        if not Is.list_arithmetic_input(leaf):
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, self._refused(op, name, field))
        return field.with_leaf(self.LIST_OPS[op].leaf_supertype(leaf, number))


class TrueDivField:
    """Give the planner type of `/`.

    A true division gives a float: the narrowest float that holds the two
    inputs, with `Float16` and `Float32` kept beside small integers. Two
    structs divide field by field. A list or an array divides its leaf, the
    leaf of the list first. A `Decimal` stays beside an integer. A `Duration`
    divided by a number stays. A pair that the arms refuse raises `PolarsRefusal`.

    Example:
        >>> TrueDivField().of(DType("Int8"), DType("Float32"))
        Float32
    """

    NARROW_BESIDE: ClassVar[dict[str, dict[str, str]]] = {
        "Float16": {"UInt8": "Float16", "Int8": "Float16", "UInt16": "Float32", "Int16": "Float32"},
        "Float32": {"UInt8": "Float32", "Int8": "Float32", "UInt16": "Float32", "Int16": "Float32"},
    }

    def of(self, left: DType, right: DType) -> DType:
        """Give the type of `left / right`.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The type.

        Example:
            >>> TrueDivField().of(DType("Duration", ("ms",)), DType("Duration", ("us",)))
            Float64
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField().of(struct, DType("Int8"))
            Struct({'a': Float64})
        """
        struct = StructBesideNumber().joined(Operator.TRUE_DIVIDE, left, right)
        if struct is not None:
            return struct
        if left.name == "Struct" and self._divides_fields(right):
            return self._struct(left, right)
        ListArithmeticCheck().refuse(Operator.TRUE_DIVIDE, left, right)
        for kind in ("List", "Array"):
            nested, other = (left, right) if left.name == kind else (right, left)
            if nested.name == kind:
                return nested.with_leaf(self.of(nested.leaf(), other.leaf()))
        flat = self._flat_number(left, right)
        return flat if flat is not None else self._other(left, right)

    @polars_releases
    def _divides_fields(self, right: DType) -> bool:
        """Say whether a struct on the left divides field by field: beside a struct.

        A struct beside a number of the kernels has the rule of
        `StructBesideNumber`. Beside any other type, the struct keeps its type.

        Args:
            right: The right type.

        Returns:
            Whether the right type is a struct.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField()._divides_fields(DType("Decimal", (10, 2)))
            False
        """
        return right.name == "Struct"

    @_divides_fields.before("2.0.0")
    def _divides_by_numbers(self, right: DType) -> bool:
        """Say whether a struct on the left divides field by field, as the older releases do.

        Args:
            right: The right type.

        Returns:
            Whether the right type is a struct or a number.

        Example:
            >>> TrueDivField()._divides_by_numbers(DType("Decimal", (10, 2)))
            True
        """
        return right.name == "Struct" or Is.numeric(right)

    def _struct(self, left: DType, right: DType) -> DType:
        """Divide a struct by a struct or by a number, field by field.

        A right struct of one field divides every field. Two structs of other
        lengths refuse. Each field and the divisor must be numbers.

        Args:
            left: The left struct.
            right: The right struct or number.

        Returns:
            The struct type.

        Raises:
            PolarsRefusal: The lengths differ, or a field is not a number.

        Example:
            >>> pair = DType("Struct", ((("a", DType("Int8")), ("b", DType("Float32"))),))
            >>> TrueDivField()._struct(pair, DType("Int8"))
            Struct({'a': Float64, 'b': Float32})
        """
        fields = left.fields()
        divisors = [field for _, field in right.fields()] if right.name == "Struct" else [right]
        if right.name == "Struct" and len(divisors) not in {len(fields), 1}:
            message = f"cannot div two structs of different length: {left!r}, {right!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        divided = []
        for index, (name, field) in enumerate(fields):
            divisor = divisors[min(index, len(divisors) - 1)]
            if not (Is.numeric(field) and Is.numeric(divisor)):
                message = f"cannot div two structs with non-numeric fields: {field!r}, {divisor!r}"
                raise PolarsRefusal(
                    ErrorClass.INVALID_OPERATION,
                    message,
                )
            divided.append((name, self.of(field, divisor)))
        return DType("Struct", (tuple(divided),))

    def _flat_number(self, left: DType, right: DType) -> DType | None:
        """Give the type of a division of a `Boolean` or a narrow float.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The type, or `None` where no arm of this group applies.

        Example:
            >>> TrueDivField()._flat_number(DType("Float16"), DType("Int16"))
            Float32
            >>> TrueDivField()._flat_number(DType("Float32"), DType.dynamic_int(1))
            Float32
        """
        if left.name == "Boolean":
            return self._boolean(right)
        if left.name not in self.NARROW_BESIDE:
            return None
        beside = self.NARROW_BESIDE[left.name].get(right.name)
        if beside is not None:
            return DType(beside)
        if Is.dynamic_int(right):
            return left
        if Is.integer(right):
            return DType("Float64")
        return right if right.name in {"Float32", "Float64"} else left

    def _boolean(self, right: DType) -> DType | None:
        """Give the type of a `Boolean` divided by a type.

        Args:
            right: The right type.

        Returns:
            A narrow float as it is, `Float64` for a number or a `Boolean`, and `None` otherwise.

        Example:
            >>> TrueDivField()._boolean(DType("Decimal", (10, 2)))
            Float64
        """
        if right.name in self.NARROW_BESIDE:
            return right
        if Is.numeric(right) or right.name == "Boolean":
            return DType("Float64")
        return None

    def _other(self, left: DType, right: DType) -> DType:
        """Give the type of a division of a type that is not a `Boolean` or a narrow float.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            The type.

        Raises:
            PolarsRefusal: An input is a `String`.

        Example:
            >>> TrueDivField()._other(DType("UInt16"), DType("Float16"))
            Float32
        """
        if "String" in {left.name, right.name}:
            message = "division with 'String' datatypes is not allowed"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if left.name == "Decimal":
            return self._decimal(left, right)
        narrow = self.NARROW_BESIDE.get(right.name, {})
        if left.name in narrow:
            return DType(narrow[left.name])
        if Is.primitive_numeric(left):
            return self._number_by(left, right)
        return self._duration_or_moment(left, right)

    @polars_releases
    def _number_by(self, left: DType, right: DType) -> DType:
        """Give the type of a number divided by a type: the type of a decimal divisor, or `Float64`.

        A `Decimal` divisor gives the type of `_beside_number`.

        Args:
            left: The number.
            right: The divisor.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField()._number_by(DType("Int8"), DType("Decimal", (5, 2)))
            Decimal(precision=38, scale=2)
        """
        if right.name == "Decimal":
            return self._beside_number(right, left)
        return DType("Float64")

    @_number_by.before("2.0.0")
    def _float_quotient(self, left: DType, right: DType) -> DType:
        """Give the type of a number divided by a type in the older releases: `Float64`.

        Args:
            left: The number.
            right: The divisor.

        Returns:
            `Float64`.

        Example:
            >>> TrueDivField()._float_quotient(DType("Int8"), DType("Decimal", (5, 2)))
            Float64
        """
        del left, right
        return DType("Float64")

    def _decimal(self, left: DType, right: DType) -> DType:
        """Give the type of a `Decimal` divided by a type.

        Args:
            left: The left decimal.
            right: The right type.

        Returns:
            A decimal of the largest precision beside a decimal, and the type
            of `_beside_number` otherwise.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.44.2"):
            ...     TrueDivField()._decimal(DType("Decimal", (10, 2)), DType("Int64"))
            Decimal(precision=10, scale=2)
        """
        if right.name == "Decimal":
            return ArithmeticField().decimal(left, right)
        return self._beside_number(left, right)

    @polars_releases
    def _beside_number(self, left: DType, right: DType) -> DType:
        """Give the type of a `Decimal` beside a type that is not a `Decimal`, in a division.

        Args:
            left: The decimal.
            right: The other type.

        Returns:
            `Float64` beside a float, the decimal of the largest precision at
            its scale beside an integer, and the decimal itself beside a type
            that is not a number.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> decimal = DType("Decimal", (10, 2))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField()._beside_number(decimal, DType("Int8"))
            Decimal(precision=38, scale=2)
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField()._beside_number(decimal, DType("Time"))
            Decimal(precision=10, scale=2)
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     TrueDivField()._beside_number(decimal, DType("Float32"))
            Float64
        """
        if Is.float(right):
            return DType("Float64")
        if not Is.primitive_numeric(right):
            return left
        return Decimals().widest(Parameter().integer(left, 1))

    @_beside_number.before("2.0.0")
    def _float_or_decimal(self, left: DType, right: DType) -> DType:
        """Give the type of a `Decimal` divided by another type, in polars 1.43 and 1.44.

        Args:
            left: The left decimal.
            right: The right type.

        Returns:
            `Float64` beside a float, and the left decimal otherwise.

        Example:
            >>> TrueDivField()._float_or_decimal(DType("Decimal", (10, 2)), DType("Float32"))
            Float64
        """
        return DType("Float64") if Is.float(right) else left

    @_beside_number.before("1.43.0")
    def _keeps_decimal(self, left: DType, right: DType) -> DType:
        """Keep the left decimal beside every type, a float included, as the older releases do.

        Args:
            left: The left decimal.
            right: The right type.

        Returns:
            The left decimal.

        Example:
            >>> TrueDivField()._keeps_decimal(DType("Decimal", (10, 2)), DType("Float32"))
            Decimal(precision=10, scale=2)
        """
        del right
        return left

    def _duration_or_moment(self, left: DType, right: DType) -> DType:
        """Give the type of a division of a `Duration`, and refuse a moment.

        Args:
            left: The left type.
            right: The right type.

        Returns:
            `Float64` for two durations, the duration beside a number, and
            the left type for any other left type.

        Raises:
            PolarsRefusal: The left type is a moment, or a duration meets another type.

        Example:
            >>> TrueDivField()._duration_or_moment(DType("Duration", ("us",)), DType("Int8"))
            Duration(time_unit='us')
        """
        if left.name == "Duration" and right.name == "Duration":
            return DType("Float64")
        if left.name == "Duration" and Is.primitive_numeric(right):
            return left
        if left.name == "Duration" or left.name in Is.MOMENTS:
            message = f"true division of {left!r} with {right!r} is not allowed"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return left


class ExprFields:
    """Give the planner type of each node that is not a function.

    A binary operator, a `when`, a cast and a literal each have a rule here.
    Sorts, filters, slices and gathers keep the type of their input. The
    aggregations are in `AggFields`.

    Example:
        >>> date, int8 = Operand.column(DType("Date")), Operand.column(DType("Int8"))
        >>> ExprFields().binary(Operator.LT, date, int8)
        Boolean
    """

    LOGICAL: ClassVar[frozenset[Operator]] = frozenset({Operator.LOGICAL_AND, Operator.LOGICAL_OR})

    def binary(self, op: Operator, left: Operand, right: Operand) -> DType:
        """Give the type of a binary operator.

        A comparison and a logical operator give `Boolean`, whatever the
        inputs. `/` follows `TrueDivField`, and the rest follows
        `ArithmeticField`. A pair that they refuse raises `PolarsRefusal`.

        Args:
            op: The operator.
            left: The left input.
            right: The right input.

        Returns:
            The type.

        Example:
            >>> int8 = Operand.column(DType("Int8"))
            >>> ExprFields().binary(Operator.TRUE_DIVIDE, int8, Operand.of_value(2))
            Float64
        """
        if op.is_comparison() or op in self.LOGICAL:
            return DType("Boolean")
        if op is Operator.TRUE_DIVIDE:
            return TrueDivField().of(left.dtype, right.dtype)
        return ArithmeticField().of(op, left, right)

    def ternary(self, truthy: DType, falsy: DType) -> DType:
        """Give the type of `when(...).then(truthy).otherwise(falsy)`.

        Two types with no supertype refuse with `PolarsRefusal`.

        Args:
            truthy: The type of the `then` input.
            falsy: The type of the `otherwise` input.

        Returns:
            The type of `otherwise` where `then` is `Null`, and their supertype otherwise.

        Example:
            >>> ExprFields().ternary(DType("Null"), DType.dynamic_int(1))
            Unknown(int, 1)
        """
        if truthy.name == "Null":
            return falsy
        return Supertype().require(truthy, falsy)

    def append(self, values: DType, other: DType) -> DType:
        """Give the type of `append`: the supertype of its two inputs.

        Polars casts both inputs to their supertype when it converts the
        expression, before the node types them, so the planner reads that
        supertype. Two inputs with no supertype refuse the plan. Before
        polars 1.40.0, the node types its inputs itself (`_joined`).

        Args:
            values: The type of the first input.
            other: The type of the second input.

        Returns:
            The supertype.

        Example:
            >>> ExprFields().append(DType("Int8"), DType("Int16"))
            Int16
        """
        return self._joined(values, Supertype().require(values, other))

    @polars_releases
    def _joined(self, values: DType, joined: DType) -> DType:
        """Give the type of `append` from the supertype of its inputs: that supertype.

        Args:
            values: The type of the first input.
            joined: The supertype of the two inputs.

        Returns:
            The supertype.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     ExprFields()._joined(DType("Categorical"), DType("String"))
            String
        """
        del values
        return joined

    @_joined.before("1.40.0")
    def _mapped(self, values: DType, joined: DType) -> DType:
        """Give the type of `append` in the older releases, as the node maps its inputs to one type.

        A literal type of the supertype takes the type that it takes alone,
        and a `Categorical` beside text keeps its categories.

        Args:
            values: The type of the first input.
            joined: The supertype of the two inputs.

        Returns:
            The type.

        Example:
            >>> ExprFields()._mapped(DType("Categorical"), DType("String"))
            Categorical
            >>> ExprFields()._mapped(DType.dynamic_int(1), DType.dynamic_int(1))
            Int32
        """
        if values.name == "Categorical" and joined.name == "String":
            return values
        return joined.materialized() if Is.unknown(joined) else joined

    def when(self, predicate: DType, truthy: DType, falsy: DType) -> DType:
        """Give the type of a `when` node from its three inputs: the type of `ternary`.

        The mask does not take part in the type.

        Args:
            predicate: The type of the mask.
            truthy: The type of `then`.
            falsy: The type of `otherwise`.

        Returns:
            The type.

        Example:
            >>> ExprFields().when(DType("Boolean"), DType("Int8"), DType("Int16"))
            Int16
        """
        del predicate
        return self.ternary(truthy, falsy)

    def cast(self, source: DType, target: DType) -> DType:
        """Give the type of a cast: its target, whatever its input.

        Args:
            source: The type of the input.
            target: The type of the cast.

        Returns:
            The target.

        Example:
            >>> ExprFields().cast(DType("String"), DType("Date"))
            Date
        """
        return target

    def literal(self, operand: Operand) -> DType:
        """Give the type of a literal node.

        Args:
            operand: The literal.

        Returns:
            Its type: `Unknown` for a Python number with no data type.

        Example:
            >>> ExprFields().literal(Operand.of_value(2**40))
            Unknown(int, 1099511627776)
        """
        return operand.dtype

    def explode(self, dtype: DType) -> DType:
        """Give the type of `explode`: the inner type of a list or an array.

        Args:
            dtype: The input type.

        Returns:
            The inner type, or the input type where it is not a list or an array.

        Example:
            >>> ExprFields().explode(DType("List", (DType("Int8"),)))
            Int8
        """
        return dtype.inner() if dtype.name in Is.LISTS else dtype

    def over(self, dtype: DType, *, join: bool, scalar: bool) -> DType:
        """Give the type of `over`.

        With the mapping `join`, a function that does not give one value per
        group gives a list for each row.

        Args:
            dtype: The type of the function.
            join: Whether the mapping strategy is `join`.
            scalar: Whether the function gives one value per group.

        Returns:
            The type.

        Example:
            >>> ExprFields().over(DType("Int8"), join=True, scalar=False)
            List(Int8)
        """
        return DType("List", (dtype,)) if join and not scalar else dtype

    def rolling(self, dtype: DType, *, scalar: bool) -> DType:
        """Give the type of `rolling` over a function.

        Args:
            dtype: The type of the function.
            scalar: Whether the function gives one value per window.

        Returns:
            The type, in a list where the function gives more than one value.

        Example:
            >>> ExprFields().rolling(DType("Int8"), scalar=True)
            Int8
        """
        return dtype if scalar else DType("List", (dtype,))

    def length(self) -> DType:
        """Give the type of `pl.len()`.

        Returns:
            `UInt32`.

        Example:
            >>> ExprFields().length()
            UInt32
        """
        return DType.idx()

    def bound(self, dtype: DType) -> DType:
        """Give the type of the literal that `lower_bound` or `upper_bound` becomes.

        When polars converts the expression, it puts a literal in place of the
        node: the least or the greatest value of the physical type of the
        input. So a `Date` gives an `Int32`. Only a number has such a value. A
        literal with no data type has no physical type yet.

        Args:
            dtype: The planner type of the input.

        Returns:
            The physical type.

        Raises:
            PolarsRefusal: The physical type is not a number.

        Example:
            >>> ExprFields().bound(DType("Decimal", (10, 2)))
            Int128
            >>> ExprFields().bound(DType("String"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: cannot determine the bound ...
        """
        physical = dtype.physical()
        if Is.unknown(physical) or not Is.primitive_numeric(physical):
            message = f"cannot determine the bound of the dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return physical


class AggFields:
    """Give the planner type of each aggregation node.

    These are the aggregations that polars stores as `Agg` nodes, not as
    functions: `min`, `max`, `first`, `last`, `item`, `sum`, `mean`,
    `median`, `std`, `var`, `implode`, `n_unique`, `count` and `agg_groups`.

    Example:
        >>> AggFields().sum(DType("Int8")), AggFields().mean(DType("Date"))
        (Int64, Datetime(time_unit='us', time_zone=None))
    """

    UNSUMMABLE: ClassVar[frozenset[str]] = frozenset({
        "String",
        "Binary",
        "List",
        "Array",
        "Struct",
    })

    def same(self, dtype: DType) -> DType:
        """Give the type of `min`, `max`, `first`, `last` and `item`.

        Args:
            dtype: The input type.

        Returns:
            The input type.

        Example:
            >>> AggFields().same(DType("String"))
            String
        """
        return dtype

    def sum(self, dtype: DType) -> DType:
        """Give the type of `sum`.

        Args:
            dtype: The input type.

        Returns:
            `UInt32` for a `Boolean`, `Int64` for a small integer, a decimal
            of the largest precision (`_decimal`), and the input type otherwise.

        Raises:
            PolarsRefusal: The input is text, binary data, a list, an array or a struct.

        Example:
            >>> AggFields().sum(DType("UInt8"))
            Int64
        """
        if dtype.name in self.UNSUMMABLE:
            message = f"`sum` operation not supported for dtype `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if dtype.name == "Decimal":
            return self._decimal(dtype)
        return FieldsMapper(dtype).sum_dtype()

    @polars_releases
    def _decimal(self, dtype: DType) -> DType:
        """Give the type of the sum of a decimal: the decimal of the largest precision at its scale.

        Args:
            dtype: The decimal.

        Returns:
            The decimal of precision 38.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     AggFields()._decimal(DType("Decimal", (10, 2)))
            Decimal(precision=38, scale=2)
        """
        return Decimals().widest(Parameter().integer(dtype, 1))

    @_decimal.before("1.40.0")
    def _kept(self, dtype: DType) -> DType:
        """Give the type of the sum of a decimal in the older releases: the decimal itself.

        Args:
            dtype: The decimal.

        Returns:
            The same decimal.

        Example:
            >>> AggFields()._kept(DType("Decimal", (10, 2)))
            Decimal(precision=10, scale=2)
        """
        return dtype

    def mean(self, dtype: DType) -> DType:
        """Give the type of `mean`, `median` and `std`.

        Args:
            dtype: The input type.

        Returns:
            The type of `FieldsMapper.moment_dtype`.

        Example:
            >>> AggFields().mean(DType("Int8"))
            Float64
        """
        return FieldsMapper(dtype).moment_dtype()

    @polars_releases
    def var(self, dtype: DType) -> DType:
        """Give the type of `var`.

        Args:
            dtype: The input type.

        Returns:
            The type of `FieldsMapper.moment_dtype`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     AggFields().var(DType("Duration", ("us",)))
            Duration(time_unit='us')
        """
        return FieldsMapper(dtype).moment_dtype()

    @var.before("2.0.0")
    def _refused_duration(self, dtype: DType) -> DType:
        """Give the type of `var` in the older releases: that of `FieldsMapper.var_dtype`.

        Args:
            dtype: The input type.

        Returns:
            The type, which refuses a `Duration`.

        Example:
            >>> AggFields()._refused_duration(DType("Float32"))
            Float32
        """
        return FieldsMapper(dtype).var_dtype()

    def implode(self, dtype: DType) -> DType:
        """Give the type of `implode`: a list of the input type.

        Args:
            dtype: The input type.

        Returns:
            The list type.

        Example:
            >>> AggFields().implode(DType("Int8"))
            List(Int8)
        """
        return DType("List", (dtype,))

    def count(self, dtype: DType) -> DType:
        """Give the type of `count`, `len` and `n_unique`.

        Args:
            dtype: The input type.

        Returns:
            `UInt32`.

        Example:
            >>> AggFields().count(DType("String"))
            UInt32
        """
        return DType.idx()

    def agg_groups(self, dtype: DType) -> DType:
        """Give the type of `agg_groups`: a list of row indices.

        Args:
            dtype: The input type.

        Returns:
            `List(UInt32)`.

        Example:
            >>> AggFields().agg_groups(DType("Int8"))
            List(UInt32)
        """
        return DType("List", (DType.idx(),))


class ProjectionSchema:
    """Give the type that the schema of a `select` or of a `group_by().agg()` shows for a node.

    Polars computes the schema of a projection from the type function of each
    expression, on the tree before the cast pass. It then gives a literal type
    the type that it takes alone, so `pl.lit(1)` shows `Int32`. In `agg`, an
    expression that does not give one value for each group shows a list of its
    type. The type function itself does not read the context: it gives the same
    type in both.

    Example:
        >>> ProjectionSchema().select(DType.dynamic_int(2**40))
        Int64
        >>> ProjectionSchema().agg(DType("Int8"), scalar=False)
        List(Int8)
    """

    def select(self, dtype: DType) -> DType:
        """Give the type in the schema of a `select`.

        Args:
            dtype: The type of the expression.

        Returns:
            The type with every literal type materialized, and `Unknown` as it is.

        Example:
            >>> ProjectionSchema().select(DType("List", (DType.unknown("float"),)))
            List(Float64)
        """
        return dtype.materialized()

    def agg(self, dtype: DType, *, scalar: bool) -> DType:
        """Give the type in the schema of a `group_by().agg()`.

        Args:
            dtype: The type of the expression.
            scalar: Whether the expression gives one value for each group.

        Returns:
            The type, in a list where the expression gives more than one value.

        Example:
            >>> ProjectionSchema().agg(DType.dynamic_int(1), scalar=True)
            Int32
        """
        materialized = self.select(dtype)
        return materialized if scalar else DType("List", (materialized,))
