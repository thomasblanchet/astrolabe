"""The kernels of the `list` namespace, which work on the values inside each list.

A function of the `list` namespace reads its first input as a list column.
Data of another type stops it with a `SchemaError`. Most functions then run a
kernel of the inner type on the values of each list: `list.sum` sums the
values, `list.mean` takes their mean, and `list.diff` subtracts each value
from the next one. That kernel picks its work by the inner type, as any
kernel picks it by the type of its data, and it can refuse an inner type. So
the run type of such a function is a function of the inner type of the data,
and it can differ from the planner type. For example, the planner gives
`list.std` of dates the type `Datetime`, and the kernel gives `Float64`.

The functions of several inputs, such as `list.get`, also read the number of
rows of each input. Each one checks the lengths in its own way, with its own
error, so each has a class here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.cast_pass import CastFold
from astrolabe._rules.dtypes import DType, Is, Parameter
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.dispatch import (
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    Refuse,
    RunCast,
    Same,
)
from astrolabe._rules.operators import NumericOp
from astrolabe._rules.supertype import Supertype
from astrolabe._rules.type_functions import StructOf

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class Elements:
    """Run a kernel of the inner type on the values of each list or array, and give its type.

    A function such as `list.sum` reads the values of each list as one series
    of the inner type, and runs a kernel on that series. `kernel` is the
    dispatch of that kernel on the inner type. Where `listed` is true, each
    list gives a list again, as `list.diff` does, so the output is a list of
    the type that the kernel gives.

    Example:
        >>> sums = Elements(ListKernels.VALUE_SUM)
        >>> sums.give(Data(DType("List", (DType("Int8"),))), ListKernels.SUM)
        Int64
    """

    kernel: Dispatch
    listed: bool = False

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the kernel refuses the inner type.

        Args:
            data: The data, of a `List` or an `Array` type.
            kernel: The dispatch of the function, which this output does not read.

        Returns:
            The type.

        Example:
            >>> differences = Elements(ListKernels.VALUE_DIFF, listed=True)
            >>> differences.give(Data(DType("List", (DType("UInt8"),))), ListKernels.DIFF)
            List(Int16)
        """
        del kernel
        output = self.kernel.output(Data(data.dtype.inner()))
        return DType("List", (output,)) if self.listed else output


class Difference:
    """Give the type of the `diff` of a series: the series minus the series shifted.

    An unsigned integer first casts to the signed type that holds the
    difference of two of its values. Then the kernel subtracts two series of
    one type with the arithmetic of two series (`SeriesArithmetic`), which
    refuses a type that has no subtraction, such as `Boolean` or `String`.

    Example:
        >>> Difference().give(Data(DType("Date")), ListKernels.VALUE_DIFF)
        Duration(time_unit='us')
    """

    SIGNED: ClassVar[dict[str, str]] = {
        "UInt8": "Int16",
        "UInt16": "Int32",
        "UInt32": "Int64",
        "UInt64": "Int64",
    }

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the difference.

        It raises `PolarsRefusal` where the subtraction refuses the type.

        Args:
            data: The data of the series.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> Difference().give(Data(DType("String")), ListKernels.VALUE_DIFF)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        del kernel
        signed = self.SIGNED.get(data.dtype.name)
        values = data if signed is None else RunCast().of(data, DType(signed))
        return SeriesArithmetic(NumericOp.SUB).output(values, values)


@dataclass(frozen=True)
class AsArray:
    """Give an `Array` of the inner type of a list, as the cast of `list.to_array` gives it.

    The kernel casts the lists to arrays of one width. A list of another
    length makes the cast fail, which is a fact of the data.

    Example:
        >>> lists = Data(DType("List", (DType("Date"),)))
        >>> AsArray(2).give(lists, ListKernels().to_array(2))
        Array(Date, shape=(2,))
    """

    width: int

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the array type.

        It raises `PolarsRefusal` where the cast has no kernel.

        Args:
            data: The data, of a `List` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> nested = Data(DType("List", (DType("Array", (DType("Int8"), 2)),)))
            >>> AsArray(3).give(nested, ListKernels().to_array(3)).args[1]
            3
        """
        del kernel
        target = DType("Array", (data.dtype.inner(), self.width))
        return RunCast().of(data, target).dtype


@dataclass(frozen=True)
class NamedFields:
    """Give a struct with one field of the inner type of a list for each name, as `list.to_struct`.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> NamedFields(("a", "b")).give(lists, ListKernels().to_struct(("a", "b")))
        Struct({'a': Int8, 'b': Int8})
    """

    names: Sequence[str]

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the struct type.

        Args:
            data: The data, of a `List` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> lists = Data(DType("List", (DType("Date"),)))
            >>> NamedFields(()).give(lists, ListKernels().to_struct(()))
            Struct({})
        """
        del kernel
        return StructOf().of(data.dtype.inner(), self.names)


class ListKernels:
    """Hold the kernels of the functions of the `list` namespace that read one input.

    Each kernel is a dispatch on the data of the list column. Its one arm
    takes a `List`, and any other type raises a `SchemaError`. The arm gives a
    fixed type, the type of the list itself, or the type that a kernel of the
    inner type gives (`Elements`). The kernels of the inner type follow the
    source:

    - `sum` counts a `Boolean`, widens an integer of 8 or 16 bits to `Int64`,
      and keeps the other numbers, a `Decimal`, a `Duration` and `Null`. It
      refuses every other type.
    - `mean` and `median` give a float type for a number, a `Datetime` in
      microseconds for a `Date`, and keep the other temporal types. The mean
      of `Float16` values panics, because that kernel has no arm for it.
    - `std` keeps a narrow float and a `Duration`, and gives `Float64`
      otherwise. `var` refuses a temporal type.
    - `min` and `max` keep the inner type, and refuse a nested type.
    - `diff` subtracts the values of each list (`Difference`).

    Example:
        >>> listed = Data(DType("List", (DType("Date"),)))
        >>> ListKernels.STD.output(listed), ListKernels.MEDIAN.output(listed)
        (Float64, Datetime(time_unit='us', time_zone=None))
    """

    LISTS: ClassVar[Kinds] = Kinds("List")
    WIDENED: ClassVar[Kinds] = Kinds(*Is.SMALL_INTEGERS)
    SUMMABLE: ClassVar[Kinds] = Kinds(
        *Is.SIGNED,
        *Is.UNSIGNED,
        *Is.FLOATS,
        "Decimal",
        "Duration",
        "Null",
    )
    NARROW_FLOATS: ClassVar[Kinds] = Kinds("Float16", "Float32")
    NESTED: ClassVar[Kinds] = Kinds("List", "Array", "Struct", "Object")
    MICROSECONDS: ClassVar[DType] = DType("Datetime", ("us", None))

    VALUE_SUM: ClassVar[Dispatch] = Dispatch(
        "sum",
        (
            Arm(Kinds("Boolean"), Fixed(DType.idx())),
            Arm(WIDENED, Fixed(DType("Int64"))),
            Arm(SUMMABLE, Same()),
        ),
        Refuse(),
    )
    VALUE_MEAN: ClassVar[Dispatch] = Dispatch(
        "mean",
        (
            Arm(Kinds("Float16"), Refuse(Stop.PANICS)),
            Arm(Kinds("Float32"), Same()),
            Arm(Kinds("Date"), Fixed(MICROSECONDS)),
            Arm(Kinds("Datetime", "Duration", "Time"), Same()),
        ),
        Fixed(DType("Float64")),
    )
    VALUE_MEDIAN: ClassVar[Dispatch] = Dispatch(
        "median",
        (
            Arm(NARROW_FLOATS, Same()),
            Arm(Kinds("Date"), Fixed(MICROSECONDS)),
            Arm(Kinds("Datetime", "Duration", "Time"), Same()),
        ),
        Fixed(DType("Float64")),
    )
    VALUE_STD: ClassVar[Dispatch] = Dispatch(
        "std",
        (Arm(NARROW_FLOATS, Same()), Arm(Kinds("Duration"), Same())),
        Fixed(DType("Float64")),
    )
    VALUE_VAR: ClassVar[Dispatch] = Dispatch(
        "var",
        (Arm(NARROW_FLOATS, Same()), Arm(Kinds(*Is.TEMPORAL), Refuse())),
        Fixed(DType("Float64")),
    )
    VALUE_EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "min",
        (Arm(NESTED, Refuse()),),
        Same(),
    )
    VALUE_DIFF: ClassVar[Dispatch] = Dispatch("diff", (), Difference())

    LENGTH: ClassVar[Dispatch] = Dispatch(
        "list.len",
        (Arm(LISTS, Fixed(DType.idx())),),
        Refuse(ErrorClass.SCHEMA),
    )
    SUM: ClassVar[Dispatch] = Dispatch(
        "list.sum",
        (Arm(LISTS, Elements(VALUE_SUM)),),
        Refuse(ErrorClass.SCHEMA),
    )
    MEAN: ClassVar[Dispatch] = Dispatch(
        "list.mean",
        (Arm(LISTS, Elements(VALUE_MEAN)),),
        Refuse(ErrorClass.SCHEMA),
    )
    MEDIAN: ClassVar[Dispatch] = Dispatch(
        "list.median",
        (Arm(LISTS, Elements(VALUE_MEDIAN)),),
        Refuse(ErrorClass.SCHEMA),
    )
    STD: ClassVar[Dispatch] = Dispatch(
        "list.std",
        (Arm(LISTS, Elements(VALUE_STD)),),
        Refuse(ErrorClass.SCHEMA),
    )
    VAR: ClassVar[Dispatch] = Dispatch(
        "list.var",
        (Arm(LISTS, Elements(VALUE_VAR)),),
        Refuse(ErrorClass.SCHEMA),
    )
    EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "list.min",
        (Arm(LISTS, Elements(VALUE_EXTREMUM)),),
        Refuse(ErrorClass.SCHEMA),
    )
    ARG_EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "list.arg_min",
        (Arm(LISTS, Fixed(DType.idx())),),
        Refuse(ErrorClass.SCHEMA),
    )
    DIFF: ClassVar[Dispatch] = Dispatch(
        "list.diff",
        (Arm(LISTS, Elements(VALUE_DIFF, listed=True)),),
        Refuse(ErrorClass.SCHEMA),
    )
    SAME: ClassVar[Dispatch] = Dispatch(
        "list.sort",
        (Arm(LISTS, Same()),),
        Refuse(ErrorClass.SCHEMA),
    )

    def fails_out_of_bounds(self, null_on_oob: bool) -> bool:
        """Say whether `list.get`, `list.gather` or `arr.get` can fail on one value.

        Where the option `null_on_oob` is off, an index outside a list raises.
        That is a fact of the data, and it makes polars run the node once for
        each group in `agg`.

        Args:
            null_on_oob: Whether an index outside a list gives a null.

        Returns:
            Whether the kernel can fail on one value.

        Example:
            >>> ListKernels().fails_out_of_bounds(False)
            True
        """
        return not null_on_oob

    def get(self, null_on_oob: bool) -> ElementGet:
        """Give the kernel of `list.get`, `list.first` and `list.last`.

        Args:
            null_on_oob: Whether an index outside a list gives a null.

        Returns:
            The kernel.

        Example:
            >>> ListKernels().get(null_on_oob=True)
            ElementGet(container='List', null_on_oob=True)
        """
        return ElementGet("List", null_on_oob=null_on_oob)

    def sample(self, *, fraction: bool, replaces: bool) -> Sampled:
        """Give the kernel of `list.sample`.

        Args:
            fraction: Whether the second input is a fraction, and not a number of values.
            replaces: Whether the sample can take a value more than once.

        Returns:
            The kernel.

        Example:
            >>> ListKernels().sample(fraction=True, replaces=False)
            Sampled(fraction=True, replaces=False)
        """
        return Sampled(fraction=fraction, replaces=replaces)

    def to_array(self, width: int) -> Dispatch:
        """Give the kernel of `list.to_array`.

        A first input that is not a list raises a `ComputeError`.

        Args:
            width: The width of the arrays.

        Returns:
            The dispatch.

        Example:
            >>> ListKernels().to_array(2).output(Data(DType("Int8")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        arms = (Arm(self.LISTS, AsArray(width)),)
        return Dispatch("list.to_array", arms, Refuse(ErrorClass.COMPUTE))

    def to_struct(self, names: Sequence[str]) -> Dispatch:
        """Give the kernel of `list.to_struct`.

        Args:
            names: The names of the fields.

        Returns:
            The dispatch.

        Example:
            >>> ListKernels().to_struct(("a",)).output(Data(DType("Int8")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        arms = (Arm(self.LISTS, NamedFields(names)),)
        return Dispatch("list.to_struct", arms, Refuse(ErrorClass.SCHEMA))


class Container:
    """Read the first input of a function of the `list` or the `arr` namespace as its kind of data.

    A function of the `list` namespace reads its first input as a `List`
    column, and a function of the `arr` namespace as an `Array` column. Data
    of another kind raises a `SchemaError`.

    Example:
        >>> Container("List").checked(Data(DType("List", (DType("Int8"),))))
        Data(dtype=List(Int8), nulls=False, one_row=False)
        >>> Container("Array").checked(Data(DType("Int8")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
    """

    def __init__(self, kind: str) -> None:
        """Keep the kind of data that the function reads.

        Args:
            kind: `List` or `Array`.

        Example:
            >>> Container("List")._kind
            'List'
        """
        self._kind = kind

    def checked(self, data: Data) -> Data:
        """Give the data, or refuse data of another kind.

        Args:
            data: The data of the first input.

        Returns:
            The same data.

        Raises:
            PolarsRefusal: The data is not of the kind.

        Example:
            >>> Container("Array").checked(Data(DType("Array", (DType("Int8"), 2)))).dtype
            Array(Int8, shape=(2,))
        """
        if data.dtype.name != self._kind:
            message = f"invalid series dtype: expected `{self._kind}`, got `{data.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return data


@dataclass(frozen=True)
class CastInto:
    """Cast an argument of a kernel to one type, and read it as that type, as a kernel does.

    A kernel such as `list.get` casts its index to `Int64` with a cast that is
    not strict (`RunCast`), and then reads the result as `Int64` data. A cast
    with no kernel raises its error. A struct casts field by field, so it
    stays a struct, and the read then fails: with a panic where `unwrap` is
    set, and with a `SchemaError` otherwise.

    Example:
        >>> CastInto(DType("Int64")).of(Data(DType("Date"))).dtype
        Int64
        >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
        >>> CastInto(DType("Int64"), unwrap=True).of(struct)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    target: DType
    unwrap: bool = False

    def of(self, data: Data) -> Data:
        """Give the data after the cast.

        It raises `PolarsRefusal` where the cast has no kernel.

        Args:
            data: The data of the argument.

        Returns:
            The data, of the target type.

        Raises:
            PolarsRefusal: The cast gives another type.

        Example:
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> CastInto(DType("Int64")).of(struct)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        cast = RunCast().of(data, self.target)
        if cast.dtype.same(self.target):
            return cast
        message = f"invalid series dtype: expected `{self.target!r}`, got `{cast.dtype!r}`"
        raise PolarsRefusal(Stop.PANICS if self.unwrap else ErrorClass.SCHEMA, message)


class Spread:
    """Say whether two inputs of a kernel have lengths that it takes: equal, or one of them 1.

    Many kernels spread an input of one row to the length of the other input,
    and refuse two other lengths. Where the code that evaluates the node does
    not know the lengths, they fit.

    Example:
        >>> Spread().fits([5, 1], 0, 1), Spread().fits([5, 2], 0, 1), Spread().fits([], 0, 1)
        (True, False, True)
    """

    def fits(self, lengths: Sequence[int], first: int, second: int) -> bool:
        """Say whether the lengths of two inputs fit.

        Args:
            lengths: The number of rows of each input, or none where they are not known.
            first: The place of the first input.
            second: The place of the second input.

        Returns:
            Whether they fit.

        Example:
            >>> Spread().fits([2, 2, 3], 0, 1)
            True
        """
        if (  # pragma: no cover -- the run passes the rows of each input
            len(lengths) <= max(first, second)
        ):
            return True
        pair = (lengths[first], lengths[second])
        return pair[0] == pair[1] or 1 in pair


@dataclass(frozen=True)
class ElementGet:
    """Give the type of `list.get` and `arr.get`: the value at an index of each list or array.

    The kernel reads its first input as a list or an array (`Container`), and
    casts the index to `Int64` (`CastInto`). An index of one row reads one
    place of every list, an index of the length of the lists reads one place
    of each list, and a list of one row reads each index. Other lengths raise
    a `ComputeError`. The value has the inner type. Whether the index is
    inside each list is a fact of the data.

    An array has a width in its type, so where `null_on_oob` is off, the
    kernel of arrays refuses an index outside the width that the code fixes:
    a literal index, at every run. It reads an index of the length of the
    arrays as places, and a null place is outside the array, so an index that
    holds only nulls raises too.

    Example:
        >>> lists = Data(DType("List", (DType("Date"),)))
        >>> ElementGet("List", null_on_oob=False).output([lists, Data(DType("UInt8"))], [5, 1], [])
        Date
        >>> ElementGet("List", null_on_oob=False).output([lists, Data(DType("UInt8"))], [5, 2], [])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    container: str
    null_on_oob: bool

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not of the kind,
        where the cast of the index fails, or where an array refuses its
        index (`_places`).

        Args:
            inputs: The data of the lists and of the index.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where they are not known.

        Returns:
            The inner type.

        Raises:
            PolarsRefusal: The index and the lists have lengths that do not fit.

        Example:
            >>> arrays = Data(DType("Array", (DType("Int8"), 2)))
            >>> get = ElementGet("Array", null_on_oob=False)
            >>> get.output([arrays, Data(DType("String"))], [], [])
            Int8
        """
        values = Container(self.container).checked(inputs[0])
        index = CastInto(DType("Int64"), unwrap=True).of(inputs[1])
        if not Spread().fits(lengths, 0, 1):
            message = f"the index has {lengths[1]} rows, and the input has {lengths[0]}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if self.container == "Array" and not self.null_on_oob:
            self._places(values.dtype, index, lengths, computed)
        return values.dtype.inner()

    def _places(
        self,
        dtype: DType,
        index: Data,
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> None:
        """Refuse an index that the kernel of arrays reads outside each array, at every run.

        Args:
            dtype: The array type.
            index: The data of the index, as `Int64`.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where they are not known.

        Raises:
            PolarsRefusal: A literal index lies outside the width, or an index of
                places holds only nulls.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> arrays = DType("Array", (DType("Int8"), 2))
            >>> get = ElementGet("Array", null_on_oob=False)
            >>> literal = Operand.of_value(-3)
            >>> get._places(arrays, Data(DType("Int64")), [5, 1], [Operand.column(arrays), literal])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: get index is out of bounds
        """
        literal = len(lengths) < 2 or lengths[1] == 1
        places = not literal and lengths[0] == lengths[1]
        value = LiteralIndex().of(computed[1]) if literal and len(computed) > 1 else None
        width = Parameter().integer(dtype, 1)
        outside = value is not None and not -width <= value < width
        if outside or (places and index.nulls):
            message = "get index is out of bounds"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class LiteralIndex:
    """Give the value of a literal index after the cast of a kernel to `Int64`, where it is known.

    The cast is not strict: a float drops its fraction, `True` is 1, and text
    gives its integer, or a null where it does not parse as one.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> LiteralIndex().of(Operand.of_value(1.5)), LiteralIndex().of(Operand.of_value("a"))
        (1, None)
    """

    def of(self, operand: Operand) -> int | None:
        """Give the integer value.

        Args:
            operand: The index as the run computes it.

        Returns:
            The integer, or `None` for a column, a null or text that is not an integer.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> LiteralIndex().of(Operand.of_value(True)), LiteralIndex().of(Operand.of_value("-2"))
            (1, -2)
            >>> LiteralIndex().of(Operand.of_value(None)) is None
            True
            >>> LiteralIndex().of(Operand(DType("Int8"), literal=True, value=(1,))) is None
            True
        """
        value = operand.value
        if not operand.literal or value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            digits = text.lstrip("+-")
            return int(text) if digits.isdigit() else None
        if isinstance(value, bool | int | float):
            return int(value)
        return None  # pragma: no cover -- the type function refuses an index of another type


@dataclass(frozen=True)
class Shifted:
    """Give the type of `list.shift` and `arr.shift`: each list or array, shifted by a number.

    The kernel reads its first input as a list or an array, and casts the
    number to `Int64`. The two inputs have one length, or one of them has one
    row. Other lengths raise a `ShapeError`. The output has the type of the
    first input.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> Shifted("List").output([lists, Data(DType("Boolean"))], [5, 1])
        List(Int8)
    """

    container: str

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not of the kind,
        or where the cast of the number has no kernel.

        Args:
            inputs: The data of the lists and of the number.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type of the first input.

        Raises:
            PolarsRefusal: The two inputs have lengths that do not fit.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> Shifted("List").output([lists, Data(DType("Int8"))], [5, 2])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        values = Container(self.container).checked(inputs[0])
        CastInto(DType("Int64")).of(inputs[1])
        if not Spread().fits(lengths, 0, 1):
            message = "the number of the shift and the input have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return values.dtype


@dataclass(frozen=True)
class Joined:
    """Give the type of `list.join` and `arr.join`: the text of each list, joined by a separator.

    The kernel reads its first input as a list or an array, and its
    separator as `String` data, or raises a `SchemaError`. The inner type must
    be `String`, or the kernel raises an `InvalidOperationError`. A separator of
    one row joins every list. Where the separator has more rows, the kernel of
    arrays checks that it has the length of the arrays, and raises a
    `ShapeError` otherwise. The kernel of lists pairs the rows with no check.

    Example:
        >>> texts = Data(DType("List", (DType("String"),)))
        >>> Joined("List", checks_lengths=False).output([texts, Data(DType("String"))], [5, 2])
        String
    """

    container: str
    checks_lengths: bool

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the lists and of the separator.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `String`.

        Raises:
            PolarsRefusal: The separator is not `String` data, the inner type
                is not `String`, or the separator has a length that does not fit.

        Example:
            >>> numbers = Data(DType("Array", (DType("Int8"), 2)))
            >>> Joined("Array", checks_lengths=True).output([numbers, Data(DType("String"))], [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> texts = Data(DType("Array", (DType("String"), 2)))
            >>> Joined("Array", checks_lengths=True).output([texts, Data(DType("String"))], [5, 2])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: the separator and the input ...
        """
        values = Container(self.container).checked(inputs[0])
        Container("String").checked(inputs[1])
        if values.dtype.inner().name != "String":
            message = f"join is not supported for the inner type {values.dtype.inner()!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        separated = len(lengths) > 1 and lengths[1] != 1
        if (  # pragma: no cover -- the run checks the lengths first
            self.checks_lengths and separated and lengths[0] != lengths[1]
        ):
            message = "the separator and the input have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return DType("String")


@dataclass(frozen=True)
class CountMatches:
    """Give the type of `list.count_matches` and `arr.count_matches`: how often a value occurs.

    The kernel first needs exactly one value to count, or it raises a
    `ComputeError`. It then reads its first input as a list or an array, and
    builds a series of one row from the value. A category that is not null
    has no data type of its own there, so the kernel panics. It compares the
    values inside each list with that series, as `eq_missing` of two series
    does (`Comparison`). A comparison that fails raises its own error. The
    kernel counts the matches, so the output is `UInt32`.

    Example:
        >>> texts = Data(DType("List", (DType("String"),)))
        >>> CountMatches("List").output([texts, Data(DType("String"))], [5, 1])
        UInt32
        >>> CountMatches("List").output([texts, Data(DType("Int32"))], [5, 1])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    container: str

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not of the kind,
        or where the comparison refuses the two types.

        Args:
            inputs: The data of the lists and of the value.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `UInt32`.

        Raises:
            PolarsRefusal: The value does not have exactly one row.

        Example:
            >>> numbers = Data(DType("List", (DType("Int64"),)))
            >>> CountMatches("List").output([numbers, Data(DType("Int32"))], [5, 3])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        if len(lengths) > 1 and lengths[1] != 1:
            message = f"the value to count must have exactly one row, got {lengths[1]}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        values = Container(self.container).checked(inputs[0])
        element = replace(inputs[1], one_row=True)
        if element.dtype.name in Is.CATEGORIES and not element.nulls:
            message = f"can not get dtype of {element.dtype.name} AnyValue"
            raise PolarsRefusal(Stop.PANICS, message)
        Comparison(CompareOp.EQUAL_MISSING).output(Data(values.dtype.inner()), element)
        return DType.idx()


class IsIn:
    """Give the type of the test of a value in a list or an array, as `is_in` and `contains` run it.

    The kernel takes a value and a collection: a list or an array of values.
    The two have one length, or one of them has one row, or the kernel raises
    a `ShapeError`. A collection that is not a list or an array raises an
    `InvalidOperationError`. The kernel then picks its work by the type of the
    value, and some types need a fitting inner type of the collection:

    - text needs text or categories, binary data needs binary data, and
      `Boolean` needs `Boolean`;
    - a category needs text or the same kind of category, and a `Decimal`
      needs a `Decimal`;
    - `Null` and a nested type take any inner type, because they encode
      each row;
    - a number or a temporal type reads the inner values as its own
      physical type, and polars panics where the inner physical type
      differs, unless the collection is one row of null.

    Every other pair raises an `InvalidOperationError`. The cast pass of the
    node brings most pairs to one type before the kernel runs. Two integer
    literals with no data type keep the types that they take alone, so
    `pl.lit(2**40).is_in(pl.lit(1))` panics.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> IsIn().output(Data(DType("Int8")), lists, [1, 5])
        Boolean
        >>> IsIn().output(Data(DType("String")), lists, [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    INNER: ClassVar[dict[str, frozenset[str]]] = {
        "String": frozenset({"String", "Enum", "Categorical"}),
        "Binary": frozenset({"Binary"}),
        "Boolean": frozenset({"Boolean"}),
        "Categorical": frozenset({"String", "Categorical"}),
        "Enum": frozenset({"String", "Enum"}),
        "Decimal": frozenset({"Decimal"}),
    }

    def output(self, value: Data, collection: Data, lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            value: The data of the values to look for.
            collection: The data of the lists or the arrays.
            lengths: The number of rows of the value and of the collection,
                or none where they are not known.

        Returns:
            `Boolean`.

        Raises:
            PolarsRefusal: The lengths do not fit, the collection is not a list
                or an array, or its inner type does not fit the value.

        Example:
            >>> arrays = Data(DType("Array", (DType("Int8"), 2)))
            >>> IsIn().output(Data(DType("Int8")), arrays, [5, 2])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
            >>> int8 = Data(DType("Int8"))
            >>> IsIn().output(int8, int8, [5, 5])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: is_in cannot check ...
            >>> wide = Data(DType("Int64"))
            >>> IsIn().output(wide, Data(DType("List", (DType("Int32"),))), [5, 5])
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: cannot get ref Int64 from Int32
            >>> null = Data(DType("List", (DType("Int32"),)), nulls=True)
            >>> IsIn().output(wide, null, [5, 1])
            Boolean
        """
        if not Spread().fits(lengths, 0, 1):
            message = "the values and the collection have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        if (  # pragma: no cover -- the cast pass implodes the collection
            collection.dtype.name not in Is.LISTS
        ):
            message = f"is_in cannot check {value.dtype!r} values in {collection.dtype!r} data"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        accepted = self.INNER.get(value.dtype.name)
        if (  # pragma: no cover -- the cast pass casts the collection first
            accepted is not None and collection.dtype.inner().name not in accepted
        ):
            message = f"is_in cannot check {value.dtype!r} values in {collection.dtype!r} data"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        one_row = collection.one_row or (len(lengths) > 1 and lengths[1] == 1)
        if accepted is None and not (one_row and collection.nulls):
            self._physical(value.dtype.physical(), collection.dtype.inner().physical())
        return DType("Boolean")

    def _physical(self, value: DType, inner: DType) -> None:
        """Refuse numbers whose physical type differs from the physical type of the inner values.

        Args:
            value: The physical type of the values to look for.
            inner: The physical type of the inner values of the collection.

        Raises:
            PolarsRefusal: The value is a number of another physical type than
                the inner values, where polars panics.

        Example:
            >>> IsIn()._physical(DType("Int32"), DType("Int32")) is None
            True
            >>> IsIn()._physical(DType("Int64"), DType("Int32"))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: cannot get ref Int64 from Int32
        """
        if Is.primitive_numeric(value) and not value.same(inner):
            raise PolarsRefusal(Stop.PANICS, f"cannot get ref {value!r} from {inner!r}")


@dataclass(frozen=True)
class Membership:
    """Give the type of `is_in`: whether each value is in its collection.

    The cast pass turns the second input into a list or an array. The kernel
    then tests each value of the first input in the collection (`IsIn`).

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> Membership().output([Data(DType("Int8")), lists], [5, 1])
        Boolean
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the test refuses the two inputs.

        Args:
            inputs: The data of the values and of the collection.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `Boolean`.

        Example:
            >>> wide = Data(DType("Int64"))
            >>> Membership().output([wide, Data(DType("List", (DType("Int32"),)))], [1, 1])
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: cannot get ref Int64 from Int32
        """
        return IsIn().output(inputs[0], inputs[1], lengths)


@dataclass(frozen=True)
class Contains:
    """Give the type of `list.contains` and `arr.contains`: whether each list holds a value.

    The kernel reads its first input as a list or an array, or raises a
    `SchemaError`. Then it tests the value in the collection (`IsIn`).

    Example:
        >>> lists = Data(DType("List", (DType("Date"),)))
        >>> Contains("List").output([lists, Data(DType("Date"))], [5, 1])
        Boolean
    """

    container: str

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not of the kind,
        or where the test refuses the two inputs.

        Args:
            inputs: The data of the collection and of the value.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `Boolean`.

        Example:
            >>> Contains("List").output([Data(DType("Int8")), Data(DType("Int8"))], [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        collection = Container(self.container).checked(inputs[0])
        return IsIn().output(inputs[1], collection, tuple(reversed(lengths)))


@dataclass(frozen=True)
class Gathered:
    """Give the type of `list.gather`: the values at a list of indices, for each list.

    The cast pass puts an index that is not a list in a list. The kernel
    reads both inputs as lists, or raises a `SchemaError`. The indices must
    be integers, or it raises a `ComputeError`. A list of one row reads each
    list of indices, a list of indices of one row reads every list, and two
    inputs of one length pair their rows. Other lengths raise a `ShapeError`.
    One list of indices for every list must hold signed integers, `UInt32`
    or `UInt64`, and other unsigned integers raise a `ComputeError`. The
    output has the type of the lists.

    Example:
        >>> lists = Data(DType("List", (DType("Date"),)))
        >>> indices = Data(DType("List", (DType("Int64"),)))
        >>> Gathered().output([lists, indices], [5, 1])
        List(Date)
    """

    UNSIGNED: ClassVar[frozenset[str]] = frozenset({"UInt32", "UInt64"})

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where an input is not a list.

        Args:
            inputs: The data of the lists and of the indices.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type of the lists.

        Raises:
            PolarsRefusal: The indices are not integers, or the lengths do not fit.

        Example:
            >>> lists = Data(DType("List", (DType("Date"),)))
            >>> texts = Data(DType("List", (DType("String"),)))
            >>> Gathered().output([lists, texts], [])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        values = Container("List").checked(inputs[0])
        indices = Container("List").checked(inputs[1])
        inner = indices.dtype.inner()
        if not Is.integer(inner):  # pragma: no cover -- the type function refuses it first
            message = f"cannot use dtype `{inner!r}` as an index"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if not Spread().fits(lengths, 0, 1):
            message = "the lists and the indices have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        shared = len(lengths) > 1 and lengths[0] != 1 and lengths[1] == 1
        if shared and not (Is.signed(inner) or inner.name in self.UNSIGNED):
            message = f"cannot use dtype `{inner!r}` as an index"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return values.dtype


class GatherEvery:
    """Give the type of `list.gather_every`: every n-th value of each list, from an offset.

    The kernel casts the step and the offset to `UInt32` with a strict cast
    (`StrictCount`), then reads its first input as a list. The step and the
    offset each have one row, or the length of the lists, or the kernel raises
    a `ComputeError`. The output has the type of the lists. Where the kernel
    reads a step or an offset row by row and every value is null, every row
    is null, and the output is a list of `Null`.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> numbers = [lists, Data(DType("Int64")), Data(DType("Int64"))]
        >>> GatherEvery().output(numbers, [5, 1, 5], [])
        List(Int8)
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast fails, or where the first input
        is not a list.

        Args:
            inputs: The data of the lists, of the step and of the offset.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where they are not known.

        Returns:
            The type of the lists.

        Raises:
            PolarsRefusal: The step or the offset has a length that does not fit.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> numbers = [lists, Data(DType("Int32")), Data(DType("Int32"))]
            >>> step = [Operand.column(lists.dtype), Operand.of_value(-1), Operand.of_value(0)]
            >>> GatherEvery().output(numbers, [5, 1, 1], step)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        count = StrictCount(DType.idx())
        count.check(inputs[1], computed[1] if computed else None)
        count.check(inputs[2], computed[2] if computed else None)
        values = Container("List").checked(inputs[0])
        rows, steps, offsets = tuple(lengths) if len(lengths) > 2 else (1, 1, 1)
        if not self._fits(rows, steps, offsets):
            message = "the step and the offset must have one row or the length of the lists"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if (steps, offsets) == (1, 1):
            return values.dtype
        nulled = (steps != 1 and inputs[1].nulls) or (offsets != 1 and inputs[2].nulls)
        return Sliced.EMPTY if nulled or values.nulls else values.dtype

    def _fits(self, rows: int, steps: int, offsets: int) -> bool:
        """Say whether the step and the offset have lengths that the kernel takes.

        Args:
            rows: The number of lists.
            steps: The number of rows of the step.
            offsets: The number of rows of the offset.

        Returns:
            Whether each has one row or the number of lists.

        Example:
            >>> GatherEvery()._fits(5, 5, 1), GatherEvery()._fits(1, 2, 2)
            (True, False)
        """
        return steps in {1, rows} and offsets in {1, rows}


@dataclass(frozen=True)
class StrictCount:
    """Cast a count of a kernel to its type with a strict cast, as `list.gather_every` does.

    The cast must exist for the type of the data, and the kernel then reads
    the result as that type (`CastInto`). A strict cast raises on a value that
    does not fit. The value of a column is a fact of the data, and the value of
    a literal is in the code, so a literal that does not fit raises at every
    run, as `-1` does for a count of `UInt32`.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> StrictCount(DType.idx()).check(Data(DType("Int32")), Operand.of_value(2))
        >>> StrictCount(DType.idx()).check(Data(DType("String")), Operand.of_value("a"))
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    target: DType

    def check(self, data: Data, operand: Operand | None) -> None:
        """Refuse a count that the strict cast fails on at every run.

        It raises `PolarsRefusal` where the cast has no kernel, or where a
        literal value does not fit.

        Args:
            data: The data of the count.
            operand: The count as the run computes it, or `None` where it is not known.

        Example:
            >>> StrictCount(DType.idx()).check(Data(DType("Date")), None)
        """
        CastInto(self.target).of(data)
        if operand is not None and operand.literal and not data.nulls:
            SeriesCast(data.dtype, self.target, strict=True).output([operand.value])


@dataclass(frozen=True)
class Sampled:
    """Give the type of `list.sample`: a sample of the values of each list.

    The kernel reads its first input as a list. A sample of a number of values
    casts the number to `UInt32` with a strict cast (`StrictCount`). A sample
    of a fraction casts the fraction to `Float64`, and a fraction outside 0 to
    1 raises a `ComputeError` where the sample takes each value once. The two
    inputs have one length, or one of them has one row, or the kernel raises a
    `ShapeError`. The value of a literal is in the code, so a literal that does
    not fit raises at every run. A number larger than a list is a fact of the
    data. The output has the type of the lists.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> sample = Sampled(fraction=False, replaces=False)
        >>> sample.output([lists, Data(DType("Int64"))], [5, 1], [])
        List(Int8)
    """

    fraction: bool
    replaces: bool

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not a list, or where
        the cast of the number or of the fraction fails.

        Args:
            inputs: The data of the lists and of the number or the fraction.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where they are not known.

        Returns:
            The type of the lists.

        Raises:
            PolarsRefusal: The two inputs have lengths that do not fit.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> half = [Operand.column(lists.dtype), Operand.of_value(1.5)]
            >>> share = Sampled(fraction=True, replaces=False)
            >>> share.output([lists, Data(DType("Float64"))], [5, 1], half)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        values = Container("List").checked(inputs[0])
        size = computed[1] if computed else None
        if self.fraction:
            CastInto(DType("Float64")).of(inputs[1])
            self._fraction(size)
        else:
            StrictCount(DType.idx()).check(inputs[1], size)
        if not Spread().fits(lengths, 0, 1):
            message = "the size of the sample and the lists have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return values.dtype

    def _fraction(self, size: Operand | None) -> None:
        """Refuse a literal fraction outside 0 to 1, where the sample takes each value once.

        Args:
            size: The fraction as the run computes it, or `None` where it is not known.

        Raises:
            PolarsRefusal: The fraction is a literal outside 0 to 1.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Sampled(fraction=True, replaces=True)._fraction(Operand.of_value(3))
        """
        value = None if size is None or not size.literal else size.value
        if self.replaces or isinstance(value, str) or not isinstance(value, int | float):
            return
        if not 0.0 <= float(value) <= 1.0:
            message = f"fraction must be between 0.0 and 1.0, got: {value}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class Sliced:
    """Give the type of `list.slice`: a slice of each list, from an offset and of a length.

    The kernel reads its first input as a list. It then picks a path by the
    lengths of its three inputs, and each path reads the offset and the
    length in its own way:

    - a list of one row beside a longer offset or length gives one slice for
      each row. The kernel casts the length and then the offset to `Int64`;
    - an offset and a length of one row each give one slice for every list.
      The kernel reads the offset as one number (`_number`);
    - an offset of one row beside a longer length: the length must have the
      length of the lists, or the kernel raises a `ComputeError`. It reads
      the offset as one number, and casts the length to `Int64`;
    - a longer offset beside a length of one row: the offset must have the
      length of the lists, and the kernel casts it to `Int64`;
    - otherwise both must have the length of the lists, and the kernel casts
      both to `Int64`.

    A cast reads its result as `Int64` (`CastInto`). Where the source reads
    it with no check, a struct, which casts field by field, panics. The
    output has the type of the lists. Where the kernel slices row by row and
    every slice is null, because the lists, the offsets or the lengths are all
    null, the output is a list of `Null`.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> number = Data(DType("Int64"))
        >>> Sliced().output([lists, number, number], [5, 1, 1], [])
        List(Int8)
        >>> Sliced().output([lists, Data(DType("Date")), number], [5, 1, 1], [])
        List(Int8)
    """

    NUMBERS: ClassVar[frozenset[str]] = frozenset({
        *Is.SIGNED,
        *Is.UNSIGNED,
        *Is.FLOATS,
        *Is.TEMPORAL,
        "Decimal",
        "Boolean",
        "String",
    })
    INT64: ClassVar[DType] = DType("Int64")
    READ: ClassVar[CastInto] = CastInto(INT64, unwrap=True)
    CHECKED: ClassVar[CastInto] = CastInto(INT64)
    EMPTY: ClassVar[DType] = DType("List", (DType("Null"),))

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the first input is not a list, where an
        argument has a length that does not fit, or where a cast fails.

        Args:
            inputs: The data of the lists, of the offset and of the length.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where they are not known.

        Returns:
            The type of the lists.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> number = Data(DType("Int64"))
            >>> Sliced().output([lists, number, number], [5, 2, 1], [])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
            >>> Sliced().output([lists, number, number], [1, 3, 1], [])
            List(Int8)
        """
        values = Container("List").checked(inputs[0])
        rows, offsets, counts = tuple(lengths) if len(lengths) > 2 else (1, 1, 1)
        literal = computed[1] if computed else None
        offset = self._lost(inputs[1], literal)
        count = self._lost(inputs[2], computed[2] if computed else None)
        if rows == 1 and max(offsets, counts) > 1:
            return self._spread(values, offset, count, (offsets, counts))
        self._shape(offsets, rows)
        self._shape(counts, rows)
        read = inputs[1] if offsets == 1 else offset
        self._read(read, count, (offsets, counts), literal)
        return self._sliced(values, offset, count, (offsets, counts))

    def _read(
        self,
        offset: Data,
        count: Data,
        lengths: tuple[int, int],
        literal: Operand | None,
    ) -> None:
        """Read the offset and the length on the path that their lengths pick.

        It raises `PolarsRefusal` where a read or a cast fails.

        Args:
            offset: The data of the offset: as the code gives it for an offset of
                one row, which the kernel reads as a value, and after the cast to
                `Int64` for a longer one.
            count: The data of the length.
            lengths: The number of rows of the offset and of the length.
            literal: The offset as the run computes it, or `None` where it is not known.

        Example:
            >>> number = Data(DType("Int64"))
            >>> Sliced()._read(number, number, (1, 1), None)
        """
        offsets, counts = lengths
        if offsets == 1:
            self._number(offset, literal)
        else:
            (self.READ if counts == 1 else self.CHECKED).of(offset)
        if counts != 1:
            self.READ.of(count)

    def _sliced(self, values: Data, offset: Data, count: Data, lengths: tuple[int, int]) -> DType:
        """Give the type of the slices: a list of `Null` where each slice of a row is null.

        Args:
            values: The data of the lists.
            offset: The data of the offset.
            count: The data of the length.
            lengths: The number of rows of the offset and of the length.

        Returns:
            The type.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> nulls = Data(DType("Int64"), nulls=True)
            >>> Sliced()._sliced(lists, nulls, Data(DType("Int64")), (5, 1))
            List(Null)
        """
        offsets, counts = lengths
        if (offsets, counts) == (1, 1):
            return values.dtype
        nulled = (offsets != 1 and offset.nulls) or (counts != 1 and count.nulls)
        return self.EMPTY if nulled or values.nulls else values.dtype

    def _spread(
        self,
        values: Data,
        offset: Data,
        count: Data,
        lengths: tuple[int, int],
    ) -> DType:
        """Give the type of the slices of one list, for each row of a longer offset or length.

        The kernel reads the offset and the length at each row of the longest
        of the two. A shorter one of more than one row runs out of rows, and the
        kernel panics. The kernel collects the slices into a new list, which
        holds the physical type of the values: a list of `Date` gives a list of
        `Int32`. A null list gives null lists of its own type. Where the offset
        or the length is a column, a frame of one row gives it one row, and the
        slice then keeps the type of the values. So the number of rows decides
        a type whose physical type differs.

        Args:
            values: The data of the list.
            offset: The data of the offset.
            count: The data of the length.
            lengths: The number of rows of the offset and of the length.

        Returns:
            The type of the list, or a list of `Null` where every slice is
            null.

        Raises:
            PolarsRefusal: The offset or the length has rows that do not fit.
            UndecidedError: The number of rows decides the type.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> Sliced()._spread(lists, Data(DType("Null")), Data(DType("Int8")), (1, 3))
            List(Null)
            >>> days = Data(DType("List", (DType("Date"),)))
            >>> Sliced()._spread(days, Data(DType("Int8")), Data(DType("Int8")), (3, 1))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: rows: the slice of one list of Date
            >>> Sliced()._spread(lists, Data(DType("Int8")), Data(DType("Int8")), (2, 3))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: index out of bounds ...
        """
        self.READ.of(count)
        self.READ.of(offset)
        target = max(lengths)
        if any(length not in {1, target} for length in lengths):
            message = f"index out of bounds for the {target} rows of the slice"
            raise PolarsRefusal(Stop.PANICS, message)
        if values.nulls:
            return values.dtype
        if offset.nulls or count.nulls:
            return self.EMPTY
        inner = values.dtype.inner()
        if inner.physical() != inner:
            raise UndecidedError(Undecided.ROWS, f"the slice of one list of {inner!r}")
        return values.dtype

    def _lost(self, data: Data, literal: Operand | None) -> Data:
        """Give the data of an argument, with the nulls of a literal that the cast to `Int64` loses.

        The cast is not strict, so a literal text that is not an integer becomes
        a null at every run. A literal that the cast refuses, as a list, keeps
        its data, so the cast of the kernel raises on its type.

        Args:
            data: The data of the argument.
            literal: The argument as the run computes it, or `None` where it is not known.

        Returns:
            The data.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Sliced()._lost(Data(DType("String")), Operand.of_value("a")).nulls
            True
            >>> listed = Operand(DType("List", (DType("Int64"),)), literal=True, value=[1])
            >>> Sliced()._lost(Data(listed.dtype), listed).nulls
            False
        """
        if literal is None or not literal.literal or not CastFold().nulled(literal, self.INT64):
            return data
        return replace(data, nulls=True)

    def _shape(self, length: int, rows: int) -> None:
        """Refuse an argument of more than one row that does not have the length of the lists.

        Args:
            length: The number of rows of the argument.
            rows: The number of lists.

        Raises:
            PolarsRefusal: The argument has another length.

        Example:
            >>> Sliced()._shape(1, 5)
        """
        if length not in {1, rows}:
            message = f"an argument of the slice has {length} rows, and the lists {rows}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _number(self, offset: Data, literal: Operand | None) -> None:
        """Read an offset of one row as one number, as the kernel reads a value.

        A number, a temporal value, a `Decimal` and a `Boolean` give a number,
        and so does text that parses as one. Any other type, and a null,
        raise a `ComputeError`. Whether the text of a column parses is a fact of
        the data, and the text of a literal is in the code. Polars names the
        type of the value in that error, and a categorical value has no type of
        its own there, so the kernel panics on it.

        Args:
            offset: The data of the offset.
            literal: The offset as the run computes it, or `None` where it is not known.

        Raises:
            PolarsRefusal: The offset is not a number, or holds a null.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Sliced()._number(Data(DType("String")), Operand.of_value("a"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
        """
        text = literal.value if literal is not None and literal.literal else None
        unparsed = isinstance(text, str) and not self._parses(text)
        if not offset.nulls and offset.dtype.name in {"Categorical", "Enum"}:
            message = f"can not get dtype of {offset.dtype.name} AnyValue"
            raise PolarsRefusal(Stop.PANICS, message)
        if unparsed or offset.nulls or offset.dtype.name not in self.NUMBERS:
            message = f"could not extract number from any-value of dtype: '{offset.dtype!r}'"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _parses(self, text: str) -> bool:
        """Say whether text parses as an integer or a float, as the kernel reads a number of text.

        Args:
            text: The text.

        Returns:
            Whether it parses.

        Example:
            >>> Sliced()._parses("1.5"), Sliced()._parses("a")
            (True, False)
        """
        try:
            float(text)
        except ValueError:
            return False
        return True


@dataclass(frozen=True)
class SetOperated:
    """Give the type of the set operations of `list`, such as `list.set_union`.

    The kernel reads both inputs as lists, or raises a `SchemaError`. The two
    have one length, or one of them has one row, or it raises a `ShapeError`.
    They must have one type, or it raises an `InvalidOperationError`: the cast
    pass brings them to their supertype first. The kernel then works on the
    physical values. Text and binary data work, `Boolean` raises an
    `InvalidOperationError`, and a number, a temporal type, a `Decimal` or a
    category works on its physical numbers. Any other inner type panics. The
    output has the type of the first input.

    Example:
        >>> lists = Data(DType("List", (DType("Date"),)))
        >>> SetOperated().output([lists, lists], [5, 1])
        List(Date)
        >>> nested = Data(DType("List", (DType("List", (DType("Int8"),)),)))
        >>> SetOperated().output([nested, nested], [])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    PHYSICAL: ClassVar[frozenset[str]] = frozenset({
        *Is.SIGNED,
        *Is.UNSIGNED,
        *Is.FLOATS,
        *Is.TEMPORAL,
        *Is.CATEGORIES,
        "Decimal",
        "String",
        "Binary",
    })

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where an input is not a list.

        Args:
            inputs: The data of the two lists.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type of the first input.

        Raises:
            PolarsRefusal: The lengths do not fit, the types differ, or the
                kernel does not take the inner type.

        Example:
            >>> flags = Data(DType("List", (DType("Boolean"),)))
            >>> SetOperated().output([flags, flags], [])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> numbers = Data(DType("List", (DType("Int8"),)))
            >>> SetOperated().output([numbers, Data(DType("List", (DType("Int16"),)))], [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot do 'set' ...
        """
        left = Container("List").checked(inputs[0])
        right = Container("List").checked(inputs[1])
        if not Spread().fits(lengths, 0, 1):
            message = "the two lists have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        if (  # pragma: no cover -- the cast pass casts both lists first
            not left.dtype.same(right.dtype)
        ):
            message = f"cannot do 'set' operation on dtypes: {left.dtype!r} and {right.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self._null_spread(left, right, lengths):
            message = "validity mask length must match the number of values"
            raise PolarsRefusal(Stop.PANICS, message)
        inner = left.dtype.inner()
        if inner.name == "Boolean":
            message = "boolean type not yet supported in list 'set' operations"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if inner.name not in self.PHYSICAL:
            raise PolarsRefusal(Stop.PANICS, f"not implemented for dtype {inner!r}")
        return left.dtype

    def _null_spread(self, left: Data, right: Data, lengths: Sequence[int]) -> bool:
        """Say whether a side of one null row meets a longer side, which panics.

        The kernel joins the masks of the nulls of the two sides before it
        spreads a side of one row. A side of one row whose value is null keeps
        a mask of one row, and the output of more rows then refuses it with a
        panic. A null value of a column is a fact of the data, and a side whose
        values are all null meets this at every run.

        Args:
            left: The data of the first input.
            right: The data of the second input.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            Whether it panics.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> SetOperated()._null_spread(lists, lists.with_dtype(lists.dtype), [5, 1])
            False
            >>> SetOperated()._null_spread(lists, Data(lists.dtype, nulls=True), [5, 1])
            True
        """
        if len(lengths) < 2:  # pragma: no cover -- the run passes the rows of each input
            return False
        rows = max(lengths[0], lengths[1])
        sides = ((left, lengths[0]), (right, lengths[1]))
        return rows > 1 and any(data.nulls and length == 1 for data, length in sides)


class Concatenated:
    """Give the type of `list.concat` and `concat_list`: the lists of each row, one after another.

    The kernel puts a first input that is not a list in a list of one value.
    It joins the inner type of that list with each other input: the inner
    type of a list, or the type itself of any other input. Two types with no
    supertype raise. The kernel casts each input to a list of the supertype,
    and a cast that fails raises. Each other input has the length of the first
    input, or one row, or the kernel raises a `ShapeError`. A first input of
    one row first repeats to the longest of the others.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)))
        >>> Concatenated().output([lists, Data(DType("Int16"))], [5, 1])
        List(Int16)
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where two types have no supertype, where a
        cast fails, or where an input has a length that does not fit.

        Args:
            inputs: The data of each input.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            A list of the supertype.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> Concatenated().output([lists, lists], [5, 2])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        first = inputs[0]
        if first.dtype.name != "List":
            first = first.with_dtype(DType("List", (first.dtype,)))
        inner = first.dtype.inner()
        for other in inputs[1:]:
            part = other.dtype.inner() if other.dtype.name == "List" else other.dtype
            inner = Supertype().require(inner, part)
        target = DType("List", (inner,))
        RunCast().of(first, target)
        for other in inputs[1:]:
            self._cast(other, inner, target)
        self._lengths(lengths)
        return target

    def _cast(self, other: Data, inner: DType, target: DType) -> None:
        """Cast one other input to a list of the supertype.

        An input that is not a list first casts to the supertype itself, and
        then goes in a list. That first cast raises its own error. The kernel
        then casts the list to a list of the supertype.

        It raises `PolarsRefusal` where the first cast has no kernel.

        Args:
            other: The data of the input.
            inner: The supertype.
            target: The list of the supertype.

        Raises:
            PolarsRefusal: The cast of the list fails. Polars raises a `SchemaError`.

        Example:
            >>> wide = DType("Int16")
            >>> Concatenated()._cast(Data(DType("Int8")), wide, DType("List", (wide,)))
        """
        if other.dtype.name != "List":
            cast = RunCast().of(other, inner)
            other = cast.with_dtype(DType("List", (cast.dtype,)))
        try:
            RunCast().of(other, target)
        except PolarsRefusal as refusal:
            message = f"cannot concat `{other.dtype!r}` into a list of `{inner!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message) from refusal

    def _lengths(self, lengths: Sequence[int]) -> None:
        """Check that each other input has the length of the first input, or one row.

        Args:
            lengths: The number of rows of each input, or none where they are not known.

        Raises:
            PolarsRefusal: An input has another length.

        Example:
            >>> Concatenated()._lengths([1, 3, 3])
            >>> Concatenated()._lengths([])
        """
        if not lengths:  # pragma: no cover -- the run passes the rows of each input
            return
        others = [length for length in lengths[1:] if length != 1]
        rows = max(others) if lengths[0] == 1 and others else lengths[0]
        if any(length != rows for length in others):
            message = "an input of the concatenation has another length"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
