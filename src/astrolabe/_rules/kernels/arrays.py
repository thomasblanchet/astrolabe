"""The kernels of the `arr` namespace, which work on the values inside each array.

A function of the `arr` namespace reads its first input as an array column, a
column of lists of one fixed width. Data of another type stops it with a
`SchemaError`. Most functions then run a kernel of the inner type on the
values of each array, as the functions of the `list` namespace do on each
list (`Elements`). The kernels of arrays are not those of lists: `arr.sum`
refuses a `Decimal` that `list.sum` keeps, and `arr.min` panics on every
inner type that is not an integer or a float.

The functions of several inputs, such as `arr.get`, also read the number of
rows of each input, and each one checks the lengths in its own way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Arm, Data, Dispatch, Fixed, Kinds, Refuse, Same
from astrolabe._rules.kernels.lengths import Mapped
from astrolabe._rules.kernels.lists import ElementGet, Elements
from astrolabe._rules.type_functions import ArrayClosures

if TYPE_CHECKING:
    from collections.abc import Sequence


class Relisted:
    """Give a `List` of the inner type of an array, as the cast of `arr.to_list` gives it.

    Example:
        >>> array = Data(DType("Array", (DType("Date"), 2)))
        >>> Relisted().give(array, ArrayKernels.TO_LIST)
        List(Date)
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the list type.

        Args:
            data: The data, of an `Array` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> nested = Data(DType("Array", (DType("Array", (DType("Int8"), 2)), 3)))
            >>> Relisted().give(nested, ArrayKernels.TO_LIST)
            List(Array(Int8, shape=(2,)))
        """
        del kernel
        return DType("List", (data.dtype.inner(),))


@dataclass(frozen=True)
class Places:
    """Give a struct with one field for each place of an array, as `arr.to_struct` gives it.

    The kernel reads the value at each place of the arrays, so each field has
    the inner type. `names` are the names of the fields, or `None` for the
    names `field_0`, `field_1` and on.

    Example:
        >>> array = Data(DType("Array", (DType("Int8"), 2)))
        >>> Places(None).give(array, ArrayKernels().to_struct(None))
        Struct({'field_0': Int8, 'field_1': Int8})
    """

    names: Sequence[str] | None

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the struct type.

        Args:
            data: The data, of an `Array` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> array = Data(DType("Array", (DType("Date"), 1)))
            >>> Places(("a",)).give(array, ArrayKernels().to_struct(("a",)))
            Struct({'a': Date})
        """
        del kernel
        return ArrayClosures().to_struct(self.names)(data.dtype)


class ArrayKernels:
    """Hold the kernels of the functions of the `arr` namespace that read one input.

    Each kernel is a dispatch on the data of the array column. Its one arm
    takes an `Array`, and any other type raises a `SchemaError`, except for
    `arr.to_list`, which raises a `ComputeError`. The kernels of the inner
    type follow the source:

    - `min` and `max` run on the physical numbers of an integer or a float,
      and panic on any other inner type, a `Decimal` or a `Date` included.
    - `sum` counts a `Boolean`, widens an integer of 8 or 16 bits to `Int64`,
      and keeps the other integers and floats. Any other inner type raises a
      `ComputeError`.
    - `mean`, `median` and `std` keep a narrow float and a `Duration`, and give
      `Float64` otherwise. `var` gives a `Duration` in milliseconds.

    Example:
        >>> array = Data(DType("Array", (DType("Date"), 2)))
        >>> ArrayKernels.MOMENT.output(array)
        Float64
        >>> ArrayKernels.EXTREMUM.output(array)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    ARRAYS: ClassVar[Kinds] = Kinds("Array")
    NUMBERS: ClassVar[Kinds] = Kinds(*Is.SIGNED, *Is.UNSIGNED, *Is.FLOATS)
    NARROW_FLOATS: ClassVar[Kinds] = Kinds("Float16", "Float32")

    VALUE_EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "arr.min",
        (Arm(NUMBERS, Same()),),
        Refuse(Stop.PANICS),
    )
    VALUE_SUM: ClassVar[Dispatch] = Dispatch(
        "arr.sum",
        (
            Arm(Kinds("Boolean"), Fixed(DType.idx())),
            Arm(Kinds(*Is.SMALL_INTEGERS), Fixed(DType("Int64"))),
            Arm(NUMBERS, Same()),
        ),
        Refuse(ErrorClass.COMPUTE),
    )
    VALUE_MOMENT: ClassVar[Dispatch] = Dispatch(
        "arr.mean",
        (Arm(NARROW_FLOATS, Same()), Arm(Kinds("Duration"), Same())),
        Fixed(DType("Float64")),
    )
    VALUE_VAR: ClassVar[Dispatch] = Dispatch(
        "arr.var",
        (
            Arm(NARROW_FLOATS, Same()),
            Arm(Kinds("Duration"), Fixed(DType("Duration", ("ms",)))),
        ),
        Fixed(DType("Float64")),
    )
    VALUE: ClassVar[Dispatch] = Dispatch("arr.explode", (), Same())

    LENGTH: ClassVar[Dispatch] = Dispatch(
        "arr.len",
        (Arm(ARRAYS, Fixed(DType.idx())),),
        Refuse(ErrorClass.SCHEMA),
    )
    EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "arr.min",
        (Arm(ARRAYS, Elements(VALUE_EXTREMUM)),),
        Refuse(ErrorClass.SCHEMA),
    )
    SUM: ClassVar[Dispatch] = Dispatch(
        "arr.sum",
        (Arm(ARRAYS, Elements(VALUE_SUM)),),
        Refuse(ErrorClass.SCHEMA),
    )
    MOMENT: ClassVar[Dispatch] = Dispatch(
        "arr.mean",
        (Arm(ARRAYS, Elements(VALUE_MOMENT)),),
        Refuse(ErrorClass.SCHEMA),
    )
    VAR: ClassVar[Dispatch] = Dispatch(
        "arr.var",
        (Arm(ARRAYS, Elements(VALUE_VAR)),),
        Refuse(ErrorClass.SCHEMA),
    )
    ARG_EXTREMUM: ClassVar[Dispatch] = Dispatch(
        "arr.arg_min",
        (Arm(ARRAYS, Fixed(DType.idx())),),
        Refuse(ErrorClass.SCHEMA),
    )
    SAME: ClassVar[Dispatch] = Dispatch(
        "arr.sort",
        (Arm(ARRAYS, Same()),),
        Refuse(ErrorClass.SCHEMA),
    )
    TO_LIST: ClassVar[Dispatch] = Dispatch(
        "arr.to_list",
        (Arm(ARRAYS, Relisted()),),
        Refuse(ErrorClass.COMPUTE),
    )
    EXPLODE: ClassVar[Dispatch] = Dispatch(
        "arr.explode",
        (Arm(ARRAYS, Elements(VALUE)),),
        Refuse(ErrorClass.SCHEMA),
    )

    def get(self, null_on_oob: bool) -> ElementGet:
        """Give the kernel of `arr.get`, `arr.first` and `arr.last`.

        Args:
            null_on_oob: Whether an index outside an array gives a null.

        Returns:
            The kernel.

        Example:
            >>> ArrayKernels().get(null_on_oob=False)
            ElementGet(container='Array', null_on_oob=False)
        """
        return ElementGet("Array", null_on_oob=null_on_oob)

    def to_struct(self, names: Sequence[str] | None) -> Dispatch:
        """Give the kernel of `arr.to_struct`.

        Args:
            names: The names of the fields, or `None` for the default names.

        Returns:
            The dispatch.

        Example:
            >>> ArrayKernels().to_struct(None).output(Data(DType("Int8")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        arms = (Arm(self.ARRAYS, Places(names)),)
        return Dispatch("arr.to_struct", arms, Refuse(ErrorClass.SCHEMA))

    def slice(self, offset: int, length: int) -> Dispatch:
        """Give the kernel of `arr.slice`, `arr.head` and `arr.tail` that keep the array.

        The output is an array of the inner type, as wide as the part of the
        array that the slice covers (`ArrayClosures.slice`).

        Args:
            offset: The offset, which counts from the end where it is negative.
            length: The length.

        Returns:
            The dispatch.

        Example:
            >>> pair = Data(DType("Array", (DType("Int8"), 2)))
            >>> ArrayKernels().slice(-1, 5).output(pair)
            Array(Int8, shape=(1,))
        """
        arms = (Arm(self.ARRAYS, Mapped(ArrayClosures().slice(offset, length))),)
        return Dispatch("arr.slice", arms, Refuse(ErrorClass.INVALID_OPERATION))


class ArrayConcat:
    """Give the type of `concat_arr`: the values of each input of a row, as one array.

    The kernel reads the type of the data of each input: an array of a
    width, or a value of any other type, which counts as an array of width 1.
    The inner types must all be the inner type of the first input, or it
    raises a `SchemaError` (`ArrayClosures.concat`). The inputs that have more
    than one row must all have one length, or it raises a `ShapeError`.

    Example:
        >>> pair = Data(DType("Array", (DType("Int8"), 2)))
        >>> ArrayConcat().output([pair, Data(DType("Int8"))], [5, 1])
        Array(Int8, shape=(3,))
        >>> ArrayConcat().output([pair, pair], [5, 2])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where an inner type differs from the first.

        Args:
            inputs: The data of each input.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The array type.

        Raises:
            PolarsRefusal: Two inputs of more than one row have other lengths.

        Example:
            >>> ArrayConcat().output([Data(DType("Date"))], [])
            Array(Date, shape=(1,))
        """
        dtype = ArrayClosures().concat(tuple(data.dtype for data in inputs))
        heights = {length for length in lengths if length != 1}
        if len(heights) > 1:
            message = "concat_arr: the inputs have other lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return dtype
