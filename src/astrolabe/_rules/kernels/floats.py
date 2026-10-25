"""The float family of kernels: cast a non-float input to `Float64`, then run on floats.

`sqrt`, `cbrt`, `exp`, `log1p`, `log` and the trigonometric functions run their
kernel on a float, and keep the width of `Float16`, `Float32` and `Float64`.
Another input casts to `Float64`, and the kernel calls itself on the result. A
cast of a `Struct` to a number casts each field and gives a `Struct` again, so
`sqrt`, `cbrt`, `exp`, `log1p` and `log` call themselves without end over a
struct with values, and the process ends with a stack overflow. A struct of
only nulls casts to a plain `Float64`, so it runs. The trigonometric functions
cast only a number, and raise on any other type. The tests of a value, such
as `is_nan`, give a `Boolean` for each number and cast nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import (
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    Physical,
    Recast,
    Refuse,
    RunCast,
    Same,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class Log:
    """Give the type of `log`, the logarithm of a value to a base, at the data of its two inputs.

    The kernel first checks the lengths of the two inputs: they are equal,
    or one of them has one row. Two other lengths raise an
    `InvalidOperationError`. Then it runs on two floats of one type, and
    keeps that type. Where the value is a float and the base is not, it
    casts the base to the type of the value. Where only the base is a float,
    it casts the value to the type of the base. Where neither is a float, it
    casts the value to `Float64`. Each cast is not strict, and a cast with
    no kernel panics. After each cast, the kernel calls itself. A struct
    with values casts to a struct again, so the kernel calls itself without
    end, and the process ends with a stack overflow.

    Example:
        >>> Log().output([Data(DType("Int8")), Data(DType("Float32"))], [3, 1])
        Float32
        >>> Log().output([Data(DType("Int8")), Data(DType("Int8"))], [3, 2])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot do a binary ...
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast has no kernel, or where the
        kernel calls itself without end.

        Args:
            inputs: The data of the value and of the base.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The two inputs have other lengths, and neither has one row.

        Example:
            >>> Log().output([Data(DType("Float64")), Data(DType("Boolean"))], [])
            Float64
        """
        others = {length for length in lengths if length != 1}
        if len(others) > 1:
            message = f"cannot do a binary operation on columns of different lengths: {lengths}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return self._log(inputs[0], inputs[1])

    def _log(self, value: Data, base: Data) -> DType:
        """Give the type of the logarithm, after the casts that bring the two inputs to one float.

        It raises `PolarsRefusal` where a cast gives the input its type back,
        so that the kernel calls itself without end (`_recurses`).

        Args:
            value: The data of the value.
            base: The data of the base.

        Returns:
            The type.

        Example:
            >>> Log()._log(Data(DType("Date")), Data(DType("Int32")))
            Float64
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> Log()._log(struct, Data(DType("Float32")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: crashes: log calls itself ...
        """
        if Is.float(value.dtype) and value.dtype.same(base.dtype):
            return value.dtype
        if Is.float(value.dtype):
            cast = RunCast(unwrap=True).of(base, value.dtype)
            self._recurses(cast, base)
            return self._log(value, cast)
        target = base.dtype if Is.float(base.dtype) else DType("Float64")
        cast = RunCast(unwrap=True).of(value, target)
        self._recurses(cast, value)
        return self._log(cast, base)

    def _recurses(self, cast: Data, before: Data) -> None:
        """Refuse a cast that gives the input its type back, as a cast of a struct to a float does.

        Args:
            cast: The data after the cast.
            before: The data before the cast.

        Raises:
            PolarsRefusal: The cast changed nothing, so the kernel calls itself without end.

        Example:
            >>> Log()._recurses(Data(DType("Float64")), Data(DType("Int8"))) is None
            True
        """
        if cast.dtype.same(before.dtype):
            message = f"log calls itself on {cast.dtype!r} without end"
            raise PolarsRefusal(Stop.CRASHES, message)


class FloatKernels:
    """Hold the dispatch of each kernel of the float family.

    `SQRT` and `CBRT` keep a float and cast every other input to `Float64`, and
    a cast with no kernel raises. `EXP` and `LOG1P` cast a `Decimal` to
    `Float64`, then read the physical type: an integer, such as the days of a
    `Date`, gives `Float64`, and a float keeps its width. They cast every other
    physical type to `Float64`, and a cast with no kernel panics.
    `TRIGONOMETRY` keeps a float, casts another number to `Float64`, and raises
    on every other type.

    `IS_NAN`, `IS_FINITE` and `IS_INFINITE` give a `Boolean` for a number of
    a primitive type and for `Null`, and raise on every other type, a
    `Decimal` included. `IS_NOT_NAN` is the same, but it raises on `Null`.

    Example:
        >>> FloatKernels.SQRT.output(Data(DType("String")))
        Float64
        >>> FloatKernels.EXP.output(Data(DType("Date")))
        Float64
        >>> FloatKernels.TRIGONOMETRY.output(Data(DType("Boolean")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    FLOAT: ClassVar[Kinds] = Kinds(*Is.FLOATS)
    SQRT: ClassVar[Dispatch] = Dispatch(
        "sqrt",
        (Arm(FLOAT, Same()),),
        Recast(DType("Float64")),
    )
    CBRT: ClassVar[Dispatch] = Dispatch(
        "cbrt",
        (Arm(FLOAT, Same()),),
        Recast(DType("Float64")),
    )
    EXP: ClassVar[Dispatch] = Dispatch(
        "exp",
        (
            Arm(
                Kinds("Decimal"),
                Recast(DType("Float64"), unwrap=True),
            ),
            Arm(Physical(Is.integer), Fixed(DType("Float64"))),
            Arm(Physical(FLOAT), Same()),
        ),
        Recast(DType("Float64"), unwrap=True, physical=True),
    )
    LOG1P: ClassVar[Dispatch] = Dispatch(
        "log1p",
        (
            Arm(
                Kinds("Decimal"),
                Recast(DType("Float64"), unwrap=True),
            ),
            Arm(Physical(Is.integer), Fixed(DType("Float64"))),
            Arm(Physical(FLOAT), Same()),
        ),
        Recast(DType("Float64"), unwrap=True, physical=True),
    )
    TRIGONOMETRY: ClassVar[Dispatch] = Dispatch(
        "trigonometry",
        (
            Arm(FLOAT, Same()),
            Arm(
                Is.primitive_numeric,
                Recast(DType("Float64")),
            ),
        ),
        Refuse(),
    )
    BOOLEAN: ClassVar[Fixed] = Fixed(DType("Boolean"))
    IS_NAN: ClassVar[Dispatch] = Dispatch(
        "is_nan",
        (
            Arm(Is.primitive_numeric, BOOLEAN),
            Arm(Kinds("Null"), BOOLEAN),
        ),
        Refuse(),
    )
    IS_NOT_NAN: ClassVar[Dispatch] = Dispatch(
        "is_not_nan",
        (Arm(Is.primitive_numeric, BOOLEAN),),
        Refuse(),
    )
    IS_FINITE: ClassVar[Dispatch] = Dispatch("is_finite", IS_NAN.arms, Refuse())
    IS_INFINITE: ClassVar[Dispatch] = Dispatch("is_infinite", IS_NAN.arms, Refuse())
