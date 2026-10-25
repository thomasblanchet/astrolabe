"""The bitwise and logic kernels: `&`, `|`, `^`, the logical operators and the bitwise functions.

The bitwise operators of two series accept only a `Boolean` or an integer on
the left, and the same type on the right. A series of one row on the right
first casts to the left type. The logical operators of an expression cast both
sides to `Boolean` first. The bitwise functions of one input, `not`, the counts
of bits and the bitwise reductions, dispatch on the type of the input.
"""

from __future__ import annotations

import enum
from typing import ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal
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


class BitOp(enum.Enum):
    """Name the bitwise operators of two series.

    The value is the name of the method of polars.

    Example:
        >>> BitOp.XOR.value
        'bitxor'
    """

    AND = "bitand"
    OR = "bitor"
    XOR = "bitxor"


class BitOps:
    """Give the type of `&`, `|` or `^` of two series.

    The left side must be `Boolean` or an integer, and the right side must
    have the same type. Beside an integer, a right side of one row casts to
    the left type first. A float, a `Null` or any other left type raises,
    and so does a nested type.

    Example:
        >>> BitOps(BitOp.AND).output(Data(DType("Int8")), Data(DType("Int8")))
        Int8
        >>> BitOps(BitOp.AND).output(Data(DType("Int8")), Data(DType("Int16"), one_row=True))
        Int8
    """

    def __init__(self, op: BitOp) -> None:
        """Keep the operator.

        Args:
            op: The operator.

        Example:
            >>> BitOps(BitOp.OR)._op
            <BitOp.OR: 'bitor'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the left input.

        It raises `PolarsRefusal` where the right type differs (`_unpack`), or
        a cast of a right side of one row has no kernel.

        Raises:
            PolarsRefusal: The left type has no bitwise kernel.

        Example:
            >>> double = Data(DType("Float64"))
            >>> BitOps(BitOp.XOR).output(double, double)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        dtype = left.dtype
        if dtype.name == "Boolean":
            self._unpack(dtype, right.dtype)
            return dtype
        if Is.integer(dtype):
            other = RunCast().of(right, dtype) if right.one_row else right
            self._unpack(dtype, other.dtype)
            return dtype
        message = f"{self._op.value} is not defined for {dtype!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _unpack(self, left: DType, right: DType) -> None:
        """Raise where the right type differs from the left, as a bitwise operator of polars does.

        Args:
            left: The left type.
            right: The right type.

        Raises:
            PolarsRefusal: The types differ.

        Example:
            >>> BitOps(BitOp.AND)._unpack(DType("Boolean"), DType("Boolean")) is None
            True
        """
        if not left.same(right):
            message = f"cannot unpack {right!r} into {left!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


class LogicalOps:
    """Give the type of a logical `and` or `or` of an expression.

    The operator casts both sides to `Boolean`, and then runs the bitwise
    operator of two series, `BitOps`. A cast that has no kernel raises. A
    struct casts each field, so a struct with values reaches the bitwise
    operator as a struct and raises there.

    Example:
        >>> LogicalOps(BitOp.AND).output(Data(DType("Int8")), Data(DType("Float32")))
        Boolean
    """

    def __init__(self, op: BitOp) -> None:
        """Keep the operator.

        Args:
            op: `AND` for the logical and, `OR` for the logical or.

        Example:
            >>> LogicalOps(BitOp.OR)._op
            <BitOp.OR: 'bitor'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast to `Boolean` fails, or the bitwise operator fails.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            `Boolean`.

        Example:
            >>> LogicalOps(BitOp.OR).output(Data(DType("Date")), Data(DType("Boolean")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        cast = RunCast()
        boolean = DType("Boolean")
        return BitOps(self._op).output(cast.of(left, boolean), cast.of(right, boolean))


class BitwiseKernels:
    """Hold the dispatch of the bitwise kernels of one input.

    `NOT` is `not_`: it negates a `Boolean` and the bits of an integer. `COUNT`
    is each count of bits, such as `bitwise_count_ones`: it accepts a
    `Boolean`, an integer or a float and gives `UInt32`. `REDUCE` is each
    bitwise reduction, such as `bitwise_and`: it keeps the type of a `Boolean`,
    an integer or a float, and every other type raises.

    Example:
        >>> BitwiseKernels.COUNT.output(Data(DType("Float64")))
        UInt32
        >>> BitwiseKernels.REDUCE.output(Data(DType("Float32")))
        Float32
    """

    NOT: ClassVar[Dispatch] = Dispatch(
        "not",
        (
            Arm(Kinds("Boolean"), Same()),
            Arm(Is.integer, Same()),
        ),
        Refuse(),
    )
    COUNT: ClassVar[Dispatch] = Dispatch(
        "bitwise count",
        (
            Arm(Kinds("Boolean"), Fixed(DType("UInt32"))),
            Arm(Is.integer, Fixed(DType("UInt32"))),
            Arm(Is.float, Fixed(DType("UInt32"))),
        ),
        Refuse(),
    )
    REDUCE: ClassVar[Dispatch] = Dispatch(
        "bitwise reduce",
        (
            Arm(Kinds("Boolean"), Same()),
            Arm(Is.integer, Same()),
            Arm(Is.float, Same()),
        ),
        Refuse(),
    )
