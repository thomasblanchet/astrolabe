"""The kernels that read how the data is laid out, and not what its values mean.

`is_null` and `is_not_null` read the validity of each value, `hash` hashes the
bytes of each value, `set_sorted` marks the data as sorted, and `to_physical`
and `reinterpret` read the integers or the floats that hold the values. So
these kernels accept every data type, or every number, with no cast.
`reinterpret` reads the bits of each value as the type that polars computes
when it converts the expression, so its kernel is built from that type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from astrolabe._rules.dtypes import BitWidth, DType, Is
from astrolabe._rules.failures import ErrorClass
from astrolabe._rules.kernels.dispatch import Arm, Dispatch, Fixed, Refuse, Same, ToPhysical


@dataclass(frozen=True)
class SameNumber:
    """Test whether a data type is one number type, with all its parameters.

    Example:
        >>> SameNumber(DType("Decimal", (10, 2)))(DType("Decimal", (10, 2)))
        True
        >>> SameNumber(DType("Date"))(DType("Date"))
        False
    """

    target: DType

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type is a number and is the target.

        Args:
            dtype: The data type.

        Returns:
            Whether it is.

        Example:
            >>> SameNumber(DType("Int8"))(DType("UInt8"))
            False
        """
        return Is.numeric(dtype) and dtype == self.target


@dataclass(frozen=True)
class SameWidth:
    """Test whether a data type is a primitive number of the width of the target (`BitWidth`).

    Example:
        >>> SameWidth(DType("Float32"))(DType("Int32")), SameWidth(DType("Float32"))(DType("Int64"))
        (True, False)
    """

    target: DType

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type has the width of the target.

        Args:
            dtype: The data type.

        Returns:
            Whether it has, where both are primitive numbers.

        Example:
            >>> SameWidth(DType("Date"))(DType("Int32"))
            False
        """
        width = BitWidth().of(dtype)
        return width is not None and width == BitWidth().of(self.target)


class LayoutKernels:
    """Hold the dispatch of each kernel that reads the layout of the data.

    `IS_NULL` gives a `Boolean` for every type, and so does `is_not_null`.
    `HASH` gives a `UInt64` for every type. `SET_SORTED` keeps every type.
    `TO_PHYSICAL` gives the physical type of every type, for example `Int32`
    for a `Date` and `List(Int64)` for a `List(Time)`.

    Example:
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> LayoutKernels.HASH.output(Data(DType("Struct", ((("a", DType("Date")),),))))
        UInt64
        >>> LayoutKernels.TO_PHYSICAL.output(Data(DType("Duration", ("ms",))))
        Int64
    """

    IS_NULL: ClassVar[Dispatch] = Dispatch("is_null", (), Fixed(DType("Boolean")))
    HASH: ClassVar[Dispatch] = Dispatch("hash", (), Fixed(DType("UInt64")))
    SET_SORTED: ClassVar[Dispatch] = Dispatch("set_sorted", (), Same())
    TO_PHYSICAL: ClassVar[Dispatch] = Dispatch("to_physical", (), ToPhysical())

    def reinterpret(self, target: DType) -> Dispatch:
        """Give the kernel of `reinterpret`, which reads the bits of each value as the target.

        The target is the type that polars computes from the planner type
        of the input when it converts the expression. Data of the target
        keeps its type, where the target is a number. Data of a primitive
        number of the width of the target takes the target. Any other data
        raises a `ComputeError`. So a literal with no data type, whose
        planner type is not the type of its data, raises.

        Args:
            target: The type that the function of the node holds.

        Returns:
            The dispatch.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> LayoutKernels().reinterpret(DType("Int8")).output(Data(DType("UInt8")))
            Int8
            >>> LayoutKernels().reinterpret(DType.dynamic_int(1)).output(Data(DType("Int32")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: reinterpret does not accept Int32
        """
        arms = (
            Arm(SameNumber(target), Same()),
            Arm(SameWidth(target), Fixed(target)),
        )
        return Dispatch("reinterpret", arms, Refuse(ErrorClass.COMPUTE))
