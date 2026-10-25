"""The arithmetic of temporal data: a `Date`, a `Datetime`, a `Duration` or a `Time` on the left.

Each logical type of polars has its own arithmetic for an operation where it is
on the left side. A date minus a date gives a duration, a duration times a
number gives a duration, and most other pairs raise. The arithmetic of two
series reaches these classes after it coerces both sides, so the time units of
the two sides already agree in most cases. Where a unit still differs,
`Datetime` panics on an assertion and `Duration` raises, with `PolarsRefusal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.operators import NumericOp

if TYPE_CHECKING:
    from collections.abc import Sequence


class DateOps:
    """Give the type of an operation with a `Date` series on the left.

    A date minus a date casts both to `Datetime("us")` and gives
    `Duration("us")`. A date plus or minus a duration runs on `Datetime("us")`
    and casts the result back to `Date`. Every other operation raises.

    Example:
        >>> DateOps(NumericOp.SUB).output(Data(DType("Date")), Data(DType("Date")))
        Duration(time_unit='us')
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> DateOps(NumericOp.ADD)._op
            <NumericOp.ADD: 'add'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The `Date` input.
            right: The other input.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The operation is not defined for the pair.

        Example:
            >>> DateOps(NumericOp.ADD).output(Data(DType("Date")), Data(DType("Duration", ("ms",))))
            Date
        """
        other = right.dtype.name
        if self._op is NumericOp.SUB and other == "Date":
            return DType("Duration", (TimeUnit.DIFFERENCE["Date"],))
        if self._op in {NumericOp.ADD, NumericOp.SUB} and other == "Duration":
            return DType("Date")
        message = f"{self._op.value} of Date and {right.dtype!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class DatetimeOps:
    """Give the type of an operation with a `Datetime` series on the left.

    A datetime minus a datetime gives a duration of its unit, and a datetime
    plus or minus a duration keeps its type. The kernel asserts that the
    units and the zones agree, and the assertion panics where they differ.
    Every other operation raises.

    Example:
        >>> moment = DType("Datetime", ("ms", "UTC"))
        >>> DatetimeOps(NumericOp.SUB).output(Data(moment), Data(moment))
        Duration(time_unit='ms')
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> DatetimeOps(NumericOp.SUB)._op
            <NumericOp.SUB: 'sub'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The `Datetime` input.
            right: The other input.

        Returns:
            The type of the output.

        It raises `PolarsRefusal` where an assertion on the units fails (`_assert`).

        Raises:
            PolarsRefusal: The operation is not defined for the pair.

        Example:
            >>> moment = DType("Datetime", ("ms", None))
            >>> DatetimeOps(NumericOp.ADD).output(Data(moment), Data(DType("Duration", ("us",))))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: the units differ
        """
        unit = Parameter().unit(left.dtype)
        other = right.dtype
        if self._op is NumericOp.SUB and other.name == "Datetime":
            self._assert(left.dtype.args == other.args)
            return DType("Duration", (unit,))
        if self._op in {NumericOp.ADD, NumericOp.SUB} and other.name == "Duration":
            self._assert(Parameter().unit(other) == unit)
            return left.dtype
        message = f"{self._op.value} of {left.dtype!r} and {other!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _assert(self, holds: bool) -> None:
        """Panic where the units or the zones of the two sides differ, as polars does.

        Args:
            holds: Whether the units and the zones agree.

        Raises:
            PolarsRefusal: They do not agree.

        Example:
            >>> DatetimeOps(NumericOp.SUB)._assert(True) is None
            True
        """
        if not holds:  # pragma: no cover -- the cast pass casts to one unit
            raise PolarsRefusal(Stop.PANICS, "the units differ")


class DurationOps:
    """Give the type of an operation with a `Duration` series on the left.

    A duration plus or minus a duration of the same unit keeps its type. A
    duration plus a date gives a date, and plus a datetime of the same unit
    gives the datetime. A duration times or divided by a number keeps its
    type. A duration divided by a duration gives `Float64`, after a cast of
    the right side to the unit of the left. The remainder needs two equal
    types. Every other operation raises.

    Example:
        >>> span = DType("Duration", ("us",))
        >>> DurationOps(NumericOp.MUL).output(Data(span), Data(DType("Float32")))
        Duration(time_unit='us')
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> DurationOps(NumericOp.REM)._op
            <NumericOp.REM: 'rem'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The `Duration` input.
            right: The other input.

        Returns:
            The type of the output.

        It raises `PolarsRefusal` where a sum, a difference, a product or a
        quotient is not defined for the pair.

        Raises:
            PolarsRefusal: The types of a remainder differ.

        Example:
            >>> span = DType("Duration", ("us",))
            >>> DurationOps(NumericOp.DIV).output(Data(span), Data(DType("Duration", ("ms",))))
            Float64
        """
        if self._op in {NumericOp.ADD, NumericOp.SUB}:
            return self._add_or_sub(left.dtype, right.dtype)
        if self._op in {NumericOp.MUL, NumericOp.DIV}:
            return self._scale(left.dtype, right.dtype)
        if not left.dtype.same(right.dtype):
            message = "the types of a remainder differ"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return left.dtype

    def _add_or_sub(self, left: DType, right: DType) -> DType:
        """Give the type of a sum or a difference.

        Args:
            left: The `Duration` type.
            right: The other type.

        Returns:
            The type of the output.

        It raises `PolarsRefusal` where the units differ (`_same_unit`).

        Raises:
            PolarsRefusal: The pair is not defined.

        Example:
            >>> span = DType("Duration", ("ns",))
            >>> DurationOps(NumericOp.ADD)._add_or_sub(span, DType("Datetime", ("ns", "UTC")))
            Datetime(time_unit='ns', time_zone='UTC')
        """
        unit = Parameter().unit(left)
        if right.name == "Duration":
            self._same_unit(unit, right)
            return left
        if self._op is NumericOp.ADD and right.name == "Date":
            return right
        if self._op is NumericOp.ADD and right.name == "Datetime":
            self._same_unit(unit, right)
            return right
        message = f"{self._op.value} of {left!r} and {right!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _scale(self, left: DType, right: DType) -> DType:
        """Give the type of a product or a quotient.

        Args:
            left: The `Duration` type.
            right: The other type.

        Returns:
            The `Duration` type for a number, and `Float64` for a quotient of two durations.

        Raises:
            PolarsRefusal: The other type is not a number, or not a duration in a quotient.

        Example:
            >>> span = DType("Duration", ("ms",))
            >>> DurationOps(NumericOp.MUL)._scale(span, DType("UInt128"))
            Duration(time_unit='ms')
        """
        if self._op is NumericOp.DIV and right.name == "Duration":
            return DType("Float64")
        if Is.integer(right) or Is.float(right):
            return left
        message = f"{self._op.value} of {left!r} and {right!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _same_unit(self, unit: str, other: DType) -> None:
        """Raise where the other type has another time unit.

        Args:
            unit: The unit of the duration.
            other: The other temporal type.

        Raises:
            PolarsRefusal: The units differ.

        Example:
            >>> span = DType("Duration", ("ms",))
            >>> DurationOps(NumericOp.ADD)._same_unit("ms", span) is None
            True
            >>> DurationOps(NumericOp.ADD)._same_unit("ns", span)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: units are different
        """
        if Parameter().unit(other) != unit:  # pragma: no cover -- the cast pass casts to one unit
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, "units are different")


class TimeOps:
    """Give the type of an operation with a `Time` series on the left.

    A time minus a time gives `Duration("ns")`. Every other operation raises.

    Example:
        >>> TimeOps(NumericOp.SUB).output(Data(DType("Time")), Data(DType("Time")))
        Duration(time_unit='ns')
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> TimeOps(NumericOp.MUL)._op
            <NumericOp.MUL: 'mul'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The `Time` input.
            right: The other input.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The operation is not a difference of two times.

        Example:
            >>> clock = Data(DType("Time"))
            >>> TimeOps(NumericOp.ADD).output(clock, clock)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if self._op is NumericOp.SUB and right.dtype.name == "Time":
            return DType("Duration", (TimeUnit.DIFFERENCE["Time"],))
        message = f"{self._op.value} of Time and {right.dtype!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


@dataclass(frozen=True)
class TemporalParts:
    """Give the type of `pl.datetime` and `pl.duration` from their parts: the type of the options.

    The kernel of `pl.datetime` casts each part with the cast of a series,
    which does not refuse a value: the year and the microsecond to `Int32`,
    the month, the day, the hour, the minute and the second to `Int8`, and
    the ambiguity to text. It then reads each part at that type, and refuses
    a part that the cast leaves at another type, as a struct. It multiplies
    the microsecond by 1000 first, and the arithmetic of a struct panics. A
    part shorter
    than the longest repeats its first value. The kernel of `pl.duration`
    reads parts that the cast pass already cast, and adds them with the
    arithmetic of series, which refuses two lengths of more than one row
    that differ.

    Example:
        >>> parts = TemporalParts.datetime(DType("Datetime", ("us", None)))
        >>> parts.output([Data(DType("String"))] * 8, [5] * 8)
        Datetime(time_unit='us', time_zone=None)
    """

    DATETIME: ClassVar[tuple[DType, ...]] = (
        DType("Int32"),
        DType("Int8"),
        DType("Int8"),
        DType("Int8"),
        DType("Int8"),
        DType("Int8"),
        DType("Int32"),
        DType("String"),
    )
    MICROSECOND: ClassVar[int] = 6
    dtype: DType
    targets: tuple[DType, ...] = ()
    arithmetic: bool = False

    @classmethod
    def datetime(cls, dtype: DType) -> TemporalParts:
        """Give the kernel of `pl.datetime`.

        Args:
            dtype: The `Datetime` that the options of the node give.

        Returns:
            The kernel.

        Example:
            >>> len(TemporalParts.datetime(DType("Datetime", ("us", None))).targets)
            8
        """
        return cls(dtype, cls.DATETIME)

    @classmethod
    def duration(cls, dtype: DType) -> TemporalParts:
        """Give the kernel of `pl.duration`.

        Args:
            dtype: The `Duration` that the options of the node give.

        Returns:
            The kernel.

        Example:
            >>> TemporalParts.duration(DType("Duration", ("us",))).arithmetic
            True
        """
        return cls(dtype, arithmetic=True)

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each part.
            lengths: The rows of each part, or none where they are not known.

        Returns:
            The type of the options.

        Raises:
            PolarsRefusal: A part has no cast to its target, the cast leaves it
                at another type, or two parts of `pl.duration` have other lengths.

        Example:
            >>> parts = TemporalParts.datetime(DType("Datetime", ("us", None)))
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> parts.output([struct] * 8, [5] * 8)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
            >>> TemporalParts.duration(DType("Duration", ("us",))).output([], [3, 5])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot do arithmetic ...
        """
        long = {length for length in lengths if length != 1}
        if self.arithmetic and len(long) > 1:
            message = "cannot do arithmetic operation on series of different lengths"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        for place, (data, target) in enumerate(zip(inputs, self.targets, strict=False)):
            cast = RunCast().of(data, target).dtype
            if cast != target and place == self.MICROSECOND:
                raise PolarsRefusal(Stop.PANICS, f"the arithmetic of {cast!r} with a number")
            if cast != target:
                message = f"invalid series dtype: expected `{target!r}`, got `{cast!r}`"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return self.dtype
