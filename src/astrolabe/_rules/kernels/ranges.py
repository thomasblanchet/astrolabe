"""The kernels of the range functions, such as `int_range` and `date_ranges`.

A range function builds a run of values from its bounds. A function of one
range, such as `int_range`, reads one value from each bound and gives a column
of the range. A function of many ranges, such as `int_ranges`, reads the bounds
row by row and gives a list for each row. Each kernel checks the rows of its
inputs, reads the types of the bounds, and gives the type of the values. The
cast pass has already cast the bounds of the integer and the date ranges. The
number of values depends on the values of the bounds, so no kernel here
decides it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.kernels.lengths import Counted, Mapped
from astrolabe._rules.type_functions import FieldsMapper, LengthClosures

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


class RangeRows:
    """Check the rows of the inputs of a range function.

    A function of one range needs one value in each bound. A function of
    many ranges reads its inputs row by row: each input that does not have
    one row must have the rows of the others, and an input of one row
    spreads. A length that the caller does not know passes.

    Example:
        >>> RangeRows().matched((3, 1, 3), ("start", "end", "step"))
    """

    def single(self, lengths: Sequence[int], names: Sequence[str]) -> None:
        """Refuse a bound that does not hold exactly one value.

        Args:
            lengths: The rows of each input, or none where they are not known.
            names: The name of each bound, for the message.

        Raises:
            PolarsRefusal: A bound has another number of rows.

        Example:
            >>> RangeRows().single((1, 3), ("start", "end"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: `end` must contain ...
        """
        for name, length in zip(names, lengths, strict=False):
            if length != 1:  # pragma: no cover -- the conversion checks it first
                message = f"`{name}` must contain exactly one value, got {length} values"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def matched(self, lengths: Sequence[int], names: Sequence[str]) -> None:
        """Refuse inputs whose rows do not match, where no input of one row spreads.

        Args:
            lengths: The rows of each input, or none where they are not known.
            names: The name of each input, for the message.

        Raises:
            PolarsRefusal: Two inputs of more than one row differ.

        Example:
            >>> RangeRows().matched((3, 2), ("start", "end"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: lengths of ...
        """
        longer = {length for length in lengths[: len(names)] if length != 1}
        if len(longer) > 1:
            listed = ", ".join(names)
            raise PolarsRefusal(ErrorClass.COMPUTE, f"lengths of {listed} do not match")


@dataclass(frozen=True)
class IntRange:
    """Give the type of `int_range`: the integer type of the call.

    Each bound holds one value of that type, which the cast pass gave it. A
    null bound raises, and so does a step of zero.

    Example:
        >>> IntRange(DType("Int64"), 1).output([Data(DType("Int64"))] * 2, (1, 1))
        Int64
    """

    dtype: DType
    step: int

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a bound does not hold one value (`RangeRows`).

        Args:
            inputs: The data of the two bounds.
            lengths: The rows of each bound, or none where they are not known.

        Returns:
            The type of the call.

        Raises:
            PolarsRefusal: A bound is null, or the step is zero.

        Example:
            >>> null = Data(DType("Int64"), nulls=True)
            >>> IntRange(DType("Int64"), 1).output([null, null], (1, 1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: invalid null input ...
            >>> bound = Data(DType("Int64"))
            >>> IntRange(DType("Int64"), 0).output([bound, bound], (1, 1))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: step must not be zero
        """
        RangeRows().single(lengths, ("start", "end"))
        if any(data.nulls for data in inputs[:2]):
            raise PolarsRefusal(ErrorClass.COMPUTE, "invalid null input for `int_range`")
        if self.step == 0:
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, "step must not be zero")
        return self.dtype


@dataclass(frozen=True)
class IntRanges:
    """Give the type of `int_ranges`: a list of the integer type of the call.

    The cast pass gave each input the type `Int64`. A null input gives a null
    list. A literal step of zero panics, at every row.

    Example:
        >>> IntRanges(DType("Int16")).output([Data(DType("Int64"))] * 3, (4, 1, 4), ())
        List(Int16)
    """

    dtype: DType

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the rows of the inputs do not match (`RangeRows`).

        Args:
            inputs: The data of the start, the end and the step.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The step is a literal of zero, where polars panics.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> step = Operand.typed_literal(0, DType("Int64"))
            >>> IntRanges(DType("Int64")).output([Data(DType("Int64"))] * 3, (), (step,) * 3)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        del inputs
        RangeRows().matched(lengths, ("start", "end", "step"))
        step = computed[2] if len(computed) > 2 else None
        if step is not None and step.literal and step.value == 0:
            raise PolarsRefusal(Stop.PANICS, "assertion failed: step != 0")
        return DType("List", (self.dtype,))


class LinearSpace:
    """Give the type of `linear_space`, from the first value of each of its three inputs.

    Each input must hold one value. The number of samples must read as an
    integer of zero or more: a null, a value of a category, of bytes or of a
    nested type, and a literal of a negative number or of a text that holds
    no number raise. A text reads as an integer, or else as a float. The type
    of the values comes from the bounds as the type function gives it
    (`FieldsMapper`), except that two bounds of `Float16` give `Float64`. A
    null bound has the type `Null`, which raises.

    Example:
        >>> inputs = [Data(DType("Int8")), Data(DType("Int8")), Data(DType("Int64"))]
        >>> LinearSpace().output(inputs, (1, 1, 1), ())
        Float64
    """

    NO_NUMBER: ClassVar[frozenset[str]] = frozenset({
        "Categorical",
        "Enum",
        "Binary",
        "List",
        "Array",
        "Struct",
        "Null",
    })
    INTEGER: ClassVar[re.Pattern[str]] = re.compile(r"[+-]?[0-9]+")
    UPPER: ClassVar[int] = 2**64

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where an input does not hold one value
        (`RangeRows`), or where the bounds have no type of values.

        Args:
            inputs: The data of the start, the end and the number of samples.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The number of samples does not read as an integer of
                zero or more.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Data(DType("Int8")), Data(DType("Int8")), Data(DType("Int32"))]
            >>> LinearSpace().output(inputs, (1, 1, 1), [Operand.of_value(-1)] * 3)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: 'num_samples' must be ...
        """
        RangeRows().single(lengths, ("start", "end", "num_samples"))
        count = inputs[2]
        literal = computed[2] if len(computed) > 2 and computed[2].literal else None
        unread = literal is not None and not self._count(literal.value)
        if count.nulls or count.dtype.name in self.NO_NUMBER or unread:
            message = f"'num_samples' must be non-negative integer, got {count.dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        start, end = (DType("Null") if data.nulls else data.dtype for data in inputs[:2])
        dtype = FieldsMapper(start, end).linear_space_dtype()
        return DType("Float64") if dtype.name == "Float16" else dtype

    def _count(self, value: object) -> bool:
        """Say whether a literal value reads as an integer of zero or more.

        A float reads as its whole part, so it must be more than -1. A text
        reads as an integer, or else as a float.

        Args:
            value: The value of the literal.

        Returns:
            Whether it reads.

        Example:
            >>> LinearSpace()._count("3"), LinearSpace()._count("a"), LinearSpace()._count(-0.5)
            (True, False, True)
        """
        if isinstance(value, str):
            value = self._number(value)
        if isinstance(value, bool):
            return True
        if isinstance(value, int):
            return 0 <= value < self.UPPER
        if isinstance(value, float):
            return -1 < value < self.UPPER
        return False

    def _number(self, text: str) -> int | float | None:
        """Read a text as polars reads it into a number: an integer, or else a float.

        Args:
            text: The text.

        Returns:
            The number, or `None` where the text holds none.

        Example:
            >>> [LinearSpace()._number(text) for text in ("+7", "1e3", " 1")]
            [7, 1000.0, None]
        """
        if self.INTEGER.fullmatch(text):
            return int(text)
        if text != text.strip() or "_" in text:
            return None
        try:
            return float(text)
        except ValueError:
            return None


@dataclass(frozen=True)
class LinearSpaces:
    """Give the type of `linear_spaces`: a list or an array of the values of each row.

    The number of samples casts to `UInt64` with a strict cast (`Counted`),
    or comes from the width of the arrays. The inputs then read row by row
    (`RangeRows`). Two bounds of `Float32` keep it, and two bounds of one
    temporal type keep it, where a `Date` becomes a `Datetime` in
    microseconds. Bounds of other numbers cast to `Float64` with a strict
    cast, and any other pair raises. A null input gives a null row.

    Example:
        >>> inputs = [Data(DType("Float16")), Data(DType("Float16")), Data(DType("Int8"))]
        >>> LinearSpaces(None).output(inputs, (3, 3, 1), ())
        List(Float64)
    """

    COUNT: ClassVar[Counted] = Counted((), Mapped(LengthClosures().same))

    array_width: int | None

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the count does not cast, where the rows
        of the inputs do not match, or where the bounds have no type of values.

        Args:
            inputs: The data of the start, the end and the number of samples.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Example:
            >>> inputs = [Data(DType("Date")), Data(DType("Date")), Data(DType("Int8"))]
            >>> LinearSpaces(2).output(inputs, (), ())
            Array(Datetime(time_unit='us', time_zone=None), shape=(2,))
        """
        names = ("start", "end")
        if self.array_width is None:
            literal = computed[2] if len(computed) > 2 and computed[2].literal else None
            self.COUNT.cast(inputs[2], literal)
            names = ("start", "end", "num_samples")
        RangeRows().matched(lengths, names)
        start, end = inputs[0].dtype, inputs[1].dtype
        inner = FieldsMapper(start, end).linear_space_dtype()
        if inner.name == "Float16":
            inner = DType("Float64")
        if inner.name == "Float64":
            RunCast().of(inputs[0], inner)
            RunCast().of(inputs[1], inner)
        if self.array_width is None:
            return DType("List", (inner,))
        return DType("Array", (inner, self.array_width))


@dataclass(frozen=True)
class TemporalRange:
    """Give the type of a range of dates, of datetimes or of times.

    A range of dates casts each bound to a `Datetime` in microseconds, and a
    range of times casts each bound to `Time`, with a cast that is not
    strict. A struct stays a struct in that cast, and its physical values
    are not integers: a function of one range raises, and a function of many
    ranges panics. A range of datetimes reads its bounds as the cast pass gave
    them, and the start must then be a `Datetime`. A function of one range
    needs one value in each bound, and a null bound raises. A literal bound
    whose value the cast loses, such as `-1` cast to `Time`, becomes null
    too. A function of many ranges reads its bounds row by row, and a null
    bound gives a null list. The values take the type `target`, or the type
    of the start where it has none.

    Example:
        >>> dates = TemporalRange(DType("Date"), single=False, cast=DType("Datetime", ("us", None)))
        >>> dates.output([Data(DType("Date")), Data(DType("Date"))], (3, 1), ())
        List(Date)
    """

    target: DType | None
    single: bool
    cast: DType | None = None

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast of a bound fails, or where the
        rows of the bounds do not fit (`RangeRows`).

        Args:
            inputs: The data of the two bounds.
            lengths: The rows of each bound, or none where they are not known.
            computed: Each bound as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The start of a range of datetimes is not a `Datetime`,
                or the bound of a function of one range is null.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> times = TemporalRange(DType("Time"), single=True, cast=DType("Time"))
            >>> inputs = [Data(DType("Int32")), Data(DType("Time"))]
            >>> times.output(inputs, (1, 1), (Operand.of_value(-1),))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: start is an out-of-range time.
        """
        bounds = [self._cast(data) for data in inputs[:2]]
        if self.single:
            RangeRows().single(lengths, ("start", "end"))
        dtype = self._values_type(bounds[0])
        if not self.single:
            RangeRows().matched(lengths, ("start", "end"))
            return DType("List", (dtype,))
        for place, (name, data) in enumerate(zip(("start", "end"), bounds, strict=True)):
            if self._value(data, computed, place).nulls:
                raise PolarsRefusal(ErrorClass.COMPUTE, f"{name} is an out-of-range time.")
        return dtype

    def _values_type(self, start: Data) -> DType:
        """Give the type of the values: `target`, or the type of a `Datetime` start.

        Args:
            start: The data of the start, after the cast of the kernel.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The range takes the type of its start, and the start
                is not a `Datetime`.

        Example:
            >>> RangeKernels.DATETIME._values_type(Data(DType("Date")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: expected Datetime input, ...
        """
        dtype = self.target or start.dtype
        if (  # pragma: no cover -- the cast pass casts the bounds first
            dtype.name != "Datetime" and self.target is None
        ):
            message = f"expected Datetime input, got {dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return dtype

    def _value(self, data: Data, computed: Sequence[Operand], place: int) -> Data:
        """Give a bound of one value, null where the cast of the kernel loses its literal value.

        Args:
            data: The data of the bound after the cast.
            computed: Each bound as the run computes it, or none.
            place: The place of the bound.

        Returns:
            The data of the bound.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> times = TemporalRange(DType("Time"), single=True, cast=DType("Time"))
            >>> times._value(Data(DType("Time")), (Operand.of_value(300),), 0).nulls
            False
        """
        literal = computed[place] if place < len(computed) else None
        if self.cast is None or literal is None or not literal.literal:
            return data
        try:
            SeriesCast(literal.dtype.materialized(), self.cast, strict=True).output([literal.value])
        except PolarsRefusal:
            return Data(data.dtype, nulls=True)
        return data

    def _cast(self, data: Data) -> Data:
        """Give a bound after the cast of the kernel, where it has one.

        Args:
            data: The data of the bound.

        Returns:
            The data after the cast.

        Raises:
            PolarsRefusal: The cast gives a struct, where polars raises or panics.

        Example:
            >>> times = TemporalRange(DType("Time"), single=True, cast=DType("Time"))
            >>> times._cast(Data(DType("Datetime", ("ms", None)))).dtype
            Time
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> times._cast(Data(struct))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: start is an out-of-range time.
        """
        if self.cast is None:
            return data
        cast = RunCast().of(data, self.cast)
        if cast.dtype.name == self.cast.name:
            return cast
        if self.single:
            raise PolarsRefusal(ErrorClass.COMPUTE, "start is an out-of-range time.")
        message = f"invalid series dtype: expected `Int64`, got `{cast.dtype!r}`"
        raise PolarsRefusal(Stop.PANICS, message)


class RangeKernels:
    """Give the kernel of each range of moments that has no option of its own.

    Example:
        >>> RangeKernels.TIMES.target
        Time
    """

    MICROSECONDS: ClassVar[DType] = DType("Datetime", ("us", None))
    DATE: ClassVar[TemporalRange] = TemporalRange(DType("Date"), single=True, cast=MICROSECONDS)
    DATES: ClassVar[TemporalRange] = TemporalRange(DType("Date"), single=False, cast=MICROSECONDS)
    DATETIME: ClassVar[TemporalRange] = TemporalRange(None, single=True)
    DATETIMES: ClassVar[TemporalRange] = TemporalRange(None, single=False)
    TIME: ClassVar[TemporalRange] = TemporalRange(DType("Time"), single=True, cast=DType("Time"))
    TIMES: ClassVar[TemporalRange] = TemporalRange(DType("Time"), single=False, cast=DType("Time"))
