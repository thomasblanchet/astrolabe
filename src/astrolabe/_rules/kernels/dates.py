"""The kernels of the `dt` namespace: the parts of a date or a time, and the functions on moments.

A function of the `dt` namespace reads the data type of its input first. Most
of them accept a few temporal kinds, and raise on every other type: `year`
reads a `Date` or a `Datetime`, `hour` reads a `Datetime` or a `Time`, and
`total_days` reads a `Duration`. The part of a moment has a fixed type, for
example `Int32` for a year and `Int8` for a month. The functions that change a
moment keep its type, or set its unit or its zone.

A few kernels read more than the type:

- `to_string` writes each value with a format of `chrono`, the library of
  dates of Rust. A format can name a part that a type does not have, for
  example the hour of a `Date` (`StrftimeFormat`).
- `timestamp` casts its input to a `Datetime`, so it accepts every type that
  casts to one (`Epoch`).
- `truncate`, `round` and `offset_by` read a duration from text
  (`DurationText`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.binary import OperatorKernel
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
from astrolabe._rules.operators import Operator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class WithUnit:
    """Give a `Datetime` or a `Duration` in another time unit, as the cast of its unit does.

    A `Datetime` keeps its zone. The arm of the dispatch accepts only these
    two kinds.

    Example:
        >>> kernel = DateKernels().cast_time_unit("ms")
        >>> WithUnit("ms").give(Data(DType("Datetime", ("us", "UTC"))), kernel)
        Datetime(time_unit='ms', time_zone='UTC')
    """

    unit: str

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type in the new unit.

        Args:
            data: The data, of a `Datetime` or a `Duration` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> kernel = DateKernels().cast_time_unit("ns")
            >>> WithUnit("ns").give(Data(DType("Duration", ("ms",))), kernel)
            Duration(time_unit='ns')
        """
        del kernel
        if data.dtype.name == "Duration":
            return DType("Duration", (self.unit,))
        return DType("Datetime", (self.unit, Parameter().text(data.dtype, 1)))


@dataclass(frozen=True)
class NaiveMoment:
    """Give a `Datetime` in the unit of the input, with no zone, as `dt.datetime` does.

    A moment with a zone first takes the wall time of its zone. The type
    keeps the unit.

    Example:
        >>> moment = Data(DType("Datetime", ("ns", "UTC")))
        >>> NaiveMoment().give(moment, DateKernels.DATETIME)
        Datetime(time_unit='ns', time_zone=None)
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data, of a `Datetime` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> moment = Data(DType("Datetime", ("ms", None)))
            >>> NaiveMoment().give(moment, DateKernels.DATETIME)
            Datetime(time_unit='ms', time_zone=None)
        """
        del kernel
        return DType("Datetime", (Parameter().unit(data.dtype), None))


@dataclass(frozen=True)
class InZone:
    """Give a `Datetime` in the unit of the input and in a new zone, as `dt.convert_time_zone` does.

    The values stay the same instants. Only the zone that shows them changes.

    Example:
        >>> moment = Data(DType("Datetime", ("us", None)))
        >>> InZone("Asia/Tokyo").give(moment, DateKernels().convert_time_zone("Asia/Tokyo"))
        Datetime(time_unit='us', time_zone='Asia/Tokyo')
    """

    zone: str

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data, of a `Datetime` type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Example:
            >>> moment = Data(DType("Datetime", ("ns", "UTC")))
            >>> InZone("UTC").give(moment, DateKernels().convert_time_zone("UTC")).args
            ('ns', 'UTC')
        """
        del kernel
        return DType("Datetime", (Parameter().unit(data.dtype), self.zone))


@dataclass(frozen=True)
class Epoch:
    """Give the `Int64` count of a unit since the epoch, as `dt.timestamp` and `dt.epoch` do.

    The kernel casts its input to a `Datetime` of the unit with no zone, with
    a cast that is not strict, and reads the integers of the result. So every
    type that casts to a `Datetime` runs: a number counts the unit, and a
    text parses as a moment or gives a null. A cast that has no kernel raises
    its own error. A cast that gives another type, as the cast of a `Struct`
    gives a `Struct`, makes the kernel panic, because it then reads the data
    as a `Datetime`.

    Example:
        >>> Epoch("us").give(Data(DType("Boolean")), DateKernels().timestamp("us"))
        Int64
        >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
        >>> Epoch("us").give(struct, DateKernels().timestamp("us"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    unit: str

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        It raises `PolarsRefusal` where the cast has no kernel (`RunCast`).

        Args:
            data: The data.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type: `Int64`.

        Raises:
            PolarsRefusal: The cast gives a type that is not a `Datetime`.

        Example:
            >>> Epoch("ms").give(Data(DType("Date")), DateKernels().timestamp("ms"))
            Int64
        """
        del kernel
        moment = RunCast().of(data, DType("Datetime", (self.unit, None)))
        if moment.dtype.name != "Datetime":
            message = f"the cast of {data.dtype!r} to a Datetime gives {moment.dtype!r}"
            raise PolarsRefusal(Stop.PANICS, message)
        return DType("Int64")


class StrftimeFormat:
    """Read a format of `chrono`, and give the parts of a moment that it writes.

    `chrono` writes a value with a format such as `%Y-%m-%d %H:%M`. Each
    specifier writes one part of a moment: a part of the date, such as `%Y`, a
    part of the time of day, such as `%H`, or the offset of a zone, such as
    `%z`. A specifier that a value does not have, as the hour of a date, makes
    the write fail. So does a specifier that `chrono` does not know, such as
    `%Q`, for every value. A padding flag, `-`, `_` or `0`, can stand between
    `%` and the specifier.

    Example:
        >>> sorted(StrftimeFormat("%Y-%m-%d %H:%M").parts())
        ['date', 'time']
        >>> sorted(StrftimeFormat("%Q").parts())
        ['unknown']
    """

    DATE: ClassVar[frozenset[str]] = frozenset("YCyqmbBhdeaAwuUWGgVjDxFv")
    TIME: ClassVar[frozenset[str]] = frozenset("HkIlPpMSfRTXr")
    LITERAL: ClassVar[frozenset[str]] = frozenset("tn%")
    PADDING: ClassVar[frozenset[str]] = frozenset("-_0")
    FRACTIONS: ClassVar[tuple[str, ...]] = (".3f", ".6f", ".9f", ".f", "3f", "6f", "9f")
    OFFSETS: ClassVar[tuple[str, ...]] = (":::z", "::z", ":z", "#z", "z", "Z")
    BOTH: ClassVar[dict[str, frozenset[str]]] = {
        "c": frozenset({"date", "time"}),
        "s": frozenset({"date", "time"}),
        "+": frozenset({"date", "time", "offset"}),
    }

    def __init__(self, text: str) -> None:
        """Keep the format.

        Args:
            text: The format.

        Example:
            >>> StrftimeFormat("%F")._text
            '%F'
        """
        self._text = text

    def parts(self) -> frozenset[str]:
        """Give the parts that the format writes: `date`, `time`, `offset`, and `unknown`.

        `unknown` stands for a specifier that `chrono` does not know, which
        fails for every value.

        Returns:
            The parts.

        Example:
            >>> sorted(StrftimeFormat("%s and %:z, 100%%").parts())
            ['date', 'offset', 'time']
            >>> sorted(StrftimeFormat("%-d.%.3f, %Y%").parts())
            ['date', 'time', 'unknown']
        """
        parts: set[str] = set()
        rest = self._text
        while "%" in rest:
            rest = rest[rest.index("%") + 1 :]
            if rest[:1] in self.PADDING:
                rest = rest[1:]
            specifier = self._specifier(rest)
            parts |= self._needs(specifier)
            rest = rest[max(len(specifier), 1) :]
        return frozenset(parts)

    def _specifier(self, rest: str) -> str:
        """Give the specifier at the start of the text after a `%`, or an empty text.

        Args:
            rest: The text after the `%` and its padding flag.

        Returns:
            The specifier, or an empty text where no specifier starts there.

        Example:
            >>> StrftimeFormat("")._specifier(".3f%Y"), StrftimeFormat("")._specifier("")
            ('.3f', '')
        """
        for specifier in (*self.FRACTIONS, *self.OFFSETS):
            if rest.startswith(specifier):
                return specifier
        return rest[:1]

    def _needs(self, specifier: str) -> frozenset[str]:
        """Give the parts that one specifier writes.

        Args:
            specifier: The specifier, with no `%`.

        Returns:
            The parts.

        Example:
            >>> sorted(StrftimeFormat("")._needs("::z")), sorted(StrftimeFormat("")._needs(":"))
            (['offset'], ['unknown'])
        """
        if specifier in self.BOTH:
            return self.BOTH[specifier]
        if specifier in self.OFFSETS:
            return frozenset({"offset"})
        if specifier in self.FRACTIONS or specifier in self.TIME:
            return frozenset({"time"})
        if specifier in self.DATE:
            return frozenset({"date"})
        if specifier in self.LITERAL:
            return frozenset()
        return frozenset({"unknown"})


@dataclass(frozen=True)
class Formatted:
    """Give the `String` of `dt.to_string` and `dt.strftime`, or the refusal of the format.

    A `Duration` accepts only the formats `iso`, `iso:strict` and `polars`,
    and raises an `InvalidOperationError` for any other. The format `polars`
    raises the same error for the other kinds. The formats `iso` and
    `iso:strict` write the parts of each kind, so they never fail. Any other
    format writes each value with `chrono` (`StrftimeFormat`). A `Date` has no
    time of day and no zone, a `Datetime` with no zone has no zone, and a
    `Time` has no date and no zone. A `Datetime` with a zone has every part.
    Where the format needs a part that the kind does not have, the write of
    each value fails. A `Date` or a `Datetime` then raises a `ComputeError`,
    and a `Time` panics. Data of only nulls writes no value, so it does not
    fail.

    Example:
        >>> Formatted("%H:%M").give(Data(DType("Time")), DateKernels().to_string("%H:%M"))
        String
        >>> Formatted("%Y").give(Data(DType("Time")), DateKernels().to_string("%Y"))
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: ...
    """

    pattern: str

    ISO: ClassVar[frozenset[str]] = frozenset({"iso", "iso:strict"})
    DURATION: ClassVar[frozenset[str]] = frozenset({"iso", "iso:strict", "polars"})
    PARTS: ClassVar[dict[str, frozenset[str]]] = {
        "Date": frozenset({"date"}),
        "Time": frozenset({"time"}),
    }

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data, of a temporal type.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type: `String`.

        Raises:
            PolarsRefusal: The format does not suit the kind, or a value of the
                kind cannot take the format.

        Example:
            >>> date = Data(DType("Date"))
            >>> Formatted("polars").give(date, DateKernels().to_string("polars"))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> Formatted("%H").give(date, DateKernels().to_string("%H"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
            >>> nulls = Data(DType("Date"), nulls=True)
            >>> Formatted("%H").give(nulls, DateKernels().to_string("%H"))
            String
        """
        del kernel
        kind = data.dtype.name
        if kind == "Duration" and self.pattern not in self.DURATION:
            message = f"format {self.pattern!r} not supported for Duration type"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if kind != "Duration" and self.pattern == "polars":
            message = f"'polars' is not a valid `to_string` format for {data.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        missing = self._missing(data.dtype)
        if missing and not data.nulls:
            message = f"cannot format {data.dtype!r} with format {self.pattern!r}"
            how = Stop.PANICS if kind == "Time" else ErrorClass.COMPUTE
            raise PolarsRefusal(how, message)
        return DType("String")

    def _missing(self, dtype: DType) -> frozenset[str]:
        """Give the parts that the format writes and that a value of the type does not have.

        Args:
            dtype: The temporal type.

        Returns:
            The parts, with `unknown` for a specifier that `chrono` does not know.

        Example:
            >>> sorted(Formatted("%Y %z")._missing(DType("Datetime", ("ms", None))))
            ['offset']
            >>> Formatted("iso")._missing(DType("Time"))
            frozenset()
        """
        if dtype.name == "Duration" or self.pattern in self.ISO:
            return frozenset()
        has = {"date", "time", "offset"}
        if dtype.name in self.PARTS:
            has = set(self.PARTS[dtype.name])
        elif Parameter().text(dtype, 1) is None:
            has = {"date", "time"}
        return StrftimeFormat(self.pattern).parts() - has


class DateKernels:
    """Hold the dispatch of each kernel of the `dt` namespace that has one input.

    A part of a date, such as `YEAR`, accepts a `Date` or a `Datetime`, and a
    part of the time of day, such as `HOUR`, accepts a `Datetime` or a `Time`.
    Every other type raises an `InvalidOperationError`. `TIME`, `DATE` and
    `DATETIME` take a part of a moment, and raise a `ComputeError` on a kind
    that does not hold it. The other kernels read an option of their node,
    such as the unit of `timestamp`, so a method builds each one.

    Example:
        >>> DateKernels.YEAR.output(Data(DType("Date")))
        Int32
        >>> DateKernels.HOUR.output(Data(DType("Date")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: hour ...
    """

    DATES: ClassVar[Kinds] = Kinds("Date", "Datetime")
    CLOCKS: ClassVar[Kinds] = Kinds("Datetime", "Time")
    INT8: ClassVar[Fixed] = Fixed(DType("Int8"))
    INT16: ClassVar[Fixed] = Fixed(DType("Int16"))
    INT32: ClassVar[Fixed] = Fixed(DType("Int32"))

    MILLENNIUM: ClassVar[Dispatch] = Dispatch("millennium", (Arm(DATES, INT32),), Refuse())
    CENTURY: ClassVar[Dispatch] = Dispatch("century", (Arm(DATES, INT32),), Refuse())
    YEAR: ClassVar[Dispatch] = Dispatch("year", (Arm(DATES, INT32),), Refuse())
    ISO_YEAR: ClassVar[Dispatch] = Dispatch("iso_year", (Arm(DATES, INT32),), Refuse())
    IS_LEAP_YEAR: ClassVar[Dispatch] = Dispatch(
        "is_leap_year",
        (Arm(DATES, Fixed(DType("Boolean"))),),
        Refuse(),
    )
    QUARTER: ClassVar[Dispatch] = Dispatch("quarter", (Arm(DATES, INT8),), Refuse())
    MONTH: ClassVar[Dispatch] = Dispatch("month", (Arm(DATES, INT8),), Refuse())
    DAYS_IN_MONTH: ClassVar[Dispatch] = Dispatch("days_in_month", (Arm(DATES, INT8),), Refuse())
    WEEK: ClassVar[Dispatch] = Dispatch("week", (Arm(DATES, INT8),), Refuse())
    WEEKDAY: ClassVar[Dispatch] = Dispatch("weekday", (Arm(DATES, INT8),), Refuse())
    DAY: ClassVar[Dispatch] = Dispatch("day", (Arm(DATES, INT8),), Refuse())
    ORDINAL_DAY: ClassVar[Dispatch] = Dispatch("ordinal_day", (Arm(DATES, INT16),), Refuse())
    HOUR: ClassVar[Dispatch] = Dispatch("hour", (Arm(CLOCKS, INT8),), Refuse())
    MINUTE: ClassVar[Dispatch] = Dispatch("minute", (Arm(CLOCKS, INT8),), Refuse())
    SECOND: ClassVar[Dispatch] = Dispatch("second", (Arm(CLOCKS, INT8),), Refuse())
    MILLISECOND: ClassVar[Dispatch] = Dispatch("millisecond", (Arm(CLOCKS, INT32),), Refuse())
    MICROSECOND: ClassVar[Dispatch] = Dispatch("microsecond", (Arm(CLOCKS, INT32),), Refuse())
    NANOSECOND: ClassVar[Dispatch] = Dispatch("nanosecond", (Arm(CLOCKS, INT32),), Refuse())
    TIME: ClassVar[Dispatch] = Dispatch(
        "time",
        (Arm(Kinds("Datetime"), Fixed(DType("Time"))), Arm(Kinds("Time"), Same())),
        Refuse(ErrorClass.COMPUTE),
    )
    DATE: ClassVar[Dispatch] = Dispatch(
        "date",
        (Arm(Kinds("Datetime"), Fixed(DType("Date"))), Arm(Kinds("Date"), Same())),
        Refuse(ErrorClass.COMPUTE),
    )
    DATETIME: ClassVar[Dispatch] = Dispatch(
        "datetime",
        (Arm(Kinds("Datetime"), NaiveMoment()),),
        Refuse(ErrorClass.COMPUTE),
    )
    MONTH_START: ClassVar[Dispatch] = Dispatch("month_start", (Arm(DATES, Same()),), Refuse())
    MONTH_END: ClassVar[Dispatch] = Dispatch("month_end", (Arm(DATES, Same()),), Refuse())

    def total(self, fractional: bool) -> Dispatch:
        """Give the kernel of `total_days`, `total_hours` and the other totals of a `Duration`.

        The kernel reads its input as a `Duration`, so another type raises a
        `SchemaError`. It gives the whole count of the unit as `Int64`, or the
        fractional count as `Float64`.

        Args:
            fractional: Whether the count keeps its fraction.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().total(True).output(Data(DType("Duration", ("ms",))))
            Float64
            >>> DateKernels().total(False).output(Data(DType("Date")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: total ...
        """
        dtype = DType("Float64" if fractional else "Int64")
        arms = (Arm(Kinds("Duration"), Fixed(dtype)),)
        return Dispatch("total", arms, Refuse(ErrorClass.SCHEMA))

    def offset(self, name: str) -> Dispatch:
        """Give the kernel of `base_utc_offset` or `dst_offset`.

        The kernel reads a `Datetime` with a zone, and gives the offset of
        each moment in that zone as a `Duration` in milliseconds. Every other
        type, a `Datetime` with no zone among them, raises an
        `InvalidOperationError`.

        Args:
            name: The name of the function.

        Returns:
            The dispatch.

        Example:
            >>> zoned = Data(DType("Datetime", ("ns", "Europe/Paris")))
            >>> DateKernels().offset("dst_offset").output(zoned)
            Duration(time_unit='ms')
        """
        arms = (Arm(self._zoned, Fixed(DType("Duration", ("ms",)))),)
        return Dispatch(name, arms, Refuse())

    def _zoned(self, dtype: DType) -> bool:
        """Say whether a type is a `Datetime` with a zone.

        Args:
            dtype: The data type.

        Returns:
            Whether it is.

        Example:
            >>> DateKernels()._zoned(DType("Datetime", ("ms", None)))
            False
        """
        return dtype.name == "Datetime" and Parameter().text(dtype, 1) is not None

    def timestamp(self, unit: str) -> Dispatch:
        """Give the kernel of `timestamp` and `epoch`.

        A `Time` and a `Duration` raise an `InvalidOperationError`. Every
        other type casts to a `Datetime` of the unit (`Epoch`).

        Args:
            unit: The unit of the count.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().timestamp("ns").output(Data(DType("String")))
            Int64
        """
        arms = (Arm(Kinds("Time", "Duration"), Refuse()),)
        return Dispatch("timestamp", arms, Epoch(unit))

    def to_string(self, pattern: str) -> Dispatch:
        """Give the kernel of `to_string` and `strftime`.

        A temporal type writes its values with the format (`Formatted`).
        Every other type raises an `InvalidOperationError`.

        Args:
            pattern: The format.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().to_string("iso").output(Data(DType("Duration", ("us",))))
            String
        """
        arms = (Arm(Kinds("Date", "Datetime", "Duration", "Time"), Formatted(pattern)),)
        return Dispatch("to_string", arms, Refuse())

    def cast_time_unit(self, unit: str) -> Dispatch:
        """Give the kernel of `cast_time_unit`.

        A `Datetime` and a `Duration` take the new unit, and the values change
        to fit it. Every other type raises a `ComputeError`, because it has no
        time unit.

        Args:
            unit: The new unit: `ms`, `us` or `ns`.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().cast_time_unit("ms").output(Data(DType("Date")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: cast_time_unit ...
        """
        arms = (Arm(Kinds("Datetime", "Duration"), WithUnit(unit)),)
        return Dispatch("cast_time_unit", arms, Refuse(ErrorClass.COMPUTE))

    def with_time_unit(self, unit: str) -> Dispatch:
        """Give the kernel of `with_time_unit`.

        A `Datetime` and a `Duration` take the new unit, and keep the
        integers that hold the values. Every other type raises a
        `ComputeError`, because it has no time unit.

        Args:
            unit: The new unit.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().with_time_unit("ns").output(Data(DType("Duration", ("ms",))))
            Duration(time_unit='ns')
        """
        arms = (Arm(Kinds("Datetime", "Duration"), WithUnit(unit)),)
        return Dispatch("with_time_unit", arms, Refuse(ErrorClass.COMPUTE))

    def convert_time_zone(self, zone: str) -> Dispatch:
        """Give the kernel of `convert_time_zone`.

        A `Datetime` takes the new zone (`InZone`). Every other type raises a
        `ComputeError`.

        Args:
            zone: The new zone.

        Returns:
            The dispatch.

        Example:
            >>> DateKernels().convert_time_zone("UTC").output(Data(DType("Date")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: convert_time_zone ...
        """
        arms = (Arm(Kinds("Datetime"), InZone(zone)),)
        return Dispatch("convert_time_zone", arms, Refuse(ErrorClass.COMPUTE))

    def replace_time_zone(self, zone: str | None) -> ReplaceTimeZone:
        """Give the kernel of `replace_time_zone`, which reads the new zone of its node.

        Args:
            zone: The new zone, or `None` for no zone.

        Returns:
            The kernel.

        Example:
            >>> DateKernels().replace_time_zone("UTC")
            ReplaceTimeZone(zone='UTC')
        """
        return ReplaceTimeZone(zone)

    def combine(self, unit: str) -> Combine:
        """Give the kernel of `combine`, which reads the unit of its node.

        Args:
            unit: The unit of the output.

        Returns:
            The kernel.

        Example:
            >>> DateKernels().combine("us")
            Combine(unit='us')
        """
        return Combine(unit)


@dataclass(frozen=True)
class ParsedDuration:
    """Hold a duration that polars reads from text, such as `1mo2d`, in its four units.

    Polars keeps the months, the weeks, the days and the nanoseconds of a
    duration apart, because a month and a day have no fixed length in a zone.
    The sign applies to the whole duration.

    Example:
        >>> DurationText("-1mo2d").parsed()
        ParsedDuration(months=1, weeks=0, days=2, nanoseconds=0, negative=True)
    """

    months: int
    weeks: int
    days: int
    nanoseconds: int
    negative: bool

    def is_zero(self) -> bool:
        """Say whether every unit is zero.

        Returns:
            Whether it is.

        Example:
            >>> DurationText("").parsed().is_zero(), DurationText("0d1ns").parsed().is_zero()
            (True, False)
        """
        return not any((self.months, self.weeks, self.days, self.nanoseconds))

    def mixes_units(self) -> bool:
        """Say whether more than one of the four units is not zero.

        Returns:
            Whether it does.

        Example:
            >>> mixed, clock = DurationText("1d1h").parsed(), DurationText("1h5m").parsed()
            >>> mixed.mixes_units(), clock.mixes_units()
            (True, False)
        """
        units = (self.months, self.weeks, self.days, self.nanoseconds)
        return sum(1 for unit in units if unit) > 1

    def has_calendar(self) -> bool:
        """Say whether the duration counts months or weeks.

        Returns:
            Whether it does.

        Example:
            >>> DurationText("2w").parsed().has_calendar()
            True
        """
        return bool(self.months or self.weeks)

    def in_unit(self, unit: str) -> int:
        """Give the length of a duration with no months and no weeks in a time unit.

        The days and the nanoseconds each convert to the unit, and the
        nanoseconds round toward zero, as polars does.

        Args:
            unit: The time unit: `ms`, `us` or `ns`.

        Returns:
            The length.

        Example:
            >>> DurationText("1d1us").parsed().in_unit("ms")
            86400000
        """
        nanoseconds_per_unit = 10**9 // TimeUnit.PER_SECOND[unit]
        days = self.days * TimeUnit.per_day(unit)
        return days + self.nanoseconds // nanoseconds_per_unit


class DurationText:
    """Read a duration from text, as the duration strings of polars: `1d`, `2h30m`, `-1mo`.

    The text is a sign, then pairs of an integer and a unit: `ns`, `us`, `ms`,
    `s`, `m`, `h`, `d`, `w`, `mo`, `q`, `y` or `i`. A text that does not have
    this form raises an `InvalidOperationError`. An empty text is a duration of
    zero.

    Example:
        >>> DurationText("1q").parsed().months
        3
        >>> DurationText("1x").parsed()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: unit 'x' ...
    """

    NANOSECONDS: ClassVar[dict[str, int]] = {
        "ns": 1,
        "us": 10**3,
        "ms": 10**6,
        "s": 10**9,
        "m": 60 * 10**9,
        "h": 3600 * 10**9,
        "i": 1,
    }
    MONTHS: ClassVar[dict[str, int]] = {"mo": 1, "q": 3, "y": 12}
    PAIR: ClassVar[re.Pattern[str]] = re.compile(r"(\d+)([A-Za-z]+)")

    def __init__(self, text: str) -> None:
        """Keep the text.

        Args:
            text: The text.

        Example:
            >>> DurationText("1d")._text
            '1d'
        """
        self._text = text

    def parsed(self) -> ParsedDuration:
        """Give the duration.

        Returns:
            The duration.

        Raises:
            PolarsRefusal: The text is not a duration.

        Example:
            >>> DurationText("+1w2d").parsed()
            ParsedDuration(months=0, weeks=1, days=2, nanoseconds=0, negative=False)
            >>> DurationText("1d-").parsed()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        negative = self._text.startswith("-")
        body = self._text[1:] if self._text[:1] in {"-", "+"} else self._text
        units = {"months": 0, "weeks": 0, "days": 0, "nanoseconds": 0}
        position = 0
        while position < len(body):
            pair = self.PAIR.match(body, position)
            if pair is None:
                message = (
                    f"the duration string {self._text!r} has no integer and unit at {position}"
                )
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            self._add(units, int(pair.group(1)), pair.group(2))
            position = pair.end()
        return ParsedDuration(negative=negative, **units)

    def _add(self, units: dict[str, int], count: int, unit: str) -> None:
        """Add one pair of an integer and a unit to the units of the duration.

        Args:
            units: The months, the weeks, the days and the nanoseconds so far.
            count: The integer.
            unit: The unit.

        Raises:
            PolarsRefusal: Polars does not know the unit.

        Example:
            >>> units = {"months": 0, "weeks": 0, "days": 0, "nanoseconds": 0}
            >>> DurationText("")._add(units, 2, "y")
            >>> units["months"]
            24
        """
        if unit in self.NANOSECONDS:
            units["nanoseconds"] += count * self.NANOSECONDS[unit]
        elif unit in self.MONTHS:
            units["months"] += count * self.MONTHS[unit]
        elif unit in {"d", "w"}:
            units["days" if unit == "d" else "weeks"] += count
        else:
            message = f"unit {unit!r} is not a unit of a duration"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class InputValues:
    """Read the value of an input of a kernel where the input is a literal of the code.

    The value of a column is a fact of the data, so a rule of the types does
    not know it. The value of a literal is in the code, so a kernel that reads
    it meets that value at every run.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> computed = [Operand.column(DType("Date")), Operand.of_value("1d")]
        >>> InputValues(computed).text(1), InputValues(computed).text(0) is None
        ('1d', True)
    """

    def __init__(self, computed: Sequence[Operand]) -> None:
        """Keep each input as the run computes it.

        Args:
            computed: The inputs, or none where they are not known.

        Example:
            >>> InputValues([])._computed
            ()
        """
        self._computed = tuple(computed)

    def literal(self, index: int) -> Operand | None:
        """Give the input at a place where it is a literal.

        Args:
            index: The place of the input.

        Returns:
            The literal, or `None` where the input is not a literal or is not known.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> InputValues([Operand.of_value(1)]).literal(0)
            Operand(dtype=Unknown(int, 1), literal=True, value=1)
        """
        if index >= len(self._computed) or not self._computed[index].literal:
            return None
        return self._computed[index]

    def text(self, index: int) -> str | None:
        """Give the text of a literal at a place.

        Args:
            index: The place of the input.

        Returns:
            The text, or `None` where the input is not a literal of text.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> InputValues([Operand.of_value(1)]).text(0) is None
            True
        """
        literal = self.literal(index)
        if literal is None or not isinstance(literal.value, str):
            return None
        return literal.value


class TextInput:
    """Refuse the data of an input that a kernel reads as text, as a read of a `String` series does.

    Example:
        >>> TextInput().require(Data(DType("String")), "every")
        >>> TextInput().require(Data(DType("Null")), "every")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
    """

    def require(self, data: Data, name: str) -> None:
        """Refuse data that is not text.

        Args:
            data: The data of the input.
            name: The name of the input, for the message.

        Raises:
            PolarsRefusal: The data is not `String` data. Polars raises a `SchemaError`.

        Example:
            >>> TextInput().require(Data(DType("String"), nulls=True), "ambiguous")
        """
        if data.dtype.name != "String":
            message = f"{name}: invalid series dtype: expected `String`, got `{data.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


class Lengths:
    """Check the lengths of the inputs of a kernel that spreads an input of one row.

    Each input has one length, or one row. Polars names the error of each
    kernel, and most raise a `ShapeError`.

    Example:
        >>> Lengths("dt.offset_by").check([5, 1])
        >>> Lengths("dt.offset_by").check([5, 2])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ShapeError: arguments for `dt.offset_by` ...
    """

    def __init__(self, name: str, how: ErrorClass | Stop = ErrorClass.SHAPE) -> None:
        """Keep the name of the function and how it refuses.

        Args:
            name: The name of the function, for the message.
            how: The class of the error, or the way that the kernel stops.

        Example:
            >>> Lengths("dt.round")._how
            <ErrorClass.SHAPE: 'ShapeError'>
        """
        self._name = name
        self._how = how

    def check(self, lengths: Sequence[int]) -> None:
        """Refuse inputs of two lengths where neither is one row.

        Args:
            lengths: The number of rows of each input, or none where they are not known.

        Raises:
            PolarsRefusal: Two inputs have other lengths, and neither has one row.

        Example:
            >>> Lengths("dt.round").check([])
        """
        others = {length for length in lengths if length != 1}
        if len(others) > 1:
            message = f"arguments for `{self._name}` have different lengths {tuple(lengths)}"
            raise PolarsRefusal(self._how, message)


@dataclass(frozen=True)
class EveryKernel:
    """Give the type of `dt.truncate` or `dt.round`, at the data of the moment and of `every`.

    The kernel reads `every` as text, so other data raises a `SchemaError`. It
    reads a `Date` or a `Datetime`, and raises an `InvalidOperationError` on
    another type. Then it checks the lengths, and it keeps the type of the
    moment. Where `every` is a literal, its text is in the code, so the kernel
    meets it at every run (`DurationText`): a text that is not a duration
    raises, and so does a negative duration. A `Datetime` with no zone or in
    `UTC` and a duration with no months and no weeks take a fast path of
    integers. There, `truncate` accepts a duration of zero, and `round`
    divides by it and panics, also over nulls. Every other case truncates each
    value that is not null with the calendar, which refuses a duration of
    zero or one that mixes its units.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> moment = Data(DType("Datetime", ("ns", "Europe/Paris")))
        >>> computed = [Operand.column(moment.dtype), Operand.of_value("1d1h")]
        >>> EveryKernel("round").output([moment, Data(DType("String"))], [5, 1], computed)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    name: str

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the moment and of `every`.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it.

        Returns:
            The type of the moment.

        Raises:
            PolarsRefusal: The moment is not a `Date` or a `Datetime`.

        Example:
            >>> day = Data(DType("Date"))
            >>> EveryKernel("truncate").output([day, Data(DType("String"))], [5, 5], [])
            Date
        """
        moment, every = inputs[0], inputs[1]
        TextInput().require(every, "every")
        if moment.dtype.name not in {"Date", "Datetime"}:
            message = f"`{self.name}` operation not supported for dtype `{moment.dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        Lengths(f"dt.{self.name}").check(lengths)
        text = InputValues(computed).text(1)
        if text is not None:
            self._duration(moment, DurationText(text).parsed())
        return moment.dtype

    def _duration(self, moment: Data, duration: ParsedDuration) -> None:
        """Refuse a duration of the code that the kernel refuses at every run.

        Args:
            moment: The data of the moment.
            duration: The duration.

        Raises:
            PolarsRefusal: The duration is negative, or the kernel refuses it.

        Example:
            >>> day = Data(DType("Date"), nulls=True)
            >>> EveryKernel("truncate")._duration(day, DurationText("").parsed())
        """
        if duration.negative:
            message = f"cannot {self.name} to a negative duration"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if self._fast(moment.dtype, duration):
            self._fast_zero(moment.dtype, duration)
        elif not moment.nulls and (duration.is_zero() or duration.mixes_units()):
            message = "the duration is zero, or it mixes month, week, day and sub-daily units"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _fast(self, dtype: DType, duration: ParsedDuration) -> bool:
        """Say whether the kernel takes its fast path of integers.

        Args:
            dtype: The type of the moment.
            duration: The duration.

        Returns:
            Whether it does: a `Datetime` with no zone or in `UTC`, and no months or weeks.

        Example:
            >>> zoned = DType("Datetime", ("ms", "UTC"))
            >>> EveryKernel("round")._fast(zoned, DurationText("1d").parsed())
            True
        """
        if dtype.name != "Datetime" or duration.has_calendar():
            return False
        return Parameter().text(dtype, 1) in {None, "UTC"}

    def _fast_zero(self, dtype: DType, duration: ParsedDuration) -> None:
        """Panic where `round` takes the fast path with a duration of zero in the unit of the data.

        Args:
            dtype: The `Datetime` type of the moment.
            duration: The duration.

        Raises:
            PolarsRefusal: The kernel divides by zero.

        Example:
            >>> moment = DType("Datetime", ("ms", None))
            >>> EveryKernel("round")._fast_zero(moment, DurationText("1d").parsed())
        """
        if self.name == "round" and duration.in_unit(Parameter().unit(dtype)) == 0:
            message = "attempt to calculate the remainder with a divisor of zero"
            raise PolarsRefusal(Stop.PANICS, message)


class OffsetBy:
    """Give the type of `dt.offset_by`, at the data of the moment and of the offset.

    The kernel reads the offset as text, so other data raises a `SchemaError`.
    Then it checks the lengths, and it reads a `Date` or a `Datetime`: another
    type raises an `InvalidOperationError`. It keeps the type of the moment.
    Where the offset is a literal, the kernel parses its text at every run
    (`DurationText`). A negative offset, a zero and mixed units all run.

    Example:
        >>> OffsetBy().output([Data(DType("Date")), Data(DType("String"))], [5, 1], [])
        Date
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the moment and of the offset.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it.

        Returns:
            The type of the moment.

        Raises:
            PolarsRefusal: The moment is not a `Date` or a `Datetime`.

        Example:
            >>> moment = Data(DType("Time"))
            >>> OffsetBy().output([moment, Data(DType("String"))], [5, 1], [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        moment, offset = inputs[0], inputs[1]
        TextInput().require(offset, "offset")
        Lengths("dt.offset_by").check(lengths)
        if moment.dtype.name not in {"Date", "Datetime"}:
            message = f"expected Date or Datetime, got {moment.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        text = InputValues(computed).text(1)
        if text is not None:
            DurationText(text).parsed()
        return moment.dtype


@dataclass(frozen=True)
class ReplaceTimeZone:
    """Give the type of `dt.replace_time_zone`, at the data of the moment and of `ambiguous`.

    The kernel reads the moment as a `Datetime`, and panics on another type.
    The planner refuses another type first. It reads `ambiguous` as text, so
    other data raises a `SchemaError`. The output takes the new zone. The kernel
    changes no value where the old zone and the new zone are one zone, and
    that zone is `UTC` or `ambiguous` is the literal `raise`. A moment with no
    zone counts as `UTC`. Otherwise it reads `ambiguous` for each value. An
    `ambiguous` of more than one row pairs its values with the values of the
    moment, with no check of the lengths, so the output takes the shorter
    length. A literal `ambiguous` must be `raise`, `earliest`, `latest` or
    `null`, or each value that is not null raises.

    Example:
        >>> moment = Data(DType("Datetime", ("ns", None)))
        >>> ReplaceTimeZone("UTC").output([moment, Data(DType("String"))], [5, 2], [])
        Datetime(time_unit='ns', time_zone='UTC')
    """

    zone: str | None

    AMBIGUOUS: ClassVar[frozenset[str]] = frozenset({"raise", "earliest", "latest", "null"})

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the moment and of `ambiguous`.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it.

        Returns:
            The `Datetime` in the new zone.

        Raises:
            PolarsRefusal: The moment is not a `Datetime`, or a literal
                `ambiguous` is not one that polars knows.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> moment = Data(DType("Datetime", ("ns", "Europe/Paris")))
            >>> computed = [Operand.column(moment.dtype), Operand.of_value("a")]
            >>> ReplaceTimeZone("UTC").output([moment, Data(DType("String"))], [5, 1], computed)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> day = Data(DType("Date"))
            >>> ReplaceTimeZone("UTC").output([day, Data(DType("String"))], [5, 1], computed)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: invalid series dtype: expected ...
        """
        moment, ambiguous = inputs[0], inputs[1]
        if (  # pragma: no cover -- the type function refuses it first
            moment.dtype.name != "Datetime"
        ):
            message = f"invalid series dtype: expected `Datetime`, got `{moment.dtype!r}`"
            raise PolarsRefusal(Stop.PANICS, message)
        TextInput().require(ambiguous, "ambiguous")
        output = DType("Datetime", (Parameter().unit(moment.dtype), self.zone))
        text = InputValues(computed).text(1)
        del lengths
        if self._unchanged(moment.dtype, text):
            return output
        if text is not None and text not in self.AMBIGUOUS and not moment.nulls:
            message = f"invalid `ambiguous` value: {text!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return output

    def _unchanged(self, dtype: DType, ambiguous: str | None) -> bool:
        """Say whether the kernel keeps the values and reads no `ambiguous`.

        Args:
            dtype: The `Datetime` type of the moment.
            ambiguous: The literal text of `ambiguous`, or `None`.

        Returns:
            Whether the old zone is the new zone, and it is `UTC` or `ambiguous` is `raise`.

        Example:
            >>> ReplaceTimeZone(None)._unchanged(DType("Datetime", ("ms", "UTC")), None)
            True
        """
        old = Parameter().text(dtype, 1) or "UTC"
        new = self.zone or "UTC"
        return old == new and (old == "UTC" or ambiguous == "raise")


@dataclass(frozen=True)
class Combine:
    """Give the type of `dt.combine`, which puts a time of day on each date.

    The kernel reads a `Date` or a `Datetime`, and raises a `ComputeError` on
    another type. It casts the moment to a `Date`, then to a `Datetime` of the
    unit with no zone. It casts the time to a `Duration` of the unit, with a
    cast that is not strict, and adds the two series, as `+` of two series
    does (`OperatorKernel`). The two series need one length, or one row. A
    moment with a zone then takes its zone back, so the sum must be a
    `Datetime`, or the kernel panics.

    Example:
        >>> Combine("ms").output([Data(DType("Date")), Data(DType("Time"))], [5, 1])
        Datetime(time_unit='ms', time_zone=None)
    """

    unit: str

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast has no kernel or the sum refuses
        the data (`RunCast`, `OperatorKernel`).

        Args:
            inputs: The data of the moment and of the time.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The moment is not a `Date` or a `Datetime`.

        Example:
            >>> moment = Data(DType("Datetime", ("us", "UTC")))
            >>> Combine("ns").output([moment, Data(DType("Int8"))], [5, 5])
            Datetime(time_unit='ns', time_zone='UTC')
            >>> Combine("ns").output([Data(DType("Int8")), Data(DType("Time"))], [5, 5])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: expected Date or Datetime, ...
        """
        moment, time = inputs[0], inputs[1]
        if (  # pragma: no cover -- the type function refuses it first
            moment.dtype.name not in {"Date", "Datetime"}
        ):
            message = f"expected Date or Datetime, got {moment.dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        zone = Parameter().text(moment.dtype, 1) if moment.dtype.name == "Datetime" else None
        start = moment.with_dtype(DType("Datetime", (self.unit, None)))
        span = RunCast().of(time, DType("Duration", (self.unit,)))
        Lengths("dt.combine", ErrorClass.INVALID_OPERATION).check(lengths)
        total = OperatorKernel(owned=False).output(Operator.PLUS, start, span)
        if zone is None:
            return total
        if total.name != "Datetime":  # pragma: no cover
            # A cast that succeeds gives a Duration, and the sum of a Datetime and a Duration
            # is a Datetime.
            message = f"the sum of a date and a time gives {total!r}"
            raise PolarsRefusal(Stop.PANICS, message)
        return DType("Datetime", (self.unit, zone))


@dataclass(frozen=True)
class DatePart:
    """Name one part of a moment that `dt.replace` sets: its place, its type, and its range.

    The kernel casts each part strictly to its type, so a literal that does not
    fit raises at every run. A literal of the right type can still name no
    moment, as a month of 13 does. Such a part refuses each value that is not
    null. A day of 31 names no moment in February only, so that case is a fact
    of the data.

    Example:
        >>> DatePart("month", 2, DType("Int8"), 1, 12).holds(13)
        False
    """

    name: str
    index: int
    dtype: DType
    low: int
    high: int

    def holds(self, value: int) -> bool:
        """Say whether a value lies in the range of the part.

        Args:
            value: The value, as an integer.

        Returns:
            Whether it does.

        Example:
            >>> DatePart("hour", 4, DType("Int8"), 0, 23).holds(0)
            True
        """
        return self.low <= value <= self.high


class StrictParts:
    """Cast the parts that `dt.replace` reads to their types, as a strict cast of each series does.

    A cast that has no kernel raises its own error, and so does a strict cast
    of a literal whose value does not fit (`SeriesCast`). A cast that gives
    another type, as the cast of a `Struct` gives a `Struct`, makes the read
    of the series raise a `SchemaError`. The kernel casts every part first,
    then it reads each one. A value of a column that does not fit is a fact of
    the data.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> part = DatePart("day", 0, DType("Int8"), 1, 31)
        >>> StrictParts([Data(DType("Int32"))], [Operand.of_value(300)]).cast([part])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    def __init__(self, inputs: Sequence[Data], computed: Sequence[Operand]) -> None:
        """Keep the data of each input and each input as the run computes it.

        Args:
            inputs: The data of each input of the node.
            computed: Each input as the run computes it.

        Example:
            >>> len(StrictParts([], [])._inputs)
            0
        """
        self._inputs = tuple(inputs)
        self._values = InputValues(computed)

    def cast(self, parts: Sequence[DatePart]) -> None:
        """Cast the parts, then read each one as its type.

        Args:
            parts: The parts, in the order that the kernel casts them.

        Raises:
            PolarsRefusal: A cast gives another type than the part.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> part = DatePart("year", 0, DType("Int32"), 1, 9999)
            >>> StrictParts([Data(struct)], []).cast([part])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        casts = [self._cast(part) for part in parts]
        for part, dtype in zip(parts, casts, strict=True):
            if dtype.name != part.dtype.name:
                message = f"{part.name}: invalid series dtype: expected `{part.dtype!r}`"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)

    def _cast(self, part: DatePart) -> DType:
        """Give the type that the strict cast of one part gives.

        Args:
            part: The part.

        Returns:
            The type.

        Example:
            >>> part = DatePart("year", 0, DType("Int32"), 1, 9999)
            >>> StrictParts([Data(DType("Boolean"))], [])._cast(part)
            Int32
        """
        data = self._inputs[part.index]
        literal = self._values.literal(part.index)
        if literal is not None:
            SeriesCast(data.dtype, part.dtype, strict=True).output([literal.value])
        return RunCast().of(data, part.dtype).dtype

    def value(self, part: DatePart) -> int | None:
        """Give the integer that a literal part holds after its cast.

        Args:
            part: The part.

        Returns:
            The integer, or `None` where the part is not a literal of a number.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> part = DatePart("day", 0, DType("Int8"), 1, 31)
            >>> StrictParts([Data(DType("Float64"))], [Operand.of_value(2.5)]).value(part)
            2
        """
        literal = self._values.literal(part.index)
        if literal is None:
            return None
        value = literal.value
        if isinstance(value, bool | int):
            return int(value)
        if isinstance(value, float) and math.isfinite(value):
            return math.trunc(value)
        return None


class Replace:
    """Give the type of `dt.replace`, which sets some parts of each moment.

    The node has nine inputs: the moment, the year, the month, the day, the
    hour, the minute, the second, the microsecond and `ambiguous`. A part that
    the call does not set is a null literal, and the moment keeps that part.
    The kernel casts the year, the month and the day strictly (`StrictParts`).
    Then it reads a `Date` or a `Datetime`, and raises an
    `InvalidOperationError` on another type. A `Date` reads no other part.
    A `Datetime` casts the parts of its time of day and `ambiguous` too, and
    checks that each input has one length, or one row. The kernel keeps the
    type of the moment. A literal part outside the range of the part names no
    moment, so it refuses each value that is not null with a `ComputeError`.
    A moment in nanoseconds holds only the years from 1677 to 2262, so a
    literal year outside them panics.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Data(DType("Date"))] + [Data(DType("Null"))] * 7 + [Data(DType("String"))]
        >>> computed = [Operand.column(DType("Date"))] + [Operand.of_value(None)] * 8
        >>> Replace().output(inputs, [5] + [1] * 8, computed)
        Date
    """

    DATE: ClassVar[tuple[DatePart, ...]] = (
        DatePart("year", 1, DType("Int32"), -262_144, 262_143),
        DatePart("month", 2, DType("Int8"), 1, 12),
        DatePart("day", 3, DType("Int8"), 1, 31),
    )
    CLOCK: ClassVar[tuple[DatePart, ...]] = (
        DatePart("hour", 4, DType("Int8"), 0, 23),
        DatePart("minute", 5, DType("Int8"), 0, 59),
        DatePart("second", 6, DType("Int8"), 0, 59),
        DatePart("microsecond", 7, DType("Int32"), 0, 1_999_999),
        DatePart("ambiguous", 8, DType("String"), 0, 0),
    )
    NANOSECOND_YEARS: ClassVar[tuple[int, int]] = (1677, 2262)

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a part does not cast (`StrictParts`).

        Args:
            inputs: The data of the nine inputs.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it.

        Returns:
            The type of the moment.

        Raises:
            PolarsRefusal: The moment is not a `Date` or a `Datetime`.

        Example:
            >>> inputs = [Data(DType("Time"))] + [Data(DType("Null"))] * 8
            >>> Replace().output(inputs, [], [])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        moment = inputs[0]
        parts = StrictParts(inputs, computed)
        parts.cast(self.DATE)
        if moment.dtype.name == "Date":
            self._date_lengths(lengths)
            self._ranges(moment, parts, self.DATE)
            return moment.dtype
        if moment.dtype.name != "Datetime":
            message = f"`round` operation not supported for dtype `{moment.dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        parts.cast(self.CLOCK)
        Lengths("dt.replace").check(lengths)
        self._ranges(moment, parts, self.DATE + self.CLOCK[:-1])
        self._nanosecond_year(moment, parts)
        return moment.dtype

    def _date_lengths(self, lengths: Sequence[int]) -> None:
        """Refuse a part of a `Date` whose length is neither one row nor the length of the moment.

        A `Date` of one row takes the length of a longer part, so it refuses nothing.

        Args:
            lengths: The number of rows of each input, or none where they are not known.

        Raises:
            PolarsRefusal: A part has another length. Polars raises a `ShapeError`.

        Example:
            >>> Replace()._date_lengths([1, 5, 2])
            >>> Replace()._date_lengths([5, 1, 2])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: the parts of `dt.replace` ...
        """
        if not lengths or lengths[0] == 1:
            return
        others = [length for length in lengths[1:4] if length not in {1, lengths[0]}]
        if others:
            message = f"the parts of `dt.replace` have the lengths {tuple(lengths[:4])}"
            raise PolarsRefusal(ErrorClass.SHAPE, message)

    def _ranges(self, moment: Data, parts: StrictParts, names: Sequence[DatePart]) -> None:
        """Refuse a literal part outside its range, where the moment holds a value.

        Args:
            moment: The data of the moment.
            parts: The parts after their casts.
            names: The parts to check.

        Raises:
            PolarsRefusal: A literal part names no moment.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> parts = StrictParts([Data(DType("Int8"))] * 3, [Operand.of_value(-1)] * 3)
            >>> Replace()._ranges(Data(DType("Date"), nulls=True), parts, Replace.DATE)
        """
        if moment.nulls:
            return
        for part in names:
            value = parts.value(part)
            if value is not None and not part.holds(value):
                message = f"invalid {part.name} {value} supplied"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def _nanosecond_year(self, moment: Data, parts: StrictParts) -> None:
        """Panic where a moment in nanoseconds takes a literal year that its integers cannot hold.

        Args:
            moment: The data of the moment, a `Datetime`.
            parts: The parts after their casts.

        Raises:
            PolarsRefusal: The year is outside the years that the unit holds.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> parts = StrictParts([Data(DType("Int32"))] * 2, [Operand.of_value(2020)] * 2)
            >>> Replace()._nanosecond_year(Data(DType("Datetime", ("ns", None))), parts)
        """
        year = parts.value(self.DATE[0])
        if moment.nulls or year is None or Parameter().unit(moment.dtype) != "ns":
            return
        low, high = self.NANOSECOND_YEARS
        if not low <= year <= high:
            message = f"the year {year} does not fit a moment in nanoseconds"
            raise PolarsRefusal(Stop.PANICS, message)


class Holidays:
    """Refuse the holidays of a function of business days that are not a `List(Date)`.

    The kernels of business days read the holidays as a list of dates in each
    row, and raise a `ComputeError` on any other type.

    Example:
        >>> Holidays().require(Data(DType("List", (DType("Date"),))))
        >>> Holidays().require(Data(DType("Date")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    LIST: ClassVar[DType] = DType("List", (DType("Date"),))

    def require(self, data: Data) -> None:
        """Refuse holidays of another type.

        Args:
            data: The data of the holidays.

        Raises:
            PolarsRefusal: The holidays are not a `List(Date)`.

        Example:
            >>> Holidays().require(Data(Holidays.LIST, nulls=True))
        """
        if data.dtype != self.LIST:  # pragma: no cover -- the type function refuses it first
            message = f"dtype of holidays list must be List(Date), got {data.dtype!r} instead"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


@dataclass(frozen=True)
class WeekMask:
    """Hold the days of the week that count as business days, from Monday to Sunday.

    A week with no business day raises a `ComputeError` in each kernel of
    business days, before it reads any input.

    Example:
        >>> WeekMask((False,) * 7).require()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    days: tuple[bool, ...]

    def require(self) -> None:
        """Refuse a week with no business day.

        Raises:
            PolarsRefusal: No day of the week is a business day.

        Example:
            >>> WeekMask((True,) * 5 + (False,) * 2).require()
        """
        if not any(self.days):
            message = "`week_mask` must have at least one business day"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


@dataclass(frozen=True)
class IsBusinessDay:
    """Give the type of `dt.is_business_day`, at the data of the dates and of the holidays.

    The kernel reads a `Date`, or a `Datetime` that it casts to a `Date`, and
    raises an `InvalidOperationError` on another type. Then it reads the
    holidays (`Holidays`), and it checks that the two inputs have one length,
    or that one has one row. It gives a `Boolean`.

    Example:
        >>> holidays = Data(Holidays.LIST)
        >>> IsBusinessDay(WeekMask((True,) * 7)).output([Data(DType("Date")), holidays], [5, 1])
        Boolean
    """

    week: WeekMask

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the dates and of the holidays.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `Boolean`.

        Raises:
            PolarsRefusal: The dates are not a `Date` or a `Datetime`.

        Example:
            >>> holidays = Data(Holidays.LIST)
            >>> week = WeekMask((True,) * 7)
            >>> IsBusinessDay(week).output([Data(DType("Time")), holidays], [5, 1])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        self.week.require()
        dates, holidays = inputs[0], inputs[1]
        if dates.dtype.name not in {"Date", "Datetime"}:
            message = f"expected date or datetime, got {dates.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        Holidays().require(holidays)
        Lengths("is_business_day").check(lengths)
        return DType("Boolean")


@dataclass(frozen=True)
class AddBusinessDays:
    """Give the type of `dt.add_business_days`, at the data of the start, `n` and the holidays.

    The kernel reads a `Date`, or a `Datetime` whose date it moves and whose
    time of day it keeps. It raises an `InvalidOperationError` on another
    type. Then it reads the holidays (`Holidays`), checks that the inputs have
    one length or one row, and reads `n`: an `Int32`, or an `Int64`, a
    `UInt32` or a `UInt64` that it casts to `Int32`. Another type of `n`
    raises an `InvalidOperationError`. It keeps the type of the start. A start
    date that is not a business day raises where the roll of the node is
    `raise`, which is a fact of the data.

    Example:
        >>> holidays = Data(Holidays.LIST)
        >>> moment = Data(DType("Datetime", ("ms", "UTC")))
        >>> week = WeekMask((True,) * 7)
        >>> AddBusinessDays(week).output([moment, Data(DType("UInt64")), holidays], [5, 1, 1])
        Datetime(time_unit='ms', time_zone='UTC')
    """

    week: WeekMask

    COUNTS: ClassVar[frozenset[str]] = frozenset({"Int32", "Int64", "UInt32", "UInt64"})

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the start, of `n` and of the holidays.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            The type of the start.

        Raises:
            PolarsRefusal: The start is not a `Date` or a `Datetime`, or `n` is
                not one of the integers that the kernel reads.

        Example:
            >>> inputs = [Data(DType("Date")), Data(DType("Int8")), Data(Holidays.LIST)]
            >>> AddBusinessDays(WeekMask((True,) * 7)).output(inputs, [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        self.week.require()
        start, count, holidays = inputs[0], inputs[1], inputs[2]
        if start.dtype.name not in {"Date", "Datetime"}:
            message = f"expected date or datetime, got {start.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        Holidays().require(holidays)
        Lengths("add_business_days").check(lengths)
        if count.dtype.name not in self.COUNTS:
            message = f"expected Int64, Int32, UInt64, or UInt32, got {count.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return start.dtype


@dataclass(frozen=True)
class BusinessDayCount:
    """Give the type of `pl.business_day_count`, at the data of the start, the end and the holidays.

    The kernel reads the start and the end as a `Date`, so any other type
    raises a `SchemaError`, a `Datetime` among them. Then it reads the
    holidays (`Holidays`), and it checks that the inputs have one length or
    one row. It gives an `Int32`.

    Example:
        >>> day, holidays = Data(DType("Date")), Data(Holidays.LIST)
        >>> BusinessDayCount(WeekMask((True,) * 7)).output([day, day, holidays], [5, 5, 1])
        Int32
    """

    week: WeekMask

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the start, of the end and of the holidays.
            lengths: The number of rows of each input, or none where they are not known.

        Returns:
            `Int32`.

        Raises:
            PolarsRefusal: The start or the end is not a `Date`.

        Example:
            >>> moment, holidays = Data(DType("Datetime", ("ms", None))), Data(Holidays.LIST)
            >>> week = WeekMask((True,) * 7)
            >>> BusinessDayCount(week).output([moment, moment, holidays], [])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        self.week.require()
        for data in inputs[:2]:
            if data.dtype.name != "Date":
                message = f"invalid series dtype: expected `Date`, got `{data.dtype!r}`"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
        Holidays().require(inputs[2])
        Lengths("business_day_count").check(lengths)
        return DType("Int32")


class BusinessKernels:
    """Build the kernels of the functions of business days, from the options of their node.

    Each function holds the days of the week that count as business days.

    Example:
        >>> BusinessKernels().is_business_day((True,) * 5 + (False,) * 2).week.days[5]
        False
    """

    def is_business_day(self, week_mask: tuple[bool, ...]) -> IsBusinessDay:
        """Give the kernel of `dt.is_business_day`.

        Args:
            week_mask: The business days, from Monday to Sunday.

        Returns:
            The kernel.

        Example:
            >>> BusinessKernels().is_business_day((True,) * 7)
            IsBusinessDay(week=WeekMask(days=(True, True, True, True, True, True, True)))
        """
        return IsBusinessDay(WeekMask(week_mask))

    def add_business_days(self, week_mask: tuple[bool, ...]) -> AddBusinessDays:
        """Give the kernel of `dt.add_business_days`.

        Args:
            week_mask: The business days, from Monday to Sunday.

        Returns:
            The kernel.

        Example:
            >>> type(BusinessKernels().add_business_days((True,) * 7)).__name__
            'AddBusinessDays'
        """
        return AddBusinessDays(WeekMask(week_mask))

    def business_day_count(self, week_mask: tuple[bool, ...]) -> BusinessDayCount:
        """Give the kernel of `pl.business_day_count`.

        Args:
            week_mask: The business days, from Monday to Sunday.

        Returns:
            The kernel.

        Example:
            >>> type(BusinessKernels().business_day_count((True,) * 7)).__name__
            'BusinessDayCount'
        """
        return BusinessDayCount(WeekMask(week_mask))
