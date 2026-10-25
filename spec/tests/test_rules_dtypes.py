"""Check the facts of the parameters in `astrolabe._rules.dtypes` against polars.

Many rules read the same facts of a data type: the range of each integer
type, the fit of an integer in a `Decimal`, the size and the join of the time
units, and the type that an integer literal takes alone. Each test here asks
polars for one fact: a series that holds the ends of a range, the wrap of an
integer that leaves it, a strict cast to a `Decimal`, the physical value of a
`Duration` and of a `Datetime`, the relaxed concat of two durations, and the
type of a literal.
"""

from __future__ import annotations

import datetime
from typing import ClassVar, Literal

import polars as pl
import pytest

from spec.tests.rule_probes import PolarsAnswer, Stopped, TreeFrame
from astrolabe._rules.dtypes import Decimals, DType, DynamicInt, IntegerRange, TimeUnit


class Values:
    """Give the parameters and the values that the tests of the facts read.

    Example:
        >>> Values.DECIMALS[0]
        (5, 2)
    """

    DECIMALS: ClassVar[tuple[tuple[int, int], ...]] = ((5, 2), (3, 3), (10, 0), (38, 2))
    INTEGERS: ClassVar[tuple[int, ...]] = (0, 9, 99, 999, 1000, -999, -1000, 10**36)
    UNITS: ClassVar[tuple[Literal["ms", "us", "ns"], ...]] = ("ms", "us", "ns")
    LITERALS: ClassVar[tuple[int, ...]] = (
        0,
        -1,
        300,
        2**31,
        -(2**31) - 1,
        2**63,
        2**64,
        -(2**63) - 1,
    )

    def integer_type(self, name: str) -> pl.DataType:
        """Give the polars data type of an integer type by its name.

        Args:
            name: The name, for example `Int8`.

        Returns:
            The data type.

        Example:
            >>> Values().integer_type("UInt16")
            UInt16
        """
        dtype: pl.DataType = getattr(pl, name)()
        return dtype


class StrictCast:
    """Run a strict cast of polars on one integer, and say whether it runs.

    A strict cast refuses a value that the target type does not hold, so it
    shows where a type holds a value.

    Example:
        >>> StrictCast().runs(999, pl.Decimal(5, 2)), StrictCast().runs(1000, pl.Decimal(5, 2))
        (True, False)
    """

    def runs(self, value: int, target: pl.DataType) -> bool:
        """Say whether the cast of a series of the value to the type runs.

        Args:
            value: The integer value.
            target: The type of the cast.

        Returns:
            Whether polars gives a series, and does not stop.

        Example:
            >>> StrictCast().runs(300, pl.Int8())
            False
        """
        series = pl.Series([value])
        cast = PolarsAnswer().result(lambda: series.cast(target))
        return not isinstance(cast, Stopped)


class TestIntegerRange:
    """Check the range of each integer type against the series of polars."""

    @pytest.mark.parametrize("name", list(IntegerRange.BOUNDS))
    def test_type_holds_the_ends_of_its_range_and_no_more(self, name: str) -> None:
        """A series of the type holds the low end and the high end, and refuses one past them."""
        low, high = IntegerRange.BOUNDS[name]
        dtype = Values().integer_type(name)
        assert pl.Series([low, high], dtype=dtype).to_list() == [low, high]
        assert IntegerRange().holds(name, low)
        assert IntegerRange().holds(name, high)
        assert not IntegerRange().holds(name, high + 1)
        past = PolarsAnswer().result(lambda: pl.Series([high + 1], dtype=dtype))
        assert isinstance(past, Stopped)

    @pytest.mark.parametrize("name", list(IntegerRange.BOUNDS))
    def test_a_sum_past_the_high_end_wraps_as_polars_wraps(self, name: str) -> None:
        """The sum of the high end and one wraps to the value that `wrapped` gives."""
        _, high = IntegerRange.BOUNDS[name]
        dtype = Values().integer_type(name)
        series = pl.Series([high], dtype=dtype) + pl.Series([1], dtype=dtype)
        assert series.to_list() == [IntegerRange().wrapped(name, high + 1)]


class TestDecimals:
    """Check the fit of an integer in a `Decimal` against a strict cast of polars."""

    @pytest.mark.parametrize(("precision", "scale"), Values.DECIMALS, ids=repr)
    def test_decimal_holds_an_integer_where_a_strict_cast_keeps_it(
        self,
        precision: int,
        scale: int,
    ) -> None:
        """A strict cast of an integer to the decimal runs exactly where the decimal holds it."""
        target = pl.Decimal(precision, scale)
        problems = []
        for value in Values.INTEGERS:
            runs = StrictCast().runs(value, target)
            if runs != Decimals().holds(value, precision, scale):
                problems.append(f"{value}: polars runs {runs}")
        assert not problems, "\n".join(problems)

    def test_a_sum_of_decimals_takes_the_widest_decimal(self) -> None:
        """The sum of a `Decimal` has the largest precision and the scale of its input."""
        frame = pl.DataFrame({"x": pl.Series([1], dtype=pl.Decimal(10, 2))})
        summed = frame.select(pl.col("x").sum()).dtypes[0]
        assert DType.of(summed) == Decimals().widest(2)


class TestTimeUnit:
    """Check the time units against the values and the joins of polars."""

    @pytest.mark.parametrize("unit", Values.UNITS)
    def test_unit_counts_its_values_in_a_second_and_in_a_day(
        self,
        unit: Literal["ms", "us", "ns"],
    ) -> None:
        """One second of a `Duration` and one day of a `Datetime` count as the unit says."""
        second = pl.Series([datetime.timedelta(seconds=1)], dtype=pl.Duration(unit))
        day = pl.Series([datetime.date(1970, 1, 2)]).cast(pl.Datetime(unit))
        assert second.to_physical().to_list() == [TimeUnit.PER_SECOND[unit]]
        assert day.to_physical().to_list() == [TimeUnit.per_day(unit)]

    @pytest.mark.parametrize("left", Values.UNITS)
    def test_two_units_join_as_the_relaxed_concat_joins_them(
        self,
        left: Literal["ms", "us", "ns"],
    ) -> None:
        """The relaxed concat of two durations takes the unit that `joined` gives."""
        problems = []
        for right in Values.UNITS:
            frames = [pl.LazyFrame(schema={"x": pl.Duration(unit)}) for unit in (left, right)]
            joined = pl.concat(frames, how="vertical_relaxed").collect_schema()["x"]
            if DType.of(joined) != DType("Duration", (TimeUnit.joined(left, right),)):
                problems.append(f"{left} and {right}: polars {joined}")
        assert not problems, "\n".join(problems)


class TestDynamicInt:
    """Check the type that an integer literal takes alone against polars."""

    @pytest.mark.parametrize("value", Values.LITERALS)
    def test_literal_alone_takes_the_type_of_its_value(self, value: int) -> None:
        """A literal with no data type takes the first type of its list that holds its value."""
        dtype = pl.select(pl.lit(value)).dtypes[0]
        assert DType.of(dtype) == DynamicInt.standalone(value)


def test_an_enum_of_many_categories_holds_its_ids_in_32_bits() -> None:
    """An `Enum` of more categories than 16 bits can number holds each id as a `UInt32`."""
    many = pl.Enum(["a", "b", "c", "d", "e", *(f"k{index}" for index in range(70_000))])
    expression = pl.col("t").cast(many).to_physical()
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)
