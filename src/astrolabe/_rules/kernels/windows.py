"""The kernels of the functions whose value at a row reads other rows of the input.

A cumulative function, such as `cum_max`, reads every row before the row. A
rolling function, such as `rolling_mean`, reads a window of rows around the
row, of a fixed number of rows or of a range of a key column (`rolling_mean_by`).
An exponentially weighted function, such as `ewm_mean`, reads every row
before the row with a weight. `shift`, `diff` and `pct_change` read the row at
a distance that a count gives. `rank`, `is_unique` and `is_first_distinct`
compare the row with the other rows. Each such kernel gives one value for each
row of its input, so its output has the length of its input.

Each kernel dispatches on the data type of its input, and its output type can
differ from the planner type: a rolling sum widens a small integer to `Int64`,
and a rolling mean casts every type that is not a float to `Float64`. A
rolling kernel checks the options of its window, and a kernel of two inputs
checks the second input, for example the key of `rolling_mean_by` or the count
of `shift`. Each class here states one kernel, or one part that several kernels
share, in the form of the other kernel modules: a `Dispatch` for a kernel of one
input, and a kernel of all the inputs (`InputsKernel`) for a kernel of two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.cast_pass import CastFold
from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.arithmetic import SeriesArithmetic
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
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
    ToPhysical,
)
from astrolabe._rules.kernels.zipping import SameTypeZip
from astrolabe._rules.operators import NumericOp
from astrolabe._rules.releases import polars_releases
from astrolabe._rules.supertype import Supertype
from astrolabe._rules.type_functions import Operand, WindowClosures

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.kernels.dispatch import Output


class WindowCheck(Protocol):
    """Check the options of the window of a rolling kernel, where the kernel reaches its window."""

    def check(self, data: Data) -> None:
        """Raise `PolarsRefusal` where the window refuses its options, its key or its data."""


@dataclass(frozen=True)
class FixedWindow:
    """Hold the options of a rolling window of a fixed number of rows, which its kernel checks.

    The kernel checks that a window needs no more rows than it holds:
    `min_periods`, the rows that a window needs to give a value, is at most
    `window_size`. It checks this only where it reaches the window of
    numbers, after its checks of the type.

    A window can hold a weight for each of its rows. The kernel of data with
    nulls has no weights, and panics. `measure` names the result that a sum of
    weights of zero leaves undefined, `variance` or `quantile`, which the
    kernel refuses before it panics where the count of weights is not the
    window size.

    Example:
        >>> FixedWindow(2, 3).check(Data(DType("Float64")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `min_periods` should be ...
    """

    window_size: int
    min_periods: int
    weights: tuple[float, ...] | None = None
    measure: str | None = None

    def check(self, data: Data) -> None:
        """Refuse a window that needs more rows than it holds, or weights that the data refuses.

        Args:
            data: The numbers of the window.

        Raises:
            PolarsRefusal: `min_periods` is more than `window_size`, or the
                weights do not fit the data or the window.

        Example:
            >>> FixedWindow(2, 2).check(Data(DType("Float64")))
            >>> FixedWindow(2, 2, (1.0,)).check(Data(DType("Float64")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: the count of weights is not the ...
            >>> FixedWindow(2, 2, (1.0, 1.0)).check(Data(DType("Float64"), nulls=True))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: weights not yet supported ...
        """
        if self.min_periods > self.window_size:
            message = "`min_periods` should be <= `window_size`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self.weights is None:
            return
        if data.nulls:
            raise PolarsRefusal(Stop.PANICS, "weights not yet supported on array with null values")
        if self.measure is not None and sum(self.weights) == 0:
            message = f"Weighted {self.measure} is undefined if weights sum to 0"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if len(self.weights) != self.window_size:
            raise PolarsRefusal(Stop.PANICS, "the count of weights is not the window size")


@dataclass(frozen=True)
class WindowDuration:
    """Hold a duration of polars, such as the window of `rolling_mean_by`.

    Polars parses a text such as `1d`, `2h` or `3i` into months, weeks, days
    and nanoseconds, a sign, and a flag that says that the text counts rows
    of an integer key (`3i`). The kernels read these parts, and not the
    length of the duration. The half life of `ewm_mean_by` has the same form.

    Example:
        >>> parts = {"months": 0, "weeks": 0, "days": 1, "nsecs": 0}
        >>> WindowDuration.of({**parts, "negative": False, "parsed_int": False}).days
        1
    """

    months: int
    weeks: int
    days: int
    nsecs: int
    negative: bool
    parsed_int: bool

    INTEGER_KEYS: ClassVar[frozenset[str]] = frozenset({"Int64", "UInt64", "Int32", "UInt32"})
    TIME_KEYS: ClassVar[frozenset[str]] = frozenset({"Datetime", "Date", "Duration", "Time"})

    @classmethod
    def of(cls, part: object) -> WindowDuration:
        """Read a duration as polars serializes it in the options of a node.

        Args:
            part: The duration, as a map of its parts.

        Returns:
            The duration.

        Raises:
            TypeError: The part is not a duration.

        Example:
            >>> WindowDuration.of("1d")
            Traceback (most recent call last):
            ...
            TypeError: no duration in '1d'
        """
        if not isinstance(part, dict):  # pragma: no cover -- polars writes a duration here
            message = f"no duration in {part!r}"
            raise TypeError(message)
        return cls(
            months=int(part["months"]),
            weeks=int(part["weeks"]),
            days=int(part["days"]),
            nsecs=int(part["nsecs"]),
            negative=bool(part["negative"]),
            parsed_int=bool(part["parsed_int"]),
        )

    def zero(self) -> bool:
        """Say whether the duration is zero.

        Returns:
            Whether each part is zero.

        Example:
            >>> WindowDuration(0, 0, 0, 0, negative=False, parsed_int=True).zero()
            True
        """
        return self.months == 0 and self.weeks == 0 and self.days == 0 and self.nsecs == 0

    def constant(self, zone: str | None) -> bool:
        """Say whether the duration has one length whatever the date, in a time zone.

        A month has no constant length. In a time zone other than `UTC`, a day
        and a week have none either, because of daylight saving time.

        Args:
            zone: The time zone of the key, or `None`.

        Returns:
            Whether the length is constant.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> day.constant(None), day.constant("Europe/Paris")
            (True, False)
        """
        if zone is None or zone == "UTC":
            return self.months == 0
        return self.months == 0 and self.weeks == 0 and self.days == 0

    def check_key(self, key: DType, name: str) -> None:
        """Refuse a duration that does not fit the type of its key column.

        An integer key needs a duration that counts rows (`2i`), and a
        temporal key needs a duration of time (`2d`). A zero duration fits an
        integer key. Every other type of key is refused.

        Args:
            key: The type of the key.
            name: The name of the option, for the message.

        Raises:
            PolarsRefusal: The duration does not fit the key.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> day.check_key(DType("Date"), "window_size")
            >>> day.check_key(DType("Int64"), "window_size")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `window_size` ...
        """
        if key.name in self.INTEGER_KEYS:
            fits = self.parsed_int or self.zero()
        elif key.name in self.TIME_KEYS:
            fits = not self.parsed_int
        else:
            fits = False
        if not fits:
            message = f"`{name}` duration does not fit a key of type {key!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


@dataclass(frozen=True)
class RollingIndex:
    """Check the index column of a `rolling` in a `select`, which places the window of each row.

    The period and the offset of the window count time, such as `2d`, or
    index steps, such as `2i`. The index must hold a value at each row, and
    each duration must fit its type (`WindowDuration`). The index must then be
    a `Date`, a `Datetime`, or an integer of 32 or 64 bits. A `Duration` or a
    `Time` index fits a duration of time, and polars then refuses its type. The
    period must be more than zero.

    Example:
        >>> day = WindowDuration(0, 0, 2, 0, negative=False, parsed_int=False)
        >>> back = replace(day, negative=True)
        >>> RollingIndex(day, back).check(Data(DType("Date")), grouped=False)
        >>> RollingIndex(day, back).check(Data(DType("Int32")), grouped=False)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    KEYS: ClassVar[frozenset[str]] = frozenset({
        "Date",
        "Datetime",
        "Int32",
        "Int64",
        "UInt32",
        "UInt64",
    })

    period: WindowDuration
    offset: WindowDuration

    def check(self, data: Data, *, grouped: bool) -> None:
        """Refuse an index that cannot place the windows.

        In `agg`, polars checks neither the period nor the fit of the
        durations, so an integer index fits a duration of time there.

        Args:
            data: The data of the index.
            grouped: Whether the `rolling` runs in `agg`.

        Raises:
            PolarsRefusal: The period is not more than zero, the index holds
                only nulls, or its type does not fit.

        Example:
            >>> day = WindowDuration(0, 0, 2, 0, negative=False, parsed_int=False)
            >>> RollingIndex(day, day).check(Data(DType("Int32")), grouped=True)
            >>> RollingIndex(day, day).check(Data(DType("Date"), nulls=True), grouped=True)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: null values in `rolling` ...
            >>> zero = WindowDuration(0, 0, 0, 0, negative=False, parsed_int=False)
            >>> RollingIndex(zero, day).check(Data(DType("Date")), grouped=False)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: rolling window period should ...
        """
        if not grouped and (self.period.zero() or self.period.negative):
            message = "rolling window period should be strictly positive"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if data.nulls:
            raise PolarsRefusal(ErrorClass.COMPUTE, "null values in `rolling` not supported")
        if not grouped:
            self.period.check_key(data.dtype, "period")
            self.offset.check_key(data.dtype, "offset")
        if data.dtype.name not in self.KEYS:
            message = f"expected a Date, a Datetime or an integer index, got {data.dtype!r}"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class SortedCheck:
    """Check that polars can tell whether a key column is sorted, as `ewm_mean_by` asks first.

    Polars reads whether the key is sorted before it runs the kernel. A key
    with one value, or with only nulls, is sorted. A struct compares its
    encoded rows, and a number, a text, a `Boolean` or a logical type with a
    physical type of numbers compares its values. Any other type compares
    each value with the next with `<=` (`Comparison`), which refuses a list.

    Example:
        >>> SortedCheck().check(Data(DType("Date")))
        >>> SortedCheck().check(Data(DType("List", (DType("Int8"),))))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    ORDERED: ClassVar[frozenset[str]] = frozenset({"Struct", "Boolean", "String", "Binary"})

    def check(self, key: Data) -> None:
        """Refuse a key whose order polars cannot read.

        Args:
            key: The data of the key.

        Example:
            >>> SortedCheck().check(Data(DType("List", (DType("Int8"),)), one_row=True))
        """
        dtype = key.dtype
        if key.nulls or key.one_row or dtype.name in self.ORDERED:
            return
        if Is.primitive_numeric(dtype.physical()) or dtype.name == "Categorical":
            return
        Comparison(CompareOp.LT_EQ).output(key, key)


class DynamicWindow:
    """Check the key and the duration of a window that a key column orders.

    The kernel checks, in order: the key has as many rows as the values; the
    duration fits the type of the key (`WindowDuration.check_key`); the
    duration is more than zero; and the key is a `Datetime`, a `Date`, or an
    integer of 32 or 64 bits. It makes these checks only where it reaches
    the window of numbers, after its checks of the type of the values.

    Example:
        >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
        >>> DynamicWindow(day, Data(DType("Date")), (3, 3)).check(Data(DType("Float64")))
        >>> numbers = Data(DType("Float64"))
        >>> DynamicWindow(day, Data(DType("Date")), (3, 1)).check(numbers)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `by` column ...
    """

    KEYS: ClassVar[frozenset[str]] = frozenset({
        "Datetime",
        "Date",
        "Int64",
        "Int32",
        "UInt64",
        "UInt32",
    })

    def __init__(self, duration: WindowDuration, key: Data, lengths: Sequence[int]) -> None:
        """Keep the duration, the key and the number of rows of the values and of the key.

        Args:
            duration: The duration of the window.
            key: The data of the key column.
            lengths: The rows of the values and of the key, or none where they are not known.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> DynamicWindow(day, Data(DType("Date")), ())._lengths
            ()
        """
        self._duration = duration
        self._key = key
        self._lengths = tuple(lengths)

    def check(self, data: Data) -> None:
        """Refuse the window where the key or the duration does not fit.

        Args:
            data: The numbers of the window, which the checks do not read.

        Raises:
            PolarsRefusal: A check of the window fails.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> numbers = Data(DType("Float64"))
            >>> DynamicWindow(day, Data(DType("Time")), (3, 3)).check(numbers)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: in `rolling_*_by` ...
        """
        del data
        if len(set(self._lengths)) > 1:
            message = "`by` column in `rolling_*_by` must be the same length as values column"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        duration = self._duration
        duration.check_key(self._key.dtype, "window_size")
        if duration.zero() or duration.negative:
            message = "`window_size` must be strictly positive"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if self._key.dtype.name not in self.KEYS:
            message = f"in `rolling_*_by` operation, `by` argument of dtype {self._key.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class Windowed:
    """Give the output of an arm of a rolling kernel after the checks of its window.

    A rolling kernel first dispatches on the type of its values, and casts
    them. It checks its window only where it reaches the window of numbers,
    so an arm that refuses the type refuses before the window.

    Example:
        >>> mean = Dispatch("rolling_mean", (), Windowed(Same(), FixedWindow(2, 3)))
        >>> mean.output(Data(DType("Float64")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `min_periods` should be ...
    """

    def __init__(self, output: Output, window: WindowCheck) -> None:
        """Keep what the arm gives, and the checks of its window.

        Args:
            output: What the arm gives after the checks.
            window: The checks of the window.

        Example:
            >>> Windowed(Same(), FixedWindow(2, 1))._window
            FixedWindow(window_size=2, min_periods=1, weights=None, measure=None)
        """
        self._output = output
        self._window = window

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Check the window, and give the type of the arm.

        Args:
            data: The data that the arm accepts.
            kernel: The dispatch that holds the arm.

        Returns:
            The type.

        Example:
            >>> sum_ = Dispatch("rolling_sum", (), Refuse())
            >>> Windowed(Fixed(DType("Int64")), FixedWindow(2, 1)).give(Data(DType("Int8")), sum_)
            Int64
        """
        self._window.check(data)
        return self._output.give(data, kernel)


class FirstRow:
    """Give `Boolean` for an input of one row, before the dispatch on its type.

    The kernel of `is_first_distinct` and of `is_last_distinct` answers
    `true` for a series of one row, whatever its type. Only a longer series
    reaches the dispatch on the type, which refuses `Null`, an `Array` and a
    list of nested values.

    Example:
        >>> first = Dispatch("is_first_distinct", (), FirstRow(Dispatch("is_first", (), Refuse())))
        >>> first.output(Data(DType("Null"), one_row=True))
        Boolean
    """

    def __init__(self, longer: Dispatch) -> None:
        """Keep the dispatch of a series of more than one row.

        Args:
            longer: The dispatch.

        Example:
            >>> FirstRow(WindowKernels.UNIQUE)._longer.name
            'is_unique'
        """
        self._longer = longer

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output.

        Args:
            data: The data of the input.
            kernel: The dispatch that holds this output, which it does not read.

        Returns:
            `Boolean`.

        Example:
            >>> first = Dispatch("is_first_distinct", (), Refuse())
            >>> FirstRow(WindowKernels.UNIQUE).give(Data(DType("Int8")), first)
            Boolean
        """
        del kernel
        if data.one_row:
            return DType("Boolean")
        return self._longer.output(data)


class ScalarKept:
    """Keep the type of an input of one value, and run the dispatch of a series otherwise.

    Polars holds a literal or an aggregation as one scalar value, and a few
    kernels, such as `reverse`, give that value back with no kernel. Only a
    series of rows reaches the kernel, so only a series meets its refusals.

    Example:
        >>> kept = Dispatch("reverse", (), ScalarKept(Dispatch("reverse", (), Refuse())))
        >>> kept.output(Data(DType("Null"), one_row=True))
        Null
    """

    def __init__(self, longer: Dispatch) -> None:
        """Keep the dispatch of a series of rows.

        Args:
            longer: The dispatch.

        Example:
            >>> ScalarKept(WindowKernels.SAME)._longer.name
            'reverse'
        """
        self._longer = longer

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output.

        Args:
            data: The data of the input.
            kernel: The dispatch that holds this output, which it does not read.

        Returns:
            The type of the input for one value, else the type of the dispatch.

        Example:
            >>> nested = DType("Array", (DType("String"), 2))
            >>> outer = Dispatch("reverse", (), Refuse())
            >>> ScalarKept(WindowKernels.SAME).give(Data(nested), outer)
            Array(String, shape=(2,))
        """
        del kernel
        if data.one_row:
            return data.dtype
        return self._longer.output(data)


@dataclass(frozen=True)
class NullFill:
    """Give the type of a fill of nulls by a constant, `zero` or `one`, over data of only nulls.

    Polars fills by a constant only where a value is null, so data with no
    null keeps its type with no work. Data of only nulls fills through its
    physical type: a number or a `Boolean` takes the constant, text and
    binary data take the empty value for `zero` and refuse `one`, and every
    other physical type refuses. A `Decimal` refuses `one` where the value 1
    does not fit its precision and scale.

    Example:
        >>> NullFill(one=False).give(Data(DType("Null")), WindowKernels.SAME)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: reverse ...
    """

    one: bool

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output, or raise the refusal of polars.

        Args:
            data: The data of the input.
            kernel: The dispatch that holds this output, whose name an error takes.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: Data of only nulls whose physical type takes no constant.

        Example:
            >>> narrow = DType("Decimal", (2, 2))
            >>> NullFill(one=True).give(Data(narrow, nulls=True), WindowKernels.SAME)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: value '1' is out of range ...
            >>> NullFill(one=True).give(Data(DType("Int8"), nulls=True), WindowKernels.SAME)
            Int8
            >>> NullFill(one=False).give(Data(DType("String"), nulls=True), WindowKernels.SAME)
            String
        """
        physical = data.dtype.physical()
        if not data.nulls:
            return data.dtype
        if physical.name == "Boolean" or Is.primitive_numeric(physical):
            self._fit(data.dtype)
            return data.dtype
        if physical.name in {"String", "Binary"} and not self.one:
            return data.dtype
        message = f"{kernel.name} does not accept {data.dtype!r}"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _fit(self, dtype: DType) -> None:
        """Raise where the value 1 does not fit a `Decimal`, as a fill by `one` needs.

        Args:
            dtype: The type of the data.

        Raises:
            PolarsRefusal: The type is a `Decimal` whose precision is at most its scale.

        Example:
            >>> NullFill(one=True)._fit(DType("Decimal", (3, 2)))
        """
        if not self.one or dtype.name != "Decimal":
            return
        precision, scale = dtype.args
        if precision <= scale:  # type: ignore[operator]
            message = f"value '1' is out of range for Decimal({precision}, {scale})"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class NonNumericArray:
    """Test whether a type is an `Array` whose inner type is not a number, which `reverse` refuses.

    Example:
        >>> NonNumericArray()(DType("Array", (DType("Date"), 2)))
        True
        >>> NonNumericArray()(DType("Array", (DType("Int8"), 2)))
        False
    """

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type is an `Array` of values that are not numbers.

        Args:
            dtype: The type.

        Returns:
            Whether it is.

        Example:
            >>> NonNumericArray()(DType("List", (DType("String"),)))
            False
        """
        return dtype.name == "Array" and not Is.primitive_numeric(dtype.inner())


class FlatList:
    """Test whether a type is a list of values that are not nested, as `is_first_distinct` needs.

    Example:
        >>> flat = DType("List", (DType("Int8"),))
        >>> FlatList()(flat), FlatList()(DType("List", (flat,)))
        (True, False)
    """

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type is a list whose inner type is not nested.

        Args:
            dtype: The type.

        Returns:
            Whether it is.

        Example:
            >>> FlatList()(DType("Array", (DType("Int8"), 2)))
            False
        """
        return dtype.name == "List" and not Is.nested(dtype.inner())


class QuantileCut:
    """Give the type of `qcut` at the data of its input: the categories of its bins.

    The kernel refuses a quantile that is NaN. It casts the input to
    `Float64`, and a cast with no kernel raises. A struct casts each field and
    stays a struct, so the kernel finds no floats and raises a `SchemaError`.
    The cast turns a value that does not fit into null, so the value of a
    literal input decides whether any float is left. An input of only nulls
    gives nulls, with no more checks (`_only_nulls`). Before polars 1.42.0,
    the kernel counts the nulls before the cast, so an input that only the
    cast makes null panics (`_finds_only_nulls`). Otherwise the kernel
    refuses a quantile outside of 0 to 1, then breaks that are not unique
    unless the call allows them, then labels that do not number one more than
    the quantiles. One row gives the same break for each quantile. The options
    of a node give NaN for a quantile that is NaN or infinite, which the rule
    cannot tell apart where the input holds only nulls.

    Example:
        >>> qcut = QuantileCut((0.5,), None, include_breaks=False)
        >>> qcut.output([Data(DType("Int8"))], (), ())
        Categorical
    """

    FLOAT64: ClassVar[DType] = DType("Float64")

    def __init__(
        self,
        probs: Sequence[float],
        labels: Sequence[str] | None,
        *,
        include_breaks: bool,
        allow_duplicates: bool = False,
    ) -> None:
        """Keep the options of the call.

        Args:
            probs: The quantiles of the breaks.
            labels: The labels of the bins, or `None`.
            include_breaks: Whether the output holds the upper break of each value.
            allow_duplicates: Whether the call accepts breaks that are not unique.

        Example:
            >>> QuantileCut((0.5,), ("a", "b"), include_breaks=True)._labels
            ('a', 'b')
        """
        self._probs = tuple(probs)
        self._labels = None if labels is None else tuple(labels)
        self._dtype = WindowClosures().qcut(include_breaks=include_breaks)
        self._allow_duplicates = allow_duplicates

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input.
            lengths: The rows of the input, which the kernel does not read.
            computed: The input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A check of the options fails, or the input has no floats.
            UndecidedError: The answer depends on whether a quantile is NaN or infinite.

        Example:
            >>> wide = QuantileCut((1.5,), None, include_breaks=False)
            >>> wide.output([Data(DType("Int8"))], (), ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: quantile should be between ...
            >>> wide.output([Data(DType("Null"))], (), ())
            Categorical
        """
        del lengths
        unknown = any(math.isnan(prob) for prob in self._probs)
        if unknown and inputs[0].nulls:
            question = "the error of qcut over a quantile that is NaN or infinite"
            raise UndecidedError(Undecided.NOT_SERIALIZED, question)
        if unknown:
            raise PolarsRefusal(ErrorClass.COMPUTE, "quantiles cannot be NaN")
        data = self._cast(inputs[0], computed)
        if data.dtype.name != "Float64":
            message = f"invalid series dtype: expected `Float64`, got `{data.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        if self._finds_only_nulls(inputs[0], data):
            return self._only_nulls()
        self._check(data)
        return self._with_values(inputs[0])

    @polars_releases
    def _finds_only_nulls(self, source: Data, floats: Data) -> bool:
        """Say whether the kernel finds only nulls, which it counts after the cast.

        Args:
            source: The data of the input, before the cast.
            floats: The data after the cast.

        Returns:
            Whether the floats hold only nulls.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> qcut = QuantileCut((0.5,), None, include_breaks=False)
            >>> text, empty = Data(DType("String")), Data(DType("Float64"), nulls=True)
            >>> with PolarsRelease.pretend("1.42.0"):
            ...     qcut._finds_only_nulls(text, empty)
            True
        """
        del source
        return floats.nulls

    @_finds_only_nulls.before("1.42.0")
    def _counts_before_the_cast(self, source: Data, floats: Data) -> bool:
        """Say whether the kernel finds only nulls, which the older releases count before the cast.

        Where only the cast makes every value null, the kernel checks the
        range of the quantiles. Then it finds no break, and it panics.

        Args:
            source: The data of the input, before the cast.
            floats: The data after the cast.

        Returns:
            Whether the input holds only nulls.

        Raises:
            PolarsRefusal: The input holds a value, and the cast makes every value null.

        Example:
            >>> qcut = QuantileCut((0.5,), None, include_breaks=False)
            >>> text, empty = Data(DType("String")), Data(DType("Float64"), nulls=True)
            >>> qcut._counts_before_the_cast(Data(DType("Null"), nulls=True), empty)
            True
            >>> qcut._counts_before_the_cast(text, Data(DType("Float64")))
            False
            >>> qcut._counts_before_the_cast(text, empty)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: qcut finds no value after the cast
        """
        if source.nulls:
            return True
        if floats.nulls:
            self._check_range()
            raise PolarsRefusal(Stop.PANICS, "qcut finds no value after the cast")
        return False

    @polars_releases
    def _only_nulls(self) -> DType:
        """Give the type of the output over an input of only nulls: the type of the options.

        Returns:
            The type, a struct where the call includes the breaks.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     QuantileCut((0.5,), None, include_breaks=True)._only_nulls().name
            'Struct'
        """
        return self._dtype

    @_only_nulls.before("1.43.0")
    def _categories_only(self) -> DType:
        """Give `Categorical` over an input of only nulls, as the older releases do, breaks or not.

        Returns:
            The type.

        Example:
            >>> QuantileCut((0.5,), None, include_breaks=True)._categories_only()
            Categorical
        """
        return DType("Categorical")

    @polars_releases
    def _with_values(self, source: Data) -> DType:
        """Give the type of the output over an input that can hold values: the type of the options.

        Args:
            source: The data of the input, before the cast.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> qcut = QuantileCut((0.5,), None, include_breaks=True)
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     qcut._with_values(Data(DType("String"))).name
            'Struct'
        """
        del source
        return self._dtype

    @_with_values.before("1.42.0")
    def _struct_or_panic(self, source: Data) -> DType:
        """Give the type over an input that can hold values, as the releases before 1.42.0 give it.

        Where the cast leaves no value, the kernel panics, so a run that ends
        gives the type of the options (`_counts_before_the_cast`).

        Args:
            source: The data of the input, before the cast.

        Returns:
            The type.

        Example:
            >>> qcut = QuantileCut((0.5,), None, include_breaks=True)
            >>> qcut._struct_or_panic(Data(DType("String"))).name
            'Struct'
        """
        del source
        return self._dtype

    @_with_values.before("1.43.0")
    def _values_decide(self, source: Data) -> DType:
        """Give the type over an input that can hold values, as the older releases give it.

        Where the call includes the breaks, the older releases give `Categorical`
        where the cast leaves no value, and the struct otherwise. A text column
        holds values that the cast turns into nulls where they do not parse, so
        its values decide the type.

        Args:
            source: The data of the input, before the cast.

        Returns:
            The type.

        Raises:
            UndecidedError: The input is text and the call includes the breaks.

        Example:
            >>> qcut = QuantileCut((0.5,), None, include_breaks=True)
            >>> qcut._values_decide(Data(DType("Int8"))).name
            'Struct'
            >>> qcut._values_decide(Data(DType("String")))
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: values: the type of qcut with breaks ...
        """
        if self._dtype.name == "Struct" and source.dtype.name == "String":
            question = "the type of qcut with breaks over text, which the values decide"
            raise UndecidedError(Undecided.VALUES, question)
        return self._dtype

    def _cast(self, data: Data, computed: Sequence[Operand]) -> Data:
        """Give the data after the cast to `Float64`, with nulls where a literal value does not fit.

        Args:
            data: The data of the input.
            computed: The input as the run computes it, or none.

        Returns:
            The data after the cast.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> qcut = QuantileCut((0.5,), None, include_breaks=False)
            >>> qcut._cast(Data(DType("String")), [Operand.of_value("a")]).nulls
            True
        """
        floats = RunCast().of(data, self.FLOAT64)
        operand = computed[0] if computed else None
        if operand is not None and operand.literal and not CastFold().keeps(operand, self.FLOAT64):
            return replace(floats, nulls=True)
        return floats

    def _check(self, data: Data) -> None:
        """Refuse a quantile outside of 0 to 1, repeated breaks, or a wrong count of labels.

        Args:
            data: The floats of the input, which decide the breaks where they hold one row.

        Raises:
            PolarsRefusal: A check fails.

        Example:
            >>> pair = QuantileCut((0.25, 0.75), None, include_breaks=False)
            >>> pair._check(Data(DType("Float64"), one_row=True))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: quantiles are not unique ...
        """
        self._check_range()
        repeated = len(set(self._probs)) < len(self._probs)
        if not self._allow_duplicates and (repeated or (data.one_row and len(self._probs) > 1)):
            message = "quantiles are not unique while allow_duplicates=False"
            raise PolarsRefusal(ErrorClass.DUPLICATE, message)
        if self._labels is not None and len(self._labels) != len(self._probs) + 1:
            raise PolarsRefusal(ErrorClass.SHAPE, "provide len(quantiles) + 1 labels")

    def _check_range(self) -> None:
        """Refuse a quantile outside of 0 to 1.

        Raises:
            PolarsRefusal: A quantile is outside of 0 to 1.

        Example:
            >>> QuantileCut((1.5,), None, include_breaks=False)._check_range()
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: quantile should be between ...
        """
        if any(not 0.0 <= prob <= 1.0 for prob in self._probs):
            message = "quantile should be between 0.0 and 1.0"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


class EwmFallback:
    """Give the last arm of `ewm_mean` and `ewm_sum`, over a type that is not a float.

    From polars 1.43.0, the kernel calls itself again with no cast, without
    end, so the process never returns. Older releases cast the input to
    `Float64` first, as `ewm_std` does, so a struct, which stays a struct,
    makes the process crash (`Recast`). The cast pass casts every other type
    to `Float64` before the kernel runs, so a struct is the one type that
    reaches this arm.

    Example:
        >>> from astrolabe._rules.releases import PolarsRelease
        >>> nested = Data(DType("Struct", ((("a", DType("Int8")),),)))
        >>> with PolarsRelease.pretend("1.42.1"):
        ...     EwmFallback().give(nested, WindowKernels.EWM_LOOP)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: crashes: ewm_mean calls ...
    """

    @polars_releases
    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Raise that the process never returns.

        Args:
            data: The data.
            kernel: The dispatch that holds the arm.

        Returns:
            Never: the arm always refuses.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> nested = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("1.43.0"):
            ...     EwmFallback().give(nested, WindowKernels.EWM_LOOP)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: hangs: ...
        """
        return Refuse(Stop.HANGS).give(data, kernel)

    @give.before("1.43.0")
    def _recast(self, data: Data, kernel: Dispatch) -> DType:
        """Cast to `Float64` and run the kernel again, as the older releases do.

        Args:
            data: The data.
            kernel: The dispatch that holds the arm.

        Returns:
            The type of the second call.

        Example:
            >>> EwmFallback()._recast(Data(DType("Int8")), WindowKernels.EWM_LOOP)
            Float64
        """
        return Recast(DType("Float64")).give(data, kernel)


class EncodedNested:
    """Accept a `List` and an `Array` in the kernel of `is_unique`, as polars does from 1.40.0.

    From that release, `is_unique` and `is_duplicated` encode each row of a
    list or an array, and hash the encoded rows. Older releases raise on
    these kinds.

    Example:
        >>> EncodedNested()._refused(DType("List", (DType("Int8"),)))
        False
    """

    @polars_releases
    def __call__(self, dtype: DType) -> bool:
        """Say whether the kernel accepts the type: a `List` or an `Array`.

        Args:
            dtype: The data type.

        Returns:
            Whether the type is a `List` or an `Array`.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> with PolarsRelease.pretend("1.40.0"):
            ...     EncodedNested()(DType("List", (DType("Int8"),)))
            True
        """
        return dtype.name in {"List", "Array"}

    @__call__.before("1.40.0")
    def _refused(self, dtype: DType) -> bool:
        """Say that the kernel of the older releases accepts no type here.

        Args:
            dtype: The data type.

        Returns:
            Always `False`.

        Example:
            >>> EncodedNested()._refused(DType("Array", (DType("Int8"), 2)))
            False
        """
        del dtype
        return False


class WindowKernels:
    """Hold the dispatch of each kernel of one input whose value at a row reads other rows.

    `INDICES` gives the index type for any type: `cum_count`, `arg_sort` and
    `rle_id` count or order the rows, and polars can compare a value with a
    value of its own type for every kind.

    `CUM_EXTREME` keeps a `Boolean`, a `Decimal` and every type whose
    physical type is a number, as `cum_min` and `cum_max` do. Every other
    type raises.

    `CUM_PROD` gives `Int64` for a `Boolean` and for an integer of 32 bits
    or fewer, keeps a wider integer and a float, and raises on every other
    type.

    `SAME` keeps every type, as `rechunk` and the fill of nulls by a
    strategy do. `REVERSE` keeps every type too, but a series of an `Array`
    whose inner type is not a number panics. The fills by `min`, `max`, `mean`, `zero` and `one`
    raise on some types only where a value is null, which is a fact of the
    data.

    `PEAKS` gives `Boolean` for a `Boolean` and for every type whose
    physical type is a number, as `peak_max` and `peak_min` do.

    `UNIQUE` gives `Boolean` for each kind that `is_unique` and
    `is_duplicated` hash: a number by its physical type, text, binary data,
    `Boolean`, `Null`, a struct, and from polars 1.40.0 a list and an array
    (`EncodedNested`).

    `FIRST_DISTINCT` gives `Boolean` for a series of one row, and otherwise
    refuses `Null`, an `Array` and a list of nested values (`FirstRow`).

    `EWM_LOOP` is `ewm_mean` and `ewm_sum`: they keep a float, and give any
    other type to their last arm (`EwmFallback`). `EWM_RECAST`
    is `ewm_std` and `ewm_var`: they cast another type to `Float64` first, so a
    struct, which stays a struct, makes the process crash. The cast pass casts
    every other input to `Float64` before these kernels run (`EwmCasts`).

    `MOMENT` is `rolling_skew` and `rolling_kurtosis`: they keep a float, give
    `Float64` for an integer, and raise on every other type.

    Example:
        >>> WindowKernels.CUM_PROD.output(Data(DType("UInt32")))
        Int64
        >>> WindowKernels.FIRST_DISTINCT.output(Data(DType("Array", (DType("Int8"), 2))))
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: is_first_distinct ...
    """

    BOOLEAN: ClassVar[DType] = DType("Boolean")
    FLOAT64: ClassVar[DType] = DType("Float64")
    INDICES: ClassVar[Dispatch] = Dispatch("arg_sort", (), Fixed(DType.idx()))
    CUM_EXTREME: ClassVar[Dispatch] = Dispatch(
        "cum_max",
        (
            Arm(Kinds("Boolean", "Decimal"), Same()),
            Arm(Physical(Is.primitive_numeric), Same()),
        ),
        Refuse(),
    )
    CUM_PROD: ClassVar[Dispatch] = Dispatch(
        "cum_prod",
        (
            Arm(
                Kinds("Boolean", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32"),
                Fixed(DType("Int64")),
            ),
            Arm(Is.primitive_numeric, Same()),
        ),
        Refuse(),
    )
    SAME: ClassVar[Dispatch] = Dispatch("reverse", (), Same())
    REVERSE: ClassVar[Dispatch] = Dispatch(
        "reverse",
        (),
        ScalarKept(Dispatch("reverse", (Arm(NonNumericArray(), Refuse(Stop.PANICS)),), Same())),
    )
    PEAKS: ClassVar[Dispatch] = Dispatch(
        "peak_max",
        (
            Arm(Kinds("Boolean"), Fixed(BOOLEAN)),
            Arm(Physical(Is.primitive_numeric), Fixed(BOOLEAN)),
        ),
        Refuse(),
    )
    UNIQUE: ClassVar[Dispatch] = Dispatch(
        "is_unique",
        (
            Arm(Physical(Kinds("Boolean", "String", "Binary", "Null", "Struct")), Fixed(BOOLEAN)),
            Arm(EncodedNested(), Fixed(BOOLEAN)),
            Arm(Physical(Is.primitive_numeric), Fixed(BOOLEAN)),
        ),
        Refuse(),
    )
    FIRST_DISTINCT: ClassVar[Dispatch] = Dispatch(
        "is_first_distinct",
        (),
        FirstRow(
            Dispatch(
                "is_first_distinct",
                (
                    Arm(Physical(Kinds("Boolean", "String", "Binary", "Struct")), Fixed(BOOLEAN)),
                    Arm(Physical(Is.primitive_numeric), Fixed(BOOLEAN)),
                    Arm(Physical(FlatList()), Fixed(BOOLEAN)),
                ),
                Refuse(),
            ),
        ),
    )
    EWM_LOOP: ClassVar[Dispatch] = Dispatch(
        "ewm_mean",
        (Arm(Is.float, Same()),),
        EwmFallback(),
    )
    EWM_RECAST: ClassVar[Dispatch] = Dispatch(
        "ewm_std",
        (Arm(Is.float, Same()),),
        Recast(FLOAT64, checked=True),
    )
    MOMENT: ClassVar[Dispatch] = Dispatch(
        "rolling_skew",
        (
            Arm(Is.float, Same()),
            Arm(Is.primitive_numeric, Fixed(FLOAT64)),
        ),
        Refuse(),
    )

    def rank(self, method: str | None) -> Dispatch:
        """Give the kernel of `rank` with a method: the type of the method, for every type.

        Args:
            method: The method, as polars serializes it, for example `Average`.

        Returns:
            The dispatch.

        Example:
            >>> WindowKernels().rank("Min").output(Data(DType("String")))
            UInt32
        """
        return Dispatch("rank", (), Fixed(WindowClosures().rank(method)))

    def fill(self, strategy: str | None) -> Dispatch:
        """Give the kernel of a fill of nulls by a strategy.

        A fill by `zero` or `one` puts a constant in each null (`NullFill`).
        Every other strategy keeps the type of the input.

        Args:
            strategy: The strategy, as polars serializes it, for example `Zero`.

        Returns:
            The dispatch.

        Example:
            >>> WindowKernels().fill("One").output(Data(DType("String")))
            String
        """
        if strategy in {"Zero", "One"}:
            return Dispatch("fill_null", (), NullFill(one=strategy == "One"))
        return self.SAME

    def interpolate(self, method: str | None) -> Dispatch:
        """Give the kernel of `interpolate` with a method.

        Both methods keep a categorical type, binary data, a struct and a
        list. The linear method keeps a float, a `Decimal` and a temporal
        type, gives `Float64` for an integer, and gives the physical type of
        every other type: a `Boolean` stays a `Boolean`, and an `Array` of
        dates becomes an `Array` of integers. The nearest method keeps every
        type whose physical type is a number, and panics on the others.

        Args:
            method: The method, as polars serializes it: `Linear` or `Nearest`.

        Returns:
            The dispatch.

        Example:
            >>> WindowKernels().interpolate("Linear").output(Data(DType("UInt8")))
            Float64
            >>> WindowKernels().interpolate("Nearest").output(Data(DType("String")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: interpolate does not accept String
        """
        kept = Arm(Kinds("Categorical", "Enum", "Binary", "Struct", "List"), Same())
        if method != "Linear":
            return Dispatch(
                "interpolate",
                (kept, Arm(Physical(Is.primitive_numeric), Same())),
                Refuse(Stop.PANICS),
            )
        arms = (
            kept,
            Arm(Is.temporal, Same()),
            Arm(Kinds("Decimal"), Same()),
            Arm(Is.float, Same()),
            Arm(Is.integer, Fixed(self.FLOAT64)),
        )
        return Dispatch("interpolate", arms, ToPhysical())

    def qcut(
        self,
        probs: Sequence[float],
        labels: Sequence[str] | None,
        *,
        include_breaks: bool,
        allow_duplicates: bool,
    ) -> QuantileCut:
        """Give the kernel of `qcut` with its options.

        Args:
            probs: The quantiles of the breaks.
            labels: The labels of the bins, or `None`.
            include_breaks: Whether the output holds the upper break of each value.
            allow_duplicates: Whether the call accepts breaks that are not unique.

        Returns:
            The kernel.

        Example:
            >>> kernels = WindowKernels()
            >>> qcut = kernels.qcut([0.5], None, include_breaks=True, allow_duplicates=False)
            >>> qcut.output([Data(DType("Date"))], (), ())
            Struct({'breakpoint': Float64, 'category': Categorical})
        """
        return QuantileCut(
            probs,
            labels,
            include_breaks=include_breaks,
            allow_duplicates=allow_duplicates,
        )


@dataclass(frozen=True)
class RollingValues:
    """Run a rolling kernel over the values that a literal input keeps after its cast to a float.

    A rolling kernel of a fixed window casts its values to a float where its
    weights need one, with a cast that turns a value that does not fit into
    null. The value of a literal input decides that: the text `"a"` becomes a
    null, and the kernel then meets data with nulls, which its weights refuse.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> weighted = RollingKernels().sum(2, 1, (1.0, 1.0))
        >>> weighted.output([Data(DType("String"))], (), [Operand.of_value("a")])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: weights not yet supported ...
    """

    FLOAT64: ClassVar[DType] = DType("Float64")

    dispatch: Dispatch

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values.
            lengths: The rows of each input, which the kernel does not read.
            computed: The values as the run computes them, or none.

        Returns:
            The type.

        Example:
            >>> RollingKernels().sum(2, 1, None).output([Data(DType("Int8"))], (), ())
            Int64
        """
        del lengths
        data = inputs[0]
        literal = computed[0] if computed and computed[0].literal else None
        if literal is not None and CastFold().nulled(literal, self.FLOAT64):
            data = replace(data, nulls=True)
        return self.dispatch.output(data)


class RollingKernels:
    """Build the kernel of each rolling function from the options of its node.

    A rolling function of a fixed window reads its window size, the rows
    that a window needs (`FixedWindow`), and for `min`, `max` and `sum` its
    weights. A function whose windows a key column orders, such as
    `rolling_mean_by`, reads the duration of its window, and checks the key
    in its kernel (`RollingBy`). Both forms dispatch on the type of the
    values in the same way, so one method builds the dispatch of each
    function for a check of a window:

    - `min` and `max` keep a `Boolean`, a temporal type and a number. With
      weights, they cast the values to a float first.
    - `sum` gives the index type for a `Boolean` and `Int64` for an integer
      of 16 bits or fewer, and keeps a wider number. With weights, it casts
      the values to a float first.
    - `mean`, `median` and `quantile` give `Datetime("us")` for a `Date`,
      keep another temporal type and `Float32` and `Float64`, and cast every
      other type to `Float64`. A struct stays a struct after the cast, and the
      kernel panics on it.
    - `var` and `std` keep `Float32` and `Float64`, and cast every other
      type to `Float64`, a temporal type too.
    - `rank` gives the type of its method for a `Boolean`, a temporal type
      and a number.

    Example:
        >>> RollingKernels().sum(2, 1, None).output([Data(DType("Boolean"))], (), ())
        UInt32
        >>> RollingKernels().mean(2, 1, None).output([Data(DType("Date"))], (), ())
        Datetime(time_unit='us', time_zone=None)
    """

    FLOATS: ClassVar[Kinds] = Kinds("Float32", "Float64")
    FLOAT64: ClassVar[DType] = DType("Float64")
    BOOLEAN: ClassVar[DType] = DType("Boolean")

    def extreme(
        self, window_size: int, min_periods: int, weights: tuple[float, ...] | None
    ) -> RollingValues:
        """Give the kernel of `rolling_min` and `rolling_max` with their options.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            weights: The weights of the rows of a window, or `None`.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().extreme(2, 1, (1.0, 2.0)).output([Data(DType("Int8"))], (), ())
            Float64
        """
        window = FixedWindow(window_size, min_periods, weights)
        return RollingValues(self.extreme_of(window, weighted=weights is not None))

    def sum(
        self, window_size: int, min_periods: int, weights: tuple[float, ...] | None
    ) -> RollingValues:
        """Give the kernel of `rolling_sum` with its options.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            weights: The weights of the rows of a window, or `None`.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().sum(2, 1, None).output([Data(DType("UInt16"))], (), ())
            Int64
        """
        window = FixedWindow(window_size, min_periods, weights)
        return RollingValues(self.sum_of(window, weighted=weights is not None))

    def mean(
        self,
        window_size: int,
        min_periods: int,
        weights: tuple[float, ...] | None,
    ) -> RollingValues:
        """Give the kernel of `rolling_mean` with its options.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            weights: The weights of the rows of a window, or `None`.

        Returns:
            The dispatch.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> RollingKernels().mean(2, 1, None).output([Data(struct)], (), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: rolling_mean does not accept ...
        """
        return RollingValues(self.mean_of(FixedWindow(window_size, min_periods, weights)))

    def quantile(
        self,
        window_size: int,
        min_periods: int,
        weights: tuple[float, ...] | None,
    ) -> RollingValues:
        """Give the kernel of `rolling_quantile` and `rolling_median` with their options.

        It dispatches as `rolling_mean` does. Weights that sum to zero leave
        the quantile undefined.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            weights: The weights of the rows of a window, or `None`.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().quantile(2, 1, (0.0, 0.0)).output([Data(DType("Int8"))], (), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: Weighted quantile is ...
        """
        return RollingValues(
            self.mean_of(FixedWindow(window_size, min_periods, weights, "quantile"))
        )

    def spread(
        self,
        window_size: int,
        min_periods: int,
        weights: tuple[float, ...] | None,
    ) -> RollingValues:
        """Give the kernel of `rolling_var` and `rolling_std` with their options.

        Weights that sum to zero leave the variance undefined.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            weights: The weights of the rows of a window, or `None`.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().spread(2, 1, None).output([Data(DType("Date"))], (), ())
            Float64
        """
        return RollingValues(
            self.spread_of(FixedWindow(window_size, min_periods, weights, "variance"))
        )

    def rank(self, window_size: int, min_periods: int, method: str | None) -> Dispatch:
        """Give the kernel of `rolling_rank` with its options.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.
            method: The method of ranking, as polars serializes it.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().rank(2, 1, "Average").output(Data(DType("Time")))
            Float64
        """
        return self.rank_of(FixedWindow(window_size, min_periods), method)

    def corr_cov(self, window_size: int, min_periods: int) -> CorrCov:
        """Give the kernel of `rolling_corr` and `rolling_cov` with their options.

        Args:
            window_size: The rows of a window.
            min_periods: The rows that a window needs to give a value.

        Returns:
            The kernel.

        Example:
            >>> RollingKernels().corr_cov(2, 3).output([Data(DType("Int8"))] * 2, (), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `min_periods` ...
        """
        return CorrCov(FixedWindow(window_size, min_periods))

    def extreme_of(self, window: WindowCheck, *, weighted: bool) -> Dispatch:
        """Give the dispatch of `min` and `max` for a check of a window.

        Args:
            window: The checks of the window.
            weighted: Whether the window has weights.

        Returns:
            The dispatch.

        Example:
            >>> extreme = RollingKernels().extreme_of(FixedWindow(2, 1), weighted=False)
            >>> extreme.output(Data(DType("Boolean")))
            Boolean
        """
        if weighted:
            return self._weighted("rolling_min", window)
        arms = (
            Arm(Kinds("Boolean"), Windowed(Fixed(self.BOOLEAN), window)),
            Arm(Is.temporal, Windowed(Same(), window)),
            Arm(Is.primitive_numeric, Windowed(Same(), window)),
        )
        return Dispatch("rolling_min", arms, Refuse())

    def sum_of(self, window: WindowCheck, *, weighted: bool) -> Dispatch:
        """Give the dispatch of `sum` for a check of a window.

        Args:
            window: The checks of the window.
            weighted: Whether the window has weights.

        Returns:
            The dispatch.

        Example:
            >>> sum_ = RollingKernels().sum_of(FixedWindow(2, 1), weighted=False)
            >>> sum_.output(Data(DType("Date")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: rolling_sum ...
        """
        if weighted:
            return self._weighted("rolling_sum", window)
        arms = (
            Arm(Kinds("Boolean"), Windowed(Fixed(DType.idx()), window)),
            Arm(Kinds(*Is.SMALL_INTEGERS), Windowed(Fixed(DType("Int64")), window)),
            Arm(Is.primitive_numeric, Windowed(Same(), window)),
        )
        return Dispatch("rolling_sum", arms, Refuse())

    def mean_of(self, window: WindowCheck) -> Dispatch:
        """Give the dispatch of `mean`, `median` and `quantile` for a check of a window.

        Args:
            window: The checks of the window.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().mean_of(FixedWindow(2, 1)).output(Data(DType("Float16")))
            Float64
        """
        arms = (
            Arm(Kinds("Date"), Windowed(Fixed(DType("Datetime", ("us", None))), window)),
            Arm(Is.temporal, Windowed(Same(), window)),
            Arm(self.FLOATS, Windowed(Same(), window)),
        )
        return Dispatch("rolling_mean", arms, Recast(self.FLOAT64, then=self._floats(window)))

    def spread_of(self, window: WindowCheck) -> Dispatch:
        """Give the dispatch of `var` and `std` for a check of a window.

        Args:
            window: The checks of the window.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().spread_of(FixedWindow(2, 1)).output(Data(DType("Float32")))
            Float32
        """
        arms = (Arm(self.FLOATS, Windowed(Same(), window)),)
        return Dispatch("rolling_var", arms, Recast(self.FLOAT64, then=self._floats(window)))

    def rank_of(self, window: WindowCheck, method: str | None) -> Dispatch:
        """Give the dispatch of `rank` for a check of a window.

        Args:
            window: The checks of the window.
            method: The method of ranking.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels().rank_of(FixedWindow(2, 1), "Min").output(Data(DType("Boolean")))
            UInt32
        """
        rank = Windowed(Fixed(WindowClosures().rank(method)), window)
        arms = (
            Arm(Kinds("Boolean"), rank),
            Arm(Is.temporal, rank),
            Arm(Is.primitive_numeric, rank),
        )
        return Dispatch("rolling_rank", arms, Refuse())

    def _weighted(self, name: str, window: WindowCheck) -> Dispatch:
        """Give the dispatch of `min`, `max` or `sum` with weights: a cast to a float first.

        The kernel casts the values to `Float64`, and keeps `Float32` and
        `Float64`. Then it needs a number, and a struct, which stays a
        struct, raises.

        Args:
            name: The name of the kernel.
            window: The checks of the window.

        Returns:
            The dispatch.

        Example:
            >>> weighted = RollingKernels()._weighted("rolling_max", FixedWindow(2, 1))
            >>> weighted.output(Data(DType("Date")))
            Float64
        """
        numbers = Dispatch(name, (Arm(Is.primitive_numeric, Windowed(Same(), window)),), Refuse())
        arms = (Arm(self.FLOATS, Windowed(Same(), window)),)
        return Dispatch(name, arms, Recast(self.FLOAT64, then=numbers))

    def _floats(self, window: WindowCheck) -> Dispatch:
        """Give the dispatch of the window of floats after a cast: it panics on any other type.

        Args:
            window: The checks of the window.

        Returns:
            The dispatch.

        Example:
            >>> RollingKernels()._floats(FixedWindow(2, 1)).output(Data(DType("Float64")))
            Float64
        """
        return Dispatch(
            "rolling_mean", (Arm(self.FLOATS, Windowed(Same(), window)),), Refuse(Stop.PANICS)
        )


class RollingBy:
    """Give the type of a rolling function whose windows a key column orders, as `rolling_mean_by`.

    The kernel reads the values and the key. `rolling_rank_by` first refuses
    a window that is not closed on the right. Then the kernel dispatches on
    the type of the values, as the function of a fixed window does
    (`RollingKernels`), and checks the key and the duration where it reaches
    the window of numbers (`DynamicWindow`).

    Example:
        >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
        >>> mean = RollingBy(RollingKernels().mean_of, day)
        >>> mean.output([Data(DType("Int8")), Data(DType("Date"))], (3, 3))
        Float64
    """

    RIGHT: ClassVar[frozenset[str]] = frozenset({"Right", "Both"})

    def __init__(
        self,
        dispatch: Callable[[WindowCheck], Dispatch],
        duration: WindowDuration,
        closed: str | None = "Right",
    ) -> None:
        """Keep the dispatch of the function, the duration of the window and its closed sides.

        Args:
            dispatch: The method of `RollingKernels` that gives the dispatch for a window.
            duration: The duration of the window.
            closed: The sides of a window that it holds, which `rolling_rank_by` checks.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> RollingBy(RollingKernels().spread_of, day)._closed
            'Right'
        """
        self._dispatch = dispatch
        self._duration = duration
        self._closed = closed

    def __repr__(self) -> str:
        """Give the name of the dispatch and the options.

        Returns:
            The text.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> RollingBy(RollingKernels().spread_of, day, "Left")  # doctest: +ELLIPSIS
            RollingBy(spread_of, WindowDuration(months=0, weeks=0, days=1, ...), 'Left')
        """
        name = getattr(self._dispatch, "__name__", repr(self._dispatch))
        return f"RollingBy({name}, {self._duration!r}, {self._closed!r})"

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the key.
            lengths: The rows of the values and of the key.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The window is not closed on the right.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> rank = RollingBy(lambda window: RollingKernels().rank_of(window, None), day, "Left")
            >>> rank.output([Data(DType("Int8")), Data(DType("Date"))], ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `rolling_rank_by` ...
        """
        if self._closed is not None and self._closed not in self.RIGHT:
            message = "`rolling_rank_by` window needs to be closed on the right side"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        values, key = inputs[0], inputs[1]
        window = DynamicWindow(self._duration, key, lengths)
        return self._dispatch(window).output(values)


class RollingByKernels:
    """Build the kernel of each rolling function whose windows a key column orders.

    Example:
        >>> day = {"months": 0, "weeks": 0, "days": 1, "nsecs": 0}
        >>> duration = WindowDuration.of({**day, "negative": False, "parsed_int": False})
        >>> RollingByKernels().sum(duration).output([Data(DType("Int8")), Data(DType("Date"))], ())
        Int64
    """

    def extreme(self, duration: WindowDuration) -> RollingBy:
        """Give the kernel of `rolling_min_by` and `rolling_max_by`.

        Args:
            duration: The duration of the window.

        Returns:
            The kernel.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> inputs = [Data(DType("Time")), Data(DType("Date"))]
            >>> RollingByKernels().extreme(day).output(inputs, ())
            Time
        """
        return RollingBy(
            lambda window: RollingKernels().extreme_of(window, weighted=False), duration
        )

    def sum(self, duration: WindowDuration) -> RollingBy:
        """Give the kernel of `rolling_sum_by`.

        Args:
            duration: The duration of the window.

        Returns:
            The kernel.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> inputs = [Data(DType("Boolean")), Data(DType("Date"))]
            >>> RollingByKernels().sum(day).output(inputs, ())
            UInt32
        """
        return RollingBy(lambda window: RollingKernels().sum_of(window, weighted=False), duration)

    def mean(self, duration: WindowDuration) -> RollingBy:
        """Give the kernel of `rolling_mean_by`, `rolling_median_by` and `rolling_quantile_by`.

        Args:
            duration: The duration of the window.

        Returns:
            The kernel.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> RollingByKernels().mean(day).output([Data(DType("Date")), Data(DType("Date"))], ())
            Datetime(time_unit='us', time_zone=None)
        """
        return RollingBy(RollingKernels().mean_of, duration)

    def spread(self, duration: WindowDuration) -> RollingBy:
        """Give the kernel of `rolling_var_by` and `rolling_std_by`.

        Args:
            duration: The duration of the window.

        Returns:
            The kernel.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> inputs = [Data(DType("Date")), Data(DType("Date"))]
            >>> RollingByKernels().spread(day).output(inputs, ())
            Float64
        """
        return RollingBy(RollingKernels().spread_of, duration)

    def rank(self, duration: WindowDuration, closed: str | None, method: str | None) -> RollingBy:
        """Give the kernel of `rolling_rank_by`.

        Args:
            duration: The duration of the window.
            closed: The sides of a window that it holds.
            method: The method of ranking.

        Returns:
            The kernel.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> rank = RollingByKernels().rank(day, "Both", "Dense")
            >>> rank.output([Data(DType("Int8")), Data(DType("Date"))], ())
            UInt32
        """
        return RollingBy(lambda window: RollingKernels().rank_of(window, method), duration, closed)


class EwmBy:
    """Give the type of `ewm_mean_by` and `ewm_sum_by`, whose weights a key column gives.

    The kernel refuses a negative half life, and a half life that has no
    constant length in the time zone of the key (`WindowDuration.constant`).
    Then it reads whether the key is sorted (`SortedCheck`), and needs a key
    with as many rows as the values. The key becomes `Int64` from a
    `Datetime`, a `Date` or an integer of 32 or 64 bits. The values keep a
    float, and an integer of 32 or 64 bits becomes `Float64`. Any other type
    of the values or of the key raises.

    Example:
        >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
        >>> EwmBy(day).output([Data(DType("Int32")), Data(DType("Date"))], (3, 3))
        Float64
    """

    KEYS: ClassVar[frozenset[str]] = frozenset({
        "Int64",
        "Datetime",
        "Date",
        "UInt64",
        "UInt32",
        "Int32",
    })
    INTEGERS: ClassVar[frozenset[str]] = frozenset({"UInt64", "UInt32", "Int64", "Int32"})

    def __init__(self, half_life: WindowDuration) -> None:
        """Keep the half life.

        Args:
            half_life: The duration after which a weight halves.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> EwmBy(day)._half_life.days
            1
        """
        self._half_life = half_life

    def __repr__(self) -> str:
        """Give the half life.

        Returns:
            The text.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> EwmBy(day)  # doctest: +ELLIPSIS
            EwmBy(WindowDuration(months=0, ...))
        """
        return f"EwmBy({self._half_life!r})"

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the key.
            lengths: The rows of the values and of the key.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A check fails.

        Example:
            >>> day = WindowDuration(0, 0, 1, 0, negative=False, parsed_int=False)
            >>> EwmBy(day).output([Data(DType("Float32")), Data(DType("Date"))], (3, 1))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        values, key = inputs[0], inputs[1]
        self._check_half_life(key.dtype)
        SortedCheck().check(key)
        if len(set(lengths)) > 1:
            message = "arguments for `ewm_mean_by` have different lengths"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        floats = Is.float(values.dtype)
        if key.dtype.name not in self.KEYS or not (floats or values.dtype.name in self.INTEGERS):
            message = (
                "expected series to be a float or an integer, and `by` to be temporal or an integer"
            )
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return values.dtype if floats else DType("Float64")

    def _check_half_life(self, key: DType) -> None:
        """Refuse a negative half life, and one with no constant length in the zone of the key.

        Args:
            key: The type of the key.

        Raises:
            PolarsRefusal: The half life does not fit.

        Example:
            >>> month = WindowDuration(1, 0, 0, 0, negative=False, parsed_int=False)
            >>> EwmBy(month)._check_half_life(DType("Date"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected `half_life` ...
        """
        if self._half_life.negative:
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, "half_life cannot be negative")
        zone = key.args[1] if key.name == "Datetime" else None
        if not self._half_life.constant(zone if isinstance(zone, str) else None):
            message = "expected `half_life` to be a constant duration"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class CountedKernel:
    """Check the count of `shift`, `diff` and `pct_change`: one value that casts to `Int64`.

    The kernel needs the count to have exactly one row. It then casts the
    count to `Int64`, and a cast with no kernel raises. A struct casts each
    field and stays a struct, so the kernel finds no `Int64` and raises a
    `SchemaError`. A count of one value of another type, such as a text,
    runs.

    Example:
        >>> CountedKernel().check(Data(DType("String")), (5, 1))
        >>> CountedKernel().check(Data(DType("Int8")), (5, 5))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: n must be a single value.
    """

    def check(self, count: Data, lengths: Sequence[int]) -> None:
        """Refuse a count of another length than one row, or of a type with no cast to `Int64`.

        Args:
            count: The data of the count.
            lengths: The rows of the values and of the count.

        Raises:
            PolarsRefusal: The count does not fit.

        Example:
            >>> CountedKernel().check(Data(DType("Binary")), (5, 1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if (  # pragma: no cover -- the conversion checks it first
            len(lengths) > 1 and lengths[1] != 1
        ):
            raise PolarsRefusal(ErrorClass.COMPUTE, "n must be a single value.")
        cast = RunCast().of(count, DType("Int64"))
        if cast.dtype.name != "Int64":
            message = f"invalid series dtype: expected `Int64`, got `{cast.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


class Shift:
    """Give the type of `shift`: the type of the values, after the check of the count.

    Polars moves each value by the count, with nulls at the edge. A count
    that is null gives a column of nulls of the type of the values.

    Example:
        >>> Shift().output([Data(DType("Date")), Data(DType("Int32"))], (5, 1))
        Date
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the count.
            lengths: The rows of the values and of the count.

        Returns:
            The type of the values.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> Shift().output([Data(struct), Data(DType("Null"))], ())
            Struct({'a': Int8})
        """
        CountedKernel().check(inputs[1], lengths)
        return inputs[0].dtype


@dataclass(frozen=True)
class ShiftFilled:
    """Give the type of `shift` with a fill value: the type of the values, after the checks.

    The kernel casts the count to `Int64`, as `shift` does (`CountedKernel`),
    and a count that is null gives a column of nulls. It then reads the fill
    by the type of the values. `Boolean`, text and a list need a fill of the
    same kind, or a null, and raise otherwise. A struct, a category and an
    object zip the fill in, which needs the type of the values
    (`SameTypeZip`). A number, a `Decimal` and a temporal type read the fill
    as a number, which runs. Every other type, such as `Binary` or an array,
    raises.

    Example:
        >>> inputs = [Data(DType("Date")), Data(DType("Int32")), Data(DType("Date"))]
        >>> ShiftFilled().output(inputs, (5, 1, 1))
        Date
    """

    SAME_KIND: ClassVar[frozenset[str]] = frozenset({"Boolean", "String", "List"})
    ZIPPED: ClassVar[frozenset[str]] = frozenset({"Struct", "Categorical", "Enum", "Object"})
    LOGICAL: ClassVar[frozenset[str]] = frozenset({*Is.TEMPORAL, "Decimal"})

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the count does not cast (`CountedKernel`).

        Args:
            inputs: The data of the values, of the count and of the fill.
            lengths: The rows of each input, which the conversion checks.

        Returns:
            The type of the values.

        Raises:
            PolarsRefusal: The fill does not fit the values, or the values
                have a type that the kernel does not shift.

        Example:
            >>> texts = Data(DType("String"))
            >>> ShiftFilled().output([texts, Data(DType("Int8")), Data(DType("Int8"))], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: fill value ...
            >>> binary = Data(DType("Binary"))
            >>> ShiftFilled().output([binary, Data(DType("Int8")), binary], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        del lengths
        values, count, fill = inputs[0], inputs[1], inputs[2]
        CountedKernel().check(count, ())
        if count.nulls:
            return values.dtype
        kind = values.dtype.name
        if kind in self.SAME_KIND:
            if fill.dtype.name not in {kind, "Null"}:
                message = f"fill value '{fill.dtype!r}' is not supported"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
        elif kind in self.ZIPPED:
            SameTypeZip().check(values.dtype, fill.dtype)
        elif not (Is.primitive_numeric(values.dtype) or kind in self.LOGICAL):
            message = f"`shift_and_fill` operation not supported for dtype `{values.dtype!r}`"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return values.dtype


@dataclass(frozen=True)
class Difference:
    """Give the type of `diff`: the difference of the values and the values moved by the count.

    The kernel checks the count (`CountedKernel`). Its cast of the count to
    `Int64` is strict, so the value of a literal count that does not fit,
    such as the text `a`, raises at every run. The kernel refuses a count
    that is null. It casts an unsigned integer to the signed integer of
    twice its width, `Int64` at most. Then it subtracts two series of that
    type (`SeriesArithmetic`): a `Date` gives a `Duration("us")`, and a
    `Boolean` or a text raises.

    `drop` says that the kernel drops the rows that the count moves out,
    instead of a null in each. A count longer than a series of two rows or
    more then leaves two parts of other lengths, and their difference raises.

    Example:
        >>> count = Operand.of_value(1)
        >>> Difference().output([Data(DType("UInt16")), Data(DType("Int32"))], (5, 1), (count,))
        Int32
        >>> Difference().output([Data(DType("Int8")), Data(DType("Null"))], (5, 1), ())
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: 'n' can not be None for diff
    """

    SIGNED: ClassVar[dict[str, str]] = {
        "UInt8": "Int16",
        "UInt16": "Int32",
        "UInt32": "Int64",
        "UInt64": "Int64",
    }

    drop: bool = False

    @classmethod
    def of(cls, behavior: str | None) -> Difference:
        """Give the kernel for the null behavior of the node, `Ignore` or `Drop`.

        Args:
            behavior: The null behavior, as polars serializes it.

        Returns:
            The kernel.

        Example:
            >>> Difference.of("Drop")
            Difference(drop=True)
        """
        return cls(drop=behavior == "Drop")

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the count.
            lengths: The rows of the values and of the count.
            computed: The values and the count as the run computes them, or
                none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The count is null.

        Example:
            >>> count = Operand.of_value("a")
            >>> Difference().output([Data(DType("Date")), Data(DType("String"))], (5, 1), (count,))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        values, count = inputs[0], inputs[1]
        CountedKernel().check(count, lengths)
        literal = computed[-1] if computed else None
        if literal is not None and literal.literal:
            SeriesCast(count.dtype, DType("Int64"), strict=True).output([literal.value])
        if count.nulls:
            raise PolarsRefusal(ErrorClass.COMPUTE, "'n' can not be None for diff")
        if literal is not None and literal.literal:
            self._dropped(literal.value, lengths)
        signed = self.SIGNED.get(values.dtype.name)
        if signed is not None:
            values = RunCast().of(values, DType(signed))
        return SeriesArithmetic(NumericOp.SUB).output(values, values)

    def _dropped(self, value: object, lengths: Sequence[int]) -> None:
        """Refuse a count that drops more rows than a series of two rows or more holds.

        Args:
            value: The value of the literal count.
            lengths: The rows of the values and of the count, or none where
                they are not known.

        Raises:
            PolarsRefusal: The parts of the series have other lengths.

        Example:
            >>> Difference(drop=True)._dropped(300, (3, 1))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot do arithmetic ...
        """
        if not self.drop or not lengths or not isinstance(value, int) or lengths[0] < 2:
            return
        if abs(value) > lengths[0]:
            message = "cannot do arithmetic operation on series of different length"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)


class PercentChange:
    """Give the type of `pct_change`: the change of each value from an earlier value.

    The kernel checks that the count has one row. It keeps a float, and casts
    any other type to `Float64` and calls itself again. A struct stays a
    struct after the cast. From polars 1.44.0 the kernel then raises, and in
    older releases it calls itself without end, so the process crashes
    (`Recast`). Then it casts the count to `Int64`, and divides two floats.

    Example:
        >>> PercentChange().output([Data(DType("Int8")), Data(DType("Int32"))], (5, 1))
        Float64
    """

    FLOATS: ClassVar[Dispatch] = Dispatch(
        "pct_change",
        (Arm(Is.float, Same()),),
        Recast(DType("Float64"), checked=True),
    )

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the count.
            lengths: The rows of the values and of the count.

        Returns:
            The float type.

        Raises:
            PolarsRefusal: The count has another length than one row.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("1.44.0"):
            ...     PercentChange().output([struct, Data(DType("Int32"))], (5, 1))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `pct_change` ...
        """
        if len(lengths) > 1 and lengths[1] != 1:
            raise PolarsRefusal(ErrorClass.COMPUTE, "n must be a single value.")
        floats = self.FLOATS.output(inputs[0])
        CountedKernel().check(inputs[1], lengths)
        return floats


class InterpolateBy:
    """Give the type of `interpolate_by`: the values interpolated along a key column.

    The kernel reads whether the key is sorted (`SortedCheck`), and needs a
    key with as many rows as the values. The key must be a float, an integer
    of 32 or 64 bits, a `Date` or a `Datetime`. The values keep `Float32` and
    `Float64`, and an integer of 32 or 64 bits becomes `Float64`. Any other
    type of the values or of the key raises.

    Example:
        >>> InterpolateBy().output([Data(DType("UInt32")), Data(DType("Date"))], (3, 3))
        Float64
    """

    KEYS: ClassVar[frozenset[str]] = frozenset({
        "Float64",
        "Float32",
        "Int64",
        "Int32",
        "UInt64",
        "UInt32",
        "Date",
        "Datetime",
    })
    FLOATS: ClassVar[frozenset[str]] = frozenset({"Float64", "Float32"})
    INTEGERS: ClassVar[frozenset[str]] = frozenset({"UInt64", "UInt32", "Int64", "Int32"})

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the key.
            lengths: The rows of the values and of the key.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A check fails.

        Example:
            >>> InterpolateBy().output([Data(DType("Float16")), Data(DType("Date"))], (3, 3))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected series ...
        """
        values, key = inputs[0].dtype, inputs[1]
        SortedCheck().check(key)
        if len(set(lengths)) > 1:
            message = "`by` column must be the same length as Series"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        known = values.name in self.FLOATS or values.name in self.INTEGERS
        if key.dtype.name not in self.KEYS or not known:
            message = "expected series to be Float64, Float32, Int64, Int32, UInt64, UInt32"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return values if values.name in self.FLOATS else DType("Float64")


class CorrCov:
    """Give the type of `rolling_corr` and `rolling_cov`: the float of the supertype of the inputs.

    The kernel joins the two types (`Supertype`), and takes the join where
    it is a float and `Float64` otherwise. It casts both inputs to that type,
    and a cast with no kernel raises. It multiplies them, which needs one
    length or a side of one row, and runs a rolling mean of the product,
    which checks the window (`FixedWindow`). A struct stays a struct after
    the cast, and the rolling mean panics on it. Where an input holds a null,
    the kernel puts the mask of the rows where both inputs are valid on each
    input. That mask has the longer length, so it panics on an input of one
    row beside a longer input. The rules know such a null where every value
    is null, or where the code decides it (`Data.holds_null`).

    Example:
        >>> pair = [Data(DType("Int8")), Data(DType("Float32"))]
        >>> CorrCov(FixedWindow(2, 2)).output(pair, (5, 5), ())
        Float32
        >>> nulls = [Data(DType("Null")), Data(DType("Int8"))]
        >>> CorrCov(FixedWindow(2, 2)).output(nulls, (1, 5), ())
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: the mask of the valid rows ...
    """

    def __init__(self, window: FixedWindow) -> None:
        """Keep the window of the rolling mean.

        Args:
            window: The window.

        Example:
            >>> CorrCov(FixedWindow(2, 2))._window.window_size
            2
        """
        self._window = window

    def __repr__(self) -> str:
        """Give the window.

        Returns:
            The text.

        Example:
            >>> CorrCov(FixedWindow(2, 2))
            CorrCov(FixedWindow(window_size=2, min_periods=2, weights=None, measure=None))
        """
        return f"CorrCov({self._window!r})"

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the two inputs.
            lengths: The rows of the two inputs.
            computed: The two inputs as the run computes them, or none where
                they are not known.

        Returns:
            The float type.

        Raises:
            PolarsRefusal: The inputs have other lengths and neither has one row,
                or an input that holds a null meets an input of another length.

        Example:
            >>> shifted = Data(DType("Float64"), some_null=True)
            >>> CorrCov(FixedWindow(2, 2)).output([shifted, Data(DType("Float64"))], (5, 1), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: the mask of the valid rows ...
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> CorrCov(FixedWindow(2, 2)).output([struct, struct], (5, 5), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        joined = Supertype().require(inputs[0].dtype, inputs[1].dtype)
        target = joined if Is.float(joined) else DType("Float64")
        left = self._cast(inputs[0], target, computed, 0)
        right = self._cast(inputs[1], target, computed, 1)
        if len({length for length in lengths if length != 1}) > 1:
            message = "cannot do arithmetic operation on series of different lengths"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        product = Data(SeriesArithmetic(NumericOp.MUL).output(left, right))
        output = RollingKernels().mean_of(self._window).output(product)
        if (left.holds_null() or right.holds_null()) and len(set(lengths)) > 1:
            message = "the mask of the valid rows has another length than an input of one row"
            raise PolarsRefusal(Stop.PANICS, message)
        return output

    def _cast(
        self,
        data: Data,
        target: DType,
        computed: Sequence[Operand],
        index: int,
    ) -> Data:
        """Cast one input to the float type, as a cast that is not strict.

        A cast with no kernel raises. The value of a literal that does not
        fit, such as the text `a`, becomes null, at every run.

        Args:
            data: The data of the input.
            target: The float type.
            computed: The inputs as the run computes them, or none.
            index: The place of the input.

        Returns:
            The data after the cast.

        Example:
            >>> text = Operand.of_value("a")
            >>> CorrCov(FixedWindow(2, 2))._cast(Data(DType("String")), DType("Float64"), [text], 0)
            Data(dtype=Float64, nulls=True, one_row=False)
        """
        cast = RunCast().of(data, target)
        operand = computed[index] if index < len(computed) else None
        if operand is not None and operand.literal and not CastFold().keeps(operand, target):
            return replace(cast, nulls=True)
        return cast
