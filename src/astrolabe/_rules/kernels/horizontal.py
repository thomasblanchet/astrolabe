"""The kernels of the functions that read several columns row by row, such as `mean_horizontal`.

`coalesce` zips its first input with each later input. `sum_horizontal`
adds its inputs with the arithmetic of two series, and skips each input of
the type `Null`. `mean_horizontal` needs numbers,
`Boolean` or `Null`, and divides that sum by the count of the values that
are not null, as a float. `max_horizontal` and `min_horizontal` compare their
inputs pair by pair. `clip` needs a number and casts its bounds to the type
of its input. `is_between` compares its input with each bound, and
`is_close` compares two numbers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.arithmetic import Coercion, SeriesArithmetic
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.dispatch import Data, Dispatch, RunCast
from astrolabe._rules.kernels.windows import WindowKernels
from astrolabe._rules.kernels.zipping import SameTypeZip
from astrolabe._rules.operators import NumericOp
from astrolabe._rules.supertype import Supertype

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class Coalesced:
    """Give the type of `coalesce`: the first input, with each null taken from the later inputs.

    Polars zips the first input with each later input in turn, and reads the
    later input as the type of the first (`SameTypeZip`). The cast pass gives
    the inputs one type, except where a cast keeps its input apart: a struct
    stays a struct, and an integer literal with no data type keeps the type
    that it takes alone.

    Example:
        >>> Coalesced().output([Data(DType("Int32")), Data(DType("Int64"))], ())
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: cannot get ref Int32 from Int64
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the run checks.

        Returns:
            The type of the first input.

        Example:
            >>> Coalesced().output([Data(DType("Null")), Data(DType("Int64"))], ())
            Null
        """
        del lengths
        first = inputs[0].dtype
        for data in inputs[1:]:
            SameTypeZip().check(first, data.dtype)
        return first


@dataclass(frozen=True)
class HorizontalSum:
    """Give the type of `sum_horizontal`: the sum of the inputs that are not of the type `Null`.

    One input of the type `Boolean` counts as the index type. Several inputs
    add with the arithmetic of two series, in the order of a parallel
    reduction (`_reduced`). That addition
    takes a fast path where the left side is a plain number and the right
    side has a physical type of numbers: it casts both sides to their
    supertype, and panics where that supertype is not a plain number, such as
    a `Decimal`, or does not exist. `sum_horizontal` casts its inputs to one
    type before, and `mean_horizontal` does not.

    With `ignore_nulls`, the default, each input with nulls first fills them
    with zero. A fill by zero refuses data of only nulls of a nested type,
    such as a `Null` input that the cast pass turned into a list.

    Example:
        >>> HorizontalSum().output([Data(DType("Int8")), Data(DType("Int8"))], ())
        Int8
    """

    FILL: ClassVar[Dispatch] = WindowKernels().fill("Zero")

    ignore_nulls: bool = True

    LOGICAL: ClassVar[frozenset[str]] = frozenset({
        "Date",
        "Datetime",
        "Duration",
        "Time",
        "Decimal",
        "Categorical",
        "Enum",
    })

    # From this number of inputs, the threads of polars decide the order of the additions.
    THREADED: ClassVar[int] = 5
    ORDER: ClassVar[str] = "the number of threads decides the order of the additions"

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where an addition refuses its inputs.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the run checks.

        Returns:
            The type.

        Example:
            >>> HorizontalSum().output([Data(DType("Boolean")), Data(DType("Null"))], ())
            UInt32
            >>> HorizontalSum().output([Data(DType("Null"))], ())
            Null
        """
        del lengths
        values = [data for data in inputs if data.dtype.name != "Null"]
        if not values:
            return inputs[0].dtype
        if len(values) == 1:
            single = values[0]
            if single.dtype.name == "Boolean":
                single = single.with_dtype(DType.idx())
            return self._filled(single).dtype
        return self._reduced(values).dtype

    def _reduced(self, values: Sequence[Data]) -> Data:
        """Give the sum of several inputs, in the order that polars adds them.

        Polars reduces three inputs or more in parallel, and the pool splits
        the inputs in halves before it adds them: `a + (b + c)` for three, and
        `(a + b) + (c + d)` for four. The order changes the type, because each
        addition takes the supertype of its two sides. From five inputs, the
        number of threads can change the order (`_any_order`).

        Args:
            values: The data of the inputs, at least one.

        Returns:
            The data of the sum.

        Example:
            >>> inputs = [Data(DType("Float16")), Data(DType("Int8")), Data(DType("UInt8"))]
            >>> HorizontalSum()._reduced(inputs).dtype
            Float32
        """
        if len(values) >= self.THREADED:
            return self._any_order(values)
        return self._halved(values)

    def _halved(self, values: Sequence[Data]) -> Data:
        """Give the sum of several inputs that the pool splits in halves down to one input.

        Args:
            values: The data of the inputs, at least one.

        Returns:
            The data of the sum.

        Example:
            >>> inputs = [Data(DType("Int8")), Data(DType("Int16"))]
            >>> HorizontalSum()._halved(inputs).dtype
            Int16
        """
        if len(values) == 1:
            return values[0]
        middle = len(values) // 2
        left = self._halved(values[:middle])
        right = self._halved(values[middle:])
        return Data(self._add(self._filled(left), self._filled(right)))

    def _any_order(self, values: Sequence[Data]) -> Data:
        """Give the sum of five inputs or more, where every order of the additions agrees.

        The number of threads of polars decides how the pool splits the
        inputs, so each order of the additions that keeps the inputs in
        their places can run. Where every order gives one type, that is the
        sum. Where every order refuses, polars refuses. Where the orders
        differ, the machine decides the answer.

        Args:
            values: The data of the inputs, at least five.

        Returns:
            The data of the sum.

        Raises:
            UndecidedError: The orders give different answers.

        Example:
            >>> HorizontalSum()._any_order([Data(DType("Int64"))] * 5).dtype
            Int64
            >>> wide = [Data(DType("Decimal", (10, 2))), Data(DType("Float64"))]
            >>> HorizontalSum()._any_order([Data(DType("Int64"))] * 3 + wide)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: process: the number of threads decides ...
            >>> HorizontalSum()._any_order([Data(DType("Date"))] * 5)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ...
        """
        answers = self._interval_sums(values)[0, len(values)]
        if len(answers) != 1:
            raise UndecidedError(Undecided.PROCESS, self.ORDER)
        answer = next(iter(answers))
        return self._halved(values) if answer is None else answer

    def _interval_sums(self, values: Sequence[Data]) -> dict[tuple[int, int], set[Data | None]]:
        """Give each sum that the inputs of each interval can have, `None` for a refusal.

        Args:
            values: The data of the inputs.

        Returns:
            The sums of the inputs from `i` to before `j`, by `(i, j)`.

        Example:
            >>> two = [Data(DType("Int8")), Data(DType("Int16"))]
            >>> HorizontalSum()._interval_sums(two)[0, 2]
            {Data(dtype=Int16, nulls=False, one_row=False)}
        """
        count = len(values)
        sums: dict[tuple[int, int], set[Data | None]] = {}
        for place, value in enumerate(values):
            sums[place, place + 1] = {value}
        for width in range(2, count + 1):
            for start in range(count - width + 1):
                sums[start, start + width] = self._joined(sums, start, start + width)
        return sums

    def _joined(
        self,
        sums: Mapping[tuple[int, int], set[Data | None]],
        start: int,
        end: int,
    ) -> set[Data | None]:
        """Give each sum of the inputs of one interval, over each last addition.

        Args:
            sums: The sums of each shorter interval.
            start: The first input of the interval.
            end: The input after the last one.

        Returns:
            The sums, `None` for a refusal.

        Example:
            >>> sums = {(0, 1): {Data(DType("Date"))}, (1, 2): {Data(DType("Date"))}}
            >>> HorizontalSum()._joined(sums, 0, 2)
            {None}
        """
        joined: set[Data | None] = set()
        for cut in range(start + 1, end):
            joined.update(
                itertools.starmap(self._pair, itertools.product(sums[start, cut], sums[cut, end]))
            )
        return joined

    def _pair(self, left: Data | None, right: Data | None) -> Data | None:
        """Give the sum of two sums, or `None` where either one or the addition refuses.

        Args:
            left: The left sum.
            right: The right sum.

        Returns:
            The sum, or `None`.

        Example:
            >>> HorizontalSum()._pair(Data(DType("Int8")), None) is None
            True
        """
        if left is None or right is None:
            return None
        try:
            return Data(self._add(self._filled(left), self._filled(right)))
        except PolarsRefusal:
            return None

    def _filled(self, data: Data) -> Data:
        """Give the data of an input after the fill of its nulls by zero, where the sum fills them.

        It raises `PolarsRefusal` where the data holds only nulls and its type
        takes no zero (`NullFill`).

        Args:
            data: The data of the input.

        Returns:
            The data, with no null where the sum fills them.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)), nulls=True)
            >>> HorizontalSum(ignore_nulls=False)._filled(lists).nulls
            True
            >>> HorizontalSum()._filled(lists)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: fill_null does not ...
        """
        if not self.ignore_nulls:
            return data
        self.FILL.output(data)
        return replace(data, nulls=False)

    def _add(self, left: Data, right: Data) -> DType:
        """Give the type of the sum of two series that the fold owns.

        Args:
            left: The sum so far.
            right: The next input.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The fast path meets a supertype that is not a plain
                number, or no supertype, where polars panics.

        Example:
            >>> decimal = DType("Decimal", (10, 2))
            >>> HorizontalSum()._add(Data(DType("Int64")), Data(decimal))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
            >>> HorizontalSum()._add(Data(decimal), Data(DType("Int64")))
            Decimal(precision=38, scale=2)
        """
        plain = left.dtype.name not in self.LOGICAL and Is.primitive_numeric(left.dtype.physical())
        if not plain or not Is.primitive_numeric(right.dtype.physical()):
            return SeriesArithmetic(NumericOp.ADD).output(left, right)
        joined = Supertype().of(left.dtype, right.dtype)
        if joined is None or not Is.primitive_numeric(joined):
            message = f"the fast addition of {left.dtype!r} and {right.dtype!r} is unreachable"
            raise PolarsRefusal(Stop.PANICS, message)
        return joined


@dataclass(frozen=True)
class HorizontalMean:
    """Give the type of `mean_horizontal`: the sum of the inputs over their count, as a float.

    Every input must be a number, a `Decimal`, a `Boolean` or `Null`. One
    input becomes `Float64` unless it is a `Float16`, a `Float32` or a
    `Decimal`. Several inputs divide their sum by a count of the type of a
    float sum, or `Float64`.

    Example:
        >>> HorizontalMean().output([Data(DType("Int8")), Data(DType("Int16"))], ())
        Float64
    """

    NARROW: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32"})
    FLOAT64: ClassVar[DType] = DType("Float64")

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the run checks.

        Returns:
            The type.

        Raises:
            PolarsRefusal: An input is not a number, or a step of the sum refuses.

        Example:
            >>> HorizontalMean().output([Data(DType("String")), Data(DType("Int8"))], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> HorizontalMean().output([Data(DType("Int8"))], ())
            Float64
            >>> HorizontalMean().output([Data(DType("Float32"))], ())
            Float32
        """
        for data in inputs:
            dtype = data.dtype
            numeric = Is.primitive_numeric(dtype) or dtype.name in {"Decimal", "Boolean", "Null"}
            if not numeric:
                message = f"'horizontal_mean' expects numeric expressions, found {dtype!r}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if len(inputs) == 1:
            dtype = inputs[0].dtype
            keeps = dtype.name in self.NARROW or dtype.name == "Decimal"
            return dtype if keeps else RunCast().of(inputs[0], self.FLOAT64).dtype
        total = HorizontalSum().output(inputs, lengths)
        count = DType(total.name) if total.name in self.NARROW else self.FLOAT64
        return SeriesArithmetic(NumericOp.DIV).output(Data(total), Data(count))


@dataclass(frozen=True)
class Clipped:
    """Give the type of `clip`: the type of its input, whose physical type must be a number.

    The bounds must have the length of the input or one row, and cast to the
    type of the input with a strict cast, so a literal bound that does not
    fit raises at every run. A struct bound stays a struct in that cast, and
    polars panics when it reads it as the type of the input, after it casts
    every bound.

    Example:
        >>> Clipped().output([Data(DType("Date")), Data(DType("Date"))], (3, 1), ())
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
            inputs: The data of the input and of each bound.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The input is not a number, a length does not fit, or a
                bound does not cast.

        Example:
            >>> Clipped().output([Data(DType("String")), Data(DType("String"))], (), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        dtype = inputs[0].dtype
        if not Is.primitive_numeric(dtype.physical()):
            message = "`clip` only supports physical numeric types"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        longer = [length for length in lengths if length != 1]
        if len(set(longer)) > 1:
            raise PolarsRefusal(ErrorClass.SHAPE, "clip: the lengths of the inputs differ")
        literals = [*computed, *([None] * len(inputs))][1 : len(inputs)]
        bounds = zip(inputs[1:], literals, strict=True)
        casts = [self._cast(bound, literal, dtype) for bound, literal in bounds]
        for cast in casts:
            if cast.dtype.name != dtype.name:
                message = f"`clip` reads a bound of {cast.dtype!r} as {dtype!r}"
                raise PolarsRefusal(Stop.PANICS, message)
        return dtype

    def _cast(self, bound: Data, literal: Operand | None, dtype: DType) -> Data:
        """Cast one bound to the type of the input, with the value of a literal bound.

        It raises `PolarsRefusal` where polars has no cast, or where a strict
        cast loses the value of a literal (`RunCast`, `SeriesCast`).

        Args:
            bound: The data of the bound.
            literal: The bound as the run computes it, or `None`.
            dtype: The type of the input.

        Returns:
            The data after the cast.

        Example:
            >>> Clipped()._cast(Data(DType("Int64")), None, DType("Date")).dtype
            Date
        """
        cast = RunCast().of(bound, dtype)
        if literal is not None and literal.literal:
            SeriesCast(literal.dtype.materialized(), dtype, strict=True).output([literal.value])
        return cast


@dataclass(frozen=True)
class Between:
    """Give the type of `is_between`: `Boolean`, from a comparison of the input with each bound.

    The kernel compares the input with the lower bound, with `>=` or `>`, and
    with the upper bound, with `<=` or `<`, as `closed` says. Each comparison
    refuses the types that it cannot compare, and then two inputs of more
    than one row that differ in length. The two masks then join with `&`,
    which spreads a mask of one row, and panics where the two masks have
    more than one row and differ in length.

    Example:
        >>> inputs = [Data(DType("Int8")), Data(DType("String")), Data(DType("Int8"))]
        >>> Between("Both").output(inputs, ())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ...
    """

    OPERATORS: ClassVar[dict[str, tuple[CompareOp, CompareOp]]] = {
        "Both": (CompareOp.GT_EQ, CompareOp.LT_EQ),
        "Left": (CompareOp.GT_EQ, CompareOp.LT),
        "Right": (CompareOp.GT, CompareOp.LT_EQ),
        "None": (CompareOp.GT, CompareOp.LT),
    }
    BOOLEAN: ClassVar[DType] = DType("Boolean")

    closed: str

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a comparison refuses its inputs or
        their lengths (`Comparison`).

        Args:
            inputs: The data of the input and of the two bounds.
            lengths: The rows of each input, or none where they are not known.

        Returns:
            `Boolean`.

        Raises:
            PolarsRefusal: The two masks have more than one row and differ in
                length, where polars panics.

        Example:
            >>> Between("None").output([Data(DType("Int8"))] * 3, (5, 1, 2))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
            >>> Between("Both").output([Data(DType("Int8"))] * 3, (1, 2, 5))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        masks = set()
        for place, op in enumerate(self.OPERATORS[self.closed], start=1):
            pair = (lengths[0], lengths[place]) if lengths else ()
            Comparison(op).output(inputs[0], inputs[place], pair)
            masks.add(max(pair, default=1))
        if len(masks - {1}) > 1:
            raise PolarsRefusal(Stop.PANICS, "`is_between` joins two masks of other lengths")
        return self.BOOLEAN


@dataclass(frozen=True)
class CloseTest:
    """Give the type of `is_close`: `Boolean`, over two numbers and two tolerances of zero or more.

    Example:
        >>> CloseTest(0.0, 1e-09).output([Data(DType("Int8")), Data(DType("Float32"))], ())
        Boolean
    """

    BOOLEAN: ClassVar[DType] = DType("Boolean")

    abs_tol: float
    rel_tol: float

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the two sides.
            lengths: The rows of each input, or none where they are not known.

        Returns:
            `Boolean`.

        Raises:
            PolarsRefusal: A tolerance is negative, a side is not a number, or
                the two sides have other lengths and neither has one row, where
                polars panics.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> CloseTest(0.0, 0.0).output([Data(struct), Data(DType("Int8"))], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> CloseTest(-1.0, 0.0).output([Data(DType("Int8")), Data(DType("Int8"))], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: `abs_tol` must be ...
        """
        for name, tolerance in (("abs_tol", self.abs_tol), ("rel_tol", self.rel_tol)):
            if tolerance < 0:
                message = f"`{name}` must be non-negative but got {tolerance}"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
        for data in inputs:
            if not Is.primitive_numeric(data.dtype) and data.dtype.name != "Decimal":
                message = f"`is_close` supports numeric types only, got {data.dtype!r}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if len({length for length in lengths if length != 1}) > 1:
            raise PolarsRefusal(Stop.PANICS, "`is_close` zips two series of other lengths")
        return self.BOOLEAN


@dataclass(frozen=True)
class HorizontalExtreme:
    """Give the type of `max_horizontal` and `min_horizontal`: the inputs folded pair by pair.

    Each step compares the two sides with `>` or `<`, and picks each row from
    one side, as a `when` does: the two sides take one type. A type that the
    comparison refuses, such as a struct, refuses the step.

    Example:
        >>> extreme = HorizontalExtreme(CompareOp.GT)
        >>> extreme.output([Data(DType("Int8")), Data(DType("UInt8"))], ())
        Int16
    """

    op: CompareOp

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a comparison or a coercion refuses a step.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the run checks.

        Returns:
            The type.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> HorizontalExtreme(CompareOp.GT).output([Data(struct), Data(struct)], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ...
        """
        del lengths
        total = inputs[0]
        for data in inputs[1:]:
            Comparison(self.op).output(total, data)
            total = Data(Coercion().of(total, data).left.dtype)
        return total.dtype
