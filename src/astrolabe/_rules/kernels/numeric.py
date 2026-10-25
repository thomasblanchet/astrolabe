"""The shared numeric kernels of one input, such as `abs`, `round`, `cut`, `cum_sum` and `sum`.

Each kernel dispatches on the data type of its input, and its output type can
differ from the planner type. `abs`, `neg`, `sign` and the roundings keep the
type of a number and refuse the rest. `cut` casts its input to `Float64` and
gives the labels of its bins. `cum_sum` widens a small integer to `Int64` and a
`Decimal` to precision 38. `rolling_mean` runs on floats and gives a temporal
input its type back. `sum` and `mean` are the reductions of a series to one
value, which a `select` of the aggregation runs. `agg` runs the sum of each
group, a kernel of its own.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is, KnownNan
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.dispatch import (
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    MaxPrecision,
    Recast,
    Refuse,
    RunCast,
    Same,
)
from astrolabe._rules.type_functions import CutBins

if TYPE_CHECKING:
    from collections.abc import Sequence


class Cut:
    """Give the type of `cut` at the data of its input: the labels of its bins, after a cast.

    The kernel first checks the breaks. A break that is NaN or infinite
    raises a `ComputeError`, and two equal breaks raise a `DuplicateError`.
    The options of a node give NaN for a break that is NaN or infinite
    (`CutBins.names`).
    It sorts the breaks, so the labels of the intervals follow the sorted
    breaks (`CutBins.sorted`), and the labels of the call must number one
    more than the breaks. Then it casts the input to `Float64`, and a cast
    with no kernel raises. A struct casts each field and stays a struct, so
    the kernel finds no floats and raises a `SchemaError`. Last, it builds
    the type of the labels (`CutBins.dtype`).

    Example:
        >>> bins = CutBins((2.0, 0.0), None, left_closed=False, include_breaks=False)
        >>> Cut(bins).give(Data(DType("Int8")), Dispatch("cut", (), Cut(bins)))
        Enum(categories=['(-inf, 0]', '(0, 2]', '(2, inf]'])
    """

    def __init__(self, bins: CutBins) -> None:
        """Keep the options of the call.

        Args:
            bins: The options.

        Example:
            >>> Cut(CutBins((0.0,), None, left_closed=True, include_breaks=False))._bins.breaks
            (0.0,)
        """
        self._bins = bins

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the breaks or the labels do not pass
        their checks, or where the input has no cast to `Float64`.

        Args:
            data: The data of the input.
            kernel: The dispatch, which this output does not read.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The input is a struct with values, which does not become floats.

        Example:
            >>> bins = CutBins((0.0,), None, left_closed=False, include_breaks=False)
            >>> cut = Dispatch("cut", (), Cut(bins))
            >>> Cut(bins).give(Data(DType("Struct", ((("a", DType("Int8")),),))), cut)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
        """
        del kernel
        bins = self._checked()
        floats = RunCast().of(data, DType("Float64"))
        if floats.dtype.name != "Float64":
            message = f"invalid series dtype: expected `Float64`, got `{floats.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return bins.dtype()

    def _checked(self) -> CutBins:
        """Check the breaks and the labels, and give the options with the breaks sorted.

        Returns:
            The options.

        Raises:
            PolarsRefusal: Two breaks are equal, or a break is infinite.

        Example:
            >>> Cut(CutBins((1.0, 1.0), None, left_closed=False, include_breaks=False))._checked()
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: DuplicateError: breaks are not unique
            >>> Cut(CutBins((math.inf,), None, left_closed=False, include_breaks=False))._checked()
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: do not include infinity in breaks
        """
        self._unknown()
        bins = self._bins.sorted()
        if self._repeats(bins.breaks):
            raise PolarsRefusal(ErrorClass.DUPLICATE, "breaks are not unique")
        if (  # pragma: no cover -- the type function refuses it first
            any(math.isinf(value) for value in bins.breaks)
        ):
            raise PolarsRefusal(ErrorClass.COMPUTE, "do not include infinity in breaks")
        bins.names()
        return bins

    def _unknown(self) -> None:
        """Refuse a break that is NaN or infinite, which the options give as NaN.

        Polars refuses a NaN first, and an infinite break after the check of
        equal breaks, each with a `ComputeError`. The options of a node read
        both as NaN (`CutBins.names`). So the class of the error is known,
        unless the other breaks repeat or more than one break is unknown. A
        NaN that the code holds (`KnownNan`) is a NaN.

        Raises:
            PolarsRefusal: One break is unknown, and the other breaks do not repeat.
            UndecidedError: The class of the error depends on the unknown values.

        Example:
            >>> bins = CutBins((math.nan, 0.0), None, left_closed=False, include_breaks=False)
            >>> Cut(bins)._unknown()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: breaks cannot be NaN or infinite
            >>> both = CutBins((math.nan, math.nan), None, left_closed=False, include_breaks=False)
            >>> Cut(both)._unknown()  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: not serialized: the error of cut over ...
            >>> Cut(CutBins((0.0,), None, left_closed=False, include_breaks=False))._unknown()
        """
        breaks = self._bins.breaks
        unknown = [value for value in breaks if math.isnan(value)]
        if not unknown:
            return
        if any(isinstance(value, KnownNan) for value in unknown):
            raise PolarsRefusal(ErrorClass.COMPUTE, "breaks cannot be NaN")
        known = sorted(value for value in breaks if not math.isnan(value))
        if (  # pragma: no cover -- the type function refuses it first
            len(unknown) > 1 or self._repeats(known)
        ):
            question = "the error of cut over breaks that are NaN or infinite"
            raise UndecidedError(Undecided.NOT_SERIALIZED, question)
        raise PolarsRefusal(ErrorClass.COMPUTE, "breaks cannot be NaN or infinite")

    def _repeats(self, breaks: Sequence[float]) -> bool:
        """Say whether two sorted breaks are equal.

        Args:
            breaks: The breaks, in ascending order.

        Returns:
            Whether two neighbors are equal.

        Example:
            >>> Cut(CutBins((), None, left_closed=False, include_breaks=False))._repeats([0.0, 0.0])
            True
        """
        return any(low == high for low, high in itertools.pairwise(breaks))


class NumericKernels:
    """Hold the dispatch of each shared numeric kernel.

    `ABS` keeps a signed integer, a float, a `Decimal` and a `Duration`,
    returns an unsigned integer as it is, and raises on every other type.

    `NEGATE` keeps a signed integer of 64 bits or fewer, a float, a
    `Decimal` and a `Duration`. Every other type raises, an `Int128` and an
    unsigned integer included.

    `KEEP_NUMBER` keeps a float, an integer and a `Decimal`, and every other
    type raises. `round`, `truncate`, `floor`, `ceil` and `sign` have this
    dispatch, and so does `round_sig_figs` with at least one digit
    (`round_sig_figs`).

    `CUM_SUM` gives `UInt32` for a `Boolean`, `Int64` for an integer of 8 or 16
    bits, keeps a wider integer, a float and a `Duration`, and gives a
    `Decimal` of precision 38.

    `ROLLING_MEAN` runs a temporal input on its physical integers and gives
    `Datetime("us")` for a `Date` and the input type for the others. It keeps
    `Float32` and `Float64` and casts every other input to `Float64`. A cast
    with no kernel raises. A struct casts to a struct, and the window kernel
    panics on it.

    `SUM` is the sum of a series. It gives `UInt32` for a `Boolean`, `Int64`
    for an integer of 8 or 16 bits, and a `Decimal` of precision 38. It keeps
    the input type for a wider integer, a float, a `Duration` and `Null`. Every
    other type raises.

    `SUM_GROUPS` is the sum of each group, which `agg` runs. It has the arms of
    `SUM`. A type that the sum of the groups does not support gives one null
    for each group, of the input type, and does not raise. So a `Date` raises
    in a `select`, and gives a null `Date` in `agg`.

    `MEAN` is the mean of a series. It gives `Float64` for an integer, a
    `Boolean` and a `Decimal`, and `Datetime("us")` for a `Date`. It keeps the
    input type for a float, a `Datetime`, a `Duration`, a `Time` and `Null`.
    Every other type raises.

    Example:
        >>> NumericKernels.NEGATE.output(Data(DType("UInt8")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: neg does not accept UInt8
        >>> NumericKernels.KEEP_NUMBER.output(Data(DType("UInt128")))
        UInt128
        >>> NumericKernels.CUM_SUM.output(Data(DType("Decimal", (10, 2))))
        Decimal(precision=38, scale=2)
        >>> NumericKernels.ROLLING_MEAN.output(Data(DType("Date")))
        Datetime(time_unit='us', time_zone=None)
        >>> NumericKernels.MEAN.output(Data(DType("Boolean")))
        Float64
        >>> NumericKernels.SUM_GROUPS.output(Data(DType("Date")))
        Date
    """

    SMALL: ClassVar[Kinds] = Kinds(*Is.SMALL_INTEGERS)
    MICROSECONDS: ClassVar[DType] = DType("Datetime", ("us", None))
    ABS: ClassVar[Dispatch] = Dispatch(
        "abs",
        (
            Arm(Is.signed, Same()),
            Arm(Is.float, Same()),
            Arm(Kinds("Decimal", "Duration"), Same()),
            Arm(Is.unsigned, Same()),
        ),
        Refuse(),
    )
    NEGATE: ClassVar[Dispatch] = Dispatch(
        "neg",
        (
            Arm(Kinds("Int8", "Int16", "Int32", "Int64"), Same()),
            Arm(Is.float, Same()),
            Arm(Kinds("Decimal", "Duration"), Same()),
        ),
        Refuse(),
    )
    KEEP_NUMBER: ClassVar[Dispatch] = Dispatch(
        "round",
        (
            Arm(Is.float, Same()),
            Arm(Kinds("Decimal"), Same()),
            Arm(Is.integer, Same()),
        ),
        Refuse(),
    )
    CUM_SUM: ClassVar[Dispatch] = Dispatch(
        "cum_sum",
        (
            Arm(Kinds("Boolean"), Fixed(DType("UInt32"))),
            Arm(SMALL, Fixed(DType("Int64"))),
            Arm(Is.primitive_numeric, Same()),
            Arm(Kinds("Decimal"), MaxPrecision()),
            Arm(Kinds("Duration"), Same()),
        ),
        Refuse(),
    )
    WINDOW: ClassVar[Dispatch] = Dispatch(
        "rolling_mean",
        (Arm(Kinds("Float32", "Float64"), Same()),),
        Refuse(Stop.PANICS),
    )
    ROLLING_MEAN: ClassVar[Dispatch] = Dispatch(
        "rolling_mean",
        (
            Arm(Kinds("Date"), Fixed(MICROSECONDS)),
            Arm(Kinds("Datetime", "Duration", "Time"), Same()),
            Arm(Kinds("Float32", "Float64"), Same()),
        ),
        Recast(DType("Float64"), then=WINDOW),
    )
    SUM: ClassVar[Dispatch] = Dispatch(
        "sum",
        (
            Arm(Kinds("Boolean"), Fixed(DType("UInt32"))),
            Arm(SMALL, Fixed(DType("Int64"))),
            Arm(Is.primitive_numeric, Same()),
            Arm(Kinds("Decimal"), MaxPrecision()),
            Arm(Kinds("Duration"), Same()),
            Arm(Kinds("Null"), Same()),
        ),
        Refuse(),
    )
    SUM_GROUPS: ClassVar[Dispatch] = Dispatch("sum", SUM.arms, Same())
    MEAN: ClassVar[Dispatch] = Dispatch(
        "mean",
        (
            Arm(Is.integer, Fixed(DType("Float64"))),
            Arm(Is.float, Same()),
            Arm(
                Kinds("Boolean", "Decimal"),
                Fixed(DType("Float64")),
            ),
            Arm(Kinds("Date"), Fixed(MICROSECONDS)),
            Arm(
                Kinds("Datetime", "Duration", "Time", "Null"),
                Same(),
            ),
        ),
        Refuse(),
    )

    def round_sig_figs(self, digits: int) -> Dispatch:
        """Give the kernel of `round_sig_figs` with a number of digits.

        The kernel checks the digits before the type, so fewer than one digit
        raises on every type.

        Args:
            digits: The number of significant digits.

        Returns:
            The dispatch.

        Example:
            >>> NumericKernels().round_sig_figs(2).output(Data(DType("Int8")))
            Int8
            >>> NumericKernels().round_sig_figs(0).output(Data(DType("Float64")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: round_sig_figs ...
        """
        if digits < 1:
            return Dispatch("round_sig_figs", (), Refuse())
        return self.KEEP_NUMBER

    def cut(
        self,
        breaks: Sequence[float],
        labels: Sequence[str] | None,
        *,
        left_closed: bool,
        include_breaks: bool,
    ) -> Dispatch:
        """Give the kernel of `cut` with its options (`Cut`).

        Args:
            breaks: The breaks, in the order of the call.
            labels: The labels of the call, or `None`.
            left_closed: Whether each bin holds its lower break and not its upper break.
            include_breaks: Whether the output holds the upper break of each value.

        Returns:
            The dispatch.

        Example:
            >>> labels = ["low", "high"]
            >>> cut = NumericKernels().cut([0.0], labels, left_closed=False, include_breaks=False)
            >>> cut.output(Data(DType("String")))
            Enum(categories=['low', 'high'])
        """
        names = None if labels is None else tuple(labels)
        bins = CutBins(tuple(breaks), names, left_closed=left_closed, include_breaks=include_breaks)
        return Dispatch("cut", (), Cut(bins))
