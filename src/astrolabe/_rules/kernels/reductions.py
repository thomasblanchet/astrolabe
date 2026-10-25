"""The kernels of the aggregations, such as `min`, `std`, `implode` or `quantile`.

An aggregation reduces its input to one value. Polars runs it with two kernels.
In a `select`, the reduction of a whole series runs, and a type that the
series of that type does not support raises. In `group_by().agg()`, the
reduction of each group runs, and a type that it does not support gives one
null of the input type for each group, with no error. So `std` raises over a
`Date` in a `select`, and gives a null `Date` in `agg`.

The reductions of one input are dispatches on the data type (`Dispatch`). The
reductions of two inputs, such as `corr` and `index_of`, read the lengths of
their inputs and check them themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.dispatch import (
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    MaxPrecision,
    Physical,
    Recast,
    Refuse,
    RunCast,
    Same,
)
from astrolabe._rules.kernels.numeric import NumericKernels

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


class Listed:
    """Give a `List` of the type of the input, as `implode` gives the values of a series.

    Example:
        >>> implode = Dispatch("implode", (), Listed())
        >>> Listed().give(Data(DType("Date")), implode)
        List(Date)
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data.
            kernel: The dispatch, which this output does not read.

        Returns:
            The list type.

        Example:
            >>> implode = Dispatch("implode", (), Listed())
            >>> Listed().give(Data(DType("List", (DType("Int8"),))), implode)
            List(List(Int8))
        """
        del kernel
        return DType("List", (data.dtype,))


class Single:
    """Give the type of the input where it holds exactly one row, as `item` needs.

    `item` gives the one value of its input. A series of another number of
    rows raises a `ComputeError`, in a `select` and in each group of `agg`.

    Example:
        >>> item = Dispatch("item", (), Single())
        >>> Single().give(Data(DType("Int8"), one_row=True), item)
        Int8
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type, or refuse a series that does not hold one row.

        Args:
            data: The data.
            kernel: The dispatch.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The series does not hold exactly one row.

        Example:
            >>> item = Dispatch("item", (), Single())
            >>> Single().give(Data(DType("Int8")), item)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: item expected a single value, ...
        """
        if not data.one_row:
            message = f"{kernel.name} expected a single value, got more"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return data.dtype


class Neighbors:
    """Give the type of `is_sorted` where it orders each value beside the next one.

    `is_sorted` checks most types with a kernel of its own, and a struct after
    it encodes each row. A `List` and an `Array` fall back to `<=` of the
    values beside the values after them. And where the call gives no
    direction, `is_sorted` first infers it with `<` and `>` of the same
    values, over any type. An ordering refuses a list, an array and a struct
    (`Comparison`). A series of one row, or of only nulls, compares nothing.

    Example:
        >>> lists = Data(DType("List", (DType("Int8"),)), one_row=True)
        >>> Neighbors().give(lists, Dispatch("is_sorted", (), Neighbors()))
        Boolean
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type, or the refusal of the comparison.

        Args:
            data: The data.
            kernel: The dispatch, which this output does not read.

        Returns:
            `Boolean`.

        Example:
            >>> lists = Data(DType("List", (DType("Int8"),)))
            >>> Neighbors().give(lists, Dispatch("is_sorted", (), Neighbors()))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        del kernel
        if not data.one_row and not data.nulls:
            Comparison(CompareOp.LT_EQ).output(data, data)
        return DType("Boolean")


class ReductionKernels:
    """Hold the dispatch of each reduction of one input.

    `SAME` keeps the type of any input, as `first`, `last` and the minimum of
    each group do. `COUNT` gives the type of a count or an index, `UInt32`, for
    any input, as `count`, `n_unique`, `arg_min` and `null_count` do.
    `EXTREMUM` is `min` and `max` of a series: a `List`, an `Array` and a
    `Struct` raise, and every other type stays.

    `MEDIAN` has the arms of `NumericKernels.MEAN`. `MOMENT_GROUPS` is the mean
    and the median of each group: the same arms, and one null of the input
    type for any other type. `STD` gives `Float64` for an integer, a `Boolean`
    and a `Decimal`, and keeps a float, a `Duration` and `Null`. `VAR` has the
    arms of `STD` without the `Duration`. `STD_GROUPS` is the standard
    deviation and the variance of each group, with the arms of `STD`.

    `IMPLODE` gives a `List` of the input type. `ITEM` needs exactly one row.
    `AGG_GROUPS` gives the row indices of each group. The reduction of a whole
    series has no groups, so it panics there (`AGG_GROUPS_WHOLE`).

    `APPROX_N_UNIQUE` refuses a `List`, an `Array` and a `Struct`. `ENTROPY`
    reads the physical type: a number gives `Float64`, apart from `Float32`,
    which stays. `PRODUCT` gives `Int64` for a `Boolean` and an integer of 32
    bits or fewer, keeps a wider integer and a float, and gives a `Decimal` of
    precision 38.

    `IS_SORTED` gives `Boolean`, and refuses a `List` or an `Array` of
    two rows or more (`Neighbors`).

    `BITWISE` is the bitwise `and`, `or` and `xor` of a series: a `Boolean`, an
    integer and a float stay. `ANY_ALL` needs a `Boolean`. `SKEW` casts its
    input to `Float64`, and panics where the cast does not give `Float64`, as
    a struct does. `SKEW_GROUPS` refuses an input that is not `Float64`.

    Example:
        >>> ReductionKernels.EXTREMUM.output(Data(DType("Struct", ((("a", DType("Int8")),),))))
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: min does not accept ...
        >>> ReductionKernels.MOMENT_GROUPS.output(Data(DType("String")))
        String
        >>> ReductionKernels.STD.output(Data(DType("Duration", ("ms",))))
        Duration(time_unit='ms')
        >>> ReductionKernels.PRODUCT.output(Data(DType("UInt8")))
        Int64
    """

    FLOAT64: ClassVar[DType] = DType("Float64")
    SAME: ClassVar[Dispatch] = Dispatch("first", (), Same())
    COUNT: ClassVar[Dispatch] = Dispatch("count", (), Fixed(DType.idx()))
    NESTED: ClassVar[Kinds] = Kinds(*Is.NESTED)
    EXTREMUM: ClassVar[Dispatch] = Dispatch("min", (Arm(NESTED, Refuse()),), Same())
    MEDIAN: ClassVar[Dispatch] = Dispatch("median", NumericKernels.MEAN.arms, Refuse())
    MOMENT_GROUPS: ClassVar[Dispatch] = Dispatch("mean", NumericKernels.MEAN.arms, Same())
    VAR_ARMS: ClassVar[tuple[Arm, ...]] = (
        Arm(Is.integer, Fixed(FLOAT64)),
        Arm(Is.float, Same()),
        Arm(Kinds("Boolean", "Decimal"), Fixed(FLOAT64)),
        Arm(Kinds("Null"), Same()),
    )
    STD_ARMS: ClassVar[tuple[Arm, ...]] = (*VAR_ARMS, Arm(Kinds("Duration"), Same()))
    STD: ClassVar[Dispatch] = Dispatch("std", STD_ARMS, Refuse())
    VAR: ClassVar[Dispatch] = Dispatch("var", VAR_ARMS, Refuse())
    STD_GROUPS: ClassVar[Dispatch] = Dispatch("std", STD_ARMS, Same())
    IMPLODE: ClassVar[Dispatch] = Dispatch("implode", (), Listed())
    ITEM: ClassVar[Dispatch] = Dispatch("item", (), Single())
    AGG_GROUPS_WHOLE: ClassVar[Dispatch] = Dispatch("agg_groups", (), Refuse(Stop.PANICS))
    AGG_GROUPS: ClassVar[Dispatch] = Dispatch(
        "agg_groups",
        (),
        Fixed(DType("List", (DType.idx(),))),
    )
    APPROX_N_UNIQUE: ClassVar[Dispatch] = Dispatch(
        "approx_n_unique",
        (Arm(NESTED, Refuse()),),
        Fixed(DType.idx()),
    )
    ENTROPY: ClassVar[Dispatch] = Dispatch(
        "entropy",
        (
            Arm(Kinds("Float32"), Same()),
            Arm(Physical(Is.primitive_numeric), Fixed(FLOAT64)),
        ),
        Refuse(),
    )
    PRODUCT: ClassVar[Dispatch] = Dispatch(
        "product",
        (
            Arm(
                Kinds("Boolean", "Int8", "UInt8", "Int16", "UInt16", "Int32", "UInt32"),
                Fixed(DType("Int64")),
            ),
            Arm(Is.primitive_numeric, Same()),
            Arm(Kinds("Decimal"), MaxPrecision()),
        ),
        Refuse(),
    )
    BITWISE: ClassVar[Dispatch] = Dispatch(
        "and_reduce",
        (
            Arm(Kinds("Boolean"), Same()),
            Arm(Is.primitive_numeric, Same()),
        ),
        Refuse(),
    )
    ANY_ALL: ClassVar[Dispatch] = Dispatch(
        "any",
        (Arm(Kinds("Boolean"), Same()),),
        Refuse(ErrorClass.SCHEMA),
    )
    SKEW_GROUPS: ClassVar[Dispatch] = Dispatch(
        "skew",
        (Arm(Kinds("Float64"), Same()),),
        Refuse(ErrorClass.SCHEMA),
    )
    SKEW: ClassVar[Dispatch] = Dispatch(
        "skew",
        (Arm(Kinds("Float64"), Same()),),
        Recast(
            FLOAT64, then=Dispatch("skew", (Arm(Kinds("Float64"), Same()),), Refuse(Stop.PANICS))
        ),
    )
    BOOLEAN: ClassVar[Dispatch] = Dispatch("has_nulls", (), Fixed(DType("Boolean")))
    IS_SORTED: ClassVar[Dispatch] = Dispatch(
        "is_sorted",
        (Arm(Kinds(*Is.LISTS), Neighbors()),),
        Fixed(DType("Boolean")),
    )

    def is_sorted(self, descending: bool | None) -> Dispatch:
        """Give the kernel of `is_sorted` with the direction of the call.

        Args:
            descending: The direction, or `None` where `is_sorted` infers it.

        Returns:
            The dispatch.

        Example:
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> ReductionKernels().is_sorted(descending=False).output(struct)
            Boolean
            >>> ReductionKernels().is_sorted(descending=None).output(struct)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if descending is None:
            return Dispatch("is_sorted", (), Neighbors())
        return self.IS_SORTED

    def bound(self, planned: DType) -> Dispatch:
        """Give the kernel of `lower_bound` and `upper_bound`: the literal of the planned type.

        Polars puts a literal in place of the node when it converts the
        expression, so the input does not run, and the data has the type of
        the literal.

        Args:
            planned: The type of the node at the planner types of its input.

        Returns:
            The dispatch.

        Example:
            >>> ReductionKernels().bound(DType("Int32")).output(Data(DType("String")))
            Int32
        """
        return Dispatch("bound", (), Fixed(planned))


class ValueAtKey:
    """Give the type of `min_by` and `max_by`: the value at the place of the least or greatest key.

    The kernel finds the place of the extremum of the key, for a key of any
    type, and gives the value at that place, or a null of the type of the
    value. So the output has the type of the value.

    Example:
        >>> key = DType("Struct", ((("a", DType("Int8")),),))
        >>> ValueAtKey().output(Data(DType("Date")), Data(key))
        Date
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the value.
            right: The data of the key, which this kernel does not read.

        Returns:
            The type of the value.

        Example:
            >>> ValueAtKey().output(Data(DType("String")), Data(DType("Null")))
            String
        """
        del right
        return left.dtype


class IndexOf:
    """Give the type of `index_of`: the place of the first row that holds the element.

    The cast pass casts the element to the type of the series. The kernel then
    checks that the element holds exactly one value, and that it has the type
    of the series. Each check raises an `InvalidOperationError`. The place is
    an index, `UInt32`, or a null.

    Example:
        >>> inputs = [Data(DType("Int8")), Data(DType("Int8"))]
        >>> IndexOf().output(inputs, [3, 1])
        UInt32
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the series and of the element.
            lengths: The number of rows of each input.

        Returns:
            `UInt32`.

        Raises:
            PolarsRefusal: The element does not hold one value, or its type differs.

        Example:
            >>> inputs = [Data(DType("Int8")), Data(DType("Int16"))]
            >>> IndexOf().output(inputs, [3, 1])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: Cannot perform ...
            >>> IndexOf().output(inputs, [3, 2])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: needle of `index_of` ...
        """
        series, element = inputs[0], inputs[1]
        if lengths and lengths[1] != 1:  # pragma: no cover -- the conversion checks it first
            message = f"needle of `index_of` can only contain a single value, found {lengths[1]}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if not series.dtype.same(element.dtype):
            message = f"Cannot perform index_of with mismatching datatypes: {series.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return DType.idx()


class Correlation:
    """Give the type of `corr` and `cov`, with the method of the call.

    The cast pass gives the two inputs their supertype. The kernel first
    checks that the inputs have one length, or that one of them has one row,
    and raises a `ShapeError` otherwise. Then it reads the type of the first
    input. A float keeps its type. The covariance runs a few integer types
    as they are, and gives `Float64` for them. Pearson runs `Int32`, `Int64`
    and `UInt32` as they are. Any other type casts both inputs to `Float64`,
    where a cast with no kernel raises. Where the cast does not give
    `Float64`, as for a struct, the kernel panics. A kernel that runs a type
    as it is reads the second input at that type too, and panics where it has
    another type. The rank correlation of Spearman ranks each input first: the
    rank of a float keeps its type, and any other rank is a `Float64`.

    Example:
        >>> inputs = [Data(DType("Float32")), Data(DType("Float32"))]
        >>> Correlation("Pearson").output(inputs, [3, 3])
        Float32
        >>> Correlation("Covariance").output([Data(DType("Int8"))] * 2, [3, 1])
        Float64
    """

    FLOAT64: ClassVar[DType] = DType("Float64")
    FLOATS: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32"})
    DIRECT: ClassVar[dict[str, frozenset[str]]] = {
        "Pearson": frozenset({"Float64", "Int32", "Int64", "UInt32"}),
        "Covariance": frozenset({"Float64", "Int32", "Int64", "UInt32", "UInt64"}),
    }

    def __init__(self, method: str) -> None:
        """Keep the method.

        Args:
            method: `Pearson`, `SpearmanRank` or `Covariance`.

        Example:
            >>> Correlation("SpearmanRank")._method
            'SpearmanRank'
        """
        self._method = method

    def __repr__(self) -> str:
        """Give the method.

        Returns:
            The text.

        Example:
            >>> Correlation("Pearson")
            Correlation('Pearson')
        """
        return f"Correlation({self._method!r})"

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the lengths differ, or where a cast or
        the kernel fails.

        Args:
            inputs: The data of the two inputs.
            lengths: The number of rows of each input.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The inputs have two lengths, and neither has one row.

        Example:
            >>> Correlation("Pearson").output([Data(DType("Int8"))] * 2, [3, 2])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
        """
        if lengths and lengths[0] != lengths[1] and 1 not in lengths:
            message = f"length mismatch in corr: {lengths[0]} and {lengths[1]}"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        left, right = inputs[0], inputs[1]
        if self._method == "SpearmanRank":
            self._coalesced(left, right, lengths)
            left, right = self._ranked(left), self._ranked(right)
        return self._pearson_or_covariance(left, right)

    def _coalesced(self, left: Data, right: Data, lengths: Sequence[int]) -> None:
        """Check that the rank correlation can join the nulls of its two inputs.

        Before it ranks them, the rank correlation of Spearman gives each input
        the nulls of the other. Where the inputs have two lengths, as one row
        beside more rows, and an input holds nulls, the validity of one does
        not fit the other, and polars panics.

        Args:
            left: The data of the first input.
            right: The data of the second input.
            lengths: The number of rows of each input.

        Raises:
            PolarsRefusal: The lengths differ, and an input holds only nulls.

        Example:
            >>> null = Data(DType("Float64"), nulls=True)
            >>> Correlation("SpearmanRank")._coalesced(null, Data(DType("Float64")), [1, 3])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: validity must be equal ...
        """
        uneven = bool(lengths) and lengths[0] != lengths[1]
        if uneven and (left.nulls or right.nulls):
            raise PolarsRefusal(Stop.PANICS, "validity must be equal to the array's length")

    def _ranked(self, data: Data) -> Data:
        """Give the data of the average rank of an input.

        Args:
            data: The data of the input.

        Returns:
            The data of the rank: the type of a float, and `Float64` otherwise.

        Example:
            >>> Correlation("SpearmanRank")._ranked(Data(DType("Int8"))).dtype
            Float64
        """
        return data if Is.float(data.dtype) else data.with_dtype(DType("Float64"))

    def _pearson_or_covariance(self, left: Data, right: Data) -> DType:
        """Give the type of the correlation of Pearson, or of the covariance, of two inputs.

        Args:
            left: The data of the first input.
            right: The data of the second input.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The kernel reads the second input at the type of the
                first, which it does not have, or the cast to `Float64` does
                not give `Float64`.

        Example:
            >>> pair = Data(DType("Float32")), Data(DType("Float64"))
            >>> Correlation("Pearson")._pearson_or_covariance(*pair)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: corr reads Float64 as Float32
        """
        name = left.dtype.name
        direct = self.DIRECT.get(self._method, self.DIRECT["Pearson"])
        if name in self.FLOATS or name in direct:
            if not right.dtype.same(left.dtype):
                message = f"corr reads {right.dtype!r} as {left.dtype!r}"
                raise PolarsRefusal(Stop.PANICS, message)
            return left.dtype if name in self.FLOATS else self.FLOAT64
        for data in (left, right):
            cast = RunCast().of(data, self.FLOAT64)
            if cast.dtype.name != "Float64":
                message = f"corr reads {cast.dtype!r} as Float64"
                raise PolarsRefusal(Stop.PANICS, message)
        return self.FLOAT64


class QuantileValue:
    """Read the quantile of `quantile`: its number of rows, its type, and the value of a literal.

    `quantile` takes one quantile, between 0 and 1. A quantile of more than one
    row raises a `ComputeError`, and so does a quantile that is null. The
    value of a column is a fact of the data, so the rules take a value that
    fits. The value of a literal is in the code, so a literal outside of 0 and
    1 meets the kernel at every run.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> QuantileValue().of([Operand.column(DType("Int8")), Operand.of_value(1.5)])
        1.5
    """

    TEXTS: ClassVar[frozenset[str]] = frozenset({"Boolean", "String", *Is.TEMPORAL})
    NULL: ClassVar[str] = "could not extract number from any-value of dtype: 'Null'"

    def single(self, lengths: Sequence[int]) -> None:
        """Refuse a quantile of more than one row.

        Args:
            lengths: The number of rows of each input, or none where they are not known.

        Raises:
            PolarsRefusal: The quantile has more than one row.

        Example:
            >>> QuantileValue().single([3, 2])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: polars does not support ...
        """
        if lengths and lengths[1] > 1:
            message = "polars does not support varying quantiles yet"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def extracted(self, quantile: Data, computed: Sequence[Operand]) -> float | None:
        """Give the quantile as a number, as a `select` reads it from its one value.

        A number, a `Boolean`, a text and a temporal value give a number, and
        any other type raises a `ComputeError`. A null of any type and a
        literal text that does not parse as a number raise too. The message
        of that error names the type of the value, and polars has no type for
        the value of a `Categorical` or an `Enum`: there it panics.

        Args:
            quantile: The data of the quantile.
            computed: Each input as the run computes it, or none where it is not known.

        Returns:
            The value of a literal, or `None` for a column.

        Raises:
            PolarsRefusal: The quantile holds no number.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> computed = [Operand.column(DType("Int8")), Operand.of_value("a")]
            >>> QuantileValue().extracted(Data(DType("String")), computed)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: could not extract number ...
        """
        if quantile.nulls:
            raise PolarsRefusal(ErrorClass.COMPUTE, self.NULL)
        name = quantile.dtype.name
        message = f"could not extract number from any-value of dtype: '{quantile.dtype!r}'"
        if name in Is.CATEGORIES:
            raise PolarsRefusal(Stop.PANICS, f"no dtype of a value of {quantile.dtype!r}")
        if not Is.numeric(quantile.dtype) and name not in self.TEXTS:
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        value = self.of(computed)
        if isinstance(value, str):
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return value

    def of(self, computed: Sequence[Operand]) -> float | str | None:
        """Give the value of a literal quantile as a number, or a text that does not parse.

        Args:
            computed: Each input as the run computes it, or none where it is not known.

        Returns:
            The number, the text, or `None` for a column or a null.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> QuantileValue().of([Operand.column(DType("Int8")), Operand.of_value(True)])
            1.0
            >>> QuantileValue().of([Operand.column(DType("Int8")), Operand.of_value(None)]) is None
            True
        """
        if len(computed) < 2 or not computed[1].literal:
            return None
        value = computed[1].value
        if isinstance(value, bool | int | float):
            return float(value)
        if not isinstance(value, str):  # pragma: no cover -- the type function refuses it first
            return None
        try:
            return float(value)
        except ValueError:
            return value

    def outside(self, value: float | str | None) -> bool:
        """Say whether a literal quantile is outside of 0 and 1.

        Args:
            value: The value, or `None` for a column, whose value fits.

        Returns:
            Whether it is outside.

        Example:
            >>> QuantileValue().outside(1.5), QuantileValue().outside(None)
            (True, False)
        """
        return isinstance(value, float) and not 0.0 <= value <= 1.0


class Quantile:
    """Give the type of `quantile` in a `select`: the quantile of a whole series.

    The kernel checks its quantile first (`QuantileValue`). A quantile that
    is a `List` gives a `List` of quantiles, where a `Decimal` keeps its type
    in the list (`QUANTILES`), and a null list panics. Then the series of an integer
    gives `Float64`, of a float its type, of a `Decimal` `Float64`, of a
    `Date` a `Datetime` in microseconds, and of another temporal type its
    type. Any other type raises an `InvalidOperationError`, and a literal
    quantile outside of 0 and 1 raises a `ComputeError`.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> computed = (Operand.column(DType("Date")), Operand.of_value(0.5))
        >>> Quantile().output([Data(DType("Date")), Data(DType("Float64"))], [3, 1], computed)
        Datetime(time_unit='us', time_zone=None)
    """

    ARMS: ClassVar[tuple[Arm, ...]] = (
        Arm(Is.integer, Fixed(DType("Float64"))),
        Arm(Is.float, Same()),
        Arm(Kinds("Decimal"), Fixed(DType("Float64"))),
        Arm(Kinds("Date"), Fixed(NumericKernels.MICROSECONDS)),
        Arm(Kinds("Datetime", "Duration", "Time"), Same()),
    )
    REDUCE: ClassVar[Dispatch] = Dispatch("quantile", ARMS, Refuse())
    QUANTILES: ClassVar[Dispatch] = Dispatch(
        "quantiles",
        (
            Arm(Is.integer, Fixed(DType("List", (DType("Float64"),)))),
            Arm(Kinds("Date"), Fixed(DType("List", (NumericKernels.MICROSECONDS,)))),
            Arm(Is.float, Listed()),
            Arm(Kinds("Decimal", "Datetime", "Duration", "Time"), Listed()),
        ),
        Refuse(),
    )

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the series and of the quantile.
            lengths: The number of rows of each input.
            computed: Each input as the run computes it.

        Returns:
            The type.

        Raises:
            PolarsRefusal: A literal quantile is outside of 0 and 1.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> computed = (Operand.column(DType("Int8")), Operand.of_value(2))
            >>> Quantile().output([Data(DType("Int8")), Data(DType("Int32"))], [3, 1], computed)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: quantile should be ...
        """
        series, quantile = inputs[0], inputs[1]
        reader = QuantileValue()
        reader.single(lengths)
        if quantile.dtype.name == "List":
            self._listed(quantile)
            return self.QUANTILES.output(series)
        value = reader.extracted(quantile, computed)
        dtype = self.REDUCE.output(series)
        if reader.outside(value):
            raise PolarsRefusal(ErrorClass.COMPUTE, "quantile should be between 0.0 and 1.0")
        return dtype

    def _listed(self, quantile: Data) -> None:
        """Check a list of quantiles: a list that is not null, whose values cast to `Float64`.

        It raises `PolarsRefusal` where the values have no cast to `Float64`
        (`RunCast`).

        Args:
            quantile: The data of the list.

        Raises:
            PolarsRefusal: The list is null, or the cast gives another type,
                as for a struct.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> Quantile()._listed(Data(DType("List", (struct,))))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
            >>> floats = DType("List", (DType("Float64"),))
            >>> Quantile()._listed(Data(floats, nulls=True))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: a null list of quantiles ...
        """
        if quantile.nulls:
            raise PolarsRefusal(Stop.PANICS, "a null list of quantiles holds no value")
        values = RunCast().of(Data(quantile.dtype.inner()), DType("Float64"))
        if values.dtype.name != "Float64":
            message = f"invalid series dtype: expected `Float64`, got `{values.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


class GroupQuantile:
    """Give the type of `quantile` in `agg`: the quantile of each group.

    The kernel reads its quantile over the whole frame, and checks it first:
    one row, and a number (`QuantileValue`), where a type that is not a
    number raises a `SchemaError` and a null raises a `ComputeError`. A
    literal series is one value, so it takes the quantile of a whole series
    (`Quantile`). Otherwise each group gives its quantile, and a type that
    has none gives a null of its type. A quantile outside of 0 and 1 gives a
    null for each group, and an integer then keeps its type, where it gives
    `Float64` otherwise. So the value of a quantile column decides the type
    of an integer.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> computed = (Operand.column(DType("Int8")), Operand.of_value(1.5))
        >>> GroupQuantile().output([Data(DType("Int8")), Data(DType("Float64"))], [3, 1], computed)
        Int8
        >>> column = (Operand.column(DType("Int8")), Operand.column(DType("Float64")))
        >>> GroupQuantile().output([Data(DType("Int8")), Data(DType("Float64"))], [3, 1], column)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.UndecidedError: values: the quantile of an integer, ...
    """

    GROUPS: ClassVar[Dispatch] = Dispatch("quantile", Quantile.ARMS, Same())
    OUTSIDE: ClassVar[Dispatch] = Dispatch(
        "quantile",
        (
            Arm(Kinds("Decimal"), Fixed(DType("Float64"))),
            Arm(Kinds("Date"), Fixed(NumericKernels.MICROSECONDS)),
        ),
        Same(),
    )

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the series and of the quantile.
            lengths: The number of rows of each input in a group.
            computed: Each input as the run computes it.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The quantile is not a number or is null, or a
                literal series meets a literal quantile outside of 0 and 1.
            UndecidedError: The series is an integer, and the quantile is
                not a literal.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> computed = (Operand.column(DType("Int8")), Operand.of_value(True))
            >>> inputs = [Data(DType("Int8")), Data(DType("Boolean"))]
            >>> GroupQuantile().output(inputs, [3, 1], computed)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: expected expression of dtype ...
        """
        series, quantile = inputs[0], inputs[1]
        reader = QuantileValue()
        reader.single(lengths)
        if not Is.numeric(quantile.dtype):
            message = (
                f"expected expression of dtype 'numeric' for quantile, got '{quantile.dtype!r}'"
            )
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        if quantile.nulls:
            raise PolarsRefusal(ErrorClass.COMPUTE, reader.NULL)
        if computed and computed[0].literal:
            return Quantile().output([series, quantile], (), computed)
        value = reader.of(computed)
        if value is None and Is.integer(series.dtype):
            question = "the quantile of an integer, whose type the value of the quantile decides"
            raise UndecidedError(Undecided.VALUES, question)
        return (self.OUTSIDE if reader.outside(value) else self.GROUPS).output(series)
