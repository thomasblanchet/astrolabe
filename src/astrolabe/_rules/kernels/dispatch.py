"""The parts that every kernel rule uses: the data, the cast at run time, the dispatch.

A kernel of polars runs on the data that reaches it. It reads the data type
of each input and the values, and it gives a series or fails. A rule of a
kernel does the same with plain values: it takes one `Data` for each input
and gives a `DType`, or it raises `PolarsRefusal` with the way that polars
fails. A kernel also casts types while it runs, and it does so with the
all-null shortcut of the cast of polars, so this module holds that cast once.
A kernel joins types with `Supertype.require`.

A kernel of one input picks its work by the data type, from a short list of
arms. For example, `abs` has one arm for each family of numbers. `Dispatch`
states such a list as data: each `Arm` names a test on the type and what the
arm gives.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.casts import CastKernel
from astrolabe._rules.dtypes import Decimals, DType, Parameter
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class Data:
    """Hold the facts of the data that reaches a kernel and decide its type.

    `dtype` is the data type of the series. It is a type of data, so it is
    never the planner type `Unknown`: a literal arrives at its materialized
    type. `nulls` says that every value is null, because the cast of polars
    then reads no data and casts to any type, and some kernels skip work on
    such a series. `one_row` says that the series has exactly one row,
    because some kernels cast or broadcast it: the bit operations cast a right
    side of one row to the type of the left side, even beside a left side of
    one row. Two inputs that both have `one_row` false have the same length,
    two rows or more. A series of type `Null` holds only nulls, so its `nulls`
    is always true. `field_nulls` says that the series is a struct whose
    fields hold only nulls, while no struct value is null: a number cast to a
    struct of one field, where the number does not fit the field. `bound` is
    the most rows that the code allows the series, where the code bounds
    them: a literal holds one row, and `head(2)` at most two. The rows of a
    column of the frame are a fact of the data, so its `bound` is `None`.
    `some_null` says that the series holds a null wherever it holds a row,
    because the code decides it, as for the first row of `shift(1)`.

    Example:
        >>> Data(DType("Int8"))
        Data(dtype=Int8, nulls=False, one_row=False)
        >>> Data(DType("Null")).nulls
        True
    """

    dtype: DType
    nulls: bool = False
    one_row: bool = False
    field_nulls: bool = False
    bound: int | None = None
    some_null: bool = False

    def __post_init__(self) -> None:
        """Mark the data of a `Null` series as only nulls.

        Example:
            >>> Data(DType("Null"), nulls=False)
            Data(dtype=Null, nulls=True, one_row=False)
        """
        if self.dtype.name == "Null":
            object.__setattr__(self, "nulls", True)

    def __repr__(self) -> str:
        """Give the facts, with `field_nulls`, `bound` and `some_null` only where they are set.

        Returns:
            The text.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> Data(struct, field_nulls=True)
            Data(dtype=Struct({'a': Int8}), nulls=False, one_row=False, field_nulls=True)
            >>> Data(DType("Int8"), some_null=True)
            Data(dtype=Int8, nulls=False, one_row=False, some_null=True)
        """
        text = f"Data(dtype={self.dtype!r}, nulls={self.nulls}, one_row={self.one_row}"
        text += ", field_nulls=True" if self.field_nulls else ""
        text += "" if self.bound is None else f", bound={self.bound}"
        return text + (", some_null=True" if self.some_null else "") + ")"

    def holds_null(self) -> bool:
        """Say whether the series holds a null wherever it holds a row: some values or all.

        Returns:
            Whether it does.

        Example:
            >>> Data(DType("Int8"), nulls=True).holds_null()
            True
            >>> Data(DType("Int8")).holds_null()
            False
        """
        return self.nulls or self.some_null

    def field(self, dtype: DType) -> Data:
        """Give the data of one field of a struct.

        A field holds only nulls where the struct values are null, or where
        the fields of the struct hold only nulls.

        Args:
            dtype: The data type of the field.

        Returns:
            The data.

        Example:
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> Data(struct, field_nulls=True).field(DType("Int8"))
            Data(dtype=Int8, nulls=True, one_row=False)
        """
        nulls = self.nulls or self.field_nulls
        inner = self.field_nulls and dtype.name == "Struct"
        return Data(dtype, nulls=nulls, one_row=self.one_row, field_nulls=inner)

    def with_dtype(self, dtype: DType) -> Data:
        """Give the same values at another data type, as after a cast.

        Args:
            dtype: The new data type.

        Returns:
            The data.

        Example:
            >>> Data(DType("Int8"), nulls=True).with_dtype(DType("Float64"))
            Data(dtype=Float64, nulls=True, one_row=False)
        """
        return replace(self, dtype=dtype)


class BinaryKernel(Protocol):
    """Give the type of a kernel of two inputs at the data that reaches it.

    A kernel of an operator, such as the addition of two series, has this
    shape. A kernel that applies another kernel to each part of its inputs,
    such as the arithmetic of two structs, takes this shape as a dependency.
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The data of the left input.
            right: The data of the right input.
        """


class InputsKernel(Protocol):
    """Give the type of a kernel that reads the data of every input and the number of rows of each.

    A kernel of this shape checks the lengths of its inputs itself, after it
    checks their types, as the parse of `str.strptime` does.
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The number of rows of each input, or none where they are
                not known.
        """


class LiteralsKernel(Protocol):
    """Give the type of a kernel that reads every input, its rows, and the value of a literal.

    The code holds the value of a literal input, so a kernel that reads that
    value meets it at every run: a text that does not parse as a duration
    raises in `dt.truncate`, at every run. The value of a column is a fact of
    the data, which no rule of the types knows.
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The number of rows of each input, or none where they are
                not known.
            computed: Each input as the run computes it: a literal with its
                value, or a column. It is empty where it is not known.
        """


class RunCast:
    """Cast the data that reaches a kernel, as the cast of a series does while a kernel runs.

    A kernel casts its input in many places, for example to bring two inputs to
    their supertype. The cast returns the series where it changes nothing,
    casts a series of only nulls to any type with no kernel, and otherwise
    needs a cast kernel for the pair. The cast does not fail on a value,
    because the casts of a kernel are not strict. `CastKernel` gives the type
    of the cast kernel, or its refusal.

    Where the cast fails, a kernel either raises the error of the cast or
    panics. Each kernel does one or the other.

    Example:
        >>> RunCast().of(Data(DType("Date")), DType("Float64")).dtype
        Float64
        >>> RunCast().of(Data(DType("Date"), nulls=True), DType("Boolean")).dtype
        Boolean
    """

    def __init__(self, *, unwrap: bool = False) -> None:
        """Keep how the kernel reads the result of the cast.

        Args:
            unwrap: Whether the kernel panics where the cast fails. A kernel
                that does not panic raises the error of the cast.

        Example:
            >>> RunCast(unwrap=True)._unwrap
            True
        """
        self._unwrap = unwrap
        self._kernel = CastKernel()

    def of(self, data: Data, target: DType) -> Data:
        """Give the data after the cast.

        Args:
            data: The data before the cast.
            target: The type of the cast.

        Returns:
            The data, with the type that the cast gives.

        Raises:
            PolarsRefusal: No cast kernel exists for the pair. The refusal is
                the refusal of the cast kernel, or a panic for a kernel that
                panics there.

        Example:
            >>> RunCast().of(Data(DType("Date")), DType("Boolean"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> RunCast(unwrap=True).of(Data(DType("Date")), DType("Boolean"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: casting from Date ...
        """
        if self._kernel.clones(data.dtype, target):
            return data
        if data.nulls:
            return data.with_dtype(self._kernel.target(data.dtype, target))
        try:
            output = self._kernel.output(data.dtype, target)
        except PolarsRefusal as refusal:
            if not self._unwrap:
                raise
            raise PolarsRefusal(Stop.PANICS, refusal.message) from refusal
        return data.with_dtype(output)


class Kinds:
    """Test a data type for one of a few kinds.

    A kind is a data type with its parameters taken away. An arm of a kernel
    often names kinds, for example `Decimal` or `Duration` for every value of
    their parameters.

    Example:
        >>> Kinds("Decimal", "Duration")(DType("Duration", ("ms",)))
        True
    """

    def __init__(self, *names: str) -> None:
        """Keep the names of the kinds.

        Args:
            *names: The names of the polars classes, for example `Date`.

        Example:
            >>> Kinds("Date")._names
            frozenset({'Date'})
        """
        self._names = frozenset(names)

    def __call__(self, dtype: DType) -> bool:
        """Say whether the type is of one of the kinds.

        Args:
            dtype: The data type.

        Returns:
            Whether it is.

        Example:
            >>> Kinds("Float32", "Float64")(DType("Float16"))
            False
        """
        return dtype.name in self._names


class Physical:
    """Test the physical type of a data type, for a kernel that reads the physical type first.

    The physical type is the type that holds the data: a `Date` holds 32-bit
    integers, and a `Decimal` holds 128-bit integers.

    Example:
        >>> Physical(Kinds("Int32"))(DType("Date"))
        True
    """

    def __init__(self, test: Callable[[DType], bool]) -> None:
        """Keep the test of the physical type.

        Args:
            test: The test.

        Example:
            >>> Physical(Kinds("Int8"))._test(DType("Int8"))
            True
        """
        self._test = test

    def __call__(self, dtype: DType) -> bool:
        """Say whether the physical type passes the test.

        Args:
            dtype: The data type.

        Returns:
            Whether it passes.

        Example:
            >>> Physical(Kinds("Int32"))(DType("Boolean"))
            False
        """
        return self._test(dtype.physical())


class Output(Protocol):
    """Give what an arm of a dispatch gives for the data that it accepts.

    It reads the dispatch itself, because an arm can cast the data and
    dispatch again, as a recursive kernel of polars does.
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output, or raise `PolarsRefusal`.

        Args:
            data: The data that the arm accepts.
            kernel: The dispatch that holds the arm.
        """


class Same:
    """Give the type of the input, for a kernel that keeps it.

    Example:
        >>> abs_kernel = Dispatch("abs", (), Refuse())
        >>> Same().give(Data(DType("Int8")), abs_kernel)
        Int8
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the input.

        Args:
            data: The data.
            kernel: The dispatch.

        Returns:
            The type of the data.

        Example:
            >>> abs_kernel = Dispatch("abs", (), Refuse())
            >>> Same().give(Data(DType("Duration", ("ms",))), abs_kernel)
            Duration(time_unit='ms')
        """
        return data.dtype


class Fixed:
    """Give one data type, whatever the type of the input.

    Example:
        >>> sum_kernel = Dispatch("sum", (), Refuse())
        >>> Fixed(DType("UInt32")).give(Data(DType("Boolean")), sum_kernel)
        UInt32
    """

    def __init__(self, dtype: DType) -> None:
        """Keep the type.

        Args:
            dtype: The type of the output.

        Example:
            >>> Fixed(DType("Float64"))._dtype
            Float64
        """
        self._dtype = dtype

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data.
            kernel: The dispatch.

        Returns:
            The fixed type.

        Example:
            >>> cum_sum = Dispatch("cum_sum", (), Refuse())
            >>> Fixed(DType("Int64")).give(Data(DType("Int8")), cum_sum)
            Int64
        """
        return self._dtype


class MaxPrecision:
    """Give a `Decimal` of the largest precision, 38, with the scale of the input.

    A kernel that adds many decimals gives this type, so that the sum fits
    (`Decimals.widest`).

    Example:
        >>> sum_kernel = Dispatch("sum", (), Refuse())
        >>> MaxPrecision().give(Data(DType("Decimal", (10, 2))), sum_kernel)
        Decimal(precision=38, scale=2)
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type.

        Args:
            data: The data, of a `Decimal` type.
            kernel: The dispatch.

        Returns:
            The `Decimal` type.

        Example:
            >>> cum_sum = Dispatch("cum_sum", (), Refuse())
            >>> MaxPrecision().give(Data(DType("Decimal", (5, 0))), cum_sum)
            Decimal(precision=38, scale=0)
        """
        return Decimals().widest(Parameter().integer(data.dtype, 1))


class ToPhysical:
    """Give the physical type of the input, for a kernel that gives the data that holds the values.

    The physical type of a `Date` is `Int32`, and a nested type takes the
    physical type of each inner type (`DType.physical`).

    Example:
        >>> to_physical = Dispatch("to_physical", (), ToPhysical())
        >>> ToPhysical().give(Data(DType("Date")), to_physical)
        Int32
    """

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the physical type.

        Args:
            data: The data.
            kernel: The dispatch.

        Returns:
            The physical type of the data.

        Example:
            >>> to_physical = Dispatch("to_physical", (), ToPhysical())
            >>> ToPhysical().give(Data(DType("List", (DType("Time"),))), to_physical)
            List(Int64)
        """
        return data.dtype.physical()


@dataclass(frozen=True)
class Recast:
    """Cast the data to a type and dispatch again, as a kernel that calls itself after a cast.

    Many kernels of polars cast an input that they do not accept and call
    themselves again, for example `sqrt` casts to `Float64`. The second call
    reads the type that the cast gives. A cast of a `Struct` to a number gives
    a `Struct` again, so the kernel calls itself without end and the stack of
    the process overflows. This arm then raises `PolarsRefusal` with the way
    `CRASHES`.

    `unwrap` says that a failed cast panics, and not that it raises. `then` is
    the dispatch of the second call where it is not the same kernel, and that
    call does not recurse. `physical` casts the physical type, for a kernel
    that reads the physical type first.

    Example:
        >>> floats = (Arm(Kinds("Float32", "Float64"), Same()),)
        >>> sqrt = Dispatch("sqrt", floats, Recast(DType("Float64")))
        >>> Recast(DType("Float64")).give(Data(DType("Int8")), sqrt)
        Float64
    """

    target: DType
    unwrap: bool = False
    then: Dispatch | None = None
    physical: bool = False

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the second call after the cast.

        It raises `PolarsRefusal` where the cast fails (`RunCast`).

        Args:
            data: The data.
            kernel: The dispatch that holds the arm, which runs again where
                `then` is `None`.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The kernel calls itself without end.

        Example:
            >>> floats = (Arm(Kinds("Float32", "Float64"), Same()),)
            >>> recast = Recast(DType("Float64"))
            >>> sqrt = Dispatch("sqrt", floats, recast)
            >>> nested = DType("Struct", ((("a", DType("Int8")),),))
            >>> recast.give(Data(nested), sqrt)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: crashes: sqrt calls ...
        """
        source = data.with_dtype(data.dtype.physical()) if self.physical else data
        cast = RunCast(unwrap=self.unwrap).of(source, self.target)
        if self.then is not None:
            return self.then.output(cast)
        if cast.dtype.same(source.dtype):
            message = f"{kernel.name} calls itself on {cast.dtype!r} without end"
            raise PolarsRefusal(Stop.CRASHES, message)
        return kernel.output(cast)


@dataclass(frozen=True)
class Refuse:
    """Fail on the data, as the last arm of a kernel that raises or panics.

    `how` is the class of the error where the arm raises, and `Stop.PANICS`
    where it panics. Most kernels raise an `InvalidOperationError`.

    Example:
        >>> refuse = Refuse()
        >>> refuse.give(Data(DType("String")), Dispatch("abs", (), refuse))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: abs ...
    """

    how: ErrorClass | Stop = ErrorClass.INVALID_OPERATION

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Raise the refusal.

        Args:
            data: The data.
            kernel: The dispatch.

        Raises:
            PolarsRefusal: Always.

        Example:
            >>> panic = Refuse(Stop.PANICS)
            >>> panic.give(Data(DType("Null")), Dispatch("rolling_mean", (), panic))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: rolling_mean ...
        """
        message = f"{kernel.name} does not accept {data.dtype!r}"
        raise PolarsRefusal(self.how, message)


@dataclass(frozen=True)
class Arm:
    """Hold one arm of a kernel: a test on the data type, and what the kernel gives where it passes.

    `accepts` tests the data type, and `output` gives the type of the arm.

    Example:
        >>> Arm(Kinds("Int8", "Int16"), Same()).accepts(DType("Int8"))
        True
    """

    accepts: Callable[[DType], bool]
    output: Output


@dataclass(frozen=True)
class Dispatch:
    """State the dispatch of a kernel of one input on the data type, as a list of arms.

    A definition of a node names its kernel with a `Dispatch`. The arms run in
    order, and the first arm that accepts the data type decides. Where no arm
    accepts it, `otherwise` decides, for example a `Refuse` that raises as the
    last arm of many kernels does.

    Example:
        >>> signed = Arm(Kinds("Int8"), Same())
        >>> unsigned = Arm(Kinds("UInt8"), Same())
        >>> abs_kernel = Dispatch("abs", (signed, unsigned), Refuse())
        >>> abs_kernel.output(Data(DType("UInt8")))
        UInt8
        >>> abs_kernel.output(Data(DType("String")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: abs ...
    """

    name: str
    arms: tuple[Arm, ...]
    otherwise: Output

    def output(self, data: Data) -> DType:
        """Give the type of the output for the data.

        It raises `PolarsRefusal` where the kernel fails on the data.

        Args:
            data: The data that reaches the kernel.

        Returns:
            The type of the output.

        Example:
            >>> arms = (Arm(Kinds("Float32"), Same()),)
            >>> Dispatch("abs", arms, Refuse()).output(Data(DType("Float32")))
            Float32
        """
        for arm in self.arms:
            if arm.accepts(data.dtype):
                return arm.output.give(data, self)
        return self.otherwise.give(data, self)


@dataclass(frozen=True)
class KernelFact:
    """Hold a fact of a kernel that the types of its inputs do not decide.

    `reason` says why no rule of the types covers it. `probe` is a Python
    expression, over the module `pl` of polars, that shows it.

    Example:
        >>> KernelFacts.SHARED_NULL_LEAF.name
        'shared null leaf'
    """

    name: str
    reason: str
    probe: str


class KernelFacts:
    """List the facts of the kernels that a rule of the types cannot give.

    A kernel rule gives the type of the run where the values fit. A few
    failures depend on the values in a way that no rule of the types sees.
    Each one is about one kernel and one shape of the values, never about a
    combination of input types.

    Example:
        >>> [fact.name for fact in KernelFacts.ALL]
        ['shared null leaf']
    """

    SHARED_NULL_LEAF: ClassVar[KernelFact] = KernelFact(
        name="shared null leaf",
        reason=(
            "A list of one row beside a column of more rows repeats to the length of"
            " the other side, and the kernel then writes into the repeated leaves in"
            " place. Where the one list holds one value and that value is null, the"
            " repeated leaves share one buffer of nulls. The kernel then finds no"
            " values that it can write, and it panics. Two values, or a value that is"
            " not null, run. The leaf type does not matter."
        ),
        probe=(
            "pl.Series('x', [1, 2], dtype=pl.Int8)"
            " + pl.Series('y', [[None]], dtype=pl.List(pl.Int8))"
        ),
    )
    ALL: ClassVar[tuple[KernelFact, ...]] = (SHARED_NULL_LEAF,)
