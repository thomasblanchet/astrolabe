"""The kernels of the functions whose output has a length of its own, such as `unique`.

Such a function gives a number of rows that its values or its options decide:
`unique` and `mode` give the distinct values, `drop_nulls` and `drop_nans`
the values that remain, `arg_where` the places of the true values, `rle` one
row for each run of equal values, and `reshape` the rows of its first
dimension. Each kernel checks the type of its input, and gives the type of
the output from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.casts import SeriesCast
from astrolabe._rules.dtypes import DType, IntegerRange, Is, Parameter
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import (
    RunCast,
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    Refuse,
    Same,
)
from astrolabe._rules.type_functions import FunctionClosures, LengthClosures

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from astrolabe._rules.type_functions import Operand


class Dimensions:
    """Read the dimensions of `reshape` from the tree of polars, `-1` for one to infer.

    The tree writes a dimension that the call gives as its size plus one, and
    a dimension to infer as `Infer`.

    Example:
        >>> Dimensions.of(["Infer", {"Specified": 3}])
        (-1, 2)
    """

    @classmethod
    def of(cls, part: object) -> tuple[int, ...]:
        """Give the dimensions of a serialized `reshape`.

        Args:
            part: The list of dimensions, as the tree writes it.

        Returns:
            The size of each dimension, or `-1` for one to infer.

        Example:
            >>> Dimensions.of([{"Specified": 5}])
            (4,)
        """
        dimensions: list[int] = []
        for entry in part if isinstance(part, list) else []:
            size = entry.get("Specified") if isinstance(entry, dict) else None
            dimensions.append(size - 1 if isinstance(size, int) else -1)
        return tuple(dimensions)


@dataclass(frozen=True)
class Mapped:
    """Give the type that a closure of polars gives for the type of the data.

    A kernel such as `rle` builds its output from the type of its input, as
    its type function does: a struct of the run lengths and the values.

    Example:
        >>> rle = Dispatch("rle", (), Mapped(FunctionClosures().rle))
        >>> rle.output(Data(DType("Int8")))
        Struct({'len': UInt32, 'value': Int8})
    """

    closure: Callable[[DType], DType]

    def give(self, data: Data, kernel: Dispatch) -> DType:
        """Give the type of the output.

        Args:
            data: The data of the input.
            kernel: The dispatch that holds this output, which it does not read.

        Returns:
            The type that the closure gives.

        Example:
            >>> Mapped(FunctionClosures().repeat_by).give(Data(DType("Date")), LengthKernels.SAME)
            List(Date)
        """
        del kernel
        return self.closure(data.dtype)


class LengthKernels:
    """Hold the dispatch of each kernel of one input whose output has a length of its own.

    `SAME` keeps every type, as `drop_nulls`, `drop_nans`, `gather_every`,
    `shuffle`, `sample`, `unique` and `mode` do. `INDICES` gives the index
    type for every type, as `arg_unique` and `unique_counts` do. `WHERE`
    needs `Boolean` data, as `arg_where` does, and refuses other data with a
    `SchemaError`. `RLE` gives a struct of the
    run lengths and the values. `hist` counts the values of a number only.

    Example:
        >>> LengthKernels.WHERE.output(Data(DType("Boolean")))
        UInt32
    """

    SAME: ClassVar[Dispatch] = Dispatch("unique", (), Same())
    INDICES: ClassVar[Dispatch] = Dispatch("arg_unique", (), Fixed(DType.idx()))
    WHERE: ClassVar[Dispatch] = Dispatch(
        "arg_where",
        (Arm(Kinds("Boolean"), Fixed(DType.idx())),),
        Refuse(ErrorClass.SCHEMA),
    )
    RLE: ClassVar[Dispatch] = Dispatch("rle", (), Mapped(FunctionClosures().rle))

    def mapped(self, closure: Callable[[DType], DType]) -> Dispatch:
        """Give a kernel that gives the type of a closure of the data, for every type.

        Args:
            closure: The closure, as the type function of the node holds it.

        Returns:
            The dispatch.

        Example:
            >>> LengthKernels().mapped(FunctionClosures().rle).output(Data(DType("Date")))
            Struct({'len': UInt32, 'value': Date})
        """
        return Dispatch("mapped", (), Mapped(closure))

    def hist(self, *, include_breakpoint: bool, include_category: bool) -> Dispatch:
        """Give the kernel of `hist`: the counts of the bins of a number.

        Args:
            include_breakpoint: Whether the output holds the upper break of each bin.
            include_category: Whether the output holds the category of each bin.

        Returns:
            The dispatch.

        Example:
            >>> hist = LengthKernels().hist(include_breakpoint=False, include_category=False)
            >>> hist.output(Data(DType("Date")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: hist ...
        """
        counts = LengthClosures().hist(
            include_breakpoint=include_breakpoint,
            include_category=include_category,
        )
        return Dispatch("hist", (Arm(Is.primitive_numeric, Fixed(counts)),), Refuse())

    def reshape(self, dimensions: Sequence[int]) -> Reshaped:
        """Give the kernel of `reshape` with its dimensions (`Reshaped`).

        Args:
            dimensions: The dimensions of the call, `-1` for one to infer.

        Returns:
            The kernel.

        Example:
            >>> LengthKernels().reshape((-1, 2)).output([Data(DType("Int8"))], (4,))
            Array(Int8, shape=(2,))
        """
        return Reshaped(tuple(dimensions))


@dataclass(frozen=True)
class Sampled:
    """Give the type of `shuffle` and `sample`: the values of the input, in another order.

    `shuffle` reads its input only. `sample` reads a count or a fraction of
    one value. A count casts to the index type with a strict cast, and a
    struct, which stays a struct, raises a `SchemaError`. A fraction casts to
    `Float64` with a cast that turns a value that does not fit into null. A
    null count or fraction gives no row. Without replacement, a literal count
    above the rows of the input raises. Only an input of one row has rows that
    the code decides: the rows of a column are a fact of the data.

    Example:
        >>> Sampled.of("Shuffle").output([Data(DType("Date"))], (3,), ())
        Date
    """

    INDEX: ClassVar[DType] = DType.idx()
    FLOAT64: ClassVar[DType] = DType("Float64")

    sample: bool
    fraction: bool = False
    with_replacement: bool = False

    @classmethod
    def of(cls, method: object) -> Sampled:
        """Give the kernel for the method of a node, as the tree writes it.

        Args:
            method: `Shuffle`, or the options of a `Sample`.

        Returns:
            The kernel.

        Example:
            >>> Sampled.of({"Sample": {"is_fraction": True, "with_replacement": False}})
            Sampled(sample=True, fraction=True, with_replacement=False)
        """
        options = method.get("Sample") if isinstance(method, dict) else None
        if not isinstance(options, dict):
            return cls(sample=False)
        fraction = options.get("is_fraction") is True
        return cls(
            sample=True, fraction=fraction, with_replacement=options.get("with_replacement") is True
        )

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values, and of the count or the fraction.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the values.

        Raises:
            PolarsRefusal: The count or the fraction does not fit.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> count = Operand.of_value(5)
            >>> sample = Sampled(sample=True)
            >>> one = Data(DType("Int8"), bound=1)
            >>> sample.output([one, Data(DType("Int32"))], (1, 1), ("", count))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: cannot take a larger sample ...
            >>> fraction = Sampled(sample=True, fraction=True)
            >>> fraction.output([Data(DType("Int8")), Data(DType("Float32"))], (3, 1), ())
            Int8
        """
        if not self.sample:
            return inputs[0].dtype
        if len(lengths) > 1 and lengths[1] != 1:
            name = "fraction" if self.fraction else "size"
            raise PolarsRefusal(ErrorClass.COMPUTE, f"Sample {name} must be a single value.")
        literal = computed[1] if len(computed) > 1 and computed[1].literal else None
        if self.fraction:
            RunCast().of(inputs[1], self.FLOAT64)
            return inputs[0].dtype
        self._count(inputs[1], literal)
        self._larger(inputs[0], literal)
        return inputs[0].dtype

    def _larger(self, values: Data, literal: Operand | None) -> None:
        """Refuse a literal count above the rows that the code allows, with no replacement.

        Args:
            values: The data of the values.
            literal: The count where it is a literal, or `None`.

        Raises:
            PolarsRefusal: The sample is larger than the values.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> one = Data(DType("Int8"), bound=1)
            >>> Sampled(sample=True)._larger(one, Operand.of_value(2))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: cannot take a larger sample ...
        """
        bound = values.bound
        value = None if literal is None else literal.value
        larger = isinstance(value, int) and bound is not None and value > bound
        if not self.with_replacement and larger:
            message = "cannot take a larger sample than the total population"
            raise PolarsRefusal(ErrorClass.SHAPE, message)

    def _count(self, count: Data, literal: Operand | None) -> None:
        """Cast the count to the index type with a strict cast, and refuse a type that stays apart.

        Args:
            count: The data of the count.
            literal: The count where it is a literal, or `None`.

        Raises:
            PolarsRefusal: The cast fails, or gives a struct.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Sampled(sample=True)._count(Data(DType("Int8")), Operand.of_value(-1))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        cast = RunCast().of(count, self.INDEX)
        if literal is not None:
            SeriesCast(literal.dtype.materialized(), self.INDEX, strict=True).output([
                literal.value
            ])
        if cast.dtype.name != self.INDEX.name:
            message = f"invalid series dtype: expected `{self.INDEX!r}`, got `{cast.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


@dataclass(frozen=True)
class Counted:
    """Give the type of a function that reads a count of one value beside its values, as `top_k`.

    Each count must have one row. It casts to `UInt64`, or to the index type
    where `index` says so, with a strict cast, so
    the value of a literal count that does not fit raises at every run, and a
    struct, which stays a struct, raises a `SchemaError`. A null count
    raises. The output takes the type that `values` gives for the values.

    Example:
        >>> top_k = Counted((1,), Mapped(lambda dtype: dtype))
        >>> top_k.output([Data(DType("Date")), Data(DType("Int8"))], (3, 2), ())
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    UINT64: ClassVar[DType] = DType("UInt64")

    places: tuple[int, ...]
    values: Mapped
    index: bool = False

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type that the values give.

        Raises:
            PolarsRefusal: A count does not have one row, or does not cast.

        Example:
            >>> top_k = Counted((1,), Mapped(lambda dtype: dtype))
            >>> top_k.output([Data(DType("Date")), Data(DType("Int8"))], (3, 1), ())
            Date
        """
        for place in self.places:
            if place < len(lengths) and lengths[place] != 1:
                raise PolarsRefusal(ErrorClass.COMPUTE, "the count must be a single value")
            self.cast(inputs[place], self._literal(computed, place))
            if inputs[place].nulls:
                raise PolarsRefusal(ErrorClass.COMPUTE, "the count can not be null")
        return self.values.give(inputs[0], LengthKernels.SAME)

    def _literal(self, computed: Sequence[Operand], place: int) -> Operand | None:
        """Give the input at a place where the run computes it as a literal, or `None`.

        Args:
            computed: Each input as the run computes it, or none.
            place: The place of the input.

        Returns:
            The literal, or `None`.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> top_k = Counted((1,), Mapped(lambda dtype: dtype))
            >>> top_k._literal([Operand.column(DType("Int8")), Operand.of_value(2)], 1).value
            2
        """
        operand = computed[place] if place < len(computed) else None
        return operand if operand is not None and operand.literal else None

    def cast(self, count: Data, literal: Operand | None) -> None:
        """Cast a count to `UInt64` with a strict cast.

        Args:
            count: The data of the count.
            literal: The count where it is a literal, or `None`.

        Raises:
            PolarsRefusal: The cast fails, or gives a struct.

        Example:
            >>> Counted((1,), Mapped(lambda dtype: dtype)).cast(Data(DType("Int8")), None)
        """
        target = DType.idx() if self.index else self.UINT64
        cast = RunCast().of(count, target)
        if literal is not None:
            SeriesCast(literal.dtype.materialized(), target, strict=True).output([literal.value])
        if cast.dtype.name != target.name:
            message = f"invalid series dtype: expected `{target!r}`, got `{cast.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


@dataclass(frozen=True)
class Appended:
    """Give the type of `append`: the values of both inputs, cast to the planned supertype.

    Polars casts both inputs to their supertype when it converts the
    expression, at the planner types, with a cast that is not strict. Some
    casts keep their input as it is:

    - a cast to a supertype that stays unknown, as for a float literal beside a `Boolean`;
    - a cast of an integer literal with no data type to a supertype that is
      itself such a literal, which leaves the literal at the type that it takes alone;
    - a cast of a struct to a type that is not a struct;
    - no cast where polars reads the input at the supertype already, so the
      input keeps the type of its data: `sqrt` of a text plans `String` and
      computes `Float64`.

    The run then appends each later input to the first input. A later input
    must have the type of the first input, or be `Null`, or differ only in
    the inner types of lists, arrays and structs by the same rule, or be a
    `Decimal` beside a `Decimal`. Every other pair raises a `SchemaError`.

    Example:
        >>> Appended().output([Data(DType("Int32")), Data(DType("UInt32"))], DType("Int64"))
        Int64
    """

    NESTED: ClassVar[frozenset[str]] = frozenset({"List", "Array"})

    def output(
        self,
        inputs: Sequence[Data],
        planned: DType,
        computed: Sequence[Operand] = (),
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where a cast has no kernel.

        Args:
            inputs: The data of the two inputs.
            planned: The supertype at the planner types.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the first input after its cast.

        Raises:
            PolarsRefusal: A later input does not append to the first input.

        Example:
            >>> Appended().output([Data(DType("Int8")), Data(DType("Int32"))], DType("Int16"))
            Int16
            >>> unknown = DType("Unknown", ("any",))
            >>> Appended().output([Data(DType("Time")), Data(DType("Int32"))], unknown)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: type Int32 is incompatible ...
            >>> from astrolabe._rules.type_functions import Operand
            >>> big, one = Operand.of_value(2**40), Operand.of_value(1)
            >>> inputs = [Data(DType("Int64")), Data(DType("Int32"))]
            >>> Appended().output(inputs, DType.dynamic_int(2**40), [big, one])
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: type Int32 is incompatible ...
        """
        types = [self._cast(data, planned, computed, place) for place, data in enumerate(inputs)]
        first = types[0]
        for other in types[1:]:
            if not self._appends(first, other):
                message = f"type {other!r} is incompatible with expected type {first!r}"
                raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return first

    def _cast(
        self,
        data: Data,
        planned: DType,
        computed: Sequence[Operand],
        place: int,
    ) -> DType:
        """Give the type of an input after its cast to the supertype.

        Args:
            data: The data of the input.
            planned: The supertype at the planner types.
            computed: Each input as the run computes it, or none.
            place: The place of the input.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> one = Operand.of_value(1)
            >>> Appended()._cast(Data(DType("Int32")), DType.dynamic_int(2**40), [one], 0)
            Int32
            >>> Appended()._cast(Data(DType("Null")), DType.dynamic_int(2**40), [], 0)
            Int64
            >>> text = Operand.column(DType("String"))
            >>> Appended()._cast(Data(DType("Float64")), DType("String"), [text], 0)
            Float64
        """
        target = planned.materialized()
        read = computed[place].dtype if place < len(computed) else None
        literal = read is not None and Is.dynamic_int(read)
        if Is.unknown(target) or (literal and Is.dynamic_int(planned)):
            return data.dtype
        if read is not None and read.same(target):
            return data.dtype
        return RunCast().of(data, target).dtype

    def _appends(self, first: DType, other: DType) -> bool:
        """Say whether a later input appends to the first input.

        Args:
            first: The type of the first input.
            other: The type of the later input.

        Returns:
            Whether it appends.

        Example:
            >>> listed = DType("List", (DType("Int8"),))
            >>> Appended()._appends(listed, DType("List", (DType("Null"),)))
            True
            >>> struct = DType("Struct", ((("a", DType("Int8")),),))
            >>> Appended()._appends(DType("Int8"), struct)
            False
            >>> Appended()._appends(struct, DType("Struct", ((("a", DType("Null")),),)))
            True
            >>> Appended()._appends(DType("Decimal", (10, 2)), DType("Decimal", (5, 1)))
            True
        """
        if other.same(first) or other.name == "Null":
            return True
        if first.name != other.name:
            return False
        if first.name in self.NESTED:
            width = first.name == "List" or first.args[1] == other.args[1]
            return width and self._appends(first.inner(), other.inner())
        if first.name == "Struct":
            pairs = zip(first.fields(), other.fields(), strict=False)
            fitting = all(self._appends(mine[1], theirs[1]) for mine, theirs in pairs)
            return len(first.fields()) == len(other.fields()) and fitting
        return first.name == "Decimal"


@dataclass(frozen=True)
class Reshaped:
    """Give the type of `reshape`: the leaf values of the input, in an array for each later size.

    The kernel counts the leaf values of its input: one for each row of flat
    data, and the width of each array for an array. The count must equal the
    product of the dimensions, or be a multiple of it where the call infers
    one dimension. The count of the values of a list is a fact of the data.

    Example:
        >>> Reshaped((2, 1)).output([Data(DType("Int8"))], (3,))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot reshape ...
    """

    dimensions: tuple[int, ...]

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input.
            lengths: The rows of the input, or none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The count of the leaf values does not fit the dimensions.

        Example:
            >>> Reshaped((2, 1)).output([Data(DType("Int8"))], (2,))
            Array(Int8, shape=(1,))
        """
        dtype = FunctionClosures().reshape(self.dimensions)(inputs[0].dtype)
        count = self._leaves(inputs[0].dtype, lengths[0]) if lengths else None
        if count is not None and not self._fits(count):
            message = f"cannot reshape array of size {count} into shape {self.dimensions}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return dtype

    def _leaves(self, dtype: DType, rows: int) -> int | None:
        """Give the count of the leaf values of a series, or `None` where its values decide it.

        Args:
            dtype: The type of the series.
            rows: The rows of the series.

        Returns:
            The count.

        Example:
            >>> Reshaped((2,))._leaves(DType("Array", (DType("Int8"), 3)), 2)
            6
            >>> Reshaped((2,))._leaves(DType("List", (DType("Int8"),)), 2) is None
            True
        """
        if dtype.name == "List":
            return None
        if dtype.name == "Array":
            width = dtype.args[1]
            return self._leaves(dtype.inner(), rows * width) if isinstance(width, int) else None
        return rows

    def _fits(self, count: int) -> bool:
        """Say whether a count of leaf values fits the dimensions.

        Args:
            count: The count of the leaf values.

        Returns:
            Whether it fits.

        Example:
            >>> Reshaped((-1, 2))._fits(4), Reshaped((-1, 2))._fits(3), Reshaped((0,))._fits(0)
            (True, False, False)
        """
        total = math.prod(size for size in self.dimensions if size >= 0)
        if total == 0:
            return False
        if -1 in self.dimensions:
            return count % total == 0
        return count == total


class Repeated:
    """Give the type of `pl.repeat`: the type of the value, with a count of an integer type.

    The count must have an integer type, or polars raises a `SchemaError`.
    Polars reads its first value as a size, so a null or a negative literal
    raises a `ComputeError`.

    Example:
        >>> Repeated().output([Data(DType("Date")), Data(DType("Float32"))], (1, 1), ())
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the value and of the count.
            lengths: The rows of each input, which the kernel does not read.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the value.

        Raises:
            PolarsRefusal: The count is not an integer, or its first value is
                not a size.

        Example:
            >>> Repeated().output([Data(DType("Date")), Data(DType("Int8"))], (1, 1), ())
            Date
        """
        del lengths
        count = inputs[1]
        if not Is.integer(count.dtype):
            message = f"expected expression of dtype 'integer', got '{count.dtype!r}'"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        if count.nulls or self._negative(computed):
            raise PolarsRefusal(ErrorClass.COMPUTE, "could not parse the count as a size")
        return inputs[0].dtype

    def _negative(self, computed: Sequence[Operand]) -> bool:
        """Say whether the count is a literal below zero.

        Args:
            computed: Each input as the run computes it, or none.

        Returns:
            Whether it is one.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Repeated()._negative([Operand.of_value(1), Operand.of_value(-1)])
            True
        """
        count = computed[1] if len(computed) > 1 else None
        if count is None or not count.literal or not isinstance(count.value, int):
            return False
        return count.value < 0


@dataclass(frozen=True)
class Extended:
    """Give the type of `extend_constant`: the type of the input, after the value and the count.

    The value and the count must hold one row each. The count casts to
    `UInt64` with a strict cast (`Counted`), and must not be null. An empty
    input takes the value as it is. Otherwise polars reads the value as one
    value with its own type, and panics for a category that is not null,
    unless the input is one value equal to it. The value then casts to the
    type of the input, and a cast that polars does not have raises.

    Example:
        >>> inputs = [Data(DType("Date")), Data(DType("Date")), Data(DType("Int8"))]
        >>> Extended().output(inputs, (3, 1, 1), ())
        Date
    """

    COUNT: ClassVar[Counted] = Counted((), Mapped(LengthClosures().same))
    CATEGORIES: ClassVar[frozenset[str]] = frozenset({"Categorical", "Enum"})
    UNBUILT: ClassVar[str] = "unexpected value while building Series of type {!r}"

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the count or the value does not cast
        (`Counted`, `RunCast`).

        Args:
            inputs: The data of the input, of the value and of the count.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The value or the count has more than one row, or the
                count is null.

        Example:
            >>> inputs = [Data(DType("Date")), Data(DType("Date")), Data(DType("Int8"))]
            >>> Extended().output(inputs, (3, 2, 1), ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
            >>> date = Data(DType("Date"))
            >>> Extended().output([date, date, Data(DType("Int8"))], (0, 1, 1), ())
            Date
        """
        if (  # pragma: no cover -- the scalar inputs refuse more rows
            lengths and (lengths[1] != 1 or lengths[2] != 1)
        ):
            raise PolarsRefusal(ErrorClass.COMPUTE, "value and n should have unit length.")
        literal = computed[2] if len(computed) > 2 and computed[2].literal else None
        self.COUNT.cast(inputs[2], literal)
        if inputs[2].nulls:
            raise PolarsRefusal(ErrorClass.COMPUTE, "n can not be None for extend_constant.")
        if lengths and lengths[0] == 0:
            self._built(inputs, computed)
            return inputs[0].dtype
        self._category(inputs, lengths, computed)
        RunCast().of(inputs[1], inputs[0].dtype)
        return inputs[0].dtype

    def _built(self, inputs: Sequence[Data], computed: Sequence[Operand]) -> None:
        """Refuse a value that polars keeps in an empty input of another type, and cannot read.

        Over an input of no row, polars keeps the value as it is, in a column
        of the type of the input. It panics where it builds that column, at
        the first read of a `select` and in the lists of `agg`, unless its
        builder takes the value (`ValueBuild`).

        Args:
            inputs: The data of the input, of the value and of the count.
            computed: Each input as the run computes it, or none.

        Raises:
            PolarsRefusal: The builder of the type of the input refuses the value.

        Example:
            >>> inputs = [Data(DType("Int8")), Data(DType("Binary")), Data(DType("Int8"))]
            >>> Extended()._built(inputs, ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: unexpected value ... of type Int8
        """
        value = inputs[1]
        literal = computed[1].value if len(computed) > 1 and computed[1].literal else None
        target = inputs[0].dtype
        if not value.nulls and ValueBuild().refuses(value.dtype, literal, target):
            raise PolarsRefusal(Stop.PANICS, self.UNBUILT.format(target))

    def _category(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> None:
        """Refuse a category value that polars reads with its own type, where it panics.

        An input of one value keeps its rows where it equals the value, with
        no read of the value. The two are equal only where they have one type,
        and their values decide it, unless both are literals.

        Args:
            inputs: The data of the input, of the value and of the count.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Raises:
            PolarsRefusal: The value is a category that is not null, and the
                input is not one value that can equal it.

        Example:
            >>> enum = Data(DType("Enum", (("a", "b"),)))
            >>> Extended()._category([enum, enum, Data(DType("Int8"))], (1, 1, 1), ()) is None
            True
            >>> Extended()._category([enum, enum, Data(DType("Int8"))], (3, 1, 1), ())
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: can not get dtype of Enum AnyValue
        """
        value = inputs[1]
        if value.dtype.name not in self.CATEGORIES or value.nulls:
            return
        one_value = inputs[0].one_row or (bool(lengths) and lengths[0] == 1)
        literals = len(computed) > 1 and computed[0].literal and computed[1].literal
        differ = literals and computed[0].value != computed[1].value
        if one_value and inputs[0].dtype.same(value.dtype) and not differ:
            return
        raise PolarsRefusal(Stop.PANICS, f"can not get dtype of {value.dtype.name} AnyValue")


class ValueBuild:
    """Say whether polars refuses to build a series of one type from one value of another type.

    Some operations keep a value of its own type in a column of another type,
    with no cast, and polars builds the column later. Its builder takes a value
    of the kind of the column, and a few others: an integer of another width
    that fits, a narrower float, a text or a category for a category, a
    decimal of no larger scale, a list for a list, and a list or an array for
    an array or a struct. It refuses any other value. The class reads no inner
    value of a nested value, so it refuses only a value of another kind.

    Example:
        >>> ValueBuild().refuses(DType("Float64"), 1.5, DType("Float32"))
        True
        >>> ValueBuild().refuses(DType("String"), "a", DType("Categorical"))
        False
    """

    OPEN: ClassVar[frozenset[str]] = frozenset({"Null", "Object", "Extension"})
    TAKEN: ClassVar[dict[str, frozenset[str]]] = {
        "Float32": frozenset({"Float32", "Float16"}),
        "Float64": frozenset({"Float64", "Float32", "Float16"}),
        "Categorical": frozenset({"String", "Categorical", "Enum"}),
        "Enum": frozenset({"String", "Categorical", "Enum"}),
        "Array": frozenset({"List", "Array"}),
        "Struct": frozenset({"Struct", "List", "Array"}),
    }
    UNITS: ClassVar[frozenset[str]] = frozenset({"Datetime", "Duration"})

    def refuses(self, value: DType, literal: object, target: DType) -> bool:
        """Say whether the builder of a type refuses a value that is not null.

        Args:
            value: The type of the value.
            literal: The Python value, or `None` where it is not a literal.
            target: The type of the series.

        Returns:
            Whether it refuses it.

        Example:
            >>> ValueBuild().refuses(DType("Int64"), 300, DType("Int8"))
            True
            >>> micro, nano = DType("Datetime", ("us", None)), DType("Datetime", ("ns", None))
            >>> ValueBuild().refuses(micro, None, nano)
            True
            >>> ValueBuild().refuses(DType("Decimal", (5, 2)), None, DType("Decimal", (10, 3)))
            False
        """
        if target.name in self.OPEN or value.name == "Null":
            return False
        if Is.integer(target):
            return not Is.integer(value) or self._too_large(literal, target)
        if target.name in self.UNITS:
            return value.name != target.name or value.args[0] != target.args[0]
        if target.name == "Decimal":
            scale = Parameter().integer(target, 1)
            return value.name != "Decimal" or Parameter().integer(value, 1) > scale
        return value.name not in self.TAKEN.get(target.name, frozenset({target.name}))

    def _too_large(self, literal: object, target: DType) -> bool:
        """Say whether a literal integer does not fit an integer type.

        Args:
            literal: The Python value, or `None`.
            target: The integer type.

        Returns:
            Whether the literal is an integer that the type does not hold.

        Example:
            >>> small = DType("UInt8")
            >>> ValueBuild()._too_large(300, small), ValueBuild()._too_large(None, small)
            (True, False)
        """
        if not isinstance(literal, int) or isinstance(literal, bool):
            return False
        return not IntegerRange().holds(target.name, literal)


@dataclass(frozen=True)
class RepeatedBy:
    """Give the type of `repeat_by`: a list of the values, with a count of each from a second input.

    The counts cast to the index type with a strict cast, so a literal count
    that does not fit raises at every run, and a struct, which stays a struct,
    raises a `SchemaError`. The kernel then checks the lengths: the values and
    the counts must have one length, or one of them one row.

    Example:
        >>> RepeatedBy().output([Data(DType("Date")), Data(DType("Int8"))], (), ())
        List(Date)
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the counts do not cast.

        Args:
            inputs: The data of the values and of the counts.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            A list of the type of the values.

        Raises:
            PolarsRefusal: The values and the counts have other lengths, and
                neither has one row.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> counts = ("", Operand.of_value(-1))
            >>> RepeatedBy().output([Data(DType("Int8")), Data(DType("Int32"))], (), counts)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        literal = computed[1] if len(computed) > 1 and computed[1].literal else None
        Counted((), Mapped(LengthClosures().same), index=True).cast(inputs[1], literal)
        if len({length for length in lengths if length != 1}) > 1:
            message = "repeat_by argument and the Series should have equal length"
            raise PolarsRefusal(ErrorClass.SHAPE, message)
        return FunctionClosures().repeat_by(inputs[0].dtype)


@dataclass(frozen=True)
class Searched:
    """Give the type of `search_sorted`: the index type, over two inputs of one searchable type.

    The values and the elements to find must have one type, as the cast pass
    makes them where they join. A categorical type searches its text.
    Text, binary data, `Boolean`, a number and a nested type can be searched.

    Example:
        >>> Searched().output([Data(DType("Null")), Data(DType("Null"))], ())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
    """

    SEARCHABLE: ClassVar[frozenset[str]] = frozenset({"String", "Binary", "Boolean"})

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the elements to find.
            lengths: The rows of each input, which the kernel does not read.

        Returns:
            The index type.

        Raises:
            PolarsRefusal: The two types differ, or the type cannot be searched.

        Example:
            >>> Searched().output([Data(DType("Date")), Data(DType("Date"))], ())
            UInt32
        """
        del lengths
        values, elements = inputs[0].dtype, inputs[1].dtype
        if not values.same(elements):
            message = f"`search_sorted` operation not supported for {values!r} and {elements!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        physical = DType("String") if values.name in {"Categorical", "Enum"} else values.physical()
        searchable = physical.name in self.SEARCHABLE or Is.primitive_numeric(physical)
        if not searchable and not Is.nested(physical):
            message = f"`search_sorted` operation not supported for {values!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return DType.idx()


@dataclass(frozen=True)
class TopKBy:
    """Give the type of `top_k_by`: the values, after the checks of the count and of the keys.

    The count must hold one value that casts to the index type (`Counted`).
    Each key must have the rows of the values.

    Example:
        >>> keys = [Data(DType("Date")), Data(DType("Int8")), Data(DType("Int8"))]
        >>> TopKBy().output(keys, (3, 1, 2), ())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values, of the count and of each key.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the values.

        Raises:
            PolarsRefusal: The count does not fit, or a key has other rows.

        Example:
            >>> keys = [Data(DType("Date")), Data(DType("Int8")), Data(DType("Int8"))]
            >>> TopKBy().output(keys, (3, 1, 3), ())
            Date
        """
        counted = Counted((1,), Mapped(LengthClosures().same), index=True)
        dtype = counted.output(inputs, lengths, computed)
        for length in lengths[2:]:
            if length != lengths[0]:
                message = "the `by` column should have the length of the source in `top_k`"
                raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return dtype
