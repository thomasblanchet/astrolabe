"""The kernels of the `str`, `bin` and `cat` namespaces, and of `concat_str` and `format`.

A function of these namespaces reads each input as one kind of series before
it does any work: the text of `str` must be `String` data, and the bytes of
`bin` must be `Binary` data. Another kind raises a `SchemaError`, whatever the
planner type of the input was. Some functions also cast an input while they
run, for example the count of `str.head` to `Int64`, and some check that
their inputs have one length, or that one of them has one row. Polars makes
these checks in a fixed order for each function, and the first check that
fails decides the error. So a kernel of several inputs here is a list of
steps in that order, and the type that the function gives (`TextKernel`).

A strict cast of a literal input meets the value of the literal at every
run, because the code holds that value. So a step that casts an input reads
the value of a literal. The value of a column is a fact of the data.

A function of `cat` reads the categories of a `Categorical` or an `Enum`
input. Most of them take the categories with no check, and panic where the
input has none.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from astrolabe._rules.casts import SeriesCast, TextParse
from astrolabe._rules.dtypes import DType, Is, Parameter
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop, Undecided, UndecidedError
from astrolabe._rules.kernels.dispatch import (
    Arm,
    Data,
    Dispatch,
    Fixed,
    Kinds,
    Refuse,
    RunCast,
    ToPhysical,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class KernelCall:
    """Hold the facts of one call of a kernel besides the types of its data.

    `lengths` gives the number of rows of each input, and is empty where the
    code that calls the kernel does not know them. `computed` gives each input
    as the run computes it: a literal with its value, or a column. It is empty
    where it is not known. `sources` gives the data of each input as it
    reaches the kernel, before the kernel casts it.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> KernelCall((5, 1), (Operand.column(DType("String")), Operand.of_value(2))).value(1)
        2
    """

    lengths: tuple[int, ...] = ()
    computed: tuple[Operand, ...] = ()
    sources: tuple[Data, ...] = ()

    def literal(self, index: int) -> Operand | None:
        """Give the input at a place where it is a literal of the code.

        Args:
            index: The place of the input.

        Returns:
            The literal, or `None` where the input is a column or is not known.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> KernelCall((), (Operand.column(DType("String")),)).literal(0) is None
            True
        """
        if index >= len(self.computed) or not self.computed[index].literal:
            return None
        return self.computed[index]

    def value(self, index: int) -> object:
        """Give the value of a literal input, or `None` where the input is not a literal.

        Args:
            index: The place of the input.

        Returns:
            The value.

        Example:
            >>> KernelCall().value(0) is None
            True
        """
        literal = self.literal(index)
        return None if literal is None else literal.value


class TextStep(Protocol):
    """Check the inputs of a kernel at one step, or change one of them, as polars does there."""

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs after the step, or raise `PolarsRefusal` where the step fails.

        Args:
            inputs: The data of each input before the step.
            call: The rows and the values of the inputs at the call of the kernel.
        """


@dataclass(frozen=True)
class Reads:
    """Read one input as a series of one kind, as `s.str()` or `s.binary()` of polars does.

    Polars reads a series of another kind as an error of the schema. The
    planner type of the input does not matter: the type of the data decides.

    Example:
        >>> Reads(0, "String").apply([Data(DType("String"))], KernelCall())
        [Data(dtype=String, nulls=False, one_row=False)]
        >>> Reads(0, "String").apply([Data(DType("Null"))], KernelCall())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
    """

    index: int
    kind: str

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the input is of the kind.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, which this step does not read.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The input is not of the kind.

        Example:
            >>> data = [Data(DType("Int8")), Data(DType("Binary"))]
            >>> Reads(1, "Binary").apply(data, KernelCall())[1]
            Data(dtype=Binary, nulls=False, one_row=False)
        """
        del call
        dtype = inputs[self.index].dtype
        if dtype.name != self.kind:
            message = f"invalid series dtype: expected `{self.kind}`, got `{dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return inputs


@dataclass(frozen=True)
class ReadsUnlessNull:
    """Read one input as `String` data, unless its type is `Null`, as `str.strip_chars` does.

    `str.strip_chars` takes an input of the type `Null` as no characters at
    all, and strips white space. An input of any other type must be `String`
    data.

    Example:
        >>> data = [Data(DType("String")), Data(DType("Null"))]
        >>> ReadsUnlessNull(1).apply(data, KernelCall())[1].dtype
        Null
    """

    index: int

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the input is `Null` or `String` data.

        It raises `PolarsRefusal` where the input is of another type (`Reads`).

        Args:
            inputs: The data of each input.
            call: The call of the kernel.

        Returns:
            The same inputs.

        Example:
            >>> data = [Data(DType("String")), Data(DType("Int8"))]
            >>> ReadsUnlessNull(1).apply(data, KernelCall())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        if inputs[self.index].dtype.name == "Null":
            return inputs
        return Reads(self.index, "String").apply(inputs, call)


@dataclass(frozen=True)
class CastsTo:
    """Cast one input to a type while the kernel runs, as `column.cast(...)` of polars does.

    The cast follows the cast of a series at run time (`RunCast`): a series
    of only nulls casts to any type, and a pair with no cast kernel raises
    the error of the cast. A strict cast also raises on a value that does not
    fit. The value of a literal is in the code, so the step checks it
    (`SeriesCast`). The value of a column is a fact of the data. The cast of a
    `Struct` to a number gives a `Struct`, so a later read of the input as the
    target raises (`Reads`).

    Example:
        >>> data = [Data(DType("String")), Data(DType("UInt8"))]
        >>> CastsTo(1, DType("Int64")).apply(data, KernelCall())[1]
        Data(dtype=Int64, nulls=False, one_row=False)
    """

    index: int
    target: DType
    strict: bool = False

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs with the one input cast.

        It raises `PolarsRefusal` where the cast has no kernel, or where a
        strict cast loses the value of a literal.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the value of a literal input.

        Returns:
            The inputs after the cast.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> call = KernelCall((3, 1), (Operand.column(DType("String")), Operand.of_value(-1)))
            >>> step = CastsTo(1, DType("UInt64"), strict=True)
            >>> step.apply([Data(DType("String")), Data(DType("Int32"))], call)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        data = inputs[self.index]
        value = call.value(self.index)
        if self.strict and value is not None and not data.nulls:
            SeriesCast(data.dtype, self.target, strict=True).output([value])
        cast = list(inputs)
        cast[self.index] = RunCast().of(data, self.target)
        return cast


@dataclass(frozen=True)
class EachCastsTo:
    """Cast every input to a type in order, as `concat_str` and `format` cast to text.

    `format` stops at the first input that holds only nulls, because its
    output is then only nulls. It casts none of the inputs after that one.

    Example:
        >>> listed = Data(DType("List", (DType("Int8"),)))
        >>> step = EachCastsTo(DType("String"), stops_at_nulls=True)
        >>> step.apply([Data(DType("Null")), listed], KernelCall())[1].dtype
        List(Int8)
    """

    target: DType
    stops_at_nulls: bool = False

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs after the casts.

        It raises `PolarsRefusal` where a cast has no kernel.

        Args:
            inputs: The data of each input.
            call: The call of the kernel.

        Returns:
            The inputs after the casts.

        Example:
            >>> EachCastsTo(DType("String")).apply([Data(DType("Int8"))], KernelCall())
            [Data(dtype=String, nulls=False, one_row=False)]
        """
        cast = list(inputs)
        for index, data in enumerate(inputs):
            if self.stops_at_nulls and data.nulls:
                return cast
            cast = CastsTo(index, self.target).apply(cast, call)
        return cast


@dataclass(frozen=True)
class SameLengths:
    """Check that the inputs have one length, where an input of one row spreads to the others.

    Polars checks the lengths with an error of its own class, which depends on
    the function. The step reads only the lengths that the code that calls the
    kernel knows.

    Example:
        >>> SameLengths(ErrorClass.SHAPE).apply([], KernelCall((3, 1, 3)))
        []
        >>> SameLengths(ErrorClass.COMPUTE).apply([], KernelCall((3, 2)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    error: ErrorClass

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where their lengths agree.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the rows of each input.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: Two inputs have other lengths, and neither has one row.

        Example:
            >>> SameLengths(ErrorClass.SHAPE).apply([], KernelCall())
            []
        """
        others = {length for length in call.lengths if length != 1}
        if len(others) > 1:
            message = f"the inputs have the lengths {list(call.lengths)}"
            raise PolarsRefusal(self.error, message)
        return inputs


@dataclass(frozen=True)
class OneRowPatterns:
    """Refuse a list of patterns for each row, which `str.contains_any` and `str.replace_many` lack.

    The cast pass implodes a flat column of patterns into one list, so only
    an input that is a list already can hold more than one row here.

    Example:
        >>> OneRowPatterns("str.contains_any", (1,)).apply([], KernelCall((3, 1)))
        []
        >>> OneRowPatterns("str.contains_any", (1,)).apply([], KernelCall((3, 3)))
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: not yet implemented: ...
    """

    name: str
    places: tuple[int, ...]

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where each list of patterns has one row.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the rows of each input.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: A list of patterns has more than one row.

        Example:
            >>> OneRowPatterns("str.replace_many", (1, 2)).apply([], KernelCall())
            []
        """
        rows = [call.lengths[place] for place in self.places if place < len(call.lengths)]
        if any(row != 1 for row in rows):
            message = f"not yet implemented: `{self.name}` with a pattern per row"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return inputs


@dataclass(frozen=True)
class PatternLengths:
    """Check the length of a pattern against the text: one row, or the length of the text.

    `str.extract_all` and `str.count_matches` read a pattern of one row as
    one pattern for every text. A pattern of another length must have one
    row for each text, and a text of one row does not spread to it.

    Example:
        >>> PatternLengths().apply([], KernelCall((1, 1)))
        []
        >>> PatternLengths().apply([], KernelCall((1, 3)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the lengths agree.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the rows of the text and of the pattern.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The pattern has more than one row, and not one for each text.

        Example:
            >>> PatternLengths().apply([], KernelCall((3, 3)))
            []
        """
        lengths = call.lengths
        if len(lengths) < 2 or lengths[1] == 1 or lengths[0] == lengths[1]:
            return inputs
        message = f"pattern's length: {lengths[1]} does not match that of the text: {lengths[0]}"
        raise PolarsRefusal(ErrorClass.COMPUTE, message)


@dataclass(frozen=True)
class SplitLengths:
    """Check the lengths of `str.split_exact` and `str.splitn`: the pattern covers the text.

    A pattern of one row splits every text. A pattern of another length splits
    the text row by row, and the struct that the function builds needs a part
    for each text. So the pattern needs at least as many rows as the text.
    The output has the rows of the text.

    Example:
        >>> SplitLengths().apply([], KernelCall((2, 3)))
        []
        >>> SplitLengths().apply([], KernelCall((3, 2)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
    """

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the pattern covers the text.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the rows of the text and of the pattern.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The pattern has more than one row, and fewer rows than the text.

        Example:
            >>> SplitLengths().apply([], KernelCall((3, 1)))
            []
        """
        lengths = call.lengths
        if len(lengths) < 2 or lengths[1] == 1 or lengths[1] >= lengths[0]:
            return inputs
        message = f"expected struct fields to have given length {lengths[0]}, got {lengths[1]}"
        raise PolarsRefusal(ErrorClass.SHAPE, message)


@dataclass(frozen=True)
class ReplaceLengths:
    """Check the lengths of the pattern and of the value of `str.replace` against the text.

    The pattern must have one row: polars does not replace with a pattern
    for each row. The value has one row, or one row for each text, and a
    text of one row does not spread to it. A replace of a count above one
    takes a value of one row only. A count of zero with a value of more rows
    returns the text with no check.

    Example:
        >>> texts = [Data(DType("String"))] * 3
        >>> ReplaceLengths(1).apply(texts, KernelCall((3, 1, 3))) == texts
        True
        >>> ReplaceLengths(-1).apply(texts, KernelCall((3, 2, 1)))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: ...
    """

    count: int

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the lengths agree.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, with the rows of the text, the pattern and the value.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The pattern has more than one row, or the value has
                more than one row and not one for each text.

        Example:
            >>> ReplaceLengths(0).apply([], KernelCall((3, 1, 2)))
            []
            >>> texts = [Data(DType("String"))] * 3
            >>> ReplaceLengths(0).apply(texts, KernelCall()) == texts
            True
            >>> ReplaceLengths(2).apply(texts, KernelCall((3, 1, 3)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: multivalue replacement ...
            >>> ReplaceLengths(1).apply(texts, KernelCall((3, 1, 2)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: replacement value length (2) ...
        """
        if len(call.lengths) < 3:  # pragma: no cover -- the run passes the rows of each input
            return inputs
        text, pattern, value = call.lengths[:3]
        if pattern != 1:
            message = "dynamic pattern length in 'str.replace' expressions is not supported yet"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if value == 1:
            self._nulls(inputs, value_read=self.count < 0)
            return inputs
        if self.count > 1:
            message = "multivalue replacement with 'n > 1' not yet supported"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if self.count == 0:
            return inputs
        self._nulls(inputs, value_read=False)
        if value != text:
            message = f"replacement value length ({value}) does not match string column length"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return inputs

    def _nulls(self, inputs: list[Data], *, value_read: bool) -> None:
        """Refuse a null pattern of one row, and a null value of one row where polars reads it.

        Polars reads the pattern first. A replace of every match reads the
        value of one row next, and a replace of a count keeps the text where
        that value is null.

        Args:
            inputs: The data of the text, the pattern and the value.
            value_read: Whether the replace reads a value of one row.

        Raises:
            PolarsRefusal: The pattern or the value that polars reads is null.

        Example:
            >>> null = Data(DType("String"), nulls=True)
            >>> ReplaceLengths(-1)._nulls([null, Data(DType("String")), null], value_read=True)
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: value cannot be 'null' ...
        """
        if inputs[1].nulls:
            message = "pattern cannot be 'null' in 'replace' expression"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        if value_read and inputs[2].nulls:
            message = "value cannot be 'null' in 'replace' expression"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)


@dataclass(frozen=True)
class IntegerTarget:
    """Check the target of `str.to_integer`, which parses text into an integer type only.

    Example:
        >>> IntegerTarget(DType("UInt8")).apply([], KernelCall())
        []
        >>> IntegerTarget(DType("Float32")).apply([], KernelCall())  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: Invalid dtype Float32
    """

    target: DType

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the target is an integer type.

        Args:
            inputs: The data of each input.
            call: The call of the kernel, which this step does not read.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The target is not an integer type.

        Example:
            >>> IntegerTarget(DType("Int128")).apply([Data(DType("String"))], KernelCall())[0].dtype
            String
        """
        del call
        if not Is.integer(self.target):
            message = f"Invalid dtype {self.target!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return inputs


@dataclass(frozen=True)
class BaseRange:
    """Check the base of `str.to_integer`, which parses each text in a base from 2 to 36.

    The kernel checks the base of each text that is not null, after the cast
    of the base to `UInt32`. A literal base outside the range raises at every
    run where the text holds a value. So does a base of a type whose values
    never fall in the range: a `Boolean` gives 0 or 1, and an `Enum` of at
    most two categories gives the index 0 or 1 of its category. A
    `Categorical` gives the id that the process gave its category, so the
    rules cannot answer. The base of any other column is a fact of the data.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> computed = (Operand.column(DType("String")), Operand.of_value(1.5))
        >>> call = KernelCall((3, 1), computed, (Data(DType("String")), Data(DType("Float64"))))
        >>> BaseRange().apply([Data(DType("String")), Data(DType("UInt32"))], call)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: `to_integer` called with ... base '1'
    """

    LOWEST: ClassVar[int] = 2
    HIGHEST: ClassVar[int] = 36

    def apply(self, inputs: list[Data], call: KernelCall) -> list[Data]:
        """Give the inputs unchanged where the base fits, or where it is a fact of the data.

        Args:
            inputs: The data of the text and of the base.
            call: The call of the kernel, with the value of a literal base and
                the type of the base before its cast.

        Returns:
            The same inputs.

        Raises:
            PolarsRefusal: The text and the base hold values, and the base is
                outside the range at every run.
            UndecidedError: The base is a `Categorical` that holds values.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> sources = (Data(DType("String")), Data(DType("Categorical")))
            >>> call = KernelCall((3, 3), (), sources)
            >>> BaseRange().apply([Data(DType("String")), Data(DType("UInt32"))], call)
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.UndecidedError: process: the ids of a Categorical base
        """
        if inputs[0].nulls or inputs[1].nulls:
            return inputs
        source = call.sources[1].dtype if len(call.sources) > 1 else inputs[1].dtype
        if source.name == "Categorical":
            raise UndecidedError(Undecided.PROCESS, "the ids of a Categorical base")
        base = self._base(call.value(1))
        if base is None and not self._never_fits(source):
            return inputs
        if base is None or not self.LOWEST <= base <= self.HIGHEST:
            message = f"`to_integer` called with invalid base '{base}'"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)
        return inputs

    def _never_fits(self, source: DType) -> bool:
        """Say whether every value of a type of base falls outside the range.

        Args:
            source: The type of the base before its cast.

        Returns:
            Whether it does: a `Boolean`, a `Decimal` with no digit before its point, or an
            `Enum` of at most two categories.

        Example:
            >>> BaseRange()._never_fits(DType("Enum", (("a", "b", "c"),)))
            False
        """
        if source.name == "Boolean":
            return True
        if source.name == "Decimal":
            precision, scale = source.args
            return precision == scale
        return source.name == "Enum" and len(Parameter().categories(source)) < self.LOWEST + 1

    def _base(self, value: object) -> int | None:
        """Give the whole number that a literal base holds after its cast to `UInt32`.

        Args:
            value: The value of the literal, or `None`.

        Returns:
            The number, or `None` where the base is not a number.

        Example:
            >>> BaseRange()._base(True), BaseRange()._base(2.9), BaseRange()._base("a") is None
            (1, 2, True)
            >>> BaseRange()._base("16")
            16
        """
        if isinstance(value, bool | int):
            return int(value)
        if isinstance(value, float) and math.isfinite(value):
            return math.trunc(value)
        if isinstance(value, str):
            return TextParse().integer(value, signed=False)
        return None


@dataclass(frozen=True)
class TextKernel:
    """Give the type of a kernel of several inputs that runs its steps in order.

    Each step checks the inputs or casts one of them (`TextStep`), in the order
    of polars. The kernel then gives one data type, because each function of
    these namespaces gives one type once its inputs pass its steps.

    Example:
        >>> contains = TextKernel(
        ...     "str.contains",
        ...     (SameLengths(ErrorClass.SHAPE), Reads(0, "String"), Reads(1, "String")),
        ...     DType("Boolean"),
        ... )
        >>> contains.output([Data(DType("String")), Data(DType("String"))], [3, 1], [])
        Boolean
        >>> contains.output([Data(DType("String")), Data(DType("Int8"))], [3, 2], [])
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ShapeError: ...
    """

    name: str
    steps: tuple[TextStep, ...]
    gives: DType

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` at the first step that fails.

        Args:
            inputs: The data of each input.
            lengths: The number of rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none where it is not known.

        Returns:
            The type.

        Example:
            >>> size = TextKernel("bin.size", (Reads(0, "Binary"),), DType("UInt32"))
            >>> size.output([Data(DType("Binary"))], [], [])
            UInt32
        """
        call = KernelCall(tuple(lengths), tuple(computed), tuple(inputs))
        data = list(inputs)
        for step in self.steps:
            data = step.apply(data, call)
        return self.gives


class TextKernels:
    """Build the kernels of the functions of the `str`, `bin` and `cat` namespaces.

    A function of one input dispatches on its data type (`Dispatch`): the
    kind that it reads gives its type, and every other type fails. A function
    of several inputs runs its steps in order (`TextKernel`). The kernels
    that read no option of their node are fixed here. The others are built
    from the option.

    Example:
        >>> TextKernels.LEN_CHARS.output(Data(DType("String")))
        UInt32
        >>> TextKernels.LEN_CHARS.output(Data(DType("Categorical")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
    """

    SHAPE: ClassVar[SameLengths] = SameLengths(ErrorClass.SHAPE)
    TEXT: ClassVar[Reads] = Reads(0, "String")
    PATTERN: ClassVar[Reads] = Reads(1, "String")
    BYTES: ClassVar[Reads] = Reads(0, "Binary")
    CATEGORIES: ClassVar[Kinds] = Kinds("Categorical", "Enum")
    STRING: ClassVar[DType] = DType("String")
    COUNT: ClassVar[tuple[TextStep, ...]] = (
        CastsTo(1, DType("Int64"), strict=True),
        Reads(1, "Int64"),
    )
    SLICE_BOUNDS: ClassVar[tuple[TextStep, ...]] = (
        CastsTo(1, DType("Int64")),
        CastsTo(2, DType("UInt64"), strict=True),
        Reads(1, "Int64"),
        Reads(2, "UInt64"),
    )

    LEN_BYTES: ClassVar[Dispatch] = Dispatch(
        "str.len_bytes",
        (Arm(Kinds("String"), Fixed(DType("UInt32"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    LEN_CHARS: ClassVar[Dispatch] = Dispatch(
        "str.len_chars",
        (Arm(Kinds("String"), Fixed(DType("UInt32"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    TO_TEXT: ClassVar[Dispatch] = Dispatch(
        "str.to_lowercase",
        (Arm(Kinds("String"), Fixed(DType("String"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    HEX_DECODE: ClassVar[Dispatch] = Dispatch(
        "str.decode",
        (Arm(Kinds("String"), Fixed(DType("Binary"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    CONTAINS: ClassVar[TextKernel] = TextKernel(
        "str.contains",
        (SHAPE, TEXT, PATTERN),
        DType("Boolean"),
    )
    FIND: ClassVar[TextKernel] = TextKernel("str.find", (SHAPE, TEXT, PATTERN), DType("UInt32"))
    COUNT_MATCHES: ClassVar[TextKernel] = TextKernel(
        "str.count_matches",
        (TEXT, PATTERN, PatternLengths()),
        DType("UInt32"),
    )
    EXTRACT: ClassVar[TextKernel] = TextKernel(
        "str.extract",
        (TEXT, PATTERN, SameLengths(ErrorClass.COMPUTE)),
        DType("String"),
    )
    EXTRACT_ALL: ClassVar[TextKernel] = TextKernel(
        "str.extract_all",
        (TEXT, PATTERN, PatternLengths()),
        DType("List", (DType("String"),)),
    )
    AFFIX: ClassVar[TextKernel] = TextKernel(
        "str.starts_with",
        (SHAPE, TEXT, PATTERN),
        DType("Boolean"),
    )
    STRIP_AFFIX: ClassVar[TextKernel] = TextKernel(
        "str.strip_prefix",
        (SHAPE, TEXT, PATTERN),
        DType("String"),
    )
    STRIP_CHARS: ClassVar[TextKernel] = TextKernel(
        "str.strip_chars",
        (SHAPE, TEXT, ReadsUnlessNull(1)),
        DType("String"),
    )
    JSON_PATH_MATCH: ClassVar[TextKernel] = TextKernel(
        "str.json_path_match",
        (SHAPE, TEXT, PATTERN),
        DType("String"),
    )
    SPLIT: ClassVar[TextKernel] = TextKernel(
        "str.split",
        (TEXT, PATTERN, SHAPE),
        DType("List", (DType("String"),)),
    )
    HEAD: ClassVar[TextKernel] = TextKernel("str.head", (SHAPE, TEXT, *COUNT), STRING)
    SLICE: ClassVar[TextKernel] = TextKernel("str.slice", (SHAPE, TEXT, *SLICE_BOUNDS), STRING)
    PAD: ClassVar[TextKernel] = TextKernel(
        "str.pad_start",
        (SHAPE, Reads(1, "UInt64"), TEXT),
        STRING,
    )
    CONTAINS_ANY: ClassVar[TextKernel] = TextKernel(
        "str.contains_any",
        (TEXT, Reads(1, "List"), SHAPE, OneRowPatterns("str.contains_any", (1,))),
        DType("Boolean"),
    )
    EXTRACT_MANY: ClassVar[TextKernel] = TextKernel(
        "str.extract_many",
        (TEXT, Reads(1, "List"), SHAPE),
        DType("List", (DType("String"),)),
    )
    FIND_MANY: ClassVar[TextKernel] = TextKernel(
        "str.find_many",
        (TEXT, Reads(1, "List"), SHAPE),
        DType("List", (DType("UInt32"),)),
    )
    REPLACE_MANY: ClassVar[TextKernel] = TextKernel(
        "str.replace_many",
        (
            TEXT,
            Reads(1, "List"),
            Reads(2, "List"),
            SHAPE,
            OneRowPatterns("str.replace_many", (1, 2)),
        ),
        STRING,
    )
    JOIN: ClassVar[TextKernel] = TextKernel("str.join", (CastsTo(0, STRING), TEXT), STRING)
    CONCAT_STR: ClassVar[TextKernel] = TextKernel(
        "concat_str", (EachCastsTo(STRING), SHAPE), STRING
    )
    FORMAT: ClassVar[TextKernel] = TextKernel(
        "format",
        (SHAPE, EachCastsTo(STRING, stops_at_nulls=True)),
        STRING,
    )
    BINARY_AFFIX: ClassVar[TextKernel] = TextKernel(
        "bin.contains",
        (BYTES, Reads(1, "Binary"), SHAPE),
        DType("Boolean"),
    )
    BINARY_HEAD: ClassVar[TextKernel] = TextKernel(
        "bin.head",
        (BYTES, *COUNT, SHAPE),
        DType("Binary"),
    )
    BINARY_SLICE: ClassVar[TextKernel] = TextKernel(
        "bin.slice",
        (BYTES, *SLICE_BOUNDS),
        DType("Binary"),
    )
    BINARY_GET: ClassVar[TextKernel] = TextKernel(
        "bin.get",
        (BYTES, CastsTo(1, DType("Int64")), Reads(1, "Int64")),
        DType("UInt8"),
    )
    BINARY_DECODE: ClassVar[Dispatch] = Dispatch(
        "bin.decode",
        (Arm(Kinds("Binary"), Fixed(DType("Binary"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    BINARY_ENCODE: ClassVar[Dispatch] = Dispatch(
        "bin.encode",
        (Arm(Kinds("Binary"), Fixed(DType("String"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    BINARY_SIZE: ClassVar[Dispatch] = Dispatch(
        "bin.size",
        (Arm(Kinds("Binary"), Fixed(DType("UInt32"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    CATEGORY_LENGTHS: ClassVar[Dispatch] = Dispatch(
        "cat.len_chars",
        (Arm(CATEGORIES, Fixed(DType("UInt32"))),),
        Refuse(Stop.PANICS),
    )
    CATEGORY_AFFIX: ClassVar[Dispatch] = Dispatch(
        "cat.starts_with",
        (Arm(CATEGORIES, Fixed(DType("Boolean"))),),
        Refuse(Stop.PANICS),
    )
    CATEGORY_SLICE: ClassVar[Dispatch] = Dispatch(
        "cat.slice",
        (Arm(CATEGORIES, Fixed(DType("String"))),),
        Refuse(Stop.PANICS),
    )
    GET_CATEGORIES: ClassVar[Dispatch] = Dispatch(
        "cat.get_categories",
        (Arm(CATEGORIES, Fixed(DType("String"))),),
        Refuse(ErrorClass.SCHEMA),
    )
    CATEGORY_PHYSICAL: ClassVar[Dispatch] = Dispatch(
        "cat.physical",
        (Arm(CATEGORIES, ToPhysical()),),
        Refuse(ErrorClass.SCHEMA),
    )

    def parse(self, target: DType) -> Dispatch:
        """Give the kernel of a function of `str` that parses text into a type that the node fixes.

        `str.to_decimal`, `str.json_decode` and `str.extract_groups` read
        `String` data and give the type of an option of the node. Whether a
        text parses is a fact of the data.

        Args:
            target: The type of the output.

        Returns:
            The dispatch.

        Example:
            >>> TextKernels().parse(DType("Int64")).output(Data(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: str.parse does not accept Int8
        """
        arms = (Arm(Kinds("String"), Fixed(target)),)
        return Dispatch("str.parse", arms, Refuse(ErrorClass.SCHEMA))

    def reinterpret(self, target: DType) -> Dispatch:
        """Give the kernel of `bin.reinterpret`: `Binary` data gives the type that the node fixes.

        Whether each value has the width of the target is a fact of the data.

        Args:
            target: The type of the output.

        Returns:
            The dispatch.

        Example:
            >>> TextKernels().reinterpret(DType("UInt8")).output(Data(DType("Binary")))
            UInt8
        """
        arms = (Arm(Kinds("Binary"), Fixed(target)),)
        return Dispatch("bin.reinterpret", arms, Refuse(ErrorClass.SCHEMA))

    def split(self, gives: DType) -> TextKernel:
        """Give the kernel of `str.split_exact` and `str.splitn`, which give a struct of parts.

        Args:
            gives: The struct that the node gives.

        Returns:
            The kernel.

        Example:
            >>> TextKernels().split(DType("Struct", ((("field_0", DType("String")),),))).name
            'str.split_exact'
        """
        return TextKernel("str.split_exact", (self.TEXT, self.PATTERN, SplitLengths()), gives)

    def replace(self, count: int) -> TextKernel:
        """Give the kernel of `str.replace` and `str.replace_all`.

        The kernel reads the text, the pattern and the value as `String`
        data, and then checks their lengths (`ReplaceLengths`).

        Args:
            count: The number of matches to replace, below zero for every match.

        Returns:
            The kernel.

        Example:
            >>> TextKernels().replace(-1).name
            'str.replace'
        """
        steps = (self.TEXT, self.PATTERN, Reads(2, "String"), ReplaceLengths(count))
        return TextKernel("str.replace", steps, self.STRING)

    def to_integer(self, target: DType | None) -> TextKernel:
        """Give the kernel of `str.to_integer`, which parses text into an integer type.

        The kernel reads the text, casts the base to `UInt32` with a strict
        cast, checks the lengths of the two inputs, then the target, and then
        the base of each text.

        Args:
            target: The target type of the node, or `None` for `Int64`.

        Returns:
            The kernel.

        Example:
            >>> TextKernels().to_integer(None).gives
            Int64
        """
        gives = DType("Int64") if target is None else target
        steps = (
            self.TEXT,
            CastsTo(1, DType("UInt32"), strict=True),
            Reads(1, "UInt32"),
            self.SHAPE,
            IntegerTarget(gives),
            BaseRange(),
        )
        return TextKernel("str.to_integer", steps, gives)

    def to_category(self, target: DType) -> Dispatch:
        """Give the kernel of `cat.to`, which reads ids of categories as a `Categorical` or `Enum`.

        The input holds the ids, so it must have the physical type of the
        target. A target that has no categories raises a `SchemaError`, and so
        does an input of another type.

        Args:
            target: The target type of the node.

        Returns:
            The dispatch.

        Example:
            >>> enum = DType("Enum", (("a",),))
            >>> TextKernels().to_category(enum).output(Data(DType("UInt8")))
            Enum(categories=['a'])
            >>> TextKernels().to_category(DType("String")).output(Data(DType("UInt32")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: cat.to does not accept UInt32
        """
        if not self.CATEGORIES(target):
            return Dispatch("cat.to", (), Refuse(ErrorClass.SCHEMA))
        physical = target.physical().name
        return Dispatch("cat.to", (Arm(Kinds(physical), Fixed(target)),), Refuse(ErrorClass.SCHEMA))
