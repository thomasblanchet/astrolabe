"""The kernels of the nodes of a kind of their own: `when`, `filter` and the nodes that move rows.

A `when` picks each row from `then` or from `otherwise` by a mask. A `filter`
keeps the rows of a mask. A sort, a slice and a gather move or pick the rows
of their input, so they give its type. Each kernel checks the types that it
reads: the mask must be `Boolean`, and the two branches of a `when` take one
type, as the arithmetic of two series does (`Coercion`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.casts import CastAbility, SeriesCast
from astrolabe._rules.kernels.arithmetic import Coercion
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.supertype import Supertype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astrolabe._rules.type_functions import Operand


@dataclass(frozen=True)
class Mask:
    """Read an input as the `Boolean` mask of a node.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Mask().read(Data(DType("Int8")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: SchemaError: invalid series dtype: ...
    """

    def read(self, mask: Data) -> None:
        """Refuse a mask that is not `Boolean`.

        Args:
            mask: The data of the mask.

        Raises:
            PolarsRefusal: The mask has another type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> Mask().read(Data(DType("Boolean")))
        """
        if mask.dtype.name != "Boolean":
            message = f"invalid series dtype: expected `Boolean`, got `{mask.dtype!r}`"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)


@dataclass(frozen=True)
class Conditional:
    """Give the type of a `when`: the type of its two branches, after a check of the mask.

    The kernel reads the mask first. Then it brings `then` and `otherwise` to
    one type, checks that the three inputs have one length, where an input
    of one row spreads, and picks each row from one of them.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> inputs = [Data(DType("Boolean")), Data(DType("Int8")), Data(DType("Int16"))]
        >>> Conditional().output(inputs, (3, 3, 3))
        Int16
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the mask, of `then` and of `otherwise`.
            lengths: The rows of each input, or none where they are not known.

        Returns:
            The type.

        Raises:
            PolarsRefusal: Two inputs of more than one row differ in length.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> inputs = [Data(DType("Boolean")), Data(DType("Date")), Data(DType("Date"))]
            >>> Conditional().output(inputs, ())
            Date
        """
        Mask().read(inputs[0])
        dtype = Coercion().of(inputs[1], inputs[2]).left.dtype
        if len({length for length in lengths if length != 1}) > 1:
            raise PolarsRefusal(
                ErrorClass.SHAPE, "the mask and the branches of `when` differ in length"
            )
        return dtype


@dataclass(frozen=True)
class Filtered:
    """Give the type of a `filter`: the type of its input, after a check of the mask.

    The mask must be `Boolean`. Then the input and the mask must have one
    length, where either one of one row spreads.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Filtered().output([Data(DType("Date")), Data(DType("Boolean"))], (3, 3))
        Date
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input and of the mask.
            lengths: The rows of each input, or none where they are not known.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The input and the mask differ in length, and neither
                has one row.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> Filtered().output([Data(DType("Int8")), Data(DType("String"))], ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        Mask().read(inputs[1])
        if lengths and 1 not in lengths[:2] and lengths[0] != lengths[1]:
            raise PolarsRefusal(ErrorClass.SHAPE, "the mask of `filter` differs in length")
        return inputs[0].dtype


class SliceArguments:
    """Give the type of a slice: the type of its input, after a read of its offset and its length.

    Polars reads the first value of the offset as an integer and of the length
    as a size. A null offset, and a negative literal length, raise. A null
    length takes every row.

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Data(DType("Date")), Data(DType("Int32")), Data(DType("Int32"))]
        >>> computed = ((), Operand.of_value(0), Operand.of_value(-1))
        >>> SliceArguments().output(inputs, (3, 1, 1), computed)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: ComputeError: unable to extract length ...
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the input, of the offset and of the length.
            lengths: The rows of each input, which the kernel does not read.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the input.

        Raises:
            PolarsRefusal: The offset is a null literal, or the length a
                negative literal.

        Example:
            >>> inputs = [Data(DType("Date")), Data(DType("Int32")), Data(DType("Null"))]
            >>> SliceArguments().output(inputs, (3, 1, 1), ())
            Date
            >>> from astrolabe._rules.type_functions import Operand
            >>> null = Operand.of_value(None)
            >>> SliceArguments().output(inputs, (3, 1, 1), (null, null, null))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: unable to extract offset ...
        """
        del lengths
        offset = computed[1] if len(computed) > 1 else None
        length = computed[2] if len(computed) > 2 else None
        if offset is not None and offset.literal and offset.value is None:
            raise PolarsRefusal(ErrorClass.COMPUTE, "unable to extract offset from a null")
        value = length.value if length is not None and length.literal else None
        if isinstance(value, int) and value < 0:
            raise PolarsRefusal(
                ErrorClass.COMPUTE, "unable to extract length from a negative value"
            )
        return inputs[0].dtype


@dataclass(frozen=True)
class ListedValues:
    """Read the lists of old and new values that `replace` and `replace_strict` take.

    The cast pass puts the old and the new values in lists. The kernel reads
    one list of each, so the input and the two lists must have one length,
    where a list of one row spreads, and each list must then have one row.
    A cast of the values reads the inner type of the list. Where the code
    holds the values, as for a literal list or the `implode` of a literal
    (`StepOperand`), a strict cast also meets each value (`SeriesCast`).

    Example:
        >>> listed = Data(DType("List", (DType("Int64"),)))
        >>> ListedValues().inner(listed)
        Data(dtype=Int64, nulls=False, one_row=False)
    """

    def lengths(self, lengths: Sequence[int], name: str) -> None:
        """Check the lengths of the input and of the lists.

        Args:
            lengths: The rows of each input, or none where they are not known.
            name: The name of the function, for the messages.

        Raises:
            PolarsRefusal: The lengths differ, or a list of values has more than one row.

        Example:
            >>> ListedValues().lengths((3, 2, 1), "replace")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ShapeError: replace: the lengths of the ...
        """
        longer = {length for length in lengths if length != 1}
        if len(longer) > 1:
            raise PolarsRefusal(ErrorClass.SHAPE, f"{name}: the lengths of the inputs differ")
        if lengths and (lengths[1] != 1 or lengths[2] != 1):
            message = f"not yet implemented: `{name}` with a replacement pattern per row"
            raise PolarsRefusal(ErrorClass.COMPUTE, message)

    def inner(self, listed: Data) -> Data:
        """Give the data of the values of a list.

        Args:
            listed: The data of the list.

        Returns:
            The data of its values.

        Example:
            >>> ListedValues().inner(Data(DType("Int8"), nulls=True))
            Data(dtype=Int8, nulls=True, one_row=False)
        """
        values = listed.dtype
        inner = values.inner() if values.name == "List" else values
        return Data(inner, nulls=listed.nulls)

    def cast(self, listed: Data, given: Operand | None, target: DType) -> Data:
        """Cast the values of one list with a strict cast.

        It raises `PolarsRefusal` where the cast has no kernel, or where a
        value that the code holds does not fit (`RunCast`, `SeriesCast`).

        Args:
            listed: The data of the list.
            given: The list as the run computes it, or `None`.
            target: The type of the cast.

        Returns:
            The data of the values after the cast. A struct cast to a type
            that is not a struct keeps its fields, so the data can hold
            another type than the target.

        Example:
            >>> listed = Data(DType("List", (DType("Int8"),)))
            >>> ListedValues().cast(listed, None, DType("Int64")).dtype
            Int64
        """
        inner = self.inner(listed)
        cast = RunCast().of(inner, target)
        literals = self._values(given)
        if literals:
            SeriesCast(inner.dtype.materialized(), target, strict=True).output(literals)
        return cast

    def _values(self, given: Operand | None) -> list[object]:
        """Give the values of a list that the code holds, or none.

        Args:
            given: The input as the run computes it, or `None`.

        Returns:
            The values: those of a literal list, which can be a list of
            several values, or of the `implode` of a literal.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> listed = DType("List", (DType("String"),))
            >>> ListedValues()._values(Operand(listed, value=("a",)))
            ['a']
            >>> pair = Operand.typed_literal([1, 2], DType("List", (DType("Int64"),)))
            >>> ListedValues()._values(pair)
            [1, 2]
        """
        if given is None:  # pragma: no cover -- the run gives each computed input
            return []
        if given.literal and given.dtype.name == "List" and isinstance(given.value, list):
            return list(given.value)
        if given.literal:
            return [given.value]
        if isinstance(given.value, tuple):
            return list(given.value)
        return []


class Replaced:
    """Give the type of `replace`: the type of its input, after the checks of the other inputs.

    The kernel reads the lists of old and new values (`ListedValues`). The
    new values must count one value or as many values as the old values, a
    count that the code holds for literal values (`_counts`). The old values
    must not repeat, a fact of the data. Each list casts its
    values to the type of the input with a strict cast, so a cast that
    polars does not have, or a literal value that does not fit, raises. A
    struct cast to a number stays a struct. The input must then compare with
    the old values at their supertype (`_mask`). The output joins the new
    values with the input, at their supertype, so a struct of new values
    gives a struct. Polars refuses a struct of several fields there, and a
    struct whose field is a struct.

    Example:
        >>> listed = Data(DType("List", (DType("Int64"),)))
        >>> Replaced().output([Data(DType("Int8")), listed, listed], (3, 1, 1), ())
        Int8
    """

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the lengths do not fit, or where a cast
        of the values fails (`ListedValues`).

        Args:
            inputs: The data of the input, of the old values and of the new values.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the input.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> listed = Data(DType("List", (DType("Int32"),)))
            >>> new = Operand.typed_literal(300, DType("List", (DType("Int32"),)))
            >>> inputs = [Data(DType("UInt8")), listed, listed]
            >>> Replaced().output(inputs, (3, 1, 1), (new, new, new))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        listed = ListedValues()
        listed.lengths(lengths[:3], "replace")
        self._counts(computed)
        target = inputs[0].dtype
        values: list[Data] = []
        for place in (1, 2):
            given = computed[place] if place < len(computed) else None
            values.append(listed.cast(inputs[place], given, target))
        self._mask(values[0].dtype, target)
        new = values[1].dtype
        if new == target:
            return target
        self._struct(new, target)
        return Supertype().require(new, target)

    def _counts(self, computed: Sequence[Operand]) -> None:
        """Refuse new values that count neither one value nor the values of the old values.

        A literal of one value counts one, and a literal series counts its
        values. The values of a column are a fact of the data.

        Args:
            computed: Each input as the run computes it, or none.

        Raises:
            PolarsRefusal: Both counts are in the code, and they do not fit.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.type_functions import Operand
            >>> old, new = Operand.of_value(1), Operand(DType("Int64"), value=(2, 3))
            >>> Replaced()._counts((Operand.column(DType("Int64")), old, new))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `new` input ...
        """
        counts = [self._count(operand) for operand in computed[1:3]]
        if len(counts) < 2 or None in counts or counts[1] in {1, counts[0]}:
            return
        message = "`new` input for `replace` must have the same length as `old` or have length 1"
        raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _count(self, operand: Operand) -> int | None:
        """Give the number of values of an input that the code holds, or `None` for a column.

        Args:
            operand: The input as the run computes it.

        Returns:
            The count.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> Replaced()._count(Operand.of_value(1))
            1
        """
        if isinstance(operand.value, tuple):
            return len(operand.value)
        return 1 if operand.literal else None

    def _mask(self, old: DType, target: DType) -> None:
        """Refuse old values that the input cannot compare with, as polars does to find its rows.

        Polars compares the input with the old values at their supertype. A
        struct of old values cast to the type of the input stays a struct. A
        struct has a supertype with a number only, and the number must cast
        to that struct (`_refusal`).

        Args:
            old: The type of the old values after their cast.
            target: The type of the input.

        Raises:
            PolarsRefusal: The two types have no supertype.

        Example:
            >>> old = DType("Struct", ((("b", DType("Boolean")),),))
            >>> Replaced()._mask(old, DType("Boolean"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: could not evaluate comparison ...
        """
        if old == target:
            return
        joined = Supertype().of(target, old)
        if joined is not None and self._refusal(joined, target) is None:
            return
        message = f"could not evaluate comparison of {target!r} and {old!r}"
        raise PolarsRefusal(ErrorClass.SCHEMA, message)

    def _struct(self, new: DType, target: DType) -> None:
        """Refuse a struct of new values that polars cannot join with an input of another kind.

        Args:
            new: The type of the new values after their cast.
            target: The type of the input.

        Raises:
            PolarsRefusal: The struct has several fields, or a field that is a struct.

        Example:
            >>> two = DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),))
            >>> Replaced()._struct(two, DType("Int8"))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: must specify ...
        """
        message = self._refusal(new, target)
        if message is not None:
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)

    def _refusal(self, joined: DType, target: DType) -> str | None:
        """Give why polars cannot cast the input to a struct that joins it with the values.

        Args:
            joined: The join of the input with the values.
            target: The type of the input.

        Returns:
            The message of polars, or `None` where the cast runs.

        Example:
            >>> Replaced()._refusal(DType("Struct", ((("a", DType("Int8")),),)), DType("Int8"))
        """
        if joined.name != "Struct" or target.name == "Struct":
            return None
        fields = joined.fields()
        if len(fields) != 1:
            return "must specify one field in the struct"
        if fields[0][1].name == "Struct":
            return "Cannot cast from struct to other types"
        return None


@dataclass(frozen=True)
class ReplacedStrict:
    """Give the type of `replace_strict`: the type of the new values, or the type the call names.

    The kernel reads the lists of old and new values (`ListedValues`). The
    old values must not repeat, and each value of the input must be among
    them where no default is given: facts of the data.

    Without a default, the old values must cast to the type of the input by
    the table of polars (`CastAbility`), then they cast to it with a strict
    cast. The new values cast to `return_dtype` with a strict cast where the
    call names one. The output has the type of the new values.

    With a default, the default must have the length of the input or one
    row. The output type is `return_dtype`, or the supertype of the new
    values and the default. The default and the new values cast to it with
    casts that are not strict, and the old values cast to the type of the
    input with a strict cast.

    Example:
        >>> listed = Data(DType("List", (DType("Float32"),)))
        >>> ReplacedStrict().output([Data(DType("Int8")), listed, listed], (3, 1, 1), ())
        Float32
    """

    return_dtype: DType | None = None

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the lengths do not fit, or where a cast
        of the values fails (`ListedValues`).

        Args:
            inputs: The data of the input, of the old values, of the new values
                and of the default where the call gives one.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The old values do not cast to the type of the input
                by the table of polars.

        Example:
            >>> texts = Data(DType("List", (DType("String"),)))
            >>> ReplacedStrict().output([Data(DType("Date")), texts, texts], (3, 1, 1), ())
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: cannot use values ...
        """
        listed = ListedValues()
        listed.lengths(lengths, "replace_strict")
        if len(inputs) > 3:
            return self._defaulted(inputs, lengths, computed)
        target = inputs[0].dtype
        old = listed.inner(inputs[1]).dtype
        if CastAbility().of(old, target) is not True:
            message = f"cannot use values of type `{old!r}` to replace values in a column of type"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, f"{message} `{target!r}`")
        listed.cast(inputs[1], self._given(computed, 1), target)
        if self.return_dtype is None:
            return listed.inner(inputs[2]).dtype.materialized()
        listed.cast(inputs[2], self._given(computed, 2), self.return_dtype)
        return self.return_dtype

    def _defaulted(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output with a default.

        It raises `PolarsRefusal` where a cast fails (`RunCast`, `ListedValues`).

        Args:
            inputs: The data of the input, of the old values, of the new values
                and of the default.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type.

        Raises:
            PolarsRefusal: The default has another length, or the new values
                and the default have no supertype.

        Example:
            >>> listed = Data(DType("List", (DType("Int8"),)))
            >>> inputs = [Data(DType("Int8")), listed, listed, Data(DType("Float32"))]
            >>> ReplacedStrict()._defaulted(inputs, (3, 1, 1, 1), ())
            Float32
            >>> ReplacedStrict()._defaulted(inputs, (1, 1, 1, 3), ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: `default` input ...
            >>> dated = [*inputs[:3], Data(DType("Date"))]
            >>> ReplacedStrict()._defaulted(dated, (3, 1, 1, 1), ())  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: failed to determine supertype ...
        """
        default = inputs[3]
        if len(lengths) > 3 and lengths[3] not in {lengths[0], 1}:
            message = "`default` input for `replace_strict` must have the same length as the input"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, f"{message} or have length 1")
        listed = ListedValues()
        new = listed.inner(inputs[2])
        joined = Supertype().of(new.dtype.materialized(), default.dtype)
        output = self.return_dtype or joined
        if output is None:
            message = f"failed to determine supertype of {new.dtype!r} and {default.dtype!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        RunCast().of(default, output)
        listed.cast(inputs[1], self._given(computed, 1), inputs[0].dtype)
        RunCast().of(new, output)
        return output

    def _given(self, computed: Sequence[Operand], place: int) -> Operand | None:
        """Give an input as the run computes it, where the call gives it.

        Args:
            computed: Each input as the run computes it, or none.
            place: The place of the input.

        Returns:
            The input, or `None`.

        Example:
            >>> ReplacedStrict()._given((), 1) is None
            True
        """
        return computed[place] if place < len(computed) else None


@dataclass(frozen=True)
class Kept:
    """Give the type of a node that moves or picks the rows of its first input, such as a sort.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Kept().output([Data(DType("Date")), Data(DType("Int8"))], ())
        Date
    """

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the first input.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the kernel does not read.

        Returns:
            The type.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> Kept().output([Data(DType("Null"))], (3,))
            Null
        """
        del lengths
        return inputs[0].dtype


@dataclass(frozen=True)
class Gathered:
    """Give the type of a gather: the type of its values, where its indices are integers.

    A literal index outside the rows of the values raises, unless the call
    asks for a null there (`null_on_oob`). A negative index counts from the
    end. Only values whose rows the code bounds, such as a literal or a
    `head(2)`, raise at every run (`Data.bound`): the rows of a column are a
    fact of the data.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Gathered().output([Data(DType("Date")), Data(DType("Float32"))], (), ())
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: expected integers as index
    """

    null_on_oob: bool = False

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the indices.
            lengths: The rows of each input, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the values.

        Raises:
            PolarsRefusal: The indices are not integers, or a literal index is
                outside the rows of the values.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> from astrolabe._rules.type_functions import Operand
            >>> index = Operand.of_value(2)
            >>> inputs = [Data(DType("Date"), bound=2), Data(DType("Int32"))]
            >>> Gathered().output(inputs, (2, 1), (Operand.column(DType("Date")), index))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: OutOfBoundsError: gather indices ...
            >>> last = Operand.of_value(-2)
            >>> Gathered().output(inputs, (2, 1), (Operand.column(DType("Date")), last))
            Date
        """
        del lengths
        if not Is.integer(inputs[1].dtype):
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, "expected integers as index")
        index = computed[1].value if len(computed) > 1 and computed[1].literal else None
        bound = inputs[0].bound
        if self.null_on_oob or not isinstance(index, int) or bound is None:
            return inputs[0].dtype
        if not -bound <= index < bound:
            raise PolarsRefusal(ErrorClass.OUT_OF_BOUNDS, "gather indices are out of bounds")
        return inputs[0].dtype


@dataclass(frozen=True)
class GatheredInGroups:
    """Give the type of a gather in `agg`, where polars runs it on the values of each group.

    `get` gives one value for each group. It casts its index to `Int64`
    with a cast that is not strict, so any type that casts reads as an
    index, and a struct, which stays a struct, panics. An index outside the
    values of a group raises a `ComputeError`, unless the call asks for a
    null there. `gather` needs integer indices, and an index outside the
    values of a group raises an `OutOfBoundsError`, except over values of
    the type `Null`. Only values whose rows the code bounds raise at every
    run (`Data.bound`).

    Example:
        >>> from astrolabe._rules.type_functions import Operand
        >>> inputs = [Data(DType("Int8")), Data(DType("Date"))]
        >>> GatheredInGroups(returns_scalar=True).output(inputs, (3, 1), ())
        Int8
    """

    INT64: ClassVar[DType] = DType("Int64")

    returns_scalar: bool
    null_on_oob: bool = False

    def output(
        self,
        inputs: Sequence[Data],
        lengths: Sequence[int],
        computed: Sequence[Operand],
    ) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of the values and of the indices.
            lengths: The rows of each input in one group, or none where they are not known.
            computed: Each input as the run computes it, or none.

        Returns:
            The type of the values.

        Raises:
            PolarsRefusal: The index does not read as an index, or a literal
                index is outside the values of a group.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> inputs = [Data(DType("Int8"), bound=1), Data(DType("Int32"))]
            >>> gather = GatheredInGroups(returns_scalar=True)
            >>> gather.output(inputs, (1, 1), ((), Operand.of_value(1)))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: ComputeError: get index is out of bounds
        """
        values = inputs[0]
        self._index(inputs[1])
        if self.null_on_oob or not self._outside(values, computed):
            return values.dtype
        if self.returns_scalar:
            raise PolarsRefusal(ErrorClass.COMPUTE, "get index is out of bounds")
        if values.dtype.name == "Null":
            return values.dtype
        raise PolarsRefusal(ErrorClass.OUT_OF_BOUNDS, "gather indices are out of bounds")

    def _index(self, index: Data) -> None:
        """Refuse an index that does not read as an index.

        Args:
            index: The data of the index.

        Raises:
            PolarsRefusal: The index of `get` stays a struct after its cast to
                `Int64`, where polars panics, or the indices of `gather` are not
                integers.

        Example:
            >>> GatheredInGroups(returns_scalar=False)._index(Data(DType("Date")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if self.returns_scalar and RunCast().of(index, self.INT64).dtype.name != "Int64":
            raise PolarsRefusal(Stop.PANICS, "called `Option::unwrap()` on a cast index")
        if not self.returns_scalar and not Is.integer(index.dtype):
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, "expected integers as index")

    def _outside(self, values: Data, computed: Sequence[Operand]) -> bool:
        """Say whether a literal index falls outside values whose rows the code bounds.

        Args:
            values: The data of the values in one group.
            computed: Each input as the run computes it, or none.

        Returns:
            Whether it does.

        Example:
            >>> from astrolabe._rules.type_functions import Operand
            >>> one = Data(DType("Int8"), bound=1)
            >>> GatheredInGroups(returns_scalar=True)._outside(one, ((), Operand.of_value(-1)))
            False
        """
        literal = computed[1].value if len(computed) > 1 and computed[1].literal else None
        bound = values.bound
        return isinstance(literal, int) and bound is not None and not -bound <= literal < bound


@dataclass(frozen=True)
class Structured:
    """Give the type of `pl.struct`: a struct of the data of each input, by the name of each.

    Example:
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> Structured(("a", "b")).output([Data(DType("Int8")), Data(DType("Int32"))], ())
        Struct({'a': Int8, 'b': Int32})
    """

    names: tuple[str, ...]

    def output(self, inputs: Sequence[Data], lengths: Sequence[int]) -> DType:
        """Give the type of the output.

        Args:
            inputs: The data of each input.
            lengths: The rows of each input, which the run checks.

        Returns:
            The struct type.

        Example:
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> Structured(("x",)).output([Data(DType("Null"))], ())
            Struct({'x': Null})
        """
        del lengths
        fields = tuple(zip(self.names, (data.dtype for data in inputs), strict=True))
        return DType("Struct", (fields,))
