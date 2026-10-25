"""The arithmetic kernels of polars: the operators of two series, the divisions and `pow`.

The addition of two series in polars does not trust its inputs. It routes
structs, lists and arrays to their own kernels, and refuses a temporal series
beside a series that is not temporal. It then coerces both sides to their
supertype, and calls the arithmetic of the left type. An eager `Series`
operation calls it directly. An expression adds a few shortcuts: an in-place
path for two columns, the cast to `Float64` of the true division, and its own
floor division. This module translates each of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import Decimals, DType, Is, Parameter, TimeUnit
from astrolabe._rules.failures import ErrorClass, PolarsRefusal, Stop
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.kernels.nested import ArrayArithmetic, ListArithmetic, StructArithmetic
from astrolabe._rules.kernels.temporal import DateOps, DatetimeOps, DurationOps, TimeOps
from astrolabe._rules.operators import NumericOp
from astrolabe._rules.supertype import Supertype

if TYPE_CHECKING:
    from astrolabe._rules.kernels.dispatch import BinaryKernel


@dataclass(frozen=True)
class Operands:
    """Hold the two inputs of a kernel after the kernel coerces them to one type.

    Example:
        >>> Operands(Data(DType("Int16")), Data(DType("Int16"))).left
        Data(dtype=Int16, nulls=False, one_row=False)
    """

    left: Data
    right: Data


class Coercion:
    """Bring two inputs to one type before a kernel runs, as the arithmetic of two series does.

    A datetime or a date beside a duration keeps its kind, and only the time
    units join. Other inputs join their leaf types, that is, the types under
    every `List` and `Array`, and a nested side then joins its whole type with
    the other side. Each input that differs from its target is cast, and the
    cast raises where no cast kernel exists. The arithmetic and the comparison
    of two series both start here.

    Example:
        >>> coerced = Coercion().of(Data(DType("Int8")), Data(DType("Float32")))
        >>> coerced.left.dtype, coerced.right.dtype
        (Float32, Float32)
    """

    def __init__(self) -> None:
        """Build the cast and the join of the run.

        Example:
            >>> Coercion()._cast is not None
            True
        """
        self._cast = RunCast()
        self._join = Supertype()

    def of(self, left: Data, right: Data) -> Operands:
        """Give the two inputs after the coercion.

        It raises `PolarsRefusal` where the types have no supertype, or a cast has no kernel.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The coerced inputs.

        Example:
            >>> listed = Data(DType("List", (DType("Int8"),)))
            >>> Coercion().of(listed, Data(DType("Int16"))).left.dtype  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        timed = self._time_units(left, right)
        if timed is not None:
            return timed
        leaf = self._join.require(left.dtype.leaf(), right.dtype.leaf())
        new_left = left.dtype.with_leaf(leaf)
        new_right = right.dtype.with_leaf(leaf)
        if {left.dtype.name, right.dtype.name} & Is.LISTS:
            new_left = self._join.require(new_left, new_right)
            new_right = new_left
        return Operands(self._to(left, new_left), self._to(right, new_right))

    def _to(self, data: Data, target: DType) -> Data:
        """Cast an input where its type differs from the target.

        It raises `PolarsRefusal` where the cast has no kernel.

        Args:
            data: The input.
            target: The target type.

        Returns:
            The input at the target type.

        Example:
            >>> Coercion()._to(Data(DType("Int8")), DType("Int8")).dtype
            Int8
        """
        if data.dtype.same(target):
            return data
        return self._cast.of(data, target)

    def _time_units(self, left: Data, right: Data) -> Operands | None:
        """Join only the time units of a temporal pair, and keep the kinds of the two sides.

        A datetime or a duration beside a duration takes the joined unit
        (`TimeUnit.joined`), and a date beside a duration stays as it is.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The inputs at the joined unit, or `None` where the pair is not one
            of a datetime or a date beside a duration, or of two durations.

        Example:
            >>> span = Data(DType("Duration", ("ns",)))
            >>> moment = Data(DType("Datetime", ("ms", None)))
            >>> timed = Coercion()._time_units(span, moment)
            >>> timed.left.dtype, timed.right.dtype
            (Duration(time_unit='ms'), Datetime(time_unit='ms', time_zone=None))
        """
        names = (left.dtype.name, right.dtype.name)
        if names in {("Date", "Duration"), ("Duration", "Date")}:
            return Operands(left, right)
        if names == ("Duration", "Datetime"):
            swapped = self._units(right, left)
            return Operands(swapped.right, swapped.left)
        if names in {("Datetime", "Duration"), ("Duration", "Duration")}:
            return self._units(left, right)
        return None

    def _units(self, left: Data, right: Data) -> Operands:
        """Give two temporal inputs at their joined unit.

        Args:
            left: The datetime or the duration.
            right: The duration.

        Returns:
            Both inputs at the joined unit (`TimeUnit.joined`).

        Example:
            >>> nano, micro = Data(DType("Duration", ("ns",))), Data(DType("Duration", ("us",)))
            >>> Coercion()._units(nano, micro).left.dtype
            Duration(time_unit='us')
        """
        parameter = Parameter()
        unit = TimeUnit.joined(parameter.unit(left.dtype), parameter.unit(right.dtype))
        left_type = DType(left.dtype.name, (unit, *left.dtype.args[1:]))
        return Operands(left.with_dtype(left_type), right.with_dtype(DType("Duration", (unit,))))


class NumOps:
    """Run the arithmetic of the left type on two coerced inputs.

    Each type of polars has its own arithmetic for the left side of an
    operation. A number needs the same type on the right and keeps it. A
    `Decimal` gives a decimal of precision 38, a `Null` gives `Null`, text and
    binary data only add, and a `Boolean` only adds into `UInt32`. A list, an
    array and each temporal type have their own kernels. Every other type
    raises.

    Example:
        >>> NumOps(NumericOp.ADD).output(Data(DType("Boolean")), Data(DType("Boolean")))
        UInt32
    """

    def __init__(self, op: NumericOp) -> None:
        """Build the kernel of each kind for the operation.

        Args:
            op: The operation.

        Example:
            >>> sorted(NumOps(NumericOp.ADD)._kinds)[:3]
            ['Array', 'Binary', 'Boolean']
        """
        self._op = op
        text = _TextOps(op)
        self._kinds: dict[str, BinaryKernel] = {
            "Decimal": _DecimalOps(op),
            "Null": _NullOps(),
            "String": text,
            "Binary": text,
            "Boolean": _BooleanOps(op),
            "List": ListArithmetic(op),
            "Array": ArrayArithmetic(op),
            "Date": DateOps(op),
            "Datetime": DatetimeOps(op),
            "Duration": DurationOps(op),
            "Time": TimeOps(op),
        }

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The coerced left input.
            right: The coerced right input.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The left type has no such arithmetic, or the right type does not fit it.

        Example:
            >>> empty = Data(DType("Struct", ((),)))
            >>> NumOps(NumericOp.REM).output(empty, Data(DType("Int8")))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> NumOps(NumericOp.ADD).output(Data(DType("Int8")), Data(DType("Int16")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ... of Int8 beside Int16
        """
        if Is.primitive_numeric(left.dtype):
            if (  # pragma: no cover -- the caller coerces the inputs
                not left.dtype.same(right.dtype)
            ):
                message = f"{self._op.value} of {left.dtype!r} beside {right.dtype!r}"
                raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
            return left.dtype
        kind = self._kinds.get(left.dtype.name)
        if kind is None:
            message = f"{self._op.value} is not defined for {left.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return kind.output(left, right)


class _DecimalOps:
    """Give the type of an operation with a `Decimal` on the left.

    The right side must be a decimal too. The output has the largest precision
    and the larger scale. A decimal has no remainder.

    Example:
        >>> ten = DType("Decimal", (10, 2))
        >>> _DecimalOps(NumericOp.MUL).output(Data(ten), Data(DType("Decimal", (5, 4))))
        Decimal(precision=38, scale=4)
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> _DecimalOps(NumericOp.ADD)._op
            <NumericOp.ADD: 'add'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The decimal input.
            right: The other input.

        Returns:
            The decimal type of the output.

        Raises:
            PolarsRefusal: The operation is a remainder, or the right side is not a decimal.

        Example:
            >>> ten = DType("Decimal", (10, 2))
            >>> _DecimalOps(NumericOp.REM).output(Data(ten), Data(ten))  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
            >>> _DecimalOps(NumericOp.ADD).output(Data(ten), Data(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: Int8 is not a decimal
        """
        if self._op is NumericOp.REM:
            message = "a decimal has no remainder"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if right.dtype.name != "Decimal":  # pragma: no cover -- the caller coerces the inputs
            message = f"{right.dtype!r} is not a decimal"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        parameter = Parameter()
        scale = max(parameter.integer(left.dtype, 1), parameter.integer(right.dtype, 1))
        return Decimals().widest(scale)


class _NullOps:
    """Give `Null` for any operation with a `Null` series on the left.

    Example:
        >>> _NullOps().output(Data(DType("Null")), Data(DType("Null")))
        Null
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give `Null`.

        Args:
            left: The `Null` input.
            right: The other input.

        Returns:
            `Null`.

        Example:
            >>> _NullOps().output(Data(DType("Null")), Data(DType("Null"), nulls=True))
            Null
        """
        return DType("Null")


class _TextOps:
    """Give the type of an operation with text on the left.

    Text concatenates with text of the same type. Every other operation raises,
    as the default arithmetic of a type of polars does.

    Example:
        >>> _TextOps(NumericOp.ADD).output(Data(DType("String")), Data(DType("String")))
        String
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> _TextOps(NumericOp.SUB)._op
            <NumericOp.SUB: 'sub'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The text input.
            right: The other input.

        Returns:
            The type of the left input.

        Raises:
            PolarsRefusal: The operation is not an addition, or the types differ.

        Example:
            >>> binary, text = Data(DType("Binary")), Data(DType("String"))
            >>> _TextOps(NumericOp.ADD).output(binary, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: ...
        """
        if self._op is not NumericOp.ADD:
            message = f"{self._op.value} is not defined for {left.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if not left.dtype.same(right.dtype):  # pragma: no cover -- the caller coerces the inputs
            message = f"cannot unpack {right.dtype!r} into {left.dtype!r}"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return left.dtype


class _BooleanOps:
    """Give the type of an operation with a `Boolean` series on the left.

    Two booleans add into the index type of polars, `UInt32`. Every other
    operation raises.

    Example:
        >>> boolean = Data(DType("Boolean"))
        >>> _BooleanOps(NumericOp.SUB).output(boolean, boolean)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: sub ...
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation.

        Example:
            >>> _BooleanOps(NumericOp.ADD)._op
            <NumericOp.ADD: 'add'>
        """
        self._op = op

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The `Boolean` input.
            right: The other input.

        Returns:
            `UInt32`.

        Raises:
            PolarsRefusal: The operation is not an addition, or the right side is not `Boolean`.

        Example:
            >>> _BooleanOps(NumericOp.ADD).output(Data(DType("Boolean")), Data(DType("Boolean")))
            UInt32
            >>> _BooleanOps(NumericOp.ADD).output(Data(DType("Boolean")), Data(DType("Int8")))
            ... # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: SchemaError: cannot unpack Int8 into Boolean
        """
        if self._op is not NumericOp.ADD:
            message = f"{self._op.value} is not defined for Boolean"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if right.dtype.name != "Boolean":  # pragma: no cover -- the caller coerces the inputs
            message = f"cannot unpack {right.dtype!r} into Boolean"
            raise PolarsRefusal(ErrorClass.SCHEMA, message)
        return DType.idx()


class SeriesArithmetic:
    """Give the type of `+`, `-`, `*`, `/` or `%` of two series, as polars runs it.

    It is the kernel that an eager `Series` operation reaches, with no cast
    pass in front of it. Two structs run field by field. A product with a
    temporal left side, or a quotient with a `Duration` left side, goes to the
    left type at once. A product with a duration on the right swaps the sides,
    and a product or a quotient with another temporal side raises. A list or an
    array runs leaf by leaf. A sum or a difference of a temporal series and a
    series that is not temporal raises. Every other pair is coerced
    (`Coercion`), and the left type runs its arithmetic (`NumOps`). `/` is the
    division of Rust: two integers give an integer.

    Example:
        >>> SeriesArithmetic(NumericOp.ADD).output(Data(DType("Int8")), Data(DType("Float32")))
        Float32
        >>> SeriesArithmetic(NumericOp.DIV).output(Data(DType("Int8")), Data(DType("Int8")))
        Int8
    """

    OPS: ClassVar[frozenset[NumericOp]] = frozenset({
        NumericOp.ADD,
        NumericOp.SUB,
        NumericOp.MUL,
        NumericOp.DIV,
        NumericOp.REM,
    })

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation. The floor division is not an operation of a series.

        Raises:
            ValueError: The operation is the floor division.

        Example:
            >>> SeriesArithmetic(NumericOp.FLOOR_DIV)
            Traceback (most recent call last):
            ...
            ValueError: floor_div is not an operation of two series
        """
        if op not in self.OPS:  # pragma: no cover -- each caller passes an operation of two series
            message = f"{op.value} is not an operation of two series"
            raise ValueError(message)
        self._op = op
        self._numbers = NumOps(op)
        self._coercion = Coercion()
        firsts = {NumericOp.MUL: self._product_first, NumericOp.DIV: self._quotient_first}
        self._first = firsts.get(op, self._no_first)

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The sum or the difference of a temporal series and a series that is not
                temporal.

        Example:
            >>> small, day = Data(DType("Int8")), Data(DType("Date"))
            >>> SeriesArithmetic(NumericOp.SUB).output(small, day)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        names = {left.dtype.name, right.dtype.name}
        if names == {"Struct"}:
            return StructArithmetic().output(left, right, self)
        temporal = self._first(left, right)
        if temporal is not None:
            return temporal
        if "List" in names:
            return ListArithmetic(self._op).output(left, right)
        if "Array" in names:
            return ArrayArithmetic(self._op).output(left, right)
        mixed = Is.temporal(left.dtype) != Is.temporal(right.dtype)
        if self._op in {NumericOp.ADD, NumericOp.SUB} and mixed:
            message = f"{self._op.value} of {left.dtype!r} and {right.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        coerced = self._coercion.of(left, right)
        return self._numbers.output(coerced.left, coerced.right)

    def right_named(self, left: Data, right: Data) -> bool:
        """Say whether the output series takes the name of the right input.

        Two structs keep the name that their arithmetic gives
        (`StructArithmetic`). An addition of text or of booleans with one row
        on the left and more rows on the right broadcasts the left value into
        the right series, so the output keeps the name of the right. A text
        that is null takes the left name. Every other kernel names its output
        after the left input, the kernels of a list or an array included, which
        do not coerce their inputs.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Whether the output takes the name of the right input.

        Example:
            >>> one = Data(DType("String"), one_row=True)
            >>> SeriesArithmetic(NumericOp.ADD).right_named(one, Data(DType("String")))
            True
            >>> listed = Data(DType("List", (DType("Int8"),)), one_row=True)
            >>> SeriesArithmetic(NumericOp.ADD).right_named(listed, Data(DType("Int8")))
            False
            >>> one = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> two = Data(DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),)))
            >>> SeriesArithmetic(NumericOp.ADD).right_named(one, two)
            True
        """
        names = {left.dtype.name, right.dtype.name}
        if names == {"Struct"}:
            return StructArithmetic().right_named(left, right)
        nested = bool(names & Is.LISTS)
        if nested or self._op is not NumericOp.ADD or not left.one_row or right.one_row:
            return False
        coerced = self._coercion.of(left, right).left
        if coerced.dtype.name == "Boolean":
            return True
        return coerced.dtype.name in {"String", "Binary"} and not coerced.nulls

    def _product_first(self, left: Data, right: Data) -> DType | None:
        """Run the temporal arms of a product, before its nested arms.

        A temporal left side runs the arithmetic of the left type with no
        coercion. A date, a datetime or a time on the right raises. A
        duration on the right swaps the sides.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the output, or `None` where no temporal arm applies.

        Raises:
            PolarsRefusal: The right side is a date, a datetime or a time.

        Example:
            >>> span = Data(DType("Duration", ("ms",)))
            >>> SeriesArithmetic(NumericOp.MUL)._product_first(Data(DType("Int8")), span)
            Duration(time_unit='ms')
        """
        if Is.temporal(left.dtype):
            return self._numbers.output(left, right)
        if right.dtype.name in Is.MOMENTS:
            message = f"mul by {right.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if right.dtype.name == "Duration":
            return self._numbers.output(right, left)
        return None

    def _quotient_first(self, left: Data, right: Data) -> DType | None:
        """Run the temporal arms of a quotient, before its nested arms.

        A duration on the left runs the arithmetic of a duration with no
        coercion. Another temporal side raises.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            The type of the output, or `None` where no temporal arm applies.

        Raises:
            PolarsRefusal: A side is temporal and the left side is not a duration.

        Example:
            >>> span = Data(DType("Duration", ("ms",)))
            >>> SeriesArithmetic(NumericOp.DIV)._quotient_first(span, Data(DType("Int8")))
            Duration(time_unit='ms')
        """
        if left.dtype.name == "Duration":
            return self._numbers.output(left, right)
        if Is.temporal(left.dtype) or Is.temporal(right.dtype):
            message = f"div of {left.dtype!r} and {right.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        return None

    def _no_first(self, left: Data, right: Data) -> DType | None:
        """Give `None`, for an operation that has no temporal arm before its nested arms.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            `None`.

        Example:
            >>> day = Data(DType("Date"))
            >>> SeriesArithmetic(NumericOp.ADD)._no_first(day, day) is None
            True
        """
        return None


class OwnedArithmetic:
    """Give the type of `+`, `-` or `*` of two columns, with the in-place path.

    An expression adds two columns that it owns. Where the left type is a
    number and the right type holds numbers, polars joins the two types, casts
    both sides and runs the kernel of a number in place. It does not check the
    join or the casts, so a missing supertype panics, and so does a supertype
    that is not a number. Every other pair runs the arithmetic of two series
    (`SeriesArithmetic`).

    Example:
        >>> OwnedArithmetic(NumericOp.ADD).output(Data(DType("Int32")), Data(DType("Date")))
        Int32
        >>> small, day = Data(DType("Int8")), Data(DType("Date"))
        >>> OwnedArithmetic(NumericOp.ADD).output(small, day)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: no supertype of Int8 ...
    """

    def __init__(self, op: NumericOp) -> None:
        """Keep the operation.

        Args:
            op: The operation: an addition, a difference or a product.

        Example:
            >>> OwnedArithmetic(NumericOp.MUL)._op
            <NumericOp.MUL: 'mul'>
        """
        self._op = op
        self._cast = RunCast(unwrap=True)

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The left column.
            right: The right column.

        Returns:
            The type of the output.

        It raises `PolarsRefusal` where the arithmetic of two series fails, or
        a cast panics.

        Raises:
            PolarsRefusal: The two types have no supertype, or it is not a number.

        Example:
            >>> small, ten = Data(DType("Int8")), Data(DType("Decimal", (10, 2)))
            >>> OwnedArithmetic(NumericOp.ADD).output(small, ten)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: panics: ...
        """
        eligible = Is.primitive_numeric(left.dtype) and Is.primitive_numeric(right.dtype.physical())
        if not eligible:
            return SeriesArithmetic(self._op).output(left, right)
        joined = Supertype().of(left.dtype, right.dtype)
        if joined is None:  # pragma: no cover -- two numbers have a supertype
            message = f"no supertype of {left.dtype!r} and {right.dtype!r}"
            raise PolarsRefusal(Stop.PANICS, message)
        self._cast.of(left, joined)
        self._cast.of(right, joined)
        if not Is.primitive_numeric(joined):
            raise PolarsRefusal(Stop.PANICS, f"{joined!r} is not a number")
        return joined


class TrueDivide:
    """Give the type of the true division `/` of an expression.

    The division of a float, a decimal, a temporal value or a nested value runs
    the division of two series as it is. Text raises. A temporal right side
    also runs as it is. Every other pair casts both sides to `Float64` first,
    so two integers give a float. The cast of a struct to a number casts each
    field, and a struct of only nulls casts to a plain `Float64`.

    Example:
        >>> TrueDivide().output(Data(DType("Int8")), Data(DType("Float32")))
        Float64
        >>> TrueDivide().output(Data(DType("Float32")), Data(DType("Int8")))
        Float32
    """

    DIRECT: ClassVar[frozenset[str]] = frozenset({
        "Decimal",
        "Float16",
        "Duration",
        "Date",
        "Datetime",
        "Float32",
        "Float64",
        "Array",
        "List",
    })

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The dividend.
            right: The divisor.

        Returns:
            The type of the output.

        It raises `PolarsRefusal` where the division of two series fails, or a
        cast to `Float64` has no kernel.

        Raises:
            PolarsRefusal: A side is text.

        Example:
            >>> small, text = Data(DType("Int8")), Data(DType("String"))
            >>> TrueDivide().output(small, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        division = SeriesArithmetic(NumericOp.DIV)
        nested_right = right.dtype.name in {"Array", "List"}
        if left.dtype.name in self.DIRECT or nested_right:
            return division.output(left, right)
        if (  # pragma: no cover -- the type function refuses text first
            "String" in {left.dtype.name, right.dtype.name}
        ):
            message = "cannot divide using strings"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if Is.temporal(right.dtype):
            return division.output(left, right)
        cast = RunCast()
        floats = DType("Float64")
        return division.output(cast.of(left, floats), cast.of(right, floats))


class FloorDivide:
    """Give the type of the floor division `//` of two series.

    Two structs run field by field, and a list or an array runs leaf by leaf.
    Otherwise the left side must be a number. The kernel then reads the right
    side as an array of the physical type of the left, with no check before, so
    a right side whose physical type differs panics. The output keeps the left
    type.

    Example:
        >>> FloorDivide().output(Data(DType("Int32")), Data(DType("Date")))
        Int32
        >>> FloorDivide().output(Data(DType("Int8")), Data(DType("Int16")))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        astrolabe._rules.failures.PolarsRefusal: panics: cannot read ...
    """

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output.

        Args:
            left: The dividend.
            right: The divisor.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: The left side is not a number, or the right side does not read as it.

        Example:
            >>> text = Data(DType("String"))
            >>> FloorDivide().output(text, text)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        names = {left.dtype.name, right.dtype.name}
        if names == {"Struct"}:
            return StructArithmetic().output(left, right, self)
        if "List" in names:
            return ListArithmetic(NumericOp.FLOOR_DIV).output(left, right)
        if "Array" in names:
            return ArrayArithmetic(NumericOp.FLOOR_DIV).output(left, right)
        if not Is.primitive_numeric(left.dtype):
            message = f"floor_div is not defined for {left.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if not right.dtype.physical().same(left.dtype):
            message = f"cannot read {right.dtype!r} as {left.dtype!r}"
            raise PolarsRefusal(Stop.PANICS, message)
        return left.dtype

    def right_named(self, left: Data, right: Data) -> bool:
        """Say whether the output series takes the name of the right input.

        Two structs keep the name that their arithmetic gives
        (`StructArithmetic`). Every other kernel names its output after the
        left input.

        Args:
            left: The left input.
            right: The right input.

        Returns:
            Whether the output takes the name of the right input.

        Example:
            >>> FloorDivide().right_named(Data(DType("Int8"), one_row=True), Data(DType("Int8")))
            False
            >>> one = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> two = Data(DType("Struct", ((("a", DType("Int8")), ("b", DType("Int8"))),)))
            >>> FloorDivide().right_named(one, two)
            True
        """
        if {left.dtype.name, right.dtype.name} == {"Struct"}:
            return StructArithmetic().right_named(left, right)
        return False


class Pow:
    """Give the type of `pow` of a base and an exponent.

    Both sides must be numbers. An integer base with a float exponent runs
    in the float type of the exponent. Otherwise the output keeps the type
    of the base: an integer exponent of an integer base casts to `UInt32`,
    and the exponent of a float base casts to the float type of the base.
    Those casts are strict, so a negative integer exponent raises, which is
    a fact of the values and not of the types.

    Example:
        >>> Pow().output(Data(DType("Int8")), Data(DType("Float32")))
        Float32
        >>> Pow().output(Data(DType("Float32")), Data(DType("Int64")))
        Float32
    """

    def output(self, base: Data, exponent: Data) -> DType:
        """Give the type of the output.

        Args:
            base: The base.
            exponent: The exponent.

        Returns:
            The type of the output.

        Raises:
            PolarsRefusal: A side is not a number.

        Example:
            >>> ten, small = Data(DType("Decimal", (10, 2))), Data(DType("Int8"))
            >>> Pow().output(ten, small)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            astrolabe._rules.failures.PolarsRefusal: InvalidOperationError: ...
        """
        if not (Is.primitive_numeric(base.dtype) and Is.primitive_numeric(exponent.dtype)):
            message = f"pow of {base.dtype!r} and {exponent.dtype!r}"
            raise PolarsRefusal(ErrorClass.INVALID_OPERATION, message)
        if Is.integer(base.dtype) and Is.float(exponent.dtype):
            return exponent.dtype
        return base.dtype
