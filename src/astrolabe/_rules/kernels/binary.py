"""The kernel of each binary operator of an expression, as polars picks it.

An expression `a + b` runs on the two columns that its inputs give. It takes an
in-place path for `+`, `-` and `*` of two columns. Otherwise it maps each
operator to a kernel of two series. The operator reads the data that arrives
after the cast pass, with no new check. This module maps each `Operator` of
polars to the kernel of the other modules of this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.kernels.arithmetic import (
    FloorDivide,
    OwnedArithmetic,
    SeriesArithmetic,
    TrueDivide,
)
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.logic import BitOp, BitOps, LogicalOps
from astrolabe._rules.kernels.dispatch import Data, RunCast
from astrolabe._rules.operators import NumericOp, Operator
from astrolabe._rules.releases import polars_releases

if TYPE_CHECKING:
    from astrolabe._rules.kernels.dispatch import BinaryKernel


class OperatorKernel:
    """Give the type of a binary operator of an expression at the data of its two inputs.

    Where both inputs are columns of rows, polars owns them and takes the
    in-place path of `+` and `-`, and of `*` of two numbers
    (`OwnedArithmetic`). An input that is one value, such as a literal or an
    aggregation, is not a column of rows. The operator then runs the kernel of
    two series.

    Example:
        >>> from astrolabe._rules.dtypes import DType
        >>> from astrolabe._rules.kernels.dispatch import Data
        >>> OperatorKernel().output(Operator.TRUE_DIVIDE, Data(DType("Int8")), Data(DType("Int8")))
        Float64
        >>> day, moment = Data(DType("Date")), Data(DType("Datetime", ("ms", None)))
        >>> OperatorKernel().output(Operator.EQ, day, moment)
        Boolean
    """

    COMPARISONS: ClassVar[dict[Operator, CompareOp]] = {
        Operator.EQ: CompareOp.EQUAL,
        Operator.EQ_VALIDITY: CompareOp.EQUAL_MISSING,
        Operator.NOT_EQ: CompareOp.NOT_EQUAL,
        Operator.NOT_EQ_VALIDITY: CompareOp.NOT_EQUAL_MISSING,
        Operator.LT: CompareOp.LT,
        Operator.LT_EQ: CompareOp.LT_EQ,
        Operator.GT: CompareOp.GT,
        Operator.GT_EQ: CompareOp.GT_EQ,
    }
    ARITHMETIC: ClassVar[dict[Operator, NumericOp]] = {
        Operator.PLUS: NumericOp.ADD,
        Operator.MINUS: NumericOp.SUB,
        Operator.MULTIPLY: NumericOp.MUL,
        Operator.RUST_DIVIDE: NumericOp.DIV,
        Operator.MODULUS: NumericOp.REM,
    }
    BITWISE: ClassVar[dict[Operator, BitOp]] = {
        Operator.AND: BitOp.AND,
        Operator.OR: BitOp.OR,
        Operator.XOR: BitOp.XOR,
    }
    LOGICAL: ClassVar[dict[Operator, BitOp]] = {
        Operator.LOGICAL_AND: BitOp.AND,
        Operator.LOGICAL_OR: BitOp.OR,
    }

    def __init__(self, *, owned: bool = True) -> None:
        """Keep whether both inputs are columns of rows that the operator owns.

        Args:
            owned: Whether both inputs are columns of rows.

        Example:
            >>> OperatorKernel(owned=False)._owned
            False
        """
        self._owned = owned

    def output(self, operator: Operator, left: Data, right: Data) -> DType:
        """Give the type of the output.

        It raises `PolarsRefusal` where the kernel fails on the data.

        Args:
            operator: The operator.
            left: The data of the left input.
            right: The data of the right input.

        Returns:
            The type of the output.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> large, span = Data(DType("Int64")), Data(DType("Duration", ("us",)))
            >>> OperatorKernel().output(Operator.FLOOR_DIVIDE, large, span)
            Int64
        """
        return self.kernel(operator, left, right).output(left, right)

    def kernel(self, operator: Operator, left: Data, right: Data) -> BinaryKernel:
        """Give the kernel that the operator runs on the data.

        Args:
            operator: The operator.
            left: The data of the left input.
            right: The data of the right input.

        Returns:
            The kernel.

        Example:
            >>> from astrolabe._rules.dtypes import DType
            >>> from astrolabe._rules.kernels.dispatch import Data
            >>> small = Data(DType("Int8"))
            >>> type(OperatorKernel().kernel(Operator.PLUS, small, small)).__name__
            'OwnedArithmetic'
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     type(OperatorKernel().kernel(Operator.PLUS, struct, small)).__name__
            'StructBesideNumberRun'
        """
        struct = self._struct(operator, left, right)
        if struct is not None:
            return struct
        if self._owned and operator in {Operator.PLUS, Operator.MINUS}:
            return OwnedArithmetic(self.ARITHMETIC[operator])
        numbers = Is.primitive_numeric(left.dtype) and Is.primitive_numeric(right.dtype)
        if self._owned and operator is Operator.MULTIPLY and numbers:
            return OwnedArithmetic(NumericOp.MUL)
        return self._applied(operator)

    @polars_releases
    def _struct(self, operator: Operator, left: Data, right: Data) -> BinaryKernel | None:
        """Give the kernel of arithmetic of a struct beside a number (`StructBesideNumberRun`).

        Args:
            operator: The operator.
            left: The data of the left input.
            right: The data of the right input.

        Returns:
            The kernel, or `None` where the pair is not a struct beside a number.

        Example:
            >>> from astrolabe._rules.releases import PolarsRelease
            >>> struct = Data(DType("Struct", ((("a", DType("Int8")),),)))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     type(OperatorKernel()._struct(Operator.PLUS, struct, Data(DType("Int8"))))
            <class 'astrolabe._rules.kernels.binary.StructBesideNumberRun'>
            >>> number = Data(DType("Int8"))
            >>> with PolarsRelease.pretend("2.0.0"):
            ...     compared = OperatorKernel()._struct(Operator.EQ, struct, number)
            ...     left = OperatorKernel()._struct(Operator.PLUS, number, struct)
            >>> compared is None, type(left).__name__
            (True, 'StructBesideNumberRun')
        """
        if not operator.is_arithmetic():
            return None
        names = (left.dtype.name, right.dtype.name)
        if names[0] == "Struct" and Is.primitive_numeric(right.dtype):
            return StructBesideNumberRun(operator, number_left=False)
        if names[1] == "Struct" and Is.primitive_numeric(left.dtype):
            return StructBesideNumberRun(operator, number_left=True)
        return None

    @_struct.before("2.0.0")
    def _no_struct(self, operator: Operator, left: Data, right: Data) -> BinaryKernel | None:
        """Give no kernel here: the older releases run a struct beside a number as other pairs.

        Args:
            operator: The operator.
            left: The data of the left input.
            right: The data of the right input.

        Returns:
            `None`.

        Example:
            >>> small = Data(DType("Int8"))
            >>> OperatorKernel()._no_struct(Operator.PLUS, small, small)
        """
        del operator, left, right
        return None

    def _applied(self, operator: Operator) -> BinaryKernel:
        """Give the kernel of two series that the operator runs where it takes no in-place path.

        Args:
            operator: The operator.

        Returns:
            The kernel.

        Example:
            >>> type(OperatorKernel()._applied(Operator.XOR)).__name__
            'BitOps'
        """
        if operator in self.COMPARISONS:
            return Comparison(self.COMPARISONS[operator])
        if operator in self.ARITHMETIC:
            return SeriesArithmetic(self.ARITHMETIC[operator])
        if operator in self.BITWISE:
            return BitOps(self.BITWISE[operator])
        if operator in self.LOGICAL:
            return LogicalOps(self.LOGICAL[operator])
        if operator is Operator.TRUE_DIVIDE:
            return TrueDivide()
        return FloorDivide()


class StructBesideNumberRun:
    """Run arithmetic of a struct beside a number field by field, as polars does from 2.0.0.

    Each field meets the number, cast to the leaf type of the field. This cast
    is not strict. A struct field, a field of the type of the number and a
    `Duration` field of a product meet the number as it is. The output is a
    struct of the results, with the names of the fields.

    Example:
        >>> struct = Data(DType("Struct", ((("a", DType("Int16")),),)))
        >>> run = StructBesideNumberRun(Operator.MINUS, number_left=False)
        >>> run.output(struct, Data(DType("Int8")))
        Struct({'a': Int16})
    """

    def __init__(self, operator: Operator, *, number_left: bool) -> None:
        """Keep the operator and the side of the number.

        Args:
            operator: The operator.
            number_left: Whether the number is the left input.

        Example:
            >>> StructBesideNumberRun(Operator.PLUS, number_left=True)._number_left
            True
        """
        self._operator = operator
        self._number_left = number_left

    def output(self, left: Data, right: Data) -> DType:
        """Give the type of the output struct.

        It raises `PolarsRefusal` where the operator fails on a field.

        Args:
            left: The data of the left input.
            right: The data of the right input.

        Returns:
            The struct of the results.

        Example:
            >>> struct = Data(DType("Struct", ((("a", DType("Float64")),),)))
            >>> run = StructBesideNumberRun(Operator.TRUE_DIVIDE, number_left=True)
            >>> run.output(Data(DType("Int8")), struct)
            Struct({'a': Float64})
        """
        struct, number = (right, left) if self._number_left else (left, right)
        fields = tuple(
            (name, self._field(struct.field(dtype), number))
            for name, dtype in struct.dtype.fields()
        )
        return DType("Struct", (fields,))

    def _field(self, field: Data, number: Data) -> DType:
        """Give the type of the operator on one field and the number.

        Args:
            field: The data of the field.
            number: The data of the number.

        Returns:
            The type of the result.

        Example:
            >>> run = StructBesideNumberRun(Operator.MULTIPLY, number_left=False)
            >>> run._field(Data(DType("Duration", ("us",))), Data(DType("Int8")))
            Duration(time_unit='us')
        """
        leaf = field.dtype.leaf()
        kept = (
            field.dtype.name == "Struct"
            or (field.dtype.name == "Duration" and self._operator is Operator.MULTIPLY)
            or number.dtype.same(leaf)
        )
        cast = number if kept else RunCast().of(number, leaf)
        pair = (cast, field) if self._number_left else (field, cast)
        return OperatorKernel(owned=False).output(self._operator, *pair)
