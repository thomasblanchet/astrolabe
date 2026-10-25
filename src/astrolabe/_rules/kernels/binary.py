"""The kernel of each binary operator of an expression, as polars picks it.

An expression `a + b` runs on the two columns that its inputs give. It takes an
in-place path for `+`, `-` and `*` of two columns. Otherwise it maps each
operator to a kernel of two series. The operator reads the data that arrives
after the cast pass, with no new check. This module maps each `Operator` of
polars to the kernel of the other modules of this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._rules.dtypes import Is
from astrolabe._rules.kernels.arithmetic import (
    FloorDivide,
    OwnedArithmetic,
    SeriesArithmetic,
    TrueDivide,
)
from astrolabe._rules.kernels.comparison import CompareOp, Comparison
from astrolabe._rules.kernels.logic import BitOp, BitOps, LogicalOps
from astrolabe._rules.operators import NumericOp, Operator

if TYPE_CHECKING:
    from astrolabe._rules.dtypes import DType
    from astrolabe._rules.kernels.dispatch import BinaryKernel, Data


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
        """
        if self._owned and operator in {Operator.PLUS, Operator.MINUS}:
            return OwnedArithmetic(self.ARITHMETIC[operator])
        numbers = Is.primitive_numeric(left.dtype) and Is.primitive_numeric(right.dtype)
        if self._owned and operator is Operator.MULTIPLY and numbers:
            return OwnedArithmetic(NumericOp.MUL)
        return self._applied(operator)

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
