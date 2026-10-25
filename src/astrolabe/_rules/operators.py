"""The binary operators of polars, and the arithmetic operations of its kernels.

An expression such as `a + b` is a node `BinaryExpr` with one `Operator`. The
planner, the cast pass and the kernels all read that operator. The tests
`is_comparison`, `is_bitwise` and `is_arithmetic` decide which rules apply.
`swapped` lets the cast pass move a literal to the right side. The kernels of
polars name their arithmetic with another enum, `NumericOp`, which also joins
the leaf types of list arithmetic. This module holds both, so that every rule
names an operator in the same way.
"""

from __future__ import annotations

import enum
from typing import ClassVar

from astrolabe._rules.dtypes import DType, Is
from astrolabe._rules.supertype import Supertype


class Operator(enum.Enum):
    """Name the binary operators of polars, with the tests that its rules read.

    The values are the names of the operators that the plan traverser of polars
    shows. Python `&`, `|` and `^` give `And`, `Or` and `Xor`. `eq_missing` and
    `ne_missing` give `EqValidity` and `NotEqValidity`. `RustDivide` is the
    division of Rust, which no Python call builds. `LogicalAnd` and `LogicalOr`
    are what `all_horizontal` and `any_horizontal` of two inputs become. The
    tests `is_comparison`, `is_bitwise` and `is_arithmetic` mirror the polars
    functions of the same names, so `LogicalAnd` and `LogicalOr` count as
    arithmetic there.

    Example:
        >>> Operator.LT.swapped(), Operator.LOGICAL_OR.is_arithmetic()
        (<Operator.GT: 'Gt'>, True)
    """

    EQ = "Eq"
    EQ_VALIDITY = "EqValidity"
    NOT_EQ = "NotEq"
    NOT_EQ_VALIDITY = "NotEqValidity"
    LT = "Lt"
    LT_EQ = "LtEq"
    GT = "Gt"
    GT_EQ = "GtEq"
    PLUS = "Plus"
    MINUS = "Minus"
    MULTIPLY = "Multiply"
    RUST_DIVIDE = "RustDivide"
    TRUE_DIVIDE = "TrueDivide"
    FLOOR_DIVIDE = "FloorDivide"
    MODULUS = "Modulus"
    AND = "And"
    OR = "Or"
    XOR = "Xor"
    LOGICAL_AND = "LogicalAnd"
    LOGICAL_OR = "LogicalOr"

    def is_comparison(self) -> bool:
        """Say whether the operator compares, the validity forms included.

        Returns:
            Whether it is `==`, `!=`, `<`, `<=`, `>`, `>=`, `eq_missing` or `ne_missing`.

        Example:
            >>> Operator.EQ_VALIDITY.is_comparison()
            True
        """
        return self in _OperatorSets.COMPARISONS

    def is_bitwise(self) -> bool:
        """Say whether the operator is `&`, `|` or `^`.

        Returns:
            Whether it is bitwise.

        Example:
            >>> Operator.XOR.is_bitwise(), Operator.LOGICAL_AND.is_bitwise()
            (True, False)
        """
        return self in _OperatorSets.BITWISE

    def is_comparison_or_bitwise(self) -> bool:
        """Say whether the operator compares or is bitwise.

        Returns:
            Whether it is one or the other.

        Example:
            >>> Operator.AND.is_comparison_or_bitwise()
            True
        """
        return self.is_comparison() or self.is_bitwise()

    def is_arithmetic(self) -> bool:
        """Say whether the operator is neither a comparison nor bitwise.

        Returns:
            Whether it is arithmetic in the sense of polars.

        Example:
            >>> Operator.MODULUS.is_arithmetic()
            True
        """
        return not self.is_comparison_or_bitwise()

    def swapped(self) -> Operator | None:
        """Give the operator for the inputs in the other order.

        It gives `None` where no operator gives the same result, as the polars
        function that swaps the operands does.

        Returns:
            The operator, or `None`.

        Example:
            >>> Operator.MINUS.swapped() is None
            True
        """
        if (  # pragma: no cover -- only a comparison asks for its swap
            self in _OperatorSets.ASYMMETRIC
        ):
            return None
        return _OperatorSets.MIRRORS.get(self, self)


class _OperatorSets:
    """Hold the sets of operators that the tests of `Operator` read.

    An enum cannot hold a class attribute of its own type, so the sets live here.

    Example:
        >>> Operator.GT in _OperatorSets.COMPARISONS
        True
    """

    COMPARISONS: ClassVar[frozenset[Operator]] = frozenset({
        Operator.EQ,
        Operator.NOT_EQ,
        Operator.LT,
        Operator.LT_EQ,
        Operator.GT,
        Operator.GT_EQ,
        Operator.EQ_VALIDITY,
        Operator.NOT_EQ_VALIDITY,
    })
    BITWISE: ClassVar[frozenset[Operator]] = frozenset({Operator.AND, Operator.OR, Operator.XOR})
    ASYMMETRIC: ClassVar[frozenset[Operator]] = frozenset({
        Operator.MINUS,
        Operator.RUST_DIVIDE,
        Operator.TRUE_DIVIDE,
        Operator.FLOOR_DIVIDE,
        Operator.MODULUS,
    })
    MIRRORS: ClassVar[dict[Operator, Operator]] = {
        Operator.LT: Operator.GT,
        Operator.GT: Operator.LT,
        Operator.LT_EQ: Operator.GT_EQ,
        Operator.GT_EQ: Operator.LT_EQ,
    }


class NumericOp(enum.Enum):
    """Name the arithmetic operations of the kernels of polars.

    The value is the name that polars puts in its messages. The kernels of two
    series and of nested values read it, and the planner reads it where it
    joins the leaf types of list arithmetic.

    Example:
        >>> NumericOp.FLOOR_DIV.value
        'floor_div'
    """

    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    REM = "rem"
    FLOOR_DIV = "floor_div"

    def leaf_supertype(self, left: DType, right: DType) -> DType:
        """Give the type of the leaves of the output, as the list arithmetic of polars joins them.

        A leaf is the type under every `List` and `Array`. The leaves join into
        their supertype. A division gives a float: the supertype where it is a
        float, and `Float64` otherwise. Two `Boolean` leaves give the index
        type of polars, `UInt32`, for every other operation.

        It raises `PolarsRefusal` where the two leaf types have no supertype,
        as `Supertype.require` does.

        Args:
            left: The leaf type of the left input.
            right: The leaf type of the right input.

        Returns:
            The leaf type of the output.

        Example:
            >>> NumericOp.DIV.leaf_supertype(DType("Int8"), DType("Int16"))
            Float64
            >>> NumericOp.SUB.leaf_supertype(DType("Boolean"), DType("Boolean"))
            UInt32
        """
        joined = Supertype().require(left, right)
        if self is NumericOp.DIV:
            return joined if Is.float(joined) else DType("Float64")
        if left.name == right.name == "Boolean":
            return DType.idx()
        return joined
