"""Check the operators of `astrolabe._rules.operators` against the plans of polars.

The plan traverser of polars shows the operator of each node `BinaryExpr`, so
each test builds a node and reads its operator: an operator of Python, a method
such as `eq_missing`, or `all_horizontal` and `any_horizontal` of two inputs,
which become the logical operators. The cast pass moves a literal on the left
of a comparison to the right, and it then swaps the operator
(`Operator.swapped`).
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pytest

from astrolabe._rules.operators import Operator

if TYPE_CHECKING:
    from collections.abc import Callable


class RootOperator:
    """Read the operator of the root node of a `select` with the plan traverser of polars.

    Example:
        >>> RootOperator().of({"x": pl.Int8()}, pl.col("x") + pl.col("x"))
        'Plus'
    """

    FLAGS: ClassVar[pl.QueryOptFlags] = pl.QueryOptFlags.none()

    def of(self, schema: dict[str, pl.DataType], expression: pl.Expr) -> str:
        """Give the name of the operator.

        Args:
            schema: The columns of the frame.
            expression: The expression, a binary node.

        Returns:
            The name of the operator in polars, for example `Plus`.

        Example:
            >>> RootOperator().of({"x": pl.Boolean()}, pl.all_horizontal("x", "x"))
            'LogicalAnd'
        """
        query = pl.LazyFrame(schema=schema).select(expression.alias("o"))
        flags = self.FLAGS._pyoptflags  # ruff: ignore[private-member-access] -- for the traverser
        view: Any = query._ldf.with_optimizations(flags).visit()  # ruff: ignore[private-member-access]
        root = view.view_expression(view.view_current_node().expr[0].node)
        return str(root.op).rsplit(".", 1)[-1]


class Builds:
    """Hold the expressions of polars that build each operator of two columns.

    `RustDivide` has no call of Python.

    Example:
        >>> len(Builds.OPERATORS), len(Builds.COMPARISONS)
        (19, 8)
    """

    OPERATORS: ClassVar[dict[Operator, Callable[[pl.Expr, pl.Expr], pl.Expr]]] = {
        Operator.PLUS: operator.add,
        Operator.MINUS: operator.sub,
        Operator.MULTIPLY: operator.mul,
        Operator.TRUE_DIVIDE: operator.truediv,
        Operator.FLOOR_DIVIDE: operator.floordiv,
        Operator.MODULUS: operator.mod,
        Operator.AND: operator.and_,
        Operator.OR: operator.or_,
        Operator.XOR: operator.xor,
        Operator.EQ: operator.eq,
        Operator.NOT_EQ: operator.ne,
        Operator.LT: operator.lt,
        Operator.LT_EQ: operator.le,
        Operator.GT: operator.gt,
        Operator.GT_EQ: operator.ge,
        Operator.EQ_VALIDITY: pl.Expr.eq_missing,
        Operator.NOT_EQ_VALIDITY: pl.Expr.ne_missing,
        Operator.LOGICAL_AND: pl.all_horizontal,
        Operator.LOGICAL_OR: pl.any_horizontal,
    }
    COMPARISONS: ClassVar[list[Operator]] = [op for op in OPERATORS if op.is_comparison()]
    LOGICAL: ClassVar[frozenset[Operator]] = frozenset({Operator.LOGICAL_AND, Operator.LOGICAL_OR})

    def schema(self, op: Operator) -> dict[str, pl.DataType]:
        """Give the columns that the node of an operator reads.

        Args:
            op: The operator.

        Returns:
            Two columns of `Boolean` for a logical operator, and of `Int8` otherwise.

        Example:
            >>> Builds().schema(Operator.LOGICAL_OR)
            {'a': Boolean, 'b': Boolean}
        """
        dtype = pl.Boolean() if op in self.LOGICAL else pl.Int8()
        return {"a": dtype, "b": dtype}


class TestOperators:
    """Check each operator against the operator that the plan of polars shows."""

    @pytest.mark.parametrize("op", list(Builds.OPERATORS), ids=lambda op: op.value)
    def test_operator_has_the_name_of_polars(self, op: Operator) -> None:
        """The node that polars builds for the operator has the operator of the same name."""
        expression = Builds.OPERATORS[op](pl.col("a"), pl.col("b"))
        assert RootOperator().of(Builds().schema(op), expression) == op.value

    @pytest.mark.parametrize("op", Builds.COMPARISONS, ids=lambda op: op.value)
    def test_comparison_with_a_literal_on_the_left_takes_the_swapped_operator(
        self,
        op: Operator,
    ) -> None:
        """The cast pass moves a literal to the right of a comparison and swaps the operator."""
        expression = Builds.OPERATORS[op](pl.lit(1), pl.col("x"))
        swapped = op.swapped()
        assert swapped is not None
        assert RootOperator().of({"x": pl.UInt8()}, expression) == swapped.value
