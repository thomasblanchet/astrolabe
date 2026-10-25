"""Compute the value of a constant expression that a user wrote, such as `2**10`.

Polars receives a value and never the text that wrote it. In `pl.col("a") + 2**10`, Python
computes 1024 first, and polars gives the result the type that the value 1024 needs. The
checker reads such a value from the abstract syntax tree (AST) of the call, so it must compute
the same arithmetic that Python computes.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import FloatExpr, IntExpr, MemberExpr, NameExpr, OpExpr, UnaryExpr, Var
from mypy.types import Instance, get_proper_type

if TYPE_CHECKING:
    from collections.abc import Callable

    from mypy.nodes import Expression
    from mypy.types import Type as MypyType

type Number = int | float | bool


class ConstantFold:
    """Compute the number that an expression of literals gives, as Python computes it.

    Polars picks the data type of a Python number from its value: `pl.lit(200)` and
    `pl.lit(2**40)` give different integer types. A user often writes a number as arithmetic,
    for example `2**10`, `-(2**63)` or `1.5 * 2`. mypy types such an expression as a plain `int`
    or `float` and keeps no value. This class reads the AST and computes the value.

    It reads int, float and bool literals, and names declared `Final` with such a value. It
    applies the unary operators `-`, `+` and `~`, and the binary arithmetic and bit operators.
    It follows the rules of Python, so `True + 1` is the int 2 and `4 / 2` is the float 2.0.

    The computation must stay cheap, because it runs inside mypy. An integer of more than
    `MAX_BITS` bits gives no value, and the class refuses a power, a product or a left shift
    whose result would pass that size before it computes it. An operation that raises in
    Python, for example a division by zero, gives no value too.

    Example:
        >>> fold = ConstantFold()
        >>> fold.value(OpExpr("**", IntExpr(2), IntExpr(10)))
        1024
        >>> fold.value(UnaryExpr("-", OpExpr("**", IntExpr(2), IntExpr(63))))
        -9223372036854775808
        >>> fold.value(OpExpr("*", FloatExpr(1.5), IntExpr(2)))
        3.0
        >>> fold.value(OpExpr("**", IntExpr(2), OpExpr("**", IntExpr(10), IntExpr(10)))) is None
        True
    """

    # The largest integer that a computation keeps. polars holds no integer of more than 128
    # bits, so this leaves room for a step that grows and then shrinks the value.
    MAX_BITS: ClassVar[int] = 1024

    _BOOLEANS: ClassVar[dict[str, bool]] = {"builtins.True": True, "builtins.False": False}

    _UNARY: ClassVar[dict[str, Callable[[Number], Number]]] = {
        "-": operator.neg,
        "+": operator.pos,
    }

    _BINARY: ClassVar[dict[str, Callable[[Number, Number], Number]]] = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
        "<<": operator.lshift,
        ">>": operator.rshift,
        "&": operator.and_,
        "|": operator.or_,
        "^": operator.xor,
    }

    def value(self, node: Expression) -> Number | None:
        """Compute the value of one expression.

        Args:
            node: The AST node.

        Returns:
            The int, float or bool value, or None if the node is not a constant expression of
            numbers, or if its value is too large or raises.

        Example:
            >>> fold = ConstantFold()
            >>> fold.value(IntExpr(5))
            5
            >>> fold.value(OpExpr("/", IntExpr(4), IntExpr(2)))
            2.0
            >>> fold.value(OpExpr("//", IntExpr(1), IntExpr(0))) is None
            True
            >>> fold.value(StrExpr("x")) is None
            True
        """
        if isinstance(node, (IntExpr, FloatExpr)):
            return node.value
        if isinstance(node, (NameExpr, MemberExpr)):
            return self._named(node)
        if isinstance(node, UnaryExpr):
            return self._unary(node)
        if isinstance(node, OpExpr):
            return self._binary(node)
        return None

    def _named(self, node: NameExpr | MemberExpr) -> Number | None:
        """Read `True`, `False`, or a name declared `Final` with a number value.

        mypy keeps the value of a `Final` name on its declaration, so the name reads as the
        value inside a larger expression too, as in `2**SHIFT`. A name at module or class level
        holds the value itself. A name inside a function holds it on its type.

        Args:
            node: The name, or an attribute such as `module.NAME`.

        Returns:
            The value, or None for any other name.

        Example:
            >>> true = NameExpr("True")
            >>> true.fullname = "builtins.True"
            >>> ConstantFold()._named(true)
            True
            >>> declared = Var("SHIFT")
            >>> declared.is_final, declared.final_value = True, 10
            >>> name = NameExpr("SHIFT")
            >>> name.node = declared
            >>> ConstantFold()._named(name)
            10
            >>> local = Var("LOCAL", literal_receiver_instance(new_type_fixture(), 3))
            >>> local.is_final = True
            >>> name.node = local
            >>> ConstantFold()._named(name)
            3
            >>> local.type = None
            >>> ConstantFold()._named(name) is None
            True
            >>> ConstantFold()._named(NameExpr("other")) is None
            True
        """
        if node.fullname in self._BOOLEANS:
            return self._BOOLEANS[node.fullname]
        declared = node.node
        if not isinstance(declared, Var) or not declared.is_final:
            return None
        final = declared.final_value
        if final is None:
            final = self._typed_value(declared.type)
        return final if isinstance(final, (int, float)) else None

    def _typed_value(self, declared: MypyType | None) -> object:
        """Read the value that mypy keeps on the type of a `Final` name.

        Args:
            declared: The type of the name, or None where mypy gives it none.

        Returns:
            The value, or None where the type keeps no value.

        Example:
            >>> ConstantFold()._typed_value(literal_receiver_instance(new_type_fixture(), 3))
            3
            >>> ConstantFold()._typed_value(None) is None
            True
        """
        proper = get_proper_type(declared)
        if not isinstance(proper, Instance) or proper.last_known_value is None:
            return None
        return proper.last_known_value.value

    def _unary(self, node: UnaryExpr) -> Number | None:
        """Compute `-x`, `+x` or `~x`.

        Python deprecates `~` on a bool, so the class reads `~` on an int only.

        Args:
            node: The unary expression.

        Returns:
            The value, or None if the operand has no value or the operator does not apply.

        Example:
            >>> ConstantFold()._unary(UnaryExpr("~", IntExpr(5)))
            -6
            >>> ConstantFold()._unary(UnaryExpr("-", NameExpr("x"))) is None
            True
            >>> ConstantFold()._unary(UnaryExpr("~", FloatExpr(1.5))) is None
            True
        """
        operand = self.value(node.expr)
        if operand is None:
            return None
        if node.op == "~":
            return self._inverted(operand)
        unary = self._UNARY.get(node.op)
        return None if unary is None else unary(operand)

    def _inverted(self, operand: Number) -> int | None:
        """Compute `~x`, which Python defines on an int and deprecates on a bool.

        Args:
            operand: The value of the operand.

        Returns:
            The value, or None for a float and a bool.

        Example:
            >>> ConstantFold()._inverted(5), ConstantFold()._inverted(True)
            (-6, None)
        """
        if isinstance(operand, bool) or not isinstance(operand, int):
            return None
        return ~operand

    def _binary(self, node: OpExpr) -> Number | None:
        """Compute a binary arithmetic or bit operation.

        Args:
            node: The binary expression.

        Returns:
            The value, or None if an operand has no value, if the operator is not arithmetic,
            if the result is too large, or if Python raises.

        Example:
            >>> ConstantFold()._binary(OpExpr("<<", IntExpr(1), IntExpr(9)))
            512
            >>> ConstantFold()._binary(OpExpr("and", IntExpr(1), IntExpr(9))) is None
            True
            >>> ConstantFold()._binary(OpExpr("&", FloatExpr(1.5), IntExpr(1))) is None
            True
            >>> ConstantFold()._binary(OpExpr("**", FloatExpr(-8.0), FloatExpr(0.5))) is None
            True
        """
        left = self.value(node.left)
        right = self.value(node.right)
        binary = self._BINARY.get(node.op)
        if left is None or right is None or binary is None:
            return None
        if not self._fits(node.op, left, right):
            return None
        try:
            result = binary(left, right)
        except (ArithmeticError, TypeError, ValueError):
            return None
        return result if self._kept(result) else None

    def _fits(self, op: str, left: Number, right: Number) -> bool:
        """Answer whether an integer operation keeps its result inside `MAX_BITS` bits.

        A power, a product and a left shift of integers can grow without limit, and Python
        computes them exactly. So the class estimates the size of the result before it
        computes it. Every other operation, and every operation with a float, stays cheap.

        Args:
            op: The operator.
            left: The value on the left.
            right: The value on the right.

        Returns:
            False where the result would pass the limit.

        Example:
            >>> fold = ConstantFold()
            >>> fold._fits("**", 2, 10), fold._fits("**", 2, 10**10)
            (True, False)
            >>> fold._fits("<<", 1, 2000), fold._fits("*", 2**600, 2**600)
            (False, False)
            >>> fold._fits("**", 2.0, 10**10), fold._fits("+", 2**1000, 2**1000)
            (True, True)
        """
        if isinstance(left, float) or isinstance(right, float):
            return True
        size = abs(left).bit_length()
        if op == "**":
            return size * right <= self.MAX_BITS
        if op == "<<":
            return size + right <= self.MAX_BITS
        if op == "*":
            return size + abs(right).bit_length() <= self.MAX_BITS
        return True

    def _kept(self, result: object) -> bool:
        """Answer whether a result is a number that the class keeps.

        Args:
            result: The result of one operation.

        Returns:
            True for a float, a bool and an int inside `MAX_BITS` bits. False for a complex
            number, which a power of a negative float can give.

        Example:
            >>> fold = ConstantFold()
            >>> fold._kept(2**10), fold._kept(2**2000), fold._kept(1.5), fold._kept(1j)
            (True, False, True, False)
        """
        if type(result) is int:
            return result.bit_length() <= self.MAX_BITS
        return type(result) in {float, bool}
