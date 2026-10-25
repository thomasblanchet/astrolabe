"""Give a power of number literals a type, where the type stub of Python gives `Any`.

Python computes `2**(N + 1)` before polars sees it, and polars picks the data type of the
result from the value. mypy types such a power as `Any`, because the stub of `int.__pow__`
gives `Any` for an exponent that is not a literal. mypy then calls no hook for an operation
with that operand, so `pl.col("u") + 2**(N + 1)` has no type. This module gives the power the
type `int` or `float` again, and the hook of the operation then reads the value from the
abstract syntax tree (AST).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import Expression
from mypy.types import AnyType, get_proper_type

from astrolabe._hooks.calls.constant_fold import ConstantFold
from astrolabe._hooks.dispatch.generic import Target

if TYPE_CHECKING:
    from mypy.nodes import Context
    from mypy.plugin import MethodContext
    from mypy.types import Type as MypyType


class ConstantPowerHook:
    """Type a power of number literals, such as `2**(N + 1)`, by the class of its value.

    The hook serves `int.__pow__` and `float.__pow__`. It changes a result only where mypy
    gives `Any` and `ConstantFold` computes the value. Any other power keeps the type that
    mypy gives it, for example `2**n` with an `int` variable `n`, whose value is not known.
    """

    _CLASSES: ClassVar[dict[type, str]] = {
        int: Target.of(int).fullname,
        float: Target.of(float).fullname,
    }

    def __call__(self, ctx: MethodContext) -> MypyType:
        """Give the type of one power.

        Args:
            ctx: The context of the call.

        Returns:
            `int` or `float` for a power whose value the checker computes, and else the type
            that mypy gives.
        """
        if not isinstance(get_proper_type(ctx.default_return_type), AnyType):
            return ctx.default_return_type
        fullname = self.class_of(ctx.context)
        if fullname is None:
            return ctx.default_return_type
        return ctx.api.named_generic_type(fullname, [])

    def class_of(self, context: Context) -> str | None:
        """Give the full name of the class of the value that a power computes.

        Args:
            context: The node of the power, for example the binary expression `2**(N + 1)`.

        Returns:
            `builtins.int` or `builtins.float`, or None where the node is not a constant
            expression of numbers.

        Example:
            >>> from mypy.nodes import IntExpr, OpExpr
            >>> power = OpExpr("**", IntExpr(2), OpExpr("+", IntExpr(9), IntExpr(1)))
            >>> ConstantPowerHook().class_of(power)
            'builtins.int'
            >>> ConstantPowerHook().class_of(OpExpr("**", IntExpr(2), IntExpr(-1)))
            'builtins.float'
            >>> ConstantPowerHook().class_of(OpExpr("**", IntExpr(2), NameExpr("n"))) is None
            True
        """
        if not isinstance(context, Expression):  # pragma: no cover
            # mypy passes the node of the power, which is an expression, at each call.
            return None
        value = ConstantFold().value(context)
        return self._CLASSES.get(type(value))
