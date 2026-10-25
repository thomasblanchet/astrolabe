"""Give the remainder parameter of each frame class the default `None`.

With this default, a frame type of one argument names a closed frame, in a
value as in an annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import NoneType, TypeVarType

from astrolabe._hooks.checked import checked_cast

if TYPE_CHECKING:
    from mypy.plugin import ClassDefContext


class RemainderDefault:
    """Let a frame type written as a value take one type argument, as an annotation does.

    `TypedDataFrame` and `TypedLazyFrame` have two type parameters: the
    schema and the remainder. In an annotation, the type-analysis hook of
    the frame class reads `TypedDataFrame[Trades]` as a closed frame, with
    `None` for the remainder. A frame type written as a value, as in
    `check_schema(raw, TypedDataFrame[Trades])`, does not reach that hook,
    and mypy checks its arguments against the parameters of the class. With
    no default, one argument for two parameters is an error. The source of
    the class cannot declare the default, because that syntax needs Python
    3.13. So this hook gives the remainder parameter the default `None`
    while mypy reads the class.

    It is a mypy class hook in the plugin layer, and it runs once for each
    frame class, during semantic analysis.
    """

    def __call__(self, ctx: ClassDefContext) -> None:
        """Give the remainder parameter of the frame class the default `None`.

        Args:
            ctx: The context that mypy gives for the definition of the class.
        """
        type_vars = ctx.cls.type_vars
        rest = checked_cast(type_vars[1], TypeVarType)
        type_vars[1] = rest.copy_modified(default=NoneType())
