"""The expression type of a bare value that a frame operation takes as a column.

A frame operation reads a bare number or Boolean, as in `select(x=1)`, as the
literal `pl.lit(1)`. Its expression type reads no column, gives one fixed data
type, and has the output name `literal`, as in polars. This module is part of
the engine, which knows nothing about mypy.
"""

from __future__ import annotations

from astrolabe._hooks.expression.character import LITERAL
from astrolabe._hooks.expression.quadruple import ExpressionType
from astrolabe._hooks.expression.table import Table


class LiteralRoot:
    """Give the expression type of a bare value that a frame operation reads as `pl.lit(value)`.

    Example:
        >>> LiteralRoot().of("Int32").name
        'literal'
    """

    def of(self, rendering: str) -> ExpressionType:
        """Give the expression type of a literal of one data type.

        Args:
            rendering: The data type of the value, as polars prints it.

        Returns:
            The expression type. It reads no column, gives the fixed
            data type, and has the output name `literal`, as in polars.

        Example:
            >>> expr = LiteralRoot().of("Int32")
            >>> expr.name, expr.character.label
            ('literal', 'literal')
        """
        return ExpressionType(
            frozenset(),
            Table.constant((), (), rendering),
            "literal",
            LITERAL,
        )
