"""The type of a declared expression: the columns it reads, its map, its name and its character.

A user declares the type of an expression that a function takes or returns,
for example with `column("price", pl.Float64)` or `typed_expr(...)`. That
declaration states four facts: the columns that the expression reads, a map
from the data types of those columns to the data type of the result, the
name of the result, and its character (row by row, aggregating, or a fixed
value). These four facts are all that a frame operation such as `select`
can observe about a declared expression.

This module is part of the engine, the pure half of the plugin that knows
nothing about mypy. Code that reads a declaration, code that encodes it for
mypy, and code that applies it to a frame all read this one value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrolabe._hooks.expression.character import Character
    from astrolabe._hooks.expression.table import Table


@dataclass(frozen=True)
class ExpressionType:
    """The type of one declared expression: needs, map, output name and character.

    The needs are every column that the expression reads. A frame must hold
    each of them, or polars raises. The table has one axis for each column
    that decides the output data type, and a cell for each combination of
    input kinds (a kind is a data type without its parameters, for example
    `Datetime` for every time unit).

    Example:
        >>> from astrolabe._hooks.expression.character import ELEMENTWISE
        >>> from astrolabe._hooks.expression.outcome import Axis
        >>> from astrolabe._hooks.expression.table import Table
        >>> identity = Table.from_cells(("a",), {("Int64",): Axis("a")})
        >>> expr = ExpressionType(frozenset({"a"}), identity, "a", ELEMENTWISE)
        >>> expr.name
        'a'
    """

    needs: frozenset[str]
    table: Table
    name: str
    character: Character

    def renamed(self, name: str) -> ExpressionType:
        """Give this type under a new output name, as a keyword of a frame operation gives it.

        A rename reads no column and changes no data type. It changes only
        the output name.

        Args:
            name: The new output name.

        Returns:
            The same type with the new output name.

        Example:
            >>> from astrolabe._hooks.expression.character import LITERAL
            >>> from astrolabe._hooks.expression.table import Table
            >>> lit = Table.constant((), (), "Int32")
            >>> value = ExpressionType(frozenset(), lit, "literal", LITERAL)
            >>> value.renamed("z").name
            'z'
        """
        return replace(self, name=name)
