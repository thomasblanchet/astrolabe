"""Read a dictionary literal from field names to data types, for a checked call.

`pl.Struct({"p": pl.Int8})` and `pl.struct(..., schema={"p": pl.Int8})`
both name fields and their data types in one dictionary. A check of
either call needs each name and the rendering of each data type, that
is, the text that polars prints for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mypy.nodes import DictExpr, StrExpr

from astrolabe._hooks.calls.dtype_expr import DtypeExpr
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from mypy.nodes import Expression


class DtypeMapping:
    """Read the entries of a dictionary from field names to data types.

    The dictionary must be a literal in the call, with a string key for each
    entry. A class with parameters, such as `pl.Datetime`, reads as the
    instance that polars builds for it, and a data type that the checker
    cannot read reads as `Unchecked`, so the names of the fields stay known.
    Any other argument, for example a variable or a `**` entry, gives no
    entries.

    Example:
        >>> DtypeMapping().entries(None, StrExpr("a")) is None
        True
    """

    UNCHECKED: ClassVar[str] = "Unchecked"

    @propagates(None)
    def entries(self, api: object, node: Expression) -> list[tuple[str, str]] | None:
        """Give each field name with the rendering of its data type, in order.

        Args:
            api: The mypy API of the type-checking phase, which resolves a
                data type.
            node: The node of the dictionary.

        Returns:
            The entries, or None if the node is not such a dictionary. None also if one field
            name is not a string literal, for example a `**` entry. A data type that the
            checker cannot read, as a variable of the type `pl.DataType`, is `Unchecked`.
        """
        mapping = not_none(node if isinstance(node, DictExpr) else None)
        entries: list[tuple[str, str]] = []
        for key, value in mapping.items:
            name = not_none(key if isinstance(key, StrExpr) else None).value
            rendering = DtypeExpr().rendering(api, value) or self.UNCHECKED
            entries.append((name, rendering))
        return entries

    def values(self, node: Expression) -> list[Expression]:
        """Give the node of each data type in the dictionary.

        Args:
            node: The node of the dictionary.

        Returns:
            The nodes of the values, or no node if the argument is not a
            dictionary literal.

        Example:
            >>> DtypeMapping().values(StrExpr("a"))
            []
        """
        if not isinstance(node, DictExpr):
            return []
        return [value for _, value in node.items]
