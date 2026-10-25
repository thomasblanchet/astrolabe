"""The one order of the read columns that every table of the engine uses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class AxisOrder:
    """Order the read columns of a table: shorter names first, then alphabetically.

    The engine types an expression with a table. The table has one axis
    for each column that the expression reads, and it maps the data types
    of those columns to the data type that the expression gives. When
    two axes describe the same cell, the lowest axis wins. So every
    table must use the same order, and each set of names must have a
    least element in it. Plain alphabetical order has no least element in
    some infinite sets of names, for example `b`, `ab`, `aab` and so on.
    Code that builds a table, or spells one as a type, reads its axes in
    this order.

    Example:
        >>> AxisOrder().arranged(["b", "aa", "a"])
        ('a', 'b', 'aa')
    """

    def arranged(self, names: Iterable[str]) -> tuple[str, ...]:
        """Put column names in axis order.

        Args:
            names: The read columns, in any order.

        Returns:
            The names, the shorter ones first, and names of one length in
            alphabetical order.

        Example:
            >>> AxisOrder().arranged({"price", "id"})
            ('id', 'price')
            >>> AxisOrder().arranged(["bb", "ab", "c"])
            ('c', 'ab', 'bb')
        """
        return tuple(sorted(names, key=lambda name: (len(name), name)))
