"""Two small checks on the names of a result's columns: a duplicate name, and a join collision.

Polars refuses a result that has two columns with the same name. Many
frame operations build their result from a list of output names, so each
of them needs the same duplicate check. A join has a second need: when
both sides hold a column with the same name, polars renames the right
one. Both rules live here, once, for every hook that builds names.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from astrolabe._hooks.errors import DuplicateColumnError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from collections.abc import Set as AbstractSet

    from astrolabe._hooks.errors import _FailContext


class DuplicateNameCheck:
    """Report an error when two columns of a result have the same name.

    Polars fails at run time when an operation gives two columns one name,
    for example `select(pl.col("a"), pl.col("b").alias("a"))`. The checker
    reports that at the call instead. A hook that builds a result from a
    list of output names gives the list to this class before it mints the
    result. The class reports at most one error for the call, and it names
    the first repeated name in sorted order, so the message is the same on
    every run.

    Example:
        >>> DuplicateNameCheck("select")._first_duplicate(["a", "b", "a"])
        'a'
        >>> DuplicateNameCheck("select")._first_duplicate(["a", "b"]) is None
        True
    """

    def __init__(self, operation: str) -> None:
        """Keep the name of the operation, for the error message.

        Args:
            operation: The operation that builds the names, for example
                `select`.
        """
        self._operation = operation

    def fails(self, ctx: _FailContext, names: Sequence[str | None]) -> bool:
        """Report the first repeated name, if the list has one.

        Args:
            ctx: The mypy context to report at. Any context with `.context`
                and `.api` works, for example a `MethodContext` or an
                `AnalyzeTypeContext`.
            names: The output name of each column of the result, in order.
                `None` marks a column whose name the checker does not know, for
                example a column of the remainder. The check skips it.

        Returns:
            True if the list has a repeated name. The method then reported an
            error.
        """
        duplicate = self._first_duplicate(names)
        if duplicate is None:
            return False
        DuplicateColumnError(self._operation, duplicate).fail(ctx)
        return True

    def _first_duplicate(self, names: Sequence[str | None]) -> str | None:
        """Find the repeated name that comes first in sorted order.

        Args:
            names: The output name of each column, in order. The method skips
                each `None`.

        Returns:
            The first repeated name in sorted order, or None if every name
            is different.

        Example:
            >>> DuplicateNameCheck("select")._first_duplicate(["b", "a", "b", "a"])
            'a'
            >>> DuplicateNameCheck("select")._first_duplicate(["b", "c", "b"])
            'b'
            >>> DuplicateNameCheck("select")._first_duplicate([None, None]) is None
            True
        """
        counts = Counter(name for name in names if name is not None)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        return duplicates[0] if duplicates else None


class CollisionSuffix:
    """Give the name of a right-side column after a join adds its suffix.

    When both sides of a join hold a column with the same name, polars keeps
    the left name and adds the join's suffix (`"_right"` by default) to the
    right one. A right column with no match on the left keeps its name. A
    hook that mints the result of a join needs this rule for each right
    column.

    Example:
        >>> CollisionSuffix.renamed("price", frozenset({"price"}), "_right")
        'price_right'
        >>> CollisionSuffix.renamed("venue", frozenset({"price"}), "_right")
        'venue'
    """

    @staticmethod
    def renamed(name: str, collision_names: AbstractSet[str], suffix: str) -> str:
        """Give the result name of one right-side column.

        It is a static method because it needs no state. A hook calls it with
        plain values.

        Args:
            name: The name of the right-side column.
            collision_names: The names that both sides hold.
            suffix: The suffix of the join.

        Returns:
            The name with the suffix added if both sides hold it. Otherwise the
            name as it is.
        """
        if name in collision_names:
            return name + suffix
        return name
