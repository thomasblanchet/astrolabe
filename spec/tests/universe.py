"""A finite universe of column sets, over which a claim can be checked by exhaustion.

A theorem in the specification quantifies over every column set, of
which there are infinitely many. A finite universe makes the same
statement checkable by running it: fix a small set of names and data
types, enumerate every column set built from them, and test the claim
on each one.

A counterexample found here is a real counterexample, because a finite
universe is a special case of the infinite one. A clean run is evidence
and not a proof, which is why the documents label such a result as
measured rather than proved.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING

from spec.tests.model import ColumnSet

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class Universe:
    """Every column set built from a fixed set of names and data types.

    Example:
        >>> u = Universe(("a",), ("Int64", "String"))
        >>> sorted(c.entries for c in u.column_sets())
        [(), (('a', 'Int64'),), (('a', 'String'),)]
    """

    names: tuple[str, ...]
    dtypes: tuple[str, ...]

    def column_sets(self) -> Iterator[ColumnSet]:
        """Enumerate every column set over these names and data types.

        A column set chooses a subset of the names, then one data type
        for each chosen name. Smaller column sets come first.

        Yields:
            Each column set exactly once.

        Example:
            >>> len(list(Universe(("a", "b"), ("Int64",)).column_sets()))
            4
            >>> u = Universe(("a", "b"), ("Int64", "String"))
            >>> len([c for c in u.column_sets() if len(c.entries) == 1])
            4
        """
        for size in range(len(self.names) + 1):
            for chosen in combinations(self.names, size):
                for assignment in product(self.dtypes, repeat=size):
                    yield ColumnSet(tuple(zip(chosen, assignment, strict=True)))

    def size(self) -> int:
        """Count the column sets this universe holds.

        Returns:
            How many column sets `column_sets` yields.

        Example:
            >>> Universe(("a", "b"), ("Int64", "String")).size()
            9
        """
        return sum(1 for _ in self.column_sets())
