"""The split binder, executable: first match wins, and nothing more.

A split divides a remainder's columns into named parts, each declared
with a per-column predicate. The binding rule is one pass: a column
goes to the first part whose predicate accepts it. This module is the
first executable model of that rule, so the split theorems and the
two rejection conditions can be measured instead of trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from spec.tests.model import ColumnSet, SelAnd, Selector, SelNot

if TYPE_CHECKING:
    from collections.abc import Iterator

    from spec.tests.universe import Universe


@dataclass(frozen=True)
class Split:
    """One split: an ordered tuple of part predicates.

    Example:
        >>> from spec.tests.model import NUMERIC, Always, NameTest, OfType
        >>> split = Split((OfType(NUMERIC), NameTest(Always())))
        >>> len(split.parts)
        2
    """

    parts: tuple[Selector, ...]

    def closed_form(self, index: int) -> Selector:
        """Give one part's ordinary selector expression.

        This is the split theorem's subject: part `i` is the columns
        satisfying its own predicate and no earlier part's, and that
        conjunction is an ordinary selector.

        Args:
            index: The part's position, first part zero.

        Returns:
            The selector the part denotes.

        Example:
            >>> from spec.tests.model import NUMERIC, Always, NameTest, OfType
            >>> split = Split((OfType(NUMERIC), NameTest(Always())))
            >>> split.closed_form(0)
            OfType(family=NumericType)
        """
        selector = self.parts[index]
        for earlier in self.parts[:index]:
            selector = SelAnd(selector, SelNot(earlier))
        return selector

    def bound(self, columns: ColumnSet) -> tuple[ColumnSet, ...]:
        """Bind a column set into the parts, first match winning.

        One pass decides every column; a column no part accepts lands
        nowhere, which `unplaced` reports and the rejection condition
        forbids.

        Args:
            columns: The remainder's actual columns.

        Returns:
            One column set per part.

        Example:
            >>> from spec.tests.model import NUMERIC, Always, NameTest, OfType
            >>> split = Split((OfType(NUMERIC), NameTest(Always())))
            >>> parts = split.bound(ColumnSet.of(n="Int64", s="String"))
            >>> [part.names() for part in parts]
            [('n',), ('s',)]
            >>> numeric = Split((OfType(NUMERIC),))
            >>> [part.names() for part in numeric.bound(ColumnSet.of(s="String"))]
            [()]
        """
        buckets: list[list[tuple[str, str]]] = [[] for _ in self.parts]
        for name, dtype in columns.entries:
            index = self._first_match(name, dtype)
            if index is not None:
                buckets[index].append((name, dtype))
        return tuple(ColumnSet.holding(bucket) for bucket in buckets)

    def _first_match(self, name: str, dtype: str) -> int | None:
        """Give the first part accepting one column.

        Args:
            name: The column's name.
            dtype: The column's data type.

        Returns:
            The part's index, or None when no part accepts it.

        Example:
            >>> from spec.tests.model import NUMERIC, OfType
            >>> Split((OfType(NUMERIC),))._first_match("s", "String") is None
            True
        """
        for index, part in enumerate(self.parts):
            if part.picks(name, dtype):
                return index
        return None

    def unplaced(self, columns: ColumnSet) -> tuple[str, ...]:
        """List the columns no part accepts.

        Args:
            columns: The remainder's actual columns.

        Returns:
            The homeless columns' names.

        Example:
            >>> from spec.tests.model import NUMERIC, OfType
            >>> Split((OfType(NUMERIC),)).unplaced(ColumnSet.of(s="String"))
            ('s',)
        """
        entries = columns.entries
        return tuple(name for name, dtype in entries if self._first_match(name, dtype) is None)

    def places_everything(self, universe: Universe) -> bool:
        """Decide the first rejection condition over a finite universe.

        The disjunction of the parts must be valid: every column the
        universe can supply finds a part.

        Args:
            universe: The names and data types callers can bring.

        Returns:
            True when no possible column is homeless.

        Example:
            >>> from spec.tests.model import NUMERIC, OfType
            >>> from spec.tests.universe import Universe
            >>> tiny = Universe(("a",), ("Int64", "String"))
            >>> Split((OfType(NUMERIC),)).places_everything(tiny)
            False
        """
        return all(index is not None for index in self._decisions(universe))

    def _decisions(self, universe: Universe) -> Iterator[int | None]:
        """Yield the first-match decision for every possible column.

        Args:
            universe: The names and data types callers can bring.

        Yields:
            One part index, or None, per possible column.

        Example:
            >>> from spec.tests.model import NUMERIC, OfType
            >>> from spec.tests.universe import Universe
            >>> tiny = Universe(("a",), ("Int64",))
            >>> list(Split((OfType(NUMERIC),))._decisions(tiny))
            [0]
        """
        for name in universe.names:
            for dtype in universe.dtypes:
                yield self._first_match(name, dtype)

    def dead_parts(self, universe: Universe) -> tuple[int, ...]:
        """Decide the second rejection condition over a finite universe.

        A part an earlier part subsumes can never hold anything: its
        closed form is unsatisfiable, so the split never answers the
        question the part was written for.

        Args:
            universe: The names and data types callers can bring.

        Returns:
            The indices of the parts no possible column reaches.

        Example:
            >>> from spec.tests.model import NUMERIC, Always, NameTest, OfType
            >>> from spec.tests.universe import Universe
            >>> tiny = Universe(("a",), ("Int64", "String"))
            >>> Split((NameTest(Always()), OfType(NUMERIC))).dead_parts(tiny)
            (1,)
        """
        alive = {index for index in self._decisions(universe) if index is not None}
        return tuple(sorted(set(range(len(self.parts))) - alive))
