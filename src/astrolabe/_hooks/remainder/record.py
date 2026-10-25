"""Column sets and records: one real remainder, and what is known about all of them.

A remainder is the columns of a frame that its type does not name.
In a function over `TypedDataFrame[Trades, Rest]`, the remainder
`Rest` is whatever other columns each caller brings. One caller
brings `venue`, and another brings nothing. A column set is one such
answer: these names, with these data types.

A record is what the checker knows about every column set that a
remainder can still be. It has two halves. The absence rule gives the
names that are certainly not in the remainder, and the type set gives
the data types that its columns can have. A record must always allow
the real columns. If a rule forgets a fact, the record allows more
column sets and still allows the real one. So the engine records less
when it is not sure, and it never guesses. Every effect states what
it does to both halves, so that a chain of operations keeps both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.remainder.absence import Absence, Never
from astrolabe._hooks.remainder.type_set import TypeSet

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ColumnSet:
    """One possible value of a remainder: each column name with its data type.

    The checker never sees the real columns of a remainder. A column
    set is the concrete answer that a record describes, and code uses
    it to state what an effect does to real columns. It also carries
    the columns of a caller whose frame type names every column: the
    columns that land in a remainder are then one known column set.

    A data type is text, for example `"Int64"` or a full rendering
    such as `"Datetime(time_unit='us', time_zone=None)"`. The entries
    stay sorted by name, so two equal sets compare equal.

    Example:
        >>> ColumnSet.of(price="Float64", venue="String").names()
        ('price', 'venue')
    """

    entries: tuple[tuple[str, str], ...]

    @classmethod
    def holding(cls, entries: Iterable[tuple[str, str]]) -> ColumnSet:
        """Build a column set from pairs, and keep each pair once.

        One name has one data type, so a repeated pair adds nothing.
        Only a join of two remainders, for example a horizontal concat,
        could give one name two data types. The checker refuses such a
        join before it builds the set, because polars refuses a frame
        with two columns of one name. If two pairs do share a name, both
        stay, so that no data type is lost without a trace.

        Args:
            entries: The name and data type pairs.

        Returns:
            The column set, sorted, with no pair twice.

        Example:
            >>> ColumnSet.holding((("a", "Int64"), ("a", "Int64"))).entries
            (('a', 'Int64'),)
        """
        return cls(tuple(sorted(dict.fromkeys(entries))))

    @classmethod
    def of(cls, **columns: str) -> ColumnSet:
        """Build a column set from keyword arguments, for a short example or test.

        No hook calls it. The examples and the tests of the remainder
        build their column sets with it, because a keyword per column
        reads more clearly than a tuple of pairs.

        Args:
            **columns: Each keyword is a column name, and its value is
                the data type.

        Returns:
            The column set of those columns.

        Example:
            >>> ColumnSet.of(a="Int64").entries
            (('a', 'Int64'),)
        """
        return cls(tuple(sorted(columns.items())))

    def names(self) -> tuple[str, ...]:
        """Give the column names.

        Returns:
            Every column name, sorted.

        Example:
            >>> ColumnSet.of(b="Int64", a="String").names()
            ('a', 'b')
        """
        return tuple(name for name, _ in self.entries)

    def type_set(self) -> TypeSet:
        """Give the data types of these columns as a type set.

        A record compares this with its own type set to decide whether
        the columns are possible.

        Returns:
            The type set of exactly the data types present.

        Example:
            >>> sorted(ColumnSet.of(a="Int64", b="Int64").type_set().members)
            ['Int64']
        """
        return TypeSet(frozenset(dtype for _, dtype in self.entries))

    def holds(self, name: str) -> bool:
        """Answer whether a column of this name is present.

        Args:
            name: The column name to look for.

        Returns:
            True if the name is present.

        Example:
            >>> ColumnSet.of(a="Int64").holds("b")
            False
        """
        return name in self.names()


@dataclass(frozen=True)
class Record:
    """What the checker knows about a remainder: its absent names and its data types.

    The checker cannot list the columns of a remainder, but it can
    prove facts about them. An operation such as `drop("tmp")` proves
    that `tmp` is absent. A declaration such as
    `other_columns(all_of=NumericType)` limits the data types. A record
    holds both kinds of fact: `absence` gives the names that are
    certainly absent, and `types` gives the data types that a column
    can have.

    A record is never stored in a type. The engine computes it from a
    remainder type each time a check needs it. The check then asks
    whether the record keeps a promise that a declaration demands,
    for example that `other_columns(without=["tmp"])` holds no `tmp`.

    Example:
        >>> from astrolabe._hooks.remainder.absence import OneOf
        >>> paid = Record(OneOf(frozenset({"a"})), TypeSet.anything())
        >>> paid.absence.holds_of("a")
        True
    """

    absence: Absence
    types: TypeSet

    @classmethod
    def unconstrained(cls) -> Record:
        """Build the record of a remainder about which nothing is known.

        This is the record of a plain `other_columns()`, which makes
        no promise.

        Returns:
            The record that proves no name absent and allows every
            data type.

        Example:
            >>> Record.unconstrained().absence.holds_of("anything")
            False
        """
        return cls(Never(), TypeSet.anything())

    def permits_nothing(self) -> bool:
        """Answer whether this remainder certainly holds no column.

        This is true if the type set is empty, or if the absence rule
        holds of every name. Such a remainder holds no column for any
        caller, so it satisfies every requirement.

        Returns:
            True if no column is possible.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> Record(Always(), TypeSet.anything()).permits_nothing()
            True
            >>> Record.unconstrained().permits_nothing()
            False
        """
        no_types = self.types.is_empty()
        return no_types or self.absence.language().complement().empty()

    def describes(self, columns: ColumnSet) -> bool:
        """Answer whether `columns` is a possible value of this remainder.

        The columns are possible if none of their names is proved
        absent and each of their data types is allowed. Tests use this
        to check a record against real columns.

        Args:
            columns: The concrete column set.

        Returns:
            True if nothing in the record rules the columns out.

        Example:
            >>> from astrolabe._hooks.remainder.absence import OneOf
            >>> r = Record(OneOf(frozenset({"a"})), TypeSet.anything())
            >>> r.describes(ColumnSet.of(a="Int64"))
            False
            >>> r.describes(ColumnSet.of(b="Int64"))
            True
        """
        names_free = not any(self.absence.holds_of(n) for n in columns.names())
        return names_free and columns.type_set().within(self.types)
