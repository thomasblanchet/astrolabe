"""Splits: divide the undeclared columns of a frame into named parts, first match first.

A remainder is the columns of a frame that its type does not name. A
function can declare its parameter as
`TypedDataFrame[Keys, SplitColumns[Numbers, Rest]]`, with
`Numbers = other_columns(all_of=NumericType)` and
`Rest = other_columns()`. Each numeric undeclared column of the
caller's frame then goes to `Numbers`, and every other one to `Rest`,
so the function can treat each part in its own way. Each part is a
test on one column. A column goes to the first part whose test
accepts it, so the order of the parts counts. Part `i` then holds
exactly the columns that pass its own test and fail every earlier
test. That is an ordinary selector, so a split needs no new kind of
remainder type.

A split can be wrong in two ways, and this module decides both
exactly. A split with a gap has no part for some possible column. A
dead part can never get a column, because earlier parts take every
column it accepts.
"""

from __future__ import annotations

from dataclasses import dataclass

from astrolabe._hooks.remainder.type_set import TypeSamples
from astrolabe._hooks.remainder.name_language import NameSet
from astrolabe._hooks.remainder.record import ColumnSet, Record
from astrolabe._hooks.checked import checked_not_none
from astrolabe._hooks.remainder.selector import NameTest, OfType, SelAnd, Selector, SelNot
from astrolabe._hooks.remainder.type_set import Family, TypeSet


@dataclass(frozen=True)
class Split:
    """One split: the test of each part, in the order the user declared them.

    Each part is a selector, a test on one column's name and data
    type. The methods place real columns into parts, give each part
    as one selector for code that narrows a remainder, and decide
    whether the split has a gap or a dead part.

    Example:
        >>> from astrolabe._hooks.remainder.absence import Always
        >>> from astrolabe._hooks.remainder.selector import NameTest, OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> split = Split((OfType(Family.numeric()), NameTest(Always())))
        >>> len(split.parts)
        2
    """

    parts: tuple[Selector, ...]

    def closed_form(self, index: int) -> Selector:
        """Give one part as a selector that needs no order.

        Part `i` gets the columns that pass its own test and fail
        every earlier test. That is the `SelAnd` of its test with the
        negation of each earlier test. Code that narrows an open
        argument's remainder to one part uses this selector.

        Args:
            index: The position of the part, from zero.

        Returns:
            The selector of the columns that the part gets.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import (
            ...     NameTest, OfType,
            ... )
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> split = Split((OfType(Family.numeric()), NameTest(Always())))
            >>> split.closed_form(0)
            OfType(family=NumericType)
        """
        selector = self.parts[index]
        for earlier in self.parts[:index]:
            selector = SelAnd(selector, SelNot(earlier))
        return selector

    def bound(self, columns: ColumnSet) -> tuple[ColumnSet, ...]:
        """Place real columns into the parts, each into the first part that accepts it.

        A caller whose frame type names every column brings known
        columns, and this gives each part its share. A column that no
        part accepts goes nowhere. `unplaced` lists such columns.

        Args:
            columns: The real columns of the remainder.

        Returns:
            One column set for each part.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import (
            ...     NameTest, OfType,
            ... )
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> split = Split((OfType(Family.numeric()), NameTest(Always())))
            >>> parts = split.bound(ColumnSet.of(n="Int64", s="String"))
            >>> [part.names() for part in parts]
            [('n',), ('s',)]
            >>> numeric = Split((OfType(Family.numeric()),))
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
        """Give the first part that accepts one column.

        Args:
            name: The name of the column.
            dtype: The data type of the column.

        Returns:
            The position of the part, or None if no part accepts the
            column.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = Split((OfType(Family.numeric()),))
            >>> numeric._first_match("s", "String") is None
            True
        """
        for index, part in enumerate(self.parts):
            if part.picks(name, dtype):
                return index
        return None

    def unplaced(self, columns: ColumnSet) -> tuple[str, ...]:
        """List the real columns that no part accepts, for an error message.

        Args:
            columns: The real columns of the remainder.

        Returns:
            The names of the columns that go nowhere.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> numeric = Split((OfType(Family.numeric()),))
            >>> numeric.unplaced(ColumnSet.of(s="String"))
            ('s',)
        """
        entries = columns.entries
        return tuple(name for name, dtype in entries if self._first_match(name, dtype) is None)

    def has_gap(self) -> bool:
        """Decide exactly whether some possible column has no part.

        Every column that any caller can bring must find a part. The
        answer comes one sample data type at a time (see
        `TypeSamples`). At each sample, the names that some part
        accepts form a regular set, and that set must hold every name.

        Returns:
            True if some possible column goes to no part.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> Split((OfType(Family.numeric()),)).has_gap()
            True
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import NameTest
            >>> Split((NameTest(Always()),)).has_gap()
            False
        """
        for dtype in self._samples():
            united = NameSet.nothing()
            for part in self.parts:
                united = united.union(part.language_for(dtype))
            if not united.complement().empty():
                return True
        return False

    def dead_parts(self) -> tuple[int, ...]:
        """Decide exactly which parts can never get a column.

        If earlier parts accept every column that a part accepts, the
        part is dead. Its selector from `closed_form` picks nothing,
        so the part is almost certainly a mistake in the declaration.

        Returns:
            The positions of the parts that no possible column
            reaches.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Always
            >>> from astrolabe._hooks.remainder.selector import (
            ...     NameTest, OfType,
            ... )
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> shadowed = Split((NameTest(Always()), OfType(Family.numeric())))
            >>> shadowed.dead_parts()
            (1,)
            >>> Split((OfType(Family.numeric()),)).dead_parts()
            ()
        """
        samples = self._samples()
        dead: list[int] = []
        for index in range(len(self.parts)):
            form = self.closed_form(index)
            if all(form.language_for(dtype).empty() for dtype in samples):
                dead.append(index)
        return tuple(dead)

    def _samples(self) -> tuple[str, ...]:
        """Give sample data types that stand for every data type, for these parts.

        Returns:
            One sample for each group of data types that the parts
            can tell apart.

        Example:
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> "Int64" in Split((OfType(Family.numeric()),))._samples()
            True
        """
        mentioned = frozenset().union(*(part.mentioned_types() for part in self.parts))
        return TypeSamples().of(mentioned)


@dataclass(frozen=True)
class SplitPredicates:
    """Turn the declaration of each part of a `SplitColumns[...]` into its test.

    The user declares each part with `other_columns(...)`, and a
    `Split` needs one selector for each part. The engine reads a
    declaration as a record: an absence rule for the names that the
    declaration keeps out, and a type set for the data types that it
    allows. The part then picks a column whose name the rule does not
    hold of, and whose data type the type set allows.

    Example:
        >>> from astrolabe._hooks.remainder.record import Record
        >>> SplitPredicates().of(Record.unconstrained())
        SelNot(inner=NameTest(rule=Never()))
    """

    def of(self, record: Record) -> Selector | None:
        """Give the selector of the columns that one declaration allows.

        Args:
            record: The record of the declaration.

        Returns:
            The selector, or None if the type set of the record takes
            some data types out of a whole kind. No family states such
            a set, so no test on data types can spell it.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Never
            >>> from astrolabe._hooks.remainder.record import Record
            >>> from astrolabe._hooks.remainder.type_set import TypeSet
            >>> ints = Record(Never(), TypeSet(frozenset({"Int64"})))
            >>> SplitPredicates().of(ints).right
            OfType(family=Int64)
        """
        named = SelNot(NameTest(record.absence))
        types = record.types
        if types == TypeSet.anything():
            return named
        if types.excluded:
            return None
        family = Family.of_members(types.members)
        return SelAnd(named, OfType(family))

    def split(self, records: tuple[Record, ...]) -> Split | None:
        """Build the split of a whole `SplitColumns[...]` declaration.

        Args:
            records: The record of each part, in the declared order.

        Returns:
            The split, or None if the test of some part cannot be
            spelled.

        Example:
            >>> from astrolabe._hooks.remainder.record import Record
            >>> SplitPredicates().split((Record.unconstrained(),)).parts
            (SelNot(inner=NameTest(rule=Never())),)
        """
        parts = [self.of(record) for record in records]
        if any(part is None for part in parts):
            return None
        return Split(tuple(map(checked_not_none, parts)))
