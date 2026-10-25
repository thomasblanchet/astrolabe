"""Effects: what one frame operation does to the columns of a remainder.

A remainder is the columns of a frame that its type does not name,
for example the columns that a caller brings beyond the ones a
function declares. A function body cannot see them, but each
operation it runs changes them in a known way. `drop("tmp")` takes
`tmp` out of them, `.name.suffix("_prev")` renames each of them, and
`select` of fixed names deletes them all. An effect states one such
change.

Each effect states its change twice, and the two must agree. `apply`
changes one real column set, that is, one concrete answer with names
and data types. `transform` changes a record, which is what the
checker knows about every column set that the remainder can still
be: the names proved absent, and the data types allowed. For every
real column set that a record allows, the transformed record must
allow the result of `apply`. Tests check this agreement, and the
checker relies on `transform` alone.

`Both` sits beside the effects. It joins two remainders into one, so
it reads two operands, and it is not an `Effect`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from astrolabe._hooks.remainder.absence import (
    AfterStripping,
    Always,
    And,
    BeforeStripping,
    EndsWith,
    Not,
    Or,
    StartsWith,
)
from astrolabe._hooks.remainder.record import ColumnSet, Record
from astrolabe._hooks.remainder.type_set import TypeSet

if TYPE_CHECKING:
    from astrolabe._hooks.remainder.selector import Selector


class Effect:
    """One change that an operation makes to a remainder, on real columns and on a record.

    Each subclass is one kind of change. A hook for a frame operation
    picks the subclass that states what the operation does to the
    undeclared columns.

    Example:
        >>> Emptied().apply(ColumnSet.of(a="Int64")).entries
        ()
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Give the columns that the operation makes from one real column set.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            The columns of the remainder after the operation.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Unchanged().apply(ColumnSet.of(a="Int64")).names()
            ('a',)
        """
        raise NotImplementedError

    def transform(self, record: Record) -> Record:
        """Give what is known about the remainder after the operation.

        The result must allow every column set that `apply` can give
        from a column set that `record` allows.

        Args:
            record: What is known about the remainder before the
                operation.

        Returns:
            What is known about the remainder after the operation.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Emptied().transform(Record.unconstrained()).absence
            Always()
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Unchanged(Effect):
    """The operation leaves the remainder as it is.

    Filtering rows, sorting and reading a declared column all leave
    the undeclared columns alone.

    Example:
        >>> Unchanged().apply(ColumnSet.of(a="Int64")).entries
        (('a', 'Int64'),)
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Give the same column set back.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            The same columns.

        Example:
            >>> Unchanged().apply(ColumnSet.of(b="String")).names()
            ('b',)
        """
        return columns

    def transform(self, record: Record) -> Record:
        """Give the same record back, since every fact still holds.

        Args:
            record: What is known about the remainder.

        Returns:
            The same record.

        Example:
            >>> Unchanged().transform(Record.unconstrained()).absence
            Never()
        """
        return record


@dataclass(frozen=True)
class Emptied(Effect):
    """The operation deletes every column of the remainder.

    A `select` of a fixed list of names does this, and so does a
    reshape that turns columns into rows.

    Example:
        >>> Emptied().transform(Record.unconstrained()).types.members
        frozenset()
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Give the empty column set.

        Args:
            columns: Not read.

        Returns:
            The column set with no columns.

        Example:
            >>> Emptied().apply(ColumnSet.of(a="Int64")).names()
            ()
        """
        del columns
        return ColumnSet(())

    def transform(self, record: Record) -> Record:
        """Prove every name absent, and allow no data type.

        Args:
            record: Not read. Nothing known before the operation
                matters once every column is gone.

        Returns:
            The record of a remainder with no columns.

        Example:
            >>> Emptied().transform(Record.unconstrained()).absence.holds_of("x")
            True
        """
        del record
        return Record(Always(), TypeSet(frozenset()))


@dataclass(frozen=True)
class Retyped(Effect):
    """Every column of the remainder keeps its name and gets one new data type.

    `count()` over every column does this: each column becomes a
    `UInt32` count of its values that are not null.

    Example:
        >>> Retyped("UInt32").apply(ColumnSet.of(a="String")).entries
        (('a', 'UInt32'),)
    """

    dtype: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Give each column the new data type, and keep each name.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            The same names, all with the new data type. Two entries
            of one name become one entry.

        Example:
            >>> Retyped("UInt32").apply(
            ...     ColumnSet.of(a="Int64", b="String")
            ... ).names()
            ('a', 'b')
        """
        return ColumnSet.holding((name, self.dtype) for name, _ in columns.entries)

    def transform(self, record: Record) -> Record:
        """Keep the absent names, and allow only the new data type.

        The absent names are the valuable half. The operation renames
        nothing, so every name that was absent stays absent.

        Args:
            record: What is known about the remainder.

        Returns:
            The same absence rule, with one allowed data type.

        Example:
            >>> retyped = Retyped("UInt32").transform(Record.unconstrained())
            >>> sorted(retyped.types.members)
            ['UInt32']
        """
        return Record(record.absence, TypeSet(frozenset({self.dtype})))


@dataclass(frozen=True)
class SuffixedBy(Effect):
    """Every column name of the remainder gets a fixed suffix.

    `.name.suffix("_prev")` over the undeclared columns does this,
    and so does a join that adds a suffix to the right frame's
    columns. This effect and `PrefixedBy` are the only two that make
    new names. The absence rules therefore have a constructor for
    each, `AfterStripping` and `BeforeStripping`, which test a name
    with the added text taken off.

    Example:
        >>> SuffixedBy("_prev").apply(ColumnSet.of(price="Int64")).names()
        ('price_prev',)
    """

    suffix: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Add the suffix to the end of every name.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            The same data types, under the longer names.

        Example:
            >>> SuffixedBy("_x").apply(ColumnSet.of(a="Int64")).entries
            (('a_x', 'Int64'),)
        """
        return ColumnSet(tuple((name + self.suffix, d) for name, d in columns.entries))

    def transform(self, record: Record) -> Record:
        """Prove a name absent if no renamed column can have that name.

        After the rename, a name can be present only if it ends with
        the suffix and the name without the suffix can have been
        present. The new rule is the opposite of that: the name does
        not end with the suffix, or the old rule proves the shorter
        name absent. The data types do not change.

        Args:
            record: What is known about the remainder.

        Returns:
            The record of the renamed remainder.

        Example:
            >>> r = SuffixedBy("_p").transform(Record.unconstrained())
            >>> r.absence.holds_of("price"), r.absence.holds_of("price_p")
            (True, False)
        """
        never_produced = Not(EndsWith(self.suffix))
        stripped = AfterStripping(self.suffix, record.absence)
        return Record(Or(never_produced, stripped), record.types)


@dataclass(frozen=True)
class PrefixedBy(Effect):
    """Every column name of the remainder gets a fixed prefix.

    This is `SuffixedBy` for the other end of the name.
    `.name.prefix("sec_")` over the undeclared columns does this.

    Example:
        >>> PrefixedBy("sec_").apply(ColumnSet.of(country="String")).names()
        ('sec_country',)
    """

    prefix: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Add the prefix to the start of every name.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            The same data types, under the longer names.

        Example:
            >>> PrefixedBy("x_").apply(ColumnSet.of(a="Int64")).entries
            (('x_a', 'Int64'),)
        """
        return ColumnSet(tuple((self.prefix + name, d) for name, d in columns.entries))

    def transform(self, record: Record) -> Record:
        """Prove a name absent if no renamed column can have that name.

        A name is absent if it does not begin with the prefix, or if
        the old rule proves the name without the prefix absent.

        Args:
            record: What is known about the remainder.

        Returns:
            The record of the renamed remainder.

        Example:
            >>> r = PrefixedBy("s_").transform(Record.unconstrained())
            >>> r.absence.holds_of("a"), r.absence.holds_of("s_a")
            (True, False)
        """
        never_produced = Not(StartsWith(self.prefix))
        stripped = BeforeStripping(self.prefix, record.absence)
        return Record(Or(never_produced, stripped), record.types)


@dataclass(frozen=True)
class Where(Effect):
    """Only the columns of the remainder that a selector picks stay.

    A selector is a test on a whole column, its name and its data
    type together, for example `cs.numeric()`. Three operations share
    this one effect. `drop(["a", "b"])` keeps the columns whose name
    is not in the list. `select(cs.numeric())` keeps the columns that
    the selector picks, and `drop(cs.numeric())` keeps the others.
    They differ only in the selector, and in whether it is negated.

    Example:
        >>> from astrolabe._hooks.remainder.selector import OfType
        >>> from astrolabe._hooks.remainder.type_set import Family
        >>> keep = Where(OfType(Family.numeric()))
        >>> keep.apply(ColumnSet.of(a="Int64", b="String")).names()
        ('a',)
    """

    selector: Selector

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Keep the columns that the selector picks.

        Args:
            columns: The columns of the remainder before the operation.

        Returns:
            Only the picked columns.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> from astrolabe._hooks.remainder.selector import NameTest
            >>> keep = Where(NameTest(StartsWith("a")))
            >>> keep.apply(ColumnSet.of(ab="Int64")).names()
            ('ab',)
        """
        kept = tuple(e for e in columns.entries if self.selector.picks(e[0], e[1]))
        return ColumnSet(kept)

    def transform(self, record: Record) -> Record:
        """Add the names that the selector certainly rejects, and narrow the data types.

        A name that the selector rejects for every allowed data type is
        absent after the operation. This is how one drop can satisfy a
        declaration that demands a whole group of names absent, for
        example every name that begins with `tmp_`. Narrowing the data
        types lets a later test on data types get an answer, for
        example after `select(cs.numeric())`.

        Args:
            record: What is known about the remainder.

        Returns:
            The record of the filtered remainder.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Never
            >>> from astrolabe._hooks.remainder.selector import NameTest
            >>> r = Where(NameTest(Never())).transform(Record.unconstrained())
            >>> r.absence.holds_of("anything")
            True
        """
        rejected = self.selector.certainly_rejects(record.types)
        return Record(
            Or(record.absence, rejected),
            self.selector.surviving_types(record.types),
        )


@dataclass(frozen=True)
class Both:
    """Two remainders joined into one.

    A horizontal concat or a join can give a frame that holds the
    undeclared columns of two frames. This is the only construct that
    joins two remainders. It reads two operands instead of one, so it
    is not an `Effect`.

    Example:
        >>> Both().apply(ColumnSet.of(a="Int64"), ColumnSet.of(b="String")).names()
        ('a', 'b')
    """

    def apply(self, left: ColumnSet, right: ColumnSet) -> ColumnSet:
        """Give every column of both operands.

        Args:
            left: The columns of one remainder.
            right: The columns of the other remainder.

        Returns:
            The columns of both, sorted by name.

        Example:
            >>> Both().apply(ColumnSet(()), ColumnSet.of(b="String")).names()
            ('b',)
        """
        return ColumnSet.holding(left.entries + right.entries)

    def transform(self, left: Record, right: Record) -> Record:
        """Prove a name absent only if both operands prove it absent.

        A column of either remainder is a column of the result. So a
        name is absent from the result only if it is absent from both,
        and a data type is allowed if either side allows it.

        Args:
            left: What is known about one remainder.
            right: What is known about the other.

        Returns:
            The record of the joined remainder.

        Example:
            >>> from astrolabe._hooks.remainder.absence import OneOf
            >>> a = Record(OneOf(frozenset({"x"})), TypeSet(frozenset({"Int64"})))
            >>> b = Record(
            ...     OneOf(frozenset({"x", "y"})), TypeSet(frozenset({"String"}))
            ... )
            >>> combined = Both().transform(a, b)
            >>> combined.absence.holds_of("x"), combined.absence.holds_of("y")
            (True, False)
        """
        return Record(
            And(left.absence, right.absence),
            left.types.combined_with(right.types),
        )
