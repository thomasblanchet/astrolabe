"""Selectors: tests on a whole column, its name and its data type together.

A polars column selector such as `cs.numeric()` or
`cs.starts_with("px_")` picks columns. When a frame operation takes
one, the checker must know what it does to the remainder, that is,
to the columns of the frame that its type does not name. Those
columns have no names that the checker can list. So a selector here
answers two kinds of question. It decides one real column, and it
states what it certainly picks or rejects among columns known only
by a record: the names proved absent and the data types allowed.

Any absence rule, a test on a name, becomes a selector through
`NameTest`. `OfType` tests the data type, which a name test cannot.
`SelNot`, `SelAnd` and `SelOr` combine selectors.

`SelectorEntailment` decides whether one selector picks no more
columns than another. `CanonicalNameFilter` and `Conjunction` give a
selector one spelling, so that two selectors that pick the same
columns give one type. They share this module with the selectors,
because the canonical form reads the containment, and the
containment reads the shape of each selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.remainder.absence import (
    Absence,
    Always,
    And,
    Matching,
    Never,
    Not,
    OneOf,
    Or,
)
from astrolabe._hooks.remainder.entailment import AbsenceEntailment
from astrolabe._hooks.remainder.rendering import StableText
from astrolabe._hooks.remainder.type_set import (
    Family,
    TypeSamples,
    TypeSet,
)

if TYPE_CHECKING:
    from astrolabe._hooks.remainder.name_language import NameSet


class Selector:
    """A test on a whole column, its name and its data type together.

    `picks` decides one real column. A remainder has no real columns
    that the checker can see, so an operation over it works from
    `certainly_picks` and `certainly_rejects`. Given what is known
    about the data types, they give the names for which the answer is
    the same whatever the column's data type. The other methods move
    a selector past a rename or a new data type, give its canonical
    form, and give the exact name set it picks at each data type, so
    that two selectors can be compared.

    Example:
        >>> from astrolabe._hooks.remainder.absence import StartsWith
        >>> NameTest(StartsWith("tmp_")).picks("tmp_x", "Int64")
        True
    """

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether this selector picks one real column.

        Args:
            name: The name of the column.
            dtype: The data type of the column, as text.

        Returns:
            True if the selector picks the column.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).picks("x", "Int64")
            True
        """
        raise NotImplementedError

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Give the names that this selector certainly picks.

        Args:
            types: The data types that the columns can have.

        Returns:
            A name rule that holds of a name only if a column of that
            name is picked for every allowed data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).certainly_picks(
            ...     TypeSet(frozenset({"Int64"}))
            ... )
            Always()
        """
        raise NotImplementedError

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Give the names that this selector certainly rejects.

        An operation that keeps what the selector picks needs this. A
        name that the selector certainly rejects is absent from the
        result.

        Args:
            types: The data types that the columns can have.

        Returns:
            A name rule that holds of a name only if a column of that
            name is rejected for every allowed data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).certainly_rejects(
            ...     TypeSet(frozenset({"String"}))
            ... )
            Always()
        """
        raise NotImplementedError

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Narrow a type set to the data types that a picked column can have.

        This default keeps every data type. That is always safe,
        because a record that allows more still allows the truth.
        Selectors that read data types give a narrower answer.

        Args:
            types: The data types that the columns can have.

        Returns:
            Every data type that a picked column can have.

        Example:
            >>> ints = TypeSet(frozenset({"Int64"}))
            >>> sorted(NameTest(Always()).surviving_types(ints).members)
            ['Int64']
        """
        return types

    def reads_only_types(self) -> bool:
        """Answer whether this selector reads no column name at all.

        The negation of a selector can narrow the data types exactly
        only if the selector reads data types alone. `SelNot` asks
        this before it narrows.

        Returns:
            True if no part of this selector reads a name.

        Example:
            >>> OfType(Family.numeric()).reads_only_types()
            True
            >>> NameTest(Always()).reads_only_types()
            False
        """
        return False

    def name_rule(self) -> Absence | None:
        """Give this selector as one rule on names, if it reads no data type.

        A selector built only from name tests, `SelNot`, `SelAnd` and
        `SelOr` depends on the name alone. So one absence rule holds
        its whole meaning. `CanonicalNameFilter` spells such a selector
        by the set of names of that rule, so two filters that pick the
        same names become the same selector.

        Returns:
            The rule, or None if any part reads a data type.

        Example:
            >>> OfType(Family.numeric()).name_rule() is None
            True
        """
        return None

    def through_suffix(self, suffix: str) -> Selector:
        """Rewrite this selector so that it runs before a rename that adds a suffix.

        A canonical form keeps its filter inside its renames. A
        selector that runs after a rename tests the new name. Moved
        before the rename, it must test the old name, and ask about
        the name that the rename will make.

        Args:
            suffix: The suffix that the rename adds.

        Returns:
            A selector that picks a column exactly if this one picks
            the same column after the suffix is added.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> numeric.through_suffix("_p") == numeric
            True
        """
        raise NotImplementedError

    def through_prefix(self, prefix: str) -> Selector:
        """Rewrite this selector so that it runs before a rename that adds a prefix.

        Args:
            prefix: The prefix that the rename adds.

        Returns:
            A selector that picks a column exactly if this one picks
            the same column after the prefix is added.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> numeric.through_prefix("s_") == numeric
            True
        """
        raise NotImplementedError

    def after_retype(self, dtype: str) -> Selector:
        """Decide every test on data types, for columns that all have one known data type.

        After an operation gives every column one data type, a
        selector that runs next knows the answer of each test on data
        types. So it becomes a test on names alone, which can move
        inside the new data type in a canonical form.

        Args:
            dtype: The data type that every column now has.

        Returns:
            A selector with no test on data types left.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).after_retype("Int64")
            NameTest(rule=Always())
        """
        raise NotImplementedError

    def canonical(self) -> Selector:
        """Give the one spelling of this selector, so that equal selectors compare equal.

        Every part that reads only names is spelled by the set of
        names it picks, however the user wrote it. If that spelling is
        a `Matching`, the rule that the user wrote stays beside it, so
        an error message can show the user's rule.

        Returns:
            A selector that picks the same columns, in its canonical
            spelling.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).canonical()
            OfType(family=NumericType)
        """
        raise NotImplementedError

    def mentioned_types(self) -> frozenset[str]:
        """Give the single data types that the tests on data types of this selector name.

        A test can name one data type of a kind, for example
        `Datetime("us")`, and not the whole kind. A question about
        every data type must then ask at each named data type, and at
        one sample for the rest of the kind (see `TypeSamples`).

        Returns:
            The text of each named single data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> NameTest(Never()).mentioned_types()
            frozenset()
        """
        raise NotImplementedError

    def language_for(self, dtype: str) -> NameSet:
        """Give the names that this selector picks among the columns of one data type.

        For one data type, the names that a selector picks form a
        regular set, that is, a set that a finite automaton decides.
        A few sample data types stand for all of them. So one name
        set for each sample describes the whole selector, and two
        selectors can be compared exactly.

        Args:
            dtype: The data type to fix.

        Returns:
            The set of names picked at that data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(Family.numeric()).language_for("Int64").accepts("any")
            True
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NameTest(Selector):
    """A selector that reads only the name of a column, through an absence rule.

    An absence rule is a test on a name, for example `StartsWith`.
    This class lets any such rule act as a selector:
    `cs.starts_with("px_")` becomes `NameTest(StartsWith("px_"))`.
    Because it reads no data type, every answer about a remainder is
    exact.

    Example:
        >>> from astrolabe._hooks.remainder.absence import OneOf
        >>> NameTest(OneOf(frozenset({"a"}))).picks("a", "String")
        True
    """

    rule: Absence

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether the rule holds of the name of the column.

        Args:
            name: The name of the column.
            dtype: Not read. A name test does not read the data type.

        Returns:
            True if the rule holds of the name.

        Example:
            >>> from astrolabe._hooks.remainder.absence import EndsWith
            >>> NameTest(EndsWith("_prev")).picks("price", "Int64")
            False
        """
        del dtype
        return self.rule.holds_of(name)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Give the names that the rule holds of, which is exact for a name test.

        Args:
            types: Not read. A name test does not read data types.

        Returns:
            The rule itself.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> NameTest(StartsWith("a")).certainly_picks(TypeSet.anything())
            StartsWith(prefix='a')
        """
        del types
        return self.rule

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Give the names that the rule does not hold of, which is exact for a name test.

        Args:
            types: Not read. A name test does not read data types.

        Returns:
            The negation of the rule.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> NameTest(StartsWith("a")).certainly_rejects(TypeSet.anything())
            Not(inner=StartsWith(prefix='a'))
        """
        del types
        return Not(self.rule)

    def through_suffix(self, suffix: str) -> Selector:
        """Test the name that the rename will make, instead of the name given.

        The new rule holds of a name if the old rule holds of that
        name with the suffix added. In general no named constructor
        states that set, so the rule is a `Matching`. This case is the
        reason that constructor exists.

        Args:
            suffix: The suffix that the rename adds.

        Returns:
            A name test on the longer name, or this test if the
            suffix is empty.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> shifted = NameTest(StartsWith("ab")).through_suffix("b_")
            >>> shifted.picks("a", "Int64")
            True
        """
        if not suffix:
            return self
        return NameTest(Matching.of(self.rule.language().after_appending(suffix)))

    def through_prefix(self, prefix: str) -> Selector:
        """Test the name that the rename will make, instead of the name given.

        Args:
            prefix: The prefix that the rename adds.

        Returns:
            A name test on the longer name, or this test if the
            prefix is empty.

        Example:
            >>> from astrolabe._hooks.remainder.absence import EndsWith
            >>> shifted = NameTest(EndsWith("_x")).through_prefix("s_")
            >>> shifted.picks("a_x", "Int64")
            True
        """
        if not prefix:
            return self
        return NameTest(Matching.of(self.rule.language().after_prepending(prefix)))

    def after_retype(self, dtype: str) -> Selector:
        """Give this selector back, since it has no test on data types.

        Args:
            dtype: Not read.

        Returns:
            This selector.

        Example:
            >>> NameTest(Always()).after_retype("Int64")
            NameTest(rule=Always())
        """
        del dtype
        return self

    def canonical(self) -> Selector:
        """Spell this test by the set of names it picks.

        Returns:
            The canonical spelling from `CanonicalNameFilter`.

        Example:
            >>> from astrolabe._hooks.remainder.absence import (
            ...     Or, StartsWith,
            ... )
            >>> a = NameTest(StartsWith("a")).canonical()
            >>> b = NameTest(Or(StartsWith("a"), StartsWith("ab"))).canonical()
            >>> a == b
            True
        """
        return CanonicalNameFilter()(self.rule)

    def name_rule(self) -> Absence | None:
        """Give the rule, since a name test already reads names alone.

        Returns:
            The rule.

        Example:
            >>> NameTest(Always()).name_rule()
            Always()
        """
        return self.rule

    def mentioned_types(self) -> frozenset[str]:
        """Give no data type, since a name test names none.

        Returns:
            The empty set.

        Example:
            >>> NameTest(Always()).mentioned_types()
            frozenset()
        """
        return frozenset()

    def language_for(self, dtype: str) -> NameSet:
        """Give the names of the rule, for every data type.

        Args:
            dtype: Not read. A name test does not read the data type.

        Returns:
            The set of names that the rule holds of.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> NameTest(StartsWith("a")).language_for("String").accepts("ab")
            True
        """
        del dtype
        return self.rule.language()


@dataclass(frozen=True)
class OfType(Selector):
    """A selector that reads only the data type of a column, for example `cs.numeric()`.

    A test on data types cannot become a rule on names, because a
    name says nothing about the data type. So its answers about a
    remainder are all or nothing. If every allowed data type is in
    the family, it certainly picks every name. If none is, it
    certainly rejects every name. Otherwise it settles no name, and
    it narrows only the data types.

    Example:
        >>> OfType(Family.numeric()).picks("anything", "Float64")
        True
    """

    family: Family

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether the family holds the data type of the column.

        Args:
            name: Not read. A test on data types does not read the
                name.
            dtype: The data type of the column.

        Returns:
            True if the family holds the data type.

        Example:
            >>> OfType(Family.numeric()).picks("price", "String")
            False
        """
        del name
        return self.family.holds(dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick every name if the family holds every allowed data type.

        Args:
            types: The data types that the columns can have.

        Returns:
            `Always` if the family holds every allowed data type. If
            not, `Never`, because no name is then certain.

        Example:
            >>> OfType(Family.numeric()).certainly_picks(TypeSet.anything())
            Never()
        """
        return Always() if types.contained_in(self.family) else Never()

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject every name if the family holds none of the allowed data types.

        Args:
            types: The data types that the columns can have.

        Returns:
            `Always` if the family holds no allowed data type, and
            `Never` if not.

        Example:
            >>> OfType(Family.numeric()).certainly_rejects(TypeSet.anything())
            Never()
        """
        return Always() if types.disjoint_from(self.family) else Never()

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep only the allowed data types that the family holds.

        This answer is exact. After `select(cs.numeric())`, it lets a
        later test on the data types of the result get an answer.

        Args:
            types: The data types that the columns can have.

        Returns:
            The allowed data types that are in the family.

        Example:
            >>> mixed = TypeSet(frozenset({"Int64", "String"}))
            >>> sorted(OfType(Family.numeric()).surviving_types(mixed).members)
            ['Int64']
        """
        return types.narrowed_to(self.family)

    def reads_only_types(self) -> bool:
        """Answer True, since this selector never reads a name.

        Returns:
            True.

        Example:
            >>> OfType(Family.numeric()).reads_only_types()
            True
        """
        return True

    def through_suffix(self, suffix: str) -> Selector:
        """Give this selector back, since a rename that adds a suffix changes no data type.

        Args:
            suffix: Not read.

        Returns:
            This selector.

        Example:
            >>> OfType(Family.numeric()).through_suffix("_p")
            OfType(family=NumericType)
        """
        del suffix
        return self

    def through_prefix(self, prefix: str) -> Selector:
        """Give this selector back, since a rename that adds a prefix changes no data type.

        Args:
            prefix: Not read.

        Returns:
            This selector.

        Example:
            >>> OfType(Family.numeric()).through_prefix("s_")
            OfType(family=NumericType)
        """
        del prefix
        return self

    def after_retype(self, dtype: str) -> Selector:
        """Decide the test now, since every column has the one known data type.

        Args:
            dtype: The data type that every column now has.

        Returns:
            A name test that picks every name, or no name.

        Example:
            >>> OfType(Family.numeric()).after_retype("String")
            NameTest(rule=Never())
        """
        return NameTest(Always() if self.family.holds(dtype) else Never())

    def canonical(self) -> Selector:
        """Give this selector back, since a family already has one spelling.

        Returns:
            This selector.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> numeric.canonical() == numeric
            True
        """
        return self

    def mentioned_types(self) -> frozenset[str]:
        """Give the members of the family that are single data types, not whole kinds.

        Returns:
            Each member whose text has parameters.

        Example:
            >>> OfType(Family.numeric()).mentioned_types()
            frozenset()
        """
        return frozenset(member for member in self.family.members if "(" in member)

    def language_for(self, dtype: str) -> NameSet:
        """Pick every name, or no name, by the data type alone.

        Args:
            dtype: The data type to fix.

        Returns:
            Every name if the family holds the data type, and no
            name if not.

        Example:
            >>> OfType(Family.numeric()).language_for("String").accepts("x")
            False
        """
        if self.family.holds(dtype):
            return Always().language()
        return Never().language()


@dataclass(frozen=True)
class SelNot(Selector):
    """A selector that picks exactly what the inner one does not, as `~cs.numeric()`.

    Example:
        >>> SelNot(OfType(Family.numeric())).picks("x", "String")
        True
    """

    inner: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer the opposite of the inner selector.

        Args:
            name: The name of the column.
            dtype: The data type of the column.

        Returns:
            True if the inner selector does not pick the column.

        Example:
            >>> SelNot(OfType(Family.numeric())).picks("x", "Int64")
            False
        """
        return not self.inner.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick the names that the inner selector certainly rejects.

        Args:
            types: The data types that the columns can have.

        Returns:
            The rule of the names that the inner selector certainly
            rejects.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> SelNot(NameTest(StartsWith("a"))).certainly_picks(
            ...     TypeSet.anything()
            ... )
            Not(inner=StartsWith(prefix='a'))
        """
        return self.inner.certainly_rejects(types)

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject the names that the inner selector certainly picks.

        Args:
            types: The data types that the columns can have.

        Returns:
            The rule of the names that the inner selector certainly
            picks.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> SelNot(NameTest(StartsWith("a"))).certainly_rejects(
            ...     TypeSet.anything()
            ... )
            StartsWith(prefix='a')
        """
        return self.inner.certainly_picks(types)

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the allowed data types that the inner selector does not pick.

        This is exact only if the inner selector reads no name. If it
        reads a name, a column of any data type can pass the negation
        by its name. Then every allowed data type stays.

        Args:
            types: The data types that the columns can have.

        Returns:
            The data types in `types` that the inner selector does not
            keep, if it reads only data types. If not, `types` itself.

        Example:
            >>> mixed = TypeSet(frozenset({"Int64", "String"}))
            >>> negated = SelNot(OfType(Family.numeric()))
            >>> sorted(negated.surviving_types(mixed).members)
            ['String']
        """
        if not self.inner.reads_only_types():
            return types
        return types.minus(self.inner.surviving_types(types))

    def reads_only_types(self) -> bool:
        """Answer whether the inner selector reads only data types.

        Returns:
            True if the inner selector reads no name.

        Example:
            >>> SelNot(OfType(Family.numeric())).reads_only_types()
            True
        """
        return self.inner.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Rewrite the inner selector so that it runs before the suffix is added.

        Args:
            suffix: The suffix that the rename adds.

        Returns:
            The negation of the rewritten inner selector.

        Example:
            >>> s = SelNot(OfType(Family.numeric()))
            >>> isinstance(s.through_suffix("_p"), SelNot)
            True
        """
        return SelNot(self.inner.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Rewrite the inner selector so that it runs before the prefix is added.

        Args:
            prefix: The prefix that the rename adds.

        Returns:
            The negation of the rewritten inner selector.

        Example:
            >>> s = SelNot(OfType(Family.numeric()))
            >>> isinstance(s.through_prefix("s_"), SelNot)
            True
        """
        return SelNot(self.inner.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Decide the tests on data types of the inner selector, for the one data type.

        Args:
            dtype: The data type that every column now has.

        Returns:
            The negation of the decided inner selector.

        Example:
            >>> s = SelNot(OfType(Family.numeric()))
            >>> isinstance(s.after_retype("Int64"), SelNot)
            True
        """
        return SelNot(self.inner.after_retype(dtype))

    def canonical(self) -> Selector:
        """Give the one spelling of this negation.

        Returns:
            The canonical name filter if the negation reads only
            names. If not, the negation of the canonical inner
            selector.

        Example:
            >>> SelNot(OfType(Family.numeric())).canonical()
            SelNot(inner=OfType(family=NumericType))
        """
        rule = self.name_rule()
        if rule is not None:
            return CanonicalNameFilter()(rule)
        return SelNot(self.inner.canonical())

    def name_rule(self) -> Absence | None:
        """Negate the name rule of the inner selector, if it has one.

        Returns:
            The negated rule, or None if the inner selector reads a
            data type.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> SelNot(NameTest(StartsWith("a"))).name_rule()
            Not(inner=StartsWith(prefix='a'))
        """
        inner = self.inner.name_rule()
        if inner is None:
            return None
        return Not(inner)

    def mentioned_types(self) -> frozenset[str]:
        """Give the single data types that the inner selector names.

        Returns:
            The named data types of the inner selector.

        Example:
            >>> SelNot(OfType(Family.numeric())).mentioned_types()
            frozenset()
        """
        return self.inner.mentioned_types()

    def language_for(self, dtype: str) -> NameSet:
        """Give every name that the inner selector does not pick at one data type.

        Args:
            dtype: The data type to fix.

        Returns:
            The complement of the inner set at that data type.

        Example:
            >>> SelNot(OfType(Family.numeric())).language_for("String").accepts("x")
            True
        """
        return self.inner.language_for(dtype).complement()


@dataclass(frozen=True)
class SelAnd(Selector):
    """A selector that picks what both inner selectors pick, as `a & b`.

    Example:
        >>> from astrolabe._hooks.remainder.absence import StartsWith
        >>> both = SelAnd(OfType(Family.numeric()), NameTest(StartsWith("p")))
        >>> both.picks("price", "Int64")
        True
    """

    left: Selector
    right: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether both inner selectors pick the column.

        Args:
            name: The name of the column.
            dtype: The data type of the column.

        Returns:
            True if both sides pick it.

        Example:
            >>> both = SelAnd(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> both.picks("x", "Int64")
            False
        """
        return self.left.picks(name, dtype) and self.right.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick a name only if both sides certainly pick it.

        Args:
            types: The data types that the columns can have.

        Returns:
            The `And` of the two rules of certainly picked names.

        Example:
            >>> from astrolabe._hooks.remainder.absence import (
            ...     EndsWith, StartsWith,
            ... )
            >>> s = SelAnd(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_picks(TypeSet.anything()).holds_of("az")
            True
        """
        return And(self.left.certainly_picks(types), self.right.certainly_picks(types))

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject a name if either side certainly rejects it.

        Args:
            types: The data types that the columns can have.

        Returns:
            The `Or` of the two rules of certainly rejected names.

        Example:
            >>> from astrolabe._hooks.remainder.absence import (
            ...     EndsWith, StartsWith,
            ... )
            >>> s = SelAnd(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_rejects(TypeSet.anything()).holds_of("bz")
            True
        """
        return Or(self.left.certainly_rejects(types), self.right.certainly_rejects(types))

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the data types that a column picked by both sides can have.

        Args:
            types: The data types that the columns can have.

        Returns:
            The intersection of the two narrowed type sets.

        Example:
            >>> s = SelAnd(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> sorted(s.surviving_types(TypeSet.anything()).members)
            []
        """
        left = self.left.surviving_types(types)
        return left.intersection(self.right.surviving_types(types))

    def reads_only_types(self) -> bool:
        """Answer whether both sides read only data types.

        Returns:
            True if neither side reads a name.

        Example:
            >>> SelAnd(
            ...     OfType(Family.numeric()), OfType(Family.temporal())
            ... ).reads_only_types()
            True
        """
        return self.left.reads_only_types() and self.right.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Rewrite each side so that it runs before the suffix is added.

        Args:
            suffix: The suffix that the rename adds.

        Returns:
            The `SelAnd` of the rewritten sides.

        Example:
            >>> s = SelAnd(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.through_suffix("_p"), SelAnd)
            True
        """
        return SelAnd(self.left.through_suffix(suffix), self.right.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Rewrite each side so that it runs before the prefix is added.

        Args:
            prefix: The prefix that the rename adds.

        Returns:
            The `SelAnd` of the rewritten sides.

        Example:
            >>> s = SelAnd(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.through_prefix("s_"), SelAnd)
            True
        """
        return SelAnd(self.left.through_prefix(prefix), self.right.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Decide the tests on data types of each side, for the one data type.

        Args:
            dtype: The data type that every column now has.

        Returns:
            The `SelAnd` of the decided sides.

        Example:
            >>> s = SelAnd(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.after_retype("Int64"), SelAnd)
            True
        """
        return SelAnd(self.left.after_retype(dtype), self.right.after_retype(dtype))

    def canonical(self) -> Selector:
        """Give the one spelling of this conjunction.

        Returns:
            The canonical name filter if both sides read only names.
            If not, the canonical sides joined by `Conjunction`, which
            keeps only the narrower side if one contains the other.

        Example:
            >>> s = SelAnd(OfType(Family.temporal()), OfType(Family.numeric()))
            >>> isinstance(s.canonical(), SelAnd)
            True
        """
        rule = self.name_rule()
        if rule is not None:
            return CanonicalNameFilter()(rule)
        return Conjunction()(self.left.canonical(), self.right.canonical())

    def name_rule(self) -> Absence | None:
        """Join the name rules of the two sides with `And`, if both have one.

        Returns:
            The joined rule, or None if either side reads a data
            type.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> pure = SelAnd(NameTest(StartsWith("a")), NameTest(Always()))
            >>> pure.name_rule() is not None
            True
            >>> SelAnd(NameTest(Always()), OfType(Family.numeric())).name_rule()
        """
        left = self.left.name_rule()
        right = self.right.name_rule()
        if left is None or right is None:
            return None
        return And(left, right)

    def mentioned_types(self) -> frozenset[str]:
        """Give the single data types that either side names.

        Returns:
            The named data types of both sides.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> SelAnd(numeric, numeric).mentioned_types()
            frozenset()
        """
        return self.left.mentioned_types() | self.right.mentioned_types()

    def language_for(self, dtype: str) -> NameSet:
        """Give the names that both sides pick at one data type.

        Args:
            dtype: The data type to fix.

        Returns:
            The intersection of the two sets at that data type.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> s = SelAnd(numeric, SelNot(numeric))
            >>> s.language_for("Int64").empty()
            True
        """
        return self.left.language_for(dtype).intersection(self.right.language_for(dtype))


@dataclass(frozen=True)
class SelOr(Selector):
    """A selector that picks what either inner selector picks, as `a | b`.

    Example:
        >>> either = SelOr(OfType(Family.numeric()), OfType(Family.temporal()))
        >>> either.picks("x", "Date")
        True
    """

    left: Selector
    right: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether either inner selector picks the column.

        Args:
            name: The name of the column.
            dtype: The data type of the column.

        Returns:
            True if at least one side picks it.

        Example:
            >>> either = SelOr(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> either.picks("x", "String")
            False
        """
        return self.left.picks(name, dtype) or self.right.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick a name if either side certainly picks it.

        Args:
            types: The data types that the columns can have.

        Returns:
            The `Or` of the two rules of certainly picked names.

        Example:
            >>> from astrolabe._hooks.remainder.absence import (
            ...     EndsWith, StartsWith,
            ... )
            >>> s = SelOr(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_picks(TypeSet.anything()).holds_of("bz")
            True
        """
        return Or(self.left.certainly_picks(types), self.right.certainly_picks(types))

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject a name only if both sides certainly reject it.

        Args:
            types: The data types that the columns can have.

        Returns:
            The `And` of the two rules of certainly rejected names.

        Example:
            >>> from astrolabe._hooks.remainder.absence import (
            ...     EndsWith, StartsWith,
            ... )
            >>> s = SelOr(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_rejects(TypeSet.anything()).holds_of("bz")
            False
        """
        return And(self.left.certainly_rejects(types), self.right.certainly_rejects(types))

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the data types that a column picked by either side can have.

        Args:
            types: The data types that the columns can have.

        Returns:
            The union of the two narrowed type sets.

        Example:
            >>> mixed = TypeSet(frozenset({"Int64", "Date", "String"}))
            >>> s = SelOr(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> sorted(s.surviving_types(mixed).members)
            ['Date', 'Int64']
        """
        left = self.left.surviving_types(types)
        return left.union(self.right.surviving_types(types))

    def reads_only_types(self) -> bool:
        """Answer whether both sides read only data types.

        Returns:
            True if neither side reads a name.

        Example:
            >>> SelOr(
            ...     OfType(Family.numeric()), NameTest(Always())
            ... ).reads_only_types()
            False
        """
        return self.left.reads_only_types() and self.right.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Rewrite each side so that it runs before the suffix is added.

        Args:
            suffix: The suffix that the rename adds.

        Returns:
            The `SelOr` of the rewritten sides.

        Example:
            >>> s = SelOr(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.through_suffix("_p"), SelOr)
            True
        """
        return SelOr(self.left.through_suffix(suffix), self.right.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Rewrite each side so that it runs before the prefix is added.

        Args:
            prefix: The prefix that the rename adds.

        Returns:
            The `SelOr` of the rewritten sides.

        Example:
            >>> s = SelOr(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.through_prefix("s_"), SelOr)
            True
        """
        return SelOr(self.left.through_prefix(prefix), self.right.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Decide the tests on data types of each side, for the one data type.

        Args:
            dtype: The data type that every column now has.

        Returns:
            The `SelOr` of the decided sides.

        Example:
            >>> s = SelOr(OfType(Family.numeric()), OfType(Family.numeric()))
            >>> isinstance(s.after_retype("Int64"), SelOr)
            True
        """
        return SelOr(self.left.after_retype(dtype), self.right.after_retype(dtype))

    def canonical(self) -> Selector:
        """Give the one spelling of this disjunction.

        Returns:
            The canonical name filter if both sides read only names.
            If not, the `SelOr` of the canonical sides, sorted by
            their text so that the order the user wrote does not
            count.

        Example:
            >>> s = SelOr(OfType(Family.temporal()), OfType(Family.numeric()))
            >>> isinstance(s.canonical(), SelOr)
            True
            >>> pure = SelOr(
            ...     NameTest(OneOf(frozenset({"a"}))),
            ...     NameTest(OneOf(frozenset({"b"}))),
            ... )
            >>> pure.canonical() == NameTest(OneOf(frozenset({"a", "b"})))
            True
        """
        rule = self.name_rule()
        if rule is not None:
            return CanonicalNameFilter()(rule)
        first, second = sorted((self.left.canonical(), self.right.canonical()), key=StableText())
        return SelOr(first, second)

    def name_rule(self) -> Absence | None:
        """Join the name rules of the two sides with `Or`, if both have one.

        Returns:
            The joined rule, or None if either side reads a data
            type.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> pure = SelOr(NameTest(StartsWith("a")), NameTest(Never()))
            >>> pure.name_rule() is not None
            True
            >>> SelOr(NameTest(Always()), OfType(Family.numeric())).name_rule()
        """
        left = self.left.name_rule()
        right = self.right.name_rule()
        if left is None or right is None:
            return None
        return Or(left, right)

    def mentioned_types(self) -> frozenset[str]:
        """Give the single data types that either side names.

        Returns:
            The named data types of both sides.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> SelOr(numeric, numeric).mentioned_types()
            frozenset()
        """
        return self.left.mentioned_types() | self.right.mentioned_types()

    def language_for(self, dtype: str) -> NameSet:
        """Give the names that either side picks at one data type.

        Args:
            dtype: The data type to fix.

        Returns:
            The union of the two sets at that data type.

        Example:
            >>> s = SelOr(OfType(Family.numeric()), OfType(Family.temporal()))
            >>> s.language_for("Date").accepts("x")
            True
        """
        return self.left.language_for(dtype).union(self.right.language_for(dtype))


class SelectorEntailment:
    """Decide whether every column that one selector picks, another picks too.

    A canonical form needs this answer in three places. Two filters
    in a row keep only the narrower one if it is inside the other. A
    filter that picks every column is deleted. A filter that picks no
    column leaves an empty remainder. Cheap checks on the shape of the
    two selectors run first. `AbsenceEntailment` decides two name
    tests, and two families compare as type sets. A check on name sets
    at a few sample data types runs last and is exact, so the whole
    answer is exact.

    A selector never changes after it is built, so the answer for one
    pair holds for the whole process. The same pairs come back often,
    because each operation on a frame normalises its remainder again,
    so each answer is kept on the class.

    Example:
        >>> from astrolabe._hooks.remainder.absence import StartsWith
        >>> narrow = NameTest(StartsWith("ab"))
        >>> SelectorEntailment()(narrow, NameTest(StartsWith("a")))
        True
    """

    # The answer for each pair of selectors that a check has decided.
    _ANSWERS: ClassVar[dict[tuple[Selector, Selector], bool]] = {}

    def __call__(self, left: Selector, right: Selector) -> bool:
        """Answer whether every column that `left` picks, `right` picks too.

        The checks run in order, and the first True answers. The cheap
        checks come first. The exact check on name sets comes last.

        Args:
            left: The selector that must be the narrower one.
            right: The selector that must be the wider one.

        Returns:
            True if the containment holds.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> SelectorEntailment()(numeric, numeric)
            True
            >>> SelectorEntailment()(numeric, OfType(Family.temporal()))
            False
        """
        pair = (left, right)
        if pair not in self._ANSWERS:
            rules = (
                self._identical,
                self._both_name_tests,
                self._both_type_tests,
                self._left_conjunction,
                self._right_conjunction,
                self._disjunctions,
                self._negations,
                self._by_automaton,
            )
            self._ANSWERS[pair] = any(rule(left, right) for rule in rules)
        return self._ANSWERS[pair]

    def _by_automaton(self, left: Selector, right: Selector) -> bool:
        """Decide the containment exactly, one sample data type at a time.

        The two selectors can tell apart only a few groups of data
        types, and one sample stands for each group (see
        `TypeSamples`). So the containment holds exactly if, at each
        sample, every name that `left` picks is a name that `right`
        picks. Each such question compares two finite automata, so
        this check is the slowest.

        Args:
            left: The selector that must be the narrower one.
            right: The selector that must be the wider one.

        Returns:
            True exactly if every column that `left` picks, `right`
            picks.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> contradiction = SelAnd(numeric, SelNot(numeric))
            >>> SelectorEntailment()._by_automaton(
            ...     contradiction, OfType(Family.temporal())
            ... )
            True
        """
        samples = TypeSamples().of(left.mentioned_types() | right.mentioned_types())
        return all(left.language_for(d).contained_in(right.language_for(d)) for d in samples)

    def _identical(self, left: Selector, right: Selector) -> bool:
        """Answer whether the two selectors are equal, which is the cheapest check.

        Args:
            left: One selector.
            right: The other.

        Returns:
            True if they are equal.

        Example:
            >>> numeric = OfType(Family.numeric())
            >>> SelectorEntailment()._identical(numeric, numeric)
            True
        """
        return left == right

    def _both_name_tests(self, left: Selector, right: Selector) -> bool:
        """Decide two name tests by comparing their name rules.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True if both are name tests, and the rule of `left` holds
            of no name that the rule of `right` does not.

        Example:
            >>> from astrolabe._hooks.remainder.absence import OneOf
            >>> SelectorEntailment()._both_name_tests(
            ...     NameTest(OneOf(frozenset({"a"}))), NameTest(Always())
            ... )
            True
        """
        if not isinstance(left, NameTest) or not isinstance(right, NameTest):
            return False
        return AbsenceEntailment()(left.rule, right.rule)

    def _both_type_tests(self, left: Selector, right: Selector) -> bool:
        """Decide two tests on data types by comparing their families.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True if both are `OfType`, and the family of `right` holds
            every data type of the family of `left`.

        Example:
            >>> ints = Family("Integer", frozenset({"Int64", "Int32"}))
            >>> SelectorEntailment()._both_type_tests(
            ...     OfType(ints), OfType(Family.numeric())
            ... )
            True
        """
        if not isinstance(left, OfType) or not isinstance(right, OfType):
            return False
        return left.family.type_set().within(right.family.type_set())

    def _left_conjunction(self, left: Selector, right: Selector) -> bool:
        """Answer True if `left` is a `SelAnd`, and one of its sides is inside `right`.

        A `SelAnd` picks no more than either side, so a containment of
        one side is enough.

        Args:
            left: The selector, which must be a `SelAnd`.
            right: The wider selector.

        Returns:
            True if either side proves the containment.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> both = SelAnd(NameTest(StartsWith("ab")), OfType(Family.numeric()))
            >>> SelectorEntailment()._left_conjunction(
            ...     both, NameTest(StartsWith("a"))
            ... )
            True
        """
        if not isinstance(left, SelAnd):
            return False
        return self(left.left, right) or self(left.right, right)

    def _right_conjunction(self, left: Selector, right: Selector) -> bool:
        """Answer True if `right` is a `SelAnd`, and `left` is inside both sides.

        Args:
            left: The narrower selector.
            right: The selector, which must be a `SelAnd`.

        Returns:
            True if both sides contain `left`.

        Example:
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> numeric = OfType(Family.numeric())
            >>> wide = SelAnd(NameTest(StartsWith("a")), numeric)
            >>> narrow = SelAnd(NameTest(StartsWith("ab")), numeric)
            >>> SelectorEntailment()._right_conjunction(narrow, wide)
            True
        """
        if not isinstance(right, SelAnd):
            return False
        return self(left, right.left) and self(left, right.right)

    def _negations(self, left: Selector, right: Selector) -> bool:
        """Decide the two negations that pick every column or none.

        The negation of a selector that picks nothing picks every
        column. The negation of a selector that picks every column
        picks nothing. Both appear after a new data type decides a
        test on data types, so this cheap check catches them.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True if `right` picks every column, or `left` picks none,
            for this reason.

        Example:
            >>> total = SelNot(NameTest(Never()))
            >>> SelectorEntailment()._negations(NameTest(Always()), total)
            True
            >>> empty = SelNot(NameTest(Always()))
            >>> SelectorEntailment()._negations(empty, NameTest(Never()))
            True
        """
        if isinstance(right, SelNot) and self(right.inner, NameTest(Never())):
            return True
        if not isinstance(left, SelNot):
            return False
        return self(NameTest(Always()), left.inner)

    def _disjunctions(self, left: Selector, right: Selector) -> bool:
        """Answer the containment if either side is a `SelOr`.

        A `SelOr` is inside a selector if both of its sides are. A
        selector is inside a `SelOr` if it is inside either side. The
        second test is enough to prove a containment, but it can miss
        one that only the whole `SelOr` gives. The exact check then
        decides it.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True if either test on a `SelOr` proves the containment.

        Example:
            >>> temporal = OfType(Family.temporal())
            >>> wide = SelOr(OfType(Family.numeric()), temporal)
            >>> SelectorEntailment()._disjunctions(temporal, wide)
            True
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> either = SelOr(
            ...     NameTest(StartsWith("ab")), NameTest(StartsWith("ac"))
            ... )
            >>> SelectorEntailment()._disjunctions(either, NameTest(StartsWith("a")))
            True
        """
        if isinstance(left, SelOr) and self(left.left, right) and self(left.right, right):
            return True
        if not isinstance(right, SelOr):
            return False
        return self(left, right.left) or self(left, right.right)


class CanonicalNameFilter:
    """Spell a selector that reads only names by the set of names it picks.

    `drop("a").drop("b")` and `drop(["a", "b"])` leave the same
    columns, so they must give the same type. So a filter that reads
    only names is spelled by the set of names it picks, and never by
    how it was built. The spelling uses a shape that a remainder type
    can show. A finite set is a list of names. A set whose complement
    is finite is the negation of that list. Any other set is a
    `Matching` rule, keyed by the signature of its automaton.

    Example:
        >>> gone = And(
        ...     Not(OneOf(frozenset({"a"}))), Not(OneOf(frozenset({"b"})))
        ... )
        >>> CanonicalNameFilter()(gone) == SelNot(
        ...     NameTest(OneOf(frozenset({"a", "b"})))
        ... )
        True
    """

    def __call__(self, rule: Absence) -> Selector:
        """Give the canonical selector that picks exactly the names of a rule.

        Args:
            rule: A rule on names alone.

        Returns:
            The list of the names if the set is finite. If not, the
            negation of the finite complement if it exists. If not, a
            `Matching` test on the set.

        Example:
            >>> CanonicalNameFilter()(OneOf(frozenset({"a"})))
            NameTest(rule=OneOf(names=frozenset({'a'})))
            >>> from astrolabe._hooks.remainder.absence import StartsWith
            >>> spelled = CanonicalNameFilter()(StartsWith("a"))
            >>> spelled == NameTest(Matching.of(StartsWith("a").language()))
            True
        """
        language = rule.language()
        names = language.finite_names()
        if names is not None:
            return NameTest(OneOf(names))
        complement = language.complement().finite_names()
        if complement is not None:
            return SelNot(NameTest(OneOf(complement)))
        return NameTest(Matching.of(language))


class Conjunction:
    """Build the selector that picks what two selectors both pick, with no part that adds nothing.

    If one selector picks no more than the other, the pair means the
    narrower one alone, so this class gives only that one. Then two
    filters in a row give the same value as one filter that picks the
    same columns, and a canonical form cannot tell them apart. If
    neither contains the other, the result is a `SelAnd` with its two
    sides in a fixed order.

    Example:
        >>> from astrolabe._hooks.remainder.absence import StartsWith
        >>> narrow = NameTest(StartsWith("ab"))
        >>> Conjunction()(NameTest(StartsWith("a")), narrow)
        NameTest(rule=StartsWith(prefix='ab'))
    """

    def __call__(self, left: Selector, right: Selector) -> Selector:
        """Give the selector that picks what both pick.

        The order of the two sides of a `SelAnd` does not change what
        it picks. A node that kept the order of the call gives two
        spellings of one selector, so the sides are sorted by their
        text.

        Args:
            left: One selector.
            right: The other.

        Returns:
            The narrower one if one contains the other. If not, their
            `SelAnd`.

        Example:
            >>> joined = Conjunction()(
            ...     OfType(Family.temporal()), OfType(Family.numeric())
            ... )
            >>> joined
            SelAnd(left=OfType(family=NumericType), right=OfType(family=TemporalType))
            >>> a, b = OfType(Family.numeric()), OfType(Family.temporal())
            >>> Conjunction()(b, a) == Conjunction()(a, b)
            True
        """
        entails = SelectorEntailment()
        if entails(left, right):
            return left
        if entails(right, left):
            return right
        first, second = sorted((left, right), key=StableText())
        return SelAnd(first, second)
