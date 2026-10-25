"""A reference implementation of the specification, used to check its claims.

The specification describes a type system over dataframe schemas. Most
of what it says is a statement about finite objects: a column set, an
absence rule, a selector, an effect. This module builds those objects
for real, so a claim in a document can be checked by running it instead
of by reading it.

Three languages appear here, and keeping them apart is the point.

- An **absence rule** tests a name. It answers "is this name provably
  absent?". `Absence` is its base class.
- A **selector** tests a whole column, name and data type together. It
  answers "does this operation pick this column?". `Selector` is its
  base class. Every name test is also a selector, and `OfType` is a
  selector that is not a name test. So the selector language is
  strictly larger than the absence language.
- A **type set** records which data types a group of columns can have.
  `TypeSet` is that class.

The specification's `matching(E)` constructor is `Matching` here. It
holds a callable, which is how an arbitrary name expression is written
in Python. Nothing a user writes produces one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from functools import reduce
from typing import TYPE_CHECKING

from greenery import Charclass, Fsm, parse

if TYPE_CHECKING:
    from collections.abc import Iterable


class StableText:
    """Render a value as text that does not depend on set ordering.

    A `frozenset` reprs its members in hash order, which differs
    between processes. Sorting a union's operands needs an order that
    does not, or two runs would canonicalise the same expression two
    ways.

    Example:
        >>> StableText()(frozenset({"b", "a"}))
        "{'a','b'}"
    """

    def __call__(self, value: object) -> str:
        """Render one value.

        A set renders its members sorted by their own rendering. A
        dataclass renders its class name and each compared field. Any
        other value renders as its repr.

        Args:
            value: The value to render.

        Returns:
            Text that is the same in every process for equal values.

        Example:
            >>> StableText()("a")
            "'a'"
            >>> StableText()(frozenset({2, 1}))
            '{1,2}'
            >>> from spec.tests.model import StartsWith
            >>> StableText()(StartsWith("a"))
            "StartsWith('a')"
        """
        if isinstance(value, frozenset):
            members = sorted(self(member) for member in value)
            return "{" + ",".join(members) + "}"
        if is_dataclass(value) and not isinstance(value, type):
            parts = [self(getattr(value, f.name)) for f in fields(value) if f.compare]
            return f"{type(value).__name__}({','.join(parts)})"
        return repr(value)


class Absence:
    """A test on a name that answers whether the name is provably absent.

    Every subclass is one constructor of the absence language. The
    language is closed under the effects, which is what lets a chain of
    operations keep proving names absent.

    Example:
        >>> rule = Or(OneOf({"a"}), StartsWith("tmp_"))
        >>> rule.holds_of("a"), rule.holds_of("tmp_x"), rule.holds_of("b")
        (True, True, False)
    """

    def holds_of(self, name: str) -> bool:
        """Answer whether this rule proves `name` absent.

        Args:
            name: The column name to test.

        Returns:
            True when the name is provably absent.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Never().holds_of("anything")
            False
        """
        raise NotImplementedError

    def language(self) -> Fsm:
        """Give the machine accepting exactly the names this rule holds of.

        Every constructor denotes a regular set, which is what makes
        containment between two rules decidable.

        Returns:
            A machine accepting this rule's names.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Always().language().accepts("anything")
            True
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Never(Absence):
    """No name is absent. This is the unconstrained remainder.

    Example:
        >>> Never().holds_of("venue")
        False
    """

    def holds_of(self, name: str) -> bool:
        """Answer False for every name.

        Args:
            name: Ignored.

        Returns:
            False, always.

        Example:
            >>> Never().holds_of("")
            False
        """
        return False

    def language(self) -> Fsm:
        """Give the machine accepting no name.

        Returns:
            The empty language.

        Example:
            >>> Never().language().accepts("")
            False
        """
        return RegularLanguage().nothing()


@dataclass(frozen=True)
class Always(Absence):
    """Every name is absent. This is the empty remainder.

    Example:
        >>> Always().holds_of("venue")
        True
    """

    def holds_of(self, name: str) -> bool:
        """Answer True for every name.

        Args:
            name: Ignored.

        Returns:
            True, always.

        Example:
            >>> Always().holds_of("")
            True
        """
        return True

    def language(self) -> Fsm:
        """Give the machine accepting every name.

        Returns:
            The total language.

        Example:
            >>> Always().language().accepts("x")
            True
        """
        return RegularLanguage().anything()


@dataclass(frozen=True)
class OneOf(Absence):
    """The names in a finite set are absent.

    Example:
        >>> OneOf(frozenset({"a", "b"})).holds_of("a")
        True
    """

    names: frozenset[str]

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` is one of the listed names.

        Args:
            name: The column name to test.

        Returns:
            True when the name is in the set.

        Example:
            >>> OneOf(frozenset({"a"})).holds_of("b")
            False
        """
        return name in self.names

    def language(self) -> Fsm:
        """Give the machine accepting exactly the listed names.

        Returns:
            The union of one literal per name.

        Example:
            >>> OneOf(frozenset({"a", "b"})).language().accepts("b")
            True
        """
        language = RegularLanguage()
        machine = language.nothing()
        for name in sorted(self.names):
            machine = machine.union(language.literal(name))
        return machine


@dataclass(frozen=True)
class StartsWith(Absence):
    """Every name that begins with a fixed string is absent.

    Example:
        >>> StartsWith("sec_").holds_of("sec_country")
        True
    """

    prefix: str

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` begins with this prefix.

        Args:
            name: The column name to test.

        Returns:
            True when the name begins with the prefix.

        Example:
            >>> StartsWith("sec_").holds_of("venue")
            False
        """
        return name.startswith(self.prefix)

    def language(self) -> Fsm:
        """Give the machine accepting every name with this prefix.

        Returns:
            The prefix, followed by anything.

        Example:
            >>> StartsWith("ab").language().accepts("abc")
            True
        """
        language = RegularLanguage()
        return language.literal(self.prefix).concatenate(language.anything())


@dataclass(frozen=True)
class EndsWith(Absence):
    """Every name that ends in a fixed string is absent.

    Example:
        >>> EndsWith("_prev").holds_of("price_prev")
        True
    """

    suffix: str

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` ends in this suffix.

        Args:
            name: The column name to test.

        Returns:
            True when the name ends in the suffix.

        Example:
            >>> EndsWith("_prev").holds_of("price")
            False
        """
        return name.endswith(self.suffix)

    def language(self) -> Fsm:
        """Give the machine accepting every name with this suffix.

        Returns:
            Anything, followed by the suffix.

        Example:
            >>> EndsWith("_p").language().accepts("x_p")
            True
        """
        language = RegularLanguage()
        return language.anything().concatenate(language.literal(self.suffix))


@dataclass(frozen=True)
class Contains(Absence):
    """Every name that holds a fixed string anywhere is absent.

    Example:
        >>> Contains("tmp").holds_of("x_tmp_y")
        True
    """

    part: str

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` holds this substring.

        Args:
            name: The column name to test.

        Returns:
            True when the substring occurs in the name.

        Example:
            >>> Contains("tmp").holds_of("venue")
            False
        """
        return self.part in name

    def language(self) -> Fsm:
        """Give the machine accepting every name holding this substring.

        Returns:
            Anything, the substring, then anything.

        Example:
            >>> Contains("bc").language().accepts("abcd")
            True
        """
        language = RegularLanguage()
        middle = language.anything().concatenate(language.literal(self.part))
        return middle.concatenate(language.anything())


@dataclass(frozen=True)
class Matching(Absence):
    """A name test the checker derived, carried as the machine that decides it.

    Moving a filter past a renaming produces a test that is not always
    one of the named shapes. This constructor is where such a test
    lands. Nothing a user writes produces one.

    Two of these are equal exactly when they accept the same names. The
    machine is excluded from comparison and the signature stands in for
    it: the signature is built from the minimal machine and is the same
    text for any two machines accepting the same names.

    Example:
        >>> language = RegularLanguage()
        >>> Matching.of(language.literal("abc")).holds_of("abc")
        True
    """

    signature: str
    machine: Fsm = field(compare=False, repr=False)

    @classmethod
    def of(cls, machine: Fsm) -> Matching:
        """Build the rule deciding exactly what a machine accepts.

        Args:
            machine: The machine to carry.

        Returns:
            A rule holding it, keyed by its canonical signature.

        Example:
            >>> language = RegularLanguage()
            >>> a = Matching.of(language.of(StartsWith("a")))
            >>> a == Matching.of(language.of(Or(StartsWith("a"), StartsWith("ab"))))
            True
        """
        return cls(RegularLanguage().signature(machine), machine)

    def holds_of(self, name: str) -> bool:
        """Answer whether the held machine accepts `name`.

        Args:
            name: The column name to test.

        Returns:
            True when the machine accepts it.

        Example:
            >>> Matching.of(RegularLanguage().literal("a")).holds_of("b")
            False
        """
        return bool(self.machine.accepts(name))

    def language(self) -> Fsm:
        """Give the machine this rule decides with.

        Returns:
            The held machine.

        Example:
            >>> Matching.of(RegularLanguage().literal("a")).language().accepts("a")
            True
        """
        return self.machine


@dataclass(frozen=True)
class Not(Absence):
    """A name is absent exactly where the inner rule does not hold.

    Example:
        >>> Not(StartsWith("a")).holds_of("b")
        True
    """

    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer the negation of the inner rule.

        Args:
            name: The column name to test.

        Returns:
            True when the inner rule is False.

        Example:
            >>> Not(Always()).holds_of("x")
            False
        """
        return not self.inner.holds_of(name)

    def language(self) -> Fsm:
        """Give the machine accepting every name the inner rule does not.

        Returns:
            The complement of the inner rule's language.

        Example:
            >>> Not(StartsWith("a")).language().accepts("b")
            True
        """
        return self.inner.language().everythingbut()


@dataclass(frozen=True)
class And(Absence):
    """A name is absent when both rules hold of it.

    Example:
        >>> And(StartsWith("a"), EndsWith("z")).holds_of("az")
        True
    """

    left: Absence
    right: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether both rules hold.

        Args:
            name: The column name to test.

        Returns:
            True when both sides hold.

        Example:
            >>> And(StartsWith("a"), EndsWith("z")).holds_of("ab")
            False
        """
        return self.left.holds_of(name) and self.right.holds_of(name)

    def language(self) -> Fsm:
        """Give the machine accepting the names both rules hold of.

        Returns:
            The intersection of the two languages.

        Example:
            >>> And(StartsWith("a"), EndsWith("z")).language().accepts("az")
            True
        """
        return self.left.language().intersection(self.right.language())


@dataclass(frozen=True)
class Or(Absence):
    """A name is absent when either rule holds of it.

    Example:
        >>> Or(OneOf(frozenset({"a"})), OneOf(frozenset({"b"}))).holds_of("b")
        True
    """

    left: Absence
    right: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether either rule holds.

        Args:
            name: The column name to test.

        Returns:
            True when at least one side holds.

        Example:
            >>> Or(Never(), Never()).holds_of("a")
            False
        """
        return self.left.holds_of(name) or self.right.holds_of(name)

    def language(self) -> Fsm:
        """Give the machine accepting the names either rule holds of.

        Returns:
            The union of the two languages.

        Example:
            >>> Or(StartsWith("a"), StartsWith("b")).language().accepts("b")
            True
        """
        return self.left.language().union(self.right.language())


@dataclass(frozen=True)
class AfterStripping(Absence):
    """A name ends in a suffix, and the inner rule holds without it.

    This constructor exists for the suffixing effect. Read it with a
    negated suffix test beside it: the pair says the name did not come
    from adding the suffix to a name the operand held.

    Example:
        >>> AfterStripping("_prev", OneOf(frozenset({"price"}))).holds_of("price_prev")
        True
    """

    suffix: str
    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` ends in the suffix and the rest is absent.

        Args:
            name: The column name to test.

        Returns:
            True when the name ends in the suffix and the inner rule
            holds of the name with that suffix removed.

        Example:
            >>> AfterStripping("_prev", Always()).holds_of("price")
            False
        """
        if not name.endswith(self.suffix):
            return False
        return self.inner.holds_of(name[: len(name) - len(self.suffix)])

    def language(self) -> Fsm:
        """Give the machine accepting the inner names with the suffix added.

        Returns:
            The inner language, followed by the suffix.

        Example:
            >>> rule = AfterStripping("_p", OneOf(frozenset({"price"})))
            >>> rule.language().accepts("price_p")
            True
        """
        return self.inner.language().concatenate(RegularLanguage().literal(self.suffix))


@dataclass(frozen=True)
class BeforeStripping(Absence):
    """A name begins with a prefix, and the inner rule holds without it.

    This is `AfterStripping` with the ends swapped. It exists for the
    prefixing effect.

    Example:
        >>> BeforeStripping("sec_", OneOf(frozenset({"country"}))).holds_of("sec_country")
        True
    """

    prefix: str
    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` begins with the prefix and the rest is absent.

        Args:
            name: The column name to test.

        Returns:
            True when the name begins with the prefix and the inner
            rule holds of the name with that prefix removed.

        Example:
            >>> BeforeStripping("sec_", Always()).holds_of("country")
            False
        """
        if not name.startswith(self.prefix):
            return False
        return self.inner.holds_of(name[len(self.prefix) :])

    def language(self) -> Fsm:
        """Give the machine accepting the inner names with the prefix added.

        Returns:
            The prefix, followed by the inner language.

        Example:
            >>> rule = BeforeStripping("s_", OneOf(frozenset({"a"})))
            >>> rule.language().accepts("s_a")
            True
        """
        return RegularLanguage().literal(self.prefix).concatenate(self.inner.language())


@dataclass(frozen=True)
class Family:
    """A named group of data types, such as every numeric type.

    A family is what `all_of=` and `none_of=` take, and what the
    `OfType` selector tests against.

    Example:
        >>> NUMERIC.name
        'NumericType'
    """

    name: str
    members: frozenset[str]

    def __repr__(self) -> str:
        """Give the family's name, so a repr does not depend on set ordering.

        A `frozenset` reprs its members in hash order, which differs
        between processes. A test identifier built from this repr would
        then differ between them too.

        Returns:
            The family's name.

        Example:
            >>> NUMERIC
            NumericType
        """
        return self.name

    def holds(self, dtype: str) -> bool:
        """Answer whether `dtype` belongs to this family.

        Args:
            dtype: The data type to test.

        Returns:
            True when the data type is a member.

        Example:
            >>> NUMERIC.holds("Int64"), NUMERIC.holds("String")
            (True, False)
        """
        return dtype in self.members


NUMERIC = Family("NumericType", frozenset({"Int64", "Int32", "Float64", "Float32"}))
TEMPORAL = Family("TemporalType", frozenset({"Date", "Datetime", "Time", "Duration"}))
EVERY_DTYPE = frozenset(NUMERIC.members | TEMPORAL.members | {"String", "Boolean"})


@dataclass(frozen=True)
class TypeSet:
    """The data types the columns of a remainder can have.

    This is the second half of what a remainder records. The first half
    is an absence rule. A requirement about data types reads this half,
    and the `OfType` selector is decided from it.

    Example:
        >>> TypeSet(frozenset({"Int64"})).contained_in(NUMERIC)
        True
    """

    members: frozenset[str]

    @classmethod
    def anything(cls) -> TypeSet:
        """Build the type set that permits every data type.

        Returns:
            The type set an unconstrained remainder carries.

        Example:
            >>> TypeSet.anything().contained_in(NUMERIC)
            False
        """
        return cls(EVERY_DTYPE)

    def contained_in(self, family: Family) -> bool:
        """Answer whether every permitted data type is in `family`.

        This decides the `all_of=` requirement.

        Args:
            family: The family to test containment against.

        Returns:
            True when no permitted type falls outside the family.

        Example:
            >>> TypeSet(frozenset({"Int64", "String"})).contained_in(NUMERIC)
            False
        """
        return all(family.holds(d) for d in self.members)

    def disjoint_from(self, family: Family) -> bool:
        """Answer whether no permitted data type is in `family`.

        This decides the `none_of=` requirement.

        Args:
            family: The family to test disjointness against.

        Returns:
            True when no permitted type falls inside the family.

        Example:
            >>> TypeSet(frozenset({"String"})).disjoint_from(NUMERIC)
            True
        """
        return not any(family.holds(d) for d in self.members)

    def narrowed_to(self, family: Family) -> TypeSet:
        """Keep only the permitted types that `family` holds.

        This is what a type selector does to the record, and it is the
        rule the specification's closure table was missing.

        Args:
            family: The family to intersect with.

        Returns:
            The type set holding the permitted types in the family.

        Example:
            >>> sorted(TypeSet(frozenset({"Int64", "String"})).narrowed_to(NUMERIC).members)
            ['Int64']
        """
        return TypeSet(frozenset(d for d in self.members if family.holds(d)))

    def combined_with(self, other: TypeSet) -> TypeSet:
        """Permit every type either set permits.

        This is what combining two remainders does to the record.

        Args:
            other: The type set to union with.

        Returns:
            The union of the two type sets.

        Example:
            >>> a = TypeSet(frozenset({"Int64"}))
            >>> sorted(a.combined_with(TypeSet(frozenset({"String"}))).members)
            ['Int64', 'String']
        """
        return TypeSet(self.members | other.members)


@dataclass(frozen=True)
class ColumnSet:
    """A finite map from column name to data type.

    This is one concrete answer to "what columns are there". A frame
    that runs has exactly one. Every description in this module denotes
    a set of these.

    Example:
        >>> ColumnSet.of(price="Float64", venue="String").names()
        ('price', 'venue')
    """

    entries: tuple[tuple[str, str], ...]

    @classmethod
    def holding(cls, entries: Iterable[tuple[str, str]]) -> ColumnSet:
        """Build a column set from entries, keeping each one once.

        A column set is a map, so one name maps to one data type and a
        repeated entry adds nothing. Two entries sharing a name and
        differing in data type are outside what any operation here
        produces, and both are kept so that nothing is silently merged.

        Args:
            entries: The name and data type pairs to hold.

        Returns:
            The column set holding them, sorted, without repeats.

        Example:
            >>> ColumnSet.holding((("a", "Int64"), ("a", "Int64"))).entries
            (('a', 'Int64'),)
        """
        return cls(tuple(sorted(dict.fromkeys(entries))))

    @classmethod
    def of(cls, **columns: str) -> ColumnSet:
        """Build a column set from keyword arguments.

        Args:
            **columns: Each keyword is a column name and its value the
                data type.

        Returns:
            The column set holding those columns.

        Example:
            >>> ColumnSet.of(a="Int64").entries
            (('a', 'Int64'),)
        """
        return cls(tuple(sorted(columns.items())))

    def names(self) -> tuple[str, ...]:
        """List the column names in order.

        Returns:
            Every column name, sorted.

        Example:
            >>> ColumnSet.of(b="Int64", a="String").names()
            ('a', 'b')
        """
        return tuple(name for name, _ in self.entries)

    def type_set(self) -> TypeSet:
        """Record which data types these columns have.

        Returns:
            The type set holding exactly the data types present.

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
            True when the name is present.

        Example:
            >>> ColumnSet.of(a="Int64").holds("b")
            False
        """
        return name in self.names()


@dataclass(frozen=True)
class Record:
    """What is recorded about a remainder: an absence rule and a type set.

    These are the two halves the specification names. Every effect must
    say what it does to both, or a chain of operations would lose track
    of one of them.

    Example:
        >>> Record(OneOf(frozenset({"a"})), TypeSet.anything()).absence.holds_of("a")
        True
    """

    absence: Absence
    types: TypeSet

    @classmethod
    def unconstrained(cls) -> Record:
        """Build the record of a remainder that promises nothing.

        Returns:
            The record with no name proved absent and every data type
            permitted.

        Example:
            >>> Record.unconstrained().absence.holds_of("anything")
            False
        """
        return cls(Never(), TypeSet.anything())

    def describes(self, columns: ColumnSet) -> bool:
        """Answer whether `columns` is a possible value of this remainder.

        A column set is possible when it uses no provably absent name
        and draws every data type from the permitted set.

        Args:
            columns: The concrete column set to test.

        Returns:
            True when nothing recorded rules the column set out.

        Example:
            >>> r = Record(OneOf(frozenset({"a"})), TypeSet.anything())
            >>> r.describes(ColumnSet.of(a="Int64"))
            False
            >>> r.describes(ColumnSet.of(b="Int64"))
            True
        """
        names_free = not any(self.absence.holds_of(n) for n in columns.names())
        return names_free and columns.type_set().members <= self.types.members


class Selector:
    """A test on a whole column, name and data type together.

    A selector says which columns an operation keeps. Every constructor
    of the absence language becomes a selector through `NameTest`.
    `OfType` is a selector that is not a name test, which is why the
    selector language is strictly larger than the absence language.

    Three methods matter, and the last two are what let a selector act
    on a record rather than on a column. `picks` decides one real
    column. `certainly_picks` and `certainly_rejects` give the names a
    selector settles without seeing the column, which is what an
    operation over a remainder has to work from.

    Example:
        >>> NameTest(StartsWith("tmp_")).picks("tmp_x", "Int64")
        True
    """

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether this selector picks the given column.

        Args:
            name: The column's name.
            dtype: The column's data type.

        Returns:
            True when the selector picks the column.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).picks("x", "Int64")
            True
        """
        raise NotImplementedError

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Give the names this selector is certain to pick.

        Args:
            types: What is known about the columns' data types.

        Returns:
            A name rule that holds only where every column of that name
            is picked, whatever its data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).certainly_picks(TypeSet(frozenset({"Int64"})))
            Always()
        """
        raise NotImplementedError

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Give the names this selector is certain to reject.

        This is what an operation over a remainder needs: a name it
        certainly rejects is provably absent from the result.

        Args:
            types: What is known about the columns' data types.

        Returns:
            A name rule that holds only where every column of that name
            is rejected, whatever its data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).certainly_rejects(TypeSet(frozenset({"String"})))
            Always()
        """
        raise NotImplementedError

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Narrow a type set to the types that can survive this selector.

        The default answer keeps every type, which is sound for any
        selector. `OfType` gives the exact answer instead.

        Args:
            types: What is known about the columns' data types.

        Returns:
            A type set holding every type a surviving column can have.

        Example:
            >>> sorted(NameTest(Always()).surviving_types(TypeSet(frozenset({"Int64"}))).members)
            ['Int64']
        """
        return types

    def reads_only_types(self) -> bool:
        """Answer whether this selector ignores column names entirely.

        Negation can give an exact type answer only for a selector that
        reads types alone, so `SelNot` asks this before narrowing.

        Returns:
            True when no part of this selector reads a name.

        Example:
            >>> OfType(NUMERIC).reads_only_types(), NameTest(Always()).reads_only_types()
            (True, False)
        """
        return False

    def name_rule(self) -> Absence | None:
        """Give this selector's decision as a pure name rule.

        A selector built only from name tests and boolean combinators
        is decided by names alone, so one absence rule carries its
        whole meaning. A combinator's `canonical` keys such a selector
        by that rule's language, which is what makes two filters that
        accept the same names the same selector.

        Returns:
            The rule, or None where any part reads a data type.

        Example:
            >>> OfType(NUMERIC).name_rule() is None
            True
        """
        return None

    def through_suffix(self, suffix: str) -> Selector:
        """Translate this selector to run before a suffix is appended.

        A selector that runs after a rename tests a name the rename has
        already changed. To move it before the rename, it must test the
        name the rename will produce.

        Args:
            suffix: The suffix the rename appends.

        Returns:
            A selector that picks a column exactly when this one picks
            the same column after the suffix is appended.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).through_suffix("_p") == OfType(NUMERIC)
            True
        """
        raise NotImplementedError

    def through_prefix(self, prefix: str) -> Selector:
        """Translate this selector to run before a prefix is prepended.

        Args:
            prefix: The prefix the rename prepends.

        Returns:
            A selector that picks a column exactly when this one picks
            the same column after the prefix is prepended.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).through_prefix("s_") == OfType(NUMERIC)
            True
        """
        raise NotImplementedError

    def after_retype(self, dtype: str) -> Selector:
        """Resolve every type test against one known data type.

        A selector running after every column took one data type can
        decide its type tests outright, which turns it into a pure name
        test.

        Args:
            dtype: The data type every column now has.

        Returns:
            A selector with no remaining type test.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).after_retype("Int64")
            NameTest(rule=Always())
        """
        raise NotImplementedError

    def canonical(self) -> Selector:
        """Give the form of this selector that compares by what it picks.

        Every name test becomes one keyed by its language, so two
        selectors picking the same columns compare equal however each
        was written. The rule a user wrote is kept beside the key, and
        an error message reports that rather than the key.

        Returns:
            An equivalent selector, keyed by language.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).canonical()
            OfType(family=NumericType)
        """
        raise NotImplementedError

    def language_for(self, dtype: str) -> Fsm:
        """Give the names this selector picks among columns of one data type.

        A selector reads a name and a data type. There are finitely
        many data types, and the names it picks for each one form a
        regular set, so a selector is decided by one machine per data
        type. That is what makes containment between two selectors
        decidable.

        Args:
            dtype: The data type to fix.

        Returns:
            A machine accepting the names picked at that data type.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> OfType(NUMERIC).language_for("Int64").accepts("anything")
            True
        """
        raise NotImplementedError


@dataclass(frozen=True)
class NameTest(Selector):
    """A selector that reads only the column's name.

    This is how every constructor of the absence language becomes a
    selector. The two languages share these constructors, and differ by
    `OfType` alone.

    Example:
        >>> NameTest(OneOf(frozenset({"a"}))).picks("a", "String")
        True
    """

    rule: Absence

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether the held name rule accepts `name`.

        Args:
            name: The column's name.
            dtype: Ignored. A name test does not read the data type.

        Returns:
            True when the held rule holds of the name.

        Example:
            >>> NameTest(EndsWith("_prev")).picks("price", "Int64")
            False
        """
        del dtype
        return self.rule.holds_of(name)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Give the names picked, which for a name test is exact.

        Args:
            types: Ignored. A name test does not read data types.

        Returns:
            The held name rule itself.

        Example:
            >>> NameTest(StartsWith("a")).certainly_picks(TypeSet.anything())
            StartsWith(prefix='a')
        """
        del types
        return self.rule

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Give the names rejected, which for a name test is exact.

        Args:
            types: Ignored. A name test does not read data types.

        Returns:
            The negation of the held name rule.

        Example:
            >>> NameTest(StartsWith("a")).certainly_rejects(TypeSet.anything())
            Not(inner=StartsWith(prefix='a'))
        """
        del types
        return Not(self.rule)

    def through_suffix(self, suffix: str) -> Selector:
        """Test the name this rename will produce, rather than the one given.

        No named constructor describes the result in general, so this
        lands in `Matching`. That is the reason the absence language
        carries that constructor at all.

        Args:
            suffix: The suffix the rename appends.

        Returns:
            A name test on the lengthened name.

        Example:
            >>> NameTest(StartsWith("ab")).through_suffix("b_").picks("a", "Int64")
            True
        """
        if not suffix:
            return self
        shifted = RegularLanguage().after_appending(self.rule.language(), suffix)
        return NameTest(Matching.of(shifted))

    def through_prefix(self, prefix: str) -> Selector:
        """Test the name this rename will produce, rather than the one given.

        Args:
            prefix: The prefix the rename prepends.

        Returns:
            A name test on the lengthened name.

        Example:
            >>> NameTest(EndsWith("_x")).through_prefix("s_").picks("a_x", "Int64")
            True
        """
        if not prefix:
            return self
        shifted = RegularLanguage().after_prepending(self.rule.language(), prefix)
        return NameTest(Matching.of(shifted))

    def after_retype(self, dtype: str) -> Selector:
        """Return this selector unchanged, since it reads no data type.

        Args:
            dtype: Ignored.

        Returns:
            This selector.

        Example:
            >>> NameTest(Always()).after_retype("Int64")
            NameTest(rule=Always())
        """
        del dtype
        return self

    def canonical(self) -> Selector:
        """Key this test by the names it accepts.

        Returns:
            The same test, carried as the machine deciding it.

        Example:
            >>> a = NameTest(StartsWith("a")).canonical()
            >>> b = NameTest(Or(StartsWith("a"), StartsWith("ab"))).canonical()
            >>> a == b
            True
        """
        machine = self.rule.language()
        return NameTest(Matching.of(machine))

    def name_rule(self) -> Absence | None:
        """Give the held rule: a name test is already a pure name rule.

        Returns:
            The held rule.

        Example:
            >>> NameTest(Always()).name_rule()
            Always()
        """
        return self.rule

    def language_for(self, dtype: str) -> Fsm:
        """Give the held rule's names, whatever the data type.

        Args:
            dtype: Ignored. A name test does not read the data type.

        Returns:
            The machine deciding the held rule.

        Example:
            >>> NameTest(StartsWith("a")).language_for("String").accepts("ab")
            True
        """
        del dtype
        return self.rule.language()


@dataclass(frozen=True)
class OfType(Selector):
    """A selector that reads only the column's data type.

    This is the one selector that is not a name test, and it is why the
    absence language cannot hold every selector: a data type cannot be
    read off a name. Its two name rules are therefore all-or-nothing.
    Either the recorded type set settles the question for every column
    at once, or it settles it for none of them.

    Example:
        >>> OfType(NUMERIC).picks("anything", "Float64")
        True
    """

    family: Family

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether the column's data type is in the family.

        Args:
            name: Ignored. A type test does not read the name.
            dtype: The column's data type.

        Returns:
            True when the data type is in the family.

        Example:
            >>> OfType(NUMERIC).picks("price", "String")
            False
        """
        del name
        return self.family.holds(dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick every name when every permitted type is in the family.

        Args:
            types: What is known about the columns' data types.

        Returns:
            `Always` when the type set is contained in the family, and
            `Never` otherwise, since no name can then be settled.

        Example:
            >>> OfType(NUMERIC).certainly_picks(TypeSet.anything())
            Never()
        """
        return Always() if types.contained_in(self.family) else Never()

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject every name when no permitted type is in the family.

        Args:
            types: What is known about the columns' data types.

        Returns:
            `Always` when the type set is disjoint from the family, and
            `Never` otherwise.

        Example:
            >>> OfType(NUMERIC).certainly_rejects(TypeSet.anything())
            Never()
        """
        return Always() if types.disjoint_from(self.family) else Never()

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep only the permitted types the family holds.

        This is the rule the specification's closure table left out. It
        is exact, and it is what makes a later type test on the result
        answerable.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The type set narrowed to the family.

        Example:
            >>> sorted(OfType(NUMERIC).surviving_types(TypeSet.anything()).members)
            ['Float32', 'Float64', 'Int32', 'Int64']
        """
        return types.narrowed_to(self.family)

    def reads_only_types(self) -> bool:
        """Answer True, since this selector never reads a name.

        Returns:
            True, always.

        Example:
            >>> OfType(NUMERIC).reads_only_types()
            True
        """
        return True

    def through_suffix(self, suffix: str) -> Selector:
        """Return this selector unchanged, since renaming changes no data type.

        Args:
            suffix: Ignored.

        Returns:
            This selector.

        Example:
            >>> OfType(NUMERIC).through_suffix("_p")
            OfType(family=NumericType)
        """
        del suffix
        return self

    def through_prefix(self, prefix: str) -> Selector:
        """Return this selector unchanged, since renaming changes no data type.

        Args:
            prefix: Ignored.

        Returns:
            This selector.

        Example:
            >>> OfType(NUMERIC).through_prefix("s_")
            OfType(family=NumericType)
        """
        del prefix
        return self

    def after_retype(self, dtype: str) -> Selector:
        """Decide the type test outright, since every column now has one type.

        Args:
            dtype: The data type every column now has.

        Returns:
            A name test that picks everything or nothing.

        Example:
            >>> OfType(NUMERIC).after_retype("String")
            NameTest(rule=Never())
        """
        return NameTest(Always() if self.family.holds(dtype) else Never())

    def canonical(self) -> Selector:
        """Return this selector unchanged, since a family is already a key.

        Returns:
            This selector.

        Example:
            >>> OfType(NUMERIC).canonical() == OfType(NUMERIC)
            True
        """
        return self

    def language_for(self, dtype: str) -> Fsm:
        """Pick every name, or none, according to the data type.

        Args:
            dtype: The data type to fix.

        Returns:
            Every name when the type is in the family, and no name
            otherwise.

        Example:
            >>> OfType(NUMERIC).language_for("String").accepts("x")
            False
        """
        language = RegularLanguage()
        return language.anything() if self.family.holds(dtype) else language.nothing()


@dataclass(frozen=True)
class SelNot(Selector):
    """A selector that picks exactly what the inner one does not.

    Example:
        >>> SelNot(OfType(NUMERIC)).picks("x", "String")
        True
    """

    inner: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer the negation of the inner selector.

        Args:
            name: The column's name.
            dtype: The column's data type.

        Returns:
            True when the inner selector does not pick the column.

        Example:
            >>> SelNot(OfType(NUMERIC)).picks("x", "Int64")
            False
        """
        return not self.inner.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick the names the inner selector certainly rejects.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The inner selector's rejection rule.

        Example:
            >>> SelNot(NameTest(StartsWith("a"))).certainly_picks(TypeSet.anything())
            Not(inner=StartsWith(prefix='a'))
        """
        return self.inner.certainly_rejects(types)

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject the names the inner selector certainly picks.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The inner selector's picking rule.

        Example:
            >>> SelNot(NameTest(StartsWith("a"))).certainly_rejects(TypeSet.anything())
            StartsWith(prefix='a')
        """
        return self.inner.certainly_picks(types)

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the permitted types the inner selector drops.

        Complementing is exact only when the inner selector reads no
        name. Otherwise a column of any type can survive, and the whole
        type set is kept.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The complement within `types` when the inner selector reads
            only types, and `types` unchanged otherwise.

        Example:
            >>> sorted(SelNot(OfType(NUMERIC)).surviving_types(TypeSet.anything()).members)
            ['Boolean', 'Date', 'Datetime', 'Duration', 'String', 'Time']
        """
        if not self.inner.reads_only_types():
            return types
        return TypeSet(types.members - self.inner.surviving_types(types).members)

    def reads_only_types(self) -> bool:
        """Answer whether the inner selector reads only types.

        Returns:
            True when the inner selector reads no name.

        Example:
            >>> SelNot(OfType(NUMERIC)).reads_only_types()
            True
        """
        return self.inner.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Translate each side to run before the suffix is appended.

        Args:
            suffix: The suffix the rename appends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelNot(OfType(NUMERIC))
            >>> isinstance(s.through_suffix("_p"), SelNot)
            True
        """
        return SelNot(self.inner.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Translate each side to run before the prefix is prepended.

        Args:
            prefix: The prefix the rename prepends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelNot(OfType(NUMERIC))
            >>> isinstance(s.through_prefix("s_"), SelNot)
            True
        """
        return SelNot(self.inner.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Resolve each side's type tests against the one data type.

        Args:
            dtype: The data type every column now has.

        Returns:
            This combinator over the resolved sides.

        Example:
            >>> s = SelNot(OfType(NUMERIC))
            >>> isinstance(s.after_retype("Int64"), SelNot)
            True
        """
        return SelNot(self.inner.after_retype(dtype))

    def canonical(self) -> Selector:
        """Key the inner selector, collapsing a pure name negation.

        Returns:
            One name test keyed by the negation's own language when
            the whole negation reads only names, and the negation of
            the inner selector's canonical form otherwise.

        Example:
            >>> SelNot(OfType(NUMERIC)).canonical()
            SelNot(inner=OfType(family=NumericType))
        """
        rule = self.name_rule()
        if rule is not None:
            return NameTest(Matching.of(rule.language()))
        return SelNot(self.inner.canonical())

    def name_rule(self) -> Absence | None:
        """Negate the inner selector's pure name rule, where it has one.

        Returns:
            The negated rule, or None where the inner selector reads
            a data type.

        Example:
            >>> SelNot(NameTest(StartsWith("a"))).name_rule()
            Not(inner=StartsWith(prefix='a'))
        """
        inner = self.inner.name_rule()
        if inner is None:
            return None
        return Not(inner)

    def language_for(self, dtype: str) -> Fsm:
        """Give the complement of the inner selector's names.

        Args:
            dtype: The data type to fix.

        Returns:
            The complement at that data type.

        Example:
            >>> SelNot(OfType(NUMERIC)).language_for("String").accepts("x")
            True
        """
        return self.inner.language_for(dtype).everythingbut()


@dataclass(frozen=True)
class SelAnd(Selector):
    """A selector that picks what both inner selectors pick.

    Example:
        >>> SelAnd(OfType(NUMERIC), NameTest(StartsWith("p"))).picks("price", "Int64")
        True
    """

    left: Selector
    right: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether both inner selectors pick the column.

        Args:
            name: The column's name.
            dtype: The column's data type.

        Returns:
            True when both sides pick it.

        Example:
            >>> SelAnd(OfType(NUMERIC), OfType(TEMPORAL)).picks("x", "Int64")
            False
        """
        return self.left.picks(name, dtype) and self.right.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick a name only when both sides certainly pick it.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The conjunction of the two picking rules.

        Example:
            >>> s = SelAnd(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_picks(TypeSet.anything()).holds_of("az")
            True
        """
        return And(self.left.certainly_picks(types), self.right.certainly_picks(types))

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject a name when either side certainly rejects it.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The disjunction of the two rejection rules.

        Example:
            >>> s = SelAnd(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_rejects(TypeSet.anything()).holds_of("bz")
            True
        """
        return Or(self.left.certainly_rejects(types), self.right.certainly_rejects(types))

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the types that survive both sides.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The intersection of the two surviving type sets.

        Example:
            >>> s = SelAnd(OfType(NUMERIC), OfType(TEMPORAL))
            >>> sorted(s.surviving_types(TypeSet.anything()).members)
            []
        """
        left = self.left.surviving_types(types).members
        return TypeSet(left & self.right.surviving_types(types).members)

    def reads_only_types(self) -> bool:
        """Answer whether both sides read only types.

        Returns:
            True when neither side reads a name.

        Example:
            >>> SelAnd(OfType(NUMERIC), OfType(TEMPORAL)).reads_only_types()
            True
        """
        return self.left.reads_only_types() and self.right.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Translate each side to run before the suffix is appended.

        Args:
            suffix: The suffix the rename appends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelAnd(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.through_suffix("_p"), SelAnd)
            True
        """
        return SelAnd(self.left.through_suffix(suffix), self.right.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Translate each side to run before the prefix is prepended.

        Args:
            prefix: The prefix the rename prepends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelAnd(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.through_prefix("s_"), SelAnd)
            True
        """
        return SelAnd(self.left.through_prefix(prefix), self.right.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Resolve each side's type tests against the one data type.

        Args:
            dtype: The data type every column now has.

        Returns:
            This combinator over the resolved sides.

        Example:
            >>> s = SelAnd(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.after_retype("Int64"), SelAnd)
            True
        """
        return SelAnd(self.left.after_retype(dtype), self.right.after_retype(dtype))

    def canonical(self) -> Selector:
        """Key both sides, collapsing a contained side or a pure name pair.

        Returns:
            One name test keyed by the conjunction's own language
            when both sides read only names, and the conjunction of
            the two canonical forms otherwise.

        Example:
            >>> s = SelAnd(OfType(TEMPORAL), OfType(NUMERIC)).canonical()
            >>> isinstance(s, SelAnd)
            True
        """
        rule = self.name_rule()
        if rule is not None:
            return NameTest(Matching.of(rule.language()))
        return Conjunction()(self.left.canonical(), self.right.canonical())

    def name_rule(self) -> Absence | None:
        """Conjoin the two sides' pure name rules, where both have one.

        Returns:
            The conjoined rule, or None where either side reads a
            data type.

        Example:
            >>> SelAnd(NameTest(Always()), OfType(NUMERIC)).name_rule()
        """
        left = self.left.name_rule()
        right = self.right.name_rule()
        if left is None or right is None:
            return None
        return And(left, right)

    def language_for(self, dtype: str) -> Fsm:
        """Give the names both sides pick at this data type.

        Args:
            dtype: The data type to fix.

        Returns:
            The intersection at that data type.

        Example:
            >>> s = SelAnd(OfType(NUMERIC), SelNot(OfType(NUMERIC)))
            >>> s.language_for("Int64").empty()
            True
        """
        left = self.left.language_for(dtype)
        return left.intersection(self.right.language_for(dtype))


@dataclass(frozen=True)
class SelOr(Selector):
    """A selector that picks what either inner selector picks.

    Example:
        >>> SelOr(OfType(NUMERIC), OfType(TEMPORAL)).picks("x", "Date")
        True
    """

    left: Selector
    right: Selector

    def picks(self, name: str, dtype: str) -> bool:
        """Answer whether either inner selector picks the column.

        Args:
            name: The column's name.
            dtype: The column's data type.

        Returns:
            True when at least one side picks it.

        Example:
            >>> SelOr(OfType(NUMERIC), OfType(TEMPORAL)).picks("x", "String")
            False
        """
        return self.left.picks(name, dtype) or self.right.picks(name, dtype)

    def certainly_picks(self, types: TypeSet) -> Absence:
        """Pick a name when either side certainly picks it.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The disjunction of the two picking rules.

        Example:
            >>> s = SelOr(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_picks(TypeSet.anything()).holds_of("bz")
            True
        """
        return Or(self.left.certainly_picks(types), self.right.certainly_picks(types))

    def certainly_rejects(self, types: TypeSet) -> Absence:
        """Reject a name only when both sides certainly reject it.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The conjunction of the two rejection rules.

        Example:
            >>> s = SelOr(NameTest(StartsWith("a")), NameTest(EndsWith("z")))
            >>> s.certainly_rejects(TypeSet.anything()).holds_of("bz")
            False
        """
        return And(self.left.certainly_rejects(types), self.right.certainly_rejects(types))

    def surviving_types(self, types: TypeSet) -> TypeSet:
        """Keep the types that survive either side.

        Args:
            types: What is known about the columns' data types.

        Returns:
            The union of the two surviving type sets.

        Example:
            >>> s = SelOr(OfType(NUMERIC), OfType(TEMPORAL))
            >>> len(sorted(s.surviving_types(TypeSet.anything()).members))
            8
        """
        left = self.left.surviving_types(types).members
        return TypeSet(left | self.right.surviving_types(types).members)

    def reads_only_types(self) -> bool:
        """Answer whether both sides read only types.

        Returns:
            True when neither side reads a name.

        Example:
            >>> SelOr(OfType(NUMERIC), NameTest(Always())).reads_only_types()
            False
        """
        return self.left.reads_only_types() and self.right.reads_only_types()

    def through_suffix(self, suffix: str) -> Selector:
        """Translate each side to run before the suffix is appended.

        Args:
            suffix: The suffix the rename appends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelOr(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.through_suffix("_p"), SelOr)
            True
        """
        return SelOr(self.left.through_suffix(suffix), self.right.through_suffix(suffix))

    def through_prefix(self, prefix: str) -> Selector:
        """Translate each side to run before the prefix is prepended.

        Args:
            prefix: The prefix the rename prepends.

        Returns:
            This combinator over the translated sides.

        Example:
            >>> s = SelOr(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.through_prefix("s_"), SelOr)
            True
        """
        return SelOr(self.left.through_prefix(prefix), self.right.through_prefix(prefix))

    def after_retype(self, dtype: str) -> Selector:
        """Resolve each side's type tests against the one data type.

        Args:
            dtype: The data type every column now has.

        Returns:
            This combinator over the resolved sides.

        Example:
            >>> s = SelOr(OfType(NUMERIC), OfType(NUMERIC))
            >>> isinstance(s.after_retype("Int64"), SelOr)
            True
        """
        return SelOr(self.left.after_retype(dtype), self.right.after_retype(dtype))

    def canonical(self) -> Selector:
        """Key both sides, collapsing a pure name pair.

        Returns:
            One name test keyed by the disjunction's own language
            when both sides read only names, and the disjunction of
            the two canonical forms, in a fixed order, otherwise.

        Example:
            >>> s = SelOr(OfType(TEMPORAL), OfType(NUMERIC)).canonical()
            >>> isinstance(s, SelOr)
            True
        """
        rule = self.name_rule()
        if rule is not None:
            return NameTest(Matching.of(rule.language()))
        first, second = sorted((self.left.canonical(), self.right.canonical()), key=StableText())
        return SelOr(first, second)

    def name_rule(self) -> Absence | None:
        """Disjoin the two sides' pure name rules, where both have one.

        Returns:
            The disjoined rule, or None where either side reads a
            data type.

        Example:
            >>> SelOr(NameTest(Always()), OfType(NUMERIC)).name_rule()
        """
        left = self.left.name_rule()
        right = self.right.name_rule()
        if left is None or right is None:
            return None
        return Or(left, right)

    def language_for(self, dtype: str) -> Fsm:
        """Give the names either side picks at this data type.

        Args:
            dtype: The data type to fix.

        Returns:
            The union at that data type.

        Example:
            >>> s = SelOr(OfType(NUMERIC), OfType(TEMPORAL))
            >>> s.language_for("Date").accepts("x")
            True
        """
        left = self.left.language_for(dtype)
        return left.union(self.right.language_for(dtype))


class Effect:
    """What an operation does to a remainder.

    An effect is two things that must agree. `apply` transforms one
    real column set. `transform` transforms the record that describes a
    whole family of them. The closure property is that the second
    always describes the result of the first.

    Example:
        >>> Emptied().apply(ColumnSet.of(a="Int64")).entries
        ()
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Transform one concrete column set.

        Args:
            columns: The column set the operation receives.

        Returns:
            The column set the operation produces.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Unchanged().apply(ColumnSet.of(a="Int64")).names()
            ('a',)
        """
        raise NotImplementedError

    def transform(self, record: Record) -> Record:
        """Transform the record that describes a remainder.

        Args:
            record: What is known about the remainder going in.

        Returns:
            What is known about the remainder coming out.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Emptied().transform(Record.unconstrained()).absence
            Always()
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Unchanged(Effect):
    """The remainder passes through untouched.

    Filtering rows, sorting and reading a declared column all do this.

    Example:
        >>> Unchanged().apply(ColumnSet.of(a="Int64")).entries
        (('a', 'Int64'),)
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Return the column set unchanged.

        Args:
            columns: The column set the operation receives.

        Returns:
            The same column set.

        Example:
            >>> Unchanged().apply(ColumnSet.of(b="String")).names()
            ('b',)
        """
        return columns

    def transform(self, record: Record) -> Record:
        """Return the record unchanged.

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
    """The remainder loses every column.

    Selecting a fixed list of names does this, and so does reshaping
    that turns columns into rows.

    Example:
        >>> Emptied().transform(Record.unconstrained()).types.members
        frozenset()
    """

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Return the empty column set.

        Args:
            columns: Ignored.

        Returns:
            The column set with no columns.

        Example:
            >>> Emptied().apply(ColumnSet.of(a="Int64")).names()
            ()
        """
        del columns
        return ColumnSet(())

    def transform(self, record: Record) -> Record:
        """Prove every name absent and permit no data type.

        Args:
            record: Ignored.

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
    """Every column keeps its name and takes one new data type.

    Counting the non-null entries of each column does this.

    Example:
        >>> Retyped("UInt32").apply(ColumnSet.of(a="String")).entries
        (('a', 'UInt32'),)
    """

    dtype: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Replace every data type, keeping every name.

        Args:
            columns: The column set the operation receives.

        Returns:
            The same names, all of the new data type. Two columns that
            shared a name become one column.

        Example:
            >>> Retyped("UInt32").apply(ColumnSet.of(a="Int64", b="String")).names()
            ('a', 'b')
        """
        return ColumnSet.holding((name, self.dtype) for name, _ in columns.entries)

    def transform(self, record: Record) -> Record:
        """Keep the absence rule and replace the type set.

        Keeping the rule is the point. An operation that renames
        nothing loses nothing about which names exist.

        Args:
            record: What is known about the remainder.

        Returns:
            The same absence rule, with one permitted data type.

        Example:
            >>> sorted(Retyped("UInt32").transform(Record.unconstrained()).types.members)
            ['UInt32']
        """
        return Record(record.absence, TypeSet(frozenset({self.dtype})))


@dataclass(frozen=True)
class SuffixedBy(Effect):
    """Every column name gains a fixed suffix.

    This is one of the two effects that invent names, which is why the
    absence language carries a stripping constructor for it.

    Example:
        >>> SuffixedBy("_prev").apply(ColumnSet.of(price="Int64")).names()
        ('price_prev',)
    """

    suffix: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Append the suffix to every name.

        Args:
            columns: The column set the operation receives.

        Returns:
            The same data types, under the lengthened names.

        Example:
            >>> SuffixedBy("_x").apply(ColumnSet.of(a="Int64")).entries
            (('a_x', 'Int64'),)
        """
        return ColumnSet(tuple((name + self.suffix, d) for name, d in columns.entries))

    def transform(self, record: Record) -> Record:
        """Prove a name absent when it cannot have come from the operand.

        A name is present only when it ends in the suffix and the name
        under that suffix was present. Negating gives the rule.

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
    """Every column name gains a fixed prefix.

    This is `SuffixedBy` with the ends swapped. Nothing in the argument
    depends on which end of the name changes.

    Example:
        >>> PrefixedBy("sec_").apply(ColumnSet.of(country="String")).names()
        ('sec_country',)
    """

    prefix: str

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Prepend the prefix to every name.

        Args:
            columns: The column set the operation receives.

        Returns:
            The same data types, under the lengthened names.

        Example:
            >>> PrefixedBy("x_").apply(ColumnSet.of(a="Int64")).entries
            (('x_a', 'Int64'),)
        """
        return ColumnSet(tuple((self.prefix + name, d) for name, d in columns.entries))

    def transform(self, record: Record) -> Record:
        """Prove a name absent when it cannot have come from the operand.

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
    """Only the columns a selector picks survive.

    This is one effect with three spellings. Removing a list of names,
    keeping what a selector picks, and dropping what it picks differ
    only in the selector and whether it is negated.

    Example:
        >>> kept = Where(OfType(NUMERIC)).apply(ColumnSet.of(a="Int64", b="String"))
        >>> kept.names()
        ('a',)
    """

    selector: Selector

    def apply(self, columns: ColumnSet) -> ColumnSet:
        """Keep the columns the selector picks.

        Args:
            columns: The column set the operation receives.

        Returns:
            The column set holding only the picked columns.

        Example:
            >>> Where(NameTest(StartsWith("a"))).apply(ColumnSet.of(ab="Int64")).names()
            ('ab',)
        """
        kept = tuple(e for e in columns.entries if self.selector.picks(e[0], e[1]))
        return ColumnSet(kept)

    def transform(self, record: Record) -> Record:
        """Add the certainly-rejected names, and narrow the type set.

        Narrowing the type set is what lets a later type test on the
        result be answered. Adding the rejected names is what lets a
        single drop satisfy a namespace requirement.

        Args:
            record: What is known about the remainder.

        Returns:
            The record of the filtered remainder.

        Example:
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
    """Two remainders held together in one.

    This is the only construct that combines two remainders. It is not
    an `Effect`, because it reads two operands rather than one.

    Example:
        >>> Both().apply(ColumnSet.of(a="Int64"), ColumnSet.of(b="String")).names()
        ('a', 'b')
    """

    def apply(self, left: ColumnSet, right: ColumnSet) -> ColumnSet:
        """Hold every column of both operands.

        Args:
            left: One operand's columns.
            right: The other operand's columns.

        Returns:
            The column set holding both, sorted by name.

        Example:
            >>> Both().apply(ColumnSet(()), ColumnSet.of(b="String")).names()
            ('b',)
        """
        return ColumnSet.holding(left.entries + right.entries)

    def transform(self, left: Record, right: Record) -> Record:
        """Prove a name absent only when both operands prove it absent.

        Args:
            left: What is known about one operand.
            right: What is known about the other.

        Returns:
            The record of the combined remainder.

        Example:
            >>> a = Record(OneOf(frozenset({"x"})), TypeSet(frozenset({"Int64"})))
            >>> b = Record(OneOf(frozenset({"x", "y"})), TypeSet(frozenset({"String"})))
            >>> combined = Both().transform(a, b)
            >>> combined.absence.holds_of("x"), combined.absence.holds_of("y")
            (True, False)
        """
        return Record(
            And(left.absence, right.absence),
            left.types.combined_with(right.types),
        )


class AbsenceEntailment:
    """Decide whether one name rule holds nowhere the other does not.

    Two rules can describe the same set of names by different routes,
    and a canonical form that compared them by shape alone would call
    two equal types different. Every constructor denotes a regular set
    of names, so containment between two of them is decidable in
    general. This class decides it for the named constructors, which is
    every rule a user can write, by rules that are cheap and exact.

    The answer is **sound**: a True is a real entailment. A False means
    "not established by these rules", which for a `Matching` holding an
    opaque test is the only honest answer.

    Example:
        >>> AbsenceEntailment()(StartsWith("ab"), StartsWith("a"))
        True
        >>> AbsenceEntailment()(StartsWith("a"), StartsWith("ab"))
        False
    """

    def __call__(self, left: Absence, right: Absence) -> bool:
        """Answer whether every name `left` holds of, `right` holds of too.

        Three cases need no rule: a rule is contained in itself, a rule
        holding of no name is contained in every rule, and every rule is
        contained in the rule holding of every name. The other rules are
        tried in order, the exact one by automaton last.

        Args:
            left: The rule that must be the narrower one.
            right: The rule that must be the wider one.

        Returns:
            True when the containment is established.

        Example:
            >>> AbsenceEntailment()(Never(), OneOf(frozenset({"a"})))
            True
            >>> AbsenceEntailment()(Always(), Always())
            True
            >>> AbsenceEntailment()(StartsWith("a"), Always())
            True
            >>> wide = Or(StartsWith("a"), StartsWith("b"))
            >>> AbsenceEntailment()(StartsWith("ab"), wide)
            True
        """
        if left == right or isinstance(left, Never) or isinstance(right, Always):
            return True
        rules = (
            self._vacuous_right,
            self._finite_left,
            self._same_affix,
            self._affix_inside_substring,
            self._left_conjunction,
            self._right_conjunction,
            self._left_disjunction,
            self._right_disjunction,
            self._by_automaton,
        )
        return any(rule(left, right) for rule in rules)

    def _by_automaton(self, left: Absence, right: Absence) -> bool:
        """Decide the containment exactly, by comparing two machines.

        Every constructor denotes a regular set of names, so this
        answers every question the cheap rules leave open. It is tried
        last because building and comparing two machines costs more
        than a string comparison, and because a rule the cheap rules
        settle keeps the shape the user wrote.

        Args:
            left: The rule that must be the narrower one.
            right: The rule that must be the wider one.

        Returns:
            True exactly when every name `left` holds of, `right` holds
            of too.

        Example:
            >>> AbsenceEntailment()._by_automaton(
            ...     Not(Not(StartsWith("a"))), StartsWith("a")
            ... )
            True
        """
        language = RegularLanguage()
        return language.contains(language.of(left), language.of(right))

    def _vacuous_right(self, left: Absence, right: Absence) -> bool:
        """An affix test with an empty affix holds of every name.

        Every name begins with the empty string, ends with it, and
        holds it, so each of the three affix constructors is total when
        its affix is empty.

        Args:
            left: Ignored.
            right: The rule to test for vacuity.

        Returns:
            True when `right` holds of every name.

        Example:
            >>> AbsenceEntailment()._vacuous_right(OneOf(frozenset()), StartsWith(""))
            True
        """
        del left
        if isinstance(right, StartsWith):
            return not right.prefix
        if isinstance(right, EndsWith):
            return not right.suffix
        return isinstance(right, Contains) and not right.part

    def _finite_left(self, left: Absence, right: Absence) -> bool:
        """A finite set is contained in a rule that holds of each member.

        This one check covers a name list against every other
        constructor, because a finite set can simply be tested.

        Args:
            left: The rule, which must be a finite name set.
            right: The rule to test each name against.

        Returns:
            True when every listed name satisfies `right`.

        Example:
            >>> names = OneOf(frozenset({"ab", "ac"}))
            >>> AbsenceEntailment()._finite_left(names, StartsWith("a"))
            True
        """
        if not isinstance(left, OneOf):
            return False
        return all(right.holds_of(name) for name in left.names)

    def _same_affix(self, left: Absence, right: Absence) -> bool:
        """A longer affix test is contained in a shorter one of the same kind.

        Every name beginning with `ab` begins with `a`, and every name
        holding `abc` holds `bc`.

        Args:
            left: The narrower affix test.
            right: The wider one.

        Returns:
            True when both are the same kind and `left` is narrower.

        Example:
            >>> AbsenceEntailment()._same_affix(EndsWith("_ab"), EndsWith("b"))
            True
            >>> AbsenceEntailment()._same_affix(StartsWith("ab"), StartsWith("a"))
            True
            >>> AbsenceEntailment()._same_affix(Contains("abc"), Contains("bc"))
            True
            >>> AbsenceEntailment()._same_affix(StartsWith("ab"), EndsWith("b"))
            False
        """
        if isinstance(left, StartsWith) and isinstance(right, StartsWith):
            return left.prefix.startswith(right.prefix)
        if isinstance(left, EndsWith) and isinstance(right, EndsWith):
            return left.suffix.endswith(right.suffix)
        if isinstance(left, Contains) and isinstance(right, Contains):
            return right.part in left.part
        return False

    def _affix_inside_substring(self, left: Absence, right: Absence) -> bool:
        """A name with a fixed affix holds every substring of that affix.

        Every name beginning with `sec_` holds `ec`, so a prefix test
        is contained in a substring test whenever the substring occurs
        in the prefix. The same holds at the other end. Any other rule
        requires no fixed text, so only the empty substring occurs in
        its affix.

        Args:
            left: The affix test.
            right: The substring test.

        Returns:
            True when `right`'s substring occurs in `left`'s affix.

        Example:
            >>> AbsenceEntailment()._affix_inside_substring(
            ...     StartsWith("sec_"), Contains("ec")
            ... )
            True
            >>> AbsenceEntailment()._affix_inside_substring(
            ...     EndsWith("_prev"), Contains("pre")
            ... )
            True
        """
        if not isinstance(right, Contains):
            return False
        affix = ""
        if isinstance(left, StartsWith):
            affix = left.prefix
        elif isinstance(left, EndsWith):
            affix = left.suffix
        return right.part in affix

    def _left_conjunction(self, left: Absence, right: Absence) -> bool:
        """A conjunction is contained in whatever either half is contained in.

        Args:
            left: The rule, which must be a conjunction.
            right: The wider rule.

        Returns:
            True when either half establishes the containment.

        Example:
            >>> both = And(StartsWith("ab"), EndsWith("z"))
            >>> AbsenceEntailment()._left_conjunction(both, StartsWith("a"))
            True
        """
        if not isinstance(left, And):
            return False
        return self(left.left, right) or self(left.right, right)

    def _right_conjunction(self, left: Absence, right: Absence) -> bool:
        """A rule is contained in a conjunction when it is in both halves.

        Args:
            left: The narrower rule.
            right: The rule, which must be a conjunction.

        Returns:
            True when both halves contain `left`.

        Example:
            >>> wide = And(StartsWith("a"), EndsWith("z"))
            >>> AbsenceEntailment()._right_conjunction(OneOf(frozenset({"az"})), wide)
            True
        """
        if not isinstance(right, And):
            return False
        return self(left, right.left) and self(left, right.right)

    def _left_disjunction(self, left: Absence, right: Absence) -> bool:
        """A disjunction is contained in a rule when both halves are.

        Args:
            left: The rule, which must be a disjunction.
            right: The wider rule.

        Returns:
            True when both halves are contained in `right`.

        Example:
            >>> either = Or(StartsWith("ab"), StartsWith("ac"))
            >>> AbsenceEntailment()._left_disjunction(either, StartsWith("a"))
            True
        """
        if not isinstance(left, Or):
            return False
        return self(left.left, right) and self(left.right, right)

    def _right_disjunction(self, left: Absence, right: Absence) -> bool:
        """A rule is contained in a disjunction when it is in either half.

        Args:
            left: The narrower rule.
            right: The rule, which must be a disjunction.

        Returns:
            True when either half contains `left`.

        Example:
            >>> either = Or(StartsWith("a"), StartsWith("b"))
            >>> AbsenceEntailment()._right_disjunction(StartsWith("bc"), either)
            True
        """
        if not isinstance(right, Or):
            return False
        return self(left, right.left) or self(left, right.right)


class SelectorEntailment:
    """Decide whether one selector picks no more columns than another.

    This lifts `AbsenceEntailment` from names to whole columns, and adds
    the one case names cannot decide: a data-type test is contained in
    another exactly when its family is.

    It is what lets a conjunction of two filters collapse to the
    narrower one. Without it, filtering on `startsWith("ab")` after
    filtering on `startsWith("a")` produces a two-part selector that
    compares unequal to the one-part selector denoting the same set.

    Example:
        >>> narrow = NameTest(StartsWith("ab"))
        >>> SelectorEntailment()(narrow, NameTest(StartsWith("a")))
        True
    """

    def __call__(self, left: Selector, right: Selector) -> bool:
        """Answer whether every column `left` picks, `right` picks too.

        A selector is contained in itself. The cheap rules are tried
        next, then the two negations that collapse to a constant: the
        complement of a selector picking nothing picks everything, and
        the complement of one picking everything picks nothing. Both
        arise from resolving a type test after a retyping, so both are
        worth deciding rather than leaving opaque. The exact decision by
        automaton comes last.

        Args:
            left: The selector that must be the narrower one.
            right: The selector that must be the wider one.

        Returns:
            True when the containment is established.

        Example:
            >>> SelectorEntailment()(OfType(NUMERIC), OfType(NUMERIC))
            True
            >>> total = SelNot(NameTest(Never()))
            >>> SelectorEntailment()(NameTest(Always()), total)
            True
            >>> wide = SelOr(OfType(NUMERIC), OfType(TEMPORAL))
            >>> SelectorEntailment()(OfType(TEMPORAL), wide)
            True
        """
        if left == right:
            return True
        rules = (
            self._both_name_tests,
            self._both_type_tests,
            self._left_conjunction,
            self._right_conjunction,
            self._left_disjunction,
            self._right_disjunction,
        )
        if any(rule(left, right) for rule in rules):
            return True
        if self._negated_right(right) or self._negated_left(left):
            return True
        return self._by_automaton(left, right)

    def _by_automaton(self, left: Selector, right: Selector) -> bool:
        """Decide the containment exactly, one data type at a time.

        A selector reads a name and a data type. There are finitely
        many data types, so the containment holds exactly when, for
        each of them, the names the narrower selector picks are among
        the names the wider one picks. Each of those is a containment
        between two regular sets.

        Args:
            left: The selector that must be the narrower one.
            right: The selector that must be the wider one.

        Returns:
            True exactly when every column `left` picks, `right` picks.

        Example:
            >>> contradiction = SelAnd(OfType(NUMERIC), SelNot(OfType(NUMERIC)))
            >>> SelectorEntailment()._by_automaton(contradiction, OfType(TEMPORAL))
            True
        """
        language = RegularLanguage()
        return all(
            language.contains(left.language_for(d), right.language_for(d))
            for d in sorted(EVERY_DTYPE)
        )

    def _both_name_tests(self, left: Selector, right: Selector) -> bool:
        """Decide two name tests by the containment of their name rules.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True when both read names and the rules are contained.

        Example:
            >>> e = SelectorEntailment()
            >>> e._both_name_tests(NameTest(OneOf(frozenset({"a"}))), NameTest(Always()))
            True
        """
        if not isinstance(left, NameTest) or not isinstance(right, NameTest):
            return False
        return AbsenceEntailment()(left.rule, right.rule)

    def _both_type_tests(self, left: Selector, right: Selector) -> bool:
        """Decide two type tests by the containment of their families.

        Args:
            left: The narrower selector.
            right: The wider one.

        Returns:
            True when both read types and the families are contained.

        Example:
            >>> ints = Family("Integer", frozenset({"Int64", "Int32"}))
            >>> SelectorEntailment()._both_type_tests(OfType(ints), OfType(NUMERIC))
            True
        """
        if not isinstance(left, OfType) or not isinstance(right, OfType):
            return False
        return left.family.members <= right.family.members

    def _left_conjunction(self, left: Selector, right: Selector) -> bool:
        """A conjunction is contained in whatever either side is contained in.

        Args:
            left: The selector, which must be a conjunction.
            right: The wider selector.

        Returns:
            True when either side establishes the containment.

        Example:
            >>> both = SelAnd(NameTest(StartsWith("ab")), OfType(NUMERIC))
            >>> SelectorEntailment()._left_conjunction(both, NameTest(StartsWith("a")))
            True
        """
        if not isinstance(left, SelAnd):
            return False
        return self(left.left, right) or self(left.right, right)

    def _right_conjunction(self, left: Selector, right: Selector) -> bool:
        """A selector is contained in a conjunction when it is in both sides.

        Args:
            left: The narrower selector.
            right: The selector, which must be a conjunction.

        Returns:
            True when both sides contain `left`.

        Example:
            >>> wide = SelAnd(NameTest(StartsWith("a")), OfType(NUMERIC))
            >>> narrow = SelAnd(NameTest(StartsWith("ab")), OfType(NUMERIC))
            >>> SelectorEntailment()._right_conjunction(narrow, wide)
            True
        """
        if not isinstance(right, SelAnd):
            return False
        return self(left, right.left) and self(left, right.right)

    def _negated_right(self, right: Selector) -> bool:
        """Answer whether the wider selector is the complement of an empty one.

        Args:
            right: The selector to test.

        Returns:
            True when it is a negation of a selector picking nothing,
            and so picks everything.

        Example:
            >>> SelectorEntailment()._negated_right(SelNot(NameTest(Never())))
            True
        """
        if not isinstance(right, SelNot):
            return False
        return self(right.inner, NameTest(Never()))

    def _negated_left(self, left: Selector) -> bool:
        """Answer whether the narrower selector is the complement of a total one.

        Args:
            left: The selector to test.

        Returns:
            True when it is a negation of a selector picking
            everything, and so picks nothing.

        Example:
            >>> SelectorEntailment()._negated_left(SelNot(NameTest(Always())))
            True
        """
        if not isinstance(left, SelNot):
            return False
        return self(NameTest(Always()), left.inner)

    def _left_disjunction(self, left: Selector, right: Selector) -> bool:
        """A disjunction is contained in a selector when both sides are.

        Args:
            left: The selector, which must be a disjunction.
            right: The wider selector.

        Returns:
            True when both sides are contained in `right`.

        Example:
            >>> either = SelOr(NameTest(StartsWith("ab")), NameTest(StartsWith("ac")))
            >>> e = SelectorEntailment()
            >>> e._left_disjunction(either, NameTest(StartsWith("a")))
            True
        """
        if not isinstance(left, SelOr):
            return False
        return self(left.left, right) and self(left.right, right)

    def _right_disjunction(self, left: Selector, right: Selector) -> bool:
        """A selector is contained in a disjunction when it is in either side.

        Args:
            left: The narrower selector.
            right: The selector, which must be a disjunction.

        Returns:
            True when either side contains `left`.

        Example:
            >>> either = SelOr(OfType(NUMERIC), OfType(TEMPORAL))
            >>> SelectorEntailment()._right_disjunction(OfType(NUMERIC), either)
            True
        """
        if not isinstance(right, SelOr):
            return False
        return self(left, right.left) or self(left, right.right)


class Conjunction:
    """Build the selector that picks what two selectors both pick.

    The point is what it does **not** build. When one selector is
    contained in the other, their conjunction is the narrower one, so
    the wider one is dropped rather than recorded. Two filters applied
    in turn then produce the same value as the single filter denoting
    the same set, and the canonical form stops distinguishing them.

    Example:
        >>> narrow = NameTest(StartsWith("ab"))
        >>> Conjunction()(NameTest(StartsWith("a")), narrow)
        NameTest(rule=StartsWith(prefix='ab'))
    """

    def __call__(self, left: Selector, right: Selector) -> Selector:
        """Give the selector picking what both pick, simplified.

        Conjunction is commutative, so a node that kept the order it was
        built in would distinguish two equal selectors. The two sides of
        a built conjunction are therefore sorted by their rendering.

        Args:
            left: One selector.
            right: The other.

        Returns:
            The narrower one when containment is established, and their
            conjunction otherwise.

        Example:
            >>> Conjunction()(OfType(TEMPORAL), OfType(NUMERIC))
            SelAnd(left=OfType(family=NumericType), right=OfType(family=TemporalType))
            >>> a, b = OfType(NUMERIC), OfType(TEMPORAL)
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


class RegularLanguage:
    """The set of names a rule holds of, as a finite automaton.

    Every constructor of the absence language denotes a regular set of
    names. That is what makes containment between two rules decidable,
    and this class is where the decision happens. It builds the machine
    for a rule, and gives two operations on machines:

    - `contains`, which decides entailment exactly, and
    - `signature`, a text form that is the same for two machines
      exactly when they accept the same names.

    The signature is what lets a canonical form compare two derived
    filters by value. It is built from the minimal machine, with its
    states numbered by a breadth-first walk and its alphabet quotiented
    by transition behaviour, so neither the order a rule was built in
    nor the letters it happens to mention can change it.

    Example:
        >>> language = RegularLanguage()
        >>> language.contains(language.of(StartsWith("ab")), language.of(StartsWith("a")))
        True
    """

    def of(self, rule: Absence) -> Fsm:
        """Build the machine accepting exactly the names a rule holds of.

        Args:
            rule: The absence rule to translate.

        Returns:
            A machine accepting that rule's names.

        Example:
            >>> RegularLanguage().of(OneOf(frozenset({"a"}))).accepts("a")
            True
        """
        return rule.language()

    def anything(self) -> Fsm:
        """Give the machine accepting every name.

        Returns:
            A machine accepting every string.

        Example:
            >>> RegularLanguage().anything().accepts("whatever")
            True
        """
        return parse(".*").to_fsm()

    def nothing(self) -> Fsm:
        """Give the machine accepting no name.

        Returns:
            A machine accepting nothing.

        Example:
            >>> RegularLanguage().nothing().accepts("")
            False
        """
        return parse("[]").to_fsm()

    def literal(self, text: str) -> Fsm:
        """Give the machine accepting exactly one string.

        Args:
            text: The string to accept.

        Returns:
            A machine accepting that string and nothing else.

        Example:
            >>> RegularLanguage().literal("a.b").accepts("a.b")
            True
            >>> RegularLanguage().literal("a.b").accepts("axb")
            False
        """
        return parse("".join(str(Charclass(((c, c),))) for c in text)).to_fsm()

    def contains(self, narrow: Fsm, wide: Fsm) -> bool:
        """Decide whether every name one machine accepts, the other accepts.

        This is exact. A False means the containment does not hold, not
        that it could not be established.

        Args:
            narrow: The machine that must accept no more.
            wide: The machine that must accept at least as much.

        Returns:
            True when the containment holds.

        Example:
            >>> lang = RegularLanguage()
            >>> lang.contains(lang.literal("ab"), lang.of(StartsWith("a")))
            True
        """
        return bool(narrow.difference(wide).empty())

    def after_appending(self, machine: Fsm, suffix: str) -> Fsm:
        """Give the names that land in `machine` once a suffix is appended.

        Args:
            machine: The machine to test the lengthened name against.
            suffix: The text the rename appends.

        Returns:
            A machine accepting each name whose lengthened form is
            accepted.

        Example:
            >>> lang = RegularLanguage()
            >>> shifted = lang.after_appending(lang.of(StartsWith("ab")), "b_")
            >>> shifted.accepts("a"), shifted.accepts("ax")
            (True, False)
        """
        return self._derived(machine.reversed(), suffix[::-1]).reversed()

    def after_prepending(self, machine: Fsm, prefix: str) -> Fsm:
        """Give the names that land in `machine` once a prefix is prepended.

        Args:
            machine: The machine to test the lengthened name against.
            prefix: The text the rename prepends.

        Returns:
            A machine accepting each name whose lengthened form is
            accepted.

        Example:
            >>> lang = RegularLanguage()
            >>> shifted = lang.after_prepending(lang.of(StartsWith("a")), "a")
            >>> shifted.accepts("anything")
            True
        """
        return self._derived(machine, prefix)

    def _derived(self, machine: Fsm, text: str) -> Fsm:
        """Give the names that follow `text` in the machine's language.

        Args:
            machine: The machine to walk into.
            text: The text to consume first.

        Returns:
            The machine reached after consuming that text.

        Example:
            >>> lang = RegularLanguage()
            >>> lang._derived(lang.literal("ab"), "a").accepts("b")
            True
        """
        for character in text:
            machine = machine.derive(character)
        return machine

    def signature(self, machine: Fsm) -> str:
        """Give text that two machines share exactly when they agree.

        Args:
            machine: The machine to render.

        Returns:
            A canonical rendering of its minimal form.

        Example:
            >>> lang = RegularLanguage()
            >>> a = lang.signature(parse("a.*").to_fsm())
            >>> a == lang.signature(parse("a.*|ab.*").to_fsm())
            True
        """
        return _Signature(machine.reduce()).text()


class _Signature:
    """One machine's canonical rendering, built once.

    Two machines accepting the same names give the same text. The
    states are numbered by a breadth-first walk from the initial state,
    and the alphabet is quotiented by transition behaviour, so a
    machine that happens to name a letter it treats like every other
    letter renders the same as one that does not.

    Example:
        >>> _Signature(parse("[]").to_fsm().reduce()).text()
        '0[.>0]'
    """

    def __init__(self, machine: Fsm) -> None:
        """Number the states and group the alphabet.

        Args:
            machine: A minimal machine to render.
        """
        self.machine = machine
        self.order = self._numbered()
        self.states = sorted(self.order, key=lambda s: self.order[s])
        self.behaviour = self._grouped()

    def _numbered(self) -> dict[object, int]:
        """Number every reachable state by a breadth-first walk.

        Each state's transitions are followed in a fixed symbol order,
        so the numbering does not depend on how the machine stores its
        map.

        Returns:
            Each state, mapped to its position in the walk.

        Example:
            >>> len(_Signature(parse("a").to_fsm().reduce())._numbered()) >= 2
            True
        """
        order, queue = {self.machine.initial: 0}, [self.machine.initial]
        while queue:
            row = self.machine.map[queue.pop(0)]
            for symbol in sorted(row, key=str):
                target = row[symbol]
                if target not in order:
                    order[target] = len(order)
                    queue.append(target)
        return order

    def _grouped(self) -> dict[tuple[int, ...], str]:
        """Group alphabet symbols that every state treats alike.

        Two machines can accept the same names over different
        alphabets. Merging symbols with identical behaviour removes
        that difference. A symbol's behaviour is its numbered target
        from every state, in order, and each group renders as the
        reduced union of its symbols.

        Returns:
            Each behaviour, mapped to the rendering of its symbols.

        Example:
            >>> grouped = _Signature(parse(".*").to_fsm().reduce())._grouped()
            >>> list(grouped.values())
            ['.']
        """
        rows: dict[tuple[int, ...], list[Charclass]] = {}
        for symbol in self.machine.alphabet:
            targets = [self.machine.map[state][symbol] for state in self.states]
            key = tuple(self.order[target] for target in targets)
            rows.setdefault(key, []).append(symbol)
        grouped: dict[tuple[int, ...], str] = {}
        for key, symbols in rows.items():
            union = reduce(lambda a, b: a.union(b), symbols)
            grouped[key] = str(union.reduce())
        return grouped

    def text(self) -> str:
        """Give the canonical rendering.

        Each state renders as its number, an `F` when it accepts, and
        its transitions, one per behaviour group.

        Returns:
            One line per state, in walk order.

        Example:
            >>> _Signature(parse(".*").to_fsm().reduce()).text()
            '0F[.>0]'
        """
        keys = sorted(self.behaviour, key=lambda k: self.behaviour[k])
        lines: list[str] = []
        for state in self.states:
            number = self.order[state]
            moves = ";".join(f"{self.behaviour[k]}>{k[number]}" for k in keys)
            accepts = "F" if state in self.machine.finals else ""
            lines.append(f"{number}{accepts}[{moves}]")
        return "|".join(lines)
