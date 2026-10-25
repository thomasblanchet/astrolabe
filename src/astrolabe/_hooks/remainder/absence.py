"""Absence rules: tests on a column name that prove the name is not in a remainder.

A remainder is the columns of a frame that its type does not name,
for example the columns that a caller brings beyond the ones a
function declares. The checker cannot list them, but it can prove
that some names are not among them. After `drop("tmp")`, the name
`tmp` is certainly absent. An absence rule is a test on a name that
holds of each name proved absent. Such a proof lets a return type
state that the remainder holds no `tmp`, and it lets a call satisfy
a declaration such as `other_columns(without=["tmp"])`.

Each class in this module is one way to build a rule. Every operation
that changes a remainder has a matching way to change its rule, so a
chain of operations keeps what it proved. Each rule also gives the
exact set of names it holds of, as a `NameSet`. That set is what lets
the engine decide whether one rule proves at least what another
demands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astrolabe._hooks.remainder.name_language import NameSet


class Absence:
    """A test on a name that holds if the name is certainly not in the remainder.

    A rule never says that a name is present. If a rule does not hold
    of a name, the column can be in the remainder or not. So a rule
    that holds of fewer names knows less, and it is always safe.

    Each subclass is one constructor, and a rule is a tree of them.
    `holds_of` tests one name, and `language` gives the whole set of
    names, which code needs to compare two rules.

    Example:
        >>> rule = Or(OneOf(frozenset({"a"})), StartsWith("tmp_"))
        >>> rule.holds_of("a"), rule.holds_of("tmp_x"), rule.holds_of("b")
        (True, True, False)
    """

    def holds_of(self, name: str) -> bool:
        """Answer whether this rule proves `name` absent.

        Args:
            name: The column name to test.

        Returns:
            True if the name is certainly absent.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Never().holds_of("anything")
            False
        """
        raise NotImplementedError

    def language(self) -> NameSet:
        """Give the set of exactly the names that this rule holds of.

        Every constructor gives a regular set, that is, a set that a
        finite automaton can decide. So two rules can always be
        compared exactly, however they are built.

        Returns:
            The set of names this rule holds of.

        Raises:
            NotImplementedError: Always. Every subclass overrides this.

        Example:
            >>> Always().language().accepts("anything")
            True
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Never(Absence):
    """No name is proved absent. This is the rule of a remainder that promises nothing.

    Example:
        >>> Never().holds_of("venue")
        False
    """

    def holds_of(self, name: str) -> bool:
        """Answer False, for every name.

        Args:
            name: Not read.

        Returns:
            False, always.

        Example:
            >>> Never().holds_of("")
            False
        """
        return False

    def language(self) -> NameSet:
        """Give the empty set of names.

        Returns:
            The set that holds no name.

        Example:
            >>> Never().language().accepts("")
            False
        """
        return NameSet.nothing()


@dataclass(frozen=True)
class Always(Absence):
    """Every name is proved absent. This is the rule of a remainder with no columns.

    Example:
        >>> Always().holds_of("venue")
        True
    """

    def holds_of(self, name: str) -> bool:
        """Answer True, for every name.

        Args:
            name: Not read.

        Returns:
            True, always.

        Example:
            >>> Always().holds_of("")
            True
        """
        return True

    def language(self) -> NameSet:
        """Give the set of every name.

        Returns:
            The set that holds every name.

        Example:
            >>> Always().language().accepts("x")
            True
        """
        return NameSet.anything()


@dataclass(frozen=True)
class OneOf(Absence):
    """The names in a finite list are absent.

    This is the rule after an operation drops a list of names, or
    after a declaration such as `other_columns(without=["a", "b"])`.

    Example:
        >>> OneOf(frozenset({"a", "b"})).holds_of("a")
        True
    """

    names: frozenset[str]

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` is in the list.

        Args:
            name: The column name to test.

        Returns:
            True if the name is in the list.

        Example:
            >>> OneOf(frozenset({"a"})).holds_of("b")
            False
        """
        return name in self.names

    def language(self) -> NameSet:
        """Give the set of exactly the listed names.

        Returns:
            The union of one single-name set for each listed name.

        Example:
            >>> OneOf(frozenset({"a", "b"})).language().accepts("b")
            True
        """
        return NameSet.of_names(self.names)


@dataclass(frozen=True)
class StartsWith(Absence):
    """Every name that begins with a fixed text is absent.

    Example:
        >>> StartsWith("sec_").holds_of("sec_country")
        True
    """

    prefix: str

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` begins with the prefix.

        Args:
            name: The column name to test.

        Returns:
            True if the name begins with the prefix.

        Example:
            >>> StartsWith("sec_").holds_of("venue")
            False
        """
        return name.startswith(self.prefix)

    def language(self) -> NameSet:
        """Give the set of every name that begins with the prefix.

        Returns:
            The prefix, followed by any text.

        Example:
            >>> StartsWith("ab").language().accepts("abc")
            True
        """
        return NameSet.literal(self.prefix).concatenate(NameSet.anything())


@dataclass(frozen=True)
class EndsWith(Absence):
    """Every name that ends with a fixed text is absent.

    Example:
        >>> EndsWith("_prev").holds_of("price_prev")
        True
    """

    suffix: str

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` ends with the suffix.

        Args:
            name: The column name to test.

        Returns:
            True if the name ends with the suffix.

        Example:
            >>> EndsWith("_prev").holds_of("price")
            False
        """
        return name.endswith(self.suffix)

    def language(self) -> NameSet:
        """Give the set of every name that ends with the suffix.

        Returns:
            Any text, followed by the suffix.

        Example:
            >>> EndsWith("_p").language().accepts("x_p")
            True
        """
        return NameSet.anything().concatenate(NameSet.literal(self.suffix))


@dataclass(frozen=True)
class Contains(Absence):
    """Every name that holds a fixed text at any position is absent.

    Example:
        >>> Contains("tmp").holds_of("x_tmp_y")
        True
    """

    part: str

    def holds_of(self, name: str) -> bool:
        """Answer whether the text occurs in `name`.

        Args:
            name: The column name to test.

        Returns:
            True if the text occurs in the name.

        Example:
            >>> Contains("tmp").holds_of("venue")
            False
        """
        return self.part in name

    def language(self) -> NameSet:
        """Give the set of every name in which the text occurs.

        Returns:
            Any text, then the fixed text, then any text.

        Example:
            >>> Contains("bc").language().accepts("abcd")
            True
        """
        middle = NameSet.anything().concatenate(NameSet.literal(self.part))
        return middle.concatenate(NameSet.anything())


@dataclass(frozen=True)
class Matching(Absence):
    """A rule with no named shape, carried as the exact set of names it holds of.

    Some rules fit none of the other constructors. A filter that the
    engine moves past a rename tests a changed name, and the result
    is often no prefix, suffix or list. A pattern selector such as
    `cs.matches("^px_")` also gives one. This constructor carries such
    a rule as its set of names. A user cannot write it directly.

    Two of these are equal if they hold of the same names, however
    each was built. So equality does not compare the set itself.
    It compares `signature`, a text that the minimal automaton of the
    set fixes.

    Example:
        >>> Matching.of(NameSet.literal("abc")).holds_of("abc")
        True
    """

    signature: str
    names: NameSet = field(compare=False, repr=False)

    @classmethod
    def of(cls, names: NameSet) -> Matching:
        """Build the rule that holds of exactly the names in a set.

        Args:
            names: The set of names.

        Returns:
            The rule, which compares by the signature of the set.

        Example:
            >>> a = Matching.of(StartsWith("a").language())
            >>> both = Or(StartsWith("a"), StartsWith("ab"))
            >>> a == Matching.of(both.language())
            True
        """
        return cls(names.signature(), names)

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` is in the carried set.

        Args:
            name: The column name to test.

        Returns:
            True if the set holds the name.

        Example:
            >>> Matching.of(NameSet.literal("a")).holds_of("b")
            False
        """
        return self.names.accepts(name)

    def language(self) -> NameSet:
        """Give the carried set.

        Returns:
            The set of names this rule holds of.

        Example:
            >>> Matching.of(NameSet.literal("a")).language().accepts("a")
            True
        """
        return self.names


@dataclass(frozen=True)
class Not(Absence):
    """A name is absent if the inner rule does not hold of it.

    An operation that keeps only some columns gives this rule. After
    a select of the names that begin with `a`, every other name is
    absent.

    Example:
        >>> Not(StartsWith("a")).holds_of("b")
        True
    """

    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer the opposite of the inner rule.

        Args:
            name: The column name to test.

        Returns:
            True if the inner rule does not hold of the name.

        Example:
            >>> Not(Always()).holds_of("x")
            False
        """
        return not self.inner.holds_of(name)

    def language(self) -> NameSet:
        """Give the set of every name that the inner rule does not hold of.

        Returns:
            The complement of the inner set.

        Example:
            >>> Not(StartsWith("a")).language().accepts("b")
            True
        """
        return self.inner.language().complement()


@dataclass(frozen=True)
class And(Absence):
    """A name is absent if both rules hold of it.

    The union of two remainders gives this rule: a name is absent
    from the union only if it is absent from both.

    Example:
        >>> And(StartsWith("a"), EndsWith("z")).holds_of("az")
        True
    """

    left: Absence
    right: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether both rules hold of `name`.

        Args:
            name: The column name to test.

        Returns:
            True if both rules hold.

        Example:
            >>> And(StartsWith("a"), EndsWith("z")).holds_of("ab")
            False
        """
        return self.left.holds_of(name) and self.right.holds_of(name)

    def language(self) -> NameSet:
        """Give the set of the names that both rules hold of.

        Returns:
            The intersection of the two sets.

        Example:
            >>> And(StartsWith("a"), EndsWith("z")).language().accepts("az")
            True
        """
        return self.left.language().intersection(self.right.language())


@dataclass(frozen=True)
class Or(Absence):
    """A name is absent if either rule holds of it.

    A second drop gives this rule: the names that each drop proved
    absent all stay absent.

    Example:
        >>> Or(OneOf(frozenset({"a"})), OneOf(frozenset({"b"}))).holds_of("b")
        True
    """

    left: Absence
    right: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether at least one rule holds of `name`.

        Args:
            name: The column name to test.

        Returns:
            True if at least one rule holds.

        Example:
            >>> Or(Never(), Never()).holds_of("a")
            False
        """
        return self.left.holds_of(name) or self.right.holds_of(name)

    def language(self) -> NameSet:
        """Give the set of the names that either rule holds of.

        Returns:
            The union of the two sets.

        Example:
            >>> Or(StartsWith("a"), StartsWith("b")).language().accepts("b")
            True
        """
        return self.left.language().union(self.right.language())


@dataclass(frozen=True)
class AfterStripping(Absence):
    """A name ends with a suffix, and the inner rule holds of the name without it.

    An operation that adds a suffix to every column name needs this
    rule, for example `.name.suffix("_prev")` over the undeclared
    columns. Before the rename, the inner rule proved some names
    absent. After it, `x_prev` is absent if `x` was absent. The
    suffix effect puts this rule beside a rule that holds of every
    name without the suffix. Together they say that the name did not come from adding
    the suffix to a name the remainder held.

    Example:
        >>> rule = AfterStripping("_prev", OneOf(frozenset({"price"})))
        >>> rule.holds_of("price_prev")
        True
    """

    suffix: str
    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` ends with the suffix and the rest is absent.

        Args:
            name: The column name to test.

        Returns:
            True if the name ends with the suffix, and the inner rule
            holds of the name without that suffix.

        Example:
            >>> AfterStripping("_prev", Always()).holds_of("price")
            False
        """
        if not name.endswith(self.suffix):
            return False
        return self.inner.holds_of(name[: len(name) - len(self.suffix)])

    def language(self) -> NameSet:
        """Give the set of the inner names, each with the suffix added.

        Returns:
            The inner set, followed by the suffix.

        Example:
            >>> rule = AfterStripping("_p", OneOf(frozenset({"price"})))
            >>> rule.language().accepts("price_p")
            True
        """
        return self.inner.language().concatenate(NameSet.literal(self.suffix))


@dataclass(frozen=True)
class BeforeStripping(Absence):
    """A name begins with a prefix, and the inner rule holds of the name without it.

    This is `AfterStripping` for the other end of the name. An
    operation that adds a prefix to every column name needs it.

    Example:
        >>> rule = BeforeStripping("sec_", OneOf(frozenset({"country"})))
        >>> rule.holds_of("sec_country")
        True
    """

    prefix: str
    inner: Absence

    def holds_of(self, name: str) -> bool:
        """Answer whether `name` begins with the prefix and the rest is absent.

        Args:
            name: The column name to test.

        Returns:
            True if the name begins with the prefix, and the inner
            rule holds of the name without that prefix.

        Example:
            >>> BeforeStripping("sec_", Always()).holds_of("country")
            False
        """
        if not name.startswith(self.prefix):
            return False
        return self.inner.holds_of(name[len(self.prefix) :])

    def language(self) -> NameSet:
        """Give the set of the inner names, each with the prefix added.

        Returns:
            The prefix, followed by the inner set.

        Example:
            >>> rule = BeforeStripping("s_", OneOf(frozenset({"a"})))
            >>> rule.language().accepts("s_a")
            True
        """
        return NameSet.literal(self.prefix).concatenate(self.inner.language())
