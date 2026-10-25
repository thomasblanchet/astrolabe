"""How the two relational requirements behave when one side is described in parts.

A remainder can be divided into named parts, and then a promise about it has to
name the parts rather than the whole. That raises a question the documents
answer and must not answer wrongly: is a promise about a union the same as a
promise about each piece?

For disjointness it is, and for containment it is not. These tests establish
both by exhaustion over a small universe, because the answer decides how a
checker may take the question apart.
"""

from __future__ import annotations

from itertools import chain, combinations, product
from typing import ClassVar

import pytest

from spec.tests.model import (
    EVERY_DTYPE,
    NUMERIC,
    TEMPORAL,
    Absence,
    AbsenceEntailment,
    Always,
    And,
    Contains,
    EndsWith,
    NameTest,
    Never,
    Not,
    OfType,
    OneOf,
    Or,
    RegularLanguage,
    SelAnd,
    Selector,
    SelectorEntailment,
    SelNot,
    SelOr,
    StartsWith,
)

UNIVERSE = ("a", "b", "c")


def subsets() -> list[frozenset[str]]:
    """Enumerate every set of names the universe allows.

    Returns:
        Each subset of `UNIVERSE`, the empty one included.
    """
    sizes = range(len(UNIVERSE) + 1)
    chosen = chain.from_iterable(combinations(UNIVERSE, k) for k in sizes)
    return [frozenset(c) for c in chosen]


ALL = subsets()


class TestDisjointnessDistributes:
    """Sharing no name with a union is sharing no name with each part."""

    def test_the_two_readings_agree_everywhere(self) -> None:
        """Over every triple of sets, the union form and the pairwise form match.

        This is what lets a checker take a `disjoint_from={A, B}`
        promise apart and answer it one part at a time.
        """
        for right in ALL:
            for first in ALL:
                self.assert_agrees(right, first)

    def assert_agrees(self, right: frozenset[str], first: frozenset[str]) -> None:
        """Check both readings against every choice of the second part.

        Args:
            right: The remainder making the promise.
            first: The first part of the split it promises about.
        """
        for second in ALL:
            union_form = not (right & (first | second))
            pairwise = not (right & first) and not (right & second)
            assert union_form == pairwise


class TestContainmentDoesNotDistribute:
    """Being within a union is not being within one part or the other."""

    def test_a_counterexample_exists(self) -> None:
        """One name in each part, and the remainder is within neither alone.

        So a `within={A, B}` promise has to be decided against the union
        and cannot be split into two smaller questions.
        """
        right = frozenset({"a", "b"})
        first, second = frozenset({"a"}), frozenset({"b"})
        assert right <= (first | second)
        assert not (right <= first)
        assert not (right <= second)

    @pytest.mark.parametrize("right", ALL, ids=lambda s: "".join(sorted(s)) or "empty")
    def test_the_union_reading_is_the_weaker_one(self, right: frozenset[str]) -> None:
        """Containment in one part always implies containment in the union.

        The implication runs one way only, which is exactly what the
        counterexample above shows.
        """
        for first, second in product(ALL, ALL):
            self.assert_one_way(right, first, second)

    def assert_one_way(
        self,
        right: frozenset[str],
        first: frozenset[str],
        second: frozenset[str],
    ) -> None:
        """Check that containment in a part implies containment in the union.

        Args:
            right: The remainder making the promise.
            first: One part of the split it promises about.
            second: The other part.
        """
        if right <= first or right <= second:
            assert right <= (first | second)


class TestContainment:
    """Containment between two name rules is decided, not approximated.

    Every constructor of the absence language denotes a regular set of
    names, so containment between two rules is a containment between
    two regular languages, which is decidable. The test answers **every**
    question, not only the ones a cheap rule settles.

    Cheap rules are tried first. They cost a string comparison where an
    automaton comparison would cost far more, and a rule they settle
    keeps the shape the user wrote, which an error message can name.
    """

    HOLDS: ClassVar[list[tuple[str, Absence, Absence]]] = [
        ("a longer prefix", StartsWith("ab"), StartsWith("a")),
        ("a longer suffix", EndsWith("_ab"), EndsWith("b")),
        ("a longer substring", Contains("abc"), Contains("bc")),
        ("a prefix inside a substring", StartsWith("sec_"), Contains("ec")),
        ("a name list, member by member", OneOf(frozenset({"ab", "ac"})), StartsWith("a")),
        ("nothing is inside everything", Never(), OneOf(frozenset({"a"}))),
        ("everything holds every rule", StartsWith("a"), Always()),
        ("an empty affix holds every name", OneOf(frozenset({"z"})), StartsWith("")),
        ("one half of a conjunction", And(StartsWith("ab"), EndsWith("z")), StartsWith("a")),
        (
            "both halves of a conjunction",
            OneOf(frozenset({"az"})),
            And(StartsWith("a"), EndsWith("z")),
        ),
        ("either half of a disjunction", StartsWith("ab"), Or(StartsWith("a"), StartsWith("b"))),
        ("a prefix inside a substring test", StartsWith("a"), Contains("a")),
        ("a suffix inside a substring test", EndsWith("a"), Contains("a")),
        ("a double negation", Not(Not(StartsWith("a"))), StartsWith("a")),
        (
            "a contradiction is inside anything",
            And(StartsWith("a"), StartsWith("b")),
            OneOf(frozenset({"q"})),
        ),
        ("a tautology holds every rule", Always(), Or(StartsWith("a"), Not(StartsWith("a")))),
    ]

    FAILS: ClassVar[list[tuple[str, Absence, Absence]]] = [
        ("a shorter prefix is not inside a longer one", StartsWith("a"), StartsWith("ab")),
        ("a substring test is not inside a prefix test", Contains("ab"), StartsWith("a")),
        ("two unrelated prefixes", StartsWith("a"), StartsWith("b")),
        ("everything is not inside a name list", Always(), OneOf(frozenset({"a"}))),
    ]

    @pytest.mark.parametrize(("label", "narrow", "wide"), HOLDS, ids=lambda x: x)
    def test_a_containment_that_holds_is_established(
        self,
        label: str,
        narrow: Absence,
        wide: Absence,
    ) -> None:
        """Each of these holds, and the test says so."""
        del label
        assert AbsenceEntailment()(narrow, wide)

    @pytest.mark.parametrize(("label", "narrow", "wide"), FAILS, ids=lambda x: x)
    def test_a_containment_that_fails_is_refused(
        self,
        label: str,
        narrow: Absence,
        wide: Absence,
    ) -> None:
        """Each of these does not hold, and the test says so."""
        del label
        assert not AbsenceEntailment()(narrow, wide)

    NAMES: ClassVar[list[str]] = [
        "",
        "a",
        "b",
        "ab",
        "ac",
        "abc",
        "_ab",
        "z",
        "az",
        "sec_x",
        "x_prev",
        "q",
    ]

    @pytest.mark.parametrize(("label", "narrow", "wide"), HOLDS, ids=lambda x: x)
    def test_an_established_containment_really_holds(
        self,
        label: str,
        narrow: Absence,
        wide: Absence,
    ) -> None:
        """A claimed containment is checked name by name.

        An entailment claimed but false would collapse two types that
        denote different frames, so this is the half that matters.
        """
        del label
        for name in self.NAMES:
            assert not narrow.holds_of(name) or wide.holds_of(name)

    @pytest.mark.parametrize(("label", "narrow", "wide"), FAILS, ids=lambda x: x)
    def test_a_refused_containment_really_fails(
        self,
        label: str,
        narrow: Absence,
        wide: Absence,
    ) -> None:
        """A refused containment has a name that witnesses the failure."""
        del label
        assert any(narrow.holds_of(n) and not wide.holds_of(n) for n in self.NAMES)

    def test_the_cheap_rules_agree_with_the_complete_one(self) -> None:
        """Where a cheap rule fires, the automaton agrees.

        The cheap rules exist for speed and for diagnostics, never to
        give a different answer.
        """
        entailment, language = AbsenceEntailment(), RegularLanguage()
        for _, narrow, wide in self.HOLDS + self.FAILS:
            exact = language.contains(language.of(narrow), language.of(wide))
            assert entailment(narrow, wide) == exact


RULE_BATTERY: tuple[Absence, ...] = (
    Never(),
    Always(),
    OneOf(frozenset({"a"})),
    OneOf(frozenset({"a", "b"})),
    OneOf(frozenset({"c"})),
    StartsWith("a"),
    Not(OneOf(frozenset({"a"}))),
    Or(OneOf(frozenset({"a"})), OneOf(frozenset({"c"}))),
)


class TestPositiveRequirements:
    """The three positive forms are decided from a remainder's rule.

    `only=[...]`, `only_prefix=` and `only_suffix=` hold of a remainder
    when every name outside their shape is provably absent, so each is
    the entailment of the shape's complement by the remainder's rule.
    """

    CASES: ClassVar[tuple[tuple[str, Absence, Absence], ...]] = (
        ("only a name list", Not(OneOf(frozenset({"a", "b"}))), OneOf(frozenset({"a", "b"}))),
        ("only a prefix", Not(StartsWith("a")), StartsWith("a")),
        ("only a suffix", Not(EndsWith("c")), EndsWith("c")),
    )

    @pytest.mark.parametrize(("label", "rule", "shape"), CASES, ids=lambda x: x)
    def test_each_positive_form_is_decided_by_its_complement(
        self,
        label: str,
        rule: Absence,
        shape: Absence,
    ) -> None:
        """A rule holding outside the shape decides the positive form."""
        del label
        assert AbsenceEntailment()(Not(shape), rule)

    @pytest.mark.parametrize(("label", "rule", "shape"), CASES, ids=lambda x: x)
    def test_a_decided_positive_form_really_holds(
        self,
        label: str,
        rule: Absence,
        shape: Absence,
    ) -> None:
        """Every name set the rule permits sits inside the shape."""
        del label
        for names in ALL:
            self.assert_positive_form(names, rule, shape)

    def assert_positive_form(self, names: frozenset[str], rule: Absence, shape: Absence) -> None:
        """Check one permitted name set against the decided shape.

        Args:
            names: One candidate name set.
            rule: The remainder's absence rule.
            shape: The decided positive form.
        """
        if names <= self.allowed(rule):
            assert all(shape.holds_of(n) for n in names)

    def test_a_rule_proving_nothing_decides_no_positive_form(self) -> None:
        """The empty rule leaves every positive form undecided."""
        for _, _, shape in self.CASES:
            assert not AbsenceEntailment()(Not(shape), Never())

    def allowed(self, rule: Absence) -> frozenset[str]:
        """Give the universe names a rule does not prove absent.

        Args:
            rule: The remainder's absence rule.

        Returns:
            The names a described column set can still hold.

        Example:
            >>> TestPositiveRequirements().allowed(Not(OneOf(frozenset({"a"}))))
            frozenset({'a'})
        """
        return frozenset(n for n in UNIVERSE if not rule.holds_of(n))


class TestRelationalDecisionsFromRules:
    """`R-10` and `R-11`, decided from an absence rule, keep `T-13`'s shape.

    The set-level fact is measured above; these sweeps re-state it in
    the record language: a disjointness promise is proven when the rule
    holds of every promised name, a containment promise when the rule
    holds of every universe name outside the promised set.
    """

    def test_disjointness_decided_from_a_rule_distributes(self) -> None:
        """Proven against a union exactly when proven against each part."""
        for rule in RULE_BATTERY:
            for first in ALL:
                self.assert_distributes(rule, first)

    def assert_distributes(self, rule: Absence, first: frozenset[str]) -> None:
        """Check the union and pairwise readings at one first part.

        Args:
            rule: The remainder's absence rule.
            first: The first part of the promised split.
        """
        for second in ALL:
            union_form = self.proven_disjoint(rule, first | second)
            pairwise = self.proven_disjoint(rule, first)
            assert union_form == (pairwise and self.proven_disjoint(rule, second))

    def test_containment_decided_from_a_rule_does_not_distribute(self) -> None:
        """A within-promise proven for a union, proven for neither part."""
        rule = OneOf(frozenset({"c"}))
        first, second = frozenset({"a"}), frozenset({"b"})
        assert self.proven_within(rule, first | second)
        assert not self.proven_within(rule, first)
        assert not self.proven_within(rule, second)

    def test_containment_in_a_part_implies_the_union(self) -> None:
        """The one direction that does hold, swept over the battery."""
        for rule in RULE_BATTERY:
            for first in ALL:
                self.assert_one_way(rule, first)

    def assert_one_way(self, rule: Absence, first: frozenset[str]) -> None:
        """Check that a part's proof carries to the union.

        Args:
            rule: The remainder's absence rule.
            first: The part whose proof is assumed.
        """
        for second in ALL:
            if self.proven_within(rule, first):
                assert self.proven_within(rule, first | second)

    def proven_disjoint(self, rule: Absence, names: frozenset[str]) -> bool:
        """Answer whether a rule proves disjointness from a name set.

        Args:
            rule: The remainder's absence rule.
            names: The promised names.

        Returns:
            True when the rule holds of every promised name.

        Example:
            >>> probe = TestRelationalDecisionsFromRules()
            >>> probe.proven_disjoint(StartsWith("a"), frozenset({"a"}))
            True
        """
        return all(rule.holds_of(n) for n in names)

    def proven_within(self, rule: Absence, names: frozenset[str]) -> bool:
        """Answer whether a rule proves containment in a name set.

        Args:
            rule: The remainder's absence rule.
            names: The promised names.

        Returns:
            True when the rule holds of every universe name outside.

        Example:
            >>> probe = TestRelationalDecisionsFromRules()
            >>> probe.proven_within(OneOf(frozenset({"c"})), frozenset({"a", "b"}))
            True
        """
        return all(rule.holds_of(n) for n in UNIVERSE if n not in names)


class TestSelectorContainment:
    """Selector containment with type tests answers soundly.

    `T-9` is a claim about selectors, not only about name rules, and
    the type test is the constructor names cannot decide. Every
    containment the cheap path establishes over this battery is
    verified column by column, over every probe name and every
    universe data type.
    """

    SELECTORS: ClassVar[tuple[Selector, ...]] = (
        NameTest(StartsWith("a")),
        NameTest(StartsWith("ab")),
        NameTest(EndsWith("_p")),
        NameTest(OneOf(frozenset({"ab"}))),
        OfType(NUMERIC),
        OfType(TEMPORAL),
        SelAnd(OfType(NUMERIC), NameTest(StartsWith("a"))),
        SelAnd(OfType(NUMERIC), NameTest(StartsWith("ab"))),
        SelOr(OfType(NUMERIC), NameTest(StartsWith("a"))),
        SelNot(OfType(NUMERIC)),
    )

    NAMES: ClassVar[tuple[str, ...]] = (
        "a",
        "ab",
        "abc",
        "b",
        "x_p",
        "ab_p",
        "",
    )

    def test_every_established_containment_holds_of_every_column(self) -> None:
        """Where the cheap path answers yes, the columns agree.

        A containment claimed but false would collapse two selectors
        that pick different columns, so this is the half that matters.
        """
        entailment = SelectorEntailment()
        for narrow, wide in product(self.SELECTORS, self.SELECTORS):
            if entailment(narrow, wide):
                self.assert_contained(narrow, wide)

    def assert_contained(self, narrow: Selector, wide: Selector) -> None:
        """Check one established containment column by column.

        Args:
            narrow: The selector claimed to pick fewer columns.
            wide: The selector claimed to pick them all.
        """
        for name, dtype in product(self.NAMES, sorted(EVERY_DTYPE)):
            assert not narrow.picks(name, dtype) or wide.picks(name, dtype)

    def test_the_type_test_cases_are_established(self) -> None:
        """The containments the proof's type device exists for fire.

        A conjunction with a type test is inside either half, a type
        test is inside a disjunction holding it, and two disjoint
        families do not contain each other.
        """
        entailment = SelectorEntailment()
        narrowed = SelAnd(OfType(NUMERIC), NameTest(StartsWith("ab")))
        assert entailment(narrowed, NameTest(StartsWith("a")))
        assert entailment(narrowed, OfType(NUMERIC))
        assert entailment(OfType(NUMERIC), SelOr(OfType(NUMERIC), NameTest(StartsWith("a"))))
        assert not entailment(OfType(NUMERIC), OfType(TEMPORAL))

    def test_the_battery_is_not_vacuous(self) -> None:
        """A fixed share of the battery's pairs is established.

        Twenty-six of the hundred pairs fire, so the column-by-column
        sweep above checks real claims, not an empty set.
        """
        entailment = SelectorEntailment()
        fired = sum(
            1
            for narrow, wide in product(self.SELECTORS, self.SELECTORS)
            if entailment(narrow, wide)
        )
        assert fired == 26


class TestSplitPairingsInheritDisjointness:
    """One promise about two whole remainders discharges every pairing.

    A part is a subset of its frame's remainder, so two disjoint
    remainders give disjoint parts, whatever the two splits were. This
    is the fact that keeps a split from multiplying the promises a
    join needs.
    """

    def test_every_part_pairing_of_disjoint_remainders_is_disjoint(self) -> None:
        """Over every disjoint pair of remainders, no pairing collides."""
        for left, right in product(ALL, ALL):
            if not left & right:
                self.assert_pairings_disjoint(left, right)

    def assert_pairings_disjoint(self, left: frozenset[str], right: frozenset[str]) -> None:
        """Check every part pairing of one disjoint remainder pair.

        Args:
            left: One remainder's names.
            right: The other remainder's names, sharing none.
        """
        for part_left in self.parts_of(left):
            for part_right in self.parts_of(right):
                assert not part_left & part_right

    def parts_of(self, whole: frozenset[str]) -> list[frozenset[str]]:
        """Give every subset of a remainder, as its possible parts.

        Args:
            whole: The remainder's names.

        Returns:
            Each part a split of the remainder can produce.

        Example:
            >>> TestSplitPairingsInheritDisjointness().parts_of(frozenset({"a"}))
            [frozenset(), frozenset({'a'})]
        """
        return [part for part in ALL if part <= whole]
