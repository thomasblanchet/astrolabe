"""Decide whether one absence rule proves at least the names that another demands.

A remainder is the columns of a frame that its type does not name.
An absence rule is a test on a column name that holds if the name is
certainly not in the remainder. Two questions compare such rules. A
call must prove what a declaration demands, for example that the
caller's remainder holds no name that begins with `tmp_`. A
canonical form must know when two rules built in different ways hold
of the same names, or it gives one type two spellings. Both
questions ask whether the names of one rule are all names of the
other.

Every rule gives a regular set of names, so a finite automaton can
always decide the question exactly. Building and comparing automata
is slow, so a short list of cheap rules for common shapes runs first.
"""

from __future__ import annotations

from astrolabe._hooks.remainder.absence import (
    Absence,
    Always,
    And,
    Contains,
    EndsWith,
    Never,
    OneOf,
    Or,
    StartsWith,
)


class AbsenceEntailment:
    """Decide whether every name that one absence rule holds of, the other holds of too.

    Code that checks a requirement asks this: the rule that a record
    proves is the narrower side, and the rule that a declaration
    demands is the wider side. Each check answers True only for a
    real containment. If a cheap check does not apply, the next one
    runs, and the automaton at the end is exact. So the whole answer
    is exact.

    Example:
        >>> AbsenceEntailment()(StartsWith("ab"), StartsWith("a"))
        True
        >>> AbsenceEntailment()(StartsWith("a"), StartsWith("ab"))
        False
    """

    def __call__(self, left: Absence, right: Absence) -> bool:
        """Answer whether every name that `left` holds of, `right` holds of too.

        The checks run in order, and the first True answers. The cheap
        checks come first. The automaton comes last.

        Args:
            left: The rule that must be the narrower one.
            right: The rule that must be the wider one.

        Returns:
            True if the containment holds.

        Example:
            >>> AbsenceEntailment()(Never(), OneOf(frozenset({"a"})))
            True
        """
        rules = (
            self._identical,
            self._empty_left,
            self._total_right,
            self._vacuous_right,
            self._finite_left,
            self._same_affix,
            self._left_conjunction,
            self._right_conjunction,
            self._disjunctions,
            self._by_automaton,
        )
        return any(rule(left, right) for rule in rules)

    def _by_automaton(self, left: Absence, right: Absence) -> bool:
        """Decide the containment exactly, by comparing the name sets of the rules.

        This check answers every case that the cheap checks leave
        open. It runs last, because it builds and compares two finite
        automata, which costs much more than a text comparison.

        Args:
            left: The rule that must be the narrower one.
            right: The rule that must be the wider one.

        Returns:
            True exactly if every name that `left` holds of, `right`
            holds of too.

        Example:
            >>> from astrolabe._hooks.remainder.absence import Not
            >>> AbsenceEntailment()._by_automaton(
            ...     Not(Not(StartsWith("a"))), StartsWith("a")
            ... )
            True
        """
        return left.language().contained_in(right.language())

    def _identical(self, left: Absence, right: Absence) -> bool:
        """Answer whether the two rules are equal, which is the cheapest check.

        Args:
            left: One rule.
            right: The other.

        Returns:
            True if they are equal.

        Example:
            >>> AbsenceEntailment()._identical(Always(), Always())
            True
        """
        return left == right

    def _empty_left(self, left: Absence, right: Absence) -> bool:
        """Answer True if `left` holds of no name, because then any rule contains it.

        Args:
            left: The rule that must hold of no name.
            right: Not read.

        Returns:
            True if `left` is `Never`.

        Example:
            >>> AbsenceEntailment()._empty_left(Never(), Never())
            True
        """
        del right
        return isinstance(left, Never)

    def _total_right(self, left: Absence, right: Absence) -> bool:
        """Answer True if `right` holds of every name, because it then contains any rule.

        Args:
            left: Not read.
            right: The rule that must hold of every name.

        Returns:
            True if `right` is `Always`.

        Example:
            >>> AbsenceEntailment()._total_right(StartsWith("a"), Always())
            True
        """
        del left
        return isinstance(right, Always)

    def _vacuous_right(self, left: Absence, right: Absence) -> bool:
        """Answer True if `right` tests for an empty text, which every name passes.

        Every name begins with the empty text, ends with it and
        contains it. So `StartsWith("")`, `EndsWith("")` and
        `Contains("")` hold of every name, and contain any rule.

        Args:
            left: Not read.
            right: The rule to test.

        Returns:
            True if `right` holds of every name for this reason.

        Example:
            >>> AbsenceEntailment()._vacuous_right(
            ...     OneOf(frozenset()), StartsWith("")
            ... )
            True
            >>> AbsenceEntailment()._vacuous_right(Never(), Contains("a"))
            False
        """
        del left
        if isinstance(right, StartsWith):
            return not right.prefix
        if isinstance(right, EndsWith):
            return not right.suffix
        return isinstance(right, Contains) and not right.part

    def _finite_left(self, left: Absence, right: Absence) -> bool:
        """Answer True if `left` is a list of names and `right` holds of each of them.

        A finite list can be tested name by name. So this one check
        covers a list against every other constructor.

        Args:
            left: The rule, which must be a `OneOf`.
            right: The rule to test each name against.

        Returns:
            True if `right` holds of every listed name.

        Example:
            >>> names = OneOf(frozenset({"ab", "ac"}))
            >>> AbsenceEntailment()._finite_left(names, StartsWith("a"))
            True
        """
        if not isinstance(left, OneOf):
            return False
        return all(right.holds_of(name) for name in left.names)

    def _same_affix(self, left: Absence, right: Absence) -> bool:
        """Answer True if a test for a longer text implies a test for a shorter one.

        Every name that begins with `ab` begins with `a`. Every name
        that ends with `xy` ends with `y`. Every name that contains
        `abc` contains `bc`. Every name that begins with `sec_`
        contains `ec`, so a prefix or a suffix test implies a substring
        test if the substring occurs in the prefix or the suffix.

        Args:
            left: The narrower text test.
            right: The wider one.

        Returns:
            True if one of these text comparisons proves the
            containment.

        Example:
            >>> AbsenceEntailment()._same_affix(EndsWith("_ab"), EndsWith("b"))
            True
            >>> AbsenceEntailment()._same_affix(StartsWith("ab"), StartsWith("a"))
            True
            >>> AbsenceEntailment()._same_affix(Contains("abc"), Contains("bc"))
            True
            >>> AbsenceEntailment()._same_affix(StartsWith("sec_"), Contains("ec"))
            True
            >>> AbsenceEntailment()._same_affix(StartsWith("a"), EndsWith("a"))
            False
        """
        if isinstance(left, StartsWith) and isinstance(right, StartsWith):
            return left.prefix.startswith(right.prefix)
        if isinstance(left, EndsWith) and isinstance(right, EndsWith):
            return left.suffix.endswith(right.suffix)
        if not isinstance(right, Contains):
            return False
        if isinstance(left, Contains):
            return right.part in left.part
        return right.part in self._affix_of(left)

    def _affix_of(self, rule: Absence) -> str:
        """Give the text that a prefix or suffix test demands.

        Args:
            rule: The rule to read.

        Returns:
            The prefix or the suffix. For any other rule, the empty
            text, in which no substring except the empty one occurs.

        Example:
            >>> AbsenceEntailment()._affix_of(EndsWith("_prev"))
            '_prev'
        """
        if isinstance(rule, StartsWith):
            return rule.prefix
        return rule.suffix if isinstance(rule, EndsWith) else ""

    def _left_conjunction(self, left: Absence, right: Absence) -> bool:
        """Answer True if `left` is an `And`, and one of its halves is inside `right`.

        An `And` holds of fewer names than either half, so a
        containment of one half is enough.

        Args:
            left: The rule, which must be an `And`.
            right: The wider rule.

        Returns:
            True if either half proves the containment.

        Example:
            >>> both = And(StartsWith("ab"), EndsWith("z"))
            >>> AbsenceEntailment()._left_conjunction(both, StartsWith("a"))
            True
        """
        if not isinstance(left, And):
            return False
        return self(left.left, right) or self(left.right, right)

    def _right_conjunction(self, left: Absence, right: Absence) -> bool:
        """Answer True if `right` is an `And`, and `left` is inside both halves.

        Args:
            left: The narrower rule.
            right: The rule, which must be an `And`.

        Returns:
            True if both halves contain `left`.

        Example:
            >>> wide = And(StartsWith("a"), EndsWith("z"))
            >>> AbsenceEntailment()._right_conjunction(
            ...     OneOf(frozenset({"az"})), wide
            ... )
            True
        """
        if not isinstance(right, And):
            return False
        return self(left, right.left) and self(left, right.right)

    def _disjunctions(self, left: Absence, right: Absence) -> bool:
        """Answer the containment if either side is an `Or`.

        An `Or` is inside a rule if both of its halves are. A rule is
        inside an `Or` if it is inside either half. The second test
        is enough to prove a containment, but it can miss one that
        only the whole `Or` gives. The automaton then decides it.

        Args:
            left: The narrower rule.
            right: The wider one.

        Returns:
            True if either test proves the containment.

        Example:
            >>> wide = Or(StartsWith("a"), StartsWith("b"))
            >>> AbsenceEntailment()._disjunctions(StartsWith("bc"), wide)
            True
            >>> either = Or(StartsWith("ab"), StartsWith("ac"))
            >>> AbsenceEntailment()._disjunctions(either, StartsWith("a"))
            True
        """
        if isinstance(left, Or) and self(left.left, right) and self(left.right, right):
            return True
        if not isinstance(right, Or):
            return False
        return self(left, right.left) or self(left, right.right)
