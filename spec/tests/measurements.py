"""The counts the specification quotes, computed rather than remembered.

Each class here answers one question of the form "over every caller, how
often is this operation exact?". A document quotes the number and names
the class, so a reader can rerun it and a drift becomes a test failure.

Every count is computed through `model.py`, not beside it. The precision
question is decided by the same `OfType` methods the model uses to
transform a record, so a measurement cannot disagree with the rule it
claims to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, combinations
from typing import TYPE_CHECKING

from spec.tests.model import Always, ColumnSet, OfType

if TYPE_CHECKING:
    from collections.abc import Iterator

    from spec.tests.model import Family
    from spec.tests.normal_form import Canonical, Expr, ExpressionSpace
    from spec.tests.universe import Universe

Meaning = tuple[tuple[tuple[str, str], ...], ...]


@dataclass(frozen=True)
class TypeTestPrecision:
    """How often a type selector answers exactly, over every caller shape.

    A caller brings some subset of a fixed set of extra columns. The
    selector `cs.numeric()` gives an exact answer for that caller when
    the caller's data types are all inside the family or all outside
    it. Anywhere in between, the answer depends on which columns the
    caller brought, and no single result type covers every caller.

    Example:
        >>> from spec.tests.model import NUMERIC
        >>> p = TypeTestPrecision((("n1", "Int64"), ("s1", "String")), NUMERIC)
        >>> p.total(), p.exact(), p.imprecise()
        (4, 3, 1)
    """

    columns: tuple[tuple[str, str], ...]
    family: Family

    def shapes(self) -> Iterator[ColumnSet]:
        """Enumerate every subset of the fixed columns a caller can bring.

        Yields:
            One column set for each subset, the empty one included.

        Example:
            >>> from spec.tests.model import NUMERIC
            >>> p = TypeTestPrecision((("n1", "Int64"),), NUMERIC)
            >>> [c.names() for c in p.shapes()]
            [(), ('n1',)]
        """
        counts = range(len(self.columns) + 1)
        for chosen in chain.from_iterable(combinations(self.columns, k) for k in counts):
            yield ColumnSet(tuple(sorted(chosen)))

    def is_exact(self, shape: ColumnSet) -> bool:
        """Answer whether the type selector settles this caller exactly.

        The question is decided by the model's own selector methods, so
        this measurement cannot drift from the rule it measures.

        Args:
            shape: The columns one caller brings.

        Returns:
            True when the selector picks every column or none of them.

        Example:
            >>> from spec.tests.model import NUMERIC
            >>> p = TypeTestPrecision((("n1", "Int64"),), NUMERIC)
            >>> p.is_exact(ColumnSet.of(n1="Int64"))
            True
        """
        selector, types = OfType(self.family), shape.type_set()
        settled_by_picking = selector.certainly_picks(types) == Always()
        return settled_by_picking or selector.certainly_rejects(types) == Always()

    def total(self) -> int:
        """Count the caller shapes.

        Returns:
            Two to the power of the number of fixed columns.

        Example:
            >>> from spec.tests.model import NUMERIC
            >>> TypeTestPrecision((("a", "Int64"), ("b", "String")), NUMERIC).total()
            4
        """
        return int(2 ** len(self.columns))

    def exact(self) -> int:
        """Count the caller shapes the selector settles exactly.

        Returns:
            How many shapes give an answer good for every caller.

        Example:
            >>> from spec.tests.model import NUMERIC
            >>> TypeTestPrecision((("a", "Int64"),), NUMERIC).exact()
            2
        """
        return sum(1 for shape in self.shapes() if self.is_exact(shape))

    def imprecise(self) -> int:
        """Count the caller shapes the selector cannot settle.

        Returns:
            How many shapes hold both a matching and a non-matching
            data type, so the result depends on the caller.

        Example:
            >>> from spec.tests.model import NUMERIC
            >>> TypeTestPrecision((("a", "Int64"), ("b", "String")), NUMERIC).imprecise()
            1
        """
        return self.total() - self.exact()


@dataclass(frozen=True)
class CanonicalPrecision:
    """How often the canonical form decides equality, over a space of expressions.

    Two expressions denoting the same thing under every binding are the
    same type. The canonical form decides that question by comparing
    two values, and it decides it soundly: equal forms always mean
    equal denotations. This class counts the other direction, which is
    where the system is incomplete.

    The count is over a finite universe, so "equal denotations" here
    means equal on every binding that universe allows. A wider universe
    can only split more classes, never fewer, so a class counted as one
    meaning here is evidence and not proof.

    Example:
        >>> from spec.tests.test_normal_form import SPACE
        >>> from spec.tests.universe import Universe
        >>> p = CanonicalPrecision(SPACE, Universe(("a",), ("Int64",)), 1)
        >>> p.meaning_classes() > 0
        True
    """

    space: ExpressionSpace
    universe: Universe
    depth: int

    def groups(self) -> dict[Meaning, list[Expr]]:
        """Group the expressions by what they denote.

        Returns:
            Each distinct meaning, mapped to the expressions with it.

        Example:
            >>> from spec.tests.test_normal_form import SPACE
            >>> from spec.tests.universe import Universe
            >>> p = CanonicalPrecision(SPACE, Universe(("a",), ("Int64",)), 1)
            >>> len(p.groups()) > 1
            True
        """
        found: dict[Meaning, list[Expr]] = {}
        for expression in self.space.up_to_depth(self.depth):
            found.setdefault(self._meaning(expression), []).append(expression)
        return found

    def _meaning(self, expression: Expr) -> Meaning:
        """Give what an expression denotes under every binding.

        Args:
            expression: The expression to evaluate.

        Returns:
            One column set per binding, as a hashable tuple.

        Example:
            >>> from spec.tests.test_normal_form import SPACE
            >>> from spec.tests.universe import Universe
            >>> from spec.tests.normal_form import Ref
            >>> p = CanonicalPrecision(SPACE, Universe(("a",), ("Int64",)), 1)
            >>> len(p._meaning(Ref("R"))) > 0
            True
        """
        bindings = self.space.bindings(self.universe)
        return tuple(tuple(sorted(expression.denote(b).entries)) for b in bindings)

    def meaning_classes(self) -> int:
        """Count the distinct meanings the space produces.

        Returns:
            How many meanings the expressions have between them.

        Example:
            >>> from spec.tests.test_normal_form import SPACE
            >>> from spec.tests.universe import Universe
            >>> p = CanonicalPrecision(SPACE, Universe(("a",), ("Int64",)), 1)
            >>> isinstance(p.meaning_classes(), int)
            True
        """
        return len(self.groups())

    def undecided(self) -> list[list[Expr]]:
        """List the meanings the canonical form splits into several forms.

        Inside one meaning, the first expression with each canonical
        form represents that form. A meaning with one form only is
        decided, so it is left out.

        Returns:
            One list of representative expressions per split meaning.

        Example:
            >>> from spec.tests.test_normal_form import SPACE
            >>> from spec.tests.universe import Universe
            >>> p = CanonicalPrecision(SPACE, Universe(("a",), ("Int64",)), 1)
            >>> isinstance(p.undecided(), list)
            True
        """
        undecided = []
        for group in self.groups().values():
            seen: dict[Canonical, Expr] = {}
            for expression in group:
                seen.setdefault(expression.normalise(), expression)
            if len(seen) > 1:
                undecided.append(list(seen.values()))
        return undecided


@dataclass(frozen=True)
class ConfirmedResidue:
    """The equalities a canonical form leaves undecided, checked twice.

    Grouping expressions by what they denote needs a universe of names
    and data types to evaluate over, and a narrow universe makes two
    expressions look equal when a longer name would separate them. So
    the count is taken in two passes: a coarse universe proposes the
    candidates, and a wider one confirms each candidate really is an
    equality the canonical form misses.

    A candidate the wider pass rejects was never an equality, so it is
    not a limit of the canonical form.

    Example:
        >>> from spec.tests.test_normal_form import SPACE, WIDE_UNIVERSE
        >>> from spec.tests.universe import Universe
        >>> wider = Universe(("", "a", "b", "ab", "xb", "axb"), ("Int64",))
        >>> isinstance(ConfirmedResidue(SPACE, WIDE_UNIVERSE, wider, 2).count(), int)
        True
    """

    space: ExpressionSpace
    coarse: Universe
    wider: Universe
    depth: int

    def candidates(self) -> list[list[Expr]]:
        """Propose the meanings the canonical form appears to split.

        Returns:
            One list of representative expressions per candidate.
        """
        return CanonicalPrecision(self.space, self.coarse, self.depth).undecided()

    def confirmed(self) -> list[list[Expr]]:
        """Keep the candidates the wider universe agrees are equalities.

        Returns:
            The candidates whose expressions still denote the same
            thing over the wider universe.
        """
        return [group for group in self.candidates() if self._agrees(group)]

    def _agrees(self, group: list[Expr]) -> bool:
        """Answer whether a candidate's expressions still agree.

        Args:
            group: Representative expressions of one candidate.

        Returns:
            True when every pair still denotes the same column set
            under every binding of the wider universe.
        """
        bindings = self.space.bindings(self.wider)
        first = group[0]
        return all(first.denote(b) == other.denote(b) for other in group[1:] for b in bindings)

    def count(self) -> int:
        """Count the confirmed equalities the canonical form misses.

        Returns:
            How many meanings survive both passes.
        """
        return len(self.confirmed())
