"""The split theorems, run on the executable binder.

The split section claims a part is an ordinary selector expression,
that binding is one first-match pass, that two malformed splits are
rejected, and that a split settles every caller a single family
requirement leaves open. Each claim runs here against `split_model`,
over a finite universe, with the counts of the measurement module
recomputed from one split of one column set.
"""

from __future__ import annotations

from spec.tests.measurements import TypeTestPrecision
from spec.tests.model import (
    NUMERIC,
    Always,
    ColumnSet,
    NameTest,
    OfType,
    StartsWith,
)
from spec.tests.split_model import Split
from spec.tests.universe import Universe

UNIVERSE = Universe(("a", "b", "sec_x"), ("Int64", "String", "Date"))

NUMERIC_REST = Split((OfType(NUMERIC), NameTest(Always())))

PREFIX_NUMERIC_REST = Split((NameTest(StartsWith("sec_")), OfType(NUMERIC), NameTest(Always())))

MIXED_COLUMNS = (
    ("n1", "Int64"),
    ("n2", "Float64"),
    ("s1", "String"),
    ("s2", "Date"),
    ("s3", "Boolean"),
)


class TestThePartIsItsClosedForm:
    """Binding equals the ordinary selector expression, part for part."""

    def test_bound_parts_equal_the_closed_forms_everywhere(self) -> None:
        """The split theorem, by exhaustion over the universe.

        For every column set and every part, the first-match binder
        gives exactly the columns the part's closed-form selector
        picks, so a split adds no expression node.
        """
        for split in (NUMERIC_REST, PREFIX_NUMERIC_REST):
            for columns in UNIVERSE.column_sets():
                self.assert_parts_match(split, columns)

    def assert_parts_match(self, split: Split, columns: ColumnSet) -> None:
        """Compare the binder with the closed forms at one column set.

        Args:
            split: The split under test.
            columns: One possible remainder content.
        """
        bound = split.bound(columns)
        for index in range(len(split.parts)):
            assert bound[index] == self.closed_pick(split, columns, index)

    def closed_pick(self, split: Split, columns: ColumnSet, index: int) -> ColumnSet:
        """Bind one part's closed form directly against the columns.

        Args:
            split: The split under test.
            columns: One possible remainder content.
            index: The part whose closed form is bound.

        Returns:
            The columns the closed form picks.
        """
        form = split.closed_form(index)
        picked = [c for c in columns.entries if form.picks(*c)]
        return ColumnSet.holding(picked)

    def test_first_match_order_carries_meaning(self) -> None:
        """Swapping two overlapping parts moves the shared columns.

        The control that shows the binder really reads order: with an
        unconstrained part first, the numeric part is provably dead.
        """
        forward = NUMERIC_REST.bound(ColumnSet.of(n="Int64", s="String"))
        assert forward[0].names() == ("n",)
        swapped = Split((NameTest(Always()), OfType(NUMERIC)))
        backward = swapped.bound(ColumnSet.of(n="Int64", s="String"))
        assert backward[0].names() == ("n", "s")
        assert backward[1].names() == ()


class TestTheTwoRejections:
    """Both malformed split shapes are decided over the universe."""

    def test_a_split_with_a_gap_is_caught(self) -> None:
        """A column no part takes has nowhere to go."""
        gapped = Split((OfType(NUMERIC),))
        assert not gapped.places_everything(UNIVERSE)
        assert gapped.unplaced(ColumnSet.of(s="String")) == ("s",)

    def test_a_subsumed_part_is_caught(self) -> None:
        """A part an earlier part covers can never hold anything."""
        shadowed = Split((NameTest(Always()), OfType(NUMERIC)))
        assert shadowed.dead_parts(UNIVERSE) == (1,)

    def test_the_lawful_splits_pass_both(self) -> None:
        """The battery's splits place everything and starve no part."""
        for split in (NUMERIC_REST, PREFIX_NUMERIC_REST):
            assert split.places_everything(UNIVERSE)
            assert split.dead_parts(UNIVERSE) == ()


class TestTheSplitSettlesEveryCaller:
    """The measured counts, recomputed from one split of one column set."""

    def test_every_part_of_every_shape_is_exact(self) -> None:
        """Thirty-two caller shapes, every part settled exactly.

        The single family requirement settles eleven of these shapes;
        the split settles all thirty-two, because each bound part's
        types sit wholly inside or wholly outside the family.
        """
        for shape in TypeTestPrecision(MIXED_COLUMNS, NUMERIC).shapes():
            numeric_part, rest = NUMERIC_REST.bound(shape)
            assert self.exact(numeric_part)
            assert self.exact(rest)

    def exact(self, part: ColumnSet) -> bool:
        """Answer whether the family selector settles one part exactly.

        Args:
            part: One bound part's columns.

        Returns:
            True when the selector picks all of it or none of it.

        Example:
            >>> TestTheSplitSettlesEveryCaller().exact(ColumnSet.of(n="Int64"))
            True
        """
        precision = TypeTestPrecision(part.entries, NUMERIC)
        return precision.is_exact(part)

    def test_the_precision_counts_come_from_the_one_split(self) -> None:
        """The per-part counts of the documents, derived, not assembled.

        Splitting the five mixed columns gives the numeric part four
        shapes and the rest eight, and neither part is ever imprecise.
        The documents' 0-of-4 and 0-of-8 are these numbers, computed
        from one split of the one column set the twenty-one came from.
        """
        numeric_part, rest = NUMERIC_REST.bound(ColumnSet.holding(MIXED_COLUMNS))
        for_numeric = TypeTestPrecision(numeric_part.entries, NUMERIC)
        for_rest = TypeTestPrecision(rest.entries, NUMERIC)
        assert (for_numeric.total(), for_numeric.imprecise()) == (4, 0)
        assert (for_rest.total(), for_rest.imprecise()) == (8, 0)
