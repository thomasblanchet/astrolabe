"""Every pathological declaration, fed to the executable rules.

The declaration grammar's corners were catalogued adversarially: for
each one, either a well-formedness rule refuses it, the fit relation
decides it, or the model shows it genuinely undecided, which is what
sends it to a design decision. This sweep pins each corner's verdict,
so a wording change that silently moves one fails a test.
"""

from __future__ import annotations


from spec.tests.fit_model import Read, Row, SliceDecl, WellFormedness
from spec.tests.table_model import QUOTIENT

NUMERIC = frozenset({"Int32", "Int64", "Float32", "Float64"})
FLOATS = frozenset({"Float32", "Float64"})
EVERY = frozenset(QUOTIENT)


class TestRefusedCorners:
    """Corners a well-formedness rule decides at the declaration."""

    def test_same_under_several_reads_is_refused(self) -> None:
        """A bare `same`, and a `wrap` of it, need exactly one read."""
        two_reads = (Read("a", NUMERIC), Read("b", NUMERIC))
        for outcome in ("same", "wrap:List"):
            decl = SliceDecl(two_reads, (Row((NUMERIC, NUMERIC), outcome),), "o", "elementwise")
            problems = WellFormedness().problems(decl)
            assert any(problem.startswith("same") for problem in problems), outcome

    def test_same_naming_a_missing_read_is_refused(self) -> None:
        """`same(n)` must name one of the reads."""
        decl = SliceDecl(
            (Read("a", NUMERIC),),
            (Row((NUMERIC,), "same:zz"),),
            None,
            "elementwise",
        )
        expected = "same(name) names a column the declaration does not read"
        assert expected in WellFormedness().problems(decl)

    def test_overlapping_rows_under_a_product_reading_are_refused(self) -> None:
        """Row disjointness is over cell tuples, the product reading.

        A row over `(EveryDType, Int64)` and a row over
        `(Int64, Int64)` claim the cell `(Int64, Int64)` twice; a row
        over `(Int64, String)` beside the first is disjoint, because
        no tuple lands in both.
        """
        reads = (Read("a", EVERY), Read("b", frozenset({"Int64", "String"})))
        colliding = SliceDecl(
            reads,
            (
                Row((EVERY, frozenset({"Int64"})), "const:Boolean"),
                Row((frozenset({"Int64"}), frozenset({"Int64"})), "const:String"),
            ),
            "o",
            "elementwise",
        )
        expected = "two rows cover the same cell. Make the rows disjoint"
        assert expected in WellFormedness().problems(colliding)
        disjoint = SliceDecl(
            reads,
            (
                Row((EVERY, frozenset({"Int64"})), "const:Boolean"),
                Row((frozenset({"Int64"}), frozenset({"String"})), "const:String"),
            ),
            "o",
            "elementwise",
        )
        assert WellFormedness().problems(disjoint) == []

    def test_a_duplicated_read_is_refused(self) -> None:
        """A schema class beside an explicit column of one name collides.

        A schema-class read expands to one named read per declared
        column, so the duplicate lands on rule 6 with no extra rule.
        """
        decl = SliceDecl(
            (Read("price", FLOATS), Read("price", frozenset({"Float64"}))),
            (),
            "o",
            "elementwise",
        )
        assert "a column is read twice. Read each column once" in WellFormedness().problems(decl)

    def test_an_empty_region_is_refused(self) -> None:
        """A region holding nothing declares an unusable slice."""
        decl = SliceDecl((Read("a", frozenset()),), (), None, "elementwise")
        expected = "a read has an empty region. Give it at least one data type"
        assert expected in WellFormedness().problems(decl)

    def test_a_second_blank_is_refused(self) -> None:
        """Two blanks would leave the binding ambiguous."""
        decl = SliceDecl((Read(None, FLOATS), Read(None, FLOATS)), (), "o", "elementwise")
        expected = "the declaration has more than one some_column read. Keep one"
        assert expected in WellFormedness().problems(decl)

    def test_a_ragged_row_is_refused(self) -> None:
        """An empty `row()` beside real reads breaks rule 2."""
        decl = SliceDecl((Read("a", NUMERIC),), (Row((), "const:Boolean"),), "o", "elementwise")
        problems = WellFormedness().problems(decl)
        uneven = "a row reads a different number of columns"
        assert any(problem.startswith(uneven) for problem in problems)

    def test_a_literal_with_a_same_outcome_is_refused(self) -> None:
        """`typed_lit(gives=same)` has no read for `same` to return."""
        decl = SliceDecl((), (Row((), "same"),), "value", "literal")
        problems = WellFormedness().problems(decl)
        assert any(problem.startswith("same") for problem in problems)


class TestDecidedCorners:
    """Corners the executable rules decide without a new rule."""

    def test_uncheckedness_is_a_reads_property_never_a_rows(self) -> None:
        """Rows cannot disagree about a read's two ends.

        The model factors `Unchecked` onto the read, which every row
        shares, so the corner asking how disjointness treats a row
        with an `Unchecked` region dissolves: rows partition the
        checked coordinates only.
        """
        decl = SliceDecl(
            (Read("a", frozenset(), unchecked=True), Read("b", NUMERIC)),
            (
                Row((frozenset(), frozenset({"Int32", "Int64"})), "const:Boolean"),
                Row((frozenset(), frozenset({"Float32", "Float64"})), "const:String"),
            ),
            "o",
            "elementwise",
        )
        assert WellFormedness().problems(decl) == []

    def test_the_blank_beside_named_reads_is_refused(self) -> None:
        """The contradiction the blank rule closes at the declaration.

        Before the rule, this declaration passed every check and no
        argument could fit it: the blank's binding demands a
        one-column argument, where the named read demands more. The
        rule now names the problem where it is written.
        """
        decl = SliceDecl(
            (Read("q", NUMERIC), Read(None, FLOATS)),
            (Row((NUMERIC, FLOATS), "const:Boolean"),),
            "o",
            "elementwise",
        )
        expected = "a some_column read stands beside another read. Declare it alone"
        assert expected in WellFormedness().problems(decl)

    def test_a_renamed_blank_is_refused(self) -> None:
        """`named=` over a blank contradicts the binding condition."""
        decl = SliceDecl(
            (Read(None, FLOATS),),
            (Row((FLOATS,), "same"),),
            "z",
            "elementwise",
        )
        problems = WellFormedness().problems(decl)
        assert "named= renames a some_column read. Delete named=" in problems

    def test_an_axis_outcome_on_an_unchecked_read_lifts_to_the_marker(self) -> None:
        """A `same` over an `Unchecked` read evaluates to the marker.

        Before the rule the model crashed here; the lifting is the
        answer: the read pins no coordinate, so the axis outcome
        gives `Unchecked` back, which is what `column(n, Unchecked)`
        declares, and a wrap of it collapses to the marker whole.
        """
        decl = SliceDecl(
            (Read("x", frozenset(), unchecked=True),),
            (Row((frozenset(),), "same"),),
            None,
            "elementwise",
        )
        assert WellFormedness().problems(decl) == []
        assert decl.outcome_at(decl.rows[0], {}) == "Unchecked"
        wrapping = SliceDecl(
            (Read("x", frozenset(), unchecked=True),),
            (Row((frozenset(),), "wrap:List"),),
            None,
            "elementwise",
        )
        assert wrapping.outcome_at(wrapping.rows[0], {}) == "Unchecked"
