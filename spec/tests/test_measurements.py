"""Every count the specification documents quote, recomputed.

A document that quotes a number without a way to recompute it is asking
to be believed. Each test here pins one such number, so a document and
its evidence cannot drift apart.

The universe for the precision measurements is five extra columns, of
which two are numeric and three are not. That choice is what makes the
sub-counts four and eight, and both appear in the documents.
"""

from __future__ import annotations

from spec.tests.measurements import TypeTestPrecision
from spec.tests.model import NUMERIC

MIXED = TypeTestPrecision(
    (
        ("n1", "Int64"),
        ("n2", "Float64"),
        ("s1", "String"),
        ("s2", "Date"),
        ("s3", "Boolean"),
    ),
    NUMERIC,
)

ALL_NUMERIC = TypeTestPrecision((("n1", "Int64"), ("n2", "Float64")), NUMERIC)

NO_NUMERIC = TypeTestPrecision(
    (("s1", "String"), ("s2", "Date"), ("s3", "Boolean")),
    NUMERIC,
)


def test_caller_shapes_number_thirty_two() -> None:
    """Five extra columns give thirty-two possible callers."""
    assert MIXED.total() == 32


def test_a_family_requirement_answers_eleven_of_thirty_two() -> None:
    """A single remainder with a family requirement settles eleven shapes.

    Four shapes hold only numeric columns and eight hold only
    non-numeric ones. The empty shape is in both counts, so eleven
    distinct shapes are settled.
    """
    assert MIXED.exact() == 11


def test_a_family_requirement_leaves_twenty_one_of_thirty_two() -> None:
    """The remaining twenty-one shapes cannot be settled.

    This is the complement of the eleven above. The specification once
    quoted twenty-eight here, which cannot hold beside eleven, since the
    two count complementary halves of the same thirty-two.
    """
    assert MIXED.imprecise() == 21
    assert MIXED.exact() + MIXED.imprecise() == MIXED.total()


def test_a_part_of_numeric_types_is_never_imprecise() -> None:
    """A part declared to hold only numeric columns answers all four shapes."""
    assert ALL_NUMERIC.total() == 4
    assert ALL_NUMERIC.imprecise() == 0


def test_a_part_of_non_numeric_types_is_never_imprecise() -> None:
    """A part declared to hold no numeric column answers all eight shapes."""
    assert NO_NUMERIC.total() == 8
    assert NO_NUMERIC.imprecise() == 0


def test_a_split_answers_every_shape() -> None:
    """Splitting by data type settles every one of the thirty-two shapes.

    A split sends the numeric columns to one part and the rest to the
    other. Each part is then exact on its own, and the two parts
    together cover every caller.
    """
    assert ALL_NUMERIC.imprecise() + NO_NUMERIC.imprecise() == 0
    assert MIXED.total() == 32
