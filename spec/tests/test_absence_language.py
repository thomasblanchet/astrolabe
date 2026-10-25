"""Closure of the record language under every effect, checked by exhaustion.

The specification records two things about a remainder: an absence rule
and a type set. It states that both stay expressible after any effect,
and it derives that claim for one effect out of seven.

These tests check the claim for all of them, and for both halves of the
record. The property is soundness of the transformation:

    if a record describes a column set,
    then the transformed record describes the transformed column set.

A failure means an effect's rule claims a fact that does not hold, which
is the one direction that is never safe.
"""

from __future__ import annotations

from itertools import product

import pytest

from spec.tests.model import (
    NUMERIC,
    TEMPORAL,
    Absence,
    AfterStripping,
    Always,
    And,
    BeforeStripping,
    Both,
    ColumnSet,
    Contains,
    Effect,
    Emptied,
    EndsWith,
    Matching,
    Never,
    NameTest,
    Not,
    OfType,
    OneOf,
    Or,
    PrefixedBy,
    Record,
    RegularLanguage,
    Retyped,
    SelAnd,
    SelNot,
    SelOr,
    StartsWith,
    SuffixedBy,
    TypeSet,
    Unchanged,
    Where,
)
from spec.tests.universe import Universe

UNIVERSE = Universe(("a", "b", "a_p"), ("Int64", "String", "Date"))

RULES: tuple[Absence, ...] = (
    Never(),
    Always(),
    OneOf(frozenset({"a"})),
    OneOf(frozenset({"a", "b"})),
    StartsWith("a"),
    EndsWith("_p"),
    Contains("_"),
    Not(OneOf(frozenset({"a"}))),
    And(StartsWith("a"), Not(EndsWith("_p"))),
    Or(OneOf(frozenset({"b"})), EndsWith("_p")),
    AfterStripping("_p", OneOf(frozenset({"a"}))),
    BeforeStripping("s_", StartsWith("a")),
    Or(Not(EndsWith("_p")), AfterStripping("_p", StartsWith("a"))),
    Matching.of(RegularLanguage().literal("a")),
)

TYPE_SETS: tuple[TypeSet, ...] = (
    TypeSet(frozenset({"Int64", "String", "Date"})),
    TypeSet(frozenset({"Int64"})),
    TypeSet(frozenset({"String", "Date"})),
    TypeSet(frozenset()),
)

EFFECTS: dict[str, Effect] = {
    "unchanged": Unchanged(),
    "emptied": Emptied(),
    "retyped": Retyped("UInt32"),
    "suffixed": SuffixedBy("_p"),
    "prefixed": PrefixedBy("s_"),
    "where-name-prefix": Where(NameTest(StartsWith("a"))),
    "where-name-not-suffix": Where(NameTest(Not(EndsWith("_p")))),
    "where-numeric": Where(OfType(NUMERIC)),
    "where-not-numeric": Where(SelNot(OfType(NUMERIC))),
    "where-temporal": Where(OfType(TEMPORAL)),
    "where-numeric-and-name": Where(SelAnd(OfType(NUMERIC), NameTest(StartsWith("a")))),
    "where-numeric-or-name": Where(SelOr(OfType(NUMERIC), NameTest(EndsWith("_p")))),
    "where-minus-a-name": Where(SelNot(NameTest(OneOf(frozenset({"a"}))))),
    "where-not-numeric-name": Where(SelNot(SelAnd(OfType(NUMERIC), NameTest(StartsWith("a"))))),
}

RECORDS: tuple[Record, ...] = tuple(Record(rule, types) for rule in RULES for types in TYPE_SETS)


def test_the_universe_holds_sixty_four_column_sets() -> None:
    """Three names over three data types give sixty-four column sets.

    The T-2 note in `OPERATIONS.md` quotes this count, so it is
    computed here rather than remembered.
    """
    assert sum(1 for _ in UNIVERSE.column_sets()) == 64


def described_by(record: Record) -> list[ColumnSet]:
    """List every column set in the universe that a record describes.

    Args:
        record: The record whose possible column sets are wanted.

    Returns:
        Each column set of `UNIVERSE` the record does not rule out.
    """
    return [c for c in UNIVERSE.column_sets() if record.describes(c)]


@pytest.mark.parametrize("effect", EFFECTS.values(), ids=EFFECTS.keys())
@pytest.mark.parametrize("record", RECORDS, ids=range(len(RECORDS)))
def test_effect_transformation_is_sound(record: Record, effect: Effect) -> None:
    """Every effect's record rule holds of the column set the effect produces.

    This is the closure property, stated as soundness and checked for
    both halves of the record at once, since `Record.describes` reads
    the absence rule and the type set together.
    """
    transformed = effect.transform(record)
    for columns in described_by(record):
        message = f"{effect} on {columns} escapes the record it was given"
        assert transformed.describes(effect.apply(columns)), message


def assert_both_is_sound(left: Record, right: Record) -> None:
    """Check the combining rule against every pair the two records allow.

    Args:
        left: What is known about one operand.
        right: What is known about the other.
    """
    combined = Both().transform(left, right)
    for one, other in product(described_by(left), described_by(right)):
        assert combined.describes(Both().apply(one, other))


@pytest.mark.parametrize("left", RECORDS, ids=range(len(RECORDS)))
def test_both_transformation_is_sound(left: Record) -> None:
    """Combining two remainders keeps every fact both operands prove."""
    for right in RECORDS:
        assert_both_is_sound(left, right)


@pytest.mark.parametrize("effect", EFFECTS.values(), ids=EFFECTS.keys())
def test_transformation_records_no_impossible_type(effect: Effect) -> None:
    """No effect permits a data type that cannot actually occur.

    A type set that grew would be a fact invented rather than kept, and
    the specification allows a record to lose a fact but never to gain
    one that does not hold.
    """
    record = Record(Never(), TypeSet(frozenset({"Int64", "String", "Date"})))
    transformed = effect.transform(record)
    for columns in described_by(record):
        produced = effect.apply(columns).type_set().members
        assert produced <= transformed.types.members
