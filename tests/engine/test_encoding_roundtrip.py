"""Check that every remainder the engine can build reads back unchanged from its mypy type.

The remainder of a frame is the set of its columns that its type does
not name. The engine holds a remainder in a canonical form: a base,
such as a declared variable, with a filter, a suffix, a prefix and a
new data type applied to it. Two remainders are the same exactly when
their canonical forms are equal. The codec writes that form as a mypy
type, and mypy then compares two remainders by their written types.

That comparison is right only if equal forms give equal types, and
if reading a written type gives back the form. These tests check both
over every expression of the bounded space that the parity tests in
`test_remainder_parity` build. Hand-built forms cover the shapes that
this space does not reach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.test.typefixture import TypeFixture

from astrolabe.remainder import Suffixed as SuffixedMarker
from astrolabe._hooks.encoding.marker_minter import MarkerMinter
from astrolabe._hooks.encoding.remainder_codec import RemainderCodec
from astrolabe._hooks.calls.dtype_constructor_readers import (
    CheckerLiteralTypeBuilder,
)
from astrolabe._hooks.remainder import absence, canonical, selector, type_set
from tests.engine.test_remainder_parity import expression_pairs
from tests.support import FakeApi, FakePlugin, NamedDeclarations

if TYPE_CHECKING:
    from mypy.types import Type as MypyType


def fresh_codec() -> RemainderCodec:
    """Build a remainder codec that reads the classes `R` and `S` as declared remainders.

    The expressions of the bounded space use these two variables.

    Returns:
        The codec, over a fake plugin and a fake API.

    Example:
        >>> fresh_codec().decoded(None) is None
        True
    """
    fixture = TypeFixture()
    minter = MarkerMinter(FakePlugin(fixture), CheckerLiteralTypeBuilder(), FakeApi(fixture))
    return RemainderCodec(minter, NamedDeclarations(frozenset({"R", "S"})))


class TestTheSpaceRoundTrips:
    """Each expression of the bounded space, in canonical form, survives the codec."""

    def test_every_normalised_form_round_trips(self) -> None:
        """Writing a canonical form and reading it back gives the same form."""
        codec = fresh_codec()
        for _, e_expr in expression_pairs():
            form = e_expr.normalise()
            self.assert_round_trip(codec, form)

    def assert_round_trip(self, codec: RemainderCodec, form: canonical.Canonical) -> None:
        """Write one form, read it back, and check that it is the same form.

        A form with no base writes as no type at all, because an empty
        remainder carries nothing.

        Args:
            codec: The codec under test.
            form: The canonical form.
        """
        spelled = codec.encoded(form)
        if spelled is None:
            assert isinstance(form.base, canonical.Nothing), form
            return
        assert codec.decoded(spelled) == form, form

    def test_equal_forms_spell_identically(self) -> None:
        """Two equal canonical forms give the same written type."""
        codec = fresh_codec()
        spellings: dict[str, str] = {}
        for _, e_expr in expression_pairs():
            form = e_expr.normalise()
            self.record(spellings, codec.encoded(form), form)

    def record(
        self,
        spellings: dict[str, str],
        spelled: MypyType | None,
        form: canonical.Canonical,
    ) -> None:
        """Check that one written type matches the type of each equal form seen before.

        Args:
            spellings: The text of each written type so far, keyed by the
                sort key of its form. This method adds `form` to it.
            spelled: The written type of `form`, or None for an empty
                remainder.
            form: The canonical form.
        """
        key = form.sort_key()
        text = str(spelled)
        if key in spellings:
            assert spellings[key] == text, form
        spellings[key] = text


class TestHandBuiltShapes:
    """Forms that the bounded space does not build also survive the codec."""

    def test_a_family_filter_round_trips(self) -> None:
        """A filter that keeps a family of data types survives."""
        keep = selector.OfType(type_set.Family.numeric())
        form = canonical.Canonical(canonical.Variable("R"), where=keep)
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert codec.decoded(spelled) == form

    def test_a_negated_family_filter_spells_except(self) -> None:
        """A filter that drops a family of data types is written `Except`, and survives."""
        drop = selector.SelNot(selector.OfType(type_set.Family.temporal()))
        form = canonical.Canonical(canonical.Variable("R"), where=drop)
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert "Except[" in str(spelled)
        assert codec.decoded(spelled) == form

    def test_a_finite_removal_spells_minus(self) -> None:
        """A filter that drops named columns is written `Minus`."""
        gone = selector.SelNot(selector.NameTest(absence.OneOf(frozenset({"z", "a"}))))
        form = canonical.Canonical(canonical.Variable("R"), where=gone)
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert "Minus[" in str(spelled)
        decoded = codec.decoded(spelled)
        assert decoded is not None
        assert decoded.where == gone

    def test_a_conjunction_filter_round_trips(self) -> None:
        """A filter that joins two tests with `and` survives."""
        both = selector.SelAnd(
            selector.OfType(type_set.Family.numeric()),
            selector.NameTest(absence.Matching.of(absence.StartsWith("p").language())),
        )
        form = canonical.Canonical(canonical.Variable("R"), where=both)
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert codec.decoded(spelled) == form

    def test_every_slot_at_once_round_trips(self) -> None:
        """A form with a filter, a suffix, a prefix and a new data type survives."""
        form = canonical.Canonical(
            canonical.Variable("R"),
            where=selector.SelNot(selector.NameTest(absence.OneOf(frozenset({"z"})))),
            suffix="_p",
            prefix="s_",
            retype="Unchecked",
        )
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert codec.decoded(spelled) == form

    def test_a_union_of_forms_with_effects_round_trips(self) -> None:
        """A union whose parts each carry a change survives.

        A union in canonical form carries no change of its own. A change
        applied to a union moves into each of its parts.
        """
        parts = (
            canonical.Canonical(canonical.Variable("R"), suffix="_p"),
            canonical.Canonical(canonical.Variable("S"), suffix="_p"),
        )
        form = canonical.Canonical.of(canonical.Union(parts))
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert codec.decoded(spelled) == form

    def test_a_three_part_union_round_trips(self) -> None:
        """A union of three parts is written as nested pairs, and reads back as one union."""
        parts = (
            canonical.Canonical(canonical.Variable("R"), suffix="_p"),
            canonical.Canonical.of(canonical.Variable("R")),
            canonical.Canonical.of(canonical.Variable("S")),
        )
        ordered = tuple(sorted(parts, key=canonical.Canonical.sort_key))
        form = canonical.Canonical(canonical.Union(ordered))
        codec = fresh_codec()
        spelled = codec.encoded(form)
        assert spelled is not None
        assert str(spelled).count("Both[") == 2
        assert codec.decoded(spelled) == form

    def test_a_literal_under_an_affix_reads_as_no_remainder(self) -> None:
        """A suffix around a literal, in place of a remainder, does not read."""
        codec = fresh_codec()
        minter = MarkerMinter(
            FakePlugin(TypeFixture()),
            CheckerLiteralTypeBuilder(),
            FakeApi(TypeFixture()),
        )
        hollow = minter.marker(SuffixedMarker, [minter.string("x"), minter.string("_p")])
        assert hollow is not None
        assert codec.decoded(hollow) is None

    def test_a_wrong_nesting_reads_as_no_remainder(self) -> None:
        """A suffix around a new data type is in the wrong order, and does not read."""
        codec = fresh_codec()
        inner = codec.encoded(canonical.Canonical(canonical.Variable("R"), retype="UInt32"))
        assert inner is not None
        minter = MarkerMinter(
            FakePlugin(TypeFixture()),
            CheckerLiteralTypeBuilder(),
            FakeApi(TypeFixture()),
        )
        twisted = minter.marker(SuffixedMarker, [inner, minter.string("_p")])
        assert twisted is not None
        assert codec.decoded(twisted) is None
