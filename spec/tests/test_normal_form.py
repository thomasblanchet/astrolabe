"""The canonical form exists, is unique, and preserves meaning.

The specification once described normalisation as rewriting continued
until no rule applies, and called that fixed point the normal form. That
description owes two arguments it never gave: that the rewriting stops,
and that it stops in the same place whichever order the rules fire in.

`normal_form.py` avoids both debts by computing the canonical form as a
function defined by structural induction. Existence and uniqueness are
then immediate, and the only claim left to check is the one that
matters:

    the canonical form denotes what the expression denotes.

These tests check that by exhaustion over a bounded space of
expressions and every binding of a small universe. They also pin the
one direction that fails, so a reader is not left believing an
equivalence the system does not decide.
"""

from __future__ import annotations


import pytest

from spec.tests.measurements import CanonicalPrecision, ConfirmedResidue
from spec.tests.model import (
    NUMERIC,
    AbsenceEntailment,
    ColumnSet,
    EndsWith,
    NameTest,
    Matching,
    Never,
    Or,
    OfType,
    RegularLanguage,
    OneOf,
    SelNot,
    StartsWith,
)
from spec.tests.normal_form import (
    Canonical,
    ExpressionSpace,
    Combined,
    Empty,
    Expr,
    Filter,
    Prefix,
    Ref,
    Retype,
    Suffix,
    Union,
    Variable,
)
from spec.tests.universe import Universe
from astrolabe._hooks.remainder.name_language import NameSet


UNIVERSE = Universe(("a", "ab"), ("Int64", "String"))
WIDE_UNIVERSE = Universe(("", "a", "ab", "ax"), ("Int64", "String"))
WIDER_UNIVERSE = Universe(("", "a", "b", "ab", "xb", "axb"), ("Int64",))
VARIABLES = ("R", "S")

SELECTORS = (
    NameTest(StartsWith("a")),
    NameTest(EndsWith("b")),
    NameTest(OneOf(frozenset({"a"}))),
    OfType(NUMERIC),
    SelNot(OfType(NUMERIC)),
)


SPACE = ExpressionSpace(VARIABLES, SELECTORS)

DEPTH_TWO = SPACE.up_to_depth(2)
ALL_BINDINGS = SPACE.bindings(UNIVERSE)


def test_the_expression_space_is_large_enough_to_be_evidence() -> None:
    """The exhaustive checks below run over a space worth calling exhaustive."""
    assert len(DEPTH_TWO) > 100
    assert len(ALL_BINDINGS) == 81


@pytest.mark.parametrize("index", range(len(DEPTH_TWO)))
def test_the_canonical_form_denotes_what_the_expression_denotes(index: int) -> None:
    """Normalising an expression never changes what it means.

    This is the one property the whole decision procedure rests on. If
    it held only sometimes, two expressions could be declared the same
    type while denoting different frames. The sweep is the full space:
    every expression of depth two, every binding.
    """
    expression = DEPTH_TWO[index]
    form = expression.normalise()
    for binding in ALL_BINDINGS:
        message = f"{expression} and its canonical form disagree under {binding}"
        assert expression.denote(binding) == form.denote(binding), message


@pytest.mark.parametrize("index", range(0, len(DEPTH_TWO), 11))
def test_normalising_is_deterministic(index: int) -> None:
    """The canonical form is a value of a function, so it never varies."""
    expression = DEPTH_TWO[index]
    assert expression.normalise() == expression.normalise()


@pytest.mark.parametrize("index", range(0, len(DEPTH_TWO), 11))
def test_the_canonical_form_has_the_one_declared_shape(index: int) -> None:
    """Every canonical form fills the same four slots over a base.

    There is no nesting to inspect and no rewriting left to do, which
    is what makes comparing two of them a comparison of values.
    """
    form = DEPTH_TWO[index].normalise()
    assert isinstance(form, Canonical)
    assert isinstance(form.suffix, str)
    assert isinstance(form.prefix, str)


@pytest.mark.parametrize("index", range(len(DEPTH_TWO)))
def test_denotation_is_additive_in_each_variable(index: int) -> None:
    """The columns of an expression split into what each variable gives (`T-24`).

    A call can bind `R` to the known columns of a closed caller and `S`
    to the remainder of an open one. The return type then splits into
    the columns that `R` gives and the remainder that `S` gives, and the
    split is exact because each effect acts on one column alone. The
    sweep binds both variables, then empties each in turn.
    """
    expression = DEPTH_TWO[index]
    nothing = ColumnSet(())
    for binding in ALL_BINDINGS:
        closed_half = expression.denote({**binding, "S": nothing})
        open_half = expression.denote({**binding, "R": nothing})
        joined = ColumnSet.holding((*closed_half.entries, *open_half.entries))
        assert expression.denote(binding) == joined, f"{expression} under {binding}"


def test_equal_canonical_forms_mean_equal_denotations() -> None:
    """Two expressions with one canonical form denote one column set.

    This is the direction the checker relies on. It follows from the
    property above, and it is checked here directly over every pair.
    """
    seen: dict[Canonical, Expr] = {}
    for expression in DEPTH_TWO:
        form = expression.normalise()
        other = seen.setdefault(form, expression)
        for binding in ALL_BINDINGS:
            assert expression.denote(binding) == other.denote(binding)


class TestUnionForgetsOrderAndGrouping:
    """A union's meaning has neither an order nor a grouping, so nor does its form."""

    def test_the_order_two_operands_were_written_in_is_forgotten(self) -> None:
        """Combining in either order gives one canonical form."""
        expected = Combined(Ref("S"), Ref("R")).normalise()
        assert Combined(Ref("R"), Ref("S")).normalise() == expected

    def test_the_grouping_of_three_operands_is_forgotten(self) -> None:
        """Nesting either way gives one canonical form."""
        left = Combined(Combined(Ref("R"), Ref("S")), Ref("T"))
        right = Combined(Ref("R"), Combined(Ref("S"), Ref("T")))
        assert left.normalise() == right.normalise()

    def test_a_union_flattens_and_an_effect_reaches_only_its_operands(self) -> None:
        """Nesting flattens, and an effect over a union folds into its parts.

        `both(both(A, B), C)` is one union of three, because the two
        groupings mean the same thing. `both(suffix(both(A, B)), C)` is
        a union of three too: the suffix folds into `A` and `B`, and
        `C` keeps its names, because the suffix never applied to it.
        """
        bare = Combined(Combined(Ref("R"), Ref("S")), Ref("T")).normalise().base
        assert isinstance(bare, Union)
        assert len(bare.parts) == 3

        inner = Suffix(Combined(Ref("R"), Ref("S")), "_p")
        wrapped = Combined(inner, Ref("T")).normalise().base
        assert isinstance(wrapped, Union)
        suffixes = {part.base: part.suffix for part in wrapped.parts}
        assert suffixes == {Variable("R"): "_p", Variable("S"): "_p", Variable("T"): ""}

    def test_a_provably_empty_operand_is_dropped(self) -> None:
        """`both(empty(R), S)` and `S` denote one frame, so one form.

        The exhaustive space never builds a union over a wrapped
        operand, so this corner needs its own case: an emptied part
        contributes no columns under any assignment, and a form that
        kept it would split one type into two spellings.
        """
        assert Combined(Empty(Ref("R")), Ref("S")).normalise() == (Ref("S").normalise())

    def test_a_union_of_empties_is_the_empty_remainder(self) -> None:
        """Dropping every operand leaves no columns at all."""
        both_empty = Combined(Empty(Ref("R")), Empty(Ref("S"))).normalise()
        assert both_empty == Empty(Ref("R")).normalise()

    def test_the_sort_key_does_not_depend_on_set_ordering(self) -> None:
        """A key built from a repr would differ between processes.

        The union's operands are sorted by this key, so an unstable key
        would canonicalise the same expression two ways.
        """
        one = Filter(Ref("R"), NameTest(OneOf(frozenset({"b", "a"})))).normalise()
        other = Filter(Ref("R"), NameTest(OneOf(frozenset({"a", "b"})))).normalise()
        assert one.sort_key() == other.sort_key()
        assert "frozenset" not in one.sort_key()


class TestUnlikeEffectsCommute:
    """Two effects of different kinds give one type in either order.

    A user can write these in either order, and both spellings must be
    acceptable to the same function. A system that distinguished them
    would reject one of two programs that do the same thing.
    """

    def test_retyping_commutes_with_filtering(self) -> None:
        """Dropping a column then counting nulls is counting nulls then dropping."""
        selector = SelNot(NameTest(OneOf(frozenset({"a"}))))
        first = Retype(Filter(Ref("R"), selector), "UInt32")
        second = Filter(Retype(Ref("R"), "UInt32"), selector)
        assert first.normalise() == second.normalise()

    def test_retyping_commutes_with_a_rename(self) -> None:
        """Changing types disturbs no name, so it can move above one."""
        expected = Suffix(Retype(Ref("R"), "UInt32"), "_p").normalise()
        assert Retype(Suffix(Ref("R"), "_p"), "UInt32").normalise() == expected

    def test_a_prefix_and_a_suffix_do_not_interact(self) -> None:
        """The two ends of a name are independent."""
        expected = Suffix(Prefix(Ref("R"), "s_"), "_p").normalise()
        assert Prefix(Suffix(Ref("R"), "_p"), "s_").normalise() == expected


class TestLikeEffectsCombine:
    """Two effects of one kind are one effect, in the shorter spelling."""

    def test_two_filters_become_one(self) -> None:
        """Each filter keeps a column only if its own test passes."""
        form = Filter(Filter(Ref("R"), SELECTORS[0]), SELECTORS[1]).normalise()
        assert form.where is not None

    def test_two_suffixes_concatenate(self) -> None:
        """Each appends to whatever name it was given."""
        assert Suffix(Suffix(Ref("R"), "_a"), "_b").normalise().suffix == "_a_b"

    def test_two_name_removals_equal_one_removal_of_both(self) -> None:
        """`Minus[Minus[R, "a"], "b"]` and `Minus[R, {"a", "b"}]` are one type."""
        one_at_a_time = Filter(
            Filter(Ref("R"), SelNot(NameTest(OneOf(frozenset({"a"}))))),
            SelNot(NameTest(OneOf(frozenset({"b"})))),
        ).normalise()
        both_at_once = Filter(Ref("R"), SelNot(NameTest(OneOf(frozenset({"a", "b"}))))).normalise()
        assert one_at_a_time == both_at_once

    def test_a_mixed_filter_keeps_its_type_test(self) -> None:
        """Only a filter reading names alone collapses to one name test."""
        form = Filter(
            Filter(Ref("R"), OfType(NUMERIC)),
            SelNot(NameTest(OneOf(frozenset({"a"})))),
        ).normalise()
        assert form.where is not None
        assert not isinstance(form.where, NameTest)

    def test_the_outer_retype_wins(self) -> None:
        """The second replaces every data type the first set."""
        assert Retype(Retype(Ref("R"), "Int64"), "UInt32").normalise().retype == "UInt32"

    def test_emptying_absorbs_everything_below_it(self) -> None:
        """No earlier effect can matter once every column is gone."""
        expected = Empty(Ref("S")).normalise()
        assert Empty(Suffix(Filter(Ref("R"), SELECTORS[0]), "_p")).normalise() == expected


class TestUnlikeEffectsThatDoNotCommute:
    """Two orders that really do give different frames must stay different."""

    def test_removing_before_and_after_a_rename_differ(self) -> None:
        """Removing `a` before suffixing removes `a_p`; after, it removes nothing."""
        drop_a = SelNot(NameTest(OneOf(frozenset({"a"}))))
        before = Suffix(Filter(Ref("R"), drop_a), "_p")
        after = Filter(Suffix(Ref("R"), "_p"), drop_a)
        binding = {"R": ColumnSet.of(a="Int64"), "S": ColumnSet(())}
        assert before.denote(binding).names() == ()
        assert after.denote(binding).names() == ("a_p",)
        assert before.normalise() != after.normalise()


class TestAnEmptyAffixNormalizesAway:
    """`M-2`: a filter through an empty affix is the filter on the remainder itself."""

    @pytest.mark.parametrize(
        "inner",
        [Ref("R"), Suffix(Ref("R"), ""), Prefix(Ref("R"), "")],
        ids=["no affix", "empty suffix", "empty prefix"],
    )
    def test_the_filter_through_an_empty_affix_is_the_plain_filter(self, inner: Expr) -> None:
        """An empty suffix or prefix renames nothing, so the filter tests the names as they are."""
        plain = Filter(Ref("R"), NameTest(StartsWith("a"))).normalise()
        assert Filter(inner, NameTest(StartsWith("a"))).normalise() == plain


class TestDecidingEquality:
    """How much of denotational equality the canonical form decides.

    Comparing two canonical forms is sound: equal forms always mean
    equal denotations. The other direction is where the system loses
    programs, and the loss is smaller than a first look suggests.

    Every constructor of the selector language denotes a regular set of
    names, and there are finitely many data types, so containment
    between two selectors is decidable. The canonical form uses a cheap
    and exact form of that test over the named constructors, which is
    every selector a user can write. What it does not decide is the
    general case, which arises only for a selector the checker itself
    derived by translating through an affix.
    """

    def test_a_redundant_filter_is_recognised(self) -> None:
        """Filtering on a prefix twice, once more narrowly, gives one form.

        A name beginning with `ab` begins with `a`, so the wider test
        removes nothing the narrower one keeps. Recognising that is
        containment between two regular sets, and for two prefix tests
        it is a single string comparison.
        """
        narrow = Filter(Ref("R"), NameTest(StartsWith("ab")))
        widened = Filter(Filter(Ref("R"), NameTest(StartsWith("a"))), NameTest(StartsWith("ab")))
        for binding in ALL_BINDINGS:
            assert narrow.denote(binding) == widened.denote(binding)
        assert narrow.normalise() == widened.normalise()

    def test_a_filter_that_picks_everything_is_dropped(self) -> None:
        """A test every column passes is the identity, and is not recorded."""
        filtered = Filter(Ref("R"), NameTest(StartsWith("")))
        assert filtered.normalise() == Ref("R").normalise()

    def test_a_filter_that_picks_nothing_empties_the_remainder(self) -> None:
        """A test no column passes leaves no columns, whatever came before."""
        filtered = Filter(Suffix(Ref("R"), "_p"), NameTest(Never()))
        assert filtered.normalise() == Empty(Ref("S")).normalise()

    def test_a_conjunction_forgets_the_order_it_was_built_in(self) -> None:
        """Two filters applied in either order give one form."""
        first = Filter(Filter(Ref("R"), SELECTORS[0]), SELECTORS[1])
        second = Filter(Filter(Ref("R"), SELECTORS[1]), SELECTORS[0])
        assert first.normalise() == second.normalise()

    def test_a_type_test_resolved_after_a_retyping_collapses(self) -> None:
        """Once every column has one type, a type test picks all or none.

        Here it picks all, so the filter is dropped and the form is the
        retyping alone.
        """
        retyped = Retype(Ref("R"), "UInt32")
        filtered = Filter(retyped, SelNot(OfType(NUMERIC)))
        assert filtered.normalise() == retyped.normalise()


class TestTheConverseHolds:
    """The canonical form decides equality, in both directions.

    Comparing two canonical forms is sound: equal forms always mean
    equal denotations. It is also complete over the space measured
    here, because the filter slot is keyed by the columns a selector
    picks rather than by the shape it was written in, and that key is
    exact.

    The measurement runs in two passes. A coarse universe proposes
    every meaning the form appears to split, and a wider one confirms
    whether each proposal is a real equality. A proposal the wider pass
    rejects was never an equality at all.
    """

    RESIDUE = ConfirmedResidue(SPACE, WIDE_UNIVERSE, WIDER_UNIVERSE, 2)

    def test_the_space_is_large_enough_to_be_evidence(self) -> None:
        """The counts below are over this space, so it is pinned first."""
        assert len(DEPTH_TWO) == 666
        assert CanonicalPrecision(SPACE, WIDE_UNIVERSE, 2).meaning_classes() == 109

    def test_no_equality_is_left_undecided(self) -> None:
        """Every meaning the coarse pass split, the wider pass refutes.

        So over this space, two expressions have one canonical form
        exactly when they denote the same thing under every binding.
        """
        assert self.RESIDUE.count() == 0

    def test_the_coarse_pass_really_does_propose_candidates(self) -> None:
        """The confirmation step is doing work, not passing vacuously.

        A coarse universe cannot separate every pair, so it proposes
        equalities that a longer name refutes. Counting those keeps the
        result above from being an artefact of an empty candidate list.
        """
        assert len(self.RESIDUE.candidates()) > 0


class TestTheConverseHoldsOverUnionsOfEffects:
    """`T-10` over a space where a union can sit above an effect.

    The space adds each union of two expressions that apply at most
    one effect to a variable. So it holds both `suffix(both(A, B))` and
    `both(suffix(A), suffix(B))`, which denote the same column sets and
    now have one canonical form, because an effect folds into the parts
    of a union.
    """

    SPACE = ExpressionSpace(VARIABLES, SELECTORS, unions_over_effects=True)
    RESIDUE = ConfirmedResidue(SPACE, WIDE_UNIVERSE, WIDER_UNIVERSE, 2)

    def test_the_space_holds_unions_over_effects(self) -> None:
        """The space is the depth-two space plus 484 unions."""
        assert len(self.SPACE.up_to_depth(2)) == len(DEPTH_TWO) + 484

    def test_an_effect_over_a_union_is_the_union_of_the_effects(self) -> None:
        """The pair that separated the two forms now has one form."""
        over = Suffix(Combined(Ref("R"), Ref("S")), "b_").normalise()
        each = Combined(Suffix(Ref("R"), "b_"), Suffix(Ref("S"), "b_")).normalise()
        assert over == each

    def test_no_equality_is_left_undecided(self) -> None:
        """Every meaning the coarse pass split, the wider pass refutes."""
        assert self.RESIDUE.count() == 0


class TestWhatDecidingCosts:
    """Deciding exactly is not free, which is why cheap rules go first."""

    def test_a_cheap_rule_and_the_exact_test_agree(self) -> None:
        """A prefix comparison and an automaton comparison give one answer."""
        language = RegularLanguage()
        narrow, wide = StartsWith("ab"), StartsWith("a")
        assert AbsenceEntailment()(narrow, wide)
        assert language.contains(language.of(narrow), language.of(wide))

    def test_a_filter_through_an_affix_is_not_a_substitution(self) -> None:
        """`M-1`: a filter moved past a suffix tests the names that the suffix completes.

        Filtering on `startsWith("ab")` after appending `"b_"` also accepts
        the name `a`, because the suffix supplies the rest of the prefix. So
        the moved test is no named shape, and it lands in a `matching`.
        """
        form = Filter(Suffix(Ref("R"), "b_"), NameTest(StartsWith("ab"))).normalise()
        assert isinstance(form.where, NameTest)
        assert isinstance(form.where.rule, Matching)
        assert [form.where.rule.holds_of(name) for name in ("a", "abx", "ax")] == [
            True,
            True,
            False,
        ]


class TestATypeSpellsAMatchingByItsNames:
    """`D-11`: a type spells a `matching` by a text that only its names fix."""

    def test_two_rules_with_the_same_names_give_one_spelling(self) -> None:
        """A prefix rule and a union that holds the same names give one text."""
        one = StartsWith("a").language()
        both = Or(StartsWith("a"), StartsWith("ab")).language()
        assert NameSet(one).spelling() == NameSet(both).spelling() == "a.*"

    def test_a_rule_with_a_loop_keeps_its_automaton(self) -> None:
        """A rule whose automaton loops on part of the characters is spelled by its signature."""
        ends = NameSet(EndsWith("_x").language())
        assert ends.spelling() == ends.signature()

    def test_a_derived_rule_is_spelled_by_an_expression(self) -> None:
        """The names that the suffix `b_` completes to `startsWith("ab")` read as `a(b.*)?`."""
        form = Filter(Suffix(Ref("R"), "b_"), NameTest(StartsWith("ab"))).normalise()
        assert isinstance(form.where, NameTest)
        assert isinstance(form.where.rule, Matching)
        assert NameSet(form.where.rule.machine).spelling() == "a(b.*)?"
        assert NameSet.from_spelling("a(b.*)?").signature() == form.where.rule.signature
