"""Check that the remainder engine gives the same answers as the Python models in `spec/tests`.

The remainder of a frame is the set of its columns that its type does
not name. The engine reasons about it with a few kinds of value: a
rule on column names, a selector that picks columns by name and data
type, a change that an operation makes, and an expression that
applies changes to a declared remainder. The models in `spec/tests`
are the executable form of the design, and the engine is the fast
code that the plugin runs.

Each test builds the same value on both sides and checks that both
give the same answer. The inputs come from small fixed lists, so each
check can try every combination. The sets of data types are written
out in full. The engine derives its own list of data types from
polars, and that longer list must not change a comparison.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from spec.tests import model, normal_form
from spec.tests.split_model import Split as ModelSplit

from astrolabe._hooks.remainder import (
    absence,
    canonical,
    effects,
    entailment,
    record,
    selector,
    split,
    type_set,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

PROBE_NAMES = (
    "",
    "a",
    "b",
    "ab",
    "abc",
    "az",
    "tmp_x",
    "x_tmp_y",
    "price",
    "price_prev",
    "sec_country",
    "a_p",
)

KINDS = ("Int64", "Int32", "Float64", "String", "Boolean", "Date")

RulePair = tuple[model.Absence, absence.Absence]
SelectorPair = tuple[model.Selector, selector.Selector]
ExprPair = tuple[normal_form.Expr, canonical.Expr]


def base_rule_pairs() -> list[RulePair]:
    """Pair each basic name rule of the model with the same rule in the engine.

    A name rule says which column names a remainder cannot hold, for
    example every name that starts with `tmp_`. A basic rule is one that
    no other rule is built from.

    Returns:
        One pair of a model rule and an engine rule for each basic form.

    Example:
        >>> len(base_rule_pairs())
        8
    """
    return [
        (model.Never(), absence.Never()),
        (model.Always(), absence.Always()),
        (
            model.OneOf(frozenset({"a", "tmp_x"})),
            absence.OneOf(frozenset({"a", "tmp_x"})),
        ),
        (model.StartsWith("tmp_"), absence.StartsWith("tmp_")),
        (model.EndsWith("_prev"), absence.EndsWith("_prev")),
        (model.Contains("mp"), absence.Contains("mp")),
        (
            model.AfterStripping("_p", model.OneOf(frozenset({"a"}))),
            absence.AfterStripping("_p", absence.OneOf(frozenset({"a"}))),
        ),
        (
            model.BeforeStripping("s_", model.OneOf(frozenset({"a"}))),
            absence.BeforeStripping("s_", absence.OneOf(frozenset({"a"}))),
        ),
    ]


def rule_pairs() -> list[RulePair]:
    """List the name rules that the tests try, basic and combined.

    Returns:
        The basic pairs, the negation of each, a sample `And`, a sample
        `Or`, one rule as a `Matching`, the form that holds a rule as an
        automaton, and a second `Contains` that holds the part of the first.

    Example:
        >>> len(rule_pairs()) > 16
        True
    """
    pairs = list(base_rule_pairs())
    pairs += [(model.Not(m), absence.Not(e)) for m, e in base_rule_pairs()]
    (m_a, e_a), (m_b, e_b) = base_rule_pairs()[3], base_rule_pairs()[4]
    pairs.append((model.And(m_a, m_b), absence.And(e_a, e_b)))
    pairs.append((model.Or(m_a, m_b), absence.Or(e_a, e_b)))
    pairs.append((
        model.Matching.of(model.RegularLanguage().of(m_a)),
        absence.Matching.of(e_a.language()),
    ))
    pairs.append((model.Contains("tmp"), absence.Contains("tmp")))
    return pairs


class TestAbsenceParity:
    """Name rules give the same answers in the model and in the engine."""

    def test_holds_of_agrees_everywhere(self) -> None:
        """Each pair of rules gives the same answer for each sample name."""
        for m_rule, e_rule in rule_pairs():
            for name in PROBE_NAMES:
                assert m_rule.holds_of(name) == e_rule.holds_of(name), (
                    m_rule,
                    name,
                )

    def test_languages_accept_the_same_names(self) -> None:
        """The automaton of each pair of rules accepts the same sample names."""
        for m_rule, e_rule in rule_pairs():
            for name in PROBE_NAMES:
                model_accepts = bool(m_rule.language().accepts(name))
                assert model_accepts == e_rule.language().accepts(name)

    def test_entailment_agrees_on_every_pair(self) -> None:
        """Both sides agree, for each two rules, on whether one implies the other."""
        pairs = rule_pairs()
        for (m_left, e_left), (m_right, e_right) in product(pairs, pairs):
            expected = model.AbsenceEntailment()(m_left, m_right)
            assert entailment.AbsenceEntailment()(e_left, e_right) == expected

    def test_matching_signatures_key_equal_languages(self) -> None:
        """Two rules that describe the same names become the same `Matching`."""
        one = absence.StartsWith("a")
        both = absence.Or(absence.StartsWith("a"), absence.StartsWith("ab"))
        assert absence.Matching.of(one.language()) == absence.Matching.of(both.language())


def model_numeric() -> model.Family:
    """Give the family of numeric data types that the model uses.

    Returns:
        The family, which holds four data types.

    Example:
        >>> model_numeric().holds("Int64")
        True
    """
    return model.NUMERIC


def engine_numeric() -> type_set.Family:
    """Build an engine family with the same members as the numeric family of the model.

    The real engine family holds every numeric data type of polars, and
    the model family holds four. With the real family, a test compares
    two different sets of data types, and not the rules.

    Returns:
        The engine family.

    Example:
        >>> engine_numeric().holds("Int64")
        True
    """
    return type_set.Family("NumericType", model.NUMERIC.members)


def selector_pairs() -> list[SelectorPair]:
    """List the selectors that the tests try, as pairs of a model and an engine selector.

    A selector picks columns by name, by data type, or both, as
    `cs.starts_with("tmp_") & cs.numeric()` does.

    Returns:
        Tests on names, a test on data types, and each way to combine them.

    Example:
        >>> len(selector_pairs())
        8
    """
    m_name = model.NameTest(model.StartsWith("tmp_"))
    e_name = selector.NameTest(absence.StartsWith("tmp_"))
    m_type = model.OfType(model_numeric())
    e_type = selector.OfType(engine_numeric())
    return [
        (m_name, e_name),
        (m_type, e_type),
        (model.SelNot(m_type), selector.SelNot(e_type)),
        (model.SelNot(m_name), selector.SelNot(e_name)),
        (model.SelAnd(m_name, m_type), selector.SelAnd(e_name, e_type)),
        (model.SelOr(m_name, m_type), selector.SelOr(e_name, e_type)),
        (
            model.NameTest(model.OneOf(frozenset({"a", "price"}))),
            selector.NameTest(absence.OneOf(frozenset({"a", "price"}))),
        ),
        (
            model.SelAnd(m_type, model.SelNot(m_type)),
            selector.SelAnd(e_type, selector.SelNot(e_type)),
        ),
    ]


def sample_type_sets() -> list[tuple[model.TypeSet, type_set.TypeSet]]:
    """List sample sets of data types, each as a model set and an engine set.

    Returns:
        One pair for each sample, the full set of data types included.

    Example:
        >>> len(sample_type_sets())
        4
    """
    samples: list[frozenset[str]] = [
        frozenset({"Int64"}),
        frozenset({"String"}),
        frozenset({"Int64", "String", "Date"}),
        frozenset(model.EVERY_DTYPE),
    ]
    return [(model.TypeSet(s), type_set.TypeSet(s)) for s in samples]


class TestSelectorParity:
    """Selectors give the same answers in the model and in the engine."""

    def test_picks_agrees_on_every_column(self) -> None:
        """Each pair of selectors picks the same sample columns."""
        for m_sel, e_sel in selector_pairs():
            for name, dtype in product(PROBE_NAMES, KINDS):
                assert m_sel.picks(name, dtype) == e_sel.picks(name, dtype)

    def test_entailment_agrees_on_every_pair(self) -> None:
        """Both sides agree, for each two selectors, on whether one picks no more than the other.

        The pairs add a selector that picks every column, the negation of
        one that picks none, and a union of a name test with a selector
        that picks none.
        """
        (m_name, e_name), (m_type, e_type) = selector_pairs()[:2]
        m_none = model.SelAnd(m_type, model.SelNot(m_type))
        e_none = selector.SelAnd(e_type, selector.SelNot(e_type))
        pairs = [
            *selector_pairs(),
            (model.SelNot(m_none), selector.SelNot(e_none)),
            (model.SelOr(m_name, m_none), selector.SelOr(e_name, e_none)),
        ]
        for (m_left, e_left), (m_right, e_right) in product(pairs, pairs):
            expected = model.SelectorEntailment()(m_left, m_right)
            assert selector.SelectorEntailment()(e_left, e_right) == expected, (m_left, m_right)

    def test_certainly_picks_and_rejects_agree(self) -> None:
        """The names that a selector picks for sure, or refuses for sure, agree."""
        for m_sel, e_sel in selector_pairs():
            for m_types, e_types in sample_type_sets():
                self.assert_same_rule(
                    m_sel.certainly_picks(m_types),
                    e_sel.certainly_picks(e_types),
                )
                self.assert_same_rule(
                    m_sel.certainly_rejects(m_types),
                    e_sel.certainly_rejects(e_types),
                )

    def assert_same_rule(self, m_rule: model.Absence, e_rule: absence.Absence) -> None:
        """Check that two name rules give the same answer for each sample name.

        Args:
            m_rule: The rule of the model.
            e_rule: The rule of the engine.
        """
        for name in PROBE_NAMES:
            assert m_rule.holds_of(name) == e_rule.holds_of(name)

    def test_surviving_types_agree(self) -> None:
        """Both sides keep the same data types of each set after a selector picks."""
        for m_sel, e_sel in selector_pairs():
            for m_types, e_types in sample_type_sets():
                expected = m_sel.surviving_types(m_types).members
                assert e_sel.surviving_types(e_types).members == expected

    def test_translations_agree(self) -> None:
        """A selector moved through a suffix, a prefix or a new data type picks alike."""
        for m_sel, e_sel in selector_pairs():
            self.assert_same_selector(m_sel.through_suffix("_p"), e_sel.through_suffix("_p"))
            self.assert_same_selector(m_sel.through_prefix("s_"), e_sel.through_prefix("s_"))
            self.assert_same_selector(m_sel.after_retype("Int64"), e_sel.after_retype("Int64"))

    def assert_same_selector(self, m_sel: model.Selector, e_sel: selector.Selector) -> None:
        """Check that two selectors pick the same sample columns.

        Args:
            m_sel: The selector of the model.
            e_sel: The selector of the engine.
        """
        for name, dtype in product(PROBE_NAMES, KINDS):
            assert m_sel.picks(name, dtype) == e_sel.picks(name, dtype)

    def test_canonical_equality_agrees(self) -> None:
        """Two selectors have equal canonical forms on one side exactly when on the other."""
        pairs = selector_pairs()
        for (m_a, e_a), (m_b, e_b) in product(pairs, pairs):
            expected = m_a.canonical() == m_b.canonical()
            assert (e_a.canonical() == e_b.canonical()) == expected


def sample_column_sets() -> list[tuple[model.ColumnSet, record.ColumnSet]]:
    """List sample sets of real columns, each with a name and a data type.

    Returns:
        One pair of a model set and an engine set for each sample, the
        empty set included.

    Example:
        >>> len(sample_column_sets())
        3
    """
    samples: list[dict[str, str]] = [
        {"price": "Float64", "venue": "String"},
        {"tmp_x": "Int64", "a": "Date"},
        {},
    ]
    return [(model.ColumnSet.of(**s), record.ColumnSet.of(**s)) for s in samples]


def effect_pairs() -> list[tuple[model.Effect, effects.Effect]]:
    """List each change that an operation can make to a remainder, on both sides.

    The changes are: keep it, empty it, give every column a new data
    type, add a suffix or a prefix to each name, and keep only the
    columns that a selector picks.

    Returns:
        One pair of a model change and an engine change for each kind.

    Example:
        >>> len(effect_pairs())
        6
    """
    m_sel = model.NameTest(model.StartsWith("tmp_"))
    e_sel = selector.NameTest(absence.StartsWith("tmp_"))
    return [
        (model.Unchanged(), effects.Unchanged()),
        (model.Emptied(), effects.Emptied()),
        (model.Retyped("UInt32"), effects.Retyped("UInt32")),
        (model.SuffixedBy("_prev"), effects.SuffixedBy("_prev")),
        (model.PrefixedBy("sec_"), effects.PrefixedBy("sec_")),
        (model.Where(m_sel), effects.Where(e_sel)),
    ]


class TestEffectParity:
    """Each change gives the same columns, and the same knowledge, on both sides."""

    def test_apply_agrees_on_every_column_set(self) -> None:
        """Each change turns each sample set of columns into the same columns."""
        for m_eff, e_eff in effect_pairs():
            for m_cols, e_cols in sample_column_sets():
                assert m_eff.apply(m_cols).entries == e_eff.apply(e_cols).entries

    def test_transform_agrees_on_records(self) -> None:
        """Each change updates a record, what is known about a remainder, in the same way."""
        m_record = model.Record(model.OneOf(frozenset({"price"})), model.TypeSet(frozenset(KINDS)))
        e_record = record.Record(
            absence.OneOf(frozenset({"price"})),
            type_set.TypeSet(frozenset(KINDS)),
        )
        for m_eff, e_eff in effect_pairs():
            m_out, e_out = m_eff.transform(m_record), e_eff.transform(e_record)
            assert e_out.types.members == m_out.types.members
            for name in PROBE_NAMES:
                assert m_out.absence.holds_of(name) == e_out.absence.holds_of(name)

    def test_both_agrees(self) -> None:
        """The union of two remainders gives the same columns on both sides."""
        (m_a, e_a), (m_b, e_b) = sample_column_sets()[0], sample_column_sets()[1]
        assert model.Both().apply(m_a, m_b).entries == effects.Both().apply(e_a, e_b).entries


def expression_pairs() -> list[ExprPair]:
    """Build every remainder expression of a small fixed space, on both sides.

    A remainder expression applies changes to a declared remainder. The
    space starts from two declared remainders, `R` and `S`, and their
    union. It then applies up to two changes to each of these three.
    Other test modules reuse this space.

    Returns:
        One pair of a model expression and an engine expression for each.

    Example:
        >>> len(expression_pairs()) > 100
        True
    """
    layer: list[ExprPair] = [
        (normal_form.Ref("R"), canonical.Ref("R")),
        (normal_form.Ref("S"), canonical.Ref("S")),
        (
            normal_form.Combined(normal_form.Ref("R"), normal_form.Ref("S")),
            canonical.Combined(canonical.Ref("R"), canonical.Ref("S")),
        ),
    ]
    found = list(layer)
    for _ in range(2):
        layer = _next_layer(layer)
        found.extend(layer)
    return found


def _next_layer(layer: list[ExprPair]) -> list[ExprPair]:
    """Apply each change to each expression of one layer.

    Args:
        layer: The pairs of expressions.

    Returns:
        Each pair with one more change, for each change.

    Example:
        >>> len(_next_layer([(normal_form.Ref("R"), canonical.Ref("R"))]))
        6
    """
    return [wrapped for pair in layer for wrapped in _wrappings(pair)]


def _wrappings(pair: ExprPair) -> list[ExprPair]:
    """Apply each change to one pair of expressions, on both sides at once.

    Args:
        pair: A model expression and the matching engine expression.

    Returns:
        One new pair for each change.

    Example:
        >>> len(_wrappings((normal_form.Ref("R"), canonical.Ref("R"))))
        6
    """
    m_inner, e_inner = pair
    m_filter = model.NameTest(model.StartsWith("ab"))
    e_filter = selector.NameTest(absence.StartsWith("ab"))
    return [
        (normal_form.Keep(m_inner), canonical.Keep(e_inner)),
        (normal_form.Empty(m_inner), canonical.Empty(e_inner)),
        (
            normal_form.Retype(m_inner, "UInt32"),
            canonical.Retype(e_inner, "UInt32"),
        ),
        (normal_form.Suffix(m_inner, "b_"), canonical.Suffix(e_inner, "b_")),
        (normal_form.Prefix(m_inner, "a"), canonical.Prefix(e_inner, "a")),
        (
            normal_form.Filter(m_inner, m_filter),
            canonical.Filter(e_inner, e_filter),
        ),
    ]


def _bindings() -> tuple[Mapping[str, model.ColumnSet], Mapping[str, record.ColumnSet]]:
    """Give real columns to the variables `R` and `S`, on both sides.

    Returns:
        The columns for the model and the same columns for the engine.

    Example:
        >>> m, e = _bindings()
        >>> m["R"].entries == e["R"].entries
        True
    """
    columns = {"R": {"abc": "Int64", "price": "Float64"}, "S": {"az": "String"}}
    m_binding = {v: model.ColumnSet.of(**cols) for v, cols in columns.items()}
    e_binding = {v: record.ColumnSet.of(**cols) for v, cols in columns.items()}
    return m_binding, e_binding


class TestCanonicalParity:
    """The canonical form and the real columns of an expression agree on both sides."""

    def test_denote_agrees_on_every_expression(self) -> None:
        """With the same columns for `R` and `S`, each expression gives the same columns."""
        m_binding, e_binding = _bindings()
        for m_expr, e_expr in expression_pairs():
            expected = m_expr.denote(m_binding).entries
            assert e_expr.denote(e_binding).entries == expected

    def test_the_canonical_form_denotes_the_columns_of_its_expression(self) -> None:
        """In the engine, the canonical form of each expression gives the same columns.

        A form carried into a larger expression, as a call carries the
        remainder of its caller, gives its own columns too.
        """
        _, e_binding = _bindings()
        for _, e_expr in expression_pairs():
            form = e_expr.normalise()
            assert form.denote(e_binding) == e_expr.denote(e_binding), e_expr
            assert canonical.Carried(form).denote(e_binding) == form.denote(e_binding)

    def test_a_graft_of_a_form_of_no_column_gives_no_column(self) -> None:
        """A declared result with no remainder column stays empty whatever the caller binds."""
        _, e_binding = _bindings()
        empty = canonical.Canonical(canonical.Nothing(), retype="UInt32")
        bound = {"R": canonical.Canonical(canonical.Variable("R"))}
        grafted = canonical.Grafted()(empty, bound)
        assert grafted is not None
        assert grafted.denote(e_binding) == record.ColumnSet(())

    def test_a_carried_union_with_effects_keeps_its_columns(self) -> None:
        """A union that applies effects to the whole union keeps its columns as canonical.

        A type that a user writes can hold such a union. The canonical form
        applies each effect to each part, then joins the parts again.
        """
        _, e_binding = _bindings()
        parts = (
            canonical.Canonical(canonical.Variable("R")),
            canonical.Canonical(canonical.Variable("S")),
        )
        e_filter = selector.NameTest(absence.StartsWith("a"))
        form = canonical.Canonical(
            canonical.Union(parts), where=e_filter, suffix="_x", retype="UInt32"
        )
        carried = canonical.Carried(form)
        assert carried.normalise().denote(e_binding) == form.denote(e_binding)

    def test_normal_form_equality_agrees(self) -> None:
        """Two expressions are the same in the engine exactly when they are in the model."""
        forms = [(m.normalise(), e.normalise()) for m, e in expression_pairs()]
        for (m_a, e_a), (m_b, e_b) in product(forms, forms):
            assert (e_a == e_b) == (m_a == m_b)

    def test_know_describes_every_denotation(self) -> None:
        """What the engine knows about each expression is true of its real columns."""
        _, e_binding = _bindings()
        records = {
            "R": record.Record.unconstrained(),
            "S": record.Record.unconstrained(),
        }
        for _, e_expr in expression_pairs():
            known = e_expr.know(records)
            assert known.describes(e_expr.denote(e_binding))

    def test_an_empty_union_operand_drops_on_both_sides(self) -> None:
        """The union of an emptied `R` and `S` becomes `S` on both sides.

        The fixed space never makes a union of changed expressions, so
        this test builds the case by hand.
        """
        m_form = normal_form.Combined(
            normal_form.Empty(normal_form.Ref("R")),
            normal_form.Ref("S"),
        ).normalise()
        assert m_form == normal_form.Ref("S").normalise()
        e_form = canonical.Combined(
            canonical.Empty(canonical.Ref("R")),
            canonical.Ref("S"),
        ).normalise()
        assert e_form == canonical.Ref("S").normalise()

    def test_two_name_removals_merge_on_both_sides(self) -> None:
        """Dropping two names one at a time gives the same form as dropping both at once.

        The merged form is the negation of one finite set of names. The
        codec writes that form as `Minus`.
        """
        gone_a = selector.SelNot(selector.NameTest(absence.OneOf(frozenset({"a"}))))
        gone_b = selector.SelNot(selector.NameTest(absence.OneOf(frozenset({"b"}))))
        gone_both = selector.SelNot(selector.NameTest(absence.OneOf(frozenset({"a", "b"}))))
        one_at_a_time = canonical.Filter(
            canonical.Filter(canonical.Ref("R"), gone_a),
            gone_b,
        ).normalise()
        both_at_once = canonical.Filter(canonical.Ref("R"), gone_both).normalise()
        assert one_at_a_time == both_at_once
        assert one_at_a_time.where == gone_both

    def test_know_applies_the_effect_record_rules(self) -> None:
        """After a suffix, a name is absent, and the same name with the suffix is not."""
        renamed = canonical.Suffix(canonical.Ref("R"), "_p")
        known = renamed.know({"R": record.Record.unconstrained()})
        assert known.absence.holds_of("price")
        assert not known.absence.holds_of("price_p")


class TestSplitParity:
    """A `SplitColumns` declaration sends each column to the same part on both sides.

    A `SplitColumns` declaration has ordered parts. Each column goes to
    the first part that accepts it. The engine also decides two faults:
    a column that no part can take, and a part that can never get one.
    """

    def split_pair(self) -> tuple[model.ColumnSet, record.ColumnSet]:
        """Give the sample columns that both sides split.

        Returns:
            The columns for the model and the same columns for the engine.
        """
        columns = {"n": "Int64", "s": "String", "tmp_x": "Float64"}
        return model.ColumnSet.of(**columns), record.ColumnSet.of(**columns)

    def test_bound_agrees(self) -> None:
        """Both sides put each column in the same part."""
        m_split = ModelSplit((model.OfType(model_numeric()), model.NameTest(model.Always())))
        e_split = split.Split((
            selector.OfType(engine_numeric()),
            selector.NameTest(absence.Always()),
        ))
        m_cols, e_cols = self.split_pair()
        m_parts = [part.entries for part in m_split.bound(m_cols)]
        e_parts = [part.entries for part in e_split.bound(e_cols)]
        assert e_parts == m_parts

    def test_unplaced_agrees(self) -> None:
        """Both sides find the same columns that no part takes."""
        m_split = ModelSplit((model.OfType(model_numeric()),))
        e_split = split.Split((selector.OfType(engine_numeric()),))
        m_cols, e_cols = self.split_pair()
        assert e_split.unplaced(e_cols) == m_split.unplaced(m_cols)

    def test_the_gap_rejection_is_decided(self) -> None:
        """A split with only a numeric part misses columns. A last part that takes all fixes it."""
        numeric_only = split.Split((selector.OfType(type_set.Family.numeric()),))
        assert numeric_only.has_gap()
        with_rest = split.Split((
            selector.OfType(type_set.Family.numeric()),
            selector.NameTest(absence.Always()),
        ))
        assert not with_rest.has_gap()

    def test_the_dead_part_rejection_is_decided(self) -> None:
        """A part after a part that takes every column can never get a column."""
        shadowed = split.Split((
            selector.NameTest(absence.Always()),
            selector.OfType(type_set.Family.numeric()),
        ))
        assert shadowed.dead_parts() == (1,)
        ordered = split.Split((
            selector.OfType(type_set.Family.numeric()),
            selector.NameTest(absence.Always()),
        ))
        assert ordered.dead_parts() == ()
