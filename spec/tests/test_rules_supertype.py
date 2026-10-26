"""Check the supertype rules of `astrolabe._rules.supertype` against polars, on every pair of types.

The rules translate the supertype of polars, the join of two data types. Polars
has one join with two flags, and each public call uses one set of flags. Each
test here runs the call that computes exactly one variant of the join, on every
ordered pair of the 27 census kinds, of the variants of their parameters that
the rules treat apart, and of the planner types of literals with no data type:

- The default join: `pl.concat(..., how="vertical_relaxed")` gives each column
  the supertype of its two types. Three frames fold it from the left.
- The join with a literal: the planner types `when(...).then(a).otherwise(b)`
  as the supertype of `a` and `b`, then gives a literal type its standalone
  type.
- The join with no cast of a number to text: the cast pass of `max_horizontal`
  folds with that flag, refuses a pair with no supertype, and casts each input
  to the join. The text of `explain()` shows the casts.
- The join that wraps a type in a list: the cast pass of `list.set_union`.
- The lossless join of two numbers: a full join with `coalesce=True` casts both
  keys to it. Its `collect_schema()` keeps the type of the left key, so the
  test reads the type of the run.

Each test takes one left type, and its assertion names every pair that fails.
"""

from __future__ import annotations

import polars as pl
import pytest

from spec.tests.rule_probes import CastPassCheck, JoinCalls, RuleAnswer, RuleKinds
from spec.tools.node_census.kinds import CensusKinds
from astrolabe._rules.dtypes import DType
from astrolabe._rules.supertype import LosslessUpcast, Supertype, SupertypeFlag
from astrolabe._rules.releases import PolarsRelease


class TestSupertype:
    """Replay the supertype rules against the calls of polars that compute each join."""

    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_default_join_matches_the_relaxed_concat(self, left: pl.DataType) -> None:
        """The relaxed concat of two frames types a column as the default supertype of its types."""
        problems = []
        for right in RuleKinds().dtypes():
            answer = JoinCalls().relaxed_concat([left, right])
            rule = Supertype().of(DType.of(left), DType.of(right))
            if answer != rule:
                problems.append(f"{left!r} | {right!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("value", RuleKinds().LITERALS, ids=repr)
    def test_join_with_a_literal_matches_the_planner_of_when(
        self,
        value: int | float | str,
    ) -> None:
        """The planner types a `when` of a column and a literal, in either order, as their join."""
        literal = RuleKinds().literal_type(value)
        problems = []
        for dtype in RuleKinds().dtypes():
            column = DType.of(dtype)
            first = pl.when(True).then(pl.col("x")).otherwise(pl.lit(value))
            second = pl.when(True).then(pl.lit(value)).otherwise(pl.col("x"))
            orders = {
                "column first": (first, column, literal),
                "literal first": (second, literal, column),
            }
            for order, (expression, left, right) in orders.items():
                answer = JoinCalls().planned(dtype, expression)
                rule = self._materialized(Supertype().of(left, right))
                if answer != rule:
                    problems.append(
                        f"{order} {dtype!r} | {value!r}: polars {answer!r}, rule {rule!r}",
                    )
        assert not problems, "\n".join(problems)

    def test_join_of_two_literals_matches_the_planner_of_when(self) -> None:
        """The planner types a `when` of two literals as their join, the first one on the left."""
        kinds = RuleKinds()
        problems = []
        for first in kinds.LITERALS:
            for second in kinds.LITERALS:
                expression = pl.when(True).then(pl.lit(first)).otherwise(pl.lit(second))
                answer = JoinCalls().planned(pl.Int8(), expression)
                joined = Supertype().of(kinds.literal_type(first), kinds.literal_type(second))
                rule = self._materialized(joined)
                if answer != rule:
                    problems.append(f"{first!r} | {second!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    @pytest.mark.skipif(
        PolarsRelease.installed() < PolarsRelease.parse("1.40.0"),
        reason="max_horizontal has no cast pass before polars 1.40.0",
    )
    def test_join_without_text_matches_the_casts_of_max_horizontal(self, left: pl.DataType) -> None:
        """The cast pass of `max_horizontal` joins with no cast of a number to text.

        Polars 1.40.0 adds that cast pass. The join itself is the same in the
        older releases, where `is_between` and `is_close` cast with it.
        """
        check = CastPassCheck(SupertypeFlag(0))
        problems = []
        for right in RuleKinds().dtypes():
            text = JoinCalls().explained((left, right), pl.max_horizontal("x0", "x1"))
            problem = check.problem((DType.of(left), DType.of(right)), text)
            if problem is not None:
                problems.append(f"{left!r} | {right!r}: {problem}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("left", RuleKinds().lists(), ids=repr)
    def test_join_that_wraps_in_a_list_matches_the_casts_of_set_union(
        self,
        left: pl.DataType,
    ) -> None:
        """The cast pass of `list.set_union` joins a list with a type by its inner type."""
        check = CastPassCheck(SupertypeFlag.default() | SupertypeFlag.ALLOW_IMPLODE_LIST)
        union = pl.col("x0").list.set_union(pl.col("x1"))
        problems = []
        for right in RuleKinds().dtypes():
            text = JoinCalls().explained((left, right), union)
            problem = check.problem((DType.of(left), DType.of(right)), text)
            if problem is not None:
                problems.append(f"{left!r} | {right!r}: {problem}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("left", RuleKinds().dtypes(), ids=repr)
    def test_lossless_join_matches_the_keys_of_a_full_join(self, left: pl.DataType) -> None:
        """A full join that coalesces its keys casts them to their lossless join, or needs one type.

        The run gives the type of the key: `collect_schema()` keeps the type of the left key.
        """
        problems = []
        for right in RuleKinds().dtypes():
            answer = JoinCalls().full_join(left, right)
            ours, theirs = DType.of(left), DType.of(right)
            upcast = LosslessUpcast().of(ours, theirs)
            rule = upcast if upcast is not None else (ours if ours.same(theirs) else None)
            if answer != rule:
                problems.append(f"{left!r} | {right!r}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    @pytest.mark.parametrize("first", CensusKinds().dtypes(), ids=repr)
    def test_fold_matches_the_relaxed_concat_of_three_frames(self, first: pl.DataType) -> None:
        """The relaxed concat of three frames folds the supertype from the left."""
        kinds = CensusKinds().dtypes()
        problems = []
        for second in kinds:
            for third in kinds:
                answer = JoinCalls().relaxed_concat([first, second, third])
                rule = self._fold([DType.of(dtype) for dtype in (first, second, third)])
                if answer != rule:
                    triple = f"{first!r} | {second!r} | {third!r}"
                    problems.append(f"{triple}: polars {answer!r}, rule {rule!r}")
        assert not problems, "\n".join(problems)

    def _materialized(self, joined: DType | None) -> DType | None:
        """Give the type that the planner shows for a join: a literal takes its standalone type.

        Args:
            joined: The join, or `None`.

        Returns:
            The type, or `None`.

        Example:
            >>> TestSupertype()._materialized(DType.dynamic_int(300))
            Int32
        """
        return None if joined is None else joined.materialized()

    def _fold(self, dtypes: list[DType]) -> DType | None:
        """Give the fold of the rules, or `None` where a step has no supertype.

        Args:
            dtypes: The types, in order.

        Returns:
            The supertype of all of them, or `None`.

        Example:
            >>> TestSupertype()._fold([DType("Int8"), DType("Date")]) is None
            True
        """
        return RuleAnswer().dtype(lambda: Supertype().fold(dtypes))
