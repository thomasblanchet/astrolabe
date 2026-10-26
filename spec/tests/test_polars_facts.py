"""What polars really does, measured, for every fact the documents assert.

The catalogue says what the type system must produce for each operation.
That answer is only as good as its picture of the library. Each test
here runs one call and asserts the columns and data types it gives, so a
claim in a document rests on a run and not on a memory.

A polars upgrade that changes one of these behaviours fails the test
that names it. That is the point: the catalogue is then wrong, and the
suite says so before a reader is misled.
"""

from __future__ import annotations

import io
import operator
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import polars as pl
import polars.selectors as cs
import pytest

from astrolabe._hooks.fill_null import RefusedKinds
from astrolabe._rules.releases import PolarsRelease
from astrolabe._hooks.row_counts import RowCounts

from astrolabe import frame
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import GroupBy
from astrolabe._rules.node_evaluation import ColumnLeaf
from astrolabe._rules.options import RowForm
from astrolabe._rules.supertype import Supertype
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import TreeEvaluation

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

JoinHow = Literal["inner", "left", "right", "full", "semi", "anti", "cross"]
ConcatHow = Literal[
    "vertical",
    "vertical_relaxed",
    "diagonal",
    "diagonal_relaxed",
    "horizontal_extend",
    "align",
    "align_full",
    "align_inner",
    "align_left",
    "align_right",
]

STRUCT_THEN_KEY: dict[str, pl.DataType] = {
    "s": pl.Struct({"k": pl.Int64}),
    "k": pl.Int64(),
}

LEFT = pl.DataFrame({"symbol": ["a"], "price": [1.0], "sector": ["x"]})
RIGHT = pl.DataFrame({"symbol": ["a"], "sector": ["y"], "country": ["z"]})

TYPED = pl.DataFrame({
    "i": pl.Series([1, 2], dtype=pl.Int64),
    "f": pl.Series([1.0, 2.0], dtype=pl.Float64),
    "s": pl.Series(["a", "b"], dtype=pl.String),
    "b": pl.Series([True, False], dtype=pl.Boolean),
    "d": pl.Series([date(2020, 1, 1)] * 2, dtype=pl.Date),
    "dt": pl.Series([datetime(2020, 1, 1)] * 2, dtype=pl.Datetime("us")),
    "t": pl.Series([time(1, 2)] * 2, dtype=pl.Time),
    "du": pl.Series([timedelta(days=1)] * 2, dtype=pl.Duration),
    "li": pl.Series([[1], [2]], dtype=pl.List(pl.Int64)),
})


def select_dtype(column: str, reducer: str, *args: float) -> str:
    """Give the data type a reducer produces in a `select` context.

    Args:
        column: Which column of `TYPED` to reduce.
        reducer: The name of the expression method to call.
        *args: Arguments the reducer needs, such as a quantile.

    Returns:
        The resulting data type, or "raises" when polars rejects it.
    """
    try:
        return str(TYPED.select(getattr(pl.col(column), reducer)(*args)).dtypes[0])
    except pl.exceptions.PolarsError:
        return "raises"


def agg_dtype(column: str, reducer: str) -> str:
    """Give the data type a reducer produces inside `group_by(...).agg(...)`.

    Args:
        column: Which column of `TYPED` to reduce.
        reducer: The name of the expression method to call.

    Returns:
        The resulting data type, or "raises" when polars rejects it.
    """
    grouped = TYPED.with_columns(g=pl.lit("k")).group_by("g")
    try:
        return str(grouped.agg(getattr(pl.col(column), reducer)()).drop("g").dtypes[0])
    except pl.exceptions.PolarsError:
        return "raises"


def frame_dtype(column: str, reducer: str) -> str:
    """Give the data type a whole-frame reducer produces for one column.

    Args:
        column: Which column of `TYPED` to read from the result.
        reducer: The name of the frame method to call.

    Returns:
        The resulting data type, or "raises" when polars rejects it.
    """
    try:
        result = getattr(TYPED, reducer)()
    except pl.exceptions.PolarsError:
        return "raises"
    return str(result.schema[column])


@dataclass(frozen=True)
class JoinCall:
    """One join spelled as a user writes it, so its columns can be read back."""

    how: JoinHow
    suffix: str = "_right"

    def columns(self) -> list[str]:
        """Run the join and give the names it produces.

        Returns:
            The result's column names, in order.
        """
        return LEFT.join(RIGHT, on="symbol", how=self.how, suffix=self.suffix).columns


class TestJoinNaming:
    """The naming rules a join follows when two column names collide."""

    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (JoinCall("inner"), ["symbol", "price", "sector", "sector_right", "country"]),
            (JoinCall("left"), ["symbol", "price", "sector", "sector_right", "country"]),
            (JoinCall("right"), ["price", "sector", "symbol", "sector_right", "country"]),
            (
                JoinCall("full"),
                ["symbol", "price", "sector", "symbol_right", "sector_right", "country"],
            ),
            (
                JoinCall("inner", "_ref"),
                ["symbol", "price", "sector", "sector_ref", "country"],
            ),
            (JoinCall("semi"), ["symbol", "price", "sector"]),
            (JoinCall("anti"), ["symbol", "price", "sector"]),
        ],
        ids=repr,
    )
    def test_only_the_right_frame_is_renamed(self, call: JoinCall, expected: list[str]) -> None:
        """The left frame keeps every name, for every join type.

        The right frame's colliding column takes the suffix. This holds
        even for `how="right"`, where only the column order changes.
        """
        assert call.columns() == expected

    def test_a_full_join_can_suffix_the_key_itself(self) -> None:
        """Without coalescing, both key columns survive, so the key can be renamed."""
        assert "symbol_right" in LEFT.join(RIGHT, on="symbol", how="full").columns

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_semi_and_anti_take_no_column_from_the_right(self, how: JoinHow) -> None:
        """These two filter rows, so no name of the right frame can collide."""
        assert LEFT.join(RIGHT, on="symbol", how=how).columns == LEFT.columns

    def test_a_suffix_is_applied_once_and_never_twice(self) -> None:
        """A renamed column that collides again raises rather than suffixing again.

        This is why the set of output names is decided in one pass. A
        rule that iterated would describe an operation polars does not
        have.
        """
        left = pl.DataFrame({"price": [1], "price_prev": [2], "k": [1]})
        right = pl.DataFrame({"k": [1], "price": [3]})
        with pytest.raises(pl.exceptions.DuplicateError):
            left.join(right, on="k", suffix="_prev")


class TestJoinSurvivingColumns:
    """Which left columns survive, which decides what a right column collides with."""

    KEYED = pl.DataFrame({"k": [1], "v": [1]})
    OTHER = pl.DataFrame({"k": [1], "v": [2]})

    @pytest.mark.parametrize("how", ["inner", "left", "full"])
    def test_coalescing_keeps_the_left_key(self, how: JoinHow) -> None:
        """For every layout but `right`, the surviving key carries the left name."""
        joined = self.KEYED.join(self.OTHER, on="k", how=how, coalesce=True)
        assert joined.columns == ["k", "v", "v_right"]

    def test_a_right_join_that_coalesces_drops_the_left_key(self) -> None:
        """The surviving key is the right frame's, and it moves after the left columns.

        This is why a collision must be read against the left columns
        that survive, and not against every column the left frame had.
        """
        joined = self.KEYED.join(self.OTHER, on="k", how="right", coalesce=True)
        assert joined.columns == ["v", "k", "v_right"]

    @pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
    def test_not_coalescing_keeps_both_keys(self, how: JoinHow) -> None:
        """Without coalescing the right key survives under a suffixed name."""
        joined = self.KEYED.join(self.OTHER, on="k", how=how, coalesce=False)
        assert joined.columns == ["k", "v", "k_right", "v_right"]

    WIDE_LEFT = pl.DataFrame({"k": [1], "v": [1], "w": [1]})
    WIDE_RIGHT = pl.DataFrame({"k": [1], "v": [2], "z": [2]})

    @pytest.mark.parametrize("how", ["inner", "left", "full"])
    def test_only_a_colliding_name_is_suffixed(self, how: JoinHow) -> None:
        """A right column colliding with no surviving left column keeps its name."""
        joined = self.WIDE_LEFT.join(self.WIDE_RIGHT, on="k", how=how, coalesce=True)
        assert joined.columns == ["k", "v", "w", "v_right", "z"]

    def test_differing_key_names_survive_by_side(self) -> None:
        """With left_on and right_on, the surviving key carries its side's name."""
        left = pl.DataFrame({"a": [1], "v": [1]})
        right = pl.DataFrame({"b": [1], "v": [2]})
        inner = left.join(right, left_on="a", right_on="b", coalesce=True)
        assert inner.columns == ["a", "v", "v_right"]
        flipped = left.join(right, left_on="a", right_on="b", how="right", coalesce=True)
        assert flipped.columns == ["v", "b", "v_right"]

    def test_two_keys_move_together_in_a_right_join(self) -> None:
        """Every coalesced key takes the right frame's position at once."""
        left = pl.DataFrame({"k1": [1], "k2": [1], "v": [1]})
        right = pl.DataFrame({"k1": [1], "k2": [1], "v": [2]})
        joined = left.join(right, on=["k1", "k2"], how="right", coalesce=True)
        assert joined.columns == ["v", "k1", "k2", "v_right"]


class TestJoinKeyTree:
    """Polars reads the tree of a join key, and not only what the key computes."""

    FRAME = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    MESSAGE = "'alias' is not allowed in a join key"
    KEYS: ClassVar[tuple[pl.Expr, ...]] = (
        pl.col("a").alias("b"),
        pl.col("a").alias("x") + 1,
        pl.struct(pl.col("a").alias("x")),
        pl.col("a").over(pl.col("b").alias("q")),
        pl.lit(1).alias("x"),
    )

    @pytest.mark.parametrize("key", KEYS, ids=str)
    def test_a_key_with_an_alias_raises_in_each_join(self, key: pl.Expr) -> None:
        """`join`, `join_asof` and the schema of a lazy join all refuse the key."""
        frame = self.FRAME
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
            frame.join(frame, on=key)
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
            frame.join(frame, left_on=key, right_on="a")
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
            frame.join_asof(frame, left_on=key, right_on="a")
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
            frame.lazy().join(frame.lazy(), on=key).collect_schema()

    def test_a_predicate_with_an_alias_raises_in_join_where(self) -> None:
        """A `join_where` predicate is a join key too, on either side of the comparison."""
        frame = self.FRAME
        for predicate in (
            pl.col("a").alias("z") > pl.col("b_right"),
            (pl.col("a") > pl.col("b_right")).alias("p"),
        ):
            with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
                frame.join_where(frame, predicate)

    ROWS: ClassVar[tuple[pl.Expr, ...]] = (
        pl.col("a").shift(1),
        pl.col("a").sort(),
        pl.col("a").over("b"),
        pl.col("a").head(1),
        pl.col("a").is_in(pl.col("b").implode()),
        pl.col("a") + pl.col("b").sum(),
        pl.lit(pl.Series([1, 2])),
    )
    KEPT: ClassVar[tuple[pl.Expr, ...]] = (
        pl.col("a").sum(),
        pl.lit(1),
        pl.col("a").is_in([1]),
        pl.when(pl.col("a") > 0).then(pl.col("a")).otherwise(0),
        pl.col("a").cast(pl.Int32) + 1,
    )

    ELEMENTWISE: ClassVar[str] = "all join key expressions must be elementwise"

    @pytest.mark.parametrize("key", ROWS, ids=str)
    def test_a_key_that_changes_the_rows_raises(self, key: pl.Expr) -> None:
        """A key that is not one value must work value by value at each of its nodes."""
        frame = self.FRAME
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.ELEMENTWISE):
            frame.join(frame, on=key)

    def test_join_asof_refuses_a_key_that_changes_the_rows_too(self) -> None:
        """`join_asof` checks its keys as `join` does."""
        frame = self.FRAME
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.ELEMENTWISE):
            frame.join_asof(frame, left_on=pl.col("a").cum_sum(), right_on="a")

    CALLS: ClassVar[dict[str, tuple[pl.Expr, bool]]] = {
        "map_batches": (pl.col("a").map_batches(lambda s: s), True),
        "map_batches that works value by value": (
            pl.col("a").map_batches(lambda s: s, is_elementwise=True),
            False,
        ),
        "map_batches of one value": (
            pl.col("a").map_batches(lambda s: s.max(), returns_scalar=True),
            False,
        ),
        "map_batches that works value by value over shift": (
            pl.col("a").shift(1).map_batches(lambda s: s, is_elementwise=True),
            True,
        ),
        "map_elements": (pl.col("a").map_elements(lambda v: v, return_dtype=pl.Int64), False),
        "rolling_map": (pl.col("a").rolling_map(lambda s: s.sum(), window_size=1), True),
        "fold": (pl.fold(pl.lit(0), operator.add, [pl.col("a"), pl.col("b")]), True),
        "fold of one value": (
            pl.fold(pl.lit(0), lambda acc, x: acc + x.sum(), [pl.col("a")], returns_scalar=True),
            False,
        ),
        "reduce": (pl.reduce(operator.add, [pl.col("a"), pl.col("b")]), True),
        "map_groups": (pl.map_groups([pl.col("a")], lambda s: s[0]), True),
        "map_groups of one value": (
            pl.map_groups([pl.col("a")], lambda s: s[0].sum(), returns_scalar=True),
            False,
        ),
        "pl.map_batches that works value by value": (
            pl.map_batches([pl.col("a"), pl.col("b")], lambda s: s[0] + s[1], is_elementwise=True),
            False,
        ),
        "name.map": (pl.col("a").name.map(lambda name: name), False),
    }

    @pytest.mark.parametrize(
        "key",
        [
            pl.cum_fold(pl.lit(0), operator.add, [pl.col("a")]),
            pl.cum_reduce(operator.add, [pl.col("a")]),
        ],
        ids=["cum_fold", "cum_reduce"],
    )
    def test_a_cumulative_fold_holds_an_alias(self, key: pl.Expr) -> None:
        """A cumulative fold names its result with an alias, which polars refuses in a key."""
        frame = self.FRAME
        with pytest.raises(pl.exceptions.InvalidOperationError, match=self.MESSAGE):
            frame.join(frame, on=key)

    @pytest.mark.parametrize("name", sorted(CALLS))
    def test_a_function_of_the_user_has_the_facts_of_its_call(self, name: str) -> None:
        """Polars reads from the call whether a function works value by value and gives one value.

        `map_batches` takes both facts from `is_elementwise=` and
        `returns_scalar=`, `map_elements` works value by value, and
        `rolling_map` does not.
        """
        key, refused = self.CALLS[name]
        frame = self.FRAME
        if refused:
            with pytest.raises(pl.exceptions.InvalidOperationError, match=self.ELEMENTWISE):
                frame.join(frame, on=key)
        else:
            assert frame.join(frame, on=key).columns[:2] == ["a", "b"]

    @pytest.mark.parametrize("key", KEPT, ids=str)
    def test_a_key_of_one_value_or_that_works_value_by_value_runs(self, key: pl.Expr) -> None:
        """Polars moves a key of one value into a column, and skips the literal of `is_in`."""
        frame = self.FRAME
        assert frame.join(frame, on=key).columns[:2] == ["a", "b"]

    @pytest.mark.parametrize(
        "key",
        [pl.col("a").cast(pl.Int64), pl.col("a") + 0, pl.col("a").name.keep()],
        ids=str,
    )
    def test_only_a_bare_column_key_merges_the_keys(self, key: pl.Expr) -> None:
        """A key that computes its column unchanged, in another tree, keeps both keys.

        A declared key fixes what the key computes, and not its tree, so
        the caller decides whether the keys merge.
        """
        frame = self.FRAME
        assert frame.join(frame, on=pl.col("a")).columns == ["a", "b", "b_right"]
        assert frame.join(frame, on=key).columns == ["a", "b", "a_right", "b_right"]

    def test_a_rename_by_name_is_no_alias(self) -> None:
        """`name.suffix` and `name.keep` rename the output, and the join runs."""
        frame = self.FRAME
        assert frame.join(frame, on=pl.col("a").name.suffix("x")).columns[:2] == ["a", "b"]
        assert frame.join(frame, on=pl.col("a").name.keep()).columns[:2] == ["a", "b"]


class TestReducerContexts:
    """A reducer has three behaviours, one per context it is written in."""

    @pytest.mark.parametrize(
        ("column", "expected"),
        [("i", "Int64"), ("f", "Float64"), ("b", "UInt32"), ("du", "Duration(time_unit='us')")],
    )
    def test_sum_accepts_these_in_a_select(self, column: str, expected: str) -> None:
        """Summing is defined for numbers, Booleans and durations."""
        assert select_dtype(column, "sum") == expected

    @pytest.mark.parametrize("column", ["s", "d", "dt", "t", "li"])
    def test_sum_rejects_these_in_a_select(self, column: str) -> None:
        """Summing text, a date, a time or a list raises in a `select`."""
        assert select_dtype(column, "sum") == "raises"

    @pytest.mark.parametrize("column", ["s", "d", "t", "li"])
    def test_a_whole_frame_sum_passes_through_what_a_select_rejects(self, column: str) -> None:
        """The whole-frame form keeps a column it cannot sum, unchanged.

        This is the sharpest of the three-context differences. The same
        column raises in a `select` and survives here, so one rule for
        both contexts must be wrong in one of them.
        """
        assert frame_dtype(column, "sum") == str(TYPED.schema[column])

    def test_mean_on_a_date_gives_a_datetime(self) -> None:
        """Averaging dates gives a datetime, in every context that accepts it."""
        assert select_dtype("d", "mean") == "Datetime(time_unit='us', time_zone=None)"
        assert agg_dtype("d", "mean") == "Datetime(time_unit='us', time_zone=None)"
        assert frame_dtype("d", "mean") == "Datetime(time_unit='us', time_zone=None)"

    def test_mean_on_text_raises_in_a_select_and_passes_elsewhere(self) -> None:
        """The `agg` and whole-frame contexts keep a text column that `select` rejects."""
        assert select_dtype("s", "mean") == "raises"
        assert agg_dtype("s", "mean") == "String"
        assert frame_dtype("s", "mean") == "String"

    @pytest.mark.parametrize("reducer", ["min", "max"])
    def test_min_and_max_reject_a_list_only_in_a_select(self, reducer: str) -> None:
        """Taking the smallest of a list column raises in a `select` alone."""
        assert select_dtype("li", reducer) == "raises"
        assert agg_dtype("li", reducer) == "List(Int64)"
        assert frame_dtype("li", reducer) == "List(Int64)"

    def test_var_rejects_a_duration_where_std_accepts_it(self) -> None:
        """The variance gate is exactly `Duration`, and it is not shared with `std`.

        An error message for this gate must name durations. A message
        naming numbers and Booleans describes a different family.
        """
        assert select_dtype("du", "var") == "raises"
        assert select_dtype("du", "std") == "Duration(time_unit='us')"

    @pytest.mark.parametrize(
        ("column", "expected"),
        [("i", "Float64"), ("d", "Datetime(time_unit='us', time_zone=None)")],
    )
    def test_quantile_maps_these(self, column: str, expected: str) -> None:
        """A quantile of numbers is a float, and of dates a datetime."""
        assert select_dtype(column, "quantile", 0.5) == expected

    @pytest.mark.parametrize("column", ["s", "b", "li"])
    def test_quantile_rejects_these(self, column: str) -> None:
        """A quantile of text, a Boolean or a list raises."""
        assert select_dtype(column, "quantile", 0.5) == "raises"

    def test_product_turns_a_boolean_into_an_integer(self) -> None:
        """Multiplying Booleans gives `Int64`, not a Boolean."""
        assert select_dtype("b", "product") == "Int64"


class TestReshape:
    """Operations whose result columns come from somewhere other than the schema."""

    PIVOTABLE = pl.DataFrame({
        "account": ["a", "a"],
        "venue": ["v", "v"],
        "region": ["EMEA", "APAC"],
        "amount": [1, 2],
    })

    def test_pivot_without_an_index_keeps_every_other_column(self) -> None:
        """Omitting `index=` uses every remaining column as the index.

        So a column outside the schema survives the call. Treating this
        as emptying the frame loses columns polars keeps.
        """
        result = self.PIVOTABLE.pivot(on="region", values="amount")
        assert result.columns == ["account", "venue", "EMEA", "APAC"]

    def test_pivot_with_an_index_keeps_only_the_index(self) -> None:
        """Naming `index=` drops every column not named."""
        result = self.PIVOTABLE.pivot(on="region", index="account", values="amount")
        assert result.columns == ["account", "EMEA", "APAC"]

    def test_a_literal_on_columns_names_the_output(self) -> None:
        """`on_columns=` lists the new columns, a value with no data included."""
        result = self.PIVOTABLE.pivot(
            on="region",
            on_columns=["EMEA", "APAC", "LATAM"],
            index="account",
            values="amount",
        )
        assert result.columns == ["account", "EMEA", "APAC", "LATAM"]

    @pytest.mark.parametrize(
        "on_columns",
        [[pl.lit("EMEA")], (pl.col("region"),), ["EMEA", pl.lit("APAC")]],
        ids=["a literal", "a column", "a text beside a literal"],
    )
    def test_an_expression_in_on_columns_raises_at_every_frame(
        self,
        on_columns: Sequence[object],
    ) -> None:
        """Polars builds a series of the values, and an expression is no value of a column name.

        In a list of expressions it makes an `Object` series, which polars
        cannot cast. Beside a string, the series refuses it.
        """
        for pivotable in (self.PIVOTABLE, self.PIVOTABLE.head(0)):
            with pytest.raises((pl.exceptions.ComputeError, TypeError)):
                pivotable.pivot(
                    on="region",
                    on_columns=on_columns,
                    index="account",
                    values="amount",
                )
            with pytest.raises((pl.exceptions.ComputeError, TypeError)):
                self.lazy_pivot_schema(pivotable, on_columns)

    def lazy_pivot_schema(self, pivotable: pl.DataFrame, on_columns: Sequence[object]) -> None:
        """Read the schema of a lazy pivot of a frame by `region`.

        Args:
            pivotable: The frame.
            on_columns: The values of `region` to make columns of.
        """
        lazy = pivotable.lazy().pivot(
            on="region",
            on_columns=on_columns,
            index="account",
            values="amount",
        )
        lazy.collect_schema()

    def test_unpivot_turns_every_other_column_into_rows(self) -> None:
        """The result is the index columns plus one name column and one value column."""
        frame = pl.DataFrame({"symbol": ["a"], "x": [1], "y": [2]})
        assert frame.unpivot(index="symbol").columns == ["symbol", "variable", "value"]


class TestAlignment:
    """What `align_frames` demands of the columns it aligns on."""

    @pytest.mark.parametrize(
        ("left_type", "right_type", "expected"),
        [
            (pl.Int32, pl.Int64, "Int64"),
            (pl.Int64, pl.UInt64, "Int128"),
            (pl.Int64, pl.Int64, "Int64"),
        ],
    )
    def test_alignment_keys_need_only_be_castable(
        self,
        left_type: pl.DataType,
        right_type: pl.DataType,
        expected: str,
    ) -> None:
        """Two key columns align when a common type exists, not only when equal.

        `Int64` and `UInt64` have no common signed or unsigned type of
        their own width, so polars widens to `Int128`.
        """
        left = pl.DataFrame({"k": pl.Series([1], dtype=left_type), "x": [1]})
        right = pl.DataFrame({"k": pl.Series([1], dtype=right_type), "y": [1]})
        aligned = pl.align_frames(left, right, on="k")
        assert str(aligned[0].schema["k"]) == expected


class TestNameCollisions:
    """Every operation that can produce a name already taken, and what it does."""

    def test_renaming_onto_a_taken_name_raises(self) -> None:
        """A rename whose destination exists raises rather than replacing."""
        with pytest.raises(pl.exceptions.DuplicateError):
            pl.DataFrame({"a": [1], "b": [2]}).rename({"a": "b"})

    def test_a_row_index_onto_a_taken_name_raises(self) -> None:
        """Adding a row index needs its own name free."""
        with pytest.raises(pl.exceptions.DuplicateError):
            pl.DataFrame({"idx": [1]}).with_row_index("idx")

    def test_unnesting_a_field_onto_a_taken_name_raises(self) -> None:
        """A struct field's name must be free in the frame it unnests into."""
        with pytest.raises(pl.exceptions.DuplicateError):
            pl.DataFrame({"s": [{"k": 1}], "k": [1]}, schema=STRUCT_THEN_KEY).unnest("s")

    def test_projecting_one_name_twice_raises(self) -> None:
        """A `select` naming a column twice raises.

        The call is written to fail, so astrolabe's own check of it is
        suppressed here. That the check fires at all is the behaviour
        this system exists to provide.
        """
        with pytest.raises(pl.exceptions.DuplicateError):
            pl.DataFrame({"a": [1]}).select("a", "a")  # type: ignore[duplicate-column]

    def test_two_produced_names_colliding_raises(self) -> None:
        """`with_columns` producing one name twice raises, in one call."""
        with pytest.raises(pl.exceptions.ComputeError):
            pl.DataFrame({"a": [1]}).with_columns(pl.lit(1).alias("z"), pl.lit(2).alias("z"))

    def test_adding_a_column_that_exists_replaces_it(self) -> None:
        """`with_columns` replaces a declared column rather than raising."""
        frame = pl.DataFrame({"a": [1]}).with_columns(a=pl.lit(2))
        assert frame.columns == ["a"]


class TestPlacement:
    """Where an operation puts the columns it produces."""

    def test_unnest_replaces_the_struct_in_place(self) -> None:
        """The struct's fields take the struct's position, not the end."""
        schema: dict[str, pl.DataType] = {
            "s": pl.Struct({"p": pl.Int64, "q": pl.String}),
            "k": pl.Int64(),
        }
        frame = pl.DataFrame({"s": [{"p": 1, "q": "t"}], "k": [1]}, schema=schema)
        assert frame.unnest("s").columns == ["p", "q", "k"]

    def test_explode_keeps_the_column_and_its_position(self) -> None:
        """Exploding a list column keeps its name and drops one level of nesting."""
        schema: dict[str, pl.DataType] = {"l": pl.List(pl.Int64), "k": pl.Int64()}
        frame = pl.DataFrame({"l": [[1, 2]], "k": [1]}, schema=schema)
        assert frame.explode("l").columns == ["l", "k"]
        assert str(frame.explode("l").schema["l"]) == "Int64"

    def test_a_row_index_is_added_at_the_front(self) -> None:
        """The index column comes first."""
        assert pl.DataFrame({"a": [1]}).with_row_index().columns == ["index", "a"]


class TestConcatenation:
    """What each concatenation strategy demands of two frames."""

    SAME = pl.DataFrame({"k": [1], "x": [1]})
    ALSO_SAME = pl.DataFrame({"k": [3], "x": [9]})
    DIFFERENT = pl.DataFrame({"k": [2], "y": ["z"]})

    @pytest.mark.parametrize("how", ["vertical", "vertical_relaxed"])
    def test_vertical_needs_the_same_columns(self, how: ConcatHow) -> None:
        """Two frames with different columns cannot be stacked vertically."""
        assert pl.concat([self.SAME, self.ALSO_SAME], how=how).columns == ["k", "x"]
        with pytest.raises(pl.exceptions.PolarsError):
            pl.concat([self.SAME, self.DIFFERENT], how=how)

    @pytest.mark.parametrize(
        "how",
        ["diagonal", "diagonal_relaxed", "align", "align_full", "align_inner"],
    )
    def test_these_take_the_union_of_the_columns(self, how: ConcatHow) -> None:
        """Every column of either frame survives, in first-seen order."""
        result = pl.concat([self.SAME, self.DIFFERENT], how=how)
        assert result.columns == ["k", "x", "y"]

    def test_horizontal_rejects_a_shared_name(self) -> None:
        """Placing two frames side by side needs their names to be disjoint.

        Polars 1.42.0 adds `horizontal_extend`. An older release refuses that
        value of `how` with `ValueError`.
        """
        recent = PolarsRelease.installed() >= PolarsRelease.parse("1.42.0")
        with pytest.raises(pl.exceptions.DuplicateError if recent else ValueError):
            pl.concat([self.SAME, self.DIFFERENT], how="horizontal_extend")


class TestWholeFrameShapes:
    """Operations whose result columns do not come from the input schema alone."""

    FRAME = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def test_describe_adds_a_statistic_column_and_retypes_numbers(self) -> None:
        """The result is one row per statistic, and a numeric column becomes a float."""
        described = self.FRAME.describe()
        assert described.columns == ["statistic", "a", "b"]
        assert str(described.schema["statistic"]) == "String"
        assert str(described.schema["a"]) == "Float64"
        assert str(described.schema["b"]) == "String"

    def test_transpose_names_columns_after_rows(self) -> None:
        """The output has one column per input row, so no schema predicts it."""
        assert self.FRAME.transpose().columns == ["column_0", "column_1"]

    def test_to_dummies_names_columns_after_values(self) -> None:
        """The output names come from the data, so no schema predicts them."""
        assert self.FRAME.to_dummies("b").columns == ["a", "b_x", "b_y"]

    def test_null_count_keeps_every_name(self) -> None:
        """Counting nulls keeps every name and gives each the same new type."""
        counted = self.FRAME.null_count()
        assert counted.columns == self.FRAME.columns
        assert {str(t) for t in counted.dtypes} == {"UInt32"}


class TestGroupByShorthands:
    """The `group_by(...).<reducer>()` forms, which are commoner than `.agg()`."""

    FRAME = pl.DataFrame({"g": ["k", "k"], "a": [1, 2], "c": [1.5, 2.5]})

    @pytest.mark.parametrize(
        "reducer",
        ["sum", "mean", "min", "max", "first", "last", "n_unique", "median"],
    )
    def test_a_shorthand_keeps_every_column_with_the_key_first(self, reducer: str) -> None:
        """The key comes first, then every other column, reduced."""
        result = getattr(self.FRAME.group_by("g"), reducer)()
        assert result.columns == ["g", "a", "c"]

    def test_len_drops_a_column_of_its_name_that_is_not_a_key(self) -> None:
        """Only a key named `len` clashes with the new column."""
        frame = pl.DataFrame({"g": [1], "len": [5]})
        assert frame.group_by("g").len().columns == ["g", "len"]
        with pytest.raises(pl.exceptions.DuplicateError):
            frame.group_by("len").len()

    def test_len_replaces_every_other_column_with_one_count(self) -> None:
        """`len` is the one shorthand that does not keep the other columns."""
        assert self.FRAME.group_by("g").len().columns == ["g", "len"]


class TestSelectorsThatReadNames:
    """Three selectors look like type tests and are name tests."""

    FRAME = pl.DataFrame({"n": [1], "s": ["a"], "d": [True]})

    def test_alpha_reads_the_name_and_not_the_data_type(self) -> None:
        """Every column here has an alphabetic name, so all three are picked."""
        assert self.FRAME.select(cs.alpha()).columns == ["n", "s", "d"]

    def test_digit_reads_the_name_and_not_the_data_type(self) -> None:
        """No column here has a numeric name, so none is picked."""
        assert self.FRAME.select(cs.digit()).columns == []

    def test_exclude_is_available_under_two_spellings(self) -> None:
        """`pl.exclude` and `cs.exclude` pick the same columns in a `select`."""
        assert self.FRAME.select(pl.exclude("n")).columns == ["s", "d"]
        assert self.FRAME.select(cs.exclude("n")).columns == ["s", "d"]


class TestDropTakesNamesAndSelectors:
    """`drop` expands each argument to names: a literal name, a column or a selector.

    The stubs of polars take names and selectors only. Polars takes a column
    and refuses any other expression at run time, so each call that passes
    an expression ignores the type of its argument.
    """

    FRAME = pl.DataFrame({"k": ["a"], "n": [1], "f": [1.5]})

    @pytest.mark.parametrize(
        ("columns", "kept"),
        [
            ((cs.numeric(),), ["k"]),
            ((pl.col("n"),), ["k", "f"]),
            (("k", cs.float()), ["n"]),
            (([cs.float(), "k"],), ["n"]),
            ((cs.temporal(),), ["k", "n", "f"]),
        ],
        ids=["selector", "column", "name and selector", "list", "selector of none"],
    )
    def test_each_argument_drops_the_columns_that_it_names(
        self,
        columns: tuple[object, ...],
        kept: list[str],
    ) -> None:
        """A selector that picks no column drops none, and does not raise."""
        assert self.FRAME.drop(*columns).columns == kept  # type: ignore[arg-type]

    def test_strict_decides_only_for_a_literal_name(self) -> None:
        """A column and a listed name of a selector require the frame to hold them."""
        assert self.FRAME.drop("x", strict=False).columns == ["k", "n", "f"]
        for required in (pl.col("x"), cs.by_name("x")):
            with pytest.raises(pl.exceptions.ColumnNotFoundError):
                self.FRAME.drop(required, strict=False)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "expression",
        [pl.col("n") + 1, pl.col("n").alias("x"), pl.lit(1)],
        ids=["binary", "alias", "literal"],
    )
    def test_an_expression_that_is_no_selector_raises_at_every_frame(
        self,
        expression: pl.Expr,
    ) -> None:
        """No name follows from such an expression, so a frame of no column raises too."""
        for receiver in (self.FRAME, pl.DataFrame()):
            with pytest.raises(pl.exceptions.InvalidOperationError, match="is not a selector"):
                receiver.drop(expression)  # type: ignore[arg-type]


class TestWholeFrameReducers:
    """Which whole-frame reducers keep every data type, and which do not."""

    def frame(self) -> pl.DataFrame:
        """Build a frame with one column of each data type under test.

        Returns:
            A one-row frame holding an integer, a float, a text, a
            Boolean and a date column.
        """
        return pl.DataFrame({
            "i": pl.Series([1], dtype=pl.Int64),
            "f": pl.Series([1.0], dtype=pl.Float64),
            "s": pl.Series(["a"], dtype=pl.String),
            "b": pl.Series([True], dtype=pl.Boolean),
            "d": pl.Series([date(2020, 1, 1)], dtype=pl.Date),
        })

    def changed_by(self, reducer: str) -> dict[str, str]:
        """Give the columns a whole-frame reducer retypes.

        Args:
            reducer: The name of the frame method to call.

        Returns:
            Each column whose type changed, mapped to its new type.
        """
        source = self.frame()
        result = getattr(source, reducer)()
        return {
            name: str(result.schema[name])
            for name in source.columns
            if str(result.schema[name]) != str(source.schema[name])
        }

    @pytest.mark.parametrize("reducer", ["min", "max"])
    def test_min_and_max_keep_every_data_type(self, reducer: str) -> None:
        """These two are the identity on types, so they need no promise."""
        assert self.changed_by(reducer) == {}

    def test_sum_keeps_every_type_except_boolean(self) -> None:
        """Summing is the identity on types apart from `Boolean`, which counts.

        `Boolean` is a nameable type, so this gate can be written as a
        requirement. That is the difference between this operation and
        a whole-frame `cast`, whose domain is no family at all.
        """
        assert self.changed_by("sum") == {"b": "UInt32"}

    @pytest.mark.parametrize("reducer", ["mean", "median"])
    def test_averaging_maps_three_types(self, reducer: str) -> None:
        """An integer becomes a float, a Boolean a float, a date a datetime."""
        assert self.changed_by(reducer) == {
            "i": "Float64",
            "b": "Float64",
            "d": "Datetime(time_unit='us', time_zone=None)",
        }

    def test_product_collapses_several_types_to_null(self) -> None:
        """Multiplying a text or a date column gives `Null`, not an error."""
        assert self.changed_by("product") == {
            "s": "Null",
            "b": "Int64",
            "d": "Null",
        }

    @pytest.mark.parametrize("reducer", ["count", "null_count"])
    def test_counting_maps_every_type_to_one(self, reducer: str) -> None:
        """These two are constant functions, so the output type is nameable."""
        assert set(self.changed_by(reducer).values()) == {"UInt32"}


class TestSharedColumnsWhenCombining:
    """What each combining strategy demands of a name both frames hold.

    This is what decides whether an operation can be typed over two
    remainders. If a shared name constrains the two data types, and the
    two remainders cannot be proven to share no name, then no signature
    can establish the precondition.
    """

    NUMBERS: ClassVar[tuple[type[pl.DataType], ...]] = (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.UInt128,
        pl.Float16,
        pl.Float32,
        pl.Float64,
    )

    def pair(self, left: pl.DataType, right: pl.DataType) -> tuple[pl.DataFrame, ...]:
        """Build two frames sharing a key and a column `x` of two types.

        Args:
            left: The data type of `x` in the first frame.
            right: The data type of `x` in the second.

        Returns:
            The two frames.
        """
        return (
            pl.DataFrame({"k": [1], "x": pl.Series([1], dtype=left)}),
            pl.DataFrame({"k": [1], "x": pl.Series([1], dtype=right)}),
        )

    @pytest.mark.parametrize("how", ["diagonal", "vertical"])
    def test_these_need_a_shared_column_to_have_one_type(self, how: ConcatHow) -> None:
        """Two widths of integer are not enough alike for the strict forms."""
        with pytest.raises(pl.exceptions.SchemaError):
            pl.concat(self.pair(pl.Int32(), pl.Int64()), how=how)

    @pytest.mark.parametrize("how", ["diagonal_relaxed", "vertical_relaxed"])
    def test_the_relaxed_forms_widen_instead(self, how: ConcatHow) -> None:
        """The relaxed forms find a supertype of the two."""
        result = pl.concat(self.pair(pl.Int32(), pl.Int64()), how=how)
        assert str(result.schema["x"]) == "Int64"

    def test_two_numbers_widen_to_the_supertype_of_the_rules(self) -> None:
        """A relaxed concat of two numeric kinds gives the supertype that the rules give."""
        for left, right in product(self.NUMBERS, repeat=2):
            result = pl.concat(self.pair(left(), right()), how="diagonal_relaxed")
            expected = Supertype().of(DType.of(left()), DType.of(right()))
            assert expected is not None, (left, right)
            assert str(result.schema["x"]) == expected.render(), (left, right)

    def test_align_joins_on_every_common_column(self) -> None:
        """With no `on=`, alignment uses every shared name as a join key.

        So a shared name whose types do not match raises, exactly as a
        join key would. Two widths of integer are castable and succeed.
        """
        aligned = pl.concat(self.pair(pl.Int32(), pl.Int64()), how="align")
        assert str(aligned.schema["x"]) == "Int64"
        left, right = self.pair(pl.Int64(), pl.Int64())
        mismatched = right.with_columns(x=pl.lit("s"))
        with pytest.raises(pl.exceptions.SchemaError):
            pl.concat([left, mismatched], how="align")

    def test_horizontal_rejects_a_shared_name_outright(self) -> None:
        """Placing frames side by side never merges; it raises.

        Before polars 1.42.0, `horizontal_extend` is not a value of `how`.
        """
        recent = PolarsRelease.installed() >= PolarsRelease.parse("1.42.0")
        with pytest.raises(pl.exceptions.DuplicateError if recent else ValueError):
            pl.concat(self.pair(pl.Int64(), pl.Int64()), how="horizontal_extend")


class TestPositionalColumnOperations:
    """Operations addressed by position rather than by name."""

    def test_replace_column_takes_the_series_name_too(self) -> None:
        """The replacement's own name wins, so this renames as well as retypes."""
        frame = pl.DataFrame({"a": [1], "b": [2]})
        replaced = frame.replace_column(0, pl.Series("zz", [9]))
        assert replaced.columns == ["zz", "b"]

    def test_by_name_accepts_both_spellings(self) -> None:
        """A list and separate arguments name the same columns."""
        frame = pl.DataFrame({"a": [1], "b": [2]})
        assert frame.select(cs.by_name("a", "b")).columns == ["a", "b"]
        assert frame.select(cs.by_name(["a", "b"])).columns == ["a", "b"]

    def test_by_name_is_strict_unless_told_otherwise(self) -> None:
        """A missing name raises, unless `require_all=False`."""
        frame = pl.DataFrame({"a": [1]})
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            frame.select(cs.by_name("a", "nope"))
        assert frame.select(cs.by_name("a", "nope", require_all=False)).columns == ["a"]


class TestOperationDomains:
    """Which data types each partial operation accepts, measured exactly.

    A requirement can state a gate only if the gate is a set of data
    types the vocabulary can name. These tests fix each gate exactly, so
    a document can say whether it is expressible instead of guessing.
    """

    NESTED: ClassVar[dict[str, pl.Series]] = {
        "List": pl.Series([[1], None], dtype=pl.List(pl.Int64)),
        "Array": pl.Series([[1], None], dtype=pl.Array(pl.Int64, 1)),
        "Struct": pl.Series([{"p": 1}, None], dtype=pl.Struct({"p": pl.Int64})),
    }
    SCALAR: ClassVar[dict[str, pl.Series]] = {
        "Int64": pl.Series([1, None], dtype=pl.Int64),
        "String": pl.Series(["a", None], dtype=pl.String),
        "Boolean": pl.Series([True, None], dtype=pl.Boolean),
        "Date": pl.Series([date(2020, 1, 1), None], dtype=pl.Date),
        "Binary": pl.Series([b"x", None], dtype=pl.Binary),
        "Null": pl.Series([None, None], dtype=pl.Null),
    }
    CATEGORIES: ClassVar[dict[str, pl.Series]] = {
        "Categorical": pl.Series(["a", None], dtype=pl.Categorical),
        "Enum": pl.Series(["a", None], dtype=pl.Enum(["a", "b"])),
    }

    def rejected_by_fill_null(self, strategy: str, *, categories: bool = False) -> set[str]:
        """Give the data types a fill strategy rejects.

        A data type is rejected when the strategy raises on a column
        of that type that holds a null.

        Args:
            strategy: The `fill_null` strategy to try.
            categories: Whether to try the categorical types in place of the others.

        Returns:
            The names of the types that raise.
        """
        probes = self.CATEGORIES if categories else {**self.SCALAR, **self.NESTED}
        rejected = set()
        for name, series in probes.items():
            try:
                pl.DataFrame({name: series}).fill_null(strategy=strategy)  # type: ignore[arg-type]
            except pl.exceptions.PolarsError:
                rejected.add(name)
        return rejected

    @pytest.mark.parametrize("strategy", ["forward", "backward"])
    def test_two_strategies_are_total(self, strategy: str) -> None:
        """Carrying a value forward or backward works for every type."""
        assert self.rejected_by_fill_null(strategy) == set()

    @pytest.mark.parametrize("strategy", ["min", "max"])
    def test_two_strategies_reject_exactly_the_nested_types(self, strategy: str) -> None:
        """The gate is exactly `NestedType`, so one family names it."""
        assert self.rejected_by_fill_null(strategy) == set(self.NESTED)

    def test_zero_rejects_the_nested_types_and_null(self) -> None:
        """A `Null` column has no zero, so the gate is a family plus one named type."""
        assert self.rejected_by_fill_null("zero") == set(self.NESTED) | {"Null"}

    def test_one_rejects_the_nested_types_and_three_more(self) -> None:
        """The gate is a family plus three named types, so a set names it."""
        assert self.rejected_by_fill_null("one") == set(self.NESTED) | {
            "String",
            "Binary",
            "Null",
        }

    def test_mean_rejects_the_nested_types_and_three_more(self) -> None:
        """The gate is a family plus three named types."""
        assert self.rejected_by_fill_null("mean") == set(self.NESTED) | {
            "String",
            "Boolean",
            "Binary",
        }

    @pytest.mark.parametrize("strategy", ["zero", "one", "mean", "min", "max"])
    def test_a_strategy_with_a_value_rejects_the_categories_from_polars_1_44(
        self,
        strategy: str,
    ) -> None:
        """From polars 1.44.0, the strategies that compute a value reject a categorical type."""
        valued = strategy in {"zero", "one", "mean"}
        recent = PolarsRelease.installed() >= PolarsRelease.parse("1.44.0")
        expected = set(self.CATEGORIES) if valued and recent else set()
        assert self.rejected_by_fill_null(strategy, categories=True) == expected

    @pytest.mark.parametrize("strategy", sorted(RefusedKinds.STRATEGIES))
    def test_the_checker_reports_the_kinds_that_each_strategy_rejects(self, strategy: str) -> None:
        """The frame `fill_null` reports a declared column of each kind that polars rejects."""
        probes = {**self.SCALAR, **self.NESTED, **self.CATEGORIES}
        rejected = self.rejected_by_fill_null(strategy)
        rejected |= self.rejected_by_fill_null(strategy, categories=True)
        assert RefusedKinds().of(strategy) & set(probes) == rejected

    def test_a_whole_frame_cast_rejects_two_of_the_three_nested_types(self) -> None:
        """`Struct` casts to text where `List` and `Array` raise.

        So the gate cuts across the nested family: a promise naming that
        family is wrong in both directions, and only a set of the two
        types that fail is right.
        """
        casts = {name: self._casts_to_text(series) for name, series in self.NESTED.items()}
        assert casts == {"List": False, "Array": False, "Struct": True}

    def _casts_to_text(self, series: pl.Series) -> bool:
        """Answer whether a column of this type casts to text.

        Args:
            series: A column of the type under test.

        Returns:
            True when the cast succeeds.
        """
        try:
            pl.DataFrame({"x": series}).cast(pl.String)
        except pl.exceptions.PolarsError:
            return False
        return True

    def test_filling_with_a_value_is_total_and_keeps_every_type_but_null(self) -> None:
        """A value whose type does not match is simply not filled.

        So this is not a supertype join, and it needs no promise: an
        `Int64` column filled with text stays `Int64`. From polars 1.40.0, a
        `Null` column takes the type of the value.
        """
        nulls = TestFrameFill.NULLS
        for name, series in {**self.SCALAR, **self.NESTED}.items():
            filled = pl.DataFrame({name: series}).fill_null("z")
            expected = "String" if name in nulls else str(series.dtype)
            assert str(filled.schema[name]) == expected


class TestFrameFill:
    """`df.fill_null(...)` fills the columns that the Python type of its value picks.

    polars reads the type of the value to pick the filled columns: a
    number fills the numeric kinds of its list, a `str` fills `String`
    and `Categorical`, a `bool`, `date`, `time`, `timedelta` or
    `datetime` fills its own kind, and from polars 1.40.0 each also fills
    `Null` (`NULLS`). The filled columns run `fill_null` as an expression,
    so a number joins each column with its bucket (`D-44`). An expression,
    or a value of any other type, fills every column through
    `select(pl.all().fill_null(value))`. A column of a kind that the value
    does not pick keeps its type and its nulls.
    """

    # One column of each kind, with a value and a null, so a strategy
    # that raises on a null meets one.
    COLUMNS: ClassVar[dict[str, pl.Series]] = {
        "Int8": pl.Series([1, None], dtype=pl.Int8),
        "Int16": pl.Series([1, None], dtype=pl.Int16),
        "Int32": pl.Series([1, None], dtype=pl.Int32),
        "Int64": pl.Series([1, None], dtype=pl.Int64),
        "Int128": pl.Series([1, None], dtype=pl.Int128),
        "UInt8": pl.Series([1, None], dtype=pl.UInt8),
        "UInt16": pl.Series([1, None], dtype=pl.UInt16),
        "UInt32": pl.Series([1, None], dtype=pl.UInt32),
        "UInt64": pl.Series([1, None], dtype=pl.UInt64),
        "UInt128": pl.Series([1, None], dtype=pl.UInt128),
        "Float16": pl.Series([1.0, None], dtype=pl.Float16),
        "Float32": pl.Series([1.0, None], dtype=pl.Float32),
        "Float64": pl.Series([1.0, None], dtype=pl.Float64),
        "Decimal": pl.Series([None, None], dtype=pl.Decimal(10, 2)),
        "Boolean": pl.Series([True, None], dtype=pl.Boolean),
        "String": pl.Series(["a", None], dtype=pl.String),
        "Categorical": pl.Series(["a", None], dtype=pl.Categorical),
        "Enum": pl.Series(["a", None], dtype=pl.Enum(["a", "b"])),
        "Binary": pl.Series([b"a", None], dtype=pl.Binary),
        "Date": pl.Series([date(2020, 1, 1), None], dtype=pl.Date),
        "Time": pl.Series([time(1), None], dtype=pl.Time),
        "Duration": pl.Series([timedelta(1), None], dtype=pl.Duration("ns")),
        "Duration_ms": pl.Series([timedelta(1), None], dtype=pl.Duration("ms")),
        "Datetime": pl.Series([datetime(2020, 1, 1), None], dtype=pl.Datetime("ns")),
        "Datetime_utc": pl.Series(
            [datetime(2020, 1, 1), None],
            dtype=pl.Datetime("ms", "UTC"),
        ),
        "Null": pl.Series([None, None], dtype=pl.Null),
        "List": pl.Series([[1], None], dtype=pl.List(pl.Int8)),
        "Array": pl.Series([[1], None], dtype=pl.Array(pl.Int8, 1)),
        "Struct": pl.Series([{"a": 1}, None], dtype=pl.Struct({"a": pl.Int8})),
    }

    # The kind `Null`, where the installed release fills it beside the kinds of a value.
    NULLS: ClassVar[frozenset[str]] = (
        frozenset({"Null"})
        if PolarsRelease.installed() >= PolarsRelease.parse("1.40.0")
        else frozenset()
    )

    # The kinds that a Python int or float fills, `Null` aside. `UInt128`
    # and `Float16` are numeric kinds that polars leaves out.
    NUMBER_KINDS: ClassVar[frozenset[str]] = frozenset({
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "Float32",
        "Float64",
        "Decimal",
    })

    # Each value that fills only columns of its own kinds, with those kinds
    # and the type that a `Null` column takes. Every filled column keeps
    # its own type.
    OWN_KINDS: ClassVar[dict[str, tuple[object, frozenset[str], str]]] = {
        "bool": (True, frozenset({"Boolean"}), "Boolean"),
        "str": ("z", frozenset({"String", "Categorical"}), "String"),
        "date": (date(2021, 1, 1), frozenset({"Date"}), "Date"),
        "time": (time(2), frozenset({"Time"}), "Time"),
    }

    NUMBERS: ClassVar[dict[str, object]] = {
        "int8": 1,
        "int16": 300,
        "neg_int8": -1,
        "int64": 2**40,
        "float": 1.5,
    }

    # What each strategy that can raise refuses, beside the strategies
    # `forward` and `backward`, which refuse nothing.
    REFUSED: ClassVar[dict[str, frozenset[str]]] = {
        "min": frozenset({"List", "Array", "Struct"}),
        "max": frozenset({"List", "Array", "Struct"}),
        "zero": frozenset({"List", "Array", "Struct", "Null"}),
        "one": frozenset({"List", "Array", "Struct", "String", "Binary", "Null"}),
        "mean": frozenset({"List", "Array", "Struct", "String", "Binary", "Boolean"}),
    }

    def run(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        expression: bool = False,
    ) -> pl.Series | None:
        """Run one fill on a frame of one column, and give the column after it.

        Args:
            name: The key of the column in `COLUMNS`.
            arguments: The arguments of `fill_null`.
            expression: Whether to run the expression form
                `select(pl.all().fill_null(...))` in place of the frame form.

        Returns:
            The column after the fill, or None where polars raises.

        Example:
            >>> TestFrameFill().run("UInt8", {"value": 300}).dtype
            UInt16
            >>> TestFrameFill().run("Null", {"strategy": "zero"}) is None
            True
        """
        frame = pl.DataFrame({"x": self.COLUMNS[name]})
        try:
            if expression:
                return frame.select(pl.all().fill_null(**arguments)).to_series()  # type: ignore[arg-type]
            return frame.fill_null(**arguments).to_series()  # type: ignore[arg-type]
        except pl.exceptions.PolarsError:
            return None

    def filled(self, name: str, value: object) -> tuple[str, bool]:
        """Fill one column with one value.

        Args:
            name: The key of the column in `COLUMNS`.
            value: The fill value.

        Returns:
            The rendering of the column's data type after the fill, and
            whether polars filled its nulls.

        Example:
            >>> TestFrameFill().filled("UInt128", 1)
            ('UInt128', False)
        """
        column = self.run(name, {"value": value})
        assert column is not None, (name, value)
        return str(column.dtype), column.null_count() == 0

    def null_fill(self, rendering: str) -> tuple[str, bool]:
        """Give what a fill gives a `Null` column: the type of the value where the release fills it.

        Args:
            rendering: The type that the value gives a `Null` column that it fills.

        Returns:
            The rendering of the type after the fill, and whether polars filled the nulls.

        Example:
            >>> TestFrameFill().null_fill("Date") in {("Date", True), ("Null", False)}
            True
        """
        return (rendering, True) if self.NULLS else ("Null", False)

    def test_a_number_fills_its_listed_kinds_and_null(self) -> None:
        """A number fills the numeric kinds of its list, and `Null`, and no other kind."""
        for name, value in product(self.COLUMNS, self.NUMBERS.values()):
            _, done = self.filled(name, value)
            assert done == (name in self.NUMBER_KINDS | self.NULLS), (name, value)

    def test_a_number_joins_each_filled_column_with_its_value(self) -> None:
        """Each filled column answers what the rules give for `pl.col(...).fill_null(value)`."""
        kinds = sorted(self.NUMBER_KINDS - {"Decimal"} | self.NULLS)
        for name, (bucket, value) in product(kinds, self.NUMBERS.items()):
            tree = ExpressionTree().of(pl.col("x").fill_null(value))
            leaf = ColumnLeaf(DType(name), RowForm.PER_ROW)
            answer = TreeEvaluation({"x": leaf}, GroupBy((3, 2))).answers(tree).run_select
            assert isinstance(answer, DType), (name, bucket)
            assert self.filled(name, value)[0] == answer.render(), (name, bucket)

    def test_a_number_keeps_the_scale_of_a_decimal(self) -> None:
        """An integer widens a `Decimal` to 38 digits, and a float gives `Float64`."""
        assert self.filled("Decimal", 1)[0] == "Decimal(precision=38, scale=2)"
        assert self.filled("Decimal", 1.5)[0] == "Float64"

    def test_a_number_leaves_uint128_and_float16_alone(self) -> None:
        """Two numeric kinds are not in the list, so a number that would change them does not."""
        assert self.filled("UInt128", 1.5) == ("UInt128", False)
        assert self.filled("UInt128", -1) == ("UInt128", False)
        assert self.filled("Float16", 1.5) == ("Float16", False)

    def test_matches_supertype_false_fills_one_kind(self) -> None:
        """Without `matches_supertype`, an int fills `Int64` and a float `Float64`, with `Null`."""
        for name, value in product(self.COLUMNS, (1, 300, 1.5)):
            kind = "Int64" if isinstance(value, int) else "Float64"
            column = self.run(name, {"value": value, "matches_supertype": False})
            assert column is not None
            assert (column.null_count() == 0) == (name in {kind} | self.NULLS), (name, value)
            expected = (
                self.filled(name, value)[0]
                if name in {kind} | self.NULLS
                else str(self.COLUMNS[name].dtype)
            )
            assert str(column.dtype) == expected, (name, value)

    def test_a_value_of_one_kind_fills_that_kind_and_null(self) -> None:
        """A `bool`, `str`, `date` or `time` fills its kinds, which keep their type, and `Null`."""
        for value, kinds, null in self.OWN_KINDS.values():
            for name, series in self.COLUMNS.items():
                rendering, done = self.filled(name, value)
                assert done == (name in kinds | self.NULLS), (name, value)
                assert rendering == (null if name in self.NULLS else str(series.dtype)), (
                    name,
                    value,
                )

    def test_a_timedelta_fills_each_duration_at_the_coarser_unit(self) -> None:
        """A `timedelta` is a `Duration('us')`, and each filled column takes the coarser unit."""
        span = timedelta(2)
        assert self.filled("Duration", span) == ("Duration(time_unit='us')", True)
        assert self.filled("Duration_ms", span) == ("Duration(time_unit='ms')", True)
        assert self.filled("Null", span) == self.null_fill("Duration(time_unit='us')")
        assert self.filled("Datetime", span) == ("Datetime(time_unit='ns', time_zone=None)", False)

    def test_a_datetime_fills_each_datetime_of_its_zone(self) -> None:
        """A `datetime` fills a `Datetime` of its own zone at the coarser unit, and `Null`."""
        naive = datetime(2021, 1, 1)
        aware = datetime(2021, 1, 1, tzinfo=UTC)
        assert self.filled("Datetime", naive) == ("Datetime(time_unit='us', time_zone=None)", True)
        naive_null = self.null_fill("Datetime(time_unit='us', time_zone=None)")
        assert self.filled("Null", naive) == naive_null
        utc = "Datetime(time_unit='ms', time_zone='UTC')"
        assert self.filled("Datetime_utc", aware) == (utc, True)
        aware_null = self.null_fill("Datetime(time_unit='us', time_zone='UTC')")
        assert self.filled("Null", aware) == aware_null
        assert self.filled("Date", naive) == ("Date", False)

    def test_a_datetime_of_another_zone_raises(self) -> None:
        """A `Datetime` column of another zone than the value makes polars raise `SchemaError`."""
        with pytest.raises(pl.exceptions.SchemaError):
            pl.DataFrame({"x": self.COLUMNS["Datetime_utc"]}).fill_null(datetime(2021, 1, 1))
        aware = datetime(2021, 1, 1, tzinfo=UTC)
        with pytest.raises(pl.exceptions.SchemaError):
            pl.DataFrame({"x": self.COLUMNS["Datetime"]}).fill_null(aware)

    def test_an_expression_fills_every_column_as_the_expression_form(self) -> None:
        """`pl.lit(v)`, a `Decimal` and `bytes` fill every column, as `pl.all().fill_null(v)` does.

        `pl.lit(1)` is therefore not the plain `1`: it fills a `Boolean`
        column, which then joins with the literal into `Int32`.
        """
        values = (pl.lit(1), pl.lit("a"), pl.lit(True), Decimal("1.5"), b"a")
        for name, value in product(self.COLUMNS, values):
            frame = self.run(name, {"value": value})
            expression = self.run(name, {"value": value}, expression=True)
            framed = None if frame is None else str(frame.dtype)
            expressed = None if expression is None else str(expression.dtype)
            assert framed == expressed, (name, value)
        assert self.filled("Boolean", pl.lit(1)) == ("Int32", True)
        assert self.filled("Boolean", 1) == ("Boolean", False)

    def test_forward_and_backward_keep_every_column(self) -> None:
        """The two strategies that carry a value raise on no kind and change no type."""
        for name, strategy in product(self.COLUMNS, ("forward", "backward")):
            column = self.run(name, {"strategy": strategy, "limit": 1})
            assert column is not None
            assert column.dtype == self.COLUMNS[name].dtype, (name, strategy)

    def test_each_other_strategy_raises_on_its_kinds_only(self) -> None:
        """`min`, `max`, `zero`, `one` and `mean` raise on their kinds and keep every other type.

        From polars 1.44.0, `zero`, `one` and `mean` also raise on a categorical type.
        """
        recent = PolarsRelease.installed() >= PolarsRelease.parse("1.44.0")
        for name, (strategy, kinds) in product(self.COLUMNS, self.REFUSED.items()):
            valued = strategy in {"zero", "one", "mean"}
            refused = kinds | {"Categorical", "Enum"} if valued and recent else kinds
            kind = name.split("_", maxsplit=1)[0]
            column = self.run(name, {"strategy": strategy})
            if kind in refused:
                assert column is None, (name, strategy)
            else:
                assert column is not None, (name, strategy)
                assert column.dtype == self.COLUMNS[name].dtype, (name, strategy)

    def test_a_strategy_raises_only_on_a_null(self) -> None:
        """A column with no null keeps its type, even under a strategy that refuses its kind."""
        frame = pl.DataFrame({"x": pl.Series([[1]], dtype=pl.List(pl.Int8))})
        assert frame.fill_null(strategy="min").schema["x"] == pl.List(pl.Int8)

    @pytest.mark.parametrize(
        "arguments",
        [
            {},
            {"value": None},
            {"value": 1, "strategy": "forward"},
            {"strategy": "min", "limit": 1},
            {"value": 1, "limit": 1},
        ],
    )
    def test_three_call_forms_raise_before_any_data(self, arguments: dict[str, object]) -> None:
        """Three call forms raise `ValueError` at the call.

        They are: no value and no strategy, a value with a strategy, and
        a `limit` without the strategy `forward` or `backward`.
        """
        frame = pl.DataFrame({"x": [1, None]})
        with pytest.raises(ValueError, match="specify"):
            frame.lazy().fill_null(**arguments)  # type: ignore[arg-type]


class TestFrameFillNan:
    """`df.fill_nan(value)` fills the float columns as `cs.float().fill_nan(value)` does.

    polars wraps a value that is not an expression in `pl.lit(...)`, and
    runs the expression form of `fill_nan` on each `Float16`, `Float32`
    and `Float64` column. Every other column keeps its type. So a float
    column joins with the value: `"a"` turns it into `String`.
    """

    FLOATS: ClassVar[frozenset[str]] = frozenset({"Float16", "Float32", "Float64"})

    VALUES: ClassVar[tuple[object, ...]] = (
        1,
        -1,
        300,
        2**40,
        2**63,
        1.5,
        float("nan"),
        "a",
        True,
        None,
        b"x",
        date(2020, 1, 1),
        datetime(2020, 1, 1),
        time(1),
        timedelta(1),
        Decimal("1.5"),
        pl.lit(1, dtype=pl.Int8),
        pl.lit("x"),
        pl.col("other"),
    )

    def schema(self, frame: pl.LazyFrame) -> dict[str, pl.DataType] | None:
        """Give the schema of a lazy frame, or None where polars raises.

        Args:
            frame: The lazy frame.

        Returns:
            The schema, or None.

        Example:
            >>> TestFrameFillNan().schema(pl.LazyFrame({"a": [1]}))
            {'a': Int64}
        """
        try:
            return dict(frame.collect_schema())
        except pl.exceptions.PolarsError:
            return None

    @pytest.mark.parametrize("kind", sorted(TestFrameFill.COLUMNS))
    def test_a_frame_fill_nan_is_a_fill_of_the_float_columns(self, kind: str) -> None:
        """The frame form and the float selection give one schema, or both raise."""
        frame = pl.LazyFrame({
            "x": TestFrameFill.COLUMNS[kind],
            "other": pl.Series([1, None], dtype=pl.Int8),
        })
        for value in self.VALUES:
            whole = self.schema(frame.fill_nan(value))  # type: ignore[arg-type]
            picked = self.schema(frame.with_columns(cs.float().fill_nan(value)))  # type: ignore[arg-type]
            assert whole == picked, (kind, value)

    def test_only_a_float_column_changes(self) -> None:
        """A text value turns a float column into `String` and keeps every other column."""
        frame = pl.DataFrame(TestFrameFill.COLUMNS)
        filled = frame.fill_nan("a")  # type: ignore[arg-type]
        changed = {name for name in frame.columns if filled.schema[name] != frame.schema[name]}
        assert changed == self.FLOATS
        assert all(filled.schema[name] == pl.String for name in changed)
        assert frame.fill_nan(1).schema == frame.schema


class TestRowOnlyGuards:
    """Three row-only calls that always fail, and are knowable before running."""

    FRAME = pl.DataFrame({"a": [1.0], "b": ["x"]})

    def test_a_non_boolean_filter_predicate_raises(self) -> None:
        """Filtering on a float is an error, decided from the schema alone."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.FRAME.filter(pl.col("a"))

    def test_an_empty_unique_subset_raises(self) -> None:
        """Grouping on no key at all is an error."""
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.unique(subset=[])

    def test_sampling_more_than_all_rows_raises_without_replacement(self) -> None:
        """A fraction above one needs replacement, and says so."""
        with pytest.raises(pl.exceptions.ShapeError):
            self.FRAME.sample(fraction=2.0)
        assert self.FRAME.sample(fraction=2.0, with_replacement=True).height == 2


class TestTemporalSelectors:
    """The temporal selectors with arguments pick single data types."""

    def frame(self) -> pl.DataFrame:
        """Build a frame of four temporal columns.

        Returns:
            Naive `us`, `ms` in UTC, `ns` in Asia/Tokyo, and a `Duration("ns")`.

        Example:
            >>> TestTemporalSelectors().frame().columns
            ['a', 'b', 'c', 'd']
        """
        stamp = datetime(2020, 1, 1)
        return pl.DataFrame([
            pl.Series("a", [stamp], dtype=pl.Datetime("us")),
            pl.Series("b", [stamp], dtype=pl.Datetime("ms", "UTC")),
            pl.Series("c", [stamp], dtype=pl.Datetime("ns", "Asia/Tokyo")),
            pl.Series("d", [timedelta(1)], dtype=pl.Duration("ns")),
        ])

    def test_listed_units_and_zones_pick_those_data_types(self) -> None:
        """A unit and a zone together pick exactly that data type."""
        frame = self.frame()
        assert frame.select(cs.datetime("us", None)).columns == ["a"]
        assert frame.select(cs.datetime(["us", "ms"], [None, "UTC"])).columns == ["a", "b"]
        assert frame.select(cs.duration("ns")).columns == ["d"]
        assert frame.select(cs.by_dtype(pl.Datetime("us"))).columns == ["a"]

    def test_a_data_type_with_parameters_picks_that_data_type_only(self) -> None:
        """`cs.duration("ms")` and `cs.by_dtype(pl.List(pl.Null))` pick one data type."""
        frame = pl.DataFrame([
            pl.Series("dus", [], dtype=pl.Duration("us")),
            pl.Series("dms", [], dtype=pl.Duration("ms")),
            pl.Series("ln", [], dtype=pl.List(pl.Null)),
            pl.Series("li", [], dtype=pl.List(pl.Int64)),
            pl.Series("e0", [], dtype=pl.Enum([])),
            pl.Series("ea", [], dtype=pl.Enum(["a"])),
            pl.Series("sa", [], dtype=pl.Struct({"a": pl.Int64})),
            pl.Series("sb", [], dtype=pl.Struct({"b": pl.Int64})),
        ])
        assert frame.select(cs.duration("ms")).columns == ["dms"]
        assert frame.select(cs.by_dtype(pl.Duration("ms"))).columns == ["dms"]
        assert frame.select(cs.by_dtype(pl.List(pl.Null))).columns == ["ln"]
        assert frame.select(cs.by_dtype(pl.Enum([]))).columns == ["e0"]
        assert frame.select(cs.by_dtype(pl.Struct({"a": pl.Int64}))).columns == ["sa"]

    def test_an_open_zone_picks_every_zone_of_the_unit(self) -> None:
        """`cs.datetime("ns")` picks the `ns` column whatever its zone."""
        frame = self.frame()
        assert frame.select(cs.datetime("ns")).columns == ["c"]
        assert frame.select(cs.datetime(time_zone="*")).columns == ["b", "c"]


class TestMarkerClassesAsDataTypes:
    """A marker class of astrolabe is an annotation, and polars cannot read it as a data type.

    `List[pl.Int64]` and `Datetime[Literal["ms"], None]` are generic
    aliases, `Unchecked` is no data type of polars, and the class
    `Decimal` of astrolabe has no precision. Polars refuses each of them
    wherever a call takes one data type as a value. A bare marker class
    other than `Decimal` subclasses a polars class, and polars reads it
    as that class in an expression and in a cast, but refuses it in a
    series and in the schema of a frame built from Python data.
    """

    REFUSED: ClassVar[dict[str, object]] = {
        "List[Int64]": frame.List[pl.Int64],
        "Array[Int64, 2]": frame.Array[pl.Int64, Literal[2]],
        "Struct[a]": frame.Struct[frame.Column[Literal["a"], pl.Int64]],
        "Datetime[ms]": frame.Datetime[Literal["ms"], None],
        "Duration[ms]": frame.Duration[Literal["ms"]],
        "Decimal[10, 2]": frame.Decimal[Literal[10], Literal[2]],
        "Enum[a]": frame.Enum[Literal["a"]],
        "Unchecked": frame.Unchecked,
        "Decimal": frame.Decimal,
    }

    def calls(self) -> dict[str, Callable[[Any], object]]:
        """Give each call that takes one data type as a value.

        Returns:
            A function for each call, which runs the call with one data
            type and gives the data type of the result.

        Example:
            >>> TestMarkerClassesAsDataTypes().calls()["cast"](pl.Int8)
            Int8
        """
        return {
            "cast": lambda t: pl.select(pl.lit(None).cast(t)).to_series().dtype,
            "series cast": lambda t: pl.Series("a", [None]).cast(t).dtype,
            "lit": lambda t: pl.select(pl.lit(None, dtype=t)).to_series().dtype,
            "repeat": lambda t: pl.select(pl.repeat(None, 1, dtype=t)).to_series().dtype,
            "struct": lambda t: pl.select(pl.struct(pl.lit(None), schema={"a": t})).dtypes[0],
            "map_batches": lambda t: self.batches(pl.col("a").map_batches(abs, return_dtype=t)),
            "series": lambda t: pl.Series("a", [None], dtype=t).dtype,
            "from_dict": lambda t: pl.from_dict({"a": [None]}, schema={"a": t}).dtypes[0],
            "read_csv": lambda t: pl.read_csv(io.StringIO("a\n1\n"), schema={"a": t}).dtypes[0],
        }

    def batches(self, expression: pl.Expr) -> object:
        """Give the data type that a lazy frame resolves for one expression.

        Args:
            expression: The expression, over an `Int64` column `a`.

        Returns:
            The data type of the result.

        Example:
            >>> TestMarkerClassesAsDataTypes().batches(pl.col("a").cast(pl.Int8))
            Int8
        """
        return pl.LazyFrame({"a": [1]}).select(expression).collect_schema()["a"]

    def raises(self, call: Callable[[Any], object], dtype: object) -> bool:
        """Answer whether one call raises for one data type.

        Args:
            call: The call, from `calls`.
            dtype: The data type to pass.

        Returns:
            True if polars raises `TypeError` or `ValueError`.

        Example:
            >>> TestMarkerClassesAsDataTypes().raises(lambda t: pl.Series([1]).cast(t), pl.Int8)
            False
        """
        try:
            call(dtype)
        except (TypeError, ValueError):
            return True
        return False

    def test_every_call_takes_a_data_type_of_polars(self) -> None:
        """The calls run, so a refusal below comes from the marker."""
        for name, call in self.calls().items():
            assert not self.raises(call, pl.Int64), name

    @pytest.mark.parametrize("marker", sorted(REFUSED))
    def test_a_marker_with_arguments_unchecked_and_decimal_are_refused(self, marker: str) -> None:
        """Every call refuses the marker, whatever the data."""
        for name, call in self.calls().items():
            assert self.raises(call, self.REFUSED[marker]), (marker, name)

    # Each bare marker class, its polars class, and one data type of the
    # same kind that is not the default instance.
    BARE: ClassVar[dict[str, tuple[Any, Any, pl.DataType]]] = {
        "List": (frame.List, pl.List, pl.List(pl.Int64)),
        "Array": (frame.Array, pl.Array, pl.Array(pl.Int64, 1)),
        "Struct": (frame.Struct, pl.Struct, pl.Struct({"a": pl.Int64})),
        "Datetime": (frame.Datetime, pl.Datetime, pl.Datetime("ms")),
        "Duration": (frame.Duration, pl.Duration, pl.Duration("ms")),
        "Enum": (frame.Enum, pl.Enum, pl.Enum(["a"])),
    }

    def test_a_bare_marker_reads_as_its_polars_class_in_an_expression(self) -> None:
        """`List` of astrolabe is `List(Null)` in a cast, and a series refuses it."""
        calls = self.calls()
        assert calls["cast"](frame.List) == pl.List(pl.Null)
        assert calls["lit"](frame.Datetime) == pl.Datetime("us")
        assert self.raises(calls["series"], frame.List)
        assert self.raises(calls["from_dict"], frame.Datetime)

    @pytest.mark.parametrize("marker", sorted(BARE))
    def test_a_bare_marker_reads_as_the_default_of_its_polars_class(self, marker: str) -> None:
        """Every call that reads one data type, other than a series, reads the polars class."""
        own, polars_class, _ = self.BARE[marker]
        calls = self.calls()
        calls["inner"] = lambda t: pl.select(pl.lit(None, dtype=pl.List(t))).to_series().dtype
        calls["field"] = lambda t: pl.select(pl.lit(None, dtype=pl.Struct({"a": t}))).dtypes[0]
        calls["from_dicts"] = lambda t: pl.from_dicts([{"a": None}], schema={"a": t}).dtypes[0]
        calls["read_ndjson"] = lambda t: pl.read_ndjson(
            io.StringIO('{"a": null}\n'), schema={"a": t}
        ).dtypes[0]
        for name in ("series", "from_dict", "read_csv"):
            del calls[name]
        for name, call in calls.items():
            assert call(own) == call(polars_class), (marker, name)

    @pytest.mark.parametrize("marker", sorted(BARE))
    def test_a_series_refuses_a_bare_marker_whatever_the_data(self, marker: str) -> None:
        """`pl.Series` raises `ValueError` for a bare marker, with data or without."""
        own, polars_class, _ = self.BARE[marker]
        for data in ([], [None]):
            assert pl.Series("a", data, dtype=polars_class).dtype == polars_class
            with pytest.raises(ValueError, match="cannot construct PySeries"):
                pl.Series("a", data, dtype=own)
        with pytest.raises(ValueError, match="cannot construct PySeries"):
            pl.Series("a", dtype=own)

    def test_a_frame_from_python_data_refuses_a_bare_marker_only_for_some_data(self) -> None:
        """The data decides: a list of values raises, and a series or a row reads the default."""
        schema = {"a": frame.Datetime}
        with pytest.raises(ValueError, match="cannot construct PySeries"):
            pl.from_dict({"a": [None]}, schema=schema)
        with pytest.raises(ValueError, match="cannot construct PySeries"):
            pl.from_records([[None]], schema=schema)
        with pytest.raises(ValueError, match="cannot construct PySeries"):
            pl.from_dicts([], schema=schema)
        default = [pl.Datetime("us")]
        assert pl.from_dict({"a": pl.Series([None])}, schema=schema).dtypes == default
        assert pl.from_records([{"a": None}], schema=schema).dtypes == default
        assert pl.from_dicts([{"a": None}], schema=schema).dtypes == default

    @pytest.mark.parametrize("marker", sorted(BARE))
    def test_a_selector_reads_a_bare_marker_as_the_default_instance(self, marker: str) -> None:
        """`cs.by_dtype(List)` picks `List(Null)`, and `cs.by_dtype(pl.List)` picks every list."""
        own, polars_class, other = self.BARE[marker]
        df = pl.DataFrame([
            pl.Series("default", [], dtype=polars_class),
            pl.Series("other", [], dtype=other),
        ])
        assert df.select(cs.by_dtype(polars_class)).columns == ["default", "other"]
        assert df.select(cs.by_dtype(own)).columns == ["default"]
        assert df.select(cs.all().exclude(own)).columns == ["other"]
        assert df.select(pl.all().exclude(own)).columns == ["other"]

    @pytest.mark.parametrize("marker", sorted(REFUSED))
    def test_a_selector_refuses_a_marker_with_arguments_unchecked_and_decimal(
        self, marker: str
    ) -> None:
        """`cs.by_dtype` and `.exclude` raise `TypeError` for each refused marker."""
        dtype = self.REFUSED[marker]
        df = pl.DataFrame({"a": [1]})
        with pytest.raises(TypeError):
            cs.by_dtype(dtype)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            df.select(cs.all().exclude(dtype))  # type: ignore[arg-type]

    @pytest.mark.parametrize("marker", sorted(set(REFUSED) - {"Unchecked", "Decimal"}))
    def test_a_nested_data_type_refuses_a_marker_with_arguments(self, marker: str) -> None:
        """`pl.List`, `pl.Array`, `pl.Struct` and `pl.Field` raise `TypeError` at the call."""
        dtype: Any = self.REFUSED[marker]
        with pytest.raises(TypeError, match="cannot parse input"):
            pl.List(dtype)
        with pytest.raises(TypeError, match="cannot parse input"):
            pl.Field("a", dtype)
        with pytest.raises(TypeError, match="cannot parse input"):
            pl.Array(dtype, 2)
        with pytest.raises(TypeError, match="cannot parse input"):
            pl.Struct({"a": dtype})

    def test_a_nested_data_type_keeps_unchecked_and_decimal_until_use(self) -> None:
        """`pl.List(Unchecked)` builds, and each use of it raises, as `pl.List(pl.Decimal)` does."""
        for dtype in (frame.Unchecked, frame.Decimal, pl.Decimal):
            for nested in (pl.List(dtype), pl.Struct([pl.Field("a", dtype)])):
                assert self.raises(self.calls()["lit"], nested)
                assert self.raises(self.calls()["series"], nested)


class TestUserClassesAsDataTypes:
    """A subclass of a polars data type that user code defines is read by its name, or refused.

    In a `dtype=`, a `return_dtype=`, a nested data type, a cast, a
    selector and the schema of a file reader, polars reads such a class,
    or an instance of it, by its name alone. A class named after a kind
    reads as the polars class of that name, whatever it subclasses. Every
    other name raises `TypeError`, and so do the names `Decimal` and
    `Extension`. `pl.Series` refuses every subclass of a kind other than
    `List`, `Array` and `Struct`, whatever its name and its data. It
    ignores a subclass of those three, so the values decide.
    """

    UNREAD: ClassVar[tuple[str, ...]] = ("Decimal", "Extension")
    NESTED: ClassVar[tuple[type[pl.DataType], ...]] = (pl.List, pl.Array, pl.Struct)

    def named(self, name: str, base: type[pl.DataType] = pl.Int64) -> Any:
        """Define a subclass of a polars data type, as user code does.

        Args:
            name: The name of the class.
            base: The polars class that it subclasses.

        Returns:
            The class.

        Example:
            >>> TestUserClassesAsDataTypes().named("Tagged").__name__
            'Tagged'
        """
        return type(name, (base,), {})

    def calls(self) -> dict[str, Callable[[Any], object]]:
        """Give each call that reads one data type by its name and runs for every kind.

        Returns:
            A function for each call, which runs the call with one data
            type and gives the data type of the result.

        Example:
            >>> TestUserClassesAsDataTypes().calls()["lit"](pl.Int8)
            Int8
        """
        return {
            "lit": lambda t: pl.select(pl.lit(None, dtype=t)).to_series().dtype,
            "inner": lambda t: pl.select(pl.lit(None, dtype=pl.List(t))).to_series().dtype,
            "map_batches": lambda t: (
                pl
                .LazyFrame({"a": [1]})
                .select(pl.col("a").map_batches(abs, return_dtype=t))
                .collect_schema()["a"]
            ),
        }

    def outcome(self, call: Callable[[Any], object], dtype: object) -> object:
        """Give what one call answers for one data type: the data type, or the error.

        Args:
            call: The call, from `calls`.
            dtype: The data type to pass.

        Returns:
            The data type of the result, or the name of the error that
            polars raises, a panic included.

        Example:
            >>> facts = TestUserClassesAsDataTypes()
            >>> facts.outcome(facts.calls()["lit"], pl.Int8)
            Int8
        """
        try:
            return call(dtype)
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt | SystemExit | GeneratorExit):
                raise
            return type(error).__name__

    def test_a_call_reads_a_user_class_by_its_name(self) -> None:
        """A class named after a kind is the polars class of that name, whatever it subclasses."""
        names = sorted(set(KindRoster().names()) - set(self.UNREAD))
        for name in names:
            polars_class = getattr(pl, name)
            for label, call in self.calls().items():
                expected = self.outcome(call, polars_class)
                assert self.outcome(call, self.named(name)) == expected, (name, label)
                assert self.outcome(call, self.named(name, pl.String)) == expected, (name, label)
        frame = pl.DataFrame({"a": [1]})
        assert frame.select(pl.col("a").cast(self.named("Float32", pl.String))).dtypes == [
            pl.Float32
        ]
        assert frame.select(cs.by_dtype(self.named("Int64"))).columns == ["a"]
        schema = {"a": self.named("Int64")}
        assert pl.read_csv(io.StringIO("a\n1\n"), schema=schema).dtypes == [pl.Int64]

    def test_a_call_reads_an_instance_by_its_name_and_its_parameters(self) -> None:
        """An instance of a class named after a kind reads as an instance of that kind."""
        stamp = self.named("Datetime", pl.Datetime)
        assert self.calls()["lit"](self.named("Int64")()) == pl.Int64
        assert self.calls()["lit"](stamp("ms")) == pl.Datetime("ms")
        assert self.calls()["lit"](stamp) == pl.Datetime("us")

    def test_a_call_refuses_every_other_name(self) -> None:
        """A name that no kind has, and `Decimal` and `Extension`, raise `TypeError`."""
        frame = pl.DataFrame({"a": [1]})
        refused = [self.named(name) for name in ("Tagged", "Utf8", *self.UNREAD)]
        for dtype in (*refused, self.named("Tagged")()):
            for call in self.calls().values():
                with pytest.raises(TypeError):
                    call(dtype)
            with pytest.raises(TypeError):
                frame.select(pl.col("a").cast(dtype))
            with pytest.raises(TypeError):
                cs.by_dtype(dtype)
            with pytest.raises(TypeError):
                pl.read_csv(io.StringIO("a\n1\n"), schema={"a": dtype})

    def test_from_dicts_reads_a_user_class_by_its_name_for_every_data(self) -> None:
        """`pl.from_dicts` raises for every other name, whatever the data and the base.

        A class named after a kind reads as the polars class of that name,
        whatever it subclasses.
        """
        data: list[list[dict[str, object]]] = [
            [{"a": 1}],
            [{"a": None}],
            [],
            [{}],
            [{"a": [1]}],
            [{"a": "x"}],
        ]
        bases: tuple[type[pl.DataType], ...] = (
            pl.Int64,
            pl.String,
            pl.Datetime,
            pl.Decimal,
            *self.NESTED,
        )
        names = ("Tagged", "Utf8", *self.UNREAD)
        for name, base, rows in product(names, bases, data):
            with pytest.raises((TypeError, ValueError)):
                pl.from_dicts(rows, schema={"a": self.named(name, base)})
        for rows in data[:2]:
            with pytest.raises(TypeError):
                pl.from_dicts(rows, schema={"a": self.named("Tagged")()})
            frame = pl.from_dicts(rows, schema={"a": self.named("Int64", pl.String)})
            assert frame.dtypes == [pl.Int64]

    def test_a_series_refuses_a_user_subclass_of_a_scalar_kind(self) -> None:
        """`pl.Series` raises `ValueError` for every name and every data."""
        for kind in KindRoster().classes():
            if kind in self.NESTED:
                continue
            for name in (kind.__name__, "Tagged"):
                dtype = self.named(name, kind)
                for data in ([], [None]):
                    with pytest.raises(ValueError, match="cannot construct PySeries"):
                        pl.Series("a", data, dtype=dtype)
                with pytest.raises(ValueError, match="cannot construct PySeries"):
                    pl.Series("a", dtype=dtype)

    def test_a_series_ignores_a_user_subclass_of_a_nested_kind(self) -> None:
        """`pl.Series` reads the data type from the values, and raises for nested values."""
        for kind in self.NESTED:
            dtype = self.named("Tagged", kind)
            assert pl.Series("a", [None], dtype=dtype).dtype == pl.Null
            assert pl.Series("a", ["x"], dtype=dtype).dtype == pl.String
            assert pl.Series("a", dtype=dtype).dtype == pl.Null
            with pytest.raises(TypeError):
                pl.Series("a", [[1]], dtype=dtype)


class TestThreadsDecideTheOrder:
    """The number of threads decides the order in which `mean_horizontal` adds five inputs."""

    SCRIPT: ClassVar[str] = (
        "import sys\n"
        "from decimal import Decimal\n"
        "import polars as pl\n"
        "frame = pl.DataFrame({\n"
        "    **{f'i{k}': [1] for k in range(3)},\n"
        "    'd': pl.Series([Decimal('1.5')], dtype=pl.Decimal(10, 2)),\n"
        "    'f': [1.5],\n"
        "})\n"
        "try:\n"
        "    print(frame.select(pl.mean_horizontal(*sys.argv[1:])).schema)\n"
        "except BaseException as error:\n"
        "    print(type(error).__name__)\n"
    )

    def answer(self, threads: int, columns: tuple[str, ...]) -> str:
        """Give what polars prints for one order of the inputs, with a number of threads.

        Args:
            threads: The number of threads of polars.
            columns: The names of the inputs, in their order.

        Returns:
            The schema, or the name of the error.
        """
        environment = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
        command = [sys.executable, "-c", self.SCRIPT, *columns]
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        return done.stdout.strip().splitlines()[-1]

    def test_one_order_panics_with_one_thread_and_answers_with_more(self) -> None:
        """The same call panics on one machine and gives `Float64` on another."""
        columns = ("i0", "i1", "i2", "d", "f")
        assert self.answer(1, columns) == "PanicException"
        assert self.answer(4, columns) == "Schema({'i0': Float64})"


class TestStructCastNeedsOneField:
    """Polars casts a value that is not a struct only to a struct of one field."""

    def test_a_number_casts_to_a_struct_of_one_field_only(self) -> None:
        """A struct of one field takes the number, and a struct of no field or two refuses it."""
        frame = pl.DataFrame({"b": [1]})
        one = pl.Struct({"x": pl.Int8})
        assert frame.select(pl.col("b").cast(one)).schema["b"] == one
        for target in (pl.Struct({}), pl.Struct({"x": pl.Int8, "y": pl.Int8})):
            with pytest.raises(pl.exceptions.InvalidOperationError, match="one field"):
                frame.select(pl.col("b").cast(target))


class TestWideningTakesTheSupertype:
    """The relaxed concats, `update` and `unpivot` widen a shared column to the supertype."""

    PAIRS: ClassVar[tuple[tuple[pl.DataType, pl.DataType, pl.DataType], ...]] = (
        (pl.Int32(), pl.Int64(), pl.Int64()),
        (pl.Int64(), pl.Int32(), pl.Int64()),
        (pl.Int64(), pl.String(), pl.String()),
        (pl.String(), pl.Int64(), pl.String()),
    )

    def frame(self, dtype: pl.DataType) -> pl.DataFrame:
        """Give a frame of a key `k` and a column `a` of one row of a data type.

        Args:
            dtype: The data type of `a`.

        Returns:
            The frame.
        """
        value = "x" if dtype == pl.String else 1
        return pl.DataFrame({"k": [1], "a": pl.Series([value], dtype=dtype)})

    @pytest.mark.parametrize(("left", "right", "widened"), PAIRS)
    def test_each_form_gives_the_supertype_in_either_order(
        self,
        left: pl.DataType,
        right: pl.DataType,
        widened: pl.DataType,
    ) -> None:
        """Each form gives the supertype of the two data types."""
        first, second = self.frame(left), self.frame(right)
        pair = [first, second]
        assert pl.concat(pair, how="vertical_relaxed").schema["a"] == widened
        assert pl.concat(pair, how="diagonal_relaxed").schema["a"] == widened
        assert first.update(second, on="k").schema["a"] == widened
        both = first.rename({"a": "b"}).join(second, on="k")
        assert both.unpivot(on=["b", "a"]).schema["value"] == widened

    STRUCTS: ClassVar[tuple[tuple[pl.DataType, pl.DataType, pl.DataType], ...]] = (
        (
            pl.Struct({"x": pl.Int8()}),
            pl.Struct({"x": pl.Int64(), "y": pl.Int64()}),
            pl.Struct({"x": pl.Int64(), "y": pl.Int64()}),
        ),
        (
            pl.Struct({"x": pl.Int64(), "y": pl.Int64()}),
            pl.Struct({"y": pl.String(), "x": pl.Int32()}),
            pl.Struct({"x": pl.Int64(), "y": pl.String()}),
        ),
        (
            pl.Struct({"y": pl.String(), "x": pl.Int32()}),
            pl.Struct({"x": pl.Int64(), "y": pl.Int64()}),
            pl.Struct({"y": pl.String(), "x": pl.Int64()}),
        ),
        (
            pl.Struct({"x": pl.Int8()}),
            pl.Struct({"y": pl.Int64(), "z": pl.String()}),
            pl.Struct({"y": pl.Int64(), "z": pl.String(), "x": pl.Int8()}),
        ),
    )

    @pytest.mark.parametrize(("left", "right", "widened"), STRUCTS, ids=str)
    def test_two_structs_widen_to_the_union_of_their_fields(
        self,
        left: pl.Struct,
        right: pl.Struct,
        widened: pl.Struct,
    ) -> None:
        """The fields of the longer struct, or of the first, then those it lacks, each widened."""
        first = pl.DataFrame({"a": pl.Series([None], dtype=left)})
        second = pl.DataFrame({"a": pl.Series([None], dtype=right)})
        assert pl.concat([first, second], how="diagonal_relaxed").schema["a"] == widened
        assert pl.concat([first, second], how="vertical_relaxed").schema["a"] == widened


class TestChunksDecideTheLength:
    """The chunks of the bound decide whether `clip` of a literal by a column answers or panics."""

    TREE: ClassVar[pl.Expr] = pl.lit(1).clip(pl.col("a"))

    def answer(self, frame: pl.DataFrame) -> str:
        """Give the schema that the `select` of the tree gives over a frame, or how it stops.

        Args:
            frame: The frame.

        Returns:
            The schema, or the name of the error.
        """
        try:
            return str(frame.select(self.TREE).schema)
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt | SystemExit | GeneratorExit):
                raise
            return type(error).__name__

    def test_one_chunk_gives_one_row_and_two_chunks_panic(self) -> None:
        """One chunk gives the length of the literal, and two chunks panic."""
        one = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        parts = [pl.DataFrame({"a": [1, 2]}), pl.DataFrame({"a": [3, 4, 5]})]
        two = pl.concat(parts, rechunk=False)
        assert one.select(self.TREE).height == 1
        assert self.answer(one) == "Schema({'literal': Int32})"
        assert self.answer(two) == "PanicException"
        assert two.lazy().select(self.TREE).collect_schema()["literal"] == pl.Int32


class TestRowCountWidths:
    """Each count of rows fits an integer of a fixed width, and polars raises past it."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame(schema={"k": pl.String, "x": pl.Int64})
    CALLS: ClassVar[dict[str, Any]] = {
        "DataFrame": FRAME,
        "LazyFrame": FRAME.lazy(),
        "GroupBy": FRAME.group_by("k"),
        "LazyGroupBy": FRAME.lazy().group_by("k"),
    }
    OTHERS: ClassVar[dict[tuple[str, str], dict[str, Any]]] = {
        ("top_k", "k"): {"by": "x"},
        ("bottom_k", "k"): {"by": "x"},
        ("gather_every", "offset"): {"n": 1},
        ("slice", "length"): {"offset": 0},
    }
    CASES: ClassVar[list[tuple[str, str, str]]] = [
        (kind, method, name)
        for kind, methods in RowCounts.KINDS.items()
        for method, ranges in methods.items()
        for name in ranges
    ]

    def run(self, kind: str, method: str, name: str, value: int) -> str | None:
        """Run one call with one count, and give the class of the error of polars.

        Args:
            kind: The kind of the receiver.
            method: The name of the method.
            name: The name of the count.
            value: The count.

        Returns:
            The class of the error, or `None` where the call runs.
        """
        keywords: dict[str, Any] = {**self.OTHERS.get((method, name), {}), name: value}
        try:
            result = getattr(self.CALLS[kind], method)(**keywords)
            if isinstance(result, pl.LazyFrame):
                result.collect()
        except (OverflowError, ValueError, pl.exceptions.PolarsError) as error:
            return type(error).__name__
        return None

    @pytest.mark.parametrize(("kind", "method", "name"), CASES)
    def test_the_checker_holds_the_range_of_each_count(
        self,
        kind: str,
        method: str,
        name: str,
    ) -> None:
        """A count at each end of its range runs, and one past it raises the stated error."""
        limits = RowCounts().ranges(kind, method)[name]
        assert self.run(kind, method, name, limits.high) is None
        assert self.run(kind, method, name, limits.high + 1) == limits.above
        if limits.zero is not None:
            assert self.run(kind, method, name, 0) == limits.zero
        if limits.low is not None:
            low = limits.low + 1 if limits.zero is not None else limits.low
            assert self.run(kind, method, name, low) is None
            assert self.run(kind, method, name, limits.low - 1) == limits.below


class TestDecimalStacks:
    """A vertical stack reads every `Decimal` as one type, and keeps the type of the first frame."""

    FIRST: ClassVar[pl.DataFrame] = pl.DataFrame({
        "d": pl.Series([Decimal(1)], dtype=pl.Decimal(10, 2)),
    })
    SECOND: ClassVar[pl.DataFrame] = pl.DataFrame({
        "d": pl.Series([Decimal(1)], dtype=pl.Decimal(38, 3)),
    })

    @pytest.mark.parametrize("how", ["vertical", "diagonal"])
    def test_a_strict_stack_keeps_the_first_type(
        self,
        how: Literal["vertical", "diagonal"],
    ) -> None:
        """The two types differ only in their parameters, so the stack runs."""
        stacked = pl.concat([self.FIRST, self.SECOND], how=how)
        assert stacked.schema["d"] == pl.Decimal(10, 2)
        assert self.FIRST.vstack(self.SECOND).schema["d"] == pl.Decimal(10, 2)

    @pytest.mark.parametrize("how", ["vertical_relaxed", "diagonal_relaxed"])
    def test_a_relaxed_stack_widens(
        self,
        how: Literal["vertical_relaxed", "diagonal_relaxed"],
    ) -> None:
        """The result takes the larger precision and the larger scale."""
        stacked = pl.concat([self.FIRST, self.SECOND], how=how)
        assert stacked.schema["d"] == pl.Decimal(38, 3)

    def test_an_align_and_a_sorted_merge_refuse(self) -> None:
        """An align strategy and `merge_sorted` compare the parameters too."""
        with pytest.raises(pl.exceptions.SchemaError):
            pl.concat([self.FIRST, self.SECOND], how="align")
        with pytest.raises(pl.exceptions.SchemaError):
            self.FIRST.merge_sorted(self.SECOND, key="d")
