"""Whole pipelines, traced through the models and run on polars.

Each scenario walks a realistic pipeline step by step: the model
predicts every expression step's output, polars executes the same
step, and the two must agree at every point. This is the seam test
the single-operation suites cannot give: frames, expressions, splits
and the lifting, meeting in one program.
"""

from __future__ import annotations

import polars as pl
import polars.selectors as cs

from spec.tests.axis_model import Grid
from spec.tests.fit_model import Abstraction, ExprType, Materializer
from spec.tests.model import NUMERIC, Always, ColumnSet, NameTest, OfType
from spec.tests.split_model import Split
from spec.tests.table_model import QUOTIENT

from astrolabe._rules.dtypes import DType
from astrolabe._rules.supertype import Supertype


class TestDeriveAggregateJoin:
    """Derive a column, aggregate a selection, join the result back."""

    TRADES = pl.DataFrame({
        "k": pl.Series([1, 1, 2], dtype=pl.Int64),
        "v": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float64),
        "s": pl.Series(["x", "y", "z"], dtype=pl.String),
    })

    def test_every_step_matches_the_model(self) -> None:
        """The pipeline's schema, predicted and executed, step by step."""
        doubled = ExprType(
            frozenset({"v"}),
            Grid.from_outputs(("v",), {("Float64",): "Float64"}),
            "z",
            "elementwise",
        )
        predicted = Materializer().result({"v": "Float64"}, doubled, "row")
        assert predicted == {"z": "Float64"}

        widened = self.TRADES.with_columns(z=(pl.col("v") * 2).alias("z"))
        assert isinstance(predicted, dict)
        assert str(widened.schema["z"]) == predicted["z"]

        grouped = widened.group_by("k").agg(cs.numeric())
        assert {n: str(t) for n, t in grouped.schema.items()} == {
            "k": "Int64",
            "v": "List(Float64)",
            "z": "List(Float64)",
        }

        joined = grouped.join(self.TRADES.select("k", "s").unique(), on="k")
        assert list(joined.schema) == ["k", "v", "z", "s"]


class TestChainFilterUnnest:
    """A four-branch chain, a filtered mask, and a struct spread."""

    FRAME = pl.DataFrame({
        "a": pl.Series([1, 2, 3], dtype=pl.Int32),
        "b": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
    })

    def test_the_fold_predicts_the_chain_and_the_rest_follows(self) -> None:
        """The supertype of the branches is the chain's schema, then the seams."""
        branches = (pl.Boolean, pl.Int32, pl.Int32, pl.Float64)
        predicted = Supertype().fold(DType.of(branch) for branch in branches).render()
        assert predicted == "Float64"

        chain = (
            pl
            .when(pl.col("a") > 1)
            .then(pl.lit(None, dtype=pl.Boolean))
            .when(pl.col("a") > 2)
            .then(pl.lit(None, dtype=pl.Int32))
            .when(pl.col("a") > 3)
            .then(pl.lit(None, dtype=pl.Int32))
            .otherwise(pl.lit(None, dtype=pl.Float64))
        )
        out = self.FRAME.select(chain.alias("c"))
        assert str(out.schema["c"]) == predicted

        filtered = self.FRAME.filter(pl.col("a") > 1)
        assert filtered.schema == self.FRAME.schema

        spread = self.FRAME.select(pl.struct("a", "b").alias("st")).select(
            pl.col("st").struct.unnest(),
        )  # type: ignore[wrong-column-shape]
        assert {n: str(t) for n, t in spread.schema.items()} == {
            "a": "Int32",
            "b": "Float64",
        }


class TestConcatSplitSelect:
    """Stack two frames, split the extras, select the numeric part."""

    def test_the_bound_part_is_what_the_selector_picks(self) -> None:
        """The split's numeric part equals the polars selection exactly."""
        top = pl.DataFrame({
            "k": pl.Series([1], dtype=pl.Int64),
            "xn": pl.Series([1], dtype=pl.Int32),
            "xs": pl.Series(["a"], dtype=pl.String),
        })
        stacked = pl.concat([top, top])
        assert stacked.schema == top.schema

        split = Split((OfType(NUMERIC), NameTest(Always())))
        extras = ColumnSet.of(xn="Int32", xs="String")
        numeric_part, rest = split.bound(extras)
        picked = stacked.select(cs.numeric().exclude("k"))
        assert tuple(picked.columns) == numeric_part.names()
        assert rest.names() == ("xs",)


class TestTheLiftingEndToEnd:
    """An `Unchecked` column's model answers, against every instance."""

    def test_the_recovered_answer_holds_at_every_running_instance(self) -> None:
        """The model recovers `UInt32`; every instance polars runs agrees.

        The expression's grid is `len_chars`' shape: defined on
        `String` alone. The model's lifted read at the marker answers
        `UInt32`, and the polars runs over every universe type either
        raise or answer exactly that.
        """
        grid = Grid.from_outputs(("p",), {("String",): "UInt32"})
        expr = ExprType(frozenset({"p"}), grid, "p", "elementwise")
        lifted = Materializer().result({"p": Abstraction.MARKER}, expr, "row")
        assert lifted == {"p": "UInt32"}

        for kind in QUOTIENT:
            observed = self.instance_answer(kind)
            assert observed in {"UInt32", "raises"}, kind

    def instance_answer(self, kind: str) -> str:
        """Run `len_chars` over one concrete instance of the column.

        Args:
            kind: The universe type the column holds this time.

        Returns:
            The output type's text, or "raises".

        Example:
            >>> TestTheLiftingEndToEnd().instance_answer("String")
            'UInt32'
        """
        try:
            series = self.instance_series(kind)
            out = pl.DataFrame({"p": series}).select(pl.col("p").str.len_chars())
            return str(out.schema["p"])
        except pl.exceptions.PolarsError:
            return "raises"

    def instance_series(self, kind: str) -> pl.Series:
        """Build a one-value column of one universe type.

        Args:
            kind: The universe type the column holds.

        Returns:
            The one-row series.

        Example:
            >>> TestTheLiftingEndToEnd().instance_series("Date").dtype
            Date
        """
        values: list[str] | list[bool] | list[int] = [1]
        if kind == "String":
            values = ["a"]
        if kind == "Boolean":
            values = [True]
        return pl.Series("p", values).cast(getattr(pl, kind)())
