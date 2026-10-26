"""What polars does with an expression at a frame, for the claims that the specification cites.

Each class measures one claim of `OPERATIONS.md` against the installed
polars: the context rule, the output name, the sites that consume an
expression, the rename of a selection, the spread of a struct, and the
functions whose output type only their code decides.
"""

from __future__ import annotations

import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import polars.selectors as cs
import pytest

from astrolabe._rules.dtypes import DType
from astrolabe._rules.kernels.lengths import ValueBuild
from astrolabe._rules.releases import PolarsRelease

if TYPE_CHECKING:
    from collections.abc import Callable


class TestContextRule:
    """One global rule relates the two contexts, keyed by the row form."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "k": [1, 1, 2],
        "a": [1, 2, 3],
        "s": ["x", "y", "z"],
    })

    def test_an_elementwise_output_implodes_into_a_list(self) -> None:
        """The aggregation context wraps an elementwise result in a list."""
        schema = self.FRAME.group_by("k").agg(pl.col("a") + 1).schema
        assert schema["a"] == pl.List(pl.Int64)

    def test_an_aggregating_output_keeps_its_type(self) -> None:
        """The aggregation context leaves a reduced result unwrapped."""
        schema = self.FRAME.group_by("k").agg(pl.col("a").sum()).schema
        assert schema["a"] == pl.Int64

    def test_a_literal_output_keeps_its_type(self) -> None:
        """The aggregation context leaves a literal unwrapped."""
        schema = self.FRAME.group_by("k").agg(pl.lit(9)).schema
        assert schema["literal"] == pl.Int32

    def test_an_elementwise_step_after_a_reduction_stays_aggregating(self) -> None:
        """`sum` then `abs` is still one value per group, unwrapped."""
        schema = self.FRAME.group_by("k").agg(pl.col("a").sum().abs()).schema
        assert schema["a"] == pl.Int64

    def test_a_window_closes_the_aggregation(self) -> None:
        """`sum().over(...)` is elementwise: full length in a `select`."""
        out = self.FRAME.select(pl.col("a").sum().over("k"))
        assert out.schema["a"] == pl.Int64
        assert out.height == 3

    def test_a_mixed_operation_broadcasts_and_is_elementwise(self) -> None:
        """Elementwise beside aggregating broadcasts the aggregate."""
        mixed = pl.col("a") - pl.col("a").mean()
        selected = self.FRAME.select(mixed)
        assert selected.schema["a"] == pl.Float64
        assert selected.height == 3
        assert self.FRAME.group_by("k").agg(mixed).schema["a"] == pl.List(pl.Float64)

    def test_a_literal_under_an_elementwise_step_stays_literal(self) -> None:
        """A negated literal is still one unwrapped value per group."""
        schema = self.FRAME.group_by("k").agg((-pl.lit(9)).alias("x")).schema
        assert schema["x"] == pl.Int32


class TestNamingAndBinding:
    """The keyword naming override, and the context-bound roots."""

    FRAME = pl.DataFrame({"a": [1, 2]})

    def test_a_keyword_argument_replaces_the_output_name_outermost(self) -> None:
        """`with_columns(z=e)` names the output `z`, an alias included."""
        plain = self.FRAME.with_columns(z=pl.col("a") + 1)
        assert list(plain.schema) == ["a", "z"]
        aliased = self.FRAME.with_columns(z=(pl.col("a") + 1).alias("q"))
        assert list(aliased.schema) == ["a", "z"]

    def test_the_element_root_is_bound_by_its_enclosing_call(self) -> None:
        """`pl.element()` types inside `list.eval` and errs outside it."""
        inside = self.FRAME.select(pl.col("a").implode().list.eval(pl.element() * 2))
        assert inside.schema["a"] == pl.List(pl.Int64)
        with pytest.raises(pl.exceptions.PolarsError):
            self.FRAME.select(pl.element())


class TestConsumerSites:
    """The sites that consume an expression without keeping its column."""

    FRAME = pl.DataFrame({"k": [1, 1, 2], "a": [1, 2, 3], "s": ["x", "y", "z"]})

    def test_a_filter_demands_a_boolean_at_schema_time(self) -> None:
        """A non-Boolean predicate raises before any data is read."""
        with pytest.raises(pl.exceptions.PolarsError):
            self.FRAME.lazy().filter(pl.col("a") + 1).collect_schema()

    def test_an_aggregating_predicate_broadcasts(self) -> None:
        """`filter` accepts a reduced Boolean: one value stands for all."""
        out = self.FRAME.filter(pl.col("a").sum() > 3)
        assert out.schema == self.FRAME.schema

    def test_a_sort_key_runs_at_any_type(self) -> None:
        """The key's cell must resolve; nothing else is demanded."""
        out = self.FRAME.sort(pl.col("s").str.len_chars())
        assert out.schema == self.FRAME.schema

    def test_a_group_by_expression_key_lands_under_its_output_name(self) -> None:
        """The key column joins the schema, named by the expression."""
        plain = self.FRAME.group_by(pl.col("a") % 2).agg(pl.col("s").first())
        assert list(plain.schema) == ["a", "s"]
        aliased = self.FRAME.group_by((pl.col("a") % 2).alias("parity")).agg(pl.col("s").first())
        assert list(aliased.schema) == ["parity", "s"]


class TestNameRewriteAdmissibility:
    """Why the name rewrites split: affixes stay injective, functions collide."""

    QUOTES = pl.DataFrame({"bid": [1.0], "ask": [2.0], "s": ["x"]})
    MIXED_CASE = pl.DataFrame({"A": [1], "a": [2]})

    def test_an_affix_rewrites_a_selection_exactly(self) -> None:
        """Suffixing every matched column keeps the names distinct."""
        out = self.QUOTES.select(cs.numeric().name.suffix("_r"))
        assert list(out.schema) == ["bid_r", "ask_r"]

    def test_a_whole_name_function_can_collide_two_columns(self) -> None:
        """Lowercasing a selection folds two known names into one.

        This is the measured reason the admissibility line falls
        where it does: an affix keeps distinct names distinct, so it
        is exact over unknown columns, where a whole-name function
        can collide two of them, so no single result type exists.
        """
        with pytest.raises(pl.exceptions.DuplicateError):
            self.MIXED_CASE.select(cs.all().name.to_lowercase())

    def test_a_whole_name_function_is_fine_on_one_known_name(self) -> None:
        """On a single named column the checker can compute the string."""
        out = self.MIXED_CASE.select(pl.col("A").name.to_lowercase())
        assert list(out.schema) == ["a"]

    def test_a_named_output_colliding_with_a_selection_raises(self) -> None:
        """A selection's outputs and a named expression share one space.

        The collision is polars' own error, so a checker over a
        remainder must prove the name absent from what the selector
        picks, the same provable-absence question the frame effects
        already decide.
        """
        with pytest.raises(pl.exceptions.DuplicateError):
            self.QUOTES.select(cs.numeric(), pl.col("s").str.len_chars().alias("bid"))
        fresh = self.QUOTES.select(cs.numeric(), pl.col("s").str.len_chars().alias("n"))
        assert list(fresh.schema) == ["bid", "ask", "n"]

    def test_a_fixed_name_over_no_picked_column_gives_no_column(self) -> None:
        """A keyword or an alias over a selection that picks nothing gives no output."""
        assert not self.QUOTES.select(x=cs.temporal()).schema
        assert not self.QUOTES.select(cs.temporal().alias("x")).schema


class TestStructUnnest:
    """The one multi-output member outside the selector roots."""

    FRAME = pl.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}).select(pl.struct("a", "b").alias("st"))

    def test_a_select_spreads_the_fields_into_columns(self) -> None:
        """The outputs are the struct type's fields, name and type."""
        out = self.FRAME.select(pl.col("st").struct.unnest())  # type: ignore[wrong-column-shape]
        assert dict(out.schema) == {"a": pl.Int64, "b": pl.Float64}

    def test_a_with_columns_overlays_the_fields_beside_the_struct(self) -> None:
        """The struct column stays, and the fields land beside it."""
        out = self.FRAME.with_columns(pl.col("st").struct.unnest())  # type: ignore[wrong-column-shape]
        assert list(out.schema) == ["st", "a", "b"]

    def test_an_alias_over_the_several_outputs_raises(self) -> None:
        """One alias cannot name several outputs."""
        with pytest.raises(pl.exceptions.PolarsError):
            self.FRAME.select(pl.col("st").struct.unnest().alias("z"))  # type: ignore[wrong-column-shape]


class TestSelectionRoots:
    """A selection evaluates once per matched column, in both contexts."""

    FRAME = pl.DataFrame({"k": [1, 1], "a": [1, 2], "b": [1.5, 2.5], "s": ["x", "y"]})

    def test_a_selection_keeps_every_column_in_a_select(self) -> None:
        """`pl.all()` gives the whole frame back, column for column."""
        assert self.FRAME.select(pl.all()).schema == self.FRAME.schema

    def test_a_selection_implodes_each_column_in_an_agg(self) -> None:
        """The context rule applies per matched column of a selection.

        `pl.all()` and `cs.numeric()` are elementwise per column, so
        the aggregation context wraps each matched column's own type
        in a `List`, the group key excluded.
        """
        schema = self.FRAME.group_by("k").agg(pl.all()).schema
        assert dict(schema) == {
            "k": pl.Int64,
            "a": pl.List(pl.Int64),
            "b": pl.List(pl.Float64),
            "s": pl.List(pl.String),
        }
        numeric = self.FRAME.group_by("k").agg(cs.numeric()).schema
        assert dict(numeric) == {
            "k": pl.Int64,
            "a": pl.List(pl.Int64),
            "b": pl.List(pl.Float64),
        }

    def test_an_aggregating_selection_stays_unwrapped(self) -> None:
        """A reducer applied to a selection is unwrapped per column."""
        schema = self.FRAME.group_by("k").agg(cs.numeric().sum()).schema
        assert dict(schema) == {"k": pl.Int64, "a": pl.Int64, "b": pl.Float64}

    def test_aliasing_a_several_column_selection_raises(self) -> None:
        """One alias over several outputs is a duplicate-name error.

        This is the artifact that once put a wrong "raises" row in the
        constructor census: the probe aliased every constructor, and
        the alias, not the selection, is what raises.
        """
        with pytest.raises(pl.exceptions.PolarsError):
            self.FRAME.group_by("k").agg(pl.all().alias("o"))
        with pytest.raises(pl.exceptions.PolarsError):
            self.FRAME.select(pl.all().alias("o"))


class TestMetaIntrospection:
    """Polars itself computes an expression's needs' names and output name."""

    def test_roots_and_output_names_for_column_expressions(self) -> None:
        """`meta` answers with the residue's name half, case by case."""
        summed = (pl.col("a") + pl.col("b")).alias("c")
        assert (summed.meta.root_names(), summed.meta.output_name()) == (["a", "b"], "c")
        assert pl.col("a").sum().meta.output_name() == "a"
        assert (pl.len().meta.root_names(), pl.len().meta.output_name()) == ([], "len")
        assert pl.lit(1).meta.output_name() == "literal"

    def test_a_condition_contributes_needs_and_not_the_name(self) -> None:
        """The `when` condition's column is a root; the branch names the output."""
        chained = pl.when(pl.col("flag")).then(pl.col("a")).otherwise(0)
        assert sorted(chained.meta.root_names()) == ["a", "flag"]
        assert chained.meta.output_name() == "a"

    def test_a_computable_rename_rewrites_the_name(self) -> None:
        """`name.suffix` gives a name the checker can compute from a literal."""
        assert pl.col("a").name.suffix("_r").meta.output_name() == "a_r"

    def test_a_selection_is_marked_and_has_no_single_name(self) -> None:
        """Selector-rooted expressions carry the multi-output flags instead."""
        selection = cs.numeric().mean()
        assert selection.meta.has_multiple_outputs()
        assert (cs.numeric() * 2).meta.has_multiple_outputs()
        assert pl.col("a", "b").meta.is_column_selection()
        with pytest.raises(pl.exceptions.ComputeError):
            selection.meta.output_name()


class TestNameNamespace:
    """The renaming namespace splits along the admissibility line."""

    MEMBERS: ClassVar[list[str]] = [
        "keep",
        "map",
        "map_fields",
        "prefix",
        "prefix_fields",
        "replace",
        "suffix",
        "suffix_fields",
        "to_lowercase",
        "to_uppercase",
    ]

    def test_the_namespace_holds_exactly_these_members(self) -> None:
        """The census's `name` rows are the namespace's members, two more from polars 2.0.0."""
        found = [n for n in dir(pl.col("x").name) if not n.startswith("_")]
        added = ["strip_prefix", "strip_suffix"]
        newer = PolarsRelease.installed() >= PolarsRelease.parse("2.0.0")
        members = sorted([*self.MEMBERS, *added]) if newer else self.MEMBERS
        assert sorted(found) == members


class TestOpaquePython:
    """Two members whose output type only running Python code can decide."""

    def test_map_elements_output_follows_the_returned_value(self) -> None:
        """Two lambdas with one signature give two schemas.

        This is the measured proof that the output type is decided by
        code, not by the call's shape: nothing a checker can read
        distinguishes the two calls.
        """
        frame = pl.LazyFrame({"a": [1, 2]})
        text = frame.select(pl.col("a").map_elements(lambda _: "s"))
        number = frame.select(pl.col("a").map_elements(lambda _: 1.5))
        assert text.collect_schema()["a"] == pl.String
        assert number.collect_schema()["a"] == pl.Float64

    def test_a_declared_return_dtype_is_the_output_type(self) -> None:
        """With `return_dtype=` given, the output type is readable."""
        frame = pl.LazyFrame({"a": [1, 2]})
        declared = pl.col("a").map_elements(lambda _: "s", return_dtype=pl.String)
        assert frame.select(declared).collect_schema()["a"] == pl.String

    def test_polars_carries_an_unknown_type_of_its_own(self) -> None:
        """The library's `Unknown` is the runtime shadow of the same gap."""
        assert isinstance(pl.Unknown, type)


class TestComparisonAgainstZero:
    """The executed domain of `> 0`, which the discussion of a declared region quotes.

    A declared region narrower than this domain is an author's choice.
    This test keeps the document's statement of the domain itself a fact.
    """

    SERIES: ClassVar[dict[str, pl.Series]] = {
        "Boolean": pl.Series([True]),
        "Int8": pl.Series([1], dtype=pl.Int8),
        "Int64": pl.Series([1], dtype=pl.Int64),
        "Int128": pl.Series([1], dtype=pl.Int128),
        "UInt8": pl.Series([1], dtype=pl.UInt8),
        "UInt128": pl.Series([1], dtype=pl.UInt128),
        "Float32": pl.Series([1.0], dtype=pl.Float32),
        "Float64": pl.Series([1.0]),
        "Decimal": pl.Series([Decimal("1.5")], dtype=pl.Decimal(10, 2)),
        "String": pl.Series(["a"]),
        "Binary": pl.Series([b"a"]),
        "Date": pl.Series([date(2020, 1, 1)]),
        "Datetime": pl.Series([datetime(2020, 1, 1)]),
        "Duration": pl.Series([timedelta(1)]),
        "Time": pl.Series([time(1)]),
        "Categorical": pl.Series(["a"], dtype=pl.Categorical),
        "Enum": pl.Series(["a"], dtype=pl.Enum(["a"])),
        "Null": pl.Series([None]),
        "List": pl.Series([[1]]),
        "Array": pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)),
        "Struct": pl.Series([{"a": 1}]),
    }
    REFUSED: ClassVar[frozenset[str]] = frozenset({
        "String",
        "Binary",
        "Categorical",
        "Enum",
        "List",
        "Array",
        "Struct",
    })

    def test_runs_on_every_kind_except_text_categories_and_nested_kinds(self) -> None:
        """`x > 0` runs for every kind but the text, category and nested kinds."""
        for label, series in self.SERIES.items():
            expected = "raises" if label in self.REFUSED else "Boolean"
            assert self.compared(pl.DataFrame({"x": series})) == expected, label

    def compared(self, frame: pl.DataFrame) -> str:
        """Give the outcome of comparing one frame's `x` against zero.

        Args:
            frame: A frame holding the column `x` to compare.

        Returns:
            The output type's text, or "raises".

        Example:
            >>> TestComparisonAgainstZero().compared(pl.DataFrame({"x": [1]}))
            'Boolean'
        """
        try:
            return str(frame.select((pl.col("x") > 0).alias("o")).schema["o"])
        except pl.exceptions.PolarsError:
            return "raises"


class TestSumWidensSmallIntegers:
    """`sum`'s output type is a real function of its input type.

    The discussion of a declared outcome quotes this: a signature
    promising "sum gives your type back" is false at `Int8`, so the
    checker's refusal of that promise is polars' own behavior.
    """

    WIDENED: ClassVar[dict[pl.DataType, pl.DataType]] = {
        pl.Int8(): pl.Int64(),
        pl.Int16(): pl.Int64(),
        pl.UInt8(): pl.Int64(),
        pl.UInt16(): pl.Int64(),
    }

    KEPT: ClassVar[tuple[pl.DataType, ...]] = (
        pl.Int32(),
        pl.Int64(),
        pl.UInt32(),
        pl.UInt64(),
        pl.Float32(),
        pl.Float64(),
    )

    def test_small_integers_widen_to_int64(self) -> None:
        """The four narrow integer types all sum to `Int64`."""
        for narrow, wide in self.WIDENED.items():
            assert self.summed(narrow) == wide, narrow

    def test_wider_numerics_keep_their_type(self) -> None:
        """From 32 bits up, `sum` is the identity on numerics."""
        for dtype in self.KEPT:
            assert self.summed(dtype) == dtype, dtype

    def summed(self, dtype: pl.DataType) -> pl.DataType:
        """Give the type `sum` produces for one input type.

        Args:
            dtype: The input column's type.

        Returns:
            The summed column's type.

        Example:
            >>> TestSumWidensSmallIntegers().summed(pl.Int8())
            Int64
        """
        frame = pl.DataFrame({"x": pl.Series([1, 2], dtype=dtype)})
        return frame.select(pl.col("x").sum()).schema["x"]


class TestFitAtEachDataType:
    """An argument that fits a declaration at some data types can differ at the others."""

    UNITS: ClassVar[tuple[str, ...]] = ("ns", "us", "ms")

    def test_a_unit_cast_agrees_with_the_column_at_one_unit_only(self) -> None:
        """`cast_time_unit("ms")` gives the unit of the column only where the column is `ms`."""
        agreeing = []
        for unit in self.UNITS:
            column = pl.Series("t", [datetime(2020, 1, 1)], dtype=pl.Datetime(unit))  # type: ignore[arg-type]
            frame = pl.DataFrame([column])
            cast = frame.select(pl.col("t").dt.cast_time_unit("ms")).schema["t"]
            if cast == frame.schema["t"]:
                agreeing.append(unit)
        assert agreeing == ["ms"]

    def test_a_cast_that_fits_a_region_gives_its_target_outside_it(self) -> None:
        """A cast to `Float32` gives `Float32` at an `Int64` column, not the type of the column."""
        frame = pl.DataFrame({"price": [1]})
        schema = frame.select(pl.col("price").cast(pl.Float32) + 1).schema
        assert frame.schema["price"] == pl.Int64
        assert schema["price"] == pl.Float32


class TestRemainderKindSplit:
    """A selection can run at some data types of a kind and raise at others."""

    STAMP: ClassVar[datetime] = datetime(2020, 1, 1, tzinfo=UTC)
    NAIVE: ClassVar[pl.DataFrame] = pl.DataFrame({"t": [datetime(2020, 1, 1)]})
    ZONED: ClassVar[pl.DataFrame] = NAIVE.with_columns(pl.col("t").dt.replace_time_zone("UTC"))

    def test_a_zoned_fill_raises_at_a_naive_column_only(self) -> None:
        """`cs.datetime().fill_null(value)` with a zoned value runs only where the zones agree."""
        with pytest.raises(pl.exceptions.SchemaError):
            self.NAIVE.with_columns(cs.datetime().fill_null(self.STAMP))
        filled = self.ZONED.with_columns(cs.datetime().fill_null(self.STAMP))
        assert filled.schema["t"] == pl.Datetime("us", "UTC")

    def test_a_zoned_comparison_raises_at_a_naive_column_only(self) -> None:
        """`cs.datetime() > value` with a zoned value runs only where the zones agree."""
        with pytest.raises(pl.exceptions.SchemaError):
            self.NAIVE.select(cs.datetime() > self.STAMP)
        assert self.ZONED.select(cs.datetime() > self.STAMP).schema["t"] == pl.Boolean


class TestEngines:
    """An eager frame and a lazy frame run different passes of the optimizer."""

    LISTS: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 2],
        "l": pl.Series([[1], [2], [3]], dtype=pl.List(pl.Int16)),
    })
    ROWS: ClassVar[pl.DataFrame] = pl.DataFrame({
        "a": [1, -2, 3],
        "o": pl.Series([0, 1, 0], dtype=pl.Int8),
    })
    HANG: ClassVar[str] = (
        "import polars as pl\n"
        "frame = pl.LazyFrame({'a': [1, -2, 3], 'o': pl.Series([0, 1, 0], dtype=pl.Int8)})\n"
        "frame.select(pl.col('a').abs().slice(pl.col('o'), 2)).collect(engine='in-memory')\n"
    )

    @pytest.mark.skipif(
        PolarsRelease.installed() >= PolarsRelease.parse("2.0.0"),
        reason="polars 2.0.0 makes an eager agg simplify its expressions",
    )
    def test_an_eager_agg_runs_no_optimizer_before_2_0(self) -> None:
        """`l & True` raises in an eager `agg`, where a lazy `agg` simplifies it to `l`."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.LISTS.group_by("g").agg(pl.col("l") & True)
        lazy = self.LISTS.lazy().group_by("g").agg(pl.col("l") & True).collect(engine="in-memory")
        assert lazy.schema["l"] == pl.List(pl.List(pl.Int16))

    @pytest.mark.skipif(
        PolarsRelease.installed() >= PolarsRelease.parse("2.0.0"),
        reason="polars 2.0.0 makes an eager agg simplify its expressions",
    )
    def test_an_eager_agg_negates_the_literal_that_a_lazy_agg_folds_before_2_0(self) -> None:
        """The negation of `2**31` gives `Int64` in an eager `agg` and `Int32` in a lazy one."""
        eager = self.LISTS.group_by("g").agg(-pl.lit(2**31))
        lazy = self.LISTS.lazy().group_by("g").agg(-pl.lit(2**31)).collect(engine="in-memory")
        assert eager.schema["literal"] == pl.Int64
        assert lazy.schema["literal"] == pl.Int32

    @pytest.mark.skipif(
        PolarsRelease.installed() < PolarsRelease.parse("2.0.0"),
        reason="polars 2.0.0 makes an eager agg simplify its expressions",
    )
    def test_an_eager_agg_simplifies_as_a_lazy_agg(self) -> None:
        """An eager `agg` simplifies `l & True` to `l`, and folds the negation of `2**31`."""
        eager = self.LISTS.group_by("g").agg(pl.col("l") & True, -pl.lit(2**31))
        assert eager.schema["l"] == pl.List(pl.List(pl.Int16))
        assert eager.schema["literal"] == pl.Int32

    def test_an_eager_slice_refuses_an_offset_that_is_a_column(self) -> None:
        """An eager frame reads the offset of a slice as one value."""
        with pytest.raises(pl.exceptions.ComputeError):
            self.ROWS.select(pl.col("a").slice(pl.col("o"), 2))

    @pytest.mark.skipif(
        PolarsRelease.installed() < PolarsRelease.parse("1.40.0"),
        reason="polars 1.40.0 makes an offset that is not a literal a place of a slice",
    )
    def test_a_lazy_slice_by_a_column_leaves_the_tree_over_a_column(self) -> None:
        """The slice pushdown deletes the slice, and the column comes back whole."""
        out = self.ROWS.lazy().select(pl.col("a").slice(pl.col("o"), 2)).collect(engine="in-memory")
        assert out.schema["a"] == pl.Int64
        assert out.height == self.ROWS.height

    @pytest.mark.skipif(
        PolarsRelease.installed() >= PolarsRelease.parse("1.40.0"),
        reason="polars 1.40.0 makes an offset that is not a literal a place of a slice",
    )
    def test_a_lazy_slice_by_a_column_raises_as_an_eager_one_before_1_40(self) -> None:
        """The older releases keep the slice, whose offset of more than one row raises."""
        with pytest.raises(pl.exceptions.ComputeError):
            self.ROWS.lazy().select(pl.col("a").slice(pl.col("o"), 2)).collect(engine="in-memory")

    @pytest.mark.skipif(
        PolarsRelease.installed() < PolarsRelease.parse("1.40.0"),
        reason="polars 1.40.0 makes an offset that is not a literal a place of a slice",
    )
    def test_a_lazy_slice_by_a_column_hangs_over_an_elementwise_value(self) -> None:
        """The slice pushdown makes a cycle below `abs`, and the query never returns."""
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                [sys.executable, "-c", self.HANG],
                capture_output=True,
                check=False,
                timeout=5,
            )


class TestNullBound:
    """A comparison with a null literal beside one value becomes a null, and skips its receiver."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"s": ["a", "b"]})

    def test_a_scalar_receiver_beside_a_null_bound_does_not_run(self) -> None:
        """The conversion puts a null in place of the node, so `mean` over text never runs."""
        out = self.FRAME.select(pl.col("s").mean().is_between(None, 2).alias("o"))
        assert out.schema["o"] == pl.Boolean

    def test_a_receiver_of_one_value_per_row_still_runs(self) -> None:
        """The null repeats to the length of the receiver, so `abs` over text still runs."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.FRAME.select(pl.col("s").abs().is_between(None, 2).alias("o"))


class TestQuantileColumn:
    """The value of a quantile column decides the type of the quantile of an integer in `agg`."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame(
        {"g": [1, 1, 2], "i": pl.Series([1, 2, 3], dtype=pl.Int8)},
    )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.5, pl.Float64), (1.5, pl.Int8), (float("nan"), pl.Int8)],
    )
    def test_the_value_decides_the_type(self, value: float, expected: pl.DataType) -> None:
        """A quantile outside of 0 and 1 gives nulls, and an integer then keeps its type."""
        frame = self.FRAME.with_columns(q=pl.lit(value))
        out = frame.group_by("g").agg(pl.col("i").quantile(pl.col("q").first()))
        assert out.schema["i"] == expected

    def test_a_null_quantile_raises(self) -> None:
        """A quantile that is null holds no number, in a `select` and in `agg`."""
        quantile = pl.col("i").quantile(pl.lit(None, dtype=pl.Float64))  # type: ignore[invalid-call-shape]
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.select(quantile)
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.group_by("g").agg(quantile)


class TestRowsDecideAType:
    """The number of rows decides the type of a slice of one list beside a column of offsets."""

    SLICE: ClassVar[pl.Expr] = pl.col("d").implode().list.slice(pl.col("i"), 1)

    @pytest.mark.parametrize(
        ("rows", "expected"),
        [(1, pl.List(pl.Date)), (2, pl.List(pl.Int32))],
    )
    def test_the_rows_decide_the_type(self, rows: int, expected: pl.DataType) -> None:
        """Over one row the offsets have the one row of the list, and the slice keeps the dates.

        Before polars 1.41.0, the kernel does not slice one list for each row of
        a longer offset, so two rows raise.
        """
        days = pl.Series([date(2020, 1, 1)] * rows)
        frame = pl.DataFrame({"d": days, "i": pl.Series([0] * rows, dtype=pl.Int32)})
        if rows > 1 and PolarsRelease.installed() < PolarsRelease.parse("1.41.0"):
            with pytest.raises(pl.exceptions.ComputeError):
                frame.select(self.SLICE)
            return
        assert frame.select(self.SLICE).schema["d"] == expected


class TestStoredType:
    """A lazy frame plans each step at the stored type of a column, and runs on its data.

    Before polars 2.0.0, the product of an `Int8` and a `Decimal(10, 2)` is
    planned as `Decimal(10, 2)` and runs as `Decimal(38, 2)`. From that
    release, the planner gives `Decimal(38, 2)` too, so the stored type and
    the type of the data agree.
    """

    OLDER: ClassVar[bool] = PolarsRelease.installed() < PolarsRelease.parse("2.0.0")

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "i8": pl.Series([1, None], dtype=pl.Int8),
        "dec": pl.Series([Decimal("1.5"), Decimal("2.5")], dtype=pl.Decimal(10, 2)),
    })

    def planned(self) -> pl.LazyFrame:
        """Give a lazy frame whose column `d` has a stored type other than the type of its data.

        Returns:
            The frame.
        """
        return self.FRAME.lazy().with_columns(d=pl.col("i8") * pl.col("dec"))

    def test_the_frame_stores_the_planner_type(self) -> None:
        """`collect_schema()` gives the planner type of `d`, and `collect()` gives its run type."""
        stored = pl.Decimal(10, 2) if self.OLDER else pl.Decimal(38, 2)
        assert self.planned().collect_schema()["d"] == stored
        assert self.planned().collect(engine="in-memory").schema["d"] == pl.Decimal(38, 2)

    def test_a_later_step_plans_at_the_stored_type(self) -> None:
        """A fill of `i8` by `d` casts to the stored type, and an eager fill to the run type."""
        filled = pl.col("i8").fill_null(pl.col("d")).alias("o")
        stored = pl.Decimal(10, 2) if self.OLDER else pl.Decimal(38, 2)
        assert self.planned().select(filled).collect(engine="in-memory").schema["o"] == stored
        eager = self.FRAME.with_columns(d=pl.col("i8") * pl.col("dec"))
        assert eager.select(filled).schema["o"] == pl.Decimal(38, 2)

    def test_a_join_checks_its_keys_at_both_types(self) -> None:
        """The plan compares the stored types of the keys, and the run the types of their data.

        From polars 2.0.0, both are `Decimal(38, 2)`, so the plan refuses the
        narrow key, and the wide key runs.
        """
        narrow = pl.LazyFrame({"d": pl.Series([Decimal("1.5")], dtype=pl.Decimal(10, 2))})
        wide = pl.LazyFrame({"d": pl.Series([Decimal("1.5")], dtype=pl.Decimal(38, 2))})
        refused, kept = (wide, narrow) if self.OLDER else (narrow, wide)
        with pytest.raises(pl.exceptions.SchemaError):
            self.planned().join(refused, on="d").collect_schema()
        assert self.planned().join(kept, on="d").collect_schema()["d"] == kept.collect_schema()["d"]
        join = self.planned().join(kept, on="d")
        if self.OLDER:
            with pytest.raises(pl.exceptions.ComputeError):
                join.collect(engine="in-memory")
        else:
            assert join.collect(engine="in-memory").schema["d"] == pl.Decimal(38, 2)


class TestLengthsTheValuesDecide:
    """A length that the values decide meets a check of lengths, and the data decides the answer."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"g": [1, 1, 2, 2], "x": [1, 5, 2, 6]})

    def test_item_runs_where_the_filter_keeps_one_row(self) -> None:
        """`item` over a `filter` runs where one row is left, and raises otherwise."""
        one = pl.col("x").filter(pl.col("x") > 5).item()
        assert self.FRAME.select(one).schema["x"] == pl.Int64
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.select(pl.col("x").filter(pl.col("x") > 1).item())
        empty = pl.col("x").filter(pl.col("x") > 9).item(allow_empty=True)
        assert self.FRAME.select(empty).schema["x"] == pl.Int64

    def test_item_runs_in_each_group_that_keeps_one_row(self) -> None:
        """In `agg`, `item` over a `filter` runs where each group keeps one row."""
        one = pl.col("x").filter(pl.col("x") > 3).item()
        assert self.FRAME.group_by("g").agg(one).schema["x"] == pl.Int64
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.group_by("g").agg(pl.col("x").filter(pl.col("x") > 0).item())

    def test_a_filter_meets_a_fixed_length_where_the_data_gives_it(self) -> None:
        """A `filter` beside `head(2)` runs where the mask keeps two rows."""
        two = pl.col("x").filter(pl.col("x") > 2) + pl.col("x").head(2)
        assert self.FRAME.select(two).height == 2
        with pytest.raises(pl.exceptions.ShapeError):
            self.FRAME.select(pl.col("x").filter(pl.col("x") > 0) + pl.col("x").head(2))

    def test_a_count_of_unique_values_runs_where_one_value_is_left(self) -> None:
        """`pct_change` needs a count of one row, which `unique` gives where the values agree."""
        frame = self.FRAME.with_columns(n=pl.lit(1))
        assert frame.select(pl.col("x").pct_change(pl.col("n").unique())).schema["x"] == pl.Float64
        with pytest.raises(pl.exceptions.ComputeError):
            self.FRAME.select(pl.col("x").pct_change(pl.col("x").unique()))

    def test_a_key_of_unique_values_sorts_where_each_group_meets_the_value(self) -> None:
        """In `agg`, `sort_by` runs where the key gives each group the rows of the value."""
        one = pl.col("x").head(1).sort_by(pl.col("g").unique())
        assert self.FRAME.group_by("g").agg(one).schema["x"] == pl.List(pl.Int64)
        with pytest.raises(pl.exceptions.ShapeError):
            self.FRAME.group_by("g").agg(pl.col("x").sort_by(pl.col("g").unique()))


class TestElementwiseKeys:
    """In `agg`, `over` takes only a key whose every node works value by value."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3], "k": [1, 1, 2]})

    @pytest.mark.parametrize("key", [pl.col("k") + 1, pl.col("k").abs(), pl.lit(1)])
    def test_an_elementwise_key_runs(self, key: pl.Expr) -> None:
        """A column, a literal and an elementwise function of them run."""
        out = self.FRAME.group_by("g").agg(pl.col("x").over(key))
        assert out.schema["x"] == pl.List(pl.Int64)

    @pytest.mark.parametrize(
        "key", [pl.col("k").sort(), pl.col("k").cum_sum(), pl.col("k").first()]
    )
    def test_another_key_raises(self, key: pl.Expr) -> None:
        """A key with a node that is not elementwise raises, though it keeps the rows."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.FRAME.group_by("g").agg(pl.col("x").over(key))


class TestBoundedRows:
    """A slice of a literal length holds at most that many rows, on every frame."""

    @pytest.mark.parametrize("rows", [1, 3, 400])
    def test_an_index_past_a_head_raises_on_every_frame(self, rows: int) -> None:
        """`head(2).gather(300)` raises, and `gather(300)` runs where the frame is long enough."""
        frame = pl.DataFrame({"x": list(range(rows))})
        calls: list[Callable[[pl.Expr], pl.Expr]] = [lambda x: x.gather(300), lambda x: x.get(300)]
        for call in calls:
            with pytest.raises(pl.exceptions.OutOfBoundsError):
                frame.select(call(pl.col("x").head(2)))
        with pytest.raises(pl.exceptions.ShapeError):
            frame.select(pl.col("x").head(2).sample(300))  # type: ignore[invalid-call-shape]
        if rows > 300:
            assert frame.select(pl.col("x").gather(300)).height == 1


class TestNullsThroughANode:
    """A node that keeps the nulls of its input gives only nulls, which polars casts to any type."""

    def test_a_cast_after_a_slice_of_a_null_runs(self) -> None:
        """The cast of `dt.epoch` from a `Decimal` runs over only nulls, and not over a value."""
        frame = pl.DataFrame({"x": [1]})
        null = pl.lit(None, dtype=pl.Decimal(10, 2))
        assert frame.select(null.head(2).dt.epoch()).schema["literal"] == pl.Int64
        assert frame.select((null + 1).dt.epoch()).schema["literal"] == pl.Int64
        with pytest.raises(pl.exceptions.InvalidOperationError):
            frame.select(null.fill_null(1).dt.epoch())  # type: ignore[invalid-call-shape]


class TestSliceOfNoRow:
    """A slice of no row holds no value, so polars casts it to any type, as it casts nulls."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 2],
        "l": [[1], [2], [3]],
        "t": ["a", "b", "c"],
    })

    def test_a_comparison_casts_a_list_of_no_row(self) -> None:
        """A comparison casts both sides to their supertype, and a list of no row casts to text."""
        compared = self.FRAME.select(pl.col("l").head(0) == pl.col("t").first())
        assert compared.schema["l"] == pl.Boolean
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.FRAME.select(pl.col("l").head(1) == pl.col("t").first())

    def test_a_cast_in_agg_reads_the_rows_that_the_slice_drops_where_it_is_not_strict(self) -> None:
        """In `agg`, a strict cast reads each group, and another cast reads the whole column."""
        grouped = self.FRAME.group_by("g")
        assert grouped.agg(pl.col("t").head(0).any()).schema["t"] == pl.Boolean
        with pytest.raises(pl.exceptions.InvalidOperationError):
            grouped.agg(pl.col("l").head(0).ewm_mean(alpha=0.5))


class TestExtendOverNoRow:
    """Over an input of no row, `extend_constant` keeps its value as it is, and panics at a read.

    Polars builds the column of the type of the input from the value with no
    cast, and its builder refuses a value of another kind (`ValueBuild`).
    """

    CASES: ClassVar[tuple[tuple[pl.DataType, object], ...]] = (
        (pl.Int8(), 1),
        (pl.Int8(), 300),
        (pl.Int8(), True),
        (pl.Int8(), b"a"),
        (pl.Float32(), 1.5),
        (pl.Float64(), 1.5),
        (pl.String(), 1),
        (pl.Categorical(), "x"),
        (pl.Enum(["a"]), "a"),
        (pl.Datetime("ns"), datetime(2020, 1, 1)),
        (pl.Datetime("us"), datetime(2020, 1, 1)),
        (pl.Decimal(10, 2), Decimal("1.5")),
        (pl.List(pl.Int64()), [1]),
    )

    @pytest.mark.parametrize(("dtype", "value"), CASES, ids=repr)
    def test_a_read_panics_where_the_builder_refuses_the_value(
        self,
        dtype: pl.DataType,
        value: object,
    ) -> None:
        """A print in a `select` and the lists of `agg` panic exactly where the builder refuses."""
        frame = pl.DataFrame({"g": [1, 2], "x": pl.Series([None, None], dtype=dtype)})
        extended = pl.col("x").head(0).extend_constant(value, 3)
        literal = pl.select(pl.lit(value)).dtypes[0]
        refused = ValueBuild().refuses(DType.of(literal), value, DType.of(dtype))
        assert frame.select(extended).schema["x"] == dtype
        for read in (
            lambda: str(frame.select(extended)),
            lambda: frame.group_by("g").agg(extended),
        ):
            if refused:
                with pytest.raises(pl.exceptions.PanicException):
                    read()
            else:
                read()


class TestCumulativeEvalInAgg:
    """In `agg`, `cumulative_eval` gives one list for each group, even over one value for each."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3]})

    def test_one_value_for_each_group_gives_a_list(self) -> None:
        """The run gives a list where the planner gives the type of one value."""
        evaluated = pl.col("x").first().cumulative_eval(pl.element().sum())
        assert self.FRAME.group_by("g").agg(evaluated).schema["x"] == pl.List(pl.Int64)
        planned = self.FRAME.lazy().group_by("g").agg(evaluated).collect_schema()
        assert planned["x"] == pl.Int64


class TestLiteralTextParse:
    """A parse of a literal text meets the value of the text at every run."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"g": [1, 1, 2], "s": ["2020-01-01"] * 3})

    def test_a_text_that_cannot_fit_the_format_raises(self) -> None:
        """A strict parse of `"a"` raises, and a parse that is not strict gives a null."""
        with pytest.raises(pl.exceptions.InvalidOperationError):
            pl.select(pl.lit("a").str.to_date("%Y-%m-%d"))  # type: ignore[invalid-call-shape]
        lenient = pl.select(pl.lit("a").str.to_date("%Y-%m-%d", strict=False))
        assert lenient.schema["literal"] == pl.Date

    PARTIAL_TIMES: ClassVar[dict[str, str]] = {
        "%H": "01",
        "%M": "02",
        "%S": "03",
        "%I:%M": "01:02",
        "%l:%M": " 1:02",
    }
    WHOLE_TIMES: ClassVar[dict[str, str]] = {
        "%H:%M": "01:02",
        "%k:%M": " 1:02",
        "%I:%M %p": "01:02 am",
        "%R": "01:02",
        "%T": "01:02:03",
    }

    @pytest.mark.parametrize("form", sorted(PARTIAL_TIMES))
    def test_a_time_without_its_hour_or_its_minute_raises(self, form: str) -> None:
        """A time needs its hour and its minute, and a 12-hour clock its half of the day."""
        parsed = pl.lit(self.PARTIAL_TIMES[form]).str.strptime(pl.Time, form)
        with pytest.raises(pl.exceptions.InvalidOperationError):
            pl.select(parsed)

    @pytest.mark.parametrize("form", sorted(WHOLE_TIMES))
    def test_a_time_with_its_hour_and_its_minute_parses(self, form: str) -> None:
        """A format that gives the hour and the minute parses a text of its shape."""
        parsed = pl.lit(self.WHOLE_TIMES[form]).str.strptime(pl.Time, form)
        assert pl.select(parsed).schema["literal"] == pl.Time

    def test_a_parse_of_a_literal_text_is_elementwise_in_agg(self) -> None:
        """The parse gives one row beside an `ambiguous` of one value for each group."""
        parsed = pl.lit("2020-01-01").str.to_datetime(ambiguous=pl.col("s").first())
        with pytest.raises(pl.exceptions.InvalidOperationError):
            self.FRAME.group_by("g").agg(parsed)
        assert self.FRAME.select(parsed).schema["literal"] == pl.Datetime("us")


class TestSortByOverARebuiltList:
    """In `agg`, `sort_by` sorts a list that a node rebuilt by the list of its one key."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({"g": [1, 1, 1, 2, 2], "x": [1, 2, 3, 4, 5]})

    @pytest.mark.skipif(
        PolarsRelease.installed() >= PolarsRelease.parse("2.0.0"),
        reason="polars 2.0.0 checks the group lengths of the key beside a rebuilt list",
    )
    def test_a_key_of_one_value_for_each_group_panics_before_2_0(self) -> None:
        """The key gives no list, and polars panics."""
        sorted_ = pl.col("x").cum_sum().sort_by(pl.col("x").first())
        with pytest.raises(pl.exceptions.PanicException):
            self.FRAME.group_by("g").agg(sorted_)

    @pytest.mark.skipif(
        PolarsRelease.installed() < PolarsRelease.parse("2.0.0"),
        reason="polars 2.0.0 checks the group lengths of the key beside a rebuilt list",
    )
    def test_a_key_of_one_value_for_each_group_has_other_group_lengths(self) -> None:
        """The key has one row in each group, where the list has more, and polars refuses it."""
        sorted_ = pl.col("x").cum_sum().sort_by(pl.col("x").first())
        with pytest.raises(pl.exceptions.ShapeError):
            self.FRAME.group_by("g").agg(sorted_)

    def test_a_key_of_one_value_per_row_runs(self) -> None:
        """A key of one value per row gives a list for each group."""
        sorted_ = pl.col("x").cum_sum().sort_by(pl.col("x"))
        assert self.FRAME.group_by("g").agg(sorted_).schema["x"] == pl.List(pl.Int64)
