"""Check the evaluator of whole trees against polars, on trees that each path of it reads.

The evaluator computes the state of each node of a tree from the states of
its inputs (`TreeEvaluation`). The random trees of `spec/tools/rules/tree_check.py`
measure it at scale. These tests reach the paths that a random tree of census
nodes rarely builds: an `eval` or a struct `eval` at the root or below
another node, a literal of each form, a literal of a series of several values
beside a column and a slice, a node that refuses below another node,
`pl.element()` and `pl.field()` outside their context, a node over an input
that answers in `agg` and refuses in a `select`, a horizontal sum in the order
of a parallel reduction, a `replace` by a struct that a cast keeps, a typed
literal as the key of a window, an input that a node reads in a form that the
Python API does not write, and a node or a literal that the rules leave
unchecked.

Each case builds one tree over a frame of two groups of other sizes, and
compares the four answers of polars with the four answers of the evaluator:
the planner type and the run type, in a `select` and in `agg`.
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, ClassVar

import polars as pl
import pytest
from polars.io.plugins import register_io_source

from astrolabe._rules.dtypes import DType
from astrolabe._rules.execution import GroupBy, QueryFlags
from astrolabe._rules.failures import UndecidedError
from astrolabe._rules.node_evaluation import ColumnLeaf, Unchecked
from astrolabe._rules.options import Height, RowForm
from astrolabe._rules.releases import PolarsRelease
from astrolabe._rules.tree import ExpressionTree
from astrolabe._rules.tree_evaluation import JoinKeyRows, LiteralParse, NodeTraits, TreeEvaluation
from spec.tests.rule_probes import PolarsAnswer, Stopped, TemplateCall, TreeColumns, TreeFrame

if TYPE_CHECKING:
    from collections.abc import Iterator

CASES: dict[str, pl.Expr] = {
    "eval at the root": pl.col("l").list.eval(pl.element() + 1),
    "eval below another node": pl.col("l").list.eval(pl.element() + 1).list.len(),
    "array eval": pl.col("a").arr.eval(pl.element() * 2),
    "array eval over a list": pl.col("l").arr.eval(pl.element()),
    "cumulative eval": pl.col("x").cumulative_eval(pl.element().sum()),
    "cumulative eval of a body that works row by row": pl.col("x").cumulative_eval(  # type: ignore[invalid-call-shape]
        pl.element() * 2,
    ),
    "cumulative eval of a head": pl.col("x").cumulative_eval(pl.element().head(1)),  # type: ignore[invalid-call-shape]
    "cumulative eval of an implode": pl.col("x").cumulative_eval(pl.element().implode()),
    "cumulative eval of a literal": pl.col("x").cumulative_eval(pl.lit(1)),
    "eval that refuses": pl.col("l").list.eval(pl.element().str.len_bytes()),
    "struct eval at the root": pl.col("s").struct.with_fields(pl.field("a") + 1),
    "struct eval below another node": (
        pl.col("s").struct.with_fields(pl.field("a") + 1).struct.field("a")
    ),
    "struct eval over a number": pl.col("x").struct.with_fields(pl.field("a")),
    "struct eval of an alias of a field": pl.col("s").struct.with_fields(pl.field("a").alias("b")),
    "struct eval of a struct of fields": pl.col("s").struct.with_fields(
        b=pl.struct(pl.field("a") + 1, pl.field("a").alias("c")),
    ),
    "struct eval of a head as a key of over": pl
    .col("x")
    .sum()
    .over(
        pl.col("s").head(5).struct.with_fields(pl.field("a")),
    ),
    "struct eval of a list eval as a key of over": pl
    .col("x")
    .sum()
    .over(
        pl.col("s").struct.with_fields(b=pl.concat_list(pl.field("a")).list.eval(pl.element() + 1)),
    ),
    "struct eval of an is_in of a literal as a key of over": pl
    .col("x")
    .sum()
    .over(
        pl.col("s").struct.with_fields(b=pl.field("a").is_in([1, 2])),
    ),
    "struct eval of a struct eval as a key of over": pl
    .col("x")
    .sum()
    .over(
        pl.col("s").struct.with_fields(
            b=pl.struct(pl.field("a")).struct.with_fields(pl.field("a") + 1),
        ),
    ),
    "text literal": pl.col("t") + pl.lit("a"),
    "a range of the height of the frame beside a column": pl.int_range(pl.len()) + pl.col("x"),
    "a search of the unique values": pl.col("x").search_sorted(pl.col("x").unique()),
    "a gather by the unique keys": pl.col("x").gather(pl.col("g").unique() - 1),
    "an over ordered by a text column": pl.col("x").cum_sum().over("g", order_by="t"),
    "literal of a struct": pl.col("x") + pl.lit({"a": 1}),
    "literal of a list": pl.col("l").list.concat(pl.lit([7, 8])),
    "literal of an array": pl.col("a") == pl.lit([1, 2], dtype=pl.Array(pl.Int64, 2)),
    "literal of an enum": pl.col("t").cast(pl.Enum(["a", "b", "c", "d", "e"]))
    == pl.lit(
        "a",
        dtype=pl.Enum(["a", "b", "c", "d", "e"]),
    ),
    "literal of bytes": pl.col("t").cast(pl.Binary) + pl.lit(b"x"),
    "a node over an input that answers only in agg": pl.col("t").mean() == pl.lit("a"),
    "refusal below another node": (pl.col("t") + 1).abs(),
    "an element outside an eval": pl.element() + 1,
    "a replace by a struct that the cast to a number keeps": (
        pl.col("x").cast(pl.Int8).replace(1, pl.col("s").first())
    ),
    "a mean of three inputs that adds the last two first": pl.mean_horizontal(
        pl.col("g").cast(pl.Float16),
        pl.col("g").cast(pl.Int8),
        pl.col("g").cast(pl.UInt8),
    ),
    "a field outside a struct eval": pl.field("a") + 1,
    "rename below another node": pl.col("x").name.suffix("_y").abs(),
    "a kept name over a literal": (pl.lit(1) + 1).name.keep(),  # type: ignore[invalid-call-shape]
    "a kept name over a column": (pl.lit(1) + pl.col("x")).name.keep(),
    "a kept name as a field of a struct": pl.struct(
        pl.lit(1).over("x").name.keep(),
        pl.col("t").alias("z").name.keep(),
    ),
    "literal of a parametric type": pl.col("x") + pl.lit(1, dtype=pl.Datetime("ms")),
    "eval over a literal of a parametric type": pl.col("l").list.eval(
        pl.element() + pl.lit(1, dtype=pl.Datetime("ms")),
    ),
    "null of a list": pl.lit(None, dtype=pl.List(pl.Datetime("ms"))),
    "null of a struct": pl.lit(None, dtype=pl.Struct({"a": pl.Duration("us")})),
    "null of an enum": pl.lit(None, dtype=pl.Enum(["a", "b"])),
    "null of an enum with no category": pl.lit(None, dtype=pl.Enum),
    "null of a list beside a column": pl.col("l").list.concat(
        pl.lit(None, dtype=pl.List(pl.Int64)),
    ),
    "an or that the optimizer folds below another node": pl.col("a") & (pl.col("g").first() | True),
    "an and that the optimizer folds below a node": pl.col("a") | (pl.col("g").first() & False),
    "an input that the optimizer keeps below another node": pl.col("a") & (pl.lit(True) & True),
    "an or of a column below another node": pl.col("a") & (pl.col("g").cast(pl.Boolean) | True),
    "literal of a series of one value": pl.col("x") + pl.lit(pl.Series([1], dtype=pl.Int8)),
    "literal of a series of the height of the frame": pl.col("x") + pl.lit(pl.Series([1] * 5)),
    "literal of a series beside a slice of its length": (
        pl.col("x").head(3) + pl.lit(pl.Series([1, 2, 3]))
    ),
    "literal of a series beside a slice of another length": (
        pl.col("x").head(2) + pl.lit(pl.Series([1, 2, 3]))
    ),
    "explode of a flat column beside a slice": pl.col("x").head(2) + pl.col("x").explode(),
    "slice of one list beside a column offset": pl.col("l").first().list.slice(pl.col("x"), 1),
    "slice of a null list beside a column offset": pl.lit(None, dtype=pl.List(pl.Date)).list.slice(
        pl.col("x"), 1
    ),
    "a count of unique values": pl.col("x").pct_change(pl.col("g").head(2).unique()),
    "a count that two literals give": (
        pl.col("x").first().diff(pl.lit(1) + 0).cast(pl.List(pl.Date))
    ),
    "a sort by unique values": pl.col("x").head(1).sort_by(pl.col("g").head(2).unique()),
    "a sort of rebuilt groups by a slice": pl.col("x").cum_sum().sort_by(pl.col("x").head(2)),
    "a window by a key in a literal set": pl.col("x").over(pl.col("g").is_in([1, 2])),
    "an append after a node that plans at the supertype": pl.col("t").sqrt().append(pl.col("t")),
    "an append after a node that plans at another type": pl
    .col("x")
    .cast(pl.Float64)
    .append(
        pl.col("t"),
    ),
    "a window by a key in a set of a column": pl.col("x").over(pl.col("g").is_in(pl.col("x"))),
    "a deviation of one value": pl.col("x").first().std().cast(pl.List(pl.Date)),
    "a deviation over the rows that its freedom takes": pl
    .col("x")
    .head(1)
    .var()
    .cast(pl.List(pl.Date)),
    "a weighted deviation of one value": pl
    .col("x")
    .first()
    .ewm_std(alpha=0.5)
    .cast(pl.List(pl.Date)),
    "a weighted mean over fewer rows than it needs": (
        pl.col("x").head(2).ewm_mean(alpha=0.5, min_samples=3).cast(pl.List(pl.Date))
    ),
    "a weighted deviation in agg beside text": pl.col("t").ewm_std(alpha=0.5) - pl.col("t"),
    "when beside a literal of a series": pl
    .when(pl.col("x") > 0)
    .then(pl.lit(pl.Series([1] * 5)))
    .otherwise(pl.col("x")),
    "literal of a series of one null": pl.col("t") + pl.lit(pl.Series([None], dtype=pl.String)),
    "literal of a series of two values": pl.col("x") + pl.lit(pl.Series([1, 2])),
    "replace of lists of values": pl.col("x").replace([1, 2], [3, 4]),
    "replace by a dictionary": pl.col("x").replace({1: 3, 2: 4}),
    "replace by a value that does not fit": pl.col("x").cast(pl.Int8).replace([1], [1000]),
    "replace by a struct of two fields": pl.col("x").replace(1, pl.struct("x", "g").first()),
    "replace by a struct of a struct": pl.col("x").replace(1, pl.struct(pl.col("s")).first()),
    "a format with no input": pl.format("no placeholder"),
    "a negated wide literal": -pl.lit(2**31),
    "a negated literal of the widest bucket": -pl.lit(2**63),
    "a negated wide literal beside a temporal column": pl
    .when(pl.col("x") > 0)
    .then(pl.col("x").cast(pl.Date))
    .otherwise(-pl.lit(2**31)),
    "a negated typed literal": -pl.lit(5, dtype=pl.Int8),
    "a negated typed literal at the end of its range": -pl.lit(-128, dtype=pl.Int8),
    "a negated float literal": -pl.lit(1.5, dtype=pl.Float32),
    "a negated unsigned literal": -pl.lit(5, dtype=pl.UInt8),  # type: ignore[invalid-call-shape]
    "a shift of nulls with a dynamic fill": pl.lit(None).shift(1, fill_value=300),
    "a shift of nulls with a dynamic fill beside a column": (
        pl.lit(None).shift(1, fill_value=300) + pl.col("g").cast(pl.UInt8)
    ),
    "a shift of nulls with a wide fill beside a column": (
        pl.col("g").cast(pl.UInt8) + pl.lit(None).shift(1, fill_value=2**40)
    ),
    "a shift of nulls with a float fill beside a column": (
        pl.lit(None).shift(1, fill_value=0.5) + pl.col("g").cast(pl.UInt8)
    ),
    "a named column in an eval": pl.col("l").list.eval(pl.col("x")),
    "a window over a key of a fold of literals": pl.col("x").over(pl.lit(1) + pl.lit(2)),
    "a window over a key of an eval": pl.col("x").over(pl.col("l").list.eval(pl.element() + 1)),
    "a window over a key of a struct eval": pl.col("x").over(
        pl.col("s").struct.with_fields(pl.field("a") + 1),
    ),
    "a window over a key of a series literal": pl.col("x").over(pl.lit(pl.Series([1, 2, 3, 4, 5]))),
    "a window over a key of the length": pl.col("x").over(pl.len()),
    "a window over a key of a zoned literal": pl.col("s").over(
        pl.lit(dt.datetime(2024, 1, 2, tzinfo=dt.UTC))
    ),
    "a parse of a zoned literal by a format of two rows": TemplateCall(
        "pl.col('x').str.to_datetime()"
    )(
        pl.lit(dt.datetime(2024, 1, 2, tzinfo=dt.UTC)),
        pl.col("t").head(2),
    ),
    "a named column in a cumulative eval": pl.col("x").cumulative_eval(pl.col("x").sum()),  # type: ignore[invalid-call-shape]
    "a column of the frame in a struct eval": pl.col("s").struct.with_fields(pl.col("x")),
    "business days with the holidays as a series": pl.business_day_count(
        pl.col("x").cast(pl.Date),
        pl.col("g").cast(pl.Date),
    ),
}
DAY: pl.Expr = pl.col("x").cast(pl.Date)
MOMENT: pl.Expr = pl.col("x").cast(pl.Datetime("us"))
CLOCK: pl.Expr = pl.col("x").cast(pl.Time)
ONE_VALUE: dict[str, pl.Expr] = {
    "the count of shift": pl.col("x").shift(pl.col("g")),
    "the fill of shift": pl.col("x").shift(1, fill_value=pl.col("g")),
    "the count of diff": pl.col("x").diff(pl.col("g")),
    "the value of repeat": pl.repeat(pl.col("g"), 2),  # type: ignore[invalid-call-shape]
    "the count of repeat": pl.repeat(1, pl.col("g")),  # type: ignore[invalid-call-shape]
    "the value of extend_constant": pl.col("x").extend_constant(pl.col("g"), 2),
    "the count of extend_constant": pl.col("x").extend_constant(1, pl.col("g")),
    "the value of index_of": pl.col("x").index_of(pl.col("g")),
    "the start of int_range": pl.int_range(pl.col("g"), 3),  # type: ignore[invalid-call-shape]
    "the stop of int_range": pl.int_range(0, pl.col("g")),  # type: ignore[invalid-call-shape]
    "the start of date_range": pl.date_range(DAY, DAY.max(), eager=False),
    "the end of date_range": pl.date_range(DAY.min(), DAY, eager=False),
    "the start of datetime_range": pl.datetime_range(MOMENT, MOMENT.max(), eager=False),
    "the end of datetime_range": pl.datetime_range(MOMENT.min(), MOMENT, eager=False),
    "the start of time_range": pl.time_range(CLOCK, CLOCK.max(), eager=False),
    "the end of time_range": pl.time_range(CLOCK.min(), CLOCK, eager=False),
    "the start of linear_space": pl.linear_space(pl.col("g"), 3, 3),  # type: ignore[invalid-call-shape]
    "the end of linear_space": pl.linear_space(0, pl.col("g"), 3),  # type: ignore[invalid-call-shape]
    "the count of linear_space": pl.linear_space(0, 3, pl.col("g")),  # type: ignore[invalid-call-shape]
    "the body of cumulative_eval": pl.col("x").cumulative_eval(pl.element() * 2),  # type: ignore[invalid-call-shape]
    "the body of cumulative_eval over text": pl.col("t").cumulative_eval(pl.element() * 2),  # type: ignore[invalid-call-shape]
}
UNCHECKED: dict[str, pl.Expr] = {
    "literal of a category": pl.col("t") + pl.lit("a", dtype=pl.Categorical),
    "cast to the data type of a column": pl.col("x").cast(pl.dtype_of("g")),
    "evaluation that the rules leave unchecked": pl.col("l").list.eval(
        pl.element().cast(pl.String) + pl.lit("a", dtype=pl.Categorical),
    ),
}


WIDE_CASES: dict[str, tuple[dict[str, pl.Series], pl.Expr]] = {
    "a slice of a literal by a list eval and a cumulative count": (
        {"f": pl.Series([[1.0, -1.0]] * 5, dtype=pl.List(pl.Float32)), "dec": TreeColumns.DECIMALS},
        TemplateCall("pl.tail('x')")(
            pl.lit(1),
            pl.col("f").list.filter(pl.element() > 0),
            pl.col("dec").cum_count(),
        ),
    ),
    "a rolling window of one list by an index of moments": (
        {"z": TreeColumns.ZONED, "m": TreeColumns.MOMENTS},
        TemplateCall("pl.col('x').rolling(index_column='t', period='2d')")(
            pl.col("z").implode(),
            pl.col("m"),
        ).alias("y"),
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_evaluation_of_a_tree_gives_the_answers_of_polars(case: str) -> None:
    """The four answers of the evaluator match polars, for a tree that reaches one path."""
    expression = CASES[case]
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)


@pytest.mark.parametrize("case", sorted(WIDE_CASES))
def test_the_evaluation_of_a_tree_over_more_data_types_gives_the_answers_of_polars(
    case: str,
) -> None:
    """The four answers match polars, for a tree that reads columns that the frame adds."""
    columns, expression = WIDE_CASES[case]
    frame = TreeFrame(columns)
    assert frame.rules(expression) == frame.polars(expression)


@pytest.mark.parametrize("case", sorted(ONE_VALUE))
def test_an_input_of_one_value_is_refused_with_the_message_of_polars(case: str) -> None:
    """A column where an input must give one value gets the refusal of polars and its message."""
    expression = ONE_VALUE[case]
    message = TreeFrame().polars_message(expression)
    assert message
    assert TreeFrame().rules_message(expression) == message


@pytest.mark.parametrize("case", sorted(UNCHECKED))
def test_the_evaluation_leaves_a_tree_unchecked(case: str) -> None:
    """The evaluator gives no answer where the rules do not translate a node or a literal."""
    assert set(TreeFrame().rules(UNCHECKED[case])) == {"unchecked"}


@pytest.mark.parametrize(
    "expression",
    [pl.col("l").list.item(), pl.col("l").list.head(1).list.item()],
    ids=["item", "item of a head"],
)
def test_an_eval_over_lists_of_one_value_gives_their_run(expression: pl.Expr) -> None:
    """The run of an `eval` that needs lists of one value gives the answer of polars over such data.

    The number of values in each list is a fact of the data, so the evaluator
    answers as polars does where each list holds one value.
    """
    frame = pl.DataFrame({"l": [[{"a": 1}], [{"a": 2}]]})
    leaf = ColumnLeaf(DType.of(frame.schema["l"]), RowForm.PER_ROW)
    answers = TreeEvaluation({"l": leaf}, GroupBy((3, 2))).answers(ExpressionTree().of(expression))
    assert answers.run_select == DType.of(frame.select(expression).schema["l"])


def test_a_node_with_no_definition_is_unchecked() -> None:
    """A node that no definition states gives no answer, in every pass and context."""
    tree = ExpressionTree().read({"Function": {"input": [{"Column": "x"}], "function": "Nothing"}})
    columns = {"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}
    answers = TreeEvaluation(columns, GroupBy((3, 2))).answers(tree)
    assert {type(answer) for answer in vars(answers).values()} == {Unchecked}


def test_a_node_with_no_definition_leaves_a_join_key_unknown() -> None:
    """A node that no definition states has no known height, and leaves a join key open."""
    tree = ExpressionTree().read({"Function": {"input": [{"Column": "x"}], "function": "Nothing"}})
    evaluation = TreeEvaluation({"x": ColumnLeaf(DType("Int8"), RowForm.PER_ROW)}, GroupBy((3,)))
    assert NodeTraits(evaluation).height(tree) is Height.UNKNOWN
    assert JoinKeyRows(evaluation).refuses(tree) is None


def test_a_literal_of_another_form_is_not_read() -> None:
    """A literal whose body is not a map of one entry has no leaf."""
    with pytest.raises(UndecidedError):
        LiteralParse().of("a")


STORED_FRAME = pl.DataFrame({
    "i8": pl.Series([1, None, 3], dtype=pl.Int8),
    "i32": pl.Series([1, 2, 3], dtype=pl.Int32),
    "f32": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32),
    "dec": pl.Series([1, 2, 3], dtype=pl.Decimal(10, 2)),
})
# The steps that give `d` a stored type other than the type of its data. From
# polars 2.0.0, the planner widens the precision of the two steps of a decimal
# too, so only the fill of another kind keeps the two types apart.
STORED_FIRSTS: dict[str, pl.Expr] = {
    "a shift filled by another kind": pl.col("i8").shift(1, fill_value=pl.col("i32").first()),
    **(
        {
            "a product with a decimal": pl.col("i8") * pl.col("dec"),
            "a cumulative sum of a decimal": pl.col("dec").cum_sum(),
        }
        if PolarsRelease.installed() < PolarsRelease.parse("2.0.0")
        else {}
    ),
}
STORED_SECONDS: dict[str, pl.Expr] = {
    "fill_null": pl.col("i8").fill_null(pl.col("d")),
    "add": pl.col("d") + pl.col("dec"),
    "cum_sum": pl.col("d").cum_sum(),
    "mul": pl.col("d") * 2,
    "coalesce": pl.coalesce(pl.col("d"), pl.col("i8")),
    "abs": pl.col("d").abs(),
    "truediv": pl.col("d") / pl.col("i32"),
    "mean": pl.col("d").mean(),
    "max_horizontal": pl.max_horizontal(pl.col("d"), pl.col("f32")),
}


# The fill of `i8` reads the data of `d`, of another type than its plan, only
# where `i8` holds a null, so the values decide whether the kernel panics.
STORED_PAIRS = [
    (first, second)
    for first in sorted(STORED_FIRSTS)
    for second in sorted(STORED_SECONDS)
    if (first, second) != ("a shift filled by another kind", "fill_null")
]


@pytest.mark.parametrize(("first", "second"), STORED_PAIRS)
def test_a_leaf_with_a_stored_type_gives_the_answer_of_polars(first: str, second: str) -> None:
    """A step of a lazy frame plans at the stored type of a column, and runs on its data.

    The first step gives the column `d` a stored type other than the type of
    its data. The leaf of `d` holds both, and the evaluator gives the type
    of the second step as polars runs it, or says that it stops.

    Args:
        first: The step that makes `d`.
        second: The step that reads `d`.
    """
    lazy = STORED_FRAME.lazy().with_columns(d=STORED_FIRSTS[first])
    stored, run = lazy.collect_schema()["d"], lazy.collect(engine="in-memory").schema["d"]
    assert stored != run
    leaves = {
        name: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
        for name, dtype in STORED_FRAME.schema.items()
    }
    leaves["d"] = ColumnLeaf(DType.of(run), RowForm.PER_ROW, planned=DType.of(stored))
    expression = STORED_SECONDS[second].alias("o")
    answers = TreeEvaluation(leaves, GroupBy((2, 1), QueryFlags.lazy())).answers(
        ExpressionTree().of(expression)
    )
    run = PolarsAnswer().result(
        lambda: lazy.select(expression).collect(engine="in-memory").schema["o"]
    )
    if isinstance(run, Stopped):
        assert not isinstance(answers.run_select, DType), answers.run_select
    else:
        assert answers.run_select == DType.of(run)
    assert answers.planner_select == DType.of(lazy.select(expression).collect_schema()["o"])


class SplitDecimal:
    """Give a lazy frame whose column `x` declares `Decimal(10, 2)` and holds `Int32` data."""

    FRAME: ClassVar[pl.DataFrame] = pl.DataFrame({
        "g": [1, 1, 1, 2, 2],
        "x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int32),
        "d": pl.Series([1, 2, 3, 4, 5], dtype=pl.Decimal(10, 2)),
    })

    def lazy(self) -> pl.LazyFrame:
        """Give the frame, which polars plans at the declared types.

        Returns:
            The lazy frame.
        """

        def source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            """Give the frame, with the columns that the query reads.

            Args:
                with_columns: The columns that the query reads, or `None` for all.
                predicate: A filter that the query pushes down, which the frame ignores.
                n_rows: A limit that the query pushes down, which the frame ignores.
                batch_size: The size of a batch, which the frame ignores.

            Yields:
                The frame.
            """
            del predicate, n_rows, batch_size
            yield self.FRAME if with_columns is None else self.FRAME.select(with_columns)

        schema = {"g": pl.Int64(), "x": pl.Decimal(10, 2), "d": pl.Decimal(10, 2)}
        return register_io_source(io_source=source, schema=schema)


ARITHMETIC_PATHS: dict[str, pl.Expr] = {
    "flat": pl.col("x") + pl.col("d"),
    "lists": pl.col("x") + pl.col("d").cum_sum(),
    "groups": pl.col("x").first() + pl.col("d"),
}


@pytest.mark.parametrize("path", sorted(ARITHMETIC_PATHS))
def test_arithmetic_in_agg_runs_in_place_only_on_flat_values(path: str) -> None:
    """In `agg`, `+` writes into its left input only on the path over flat values.

    The data of `x` is `Int32` where polars plans `Decimal(10, 2)`, so the
    arithmetic in place panics, and the arithmetic of two series widens.

    Args:
        path: The path that the run in `agg` takes.
    """
    expression = ARITHMETIC_PATHS[path].alias("o")
    decimal = DType.of(pl.Decimal(10, 2))
    leaves = {
        "x": ColumnLeaf(DType("Int32"), RowForm.PER_ROW, planned=decimal),
        "d": ColumnLeaf(decimal, RowForm.PER_ROW),
    }
    answers = TreeEvaluation(leaves, GroupBy((3, 2))).answers(ExpressionTree().of(expression))
    query = SplitDecimal().lazy().group_by("g").agg(expression)
    try:
        polars: object = DType.of(query.collect(engine="in-memory").schema["o"])
    except BaseException as error:  # a panic is not an Exception
        polars = type(error).__name__
    rules = answers.run_agg
    assert (rules if isinstance(rules, DType) else "PanicException") == polars


ROOT_FRAME = pl.DataFrame({
    "d": pl.Series([1, 2, 3]).cast(pl.Datetime("ms")),
    "y": pl.Series([1, 2, 3], dtype=pl.UInt128),
})
SEARCH = pl.col("d").ewm_var(alpha=0.5).search_sorted(pl.col("y"))
RENAMED_ROOTS: dict[str, pl.Expr] = {
    "no rename": SEARCH,
    "alias": SEARCH.alias("o"),
    "two aliases": SEARCH.alias("a").alias("o"),
    "name.keep": SEARCH.name.keep(),
    "name.suffix": SEARCH.name.suffix("_s"),
    "a node above": SEARCH.alias("z").abs(),
}


@pytest.mark.parametrize("case", sorted(RENAMED_ROOTS))
def test_a_rename_at_the_root_leaves_the_node_below_it_the_root(case: str) -> None:
    """The cast pass visits the root of a `select` once, and a rename above it keeps it the root.

    At the root, `search_sorted` reads `ewm_var` at its type after its own
    visit, `Float64`, which joins `UInt128`. Below another node, it reads the
    planner type of `ewm_var` too, `Datetime`, which joins no `UInt128`.

    Args:
        case: The rename at the root, or a node above.
    """
    expression = RENAMED_ROOTS[case]
    # `.name.keep()` names its output after the column `d`, which `output_name` does not give.
    name = expression.meta.output_name(raise_if_undetermined=False) or "d"
    leaves = {
        column: ColumnLeaf(DType.of(dtype), RowForm.PER_ROW)
        for column, dtype in ROOT_FRAME.schema.items()
    }
    tree = ExpressionTree().of(expression)
    answers = TreeEvaluation(leaves, GroupBy((3,))).answers(tree)
    select = ROOT_FRAME.lazy().select(expression)
    polars = PolarsAnswer().answer(lambda: select.collect_schema()[name])
    rule = answers.planner_select
    assert rule == polars if isinstance(rule, DType) else isinstance(polars, Stopped)
