"""Check the definitions of the functions of several inputs against polars, node by node.

The family holds the functions that read one row of each input to give one
row: `coalesce`, the horizontal functions such as `max_horizontal`, `clip`,
`is_between`, `is_close`, `is_in`, `arctan2`, `pow`, `replace` and
`pl.struct`. Each test builds the node of polars over the inputs of one cell,
and compares the four answers of polars with the four answers of the rules
(`CellCheck`): the planner type and the run type, in a `select` and in `agg`
over two groups of other sizes.

A node of two inputs takes about 500 pairs beside a typical input at each
place (`PairSample`), and a node of three inputs about 500 cells
(`TripleSample`). The cast pass casts the input of `is_between` to the type
of its bounds with a strict cast, and `clip` casts its bounds to the type of
its input, so where the rules refuse their run, polars runs them over text
that parses as a number: other text fails that cast first, a fact of the
values. `clip` leaves out a `Categorical` input, whose cast of an integer
bound depends on the categories of the process.

The old values of `replace` must not repeat, and each
column of the test frames repeats its three values over five rows, so the
sample of `replace` leaves out each column of one value per row at that
place. Its new values must count one value or as many as the old values.
The rules count the values of a literal, and the rows of a column are a fact
of the data, so the sample leaves out each column of more than one row at
the place of the new values. It also leaves out a `Categorical` input, whose
cast of the old values depends on the categories of the process.

The sample leaves out some cells where polars casts the values of the test
frames before the check that the rules refuse, and a value that does not fit
the cast raises first (`ManyInputNodes.EDGES`). `is_in` casts the values of
a list to `Decimal(3, 3)`, `clip` casts a struct bound, and `replace` casts
its old values to the type of its input with a strict cast. A value that
fits gives the error of the rules, so the class of the error is a fact of
the values.

Some forms need a whole tree over the frame of `TreeFrame`
(`ManyInputNodes.TREES`). `replace_strict` fails for a value of the input
that its old values do not hold, a fact of the data, and the census holds its
mapping as a series, which the rules leave unchecked. So each of its trees has
old values that hold every value of the input, or a default.
`all_horizontal` and `any_horizontal` of other than two inputs become other
nodes while polars converts the expression. `is_between` joins its two masks
with `&`, and two trees check masks of other lengths beside an input of one
row.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import CellCheck, Input, Inputs, NodeSample, TreeFrame, Values
from astrolabe._rules.definitions.many_inputs import ManyInputs


class ManyInputNodes:
    """Hold the nodes of the tests, each with its call and its typical inputs.

    Example:
        >>> sorted(ManyInputNodes.NODES)[:3]
        ['all_horizontal', 'any_horizontal', 'arctan2']
    """

    EDGES: ClassVar[frozenset[str]] = frozenset({
        (
            "clip both-Decimal(precision=3, scale=3)/one row, Struct({'a': Int8})/one value, "
            "Enum(categories=['a', 'b'])/one row"
        ),
        "is_in-Decimal(precision=3, scale=3)/other length, List(Int64)/per row",
        (
            "replace-Decimal(precision=38, scale=0)/one row, Struct({'b': String, 'a': Int8})/one "
            "value, Struct({'b': Int8})/one row"
        ),
        (
            "replace-Decimal(precision=3, scale=3)/one row, Struct({'a': Int8})/one value, "
            "Enum(categories=['a', 'b'])/one row"
        ),
        (
            "replace-UInt128/one value, Struct({'b': String, 'a': Int8})/one row, "
            "Duration(time_unit='ns')/one value"
        ),
        (
            "replace-UInt32/one value, List(Struct({'a': Int8}))/one row, "
            "List(Struct({'a': Int8}))/one value"
        ),
    })
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")
    FLOATS: ClassVar[Input] = Inputs().labelled("Float64/per row")
    BOOLEANS: ClassVar[Input] = Inputs().labelled("Boolean/per row")
    LISTS: ClassVar[Input] = Inputs().labelled("List(Int64)/per row")
    NULLS: ClassVar[Input] = Inputs().labelled("Null/per row")
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    WIDE: ClassVar[Input] = Inputs().labelled("lit int 300/direct")

    NODES: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...]]]] = {
        "coalesce": (CellCheck(ManyInputs.COALESCE, pl.coalesce), (INTEGERS, INTEGERS)),
        "coalesce beside a literal": (
            CellCheck(ManyInputs.COALESCE, pl.coalesce, Values.NUMBER_TEXT),
            (ONE, ONE),
        ),
        "max_horizontal": (
            CellCheck(ManyInputs.MAX_HORIZONTAL, pl.max_horizontal),
            (INTEGERS, INTEGERS),
        ),
        "min_horizontal": (
            CellCheck(ManyInputs.MIN_HORIZONTAL, pl.min_horizontal),
            (INTEGERS, INTEGERS),
        ),
        "sum_horizontal": (
            CellCheck(ManyInputs.SUM_HORIZONTAL, pl.sum_horizontal),
            (INTEGERS, INTEGERS),
        ),
        "sum_horizontal beside a null": (
            CellCheck(ManyInputs.SUM_HORIZONTAL, pl.sum_horizontal),
            (NULLS, NULLS),
        ),
        "mean_horizontal": (
            CellCheck(ManyInputs.MEAN_HORIZONTAL, pl.mean_horizontal),
            (INTEGERS, INTEGERS),
        ),
        "all_horizontal": (
            CellCheck(ManyInputs.ALL_HORIZONTAL, pl.all_horizontal),
            (BOOLEANS, BOOLEANS),
        ),
        "any_horizontal": (
            CellCheck(ManyInputs.ANY_HORIZONTAL, pl.any_horizontal),
            (BOOLEANS, BOOLEANS),
        ),
        "clip": (
            CellCheck(ManyInputs.CLIP, lambda x, y: x.clip(y), Values.NUMBER_TEXT),
            (INTEGERS, ONE),
        ),
        "clip both": (
            CellCheck(ManyInputs.CLIP, lambda x, y, z: x.clip(y, z), Values.NUMBER_TEXT),
            (INTEGERS, ONE, WIDE),
        ),
        "is_between": (
            CellCheck(
                ManyInputs.IS_BETWEEN,
                lambda x, y, z: x.is_between(y, z),
                Values.NUMBER_TEXT,
            ),
            (INTEGERS, ONE, WIDE),
        ),
        "is_close": (CellCheck(ManyInputs.IS_CLOSE, lambda x, y: x.is_close(y)), (FLOATS, FLOATS)),
        "is_in": (CellCheck(ManyInputs.IS_IN, lambda x, y: x.is_in(y)), (INTEGERS, LISTS)),
        "is_in of a literal": (CellCheck(ManyInputs.IS_IN, lambda x, y: x.is_in(y)), (ONE, ONE)),
        "arctan2": (CellCheck(ManyInputs.ATAN2, pl.arctan2), (FLOATS, FLOATS)),
        "pow": (CellCheck(ManyInputs.POW, lambda x, y: x.pow(y)), (INTEGERS, ONE)),
        "replace": (
            CellCheck(ManyInputs.REPLACE, lambda x, y, z: x.replace(y, z)),
            (INTEGERS, ONE, WIDE),
        ),
        "struct": (CellCheck(ManyInputs.AS_STRUCT, pl.struct), (INTEGERS, FLOATS)),
    }

    TREES: ClassVar[dict[str, pl.Expr]] = {
        "all_horizontal of one input": pl.all_horizontal("x"),
        "all_horizontal of text": pl.all_horizontal("t"),
        "any_horizontal of three inputs": pl.any_horizontal("x", "g", pl.col("x") > 1),
        "is_between with masks of other lengths": pl
        .col("x")
        .first()
        .is_between(pl.col("x").head(2), pl.col("x")),
        "is_between with a mask of one row": pl
        .col("x")
        .first()
        .is_between(pl.col("x").head(2), pl.col("x").last()),
        "to the type of the new values": pl.col("x").replace_strict(
            pl.col("x"),
            pl.col("x").cast(pl.Float32),
        ),
        "with a return type": pl.col("x").replace_strict(
            pl.col("x"),
            pl.col("x"),
            return_dtype=pl.Int16,
        ),
        "with a default": pl.col("x").replace_strict(pl.lit(1), pl.lit(2.5), default=pl.lit(0)),
        "with a default and a return type": pl.col("x").replace_strict(
            pl.lit(1),
            pl.lit(2),
            default=pl.lit(0),
            return_dtype=pl.Int8,
        ),
        "old values that do not cast": pl.col("x").replace_strict(pl.lit("a"), pl.lit(1)),
        "a literal old value that does not fit": pl
        .col("x")
        .cast(pl.UInt8)
        .replace_strict(
            pl.lit(-1),
            pl.lit(2),
            default=pl.lit(0),
        ),
        "a default of another length": pl
        .col("x")
        .first()
        .replace_strict(
            pl.lit(1),
            pl.lit(2),
            default=pl.col("x"),
        ),
        "a struct default beside text values": pl.col("x").replace_strict(
            pl.lit(1),
            pl.lit("a"),
            default=pl.col("s"),
        ),
    }

    @classmethod
    def cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(ManyInputNodes.cells()) < 10000
            True
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical) in cls.NODES.items():
            sample = NodeSample().cells(len(typical), typical)
            cells.extend((name, cell) for cell in sample if not cls._left_out(name, cell))
        return cells

    @classmethod
    def _left_out(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether the sample leaves a cell out, for one of the reasons of the module.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the sample leaves the cell out.

        Example:
            >>> cell = (ManyInputNodes.INTEGERS, ManyInputNodes.ONE, ManyInputNodes.INTEGERS)
            >>> ManyInputNodes._left_out("replace", cell)
            True
        """
        if cls.ids([(name, cell)])[0] in cls.EDGES:
            return True
        if name == "is_in of a literal":
            return cell[0].label == "lit null/direct"
        if name.startswith("clip"):
            return cell[0].label.startswith("Categorical/")
        if name != "replace":
            return False
        rows = ("/per row", "/other length")
        if cell[1].label.endswith("/per row") or cell[2].label.endswith(rows):
            return True
        return cell[0].label.startswith("Categorical/")

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> ManyInputNodes.ids([("pow", (ManyInputNodes.INTEGERS, ManyInputNodes.ONE))])
            ['pow-Int64/per row, lit int 1/direct']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    ManyInputNodes.cells(),
    ids=ManyInputNodes.ids(ManyInputNodes.cells()),
)
def test_function_of_many_inputs_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a function of several inputs match polars, at one cell of its sample."""
    problems = ManyInputNodes.NODES[node][0].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("case", sorted(ManyInputNodes.TREES))
def test_a_tree_of_the_family_gives_the_answers_of_polars(case: str) -> None:
    """The four answers of a whole tree match polars, for a form that a cell cannot build."""
    expression = ManyInputNodes.TREES[case]
    assert TreeFrame().rules(expression) == TreeFrame().polars(expression)
