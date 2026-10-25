"""Check the definitions of `astrolabe._rules.definitions` against polars, node by node.

A definition names the parts of the vocabulary that the rules of one node
use, and `NodeEvaluation` gives the four answers of the node from it: the
planner type and the run type, in a `select` and in `group_by().agg()`. Each
test here builds the node of polars over inputs of every form, and compares
the four answers of polars with the four answers of the rules (`CellCheck`).

An input is a column of each type of `RuleKinds` in each of the four row
forms, or a literal of each class of the census, direct or derived. A node of
one input takes each input. The nodes of the pilot of two inputs take each
input at the left, beside each input at the right. Every other node takes
the sample of its family, about 500 cells for each node (`NodeSample`): every
input for a node of one input, and about 500 pairs for a node of two inputs.
A failure names the node and the cell.

A node that reads an option, such as the target of a cast, runs at several
values of the option. The rules read the option from the tree that polars
builds for the call. A call that differs from the main call of its node only
by an option takes a smaller sample.

The frame has two groups of different sizes, 3 and 2 rows, because one group
hides the checks of polars between groups. A run answer is the first of the
samples of values that runs, as in the census: a value that does not fit one
sample is a fact of the data, not of the types. So the samples hold text that
parses as a number, as a date, as a time, and as a datetime with an offset,
and a NaN as the first float (`LiveFrames`). Where the rules refuse, one
sample is enough, because a type error refuses every sample. A cell where
the rules say that the run ends the process or never returns does not run
here. The tests of the kernels run those cells in a subprocess.
"""

from __future__ import annotations

import json
import math
from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    PairSample,
    PolarsCell,
    TemplateCall,
    Values,
)
from astrolabe._rules.definitions.aggregations import Aggregations
from astrolabe._rules.definitions.binary_operators import BinaryOperators
from astrolabe._rules.definitions.elementwise import Elementwise
from astrolabe._rules.definitions.length_preserving import LengthPreserving
from astrolabe._rules.definitions.strings import Strings
from astrolabe._rules.definitions.structs import Structs
from astrolabe._rules.definitions.temporal import Temporal
from astrolabe._rules.definitions.node_options import NodeOptions
from astrolabe._rules.dtypes import DType
from astrolabe._rules.node_evaluation import LiteralLeaf, NodeEvaluation, Unchecked
from astrolabe._rules.tree import NonFinite


class Nodes:
    """Hold the nodes of the tests: each definition with the call of polars that builds its node.

    A node that reads an option runs at several values of it: the targets and
    the options of a cast, the units of `dt.cast_time_unit`, the names of
    `struct.field`, and the direction of `cum_sum`. The calls of `strptime`
    run on a sample of pairs of inputs (`PairSample`), each with the sample
    of values whose text its format parses. Their typical text is a `String`
    column, and their typical second input is the literal `raise`.

    The elementwise functions of one input take every input, and their calls
    at other values of an option take a smaller sample (`NodeSample`). `log`
    takes a sample of about 500 pairs of a value and a base. Its typical
    value is a `Float32` column, and its typical base is a float literal, as
    in `log()` and `log10()`.

    Each binary operator takes a sample of about 500 pairs. The typical
    inputs of arithmetic and of a comparison are an `Int64` column and the
    literal `1`. The typical inputs of `&`, `|` and `^` are a `Boolean`
    column and the literal `True`.

    Example:
        >>> sorted(Nodes.ONE_INPUT)[:4]
        ['abs', 'cast Array(Int16, 2)', 'cast Boolean', 'cast Date']
        >>> Nodes.BASE.label
        'lit float/direct'
    """

    ONE_INPUT: ClassVar[dict[str, CellCheck]] = {
        "abs": CellCheck(Elementwise.ABS, lambda x: x.abs()),
        "sqrt": CellCheck(Elementwise.SQRT, lambda x: x.sqrt()),
        "cum_sum": CellCheck(LengthPreserving.CUM_SUM, lambda x: x.cum_sum()),
        "cum_sum reverse": CellCheck(LengthPreserving.CUM_SUM, lambda x: x.cum_sum(reverse=True)),
        "sum": CellCheck(Aggregations.SUM, lambda x: x.sum()),
        "cast Int32": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Int32)),
        "cast Int8 not strict": CellCheck(
            Elementwise.CAST, lambda x: x.cast(pl.Int8, strict=False)
        ),
        "cast UInt8 wrapping": CellCheck(
            Elementwise.CAST,
            lambda x: x.cast(pl.UInt8, wrap_numerical=True),
        ),
        "cast Float32": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Float32)),
        "cast String": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.String)),
        "cast Boolean": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Boolean)),
        "cast Date": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Date)),
        "cast Datetime(ms, UTC)": CellCheck(
            Elementwise.CAST,
            lambda x: x.cast(pl.Datetime("ms", "UTC")),
        ),
        "cast Duration(ns)": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Duration("ns"))),
        "cast Decimal(5, 2)": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Decimal(5, 2))),
        "cast Enum": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Enum(["a", "b"]))),
        "cast List(Int16)": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.List(pl.Int16))),
        "cast Array(Int16, 2)": CellCheck(
            Elementwise.CAST,
            lambda x: x.cast(pl.Array(pl.Int16, 2)),
        ),
        "cast Struct": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Struct({"a": pl.Int32}))),
        "cast Null": CellCheck(Elementwise.CAST, lambda x: x.cast(pl.Null)),
        "cast_time_unit ms": CellCheck(
            Temporal.CAST_TIME_UNIT,
            lambda x: x.dt.cast_time_unit("ms"),
        ),
        "cast_time_unit us": CellCheck(
            Temporal.CAST_TIME_UNIT,
            lambda x: x.dt.cast_time_unit("us"),
        ),
        "cast_time_unit ns": CellCheck(
            Temporal.CAST_TIME_UNIT,
            lambda x: x.dt.cast_time_unit("ns"),
        ),
        "struct.field a": CellCheck(Structs.STRUCT_FIELD, lambda x: x.struct.field("a")),
        "struct.field b": CellCheck(Structs.STRUCT_FIELD, lambda x: x.struct.field("b")),
    }
    TWO_INPUTS: ClassVar[dict[str, CellCheck]] = {
        "add": CellCheck(BinaryOperators.ADD, lambda x, y: x + y),
        "fill_null": CellCheck(Elementwise.FILL_NULL, lambda x, y: x.fill_null(y)),
    }
    STRPTIME: ClassVar[dict[str, tuple[str, Values]]] = {
        "to_date %Y-%m-%d": ("pl.col('x').str.to_date('%Y-%m-%d')", Values.DATE_TEXT),
        "to_date inferred": ("pl.col('x').str.to_date()", Values.DATE_TEXT),
        "to_datetime ms %Y-%m-%d": (
            "pl.col('x').str.to_datetime('%Y-%m-%d', time_unit='ms')",
            Values.DATE_TEXT,
        ),
        "to_datetime inferred": ("pl.col('x').str.to_datetime()", Values.DATE_TEXT),
        "to_datetime not strict": (
            "pl.col('x').str.to_datetime('%Y-%m-%d', strict=False)",
            Values.DATE_TEXT,
        ),
        "strptime ns %:z": (
            "pl.col('x').str.strptime(pl.Datetime('ns'), '%Y-%m-%dT%H:%M:%S%:z')",
            Values.ZONED_TEXT,
        ),
        "to_time %H:%M": ("pl.col('x').str.to_time('%H:%M')", Values.TIME_TEXT),
    }
    TEXT: ClassVar[Input] = Inputs().columns()[13 * 4]
    RAISE: ClassVar[Input] = Input("lit raise", LiteralLeaf("raise"), pl.lit("raise"))
    ELEMENTWISE: ClassVar[dict[str, CellCheck]] = {
        "cbrt": CellCheck(Elementwise.CBRT, lambda x: x.cbrt()),
        "exp": CellCheck(Elementwise.EXP, lambda x: x.exp()),
        "log1p": CellCheck(Elementwise.LOG1P, lambda x: x.log1p()),
        "sin": CellCheck(Elementwise.SIN, lambda x: x.sin()),
        "cos": CellCheck(Elementwise.COS, lambda x: x.cos()),
        "tan": CellCheck(Elementwise.TAN, lambda x: x.tan()),
        "cot": CellCheck(Elementwise.COT, lambda x: x.cot()),
        "arcsin": CellCheck(Elementwise.ARCSIN, lambda x: x.arcsin()),
        "arccos": CellCheck(Elementwise.ARCCOS, lambda x: x.arccos()),
        "arctan": CellCheck(Elementwise.ARCTAN, lambda x: x.arctan()),
        "sinh": CellCheck(Elementwise.SINH, lambda x: x.sinh()),
        "cosh": CellCheck(Elementwise.COSH, lambda x: x.cosh()),
        "tanh": CellCheck(Elementwise.TANH, lambda x: x.tanh()),
        "arcsinh": CellCheck(Elementwise.ARCSINH, lambda x: x.arcsinh()),
        "arccosh": CellCheck(Elementwise.ARCCOSH, lambda x: x.arccosh()),
        "arctanh": CellCheck(Elementwise.ARCTANH, lambda x: x.arctanh()),
        "degrees": CellCheck(Elementwise.DEGREES, lambda x: x.degrees()),
        "radians": CellCheck(Elementwise.RADIANS, lambda x: x.radians()),
        "neg": CellCheck(Elementwise.NEGATE, lambda x: x.neg()),
        "sign": CellCheck(Elementwise.SIGN, lambda x: x.sign()),
        "floor": CellCheck(Elementwise.FLOOR, lambda x: x.floor()),
        "ceil": CellCheck(Elementwise.CEIL, lambda x: x.ceil()),
        "round": CellCheck(Elementwise.ROUND, lambda x: x.round()),
        "truncate": CellCheck(Elementwise.TRUNCATE, lambda x: x.truncate()),
        "round_sig_figs": CellCheck(Elementwise.ROUND_SIG_FIGS, lambda x: x.round_sig_figs(2)),
        "is_nan": CellCheck(Elementwise.IS_NAN, lambda x: x.is_nan()),
        "is_not_nan": CellCheck(Elementwise.IS_NOT_NAN, lambda x: x.is_not_nan()),
        "is_finite": CellCheck(Elementwise.IS_FINITE, lambda x: x.is_finite()),
        "is_infinite": CellCheck(Elementwise.IS_INFINITE, lambda x: x.is_infinite()),
        "is_null": CellCheck(Elementwise.IS_NULL, lambda x: x.is_null()),
        "is_not_null": CellCheck(Elementwise.IS_NOT_NULL, lambda x: x.is_not_null()),
        "not_": CellCheck(Elementwise.NOT, lambda x: x.not_()),
        "bitwise_count_ones": CellCheck(
            Elementwise.BITWISE_COUNT_ONES,
            lambda x: x.bitwise_count_ones(),
        ),
        "bitwise_count_zeros": CellCheck(
            Elementwise.BITWISE_COUNT_ZEROS,
            lambda x: x.bitwise_count_zeros(),
        ),
        "bitwise_leading_ones": CellCheck(
            Elementwise.BITWISE_LEADING_ONES,
            lambda x: x.bitwise_leading_ones(),
        ),
        "bitwise_leading_zeros": CellCheck(
            Elementwise.BITWISE_LEADING_ZEROS,
            lambda x: x.bitwise_leading_zeros(),
        ),
        "bitwise_trailing_ones": CellCheck(
            Elementwise.BITWISE_TRAILING_ONES,
            lambda x: x.bitwise_trailing_ones(),
        ),
        "bitwise_trailing_zeros": CellCheck(
            Elementwise.BITWISE_TRAILING_ZEROS,
            lambda x: x.bitwise_trailing_zeros(),
        ),
        "hash": CellCheck(Elementwise.HASH, lambda x: x.hash()),
        "to_physical": CellCheck(Elementwise.TO_PHYSICAL, lambda x: x.to_physical()),
        "set_sorted": CellCheck(Elementwise.SET_SORTED, lambda x: x.set_sorted()),
        "reinterpret signed": CellCheck(
            Elementwise.REINTERPRET,
            lambda x: x.reinterpret(signed=True),
        ),
        "cut": CellCheck(Elementwise.CUT, lambda x: x.cut([0, 2])),
    }
    ELEMENTWISE_OPTIONS: ClassVar[dict[str, CellCheck]] = {
        "round 2 half away from zero": CellCheck(
            Elementwise.ROUND,
            lambda x: x.round(2, mode="half_away_from_zero"),
        ),
        "truncate 2": CellCheck(Elementwise.TRUNCATE, lambda x: x.truncate(2)),
        "round_sig_figs 0": CellCheck(Elementwise.ROUND_SIG_FIGS, lambda x: x.round_sig_figs(0)),
        "hash with seeds": CellCheck(Elementwise.HASH, lambda x: x.hash(1, 2, 3, 4)),
        "set_sorted descending": CellCheck(
            Elementwise.SET_SORTED,
            lambda x: x.set_sorted(descending=True),
        ),
        "reinterpret unsigned": CellCheck(
            Elementwise.REINTERPRET,
            lambda x: x.reinterpret(signed=False),
        ),
        "reinterpret Float32": CellCheck(
            Elementwise.REINTERPRET,
            lambda x: x.reinterpret(dtype=pl.Float32),
        ),
        "reinterpret UInt64": CellCheck(
            Elementwise.REINTERPRET,
            lambda x: x.reinterpret(dtype=pl.UInt64),
        ),
        "reinterpret Decimal": CellCheck(
            Elementwise.REINTERPRET,
            lambda x: x.reinterpret(dtype=pl.Decimal(10, 2)),
        ),
        "cut unsorted": CellCheck(Elementwise.CUT, lambda x: x.cut([2, 0])),
        "cut labels": CellCheck(Elementwise.CUT, lambda x: x.cut([0, 2], labels=["a", "b", "c"])),
        "cut left closed": CellCheck(Elementwise.CUT, lambda x: x.cut([0, 2], left_closed=True)),
        "cut with breaks": CellCheck(
            Elementwise.CUT,
            lambda x: x.cut([0, 2], include_breaks=True),
        ),
        "cut far breaks": CellCheck(Elementwise.CUT, lambda x: x.cut([0.5, 1e20, -1e-7])),
        "cut equal breaks": CellCheck(Elementwise.CUT, lambda x: x.cut([1, 1])),
        "cut infinite break with labels": CellCheck(
            Elementwise.CUT,
            lambda x: x.cut([0, float("inf")], labels=["a", "b", "c"]),
        ),
        "cut few labels": CellCheck(Elementwise.CUT, lambda x: x.cut([0, 2], labels=["a"])),
        "cut equal labels": CellCheck(
            Elementwise.CUT,
            lambda x: x.cut([0, 2], labels=["a", "a", "b"]),
        ),
    }
    LOG: ClassVar[CellCheck] = CellCheck(Elementwise.LOG, TemplateCall("pl.col('x').log()"))
    NUMBERS: ClassVar[tuple[Input, Input]] = (
        Inputs().labelled("Int64/per row"),
        Inputs().labelled("lit int 1/direct"),
    )
    TRUTHS: ClassVar[tuple[Input, Input]] = (
        Inputs().labelled("Boolean/per row"),
        Inputs().labelled("lit bool/direct"),
    )
    OPERATORS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, Input]]]] = {
        "sub": (CellCheck(BinaryOperators.SUBTRACT, lambda x, y: x - y), NUMBERS),
        "mul": (CellCheck(BinaryOperators.MULTIPLY, lambda x, y: x * y), NUMBERS),
        "truediv": (CellCheck(BinaryOperators.TRUE_DIVIDE, lambda x, y: x / y), NUMBERS),
        "floordiv": (CellCheck(BinaryOperators.FLOOR_DIVIDE, lambda x, y: x // y), NUMBERS),
        "mod": (CellCheck(BinaryOperators.MODULUS, lambda x, y: x % y), NUMBERS),
        "eq": (CellCheck(BinaryOperators.EQ, lambda x, y: x == y), NUMBERS),
        "eq_missing": (
            CellCheck(BinaryOperators.EQ_MISSING, lambda x, y: x.eq_missing(y)),
            NUMBERS,
        ),
        "ne": (CellCheck(BinaryOperators.NOT_EQ, lambda x, y: x != y), NUMBERS),
        "ne_missing": (
            CellCheck(BinaryOperators.NOT_EQ_MISSING, lambda x, y: x.ne_missing(y)),
            NUMBERS,
        ),
        "lt": (CellCheck(BinaryOperators.LT, lambda x, y: x < y), NUMBERS),
        "le": (CellCheck(BinaryOperators.LT_EQ, lambda x, y: x <= y), NUMBERS),
        "gt": (CellCheck(BinaryOperators.GT, lambda x, y: x > y), NUMBERS),
        "ge": (CellCheck(BinaryOperators.GT_EQ, lambda x, y: x >= y), NUMBERS),
        "and": (CellCheck(BinaryOperators.AND, lambda x, y: x & y), TRUTHS),
        "or": (CellCheck(BinaryOperators.OR, lambda x, y: x | y), TRUTHS),
        "xor": (CellCheck(BinaryOperators.XOR, lambda x, y: x ^ y), TRUTHS),
    }

    @classmethod
    def operator_cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give the cells of the binary operators: each operator beside each cell of its sample.

        Returns:
            The name of the operator and the cell, for each cell.

        Example:
            >>> cells = Nodes.operator_cells()
            >>> len(cells), cells[0][0]
            (8000, 'sub')
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical) in cls.OPERATORS.items():
            cells.extend((name, cell) for cell in NodeSample().cells(2, typical))
        return cells

    FLOAT: ClassVar[Input] = Inputs().labelled("Float32/per row")
    BASE: ClassVar[Input] = Inputs().labelled("lit float/direct")


@pytest.mark.parametrize("node", list(Nodes.ONE_INPUT))
@pytest.mark.parametrize("entry", Inputs().all(), ids=lambda entry: entry.label)
def test_node_of_one_input_gives_the_answers_of_polars(node: str, entry: Input) -> None:
    """The four answers of a node of one input match polars, at one input."""
    problems = Nodes.ONE_INPUT[node].problems([entry])
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.TWO_INPUTS))
@pytest.mark.parametrize("left", Inputs().all(), ids=lambda entry: entry.label)
def test_node_of_two_inputs_gives_the_answers_of_polars(node: str, left: Input) -> None:
    """The four answers of a node of two inputs match polars: one left input beside each input."""
    check = Nodes.TWO_INPUTS[node]
    problems: list[str] = []
    for right in Inputs().all():
        problems.extend(check.problems([left, right]))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.STRPTIME))
def test_strptime_gives_the_answers_of_polars_on_a_sample(node: str) -> None:
    """The four answers of `str.strptime` match polars on a sample of about 500 pairs of inputs.

    The literal text of the sample is a text that the format of the call parses.
    """
    code, values = Nodes.STRPTIME[node]
    check = CellCheck(Strings.STRPTIME, TemplateCall(code))
    text = values.text()
    assert text is not None
    sample = PairSample(Inputs().with_text(text), Nodes.TEXT, Nodes.RAISE)
    problems: list[str] = []
    for left, right in sample.pairs():
        problems.extend(check.problems([left, right]))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.ELEMENTWISE))
@pytest.mark.parametrize("cell", NodeSample().cells(1), ids=CellCheck.label)
def test_elementwise_node_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of an elementwise function of one input match polars, at one input."""
    problems = Nodes.ELEMENTWISE[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("node", list(Nodes.ELEMENTWISE_OPTIONS))
@pytest.mark.parametrize("cell", NodeSample().options(), ids=CellCheck.label)
def test_elementwise_option_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of an elementwise function at another value of an option match polars."""
    problems = Nodes.ELEMENTWISE_OPTIONS[node].problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "cell",
    NodeSample().cells(2, (Nodes.FLOAT, Nodes.BASE)),
    ids=CellCheck.label,
)
def test_log_gives_the_answers_of_polars(cell: tuple[Input, ...]) -> None:
    """The four answers of `log` match polars, at one value and one base of its sample."""
    problems = Nodes.LOG.problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    ("node", "cell"),
    Nodes.operator_cells(),
    ids=[f"{node}-{CellCheck.label(cell)}" for node, cell in Nodes.operator_cells()],
)
def test_binary_operator_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of a binary operator match polars, at one cell of its sample."""
    check, _ = Nodes.OPERATORS[node]
    problems = check.problems(list(cell))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("breaks", [[0, math.inf], [-math.inf, 0], [math.nan, 0]], ids=repr)
def test_cut_over_a_break_that_the_tree_does_not_hold_is_unchecked(breaks: list[float]) -> None:
    """The rules give `Unchecked` for `cut` with no labels and a break that is NaN or infinite.

    The JSON form of the tree writes each such break as null, so the rules do
    not know the interval of the break, and polars gives its label.
    """
    node = pl.col("c10").cut(breaks)
    definition = Elementwise.CUT.at(NodeOptions(json.loads(node.meta.serialize(format="json"))))
    leaf = Inputs().labelled("Float64/per row").leaf
    answers = NodeEvaluation(definition, [leaf], CellCheck.GROUPS).answers()
    unchecked = [answers.planner_select, answers.planner_agg, answers.run_select, answers.run_agg]
    assert all(isinstance(answer, Unchecked) for answer in unchecked), unchecked
    planned = PolarsCell(node.alias("o")).planner(agg=False)
    assert isinstance(planned, DType)


def test_cut_over_a_nan_break_that_the_code_holds_gives_the_answers_of_polars() -> None:
    """A NaN that the code holds reaches the rules through its stand-in, so its bin is known.

    The planner names the bin `NaN`, and the run raises `ComputeError` at
    every frame.
    """
    written = pl.col("c10").cut([NonFinite().of("nan"), 0.0])
    tree = json.loads(written.meta.serialize(format="json"), parse_float=NonFinite().parsed)
    definition = Elementwise.CUT.at(NodeOptions(tree))
    leaf = Inputs().labelled("Float64/per row").leaf
    answers = NodeEvaluation(definition, [leaf], CellCheck.GROUPS).answers()
    cell = PolarsCell(pl.col("c10").cut([math.nan, 0.0]).alias("o"))  # type: ignore[invalid-call-shape]
    assert answers.planner_select == cell.planner(agg=False)
    assert str(answers.run_select) == str(cell.run(agg=False, every=False))
