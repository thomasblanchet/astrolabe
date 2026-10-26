"""Check the definitions of the range functions against polars, node by node.

The family holds the functions of one range, such as `int_range` and
`date_range`, and the functions of many ranges, such as `int_ranges` and
`date_ranges`. Each test builds the node of polars over the inputs of one
cell, and compares the four answers of polars with the four answers of the
rules (`CellCheck`): the planner type and the run type, in a `select` and in
`agg` over two groups of other sizes.

A node of two inputs takes about 500 pairs beside a typical input at each
place (`PairSample`), and a node of three inputs about 500 cells
(`TripleSample`). A range holds as many values as its bounds or its count
say, so the integer ranges and the linear spaces leave out each input of a
large value (`RangeNodes.LARGE`): the literal `2**40`, the derived `-1`, which
polars can wrap to a large unsigned number, and the temporal columns, whose
physical values are large. Such a range takes more memory than the machine
holds, and the rules read only the types. The step of `int_ranges` leaves out
a `Decimal` whose precision equals its scale (`RangeNodes.ZERO_STEPS`): the
cast to `Int64` rounds each value half to even, so each value of the samples
below one half casts to a step of zero, where polars panics. The large
sample casts to a step of 1, and its bounds make ranges that take more
memory than the machine holds. A value further from zero than one half
casts to a step of 1 or -1, so the panic is a fact of the values, which the
rules accept. An answer that no sample shows before a release of polars is
left out (`RangeNodes.UNSHOWN`), and a test of its own checks it.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    PolarsAnswer,
    Stopped,
    Values,
)
from astrolabe._rules.definitions.node_options import NodeOptions
from astrolabe._rules.definitions.ranges import Ranges
from astrolabe._rules.dtypes import DType
from astrolabe._rules.node_evaluation import (
    ColumnLeaf,
    Leaf,
    LiteralLeaf,
    NodeEvaluation,
    Refused,
)
from astrolabe._rules.options import RowForm
from astrolabe._rules.releases import PolarsRelease


class RangeNodes:
    """Hold the nodes of the tests, each with its call, its typical inputs and its sample.

    `allocates` says that the values of the bounds or of the count decide the
    memory of the run, so the sample leaves out the inputs of `LARGE`.

    Example:
        >>> sorted(RangeNodes.NODES)[:3]
        ['date_range', 'date_ranges', 'datetime_range']
        >>> "lit int 2**40/derived" in RangeNodes.LARGE
        True
    """

    LARGE: ClassVar[frozenset[str]] = frozenset(
        entry.label
        for entry in Inputs().all()
        if entry.label.startswith(("Datetime(", "Duration(", "Time/", "lit int 2**40"))
        or entry.label == "lit int -1/derived"
    )
    # The answers that no sample shows before a release of polars, by the call and
    # the cell, each with that release. A place that ends at `run` names both
    # runs. A test of its own checks each one
    # (`test_int_range_over_a_struct_of_a_date_stops_as_the_rules_say`).
    UNSHOWN: ClassVar[dict[tuple[str, str], str]] = {
        ("int_range UInt8", "Struct({'a': Date})/one value, lit int 1/direct: run"): "1.40.0",
        ("int_range UInt8", "Int64/one value, Struct({'a': Date})/one value: run"): "1.40.0",
    }
    ZERO_STEPS: ClassVar[frozenset[str]] = frozenset(
        entry.label
        for entry in Inputs().all()
        if entry.label.startswith("Decimal(precision=3, scale=3)")
    )
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    INTEGER: ClassVar[Input] = Inputs().labelled("Int64/one value")
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")
    FLOAT: ClassVar[Input] = Inputs().labelled("Float64/one value")
    FLOATS: ClassVar[Input] = Inputs().labelled("Float64/per row")
    DATE: ClassVar[Input] = Inputs().labelled("Date/one value")
    DATES: ClassVar[Input] = Inputs().labelled("Date/per row")
    DATETIME: ClassVar[Input] = Inputs().labelled(
        "Datetime(time_unit='ms', time_zone=None)/one value"
    )
    DATETIMES: ClassVar[Input] = Inputs().labelled(
        "Datetime(time_unit='ms', time_zone=None)/per row"
    )
    TIME: ClassVar[Input] = Inputs().labelled("Time/one value")
    TIMES: ClassVar[Input] = Inputs().labelled("Time/per row")

    NODES: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...], bool]]] = {
        "int_range": (
            CellCheck(Ranges.INT_RANGE, pl.int_range, Values.NUMBER_TEXT),
            (INTEGER, ONE),
            True,
        ),
        "int_range UInt8": (
            CellCheck(
                Ranges.INT_RANGE,
                lambda x, y: pl.int_range(x, y, dtype=pl.UInt8),
                Values.NUMBER_TEXT,
            ),
            (INTEGER, ONE),
            True,
        ),
        "int_ranges": (
            CellCheck(Ranges.INT_RANGES, pl.int_ranges, Values.NUMBER_TEXT),
            (INTEGERS, INTEGERS, ONE),
            True,
        ),
        "linear_space": (
            CellCheck(Ranges.LINEAR_SPACE, pl.linear_space),
            (FLOAT, FLOAT, ONE),
            True,
        ),
        "linear_spaces": (
            CellCheck(Ranges.LINEAR_SPACES, pl.linear_spaces, Values.NUMBER_TEXT),
            (FLOATS, FLOATS, ONE),
            True,
        ),
        "date_range": (
            CellCheck(Ranges.DATE_RANGE, pl.date_range),
            (DATE, DATE),
            False,
        ),
        "date_ranges": (
            CellCheck(Ranges.DATE_RANGES, pl.date_ranges),
            (DATES, DATES),
            False,
        ),
        "datetime_range": (
            CellCheck(Ranges.DATETIME_RANGE, pl.datetime_range),
            (DATETIME, DATETIME),
            False,
        ),
        "datetime_range ns UTC": (
            CellCheck(
                Ranges.DATETIME_RANGE,
                lambda x, y: pl.datetime_range(x, y, "90ns", time_unit=None, time_zone="UTC"),
            ),
            (DATE, DATE),
            False,
        ),
        "datetime_ranges": (
            CellCheck(Ranges.DATETIME_RANGES, pl.datetime_ranges),
            (DATETIMES, DATETIMES),
            False,
        ),
        "time_range": (
            CellCheck(Ranges.TIME_RANGE, pl.time_range),
            (TIME, TIME),
            False,
        ),
        "time_ranges": (
            CellCheck(Ranges.TIME_RANGES, pl.time_ranges),
            (TIMES, TIMES),
            False,
        ),
    }

    @classmethod
    def cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(RangeNodes.cells()) > 5000
            True
        """
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, (_, typical, _) in cls.NODES.items():
            sample = NodeSample().cells(len(typical), typical)
            cells.extend((name, cell) for cell in sample if not cls._left_out(name, cell))
        return cells

    @classmethod
    def _left_out(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether the sample leaves a cell out: a large range, or a step of zero.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the sample leaves the cell out.

        Example:
            >>> large = Inputs().labelled("lit int 2**40/direct")
            >>> RangeNodes._left_out("int_range", (RangeNodes.INTEGER, large))
            True
        """
        if cls.NODES[name][2] and any(entry.label in cls.LARGE for entry in cell):
            return True
        return name == "int_ranges" and cell[2].label in cls.ZERO_STEPS

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> RangeNodes.ids([("date_range", (RangeNodes.DATE, RangeNodes.DATE))])
            ['date_range-Date/one value, Date/one value']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    RangeNodes.cells(),
    ids=RangeNodes.ids(RangeNodes.cells()),
)
def test_range_gives_the_answers_of_polars(node: str, cell: tuple[Input, ...]) -> None:
    """The four answers of a range function match polars, at one cell of its sample.

    An answer that no sample shows in the installed release is left out (`RangeNodes.UNSHOWN`).
    """
    problems = RangeNodes.NODES[node][0].problems(list(cell))
    installed = PolarsRelease.installed()
    unshown = {
        place
        for (call, place), release in RangeNodes.UNSHOWN.items()
        if call == node and installed < PolarsRelease.parse(release)
    }
    shown = [problem for problem in problems if not problem.startswith(tuple(unshown))]
    assert not shown, "\n".join(shown)


def test_int_range_over_a_struct_of_a_date_stops_as_the_rules_say() -> None:
    """`int_range` over a struct of one `Date` stops in the way of the rules, where the date fits.

    Before polars 1.40.0, the cast pass casts each bound to the type of the
    call with a strict cast, and a struct casts field by field. A date that
    fits `UInt8` leaves a struct, and the kernel panics over it. Each date of
    the samples is too large for `UInt8`, so the strict cast raises first,
    and no sample shows the panic (`RangeNodes.UNSHOWN`). From polars 1.40.0,
    the conversion refuses a struct, whatever its values.
    """
    node = pl.int_range(pl.col("s").first(), 1, dtype=pl.UInt8)
    tree = json.loads(node.meta.serialize(format="json"))
    definition = Ranges.INT_RANGE.at(NodeOptions(tree))
    struct = DType("Struct", ((("a", DType("Date")),),))
    leaves: list[Leaf] = [ColumnLeaf(struct, RowForm.ONE_VALUE), LiteralLeaf(1)]
    rules = NodeEvaluation(definition, leaves, CellCheck.GROUPS).answers().run_select
    frame = pl.DataFrame({"s": [{"a": dt.date(1970, 1, 2)}]})
    ran = PolarsAnswer().answer(lambda: frame.select(node.alias("o")).schema["o"])
    assert isinstance(rules, Refused)
    assert ran == Stopped(rules.stop, None if rules.error is None else rules.error.value)
