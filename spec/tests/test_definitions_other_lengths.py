"""Check the definitions of the functions whose output has a length of its own, node by node.

The family holds the functions that give a number of rows that their values
or their options decide, such as `unique`, `drop_nulls`, `top_k`, `repeat`,
`extend_constant` and `append`. Each test builds the node of polars over the
inputs of one cell, and compares the four answers of polars with the four
answers of the rules (`CellCheck`): the planner type and the run type, in a
`select` and in `agg` over two groups of other sizes.

A node of one input takes every input (`NodeSample`), a node of two inputs
about 500 pairs beside a typical input at each place (`PairSample`), and a
node of three inputs about 500 cells (`TripleSample`). A count decides the
length of `repeat`, `extend_constant`, `repeat_by`, `top_k`, `top_k_by` and
`sample`. So their count leaves out each input of a large value
(`OtherLengthNodes.LARGE`): the literal `2**40`, the derived `-1`, which
polars can wrap to a large unsigned number, and the temporal columns, whose
physical values are large. Their runs skip the samples of large values too.
Such a count takes more memory than the machine holds, and the rules read
only its type. A count casts to an integer with a strict cast, so where the
rules refuse a run, polars runs it over text that parses as a number.

Three facts of the data decide some cells, so the samples leave them out:
the count `300` of `sample` against the five rows of the test frames, and the
leaves of a list, whose number `reshape` reads.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import (
    CellCheck,
    Input,
    Inputs,
    NodeSample,
    Values,
)
from astrolabe._rules.definitions.other_lengths import OtherLengths


class OtherLengthNodes:
    """Hold the nodes of the tests, each with its call, its typical inputs and its count.

    Example:
        >>> sorted(OtherLengthNodes.ONE_INPUT)[:3]
        ['arg_true', 'arg_unique', 'arg_where']
        >>> OtherLengthNodes.MANY_INPUTS["repeat"][2]
        1
    """

    LARGE: ClassVar[frozenset[str]] = frozenset(
        entry.label
        for entry in Inputs().all()
        if entry.label.startswith(("Datetime(", "Duration(", "Time/", "lit int 2**40"))
        or entry.label == "lit int -1/derived"
    )
    SKIP: ClassVar[frozenset[Values]] = frozenset({Values.LARGE})
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")
    INTEGER: ClassVar[Input] = Inputs().labelled("Int64/one value")
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")
    CATEGORY: ClassVar[Input] = Inputs().labelled("Enum(categories=['a', 'b', 'c'])/one value")

    ONE_INPUT: ClassVar[dict[str, CellCheck]] = {
        "unique": CellCheck(OtherLengths.UNIQUE, lambda x: x.unique()),
        "mode": CellCheck(OtherLengths.MODE, lambda x: x.mode()),
        "arg_unique": CellCheck(OtherLengths.ARG_UNIQUE, lambda x: x.arg_unique()),
        "unique_counts": CellCheck(OtherLengths.UNIQUE_COUNTS, lambda x: x.unique_counts()),
        "drop_nulls": CellCheck(OtherLengths.DROP_NULLS, lambda x: x.drop_nulls()),
        "drop_nans": CellCheck(OtherLengths.DROP_NANS, lambda x: x.drop_nans()),
        "arg_where": CellCheck(OtherLengths.ARG_WHERE, pl.arg_where),
        "arg_true": CellCheck(OtherLengths.ARG_WHERE, lambda x: x.arg_true()),
        "gather_every": CellCheck(OtherLengths.GATHER_EVERY, lambda x: x.gather_every(2)),
        "rle": CellCheck(OtherLengths.RLE, lambda x: x.rle()),
        "shuffle": CellCheck(OtherLengths.RANDOM, lambda x: x.shuffle()),
        "hist": CellCheck(OtherLengths.HIST, lambda x: x.hist()),
        "reshape": CellCheck(OtherLengths.RESHAPE, lambda x: x.reshape((2, 1))),
        "value_counts": CellCheck(OtherLengths.VALUE_COUNTS, lambda x: x.value_counts()),
    }
    MANY_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...], int | None]]] = {
        "sample": (
            CellCheck(OtherLengths.RANDOM, lambda x, y: x.sample(y), Values.NUMBER_TEXT, skip=SKIP),
            (INTEGERS, ONE),
            1,
        ),
        "top_k": (
            CellCheck(OtherLengths.TOP_K, lambda x, y: x.top_k(y), Values.NUMBER_TEXT, skip=SKIP),
            (INTEGERS, ONE),
            1,
        ),
        "top_k_by": (
            CellCheck(
                OtherLengths.TOP_K_BY,
                lambda x, k, by: x.top_k_by(by, k),
                Values.NUMBER_TEXT,
                skip=SKIP,
            ),
            (INTEGERS, ONE, INTEGERS),
            1,
        ),
        "repeat": (
            CellCheck(OtherLengths.REPEAT, pl.repeat, skip=SKIP),
            (INTEGER, ONE),
            1,
        ),
        "extend_constant": (
            CellCheck(
                OtherLengths.EXTEND_CONSTANT,
                lambda x, y, z: x.extend_constant(y, z),
                skip=SKIP,
            ),
            (INTEGERS, ONE, ONE),
            2,
        ),
        "extend_constant by a category": (
            CellCheck(
                OtherLengths.EXTEND_CONSTANT,
                lambda x, y, z: x.extend_constant(y, z),
                skip=SKIP,
            ),
            (INTEGERS, CATEGORY, ONE),
            2,
        ),
        "repeat_by": (
            CellCheck(
                OtherLengths.REPEAT_BY, lambda x, y: x.repeat_by(y), Values.NUMBER_TEXT, skip=SKIP
            ),
            (INTEGERS, ONE),
            1,
        ),
        "search_sorted": (
            CellCheck(OtherLengths.SEARCH_SORTED, lambda x, y: x.search_sorted(y)),
            (INTEGERS, ONE),
            None,
        ),
        "append": (
            CellCheck(OtherLengths.APPEND, lambda x, y: x.append(y)),
            (INTEGERS, INTEGERS),
            None,
        ),
        "append beside a literal": (
            CellCheck(OtherLengths.APPEND, lambda x, y: x.append(y)),
            (ONE, ONE),
            None,
        ),
    }

    @classmethod
    def cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(OtherLengthNodes.cells()) > 6000
            True
        """
        samples = {name: NodeSample().cells(1) for name in cls.ONE_INPUT}
        for name, (_, typical, _) in cls.MANY_INPUTS.items():
            samples[name] = NodeSample().cells(len(typical), typical)
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, sample in samples.items():
            cells.extend((name, cell) for cell in sample if not cls._left_out(name, cell))
        return cells

    @classmethod
    def _left_out(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether the sample leaves a cell out: a fact of the data decides it.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the sample leaves the cell out.

        Example:
            >>> cell = (OtherLengthNodes.INTEGERS, Inputs().labelled("lit int 300/direct"))
            >>> OtherLengthNodes._left_out("sample", cell)
            True
        """
        if name == "reshape":
            return "List(" in cell[0].label
        if name == "sample" and cell[1].label.startswith("lit int 300/"):
            return True
        count = cls.MANY_INPUTS[name][2] if name in cls.MANY_INPUTS else None
        return count is not None and cell[count].label in cls.LARGE

    @classmethod
    def check(cls, name: str) -> CellCheck:
        """Give the comparison of the rules with polars for the node of a name.

        Args:
            name: The name of the node.

        Returns:
            The comparison.

        Example:
            >>> OtherLengthNodes.check("append").label(())
            ''
        """
        if name in cls.ONE_INPUT:
            return cls.ONE_INPUT[name]
        return cls.MANY_INPUTS[name][0]

    @classmethod
    def ids(cls, cells: list[tuple[str, tuple[Input, ...]]]) -> list[str]:
        """Give the identifier of each test: the node and the cell.

        Args:
            cells: The name of the node and the cell, for each cell.

        Returns:
            The identifiers.

        Example:
            >>> OtherLengthNodes.ids([("unique", (OtherLengthNodes.INTEGERS,))])
            ['unique-Int64/per row']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    OtherLengthNodes.cells(),
    ids=OtherLengthNodes.ids(OtherLengthNodes.cells()),
)
def test_function_of_another_length_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a function of another length match polars, at one cell of its sample."""
    problems = OtherLengthNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)
