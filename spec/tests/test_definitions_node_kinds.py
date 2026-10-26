"""Check the definitions of the nodes of a kind of their own against polars, node by node.

The family holds the nodes that are not functions: `when`, `filter`, `sort`,
`sort_by`, `slice`, `gather`, `explode`, `over`, `rolling`, `alias` and the
renames of `.name`. Each
test builds the node of polars over the inputs of one cell, and compares the
four answers of polars with the four answers of the rules (`CellCheck`): the
planner type and the run type, in a `select` and in `agg` over two groups of
other sizes.

A node of one input takes every input (`NodeSample`), a node of two inputs
about 500 pairs beside a typical input at each place (`PairSample`), and a
node of three inputs about 500 cells (`TripleSample`).

Two facts of the test frames leave cells out. An offset or a length of
`slice` with more than one row makes `collect()` hang (`spec/POLARS_BEHAVIORS.md`), so
those places take inputs of one row only. And each column of the test frames
repeats its three values over five rows, so a column of one value per row is
not sorted, which the index of `rolling` must be: that place leaves such
columns out. The keys of `over` of one value per row repeat their values, so
the groups of the keys hold one or two rows, and whether a function of one
row or of another length keeps the rows of each group is a fact of the
values: `over` leaves out those functions beside such keys. The test frames
hold five rows, so a literal index of `300` or of `2**40` falls outside them:
`gather` and `get` leave out those indices. A further fact is a
defect of polars: `sort` over one value for
each group reads out of bounds in `agg` (`spec/POLARS_BEHAVIORS.md`), and can crash
or allocate memory with no bound over values of a variable size: text,
bytes, categories and nested types. So `sort` leaves out every input of one
value of such a type. `.name.keep()` over a literal reads no column, which
polars refuses while it converts the whole tree, so the sample of the node
leaves out the literals, and the test of the tree evaluation checks them.

Each test names the node and the cell, so a failure names both.
"""

from __future__ import annotations

from typing import ClassVar

import polars as pl
import pytest

from spec.tests.rule_probes import ApiGate, CellCheck, Input, Inputs, NodeSample
from astrolabe._rules.definitions.node_kinds import NodeKinds


class KindNodes:
    """Hold the nodes of the tests, each with its call and its typical inputs.

    Example:
        >>> sorted(KindNodes.ONE_INPUT)[:2]
        ['alias', 'explode']
    """

    VARIABLE_SIZE: ClassVar[tuple[str, ...]] = (
        "String/",
        "Binary/",
        "Categorical/",
        "Enum(",
        "List(",
        "Array(",
        "Struct(",
    )
    INTEGERS: ClassVar[Input] = Inputs().labelled("Int64/per row")
    BOOLEANS: ClassVar[Input] = Inputs().labelled("Boolean/per row")
    DATE: ClassVar[Input] = Inputs().labelled("Date/one value")
    ONE: ClassVar[Input] = Inputs().labelled("lit int 1/direct")

    ONE_INPUT: ClassVar[dict[str, CellCheck]] = ApiGate().kept({
        "sort": CellCheck(NodeKinds.SORT, lambda x: x.sort()),
        "explode": CellCheck(NodeKinds.EXPLODE, lambda x: x.explode()),
        "flatten": CellCheck(NodeKinds.EXPLODE, lambda x: x.flatten()),
        "alias": CellCheck(NodeKinds.ALIAS, lambda x: x.alias("y")),
        "name.keep": CellCheck(NodeKinds.KEEP_NAME, lambda x: x.name.keep()),
        "name.suffix": CellCheck(NodeKinds.RENAME_ALIAS, lambda x: x.name.suffix("_y")),
    })
    MANY_INPUTS: ClassVar[dict[str, tuple[CellCheck, tuple[Input, ...]]]] = {
        "when": (
            CellCheck(NodeKinds.WHEN, lambda x, y, z: pl.when(x).then(y).otherwise(z)),
            (BOOLEANS, INTEGERS, INTEGERS),
        ),
        "filter": (CellCheck(NodeKinds.FILTER, lambda x, y: x.filter(y)), (INTEGERS, BOOLEANS)),
        "sort_by": (CellCheck(NodeKinds.SORT_BY, lambda x, y: x.sort_by(y)), (INTEGERS, INTEGERS)),
        "slice": (CellCheck(NodeKinds.SLICE, lambda x, y, z: x.slice(y, z)), (INTEGERS, ONE, ONE)),
        "gather": (CellCheck(NodeKinds.GATHER, lambda x, y: x.gather(y)), (INTEGERS, ONE)),
        "get": (CellCheck(NodeKinds.GATHER, lambda x, y: x.get(y)), (INTEGERS, ONE)),
        "over": (CellCheck(NodeKinds.OVER, lambda x, y: x.over(y)), (INTEGERS, INTEGERS)),
        "rolling": (
            CellCheck(NodeKinds.ROLLING, lambda x, y: x.rolling(index_column=y, period="2d")),
            (INTEGERS, DATE),
        ),
    }

    @classmethod
    def cells(cls) -> list[tuple[str, tuple[Input, ...]]]:
        """Give each node beside each cell of its sample.

        Returns:
            The name of the node and the cell, for each cell.

        Example:
            >>> len(KindNodes.cells()) > 3000
            True
        """
        samples = {name: NodeSample().cells(1) for name in cls.ONE_INPUT}
        for name, (_, typical) in cls.MANY_INPUTS.items():
            samples[name] = NodeSample().cells(len(typical), typical)
        cells: list[tuple[str, tuple[Input, ...]]] = []
        for name, sample in samples.items():
            cells.extend((name, cell) for cell in sample if not cls._left_out(name, cell))
        return cells

    @classmethod
    def _unsafe(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether `sort` reads out of bounds over a cell, where polars can crash.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the sample leaves the cell out.

        Example:
            >>> KindNodes._unsafe("sort", (Inputs().labelled("List(Int64)/one value"),))
            True
        """
        label = cell[0].label
        variable = label.startswith(cls.VARIABLE_SIZE)
        return name == "sort" and variable and label.endswith("/one value")

    @classmethod
    def _left_out(cls, name: str, cell: tuple[Input, ...]) -> bool:
        """Say whether the sample leaves a cell out: a fact of the test frames decides it.

        Args:
            name: The name of the node.
            cell: The cell.

        Returns:
            Whether the sample leaves the cell out.

        Example:
            >>> KindNodes._left_out("rolling", (KindNodes.INTEGERS, KindNodes.INTEGERS))
            True
        """
        rows = ("/per row", "/other length")
        if cls._unsafe(name, cell):
            return True
        if name == "slice":
            return any(entry.label.endswith(rows) for entry in cell[1:])
        if name == "name.keep":
            return cell[0].label.startswith("lit ")
        if name in {"gather", "get"}:
            return cell[1].label.startswith(("lit int 300/", "lit int 2**40/"))
        if name == "over":
            shaped = cell[0].label.endswith(("/one row", "/other length"))
            return shaped and cell[1].label.endswith("/per row")
        return name == "rolling" and cell[1].label.endswith("/per row")

    @classmethod
    def check(cls, name: str) -> CellCheck:
        """Give the comparison of the rules with polars for the node of a name.

        Args:
            name: The name of the node.

        Returns:
            The comparison.

        Example:
            >>> KindNodes.check("sort").label(())
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
            >>> KindNodes.ids([("sort", (KindNodes.INTEGERS,))])
            ['sort-Int64/per row']
        """
        return [f"{name}-{CellCheck.label(cell)}" for name, cell in cells]


@pytest.mark.parametrize(
    ("node", "cell"),
    KindNodes.cells(),
    ids=KindNodes.ids(KindNodes.cells()),
)
def test_node_of_its_own_kind_gives_the_answers_of_polars(
    node: str,
    cell: tuple[Input, ...],
) -> None:
    """The four answers of a node of a kind of its own match polars, at one cell of its sample."""
    problems = KindNodes.check(node).problems(list(cell))
    assert not problems, "\n".join(problems)
