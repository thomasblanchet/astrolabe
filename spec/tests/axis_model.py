"""Multi-axis tables: the expression type map over several columns.

`table_model.py` holds the one-axis table. This module holds the
general object: a grid over any number of read columns, one axis per
column the output depends on. It exists so the axis half of the
composition theorem, the fit rule's axis-count mismatch, and the open
chain's axis growth can be measured instead of assumed.

Ground truth is the cell map, exactly as for one axis: equality of
grids is equality of their outputs, and the canonical spelling is
derived from the outputs alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from spec.tests.table_model import QUOTIENT, Table

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

Cell = tuple[str, ...]


class AxisOrder:
    """Order the read columns of a grid: shorter names first, then alphabetically.

    `D-35` fixes this order. Axis `i` is the `i`-th name in it.

    Example:
        >>> AxisOrder().arranged(["b", "aa", "a"])
        ('a', 'b', 'aa')
    """

    def arranged(self, names: Iterable[str]) -> tuple[str, ...]:
        """Put column names in axis order.

        Args:
            names: The read columns, in any order.

        Returns:
            The names, shorter first, then alphabetically.

        Example:
            >>> AxisOrder().arranged({"price", "id"})
            ('id', 'price')
        """
        return tuple(sorted(names, key=lambda name: (len(name), name)))


@dataclass(frozen=True)
class Grid:
    """One expression type map over named columns, stored extensionally.

    The axes are the columns the output value depends on, by name and
    in sorted order, so two grids over one column set line up cell for
    cell.

    Example:
        >>> grid = Grid.constant(("a",), "UInt32")
        >>> grid.output_at(("Int64",))
        'UInt32'
    """

    columns: tuple[str, ...]
    outputs: tuple[tuple[Cell, str], ...]

    @classmethod
    def from_outputs(cls, columns: tuple[str, ...], outputs: Mapping[Cell, str]) -> Grid:
        """Build a grid from measured cell-to-output pairs.

        Args:
            columns: The read columns, in any order.
            outputs: Each defined cell mapped to its output type.

        Each cell's coordinates are reordered from the given column
        order to the sorted one.

        Returns:
            The grid, columns and cells sorted.

        Example:
            >>> Grid.from_outputs(("a",), {("Int64",): "Int64"}).columns
            ('a',)
            >>> Grid.from_outputs(("b", "a"), {("Int64", "String"): "Int64"}).outputs
            ((('String', 'Int64'), 'Int64'),)
        """
        order = AxisOrder().arranged(columns)
        rearranged: dict[Cell, str] = {}
        for cell, out in outputs.items():
            at = dict(zip(columns, cell, strict=True))
            rearranged[tuple(at[name] for name in order)] = out
        return cls(order, tuple(sorted(rearranged.items())))

    @classmethod
    def constant(cls, columns: tuple[str, ...], output: str) -> Grid:
        """Build the grid answering one type at every cell.

        Args:
            columns: The read columns.
            output: The one output type.

        Returns:
            The total constant grid.

        Example:
            >>> Grid.constant(("a", "b"), "Boolean").axes()
            2
        """
        cells = product(QUOTIENT, repeat=len(columns))
        return cls.from_outputs(columns, {cell: output for cell in cells})

    @classmethod
    def lifted(cls, column: str, table: Table) -> Grid:
        """Lift a one-axis table into a grid over one named column.

        Args:
            column: The read column's name.
            table: The one-axis table.

        Returns:
            The one-column grid with the table's cells.

        Example:
            >>> from spec.tests.table_model import Rule
            >>> one = Table.from_rules([Rule(frozenset({"Int64"}), "same")])
            >>> Grid.lifted("a", one).output_at(("Int64",))
            'Int64'
        """
        outputs: dict[Cell, str] = {}
        for cell in QUOTIENT:
            out = table.output_at(cell)
            if out is not None:
                outputs[cell,] = out
        return cls.from_outputs((column,), outputs)

    def axes(self) -> int:
        """Give the number of axes.

        Returns:
            One per read column.

        Example:
            >>> Grid.constant(("a", "b"), "Boolean").axes()
            2
        """
        return len(self.columns)

    def output_at(self, cell: Cell) -> str | None:
        """Read one cell.

        Args:
            cell: One coordinate per column, in stored order.

        Returns:
            The output type, or None for an error cell.

        Example:
            >>> Grid.constant(("a",), "Int64").output_at(("String",))
            'Int64'
        """
        return dict(self.outputs).get(cell)

    def read_at(self, assignment: Mapping[str, str]) -> str | None:
        """Read the grid at a wider assignment of columns to types.

        This is the fit rule's axis-count answer: a table is read by
        its own axes' coordinates, and a column the table does not
        depend on contributes no coordinate, so any assignment naming
        at least the grid's columns reads one cell.

        Args:
            assignment: Column names mapped to types, the grid's
                columns among them.

        Returns:
            The output type, or None for an error cell.

        Example:
            >>> grid = Grid.constant(("a",), "Int64")
            >>> grid.read_at({"a": "String", "b": "Date"})
            'Int64'
        """
        return self.output_at(tuple(assignment[name] for name in self.columns))

    def combined(self, other: Grid, joiner: Mapping[tuple[str, str], str]) -> Grid:
        """Combine two grids under a binary type rule, axes united.

        Each cell of the united column space reads both operands at
        its own coordinates and pushes the pair through the rule. An
        axis appears exactly when one operand depends on the column.

        Args:
            other: The other operand's grid.
            joiner: The binary rule, unordered pairs to outputs.

        Returns:
            The combined grid over the united columns.

        Example:
            >>> left = Grid.constant(("a",), "Int64")
            >>> right = Grid.constant(("b",), "Int64")
            >>> both = left.combined(right, {("Int64", "Int64"): "Int64"})
            >>> both.columns
            ('a', 'b')
            >>> left.combined(left, {}).outputs
            ()
        """
        columns = AxisOrder().arranged(set(self.columns) | set(other.columns))
        outputs: dict[Cell, str] = {}
        for cell in product(QUOTIENT, repeat=len(columns)):
            assignment = dict(zip(columns, cell, strict=True))
            mine = self.read_at(assignment)
            theirs = other.read_at(assignment)
            if mine is None or theirs is None:
                continue
            out = joiner.get((min(mine, theirs), max(mine, theirs)))
            if out is not None:
                outputs[cell] = out
        return Grid.from_outputs(columns, outputs)

    def spelling(self) -> tuple[tuple[str, tuple[Cell, ...]], ...]:
        """Give the canonical spelling: outcomes, each with its cells.

        Classification prefers an axis over a wrap and a wrap over a
        constant, the lowest axis winning ties, exactly the one-axis
        rules with the axis index made explicit.

        Returns:
            Sorted (outcome, sorted cells) pairs; error by absence.

        Example:
            >>> Grid.constant(("a",), "UInt32").spelling()[0][0]
            'const:UInt32'
        """
        regions: dict[str, list[Cell]] = {}
        for cell, output in self.outputs:
            regions.setdefault(self._classified(cell, output), []).append(cell)
        return tuple(sorted((k, tuple(sorted(v))) for k, v in regions.items()))

    def _classified(self, cell: Cell, output: str) -> str:
        """Express one cell's output relative to its coordinates.

        An axis label says the output is the input's own data type at
        that axis, parameters included. A wrap label says the output
        wraps that input in a constructor. Every axis is tried before
        any wrap, so an axis outranks a wrap whatever their positions,
        as `D-37` says. The lowest axis wins a tie.

        Args:
            cell: The input coordinates.
            output: The measured output type.

        Returns:
            "axis:i", "wrap:C:i", or the constant output.

        Example:
            >>> Grid(("a", "b"), ())._classified(("Int64", "Int32"), "Int64")
            'axis:0'
            >>> Grid(("a", "b"), ())._classified(("Int64", "List"), "List(Int64)")
            'axis:1'
            >>> Grid(("a",), ())._classified(("Int64",), "List(Int64)")
            'wrap:List:0'
            >>> Grid(("a",), ())._classified(("Int64",), "String")
            'const:String'
        """
        for index, coordinate in enumerate(cell):
            if output == coordinate or output.startswith(coordinate + "("):
                return f"axis:{index}"
        for index, coordinate in enumerate(cell):
            if f"({coordinate})" in output:
                constructor = output.split(f"({coordinate})", maxsplit=1)[0]
                return f"wrap:{constructor}:{index}"
        return "const:" + output
