"""The table: how a declared expression's output data type depends on its inputs.

A user declares the type of an expression with `typed_expr(...)` and its
kin: the columns that it reads, and what it gives for their data types.
The checker holds that declaration as a table. It has one axis for each
column that decides the output, and one cell for each combination of
kinds on those axes. A kind is a data type without its parameters, for
example `Datetime` for every time unit. Each cell holds an outcome (see
`Outcome`). A cell that is not there means that the declaration gives no
answer for those inputs.

The cells are the whole truth about a table. Two tables are one type
exactly when their canonical spellings agree, and the spelling comes from
the cells alone. So two declarations that say the same get one type,
however a user wrote them.

This module is part of the engine, which knows nothing about mypy.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.expression.axis_order import AxisOrder
from astrolabe._hooks.expression.outcome import UNCHECKED, Axis, Const, Outcome, Wrap

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astrolabe._hooks.expression.rendering import Rendered

Cell = tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """The data type that an expression gives, for each combination of input kinds.

    Every declared expression carries one table. Code that reads a
    declaration builds it from the declared regions and outcomes. Code
    that applies the declared type to a frame reads it at the frame's
    data types.

    `columns` holds the axes, sorted by `AxisOrder`, so two tables over
    the same columns line up cell for cell. `cells` holds each defined
    cell with its outcome, sorted by coordinates. A column that a
    declaration leaves `Unchecked` has one cell coordinate, `Unchecked`,
    that stands for every kind. This module calls such an axis
    trusted.

    Example:
        >>> from astrolabe._hooks.expression.rendering import Rendered
        >>> table = Table.constant(("a",), ("Int64",), "Boolean")
        >>> table.read_at({"a": Rendered.bare("Int64")})
        'Boolean'
    """

    columns: tuple[str, ...]
    cells: tuple[tuple[Cell, Outcome], ...]

    @classmethod
    def from_cells(cls, columns: tuple[str, ...], cells: Mapping[Cell, Outcome]) -> Table:
        """Build a table from cells that already hold outcomes.

        The caller can list the columns in any order. This method
        sorts them and each cell, so the stored form is canonical.

        Args:
            columns: The read columns, in any order.
            cells: Each defined cell mapped to its outcome. The
                coordinates follow the order of `columns`.

        Returns:
            The table, with its columns and cells sorted.

        Example:
            >>> built = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> built.columns
            ('a',)
        """
        order = AxisOrder().arranged(columns)
        arranged = {cls._arranged(columns, order, cell): outcome for cell, outcome in cells.items()}
        ordered = tuple(sorted(arranged.items(), key=operator.itemgetter(0)))
        return cls(order, ordered)

    @classmethod
    def from_outputs(
        cls,
        columns: tuple[str, ...],
        outputs: Mapping[Cell, str],
    ) -> Table:
        """Build a table from the output data type at each cell, as a declaration gives it.

        A declaration gives a plain text for each cell, for example
        `Int64`. `Classifier` turns each text into an outcome relative
        to the cell's inputs: `Int64` at an `Int64` input becomes "the
        input's own type". So one function always has one spelling,
        whatever the declaration wrote.

        Args:
            columns: The read columns, in any order.
            outputs: Each defined cell mapped to its output text.

        Returns:
            The table, with canonical outcomes.

        Example:
            >>> grid = Table.from_outputs(("a",), {("Int64",): "Int64"})
            >>> grid.cells[0][1]
            Axis(column='a')
        """
        order = AxisOrder().arranged(columns)
        classified = {
            cls._arranged(columns, order, cell): Classifier(order).classified(
                cls._arranged(columns, order, cell),
                text,
            )
            for cell, text in outputs.items()
        }
        ordered = tuple(sorted(classified.items(), key=operator.itemgetter(0)))
        return cls(order, ordered)

    @classmethod
    def constant(cls, columns: tuple[str, ...], cell: Cell, dtype: str) -> Table:
        """Build a table with one cell, which gives one fixed data type.

        Args:
            columns: The read columns.
            cell: The coordinates of the one cell.
            dtype: The fixed output data type.

        Returns:
            The table with one cell.

        Example:
            >>> Table.constant((), (), "UInt32").read_at({})
            'UInt32'
        """
        return cls.from_cells(columns, {cell: Const(dtype)})

    @classmethod
    def _arranged(cls, columns: tuple[str, ...], order: tuple[str, ...], cell: Cell) -> Cell:
        """Put one cell's coordinates in the order that the table stores.

        Args:
            columns: The column order of the given coordinates.
            order: The axis order that the table stores.
            cell: The coordinates, one for each column.

        Returns:
            The coordinates in the stored order.

        Example:
            >>> Table._arranged(("b", "a"), ("a", "b"), ("Int64", "String"))
            ('String', 'Int64')
        """
        at = dict(zip(columns, cell, strict=True))
        return tuple(at[name] for name in order)

    def outcome_at(self, cell: Cell) -> Outcome | None:
        """Give the outcome of one cell.

        Args:
            cell: The coordinates, in the stored order.

        Returns:
            The outcome, or None where the table has no cell there.

        Example:
            >>> table = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> table.outcome_at(("String",)) is None
            True
        """
        return self._outcomes.get(cell)

    @cached_property
    def _outcomes(self) -> dict[Cell, Outcome]:
        """Index the cells by their coordinates, once for each table.

        Returns:
            Each defined cell mapped to its outcome.

        Example:
            >>> Table.from_cells(("a",), {("Int64",): Axis("a")})._outcomes
            {('Int64',): Axis(column='a')}
        """
        return dict(self.cells)

    def read_at(self, assignment: Mapping[str, Rendered]) -> str | None:
        """Give the output data type for the data types that a frame declares.

        This is how code that applies an expression to a frame gets
        the output column's data type. The read uses only the table's
        own axes. The assignment can name more columns, and those
        columns change nothing.

        Args:
            assignment: Each column name mapped to its rendering, the
                text that polars prints for its data type. It must
                hold every column of the table.

        Returns:
            The rendering of the output data type. None where the
            declaration gives no answer for these inputs. `Unchecked`
            where an input kind is one that the checker does not know
            (see `UnknownKind`).

        Example:
            >>> from astrolabe._hooks.expression.rendering import Rendered
            >>> table = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> wide = {"a": Rendered.bare("Int64"), "b": Rendered.bare("Date")}
            >>> table.read_at(wide)
            'Int64'
            >>> table.read_at({"a": Rendered.bare("String")}) is None
            True
            >>> table.read_at({"a": Rendered.bare("Object")})
            'Unchecked'
        """
        cell = self._coordinates(assignment)
        outcome = self.outcome_at(cell)
        if outcome is None:
            return UnknownKind().answer_at(cell)
        return outcome.evaluated(assignment)

    def _coordinates(self, assignment: Mapping[str, Rendered]) -> Cell:
        """Give the coordinates of the cell that one assignment reads.

        A trusted axis (one that a declaration left `Unchecked`) has
        one coordinate, `Unchecked`. Whatever data type the frame
        gives that column, the read uses the `Unchecked` coordinate.

        Args:
            assignment: Each column name mapped to its rendering.

        Returns:
            The coordinates, in the stored order.

        Example:
            >>> from astrolabe._hooks.expression.rendering import Rendered
            >>> trusted = Table.from_cells(("q",), {(UNCHECKED,): Axis("q")})
            >>> trusted._coordinates({"q": Rendered.bare("Int64")})
            ('Unchecked',)
        """
        trusted = self.trusted_axes()
        coordinates: list[str] = []
        for name in self.columns:
            coordinates.append(UNCHECKED if name in trusted else assignment[name].kind)
        return tuple(coordinates)

    def trusted_axes(self) -> frozenset[str]:
        """Give the axes that a declaration left `Unchecked`.

        A user can declare an expression that reads a column of any
        data type. Its table then has one coordinate, `Unchecked`, on
        that axis, and the coordinate stands for every kind at once. A
        read at any rendering uses it.

        Returns:
            The trusted columns.

        Example:
            >>> trusted = Table.from_cells(("q",), {(UNCHECKED,): Axis("q")})
            >>> trusted.trusted_axes()
            frozenset({'q'})
            >>> Table.from_cells(("a",), {("Int64",): Axis("a")}).trusted_axes()
            frozenset()
        """
        return self._trusted

    @cached_property
    def _trusted(self) -> frozenset[str]:
        """Find the trusted axes, once for each table.

        Returns:
            The columns whose only coordinate is `Unchecked`.

        Example:
            >>> Table.from_cells(("q",), {(UNCHECKED,): Axis("q")})._trusted
            frozenset({'q'})
        """
        return frozenset(name for name in self.columns if self.projection(name) == {UNCHECKED})

    def projection(self, column: str) -> frozenset[str]:
        """Give the kinds that the cells cover on one axis.

        Args:
            column: The column of the axis.

        Returns:
            The coordinates on that axis.

        Example:
            >>> one = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> one.projection("a")
            frozenset({'Int64'})
        """
        return self._projections[column]

    @cached_property
    def _projections(self) -> dict[str, frozenset[str]]:
        """Give the coordinates on every axis, once for each table.

        Returns:
            Each column mapped to the kinds that its cells cover.

        Example:
            >>> Table.from_cells(("a",), {("Int64",): Axis("a")})._projections
            {'a': frozenset({'Int64'})}
        """
        coordinates = zip(*(cell for cell, _ in self.cells), strict=True)
        # A table with no cell has no coordinate on any axis, so zip gives nothing.
        found = dict(zip(self.columns, map(frozenset, coordinates), strict=False))
        return {name: found.get(name, frozenset()) for name in self.columns}

    def spelling(self) -> tuple[tuple[str, tuple[Cell, ...]], ...]:
        """Give the canonical spelling, which decides whether two tables are equal.

        The spelling groups the cells by outcome. It depends only on
        the cells, so two tables with the same cells have the same
        spelling, however they were built.

        Returns:
            Pairs of an outcome's spelling and its sorted cells,
            sorted.

        Example:
            >>> table = Table.from_cells(("a",), {("Int64",): Axis("a")})
            >>> table.spelling()
            (('axis:a', (('Int64',),)),)
        """
        regions: dict[str, list[Cell]] = {}
        for cell, outcome in self.cells:
            regions.setdefault(outcome.spelling(), []).append(cell)
        return tuple(sorted((label, tuple(sorted(cells))) for label, cells in regions.items()))


class UnknownKind:
    """The answer at a cell whose kind the checker does not know.

    A declaration names its regions with the kinds that the checker
    knows: the scalar kinds, `List`, `Array` and `Struct`. A cell that
    a table does not hold at such a kind is a place where the
    declaration gives no answer, and the checker reports an error. A
    kind outside that set, for example `Object`, is one that no
    declaration names and no rule of the checker reads. So a cell at
    such a kind reads as `Unchecked`: the checker knows nothing about
    the kind and says so. Code that applies the expression to a frame
    treats that `Unchecked` as a gap in the checker, and reports it
    only where the settings ask for such reports.

    Example:
        >>> UnknownKind().answer_at(("Object",))
        'Unchecked'
        >>> UnknownKind().answer_at(("String",)) is None
        True
        >>> UnknownKind().answer_at(("List",)) is None
        True
    """

    # The kinds that the checker knows.
    KNOWN_KINDS: ClassVar[frozenset[str]] = frozenset({
        "Array",
        "Binary",
        "Boolean",
        "Categorical",
        "Date",
        "Datetime",
        "Decimal",
        "Duration",
        "Enum",
        "Float16",
        "Float32",
        "Float64",
        "Int128",
        "Int16",
        "Int32",
        "Int64",
        "Int8",
        "List",
        "Null",
        "String",
        "Struct",
        "Time",
        "UInt128",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt8",
    })

    def answer_at(self, cell: Cell) -> str | None:
        """Give the answer at a cell that the table does not hold.

        Args:
            cell: The coordinates.

        Returns:
            `Unchecked` where a coordinate is a kind that the checker
            does not know, else None, which means that the declaration
            gives no answer.
        """
        if any(kind not in self.KNOWN_KINDS for kind in cell):
            return UNCHECKED
        return None


@dataclass(frozen=True)
class Classifier:
    """Turn an output data type into an outcome relative to the cell's inputs.

    A declaration gives a plain text for each cell, for example
    `List(Int64)` at an `Int64` input. As a fixed type, that text says
    nothing for other element types. As an outcome relative to the
    input, "a `List` of the input's own type", it carries the rule. The
    classifier finds that outcome.

    One text can fit several outcomes, so the classifier tries them in
    a fixed order: the input's own type, then a container around it,
    then a fixed type. Where two axes fit, the first axis wins. The
    cells of a declaration are kinds, that is, data types with their
    parameters taken away. So a text with parameters, such as
    `Decimal(precision=38, scale=2)`, is a fixed type, never the
    input's own type.

    Example:
        >>> Classifier(("a",)).classified(("Int64",), "Int64")
        Axis(column='a')
        >>> wide = "Decimal(precision=38, scale=2)"
        >>> Classifier(("d",)).classified(("Decimal",), wide)
        Const(dtype='Decimal(precision=38, scale=2)')
    """

    columns: tuple[str, ...]

    def classified(self, cell: Cell, text: str) -> Outcome:
        """Give the canonical outcome for one cell's output text.

        Args:
            cell: The coordinates, in the stored order.
            text: The output text.

        Returns:
            The first outcome that fits, in the fixed order, else a
            `Const`.

        Example:
            >>> Classifier(("a",)).classified(("Int64",), "List(Int64)")
            Wrap(constructor='List', inner=Axis(column='a'))
        """
        for finder in (self._as_axis, self._as_wrap):
            outcome = finder(cell, text)
            if outcome is not None:
                return outcome
        return Const(text)

    def _as_axis(self, cell: Cell, text: str) -> Outcome | None:
        """Read the text as the own type of one input.

        Args:
            cell: The coordinates, in the stored order.
            text: The output text.

        Returns:
            An `Axis` for the first column that fits, or None.

        Example:
            >>> Classifier(("a", "b")).classified(("Int64", "Int64"), "Int64")
            Axis(column='a')
            >>> micro = "Datetime(time_unit='us', time_zone=None)"
            >>> Classifier(("t",))._as_axis(("Datetime",), micro) is None
            True
        """
        for name, kind in zip(self.columns, cell, strict=True):
            if text == kind:
                return Axis(name)
        return None

    def _as_wrap(self, cell: Cell, text: str) -> Outcome | None:
        """Read the text as a container around the type of one input.

        Args:
            cell: The coordinates, in the stored order.
            text: The output text.

        Returns:
            A `Wrap` around the first column that fits, or None.

        Example:
            >>> Classifier(("a",))._as_wrap(("Int64",), "List(Int64)")
            Wrap(constructor='List', inner=Axis(column='a'))
        """
        for name, kind in zip(self.columns, cell, strict=True):
            if f"({kind})" in text:
                head = text.split(f"({kind})", maxsplit=1)[0]
                return Wrap(head, Axis(name))
        return None
