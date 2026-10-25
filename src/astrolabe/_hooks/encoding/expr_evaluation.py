"""Find the column that an expression argument gives on one frame.

A frame operation such as `select`, `with_columns` or `agg` takes expressions as
arguments, and its result has one column for each of them. To type that result,
a hook reads the type of each argument back into an expression type, gives the
engine the columns of the frame, lets the engine apply the expression type to
them, and turns the answer back into a column. The engine calls that application
"materialize".

This module holds that path once. Every operation that takes expressions goes
through it, so all of them read their arguments in the same way and give the
same answer for the same argument. If the engine finds no answer, the path gives
a short text that says why, and the hook turns that text into its error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from astrolabe._hooks.encoding.expr_codec import QuadrupleReader
from astrolabe._hooks.encoding.schema_renderings import (
    MaterializedColumn,
    SchemaRenderings,
)
from astrolabe._hooks.expression.kind import KindRoster
from astrolabe._hooks.expression.materialize import Materializer
from astrolabe._hooks.expression.selection import SelectionMaterializer
from astrolabe._hooks.propagate import not_none, propagates

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mypy.types import Type as MypyType

    from astrolabe._hooks.domain.column import ColumnRef
    from astrolabe._hooks.encoding.marker_minter import MarkerMinter
    from astrolabe._hooks.expression.quadruple import ExpressionType
    from astrolabe._hooks.expression.selection import RemainderEffect, SelectionType
    from astrolabe._hooks.remainder.record import Record


class ExpressionArgument:
    """Read the type of an argument back into an expression type, with no access to the plugin.

    An argument can be a declared expression type in its full or its short
    form. Code that needs only the expression type, and builds nothing, uses
    this class.

    Example:
        >>> from mypy.types import UninhabitedType
        >>> ExpressionArgument().decoded(UninhabitedType()) is None
        True
    """

    def decoded(self, typ: MypyType) -> ExpressionType | None:
        """Read the type of one argument back into an expression type.

        Args:
            typ: The type of the argument.

        Returns:
            The expression type, or None for a type that is not an expression.
            The caller then falls back: it gives the plain frame, or keeps the
            column under `Unchecked`.
        """
        return QuadrupleReader().decoded(typ)


class ExpressionEvaluation:
    """Apply an expression argument to the columns of one frame, and give the columns that it makes.

    A hook for an operation that takes expressions uses this class for every
    step between the mypy type of an argument and the columns of its result. The
    class reads the argument, turns the declared columns of the frame into the
    text form that the engine reads, asks the engine, and builds a `ColumnRef`
    from each answer. It handles plain expressions, selections that give one
    column for each picked column, and the fields that `.struct.unnest()`
    spreads. A selection is a selector expression such as `cs.numeric()`, with
    any operations on it.

    If the engine finds no answer, a method gives a short text in place of the
    column, and the calling hook turns that text into its error.

    Example:
        >>> from astrolabe._hooks.expression.roots import LiteralRoot
        >>> evaluation = ExpressionEvaluation(new_marker_minter(new_type_fixture()))
        >>> evaluation.materialized(LiteralRoot().of("Int32"), [], "row").name
        'literal'
    """

    # The scalar kinds that a declared table can name one by one.
    UNIVERSE: ClassVar[tuple[str, ...]] = (
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
        "Null",
        "String",
        "Time",
        "UInt128",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt8",
    )
    # The kinds outside that list, which an `Unchecked` column can still
    # hold (see `Materializer`).
    _NESTED: ClassVar[tuple[str, ...]] = tuple(
        sorted(frozenset(KindRoster().names()) - frozenset(UNIVERSE)),
    )

    def __init__(self, minter: MarkerMinter) -> None:
        """Keep the builder that turns an answer back into a column.

        Args:
            minter: The marker minter for the current phase.
        """
        self._column = MaterializedColumn(minter)

    def materialized(
        self,
        expression: ExpressionType,
        columns: Sequence[ColumnRef],
        context: str,
    ) -> ColumnRef | str:
        """Apply one expression type to the declared columns of one frame.

        Args:
            expression: The expression type of the argument.
            columns: The declared columns of the frame.
            context: "row" for a row context such as `select`, or "agg" for the
                aggregation context of `agg`.

        Returns:
            The output column, or a text that says why the engine found no
            answer. The calling hook turns that text into its error. An answer
            whose rendering has no type is such a case. A rendering is the text
            that polars shows for a data type.

        Example:
            >>> from astrolabe._hooks.expression.character import ELEMENTWISE
            >>> from astrolabe._hooks.expression.outcome import Const
            >>> from astrolabe._hooks.expression.quadruple import ExpressionType
            >>> from astrolabe._hooks.expression.table import Table
            >>> fixture = new_type_fixture()
            >>> evaluation = ExpressionEvaluation(new_marker_minter(fixture))
            >>> a = column_ref(fixture, "a", int_dtype(fixture))
            >>> broken = Table.from_cells(("a",), {("Int64",): Const("Nope(")})
            >>> expression = ExpressionType(frozenset({"a"}), broken, "a", ELEMENTWISE)
            >>> evaluation.materialized(expression, [a], "row")
            "the declared result 'Nope(' is not a data type"
        """
        schema = SchemaRenderings().of(columns)
        result = Materializer(self.UNIVERSE, self._NESTED).result(
            schema,
            expression,
            context,
        )
        if isinstance(result, str):
            return result
        name, rendering = next(iter(result.items()))
        column = self._column.built(name, rendering)
        if column is None:
            return f"the declared result '{rendering}' is not a data type"
        return column

    def selection_outputs(
        self,
        selection: SelectionType,
        columns: Sequence[ColumnRef],
        context: str,
    ) -> list[ColumnRef] | str:
        """Apply one declared selection to the declared columns of one frame.

        A selection gives one output column for each column that its selector
        picks.

        Args:
            selection: The selection type of the argument.
            columns: The declared columns that the selection picks from.
            context: "row" or "agg".

        Returns:
            The output columns, in the order of the frame, or the name of a
            picked column where the selection has no answer.
        """
        schema = SchemaRenderings().of(columns)
        materializer = SelectionMaterializer(self.UNIVERSE)
        outputs = materializer.outputs(schema, selection, context)
        if isinstance(outputs, str):
            return outputs
        stored_schema = SchemaRenderings().stored(columns)
        stored = outputs
        if stored_schema != schema:
            stored = materializer.outputs(stored_schema, selection, context)
        return self._spliced_all(outputs, outputs if isinstance(stored, str) else stored)

    def partially_picked(self, selection: SelectionType, columns: Sequence[ColumnRef]) -> list[str]:
        """Give the declared columns that one declared selection picks for some data only.

        A column declared `Unchecked` can hold any kind, so a selector that
        tests the data type can pick it for some data and not for other data.

        Args:
            selection: The selection type of the argument.
            columns: The declared columns that the selection picks from.

        Returns:
            The names, in the order of the frame.

        Example:
            >>> from astrolabe._hooks.expression.selection import SelectionType
            >>> from astrolabe._hooks.remainder.selector import OfType
            >>> from astrolabe._hooks.remainder.type_set import Family
            >>> fixture = new_type_fixture()
            >>> evaluation = ExpressionEvaluation(new_marker_minter(fixture))
            >>> u = column_ref(fixture, "u", unchecked_instance(fixture))
            >>> numeric = SelectionType.identity(OfType(Family.numeric()))
            >>> evaluation.partially_picked(numeric, [u])
            ['u']
        """
        schema = SchemaRenderings().of(columns)
        return SelectionMaterializer(self.UNIVERSE).partial(schema, selection)

    def selection_effect(
        self,
        selection: SelectionType,
        record: Record,
        context: str,
    ) -> RemainderEffect | str:
        """Give what one declared selection does to the remainder of an open frame.

        The remainder is the part of an open frame that its type does not name.
        A selection can pick columns there too, so the result has a remainder
        that the selection changed: kept, retyped or renamed.

        Args:
            selection: The selection type of the argument.
            record: What the checker knows about the remainder: the names that
                it cannot hold and the data types that its columns can have.
            context: "row" or "agg".

        Returns:
            The effect on the remainder, or a text that says why the engine
            found no answer.
        """
        return SelectionMaterializer(self.UNIVERSE).remainder(record, selection, context)

    @propagates("a declared result is not a data type")
    def _spliced_all(
        self,
        outputs: Mapping[str, str],
        stored: Mapping[str, str],
    ) -> list[ColumnRef] | str:
        """Turn every answer of the engine into a column, with the type that a lazy frame stores.

        Args:
            outputs: The answers: each output name with its rendering.
            stored: The answers at the stored types of the columns, by name.

        Returns:
            The columns, or a text that says that a rendering has no type.

        Example:
            >>> fixture = new_type_fixture()
            >>> evaluation = ExpressionEvaluation(new_marker_minter(fixture))
            >>> answers = {"a": "Int64", "b": "String"}
            >>> [c.name for c in evaluation._spliced_all(answers, answers)]
            ['a', 'b']
        """
        columns: list[ColumnRef] = []
        for name, rendering in outputs.items():
            kept = stored.get(name, rendering)
            if kept == rendering:
                column = self._column.built(name, rendering)
            else:
                column = self._column.planned(name, rendering, kept)
            columns.append(not_none(column))
        return columns
